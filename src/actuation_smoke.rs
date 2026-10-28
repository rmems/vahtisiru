//! Maintainer-only hardware actuation smoke (RM-1909).
//!
//! Crate-private. The read-only smoke (`--gpu-hardware-smoke`) proves NVML
//! identity, telemetry, and power-limit *reads*. This module proves the real
//! brake path on physical hardware: it applies the relay brake through the
//! same UUID-scoped `nvidia-smi` command plan the supervisor uses, observes
//! the resulting limit, waits for the documented healthy streak, releases,
//! and verifies the device is back at its default.
//!
//! The sequence is a pure function of an [`ActuationSmokeBackend`] so every
//! refusal, verification, and cleanup path is exercised in software-only CI
//! with a fake backend. Only the `vahtisiru` executable wires the NVML /
//! `sudo -n nvidia-smi` backend (see `crate::daemon`), and it takes the
//! supervisor's single-instance lock first. Normal CI never runs the hardware
//! path.
//!
//! Fail-closed rules (no mutation unless every pre-check passes):
//!
//! 1. The resolved GPU UUID must equal the operator-supplied UUID.
//! 2. Current and default limits must be readable, positive, and the current
//!    limit must sit at/above the default — a leftover relay brake or a foreign
//!    sub-default cap is never adopted.
//! 3. The planned apply command must pin `-i <UUID>` before `-pl <watts>` and
//!    target exactly the brake wattage; an unscoped `nvidia-smi -pl` plan is a
//!    refusal.
//! 4. Non-interactive `sudo -n` must already permit both the apply and the
//!    release command (checked with `sudo -l`, which runs nothing).
//!
//! After the brake is applied, any failure (unexpected limit, unhealthy
//! sample, release error) triggers one cleanup release through the production
//! actuator — which itself refuses to restore a limit it does not recognise
//! as the relay's brake. Success is claimed only after the post-release limit
//! has been observed at the device default; otherwise the result names the
//! explicit operator recovery command.

use crate::safety::{
    ActuatorError, BRAKE_FRACTION, BRAKE_MATCH_TOLERANCE_W, PowerLimitObservation, SafetyActuator,
    classify_power_limit, unambiguous_brake_target,
};
use std::fmt;
use std::io::Write;

/// Hardware seam for [`run_actuation_smoke`].
///
/// The production implementation lives in the executable and is backed by
/// `GpuTarget` + `NvmlActuator` + `HardwareBridge`; tests use a scripted fake.
pub(crate) trait ActuationSmokeBackend {
    /// The UUID the relay resolved at startup (NVML index 0, round-tripped).
    fn resolved_uuid(&self) -> &str;

    /// The exact argument vector the actuator would hand to `timeout` to set
    /// `limit_w`, or `None` when no target is resolved (mutation refused).
    fn planned_power_limit_command(&self, limit_w: u32) -> Option<Vec<String>>;

    /// Verify, without running `nvidia-smi`, that non-interactive sudo permits
    /// setting `limit_w` on the resolved device.
    fn check_sudo_permission(&self, limit_w: u32) -> Result<(), String>;

    /// Verify, without running `nvidia-smi`, that non-interactive sudo does
    /// **not** permit the unscoped `nvidia-smi -pl <limit_w>` form. A bare or
    /// wildcard grant would pass the scoped probe while violating the
    /// UUID-only privilege boundary. `Ok(())` only when sudo denies it.
    fn check_sudo_denies_unscoped(&self, limit_w: u32) -> Result<(), String>;

    /// The relay's actuator for the resolved device.
    fn actuator(&self) -> &dyn SafetyActuator;

    /// Acquire one real telemetry sample and classify it under the device
    /// policy. `Ok(())` only for a real (non-simulated) `Ok` classification.
    fn observe_healthy_sample(&mut self) -> Result<(), String>;
}

/// Where the smoke stopped. Everything before [`Self::Apply`] is a refusal
/// that never mutated hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SmokeStage {
    /// Caller configuration was invalid (e.g. zero release streak).
    Config,
    /// Operator UUID did not match the resolved target.
    Identity,
    /// Pre-test power limits were unreadable or not at the device default.
    PreState,
    /// The planned apply command was absent or not UUID-scoped.
    CommandPlan,
    /// `sudo -n` does not permit the apply or release command.
    SudoPermission,
    /// The actuator's apply call failed.
    Apply,
    /// The post-apply limit did not match the brake target.
    ApplyObservation,
    /// A sample in the healthy streak was not a real `Ok`.
    HealthyStreak,
    /// The actuator's release call failed.
    Release,
    /// The post-release limit did not return to the device default.
    ReleaseObservation,
}

impl SmokeStage {
    /// Stable log label.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::Identity => "identity",
            Self::PreState => "pre_state",
            Self::CommandPlan => "command_plan",
            Self::SudoPermission => "sudo_permission",
            Self::Apply => "apply",
            Self::ApplyObservation => "apply_observation",
            Self::HealthyStreak => "healthy_streak",
            Self::Release => "release",
            Self::ReleaseObservation => "release_observation",
        }
    }

    /// Whether the hardware may have been mutated before this stage failed.
    const fn after_mutation(self) -> bool {
        matches!(
            self,
            Self::Apply
                | Self::ApplyObservation
                | Self::HealthyStreak
                | Self::Release
                | Self::ReleaseObservation
        )
    }
}

/// Outcome of the one cleanup release attempted after a post-apply failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CleanupOutcome {
    /// No mutation happened, so nothing to clean up.
    NotNeeded,
    /// Release ran and the limit was observed back at the device default.
    Restored {
        /// Observed post-cleanup current limit in watts.
        current_w: u32,
    },
    /// The brake was deliberately left engaged because telemetry was not
    /// healthy; releasing would remove protection the device may need. A
    /// supervisor started afterwards adopts and releases it through normal
    /// hysteresis.
    Held {
        /// Observed current limit (the brake) in watts, if readable.
        current_w: Option<u32>,
    },
    /// Release failed or the post-cleanup limit is not at default. The
    /// operator must run the documented recovery procedure.
    Failed {
        /// Why cleanup could not be confirmed.
        reason: String,
        /// Last observed current limit, if readable.
        current_w: Option<u32>,
    },
}

/// A failed or refused smoke, with its cleanup result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SmokeFailure {
    /// Stage that failed.
    pub(crate) stage: SmokeStage,
    /// Human-readable reason.
    pub(crate) reason: String,
    /// Cleanup result (always [`CleanupOutcome::NotNeeded`] for refusals).
    pub(crate) cleanup: CleanupOutcome,
}

impl fmt::Display for SmokeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU actuation smoke FAILED at {}: {}",
            self.stage.as_str(),
            self.reason
        )?;
        match &self.cleanup {
            CleanupOutcome::NotNeeded => write!(f, " (no hardware mutation performed)"),
            CleanupOutcome::Restored { current_w } => {
                write!(f, " (cleanup restored default; current={current_w} W)")
            }
            CleanupOutcome::Held { current_w } => write!(
                f,
                " (BRAKE HELD: telemetry not healthy, limit left at {}; release only once the device is healthy)",
                fmt_w(*current_w)
            ),
            CleanupOutcome::Failed { reason, current_w } => write!(
                f,
                " (CLEANUP FAILED: {reason}; last observed current={}; run the operator recovery procedure)",
                fmt_w(*current_w)
            ),
        }
    }
}

impl std::error::Error for SmokeFailure {}

/// Successful run record. Every field was observed, not assumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SmokeReport {
    /// Validated GPU UUID.
    pub(crate) uuid: String,
    /// Pre-test current limit in watts.
    pub(crate) pre_current_w: u32,
    /// Pre-test default limit in watts.
    pub(crate) pre_default_w: u32,
    /// Brake target the apply command carried, in watts.
    pub(crate) apply_target_w: u32,
    /// Current limit observed after apply, in watts.
    pub(crate) applied_current_w: u32,
    /// Consecutive real `Ok` samples observed before release.
    pub(crate) healthy_streak: u32,
    /// Current limit observed after release, in watts.
    pub(crate) released_current_w: u32,
}

/// Operator-facing recovery command for a device left at a non-default limit.
///
/// Uses the same UUID-scoped form as the relay so a tightly scoped sudo rule
/// still matches. Printed *before* apply so an interrupted log already carries
/// it.
pub(crate) fn recovery_command(uuid: &str, default_w: u32) -> String {
    format!("sudo -n nvidia-smi -i {uuid} -pl {default_w}")
}

/// Argument vector for a read-only `sudo -l` permission check of the exact
/// power-limit command the actuator would run. `sudo -l <command>` exits 0
/// only when the rule permits it and never executes the command.
pub(crate) fn sudo_permission_check_args(uuid: &str, limit_w: u32) -> Vec<String> {
    vec![
        "-k".to_string(),
        "2".to_string(),
        "5s".to_string(),
        "sudo".to_string(),
        "-n".to_string(),
        "-l".to_string(),
        "--".to_string(),
        "nvidia-smi".to_string(),
        "-i".to_string(),
        uuid.to_string(),
        "-pl".to_string(),
        limit_w.to_string(),
    ]
}

/// Argument vector for the negative `sudo -l` probe: the *unscoped*
/// `nvidia-smi -pl <limit_w>` form, which a correctly scoped sudoers rule must
/// deny. Runs nothing either way.
pub(crate) fn sudo_unscoped_probe_args(limit_w: u32) -> Vec<String> {
    vec![
        "-k".to_string(),
        "2".to_string(),
        "5s".to_string(),
        "sudo".to_string(),
        "-n".to_string(),
        "-l".to_string(),
        "--".to_string(),
        "nvidia-smi".to_string(),
        "-pl".to_string(),
        limit_w.to_string(),
    ]
}

/// Pure check that a planned argument vector is the scoped
/// `nvidia-smi -i <uuid> -pl <limit_w>` form and nothing else sets a limit.
pub(crate) fn validate_scoped_plan(
    plan: &[String],
    uuid: &str,
    limit_w: u32,
) -> Result<(), String> {
    let pl_positions: Vec<usize> = plan
        .iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == "-pl")
        .map(|(i, _)| i)
        .collect();
    let [pl_pos] = pl_positions[..] else {
        return Err(format!(
            "planned command must set exactly one power limit; found {} `-pl` selectors",
            pl_positions.len()
        ));
    };
    let i_pos = plan
        .iter()
        .position(|a| a == "-i")
        .ok_or("planned command has no `-i <UUID>` device selector (unscoped `-pl`)")?;
    let selected = plan.get(i_pos + 1).map(String::as_str);
    let carried = plan.get(pl_pos + 1).map(String::as_str);
    let limit = limit_w.to_string();
    let problem = if i_pos >= pl_pos {
        "sets `-pl` before the `-i <UUID>` selector".to_string()
    } else if selected != Some(uuid) {
        format!("selects {selected:?}, not the validated UUID {uuid}")
    } else if carried != Some(limit.as_str()) {
        format!("carries limit {carried:?}, expected {limit_w} W")
    } else if !plan.iter().any(|a| a == "nvidia-smi") {
        "does not invoke nvidia-smi".to_string()
    } else {
        return Ok(());
    };
    Err(format!("planned command {problem}"))
}

fn within_tolerance(observed: Option<u32>, expected: u32) -> Option<u32> {
    observed.filter(|w| w.abs_diff(expected) <= BRAKE_MATCH_TOLERANCE_W)
}

fn refuse(stage: SmokeStage, reason: impl Into<String>) -> SmokeFailure {
    debug_assert!(!stage.after_mutation());
    SmokeFailure {
        stage,
        reason: reason.into(),
        cleanup: CleanupOutcome::NotNeeded,
    }
}

fn actuator_reason(err: &ActuatorError) -> String {
    format!("actuator error: {err:?}")
}

fn fmt_w(w: Option<u32>) -> String {
    w.map_or("unreadable".to_string(), |w| format!("{w} W"))
}

/// Run the apply → observe → healthy streak → release → observe sequence.
///
/// `expected_uuid` is the operator's explicit confirmation of which device may
/// be mutated. `release_ok_streak` is the configured healthy streak (the
/// supervisor's `--safety-release-ok-streak`). Progress is written to `out` as
/// it is observed so an interrupted run still leaves a usable log.
pub(crate) fn run_actuation_smoke<B: ActuationSmokeBackend>(
    backend: &mut B,
    expected_uuid: &str,
    release_ok_streak: u32,
    out: &mut dyn Write,
) -> Result<SmokeReport, SmokeFailure> {
    if release_ok_streak == 0 {
        return Err(refuse(
            SmokeStage::Config,
            "release Ok streak must be positive",
        ));
    }
    let uuid = backend.resolved_uuid().to_string();
    let _ = writeln!(out, "Resolved GPU UUID: {uuid}");
    // NVML UUIDs are hex; `is_valid_gpu_uuid` accepts either case, so compare
    // the same way rather than refusing an operator's capitalised copy.
    if !uuid.eq_ignore_ascii_case(expected_uuid) {
        return Err(refuse(
            SmokeStage::Identity,
            format!("operator confirmed {expected_uuid} but NVML index 0 resolved to {uuid}"),
        ));
    }

    let mut smoke = Smoke {
        backend,
        uuid,
        default_w: 0,
        applied_w: None,
        out,
    };
    let (pre_current_w, target_w) = smoke.check_pre_state()?;
    smoke.check_command_plans(target_w)?;
    smoke.check_sudo(target_w)?;

    let applied_current_w = smoke.apply_and_observe(target_w)?;
    smoke.healthy_streak(release_ok_streak)?;
    let released_current_w = smoke.release_and_observe()?;

    Ok(SmokeReport {
        uuid: smoke.uuid,
        pre_current_w,
        pre_default_w: smoke.default_w,
        apply_target_w: target_w,
        applied_current_w,
        healthy_streak: release_ok_streak,
        released_current_w,
    })
}

/// One in-flight smoke run: the backend plus the facts every stage shares.
struct Smoke<'a, B: ActuationSmokeBackend> {
    backend: &'a mut B,
    uuid: String,
    /// Device default in watts; set by [`Self::check_pre_state`].
    default_w: u32,
    /// Limit observed right after apply; set by [`Self::apply_and_observe`].
    /// Reused by the hold-the-brake report so that path makes no further
    /// (unbounded) NVML query.
    applied_w: Option<u32>,
    out: &'a mut dyn Write,
}

impl<B: ActuationSmokeBackend> Smoke<'_, B> {
    fn limits(&self) -> (Option<u32>, Option<u32>) {
        self.backend.actuator().query_power_limits_w()
    }

    /// Require the device at its default, owned by nobody. Returns
    /// `(pre_current_w, brake_target_w)` and records the default.
    fn check_pre_state(&mut self) -> Result<(u32, u32), SmokeFailure> {
        let (current, default) = self.limits();
        let _ = writeln!(
            self.out,
            "Pre-test power limits: current={} default={}",
            fmt_w(current),
            fmt_w(default)
        );
        let (current_w, default_w) = match classify_power_limit(current, default, BRAKE_FRACTION) {
            PowerLimitObservation::AtOrAboveDefault {
                current_w,
                default_w,
            } if current_w > 0 && default_w > 0 => (current_w, default_w),
            PowerLimitObservation::AtOrAboveDefault { .. } => {
                return Err(refuse(
                    SmokeStage::PreState,
                    "NVML reported a non-positive power limit",
                ));
            }
            PowerLimitObservation::Unreadable => {
                return Err(refuse(
                    SmokeStage::PreState,
                    "current/default power limit unreadable or brake target ambiguous",
                ));
            }
            PowerLimitObservation::RelayOwnedBrake(m) => {
                return Err(refuse(
                    SmokeStage::PreState,
                    format!(
                        "a relay-owned brake is already engaged (current={} W, default={} W); recover with `{}` before re-running",
                        m.current_w,
                        m.default_w,
                        recovery_command(&self.uuid, m.default_w)
                    ),
                ));
            }
            PowerLimitObservation::ForeignSubDefaultCap {
                current_w,
                default_w,
                ..
            } => {
                return Err(refuse(
                    SmokeStage::PreState,
                    format!(
                        "foreign sub-default power cap (current={current_w} W, default={default_w} W); refusing to take ownership"
                    ),
                ));
            }
        };
        let target_w = unambiguous_brake_target(default_w, BRAKE_FRACTION)
            .ok_or_else(|| refuse(SmokeStage::PreState, "brake target is ambiguous"))?;
        self.default_w = default_w;
        let _ = writeln!(
            self.out,
            "Apply target: {target_w} W (default x {BRAKE_FRACTION})"
        );
        let _ = writeln!(
            self.out,
            "Operator recovery if this run is interrupted: {}",
            recovery_command(&self.uuid, default_w)
        );
        Ok((current_w, target_w))
    }

    /// Both the apply and the release plan must be the validated scoped path.
    fn check_command_plans(&mut self, target_w: u32) -> Result<(), SmokeFailure> {
        for limit_w in [target_w, self.default_w] {
            let plan = self
                .backend
                .planned_power_limit_command(limit_w)
                .ok_or_else(|| refuse(SmokeStage::CommandPlan, "actuator plans no command"))?;
            validate_scoped_plan(&plan, &self.uuid, limit_w)
                .map_err(|reason| refuse(SmokeStage::CommandPlan, reason))?;
            let _ = writeln!(
                self.out,
                "Planned command ({limit_w} W): timeout {}",
                plan.join(" ")
            );
        }
        Ok(())
    }

    /// Both apply and release must already be permitted non-interactively.
    fn check_sudo(&mut self, target_w: u32) -> Result<(), SmokeFailure> {
        for limit_w in [target_w, self.default_w] {
            self.backend
                .check_sudo_permission(limit_w)
                .map_err(|reason| refuse(SmokeStage::SudoPermission, reason))?;
        }
        // A grant that also permits the unscoped form is wider than the
        // documented boundary; refuse to mutate on such a host.
        self.backend
            .check_sudo_denies_unscoped(target_w)
            .map_err(|reason| refuse(SmokeStage::SudoPermission, reason))?;
        let _ = writeln!(
            self.out,
            "sudo -n permits the UUID-scoped apply and release commands and denies the unscoped form"
        );
        Ok(())
    }

    /// Apply the brake and require the read-back within tolerance of target.
    fn apply_and_observe(&mut self, target_w: u32) -> Result<u32, SmokeFailure> {
        let _ = writeln!(self.out, "Applying relay brake...");
        if let Err(err) = self
            .backend
            .actuator()
            .apply_emergency_brake(BRAKE_FRACTION)
        {
            // The command may have partially run; observe before reporting.
            return Err(self.fail_after_mutation(SmokeStage::Apply, actuator_reason(&err)));
        }
        let (observed, _) = self.limits();
        self.applied_w = observed;
        let _ = writeln!(self.out, "Post-apply current limit: {}", fmt_w(observed));
        within_tolerance(observed, target_w).ok_or_else(|| {
            self.fail_after_mutation(
                SmokeStage::ApplyObservation,
                format!(
                    "expected {target_w} W (+/-{BRAKE_MATCH_TOLERANCE_W}), observed {}",
                    fmt_w(observed)
                ),
            )
        })
    }

    /// Require `streak` consecutive real `Ok` samples — the supervisor's
    /// release hysteresis.
    fn healthy_streak(&mut self, streak: u32) -> Result<(), SmokeFailure> {
        for n in 1..=streak {
            if let Err(reason) = self.backend.observe_healthy_sample() {
                return Err(
                    self.fail_holding_brake(format!("sample {n}/{streak} not healthy: {reason}"))
                );
            }
            let _ = writeln!(self.out, "Healthy real sample {n}/{streak}");
        }
        Ok(())
    }

    /// Release the brake and require the read-back within tolerance of default.
    fn release_and_observe(&mut self) -> Result<u32, SmokeFailure> {
        let _ = writeln!(self.out, "Releasing relay brake...");
        if let Err(err) = self.backend.actuator().release_emergency_brake() {
            return Err(self.fail_after_mutation(SmokeStage::Release, actuator_reason(&err)));
        }
        let (observed, _) = self.limits();
        let _ = writeln!(self.out, "Post-release current limit: {}", fmt_w(observed));
        within_tolerance(observed, self.default_w).ok_or_else(|| {
            self.fail_after_mutation(
                SmokeStage::ReleaseObservation,
                format!(
                    "expected default {} W (+/-{BRAKE_MATCH_TOLERANCE_W}), observed {}",
                    self.default_w,
                    fmt_w(observed)
                ),
            )
        })
    }

    /// Record a post-apply failure, attempt one verified cleanup, and name the
    /// operator recovery when the device cannot be confirmed at default.
    fn fail_after_mutation(&mut self, stage: SmokeStage, reason: String) -> SmokeFailure {
        debug_assert!(stage.after_mutation());
        let _ = writeln!(self.out, "FAILURE at {}: {reason}", stage.as_str());
        let cleanup = self.cleanup();
        if let CleanupOutcome::Failed { reason, .. } = &cleanup {
            let _ = writeln!(
                self.out,
                "CLEANUP FAILED: {reason}. Operator recovery: {}",
                recovery_command(&self.uuid, self.default_w)
            );
        }
        SmokeFailure {
            stage,
            reason,
            cleanup,
        }
    }

    /// An unhealthy sample while braked is exactly the situation the brake
    /// exists for: do **not** release. Leave the relay-owned brake engaged,
    /// report it, and let the operator (or a later supervisor, which adopts a
    /// relay-owned brake at startup) release once telemetry is healthy.
    fn fail_holding_brake(&mut self, reason: String) -> SmokeFailure {
        let _ = writeln!(
            self.out,
            "FAILURE at {}: {reason}",
            SmokeStage::HealthyStreak.as_str()
        );
        // Report the limit observed after apply rather than querying NVML
        // again: this path is reached when a driver read may already be
        // wedged, and it must not block on another unbounded call.
        let current_w = self.applied_w;
        let _ = writeln!(
            self.out,
            "BRAKE HELD: telemetry not healthy; leaving limit at {} (as observed after apply). Release only once healthy: {}",
            fmt_w(current_w),
            recovery_command(&self.uuid, self.default_w)
        );
        SmokeFailure {
            stage: SmokeStage::HealthyStreak,
            reason,
            cleanup: CleanupOutcome::Held { current_w },
        }
    }

    /// One release attempt through the production actuator, then a read-back.
    /// The actuator refuses limits it does not own, so a foreign wattage ends
    /// in [`CleanupOutcome::Failed`] by design.
    fn cleanup(&mut self) -> CleanupOutcome {
        let (before, _) = self.limits();
        if let Some(w) = within_tolerance(before, self.default_w) {
            let _ = writeln!(self.out, "Cleanup: device already at default ({w} W)");
            return CleanupOutcome::Restored { current_w: w };
        }
        let _ = writeln!(self.out, "Cleanup: attempting release...");
        if let Err(err) = self.backend.actuator().release_emergency_brake() {
            return CleanupOutcome::Failed {
                reason: actuator_reason(&err),
                current_w: before,
            };
        }
        let (after, _) = self.limits();
        match within_tolerance(after, self.default_w) {
            Some(w) => {
                let _ = writeln!(self.out, "Cleanup: restored default ({w} W)");
                CleanupOutcome::Restored { current_w: w }
            }
            None => CleanupOutcome::Failed {
                reason: format!(
                    "release ran but limit is {} not default {} W",
                    fmt_w(after),
                    self.default_w
                ),
                current_w: after,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    const UUID: &str = "GPU-12345678-1234-1234-1234-123456789abc";

    /// Scripted fake that mirrors `NvmlActuator`'s ownership rules: apply
    /// moves the limit to the brake target; release restores default only when
    /// the current limit is the relay's own brake (otherwise it refuses).
    struct ScriptedActuator {
        state: Mutex<ScriptState>,
    }

    struct ScriptState {
        current_w: Option<u32>,
        default_w: Option<u32>,
        apply_calls: u32,
        release_calls: u32,
        /// Limit the fake "device" lands on after apply (None = target).
        apply_lands_on: Option<Option<u32>>,
        /// Limit the fake "device" lands on after release (None = default).
        release_lands_on: Option<Option<u32>>,
        fail_apply: bool,
        fail_release: bool,
    }

    impl SafetyActuator for ScriptedActuator {
        fn apply_emergency_brake(&self, pct: f32) -> Result<(), ActuatorError> {
            assert_eq!(pct, BRAKE_FRACTION);
            let mut s = self.state.lock().unwrap();
            s.apply_calls += 1;
            if s.fail_apply {
                return Err(ActuatorError::CommandFailed("apply boom".into()));
            }
            let default_w = s.default_w.ok_or(ActuatorError::PowerLimitUnavailable)?;
            s.current_w = match s.apply_lands_on {
                Some(forced) => forced,
                None => Some(default_w / 2),
            };
            Ok(())
        }
        fn release_emergency_brake(&self) -> Result<(), ActuatorError> {
            let mut s = self.state.lock().unwrap();
            s.release_calls += 1;
            if s.fail_release {
                return Err(ActuatorError::CommandFailed("release boom".into()));
            }
            // Same gate as production `plan_brake_release`.
            match classify_power_limit(s.current_w, s.default_w, BRAKE_FRACTION) {
                PowerLimitObservation::RelayOwnedBrake(_) => {}
                _ => {
                    return Err(ActuatorError::CommandFailed(
                        "current power limit does not match relay brake; refusing restore".into(),
                    ));
                }
            }
            s.current_w = match s.release_lands_on {
                Some(forced) => forced,
                None => s.default_w,
            };
            Ok(())
        }
        fn query_power_limits_w(&self) -> (Option<u32>, Option<u32>) {
            let s = self.state.lock().unwrap();
            (s.current_w, s.default_w)
        }
    }

    struct FakeBackend {
        uuid: String,
        actuator: ScriptedActuator,
        plan_override: Option<Option<Vec<String>>>,
        sudo_ok: bool,
        sudo_denies_unscoped: bool,
        health: VecDeque<Result<(), String>>,
        health_calls: u32,
    }

    impl FakeBackend {
        fn healthy(current_w: u32, default_w: u32) -> Self {
            Self {
                uuid: UUID.to_string(),
                actuator: ScriptedActuator {
                    state: Mutex::new(ScriptState {
                        current_w: Some(current_w),
                        default_w: Some(default_w),
                        apply_calls: 0,
                        release_calls: 0,
                        apply_lands_on: None,
                        release_lands_on: None,
                        fail_apply: false,
                        fail_release: false,
                    }),
                },
                plan_override: None,
                sudo_ok: true,
                sudo_denies_unscoped: true,
                health: VecDeque::new(),
                health_calls: 0,
            }
        }
        fn apply_calls(&self) -> u32 {
            self.actuator.state.lock().unwrap().apply_calls
        }
        fn release_calls(&self) -> u32 {
            self.actuator.state.lock().unwrap().release_calls
        }
        fn current_w(&self) -> Option<u32> {
            self.actuator.state.lock().unwrap().current_w
        }
        fn with_state(self, f: impl FnOnce(&mut ScriptState)) -> Self {
            f(&mut self.actuator.state.lock().unwrap());
            self
        }
    }

    impl ActuationSmokeBackend for FakeBackend {
        fn resolved_uuid(&self) -> &str {
            &self.uuid
        }
        fn planned_power_limit_command(&self, limit_w: u32) -> Option<Vec<String>> {
            match &self.plan_override {
                Some(plan) => plan.clone(),
                None => Some(
                    [
                        "-k",
                        "2",
                        "5s",
                        "sudo",
                        "-n",
                        "nvidia-smi",
                        "-i",
                        UUID,
                        "-pl",
                    ]
                    .iter()
                    .map(|s| s.to_string())
                    .chain(std::iter::once(limit_w.to_string()))
                    .collect(),
                ),
            }
        }
        fn check_sudo_permission(&self, _limit_w: u32) -> Result<(), String> {
            if self.sudo_ok {
                Ok(())
            } else {
                Err("sudo -n -l refused".into())
            }
        }
        fn check_sudo_denies_unscoped(&self, _limit_w: u32) -> Result<(), String> {
            if self.sudo_denies_unscoped {
                Ok(())
            } else {
                Err("sudo -n -l permits unscoped nvidia-smi -pl".into())
            }
        }
        fn actuator(&self) -> &dyn SafetyActuator {
            &self.actuator
        }
        fn observe_healthy_sample(&mut self) -> Result<(), String> {
            self.health_calls += 1;
            self.health.pop_front().unwrap_or(Ok(()))
        }
    }

    fn run(backend: &mut FakeBackend) -> (Result<SmokeReport, SmokeFailure>, String) {
        let mut out = Vec::new();
        let result = run_actuation_smoke(backend, UUID, 3, &mut out);
        (result, String::from_utf8(out).unwrap())
    }

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    /// A refusal names its stage and never touched the actuator.
    fn assert_refused(backend: &FakeBackend, failure: &SmokeFailure, stage: SmokeStage) {
        assert_eq!(failure.stage, stage);
        assert_eq!(failure.cleanup, CleanupOutcome::NotNeeded);
        assert_eq!(
            (backend.apply_calls(), backend.release_calls()),
            (0, 0),
            "refusal must not apply or release"
        );
    }

    #[test]
    fn happy_path_applies_waits_for_streak_releases_and_records_every_observation() {
        let mut backend = FakeBackend::healthy(400, 400);
        let (result, _) = run(&mut backend);
        assert_eq!(
            result.unwrap(),
            SmokeReport {
                uuid: UUID.to_string(),
                pre_current_w: 400,
                pre_default_w: 400,
                apply_target_w: 200,
                applied_current_w: 200,
                healthy_streak: 3,
                released_current_w: 400,
            }
        );
        assert_eq!(
            (
                backend.apply_calls(),
                backend.release_calls(),
                backend.health_calls,
                backend.current_w()
            ),
            (1, 1, 3, Some(400))
        );
    }

    #[test]
    fn log_carries_recovery_before_apply_and_both_scoped_plans() {
        let mut backend = FakeBackend::healthy(400, 400);
        let (_, log) = run(&mut backend);
        let recovery_pos = log.find("Operator recovery").unwrap();
        let apply_pos = log.find("Applying relay brake").unwrap();
        assert!(recovery_pos < apply_pos);
        assert!(log.contains(&format!("-i {UUID} -pl 200")));
        assert!(log.contains(&format!("-i {UUID} -pl 400")));
    }

    #[test]
    fn identity_mismatch_refuses_before_any_read_or_mutation() {
        let mut backend = FakeBackend::healthy(400, 400);
        backend.uuid = "GPU-ffffffff-1234-1234-1234-123456789abc".to_string();
        let (result, _) = run(&mut backend);
        assert_refused(&backend, &result.unwrap_err(), SmokeStage::Identity);
    }

    #[test]
    fn leftover_relay_brake_refuses_and_names_recovery() {
        let mut backend = FakeBackend::healthy(200, 400);
        let (result, _) = run(&mut backend);
        let failure = result.unwrap_err();
        assert_refused(&backend, &failure, SmokeStage::PreState);
        assert!(failure.reason.contains(&recovery_command(UUID, 400)));
    }

    #[test]
    fn foreign_sub_default_cap_refuses() {
        let mut backend = FakeBackend::healthy(300, 400);
        let (result, _) = run(&mut backend);
        assert_refused(&backend, &result.unwrap_err(), SmokeStage::PreState);
    }

    #[test]
    fn unreadable_or_zero_limits_refuse() {
        for (current, default) in [(None, Some(400)), (Some(400), None), (None, None)] {
            let mut backend = FakeBackend::healthy(400, 400).with_state(|s| {
                s.current_w = current;
                s.default_w = default;
            });
            let (result, _) = run(&mut backend);
            assert_refused(&backend, &result.unwrap_err(), SmokeStage::PreState);
        }
    }

    #[test]
    fn tiny_default_makes_the_brake_band_ambiguous_and_refuses() {
        let mut backend = FakeBackend::healthy(8, 8);
        let (result, _) = run(&mut backend);
        assert_refused(&backend, &result.unwrap_err(), SmokeStage::PreState);
    }

    #[test]
    fn zero_streak_is_a_config_refusal() {
        let mut backend = FakeBackend::healthy(400, 400);
        let mut out = Vec::new();
        let failure = run_actuation_smoke(&mut backend, UUID, 0, &mut out).unwrap_err();
        assert_refused(&backend, &failure, SmokeStage::Config);
    }

    #[test]
    fn missing_plan_refuses() {
        let mut backend = FakeBackend::healthy(400, 400);
        backend.plan_override = Some(None);
        let (result, _) = run(&mut backend);
        assert_refused(&backend, &result.unwrap_err(), SmokeStage::CommandPlan);
    }

    #[test]
    fn unscoped_plan_refuses() {
        let mut backend = FakeBackend::healthy(400, 400);
        backend.plan_override = Some(Some(args(&[
            "-k",
            "2",
            "5s",
            "sudo",
            "-n",
            "nvidia-smi",
            "-pl",
            "200",
        ])));
        let (result, _) = run(&mut backend);
        let failure = result.unwrap_err();
        assert_refused(&backend, &failure, SmokeStage::CommandPlan);
        assert!(failure.reason.contains("unscoped"));
    }

    #[test]
    fn sudo_refusal_prevents_mutation() {
        let mut backend = FakeBackend::healthy(400, 400);
        backend.sudo_ok = false;
        let (result, _) = run(&mut backend);
        assert_refused(&backend, &result.unwrap_err(), SmokeStage::SudoPermission);
    }

    #[test]
    fn unscoped_sudo_grant_prevents_mutation() {
        // Scoped form permitted, but so is the unscoped one: too wide, refuse.
        let mut backend = FakeBackend::healthy(400, 400);
        backend.sudo_denies_unscoped = false;
        let (result, _) = run(&mut backend);
        let failure = result.unwrap_err();
        assert_refused(&backend, &failure, SmokeStage::SudoPermission);
        assert!(failure.reason.contains("unscoped"));
    }

    #[test]
    fn unscoped_probe_is_a_list_only_invocation_without_a_device_selector() {
        let args = sudo_unscoped_probe_args(200);
        assert_eq!(&args[3..7], ["sudo", "-n", "-l", "--"]);
        assert!(!args.iter().any(|a| a == "-i"));
        assert_eq!(&args[7..], ["nvidia-smi", "-pl", "200"]);
    }

    #[test]
    fn apply_error_is_verified_and_reported_without_claiming_success() {
        let mut backend = FakeBackend::healthy(400, 400).with_state(|s| s.fail_apply = true);
        let (result, _) = run(&mut backend);
        let failure = result.unwrap_err();
        assert_eq!(failure.stage, SmokeStage::Apply);
        // Device never moved, so cleanup observes default without releasing.
        assert_eq!(failure.cleanup, CleanupOutcome::Restored { current_w: 400 });
        assert_eq!(backend.release_calls(), 0);
    }

    #[test]
    fn unexpected_post_apply_limit_is_not_force_restored() {
        // 150 W is neither the brake nor the default: production refuses to
        // own it, so cleanup must report failure and defer to the operator.
        let mut backend =
            FakeBackend::healthy(400, 400).with_state(|s| s.apply_lands_on = Some(Some(150)));
        let (result, log) = run(&mut backend);
        let failure = result.unwrap_err();
        assert_eq!(failure.stage, SmokeStage::ApplyObservation);
        assert!(matches!(
            failure.cleanup,
            CleanupOutcome::Failed {
                current_w: Some(150),
                ..
            }
        ));
        assert_eq!(backend.health_calls, 0, "no streak after a bad apply");
        assert!(log.contains(&recovery_command(UUID, 400)));
    }

    #[test]
    fn unhealthy_sample_in_streak_holds_the_brake_and_fails() {
        let mut backend = FakeBackend::healthy(400, 400);
        backend.health = VecDeque::from([Ok(()), Err("Critical(temp)".to_string())]);
        let (result, log) = run(&mut backend);
        let failure = result.unwrap_err();
        assert_eq!(failure.stage, SmokeStage::HealthyStreak);
        assert!(failure.reason.contains("sample 2/3"));
        // Unhealthy while braked: protection stays on, nothing is released.
        assert_eq!(
            failure.cleanup,
            CleanupOutcome::Held {
                current_w: Some(200)
            }
        );
        assert_eq!(
            (backend.release_calls(), backend.current_w()),
            (0, Some(200))
        );
        assert!(log.contains("BRAKE HELD"));
        assert!(failure.to_string().contains("BRAKE HELD"));
    }

    #[test]
    fn operator_uuid_matches_case_insensitively() {
        let mut backend = FakeBackend::healthy(400, 400);
        let mut out = Vec::new();
        let upper = UUID.to_ascii_uppercase();
        assert!(run_actuation_smoke(&mut backend, &upper, 3, &mut out).is_ok());
    }

    #[test]
    fn release_error_reports_cleanup_failure_and_recovery() {
        let mut backend = FakeBackend::healthy(400, 400).with_state(|s| s.fail_release = true);
        let (result, log) = run(&mut backend);
        let failure = result.unwrap_err();
        assert_eq!(failure.stage, SmokeStage::Release);
        assert!(matches!(
            failure.cleanup,
            CleanupOutcome::Failed {
                current_w: Some(200),
                ..
            }
        ));
        assert!(log.contains(&recovery_command(UUID, 400)));
        assert!(failure.to_string().contains("CLEANUP FAILED"));
    }

    #[test]
    fn release_observed_at_wrong_limit_is_not_success() {
        let mut backend =
            FakeBackend::healthy(400, 400).with_state(|s| s.release_lands_on = Some(Some(350)));
        let (result, _) = run(&mut backend);
        let failure = result.unwrap_err();
        assert_eq!(failure.stage, SmokeStage::ReleaseObservation);
        // 350 W is a foreign cap now; the actuator refuses to touch it.
        assert!(matches!(
            failure.cleanup,
            CleanupOutcome::Failed {
                current_w: Some(350),
                ..
            }
        ));
    }

    #[test]
    fn tolerance_accepts_two_watt_observation_drift() {
        let mut backend = FakeBackend::healthy(400, 400).with_state(|s| {
            s.apply_lands_on = Some(Some(202));
            s.release_lands_on = Some(Some(398));
        });
        let (result, _) = run(&mut backend);
        let report = result.unwrap();
        assert_eq!(
            (report.applied_current_w, report.released_current_w),
            (202, 398)
        );
    }

    #[test]
    fn scoped_plan_validator_accepts_the_canonical_form_only_at_the_planned_limit() {
        let ok = args(&["sudo", "-n", "nvidia-smi", "-i", UUID, "-pl", "200"]);
        validate_scoped_plan(&ok, UUID, 200).unwrap();
        assert!(validate_scoped_plan(&ok, UUID, 201).is_err());
        assert!(validate_scoped_plan(&ok, "GPU-other", 200).is_err());
    }

    #[test]
    fn scoped_plan_validator_rejects_reordered_doubled_and_non_smi_plans() {
        let reordered = args(&["nvidia-smi", "-pl", "200", "-i", UUID]);
        assert!(validate_scoped_plan(&reordered, UUID, 200).is_err());
        let doubled = args(&["nvidia-smi", "-i", UUID, "-pl", "200", "-pl", "300"]);
        assert!(validate_scoped_plan(&doubled, UUID, 200).is_err());
        let no_smi = args(&["echo", "-i", UUID, "-pl", "200"]);
        assert!(validate_scoped_plan(&no_smi, UUID, 200).is_err());
    }

    #[test]
    fn sudo_check_is_a_list_only_invocation_of_the_scoped_command() {
        let args = sudo_permission_check_args(UUID, 200);
        assert_eq!(&args[3..7], ["sudo", "-n", "-l", "--"]);
        validate_scoped_plan(&args, UUID, 200).unwrap();
    }

    #[test]
    fn recovery_command_is_uuid_scoped() {
        assert_eq!(
            recovery_command(UUID, 400),
            format!("sudo -n nvidia-smi -i {UUID} -pl 400")
        );
    }
}
