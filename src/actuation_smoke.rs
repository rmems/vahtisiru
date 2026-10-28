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
//! `sudo -n nvidia-smi` backend (see `crate::daemon`). Normal CI never runs
//! the hardware path.
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
//! sample, release error) triggers one cleanup release attempt. Success is
//! claimed only after the post-release limit has been observed at the device
//! default; otherwise the result names the explicit operator recovery command.

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
        !matches!(
            self,
            Self::Identity | Self::PreState | Self::CommandPlan | Self::SudoPermission
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
            CleanupOutcome::Failed { reason, current_w } => write!(
                f,
                " (CLEANUP FAILED: {reason}; last observed current={}; run the operator recovery procedure)",
                current_w.map_or("unreadable".to_string(), |w| format!("{w} W"))
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
    let Some(i_pos) = plan.iter().position(|a| a == "-i") else {
        return Err("planned command has no `-i <UUID>` device selector (unscoped `-pl`)".into());
    };
    if i_pos >= pl_pos {
        return Err("planned command sets `-pl` before the `-i <UUID>` selector".into());
    }
    if plan.get(i_pos + 1).map(String::as_str) != Some(uuid) {
        return Err(format!(
            "planned command selects {:?}, not the validated UUID {uuid}",
            plan.get(i_pos + 1)
        ));
    }
    if plan.get(pl_pos + 1).map(String::as_str) != Some(limit_w.to_string().as_str()) {
        return Err(format!(
            "planned command carries limit {:?}, expected {limit_w} W",
            plan.get(pl_pos + 1)
        ));
    }
    if !plan.iter().any(|a| a == "nvidia-smi") {
        return Err("planned command does not invoke nvidia-smi".into());
    }
    Ok(())
}

fn within_tolerance(observed: u32, expected: u32) -> bool {
    observed.abs_diff(expected) <= BRAKE_MATCH_TOLERANCE_W
}

fn refuse(stage: SmokeStage, reason: impl Into<String>) -> SmokeFailure {
    SmokeFailure {
        stage,
        reason: reason.into(),
        cleanup: CleanupOutcome::NotNeeded,
    }
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
    let uuid = backend.resolved_uuid().to_string();
    let _ = writeln!(out, "Resolved GPU UUID: {uuid}");
    if uuid != expected_uuid {
        return Err(refuse(
            SmokeStage::Identity,
            format!("operator confirmed {expected_uuid} but NVML index 0 resolved to {uuid}"),
        ));
    }
    if release_ok_streak == 0 {
        return Err(refuse(
            SmokeStage::PreState,
            "release Ok streak must be positive",
        ));
    }

    // Pre-state: must be at the device default and owned by nobody.
    let (current, default) = backend.actuator().query_power_limits_w();
    let _ = writeln!(
        out,
        "Pre-test power limits: current={} default={}",
        fmt_w(current),
        fmt_w(default)
    );
    let (pre_current_w, pre_default_w) = match classify_power_limit(
        current,
        default,
        BRAKE_FRACTION,
    ) {
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
                    recovery_command(&uuid, m.default_w)
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
    let apply_target_w = unambiguous_brake_target(pre_default_w, BRAKE_FRACTION)
        .ok_or_else(|| refuse(SmokeStage::PreState, "brake target is ambiguous"))?;
    let _ = writeln!(
        out,
        "Apply target: {apply_target_w} W (default x {BRAKE_FRACTION})"
    );
    let _ = writeln!(
        out,
        "Operator recovery if this run is interrupted: {}",
        recovery_command(&uuid, pre_default_w)
    );

    // Command plan: must be the validated UUID-scoped path.
    for limit_w in [apply_target_w, pre_default_w] {
        let plan = backend
            .planned_power_limit_command(limit_w)
            .ok_or_else(|| refuse(SmokeStage::CommandPlan, "actuator plans no command"))?;
        validate_scoped_plan(&plan, &uuid, limit_w)
            .map_err(|reason| refuse(SmokeStage::CommandPlan, reason))?;
        let _ = writeln!(
            out,
            "Planned command ({limit_w} W): timeout {}",
            plan.join(" ")
        );
    }

    // Sudo: both apply and release must already be permitted non-interactively.
    for limit_w in [apply_target_w, pre_default_w] {
        backend
            .check_sudo_permission(limit_w)
            .map_err(|reason| refuse(SmokeStage::SudoPermission, reason))?;
    }
    let _ = writeln!(out, "sudo -n permits apply and release commands");

    // Apply.
    let _ = writeln!(out, "Applying relay brake...");
    if let Err(err) = backend.actuator().apply_emergency_brake(BRAKE_FRACTION) {
        // The command may have partially run; observe before reporting.
        return Err(fail_after_mutation(
            backend,
            &uuid,
            SmokeStage::Apply,
            actuator_reason(&err),
            pre_default_w,
            out,
        ));
    }
    let (applied_current, _) = backend.actuator().query_power_limits_w();
    let _ = writeln!(out, "Post-apply current limit: {}", fmt_w(applied_current));
    let applied_current_w = match applied_current {
        Some(w) if within_tolerance(w, apply_target_w) => w,
        observed => {
            return Err(fail_after_mutation(
                backend,
                &uuid,
                SmokeStage::ApplyObservation,
                format!(
                    "expected {apply_target_w} W (+/-{BRAKE_MATCH_TOLERANCE_W}), observed {}",
                    fmt_w(observed)
                ),
                pre_default_w,
                out,
            ));
        }
    };

    // Healthy streak before release — same hysteresis the supervisor uses.
    for n in 1..=release_ok_streak {
        if let Err(reason) = backend.observe_healthy_sample() {
            return Err(fail_after_mutation(
                backend,
                &uuid,
                SmokeStage::HealthyStreak,
                format!("sample {n}/{release_ok_streak} not healthy: {reason}"),
                pre_default_w,
                out,
            ));
        }
        let _ = writeln!(out, "Healthy real sample {n}/{release_ok_streak}");
    }

    // Release and observe.
    let _ = writeln!(out, "Releasing relay brake...");
    if let Err(err) = backend.actuator().release_emergency_brake() {
        return Err(fail_after_mutation(
            backend,
            &uuid,
            SmokeStage::Release,
            actuator_reason(&err),
            pre_default_w,
            out,
        ));
    }
    let (released_current, _) = backend.actuator().query_power_limits_w();
    let _ = writeln!(
        out,
        "Post-release current limit: {}",
        fmt_w(released_current)
    );
    let released_current_w = match released_current {
        Some(w) if within_tolerance(w, pre_default_w) => w,
        observed => {
            return Err(fail_after_mutation(
                backend,
                &uuid,
                SmokeStage::ReleaseObservation,
                format!(
                    "expected default {pre_default_w} W (+/-{BRAKE_MATCH_TOLERANCE_W}), observed {}",
                    fmt_w(observed)
                ),
                pre_default_w,
                out,
            ));
        }
    };

    Ok(SmokeReport {
        uuid,
        pre_current_w,
        pre_default_w,
        apply_target_w,
        applied_current_w,
        healthy_streak: release_ok_streak,
        released_current_w,
    })
}

fn actuator_reason(err: &ActuatorError) -> String {
    format!("actuator error: {err:?}")
}

fn fmt_w(w: Option<u32>) -> String {
    w.map_or("unreadable".to_string(), |w| format!("{w} W"))
}

/// Attempt one cleanup release after a post-apply failure and verify it.
fn fail_after_mutation<B: ActuationSmokeBackend>(
    backend: &mut B,
    uuid: &str,
    stage: SmokeStage,
    reason: String,
    default_w: u32,
    out: &mut dyn Write,
) -> SmokeFailure {
    debug_assert!(stage.after_mutation());
    let _ = writeln!(out, "FAILURE at {}: {reason}", stage.as_str());
    let (before, _) = backend.actuator().query_power_limits_w();
    let cleanup = match before {
        Some(w) if within_tolerance(w, default_w) => {
            let _ = writeln!(out, "Cleanup: device already at default ({w} W)");
            CleanupOutcome::Restored { current_w: w }
        }
        _ => {
            let _ = writeln!(out, "Cleanup: attempting release...");
            match backend.actuator().release_emergency_brake() {
                Ok(()) => {
                    let (after, _) = backend.actuator().query_power_limits_w();
                    match after {
                        Some(w) if within_tolerance(w, default_w) => {
                            let _ = writeln!(out, "Cleanup: restored default ({w} W)");
                            CleanupOutcome::Restored { current_w: w }
                        }
                        observed => CleanupOutcome::Failed {
                            reason: format!(
                                "release ran but limit is {} not default {default_w} W",
                                fmt_w(observed)
                            ),
                            current_w: observed,
                        },
                    }
                }
                Err(err) => CleanupOutcome::Failed {
                    reason: actuator_reason(&err),
                    current_w: before,
                },
            }
        }
    };
    if let CleanupOutcome::Failed { reason, .. } = &cleanup {
        let _ = writeln!(
            out,
            "CLEANUP FAILED: {reason}. Operator recovery: {}",
            recovery_command(uuid, default_w)
        );
    }
    SmokeFailure {
        stage,
        reason,
        cleanup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    const UUID: &str = "GPU-12345678-1234-1234-1234-123456789abc";

    /// Scripted fake: limits move on apply/release unless overridden.
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
            s.current_w = match s.apply_lands_on {
                Some(forced) => forced,
                None => Some(s.default_w.unwrap() / 2),
            };
            Ok(())
        }
        fn release_emergency_brake(&self) -> Result<(), ActuatorError> {
            let mut s = self.state.lock().unwrap();
            s.release_calls += 1;
            if s.fail_release {
                return Err(ActuatorError::CommandFailed("release boom".into()));
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

    fn assert_refused(backend: &FakeBackend, failure: &SmokeFailure, stage: SmokeStage) {
        assert_eq!(failure.stage, stage);
        assert_eq!(failure.cleanup, CleanupOutcome::NotNeeded);
        assert_eq!(backend.apply_calls(), 0, "refusal must not apply");
        assert_eq!(backend.release_calls(), 0, "refusal must not release");
    }

    #[test]
    fn happy_path_applies_waits_for_streak_releases_and_records_every_observation() {
        let mut backend = FakeBackend::healthy(400, 400);
        let (result, log) = run(&mut backend);
        let report = result.unwrap();
        assert_eq!(
            report,
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
        assert_eq!(backend.apply_calls(), 1);
        assert_eq!(backend.release_calls(), 1);
        assert_eq!(backend.health_calls, 3);
        assert_eq!(backend.current_w(), Some(400));
        // The log carries the recovery command before apply and the scoped plan.
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
        // Tiny defaults make the brake band ambiguous -> Unreadable -> refuse.
        let mut backend = FakeBackend::healthy(8, 8);
        let (result, _) = run(&mut backend);
        assert_refused(&backend, &result.unwrap_err(), SmokeStage::PreState);
    }

    #[test]
    fn zero_streak_refuses() {
        let mut backend = FakeBackend::healthy(400, 400);
        let mut out = Vec::new();
        let failure = run_actuation_smoke(&mut backend, UUID, 0, &mut out).unwrap_err();
        assert_refused(&backend, &failure, SmokeStage::PreState);
    }

    #[test]
    fn missing_or_unscoped_plan_refuses() {
        let mut backend = FakeBackend::healthy(400, 400);
        backend.plan_override = Some(None);
        let (result, _) = run(&mut backend);
        assert_refused(&backend, &result.unwrap_err(), SmokeStage::CommandPlan);

        let unscoped = ["-k", "2", "5s", "sudo", "-n", "nvidia-smi", "-pl", "200"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut backend = FakeBackend::healthy(400, 400);
        backend.plan_override = Some(Some(unscoped));
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
    fn unexpected_post_apply_limit_triggers_cleanup_release() {
        let mut backend =
            FakeBackend::healthy(400, 400).with_state(|s| s.apply_lands_on = Some(Some(150)));
        let (result, log) = run(&mut backend);
        let failure = result.unwrap_err();
        assert_eq!(failure.stage, SmokeStage::ApplyObservation);
        assert_eq!(failure.cleanup, CleanupOutcome::Restored { current_w: 400 });
        assert_eq!(backend.release_calls(), 1);
        assert_eq!(backend.health_calls, 0, "no streak after a bad apply");
        assert!(log.contains("Cleanup: restored default"));
    }

    #[test]
    fn unhealthy_sample_in_streak_releases_and_fails() {
        let mut backend = FakeBackend::healthy(400, 400);
        backend.health = VecDeque::from([Ok(()), Err("Critical(temp)".to_string())]);
        let (result, _) = run(&mut backend);
        let failure = result.unwrap_err();
        assert_eq!(failure.stage, SmokeStage::HealthyStreak);
        assert!(failure.reason.contains("sample 2/3"));
        assert_eq!(failure.cleanup, CleanupOutcome::Restored { current_w: 400 });
        assert_eq!(backend.release_calls(), 1);
        assert_eq!(backend.current_w(), Some(400));
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
        // Cleanup release lands on 350 again -> cannot confirm default.
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
        assert_eq!(report.applied_current_w, 202);
        assert_eq!(report.released_current_w, 398);
    }

    #[test]
    fn scoped_plan_validator_rejects_every_unscoped_shape() {
        let ok: Vec<String> = ["sudo", "-n", "nvidia-smi", "-i", UUID, "-pl", "200"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        validate_scoped_plan(&ok, UUID, 200).unwrap();
        assert!(validate_scoped_plan(&ok, UUID, 201).is_err());
        assert!(validate_scoped_plan(&ok, "GPU-other", 200).is_err());

        let reordered: Vec<String> = ["nvidia-smi", "-pl", "200", "-i", UUID]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(validate_scoped_plan(&reordered, UUID, 200).is_err());

        let doubled: Vec<String> = ["nvidia-smi", "-i", UUID, "-pl", "200", "-pl", "300"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(validate_scoped_plan(&doubled, UUID, 200).is_err());

        let no_smi: Vec<String> = ["echo", "-i", UUID, "-pl", "200"]
            .iter()
            .map(|s| s.to_string())
            .collect();
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
