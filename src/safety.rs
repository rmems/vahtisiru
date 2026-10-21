//! Isolated hardware-safety failure domain.
//!
//! Classification, hysteresis, and brake *intent* are pure functions of a
//! [`TelemetryFrame`] plus machine state. This module has no corpus-ipc,
//! Brainstem, NVML, subprocess, or async-runtime dependency. Privileged
//! power-limit actuation is performed by the `vahtisiru` executable
//! through [`SafetyActuator`]; sensory publication lives in [`crate::publish`]
//! and must never be awaited on this path.
//!
//! Named states and hysteresis are documented in the repository file
//! `docs/safety.md`. This machine does not run neural computation.

use crate::telemetry::{
    SampleValidity, TelemetryFrame, TelemetrySample, TelemetrySource, UnixMillis,
};
use std::sync::Mutex;

mod policy;
pub use policy::{PolicyError, SafetyPolicy, SafetyPolicyConfig};

/// Consecutive real [`AssessmentKind::Ok`] evaluations required before a release intent.
pub const RELEASE_OK_STREAK: u32 = 3;
/// Emergency-brake fraction of the device default power limit.
pub const BRAKE_FRACTION: f32 = 0.5;
/// Wattage tolerance when matching a leftover relay-owned brake target.
pub const BRAKE_MATCH_TOLERANCE_W: u32 = 2;
/// Thermal warning band (°C), inclusive of this threshold.
pub const TEMP_WARN_C: f32 = 75.0;
/// Thermal critical threshold (°C).
pub const TEMP_CRITICAL_C: f32 = 85.0;
#[cfg(test)]
const POWER_WARN_W: f32 = 300.0;
#[cfg(test)]
const POWER_CRITICAL_W: f32 = 350.0;

/// Instantaneous Ok / Warn / Critical used by logs and [`instant_status`].
///
/// Missing, invalid, and stale safety samples are [`Self::Critical`] (fail
/// closed). Simulated software-only frames are [`Self::Ok`] — there is no
/// real GPU to protect. Stateful relay names live in [`SafetyState`].
#[derive(Debug, Clone, PartialEq)]
pub enum SafetyStatus {
    /// Safety samples present and below warn thresholds (or software-fallback).
    Ok,
    /// Thermal or power warn band; reason is human-readable.
    Warn(String),
    /// Fail-closed or critical thermal/power; reason is human-readable.
    Critical(String),
}

/// Named relay/safety state. Observable without querying neural runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafetyState {
    /// Real NVML, valid, below warn, brake not engaged.
    HealthyReal,
    /// Real, valid, warn band. Brake is not newly applied (held if already on).
    Warning,
    /// Thermal/power critical; brake desired (and engaged once actuation succeeds).
    CriticalBraked,
    /// Brake still engaged; consecutive real Ok streak toward release.
    Recovering,
    /// Safety-critical sample missing (`NvmlUnavailable` or sensor dropout).
    TelemetryMissing,
    /// Safety-critical sample older than its stale threshold.
    TelemetryStale,
    /// Safety-critical sample non-finite or out of engineering range.
    TelemetryInvalid,
    /// `--force-software-only` (`TelemetrySource::SoftwareFallback`).
    SimulatedSoftwareOnly,
    /// Last apply/release command failed. Policy still evaluates underneath.
    ActuatorFailure,
}

impl SafetyState {
    /// All named states, in stable metric-id order.
    pub const ALL: [Self; 9] = [
        Self::HealthyReal,
        Self::Warning,
        Self::CriticalBraked,
        Self::Recovering,
        Self::TelemetryMissing,
        Self::TelemetryStale,
        Self::TelemetryInvalid,
        Self::SimulatedSoftwareOnly,
        Self::ActuatorFailure,
    ];

    /// Stable Prometheus numeric id (0…8). Do not reorder [`Self::ALL`].
    #[must_use]
    pub const fn as_id(self) -> u8 {
        match self {
            Self::HealthyReal => 0,
            Self::Warning => 1,
            Self::CriticalBraked => 2,
            Self::Recovering => 3,
            Self::TelemetryMissing => 4,
            Self::TelemetryStale => 5,
            Self::TelemetryInvalid => 6,
            Self::SimulatedSoftwareOnly => 7,
            Self::ActuatorFailure => 8,
        }
    }

    /// Prometheus `state` label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HealthyReal => "healthy_real",
            Self::Warning => "warning",
            Self::CriticalBraked => "critical_braked",
            Self::Recovering => "recovering",
            Self::TelemetryMissing => "telemetry_missing",
            Self::TelemetryStale => "telemetry_stale",
            Self::TelemetryInvalid => "telemetry_invalid",
            Self::SimulatedSoftwareOnly => "simulated_software_only",
            Self::ActuatorFailure => "actuator_failure",
        }
    }
}

/// Closed, machine-readable reason for a safety classification or event.
///
/// Every [`FrameAssessment`] and [`SafetySnapshot`] carries one. Unlike the
/// human-readable `reason` strings, this vocabulary is stable and bounded, so
/// it is safe as a Prometheus label (`safety_trips_total{reason=...}`) and as
/// a test oracle. Do not derive codes from free-text messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafetyReason {
    /// Valid real telemetry below every warn threshold.
    Healthy,
    /// `TelemetrySource::SoftwareFallback`: thresholds are not evaluated.
    Simulated,
    /// GPU temperature exceeds the critical limit.
    ThermalCritical,
    /// Board power exceeds the critical limit.
    PowerCritical,
    /// GPU temperature is in the warn band.
    ThermalWarn,
    /// Board power is in the warn band.
    PowerWarn,
    /// `gpu_temp_c` sample missing (dropout, NVML unavailable, or Valid+`None`).
    TempMissing,
    /// `power_w` sample missing.
    PowerMissing,
    /// `gpu_temp_c` non-finite, out of range, or future-dated.
    TempInvalid,
    /// `power_w` non-finite, out of range, or future-dated.
    PowerInvalid,
    /// `gpu_temp_c` older than the telemetry contract stale threshold.
    TempStale,
    /// `power_w` older than the telemetry contract stale threshold.
    PowerStale,
    /// Sample age or acquisition interval exceeds the configured policy.
    PolicyStale,
    /// No device default and no explicit power limits: cannot authorize real health.
    PowerPolicyUnavailable,
    /// Machine constructed; no frame evaluated yet (fail closed).
    NoTelemetryYet,
    /// Relay-owned brake adopted at startup.
    LeftoverBrake,
    /// Actuator reported a successful release.
    BrakeReleased,
    /// Actuator apply attempt failed.
    ApplyFailed,
    /// Actuator release attempt failed.
    ReleaseFailed,
}

impl SafetyReason {
    /// Every reason code, in stable order. Prometheus cardinality is this length.
    pub const ALL: [Self; 19] = [
        Self::Healthy,
        Self::Simulated,
        Self::ThermalCritical,
        Self::PowerCritical,
        Self::ThermalWarn,
        Self::PowerWarn,
        Self::TempMissing,
        Self::PowerMissing,
        Self::TempInvalid,
        Self::PowerInvalid,
        Self::TempStale,
        Self::PowerStale,
        Self::PolicyStale,
        Self::PowerPolicyUnavailable,
        Self::NoTelemetryYet,
        Self::LeftoverBrake,
        Self::BrakeReleased,
        Self::ApplyFailed,
        Self::ReleaseFailed,
    ];

    /// Stable snake_case label for logs and Prometheus.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Simulated => "simulated",
            Self::ThermalCritical => "thermal_critical",
            Self::PowerCritical => "power_critical",
            Self::ThermalWarn => "thermal_warn",
            Self::PowerWarn => "power_warn",
            Self::TempMissing => "temp_missing",
            Self::PowerMissing => "power_missing",
            Self::TempInvalid => "temp_invalid",
            Self::PowerInvalid => "power_invalid",
            Self::TempStale => "temp_stale",
            Self::PowerStale => "power_stale",
            Self::PolicyStale => "policy_stale",
            Self::PowerPolicyUnavailable => "power_policy_unavailable",
            Self::NoTelemetryYet => "no_telemetry_yet",
            Self::LeftoverBrake => "leftover_brake",
            Self::BrakeReleased => "brake_released",
            Self::ApplyFailed => "apply_failed",
            Self::ReleaseFailed => "release_failed",
        }
    }
}

/// Brake command the supervisor may dispatch to a [`SafetyActuator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrakeIntent {
    /// No apply/release this step.
    None,
    /// Supervisor should apply the emergency brake.
    Apply,
    /// Supervisor should release the emergency brake.
    Release,
}

/// Result of a privileged apply/release attempt. Fed back into the machine;
/// the machine never executes the command itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActuatorOutcome {
    /// Brake apply succeeded.
    Applied,
    /// Brake apply failed; the string is a log-oriented reason.
    ApplyFailed(String),
    /// Brake release succeeded.
    Released,
    /// Brake release failed; the string is a log-oriented reason.
    ReleaseFailed(String),
}

/// Stateless classification of one frame (no hysteresis, no actuator).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssessmentKind {
    /// [`crate::telemetry::TelemetrySource::SoftwareFallback`]; skips thresholds.
    Simulated,
    /// Safety-critical sample missing.
    Missing,
    /// Safety-critical sample non-finite or out of engineering range.
    Invalid,
    /// Safety-critical sample older than its stale threshold.
    Stale,
    /// Valid samples above critical thermal/power limits.
    Critical,
    /// Valid samples in the warn band.
    Warn,
    /// Valid samples below warn thresholds.
    Ok,
}

/// Frame assessment with a human-readable reason for logs/metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameAssessment {
    /// Stateless kind (no hysteresis).
    pub kind: AssessmentKind,
    /// Closed machine-readable reason code.
    pub code: SafetyReason,
    /// Human-readable reason suitable for logs (never a metric label).
    pub reason: String,
}

/// Observable snapshot after [`SafetyMachine::evaluate`] or actuator feedback.
#[derive(Debug, Clone, PartialEq)]
pub struct SafetySnapshot {
    /// Reported state (actuator-failure overlays policy when the last command failed).
    pub state: SafetyState,
    /// Policy classification ignoring actuator overlay.
    pub policy_state: SafetyState,
    /// Whether the last successful apply is still claimed.
    pub brake_engaged: bool,
    /// Whether policy currently wants the brake on.
    pub desired_brake: bool,
    /// Apply / release / none derived from desired vs engaged.
    pub intent: BrakeIntent,
    /// Consecutive real Ok evaluations while braked (release hysteresis).
    pub hysteresis_ok_count: u32,
    /// Reason from the latest classification or actuator feedback.
    pub last_reason: String,
    /// Closed reason code matching [`Self::last_reason`].
    pub reason_code: SafetyReason,
    /// Deterministic trip event: `Some(code)` on the [`SafetyMachine::evaluate`]
    /// step that opens a fail-closed episode (critical/missing/invalid/stale
    /// while no episode is open), else `None`. Never set by actuator feedback.
    pub trip: Option<SafetyReason>,
    /// Last apply/release error, if the overlay is active.
    pub last_actuator_error: Option<String>,
    /// Named-state change this step, if any.
    pub transition: Option<(SafetyState, SafetyState)>,
    /// True when this step recorded a new actuator failure.
    pub actuator_failed: bool,
}

impl SafetySnapshot {
    /// Derive [`BrakeIntent`] from desired vs currently engaged brake.
    #[must_use]
    pub fn intent_from(desired_brake: bool, brake_engaged: bool) -> BrakeIntent {
        match (desired_brake, brake_engaged) {
            (true, false) => BrakeIntent::Apply,
            (false, true) => BrakeIntent::Release,
            _ => BrakeIntent::None,
        }
    }
}

/// Deterministic safety state machine. Not coupled to IPC or GPU actuation.
///
/// ```
/// use vahtisiru::safety::{SafetyMachine, SafetyState};
/// use vahtisiru::telemetry::{assess, fixtures};
///
/// let policy = vahtisiru::safety::SafetyPolicyConfig::default()
///     .resolve(Some(400.0)).unwrap(); // Example device default: 400 W
/// let mut machine = SafetyMachine::with_policy(policy);
/// let frame = assess(&fixtures::healthy_real(), fixtures::NOW);
/// let snap = machine.evaluate(&frame);
/// assert_eq!(snap.state, SafetyState::HealthyReal);
/// ```
#[derive(Debug, Clone)]
pub struct SafetyMachine {
    policy: SafetyPolicy,
    brake_engaged: bool,
    ok_count: u32,
    just_released: bool,
    last_actuator_error: Option<String>,
    last_policy_state: SafetyState,
    state: SafetyState,
    last_reason: String,
    reason_code: SafetyReason,
    trip_latched: bool,
    apply_pending: bool,
    transitions_total: u64,
    actuator_failures_total: u64,
    trips_total: u64,
}

impl Default for SafetyMachine {
    /// Same as [`Self::new`]: fail-closed (`TelemetryMissing`) until the first frame.
    fn default() -> Self {
        Self::new()
    }
}

impl SafetyMachine {
    /// Fail-closed without a device power envelope. For real telemetry use
    /// [`Self::with_policy`]; software-only frames remain simulated.
    #[must_use]
    pub fn new() -> Self {
        Self::with_policy(
            SafetyPolicyConfig::default()
                .resolve(None)
                .expect("valid default policy"),
        )
    }

    /// Construct with a validated immutable policy. No hardware access.
    #[must_use]
    pub fn with_policy(policy: SafetyPolicy) -> Self {
        Self {
            policy,
            brake_engaged: false,
            ok_count: 0,
            just_released: false,
            last_actuator_error: None,
            last_policy_state: SafetyState::TelemetryMissing,
            state: SafetyState::TelemetryMissing,
            last_reason: "no telemetry evaluated yet".to_string(),
            reason_code: SafetyReason::NoTelemetryYet,
            trip_latched: false,
            apply_pending: false,
            transitions_total: 0,
            actuator_failures_total: 0,
            trips_total: 0,
        }
    }

    /// Consecutive named-state changes observed by this machine.
    #[must_use]
    pub fn transitions_total(&self) -> u64 {
        self.transitions_total
    }

    /// Actuator apply/release failures recorded by this machine.
    #[must_use]
    pub fn actuator_failures_total(&self) -> u64 {
        self.actuator_failures_total
    }

    /// Deterministic trip events (fail-closed episodes opened) observed.
    #[must_use]
    pub fn trips_total(&self) -> u64 {
        self.trips_total
    }

    /// Seed leftover hardware throttle detected at process start.
    pub fn seed_brake_applied(&mut self) {
        self.brake_engaged = true;
        self.ok_count = 0;
        self.just_released = false;
        self.last_policy_state = SafetyState::Recovering;
        self.state = SafetyState::Recovering;
        self.last_reason = "leftover emergency brake detected at startup".to_string();
        self.reason_code = SafetyReason::LeftoverBrake;
        // Adoption is not a new trip; release still requires the Ok streak.
        self.trip_latched = true;
    }

    /// Current snapshot without advancing hysteresis.
    #[must_use]
    pub fn snapshot(&self) -> SafetySnapshot {
        self.compose_snapshot(false, None)
    }

    /// Evaluate one telemetry frame. Does not publish, sleep, or actuate.
    #[must_use]
    pub fn evaluate(&mut self, frame: &TelemetryFrame) -> SafetySnapshot {
        let assessment = classify_frame_with_policy(frame, &self.policy);
        let just_released = self.just_released;
        self.just_released = false;

        let (policy_state, desired_brake) = self.policy_for(&assessment, just_released);
        if policy_state == SafetyState::HealthyReal && !desired_brake {
            // Healthy Ok must not keep a stale ActuatorFailure overlay forever.
            self.last_actuator_error = None;
        }
        self.last_policy_state = policy_state;
        self.last_reason = assessment.reason;
        self.reason_code = assessment.code;
        let trip = self.update_trip_latch(assessment.kind, desired_brake);
        self.compose_and_store(desired_brake, false, trip)
    }

    /// The supervisor actually dispatched an apply (worker spawned). Until
    /// [`Self::record_actuator`] reports `Applied` or `ApplyFailed`, the brake
    /// may still come on, so the open trip episode is held. An `Apply` intent
    /// that is never dispatched (for example dropped after the NVML gate was
    /// busy) does not hold the episode. Bookkeeping only; never affects brake
    /// decisions.
    pub fn apply_dispatched(&mut self) {
        self.apply_pending = true;
    }

    /// Trip bookkeeping only; never affects brake decisions.
    ///
    /// A trip is a fail-closed classification (critical, missing, invalid,
    /// stale) while no trip episode is open. The episode stays open until an
    /// evaluation finds the brake neither wanted, engaged, nor with a
    /// dispatched apply still in flight (see [`Self::apply_dispatched`]), so
    /// holding,
    /// re-applying after release, recovery, and late actuator acks (handled
    /// in [`Self::record_actuator`], which never touches the latch) are not
    /// new trips.
    fn update_trip_latch(
        &mut self,
        kind: AssessmentKind,
        desired_brake: bool,
    ) -> Option<SafetyReason> {
        let fail_closed = matches!(
            kind,
            AssessmentKind::Critical
                | AssessmentKind::Missing
                | AssessmentKind::Invalid
                | AssessmentKind::Stale
        );
        let trip = (fail_closed && !self.trip_latched).then_some(self.reason_code);
        if trip.is_some() {
            self.trip_latched = true;
            self.trips_total = self.trips_total.saturating_add(1);
        } else if self.trip_episode_can_close(desired_brake) {
            self.trip_latched = false;
        }
        trip
    }

    /// The brake is off for real: not wanted, not engaged, no apply in flight.
    fn trip_episode_can_close(&self, desired_brake: bool) -> bool {
        !desired_brake && !self.brake_engaged && !self.apply_pending
    }

    /// Record a completed actuator attempt. Does not re-read telemetry.
    #[must_use]
    pub fn record_actuator(&mut self, outcome: ActuatorOutcome) -> SafetySnapshot {
        let mut actuator_failed = false;
        match outcome {
            ActuatorOutcome::Applied => {
                self.apply_pending = false;
                self.brake_engaged = true;
                self.last_actuator_error = None;
                self.just_released = false;
            }
            ActuatorOutcome::ApplyFailed(err) => {
                self.apply_pending = false;
                self.reason_code = SafetyReason::ApplyFailed;
                self.last_reason = format!("brake apply failed: {err}");
                self.brake_engaged = false;
                self.last_actuator_error = Some(err);
                self.actuator_failures_total = self.actuator_failures_total.saturating_add(1);
                actuator_failed = true;
            }
            ActuatorOutcome::Released => {
                self.brake_engaged = false;
                self.ok_count = 0;
                self.just_released = true;
                self.last_actuator_error = None;
                // Optimistic until the supervisor re-evaluates post-release
                // telemetry. Warn/Critical on that frame re-applies.
                self.last_policy_state = SafetyState::HealthyReal;
                self.last_reason = "brake released".to_string();
                self.reason_code = SafetyReason::BrakeReleased;
            }
            ActuatorOutcome::ReleaseFailed(err) => {
                self.reason_code = SafetyReason::ReleaseFailed;
                self.last_reason = format!("brake release failed: {err}");
                self.last_actuator_error = Some(err);
                self.actuator_failures_total = self.actuator_failures_total.saturating_add(1);
                actuator_failed = true;
            }
        }
        self.compose_and_store(self.desired_from_last_policy(), actuator_failed, None)
    }

    fn policy_for(
        &mut self,
        assessment: &FrameAssessment,
        just_released: bool,
    ) -> (SafetyState, bool) {
        match assessment.kind {
            AssessmentKind::Simulated => {
                self.ok_count = 0;
                (SafetyState::SimulatedSoftwareOnly, self.brake_engaged)
            }
            AssessmentKind::Missing => self.fail_closed(SafetyState::TelemetryMissing),
            AssessmentKind::Invalid => self.fail_closed(SafetyState::TelemetryInvalid),
            AssessmentKind::Stale => self.fail_closed(SafetyState::TelemetryStale),
            AssessmentKind::Critical => self.fail_closed(SafetyState::CriticalBraked),
            AssessmentKind::Warn => {
                self.ok_count = 0;
                let desired = self.brake_engaged || just_released;
                (SafetyState::Warning, desired)
            }
            AssessmentKind::Ok if self.brake_engaged => {
                self.ok_count = self.ok_count.saturating_add(1);
                let desired = self.ok_count < self.policy.config.release_ok_streak;
                (SafetyState::Recovering, desired)
            }
            AssessmentKind::Ok => {
                self.ok_count = 0;
                (SafetyState::HealthyReal, false)
            }
        }
    }

    fn fail_closed(&mut self, state: SafetyState) -> (SafetyState, bool) {
        self.ok_count = 0;
        (state, true)
    }

    fn desired_from_last_policy(&self) -> bool {
        match self.last_policy_state {
            SafetyState::HealthyReal => false,
            SafetyState::SimulatedSoftwareOnly => self.brake_engaged,
            SafetyState::Warning => self.brake_engaged || self.just_released,
            SafetyState::Recovering => self.ok_count < self.policy.config.release_ok_streak,
            SafetyState::CriticalBraked
            | SafetyState::TelemetryMissing
            | SafetyState::TelemetryStale
            | SafetyState::TelemetryInvalid
            | SafetyState::ActuatorFailure => true,
        }
    }

    fn compose_and_store(
        &mut self,
        desired_brake: bool,
        actuator_failed: bool,
        trip: Option<SafetyReason>,
    ) -> SafetySnapshot {
        let prev = self.state;
        let reported = if self.last_actuator_error.is_some() {
            SafetyState::ActuatorFailure
        } else {
            self.last_policy_state
        };
        let transition = if prev != reported {
            self.transitions_total = self.transitions_total.saturating_add(1);
            Some((prev, reported))
        } else {
            None
        };
        self.state = reported;
        SafetySnapshot {
            state: reported,
            policy_state: self.last_policy_state,
            brake_engaged: self.brake_engaged,
            desired_brake,
            intent: SafetySnapshot::intent_from(desired_brake, self.brake_engaged),
            hysteresis_ok_count: self.ok_count,
            last_reason: self.last_reason.clone(),
            reason_code: self.reason_code,
            trip,
            last_actuator_error: self.last_actuator_error.clone(),
            transition,
            actuator_failed,
        }
    }

    fn compose_snapshot(
        &self,
        actuator_failed: bool,
        transition: Option<(SafetyState, SafetyState)>,
    ) -> SafetySnapshot {
        let desired = self.desired_from_last_policy();
        SafetySnapshot {
            state: self.state,
            policy_state: self.last_policy_state,
            brake_engaged: self.brake_engaged,
            desired_brake: desired,
            intent: SafetySnapshot::intent_from(desired, self.brake_engaged),
            hysteresis_ok_count: self.ok_count,
            last_reason: self.last_reason.clone(),
            reason_code: self.reason_code,
            trip: None,
            last_actuator_error: self.last_actuator_error.clone(),
            transition,
            actuator_failed,
        }
    }
}

/// Instantaneous status (no hysteresis). Same classification as [`classify_frame`].
#[must_use]
pub fn instant_status(frame: &TelemetryFrame) -> (SafetyStatus, bool) {
    let policy = SafetyPolicyConfig::default()
        .resolve(None)
        .expect("valid default policy");
    instant_status_with_policy(frame, &policy)
}

/// Instantaneous status using a validated operator/device envelope.
#[must_use]
pub fn instant_status_with_policy(
    frame: &TelemetryFrame,
    policy: &SafetyPolicy,
) -> (SafetyStatus, bool) {
    let assessment = classify_frame_with_policy(frame, policy);
    match assessment.kind {
        AssessmentKind::Simulated => (SafetyStatus::Ok, true),
        AssessmentKind::Missing
        | AssessmentKind::Invalid
        | AssessmentKind::Stale
        | AssessmentKind::Critical => (SafetyStatus::Critical(assessment.reason), false),
        AssessmentKind::Warn => (SafetyStatus::Warn(assessment.reason), false),
        AssessmentKind::Ok => (SafetyStatus::Ok, false),
    }
}

/// Classify without a device envelope: real telemetry fails closed.
/// Use [`classify_frame_with_policy`] to authorize healthy real readings.
#[must_use]
pub fn classify_frame(frame: &TelemetryFrame) -> FrameAssessment {
    let policy = SafetyPolicyConfig::default()
        .resolve(None)
        .expect("valid default policy");
    classify_frame_with_policy(frame, &policy)
}

/// Classify using validated operator/device policy; does not access hardware.
#[must_use]
pub fn classify_frame_with_policy(
    frame: &TelemetryFrame,
    policy: &SafetyPolicy,
) -> FrameAssessment {
    if frame.source == TelemetrySource::SoftwareFallback {
        return FrameAssessment {
            kind: AssessmentKind::Simulated,
            code: SafetyReason::Simulated,
            reason: "telemetry source is SoftwareFallback".to_string(),
        };
    }

    if let Some(fault) = worst_safety_fault(frame) {
        return fault;
    }
    let (gpu_temp_c, power_w) = match safety_readings(frame) {
        Ok(v) => v,
        Err(fault) => return fault,
    };
    if exceeds_freshness_policy(frame, policy) {
        return FrameAssessment {
            kind: AssessmentKind::Stale,
            code: SafetyReason::PolicyStale,
            reason: "sample age or acquisition interval exceeds safety policy".into(),
        };
    }
    let Some(power_limits_w) = policy.power_limits_w else {
        return FrameAssessment {
            kind: AssessmentKind::Missing,
            code: SafetyReason::PowerPolicyUnavailable,
            reason: "power policy unavailable: provide device default or explicit limits".into(),
        };
    };
    classify_thresholds(gpu_temp_c, power_w, &policy.config, power_limits_w)
}

/// Sample age or declared acquisition interval exceeds the configured policy.
fn exceeds_freshness_policy(frame: &TelemetryFrame, policy: &SafetyPolicy) -> bool {
    frame.acquisition_cadence_ms > policy.config.max_acquisition_interval_ms
        || [&frame.gpu_temp_c, &frame.power_w].iter().any(|sample| {
            // Explicit age: a future sample has no age and is never "fresh".
            // (Future samples are already rejected as invalid above.)
            frame
                .acquired_at
                .checked_sub(sample.observed_at)
                .is_none_or(|age| age >= policy.config.max_sample_age_ms)
        })
}

/// Threshold classification of valid, fresh readings. Order is normative:
/// thermal critical, power critical, thermal warn, power warn, healthy.
/// Critical is exceeded strictly; warn is inclusive.
fn classify_thresholds(
    gpu_temp_c: f32,
    power_w: f32,
    config: &SafetyPolicyConfig,
    (power_warn_w, power_critical_w): (f32, f32),
) -> FrameAssessment {
    let temp_warn_c = config.temp_warn_c;
    let temp_critical_c = config.temp_critical_c;
    let (kind, code, reason) = if gpu_temp_c > temp_critical_c {
        (
            AssessmentKind::Critical,
            SafetyReason::ThermalCritical,
            format!("GPU thermal: {gpu_temp_c:.0}°C exceeds {temp_critical_c:.0}°C"),
        )
    } else if power_w > power_critical_w {
        (
            AssessmentKind::Critical,
            SafetyReason::PowerCritical,
            format!("GPU power: {power_w:.0}W exceeds {power_critical_w:.0}W safety limit"),
        )
    } else if gpu_temp_c >= temp_warn_c {
        (
            AssessmentKind::Warn,
            SafetyReason::ThermalWarn,
            format!("GPU thermal: {gpu_temp_c:.0}°C approaching {temp_critical_c:.0}°C limit"),
        )
    } else if power_w >= power_warn_w {
        (
            AssessmentKind::Warn,
            SafetyReason::PowerWarn,
            format!("GPU power: {power_w:.0}W approaching safety limit"),
        )
    } else {
        (
            AssessmentKind::Ok,
            SafetyReason::Healthy,
            "healthy real telemetry".to_string(),
        )
    };
    FrameAssessment { kind, code, reason }
}

/// Direct warn-path helper: missing/invalid/stale fail closed as Critical.
/// Kept so the fail-closed warn regression does not skip via `Option::?`.
#[cfg(test)]
#[must_use]
fn warn_from_frame(frame: &TelemetryFrame) -> Option<SafetyStatus> {
    match classify_frame_with_policy(frame, &test_policy()).kind {
        AssessmentKind::Missing
        | AssessmentKind::Invalid
        | AssessmentKind::Stale
        | AssessmentKind::Critical => {
            let (status, _) = instant_status_with_policy(frame, &test_policy());
            Some(status)
        }
        AssessmentKind::Warn => {
            let (status, _) = instant_status_with_policy(frame, &test_policy());
            Some(status)
        }
        AssessmentKind::Ok | AssessmentKind::Simulated => None,
    }
}

/// Safety-critical telemetry channel. Typed so reason codes never depend on
/// message wording (GH#42 review).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SafetyChannel {
    GpuTemp,
    Power,
}

impl SafetyChannel {
    const fn name(self) -> &'static str {
        match self {
            Self::GpuTemp => "gpu_temp_c",
            Self::Power => "power_w",
        }
    }

    const fn signal(self) -> crate::telemetry::SignalId {
        match self {
            Self::GpuTemp => crate::telemetry::SignalId::GpuTempC,
            Self::Power => crate::telemetry::SignalId::PowerW,
        }
    }

    /// Finite, inside this channel's contract range, labelled as this channel
    /// (a relabelled sample cannot borrow another signal's range), and not
    /// observed after its frame was acquired. Zero tolerance, matching
    /// `TelemetrySample::from_raw` (`observed_at > now` is invalid).
    fn accepts(self, sample: &TelemetrySample<f32>, v: f32, acquired_at: UnixMillis) -> bool {
        let spec = crate::telemetry::signal_spec(self.signal());
        sample.signal == self.signal()
            && v.is_finite()
            && (spec.min..=spec.max).contains(&v)
            && sample.observed_at <= acquired_at
    }
}

/// Typed sample fault: which channel, and whether it is missing/invalid/stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleFault {
    Missing,
    Invalid,
    Stale,
}

impl SampleFault {
    /// Worst-fault ranking: missing > invalid > stale.
    const fn rank(self) -> u8 {
        match self {
            Self::Missing => 3,
            Self::Invalid => 2,
            Self::Stale => 1,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Invalid => "invalid",
            Self::Stale => "stale",
        }
    }

    const fn kind(self) -> AssessmentKind {
        match self {
            Self::Missing => AssessmentKind::Missing,
            Self::Invalid => AssessmentKind::Invalid,
            Self::Stale => AssessmentKind::Stale,
        }
    }

    /// Reason code from the typed (channel, fault) pair, never from text.
    const fn code(self, channel: SafetyChannel) -> SafetyReason {
        match (self, channel) {
            (Self::Missing, SafetyChannel::GpuTemp) => SafetyReason::TempMissing,
            (Self::Missing, SafetyChannel::Power) => SafetyReason::PowerMissing,
            (Self::Invalid, SafetyChannel::GpuTemp) => SafetyReason::TempInvalid,
            (Self::Invalid, SafetyChannel::Power) => SafetyReason::PowerInvalid,
            (Self::Stale, SafetyChannel::GpuTemp) => SafetyReason::TempStale,
            (Self::Stale, SafetyChannel::Power) => SafetyReason::PowerStale,
        }
    }

    fn assessment(self, channel: SafetyChannel) -> FrameAssessment {
        FrameAssessment {
            kind: self.kind(),
            code: self.code(channel),
            reason: format!("Invalid telemetry: {} {}", channel.name(), self.label()),
        }
    }
}

/// Typed fault for one safety sample, or `None` when it is valid with a value.
///
/// `Valid` is re-checked here, independent of how the sample was built:
/// fields are public and the type is deserializable, so a sample can claim
/// `Valid` with NaN, ±inf, or an out-of-range value. Such a sample would
/// otherwise slip through the threshold comparisons (NaN compares false)
/// and fail open. It is classified `Invalid` instead.
fn sample_fault(
    channel: SafetyChannel,
    sample: &TelemetrySample<f32>,
    acquired_at: UnixMillis,
) -> Option<SampleFault> {
    let rejected = |v| !channel.accepts(sample, v, acquired_at);
    match sample.validity {
        SampleValidity::Valid if sample.value.is_some_and(rejected) => Some(SampleFault::Invalid),
        SampleValidity::Valid if sample.value.is_some() => None,
        SampleValidity::Valid | SampleValidity::Missing => Some(SampleFault::Missing),
        SampleValidity::Invalid => Some(SampleFault::Invalid),
        SampleValidity::Stale => Some(SampleFault::Stale),
    }
}

fn safety_value(
    channel: SafetyChannel,
    sample: &TelemetrySample<f32>,
    acquired_at: UnixMillis,
) -> Result<f32, FrameAssessment> {
    match (sample_fault(channel, sample, acquired_at), sample.value) {
        (None, Some(v)) => Ok(v),
        (Some(fault), _) => Err(fault.assessment(channel)),
        (None, None) => Err(SampleFault::Missing.assessment(channel)),
    }
}

fn safety_readings(frame: &TelemetryFrame) -> Result<(f32, f32), FrameAssessment> {
    Ok((
        safety_value(SafetyChannel::GpuTemp, &frame.gpu_temp_c, frame.acquired_at)?,
        safety_value(SafetyChannel::Power, &frame.power_w, frame.acquired_at)?,
    ))
}

fn sample_fault_rank(
    sample: &TelemetrySample<f32>,
    channel: SafetyChannel,
    acquired_at: UnixMillis,
) -> Option<(u8, FrameAssessment)> {
    sample_fault(channel, sample, acquired_at)
        .map(|fault| (fault.rank(), fault.assessment(channel)))
}

fn worst_safety_fault(frame: &TelemetryFrame) -> Option<FrameAssessment> {
    let temp = sample_fault_rank(&frame.gpu_temp_c, SafetyChannel::GpuTemp, frame.acquired_at);
    let power = sample_fault_rank(&frame.power_w, SafetyChannel::Power, frame.acquired_at);
    match (temp, power) {
        (None, None) => None,
        (Some((_, a)), None) | (None, Some((_, a))) => Some(a),
        (Some((rt, at)), Some((rp, ap))) => {
            if rt >= rp {
                Some(at)
            } else {
                Some(ap)
            }
        }
    }
}

// ── Actuation boundary (GH#46) ──────────────────────────────────────

/// Typed, observable failure of a privileged actuation attempt.
///
/// Represented explicitly (never a bare `String` at the actuator boundary) so
/// the supervisor can surface actuator failures — the `SafetyMachine` feeds
/// these into the [`SafetyState::ActuatorFailure`] observability path (GH#42).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActuatorError {
    /// A real power limit could not be queried; refusing an arbitrary fallback.
    PowerLimitUnavailable,
    /// The actuation command failed (subprocess spawn error, timeout, missing
    /// passwordless sudo, or a non-zero exit).
    CommandFailed(String),
}

impl std::fmt::Display for ActuatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PowerLimitUnavailable => write!(
                f,
                "power limit unavailable via backend; refusing arbitrary fallback"
            ),
            Self::CommandFailed(msg) => write!(f, "actuation command failed: {msg}"),
        }
    }
}

impl std::error::Error for ActuatorError {}

/// A detected engaged brake: the current, default, and expected-brake limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrakeMatch {
    /// Current power limit in watts.
    pub current_w: u32,
    /// Device default power limit in watts.
    pub default_w: u32,
    /// Expected emergency-brake target in watts (`pct` of default).
    pub expected_w: u32,
}

/// Classification of the device power limit at process start (crash/restart).
///
/// Used to adopt a leftover **relay-owned** 50% brake without treating an
/// operator-configured sub-default cap as something the relay may auto-release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerLimitObservation {
    /// Current and/or default limit could not be read. Do not seed; first-frame
    /// evaluation fail-closes if telemetry is also missing.
    Unreadable,
    /// At or above the device default (within [`BRAKE_MATCH_TOLERANCE_W`]).
    AtOrAboveDefault {
        /// Current observed power limit in watts.
        current_w: u32,
        /// Default device power limit in watts.
        default_w: u32,
    },
    /// Current limit matches this relay's expected brake target. Adopt and
    /// recover only through the normal Ok-streak hysteresis.
    RelayOwnedBrake(BrakeMatch),
    /// Below default but not the relay target. Leave unchanged; never auto-release.
    ForeignSubDefaultCap {
        /// Current observed power limit in watts.
        current_w: u32,
        /// Default device power limit in watts.
        default_w: u32,
        /// Expected emergency brake target in watts.
        expected_brake_w: u32,
    },
}

impl PowerLimitObservation {
    /// Stable log/metric label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unreadable => "unreadable",
            Self::AtOrAboveDefault { .. } => "at_or_above_default",
            Self::RelayOwnedBrake(_) => "relay_owned_brake",
            Self::ForeignSubDefaultCap { .. } => "foreign_sub_default_cap",
        }
    }

    /// Whether startup should call [`SafetyMachine::seed_brake_applied`].
    #[must_use]
    pub const fn should_seed_leftover_brake(self) -> bool {
        matches!(self, Self::RelayOwnedBrake(_))
    }
}

// Both ±2 W bands must be disjoint; otherwise a default or foreign cap
// could be classified as a relay-owned brake. Also reject zero/invalid targets.
pub(crate) fn unambiguous_brake_target(default_w: u32, pct: f32) -> Option<u32> {
    if !pct.is_finite() || !(0.1..1.0).contains(&pct) {
        return None;
    }
    let target = (default_w as f32 * pct) as u32;
    (target > 0 && default_w.saturating_sub(target) > 2 * BRAKE_MATCH_TOLERANCE_W).then_some(target)
}

/// Classify current vs default power limits against this relay's brake fraction.
///
/// Pure function: no NVML, no mutation. Missing limits or overlapping
/// default/target tolerance bands are unreadable.
#[must_use]
pub fn classify_power_limit(
    current_w: Option<u32>,
    default_w: Option<u32>,
    brake_fraction: f32,
) -> PowerLimitObservation {
    let (Some(current_w), Some(default_w)) = (current_w, default_w) else {
        return PowerLimitObservation::Unreadable;
    };
    let Some(expected_brake_w) = unambiguous_brake_target(default_w, brake_fraction) else {
        return PowerLimitObservation::Unreadable;
    };
    if current_w.abs_diff(expected_brake_w) <= BRAKE_MATCH_TOLERANCE_W {
        return PowerLimitObservation::RelayOwnedBrake(BrakeMatch {
            current_w,
            default_w,
            expected_w: expected_brake_w,
        });
    }
    if current_w.saturating_add(BRAKE_MATCH_TOLERANCE_W) >= default_w {
        return PowerLimitObservation::AtOrAboveDefault {
            current_w,
            default_w,
        };
    }
    PowerLimitObservation::ForeignSubDefaultCap {
        current_w,
        default_w,
        expected_brake_w,
    }
}

/// The privileged hardware-safety actuation boundary.
///
/// Implementations apply/release a hardware power-limit brake and detect a
/// leftover brake at startup. The NVML/`nvidia-smi` backend is private to the
/// `vahtisiru` executable; [`FakeActuator`] provides a deterministic test
/// double. This trait is the actuation half of GH#46: the [`SafetyMachine`]
/// decides *intent* ([`BrakeIntent`]) and never actuates, while implementations
/// here perform the privileged side effect and report typed [`ActuatorError`]s.
///
/// Implementations must be `Send + Sync` so the supervisor can drive them from
/// blocking worker tasks without stalling the telemetry loop.
pub trait SafetyActuator: Send + Sync {
    /// Engage the emergency brake, throttling to `pct` of the device *default*
    /// power limit. Must fail closed with [`ActuatorError`] rather than
    /// inventing a hardcoded wattage.
    fn apply_emergency_brake(&self, pct: f32) -> Result<(), ActuatorError>;

    /// Release the emergency brake, restoring the device default power limit.
    fn release_emergency_brake(&self) -> Result<(), ActuatorError>;

    /// Current and device-default power limits in watts, if the backend can
    /// read them. Either side may be `None` (unreadable).
    fn query_power_limits_w(&self) -> (Option<u32>, Option<u32>);

    /// Detect a leftover brake matching this relay's `pct` target (e.g. after a
    /// crash/restart), so the supervisor can adopt and later release it. Returns
    /// `None` when no matching brake is present or the limits cannot be queried.
    fn detect_engaged_brake(&self, pct: f32) -> Option<BrakeMatch> {
        let (current_w, default_w) = self.query_power_limits_w();
        match classify_power_limit(current_w, default_w, pct) {
            PowerLimitObservation::RelayOwnedBrake(m) => Some(m),
            _ => None,
        }
    }
}

/// Deterministic in-memory [`SafetyActuator`] for tests.
///
/// Records apply/release call counts, tracks engaged state, and can be
/// configured to fail apply and/or release to exercise fail-closed paths.
#[derive(Debug, Default)]
pub struct FakeActuator {
    state: Mutex<FakeState>,
}

#[derive(Debug, Default, Clone)]
struct FakeState {
    engaged: bool,
    apply_calls: u32,
    release_calls: u32,
    fail_apply: Option<ActuatorError>,
    fail_release: Option<ActuatorError>,
    current_w: Option<u32>,
    default_w: Option<u32>,
}

impl FakeActuator {
    /// A fake with no brake engaged and no injected failures.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure [`SafetyActuator::apply_emergency_brake`] to fail with `err`.
    pub fn set_apply_failure(&self, err: Option<ActuatorError>) {
        self.state.lock().unwrap().fail_apply = err;
    }

    /// Configure [`SafetyActuator::release_emergency_brake`] to fail with `err`.
    pub fn set_release_failure(&self, err: Option<ActuatorError>) {
        self.state.lock().unwrap().fail_release = err;
    }

    /// Configure the current and default power limits reported by
    /// [`SafetyActuator::query_power_limits_w`].
    pub fn set_power_limits(&self, current_w: Option<u32>, default_w: Option<u32>) {
        let mut state = self.state.lock().unwrap();
        state.current_w = current_w;
        state.default_w = default_w;
    }

    /// Configure limits so [`SafetyActuator::detect_engaged_brake`] returns
    /// `detected` (a relay-owned match, or `None` for unreadable).
    pub fn set_detected_brake(&self, detected: Option<BrakeMatch>) {
        match detected {
            Some(m) => self.set_power_limits(Some(m.current_w), Some(m.default_w)),
            None => self.set_power_limits(None, None),
        }
    }

    /// Whether the fake brake is currently engaged.
    #[must_use]
    pub fn is_engaged(&self) -> bool {
        self.state.lock().unwrap().engaged
    }

    /// Number of apply calls received.
    #[must_use]
    pub fn apply_calls(&self) -> u32 {
        self.state.lock().unwrap().apply_calls
    }

    /// Number of release calls received.
    #[must_use]
    pub fn release_calls(&self) -> u32 {
        self.state.lock().unwrap().release_calls
    }
}

impl SafetyActuator for FakeActuator {
    fn apply_emergency_brake(&self, _pct: f32) -> Result<(), ActuatorError> {
        let mut state = self.state.lock().unwrap();
        state.apply_calls += 1;
        if let Some(err) = state.fail_apply.clone() {
            return Err(err);
        }
        state.engaged = true;
        Ok(())
    }

    fn release_emergency_brake(&self) -> Result<(), ActuatorError> {
        let mut state = self.state.lock().unwrap();
        state.release_calls += 1;
        if let Some(err) = state.fail_release.clone() {
            return Err(err);
        }
        state.engaged = false;
        Ok(())
    }

    fn query_power_limits_w(&self) -> (Option<u32>, Option<u32>) {
        let state = self.state.lock().unwrap();
        (state.current_w, state.default_w)
    }
}

#[cfg(test)]
mod actuator_tests {
    use super::*;

    #[test]
    fn fake_actuator_tracks_engage_release() {
        let fake = FakeActuator::new();
        assert!(!fake.is_engaged());
        fake.apply_emergency_brake(BRAKE_FRACTION).unwrap();
        assert!(fake.is_engaged());
        assert_eq!(fake.apply_calls(), 1);
        fake.release_emergency_brake().unwrap();
        assert!(!fake.is_engaged());
        assert_eq!(fake.release_calls(), 1);
    }

    #[test]
    fn fake_actuator_injects_failures() {
        let fake = FakeActuator::new();
        fake.set_apply_failure(Some(ActuatorError::PowerLimitUnavailable));
        assert_eq!(
            fake.apply_emergency_brake(BRAKE_FRACTION),
            Err(ActuatorError::PowerLimitUnavailable)
        );
        assert!(!fake.is_engaged());
        assert_eq!(fake.apply_calls(), 1);

        fake.set_apply_failure(None);
        fake.apply_emergency_brake(BRAKE_FRACTION).unwrap();
        assert!(fake.is_engaged());

        fake.set_release_failure(Some(ActuatorError::CommandFailed("boom".into())));
        assert!(fake.release_emergency_brake().is_err());
        assert!(fake.is_engaged());
    }

    #[test]
    fn fake_actuator_reports_detected_brake() {
        let fake = FakeActuator::new();
        assert_eq!(fake.detect_engaged_brake(BRAKE_FRACTION), None);
        let m = BrakeMatch {
            current_w: 150,
            default_w: 300,
            expected_w: 150,
        };
        fake.set_detected_brake(Some(m));
        assert_eq!(fake.detect_engaged_brake(BRAKE_FRACTION), Some(m));
    }

    #[test]
    fn fake_actuator_foreign_cap_is_not_a_relay_owned_brake() {
        let fake = FakeActuator::new();
        fake.set_power_limits(Some(200), Some(300));
        assert_eq!(fake.detect_engaged_brake(BRAKE_FRACTION), None);
        assert_eq!(
            classify_power_limit(Some(200), Some(300), BRAKE_FRACTION).as_str(),
            "foreign_sub_default_cap"
        );
    }

    #[test]
    fn actuator_error_display_is_stable() {
        assert!(
            ActuatorError::PowerLimitUnavailable
                .to_string()
                .contains("power limit unavailable")
        );
        assert!(
            ActuatorError::CommandFailed("x".into())
                .to_string()
                .contains("x")
        );
    }

    /// End-to-end: the pure `SafetyMachine` policy driving a `FakeActuator`,
    /// with no GPU and no subprocess.
    #[test]
    fn machine_drives_fake_actuator_critical_then_recovery() {
        use crate::telemetry::{assess, fixtures};

        fn frame(temp_c: f32, power_w: f32) -> TelemetryFrame {
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(temp_c);
            raw.power_w = Some(power_w);
            assess(&raw, fixtures::NOW)
        }

        let fake = FakeActuator::new();
        let mut machine = crate::safety::test_machine();

        // Overheat → Apply intent → drive the fake actuator.
        let snap = machine.evaluate(&frame(95.0, 200.0));
        assert_eq!(snap.intent, BrakeIntent::Apply);
        let outcome = match fake.apply_emergency_brake(BRAKE_FRACTION) {
            Ok(()) => ActuatorOutcome::Applied,
            Err(e) => ActuatorOutcome::ApplyFailed(e.to_string()),
        };
        let snap = machine.record_actuator(outcome);
        assert!(snap.brake_engaged);
        assert!(fake.is_engaged());

        // Three real Ok readings → Release intent → drive the fake actuator.
        let cool = frame(60.0, 150.0);
        let _ = machine.evaluate(&cool);
        let _ = machine.evaluate(&cool);
        let snap = machine.evaluate(&cool);
        assert_eq!(snap.intent, BrakeIntent::Release);
        let outcome = match fake.release_emergency_brake() {
            Ok(()) => ActuatorOutcome::Released,
            Err(e) => ActuatorOutcome::ReleaseFailed(e.to_string()),
        };
        let snap = machine.record_actuator(outcome);
        assert!(!snap.brake_engaged);
        assert!(!fake.is_engaged());
        assert_eq!(fake.apply_calls(), 1);
        assert_eq!(fake.release_calls(), 1);
    }
}

#[cfg(test)]
pub(crate) fn test_policy() -> SafetyPolicy {
    SafetyPolicyConfig {
        power_warn_w: Some(300.0),
        power_critical_w: Some(350.0),
        ..Default::default()
    }
    .resolve(None)
    .unwrap()
}

#[cfg(test)]
pub(crate) fn test_machine() -> SafetyMachine {
    SafetyMachine::with_policy(test_policy())
}

#[cfg(test)]
mod tests {
    fn classify_frame_with_policy_test(frame: &super::TelemetryFrame) -> super::FrameAssessment {
        super::classify_frame_with_policy(frame, &super::test_policy())
    }
    fn instant_status_test(frame: &super::TelemetryFrame) -> (super::SafetyStatus, bool) {
        super::instant_status_with_policy(frame, &super::test_policy())
    }

    use super::*;
    use crate::telemetry::{SampleValidity, assess, fixtures};

    fn nvml_temp_power(temp_c: f32, power_w: f32) -> TelemetryFrame {
        let mut raw = fixtures::healthy_real();
        raw.gpu_temp_c = Some(temp_c);
        raw.power_w = Some(power_w);
        assess(&raw, fixtures::NOW)
    }

    fn eval_once(raw: crate::telemetry::RawTelemetry) -> SafetySnapshot {
        let mut machine = crate::safety::test_machine();
        machine.evaluate(&assess(&raw, fixtures::NOW))
    }

    #[test]
    fn new_is_fail_closed_telemetry_missing() {
        let snap = crate::safety::test_machine().snapshot();
        assert_eq!(snap.state, SafetyState::TelemetryMissing);
        assert_eq!(snap.policy_state, SafetyState::TelemetryMissing);
        assert!(snap.desired_brake);
        assert!(!snap.brake_engaged);
        assert_eq!(snap.intent, BrakeIntent::Apply);
        assert_eq!(snap.last_reason, "no telemetry evaluated yet");
    }

    #[test]
    fn healthy_real_is_named_state() {
        let snap = eval_once(fixtures::healthy_real());
        assert_eq!(snap.state, SafetyState::HealthyReal);
        assert_eq!(snap.policy_state, SafetyState::HealthyReal);
        assert!(!snap.brake_engaged);
        assert!(!snap.desired_brake);
        assert_eq!(snap.intent, BrakeIntent::None);
    }

    #[test]
    fn warning_does_not_apply_brake_when_released() {
        let snap = eval_once({
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(78.0);
            raw
        });
        assert_eq!(snap.state, SafetyState::Warning);
        assert!(!snap.desired_brake);
        assert_eq!(snap.intent, BrakeIntent::None);
    }

    #[test]
    fn warn_band_includes_exact_thresholds() {
        let at_temp = eval_once({
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(TEMP_WARN_C);
            raw
        });
        assert_eq!(at_temp.state, SafetyState::Warning);

        let at_power = eval_once({
            let mut raw = fixtures::healthy_real();
            raw.power_w = Some(POWER_WARN_W);
            raw
        });
        assert_eq!(at_power.state, SafetyState::Warning);

        let just_below = eval_once({
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(74.9);
            raw.power_w = Some(299.9);
            raw
        });
        assert_eq!(just_below.state, SafetyState::HealthyReal);
    }

    #[test]
    fn critical_requests_brake() {
        let snap = eval_once({
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(90.0);
            raw
        });
        assert_eq!(snap.state, SafetyState::CriticalBraked);
        assert!(snap.desired_brake);
        assert_eq!(snap.intent, BrakeIntent::Apply);
    }

    #[test]
    fn missing_stale_invalid_and_unavailable_fail_closed() {
        let missing = eval_once(fixtures::sensor_dropout());
        assert_eq!(snap_policy(&missing), SafetyState::TelemetryMissing);
        assert_eq!(missing.intent, BrakeIntent::Apply);

        let stale = eval_once(fixtures::stale());
        assert_eq!(snap_policy(&stale), SafetyState::TelemetryStale);
        assert_eq!(stale.intent, BrakeIntent::Apply);

        let invalid = eval_once(fixtures::out_of_range());
        assert_eq!(snap_policy(&invalid), SafetyState::TelemetryInvalid);
        assert_eq!(invalid.intent, BrakeIntent::Apply);

        let unavail = eval_once(fixtures::nvml_unavailable());
        assert_eq!(snap_policy(&unavail), SafetyState::TelemetryMissing);
        assert_eq!(unavail.intent, BrakeIntent::Apply);
        assert!(unavail.last_reason.contains("missing"));
    }

    fn snap_policy(snap: &SafetySnapshot) -> SafetyState {
        snap.policy_state
    }

    #[test]
    fn simulated_skips_thresholds_and_does_not_apply() {
        let mut raw = fixtures::software_fallback();
        raw.gpu_temp_c = Some(99.0);
        raw.power_w = Some(400.0);
        let snap = eval_once(raw);
        assert_eq!(snap.state, SafetyState::SimulatedSoftwareOnly);
        assert_eq!(snap.intent, BrakeIntent::None);
        let (status, is_sim) =
            instant_status_test(&assess(&fixtures::software_fallback(), fixtures::NOW));
        assert_eq!(status, SafetyStatus::Ok);
        assert!(is_sim);
    }

    #[test]
    fn simulated_holds_existing_brake_and_resets_hysteresis() {
        let mut machine = crate::safety::test_machine();
        let critical = nvml_temp_power(90.0, 200.0);
        let snap = machine.evaluate(&critical);
        assert_eq!(snap.intent, BrakeIntent::Apply);
        let snap = machine.record_actuator(ActuatorOutcome::Applied);
        assert!(snap.brake_engaged);

        let sim = assess(&fixtures::software_fallback(), fixtures::NOW);
        let snap = machine.evaluate(&sim);
        assert_eq!(snap.state, SafetyState::SimulatedSoftwareOnly);
        assert!(snap.brake_engaged);
        assert!(snap.desired_brake);
        assert_eq!(snap.intent, BrakeIntent::None);
        assert_eq!(snap.hysteresis_ok_count, 0);
    }

    #[test]
    fn hysteresis_requires_three_real_ok_before_release() {
        let mut machine = crate::safety::test_machine();
        let _ = machine.evaluate(&nvml_temp_power(90.0, 200.0));
        let _ = machine.record_actuator(ActuatorOutcome::Applied);

        let ok = nvml_temp_power(65.0, 200.0);
        let first = machine.evaluate(&ok);
        assert_eq!(first.state, SafetyState::Recovering);
        assert_eq!(first.hysteresis_ok_count, 1);
        assert_eq!(first.intent, BrakeIntent::None);

        let second = machine.evaluate(&ok);
        assert_eq!(second.hysteresis_ok_count, 2);
        assert_eq!(second.intent, BrakeIntent::None);

        let third = machine.evaluate(&ok);
        assert_eq!(third.state, SafetyState::Recovering);
        assert_eq!(third.hysteresis_ok_count, 3);
        assert!(!third.desired_brake);
        assert_eq!(third.intent, BrakeIntent::Release);
    }

    #[test]
    fn warn_or_critical_resets_hysteresis_streak() {
        let mut machine = crate::safety::test_machine();
        let _ = machine.evaluate(&nvml_temp_power(90.0, 200.0));
        let _ = machine.record_actuator(ActuatorOutcome::Applied);
        let ok = nvml_temp_power(65.0, 200.0);
        let _ = machine.evaluate(&ok);
        let _ = machine.evaluate(&ok);
        assert_eq!(machine.snapshot().hysteresis_ok_count, 2);

        let warn = nvml_temp_power(78.0, 200.0);
        let snap = machine.evaluate(&warn);
        assert_eq!(snap.state, SafetyState::Warning);
        assert_eq!(snap.hysteresis_ok_count, 0);
        assert!(snap.brake_engaged);
        assert_eq!(snap.intent, BrakeIntent::None);
    }

    #[test]
    fn warn_after_release_reapplies_brake() {
        let mut machine = crate::safety::test_machine();
        let _ = machine.evaluate(&nvml_temp_power(90.0, 200.0));
        let _ = machine.record_actuator(ActuatorOutcome::Applied);
        let ok = nvml_temp_power(65.0, 200.0);
        let _ = machine.evaluate(&ok);
        let _ = machine.evaluate(&ok);
        let _ = machine.evaluate(&ok);
        let _ = machine.record_actuator(ActuatorOutcome::Released);
        assert!(!machine.snapshot().brake_engaged);

        let warn = nvml_temp_power(78.0, 200.0);
        let snap = machine.evaluate(&warn);
        assert_eq!(snap.state, SafetyState::Warning);
        assert!(snap.desired_brake);
        assert_eq!(snap.intent, BrakeIntent::Apply);
    }

    #[test]
    fn ok_after_release_is_healthy_real() {
        let mut machine = crate::safety::test_machine();
        let _ = machine.evaluate(&nvml_temp_power(90.0, 200.0));
        let _ = machine.record_actuator(ActuatorOutcome::Applied);
        let ok = nvml_temp_power(65.0, 200.0);
        let _ = machine.evaluate(&ok);
        let _ = machine.evaluate(&ok);
        let _ = machine.evaluate(&ok);
        let _ = machine.record_actuator(ActuatorOutcome::Released);
        let snap = machine.evaluate(&ok);
        assert_eq!(snap.state, SafetyState::HealthyReal);
        assert_eq!(snap.intent, BrakeIntent::None);
    }

    #[test]
    fn actuator_failure_is_named_state_and_does_not_stop_evaluation() {
        let mut machine = crate::safety::test_machine();
        let critical = nvml_temp_power(90.0, 200.0);
        let _ = machine.evaluate(&critical);
        let snap = machine.record_actuator(ActuatorOutcome::ApplyFailed(
            "nvidia-smi -pl failed".to_string(),
        ));
        assert_eq!(snap.state, SafetyState::ActuatorFailure);
        assert_eq!(snap.policy_state, SafetyState::CriticalBraked);
        assert!(snap.actuator_failed);
        assert_eq!(machine.actuator_failures_total(), 1);
        assert!(!snap.brake_engaged);
        assert_eq!(snap.intent, BrakeIntent::Apply);

        let still = machine.evaluate(&critical);
        assert_eq!(still.state, SafetyState::ActuatorFailure);
        assert_eq!(still.policy_state, SafetyState::CriticalBraked);
        assert_eq!(still.intent, BrakeIntent::Apply);

        let recovered = machine.record_actuator(ActuatorOutcome::Applied);
        assert!(!recovered.actuator_failed);
        let after = machine.evaluate(&critical);
        assert_eq!(after.state, SafetyState::CriticalBraked);
        assert!(after.brake_engaged);
        assert_eq!(after.intent, BrakeIntent::None);
    }

    #[test]
    fn healthy_evaluate_clears_stale_actuator_failure_overlay() {
        let mut machine = crate::safety::test_machine();
        let critical = nvml_temp_power(90.0, 200.0);
        let _ = machine.evaluate(&critical);
        let _ = machine.record_actuator(ActuatorOutcome::ApplyFailed(
            "nvidia-smi -pl failed".to_string(),
        ));
        assert_eq!(machine.snapshot().state, SafetyState::ActuatorFailure);

        let healthy = machine.evaluate(&nvml_temp_power(65.0, 200.0));
        assert_eq!(healthy.policy_state, SafetyState::HealthyReal);
        assert!(!healthy.desired_brake);
        assert_eq!(healthy.state, SafetyState::HealthyReal);
        assert_eq!(healthy.last_actuator_error, None);
        assert_eq!(healthy.intent, BrakeIntent::None);
        assert_eq!(machine.actuator_failures_total(), 1);
    }

    #[test]
    fn release_failure_keeps_brake_and_retries() {
        let mut machine = crate::safety::test_machine();
        let _ = machine.evaluate(&nvml_temp_power(90.0, 200.0));
        let _ = machine.record_actuator(ActuatorOutcome::Applied);
        let ok = nvml_temp_power(65.0, 200.0);
        let _ = machine.evaluate(&ok);
        let _ = machine.evaluate(&ok);
        let release = machine.evaluate(&ok);
        assert_eq!(release.intent, BrakeIntent::Release);

        let snap = machine.record_actuator(ActuatorOutcome::ReleaseFailed("sudo denied".into()));
        assert_eq!(snap.state, SafetyState::ActuatorFailure);
        assert!(snap.brake_engaged);
        assert_eq!(snap.intent, BrakeIntent::Release);
        assert_eq!(machine.actuator_failures_total(), 1);
    }

    #[test]
    fn leftover_brake_seed_enters_recovering() {
        let mut machine = crate::safety::test_machine();
        machine.seed_brake_applied();
        let snap = machine.evaluate(&nvml_temp_power(65.0, 200.0));
        assert_eq!(snap.state, SafetyState::Recovering);
        assert!(snap.brake_engaged);
        assert_eq!(snap.hysteresis_ok_count, 1);
    }

    #[test]
    fn leftover_brake_plus_simulated_never_releases() {
        let mut machine = crate::safety::test_machine();
        machine.seed_brake_applied();
        let sim = assess(&fixtures::software_fallback(), fixtures::NOW);
        for _ in 0..RELEASE_OK_STREAK + 2 {
            let snap = machine.evaluate(&sim);
            assert_eq!(snap.state, SafetyState::SimulatedSoftwareOnly);
            assert!(snap.brake_engaged);
            assert!(snap.desired_brake);
            assert_eq!(snap.intent, BrakeIntent::None);
            assert_eq!(snap.hysteresis_ok_count, 0);
        }
    }

    #[test]
    fn nvml_old_magic_is_healthy_real_not_simulated() {
        let snap = eval_once(fixtures::nvml_looks_like_old_magic());
        assert_eq!(snap.state, SafetyState::HealthyReal);
        let (status, is_sim) = instant_status_test(&assess(
            &fixtures::nvml_looks_like_old_magic(),
            fixtures::NOW,
        ));
        assert_eq!(status, SafetyStatus::Ok);
        assert!(!is_sim);
    }

    #[test]
    fn warn_from_frame_fail_closes_on_missing_temp_or_power() {
        let mut raw = fixtures::healthy_real();
        raw.gpu_temp_c = None;
        let frame = assess(&raw, fixtures::NOW);
        assert!(
            matches!(warn_from_frame(&frame), Some(SafetyStatus::Critical(ref msg)) if msg.contains("gpu_temp_c"))
        );

        let mut raw = fixtures::healthy_real();
        raw.power_w = None;
        let frame = assess(&raw, fixtures::NOW);
        assert!(
            matches!(warn_from_frame(&frame), Some(SafetyStatus::Critical(ref msg)) if msg.contains("power_w"))
        );

        let mut frame = nvml_temp_power(78.0, 200.0);
        frame.power_w.value = None;
        frame.power_w.validity = SampleValidity::Missing;
        assert!(
            matches!(warn_from_frame(&frame), Some(SafetyStatus::Critical(ref msg)) if msg.contains("power_w"))
        );
        let (status, is_sim) = instant_status_test(&frame);
        assert!(matches!(status, SafetyStatus::Critical(_)));
        assert!(!matches!(status, SafetyStatus::Warn(_)));
        assert!(!is_sim);

        let mut frame = nvml_temp_power(78.0, 200.0);
        frame.gpu_temp_c.value = None;
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Valid);
        assert!(
            matches!(warn_from_frame(&frame), Some(SafetyStatus::Critical(ref msg)) if msg.contains("gpu_temp_c"))
        );
    }

    #[test]
    fn mixed_faults_prefer_missing_over_stale() {
        let mut frame = nvml_temp_power(65.0, 200.0);
        frame.gpu_temp_c.validity = SampleValidity::Stale;
        frame.power_w.value = None;
        frame.power_w.validity = SampleValidity::Missing;
        let assessment = classify_frame_with_policy_test(&frame);
        assert_eq!(assessment.kind, AssessmentKind::Missing);
        assert!(assessment.reason.contains("power_w"));
    }

    /// Reason codes come from typed (channel, fault) data. This pins both the
    /// code and the exact log wording for every safety channel and fault, so
    /// rewording a message can neither silently change a code nor go unseen.
    #[test]
    fn fault_reason_codes_are_typed_not_derived_from_message_text() {
        let cases = [
            (
                SampleValidity::Missing,
                true,
                "missing",
                SafetyReason::TempMissing,
                SafetyReason::PowerMissing,
            ),
            (
                SampleValidity::Valid,
                false,
                "missing",
                SafetyReason::TempMissing,
                SafetyReason::PowerMissing,
            ),
            (
                SampleValidity::Invalid,
                true,
                "invalid",
                SafetyReason::TempInvalid,
                SafetyReason::PowerInvalid,
            ),
            (
                SampleValidity::Stale,
                true,
                "stale",
                SafetyReason::TempStale,
                SafetyReason::PowerStale,
            ),
        ];
        for (validity, has_value, label, temp_code, power_code) in cases {
            for (power, code, name) in [
                (false, temp_code, "gpu_temp_c"),
                (true, power_code, "power_w"),
            ] {
                let mut frame = nvml_temp_power(60.0, 200.0);
                let sample = if power {
                    &mut frame.power_w
                } else {
                    &mut frame.gpu_temp_c
                };
                sample.validity = validity;
                if !has_value {
                    sample.value = None;
                }
                let a = classify_frame_with_policy_test(&frame);
                assert_eq!(a.code, code, "{name} {validity:?}");
                assert_eq!(a.reason, format!("Invalid telemetry: {name} {label}"));
            }
        }
        // The typed mapping is independent of any message: the code for a
        // fault on the power channel never mentions temperature.
        for fault in [
            SampleFault::Missing,
            SampleFault::Invalid,
            SampleFault::Stale,
        ] {
            assert_ne!(
                fault.code(SafetyChannel::Power),
                fault.code(SafetyChannel::GpuTemp)
            );
            assert_eq!(fault.assessment(SafetyChannel::Power).kind, fault.kind());
        }
    }

    #[test]
    fn transitions_are_counted() {
        let mut machine = crate::safety::test_machine();
        let _ = machine.evaluate(&nvml_temp_power(65.0, 200.0));
        let before = machine.transitions_total();
        let snap = machine.evaluate(&nvml_temp_power(90.0, 200.0));
        assert_eq!(
            snap.transition,
            Some((SafetyState::HealthyReal, SafetyState::CriticalBraked))
        );
        assert_eq!(machine.transitions_total(), before + 1);
        let again = machine.evaluate(&nvml_temp_power(91.0, 200.0));
        assert_eq!(again.transition, None);
        assert_eq!(machine.transitions_total(), before + 1);
    }

    #[test]
    fn safety_state_ids_are_stable() {
        for (i, state) in SafetyState::ALL.iter().enumerate() {
            assert_eq!(state.as_id() as usize, i);
        }
        assert_eq!(SafetyState::ActuatorFailure.as_str(), "actuator_failure");
    }

    #[test]
    fn classify_power_limit_distinguishes_relay_owned_from_operator_cap() {
        assert_eq!(
            classify_power_limit(None, None, BRAKE_FRACTION),
            PowerLimitObservation::Unreadable
        );
        assert_eq!(
            classify_power_limit(Some(150), None, BRAKE_FRACTION),
            PowerLimitObservation::Unreadable
        );
        assert_eq!(
            classify_power_limit(None, Some(300), BRAKE_FRACTION),
            PowerLimitObservation::Unreadable
        );

        let owned = classify_power_limit(Some(150), Some(300), BRAKE_FRACTION);
        assert_eq!(
            owned,
            PowerLimitObservation::RelayOwnedBrake(BrakeMatch {
                current_w: 150,
                default_w: 300,
                expected_w: 150,
            })
        );
        assert!(owned.should_seed_leftover_brake());
        assert_eq!(owned.as_str(), "relay_owned_brake");

        let near = classify_power_limit(Some(149), Some(300), BRAKE_FRACTION);
        assert!(matches!(near, PowerLimitObservation::RelayOwnedBrake(_)));

        let foreign = classify_power_limit(Some(200), Some(300), BRAKE_FRACTION);
        assert_eq!(
            foreign,
            PowerLimitObservation::ForeignSubDefaultCap {
                current_w: 200,
                default_w: 300,
                expected_brake_w: 150,
            }
        );
        assert!(!foreign.should_seed_leftover_brake());

        let at_default = classify_power_limit(Some(300), Some(300), BRAKE_FRACTION);
        assert_eq!(
            at_default,
            PowerLimitObservation::AtOrAboveDefault {
                current_w: 300,
                default_w: 300,
            }
        );
        assert!(!at_default.should_seed_leftover_brake());

        let slightly_under_default = classify_power_limit(Some(298), Some(300), BRAKE_FRACTION);
        assert!(matches!(
            slightly_under_default,
            PowerLimitObservation::AtOrAboveDefault { .. }
        ));
    }

    #[test]
    fn default_machine_matches_new() {
        let snap = SafetyMachine::default().snapshot();
        assert_eq!(snap.state, SafetyState::TelemetryMissing);
        assert_eq!(snap.intent, BrakeIntent::Apply);
    }

    #[test]
    fn exact_critical_thresholds_are_exclusive() {
        let at_temp = eval_once({
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(TEMP_CRITICAL_C);
            raw
        });
        assert_eq!(at_temp.state, SafetyState::Warning);

        let over_temp = eval_once({
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(TEMP_CRITICAL_C + 0.1);
            raw
        });
        assert_eq!(over_temp.state, SafetyState::CriticalBraked);
        assert_eq!(over_temp.intent, BrakeIntent::Apply);

        let at_power = eval_once({
            let mut raw = fixtures::healthy_real();
            raw.power_w = Some(POWER_CRITICAL_W);
            raw
        });
        assert_eq!(at_power.state, SafetyState::Warning);

        let over_power = eval_once({
            let mut raw = fixtures::healthy_real();
            raw.power_w = Some(POWER_CRITICAL_W + 0.1);
            raw
        });
        assert_eq!(over_power.state, SafetyState::CriticalBraked);
    }

    #[test]
    fn power_critical_wins_over_temp_warn() {
        let snap = eval_once({
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(78.0);
            raw.power_w = Some(360.0);
            raw
        });
        assert_eq!(snap.state, SafetyState::CriticalBraked);
        assert!(snap.last_reason.contains("power"));
    }

    #[test]
    fn non_finite_is_named_telemetry_invalid() {
        let snap = eval_once(fixtures::non_finite());
        assert_eq!(snap.state, SafetyState::TelemetryInvalid);
        assert_eq!(snap.intent, BrakeIntent::Apply);
    }

    #[test]
    fn mixed_faults_rank_missing_over_invalid_over_stale() {
        let mut invalid_over_stale = nvml_temp_power(65.0, 200.0);
        invalid_over_stale.gpu_temp_c.validity = SampleValidity::Stale;
        invalid_over_stale.power_w.validity = SampleValidity::Invalid;
        let assessment = classify_frame_with_policy_test(&invalid_over_stale);
        assert_eq!(assessment.kind, AssessmentKind::Invalid);
        assert!(assessment.reason.contains("power_w"));

        let mut missing_over_invalid = nvml_temp_power(65.0, 200.0);
        missing_over_invalid.gpu_temp_c.validity = SampleValidity::Invalid;
        missing_over_invalid.power_w.value = None;
        missing_over_invalid.power_w.validity = SampleValidity::Missing;
        let assessment = classify_frame_with_policy_test(&missing_over_invalid);
        assert_eq!(assessment.kind, AssessmentKind::Missing);
        assert!(assessment.reason.contains("power_w"));
    }

    #[test]
    fn equal_fault_rank_prefers_gpu_temp() {
        let mut both_stale = nvml_temp_power(65.0, 200.0);
        both_stale.gpu_temp_c.validity = SampleValidity::Stale;
        both_stale.power_w.validity = SampleValidity::Stale;
        let assessment = classify_frame_with_policy_test(&both_stale);
        assert_eq!(assessment.kind, AssessmentKind::Stale);
        assert!(assessment.reason.contains("gpu_temp_c"));

        let mut both_invalid = nvml_temp_power(65.0, 200.0);
        both_invalid.gpu_temp_c.validity = SampleValidity::Invalid;
        both_invalid.power_w.validity = SampleValidity::Invalid;
        let assessment = classify_frame_with_policy_test(&both_invalid);
        assert_eq!(assessment.kind, AssessmentKind::Invalid);
        assert!(assessment.reason.contains("gpu_temp_c"));

        let mut both_missing = nvml_temp_power(65.0, 200.0);
        both_missing.gpu_temp_c.value = None;
        both_missing.gpu_temp_c.validity = SampleValidity::Missing;
        both_missing.power_w.value = None;
        both_missing.power_w.validity = SampleValidity::Missing;
        let assessment = classify_frame_with_policy_test(&both_missing);
        assert_eq!(assessment.kind, AssessmentKind::Missing);
        assert!(assessment.reason.contains("gpu_temp_c"));
    }

    #[test]
    fn warn_from_frame_is_none_for_ok_and_simulated() {
        assert_eq!(
            warn_from_frame(&assess(&fixtures::healthy_real(), fixtures::NOW)),
            None
        );
        assert_eq!(
            warn_from_frame(&assess(&fixtures::software_fallback(), fixtures::NOW)),
            None
        );
    }

    #[test]
    fn leftover_brake_plus_simulated_holds_and_does_not_apply() {
        let mut machine = crate::safety::test_machine();
        machine.seed_brake_applied();
        let snap = machine.evaluate(&assess(&fixtures::software_fallback(), fixtures::NOW));
        assert_eq!(snap.state, SafetyState::SimulatedSoftwareOnly);
        assert!(snap.brake_engaged);
        assert!(snap.desired_brake);
        assert_eq!(snap.intent, BrakeIntent::None);
        assert_eq!(snap.hysteresis_ok_count, 0);
    }
}
