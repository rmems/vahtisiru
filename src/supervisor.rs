//! Testable supervisor step: one tick of the safety loop with every external
//! dependency injected (actuator, publisher, telemetry acquirer, clock).
//!
//! The `vahtisiru` executable (`daemon.rs`) owns process plumbing — CLI,
//! lockfile, Prometheus bind, NVML target resolution, signals — and drives
//! [`Supervisor::step`] on its tick cadence. Tests drive the same step with
//! [`FakeActuator`](crate::safety::FakeActuator) and a scripted/hanging
//! [`TelemetryAcquirer`] so the fault table (hung acquire, apply/release
//! failure, apply-in-flight overlap, slow publisher, simulated source while
//! braked) runs without a GPU, `sudo`, or `nvidia-smi` (RM-1459).

use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task::JoinHandle;

use crate::cpu::{self, RelayMetrics};
use crate::gpu::HardwareBridge;
use crate::nvml_gate::{NvmlGate, NvmlRunError};
use crate::publish::{SensoryPublisher, evaluate_then_try_publish};
use crate::safety::{
    ActuatorError, ActuatorOutcome, BRAKE_FRACTION, BrakeIntent, PowerLimitObservation,
    SafetyActuator, SafetyMachine, SafetySnapshot, SafetyState, classify_power_limit,
};
use crate::shutdown::{
    InFlightActuation, InFlightJoin, SHUTDOWN_ACTUATOR_TIMEOUT, SHUTDOWN_METRICS_TIMEOUT,
    ShutdownActuation, ShutdownPlan, ShutdownReason, join_in_flight, plan_shutdown,
    shutdown_metrics_collector,
};
use crate::telemetry::{
    RawTelemetry, SampleClock, SampleValidity, TelemetryFrame, TelemetrySample, TelemetrySource,
    UnixMillis, assess_with_clock,
};

/// Bound on any single NVML wait (telemetry read, startup query, actuation).
pub(crate) const NVML_TIMEOUT: Duration = Duration::from_secs(2);

/// Blocking raw-telemetry source. Production is [`HardwareBridge`] (NVML);
/// tests inject scripted frames or a worker that hangs.
///
/// Calls may block indefinitely — the supervisor runs them under the
/// [`NvmlGate`] with a timeout, except for the software-only path which is
/// pure computation and runs inline.
pub(crate) trait TelemetryAcquirer: Send + Sync + 'static {
    fn acquire_raw(&self, force_software_only: bool) -> RawTelemetry;
}

impl TelemetryAcquirer for HardwareBridge {
    fn acquire_raw(&self, force_software_only: bool) -> RawTelemetry {
        HardwareBridge::acquire_raw(self, force_software_only)
    }
}

/// Wall-clock source for frame receive timestamps (injectable in tests).
pub(crate) type Clock = Arc<dyn Fn() -> UnixMillis + Send + Sync>;

/// Static supervisor settings derived from the CLI.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SupervisorConfig {
    pub force_software_only: bool,
    /// Declared acquisition cadence (ms); drives stale-after thresholds.
    pub cadence_ms: u64,
    /// Bound on each gated NVML wait.
    pub nvml_timeout: Duration,
    /// Print the live dashboard line to stdout each step.
    pub dashboard: bool,
}

/// Injected collaborators for a [`Supervisor`].
pub(crate) struct SupervisorParts {
    pub machine: SafetyMachine,
    pub sample_clock: SampleClock,
    pub publisher: Box<dyn SensoryPublisher>,
    pub actuator: Arc<dyn SafetyActuator>,
    pub acquirer: Arc<dyn TelemetryAcquirer>,
    pub gate: NvmlGate,
    pub clock: Clock,
    pub relay_metrics: Arc<Mutex<RelayMetrics>>,
}

/// What one [`Supervisor::step`] observed and decided.
#[derive(Debug, Clone)]
pub(crate) struct StepReport {
    pub step: u64,
    pub source: TelemetrySource,
    /// Machine snapshot after this step's evaluation and actuator feedback.
    pub snapshot: SafetySnapshot,
    /// Actuator outcome joined and recorded during this step, if any.
    pub actuation: Option<ActuatorOutcome>,
    /// Privileged actuation still running when the step returned.
    pub in_flight: InFlightActuation,
    /// Whether this step spawned a new actuation worker.
    pub dispatched: bool,
    /// Whether an actuation intent is parked because the NVML gate was busy.
    pub pending_intent: Option<BrakeIntent>,
}

/// Handle to an in-flight privileged actuation attempt.
type ActuationTask = JoinHandle<Result<(), ActuatorError>>;

/// One safety-loop supervisor with all hardware and IPC behind traits.
pub(crate) struct Supervisor {
    config: SupervisorConfig,
    machine: SafetyMachine,
    sample_clock: SampleClock,
    publisher: Box<dyn SensoryPublisher>,
    actuator: Arc<dyn SafetyActuator>,
    acquirer: Arc<dyn TelemetryAcquirer>,
    gate: NvmlGate,
    clock: Clock,
    relay_metrics: Arc<Mutex<RelayMetrics>>,
    warned_brake_held_sim: bool,
    warned_nvml_busy: bool,
    brake_task: Option<ActuationTask>,
    release_task: Option<ActuationTask>,
    pending_intent: Option<BrakeIntent>,
    step_count: u64,
}

impl Supervisor {
    pub(crate) fn new(config: SupervisorConfig, parts: SupervisorParts) -> Self {
        Self {
            config,
            machine: parts.machine,
            sample_clock: parts.sample_clock,
            publisher: parts.publisher,
            actuator: parts.actuator,
            acquirer: parts.acquirer,
            gate: parts.gate,
            clock: parts.clock,
            relay_metrics: parts.relay_metrics,
            warned_brake_held_sim: false,
            warned_nvml_busy: false,
            brake_task: None,
            release_task: None,
            pending_intent: None,
            step_count: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> SafetySnapshot {
        self.machine.snapshot()
    }

    #[cfg(test)]
    pub(crate) fn machine(&self) -> &SafetyMachine {
        &self.machine
    }

    pub(crate) fn in_flight(&self) -> InFlightActuation {
        InFlightActuation::from_tasks(self.brake_task.is_some(), self.release_task.is_some())
    }

    /// Seed only a power cap that unambiguously matches this relay's brake.
    /// The read-only startup query runs in software-only mode too, so simulated
    /// telemetry cannot hide or authorize release of a real persistent brake.
    pub(crate) fn seed_startup_brake(&mut self, current_w: Option<u32>, default_w: Option<u32>) {
        match classify_power_limit(current_w, default_w, BRAKE_FRACTION) {
            PowerLimitObservation::RelayOwnedBrake(m) => {
                eprintln!(
                    "[relay] WARNING: GPU power limit {}W matches expected emergency brake \
                     target {}W (default {}W); will auto-release after Ok streak",
                    m.current_w, m.expected_w, m.default_w
                );
                self.machine.seed_brake_applied();
                let seed = self.machine.snapshot();
                let mut metrics = self.relay_metrics.lock().unwrap();
                cpu::record_safety_snapshot(&mut metrics, &seed);
            }
            PowerLimitObservation::ForeignSubDefaultCap {
                current_w,
                default_w,
                expected_brake_w,
            } => {
                eprintln!(
                    "[relay] GPU power limit {current_w}W is below default {default_w}W but does not \
                     match relay brake target {expected_brake_w}W; leaving operator/device cap unchanged"
                );
            }
            PowerLimitObservation::Unreadable => {
                tracing::info!(
                    "power limits unreadable at startup; first-frame eval will fail-closed if needed"
                );
            }
            PowerLimitObservation::AtOrAboveDefault { .. } => {}
        }
    }

    /// Run one tick: reconcile finished actuations, acquire + assess telemetry,
    /// evaluate on cadence, publish best-effort, and dispatch any intent.
    ///
    /// Never awaits the actuator or the publisher; the only awaits are the
    /// gate-bounded telemetry read and joining a worker that has already
    /// finished. Sleeping between ticks is the caller's job.
    pub(crate) async fn step(&mut self) -> StepReport {
        self.step_count += 1;
        let step_count = self.step_count;
        let mut dispatched = false;

        // A previously blocked emergency brake gets first claim on a gate
        // that became free during the sleep. Otherwise the next read can
        // repeatedly reacquire it and starve actuation.
        if self.pending_intent == Some(BrakeIntent::Apply) {
            dispatched |= self.dispatch_due_or_pending(false);
        }

        let raw = self.acquire_raw_with_timeout().await;
        let telemetry = self.assess(&raw);

        // Reconcile workers that finished during the sleep. A success already
        // re-read and re-evaluated telemetry; a failure is recorded as-is.
        // At most one worker is ever in flight, so at most one slot settles.
        let mut actuation = None;
        for kind in [ActuationKind::Apply, ActuationKind::Release] {
            actuation = self.reconcile_finished(kind).await.or(actuation);
        }
        let succeeded = actuation.as_ref().is_some_and(outcome_succeeded);
        let failed = actuation.is_some() && !succeeded;

        let dispatch_due = succeeded || self.evaluate_or_publish(step_count, &telemetry, failed);
        dispatched |= self.dispatch_due_or_pending(dispatch_due);

        {
            let mut metrics = self.relay_metrics.lock().unwrap();
            metrics.telemetry_acquired_at = Some(telemetry.acquired_at);
        }

        let snapshot = self.machine.snapshot();
        if self.config.dashboard {
            print_dashboard(&telemetry, step_count, snapshot.state);
        }

        StepReport {
            step: step_count,
            source: telemetry.source,
            snapshot,
            actuation,
            in_flight: self.in_flight(),
            dispatched,
            pending_intent: self.pending_intent,
        }
    }

    /// Join the worker in `kind`'s slot if it has finished. On success the
    /// outcome is recorded and telemetry is re-read and re-evaluated before
    /// any further dispatch; on failure only the outcome is recorded.
    async fn reconcile_finished(&mut self, kind: ActuationKind) -> Option<ActuatorOutcome> {
        let slot = match kind {
            ActuationKind::Apply => &mut self.brake_task,
            ActuationKind::Release => &mut self.release_task,
        };
        if !slot.as_ref().is_some_and(JoinHandle::is_finished) {
            return None;
        }
        let task = slot.take().expect("finished task exists");
        let outcome = actuation_outcome(kind, task.await);
        if outcome_succeeded(&outcome) {
            self.reassess_after_actuation(outcome.clone()).await;
        } else {
            let snap = self.machine.record_actuator(outcome.clone());
            self.store_safety(&snap);
        }
        Some(outcome)
    }

    /// Evaluate the frame on the safety cadence, otherwise only publish it.
    /// Returns whether the resulting intent is due for dispatch.
    ///
    /// The first acquired frame is evaluated immediately (fail-closed
    /// startup); later evaluations keep the every-10-ticks cadence. Never
    /// spawn from the pre-telemetry snapshot: SoftwareFallback holds rather
    /// than applies, which only `classify_frame` can decide.
    fn evaluate_or_publish(
        &mut self,
        step_count: u64,
        telemetry: &TelemetryFrame,
        actuation_failed_this_tick: bool,
    ) -> bool {
        if !should_evaluate_tick(step_count, telemetry.source) {
            // Publication is outside the safety critical path and never awaited.
            // Same mapping-time basis as evaluate_then_try_publish.
            let _ = self.publisher.try_publish(&telemetry.to_sensory_mapping());
            return false;
        }
        let prior_intent = self.machine.snapshot().intent;
        let (snap, pub_res) =
            evaluate_then_try_publish(&mut self.machine, telemetry, self.publisher.as_ref());
        let _ = pub_res;
        self.store_safety(&snap);
        should_dispatch_intent(
            step_count,
            prior_intent,
            snap.intent,
            actuation_failed_this_tick,
        )
    }

    fn assess(&mut self, raw: &RawTelemetry) -> TelemetryFrame {
        assess_with_clock(
            raw,
            (self.clock)(),
            self.config.cadence_ms,
            &mut self.sample_clock,
        )
    }

    /// Bounded single-flight acquisition for both per-tick and post-actuation reads.
    async fn acquire_raw_with_timeout(&mut self) -> RawTelemetry {
        if self.config.force_software_only {
            return self.acquirer.acquire_raw(true);
        }
        let acquirer = Arc::clone(&self.acquirer);
        let result = self
            .gate
            .run(self.config.nvml_timeout, move || {
                acquirer.acquire_raw(false)
            })
            .await;
        if should_log_acquisition_error(&mut self.warned_nvml_busy, result.as_ref().err()) {
            tracing::warn!(error = ?result.as_ref().err(), "NVML acquisition unavailable; treating as unavailable");
        }
        match result {
            Ok(raw) => raw,
            Err(_) => RawTelemetry::nvml_unavailable((self.clock)()),
        }
    }

    /// Reassess telemetry after either successful hardware mutation before
    /// dispatching any further safety intent.
    async fn reassess_after_actuation(&mut self, outcome: ActuatorOutcome) {
        let snap = self.machine.record_actuator(outcome);
        self.store_safety(&snap);
        let raw = self.acquire_raw_with_timeout().await;
        let frame = self.assess(&raw);
        let (snap, pub_res) =
            evaluate_then_try_publish(&mut self.machine, &frame, self.publisher.as_ref());
        let _ = pub_res;
        self.store_safety(&snap);
    }

    fn store_safety(&mut self, snap: &SafetySnapshot) {
        log_safety_snapshot(snap, &mut self.warned_brake_held_sim);
        let mut metrics = self.relay_metrics.lock().unwrap();
        cpu::record_safety_snapshot(&mut metrics, snap);
    }

    /// Retry only intents skipped because another NVML worker held the gate.
    /// Reconcile with the latest safety snapshot before any retry.
    /// Returns whether a worker was spawned.
    fn dispatch_due_or_pending(&mut self, due: bool) -> bool {
        let snap = self.machine.snapshot();
        if self.pending_intent != Some(snap.intent) {
            self.pending_intent = None;
        }
        if !due && self.pending_intent.is_none() {
            return false;
        }
        let result = spawn_intent(
            &snap,
            &self.actuator,
            &self.gate,
            (&mut self.brake_task, &mut self.release_task),
        );
        self.pending_intent = match result {
            DispatchResult::GateBusy => Some(snap.intent),
            DispatchResult::Spawned
            | DispatchResult::AlreadyInFlight
            | DispatchResult::NoIntent => None,
        };
        result == DispatchResult::Spawned
    }

    /// Fail-closed orderly shutdown: never issues a new power-limit restore;
    /// awaits (bounded) only an actuation that was already in flight.
    pub(crate) async fn shutdown(
        &mut self,
        reason: ShutdownReason,
        metrics_shutdown: &tokio::sync::watch::Sender<bool>,
        metrics_task: JoinHandle<()>,
    ) -> ShutdownPlan {
        let snap = self.machine.snapshot();
        let in_flight = self.in_flight();
        let plan = plan_shutdown(reason, &snap, in_flight);
        log_shutdown_plan(&snap, in_flight, &plan);
        {
            let mut metrics = self.relay_metrics.lock().unwrap();
            cpu::record_shutdown(&mut metrics, &plan);
        }

        // Await (bounded) only the actuation the plan authorizes; abort the
        // other slot. Idle aborts both and never starts a new actuation.
        let awaited = match plan.actuation {
            ShutdownActuation::Idle => None,
            ShutdownActuation::AwaitApply => Some(ActuationKind::Apply),
            ShutdownActuation::AwaitAuthorizedRelease => Some(ActuationKind::Release),
        };
        for kind in [ActuationKind::Apply, ActuationKind::Release] {
            self.settle_slot(kind, awaited == Some(kind)).await;
        }

        shutdown_metrics_collector(metrics_shutdown, metrics_task, SHUTDOWN_METRICS_TIMEOUT).await;
        plan
    }

    /// Empty one worker slot on shutdown: join it (bounded) when authorized,
    /// otherwise abort it.
    async fn settle_slot(&mut self, kind: ActuationKind, authorized: bool) {
        let slot = match kind {
            ActuationKind::Apply => &mut self.brake_task,
            ActuationKind::Release => &mut self.release_task,
        };
        match (slot.take(), authorized) {
            (None, _) => {}
            (Some(task), true) => self.await_in_flight(kind, task).await,
            (Some(task), false) => task.abort(),
        }
    }

    /// Bounded join of an already-running actuation during shutdown; the
    /// outcome is recorded, a timeout leaves hardware and machine unchanged.
    async fn await_in_flight(&mut self, kind: ActuationKind, task: ActuationTask) {
        let label = kind.label();
        match join_in_flight(task, SHUTDOWN_ACTUATOR_TIMEOUT).await {
            InFlightJoin::Succeeded => {
                let snap = self.machine.record_actuator(kind.success());
                self.store_safety(&snap);
            }
            InFlightJoin::Failed(e) | InFlightJoin::Panicked(e) => {
                eprintln!("[relay] shutdown: in-flight {label} did not succeed: {e}");
                let snap = self.machine.record_actuator(kind.failure(e));
                self.store_safety(&snap);
            }
            InFlightJoin::TimedOut => {
                let consequence = match kind {
                    ActuationKind::Apply => "leaving hardware unchanged",
                    ActuationKind::Release => "leaving brake engaged",
                };
                eprintln!("[relay] shutdown: in-flight {label} timed out; {consequence}");
            }
        }
    }
}

/// Which privileged actuation a worker slot carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActuationKind {
    Apply,
    Release,
}

impl ActuationKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Apply => "apply",
            Self::Release => "release",
        }
    }

    fn success(self) -> ActuatorOutcome {
        match self {
            Self::Apply => ActuatorOutcome::Applied,
            Self::Release => ActuatorOutcome::Released,
        }
    }

    fn failure(self, reason: String) -> ActuatorOutcome {
        match self {
            Self::Apply => ActuatorOutcome::ApplyFailed(reason),
            Self::Release => ActuatorOutcome::ReleaseFailed(reason),
        }
    }
}

fn outcome_succeeded(outcome: &ActuatorOutcome) -> bool {
    matches!(
        outcome,
        ActuatorOutcome::Applied | ActuatorOutcome::Released
    )
}

/// Map a joined worker result onto the machine's outcome vocabulary. A
/// panicking worker is an actuation failure, never a supervisor crash.
fn actuation_outcome(
    kind: ActuationKind,
    joined: Result<Result<(), ActuatorError>, tokio::task::JoinError>,
) -> ActuatorOutcome {
    match joined {
        Ok(Ok(())) => kind.success(),
        Ok(Err(e)) => {
            eprintln!("[relay] Emergency brake {} failed: {e}", kind.label());
            kind.failure(e.to_string())
        }
        Err(e) => {
            eprintln!("[relay] Brake {} task panicked: {e}", kind.label());
            kind.failure(e.to_string())
        }
    }
}

fn log_shutdown_plan(snap: &SafetySnapshot, in_flight: InFlightActuation, plan: &ShutdownPlan) {
    println!();
    eprintln!(
        "[relay] shutdown reason={} state={} policy_state={} brake_engaged={} desired_brake={} \
         in_flight={} unresolved_brake={} unresolved_actuator={} actuation={} ({})",
        plan.reason.as_str(),
        snap.state.as_str(),
        snap.policy_state.as_str(),
        snap.brake_engaged,
        snap.desired_brake,
        in_flight.as_str(),
        plan.unresolved_brake,
        plan.unresolved_actuator,
        plan.actuation.as_str(),
        plan.summary,
    );
    tracing::info!(
        reason = plan.reason.as_str(),
        state = snap.state.as_str(),
        policy_state = snap.policy_state.as_str(),
        brake_engaged = snap.brake_engaged,
        desired_brake = snap.desired_brake,
        in_flight = in_flight.as_str(),
        unresolved_brake = plan.unresolved_brake,
        unresolved_actuator = plan.unresolved_actuator,
        actuation = plan.actuation.as_str(),
        summary = plan.summary,
        "orderly shutdown (fail-closed: no new power-limit restore)"
    );
}

fn should_evaluate_tick(step_count: u64, source: TelemetrySource) -> bool {
    step_count == 1 || step_count.is_multiple_of(10) || source == TelemetrySource::NvmlUnavailable
}

fn should_dispatch_intent(
    step_count: u64,
    prior: BrakeIntent,
    current: BrakeIntent,
    actuation_failed_this_tick: bool,
) -> bool {
    if current == BrakeIntent::None {
        return false;
    }
    if current != prior {
        return true;
    }
    !actuation_failed_this_tick && (step_count == 1 || step_count.is_multiple_of(10))
}

fn should_log_acquisition_error(warned_busy: &mut bool, error: Option<&NvmlRunError>) -> bool {
    match error {
        Some(NvmlRunError::Busy) if *warned_busy => false,
        Some(NvmlRunError::Busy) => {
            *warned_busy = true;
            true
        }
        Some(_) => {
            *warned_busy = false;
            true
        }
        None => {
            *warned_busy = false;
            false
        }
    }
}

fn log_safety_snapshot(snap: &SafetySnapshot, warned_brake_held_sim: &mut bool) {
    if let Some((from, to)) = snap.transition {
        eprintln!(
            "[relay] safety {} → {} ({})",
            from.as_str(),
            to.as_str(),
            snap.last_reason
        );
    }
    if let Some(level) = alert_level_on_transition(snap) {
        eprintln!("[relay] SAFETY {level}: {}", snap.last_reason);
    }
    // A newly recorded failure always carries its error and reports
    // ActuatorFailure (see SafetyMachine::compose_and_store).
    if snap.actuator_failed
        && let Some(err) = &snap.last_actuator_error
    {
        eprintln!("[relay] SAFETY actuator failure: {err}");
    }
    *warned_brake_held_sim = brake_held_sim_warned(snap, *warned_brake_held_sim);
}

/// Warn once per episode that simulated telemetry is holding a real brake;
/// the flag resets on the first state that is not a held-simulated brake.
fn brake_held_sim_warned(snap: &SafetySnapshot, already_warned: bool) -> bool {
    match snap.state {
        SafetyState::SimulatedSoftwareOnly if snap.brake_engaged => {
            if !already_warned {
                eprintln!(
                    "[relay] SAFETY: brake held — telemetry is simulated (no real GPU readings to confirm safe release)"
                );
            }
            true
        }
        SafetyState::HealthyReal | SafetyState::Recovering | SafetyState::SimulatedSoftwareOnly => {
            false
        }
        _ => already_warned,
    }
}

/// Alert line emitted only when a step enters a critical or warning state.
fn alert_level_on_transition(snap: &SafetySnapshot) -> Option<&'static str> {
    snap.transition?;
    match snap.state {
        SafetyState::CriticalBraked
        | SafetyState::TelemetryMissing
        | SafetyState::TelemetryStale
        | SafetyState::TelemetryInvalid => Some("CRITICAL"),
        SafetyState::Warning => Some("WARN"),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchResult {
    Spawned,
    GateBusy,
    AlreadyInFlight,
    NoIntent,
}

/// Dispatch the machine's [`BrakeIntent`] onto a blocking worker so the
/// telemetry loop is never stalled by `nvidia-smi`. Actuation always goes
/// through the [`SafetyActuator`] boundary; the outcome is fed back into the
/// pure [`SafetyMachine`] by the caller.
fn spawn_intent(
    snap: &SafetySnapshot,
    actuator: &Arc<dyn SafetyActuator>,
    gate: &NvmlGate,
    slots: (&mut Option<ActuationTask>, &mut Option<ActuationTask>),
) -> DispatchResult {
    let (brake_task, release_task) = slots;
    if brake_task.is_some() || release_task.is_some() {
        return DispatchResult::AlreadyInFlight;
    }
    let slot = match snap.intent {
        BrakeIntent::Apply => brake_task,
        BrakeIntent::Release => release_task,
        BrakeIntent::None => return DispatchResult::NoIntent,
    };
    let Ok(permit) = gate.try_enter() else {
        // A wedged driver has not attempted a new actuation. Preserve the
        // machine's intent without inventing an actuator failure outcome.
        return DispatchResult::GateBusy;
    };
    let actuator = Arc::clone(actuator);
    let intent = snap.intent;
    *slot = Some(tokio::task::spawn_blocking(move || {
        let _permit = permit;
        match intent {
            BrakeIntent::Apply => actuator.apply_emergency_brake(BRAKE_FRACTION),
            BrakeIntent::Release => actuator.release_emergency_brake(),
            BrakeIntent::None => unreachable!("only actuation intents acquire a slot"),
        }
    }));
    DispatchResult::Spawned
}

fn print_dashboard(frame: &TelemetryFrame, step: u64, safety: SafetyState) {
    let pwr = format_live_reading(&frame.power_w, |w| format!("{w:5.1}W"));
    let vddcr = format_live_reading(&frame.vddcr_gfx_v, |v| format!("{v:.3}V"));
    let tag = match frame.source {
        TelemetrySource::SoftwareFallback => " [sim]",
        TelemetrySource::NvmlUnavailable => " [unavail]",
        TelemetrySource::Nvml => "",
    };
    print!(
        "\r[Step {step}] Pwr: {pwr} | Vddcr: {vddcr}{tag} | safety: {}   ",
        safety.as_str()
    );
    let _ = io::stdout().flush();
}

/// Live hardware display: only [`SampleValidity::Valid`] + present values.
fn format_live_reading(sample: &TelemetrySample<f32>, fmt_val: impl Fn(f32) -> String) -> String {
    match (sample.validity, sample.value) {
        (SampleValidity::Valid, Some(v)) => fmt_val(v),
        (SampleValidity::Stale, _) => "stale".to_string(),
        (SampleValidity::Invalid, _) => "inv".to_string(),
        (SampleValidity::Missing, _) | (_, None) => "n/a".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuTarget, NvmlActuator};
    use crate::publish::{AbsentPublisher, IsolatedPublishQueue, PublishError};
    use crate::safety::FakeActuator;
    use crate::telemetry::{SensoryMapping, assess, fixtures, unix_now_ms};
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::time::Instant;

    // ---- unit tests for the pure helpers (moved from daemon.rs) ----

    #[test]
    fn unavailable_tick_is_evaluated_without_waiting_for_tenth_tick() {
        assert!(should_evaluate_tick(11, TelemetrySource::NvmlUnavailable));
        assert!(!should_evaluate_tick(11, TelemetrySource::Nvml));
        assert!(should_evaluate_tick(10, TelemetrySource::Nvml));
    }

    #[test]
    fn unchanged_brake_intent_retries_only_on_safety_cadence() {
        assert!(should_dispatch_intent(
            11,
            BrakeIntent::None,
            BrakeIntent::Apply,
            false,
        ));
        assert!(!should_dispatch_intent(
            11,
            BrakeIntent::Apply,
            BrakeIntent::Apply,
            false,
        ));
        assert!(should_dispatch_intent(
            20,
            BrakeIntent::Apply,
            BrakeIntent::Apply,
            false,
        ));
    }

    #[test]
    fn failed_actuation_does_not_retry_again_on_its_completion_tick() {
        assert!(!should_dispatch_intent(
            20,
            BrakeIntent::Apply,
            BrakeIntent::Apply,
            true,
        ));
        assert!(should_dispatch_intent(
            30,
            BrakeIntent::Apply,
            BrakeIntent::Apply,
            false,
        ));
        assert!(should_dispatch_intent(
            20,
            BrakeIntent::Release,
            BrakeIntent::Apply,
            true,
        ));
    }

    #[test]
    fn busy_nvml_warning_is_emitted_once_until_the_gate_recovers() {
        let mut warned_busy = false;
        assert!(should_log_acquisition_error(
            &mut warned_busy,
            Some(&NvmlRunError::TimedOut),
        ));
        assert!(should_log_acquisition_error(
            &mut warned_busy,
            Some(&NvmlRunError::Busy),
        ));
        for _ in 0..100 {
            assert!(!should_log_acquisition_error(
                &mut warned_busy,
                Some(&NvmlRunError::Busy),
            ));
        }
        assert!(!should_log_acquisition_error(&mut warned_busy, None));
        assert!(should_log_acquisition_error(
            &mut warned_busy,
            Some(&NvmlRunError::Busy),
        ));
    }

    #[tokio::test]
    async fn busy_nvml_gate_does_not_record_an_actuation_attempt() {
        let gate = NvmlGate::default();
        let _permit = gate.try_enter().unwrap();
        let mut machine = crate::safety::test_machine();
        let frame = assess(&fixtures::nvml_unavailable(), fixtures::NOW);
        let snap = machine.evaluate(&frame);
        let actuator: Arc<dyn SafetyActuator> = Arc::new(NvmlActuator::new(None));
        let mut brake_task = None;
        let mut release_task = None;
        spawn_intent(
            &snap,
            &actuator,
            &gate,
            (&mut brake_task, &mut release_task),
        );
        assert!(brake_task.is_none());
        assert!(release_task.is_none());
    }

    #[tokio::test]
    async fn gate_blocked_brake_dispatches_on_next_non_evaluation_tick() {
        let fake = Arc::new(FakeActuator::new());
        let mut sup = harness()
            .actuator(fake.clone())
            .frames([fixtures::nvml_unavailable()])
            .build();
        let permit = sup.gate.try_enter().unwrap();
        let snap = sup
            .machine
            .evaluate(&assess(&fixtures::nvml_unavailable(), fixtures::NOW));
        assert_eq!(snap.intent, BrakeIntent::Apply);

        // Gate busy: nothing spawned, the intent is parked.
        let spawned = sup.dispatch_due_or_pending(true);
        assert_eq!(
            (spawned, sup.pending_intent, sup.brake_task.is_some()),
            (false, Some(BrakeIntent::Apply), false)
        );
        drop(permit);

        // Gate free on a non-due tick: the parked intent dispatches.
        let spawned = sup.dispatch_due_or_pending(false);
        assert_eq!((spawned, sup.pending_intent), (true, None));
        let applied = sup.brake_task.take().unwrap().await.unwrap();
        assert_eq!((applied, fake.apply_calls()), (Ok(()), 1));
    }

    #[tokio::test]
    async fn gate_blocked_intent_clears_when_safety_intent_changes() {
        let fake = Arc::new(FakeActuator::new());
        let mut sup = harness().actuator(fake.clone()).build();
        let _ = sup
            .machine
            .evaluate(&assess(&fixtures::nvml_unavailable(), fixtures::NOW));
        let snap = sup
            .machine
            .evaluate(&assess(&fixtures::healthy_real(), fixtures::NOW));
        assert_eq!(snap.intent, BrakeIntent::None);
        sup.pending_intent = Some(BrakeIntent::Apply);

        let spawned = sup.dispatch_due_or_pending(false);
        assert_eq!(
            (
                spawned,
                sup.pending_intent,
                sup.brake_task.is_some(),
                fake.apply_calls()
            ),
            (false, None, false, 0)
        );
    }

    #[tokio::test]
    async fn busy_nvml_worker_fail_closes_tick_and_post_actuation_reads() {
        let bridge = Arc::new(HardwareBridge::new(Some(GpuTarget::from_uuid_for_test(
            "GPU-12345678-1234-1234-1234-123456789abc",
        ))));
        let mut sup = harness().acquirer(bridge).real_clock().build();
        let _permit = sup.gate.try_enter().unwrap();
        let raw = sup.acquire_raw_with_timeout().await;
        assert_eq!(raw.source, TelemetrySource::NvmlUnavailable);
        let frame = sup.assess(&raw);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Missing);
        sup.config.force_software_only = true;
        assert_eq!(
            sup.acquire_raw_with_timeout().await.source,
            TelemetrySource::SoftwareFallback
        );
    }

    #[test]
    fn dashboard_shows_only_valid_present_as_live_hardware() {
        let watts = |w: f32| format!("{w:.0}W");
        let celsius = |t: f32| format!("{t:.0}");
        let live = assess(&fixtures::healthy_real(), fixtures::NOW);
        let stale = assess(&fixtures::stale(), fixtures::NOW);
        let invalid = assess(&fixtures::out_of_range(), fixtures::NOW);
        let missing = assess(&fixtures::sensor_dropout(), fixtures::NOW);
        let shown = [
            format_live_reading(&live.power_w, watts),
            format_live_reading(&stale.power_w, watts),
            format_live_reading(&invalid.gpu_temp_c, celsius),
            format_live_reading(&missing.power_w, watts),
        ];
        assert_eq!(shown, ["200W", "stale", "inv", "n/a"]);
    }

    // ---- shutdown (moved from daemon.rs) ----

    fn metrics_collector(
        relay_metrics: &Arc<Mutex<RelayMetrics>>,
    ) -> (tokio::sync::watch::Sender<bool>, JoinHandle<()>) {
        let (tx, rx) = tokio::sync::watch::channel(false);
        let task = tokio::spawn(cpu::run_metrics_collector(Arc::clone(relay_metrics), rx));
        (tx, task)
    }

    #[tokio::test]
    async fn orderly_shutdown_leaves_seeded_brake_and_stops_metrics() {
        let mut sup = harness().build();
        sup.machine.seed_brake_applied();
        let (tx, metrics_task) = metrics_collector(&sup.relay_metrics);
        let plan = sup
            .shutdown(ShutdownReason::Sigterm, &tx, metrics_task)
            .await;
        let reason = sup.relay_metrics.lock().unwrap().shutdown_reason;
        assert_eq!(
            (
                plan.actuation,
                plan.leave_brake_engaged,
                plan.unresolved_brake,
                sup.snapshot().brake_engaged,
                reason
            ),
            (ShutdownActuation::Idle, true, true, true, Some("sigterm"))
        );
    }

    #[tokio::test]
    async fn orderly_shutdown_records_in_flight_apply_without_release() {
        let fake = Arc::new(FakeActuator::new());
        let fake_task = Arc::clone(&fake);
        let mut sup = harness().actuator(fake.clone()).build();
        let mut raw = fixtures::healthy_real();
        raw.gpu_temp_c = Some(90.0);
        let snap = sup.machine.evaluate(&assess(&raw, fixtures::NOW));
        assert_eq!(snap.intent, BrakeIntent::Apply);

        sup.brake_task = Some(tokio::task::spawn_blocking(move || {
            fake_task.apply_emergency_brake(BRAKE_FRACTION)
        }));
        tokio::time::sleep(Duration::from_millis(20)).await;

        let (tx, metrics_task) = metrics_collector(&sup.relay_metrics);
        let plan = sup
            .shutdown(ShutdownReason::Sigint, &tx, metrics_task)
            .await;
        assert_eq!(
            (plan.actuation, sup.snapshot().brake_engaged, hw(&fake)),
            (ShutdownActuation::AwaitApply, true, hw_is(true, 1, 0))
        );
    }

    // ---- fault-injection harness (RM-1459) ----

    /// Scripted telemetry: pops frames in order; the final frame repeats forever.
    /// An optional hang gate blocks every call until released.
    struct ScriptedAcquirer {
        frames: Mutex<VecDeque<RawTelemetry>>,
        last: Mutex<RawTelemetry>,
        hang: Mutex<Option<mpsc::Receiver<()>>>,
        /// False once `unhang()` ran; a woken worker then must not re-arm.
        hang_armed: AtomicBool,
        calls: Mutex<u32>,
        hang_entered: Mutex<Option<mpsc::Sender<()>>>,
    }

    impl ScriptedAcquirer {
        fn new(frames: impl IntoIterator<Item = RawTelemetry>) -> Self {
            let mut frames: VecDeque<_> = frames.into_iter().collect();
            let last = frames.pop_back().unwrap_or_else(fixtures::healthy_real);
            Self {
                frames: Mutex::new(frames),
                last: Mutex::new(last),
                hang: Mutex::new(None),
                hang_armed: AtomicBool::new(false),
                calls: Mutex::new(0),
                hang_entered: Mutex::new(None),
            }
        }

        /// Make every subsequent gated read block until `release` fires.
        /// Returns (release, entered): `entered` receives one `()` per call
        /// that started blocking.
        fn hang_until_released(&self) -> (mpsc::Sender<()>, mpsc::Receiver<()>) {
            let (release_tx, release_rx) = mpsc::channel();
            let (entered_tx, entered_rx) = mpsc::channel();
            self.hang_armed.store(true, Ordering::Release);
            *self.hang.lock().unwrap() = Some(release_rx);
            *self.hang_entered.lock().unwrap() = Some(entered_tx);
            (release_tx, entered_rx)
        }

        fn unhang(&self) {
            self.hang_armed.store(false, Ordering::Release);
            *self.hang.lock().unwrap() = None;
            *self.hang_entered.lock().unwrap() = None;
        }

        fn calls(&self) -> u32 {
            *self.calls.lock().unwrap()
        }
    }

    impl TelemetryAcquirer for ScriptedAcquirer {
        fn acquire_raw(&self, force_software_only: bool) -> RawTelemetry {
            *self.calls.lock().unwrap() += 1;
            if force_software_only {
                return fixtures::software_fallback();
            }
            // Block *without* holding our own mutex across the wait so the
            // test can call `unhang()` / `calls()` while a worker is parked.
            let entered = self.hang_entered.lock().unwrap().clone();
            let hang = self.hang.lock().unwrap().take();
            if let Some(rx) = hang {
                if let Some(tx) = entered {
                    let _ = tx.send(());
                }
                // Wait for release; a dropped sender releases too.
                let _ = rx.recv();
                // Re-arm so later calls keep blocking while the test still
                // wants it; `unhang()` in the meantime cancels the re-arm.
                if self.hang_armed.load(Ordering::Acquire) {
                    *self.hang.lock().unwrap() = Some(rx);
                }
            }
            let next = self.frames.lock().unwrap().pop_front();
            next.unwrap_or_else(|| self.last.lock().unwrap().clone())
        }
    }

    /// Actuator wrapper that parks apply/release until released, and can
    /// panic instead, on top of a [`FakeActuator`].
    struct ControlledActuator {
        inner: Arc<FakeActuator>,
        block_apply: Mutex<Option<mpsc::Receiver<()>>>,
        apply_entered: Mutex<Option<mpsc::Sender<()>>>,
        panic_apply: Mutex<bool>,
    }

    impl ControlledActuator {
        fn new(inner: Arc<FakeActuator>) -> Self {
            Self {
                inner,
                block_apply: Mutex::new(None),
                apply_entered: Mutex::new(None),
                panic_apply: Mutex::new(false),
            }
        }

        fn block_next_apply(&self) -> (mpsc::Sender<()>, mpsc::Receiver<()>) {
            let (release_tx, release_rx) = mpsc::channel();
            let (entered_tx, entered_rx) = mpsc::channel();
            *self.block_apply.lock().unwrap() = Some(release_rx);
            *self.apply_entered.lock().unwrap() = Some(entered_tx);
            (release_tx, entered_rx)
        }

        fn panic_on_apply(&self) {
            *self.panic_apply.lock().unwrap() = true;
        }
    }

    impl SafetyActuator for ControlledActuator {
        fn apply_emergency_brake(&self, pct: f32) -> Result<(), ActuatorError> {
            if *self.panic_apply.lock().unwrap() {
                panic!("injected actuator worker panic");
            }
            let block = self.block_apply.lock().unwrap().take();
            if let Some(rx) = block {
                if let Some(tx) = self.apply_entered.lock().unwrap().take() {
                    let _ = tx.send(());
                }
                let _ = rx.recv();
            }
            self.inner.apply_emergency_brake(pct)
        }
        fn release_emergency_brake(&self) -> Result<(), ActuatorError> {
            self.inner.release_emergency_brake()
        }
        fn query_power_limits_w(&self) -> (Option<u32>, Option<u32>) {
            self.inner.query_power_limits_w()
        }
    }

    /// Publisher backed by a bounded queue whose consumer never drains.
    /// Overflow is dropped, never awaited; `calls` counts publish attempts.
    struct CountingSlowPublisher {
        queue: IsolatedPublishQueue<SensoryMapping>,
        calls: Arc<Mutex<u32>>,
    }

    impl SensoryPublisher for CountingSlowPublisher {
        fn try_publish(&self, mapping: &SensoryMapping) -> Result<(), PublishError> {
            *self.calls.lock().unwrap() += 1;
            self.queue.try_publish(mapping)
        }
    }

    const FAST_TIMEOUT: Duration = Duration::from_millis(40);

    struct Harness {
        actuator: Option<Arc<dyn SafetyActuator>>,
        acquirer: Option<Arc<dyn TelemetryAcquirer>>,
        publisher: Option<Box<dyn SensoryPublisher>>,
        software_only: bool,
        real_clock: bool,
    }

    fn harness() -> Harness {
        Harness {
            actuator: None,
            acquirer: None,
            publisher: None,
            software_only: false,
            real_clock: false,
        }
    }

    impl Harness {
        fn actuator(mut self, actuator: Arc<impl SafetyActuator + 'static>) -> Self {
            self.actuator = Some(actuator);
            self
        }
        fn acquirer(mut self, acquirer: Arc<impl TelemetryAcquirer>) -> Self {
            self.acquirer = Some(acquirer);
            self
        }
        fn frames(self, frames: impl IntoIterator<Item = RawTelemetry>) -> Self {
            self.acquirer(Arc::new(ScriptedAcquirer::new(frames)))
        }
        fn publisher(mut self, publisher: impl SensoryPublisher + 'static) -> Self {
            self.publisher = Some(Box::new(publisher));
            self
        }
        fn software_only(mut self) -> Self {
            self.software_only = true;
            self
        }
        fn real_clock(mut self) -> Self {
            self.real_clock = true;
            self
        }
        fn build(self) -> Supervisor {
            let clock: Clock = if self.real_clock {
                Arc::new(unix_now_ms)
            } else {
                Arc::new(|| fixtures::NOW)
            };
            Supervisor::new(
                SupervisorConfig {
                    force_software_only: self.software_only,
                    cadence_ms: 100,
                    nvml_timeout: FAST_TIMEOUT,
                    dashboard: false,
                },
                SupervisorParts {
                    machine: crate::safety::test_machine(),
                    sample_clock: SampleClock::new(),
                    publisher: self.publisher.unwrap_or_else(|| Box::new(AbsentPublisher)),
                    actuator: self
                        .actuator
                        .unwrap_or_else(|| Arc::new(FakeActuator::new())),
                    acquirer: self
                        .acquirer
                        .unwrap_or_else(|| Arc::new(ScriptedAcquirer::new([]))),
                    gate: NvmlGate::default(),
                    clock,
                    relay_metrics: Arc::new(Mutex::new(RelayMetrics::default())),
                },
            )
        }
    }

    fn critical_temp() -> RawTelemetry {
        RawTelemetry {
            gpu_temp_c: Some(90.0),
            ..fixtures::healthy_real()
        }
    }

    /// Step until `done(report)` or `max` steps; panics on exhaustion.
    async fn step_until(
        sup: &mut Supervisor,
        max: u32,
        mut done: impl FnMut(&StepReport) -> bool,
    ) -> StepReport {
        for _ in 0..max {
            let report = sup.step().await;
            if done(&report) {
                return report;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "condition not reached within {max} steps: {:?}",
            sup.snapshot()
        );
    }

    /// Yield until an in-flight worker has finished so the next step joins it.
    async fn settle(sup: &Supervisor) {
        for task in [&sup.brake_task, &sup.release_task].into_iter().flatten() {
            while !task.is_finished() {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        }
    }

    /// Wait until the wedged blocking worker has returned its gate permit.
    async fn wait_gate_free(sup: &Supervisor) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Ok(permit) = sup.gate.try_enter() {
                drop(permit);
                return;
            }
            assert!(Instant::now() < deadline, "gate never freed");
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    const SUDO_DENIED: &str = "sudo: a password is required";

    fn sudo_denied() -> String {
        ActuatorError::CommandFailed(SUDO_DENIED.into()).to_string()
    }

    // Short names keep each expected observation on one readable line.
    use crate::safety::{BrakeIntent as I, SafetyState as S};

    // ---- comparable observations ----
    //
    // Each check compares a whole observation as one value. Any field that
    // drifts fails the test, which is stricter than asserting a subset.

    /// Hardware-double state: the claim plus the apply and release counts.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Hw {
        engaged: bool,
        applies: u32,
        releases: u32,
    }

    fn hw(fake: &FakeActuator) -> Hw {
        Hw {
            engaged: fake.is_engaged(),
            applies: fake.apply_calls(),
            releases: fake.release_calls(),
        }
    }

    /// Expected double state: claim, apply count, release count.
    const fn hw_is(engaged: bool, applies: u32, releases: u32) -> Hw {
        Hw {
            engaged,
            applies,
            releases,
        }
    }

    /// Actuator outcome without its free-text reason (panic text is unstable).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Outcome {
        Applied,
        ApplyFailed,
        Released,
        ReleaseFailed,
    }

    /// Everything one step decided.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Tick {
        source: TelemetrySource,
        state: SafetyState,
        intent: BrakeIntent,
        brake_engaged: bool,
        in_flight: InFlightActuation,
        dispatched: bool,
        pending: Option<BrakeIntent>,
        outcome: Option<Outcome>,
    }

    fn tick(r: &StepReport) -> Tick {
        Tick {
            source: r.source,
            state: r.snapshot.state,
            intent: r.snapshot.intent,
            brake_engaged: r.snapshot.brake_engaged,
            in_flight: r.in_flight,
            dispatched: r.dispatched,
            pending: r.pending_intent,
            outcome: r.actuation.as_ref().map(|o| match o {
                ActuatorOutcome::Applied => Outcome::Applied,
                ActuatorOutcome::ApplyFailed(_) => Outcome::ApplyFailed,
                ActuatorOutcome::Released => Outcome::Released,
                ActuatorOutcome::ReleaseFailed(_) => Outcome::ReleaseFailed,
            }),
        }
    }

    /// Real-telemetry tick with nothing in flight, dispatched, parked or joined.
    fn quiet(state: SafetyState, intent: BrakeIntent, brake_engaged: bool) -> Tick {
        Tick {
            source: TelemetrySource::Nvml,
            state,
            intent,
            brake_engaged,
            in_flight: InFlightActuation::None,
            dispatched: false,
            pending: None,
            outcome: None,
        }
    }

    /// Fail-closed tick: the read found NVML unavailable or the gate busy.
    fn unavailable(state: SafetyState, intent: BrakeIntent, brake_engaged: bool) -> Tick {
        Tick {
            source: TelemetrySource::NvmlUnavailable,
            ..quiet(state, intent, brake_engaged)
        }
    }

    /// Real-telemetry tick that spawned a worker for its intent.
    fn dispatching(state: SafetyState, intent: BrakeIntent, brake_engaged: bool) -> Tick {
        let in_flight = match intent {
            I::Apply => InFlightActuation::Apply,
            I::Release => InFlightActuation::Release,
            I::None => InFlightActuation::None,
        };
        Tick {
            in_flight,
            dispatched: true,
            ..quiet(state, intent, brake_engaged)
        }
    }

    /// Real-telemetry tick that joined `outcome` and dispatched nothing further.
    fn joined(
        outcome: Outcome,
        state: SafetyState,
        intent: BrakeIntent,
        brake_engaged: bool,
    ) -> Tick {
        Tick {
            outcome: Some(outcome),
            ..quiet(state, intent, brake_engaged)
        }
    }

    // Fault 1: hung NVML acquire. The tick must return within the bound, fail
    // closed, and must not invent an actuator failure; the brake is parked
    // because the wedged worker still holds the gate, then applied once the
    // driver call returns.
    #[tokio::test]
    async fn hung_acquire_fail_closes_in_bounded_time_and_defers_brake_until_gate_frees() {
        let fake = Arc::new(FakeActuator::new());
        // Hold the eventual apply in flight so its dispatch tick is
        // deterministic; the bare fake can race that tick's own read.
        let controlled = Arc::new(ControlledActuator::new(fake.clone()));
        let (release_apply, apply_entered) = controlled.block_next_apply();
        let acquirer = Arc::new(ScriptedAcquirer::new([fixtures::healthy_real()]));
        let (release, entered) = acquirer.hang_until_released();
        let mut sup = harness()
            .actuator(controlled)
            .acquirer(acquirer.clone())
            .build();
        let parked = Tick {
            pending: Some(I::Apply),
            ..unavailable(S::TelemetryMissing, I::Apply, false)
        };

        let started = Instant::now();
        let report = sup.step().await;
        let bounded = started.elapsed() < Duration::from_secs(1);
        entered.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!((bounded, tick(&report)), (true, parked));
        // No actuation attempt and no invented actuator failure.
        assert_eq!(
            (hw(&fake), sup.machine().actuator_failures_total()),
            (hw_is(false, 0, 0), 0)
        );

        // Still hung: every tick stays fail-closed and parks the intent; the
        // gate is single-flight so no second worker piles up behind the first.
        for _ in 0..3 {
            assert_eq!(tick(&sup.step().await), parked);
        }
        assert_eq!(acquirer.calls(), 1, "no second read worker");

        // Driver returns: the parked brake takes the freed gate before this
        // tick's own read, which then finds the gate held by the apply and
        // fails closed rather than spawning anything else.
        acquirer.unhang();
        release.send(()).unwrap();
        wait_gate_free(&sup).await;
        let report = sup.step().await;
        apply_entered.recv_timeout(Duration::from_secs(1)).unwrap();
        let claimed = Tick {
            in_flight: InFlightActuation::Apply,
            dispatched: true,
            ..unavailable(S::TelemetryMissing, I::Apply, false)
        };
        assert_eq!(tick(&report), claimed, "brake claims gate first");

        // The post-apply re-read reached the recovered driver (healthy frame),
        // so the machine is Recovering rather than still fail-closed.
        release_apply.send(()).unwrap();
        settle(&sup).await;
        let report = step_until(&mut sup, 50, |r| r.snapshot.brake_engaged).await;
        assert_eq!(
            (tick(&report), hw(&fake)),
            (
                joined(Outcome::Applied, S::Recovering, I::None, true),
                hw_is(true, 1, 0)
            )
        );
    }

    // Fault 2: `sudo -n` denied / nvidia-smi non-zero exit on apply. The
    // supervisor records ActuatorFailure, keeps evaluating, does not release,
    // and retries on the next safety cadence rather than on the failing tick.
    #[tokio::test]
    async fn apply_denied_by_sudo_is_actuator_failure_and_retries_on_cadence() {
        let fake = Arc::new(FakeActuator::new());
        fake.set_apply_failure(Some(ActuatorError::CommandFailed(SUDO_DENIED.into())));
        let mut sup = harness()
            .actuator(fake.clone())
            .frames([critical_temp()])
            .build();

        let report = sup.step().await;
        assert_eq!(
            tick(&report),
            dispatching(S::CriticalBraked, I::Apply, false)
        );
        settle(&sup).await;

        // Failure recorded with the exact sudo error; no retry on this tick.
        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert_eq!(
            tick(&report),
            joined(Outcome::ApplyFailed, S::ActuatorFailure, I::Apply, false)
        );
        assert_eq!(
            (report.actuation, report.snapshot.last_actuator_error),
            (
                Some(ActuatorOutcome::ApplyFailed(sudo_denied())),
                Some(sudo_denied())
            )
        );
        let exported = sup.relay_metrics.lock().unwrap().safety_state;
        assert_eq!(
            (hw(&fake), sup.machine().actuator_failures_total(), exported),
            (hw_is(false, 1, 0), 1, S::ActuatorFailure)
        );

        // Retried at the 10-tick cadence, and still denied.
        let report = step_until(&mut sup, 12, |r| r.dispatched).await;
        assert_eq!(report.step, 10);
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert_eq!(
            (report.snapshot.state, hw(&fake)),
            (S::ActuatorFailure, hw_is(false, 2, 0))
        );

        // Operator fixes sudoers: the next retry engages the brake and the
        // overlay clears on the post-apply reassessment.
        fake.set_apply_failure(None);
        let _ = step_until(&mut sup, 12, |r| r.dispatched).await;
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| r.snapshot.brake_engaged).await;
        assert_eq!(
            (tick(&report), hw(&fake)),
            (
                joined(Outcome::Applied, S::CriticalBraked, I::None, true),
                hw_is(true, 3, 0)
            )
        );
    }

    // Fault 3: release fails after the Ok streak. The brake stays claimed on
    // both the machine and the hardware double; no second release is issued
    // on the failing tick, and the retry eventually releases.
    #[tokio::test]
    async fn release_failure_keeps_brake_engaged_and_retries() {
        let fake = Arc::new(FakeActuator::new());
        fake.apply_emergency_brake(BRAKE_FRACTION).unwrap();
        fake.set_release_failure(Some(ActuatorError::CommandFailed(SUDO_DENIED.into())));
        let mut sup = harness()
            .actuator(fake.clone())
            .frames([fixtures::healthy_real()])
            .build();
        sup.seed_startup_brake(Some(200), Some(400));
        let seeded = sup.snapshot();
        assert_eq!((seeded.brake_engaged, seeded.state), (true, S::Recovering));

        // Evaluations happen at steps 1, 10, 20 → streak of 3 → Release.
        let report = step_until(&mut sup, 25, |r| r.dispatched).await;
        assert_eq!(
            (report.step, tick(&report)),
            (20, dispatching(S::Recovering, I::Release, true))
        );
        settle(&sup).await;

        // Failed release keeps the claim on machine and hardware; no retry
        // is dispatched on the failing tick.
        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert_eq!(
            (tick(&report), hw(&fake)),
            (
                joined(Outcome::ReleaseFailed, S::ActuatorFailure, I::Release, true),
                hw_is(true, 1, 1)
            )
        );

        fake.set_release_failure(None);
        let _ = step_until(&mut sup, 12, |r| r.dispatched).await;
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| !r.snapshot.brake_engaged).await;
        assert_eq!(
            (tick(&report), hw(&fake)),
            (
                joined(Outcome::Released, S::HealthyReal, I::None, false),
                hw_is(false, 1, 2)
            )
        );
    }

    // Fault 4: apply in flight while telemetry turns healthy and the machine
    // stops wanting the brake. No release may overlap the apply; the apply
    // completes, is recorded, and release only after the Ok streak.
    #[tokio::test]
    async fn healthy_telemetry_during_in_flight_apply_never_overlaps_a_release() {
        let fake = Arc::new(FakeActuator::new());
        let controlled = Arc::new(ControlledActuator::new(fake.clone()));
        let (release_apply, apply_entered) = controlled.block_next_apply();
        let mut sup = harness()
            .actuator(controlled)
            .frames([critical_temp(), fixtures::healthy_real()])
            .build();

        let report = sup.step().await;
        assert_eq!(
            tick(&report),
            dispatching(S::CriticalBraked, I::Apply, false)
        );
        apply_entered.recv_timeout(Duration::from_secs(1)).unwrap();

        // The apply worker holds the gate: ticks fail closed to NvmlUnavailable
        // and evaluate every tick, but never spawn a second worker.
        let blocked = Tick {
            in_flight: InFlightActuation::Apply,
            ..unavailable(S::TelemetryMissing, I::Apply, false)
        };
        for _ in 0..12 {
            let report = sup.step().await;
            assert_eq!(
                (tick(&report), sup.release_task.is_some()),
                (blocked, false),
                "release must not overlap apply"
            );
        }
        assert_eq!(hw(&fake), hw_is(false, 0, 0));

        // Post-apply reassessment saw healthy real telemetry: Recovering, no
        // release yet (hysteresis), and nothing dispatched this tick.
        release_apply.send(()).unwrap();
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert_eq!(
            (tick(&report), hw(&fake)),
            (
                joined(Outcome::Applied, S::Recovering, I::None, true),
                hw_is(true, 1, 0)
            )
        );

        // Ok streak completes on cadence; exactly one release follows.
        let report = step_until(&mut sup, 40, |r| r.in_flight == InFlightActuation::Release).await;
        assert_eq!(tick(&report), dispatching(S::Recovering, I::Release, true));
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| !r.snapshot.brake_engaged).await;
        assert_eq!(
            (tick(&report), hw(&fake)),
            (
                joined(Outcome::Released, S::HealthyReal, I::None, false),
                hw_is(false, 1, 1)
            )
        );
    }

    // Fault 5: the blocking actuator worker panics. The supervisor must
    // survive, record an apply failure, and keep its intent.
    #[tokio::test]
    async fn panicking_actuator_worker_is_recorded_as_apply_failure() {
        let fake = Arc::new(FakeActuator::new());
        let controlled = Arc::new(ControlledActuator::new(fake.clone()));
        controlled.panic_on_apply();
        let mut sup = harness()
            .actuator(controlled)
            .frames([critical_temp()])
            .build();

        let report = sup.step().await;
        assert_eq!(
            tick(&report),
            dispatching(S::CriticalBraked, I::Apply, false)
        );
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert_eq!(
            tick(&report),
            joined(Outcome::ApplyFailed, S::ActuatorFailure, I::Apply, false)
        );
        // Counted once, the fake was never reached, and the gate permit was
        // dropped with the panicking worker.
        assert_eq!(
            (
                sup.machine().actuator_failures_total(),
                hw(&fake),
                sup.gate.try_enter().is_ok()
            ),
            (1, hw_is(false, 0, 0), true)
        );
    }

    // Fault 6: slow consumer on the sensory publisher. Safety steps stay fast,
    // publication attempts continue every tick, the bounded queue drops rather
    // than grows, and state transitions are unaffected.
    #[tokio::test]
    async fn slow_consumer_publisher_never_delays_or_alters_safety() {
        let (queue, consumer) = IsolatedPublishQueue::<SensoryMapping>::bounded(2).unwrap();
        let publish_calls = Arc::new(Mutex::new(0u32));
        let probe = queue.clone();
        let fake = Arc::new(FakeActuator::new());
        let mut sup = harness()
            .actuator(fake.clone())
            .publisher(CountingSlowPublisher {
                queue,
                calls: Arc::clone(&publish_calls),
            })
            .frames([fixtures::healthy_real()])
            .build();

        let started = Instant::now();
        let healthy = quiet(S::HealthyReal, I::None, false);
        for _ in 0..30 {
            assert_eq!(tick(&sup.step().await), healthy);
        }
        let fast = started.elapsed() < Duration::from_secs(2);
        let attempts = *publish_calls.lock().unwrap();
        let snapshot = probe.snapshot();
        assert_eq!(
            (
                fast,
                attempts,
                snapshot.dropped_total() >= 28,
                consumer.try_recv().is_some()
            ),
            (true, 30, true, true),
            "fast steps, one attempt per tick, bounded drops: {snapshot:?}"
        );

        // Still stalled, but a critical frame still engages the brake.
        let acquirer: Arc<ScriptedAcquirer> = Arc::new(ScriptedAcquirer::new([critical_temp()]));
        sup.acquirer = acquirer;
        let report = step_until(&mut sup, 12, |r| r.dispatched).await;
        assert_eq!(
            tick(&report),
            dispatching(S::CriticalBraked, I::Apply, false)
        );
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| r.snapshot.brake_engaged).await;
        assert_eq!(
            (tick(&report), hw(&fake)),
            (
                joined(Outcome::Applied, S::CriticalBraked, I::None, true),
                hw_is(true, 1, 0)
            )
        );
    }

    // Fault 7: --force-software-only while a real brake is left over. Simulated
    // frames must hold the brake indefinitely and never authorize a release.
    #[tokio::test]
    async fn simulated_source_holds_leftover_brake_and_never_releases() {
        let fake = Arc::new(FakeActuator::new());
        fake.apply_emergency_brake(BRAKE_FRACTION).unwrap();
        let mut sup = harness().actuator(fake.clone()).software_only().build();
        sup.seed_startup_brake(Some(200), Some(400));
        assert!(sup.snapshot().brake_engaged);

        let held = Tick {
            source: TelemetrySource::SoftwareFallback,
            ..quiet(S::SimulatedSoftwareOnly, I::None, true)
        };
        for _ in 0..45 {
            assert_eq!(tick(&sup.step().await), held);
        }
        let exported_brake = sup.relay_metrics.lock().unwrap().brake_engaged;
        assert_eq!(
            (hw(&fake), sup.warned_brake_held_sim, exported_brake),
            (hw_is(true, 1, 0), true, true),
            "brake held on hardware, operator warned, metrics report it"
        );
    }

    // Fault 8: the post-apply telemetry re-read hangs. The step stays bounded,
    // the Applied outcome is still recorded, and the re-read fail-closes.
    #[tokio::test]
    async fn hung_post_actuation_reread_is_bounded_and_fail_closed() {
        let fake = Arc::new(FakeActuator::new());
        let acquirer = Arc::new(ScriptedAcquirer::new([critical_temp()]));
        let mut sup = harness()
            .actuator(fake.clone())
            .acquirer(acquirer.clone())
            .build();

        let report = sup.step().await;
        assert_eq!(
            tick(&report),
            dispatching(S::CriticalBraked, I::Apply, false)
        );
        settle(&sup).await;
        let (release, _entered) = acquirer.hang_until_released();

        let started = Instant::now();
        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        let bounded = started.elapsed() < Duration::from_secs(1);
        let reread_fail_closed = Tick {
            source: TelemetrySource::NvmlUnavailable,
            ..joined(Outcome::Applied, S::TelemetryMissing, I::None, true)
        };
        assert_eq!(
            (bounded, tick(&report), hw(&fake)),
            (true, reread_fail_closed, hw_is(true, 1, 0))
        );
        acquirer.unhang();
        release.send(()).unwrap();
    }

    // The shutdown path on top of the same supervisor: SIGINT during an
    // in-flight release that fails leaves the brake engaged and reported.
    #[tokio::test]
    async fn shutdown_with_failing_in_flight_release_keeps_brake_and_reports_it() {
        let fake = Arc::new(FakeActuator::new());
        fake.apply_emergency_brake(BRAKE_FRACTION).unwrap();
        fake.set_release_failure(Some(ActuatorError::CommandFailed(SUDO_DENIED.into())));
        let mut sup = harness()
            .actuator(fake.clone())
            .frames([fixtures::healthy_real()])
            .build();
        sup.seed_startup_brake(Some(200), Some(400));
        let report = step_until(&mut sup, 25, |r| r.in_flight == InFlightActuation::Release).await;
        assert_eq!(tick(&report), dispatching(S::Recovering, I::Release, true));

        let (tx, metrics_task) = metrics_collector(&sup.relay_metrics);
        let plan = sup
            .shutdown(ShutdownReason::Sigint, &tx, metrics_task)
            .await;
        let snap = sup.snapshot();
        let (reason, exported_brake) = {
            let metrics = sup.relay_metrics.lock().unwrap();
            (metrics.shutdown_reason, metrics.brake_engaged)
        };
        assert_eq!(
            (plan.actuation, snap.brake_engaged, snap.state, hw(&fake)),
            (
                ShutdownActuation::AwaitAuthorizedRelease,
                true,
                S::ActuatorFailure,
                hw_is(true, 1, 1)
            )
        );
        assert_eq!((reason, exported_brake), (Some("sigint"), true));
    }
}
