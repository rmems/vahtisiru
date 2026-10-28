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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task::JoinHandle;

use crate::cpu::{self, RelayMetrics};
use crate::gpu::HardwareBridge;
use crate::nvml_gate::{NvmlGate, NvmlRunError};
use crate::publish::{SensoryPublisher, evaluate_then_try_publish, try_publish_isolated};
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

/// Monotonic millisecond source for the safety-evaluation deadline
/// (injectable in tests; never wall-clock, so NTP steps cannot skip or stall it).
pub(crate) type MonotonicClock = Arc<dyn Fn() -> u64 + Send + Sync>;

/// Production monotonic clock: milliseconds since the clock was built.
pub(crate) fn monotonic_ms_clock() -> MonotonicClock {
    let start = std::time::Instant::now();
    Arc::new(move || u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX))
}

/// Static supervisor settings derived from the CLI.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SupervisorConfig {
    pub force_software_only: bool,
    /// Declared acquisition cadence (ms); drives stale-after thresholds.
    pub cadence_ms: u64,
    /// Wall-clock safety-evaluation period (ms), independent of tick length.
    pub eval_interval_ms: u64,
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
    pub monotonic: MonotonicClock,
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

/// Sentinel for a release preempted before it reached the driver.
const RELEASE_SKIPPED: &str = "release superseded before start";

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
    monotonic: MonotonicClock,
    relay_metrics: Arc<Mutex<RelayMetrics>>,
    warned_brake_held_sim: bool,
    warned_nvml_busy: bool,
    brake_task: Option<ActuationTask>,
    release_task: Option<ActuationTask>,
    pending_intent: Option<BrakeIntent>,
    step_count: u64,
    last_eval_ms: Option<u64>,
    /// Set when policy wants the brake (`desired_brake`) while a Release is in
    /// flight: a release not yet at the driver is skipped without being recorded.
    release_superseded: Arc<AtomicBool>,
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
            monotonic: parts.monotonic,
            relay_metrics: parts.relay_metrics,
            warned_brake_held_sim: false,
            warned_nvml_busy: false,
            brake_task: None,
            release_task: None,
            pending_intent: None,
            step_count: 0,
            last_eval_ms: None,
            release_superseded: Arc::new(AtomicBool::new(false)),
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

        let now_ms = (self.monotonic)();
        let cadence_due = self
            .last_eval_ms
            .is_none_or(|last| now_ms.saturating_sub(last) >= self.config.eval_interval_ms);
        if cadence_due {
            self.last_eval_ms = Some(now_ms);
        }
        let dispatch_due = succeeded || self.evaluate_or_publish(cadence_due, &telemetry, failed);
        self.supersede_release_if_apply();
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
        let joined = task.await;
        if kind == ActuationKind::Release {
            self.release_superseded.store(false, Ordering::Release);
            if matches!(&joined, Ok(Err(ActuatorError::CommandFailed(m))) if m == RELEASE_SKIPPED) {
                // Never reached the driver: nothing changed, nothing to record.
                // The brake stays engaged and policy decides the next intent.
                return None;
            }
        }
        let outcome = actuation_outcome(kind, joined);
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
    /// startup); later evaluations fire once `eval_interval_ms` of monotonic
    /// time has elapsed, regardless of tick length or acquire latency. Never
    /// spawn from the pre-telemetry snapshot: SoftwareFallback holds rather
    /// than applies, which only `classify_frame` can decide.
    fn evaluate_or_publish(
        &mut self,
        cadence_due: bool,
        telemetry: &TelemetryFrame,
        actuation_failed_this_tick: bool,
    ) -> bool {
        if !should_evaluate_tick(cadence_due, telemetry.source) {
            // Publication is outside the safety critical path and never awaited.
            // Same mapping-time basis as evaluate_then_try_publish.
            // Panic-isolated like the evaluation path (GH#42).
            let _ = try_publish_isolated(self.publisher.as_ref(), &telemetry.to_sensory_mapping());
            return false;
        }
        let prior_intent = self.machine.snapshot().intent;
        let (snap, pub_res) =
            evaluate_then_try_publish(&mut self.machine, telemetry, self.publisher.as_ref());
        let _ = pub_res;
        self.store_safety(&snap);
        should_dispatch_intent(
            cadence_due,
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
        // This reassessment is a real evaluation: restart the deadline so the
        // next tick cannot count a second Ok immediately.
        self.last_eval_ms = Some((self.monotonic)());
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

    /// Policy wants the brake while a `Release` is still in flight (the
    /// machine still counts the brake engaged, so its intent is not `Apply`
    /// yet). Flag the release so a worker that has not reached the driver
    /// skips it. A call already inside `nvidia-smi` cannot be interrupted; its
    /// success is recorded as `Released`, which clears `brake_engaged` and
    /// makes the reassessment in the same step re-apply immediately.
    fn supersede_release_if_apply(&mut self) {
        let wants_brake = self.machine.snapshot().desired_brake;
        if !(self.release_task.is_some() && wants_brake) {
            return;
        }
        if !self.release_superseded.swap(true, Ordering::AcqRel) {
            tracing::warn!("brake desired while release in flight; superseding release");
        }
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
            (
                &mut self.brake_task,
                &mut self.release_task,
                &self.release_superseded,
            ),
        );
        self.pending_intent = match result {
            DispatchResult::GateBusy => Some(snap.intent),
            DispatchResult::Spawned
            | DispatchResult::AlreadyInFlight
            | DispatchResult::NoIntent => None,
        };
        let spawned = result == DispatchResult::Spawned;
        // An apply is in flight only once a worker was actually spawned for
        // it (GH#42); a parked-then-dropped Apply never holds the trip
        // episode open.
        if spawned && snap.intent == BrakeIntent::Apply {
            self.machine.apply_dispatched();
        }
        spawned
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

fn should_evaluate_tick(cadence_due: bool, source: TelemetrySource) -> bool {
    cadence_due || source == TelemetrySource::NvmlUnavailable
}

fn should_dispatch_intent(
    cadence_due: bool,
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
    !actuation_failed_this_tick && cadence_due
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
    slots: (
        &mut Option<ActuationTask>,
        &mut Option<ActuationTask>,
        &Arc<AtomicBool>,
    ),
) -> DispatchResult {
    let (brake_task, release_task, release_superseded) = slots;
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
    let superseded = Arc::clone(release_superseded);
    if intent == BrakeIntent::Release {
        superseded.store(false, Ordering::Release);
    }
    *slot = Some(tokio::task::spawn_blocking(move || {
        let _permit = permit;
        match intent {
            BrakeIntent::Apply => actuator.apply_emergency_brake(BRAKE_FRACTION),
            BrakeIntent::Release if superseded.load(Ordering::Acquire) => {
                Err(ActuatorError::CommandFailed(RELEASE_SKIPPED.into()))
            }
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
mod tests;
