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

        let mut evaluated_this_iter = false;
        let mut actuation_failed_this_tick = false;
        let mut dispatch_due = false;
        let mut actuation = None;

        if self
            .brake_task
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
        {
            let task = self.brake_task.take().expect("finished brake task exists");
            let outcome = match task.await {
                Ok(Ok(())) => ActuatorOutcome::Applied,
                Ok(Err(e)) => {
                    eprintln!("[relay] Emergency brake failed: {e}");
                    ActuatorOutcome::ApplyFailed(e.to_string())
                }
                Err(e) => {
                    eprintln!("[relay] Brake task panicked: {e}");
                    ActuatorOutcome::ApplyFailed(e.to_string())
                }
            };
            if matches!(outcome, ActuatorOutcome::Applied) {
                self.reassess_after_actuation(outcome.clone()).await;
                dispatch_due = true;
                evaluated_this_iter = true;
            } else {
                let snap = self.machine.record_actuator(outcome.clone());
                self.store_safety(&snap);
                actuation_failed_this_tick = true;
            }
            actuation = Some(outcome);
        }
        if self
            .release_task
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
        {
            let task = self
                .release_task
                .take()
                .expect("finished release task exists");
            let outcome = match task.await {
                Ok(Ok(())) => ActuatorOutcome::Released,
                Ok(Err(e)) => {
                    eprintln!("[relay] Brake release failed: {e}");
                    ActuatorOutcome::ReleaseFailed(e.to_string())
                }
                Err(e) => {
                    eprintln!("[relay] Brake release task panicked: {e}");
                    ActuatorOutcome::ReleaseFailed(e.to_string())
                }
            };
            if matches!(outcome, ActuatorOutcome::Released) {
                self.reassess_after_actuation(outcome.clone()).await;
                dispatch_due = true;
                evaluated_this_iter = true;
            } else {
                let snap = self.machine.record_actuator(outcome.clone());
                self.store_safety(&snap);
                actuation_failed_this_tick = true;
            }
            actuation = Some(outcome);
        }

        // First acquired frame is evaluated immediately (fail-closed startup).
        // Later evaluations keep the every-10-ticks cadence.
        // Do not spawn from the pre-telemetry snapshot: SoftwareFallback holds
        // rather than applies, which only classify_frame can decide.
        if !evaluated_this_iter && should_evaluate_tick(step_count, telemetry.source) {
            let prior_intent = self.machine.snapshot().intent;
            let (snap, pub_res) =
                evaluate_then_try_publish(&mut self.machine, &telemetry, self.publisher.as_ref());
            let _ = pub_res;
            self.store_safety(&snap);
            dispatch_due = should_dispatch_intent(
                step_count,
                prior_intent,
                snap.intent,
                actuation_failed_this_tick,
            );
        } else if !evaluated_this_iter {
            // Publication is outside the safety critical path and never awaited.
            let _ = self
                .publisher
                .try_publish(&telemetry.to_sensory_mapping_at((self.clock)()));
        }

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

        {
            let mut metrics = self.relay_metrics.lock().unwrap();
            cpu::record_shutdown(&mut metrics, &plan);
        }

        match plan.actuation {
            ShutdownActuation::Idle => {
                if let Some(task) = self.brake_task.take() {
                    task.abort();
                }
                if let Some(task) = self.release_task.take() {
                    task.abort();
                }
            }
            ShutdownActuation::AwaitApply => {
                if let Some(task) = self.brake_task.take() {
                    match join_in_flight(task, SHUTDOWN_ACTUATOR_TIMEOUT).await {
                        InFlightJoin::Succeeded => {
                            let snap = self.machine.record_actuator(ActuatorOutcome::Applied);
                            self.store_safety(&snap);
                        }
                        InFlightJoin::Failed(e) | InFlightJoin::Panicked(e) => {
                            eprintln!("[relay] shutdown: in-flight apply did not succeed: {e}");
                            let snap = self
                                .machine
                                .record_actuator(ActuatorOutcome::ApplyFailed(e));
                            self.store_safety(&snap);
                        }
                        InFlightJoin::TimedOut => {
                            eprintln!(
                                "[relay] shutdown: in-flight apply timed out; leaving hardware unchanged"
                            );
                        }
                    }
                }
                if let Some(task) = self.release_task.take() {
                    task.abort();
                }
            }
            ShutdownActuation::AwaitAuthorizedRelease => {
                if let Some(task) = self.release_task.take() {
                    match join_in_flight(task, SHUTDOWN_ACTUATOR_TIMEOUT).await {
                        InFlightJoin::Succeeded => {
                            let snap = self.machine.record_actuator(ActuatorOutcome::Released);
                            self.store_safety(&snap);
                        }
                        InFlightJoin::Failed(e) | InFlightJoin::Panicked(e) => {
                            eprintln!("[relay] shutdown: in-flight release did not succeed: {e}");
                            let snap = self
                                .machine
                                .record_actuator(ActuatorOutcome::ReleaseFailed(e));
                            self.store_safety(&snap);
                        }
                        InFlightJoin::TimedOut => {
                            eprintln!(
                                "[relay] shutdown: in-flight release timed out; leaving brake engaged"
                            );
                        }
                    }
                }
                if let Some(task) = self.brake_task.take() {
                    task.abort();
                }
            }
        }

        shutdown_metrics_collector(metrics_shutdown, metrics_task, SHUTDOWN_METRICS_TIMEOUT).await;
        plan
    }
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
    match snap.state {
        SafetyState::CriticalBraked
        | SafetyState::TelemetryMissing
        | SafetyState::TelemetryStale
        | SafetyState::TelemetryInvalid => {
            if snap.transition.is_some() {
                eprintln!("[relay] SAFETY CRITICAL: {}", snap.last_reason);
            }
        }
        SafetyState::Warning => {
            if snap.transition.is_some() {
                eprintln!("[relay] SAFETY WARN: {}", snap.last_reason);
            }
        }
        SafetyState::SimulatedSoftwareOnly if snap.brake_engaged => {
            if !*warned_brake_held_sim {
                eprintln!(
                    "[relay] SAFETY: brake held — telemetry is simulated (no real GPU readings to confirm safe release)"
                );
                *warned_brake_held_sim = true;
            }
        }
        SafetyState::ActuatorFailure => {
            if snap.actuator_failed
                && let Some(err) = &snap.last_actuator_error
            {
                eprintln!("[relay] SAFETY actuator failure: {err}");
            }
        }
        SafetyState::HealthyReal | SafetyState::Recovering | SafetyState::SimulatedSoftwareOnly => {
            *warned_brake_held_sim = false;
        }
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

        assert!(!sup.dispatch_due_or_pending(true));
        assert_eq!(sup.pending_intent, Some(BrakeIntent::Apply));
        assert!(sup.brake_task.is_none());
        drop(permit);

        assert!(sup.dispatch_due_or_pending(false));
        assert_eq!(sup.pending_intent, None);
        assert!(sup.brake_task.take().unwrap().await.unwrap().is_ok());
        assert_eq!(fake.apply_calls(), 1);
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

        assert!(!sup.dispatch_due_or_pending(false));
        assert_eq!(sup.pending_intent, None);
        assert!(sup.brake_task.is_none());
        assert_eq!(fake.apply_calls(), 0);
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
        let live = assess(&fixtures::healthy_real(), fixtures::NOW);
        assert_eq!(
            format_live_reading(&live.power_w, |w| format!("{w:.0}W")),
            "200W"
        );

        let stale = assess(&fixtures::stale(), fixtures::NOW);
        assert_eq!(
            format_live_reading(&stale.power_w, |w| format!("{w:.0}W")),
            "stale"
        );

        let invalid = assess(&fixtures::out_of_range(), fixtures::NOW);
        assert_eq!(
            format_live_reading(&invalid.gpu_temp_c, |t| format!("{t:.0}")),
            "inv"
        );

        let missing = assess(&fixtures::sensor_dropout(), fixtures::NOW);
        assert_eq!(
            format_live_reading(&missing.power_w, |w| format!("{w:.0}W")),
            "n/a"
        );
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
        assert_eq!(plan.actuation, ShutdownActuation::Idle);
        assert!(plan.leave_brake_engaged);
        assert!(plan.unresolved_brake);
        assert!(sup.snapshot().brake_engaged);
        assert_eq!(
            sup.relay_metrics.lock().unwrap().shutdown_reason,
            Some("sigterm")
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
        assert_eq!(plan.actuation, ShutdownActuation::AwaitApply);
        assert!(fake.is_engaged());
        assert!(sup.snapshot().brake_engaged);
        assert_eq!(fake.apply_calls(), 1);
        assert_eq!(fake.release_calls(), 0);
    }

    // ---- fault-injection harness (RM-1459) ----

    /// Scripted telemetry: pops frames in order; the final frame repeats forever.
    /// An optional hang gate blocks every call until released.
    struct ScriptedAcquirer {
        frames: Mutex<VecDeque<RawTelemetry>>,
        last: Mutex<RawTelemetry>,
        hang: Mutex<Option<mpsc::Receiver<()>>>,
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
            *self.hang.lock().unwrap() = Some(release_rx);
            *self.hang_entered.lock().unwrap() = Some(entered_tx);
            (release_tx, entered_rx)
        }

        fn unhang(&self) {
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
                // Re-arm so later calls keep blocking while the test wants it.
                *self.hang.lock().unwrap() = Some(rx);
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

    /// Publisher whose every call blocks the caller: a consumer that never
    /// drains and a producer that would wait on it. The supervisor must never
    /// call this on the safety path in a way that waits.
    struct CountingSlowPublisher {
        queue: IsolatedPublishQueue<SensoryMapping>,
        calls: Mutex<u32>,
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

    // Fault 1: hung NVML acquire. The tick must return within the bound, fail
    // closed, and must not invent an actuator failure; the brake is parked
    // because the wedged worker still holds the gate, then applied once the
    // driver call returns.
    #[tokio::test]
    async fn hung_acquire_fail_closes_in_bounded_time_and_defers_brake_until_gate_frees() {
        let fake = Arc::new(FakeActuator::new());
        let acquirer = Arc::new(ScriptedAcquirer::new([fixtures::healthy_real()]));
        let (release, entered) = acquirer.hang_until_released();
        let mut sup = harness()
            .actuator(fake.clone())
            .acquirer(acquirer.clone())
            .build();

        let started = Instant::now();
        let report = sup.step().await;
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "tick must be bounded"
        );
        entered.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(report.source, TelemetrySource::NvmlUnavailable);
        assert_eq!(report.snapshot.state, SafetyState::TelemetryMissing);
        assert_eq!(report.snapshot.intent, BrakeIntent::Apply);
        assert_eq!(report.pending_intent, Some(BrakeIntent::Apply));
        assert_eq!(report.in_flight, InFlightActuation::None);
        assert!(!report.dispatched);
        assert_eq!(fake.apply_calls(), 0);
        assert_eq!(sup.machine().actuator_failures_total(), 0);

        // Still hung: every tick stays fail-closed and parks the intent; the
        // gate is single-flight so no second worker piles up behind the first.
        for _ in 0..3 {
            let report = sup.step().await;
            assert_eq!(report.source, TelemetrySource::NvmlUnavailable);
            assert_eq!(report.pending_intent, Some(BrakeIntent::Apply));
            assert!(!report.dispatched);
        }
        assert_eq!(
            acquirer.calls(),
            1,
            "no new worker while the first is wedged"
        );

        // Driver returns: the parked brake takes the freed gate first.
        acquirer.unhang();
        release.send(()).unwrap();
        wait_gate_free(&sup).await;
        let report = sup.step().await;
        assert!(
            report.dispatched,
            "parked brake claims the freed gate first"
        );
        assert_eq!(report.in_flight, InFlightActuation::Apply);
        settle(&sup).await;
        let report = step_until(&mut sup, 50, |r| r.snapshot.brake_engaged).await;
        assert_eq!(report.actuation, Some(ActuatorOutcome::Applied));
        assert!(fake.is_engaged());
        assert_eq!(fake.apply_calls(), 1);
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
        assert_eq!(report.snapshot.state, SafetyState::CriticalBraked);
        assert!(report.dispatched);
        settle(&sup).await;

        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert_eq!(
            report.actuation,
            Some(ActuatorOutcome::ApplyFailed(
                ActuatorError::CommandFailed(SUDO_DENIED.into()).to_string()
            ))
        );
        assert_eq!(report.snapshot.state, SafetyState::ActuatorFailure);
        assert_eq!(
            report.snapshot.last_actuator_error.as_deref(),
            Some(
                ActuatorError::CommandFailed(SUDO_DENIED.into())
                    .to_string()
                    .as_str()
            )
        );
        assert!(!report.snapshot.brake_engaged);
        assert_eq!(report.snapshot.intent, BrakeIntent::Apply);
        assert_eq!(sup.machine().actuator_failures_total(), 1);
        assert_eq!(fake.apply_calls(), 1);
        assert_eq!(fake.release_calls(), 0);
        assert_eq!(
            sup.relay_metrics.lock().unwrap().safety_state,
            SafetyState::ActuatorFailure
        );

        // Not retried on the failing tick; retried at the 10-tick cadence.
        let before = fake.apply_calls();
        let report = step_until(&mut sup, 12, |r| r.dispatched).await;
        assert_eq!(report.step, 10);
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert_eq!(fake.apply_calls(), before + 1);
        assert_eq!(report.snapshot.state, SafetyState::ActuatorFailure);

        // Operator fixes sudoers: the next retry engages the brake and the
        // overlay clears on the post-apply reassessment.
        fake.set_apply_failure(None);
        let _ = step_until(&mut sup, 12, |r| r.dispatched).await;
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| r.snapshot.brake_engaged).await;
        assert_eq!(report.actuation, Some(ActuatorOutcome::Applied));
        assert_eq!(report.snapshot.state, SafetyState::CriticalBraked);
        assert!(fake.is_engaged());
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
        assert!(sup.snapshot().brake_engaged);
        assert_eq!(sup.snapshot().state, SafetyState::Recovering);

        // Evaluations happen at steps 1, 10, 20 → streak of 3 → Release.
        let report = step_until(&mut sup, 25, |r| r.dispatched).await;
        assert_eq!(report.step, 20);
        assert_eq!(report.snapshot.intent, BrakeIntent::Release);
        assert_eq!(report.in_flight, InFlightActuation::Release);
        settle(&sup).await;

        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert!(matches!(
            report.actuation,
            Some(ActuatorOutcome::ReleaseFailed(_))
        ));
        assert_eq!(report.snapshot.state, SafetyState::ActuatorFailure);
        assert!(
            report.snapshot.brake_engaged,
            "failed release keeps the claim"
        );
        assert!(fake.is_engaged());
        assert_eq!(fake.release_calls(), 1);
        assert_eq!(fake.apply_calls(), 1);
        assert_eq!(report.in_flight, InFlightActuation::None);
        assert!(!report.dispatched, "no retry on the failing tick");

        fake.set_release_failure(None);
        let _ = step_until(&mut sup, 12, |r| r.dispatched).await;
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| !r.snapshot.brake_engaged).await;
        assert_eq!(report.actuation, Some(ActuatorOutcome::Released));
        assert_eq!(report.snapshot.state, SafetyState::HealthyReal);
        assert!(!fake.is_engaged());
        assert_eq!(fake.release_calls(), 2);
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
        assert!(report.dispatched);
        assert_eq!(report.in_flight, InFlightActuation::Apply);
        apply_entered.recv_timeout(Duration::from_secs(1)).unwrap();

        // The apply worker holds the gate: ticks fail closed to NvmlUnavailable
        // and evaluate every tick, but never spawn a second worker.
        for _ in 0..12 {
            let report = sup.step().await;
            assert_eq!(report.in_flight, InFlightActuation::Apply);
            assert!(!report.dispatched);
            assert!(report.actuation.is_none());
            assert!(sup.release_task.is_none(), "release must not overlap apply");
        }
        assert_eq!(fake.release_calls(), 0);

        release_apply.send(()).unwrap();
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert_eq!(report.actuation, Some(ActuatorOutcome::Applied));
        assert!(report.snapshot.brake_engaged);
        assert!(fake.is_engaged());
        // Post-apply reassessment saw healthy real telemetry: Recovering, no
        // release yet (hysteresis), and nothing dispatched this tick.
        assert_eq!(report.snapshot.state, SafetyState::Recovering);
        assert_eq!(report.snapshot.intent, BrakeIntent::None);
        assert_eq!(fake.release_calls(), 0);
        assert_eq!(fake.apply_calls(), 1);

        // Ok streak completes on cadence; exactly one release follows.
        let report = step_until(&mut sup, 40, |r| r.in_flight == InFlightActuation::Release).await;
        assert_eq!(report.snapshot.intent, BrakeIntent::Release);
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| !r.snapshot.brake_engaged).await;
        assert_eq!(report.actuation, Some(ActuatorOutcome::Released));
        assert_eq!(fake.release_calls(), 1);
        assert_eq!(fake.apply_calls(), 1);
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
        assert!(report.dispatched);
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert!(matches!(
            report.actuation,
            Some(ActuatorOutcome::ApplyFailed(_))
        ));
        assert_eq!(report.snapshot.state, SafetyState::ActuatorFailure);
        assert_eq!(report.snapshot.intent, BrakeIntent::Apply);
        assert!(!report.snapshot.brake_engaged);
        assert_eq!(sup.machine().actuator_failures_total(), 1);
        assert_eq!(fake.apply_calls(), 0);
        // The gate permit was dropped with the panicking worker.
        assert!(sup.gate.try_enter().is_ok());
    }

    // Fault 6: slow consumer on the sensory publisher. Safety steps stay fast,
    // publication attempts continue every tick, the bounded queue drops rather
    // than grows, and state transitions are unaffected.
    #[tokio::test]
    async fn slow_consumer_publisher_never_delays_or_alters_safety() {
        let (queue, consumer) = IsolatedPublishQueue::<SensoryMapping>::bounded(2).unwrap();
        let probe = queue.clone();
        let fake = Arc::new(FakeActuator::new());
        let mut sup = harness()
            .actuator(fake.clone())
            .publisher(CountingSlowPublisher {
                queue,
                calls: Mutex::new(0),
            })
            .frames([fixtures::healthy_real()])
            .build();

        let started = Instant::now();
        for _ in 0..30 {
            let report = sup.step().await;
            assert_eq!(report.snapshot.state, SafetyState::HealthyReal);
        }
        assert!(started.elapsed() < Duration::from_secs(2));
        let snapshot = probe.snapshot();
        assert!(snapshot.dropped_total() >= 28, "{snapshot:?}");
        assert!(consumer.try_recv().is_some());

        // Still stalled, but a critical frame still engages the brake.
        let acquirer: Arc<ScriptedAcquirer> = Arc::new(ScriptedAcquirer::new([critical_temp()]));
        sup.acquirer = acquirer;
        let report = step_until(&mut sup, 12, |r| r.dispatched).await;
        assert_eq!(report.snapshot.state, SafetyState::CriticalBraked);
        settle(&sup).await;
        let report = step_until(&mut sup, 5, |r| r.snapshot.brake_engaged).await;
        assert_eq!(report.actuation, Some(ActuatorOutcome::Applied));
        assert!(fake.is_engaged());
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

        for _ in 0..45 {
            let report = sup.step().await;
            assert_eq!(report.source, TelemetrySource::SoftwareFallback);
            assert_eq!(report.snapshot.state, SafetyState::SimulatedSoftwareOnly);
            assert!(report.snapshot.brake_engaged);
            assert_ne!(report.snapshot.intent, BrakeIntent::Release);
            assert!(!report.dispatched);
            assert_eq!(report.in_flight, InFlightActuation::None);
        }
        assert_eq!(fake.release_calls(), 0);
        assert_eq!(fake.apply_calls(), 1);
        assert!(fake.is_engaged());
        assert!(sup.warned_brake_held_sim, "operator warned exactly once");
        assert!(sup.relay_metrics.lock().unwrap().brake_engaged);
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
        assert!(report.dispatched);
        settle(&sup).await;
        let (release, _entered) = acquirer.hang_until_released();

        let started = Instant::now();
        let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(report.actuation, Some(ActuatorOutcome::Applied));
        assert!(report.snapshot.brake_engaged);
        assert_eq!(report.snapshot.state, SafetyState::TelemetryMissing);
        assert!(fake.is_engaged());
        assert_eq!(fake.release_calls(), 0);
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
        assert_eq!(report.snapshot.intent, BrakeIntent::Release);

        let (tx, metrics_task) = metrics_collector(&sup.relay_metrics);
        let plan = sup
            .shutdown(ShutdownReason::Sigint, &tx, metrics_task)
            .await;
        assert_eq!(plan.actuation, ShutdownActuation::AwaitAuthorizedRelease);
        assert!(sup.snapshot().brake_engaged);
        assert_eq!(sup.snapshot().state, SafetyState::ActuatorFailure);
        assert!(fake.is_engaged());
        let metrics = sup.relay_metrics.lock().unwrap();
        assert_eq!(metrics.shutdown_reason, Some("sigint"));
        assert!(metrics.brake_engaged);
    }
}
