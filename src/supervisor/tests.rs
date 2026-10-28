use super::*;
use crate::gpu::{GpuTarget, NvmlActuator};
use crate::publish::{AbsentPublisher, IsolatedPublishQueue, PublishError};
use crate::safety::{FakeActuator, SafetyReason};
use crate::telemetry::{SensoryMapping, assess, fixtures, unix_now_ms};
use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc;
use std::time::Instant;

// ---- unit tests for the pure helpers (moved from daemon.rs) ----

#[test]
fn unavailable_tick_is_evaluated_without_waiting_for_tenth_tick() {
    assert!(should_evaluate_tick(
        false,
        TelemetrySource::NvmlUnavailable
    ));
    assert!(!should_evaluate_tick(false, TelemetrySource::Nvml));
    assert!(should_evaluate_tick(true, TelemetrySource::Nvml));
}

#[test]
fn unchanged_brake_intent_retries_only_on_safety_cadence() {
    assert!(should_dispatch_intent(
        false,
        BrakeIntent::None,
        BrakeIntent::Apply,
        false,
    ));
    assert!(!should_dispatch_intent(
        false,
        BrakeIntent::Apply,
        BrakeIntent::Apply,
        false,
    ));
    assert!(should_dispatch_intent(
        true,
        BrakeIntent::Apply,
        BrakeIntent::Apply,
        false,
    ));
}

#[test]
fn failed_actuation_does_not_retry_again_on_its_completion_tick() {
    assert!(!should_dispatch_intent(
        true,
        BrakeIntent::Apply,
        BrakeIntent::Apply,
        true,
    ));
    assert!(should_dispatch_intent(
        true,
        BrakeIntent::Apply,
        BrakeIntent::Apply,
        false,
    ));
    assert!(should_dispatch_intent(
        true,
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
        (
            &mut brake_task,
            &mut release_task,
            &Arc::new(AtomicBool::new(false)),
        ),
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

fn temp_frame(temp_c: f32) -> TelemetryFrame {
    let mut raw = fixtures::healthy_real();
    raw.gpu_temp_c = Some(temp_c);
    assess(&raw, fixtures::NOW)
}

/// Evaluate one frame, dispatch, and return the frame's trip reason.
fn step_trip(sup: &mut Supervisor, temp_c: f32, due: bool) -> Option<SafetyReason> {
    let snap = sup.machine.evaluate(&temp_frame(temp_c));
    sup.dispatch_due_or_pending(due);
    snap.trip
}

#[tokio::test]
async fn spawned_apply_holds_the_trip_episode_open_until_its_outcome() {
    let fake = Arc::new(FakeActuator::new());
    let mut sup = harness().actuator(fake.clone()).build();
    let critical = Some(SafetyReason::ThermalCritical);
    let first = step_trip(&mut sup, 90.0, true); // Apply spawned, no outcome yet
    let healthy = step_trip(&mut sup, 60.0, false);
    let again = step_trip(&mut sup, 90.0, true); // same episode: no new trip
    assert!(sup.brake_task.is_some(), "the apply really was spawned");
    assert_eq!(
        (first, healthy, again, sup.machine.trips_total()),
        (critical, None, None, 1)
    );
}

#[tokio::test]
async fn gate_busy_then_dropped_apply_leaves_trip_episode_closable() {
    let fake = Arc::new(FakeActuator::new());
    let mut sup = harness().actuator(fake.clone()).build();
    let permit = sup.gate.try_enter().unwrap();
    let critical = Some(SafetyReason::ThermalCritical);
    let first = step_trip(&mut sup, 90.0, true); // gate busy: Apply parked
    let dropped = step_trip(&mut sup, 60.0, false); // pending Apply dropped
    drop(permit);
    let healthy = step_trip(&mut sup, 60.0, false);
    let second = step_trip(&mut sup, 90.0, true); // gate free: Apply spawned
    assert!(
        sup.brake_task.is_some(),
        "the second apply really was spawned"
    );
    assert_eq!(
        (first, dropped, healthy, second, sup.machine.trips_total()),
        (critical, None, None, critical, 2)
    );
}

struct PanickingPublisher;

impl SensoryPublisher for PanickingPublisher {
    fn try_publish(&self, _mapping: &SensoryMapping) -> Result<(), PublishError> {
        panic!("publisher bug");
    }
}

#[tokio::test]
async fn panicking_publisher_on_a_non_evaluation_tick_does_not_unwind_step() {
    let mut sup = harness()
        .frames([fixtures::healthy_real(), fixtures::healthy_real()])
        .publisher(PanickingPublisher)
        .build();
    let _ = sup.step().await; // tick 1: evaluation tick
    let _ = sup.step().await; // tick 2: publish-only tick
    assert_eq!(sup.machine.snapshot().intent, BrakeIntent::None);
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
    block_release: Mutex<Option<mpsc::Receiver<()>>>,
    release_entered: Mutex<Option<mpsc::Sender<()>>>,
}

impl ControlledActuator {
    fn new(inner: Arc<FakeActuator>) -> Self {
        Self {
            inner,
            block_apply: Mutex::new(None),
            apply_entered: Mutex::new(None),
            panic_apply: Mutex::new(false),
            block_release: Mutex::new(None),
            release_entered: Mutex::new(None),
        }
    }

    fn block_next_apply(&self) -> (mpsc::Sender<()>, mpsc::Receiver<()>) {
        let (release_tx, release_rx) = mpsc::channel();
        let (entered_tx, entered_rx) = mpsc::channel();
        *self.block_apply.lock().unwrap() = Some(release_rx);
        *self.apply_entered.lock().unwrap() = Some(entered_tx);
        (release_tx, entered_rx)
    }

    fn block_next_release(&self) -> (mpsc::Sender<()>, mpsc::Receiver<()>) {
        let (release_tx, release_rx) = mpsc::channel();
        let (entered_tx, entered_rx) = mpsc::channel();
        *self.block_release.lock().unwrap() = Some(release_rx);
        *self.release_entered.lock().unwrap() = Some(entered_tx);
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
        let block = self.block_release.lock().unwrap().take();
        if let Some(rx) = block {
            if let Some(tx) = self.release_entered.lock().unwrap().take() {
                let _ = tx.send(());
            }
            let _ = rx.recv();
        }
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
                eval_interval_ms: 1_000,
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
                // Each step reads the clock once and advances one 100 ms tick.
                monotonic: {
                    let ms = Arc::new(AtomicU64::new(0));
                    Arc::new(move || ms.fetch_add(100, Ordering::SeqCst))
                },
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
fn joined(outcome: Outcome, state: SafetyState, intent: BrakeIntent, brake_engaged: bool) -> Tick {
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

    // Retried when the next 1 s eval deadline elapses (11th 100 ms tick
    // after the first evaluation), and still denied.
    let report = step_until(&mut sup, 12, |r| r.dispatched).await;
    assert_eq!(report.step, 11);
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

    // Evaluations happen at steps 1, 11, 21 → streak of 3 → Release.
    let report = step_until(&mut sup, 25, |r| r.dispatched).await;
    assert_eq!(
        (report.step, tick(&report)),
        (21, dispatching(S::Recovering, I::Release, true))
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

// ---- RM-1476: wall-clock eval deadline + release preemption ----

/// A tick far longer than the eval interval (slow acquire / large
/// `--step-interval-ms`) still evaluates every step, so a Critical frame
/// on step 2 trips on step 2 rather than on the 10th tick.
#[tokio::test]
async fn slow_ticks_evaluate_every_step_not_every_tenth() {
    let fake = Arc::new(FakeActuator::new());
    let mut sup = harness()
        .actuator(fake.clone())
        .frames([fixtures::healthy_real(), critical_temp()])
        .build();
    sup.monotonic = Arc::new({
        let ms = AtomicU64::new(0);
        move || ms.fetch_add(1_000, Ordering::SeqCst)
    });
    let first = sup.step().await;
    assert_eq!(first.snapshot.intent, I::None);
    let second = sup.step().await;
    assert_eq!(
        (second.step, second.snapshot.intent, second.dispatched),
        (2, I::Apply, true)
    );
}

/// Sub-deadline ticks must not evaluate; the deadline is wall-clock.
#[tokio::test]
async fn fast_ticks_wait_for_the_deadline() {
    let mut sup = harness()
        .frames([fixtures::healthy_real(), critical_temp()])
        .build();
    sup.monotonic = Arc::new({
        let ms = AtomicU64::new(0);
        move || ms.fetch_add(100, Ordering::SeqCst)
    });
    let _ = sup.step().await;
    let second = sup.step().await; // 100 ms later: publish only
    assert_eq!(second.snapshot.intent, I::None);
}

/// A release held in flight (gate busy, so telemetry fail-closes): policy flags it
/// superseded, and the instant it completes the restore is recorded
/// honestly and Apply is dispatched in the *same* step (no cadence wait,
/// no second release).
#[tokio::test]
async fn gate_busy_during_in_flight_release_reapplies_on_completion() {
    let fake = Arc::new(FakeActuator::new());
    fake.apply_emergency_brake(BRAKE_FRACTION).unwrap();
    let ctl = Arc::new(ControlledActuator::new(Arc::clone(&fake)));
    let (unblock, entered) = ctl.block_next_release();
    let mut sup = harness()
        .actuator(ctl)
        .frames([fixtures::healthy_real()])
        .build();
    sup.seed_startup_brake(Some(200), Some(400));
    sup.monotonic = Arc::new({
        let ms = AtomicU64::new(0);
        move || ms.fetch_add(1_000, Ordering::SeqCst)
    });
    let _ = step_until(&mut sup, 10, |r| r.dispatched).await;
    entered.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(sup.release_task.is_some());

    // The held release owns the NVML gate, so the next read is Busy and
    // fail-closes (NvmlUnavailable → brake desired).
    let report = sup.step().await;
    assert_eq!(report.source, TelemetrySource::NvmlUnavailable);
    assert!(sup.release_superseded.load(Ordering::Acquire));
    assert_eq!(fake.release_calls(), 0, "release still held, not restored");

    unblock.send(()).unwrap();
    settle(&sup).await;
    // Gate is free again and telemetry is still critical.
    sup.acquirer = Arc::new(ScriptedAcquirer::new([critical_temp()]));
    let report = sup.step().await;
    assert_eq!(report.actuation, Some(ActuatorOutcome::Released));
    assert_eq!(
        (report.snapshot.intent, report.dispatched),
        (I::Apply, true)
    );
    settle(&sup).await;
    let report = step_until(&mut sup, 5, |r| r.actuation.is_some()).await;
    assert_eq!(report.actuation, Some(ActuatorOutcome::Applied));
    assert_eq!(hw(&fake), hw_is(true, 2, 1));
}

/// A release worker that skipped the driver (superseded before it started)
/// is reconciled without recording an outcome: the flag clears, the brake
/// stays engaged, and no restore was issued.
#[tokio::test]
async fn skipped_release_worker_records_nothing() {
    let fake = Arc::new(FakeActuator::new());
    fake.apply_emergency_brake(BRAKE_FRACTION).unwrap();
    let mut sup = harness().actuator(fake.clone()).build();
    sup.seed_startup_brake(Some(200), Some(400));
    sup.release_superseded.store(true, Ordering::Release);
    sup.release_task = Some(tokio::task::spawn_blocking(|| {
        Err(ActuatorError::CommandFailed(RELEASE_SKIPPED.into()))
    }));
    settle(&sup).await;

    let outcome = sup.reconcile_finished(ActuationKind::Release).await;
    assert_eq!(outcome, None);
    assert!(!sup.release_superseded.load(Ordering::Acquire));
    assert!(sup.release_task.is_none());
    assert_eq!(
        (sup.snapshot().brake_engaged, hw(&fake)),
        (true, hw_is(true, 1, 0))
    );
}
