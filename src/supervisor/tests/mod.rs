use super::*;
#[cfg(unix)]
use crate::gpu::fake_nvidia_smi::{FakeBinary, IGNORES_SIGTERM};
use crate::gpu::{GpuTarget, NvmlActuator};
use crate::publish::{AbsentPublisher, IsolatedPublishQueue, PublishError};
use crate::safety::{FakeActuator, SafetyReason};
use crate::telemetry::{SensoryMapping, assess, fixtures, unix_now_ms};
use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc;
use std::time::Instant;

mod actuation_tests;
mod deadline_tests;
mod dispatch_tests;
mod misc_tests;
mod publish_tests;
mod shutdown_tests;

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
    /// Per-call sleep simulating a slow-but-obedient driver.
    delay: Duration,
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
            delay: Duration::ZERO,
        }
    }

    /// Every real read blocks the worker for `delay` before answering.
    fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
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
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
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
                acquire_deadline: FAST_TIMEOUT,
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
