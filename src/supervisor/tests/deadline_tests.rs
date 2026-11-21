//! Acquisition-deadline and probe-verification tests (RM-1475).

use super::*;

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

// Fault 1b (RM-1475): the real probe against a SIGTERM-ignoring fake
// `nvidia-smi`, at the production deadline. Evaluation runs inside 1 s, fails
// closed, and SIGKILL frees the gate so the parked brake then lands.
#[cfg(unix)]
#[tokio::test]
async fn sigterm_ignoring_nvidia_smi_cannot_stall_evaluation_past_one_second() {
    let fake_smi = FakeBinary::install(IGNORES_SIGTERM);
    let bridge = Arc::new(HardwareBridge::with_health_program_for_test(
        Some(GpuTarget::from_uuid_for_test(
            "GPU-12345678-1234-1234-1234-123456789abc",
        )),
        fake_smi.path(),
    ));
    let fake = Arc::new(FakeActuator::new());
    let mut sup = harness().actuator(fake.clone()).acquirer(bridge).build();
    sup.config.acquire_deadline = ACQUIRE_DEADLINE;
    let parked = Tick {
        pending: Some(I::Apply),
        ..unavailable(S::TelemetryMissing, I::Apply, false)
    };

    let started = Instant::now();
    let report = sup.step().await;
    let elapsed = started.elapsed();
    assert!(elapsed < Duration::from_secs(1), "{elapsed:?}");
    assert_eq!(tick(&report), parked);
    let exceeded = sup
        .relay_metrics
        .lock()
        .unwrap()
        .acquire_deadline_exceeded_total;
    assert_eq!(exceeded, 1);

    // The probe is still dying: later ticks do not wait out another deadline.
    let started = Instant::now();
    assert_eq!(tick(&sup.step().await), parked);
    assert!(started.elapsed() < Duration::from_millis(500));

    wait_gate_free(&sup).await;
    let report = sup.step().await;
    assert_eq!(report.source, TelemetrySource::NvmlUnavailable);
    assert!(report.dispatched, "brake takes the freed gate first");
    settle(&sup).await;
    assert_eq!(hw(&fake), hw_is(true, 1, 0));
}

// A post-actuation re-read shares the step's budget: once it is spent no worker
// starts and the read fails closed at once.
#[tokio::test]
async fn spent_tick_budget_fails_closed_without_starting_a_worker() {
    let acquirer = Arc::new(ScriptedAcquirer::new([fixtures::healthy_real()]));
    let mut sup = harness().acquirer(acquirer.clone()).build();
    sup.tick_deadline = Some(Instant::now());

    let raw = sup.acquire_raw_with_timeout().await;

    assert_eq!(raw.source, TelemetrySource::NvmlUnavailable);
    assert_eq!(acquirer.calls(), 0);
    let exceeded = sup
        .relay_metrics
        .lock()
        .unwrap()
        .acquire_deadline_exceeded_total;
    assert_eq!(exceeded, 1);
}

#[tokio::test]
async fn step_longer_than_its_period_counts_one_overrun() {
    let acquirer = Arc::new(ScriptedAcquirer::new([fixtures::healthy_real()]));
    let (release, _entered) = acquirer.hang_until_released();
    let mut sup = harness().acquirer(acquirer.clone()).build();
    sup.config.cadence_ms = 1;

    sup.step().await;
    let overruns = sup.relay_metrics.lock().unwrap().loop_overruns_total;
    assert_eq!(overruns, 1);

    acquirer.unhang();
    release.send(()).unwrap();
}

#[tokio::test]
async fn software_only_steps_record_no_acquire_failures_or_overruns() {
    let mut sup = harness().software_only().build();
    for _ in 0..20 {
        sup.step().await;
    }
    let metrics = sup.relay_metrics.lock().unwrap().clone();
    assert_eq!(
        (
            metrics.acquire_deadline_exceeded_total,
            metrics.loop_overruns_total
        ),
        (0, 0)
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

// A release that settled during the sleep is verified by this step's own
// read: the verification folds into it instead of paying for a second read
// out of the same budget — two reads that fit individually but not together
// can no longer starve the verification into a spurious fail-closed re-brake.
#[tokio::test]
async fn settled_release_verifies_on_the_tick_read_without_a_second_acquire() {
    let fake = Arc::new(FakeActuator::new());
    fake.apply_emergency_brake(BRAKE_FRACTION).unwrap();
    // Each read fits inside the 40 ms deadline, but two never would.
    let acquirer = Arc::new(
        ScriptedAcquirer::new([fixtures::healthy_real()]).with_delay(Duration::from_millis(20)),
    );
    let mut sup = harness()
        .actuator(fake.clone())
        .acquirer(acquirer.clone())
        .build();
    sup.seed_startup_brake(Some(200), Some(400));
    // The release worker already finished during the sleep.
    sup.release_task = Some(tokio::task::spawn_blocking(|| Ok(())));
    settle(&sup).await;

    let report = sup.step().await;

    assert_eq!(report.actuation, Some(ActuatorOutcome::Released));
    assert_eq!(
        (acquirer.calls(), report.source),
        (1, TelemetrySource::Nvml),
        "the tick read doubles as the post-release verification"
    );
    assert_eq!(
        (
            report.snapshot.brake_engaged,
            report.snapshot.intent,
            report.dispatched
        ),
        (false, I::None, false),
        "healthy post-release telemetry must not re-brake"
    );
}
