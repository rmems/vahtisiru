//! Actuation outcome/retry fault-injection tests (RM-1459).

use super::*;

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

    let outcome = sup.reconcile_finished(ActuationKind::Release, true).await;
    assert_eq!(outcome, None);
    assert!(!sup.release_superseded.load(Ordering::Acquire));
    assert!(sup.release_task.is_none());
    assert_eq!(
        (sup.snapshot().brake_engaged, hw(&fake)),
        (true, hw_is(true, 1, 0))
    );
}
