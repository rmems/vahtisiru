//! Orderly shutdown tests: the brake state survives the stop path.

use super::*;

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
