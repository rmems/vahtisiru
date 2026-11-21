//! Intent dispatch, gate-busy parking, and trip-episode tests.

use super::*;

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
