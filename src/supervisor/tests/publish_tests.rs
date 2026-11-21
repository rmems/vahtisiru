//! Publisher isolation tests: slow consumers and panics never affect safety.

use super::*;

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
