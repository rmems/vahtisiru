//! Prometheus / tracing init and scrape-time gauges for the supervisor.
//!
//! Crate-private process plumbing: not part of the public `vahtisiru` API.
//! This module does not evaluate safety policy and does not publish sensory
//! frames. Binding the metrics listener is a process-global side effect.

use metrics::{counter, gauge};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

use crate::publish::register_sensory_queue_metrics_without_queue;
use crate::safety::{SafetyReason, SafetySnapshot, SafetyState};
use crate::shutdown::ShutdownPlan;
use crate::telemetry::UnixMillis;
use crate::time::freshness_seconds_monotonic;

/// Shared telemetry + safety state populated by the main loop.
/// Freshness is computed at scrape/export time from receive/emit time
/// ([`Self::telemetry_acquired_at`] / [`Self::telemetry_received_instant`]),
/// never from source wall time.
#[derive(Debug, Clone)]
pub struct RelayMetrics {
    /// Unix-ms timestamp of the last assessed frame, if any.
    pub telemetry_acquired_at: Option<UnixMillis>,
    pub telemetry_received_instant: Option<Instant>,
    /// Reported safety state (actuator-failure overlay included).
    pub safety_state: SafetyState,
    /// Policy classification before the actuator-failure overlay.
    pub policy_state: SafetyState,
    /// Last successful brake apply still claimed by the machine.
    pub brake_engaged: bool,
    /// Consecutive real Ok evaluations while the brake is engaged.
    pub hysteresis_ok_count: u32,
    pub shutdown_reason: Option<&'static str>,
    pub shutdown_unresolved_brake: bool,
    pub shutdown_unresolved_actuator: bool,
}

impl Default for RelayMetrics {
    fn default() -> Self {
        Self {
            telemetry_acquired_at: None,
            telemetry_received_instant: None,
            safety_state: SafetyState::TelemetryMissing,
            policy_state: SafetyState::TelemetryMissing,
            brake_engaged: false,
            hysteresis_ok_count: 0,
            shutdown_reason: None,
            shutdown_unresolved_brake: false,
            shutdown_unresolved_actuator: false,
        }
    }
}

#[cfg(test)]
use crate::time::freshness_seconds;

/// Copy a safety snapshot into shared metrics and increment event counters.
pub fn record_safety_snapshot(metrics: &mut RelayMetrics, snap: &SafetySnapshot) {
    metrics.safety_state = snap.state;
    metrics.policy_state = snap.policy_state;
    metrics.brake_engaged = snap.brake_engaged;
    metrics.hysteresis_ok_count = snap.hysteresis_ok_count;
    export_safety_gauges(
        snap.state,
        snap.policy_state,
        snap.brake_engaged,
        snap.hysteresis_ok_count,
    );
    if snap.transition.is_some() {
        counter!("safety_transitions_total").increment(1);
    }
    if snap.actuator_failed {
        counter!("safety_actuator_failures_total").increment(1);
    }
    if let Some(code) = snap.trip {
        counter!("safety_trips_total", "reason" => code.as_str()).increment(1);
    }
}

/// Pre-register every `safety_trips_total{reason}` series at zero so a scrape
/// can distinguish "never tripped" from "metric missing".
pub fn register_safety_trip_series() {
    for code in SafetyReason::ALL {
        counter!("safety_trips_total", "reason" => code.as_str()).increment(0);
    }
}

/// One-hot + numeric safety gauges. Safe to call from the collector refresh.
pub fn export_safety_gauges(
    state: SafetyState,
    policy_state: SafetyState,
    brake_engaged: bool,
    hysteresis_ok_count: u32,
) {
    for s in SafetyState::ALL {
        gauge!("safety_state", "state" => s.as_str()).set(if s == state { 1.0 } else { 0.0 });
        gauge!("safety_policy_state", "state" => s.as_str()).set(if s == policy_state {
            1.0
        } else {
            0.0
        });
    }
    gauge!("safety_state_id").set(f64::from(state.as_id()));
    gauge!("safety_policy_state_id").set(f64::from(policy_state.as_id()));
    gauge!("safety_brake_engaged").set(if brake_engaged { 1.0 } else { 0.0 });
    gauge!("safety_hysteresis_ok_count").set(f64::from(hysteresis_ok_count));
}

/// Record shutdown reason and unresolved brake/actuator gauges.
pub fn record_shutdown(metrics: &mut RelayMetrics, plan: &ShutdownPlan) {
    metrics.shutdown_reason = Some(plan.reason.as_str());
    metrics.shutdown_unresolved_brake = plan.unresolved_brake;
    metrics.shutdown_unresolved_actuator = plan.unresolved_actuator;
    counter!("shutdown_total", "reason" => plan.reason.as_str()).increment(1);
    gauge!("shutdown_unresolved_brake").set(if plan.unresolved_brake { 1.0 } else { 0.0 });
    gauge!("shutdown_unresolved_actuator").set(if plan.unresolved_actuator { 1.0 } else { 0.0 });
    gauge!("shutdown_brake_left_engaged").set(if plan.leave_brake_engaged { 1.0 } else { 0.0 });
}

/// Sets up our logging and metrics engines.
/// Binds the Prometheus HTTP listener on `metrics_addr`.
/// RUST_LOG (or future log-level arg) still controls tracing via env filter where applicable.
pub fn init_telemetry(metrics_addr: std::net::SocketAddr) {
    // 1. Initialize 'tracing' for our structured logs
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();

    // We use try_set_global_default in case it's already set
    let _ = tracing::subscriber::set_global_default(subscriber);

    // 2. Initialize the Prometheus exporter for our metrics.
    // Port is always 9000 (compliance); host/IP may be customized via CLI/env.
    PrometheusBuilder::new()
        .with_http_listener(metrics_addr)
        .install()
        .expect("Failed to install Prometheus recorder");

    register_sensory_queue_metrics_without_queue();
    register_safety_trip_series();

    info!(
        "Telemetry initialized. Prometheus metrics available on http://{}/metrics",
        metrics_addr
    );
}

/// Spawns a background task to track relay telemetry metrics.
/// Reads from shared state populated by the main loop. Exits when `shutdown`
/// is set to `true` (or the sender is dropped).
pub async fn run_metrics_collector(
    metrics: Arc<Mutex<RelayMetrics>>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) {
    info!("Starting Metrics Collector...");

    loop {
        if *shutdown.borrow() {
            info!("Metrics collector stopping");
            break;
        }

        let snapshot = {
            let guard = metrics.lock().unwrap();
            guard.clone()
        };

        gauge!("telemetry_freshness_s").set(freshness_seconds_monotonic(
            snapshot.telemetry_received_instant,
            Instant::now(),
        ));
        export_safety_gauges(
            snapshot.safety_state,
            snapshot.policy_state,
            snapshot.brake_engaged,
            snapshot.hysteresis_ok_count,
        );
        // Queue gauges are updated on enqueue/dequeue/drop in `publish`.
        // Capacity and policy series are registered when the queue is created.

        tokio::select! {
            result = shutdown.changed() => {
                if result.is_err() || *shutdown.borrow() {
                    info!("Metrics collector stopping");
                    break;
                }
            }
            _ = sleep(Duration::from_secs(2)) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::publish::{DropReason, IsolatedPublishQueue, QueueFullPolicy};
    use crate::safety::{BrakeIntent, SafetyState};
    use crate::telemetry::{assess, fixtures};

    #[test]
    fn relay_metrics_default_values() {
        let m = RelayMetrics::default();
        assert_eq!(m.telemetry_acquired_at, None);
        assert_eq!(m.telemetry_received_instant, None);
        assert_eq!(m.safety_state, SafetyState::TelemetryMissing);
        assert_eq!(m.policy_state, SafetyState::TelemetryMissing);
        assert!(!m.brake_engaged);
        assert_eq!(freshness_seconds(None, 1_000), 0.0);
    }

    #[test]
    fn freshness_seconds_grows_when_export_time_advances() {
        let acquired = Some(1_000);
        let early = freshness_seconds(acquired, 2_500);
        let later = freshness_seconds(acquired, 5_000);
        assert!((early - 1.5).abs() < f64::EPSILON);
        assert!((later - 4.0).abs() < f64::EPSILON);
        assert!(later > early);
    }

    #[test]
    fn metrics_queue_snapshot_names_are_stable() {
        let (queue, _rx) =
            IsolatedPublishQueue::bounded_with_policy(4, QueueFullPolicy::DropOldest).unwrap();
        let mapping = assess(&fixtures::healthy_real(), fixtures::NOW).to_sensory_mapping();
        queue.try_enqueue(mapping.clone()).unwrap();
        queue.try_enqueue(mapping.clone()).unwrap();
        queue.try_enqueue(mapping.clone()).unwrap();
        queue.try_enqueue(mapping.clone()).unwrap();
        queue.try_enqueue(mapping).unwrap();
        let text = queue.snapshot().prometheus_exposition();
        assert!(text.contains("sensory_queue_depth 4"));
        assert!(text.contains("sensory_queue_capacity 4"));
        assert!(text.contains("sensory_queue_enqueued_total 5"));
        assert!(text.contains("sensory_queue_dropped_total{reason=\"drop_oldest\"} 1"));
        assert!(text.contains("sensory_queue_full_policy{policy=\"drop_oldest\"} 1"));
        assert_eq!(DropReason::ALL.len(), 6);
    }

    #[test]
    fn freshness_seconds_ignores_regressing_source_wall_time() {
        let received_at = Some(5_000);
        let source_regressed = Some(1_000);
        assert_eq!(freshness_seconds(received_at, 5_000), 0.0);
        assert!((freshness_seconds(source_regressed, 5_000) - 4.0).abs() < f64::EPSILON);
        // Callers must pass receive time; source age is validity, not this gauge.
        assert!(freshness_seconds(received_at, 5_000) < freshness_seconds(source_regressed, 5_000));
    }

    #[test]
    fn record_safety_snapshot_copies_state_and_brake() {
        let mut metrics = RelayMetrics::default();
        let mut machine = crate::safety::test_machine();
        let mut raw = fixtures::healthy_real();
        raw.gpu_temp_c = Some(90.0);
        let snap = machine.evaluate(&assess(&raw, fixtures::NOW));
        assert_eq!(snap.intent, BrakeIntent::Apply);
        record_safety_snapshot(&mut metrics, &snap);
        assert_eq!(metrics.safety_state, SafetyState::CriticalBraked);
        assert!(!metrics.brake_engaged);
        assert_eq!(metrics.policy_state, SafetyState::CriticalBraked);
    }

    #[test]
    fn safety_trip_and_transition_counters_are_exported_with_reason_label() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            register_safety_trip_series();
            let mut metrics = RelayMetrics::default();
            let mut machine = crate::safety::test_machine();
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(90.0);
            let snap = machine.evaluate(&assess(&raw, fixtures::NOW));
            record_safety_snapshot(&mut metrics, &snap);
            // Held critical frame: no second trip event.
            let snap = machine.evaluate(&assess(&raw, fixtures::NOW));
            record_safety_snapshot(&mut metrics, &snap);
        });
        let text = handle.render();
        assert!(
            text.contains("safety_trips_total{reason=\"thermal_critical\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("safety_trips_total{reason=\"power_missing\"} 0"),
            "{text}"
        );
        assert!(text.contains("safety_transitions_total 1"), "{text}");
        assert!(
            text.contains("safety_state{state=\"critical_braked\"} 1"),
            "{text}"
        );
        assert_eq!(
            text.matches("safety_trips_total{").count(),
            SafetyReason::ALL.len()
        );
    }

    #[test]
    fn record_shutdown_copies_unresolved_flags() {
        use crate::shutdown::{InFlightActuation, ShutdownReason, plan_shutdown};

        let mut machine = crate::safety::test_machine();
        machine.seed_brake_applied();
        let plan = plan_shutdown(
            ShutdownReason::Sigterm,
            &machine.snapshot(),
            InFlightActuation::None,
        );
        let mut metrics = RelayMetrics::default();
        record_shutdown(&mut metrics, &plan);
        assert_eq!(metrics.shutdown_reason, Some("sigterm"));
        assert!(metrics.shutdown_unresolved_brake);
        assert!(!metrics.shutdown_unresolved_actuator);
    }

    #[test]
    fn freshness_seconds_is_zero_when_acquired_at_now() {
        assert_eq!(freshness_seconds(Some(5_000), 5_000), 0.0);
        assert_eq!(freshness_seconds(Some(8_000), 5_000), 0.0);
    }

    #[test]
    fn record_safety_snapshot_tracks_hysteresis_and_actuator_overlay() {
        use crate::safety::ActuatorOutcome;

        let mut metrics = RelayMetrics::default();
        let mut machine = crate::safety::test_machine();
        let mut raw = fixtures::healthy_real();
        raw.gpu_temp_c = Some(90.0);
        let _ = machine.evaluate(&assess(&raw, fixtures::NOW));
        let failed = machine.record_actuator(ActuatorOutcome::ApplyFailed("pl".into()));
        record_safety_snapshot(&mut metrics, &failed);
        assert_eq!(metrics.safety_state, SafetyState::ActuatorFailure);
        assert_eq!(metrics.policy_state, SafetyState::CriticalBraked);
        assert!(!metrics.brake_engaged);
        assert_eq!(metrics.hysteresis_ok_count, 0);
    }
}
