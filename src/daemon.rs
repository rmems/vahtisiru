//! Supervisor loop used by the `vahtisiru` executable.
//!
//! Not part of the public library surface: Prometheus bind, process lock,
//! NVML acquisition, and privileged actuation stay here.

use clap::Parser;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time::sleep;

use crate::cpu::{self, RelayMetrics};
use crate::gpu::{GpuTarget, HardwareBridge, NvmlActuator};
use crate::nvml_gate::{NvmlGate, NvmlRunError};
use crate::publish::{
    AbsentPublisher, CorpusIpcPublisher, DEFAULT_IPC_ENDPOINT, DEFAULT_IPC_SESSION_ID, QueueConfig,
    QueueFullPolicy, SensoryPublisher, evaluate_then_try_publish, try_publish_isolated,
};
use crate::safety::{
    ActuatorError, ActuatorOutcome, BRAKE_FRACTION, BrakeIntent, PowerLimitObservation,
    SafetyActuator, SafetyMachine, SafetyPolicyConfig, SafetySnapshot, SafetyState,
    classify_power_limit,
};
use crate::shutdown::{
    InFlightActuation, InFlightJoin, SHUTDOWN_ACTUATOR_TIMEOUT, SHUTDOWN_METRICS_TIMEOUT,
    ShutdownActuation, ShutdownPlan, ShutdownReason, join_in_flight, plan_shutdown,
    shutdown_metrics_collector,
};
use crate::telemetry::{
    RawTelemetry, SampleClock, SampleValidity, TelemetryFrame, TelemetrySample, TelemetrySource,
    assess_with_clock, unix_now_ms,
};
use tokio::signal::unix::{SignalKind, signal};

#[derive(Debug)]
struct LockGuard(String);
impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Acquire the single-instance lock file, reclaiming it only when the recorded PID is dead.
fn try_acquire_lock(lock_path: &str) -> Result<LockGuard, String> {
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(lock_path)
        {
            Ok(mut file) => {
                if let Err(e) = writeln!(file, "{}", std::process::id()) {
                    return Err(format!("Failed to write PID to lock file {lock_path}: {e}"));
                }
                return Ok(LockGuard(lock_path.to_string()));
            }
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
                // A lock exists. Reclaim it ONLY when we can positively confirm
                // the recorded PID is dead. Any ambiguity (unreadable or
                // unparseable lock file) fails closed to preserve the
                // single-instance guarantee.
                let Some(recorded_pid) = std::fs::read_to_string(lock_path)
                    .ok()
                    .and_then(|content| content.trim().parse::<u32>().ok())
                else {
                    return Err(format!(
                        "Lock file {lock_path} exists but is unreadable/unparseable; refusing to start."
                    ));
                };

                if std::path::Path::new(&format!("/proc/{recorded_pid}")).exists() {
                    return Err(format!(
                        "Another instance is already active (PID: {recorded_pid})."
                    ));
                }

                // Stale lock from a dead PID: remove it and retry. If removal
                // fails, abort instead of spinning in a tight retry loop.
                if let Err(remove_err) = std::fs::remove_file(lock_path) {
                    return Err(format!(
                        "Failed to clear stale lock {lock_path}: {remove_err}"
                    ));
                }
            }
            Err(err) => return Err(format!("Failed to create lock file {lock_path}: {err}")),
        }
    }
}

const PROCESS_LOCK_PATH: &str = "/tmp/vahtisiru.lock";

/// Pre-rename lock path kept for upgrade safety: while a legacy `thalamic-relay`
/// process may still hold it, a new `vahtisiru` instance must not start, or two
/// supervisors could actuate the same GPU concurrently.
const LEGACY_PROCESS_LOCK_PATH: &str = "/tmp/thalamic_relay.lock";

/// Refuse to start while a pre-rename `thalamic-relay` instance is alive (or its
/// lock is ambiguous); clears the legacy lock once the recorded PID is dead.
fn check_legacy_lock(legacy_path: &str) -> Result<(), String> {
    let content = match std::fs::read_to_string(legacy_path) {
        Ok(content) => content,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(format!("Failed to read legacy lock {legacy_path}: {err}"));
        }
    };
    let Some(recorded_pid) = content.trim().parse::<u32>().ok() else {
        return Err(format!(
            "Legacy lock {legacy_path} is unreadable/unparseable; refusing to start."
        ));
    };
    if std::path::Path::new(&format!("/proc/{recorded_pid}")).exists() {
        return Err(format!(
            "A pre-rename thalamic-relay instance is already active (PID: {recorded_pid})."
        ));
    }
    if let Err(remove_err) = std::fs::remove_file(legacy_path) {
        return Err(format!(
            "Failed to clear stale legacy lock {legacy_path}: {remove_err}"
        ));
    }
    Ok(())
}

/// Prepared supervisor whose process lock remains owned by the caller until
/// the Tokio runtime has completed its bounded shutdown.
#[doc(hidden)]
pub struct SupervisorStart {
    cli: Cli,
    config: SafetyPolicyConfig,
    _lock_guard: LockGuard,
}

/// Run the read-only smoke when its Clap option is present. Returns `false`
/// when normal supervisor startup should continue.
#[doc(hidden)]
pub fn run_gpu_hardware_smoke_if_requested() -> Result<bool, Box<dyn std::error::Error>> {
    if !std::env::args_os().any(|arg| arg == "--gpu-hardware-smoke") {
        return Ok(false);
    }

    let cli = Cli::parse();
    if cli.gpu_hardware_smoke {
        run_gpu_hardware_smoke()?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Parse and validate configuration, then acquire the process lock before
/// starting any potentially blocking driver call.
#[doc(hidden)]
pub fn prepare() -> Result<SupervisorStart, Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let config = cli.safety_policy_config();
    // Reject contradictory operator inputs before NVML, locks, ports or workers.
    config.resolve(None)?;
    if cli.step_interval_ms > config.max_acquisition_interval_ms
        || cli.step_interval_ms.saturating_mul(10) > config.max_sample_age_ms
    {
        return Err("step interval exceeds safety acquisition limit or ten-tick evaluation exceeds sample-age limit".into());
    }

    // Block while a pre-rename instance still holds its legacy lock, then
    // acquire the process lock before any potentially blocking driver call.
    if let Err(msg) = check_legacy_lock(LEGACY_PROCESS_LOCK_PATH) {
        eprintln!("[relay] FATAL: {msg}");
        std::process::exit(1);
    }
    let lock_guard = match try_acquire_lock(PROCESS_LOCK_PATH) {
        Ok(guard) => guard,
        Err(msg) => {
            eprintln!("[relay] FATAL: {msg}");
            std::process::exit(1);
        }
    };
    Ok(SupervisorStart {
        cli,
        config,
        _lock_guard: lock_guard,
    })
}

/// Run the supervisor using a lock retained by the caller through runtime shutdown.
/// This is executable plumbing, not a reusable library API.
#[doc(hidden)]
pub async fn run(prepared: &mut SupervisorStart) -> Result<(), Box<dyn std::error::Error>> {
    let cli = &prepared.cli;
    let config = &prepared.config;
    let nvml_gate = NvmlGate::default();

    // Resolve the single GPU identity once, before constructing the adapters.
    // Both `--force-software-only` and the normal path bind the same target so
    // a real persistent brake stays addressable even under simulated telemetry.
    // On failure we continue with fail-closed telemetry and refuse mutation.
    let target = resolve_gpu_target(&nvml_gate).await;

    let bridge = HardwareBridge::new(target.clone());
    let actuator: Arc<dyn SafetyActuator> = Arc::new(NvmlActuator::new(target));
    // Read-only startup query also detects a persistent brake in software-only mode.
    let startup_actuator = Arc::clone(&actuator);
    let (current_w, default_w) = nvml_gate
        .run(NVML_TIMEOUT, move || {
            startup_actuator.query_power_limits_w()
        })
        .await
        .unwrap_or_else(|err| {
            tracing::warn!(?err, "startup NVML power-limit query unavailable");
            (None, None)
        });
    let policy = config.resolve(if cli.force_software_only {
        None
    } else {
        default_w.map(|w| w as f32)
    })?;
    let (power_warn, power_critical) = policy
        .power_limits_w()
        .map_or((None, None), |(w, c)| (Some(w), Some(c)));
    eprintln!(
        "[relay] effective_safety_policy temp_warn_c={} temp_critical_c={} power_warn_w={power_warn:?} power_critical_w={power_critical:?} power_source={} release_ok_streak={} max_sample_age_ms={} max_acquisition_interval_ms={} brake_fraction={}",
        config.temp_warn_c,
        config.temp_critical_c,
        policy.power_limit_source(),
        config.release_ok_streak,
        config.max_sample_age_ms,
        config.max_acquisition_interval_ms,
        policy.brake_fraction()
    );

    let metrics_addr = std::net::SocketAddr::new(cli.metrics_ip, 9000);
    cpu::init_telemetry(metrics_addr);

    let relay_metrics = Arc::new(Mutex::new(RelayMetrics::default()));
    let metrics_clone = Arc::clone(&relay_metrics);
    let (metrics_shutdown_tx, metrics_shutdown_rx) = tokio::sync::watch::channel(false);
    let metrics_task = tokio::spawn(async move {
        cpu::run_metrics_collector(metrics_clone, metrics_shutdown_rx).await;
    });

    println!("[relay] --- Vahtisiru ---");
    if cli.force_software_only {
        println!(
            "[relay] software-only telemetry (--force-software-only): documented idle estimates, not real GPU sensors"
        );
    } else {
        println!(
            "[relay] sensory + hardware-safety relay (probes NVML when available; NvmlUnavailable fail-closes)"
        );
    }

    let mut step_count: u64 = 0;
    let mut machine = SafetyMachine::with_policy(policy);
    let mut sample_clock = if cli.ipc_session_id.is_empty() {
        SampleClock::new()
    } else {
        SampleClock::with_session_id(cli.ipc_session_id.clone())
    };
    let publisher = build_publisher(cli);
    let mut warned_brake_held_sim = false;
    let mut warned_nvml_busy = false;
    let mut brake_task: Option<ActuationTask> = None;
    let mut release_task: Option<ActuationTask> = None;
    let mut pending_intent: Option<BrakeIntent> = None;

    seed_startup_brake(&mut machine, &relay_metrics, current_w, default_w);

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let mut sigterm = signal(SignalKind::terminate())?;

    let shutdown_reason = loop {
        step_count += 1;
        // A previously blocked emergency brake gets first claim on a gate
        // that became free during the sleep. Otherwise the next read can
        // repeatedly reacquire it and starve actuation.
        if pending_intent == Some(BrakeIntent::Apply)
            && dispatch_due_or_pending(
                &machine.snapshot(),
                &actuator,
                &nvml_gate,
                DispatchAttempt {
                    slots: (&mut brake_task, &mut release_task),
                    pending: &mut pending_intent,
                    due: false,
                },
            )
        {
            machine.apply_dispatched();
        }
        let raw = acquire_raw_with_timeout(
            &nvml_gate,
            &bridge,
            cli.force_software_only,
            &mut warned_nvml_busy,
        )
        .await;
        let telemetry =
            assess_with_clock(&raw, unix_now_ms(), cli.step_interval_ms, &mut sample_clock);

        let mut evaluated_this_iter = false;
        let mut actuation_failed_this_tick = false;
        let mut dispatch_due = false;

        if brake_task.as_ref().is_some_and(|task| task.is_finished()) {
            let task = brake_task.take().expect("finished brake task exists");
            match task.await {
                Ok(Ok(())) => {
                    reassess_after_actuation(
                        ActuatorOutcome::Applied,
                        PostActuationContext {
                            machine: &mut machine,
                            relay_metrics: &relay_metrics,
                            warned_brake_held_sim: &mut warned_brake_held_sim,
                            warned_nvml_busy: &mut warned_nvml_busy,
                            sample_clock: &mut sample_clock,
                            gate: &nvml_gate,
                            bridge: &bridge,
                            publisher: publisher.as_ref(),
                            force_software_only: cli.force_software_only,
                            cadence_ms: cli.step_interval_ms,
                        },
                    )
                    .await;
                    dispatch_due = true;
                    evaluated_this_iter = true;
                }
                Ok(Err(e)) => {
                    eprintln!("[relay] Emergency brake failed: {e}");
                    let snap = machine.record_actuator(ActuatorOutcome::ApplyFailed(e.to_string()));
                    store_safety(&relay_metrics, &snap, &mut warned_brake_held_sim);
                    actuation_failed_this_tick = true;
                }
                Err(e) => {
                    eprintln!("[relay] Brake task panicked: {e}");
                    let snap = machine.record_actuator(ActuatorOutcome::ApplyFailed(e.to_string()));
                    store_safety(&relay_metrics, &snap, &mut warned_brake_held_sim);
                    actuation_failed_this_tick = true;
                }
            }
        }
        if release_task.as_ref().is_some_and(|task| task.is_finished()) {
            let task = release_task.take().expect("finished release task exists");
            match task.await {
                Ok(Ok(())) => {
                    reassess_after_actuation(
                        ActuatorOutcome::Released,
                        PostActuationContext {
                            machine: &mut machine,
                            relay_metrics: &relay_metrics,
                            warned_brake_held_sim: &mut warned_brake_held_sim,
                            warned_nvml_busy: &mut warned_nvml_busy,
                            sample_clock: &mut sample_clock,
                            gate: &nvml_gate,
                            bridge: &bridge,
                            publisher: publisher.as_ref(),
                            force_software_only: cli.force_software_only,
                            cadence_ms: cli.step_interval_ms,
                        },
                    )
                    .await;
                    dispatch_due = true;
                    evaluated_this_iter = true;
                }
                Ok(Err(e)) => {
                    eprintln!("[relay] Brake release failed: {e}");
                    let snap =
                        machine.record_actuator(ActuatorOutcome::ReleaseFailed(e.to_string()));
                    store_safety(&relay_metrics, &snap, &mut warned_brake_held_sim);
                    actuation_failed_this_tick = true;
                }
                Err(e) => {
                    eprintln!("[relay] Brake release task panicked: {e}");
                    let snap =
                        machine.record_actuator(ActuatorOutcome::ReleaseFailed(e.to_string()));
                    store_safety(&relay_metrics, &snap, &mut warned_brake_held_sim);
                    actuation_failed_this_tick = true;
                }
            }
        }

        // First acquired frame is evaluated immediately (fail-closed startup).
        // Later evaluations keep the every-10-ticks cadence.
        // Do not spawn from the pre-telemetry snapshot: SoftwareFallback holds
        // rather than applies, which only classify_frame can decide.
        if !evaluated_this_iter && should_evaluate_tick(step_count, telemetry.source) {
            let prior_intent = machine.snapshot().intent;
            let (snap, pub_res) =
                evaluate_then_try_publish(&mut machine, &telemetry, publisher.as_ref());
            let _ = pub_res;
            store_safety(&relay_metrics, &snap, &mut warned_brake_held_sim);
            dispatch_due = should_dispatch_intent(
                step_count,
                prior_intent,
                snap.intent,
                actuation_failed_this_tick,
            );
        } else if !evaluated_this_iter {
            // Publication is outside the safety critical path and never awaited.
            let _ = try_publish_isolated(publisher.as_ref(), &telemetry.to_sensory_mapping());
        }

        if dispatch_due_or_pending(
            &machine.snapshot(),
            &actuator,
            &nvml_gate,
            DispatchAttempt {
                slots: (&mut brake_task, &mut release_task),
                pending: &mut pending_intent,
                due: dispatch_due,
            },
        ) {
            machine.apply_dispatched();
        }

        {
            let mut metrics = relay_metrics.lock().unwrap();
            metrics.telemetry_acquired_at = Some(telemetry.acquired_at);
        }

        print_dashboard(&telemetry, step_count, machine.snapshot().state);

        tokio::select! {
            _ = &mut ctrl_c => {
                break ShutdownReason::Sigint;
            }
            _ = sigterm.recv() => {
                break ShutdownReason::Sigterm;
            }
            _ = sleep(Duration::from_millis(cli.step_interval_ms)) => {}
        }
    };

    let _plan = perform_orderly_shutdown(
        shutdown_reason,
        &mut machine,
        &mut brake_task,
        &mut release_task,
        &relay_metrics,
        &mut warned_brake_held_sim,
        &metrics_shutdown_tx,
        metrics_task,
    )
    .await;
    eprintln!("[relay] runtime stopping; process lock remains held through shutdown");

    Ok(())
}

/// Perform an explicit, read-only validation of the same GPU adapters used by
/// the supervisor. This path does not start the daemon or mutate power limits.
fn run_gpu_hardware_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let list = std::process::Command::new("timeout")
        .args(["-k", "2s", "3s", "nvidia-smi", "-L"])
        .output()
        .map_err(|error| {
            std::io::Error::other(format!(
                "could not execute `timeout -k 2s 3s nvidia-smi -L`: {error}"
            ))
        })?;
    if !list.status.success() {
        return Err(std::io::Error::other(format!(
            "`nvidia-smi -L` exited with {}; NVIDIA driver/device unavailable",
            list.status
        ))
        .into());
    }
    print!("{}", String::from_utf8_lossy(&list.stdout));

    let target = GpuTarget::resolve()?;
    println!("GPU target UUID: {}", target.uuid());
    println!("Validation scope: read-only NVML telemetry and power limits; no actuation");

    let raw = HardwareBridge::new(Some(target.clone())).acquire_raw(false);
    if raw.source != TelemetrySource::Nvml {
        return Err(std::io::Error::other(format!(
            "resolved GPU {} did not produce NVML telemetry (source: {:?})",
            target.uuid(),
            raw.source
        ))
        .into());
    }
    let frame = assess_with_clock(&raw, unix_now_ms(), 100, &mut SampleClock::new());
    require_hardware_smoke_sample("GPU temperature", &frame.gpu_temp_c)?;
    require_hardware_smoke_sample("GPU board power", &frame.power_w)?;
    println!(
        "NVML telemetry on {}: temperature={:.1} C power={:.1} W",
        target.uuid(),
        frame.gpu_temp_c.value.expect("validated above"),
        frame.power_w.value.expect("validated above")
    );

    let (current_w, default_w) = NvmlActuator::new(Some(target.clone())).query_power_limits_w();
    let current_w = current_w.ok_or_else(|| {
        std::io::Error::other(format!(
            "current power limit could not be read from UUID {}",
            target.uuid()
        ))
    })?;
    let default_w = default_w.ok_or_else(|| {
        std::io::Error::other(format!(
            "default power limit could not be read from UUID {}",
            target.uuid()
        ))
    })?;
    if current_w == 0 || default_w == 0 {
        return Err(std::io::Error::other(format!(
            "NVML returned non-positive power limits for {}: current={current_w} W default={default_w} W",
            target.uuid()
        ))
        .into());
    }
    println!(
        "NVML power limits on {}: current={} W default={} W",
        target.uuid(),
        current_w,
        default_w
    );
    println!("GPU hardware smoke PASSED (read-only)");
    Ok(())
}

fn require_hardware_smoke_sample(
    label: &str,
    sample: &TelemetrySample<f32>,
) -> Result<(), std::io::Error> {
    match (sample.validity, sample.value) {
        (SampleValidity::Valid, Some(value)) if value.is_finite() => Ok(()),
        (SampleValidity::Valid, _) => Err(std::io::Error::other(format!(
            "{label} has no finite NVML value"
        ))),
        (validity, _) => Err(std::io::Error::other(format!(
            "{label} is not valid NVML telemetry ({validity:?})"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
async fn perform_orderly_shutdown(
    reason: ShutdownReason,
    machine: &mut SafetyMachine,
    brake_task: &mut Option<ActuationTask>,
    release_task: &mut Option<ActuationTask>,
    relay_metrics: &Arc<Mutex<RelayMetrics>>,
    warned_brake_held_sim: &mut bool,
    metrics_shutdown: &tokio::sync::watch::Sender<bool>,
    metrics_task: JoinHandle<()>,
) -> ShutdownPlan {
    let snap = machine.snapshot();
    let in_flight = InFlightActuation::from_tasks(brake_task.is_some(), release_task.is_some());
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
        let mut metrics = relay_metrics.lock().unwrap();
        cpu::record_shutdown(&mut metrics, &plan);
    }

    match plan.actuation {
        ShutdownActuation::Idle => {
            if let Some(task) = brake_task.take() {
                task.abort();
            }
            if let Some(task) = release_task.take() {
                task.abort();
            }
        }
        ShutdownActuation::AwaitApply => {
            if let Some(task) = brake_task.take() {
                match join_in_flight(task, SHUTDOWN_ACTUATOR_TIMEOUT).await {
                    InFlightJoin::Succeeded => {
                        let snap = machine.record_actuator(ActuatorOutcome::Applied);
                        store_safety(relay_metrics, &snap, warned_brake_held_sim);
                    }
                    InFlightJoin::Failed(e) | InFlightJoin::Panicked(e) => {
                        eprintln!("[relay] shutdown: in-flight apply did not succeed: {e}");
                        let snap = machine.record_actuator(ActuatorOutcome::ApplyFailed(e));
                        store_safety(relay_metrics, &snap, warned_brake_held_sim);
                    }
                    InFlightJoin::TimedOut => {
                        eprintln!(
                            "[relay] shutdown: in-flight apply timed out; leaving hardware unchanged"
                        );
                    }
                }
            }
            if let Some(task) = release_task.take() {
                task.abort();
            }
        }
        ShutdownActuation::AwaitAuthorizedRelease => {
            if let Some(task) = release_task.take() {
                match join_in_flight(task, SHUTDOWN_ACTUATOR_TIMEOUT).await {
                    InFlightJoin::Succeeded => {
                        let snap = machine.record_actuator(ActuatorOutcome::Released);
                        store_safety(relay_metrics, &snap, warned_brake_held_sim);
                    }
                    InFlightJoin::Failed(e) | InFlightJoin::Panicked(e) => {
                        eprintln!("[relay] shutdown: in-flight release did not succeed: {e}");
                        let snap = machine.record_actuator(ActuatorOutcome::ReleaseFailed(e));
                        store_safety(relay_metrics, &snap, warned_brake_held_sim);
                    }
                    InFlightJoin::TimedOut => {
                        eprintln!(
                            "[relay] shutdown: in-flight release timed out; leaving brake engaged"
                        );
                    }
                }
            }
            if let Some(task) = brake_task.take() {
                task.abort();
            }
        }
    }

    shutdown_metrics_collector(metrics_shutdown, metrics_task, SHUTDOWN_METRICS_TIMEOUT).await;
    plan
}

const NVML_TIMEOUT: Duration = Duration::from_secs(2);

/// Seed only a power cap that unambiguously matches this relay's brake.
/// The read-only startup query runs in software-only mode too, so simulated
/// telemetry cannot hide or authorize release of a real persistent brake.
fn seed_startup_brake(
    machine: &mut SafetyMachine,
    relay_metrics: &Arc<Mutex<RelayMetrics>>,
    current_w: Option<u32>,
    default_w: Option<u32>,
) {
    match classify_power_limit(current_w, default_w, BRAKE_FRACTION) {
        PowerLimitObservation::RelayOwnedBrake(m) => {
            eprintln!(
                "[relay] WARNING: GPU power limit {}W matches expected emergency brake \
                 target {}W (default {}W); will auto-release after Ok streak",
                m.current_w, m.expected_w, m.default_w
            );
            machine.seed_brake_applied();
            let seed = machine.snapshot();
            let mut metrics = relay_metrics.lock().unwrap();
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

async fn resolve_gpu_target(gate: &NvmlGate) -> Option<GpuTarget> {
    let resolution = match gate.run(NVML_TIMEOUT, GpuTarget::resolve).await {
        Ok(result) => result.map_err(|err| err.to_string()),
        Err(err) => Err(format!("{err:?}")),
    };
    match resolution {
        Ok(target) => {
            tracing::info!(
                gpu_uuid = target.uuid(),
                "resolved GPU target (NVML index 0)"
            );
            println!(
                "[relay] GPU target resolved: {} (NVML index 0)",
                target.uuid()
            );
            Some(target)
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                "GPU target resolution failed; continuing with fail-closed telemetry and refusing power-limit mutation"
            );
            eprintln!(
                "[relay] WARNING: GPU target resolution failed ({err}); telemetry fail-closes and power-limit mutation is refused"
            );
            None
        }
    }
}

fn should_evaluate_tick(step_count: u64, source: TelemetrySource) -> bool {
    step_count == 1 || step_count.is_multiple_of(10) || source == TelemetrySource::NvmlUnavailable
}

struct PostActuationContext<'a> {
    machine: &'a mut SafetyMachine,
    relay_metrics: &'a Arc<Mutex<RelayMetrics>>,
    warned_brake_held_sim: &'a mut bool,
    warned_nvml_busy: &'a mut bool,
    sample_clock: &'a mut SampleClock,
    gate: &'a NvmlGate,
    bridge: &'a HardwareBridge,
    publisher: &'a dyn SensoryPublisher,
    force_software_only: bool,
    cadence_ms: u64,
}

/// Reassess telemetry after either successful hardware mutation before
/// dispatching any further safety intent.
async fn reassess_after_actuation(outcome: ActuatorOutcome, ctx: PostActuationContext<'_>) {
    let snap = ctx.machine.record_actuator(outcome);
    store_safety(ctx.relay_metrics, &snap, ctx.warned_brake_held_sim);
    let raw = acquire_raw_with_timeout(
        ctx.gate,
        ctx.bridge,
        ctx.force_software_only,
        ctx.warned_nvml_busy,
    )
    .await;
    let frame = assess_with_clock(&raw, unix_now_ms(), ctx.cadence_ms, ctx.sample_clock);
    let (snap, pub_res) = evaluate_then_try_publish(ctx.machine, &frame, ctx.publisher);
    let _ = pub_res;
    store_safety(ctx.relay_metrics, &snap, ctx.warned_brake_held_sim);
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

/// Bounded single-flight acquisition for both per-tick and post-actuation reads.
async fn acquire_raw_with_timeout(
    gate: &NvmlGate,
    bridge: &HardwareBridge,
    force_software_only: bool,
    warned_busy: &mut bool,
) -> RawTelemetry {
    if force_software_only {
        return bridge.acquire_raw(true);
    }
    let bridge = bridge.clone();
    let result = gate
        .run(NVML_TIMEOUT, move || bridge.acquire_raw(false))
        .await;
    if should_log_acquisition_error(warned_busy, result.as_ref().err()) {
        tracing::warn!(error = ?result.as_ref().err(), "NVML acquisition unavailable; treating as unavailable");
    }
    match result {
        Ok(raw) => raw,
        Err(_) => RawTelemetry::nvml_unavailable(unix_now_ms()),
    }
}

fn build_publisher(cli: &Cli) -> Box<dyn SensoryPublisher> {
    if cli.ipc_disabled {
        println!("[relay] corpus-ipc: disabled; hardware safety is independent of Brainstem");
        return Box::new(AbsentPublisher);
    }
    let session_id = if cli.ipc_session_id.is_empty() {
        None
    } else {
        Some(cli.ipc_session_id.clone())
    };
    let queue_config = QueueConfig::new(cli.sensory_queue_capacity, cli.sensory_queue_full_policy)
        .expect("CLI parser already validated sensory queue capacity");
    match cli.ipc_endpoint.parse::<std::net::SocketAddr>() {
        Ok(endpoint) => match CorpusIpcPublisher::spawn(endpoint, session_id, queue_config) {
            Ok(publisher) => {
                println!(
                    "[relay] corpus-ipc: publishing IpcMessage::Stimuli to udp://{endpoint} \
                     (best-effort; safety does not wait; queue capacity={} policy={})",
                    queue_config.capacity(),
                    queue_config.policy(),
                );
                Box::new(publisher)
            }
            Err(err) => {
                eprintln!(
                    "[relay] corpus-ipc: publisher unavailable ({err}); continuing without Brainstem"
                );
                Box::new(AbsentPublisher)
            }
        },
        Err(err) => {
            eprintln!(
                "[relay] corpus-ipc: invalid --ipc-endpoint '{}': {err}; continuing without Brainstem",
                cli.ipc_endpoint
            );
            Box::new(AbsentPublisher)
        }
    }
}

fn store_safety(
    relay_metrics: &Arc<Mutex<RelayMetrics>>,
    snap: &SafetySnapshot,
    warned_brake_held_sim: &mut bool,
) {
    log_safety_snapshot(snap, warned_brake_held_sim);
    let mut metrics = relay_metrics.lock().unwrap();
    cpu::record_safety_snapshot(&mut metrics, snap);
}

/// Structured trip line; emitted once per deterministic trip episode.
fn log_safety_trip(snap: &SafetySnapshot) {
    if let Some(code) = snap.trip {
        eprintln!(
            "[relay] SAFETY TRIP reason={} state={} ({})",
            code.as_str(),
            snap.state.as_str(),
            snap.last_reason
        );
    }
}

fn log_safety_snapshot(snap: &SafetySnapshot, warned_brake_held_sim: &mut bool) {
    log_safety_trip(snap);
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

/// Handle to an in-flight privileged actuation attempt.
type ActuationTask = JoinHandle<Result<(), ActuatorError>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchResult {
    Spawned,
    GateBusy,
    AlreadyInFlight,
    NoIntent,
}

struct DispatchAttempt<'a> {
    slots: (&'a mut Option<ActuationTask>, &'a mut Option<ActuationTask>),
    pending: &'a mut Option<BrakeIntent>,
    due: bool,
}

/// Retry only intents skipped because another NVML worker held the gate.
/// Reconcile with the latest safety snapshot before any retry.
///
/// Returns `true` only when an apply worker was actually spawned; the caller
/// must then call [`SafetyMachine::apply_dispatched`].
#[must_use]
fn dispatch_due_or_pending(
    snap: &SafetySnapshot,
    actuator: &Arc<dyn SafetyActuator>,
    gate: &NvmlGate,
    attempt: DispatchAttempt<'_>,
) -> bool {
    if *attempt.pending != Some(snap.intent) {
        *attempt.pending = None;
    }
    if !attempt.due && attempt.pending.is_none() {
        return false;
    }
    let result = spawn_intent(snap, actuator, gate, attempt.slots);
    *attempt.pending = (result == DispatchResult::GateBusy).then_some(snap.intent);
    result == DispatchResult::Spawned && snap.intent == BrakeIntent::Apply
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

#[derive(Parser, Debug)]
#[command(
    name = "vahtisiru",
    version,
    about = "Sensory + deterministic hardware-safety relay (does not run neural computation)",
    long_about = "vahtisiru observes GPU telemetry, validates it, and evaluates an isolated thermal/power safety policy. It does not run a spiking neural network or own neural state.\n\nWithout --force-software-only it attempts NVML. Driver/device failure is NvmlUnavailable (fail-closed), not simulated idle. --force-software-only uses documented idle estimates tagged SoftwareFallback.\n\nBrake apply/release is best-effort: timeout + sudo -n nvidia-smi -pl on Linux (passwordless sudo for nvidia-smi). There is no control/query IPC; sensory publication is best-effort corpus-ipc UDP and Prometheus is served on :9000/metrics."
)]
struct Cli {
    /// Explicit thermal warning limit (C); not inferred from vendor capabilities.
    #[arg(long, default_value_t = 75.0, env = "VAHTISIRU_SAFETY_TEMP_WARN_C")]
    safety_temp_warn_c: f32,
    /// Explicit thermal critical limit (C).
    #[arg(long, default_value_t = 85.0, env = "VAHTISIRU_SAFETY_TEMP_CRITICAL_C")]
    safety_temp_critical_c: f32,
    /// Power warning override (W); requires a critical override.
    #[arg(long, env = "VAHTISIRU_SAFETY_POWER_WARN_W")]
    safety_power_warn_w: Option<f32>,
    /// Power critical override (W); cannot exceed a known device default.
    #[arg(long, env = "VAHTISIRU_SAFETY_POWER_CRITICAL_W")]
    safety_power_critical_w: Option<f32>,
    /// Consecutive healthy real evaluations required to release a brake.
    #[arg(long, default_value_t = 3, env = "VAHTISIRU_SAFETY_RELEASE_OK_STREAK")]
    safety_release_ok_streak: u32,
    /// Maximum sample age (ms); can tighten the 2000 ms telemetry limit.
    #[arg(
        long,
        default_value_t = 2000,
        env = "VAHTISIRU_SAFETY_MAX_SAMPLE_AGE_MS"
    )]
    safety_max_sample_age_ms: u64,
    /// Maximum declared acquisition interval (ms).
    #[arg(
        long,
        default_value_t = 100,
        env = "VAHTISIRU_SAFETY_MAX_ACQUISITION_INTERVAL_MS"
    )]
    safety_max_acquisition_interval_ms: u64,

    /// Prometheus metrics listen IP (port is always 9000 per compliance)
    #[arg(long, default_value = "127.0.0.1", env = "VAHTISIRU_METRICS_IP", value_parser = clap::value_parser!(std::net::IpAddr))]
    metrics_ip: std::net::IpAddr,

    /// Relay loop tick interval (ms); minimum 1 to prevent busy-looping
    #[arg(long, default_value_t = 100, env = "VAHTISIRU_STEP_INTERVAL_MS", value_parser = clap::value_parser!(u64).range(1..))]
    step_interval_ms: u64,

    /// Validate read-only NVIDIA hardware access and exit without starting the supervisor.
    #[arg(long)]
    gpu_hardware_smoke: bool,

    /// Force simulated idle telemetry (skip NVML). Documented estimates, not real sensors.
    /// Usable as a bare flag (`--force-software-only`) or with an explicit
    /// value (`--force-software-only=false` / `VAHTISIRU_FORCE_SOFTWARE_ONLY=false`).
    /// Distinct from NVML/driver failure, which is fail-closed `NvmlUnavailable`.
    #[arg(long, env = "VAHTISIRU_FORCE_SOFTWARE_ONLY", num_args = 0..=1, default_missing_value = "true", default_value_t = false, value_parser = clap::value_parser!(bool))]
    force_software_only: bool,

    /// UDP destination for canonical `corpus-ipc` `IpcMessage::Stimuli` datagrams.
    /// Fire-and-forget; Brainstem absence does not stall safety.
    #[arg(long, default_value = DEFAULT_IPC_ENDPOINT, env = "VAHTISIRU_IPC_ENDPOINT")]
    ipc_endpoint: String,

    /// Disable corpus-ipc publication. Hardware safety still evaluates.
    #[arg(long, env = "VAHTISIRU_IPC_DISABLED", num_args = 0..=1, default_missing_value = "true", default_value_t = false, value_parser = clap::value_parser!(bool))]
    ipc_disabled: bool,

    /// Session id stamped on each `StimulusBatch` (`session_id`). Empty keeps the telemetry-generated session id and its sequence.
    #[arg(long, default_value = DEFAULT_IPC_SESSION_ID, env = "VAHTISIRU_IPC_SESSION_ID")]
    ipc_session_id: String,

    /// Outbound sensory-queue capacity (frames). Finite; never unbounded.
    #[arg(
        long,
        default_value_t = QueueConfig::DEFAULT_CAPACITY,
        env = "VAHTISIRU_SENSORY_QUEUE_CAPACITY",
        value_parser = parse_sensory_queue_capacity
    )]
    sensory_queue_capacity: usize,

    /// Full-queue policy: `drop-oldest` (keep newest) or `reject-newest`.
    #[arg(
        long,
        default_value_t = QueueFullPolicy::DropOldest,
        env = "VAHTISIRU_SENSORY_QUEUE_FULL_POLICY"
    )]
    sensory_queue_full_policy: QueueFullPolicy,
}

fn parse_sensory_queue_capacity(s: &str) -> Result<usize, String> {
    let raw: u64 = s
        .parse()
        .map_err(|e| format!("invalid --sensory-queue-capacity: {e}"))?;
    let capacity =
        usize::try_from(raw).map_err(|_| "sensory-queue-capacity exceeds usize".to_string())?;
    QueueConfig::validate_capacity(capacity).map_err(|e| e.to_string())
}

impl Cli {
    fn safety_policy_config(&self) -> SafetyPolicyConfig {
        SafetyPolicyConfig {
            temp_warn_c: self.safety_temp_warn_c,
            temp_critical_c: self.safety_temp_critical_c,
            power_warn_w: self.safety_power_warn_w,
            power_critical_w: self.safety_power_critical_w,
            release_ok_streak: self.safety_release_ok_streak,
            max_sample_age_ms: self.safety_max_sample_age_ms,
            max_acquisition_interval_ms: self.safety_max_acquisition_interval_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        use crate::nvml_gate::NvmlRunError;

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
        let frame = crate::telemetry::assess(
            &crate::telemetry::fixtures::nvml_unavailable(),
            crate::telemetry::fixtures::NOW,
        );
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
        use crate::safety::FakeActuator;

        let gate = NvmlGate::default();
        let permit = gate.try_enter().unwrap();
        let mut machine = crate::safety::test_machine();
        let frame = crate::telemetry::assess(
            &crate::telemetry::fixtures::nvml_unavailable(),
            crate::telemetry::fixtures::NOW,
        );
        let snap = machine.evaluate(&frame);
        assert_eq!(snap.intent, BrakeIntent::Apply);
        let fake = Arc::new(FakeActuator::new());
        let actuator: Arc<dyn SafetyActuator> = fake.clone();
        let mut brake_task = None;
        let mut release_task = None;
        let mut pending = None;

        let spawned = dispatch_due_or_pending(
            &snap,
            &actuator,
            &gate,
            DispatchAttempt {
                slots: (&mut brake_task, &mut release_task),
                pending: &mut pending,
                due: true,
            },
        );
        assert!(!spawned, "a gate-blocked apply is not in flight");
        assert_eq!(pending, Some(BrakeIntent::Apply));
        assert!(brake_task.is_none());
        drop(permit);

        let spawned = dispatch_due_or_pending(
            &snap,
            &actuator,
            &gate,
            DispatchAttempt {
                slots: (&mut brake_task, &mut release_task),
                pending: &mut pending,
                due: false,
            },
        );
        assert!(spawned, "the retried apply worker was spawned");
        assert_eq!(pending, None);
        assert!(brake_task.take().unwrap().await.unwrap().is_ok());
        assert_eq!(fake.apply_calls(), 1);
    }

    #[tokio::test]
    async fn gate_blocked_intent_clears_when_safety_intent_changes() {
        use crate::safety::FakeActuator;

        let gate = NvmlGate::default();
        let fake = Arc::new(FakeActuator::new());
        let actuator: Arc<dyn SafetyActuator> = fake.clone();
        let mut machine = crate::safety::test_machine();
        let unavailable = crate::telemetry::assess(
            &crate::telemetry::fixtures::nvml_unavailable(),
            crate::telemetry::fixtures::NOW,
        );
        let _ = machine.evaluate(&unavailable);
        let healthy = crate::telemetry::assess(
            &crate::telemetry::fixtures::healthy_real(),
            crate::telemetry::fixtures::NOW,
        );
        let snap = machine.evaluate(&healthy);
        assert_eq!(snap.intent, BrakeIntent::None);
        let mut pending = Some(BrakeIntent::Apply);
        let mut brake_task = None;
        let mut release_task = None;

        let spawned = dispatch_due_or_pending(
            &snap,
            &actuator,
            &gate,
            DispatchAttempt {
                slots: (&mut brake_task, &mut release_task),
                pending: &mut pending,
                due: false,
            },
        );
        assert!(!spawned);
        assert_eq!(pending, None);
        assert!(brake_task.is_none());
        assert_eq!(fake.apply_calls(), 0);
    }

    /// Blocks review (PR #85, thread 4311678277): a gate-busy Apply that is
    /// later dropped was never in flight, so the trip episode can close and
    /// the next real critical reading is reported as a new trip.
    #[tokio::test]
    async fn gate_busy_then_dropped_apply_leaves_trip_episode_closable() {
        use crate::safety::{FakeActuator, SafetyReason};
        use crate::telemetry::{assess, fixtures};

        let frame = |temp_c| {
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(temp_c);
            assess(&raw, fixtures::NOW)
        };
        let gate = NvmlGate::default();
        let permit = gate.try_enter().unwrap();
        let actuator: Arc<dyn SafetyActuator> = Arc::new(FakeActuator::new());
        let (mut brake_task, mut release_task, mut pending) = (None, None, None);
        let mut machine = crate::safety::test_machine();
        let mut step = |machine: &mut SafetyMachine, temp_c, due| {
            let snap = machine.evaluate(&frame(temp_c));
            let attempt = DispatchAttempt {
                slots: (&mut brake_task, &mut release_task),
                pending: &mut pending,
                due,
            };
            if dispatch_due_or_pending(&snap, &actuator, &gate, attempt) {
                machine.apply_dispatched();
            }
            snap.trip
        };
        assert_eq!(
            step(&mut machine, 90.0, true),
            Some(SafetyReason::ThermalCritical)
        );
        assert_eq!(step(&mut machine, 60.0, false), None); // pending Apply dropped
        drop(permit);
        assert_eq!(step(&mut machine, 60.0, false), None);
        assert_eq!(
            step(&mut machine, 90.0, true),
            Some(SafetyReason::ThermalCritical)
        );
        assert_eq!(machine.trips_total(), 2);
        assert!(brake_task.is_some(), "the second apply really was spawned");
    }

    #[tokio::test]
    async fn busy_nvml_worker_fail_closes_tick_and_post_actuation_reads() {
        let gate = NvmlGate::default();
        let _permit = gate.try_enter().unwrap();
        let bridge = HardwareBridge::new(Some(GpuTarget::from_uuid_for_test(
            "GPU-12345678-1234-1234-1234-123456789abc",
        )));
        let mut warned_busy = false;
        let raw = acquire_raw_with_timeout(&gate, &bridge, false, &mut warned_busy).await;
        assert_eq!(raw.source, TelemetrySource::NvmlUnavailable);
        let mut clock = SampleClock::new();
        let frame = assess_with_clock(&raw, unix_now_ms(), 100, &mut clock);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Missing);
        assert_eq!(
            acquire_raw_with_timeout(&gate, &bridge, true, &mut warned_busy)
                .await
                .source,
            TelemetrySource::SoftwareFallback
        );
    }
    use clap::Parser;

    #[test]
    fn parses_custom_args_and_env_equiv() {
        // direct args
        let cli = Cli::try_parse_from([
            "vahtisiru",
            "--metrics-ip",
            "0.0.0.0",
            "--step-interval-ms",
            "50",
            "--force-software-only",
        ])
        .unwrap();
        assert_eq!(
            cli.metrics_ip,
            "0.0.0.0".parse::<std::net::IpAddr>().unwrap()
        );
        assert_eq!(cli.step_interval_ms, 50);
        assert!(cli.force_software_only);
    }

    #[test]
    fn parses_defaults() {
        let cli = Cli::try_parse_from(["vahtisiru"]).unwrap();
        assert_eq!(
            cli.metrics_ip,
            "127.0.0.1".parse::<std::net::IpAddr>().unwrap()
        );
        assert_eq!(cli.step_interval_ms, 100);
        assert!(!cli.force_software_only);
        assert_eq!(cli.ipc_endpoint, DEFAULT_IPC_ENDPOINT);
        assert!(!cli.ipc_disabled);
        assert_eq!(cli.ipc_session_id, DEFAULT_IPC_SESSION_ID);
        assert_eq!(cli.sensory_queue_capacity, QueueConfig::DEFAULT_CAPACITY);
        assert_eq!(cli.sensory_queue_full_policy, QueueFullPolicy::DropOldest);
    }

    #[test]
    fn parses_sensory_queue_config() {
        let cli = Cli::try_parse_from([
            "vahtisiru",
            "--sensory-queue-capacity",
            "8",
            "--sensory-queue-full-policy",
            "reject-newest",
        ])
        .unwrap();
        assert_eq!(cli.sensory_queue_capacity, 8);
        assert_eq!(cli.sensory_queue_full_policy, QueueFullPolicy::RejectNewest);
    }

    #[test]
    fn rejects_zero_queue_capacity() {
        assert!(Cli::try_parse_from(["vahtisiru", "--sensory-queue-capacity", "0"]).is_err());
    }

    #[test]
    fn rejects_unknown_queue_policy() {
        assert!(
            Cli::try_parse_from(["vahtisiru", "--sensory-queue-full-policy", "drop-random"])
                .is_err()
        );
    }

    #[test]
    fn parses_ipc_flags() {
        let cli = Cli::try_parse_from([
            "vahtisiru",
            "--ipc-endpoint",
            "127.0.0.1:9911",
            "--ipc-disabled",
            "--ipc-session-id",
            "lab-1",
        ])
        .unwrap();
        assert_eq!(cli.ipc_endpoint, "127.0.0.1:9911");
        assert!(cli.ipc_disabled);
        assert_eq!(cli.ipc_session_id, "lab-1");
    }

    #[test]
    fn parses_force_software_only_false() {
        let cli = Cli::try_parse_from(["vahtisiru", "--force-software-only=false"]).unwrap();
        assert!(!cli.force_software_only);
    }

    #[test]
    fn lock_guard_created_and_removed() {
        let lock_path = "/tmp/vahtisiru_test_created.lock";
        let _ = std::fs::remove_file(lock_path);
        let guard = try_acquire_lock(lock_path).unwrap();
        assert!(std::path::Path::new(lock_path).exists());
        drop(guard);
        assert!(!std::path::Path::new(lock_path).exists());
    }

    #[test]
    fn lock_guard_rejects_active_pid() {
        let lock_path = "/tmp/vahtisiru_test_active.lock";
        let _ = std::fs::remove_file(lock_path);
        std::fs::write(lock_path, std::process::id().to_string()).unwrap();
        let err = try_acquire_lock(lock_path).unwrap_err();
        assert!(
            err.contains("already active"),
            "expected active-instance error, got: {err}"
        );
        let _ = std::fs::remove_file(lock_path);
    }

    #[test]
    fn lock_guard_reclaims_stale_lock() {
        let lock_path = "/tmp/vahtisiru_test_stale.lock";
        let _ = std::fs::remove_file(lock_path);
        std::fs::write(lock_path, "0").unwrap();
        let guard = try_acquire_lock(lock_path).unwrap();
        let content = std::fs::read_to_string(lock_path).unwrap();
        assert_eq!(content.trim(), std::process::id().to_string());
        drop(guard);
        assert!(!std::path::Path::new(lock_path).exists());
    }

    #[test]
    fn legacy_lock_rejects_active_pid_and_reclaims_stale() {
        let lock_path = "/tmp/vahtisiru_test_legacy_active.lock";
        let _ = std::fs::remove_file(lock_path);
        std::fs::write(lock_path, std::process::id().to_string()).unwrap();
        let err = check_legacy_lock(lock_path).unwrap_err();
        assert!(
            err.contains("already active"),
            "expected active-instance error, got: {err}"
        );
        let _ = std::fs::remove_file(lock_path);

        std::fs::write(lock_path, "0").unwrap();
        check_legacy_lock(lock_path).unwrap();
        assert!(!std::path::Path::new(lock_path).exists());
    }

    #[test]
    fn prepared_process_lock_outlives_supervisor_future_and_runtime() {
        let path = "/tmp/vahtisiru_test_prepared_runtime.lock";
        let _ = std::fs::remove_file(path);
        let cli = Cli::parse_from(["vahtisiru"]);
        let prepared = SupervisorStart {
            config: cli.safety_policy_config(),
            cli,
            _lock_guard: try_acquire_lock(path).unwrap(),
        };
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let _ = &prepared;
        });
        assert!(try_acquire_lock(path).is_err());
        runtime.shutdown_timeout(Duration::from_millis(10));
        assert!(try_acquire_lock(path).is_err());
        drop(prepared);
        assert!(!std::path::Path::new(path).exists());
    }

    #[test]
    fn dashboard_shows_only_valid_present_as_live_hardware() {
        use crate::telemetry::{assess, fixtures};

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

    #[tokio::test]
    async fn orderly_shutdown_leaves_seeded_brake_and_stops_metrics() {
        let mut machine = crate::safety::test_machine();
        machine.seed_brake_applied();
        let mut brake_task = None;
        let mut release_task = None;
        let relay_metrics = Arc::new(Mutex::new(RelayMetrics::default()));
        let (tx, rx) = tokio::sync::watch::channel(false);
        let metrics_task = tokio::spawn(cpu::run_metrics_collector(Arc::clone(&relay_metrics), rx));
        let mut warned = false;
        let plan = perform_orderly_shutdown(
            ShutdownReason::Sigterm,
            &mut machine,
            &mut brake_task,
            &mut release_task,
            &relay_metrics,
            &mut warned,
            &tx,
            metrics_task,
        )
        .await;
        assert_eq!(plan.actuation, ShutdownActuation::Idle);
        assert!(plan.leave_brake_engaged);
        assert!(plan.unresolved_brake);
        assert!(machine.snapshot().brake_engaged);
        assert_eq!(
            relay_metrics.lock().unwrap().shutdown_reason,
            Some("sigterm")
        );
    }

    #[tokio::test]
    async fn orderly_shutdown_records_in_flight_apply_without_release() {
        use crate::safety::FakeActuator;
        use crate::telemetry::{assess, fixtures};

        let fake = Arc::new(FakeActuator::new());
        let fake_task = Arc::clone(&fake);
        let mut machine = crate::safety::test_machine();
        let mut raw = fixtures::healthy_real();
        raw.gpu_temp_c = Some(90.0);
        let snap = machine.evaluate(&assess(&raw, fixtures::NOW));
        assert_eq!(snap.intent, BrakeIntent::Apply);

        let mut brake_task = Some(tokio::task::spawn_blocking(move || {
            fake_task.apply_emergency_brake(BRAKE_FRACTION)
        }));
        tokio::time::sleep(Duration::from_millis(20)).await;

        let mut release_task = None;
        let relay_metrics = Arc::new(Mutex::new(RelayMetrics::default()));
        let (tx, rx) = tokio::sync::watch::channel(false);
        let metrics_task = tokio::spawn(cpu::run_metrics_collector(Arc::clone(&relay_metrics), rx));
        let mut warned = false;
        let plan = perform_orderly_shutdown(
            ShutdownReason::Sigint,
            &mut machine,
            &mut brake_task,
            &mut release_task,
            &relay_metrics,
            &mut warned,
            &tx,
            metrics_task,
        )
        .await;
        assert_eq!(plan.actuation, ShutdownActuation::AwaitApply);
        assert!(fake.is_engaged());
        assert!(machine.snapshot().brake_engaged);
        assert_eq!(fake.apply_calls(), 1);
        assert_eq!(fake.release_calls(), 0);
    }

    #[tokio::test]
    async fn orderly_shutdown_then_lock_guard_releases_file() {
        let lock_path = "/tmp/vahtisiru_test_shutdown.lock";
        let _ = std::fs::remove_file(lock_path);
        let guard = try_acquire_lock(lock_path).unwrap();
        assert!(std::path::Path::new(lock_path).exists());

        let mut machine = crate::safety::test_machine();
        let mut brake_task = None;
        let mut release_task = None;
        let relay_metrics = Arc::new(Mutex::new(RelayMetrics::default()));
        let (tx, rx) = tokio::sync::watch::channel(false);
        let metrics_task = tokio::spawn(cpu::run_metrics_collector(Arc::clone(&relay_metrics), rx));
        let mut warned = false;
        let _ = perform_orderly_shutdown(
            ShutdownReason::Sigint,
            &mut machine,
            &mut brake_task,
            &mut release_task,
            &relay_metrics,
            &mut warned,
            &tx,
            metrics_task,
        )
        .await;
        assert!(std::path::Path::new(lock_path).exists());
        drop(guard);
        assert!(!std::path::Path::new(lock_path).exists());
    }
}
