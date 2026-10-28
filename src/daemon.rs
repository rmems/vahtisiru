//! Process plumbing for the `vahtisiru` executable.
//!
//! Not part of the public library surface: CLI, process lock, Prometheus
//! bind, NVML target resolution, and signal handling stay here. The loop body
//! itself is [`crate::supervisor::Supervisor::step`], which takes every
//! hardware/IPC dependency through a trait so it is testable without a GPU.

use clap::Parser;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;

use crate::actuation_smoke::{
    ActuationSmokeBackend, run_actuation_smoke, sudo_permission_check_args,
};
use crate::cpu::{self, RelayMetrics};
use crate::gpu::{GpuTarget, HardwareBridge, NvmlActuator, is_valid_gpu_uuid};
use crate::nvml_gate::NvmlGate;
use crate::publish::{
    AbsentPublisher, CorpusIpcPublisher, DEFAULT_IPC_ENDPOINT, DEFAULT_IPC_SESSION_ID, QueueConfig,
    QueueFullPolicy, SensoryPublisher,
};
use crate::safety::{
    SafetyActuator, SafetyMachine, SafetyPolicy, SafetyPolicyConfig, SafetyStatus,
    instant_status_with_policy,
};
use crate::shutdown::ShutdownReason;
use crate::supervisor::{NVML_TIMEOUT, Supervisor, SupervisorConfig, SupervisorParts};
use crate::telemetry::{
    SampleClock, SampleValidity, TelemetrySample, TelemetrySource, assess_with_clock, unix_now_ms,
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

/// Run the read-only or maintainer-only actuation smoke when its Clap option
/// is present. Returns `false` when normal supervisor startup should continue.
#[doc(hidden)]
pub fn run_gpu_hardware_smoke_if_requested() -> Result<bool, Box<dyn std::error::Error>> {
    if !std::env::args_os()
        .any(|arg| arg == "--gpu-hardware-smoke" || arg == "--gpu-hardware-actuation-smoke")
    {
        return Ok(false);
    }

    let cli = Cli::parse();
    if cli.gpu_hardware_smoke {
        run_gpu_hardware_smoke()?;
        Ok(true)
    } else if cli.gpu_hardware_actuation_smoke {
        let expected_uuid = cli
            .gpu_hardware_actuation_uuid
            .clone()
            .ok_or("--gpu-hardware-actuation-smoke requires --gpu-hardware-actuation-uuid")?;
        run_gpu_hardware_actuation_smoke(&cli, &expected_uuid)?;
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
    // Reject contradictory operator inputs before NVML, locks, ports or workers.
    let config = cli.validated_safety_policy_config()?;

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

    let bridge = Arc::new(HardwareBridge::new(target.clone()));
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

    let sample_clock = if cli.ipc_session_id.is_empty() {
        SampleClock::new()
    } else {
        SampleClock::with_session_id(cli.ipc_session_id.clone())
    };
    let mut supervisor = Supervisor::new(
        SupervisorConfig {
            force_software_only: cli.force_software_only,
            cadence_ms: cli.step_interval_ms,
            nvml_timeout: NVML_TIMEOUT,
            dashboard: true,
        },
        SupervisorParts {
            machine: SafetyMachine::with_policy(policy),
            sample_clock,
            publisher: build_publisher(cli),
            actuator,
            acquirer: bridge,
            gate: nvml_gate,
            clock: Arc::new(unix_now_ms),
            relay_metrics,
        },
    );
    supervisor.seed_startup_brake(current_w, default_w);

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let mut sigterm = signal(SignalKind::terminate())?;

    let shutdown_reason = loop {
        let report = supervisor.step().await;
        tracing::debug!(
            step = report.step,
            source = ?report.source,
            state = report.snapshot.state.as_str(),
            intent = ?report.snapshot.intent,
            actuation = ?report.actuation,
            in_flight = report.in_flight.as_str(),
            dispatched = report.dispatched,
            pending_intent = ?report.pending_intent,
            "supervisor step"
        );

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

    let _plan = supervisor
        .shutdown(shutdown_reason, &metrics_shutdown_tx, metrics_task)
        .await;
    eprintln!("[relay] runtime stopping; process lock remains held through shutdown");

    Ok(())
}

/// Perform an explicit, read-only validation of the same GPU adapters used by
/// the supervisor. This path does not start the daemon or mutate power limits.
fn run_gpu_hardware_smoke() -> Result<(), Box<dyn std::error::Error>> {
    list_nvidia_devices()?;

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

/// Run `timeout -k 2s 3s nvidia-smi -L` and echo it, so the job log records
/// the driver's own view of the device inventory before any NVML call.
fn list_nvidia_devices() -> Result<(), Box<dyn std::error::Error>> {
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
    Ok(())
}

/// Production [`ActuationSmokeBackend`]: the same `GpuTarget`, `NvmlActuator`,
/// and `HardwareBridge` the supervisor uses, plus a `sudo -l` permission probe.
struct NvmlActuationSmokeBackend {
    target: GpuTarget,
    actuator: NvmlActuator,
    bridge: HardwareBridge,
    policy: SafetyPolicy,
    clock: SampleClock,
    sample_cadence_ms: u64,
}

impl ActuationSmokeBackend for NvmlActuationSmokeBackend {
    fn resolved_uuid(&self) -> &str {
        self.target.uuid()
    }

    fn planned_power_limit_command(&self, limit_w: u32) -> Option<Vec<String>> {
        self.actuator.planned_command(limit_w)
    }

    fn check_sudo_permission(&self, limit_w: u32) -> Result<(), String> {
        let args = sudo_permission_check_args(self.target.uuid(), limit_w);
        let status = std::process::Command::new("timeout")
            .args(&args)
            .stdout(std::process::Stdio::null())
            .status()
            .map_err(|e| format!("could not execute `timeout {}`: {e}", args.join(" ")))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "`sudo -n -l -- nvidia-smi -i {} -pl {limit_w}` exited with {status}; the sudoers rule must permit this exact command non-interactively",
                self.target.uuid()
            ))
        }
    }

    fn actuator(&self) -> &dyn SafetyActuator {
        &self.actuator
    }

    fn observe_healthy_sample(&mut self) -> Result<(), String> {
        std::thread::sleep(Duration::from_millis(self.sample_cadence_ms));
        let raw = self.bridge.acquire_raw(false);
        let frame = assess_with_clock(&raw, unix_now_ms(), self.sample_cadence_ms, &mut self.clock);
        match instant_status_with_policy(&frame, &self.policy) {
            (SafetyStatus::Ok, false) => {
                println!(
                    "  sample: temperature={:?} C power={:?} W source={:?}",
                    frame.gpu_temp_c.value, frame.power_w.value, frame.source
                );
                Ok(())
            }
            (SafetyStatus::Ok, true) => Err("sample is simulated, not real NVML".into()),
            (status, _) => Err(format!("{status:?}")),
        }
    }
}

/// Maintainer-only hardware actuation smoke (RM-1909). Applies and releases the
/// relay brake on the operator-confirmed GPU through the production actuator,
/// verifying every step. See `docs/gpu-hardware-validation.md`.
fn run_gpu_hardware_actuation_smoke(
    cli: &Cli,
    expected_uuid: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !is_valid_gpu_uuid(expected_uuid) {
        return Err(format!(
            "--gpu-hardware-actuation-uuid {expected_uuid:?} is not a physical NVML GPU UUID (GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)"
        )
        .into());
    }
    if cli.force_software_only {
        return Err("actuation smoke requires real NVML; --force-software-only is refused".into());
    }
    // Same operator-input validation as `prepare`, before any lock or NVML
    // call: a step interval above the acquisition limit would brake the device
    // and then classify every streak sample as stale.
    let config = cli.validated_safety_policy_config()?;
    if let Ok(sha) = std::env::var("GITHUB_SHA") {
        println!("Trusted commit: {sha}");
    }
    // Take the supervisor's single-instance lock before touching the device.
    // A running relay drives the same brake target; letting the smoke apply
    // and release beneath it could lift a real brake mid-fault. Held until
    // the smoke returns, including through cleanup.
    check_legacy_lock(LEGACY_PROCESS_LOCK_PATH)
        .map_err(|msg| format!("{msg} Stop the supervisor before the actuation smoke."))?;
    let _lock_guard = try_acquire_lock(PROCESS_LOCK_PATH).map_err(|msg| {
        format!("{msg} Stop the running vahtisiru supervisor before the actuation smoke.")
    })?;
    println!("Process lock held: {PROCESS_LOCK_PATH} (no supervisor is running)");
    list_nvidia_devices()?;

    let target = GpuTarget::resolve()?;
    let actuator = NvmlActuator::new(Some(target.clone()));
    // Resolve the policy against the device default like the supervisor does,
    // so "healthy" means the same thing here as in production.
    let (_, default_w) = actuator.query_power_limits_w();
    let policy = config.resolve(default_w.map(|w| w as f32))?;
    println!(
        "Validation scope: ACTUATION — applies and releases the relay brake on {} (release_ok_streak={})",
        target.uuid(),
        cli.safety_release_ok_streak
    );

    let mut backend = NvmlActuationSmokeBackend {
        bridge: HardwareBridge::new(Some(target.clone())),
        target,
        actuator,
        policy,
        clock: SampleClock::new(),
        sample_cadence_ms: cli.step_interval_ms,
    };
    let mut stdout = io::stdout().lock();
    let report = run_actuation_smoke(
        &mut backend,
        expected_uuid,
        cli.safety_release_ok_streak,
        &mut stdout,
    )?;
    println!(
        "GPU actuation smoke PASSED: uuid={} pre_current={} W pre_default={} W apply_target={} W applied={} W healthy_streak={} released={} W cleanup=not_needed",
        report.uuid,
        report.pre_current_w,
        report.pre_default_w,
        report.apply_target_w,
        report.applied_current_w,
        report.healthy_streak,
        report.released_current_w
    );
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
    #[arg(long, conflicts_with = "gpu_hardware_actuation_smoke")]
    gpu_hardware_smoke: bool,

    /// MAINTAINER ONLY: apply and release the relay brake on real hardware, then exit.
    /// Mutates the GPU power limit. Requires --gpu-hardware-actuation-uuid.
    #[arg(long, requires = "gpu_hardware_actuation_uuid")]
    gpu_hardware_actuation_smoke: bool,

    /// Explicit confirmation of the GPU UUID the actuation smoke may mutate.
    /// Must equal the UUID NVML index 0 resolves to.
    #[arg(
        long,
        requires = "gpu_hardware_actuation_smoke",
        value_name = "GPU-UUID"
    )]
    gpu_hardware_actuation_uuid: Option<String>,

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
    /// The policy config, after rejecting contradictory operator inputs and a
    /// step interval the safety policy cannot keep fresh. Shared by the
    /// supervisor and the actuation smoke so the two cannot drift.
    fn validated_safety_policy_config(
        &self,
    ) -> Result<SafetyPolicyConfig, Box<dyn std::error::Error>> {
        let config = self.safety_policy_config();
        config.resolve(None)?;
        if self.step_interval_ms > config.max_acquisition_interval_ms
            || self.step_interval_ms.saturating_mul(10) > config.max_sample_age_ms
        {
            return Err("step interval exceeds safety acquisition limit or ten-tick evaluation exceeds sample-age limit".into());
        }
        Ok(config)
    }

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

    #[tokio::test]
    async fn orderly_shutdown_then_lock_guard_releases_file() {
        let lock_path = "/tmp/vahtisiru_test_shutdown.lock";
        let _ = std::fs::remove_file(lock_path);
        let guard = try_acquire_lock(lock_path).unwrap();
        assert!(std::path::Path::new(lock_path).exists());

        // Production adapters with no GPU target: NvmlActuator refuses to
        // mutate and HardwareBridge fail-closes. Nothing here touches hardware.
        let relay_metrics = Arc::new(Mutex::new(RelayMetrics::default()));
        let mut supervisor = Supervisor::new(
            SupervisorConfig {
                force_software_only: true,
                cadence_ms: 100,
                nvml_timeout: NVML_TIMEOUT,
                dashboard: false,
            },
            SupervisorParts {
                machine: crate::safety::test_machine(),
                sample_clock: SampleClock::new(),
                publisher: Box::new(AbsentPublisher),
                actuator: Arc::new(NvmlActuator::new(None)),
                acquirer: Arc::new(HardwareBridge::new(None)),
                gate: NvmlGate::default(),
                clock: Arc::new(unix_now_ms),
                relay_metrics: Arc::clone(&relay_metrics),
            },
        );
        let (tx, rx) = tokio::sync::watch::channel(false);
        let metrics_task = tokio::spawn(cpu::run_metrics_collector(Arc::clone(&relay_metrics), rx));
        let _ = supervisor
            .shutdown(ShutdownReason::Sigint, &tx, metrics_task)
            .await;
        assert!(std::path::Path::new(lock_path).exists());
        drop(guard);
        assert!(!std::path::Path::new(lock_path).exists());
    }
}
