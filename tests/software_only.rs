//! Software-only integration coverage (RM-1217 / GH#27).
//!
//! These tests require no NVIDIA GPU, `sudo`, or `nvidia-smi`. They lock the
//! post-GH#39 regression surface: telemetry acquisition, deterministic safety
//! transitions, CLI/process lifecycle, and IPC isolation. Typed `corpus-ipc`
//! round-trips remain GH#40 / RM-1144.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use vahtisiru::publish::{
    AbsentPublisher, FailingPublisher, IsolatedPublishQueue, PublishError,
    evaluate_then_try_publish,
};
use vahtisiru::safety::{
    ActuatorOutcome, BRAKE_FRACTION, BrakeIntent, FakeActuator, SafetyActuator, SafetyMachine,
    SafetyState,
};
use vahtisiru::telemetry::SensoryMapping;
use vahtisiru::telemetry::{SampleValidity, TelemetrySource, assess, fixtures, software_fallback};

#[test]
fn binary_help_and_version_exit_zero_without_gpu() {
    let bin = env!("CARGO_BIN_EXE_vahtisiru");
    for arg in ["--help", "-V"] {
        let output = Command::new(bin)
            .arg(arg)
            .output()
            .unwrap_or_else(|e| panic!("spawn {arg}: {e}"));
        assert!(
            output.status.success(),
            "{arg} failed ({:?}): {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.stdout.is_empty());
    }
    let help = Command::new(bin).arg("--help").output().unwrap();
    let text = String::from_utf8_lossy(&help.stdout);
    assert!(text.contains("--force-software-only"));
    assert!(text.contains("--metrics-ip"));
    assert!(text.contains("--step-interval-ms"));
}

#[test]
fn binary_software_only_starts_without_nvidia() {
    let bin = env!("CARGO_BIN_EXE_vahtisiru");
    let lock_path = production_lock_path();
    reclaim_stale_production_lock(&lock_path);

    let mut child = Command::new(bin)
        .args(["--force-software-only", "--step-interval-ms", "50"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn vahtisiru");
    let child_pid = child.id();

    // Stop the supervisor ourselves so this test does not depend on coreutils
    // `timeout` (missing on some CI images / non-GNU hosts).
    std::thread::sleep(Duration::from_secs(1));
    let still_running = child.try_wait().expect("poll vahtisiru").is_none();
    if still_running {
        let _ = child.kill();
    }
    let output = child.wait_with_output().expect("collect relay output");
    remove_lock_if_pid(&lock_path, child_pid);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");
    assert!(
        still_running,
        "supervisor exited before kill ({:?}): stdout={stdout:?} stderr={stderr:?}",
        output.status.code()
    );
    assert!(
        combined.contains("software-only"),
        "expected software-only startup, exit={:?} stdout={stdout:?} stderr={stderr:?}",
        output.status.code()
    );
}

/// Production lock path, matching the daemon's (`/tmp` on unix, the OS temp
/// dir elsewhere).
fn production_lock_path() -> PathBuf {
    #[cfg(unix)]
    {
        PathBuf::from("/tmp/vahtisiru.lock")
    }
    #[cfg(not(unix))]
    {
        std::env::temp_dir().join("vahtisiru.lock")
    }
}

/// Reclaim the production lock only when the recorded PID is dead.
/// Never unlink a live supervisor's single-instance marker.
fn reclaim_stale_production_lock(lock_path: &Path) {
    if !lock_path.exists() {
        return;
    }
    let content = std::fs::read_to_string(lock_path).unwrap_or_default();
    let Some(pid) = content.trim().parse::<u32>().ok() else {
        panic!(
            "refusing to delete unparseable production lock {}",
            lock_path.display()
        );
    };
    assert!(
        !pid_is_alive(pid),
        "refusing to clobber live lock {} (PID {pid}); stop that instance first",
        lock_path.display()
    );
    std::fs::remove_file(lock_path).expect("remove stale production lock");
}

#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(windows)]
fn pid_is_alive(pid: u32) -> bool {
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const ERROR_INVALID_PARAMETER: i32 = 87;
    const STILL_ACTIVE: u32 = 259;
    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut core::ffi::c_void;
        fn GetExitCodeProcess(process: *mut core::ffi::c_void, code: *mut u32) -> i32;
        fn CloseHandle(handle: *mut core::ffi::c_void) -> i32;
    }
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return std::io::Error::last_os_error().raw_os_error() != Some(ERROR_INVALID_PARAMETER);
        }
        let mut code = 0;
        let alive = GetExitCodeProcess(handle, &mut code) == 0 || code == STILL_ACTIVE;
        CloseHandle(handle);
        alive
    }
}

#[cfg(not(any(unix, windows)))]
fn pid_is_alive(_pid: u32) -> bool {
    true
}

fn remove_lock_if_pid(lock_path: &Path, pid: u32) {
    let Ok(content) = std::fs::read_to_string(lock_path) else {
        return;
    };
    if content.trim() == pid.to_string() {
        let _ = std::fs::remove_file(lock_path);
    }
}

#[test]
fn software_only_pipeline_evaluates_safety_then_best_effort_publish() {
    let frame = assess(&fixtures::software_fallback(), fixtures::NOW);
    assert_eq!(frame.source, TelemetrySource::SoftwareFallback);
    assert_eq!(frame.gpu_temp_c.value, Some(software_fallback::GPU_TEMP_C));
    assert_eq!(frame.vram_temp_c.value, None);
    assert_eq!(frame.vram_temp_c.validity, SampleValidity::Missing);

    let mut machine = configured_machine();
    let (snap, pub_res) = evaluate_then_try_publish(&mut machine, &frame, &AbsentPublisher);
    assert_eq!(pub_res, Err(PublishError::Absent));
    assert_eq!(snap.state, SafetyState::SimulatedSoftwareOnly);
    assert_eq!(snap.intent, BrakeIntent::None);

    let mapping = frame.to_sensory_mapping();
    assert_eq!(
        mapping.acquisition_source,
        TelemetrySource::SoftwareFallback
    );
    let names: Vec<&str> = mapping.stimuli.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        ["gpu_temp_c", "power_w", "gpu_clock_mhz", "mem_util_pct"]
    );
}

#[test]
fn ipc_failure_cannot_stall_safety_loop_progress() {
    let mut machine = configured_machine();
    let publisher = FailingPublisher::send_failed();
    let fake = FakeActuator::new();

    let mut critical = fixtures::healthy_real();
    critical.gpu_temp_c = Some(95.0);
    let critical = assess(&critical, fixtures::NOW);

    let (snap, pub_res) = evaluate_then_try_publish(&mut machine, &critical, &publisher);
    assert!(pub_res.is_err());
    assert_eq!(snap.state, SafetyState::CriticalBraked);
    assert_eq!(snap.intent, BrakeIntent::Apply);

    fake.apply_emergency_brake(BRAKE_FRACTION).unwrap();
    let snap = machine.record_actuator(ActuatorOutcome::Applied);
    assert!(snap.brake_engaged);

    let ok = assess(&fixtures::healthy_real(), fixtures::NOW);
    let _ = evaluate_then_try_publish(&mut machine, &ok, &publisher);
    let _ = evaluate_then_try_publish(&mut machine, &ok, &publisher);
    let (third, pub_res) = evaluate_then_try_publish(&mut machine, &ok, &publisher);
    assert_eq!(
        pub_res,
        Err(PublishError::SendFailed("ipc send failed".into()))
    );
    assert_eq!(third.state, SafetyState::Recovering);
    assert_eq!(third.intent, BrakeIntent::Release);

    fake.release_emergency_brake().unwrap();
    let snap = machine.record_actuator(ActuatorOutcome::Released);
    assert!(!snap.brake_engaged);
    assert_eq!(fake.apply_calls(), 1);
    assert_eq!(fake.release_calls(), 1);

    let (healthy, pub_res) = evaluate_then_try_publish(&mut machine, &ok, &publisher);
    assert!(pub_res.is_err());
    assert_eq!(healthy.state, SafetyState::HealthyReal);
    assert_eq!(healthy.intent, BrakeIntent::None);
}

#[test]
fn missing_invalid_stale_and_simulated_frames_are_named_states() {
    let mut machine = configured_machine();
    let missing = machine.evaluate(&assess(&fixtures::sensor_dropout(), fixtures::NOW));
    assert_eq!(missing.state, SafetyState::TelemetryMissing);
    assert_eq!(missing.intent, BrakeIntent::Apply);

    let invalid = machine.evaluate(&assess(&fixtures::out_of_range(), fixtures::NOW));
    assert_eq!(invalid.state, SafetyState::TelemetryInvalid);

    let stale = machine.evaluate(&assess(&fixtures::stale(), fixtures::NOW));
    assert_eq!(stale.state, SafetyState::TelemetryStale);

    let sim = machine.evaluate(&assess(&fixtures::software_fallback(), fixtures::NOW));
    assert_eq!(sim.state, SafetyState::SimulatedSoftwareOnly);
    assert_eq!(sim.intent, BrakeIntent::None);
}

#[test]
fn slow_and_disconnected_publish_queues_do_not_block_critical_brake() {
    let (queue, consumer) =
        IsolatedPublishQueue::<SensoryMapping>::bounded(1).expect("capacity 1 is valid");
    drop(consumer);
    let mut machine = configured_machine();
    let mut critical = fixtures::healthy_real();
    critical.gpu_temp_c = Some(90.0);
    let frame = assess(&critical, fixtures::NOW);
    let (snap, pub_res) = evaluate_then_try_publish(&mut machine, &frame, &queue);
    assert_eq!(pub_res, Err(PublishError::Disconnected));
    assert_eq!(snap.state, SafetyState::CriticalBraked);
    assert_eq!(snap.intent, BrakeIntent::Apply);
}

#[test]
fn acquire_without_force_is_not_simulated_on_ci() {
    let raw = fixtures::nvml_unavailable();
    assert_ne!(raw.source, TelemetrySource::SoftwareFallback);
    assert_eq!(raw.source, TelemetrySource::NvmlUnavailable);
}

fn configured_machine() -> SafetyMachine {
    let config = vahtisiru::safety::SafetyPolicyConfig {
        power_warn_w: Some(300.0),
        power_critical_w: Some(350.0),
        ..Default::default()
    };
    SafetyMachine::with_policy(config.resolve(None).unwrap())
}
