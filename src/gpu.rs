//! GPU telemetry acquisition and privileged NVML / `nvidia-smi` actuation.
//!
//! Crate-private: the `vahtisiru` executable uses this adapter; it is not
//! public library API.
//!
//! Raw NVML acquisition lives on [`HardwareBridge`]; the `nvidia-smi`
//! brake/release backend lives on [`NvmlActuator`]. Validation, normalization,
//! freshness, and provenance live in [`crate::telemetry`]. Deterministic safety
//! classification, hysteresis, and the [`crate::safety::SafetyActuator`]
//! boundary live in [`crate::safety`].
//!
//! [`NvmlActuator`] is a hardware *adapter*: it implements
//! [`crate::safety::SafetyActuator`] against NVML / `nvidia-smi`, but defines
//! none of the generic safety semantics (thresholds, hysteresis, fail-closed
//! policy) — those belong to [`crate::safety`].
//!
//! Apply/release uses `sudo -n nvidia-smi -i <UUID> -pl` (passwordless sudo,
//! Linux). Every NVML read and every `nvidia-smi` command is scoped to a
//! single [`GpuTarget`] resolved once at startup (NVML index 0), so the relay
//! never silently drifts to another device. Acquisition without
//! `--force-software-only` that cannot open NVML is
//! [`crate::telemetry::TelemetrySource::NvmlUnavailable`], not simulated idle.

use crate::safety::{ActuatorError, SafetyActuator};
#[cfg(test)]
use crate::safety::{SafetyStatus, instant_status_with_policy};
#[cfg(test)]
use crate::telemetry::DEFAULT_ACQUISITION_CADENCE_MS;
use crate::telemetry::{RawTelemetry, TelemetrySource, TimestampOrigin, unix_now_ms};
#[cfg(test)]
use crate::telemetry::{SampleClock, TelemetryFrame, assess_with_clock};
use lazy_static::lazy_static;
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};

lazy_static! {
    static ref NVML: Option<Nvml> = Nvml::init().ok();
}

// ── GPU identity (GH#70 / RM-1850) ──────────────────────────────────

/// A single validated GPU identity that scopes every NVML read and every
/// `nvidia-smi` power-limit command for the life of the process.
///
/// Resolved exactly once at startup ([`GpuTarget::resolve`]) from NVML index 0.
/// After resolution the relay addresses the device by its globally unique,
/// immutable NVML UUID (`device_by_uuid`) rather than by index, so a changing
/// device enumeration cannot silently redirect telemetry or a brake to a
/// different card. This release does not select among multiple GPUs: it always
/// binds index 0 and never falls back to another device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTarget {
    uuid: String,
}

/// Why startup GPU identity resolution failed. Reported so the supervisor can
/// log an explicit reason and continue with fail-closed telemetry and no
/// mutation (see [`GpuTarget::resolve`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetError {
    /// The NVML library could not be initialized (no driver / `libnvidia-ml.so`).
    NvmlInit,
    /// NVML index 0 could not be looked up (no device at that index).
    DeviceLookup(String),
    /// The device UUID could not be read from NVML.
    UuidRead(String),
    /// The UUID NVML reported is not a well-formed physical-GPU identifier.
    UuidInvalid(String),
    /// The reported UUID did not round-trip back through `device_by_uuid`.
    UuidRoundTrip(String),
}

impl std::fmt::Display for TargetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NvmlInit => write!(
                f,
                "NVML initialization failed (no NVIDIA driver / libnvidia-ml.so)"
            ),
            Self::DeviceLookup(msg) => write!(f, "NVML device index 0 lookup failed: {msg}"),
            Self::UuidRead(msg) => write!(f, "reading GPU UUID from NVML failed: {msg}"),
            Self::UuidInvalid(uuid) => write!(f, "NVML reported a malformed GPU UUID: {uuid:?}"),
            Self::UuidRoundTrip(msg) => {
                write!(
                    f,
                    "GPU UUID did not resolve back through device_by_uuid: {msg}"
                )
            }
        }
    }
}

impl std::error::Error for TargetError {}

impl GpuTarget {
    /// The validated, NVML-canonical GPU UUID (e.g. `GPU-xxxxxxxx-...`).
    #[must_use]
    pub fn uuid(&self) -> &str {
        &self.uuid
    }

    /// Resolve the single GPU identity once, from NVML index 0.
    ///
    /// Initializes NVML through the shared lazy handle, looks up index 0 a
    /// single time, reads its UUID, validates the UUID shape, and confirms the
    /// UUID resolves back to a device via `device_by_uuid`. Any step failing is
    /// an explicit [`TargetError`]; this function never tries another device.
    pub fn resolve() -> Result<Self, TargetError> {
        let nvml = NVML.as_ref().ok_or(TargetError::NvmlInit)?;
        let device = nvml
            .device_by_index(0)
            .map_err(|e| TargetError::DeviceLookup(e.to_string()))?;
        let uuid = device
            .uuid()
            .map_err(|e| TargetError::UuidRead(e.to_string()))?;
        if !is_valid_gpu_uuid(&uuid) {
            return Err(TargetError::UuidInvalid(uuid));
        }
        // Confirm the identity round-trips: device_by_uuid must accept it. This
        // catches a UUID that reads back but cannot be re-addressed.
        nvml.device_by_uuid(uuid.clone())
            .map_err(|e| TargetError::UuidRoundTrip(e.to_string()))?;
        Ok(Self { uuid })
    }

    /// Construct a target from a pre-validated UUID string (test/wiring helper).
    ///
    /// Does not touch NVML; callers outside `resolve` are responsible for the
    /// UUID being a real device identity.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_uuid_for_test(uuid: impl Into<String>) -> Self {
        Self { uuid: uuid.into() }
    }
}

/// Validate an NVML physical-GPU UUID: a `GPU-` prefix followed by a canonical
/// 8-4-4-4-12 lowercase-or-uppercase hex UUID. MIG / non-`GPU-` identities are
/// rejected — this relay scopes a whole physical device, not a MIG slice.
pub(crate) fn is_valid_gpu_uuid(uuid: &str) -> bool {
    let Some(body) = uuid.strip_prefix("GPU-") else {
        return false;
    };
    let groups = [8usize, 4, 4, 4, 12];
    let mut parts = body.split('-');
    for &expected in &groups {
        let Some(part) = parts.next() else {
            return false;
        };
        if part.len() != expected || !part.bytes().all(|b| b.is_ascii_hexdigit()) {
            return false;
        }
    }
    parts.next().is_none()
}

// ── Hardware Bridge ─────────────────────────────────────────────────

/// GPU telemetry acquisition facade (NVML + software-fallback / unavailable).
///
/// This type does **not** run inference or own neural state. It only reads
/// sensors (or documented estimates) and forwards frames to
/// [`crate::telemetry`] validation. Privileged power-limit mutation lives on
/// [`NvmlActuator`].
///
/// Holds the [`GpuTarget`] resolved at startup. When the target is absent
/// (identity resolution failed) real NVML acquisition is skipped and telemetry
/// fail-closes to [`TelemetrySource::NvmlUnavailable`]; forced software-only
/// still returns documented idle estimates.
#[derive(Debug, Clone)]
pub struct HardwareBridge {
    target: Option<GpuTarget>,
}

impl HardwareBridge {
    /// Construct the acquisition facade scoped to `target`.
    ///
    /// A `None` target means startup identity resolution failed: real NVML
    /// acquisition is skipped and telemetry fail-closes to
    /// [`TelemetrySource::NvmlUnavailable`] (forced software-only still returns
    /// documented idle estimates).
    #[must_use]
    pub fn new(target: Option<GpuTarget>) -> Self {
        Self { target }
    }

    /// Read telemetry, but if `force_software` is true, always use the simulated
    /// fallback (never attempt real NVML/nvidia-smi). This implements the
    /// `--force-software-only` CLI flag for #11.
    #[cfg(test)]
    pub fn read_telemetry_force(&self, force_software: bool) -> TelemetryFrame {
        self.read_telemetry_with(force_software, DEFAULT_ACQUISITION_CADENCE_MS)
    }

    /// Acquire + assess with the configured supervisor tick interval.
    #[cfg(test)]
    pub fn read_telemetry_with(&self, force_software: bool, cadence_ms: u64) -> TelemetryFrame {
        self.read_telemetry_with_clock(force_software, cadence_ms, &mut SampleClock::new())
    }

    /// Acquire + assess through a shared session [`SampleClock`].
    #[cfg(test)]
    pub fn read_telemetry_with_clock(
        &self,
        force_software: bool,
        cadence_ms: u64,
        clock: &mut SampleClock,
    ) -> TelemetryFrame {
        let raw = self.acquire_raw(force_software);
        assess_with_clock(&raw, unix_now_ms(), cadence_ms, clock)
    }

    /// Raw acquisition only — no validation, no silent zeros for missing sensors.
    ///
    /// `SoftwareFallback` is used only when `force_software` is true.
    /// NVML/driver/device failure (or an absent [`GpuTarget`]) is
    /// [`TelemetrySource::NvmlUnavailable`].
    pub fn acquire_raw(&self, force_software: bool) -> RawTelemetry {
        if force_software {
            return RawTelemetry::software_fallback(unix_now_ms());
        }
        self.target
            .as_ref()
            .and_then(Self::read_nvml)
            .unwrap_or_else(|| RawTelemetry::nvml_unavailable(unix_now_ms()))
    }

    /// Returns true if the NVIDIA driver is responsive and the GPU is healthy.
    /// Uses a tight timeout to prevent blocking the supervisor if the driver is "wedged".
    pub fn is_gpu_healthy() -> bool {
        let output = std::process::Command::new("timeout")
            .args(["-k", "1s", "1s", "nvidia-smi", "-L"])
            .output();

        match output {
            Ok(out) => out.status.success(),
            Err(_) => false,
        }
    }

    fn read_nvml(target: &GpuTarget) -> Option<RawTelemetry> {
        use std::sync::atomic::{AtomicBool, Ordering};
        // Only log on the healthy -> unhealthy transition; the telemetry loop
        // calls this ~10x/sec, so an unconditional print would flood stdout
        // while the driver stays wedged.
        static WAS_UNHEALTHY: AtomicBool = AtomicBool::new(false);

        if !Self::is_gpu_healthy() {
            if !WAS_UNHEALTHY.swap(true, Ordering::Relaxed) {
                println!(
                    "[hardware_bridge] nvidia-smi hung. Treating telemetry as unavailable (fail closed)."
                );
            }
            return None;
        }
        WAS_UNHEALTHY.store(false, Ordering::Relaxed);

        let nvml = NVML.as_ref()?;
        // Address the device by its validated UUID, not by index, so telemetry
        // stays bound to the GPU chosen at startup even if enumeration shifts.
        let device = nvml.device_by_uuid(target.uuid.clone()).ok()?;
        let observed_at = unix_now_ms();

        let gpu_temp_c = device
            .temperature(TemperatureSensor::Gpu)
            .ok()
            .map(|t| t as f32);
        let power_w = device.power_usage().ok().map(|mw| mw as f32 / 1000.0);
        let gpu_clock_mhz = device.clock_info(Clock::Graphics).ok().map(|c| c as f32);
        let mem_clock_mhz = device.clock_info(Clock::Memory).ok().map(|c| c as f32);
        let fan_speed_pct = device.fan_speed(0).ok().map(|s| s as f32);
        let mem_util_pct = device.utilization_rates().ok().map(|u| u.memory as f32);
        let vddcr_gfx_v = power_w.map(derive_vddcr_gfx_v);

        Some(RawTelemetry {
            source_unix_ms: Some(observed_at),
            timestamp_origin: TimestampOrigin::LiveAcquire,
            source: TelemetrySource::Nvml,
            gpu_temp_c,
            // nvml-wrapper 0.10 only exposes TemperatureSensor::Gpu. Do not
            // fabricate VRAM temp as gpu+8 — leave it missing.
            vram_temp_c: None,
            power_w,
            vddcr_gfx_v,
            gpu_clock_mhz,
            mem_clock_mhz,
            fan_speed_pct,
            mem_util_pct,
        })
    }

    /// Check GPU safety thresholds against a validated frame.
    ///
    /// Instantaneous Ok/Warn/Critical only — stateful hysteresis lives in
    /// [`crate::safety::SafetyMachine`]. Simulation is
    /// [`TelemetrySource::SoftwareFallback`] only (forced software-only).
    /// [`TelemetrySource::NvmlUnavailable`] is **not** simulated: missing
    /// safety samples fail closed.
    ///
    /// Returns `(SafetyStatus, is_simulated)`.
    #[cfg(test)]
    pub fn check_safety(frame: &TelemetryFrame) -> (SafetyStatus, bool) {
        instant_status_with_policy(frame, &crate::safety::test_policy())
    }
}

// ── NVML / nvidia-smi safety actuator (privileged backend, GH#46) ───

/// [`SafetyActuator`] backed by NVML (for reads) and `nvidia-smi` (for the
/// privileged power-limit mutation).
///
/// A hardware adapter only: the emergency-brake *policy* (when to brake,
/// hysteresis, fail-closed rules) is owned by [`crate::safety`]. This type just
/// applies/releases/detects a power-limit brake and reports typed
/// [`ActuatorError`]s.
///
/// Scoped to the startup [`GpuTarget`]. Reads (`query_power_limits_w`) and
/// every `nvidia-smi` command are pinned to that UUID. When the target is
/// absent (identity resolution failed) reads return `None` and apply/release
/// refuse without running any command.
#[derive(Debug, Clone, Default)]
pub struct NvmlActuator {
    target: Option<GpuTarget>,
}

impl NvmlActuator {
    /// Construct the NVML / `nvidia-smi` actuator adapter scoped to `target`.
    ///
    /// Construction does not open a device. Apply/release still require a live
    /// NVIDIA GPU and passwordless `sudo -n nvidia-smi` (see crate-level docs).
    /// A `None` target means startup identity resolution failed: reads report
    /// `None` and mutation is refused.
    #[must_use]
    pub fn new(target: Option<GpuTarget>) -> Self {
        Self { target }
    }

    /// The exact `nvidia-smi` argument vector this actuator would run to set
    /// `limit_w`, or `None` when no [`GpuTarget`] is resolved (mutation refused,
    /// no command).
    ///
    /// This is the single command-planning seam the mutation paths use, so the
    /// UUID scoping of a real apply/release is observable without a GPU or a
    /// subprocess. Pure and side-effect free.
    pub(crate) fn planned_command(&self, limit_w: u32) -> Option<Vec<String>> {
        self.target
            .as_ref()
            .map(|target| power_limit_command_args(target, limit_w))
    }

    /// Execute a planned power-limit command, or refuse (no command) when the
    /// target is absent. Keeps `-i <UUID>` scoping on every real mutation.
    fn run_power_limit(&self, limit_w: u32) -> Result<(), ActuatorError> {
        let args = self.planned_command(limit_w).ok_or_else(no_target_error)?;
        run_power_limit_command(&args, limit_w)
    }
}

impl SafetyActuator for NvmlActuator {
    /// CLOSED LOOP CONTROL: The Emergency Brake.
    /// Throttles GPU power to `pct` of the device's *default* power limit
    /// (not the current limit) to avoid compounding throttle across restarts.
    /// Fails closed if NVML cannot report a real limit — never invents a hardcoded wattage.
    /// Refuses without running any command if the [`GpuTarget`] is absent.
    fn apply_emergency_brake(&self, pct: f32) -> Result<(), ActuatorError> {
        let target = self.target.as_ref().ok_or_else(no_target_error)?;
        let Some(target_pl) = plan_brake_apply(
            query_power_limit_w(target),
            query_default_power_limit_w(target),
            pct,
        )?
        else {
            return Ok(());
        };
        println!("[hardware_bridge] EMERGENCY BRAKE: setting PL to {target_pl}W");

        self.run_power_limit(target_pl)
    }

    /// Release the emergency brake — restore GPU power limit to its default.
    /// Fails closed if the default cannot be queried (never restores a fabricated wattage).
    /// Refuses without running any command if the [`GpuTarget`] is absent.
    fn release_emergency_brake(&self) -> Result<(), ActuatorError> {
        let target = self.target.as_ref().ok_or_else(no_target_error)?;
        let default_limit = plan_brake_release(
            query_power_limit_w(target),
            query_default_power_limit_w(target),
        )?;
        println!(
            "[hardware_bridge] RELEASING BRAKE: Restoring PL to {}W (device default)",
            default_limit
        );

        self.run_power_limit(default_limit)
    }

    fn query_power_limits_w(&self) -> (Option<u32>, Option<u32>) {
        match self.target.as_ref() {
            Some(target) => (
                query_power_limit_w(target),
                query_default_power_limit_w(target),
            ),
            None => (None, None),
        }
    }
}

/// Typed refusal when actuation is attempted without a resolved [`GpuTarget`].
fn no_target_error() -> ActuatorError {
    ActuatorError::CommandFailed(
        "no GPU target resolved at startup; refusing power-limit mutation".into(),
    )
}

// Read-only command planning: never adopt a lower operator cap as a relay brake.
fn plan_brake_apply(
    current: Option<u32>,
    default: Option<u32>,
    pct: f32,
) -> Result<Option<u32>, ActuatorError> {
    if pct != crate::safety::BRAKE_FRACTION {
        return Err(ActuatorError::CommandFailed(
            "unsupported brake fraction".into(),
        ));
    }
    let (Some(current), Some(default)) = (current.filter(|w| *w > 0), default.filter(|w| *w > 0))
    else {
        return Err(ActuatorError::PowerLimitUnavailable);
    };
    let target = crate::safety::unambiguous_brake_target(default, pct)
        .ok_or(ActuatorError::PowerLimitUnavailable)?;
    if current < target.saturating_sub(crate::safety::BRAKE_MATCH_TOLERANCE_W) {
        return Err(ActuatorError::CommandFailed(
            "power limit is below relay target; preserving operator/device cap".into(),
        ));
    }
    if current.abs_diff(target) <= crate::safety::BRAKE_MATCH_TOLERANCE_W {
        return Ok(None);
    }
    // A sub-default foreign cap above target also must not later be raised to
    // default by release. Refuse to claim ownership of any foreign cap.
    if current < default.saturating_sub(crate::safety::BRAKE_MATCH_TOLERANCE_W) {
        return Err(ActuatorError::CommandFailed(
            "foreign sub-default power cap; refusing relay ownership".into(),
        ));
    }
    Ok(Some(target))
}

fn plan_brake_release(current: Option<u32>, default: Option<u32>) -> Result<u32, ActuatorError> {
    if current.is_none_or(|w| w == 0) || default.is_none_or(|w| w == 0) {
        return Err(ActuatorError::PowerLimitUnavailable);
    }
    match crate::safety::classify_power_limit(current, default, crate::safety::BRAKE_FRACTION) {
        crate::safety::PowerLimitObservation::RelayOwnedBrake(m)
            if m.default_w > 0 && m.current_w > 0 =>
        {
            Ok(m.default_w)
        }
        _ => Err(ActuatorError::CommandFailed(
            "current power limit does not match relay brake; refusing restore".into(),
        )),
    }
}

/// Build the scoped `nvidia-smi` power-limit argument vector for `target`.
///
/// Pure and side-effect free so tests can assert the exact selector and
/// argument order without a GPU. Always pins `-i <UUID>` before `-pl <watts>`.
fn power_limit_command_args(target: &GpuTarget, limit_w: u32) -> Vec<String> {
    // -k 2: escalate SIGTERM -> SIGKILL so a wedged nvidia-smi cannot stall forever.
    vec![
        "-k".to_string(),
        "2".to_string(),
        "5s".to_string(),
        "sudo".to_string(),
        "-n".to_string(),
        "nvidia-smi".to_string(),
        "-i".to_string(),
        target.uuid.clone(),
        "-pl".to_string(),
        limit_w.to_string(),
    ]
}

/// Execute a pre-built power-limit command via `timeout` + non-interactive
/// `sudo -n` so a password prompt or wedged nvidia-smi cannot stall the relay
/// loop indefinitely. `args` is the exact vector from
/// [`power_limit_command_args`] (already scoped to the target via `-i <UUID>`).
fn run_power_limit_command(args: &[String], limit_w: u32) -> Result<(), ActuatorError> {
    let status = std::process::Command::new("timeout")
        .args(args)
        .status()
        .map_err(|e| ActuatorError::CommandFailed(format!("Failed to exec nvidia-smi: {e}")))?;

    if !status.success() {
        return Err(ActuatorError::CommandFailed(format!(
            "nvidia-smi -pl {limit_w} failed (timeout, missing passwordless sudo, or command error)"
        )));
    }
    Ok(())
}

/// Query the target GPU's current power management limit in watts via NVML.
fn query_power_limit_w(target: &GpuTarget) -> Option<u32> {
    let nvml = NVML.as_ref()?;
    let device = nvml.device_by_uuid(target.uuid.clone()).ok()?;
    let limit_mw = device.power_management_limit().ok()?;
    Some(limit_mw / 1000)
}

/// Query the target GPU's default (enforced) power management limit in watts via NVML.
fn query_default_power_limit_w(target: &GpuTarget) -> Option<u32> {
    let nvml = NVML.as_ref()?;
    let device = nvml.device_by_uuid(target.uuid.clone()).ok()?;
    let limit_mw = device.power_management_limit_default().ok()?;
    Some(limit_mw / 1000)
}

/// Derive an observability-only Vcore estimate from board power.
/// This is not an NVML voltage sensor; [`crate::telemetry::SignalOrigin::Derived`].
pub(crate) fn derive_vddcr_gfx_v(power_w: f32) -> f32 {
    let p_idle = 50.0_f32;
    let p_tdp = 300.0_f32;
    let v_idle = 0.70_f32;
    let v_tdp = 1.05_f32;
    let t = ((power_w - p_idle) / (p_tdp - p_idle)).clamp(0.0, 1.0);
    v_idle + t * (v_tdp - v_idle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::{SampleValidity, TelemetrySource, assess, fixtures, software_fallback};

    /// A canonical, well-formed physical-GPU UUID for hardware-free wiring.
    const TEST_UUID: &str = "GPU-12345678-1234-1234-1234-123456789abc";

    /// Bridge scoped to a synthetic target (never touches NVML in these tests
    /// because they all run software-only or without a real device).
    fn test_bridge() -> HardwareBridge {
        HardwareBridge::new(Some(GpuTarget::from_uuid_for_test(TEST_UUID)))
    }

    fn nvml_temp_power(temp_c: f32, power_w: f32) -> TelemetryFrame {
        let mut raw = fixtures::healthy_real();
        raw.gpu_temp_c = Some(temp_c);
        raw.power_w = Some(power_w);
        assess(&raw, fixtures::NOW)
    }

    #[test]
    fn test_raw_software_fallback_never_uses_silent_zero_for_missing_vram() {
        let raw = RawTelemetry::software_fallback(fixtures::NOW);
        assert_eq!(raw.source, TelemetrySource::SoftwareFallback);
        assert_eq!(raw.vram_temp_c, None);
        assert_eq!(raw.mem_util_pct, Some(0.0));
        assert_eq!(raw.gpu_temp_c, Some(software_fallback::GPU_TEMP_C));
    }

    #[test]
    fn test_safety_ok_on_simulated_values() {
        let frame = assess(&fixtures::software_fallback(), fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert_eq!(status, SafetyStatus::Ok);
        assert!(is_sim);
    }

    #[test]
    fn test_safety_does_not_infer_simulation_from_old_magic_values() {
        let frame = assess(&fixtures::nvml_looks_like_old_magic(), fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert_eq!(status, SafetyStatus::Ok);
        assert!(!is_sim);
        assert_eq!(frame.source, TelemetrySource::Nvml);
        assert_eq!(frame.gpu_temp_c.value, Some(0.0));
        assert_eq!(frame.power_w.value, Some(25.0));
    }

    #[test]
    fn test_safety_warn_on_elevated_temp() {
        let frame = nvml_temp_power(78.0, 200.0);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Warn(_)));
        assert!(!is_sim);
    }

    #[test]
    fn test_safety_warn_on_elevated_power() {
        let frame = nvml_temp_power(70.0, 320.0);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Warn(_)));
        assert!(!is_sim);
    }

    #[test]
    fn test_safety_critical_on_high_temp() {
        let frame = nvml_temp_power(90.0, 200.0);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(_)));
        assert!(!is_sim);
    }

    #[test]
    fn test_safety_critical_on_high_power() {
        let frame = nvml_temp_power(70.0, 360.0);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(_)));
        assert!(!is_sim);
    }

    #[test]
    fn test_safety_ok_on_normal_telemetry() {
        let frame = nvml_temp_power(65.0, 200.0);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert_eq!(status, SafetyStatus::Ok);
        assert!(!is_sim);
    }

    #[test]
    fn test_safety_critical_on_unknown_power_with_real_temperature() {
        let frame = assess(&fixtures::sensor_dropout(), fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(ref msg) if msg.contains("power_w")));
        assert!(!is_sim);
        assert_eq!(frame.power_w.validity, SampleValidity::Missing);
        assert_eq!(frame.power_w.value, None);
    }

    #[test]
    fn test_check_safety_fail_closes_on_missing_temp_or_power() {
        let mut raw = fixtures::healthy_real();
        raw.gpu_temp_c = None;
        let frame = assess(&raw, fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(ref msg) if msg.contains("gpu_temp_c")));
        assert!(!is_sim);

        let mut raw = fixtures::healthy_real();
        raw.power_w = None;
        let frame = assess(&raw, fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(ref msg) if msg.contains("power_w")));
        assert!(!is_sim);

        // Warn-band temperature with missing power must not become Warn or Ok.
        let mut frame = nvml_temp_power(78.0, 200.0);
        frame.power_w.value = None;
        frame.power_w.validity = SampleValidity::Missing;
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(_)));
        assert!(!matches!(status, SafetyStatus::Warn(_)));
        assert!(!is_sim);

        let mut frame = nvml_temp_power(78.0, 200.0);
        frame.gpu_temp_c.value = None;
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Valid);
        let (status, _) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(ref msg) if msg.contains("gpu_temp_c")));
    }

    #[test]
    fn test_safety_critical_on_non_finite_telemetry() {
        let frame = assess(&fixtures::non_finite(), fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(_)));
        assert!(!is_sim);

        let mut raw = fixtures::healthy_real();
        raw.gpu_temp_c = Some(f32::NAN);
        let frame = assess(&raw, fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(_)));
        assert!(!is_sim);

        let mut raw = fixtures::healthy_real();
        raw.power_w = Some(f32::INFINITY);
        let frame = assess(&raw, fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(_)));
        assert!(!is_sim);
    }

    #[test]
    fn test_safety_critical_on_stale_and_out_of_range() {
        let stale = assess(&fixtures::stale(), fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&stale);
        assert!(matches!(status, SafetyStatus::Critical(ref msg) if msg.contains("stale")));
        assert!(!is_sim);

        let oor = assess(&fixtures::out_of_range(), fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&oor);
        assert!(matches!(status, SafetyStatus::Critical(ref msg) if msg.contains("invalid")));
        assert!(!is_sim);
    }

    #[test]
    fn test_acquire_raw_force_software_is_fallback_not_unavailable() {
        let bridge = test_bridge();
        let raw = bridge.acquire_raw(true);
        assert_eq!(raw.source, TelemetrySource::SoftwareFallback);
        let unavail = bridge.acquire_raw(false);
        assert!(
            matches!(
                unavail.source,
                TelemetrySource::Nvml | TelemetrySource::NvmlUnavailable
            ),
            "acquire_raw(false) must not be SoftwareFallback, got {:?}",
            unavail.source
        );
        assert_ne!(unavail.source, TelemetrySource::SoftwareFallback);
    }

    #[test]
    fn test_nvml_unavailable_fail_closes_safety() {
        let frame = assess(&fixtures::nvml_unavailable(), fixtures::NOW);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert!(matches!(status, SafetyStatus::Critical(ref msg) if msg.contains("missing")));
        assert!(!is_sim);
        assert_eq!(frame.source, TelemetrySource::NvmlUnavailable);
        assert_eq!(frame.gpu_temp_c.value, None);
        assert_eq!(frame.power_w.value, None);
    }

    #[test]
    fn test_read_telemetry_with_records_configured_cadence() {
        let frame = test_bridge().read_telemetry_with(true, 50);
        assert_eq!(frame.acquisition_cadence_ms, 50);
        assert_eq!(frame.source, TelemetrySource::SoftwareFallback);
    }

    #[test]
    fn test_read_telemetry_force_software_only() {
        let frame = test_bridge().read_telemetry_force(true);
        assert_eq!(frame.source, TelemetrySource::SoftwareFallback);
        assert_eq!(frame.power_w.value, Some(software_fallback::POWER_W));
        assert_eq!(frame.gpu_temp_c.value, Some(software_fallback::GPU_TEMP_C));
        assert_eq!(
            frame.gpu_clock_mhz.value,
            Some(software_fallback::GPU_CLOCK_MHZ)
        );
        assert_eq!(
            frame.mem_clock_mhz.value,
            Some(software_fallback::MEM_CLOCK_MHZ)
        );
        assert_eq!(
            frame.fan_speed_pct.value,
            Some(software_fallback::FAN_SPEED_PCT)
        );
        assert_eq!(
            frame.mem_util_pct.value,
            Some(software_fallback::MEM_UTIL_PCT)
        );
        assert_eq!(frame.vram_temp_c.value, None);
        assert_eq!(frame.vram_temp_c.validity, SampleValidity::Missing);
        let (status, is_sim) = HardwareBridge::check_safety(&frame);
        assert_eq!(status, SafetyStatus::Ok);
        assert!(is_sim);
    }

    #[test]
    fn test_is_gpu_healthy_does_not_panic() {
        let result = std::panic::catch_unwind(HardwareBridge::is_gpu_healthy);
        assert!(result.is_ok(), "is_gpu_healthy should not panic");
    }

    /// Without a real GPU (no NVML), actuation must fail closed with a typed
    /// [`ActuatorError`] rather than inventing a wattage or panicking.
    #[test]
    fn test_nvml_actuator_fails_closed_without_gpu() {
        // In CI there is no NVIDIA device, so NVML queries return None and every
        // actuation path must surface a typed error / no match.
        let target = GpuTarget::from_uuid_for_test(TEST_UUID);
        if query_default_power_limit_w(&target).is_some() || query_power_limit_w(&target).is_some()
        {
            // A real GPU is present (unusual in CI) — skip the fail-closed asserts.
            return;
        }
        let actuator = NvmlActuator::new(Some(target));
        assert_eq!(
            actuator.apply_emergency_brake(0.5),
            Err(ActuatorError::PowerLimitUnavailable)
        );
        assert_eq!(
            actuator.release_emergency_brake(),
            Err(ActuatorError::PowerLimitUnavailable)
        );
        assert_eq!(actuator.detect_engaged_brake(0.5), None);
    }

    #[test]
    fn test_sensory_mapping_from_software_fallback_carries_provenance() {
        let frame = test_bridge().read_telemetry_force(true);
        let mapping = frame.to_sensory_mapping();
        assert_eq!(
            mapping.acquisition_source,
            TelemetrySource::SoftwareFallback
        );
        assert!(
            mapping
                .stimuli
                .iter()
                .all(|s| s.source == TelemetrySource::SoftwareFallback)
        );
        let util = mapping
            .stimuli
            .iter()
            .find(|s| s.name == "mem_util_pct")
            .unwrap();
        assert_eq!(util.raw, Some(0.0));
        assert_eq!(util.normalized, Some(0.0));
        assert_eq!(util.validity, SampleValidity::Valid);
        assert_eq!(mapping.timestamp_origin, TimestampOrigin::Simulated);
        assert_eq!(mapping.batch_id, 0);
        assert!(!mapping.session_id.is_empty());
        assert!(mapping.source_unix_ms.is_some());
        assert!(
            mapping.received_at_unix_ms >= mapping.source_unix_ms.unwrap(),
            "receive time is at or after source time on the live simulated path"
        );
    }

    #[test]
    fn test_software_fallback_frames_share_sample_clock_sequence() {
        let mut clock = SampleClock::with_session_id("hw-session");
        let bridge = test_bridge();
        let a = bridge.read_telemetry_with_clock(true, 50, &mut clock);
        let b = bridge.read_telemetry_with_clock(true, 50, &mut clock);
        assert_eq!(a.session_id, "hw-session");
        assert_eq!(a.session_id, b.session_id);
        assert_eq!(a.batch_id, 0);
        assert_eq!(b.batch_id, 1);
        assert_eq!(a.timestamp_origin, TimestampOrigin::Simulated);
        assert_eq!(b.timestamp_origin, TimestampOrigin::Simulated);
        assert!(b.received_at_unix_ms >= a.received_at_unix_ms);
    }

    #[test]
    fn test_derive_vddcr_gfx_v_is_deterministic() {
        assert!((derive_vddcr_gfx_v(50.0) - 0.70).abs() < 1e-6);
        assert!((derive_vddcr_gfx_v(0.0) - 0.70).abs() < 1e-6);
        assert!((derive_vddcr_gfx_v(300.0) - 1.05).abs() < 1e-6);
        assert!((derive_vddcr_gfx_v(400.0) - 1.05).abs() < 1e-6);
        let mid = derive_vddcr_gfx_v(175.0);
        assert!((mid - 0.875).abs() < 1e-6);
    }

    #[test]
    fn test_read_telemetry_without_force_is_never_software_fallback() {
        let frame = test_bridge().read_telemetry_force(false);
        assert_ne!(frame.source, TelemetrySource::SoftwareFallback);
        assert!(matches!(
            frame.source,
            TelemetrySource::Nvml | TelemetrySource::NvmlUnavailable
        ));
    }
}

#[cfg(test)]
mod power_limit_ownership_tests {
    use super::*;

    #[test]
    fn apply_never_adopts_an_operator_cap_below_the_relay_target() {
        assert!(plan_brake_apply(Some(100), Some(400), 0.5).is_err());
        assert!(plan_brake_apply(Some(300), Some(400), 0.5).is_err());
        assert_eq!(plan_brake_apply(Some(200), Some(400), 0.5).unwrap(), None);
        assert_eq!(
            plan_brake_apply(Some(400), Some(400), 0.5).unwrap(),
            Some(200)
        );
    }

    #[test]
    fn apply_rejects_fractions_other_than_the_fixed_brake() {
        let err = plan_brake_apply(Some(400), Some(400), 0.6).unwrap_err();
        assert!(
            matches!(err, ActuatorError::CommandFailed(ref msg) if msg.contains("unsupported brake fraction"))
        );
        assert!(plan_brake_apply(Some(400), Some(400), 0.0).is_err());
        assert_eq!(
            plan_brake_apply(Some(400), Some(400), crate::safety::BRAKE_FRACTION).unwrap(),
            Some(200)
        );
    }

    #[test]
    fn foreign_cap_failure_never_authorizes_a_later_default_restore() {
        use crate::safety::{ActuatorOutcome, BrakeIntent};
        use crate::telemetry::{assess, fixtures};
        for cap in [100, 300] {
            let mut machine = crate::safety::test_machine();
            let mut raw = fixtures::healthy_real();
            raw.gpu_temp_c = Some(95.0);
            assert_eq!(
                machine.evaluate(&assess(&raw, fixtures::NOW)).intent,
                BrakeIntent::Apply
            );
            let err = plan_brake_apply(Some(cap), Some(400), 0.5).unwrap_err();
            let _ = machine.record_actuator(ActuatorOutcome::ApplyFailed(err.to_string()));
            for _ in 0..5 {
                let snap = machine.evaluate(&assess(&fixtures::healthy_real(), fixtures::NOW));
                assert_eq!(snap.intent, BrakeIntent::None);
                assert!(!snap.brake_engaged);
            }
        }
    }

    #[test]
    fn missing_default_cannot_compound_a_previous_brake() {
        assert!(plan_brake_apply(Some(200), None, 0.5).is_err());
        assert!(plan_brake_apply(None, Some(400), 0.5).is_err());
    }

    #[test]
    fn release_requires_current_power_to_still_match_the_relay_brake() {
        assert_eq!(plan_brake_release(Some(200), Some(400)).unwrap(), 400);
        assert!(plan_brake_release(Some(100), Some(400)).is_err());
        assert!(plan_brake_release(Some(300), Some(400)).is_err());
        assert_eq!(
            plan_brake_release(None, Some(400)),
            Err(ActuatorError::PowerLimitUnavailable)
        );
        assert_eq!(
            plan_brake_release(Some(200), None),
            Err(ActuatorError::PowerLimitUnavailable)
        );
        assert_eq!(
            plan_brake_release(None, None),
            Err(ActuatorError::PowerLimitUnavailable)
        );
    }
}

#[cfg(test)]
mod ambiguous_power_limit_tests {
    use super::*;
    #[test]
    fn tiny_limits_cannot_be_mistaken_for_a_relay_owned_brake() {
        for default in 1..=8 {
            assert!(plan_brake_apply(Some(default), Some(default), 0.5).is_err());
            assert!(plan_brake_release(Some(default / 2), Some(default)).is_err());
            assert_eq!(
                crate::safety::classify_power_limit(Some(default), Some(default), 0.5),
                crate::safety::PowerLimitObservation::Unreadable
            );
        }
        assert_eq!(plan_brake_apply(Some(10), Some(10), 0.5).unwrap(), Some(5));
        assert_eq!(plan_brake_release(Some(5), Some(10)).unwrap(), 10);
    }
}

#[cfg(test)]
mod gpu_target_tests {
    use super::*;
    use crate::safety::BRAKE_FRACTION;

    /// A canonical, well-formed physical-GPU UUID.
    const OK_UUID: &str = "GPU-abcdef01-2345-6789-abcd-ef0123456789";

    #[test]
    fn accepts_canonical_physical_gpu_uuids() {
        assert!(is_valid_gpu_uuid(OK_UUID));
        // Uppercase hex is equally valid.
        assert!(is_valid_gpu_uuid(
            "GPU-ABCDEF01-2345-6789-ABCD-EF0123456789"
        ));
        // Mixed case.
        assert!(is_valid_gpu_uuid(
            "GPU-AbCdEf01-2345-6789-abcd-EF0123456789"
        ));
    }

    #[test]
    fn rejects_malformed_or_non_gpu_uuids() {
        // Missing GPU- prefix.
        assert!(!is_valid_gpu_uuid("abcdef01-2345-6789-abcd-ef0123456789"));
        // MIG identity is not a whole-device UUID.
        assert!(!is_valid_gpu_uuid(
            "MIG-abcdef01-2345-6789-abcd-ef0123456789"
        ));
        // Empty.
        assert!(!is_valid_gpu_uuid(""));
        // Just the prefix.
        assert!(!is_valid_gpu_uuid("GPU-"));
        // Wrong group lengths.
        assert!(!is_valid_gpu_uuid("GPU-abcd-2345-6789-abcd-ef0123456789"));
        assert!(!is_valid_gpu_uuid("GPU-abcdef01-2345-6789-abcd-ef0123456"));
        // Non-hex characters.
        assert!(!is_valid_gpu_uuid(
            "GPU-ghijkl01-2345-6789-abcd-ef0123456789"
        ));
        // Trailing extra group.
        assert!(!is_valid_gpu_uuid(
            "GPU-abcdef01-2345-6789-abcd-ef0123456789-0000"
        ));
        // Missing a group.
        assert!(!is_valid_gpu_uuid("GPU-abcdef01-2345-6789-abcd"));
    }

    #[test]
    fn command_selector_uses_exact_arg_order_and_scopes_by_uuid() {
        let target = GpuTarget::from_uuid_for_test(OK_UUID);
        let args = power_limit_command_args(&target, 200);
        assert_eq!(
            args,
            vec![
                "-k".to_string(),
                "2".to_string(),
                "5s".to_string(),
                "sudo".to_string(),
                "-n".to_string(),
                "nvidia-smi".to_string(),
                "-i".to_string(),
                OK_UUID.to_string(),
                "-pl".to_string(),
                "200".to_string(),
            ]
        );
        // The device selector `-i <UUID>` must precede the `-pl <watts>` pair.
        let i_pos = args.iter().position(|a| a == "-i").unwrap();
        let pl_pos = args.iter().position(|a| a == "-pl").unwrap();
        assert!(i_pos < pl_pos);
        assert_eq!(args[i_pos + 1], OK_UUID);
    }

    /// Assert `args` is the canonical `-i <UUID> -pl <watts>` invocation.
    fn assert_scoped_command(args: &[String], uuid: &str, watts: &str) {
        let i_pos = args.iter().position(|a| a == "-i").expect("has -i");
        let pl_pos = args.iter().position(|a| a == "-pl").expect("has -pl");
        assert!(i_pos < pl_pos, "-i must precede -pl");
        assert_eq!(
            args[i_pos + 1],
            uuid,
            "device selector must be the target UUID"
        );
        assert_eq!(args.last().unwrap(), watts);
    }

    #[test]
    fn apply_command_path_preserves_uuid_scoping_at_the_planned_wattage() {
        // plan_brake_apply picks 50% of a 400 W default -> 200 W. The actuator's
        // own command plan (the seam apply_emergency_brake runs) must carry that
        // wattage against the scoped UUID.
        let planned = plan_brake_apply(Some(400), Some(400), BRAKE_FRACTION)
            .unwrap()
            .expect("apply should target a wattage");
        assert_eq!(planned, 200);
        let actuator = NvmlActuator::new(Some(GpuTarget::from_uuid_for_test(OK_UUID)));
        let args = actuator
            .planned_command(planned)
            .expect("resolved target plans a command");
        assert_scoped_command(&args, OK_UUID, "200");
    }

    #[test]
    fn release_command_path_preserves_uuid_scoping_at_the_default_wattage() {
        // plan_brake_release restores the device default (400 W) when current
        // still matches the relay's 200 W brake target. The actuator's command
        // plan for release must carry that wattage against the scoped UUID.
        let default_w = plan_brake_release(Some(200), Some(400)).unwrap();
        assert_eq!(default_w, 400);
        let actuator = NvmlActuator::new(Some(GpuTarget::from_uuid_for_test(OK_UUID)));
        let args = actuator
            .planned_command(default_w)
            .expect("resolved target plans a command");
        assert_scoped_command(&args, OK_UUID, "400");
    }

    #[test]
    fn actuator_without_target_plans_no_command() {
        // The command seam itself refuses without a target, so no mutation path
        // can build (or run) a command when identity resolution failed.
        let actuator = NvmlActuator::new(None);
        assert_eq!(actuator.planned_command(200), None);
    }

    #[test]
    fn apply_planner_preserves_a_sub_target_operator_cap() {
        // A cap below the relay target must never be adopted/lowered by apply.
        let err = plan_brake_apply(Some(100), Some(400), BRAKE_FRACTION).unwrap_err();
        assert!(matches!(err, ActuatorError::CommandFailed(_)));
    }

    #[test]
    fn release_planner_refuses_when_current_no_longer_matches_the_brake() {
        // A foreign cap that is not the relay target must not be restored.
        let err = plan_brake_release(Some(300), Some(400)).unwrap_err();
        assert!(matches!(err, ActuatorError::CommandFailed(_)));
    }

    #[test]
    fn actuator_without_target_refuses_apply_and_release_without_a_command() {
        let actuator = NvmlActuator::new(None);
        // Both mutations must fail closed with a typed no-target refusal and,
        // by construction, never build or run a command.
        let apply = actuator.apply_emergency_brake(BRAKE_FRACTION).unwrap_err();
        assert!(
            matches!(apply, ActuatorError::CommandFailed(ref m) if m.contains("no GPU target")),
            "unexpected apply error: {apply:?}"
        );
        let release = actuator.release_emergency_brake().unwrap_err();
        assert!(
            matches!(release, ActuatorError::CommandFailed(ref m) if m.contains("no GPU target")),
            "unexpected release error: {release:?}"
        );
        // No target -> both limits unreadable, so no leftover brake is detected.
        assert_eq!(actuator.query_power_limits_w(), (None, None));
        assert_eq!(actuator.detect_engaged_brake(BRAKE_FRACTION), None);
    }

    #[test]
    fn bridge_without_target_fail_closes_real_telemetry() {
        let bridge = HardwareBridge::new(None);
        // Real acquisition without a resolved target never touches NVML and
        // fail-closes to NvmlUnavailable (not SoftwareFallback).
        let raw = bridge.acquire_raw(false);
        assert_eq!(raw.source, TelemetrySource::NvmlUnavailable);
        // Forced software-only still returns documented idle estimates.
        let sim = bridge.acquire_raw(true);
        assert_eq!(sim.source, TelemetrySource::SoftwareFallback);
    }

    #[test]
    fn target_uuid_accessor_returns_the_validated_string() {
        let target = GpuTarget::from_uuid_for_test(OK_UUID);
        assert_eq!(target.uuid(), OK_UUID);
    }
}
