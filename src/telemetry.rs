//! Explicit telemetry validity, freshness, normalization, and provenance contract.
//!
//! Invalid, missing, stale, and simulated data is represented by
//! [`SampleValidity`] and [`TelemetrySource`], never inferred from magic numeric
//! values. Missing or invalid readings stay [`None`] and are never silently
//! converted into a legitimate numeric zero.
//!
//! This module owns the typed mapping surface toward `corpus-ipc` ([#40](https://github.com/rmems/vahtisiru/issues/40)):
//! [`SensoryMapping`] / [`MappedStimulus`]. Transport and the canonical
//! `StimulusBatch` conversion live in [`crate::publish`].
//!
//! Frame ordering and timestamp provenance live in [`crate::time`]: every
//! assessed frame is stamped with a session id + strictly increasing
//! `batch_id`. Source wall time is preserved separately from receive/emit time.

use serde::{Deserialize, Serialize};

pub use crate::time::{
    FrameTiming, SampleClock, SourceTimeStatus, TimestampOrigin, UnixMillis, unix_now_ms,
};

/// Provenance of a telemetry acquisition path.
///
/// Simulated / software-fallback data is tagged here explicitly. It is never
/// inferred from numeric conventions such as `temperature <= 0 && power <= 25`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TelemetrySource {
    /// NVIDIA Management Library (NVML) on a live device.
    Nvml,
    /// Documented software-only idle estimates (`--force-software-only`).
    SoftwareFallback,
    /// NVML/driver/device lookup failed. Not confirmed software-only.
    /// Safety must fail closed (missing safety samples), not skip as simulated.
    NvmlUnavailable,
}

/// Validity of a single sample. Orthogonal to [`TelemetrySource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SampleValidity {
    /// Finite, in-range, and fresh enough to use.
    Valid,
    /// Sensor was not read (dropout, unsupported, or not applicable).
    Missing,
    /// Non-finite or outside the documented engineering range.
    Invalid,
    /// Value may be real but is older than the signal's stale threshold.
    Stale,
}

/// Engineering unit of a sample's `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Unit {
    /// Degrees Celsius.
    Celsius,
    /// Watts.
    Watt,
    /// Volts.
    Volt,
    /// Megahertz.
    Megahertz,
    /// Percent (0–100).
    Percent,
}

/// How a signal may be consumed. See `docs/telemetry.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalClass {
    /// Candidate sensory input for downstream runtimes (via corpus-ipc).
    RuntimeInput,
    /// Consumed only by hardware-safety policy.
    SafetyOnly,
    /// Dashboards / Prometheus / logs; not a sensory or safety input.
    ObservabilityOnly,
    /// Both a safety signal and a runtime-input candidate.
    Both,
}

impl SignalClass {
    /// Whether this signal may appear in the corpus-ipc sensory mapping.
    #[must_use]
    pub const fn includes_runtime_input(self) -> bool {
        matches!(self, Self::RuntimeInput | Self::Both)
    }

    /// Whether safety policy may consume this signal.
    #[must_use]
    pub const fn includes_safety(self) -> bool {
        matches!(self, Self::SafetyOnly | Self::Both)
    }
}

/// Whether the engineering value is measured or computed from another signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalOrigin {
    /// Read from a hardware sensor.
    Measured,
    /// Computed from another signal (not a sensor reading).
    Derived,
}

/// Deterministic model-input normalization. Applied only to [`SampleValidity::Valid`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Normalization {
    /// Pass-through of the engineering value (not used for current GPU signals).
    Identity,
    /// `(value - min) / (max - min)` clamped to `[0, 1]`.
    Linear {
        /// Lower bound of the linear mapping (engineering units).
        min: f32,
        /// Upper bound of the linear mapping (engineering units).
        max: f32,
    },
}

impl Normalization {
    /// Normalize a finite engineering value. The caller must have already
    /// established [`SampleValidity::Valid`]; this never maps missing data.
    #[must_use]
    pub fn apply(self, value: f32) -> f32 {
        match self {
            Self::Identity => value,
            Self::Linear { min, max } => {
                let span = max - min;
                if !span.is_finite() || span.abs() < f32::EPSILON {
                    0.0
                } else {
                    ((value - min) / span).clamp(0.0, 1.0)
                }
            }
        }
    }
}

/// Stable signal identifiers. Names are the corpus-ipc mapping keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalId {
    /// GPU die temperature (`gpu_temp_c`).
    GpuTempC,
    /// VRAM temperature when the sensor exists (`vram_temp_c`).
    VramTempC,
    /// Board power (`power_w`).
    PowerW,
    /// Derived Vcore estimate (`vddcr_gfx_v`); not an NVML voltage sensor.
    VddcrGfxV,
    /// Graphics clock (`gpu_clock_mhz`).
    GpuClockMhz,
    /// Memory clock (`mem_clock_mhz`).
    MemClockMhz,
    /// Fan speed (`fan_speed_pct`).
    FanSpeedPct,
    /// Memory utilization (`mem_util_pct`).
    MemUtilPct,
}

impl SignalId {
    /// Canonical snake_case name for docs and corpus-ipc mapping.
    #[must_use]
    pub const fn name(self) -> &'static str {
        signal_spec(self).name
    }
}

/// Inventory of every GPU reading currently acquired by this crate.
pub const ALL_SIGNALS: [SignalId; 8] = [
    SignalId::GpuTempC,
    SignalId::VramTempC,
    SignalId::PowerW,
    SignalId::VddcrGfxV,
    SignalId::GpuClockMhz,
    SignalId::MemClockMhz,
    SignalId::FanSpeedPct,
    SignalId::MemUtilPct,
];

/// Default supervisor acquisition cadence (matches `--step-interval-ms` default).
pub const DEFAULT_ACQUISITION_CADENCE_MS: u64 = 100;
/// Wall-clock safety evaluation period in the supervisor (monotonic deadline, independent of tick length).
pub const SAFETY_EVAL_CADENCE_MS: u64 = 1_000;
/// Stale threshold for safety-critical signals (2× safety cadence).
pub const SAFETY_STALE_AFTER_MS: u64 = 2_000;
/// Stale threshold for non-safety signals.
pub const OBSERVABILITY_STALE_AFTER_MS: u64 = 5_000;

/// Number of acquisition ticks between regular supervisor safety evaluations.
pub const SAFETY_EVAL_TICKS: u64 = 10;

/// Return the freshness threshold for `signal` at the configured acquisition
/// cadence. Safety channels remain fresh for two complete evaluation periods,
/// including when `--step-interval-ms` differs from its default.
#[must_use]
pub const fn stale_after_ms(signal: SignalId, acquisition_cadence_ms: u64) -> u64 {
    if signal_spec(signal).class.includes_safety() {
        (if acquisition_cadence_ms == 0 {
            1
        } else {
            acquisition_cadence_ms
        })
        .saturating_mul(SAFETY_EVAL_TICKS)
        .saturating_mul(2)
    } else {
        signal_spec(signal).stale_after_ms
    }
}

/// Per-signal contract: unit, range, origin, class, normalization, cadence, stale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignalSpec {
    /// Stable identifier for this signal.
    pub id: SignalId,
    /// Canonical snake_case name (corpus-ipc mapping key).
    pub name: &'static str,
    /// Engineering unit of [`TelemetrySample::value`].
    pub unit: Unit,
    /// Inclusive engineering-range minimum; below this is [`SampleValidity::Invalid`].
    pub min: f32,
    /// Inclusive engineering-range maximum; above this is [`SampleValidity::Invalid`].
    pub max: f32,
    /// Measured vs derived.
    pub origin: SignalOrigin,
    /// How the signal may be consumed.
    pub class: SignalClass,
    /// Model-input mapping applied only to [`SampleValidity::Valid`] values.
    pub normalization: Normalization,
    /// Documented acquisition cadence (ms). Actual cadence is on the frame.
    pub cadence_ms: u64,
    /// Age after which a finite value becomes [`SampleValidity::Stale`].
    pub stale_after_ms: u64,
}

/// Look up the static contract for `id`.
#[must_use]
pub const fn signal_spec(id: SignalId) -> SignalSpec {
    match id {
        SignalId::GpuTempC => SignalSpec {
            id,
            name: "gpu_temp_c",
            unit: Unit::Celsius,
            min: 0.0,
            max: 125.0,
            origin: SignalOrigin::Measured,
            class: SignalClass::Both,
            normalization: Normalization::Linear {
                min: 0.0,
                max: 100.0,
            },
            cadence_ms: DEFAULT_ACQUISITION_CADENCE_MS,
            stale_after_ms: SAFETY_STALE_AFTER_MS,
        },
        SignalId::VramTempC => SignalSpec {
            id,
            name: "vram_temp_c",
            unit: Unit::Celsius,
            min: 0.0,
            max: 125.0,
            origin: SignalOrigin::Measured,
            class: SignalClass::ObservabilityOnly,
            normalization: Normalization::Linear {
                min: 0.0,
                max: 100.0,
            },
            cadence_ms: DEFAULT_ACQUISITION_CADENCE_MS,
            stale_after_ms: OBSERVABILITY_STALE_AFTER_MS,
        },
        SignalId::PowerW => SignalSpec {
            id,
            name: "power_w",
            unit: Unit::Watt,
            min: 0.0,
            max: 2_000.0,
            origin: SignalOrigin::Measured,
            class: SignalClass::Both,
            normalization: Normalization::Linear {
                min: 0.0,
                max: 350.0,
            },
            cadence_ms: DEFAULT_ACQUISITION_CADENCE_MS,
            stale_after_ms: SAFETY_STALE_AFTER_MS,
        },
        SignalId::VddcrGfxV => SignalSpec {
            id,
            name: "vddcr_gfx_v",
            unit: Unit::Volt,
            min: 0.4,
            max: 1.5,
            origin: SignalOrigin::Derived,
            class: SignalClass::ObservabilityOnly,
            normalization: Normalization::Linear { min: 0.5, max: 1.2 },
            cadence_ms: DEFAULT_ACQUISITION_CADENCE_MS,
            stale_after_ms: OBSERVABILITY_STALE_AFTER_MS,
        },
        SignalId::GpuClockMhz => SignalSpec {
            id,
            name: "gpu_clock_mhz",
            unit: Unit::Megahertz,
            min: 0.0,
            max: 3_000.0,
            origin: SignalOrigin::Measured,
            class: SignalClass::RuntimeInput,
            normalization: Normalization::Linear {
                min: 0.0,
                max: 2_500.0,
            },
            cadence_ms: DEFAULT_ACQUISITION_CADENCE_MS,
            stale_after_ms: OBSERVABILITY_STALE_AFTER_MS,
        },
        SignalId::MemClockMhz => SignalSpec {
            id,
            name: "mem_clock_mhz",
            unit: Unit::Megahertz,
            min: 0.0,
            max: 12_000.0,
            origin: SignalOrigin::Measured,
            class: SignalClass::ObservabilityOnly,
            normalization: Normalization::Linear {
                min: 0.0,
                max: 10_000.0,
            },
            cadence_ms: DEFAULT_ACQUISITION_CADENCE_MS,
            stale_after_ms: OBSERVABILITY_STALE_AFTER_MS,
        },
        SignalId::FanSpeedPct => SignalSpec {
            id,
            name: "fan_speed_pct",
            unit: Unit::Percent,
            min: 0.0,
            max: 100.0,
            origin: SignalOrigin::Measured,
            class: SignalClass::ObservabilityOnly,
            normalization: Normalization::Linear {
                min: 0.0,
                max: 100.0,
            },
            cadence_ms: DEFAULT_ACQUISITION_CADENCE_MS,
            stale_after_ms: OBSERVABILITY_STALE_AFTER_MS,
        },
        SignalId::MemUtilPct => SignalSpec {
            id,
            name: "mem_util_pct",
            unit: Unit::Percent,
            min: 0.0,
            max: 100.0,
            origin: SignalOrigin::Measured,
            class: SignalClass::RuntimeInput,
            normalization: Normalization::Linear {
                min: 0.0,
                max: 100.0,
            },
            cadence_ms: DEFAULT_ACQUISITION_CADENCE_MS,
            stale_after_ms: OBSERVABILITY_STALE_AFTER_MS,
        },
    }
}

/// Documented software-only idle estimates. These are **not** real sensors.
/// Provenance must be [`TelemetrySource::SoftwareFallback`].
pub mod software_fallback {
    /// Typical idle die temperature estimate (°C). Not a magic "no GPU" flag.
    pub const GPU_TEMP_C: f32 = 35.0;
    /// Typical idle board power estimate (W).
    pub const POWER_W: f32 = 25.0;
    /// Typical idle Vcore estimate (V). Derived, not measured.
    pub const VDDCR_GFX_V: f32 = 0.7;
    /// Typical idle graphics clock estimate (MHz).
    pub const GPU_CLOCK_MHZ: f32 = 210.0;
    /// Typical idle memory clock estimate (MHz).
    pub const MEM_CLOCK_MHZ: f32 = 405.0;
    /// Typical idle fan speed estimate (%).
    pub const FAN_SPEED_PCT: f32 = 30.0;
    /// Legitimate idle utilization of 0%, distinguishable from [`None`] missing.
    pub const MEM_UTIL_PCT: f32 = 0.0;
}

/// One typed sample. `value` is engineering units when present.
///
/// `value` is `None` for missing and non-finite readings. Out-of-range and
/// stale readings keep the raw number for observability but are not [`SampleValidity::Valid`].
/// `observed_at` is the source wall time when present, otherwise receive time
/// (missing CSV timestamp). Frame-level source vs receive split lives on
/// [`TelemetryFrame`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySample<T> {
    /// Which inventory signal this sample belongs to.
    pub signal: SignalId,
    /// Engineering-unit value. `None` for missing and non-finite readings.
    pub value: Option<T>,
    /// Acquisition timestamp (unix ms).
    pub observed_at: UnixMillis,
    /// Provenance of the acquisition path.
    pub source: TelemetrySource,
    /// Validity after range/freshness checks.
    pub validity: SampleValidity,
    /// Engineering unit of `value`.
    pub unit: Unit,
}

impl TelemetrySample<f32> {
    /// Validate a raw optional reading against the signal contract.
    ///
    /// Non-finite values become `value: None` + [`SampleValidity::Invalid`].
    /// Out-of-range values keep the raw number with [`SampleValidity::Invalid`].
    /// Missing stays `None` — never a silent `0.0`.
    #[must_use]
    pub fn from_raw(
        signal: SignalId,
        raw: Option<f32>,
        observed_at: UnixMillis,
        source: TelemetrySource,
        now: UnixMillis,
    ) -> Self {
        let spec = signal_spec(signal);
        Self::from_raw_with_stale_after(signal, raw, observed_at, source, now, spec.stale_after_ms)
    }

    fn from_raw_with_stale_after(
        signal: SignalId,
        raw: Option<f32>,
        observed_at: UnixMillis,
        source: TelemetrySource,
        now: UnixMillis,
        stale_after_ms: u64,
    ) -> Self {
        let spec = signal_spec(signal);
        let (value, validity) = if observed_at > now {
            match raw {
                None => (None, SampleValidity::Missing),
                Some(v) if !v.is_finite() => (None, SampleValidity::Invalid),
                Some(v) => (Some(v), SampleValidity::Invalid),
            }
        } else {
            let age_ms = now.saturating_sub(observed_at);
            match raw {
                None => (None, SampleValidity::Missing),
                Some(v) if !v.is_finite() => (None, SampleValidity::Invalid),
                Some(v) if v < spec.min || v > spec.max => (Some(v), SampleValidity::Invalid),
                Some(v) if age_ms >= stale_after_ms => (Some(v), SampleValidity::Stale),
                Some(v) => (Some(v), SampleValidity::Valid),
            }
        };
        Self {
            signal,
            value,
            observed_at,
            source,
            validity,
            unit: spec.unit,
        }
    }

    /// Model-input value in `[0, 1]`. `None` unless this sample is [`SampleValidity::Valid`].
    #[must_use]
    pub fn normalized(&self) -> Option<f32> {
        if self.validity != SampleValidity::Valid {
            return None;
        }
        self.value
            .map(|v| signal_spec(self.signal).normalization.apply(v))
    }

    /// Re-evaluate future/stale validity at `now` without re-acquiring.
    ///
    /// Missing stays missing. Samples with no engineering value keep their
    /// validity (except a future timestamp on a non-missing empty sample is
    /// [`SampleValidity::Invalid`]). Finite values are re-run through
    /// [`Self::from_raw`].
    #[must_use]
    pub fn at_time(&self, now: UnixMillis) -> Self {
        if self.value.is_none() {
            return self.clone();
        }
        Self::from_raw(self.signal, self.value, self.observed_at, self.source, now)
    }

    fn at_time_from(&self, now: UnixMillis, received_at: UnixMillis, stale_after_ms: u64) -> Self {
        if self.value.is_none() || self.validity != SampleValidity::Valid {
            return self.clone();
        }
        let mut sample = self.clone();
        sample.validity = if now.saturating_sub(received_at) >= stale_after_ms {
            SampleValidity::Stale
        } else {
            SampleValidity::Valid
        };
        sample
    }
}

/// Raw acquired readings **before** validation / normalization.
///
/// Every field is `Option`: absence means the sensor was not read. Acquisition
/// must not write `0.0` or `NaN` as a stand-in for "no data".
///
/// `source_unix_ms` is the producer's wall timestamp (NVML capture, CSV cell,
/// or simulated emit time). It is **not** the sample-sequence key; missing
/// cells stay `None` rather than a silent `0`.
#[derive(Debug, Clone, PartialEq)]
pub struct RawTelemetry {
    /// Producer timestamp in unix ms, if available.
    pub source_unix_ms: Option<UnixMillis>,
    /// Origin of the timestamp.
    pub timestamp_origin: TimestampOrigin,
    /// Provenance of this raw bag.
    pub source: TelemetrySource,
    /// GPU die temperature (°C), if read.
    pub gpu_temp_c: Option<f32>,
    /// VRAM temperature (°C), if the sensor exists and was read.
    pub vram_temp_c: Option<f32>,
    /// Board power (W), if read.
    pub power_w: Option<f32>,
    /// Derived Vcore estimate (V), if computed.
    pub vddcr_gfx_v: Option<f32>,
    /// Graphics clock (MHz), if read.
    pub gpu_clock_mhz: Option<f32>,
    /// Memory clock (MHz), if read.
    pub mem_clock_mhz: Option<f32>,
    /// Fan speed (%), if read.
    pub fan_speed_pct: Option<f32>,
    /// Memory utilization (%), if read.
    pub mem_util_pct: Option<f32>,
}

impl RawTelemetry {
    /// Documented software-only idle estimates with explicit fallback provenance.
    #[must_use]
    pub fn software_fallback(observed_at: UnixMillis) -> Self {
        Self {
            source_unix_ms: Some(observed_at),
            timestamp_origin: TimestampOrigin::Simulated,
            source: TelemetrySource::SoftwareFallback,
            gpu_temp_c: Some(software_fallback::GPU_TEMP_C),
            vram_temp_c: None,
            power_w: Some(software_fallback::POWER_W),
            vddcr_gfx_v: Some(software_fallback::VDDCR_GFX_V),
            gpu_clock_mhz: Some(software_fallback::GPU_CLOCK_MHZ),
            mem_clock_mhz: Some(software_fallback::MEM_CLOCK_MHZ),
            fan_speed_pct: Some(software_fallback::FAN_SPEED_PCT),
            mem_util_pct: Some(software_fallback::MEM_UTIL_PCT),
        }
    }

    /// NVML/driver unavailable: every channel missing, not software-idle estimates.
    #[must_use]
    pub fn nvml_unavailable(observed_at: UnixMillis) -> Self {
        Self {
            source_unix_ms: Some(observed_at),
            timestamp_origin: TimestampOrigin::LiveAcquire,
            source: TelemetrySource::NvmlUnavailable,
            gpu_temp_c: None,
            vram_temp_c: None,
            power_w: None,
            vddcr_gfx_v: None,
            gpu_clock_mhz: None,
            mem_clock_mhz: None,
            fan_speed_pct: None,
            mem_util_pct: None,
        }
    }

    fn value(&self, id: SignalId) -> Option<f32> {
        match id {
            SignalId::GpuTempC => self.gpu_temp_c,
            SignalId::VramTempC => self.vram_temp_c,
            SignalId::PowerW => self.power_w,
            SignalId::VddcrGfxV => self.vddcr_gfx_v,
            SignalId::GpuClockMhz => self.gpu_clock_mhz,
            SignalId::MemClockMhz => self.mem_clock_mhz,
            SignalId::FanSpeedPct => self.fan_speed_pct,
            SignalId::MemUtilPct => self.mem_util_pct,
        }
    }
}

/// Validated frame: per-signal [`TelemetrySample`] plus frame-level provenance.
///
/// Timing: [`Self::batch_id`] is the monotonic sample sequence (corpus-ipc
/// `batch_id`); [`Self::source_unix_ms`] is the original wall time;
/// [`Self::acquired_at`] / [`Self::received_at_unix_ms`] is receive/assess
/// time (the freshness basis).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryFrame {
    /// Relay session identifier.
    pub session_id: String,
    /// Monotonic batch index within session.
    pub batch_id: u64,
    /// Producer timestamp in unix ms, if available.
    pub source_unix_ms: Option<UnixMillis>,
    /// Receive timestamp in unix ms.
    pub received_at_unix_ms: UnixMillis,
    /// Origin of the timestamp.
    pub timestamp_origin: TimestampOrigin,
    /// Anomaly status of source vs receive time.
    pub source_time_status: SourceTimeStatus,
    /// Receive/assess time. Freshness uses this, not source wall time.
    pub acquired_at: UnixMillis,
    /// Provenance shared by every sample in this frame.
    pub source: TelemetrySource,
    /// Actual supervisor acquisition interval (`--step-interval-ms`), not the
    /// documented default in [`signal_spec`].
    pub acquisition_cadence_ms: u64,
    /// GPU die temperature sample.
    pub gpu_temp_c: TelemetrySample<f32>,
    /// VRAM temperature sample (often [`SampleValidity::Missing`]).
    pub vram_temp_c: TelemetrySample<f32>,
    /// Board power sample.
    pub power_w: TelemetrySample<f32>,
    /// Derived Vcore estimate sample.
    pub vddcr_gfx_v: TelemetrySample<f32>,
    /// Graphics clock sample.
    pub gpu_clock_mhz: TelemetrySample<f32>,
    /// Memory clock sample.
    pub mem_clock_mhz: TelemetrySample<f32>,
    /// Fan speed sample.
    pub fan_speed_pct: TelemetrySample<f32>,
    /// Memory utilization sample.
    pub mem_util_pct: TelemetrySample<f32>,
}

impl TelemetryFrame {
    /// Validate and stamp every raw channel. Does not mutate the raw bag.
    /// Uses [`DEFAULT_ACQUISITION_CADENCE_MS`] as the actual cadence.
    ///
    /// Uses an ephemeral [`SampleClock`] (new session per call). Production
    /// and multi-frame tests should use [`Self::from_raw_with_clock`].
    #[must_use]
    pub fn from_raw(raw: &RawTelemetry, now: UnixMillis) -> Self {
        Self::from_raw_with_cadence(raw, now, DEFAULT_ACQUISITION_CADENCE_MS)
    }

    /// Like [`Self::from_raw`] but records the configured acquisition interval.
    #[must_use]
    pub fn from_raw_with_cadence(raw: &RawTelemetry, now: UnixMillis, cadence_ms: u64) -> Self {
        Self::from_raw_with_clock(raw, now, cadence_ms, &mut SampleClock::new())
    }

    /// Validate, record cadence, and stamp from a shared session clock.
    ///
    /// Live collectors, software-fallback, and CSV/replay must share the same
    /// `clock` so `batch_id` is strictly increasing within one relay session.
    #[must_use]
    pub fn from_raw_with_clock(
        raw: &RawTelemetry,
        now: UnixMillis,
        cadence_ms: u64,
        clock: &mut SampleClock,
    ) -> Self {
        let timing = clock.stamp(raw.source_unix_ms, now, raw.timestamp_origin);
        let observed_at = raw.source_unix_ms.unwrap_or(now);
        let sample = |id: SignalId| {
            TelemetrySample::from_raw_with_stale_after(
                id,
                raw.value(id),
                observed_at,
                raw.source,
                now,
                stale_after_ms(id, cadence_ms),
            )
        };
        Self {
            session_id: timing.session_id,
            batch_id: timing.batch_id,
            source_unix_ms: timing.source_unix_ms,
            received_at_unix_ms: timing.received_at_unix_ms,
            timestamp_origin: timing.timestamp_origin,
            source_time_status: timing.source_time_status,
            acquired_at: timing.received_at_unix_ms,
            source: raw.source,
            acquisition_cadence_ms: cadence_ms.max(1),
            gpu_temp_c: sample(SignalId::GpuTempC),
            vram_temp_c: sample(SignalId::VramTempC),
            power_w: sample(SignalId::PowerW),
            vddcr_gfx_v: sample(SignalId::VddcrGfxV),
            gpu_clock_mhz: sample(SignalId::GpuClockMhz),
            mem_clock_mhz: sample(SignalId::MemClockMhz),
            fan_speed_pct: sample(SignalId::FanSpeedPct),
            mem_util_pct: sample(SignalId::MemUtilPct),
        }
    }

    /// Frame-level timing view (corpus-ipc `session_id` / `batch_id` names).
    #[must_use]
    pub fn timing(&self) -> FrameTiming {
        FrameTiming {
            session_id: self.session_id.clone(),
            batch_id: self.batch_id,
            source_unix_ms: self.source_unix_ms,
            received_at_unix_ms: self.received_at_unix_ms,
            timestamp_origin: self.timestamp_origin,
            source_time_status: self.source_time_status,
        }
    }

    /// All samples in inventory order (includes observability-only).
    #[must_use]
    pub fn samples(&self) -> [&TelemetrySample<f32>; 8] {
        [
            &self.gpu_temp_c,
            &self.vram_temp_c,
            &self.power_w,
            &self.vddcr_gfx_v,
            &self.gpu_clock_mhz,
            &self.mem_clock_mhz,
            &self.fan_speed_pct,
            &self.mem_util_pct,
        ]
    }

    /// Deterministic mapping toward corpus-ipc (`#40`) using wall-clock now.
    #[must_use]
    pub fn to_sensory_mapping(&self) -> SensoryMapping {
        self.to_sensory_mapping_at(unix_now_ms())
    }

    /// Mapping at an explicit instant so held frames can go stale.
    ///
    /// Includes only runtime-input candidates ([`SignalClass::RuntimeInput`] or
    /// [`SignalClass::Both`]). Observability-only channels are omitted (no filler).
    /// Normalized values are `None` unless the sample is [`SampleValidity::Valid`]
    /// *at `now`*. Each stimulus carries `stale_after_ms` and the actual
    /// `cadence_ms` so corpus-ipc consumers need not duplicate the inventory.
    #[must_use]
    pub fn to_sensory_mapping_at(&self, now: UnixMillis) -> SensoryMapping {
        let stimuli = self
            .samples()
            .into_iter()
            .filter(|s| signal_spec(s.signal).class.includes_runtime_input())
            .map(|s| {
                let stale_after_ms = stale_after_ms(s.signal, self.acquisition_cadence_ms);
                MappedStimulus::from_sample(
                    &s.at_time_from(now, self.received_at_unix_ms, stale_after_ms),
                    self.acquisition_cadence_ms,
                    stale_after_ms,
                )
            })
            .collect();
        SensoryMapping {
            session_id: self.session_id.clone(),
            batch_id: self.batch_id,
            source_unix_ms: self.source_unix_ms,
            received_at_unix_ms: self.received_at_unix_ms,
            emitted_at_unix_ms: now,
            timestamp_origin: self.timestamp_origin,
            source_time_status: self.source_time_status,
            acquisition_source: self.source,
            acquisition_cadence_ms: self.acquisition_cadence_ms,
            stimuli,
        }
    }

    /// Full raw snapshot for observability, independent of sensory normalization.
    #[must_use]
    pub fn to_observability_snapshot(&self) -> ObservabilitySnapshot {
        self.to_observability_snapshot_at(unix_now_ms())
    }

    /// Observability snapshot re-evaluated at `now`.
    #[must_use]
    pub fn to_observability_snapshot_at(&self, now: UnixMillis) -> ObservabilitySnapshot {
        ObservabilitySnapshot {
            session_id: self.session_id.clone(),
            batch_id: self.batch_id,
            source_unix_ms: self.source_unix_ms,
            received_at_unix_ms: self.received_at_unix_ms,
            emitted_at_unix_ms: now,
            timestamp_origin: self.timestamp_origin,
            source_time_status: self.source_time_status,
            acquisition_source: self.source,
            acquisition_cadence_ms: self.acquisition_cadence_ms,
            samples: self
                .samples()
                .into_iter()
                .map(|s| {
                    let stale_after_ms = stale_after_ms(s.signal, self.acquisition_cadence_ms);
                    MappedStimulus::from_sample(
                        &s.at_time_from(now, self.received_at_unix_ms, stale_after_ms),
                        self.acquisition_cadence_ms,
                        stale_after_ms,
                    )
                })
                .collect(),
        }
    }
}

/// Validate raw acquisition into a [`TelemetryFrame`].
#[must_use]
pub fn assess(raw: &RawTelemetry, now: UnixMillis) -> TelemetryFrame {
    TelemetryFrame::from_raw(raw, now)
}

/// Validate raw acquisition and record the configured acquisition cadence.
#[must_use]
pub fn assess_with_cadence(raw: &RawTelemetry, now: UnixMillis, cadence_ms: u64) -> TelemetryFrame {
    TelemetryFrame::from_raw_with_cadence(raw, now, cadence_ms)
}

/// Validate raw acquisition through a shared [`SampleClock`].
#[must_use]
pub fn assess_with_clock(
    raw: &RawTelemetry,
    now: UnixMillis,
    cadence_ms: u64,
    clock: &mut SampleClock,
) -> TelemetryFrame {
    TelemetryFrame::from_raw_with_clock(raw, now, cadence_ms, clock)
}

/// One channel in the corpus-ipc mapping surface (not a wire type).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MappedStimulus {
    /// Inventory identifier.
    pub signal: SignalId,
    /// Canonical snake_case name.
    pub name: String,
    /// How this signal may be consumed.
    pub classification: SignalClass,
    /// Engineering unit of `raw`.
    pub unit: Unit,
    /// Acquisition provenance.
    pub source: TelemetrySource,
    /// Validity at mapping time.
    pub validity: SampleValidity,
    /// Engineering-unit raw value. `None` if missing or non-finite.
    pub raw: Option<f32>,
    /// `[0, 1]` model-input. `None` unless [`SampleValidity::Valid`].
    pub normalized: Option<f32>,
    /// Per-signal stale threshold (ms). Consumers need not copy the inventory.
    pub stale_after_ms: u64,
    /// Actual acquisition cadence for this frame (`--step-interval-ms`).
    pub cadence_ms: u64,
}

impl MappedStimulus {
    fn from_sample(sample: &TelemetrySample<f32>, cadence_ms: u64, stale_after_ms: u64) -> Self {
        let spec = signal_spec(sample.signal);
        Self {
            signal: sample.signal,
            name: spec.name.to_string(),
            classification: spec.class,
            unit: sample.unit,
            source: sample.source,
            validity: sample.validity,
            raw: sample.value,
            normalized: sample.normalized(),
            stale_after_ms,
            cadence_ms,
        }
    }
}

/// Typed mapping hook for `#40`. Converted to `corpus-ipc` `StimulusBatch`
/// in [`crate::publish`] — not a wire-schema duplicate.
///
/// `session_id` / `batch_id` match corpus-ipc `StimulusBatch`. Source wall
/// time and receive/emit time are separate so RM-1144 can put emit time on
/// the wire `timestamp` and source time in metadata without guessing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensoryMapping {
    /// Relay session identifier.
    pub session_id: String,
    /// Monotonic batch index within session.
    pub batch_id: u64,
    /// Producer timestamp in unix ms, if available.
    pub source_unix_ms: Option<UnixMillis>,
    /// Receive timestamp in unix ms.
    pub received_at_unix_ms: UnixMillis,
    /// Mapping-time instant (`now` passed to [`TelemetryFrame::to_sensory_mapping_at`]).
    pub emitted_at_unix_ms: UnixMillis,
    /// Origin of the timestamp.
    pub timestamp_origin: TimestampOrigin,
    /// Anomaly status of source vs receive time.
    pub source_time_status: SourceTimeStatus,
    /// Telemetry acquisition source.
    pub acquisition_source: TelemetrySource,
    /// Actual acquisition cadence for this frame.
    pub acquisition_cadence_ms: u64,
    /// Runtime-input / Both stimuli only (no observability-only filler).
    pub stimuli: Vec<MappedStimulus>,
}

/// Raw-preserving observability view of an entire frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservabilitySnapshot {
    /// Relay session identifier.
    pub session_id: String,
    /// Monotonic batch index within session.
    pub batch_id: u64,
    /// Producer timestamp in unix ms, if available.
    pub source_unix_ms: Option<UnixMillis>,
    /// Receive timestamp in unix ms.
    pub received_at_unix_ms: UnixMillis,
    /// Mapping-time instant.
    pub emitted_at_unix_ms: UnixMillis,
    /// Origin of the timestamp.
    pub timestamp_origin: TimestampOrigin,
    /// Anomaly status of source vs receive time.
    pub source_time_status: SourceTimeStatus,
    /// Telemetry acquisition source.
    pub acquisition_source: TelemetrySource,
    /// Actual acquisition cadence for this frame.
    pub acquisition_cadence_ms: u64,
    /// Every inventory signal, raw preserved independently of normalization.
    pub samples: Vec<MappedStimulus>,
}

/// Deterministic fixtures for the contract (healthy, fallback, stale, dropout, …).
pub mod fixtures {
    use super::{RawTelemetry, TelemetrySource, TimestampOrigin, UnixMillis};
    use crate::time::parse_source_timestamp_field;

    /// Fixed timestamp so tests do not depend on wall clock.
    pub const NOW: UnixMillis = 1_700_000_000_000;

    /// CSV header shared by replay fixtures: source time then two safety signals.
    pub const CSV_HEADER: &str = "source_unix_ms,gpu_temp_c,power_w";

    /// Duplicate, backward, missing, and very large source timestamps.
    pub const CSV_SOURCE_TIME_FIXTURE: &str = "\
source_unix_ms,gpu_temp_c,power_w
1700000000000,65,200
1700000000000,66,201
1699999990000,64,190
,65,200
18446744073709551615,65,200
";

    fn nvml_at(source_unix_ms: Option<UnixMillis>) -> RawTelemetry {
        RawTelemetry {
            source_unix_ms,
            timestamp_origin: TimestampOrigin::LiveAcquire,
            source: TelemetrySource::Nvml,
            gpu_temp_c: Some(65.0),
            vram_temp_c: Some(72.0),
            power_w: Some(200.0),
            vddcr_gfx_v: Some(0.91),
            gpu_clock_mhz: Some(1_500.0),
            mem_clock_mhz: Some(6_000.0),
            fan_speed_pct: Some(40.0),
            mem_util_pct: Some(0.0),
        }
    }

    /// Healthy NVML-like readings, including a legitimate `mem_util_pct = 0.0`.
    #[must_use]
    pub fn healthy_real() -> RawTelemetry {
        nvml_at(Some(NOW))
    }

    /// Software-only idle estimates (explicit fallback provenance).
    #[must_use]
    pub fn software_fallback() -> RawTelemetry {
        RawTelemetry::software_fallback(NOW)
    }

    /// NVML unavailable: all channels missing, fail-closed safety path.
    #[must_use]
    pub fn nvml_unavailable() -> RawTelemetry {
        RawTelemetry::nvml_unavailable(NOW)
    }

    /// Same engineering values as [`healthy_real`] but observed 10s ago.
    #[must_use]
    pub fn stale() -> RawTelemetry {
        RawTelemetry {
            source_unix_ms: Some(NOW.saturating_sub(10_000)),
            ..healthy_real()
        }
    }

    /// NVML path with power sensor dropout (`None`, not `0.0` or `NaN`).
    #[must_use]
    pub fn sensor_dropout() -> RawTelemetry {
        RawTelemetry {
            power_w: None,
            vddcr_gfx_v: None,
            fan_speed_pct: None,
            ..healthy_real()
        }
    }

    /// Non-finite power; temperature still present.
    #[must_use]
    pub fn non_finite() -> RawTelemetry {
        RawTelemetry {
            power_w: Some(f32::NAN),
            ..healthy_real()
        }
    }

    /// GPU temperature far outside the documented engineering range.
    #[must_use]
    pub fn out_of_range() -> RawTelemetry {
        RawTelemetry {
            gpu_temp_c: Some(200.0),
            ..healthy_real()
        }
    }

    /// NVML readings that match the *old* magic-value heuristic (`temp <= 0 && power <= 25`).
    /// These must **not** be classified as simulated.
    #[must_use]
    pub fn nvml_looks_like_old_magic() -> RawTelemetry {
        RawTelemetry {
            gpu_temp_c: Some(0.0),
            power_w: Some(25.0),
            vddcr_gfx_v: Some(0.7),
            gpu_clock_mhz: Some(210.0),
            mem_clock_mhz: Some(405.0),
            fan_speed_pct: Some(30.0),
            mem_util_pct: Some(0.0),
            vram_temp_c: None,
            source_unix_ms: Some(NOW),
            timestamp_origin: TimestampOrigin::LiveAcquire,
            source: TelemetrySource::Nvml,
        }
    }

    /// Duplicate of [`healthy_real`]'s source timestamp (CSV replay of a repeated row).
    #[must_use]
    pub fn duplicate_source_timestamp() -> RawTelemetry {
        RawTelemetry {
            timestamp_origin: TimestampOrigin::CsvSource,
            source_unix_ms: Some(NOW),
            ..healthy_real()
        }
    }

    /// Source timestamp earlier than [`NOW`] (CSV row arrived out of order).
    #[must_use]
    pub fn backward_source_timestamp() -> RawTelemetry {
        RawTelemetry {
            timestamp_origin: TimestampOrigin::CsvSource,
            source_unix_ms: Some(NOW.saturating_sub(1_000)),
            ..healthy_real()
        }
    }

    /// CSV cell left blank — no source wall time.
    #[must_use]
    pub fn missing_source_timestamp() -> RawTelemetry {
        RawTelemetry {
            timestamp_origin: TimestampOrigin::CsvSource,
            source_unix_ms: None,
            ..healthy_real()
        }
    }

    /// Source timestamp of `u64::MAX` (ns mistakenly stored as ms, or overflow).
    #[must_use]
    pub fn very_large_source_timestamp() -> RawTelemetry {
        RawTelemetry {
            timestamp_origin: TimestampOrigin::CsvSource,
            source_unix_ms: Some(u64::MAX),
            ..healthy_real()
        }
    }

    /// Parse one `source_unix_ms,gpu_temp_c,power_w` CSV row into a raw bag.
    ///
    /// Empty source cell → missing source time. Used so CSV/replay and live
    /// collectors share [`crate::time::SampleClock`] stamping.
    pub fn raw_from_csv_row(line: &str) -> Result<RawTelemetry, String> {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() != 3 {
            return Err(format!(
                "expected 3 CSV columns (source_unix_ms,gpu_temp_c,power_w), got {}: {line}",
                cols.len()
            ));
        }
        let source_unix_ms = parse_source_timestamp_field(cols[0])?;
        let gpu_temp_c = parse_optional_f32(cols[1], "gpu_temp_c")?;
        let power_w = parse_optional_f32(cols[2], "power_w")?;
        Ok(RawTelemetry {
            source_unix_ms,
            timestamp_origin: TimestampOrigin::CsvSource,
            source: TelemetrySource::SoftwareFallback,
            gpu_temp_c,
            vram_temp_c: None,
            power_w,
            vddcr_gfx_v: None,
            gpu_clock_mhz: None,
            mem_clock_mhz: None,
            fan_speed_pct: None,
            mem_util_pct: None,
        })
    }

    fn parse_optional_f32(field: &str, name: &str) -> Result<Option<f32>, String> {
        let trimmed = field.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        trimmed
            .parse::<f32>()
            .map(Some)
            .map_err(|e| format!("invalid {name} {trimmed:?}: {e}"))
    }

    /// Data rows of [`CSV_SOURCE_TIME_FIXTURE`] (header skipped).
    pub fn csv_source_time_rows() -> impl Iterator<Item = Result<RawTelemetry, String>> {
        CSV_SOURCE_TIME_FIXTURE
            .lines()
            .filter(|l| !l.is_empty())
            .skip(1)
            .map(raw_from_csv_row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fixtures::NOW;

    fn assess_now(raw: &RawTelemetry) -> TelemetryFrame {
        assess(raw, NOW)
    }

    #[test]
    fn healthy_real_is_valid_nvml_with_legitimate_zero_util() {
        let frame = assess_now(&fixtures::healthy_real());
        assert_eq!(frame.source, TelemetrySource::Nvml);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Valid);
        assert_eq!(frame.gpu_temp_c.value, Some(65.0));
        assert_eq!(frame.power_w.validity, SampleValidity::Valid);
        assert_eq!(frame.mem_util_pct.validity, SampleValidity::Valid);
        assert_eq!(frame.mem_util_pct.value, Some(0.0));
        assert_eq!(frame.mem_util_pct.normalized(), Some(0.0));
        assert_ne!(frame.mem_util_pct.validity, SampleValidity::Missing);
    }

    #[test]
    fn software_fallback_is_explicit_source_not_magic_values() {
        let frame = assess_now(&fixtures::software_fallback());
        assert_eq!(frame.source, TelemetrySource::SoftwareFallback);
        assert_eq!(frame.gpu_temp_c.value, Some(software_fallback::GPU_TEMP_C));
        assert_eq!(frame.power_w.value, Some(software_fallback::POWER_W));
        assert_eq!(frame.vram_temp_c.value, None);
        assert_eq!(frame.vram_temp_c.validity, SampleValidity::Missing);
        assert_eq!(frame.gpu_temp_c.source, TelemetrySource::SoftwareFallback);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Valid);
        // Fallback idle is 35°C / 25W — the old heuristic would have treated
        // only temp<=0 && power<=25 as simulated. Provenance is the contract.
        assert!(frame.gpu_temp_c.value.unwrap() > 0.0);
    }

    #[test]
    fn nvml_matching_old_magic_is_not_simulated() {
        let frame = assess_now(&fixtures::nvml_looks_like_old_magic());
        assert_eq!(frame.source, TelemetrySource::Nvml);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Valid);
        assert_eq!(frame.gpu_temp_c.value, Some(0.0));
        assert_eq!(frame.power_w.value, Some(25.0));
        assert_eq!(frame.gpu_temp_c.normalized(), Some(0.0));
    }

    #[test]
    fn stale_samples_keep_raw_but_are_not_valid() {
        let frame = assess(&fixtures::stale(), NOW);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Stale);
        assert_eq!(frame.gpu_temp_c.value, Some(65.0));
        assert_eq!(frame.gpu_temp_c.normalized(), None);
        assert_eq!(frame.power_w.validity, SampleValidity::Stale);
        assert_eq!(frame.mem_util_pct.validity, SampleValidity::Stale);
    }

    #[test]
    fn sensor_dropout_is_missing_not_zero() {
        let frame = assess_now(&fixtures::sensor_dropout());
        assert_eq!(frame.source, TelemetrySource::Nvml);
        assert_eq!(frame.power_w.value, None);
        assert_eq!(frame.power_w.validity, SampleValidity::Missing);
        assert_eq!(frame.power_w.normalized(), None);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Valid);
        assert_eq!(frame.vddcr_gfx_v.validity, SampleValidity::Missing);
        assert_eq!(frame.fan_speed_pct.validity, SampleValidity::Missing);
    }

    #[test]
    fn non_finite_is_invalid_with_none_value() {
        let frame = assess_now(&fixtures::non_finite());
        assert_eq!(frame.power_w.validity, SampleValidity::Invalid);
        assert_eq!(frame.power_w.value, None);
        assert_eq!(frame.power_w.normalized(), None);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Valid);
    }

    #[test]
    fn out_of_range_keeps_raw_and_is_invalid() {
        let frame = assess_now(&fixtures::out_of_range());
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Invalid);
        assert_eq!(frame.gpu_temp_c.value, Some(200.0));
        assert_eq!(frame.gpu_temp_c.normalized(), None);
    }

    #[test]
    fn normalization_is_deterministic() {
        let frame = assess_now(&fixtures::healthy_real());
        assert_eq!(frame.gpu_temp_c.normalized(), Some(0.65));
        let expected_power = 200.0 / 350.0;
        assert!((frame.power_w.normalized().unwrap() - expected_power).abs() < 1e-6);
        assert_eq!(frame.mem_util_pct.normalized(), Some(0.0));
    }

    #[test]
    fn sensory_mapping_omits_observability_only_without_filler() {
        let frame = assess_now(&fixtures::healthy_real());
        let mapping = frame.to_sensory_mapping_at(NOW);
        assert_eq!(mapping.acquisition_source, TelemetrySource::Nvml);
        let names: Vec<&str> = mapping.stimuli.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["gpu_temp_c", "power_w", "gpu_clock_mhz", "mem_util_pct"]
        );
        assert!(!names.contains(&"vddcr_gfx_v"));
        assert!(!names.contains(&"fan_speed_pct"));
        let util = mapping
            .stimuli
            .iter()
            .find(|s| s.signal == SignalId::MemUtilPct)
            .unwrap();
        assert_eq!(util.raw, Some(0.0));
        assert_eq!(util.normalized, Some(0.0));
        assert_eq!(util.validity, SampleValidity::Valid);
    }

    #[test]
    fn observability_snapshot_preserves_raw_independently_of_normalization() {
        let frame = assess_now(&fixtures::out_of_range());
        let snap = frame.to_observability_snapshot_at(NOW);
        assert_eq!(snap.samples.len(), ALL_SIGNALS.len());
        let temp = snap
            .samples
            .iter()
            .find(|s| s.signal == SignalId::GpuTempC)
            .unwrap();
        assert_eq!(temp.raw, Some(200.0));
        assert_eq!(temp.normalized, None);
        assert_eq!(temp.validity, SampleValidity::Invalid);
    }

    #[test]
    fn missing_runtime_input_is_not_normalized_to_zero() {
        let frame = assess_now(&fixtures::sensor_dropout());
        let mapping = frame.to_sensory_mapping_at(NOW);
        let power = mapping
            .stimuli
            .iter()
            .find(|s| s.signal == SignalId::PowerW)
            .unwrap();
        assert_eq!(power.raw, None);
        assert_eq!(power.normalized, None);
        assert_eq!(power.validity, SampleValidity::Missing);
    }

    #[test]
    fn inventory_classifies_every_gpu_signal() {
        for id in ALL_SIGNALS {
            let spec = signal_spec(id);
            assert_eq!(spec.id, id);
            assert!(!spec.name.is_empty());
            assert!(spec.max > spec.min);
            assert!(spec.cadence_ms > 0);
            assert!(spec.stale_after_ms > 0);
        }
        assert_eq!(signal_spec(SignalId::GpuTempC).class, SignalClass::Both);
        assert_eq!(signal_spec(SignalId::PowerW).class, SignalClass::Both);
        assert_eq!(
            signal_spec(SignalId::GpuClockMhz).class,
            SignalClass::RuntimeInput
        );
        assert_eq!(
            signal_spec(SignalId::MemUtilPct).class,
            SignalClass::RuntimeInput
        );
        assert_eq!(
            signal_spec(SignalId::VddcrGfxV).class,
            SignalClass::ObservabilityOnly
        );
        assert_eq!(
            signal_spec(SignalId::VddcrGfxV).origin,
            SignalOrigin::Derived
        );
        assert_eq!(
            signal_spec(SignalId::VramTempC).class,
            SignalClass::ObservabilityOnly
        );
        assert_eq!(
            signal_spec(SignalId::MemClockMhz).class,
            SignalClass::ObservabilityOnly
        );
        assert_eq!(
            signal_spec(SignalId::FanSpeedPct).class,
            SignalClass::ObservabilityOnly
        );
    }

    #[test]
    fn future_observed_at_is_invalid_not_valid() {
        let mut raw = fixtures::healthy_real();
        raw.source_unix_ms = Some(NOW + 5_000);
        let frame = assess(&raw, NOW);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Invalid);
        assert_eq!(frame.gpu_temp_c.value, Some(65.0));
        assert_eq!(frame.power_w.validity, SampleValidity::Invalid);
        assert_eq!(frame.gpu_temp_c.normalized(), None);
        assert_eq!(frame.source_time_status, SourceTimeStatus::Future);
        assert_eq!(frame.source_unix_ms, Some(NOW + 5_000));
        assert_eq!(frame.received_at_unix_ms, NOW);
    }

    #[test]
    fn mapping_carries_session_batch_and_split_timestamps() {
        let mut clock = SampleClock::with_session_id("map-sess");
        let frame = assess_with_clock(&fixtures::healthy_real(), NOW, 100, &mut clock);
        let mapping = frame.to_sensory_mapping_at(NOW + 50);
        assert_eq!(mapping.session_id, "map-sess");
        assert_eq!(mapping.batch_id, 0);
        assert_eq!(mapping.source_unix_ms, Some(NOW));
        assert_eq!(mapping.received_at_unix_ms, NOW);
        assert_eq!(mapping.emitted_at_unix_ms, NOW + 50);
        assert_ne!(mapping.source_unix_ms.unwrap(), mapping.emitted_at_unix_ms);
        assert_eq!(mapping.timestamp_origin, TimestampOrigin::LiveAcquire);
        assert_eq!(mapping.source_time_status, SourceTimeStatus::InOrder);
    }

    #[test]
    fn csv_fixture_rows_share_clock_with_live_and_flag_source_anomalies() {
        let mut clock = SampleClock::with_session_id("csv-live");
        let live = assess_with_clock(&fixtures::healthy_real(), NOW, 100, &mut clock);
        assert_eq!(live.timestamp_origin, TimestampOrigin::LiveAcquire);
        assert_eq!(live.batch_id, 0);

        let rows: Vec<RawTelemetry> = fixtures::csv_source_time_rows()
            .collect::<Result<Vec<_>, _>>()
            .expect("CSV fixture must parse");
        assert_eq!(rows.len(), 5);

        let stamped: Vec<TelemetryFrame> = rows
            .iter()
            .map(|raw| assess_with_clock(raw, NOW, 100, &mut clock))
            .collect();
        assert!(stamped.iter().all(|f| f.session_id == live.session_id));
        assert_eq!(stamped[0].batch_id, 1);
        assert_eq!(stamped[1].batch_id, 2);
        assert_eq!(stamped[2].batch_id, 3);
        assert_eq!(stamped[3].batch_id, 4);
        assert_eq!(stamped[4].batch_id, 5);
        assert!(stamped.windows(2).all(|w| w[1].batch_id > w[0].batch_id));

        assert_eq!(stamped[0].source_time_status, SourceTimeStatus::Duplicate);
        assert_eq!(stamped[1].source_time_status, SourceTimeStatus::Duplicate);
        assert_eq!(stamped[2].source_time_status, SourceTimeStatus::Backward);
        assert_eq!(stamped[3].source_time_status, SourceTimeStatus::Missing);
        assert_eq!(stamped[4].source_time_status, SourceTimeStatus::Future);
        assert_eq!(stamped[4].gpu_temp_c.validity, SampleValidity::Invalid);
        assert_eq!(stamped[3].source_unix_ms, None);
        // Missing source uses receive time for sample validity (age ≈ 0).
        assert_eq!(stamped[3].gpu_temp_c.observed_at, NOW);
        assert_eq!(stamped[3].gpu_temp_c.validity, SampleValidity::Valid);
        assert_eq!(stamped[0].timestamp_origin, TimestampOrigin::CsvSource);
        assert_eq!(
            fixtures::CSV_SOURCE_TIME_FIXTURE.lines().next().unwrap(),
            fixtures::CSV_HEADER
        );
    }

    #[test]
    fn named_source_timestamp_fixtures_cover_duplicate_backward_missing_large() {
        assert_eq!(
            fixtures::duplicate_source_timestamp().source_unix_ms,
            Some(NOW)
        );
        assert_eq!(
            fixtures::backward_source_timestamp().source_unix_ms,
            Some(NOW - 1_000)
        );
        assert_eq!(fixtures::missing_source_timestamp().source_unix_ms, None);
        assert_eq!(
            fixtures::very_large_source_timestamp().source_unix_ms,
            Some(u64::MAX)
        );
    }

    #[test]
    fn mapping_re_evaluates_stale_and_carries_thresholds() {
        let frame = assess_now(&fixtures::healthy_real());
        let fresh = frame.to_sensory_mapping_at(NOW);
        let temp = fresh
            .stimuli
            .iter()
            .find(|s| s.signal == SignalId::GpuTempC)
            .unwrap();
        assert_eq!(temp.validity, SampleValidity::Valid);
        assert_eq!(temp.normalized, Some(0.65));
        assert_eq!(temp.stale_after_ms, SAFETY_STALE_AFTER_MS);
        assert_eq!(temp.cadence_ms, DEFAULT_ACQUISITION_CADENCE_MS);
        assert_eq!(fresh.acquisition_cadence_ms, DEFAULT_ACQUISITION_CADENCE_MS);

        let later = NOW + SAFETY_STALE_AFTER_MS;
        let mapping = frame.to_sensory_mapping_at(later);
        let temp = mapping
            .stimuli
            .iter()
            .find(|s| s.signal == SignalId::GpuTempC)
            .unwrap();
        assert_eq!(temp.validity, SampleValidity::Stale);
        assert_eq!(temp.raw, Some(65.0));
        assert_eq!(temp.normalized, None);
    }

    #[test]
    fn mapping_reports_configured_acquisition_cadence() {
        let frame = assess_with_cadence(&fixtures::healthy_real(), NOW, 50);
        assert_eq!(frame.acquisition_cadence_ms, 50);
        assert_eq!(
            signal_spec(SignalId::GpuTempC).cadence_ms,
            DEFAULT_ACQUISITION_CADENCE_MS
        );
        let mapping = frame.to_sensory_mapping_at(NOW);
        assert_eq!(mapping.acquisition_cadence_ms, 50);
        assert!(mapping.stimuli.iter().all(|s| s.cadence_ms == 50));
    }

    #[test]
    fn safety_staleness_tracks_50_and_500_ms_cadences_at_the_boundary() {
        for (cadence_ms, threshold_ms) in [(50, 1_000), (500, 10_000)] {
            let frame = assess_with_cadence(&fixtures::healthy_real(), NOW, cadence_ms);
            let just_fresh = frame.to_sensory_mapping_at(NOW + threshold_ms - 1);
            let at_boundary = frame.to_sensory_mapping_at(NOW + threshold_ms);
            for signal in [SignalId::GpuTempC, SignalId::PowerW] {
                let fresh = just_fresh
                    .stimuli
                    .iter()
                    .find(|stimulus| stimulus.signal == signal)
                    .unwrap();
                let stale = at_boundary
                    .stimuli
                    .iter()
                    .find(|stimulus| stimulus.signal == signal)
                    .unwrap();
                assert_eq!(fresh.stale_after_ms, threshold_ms);
                assert_eq!(fresh.validity, SampleValidity::Valid);
                assert_eq!(stale.validity, SampleValidity::Stale);
            }
        }
    }

    #[test]
    fn source_stale_frame_stays_stale_for_safety_and_mapping() {
        let mut jumped_source = fixtures::healthy_real();
        jumped_source.source_unix_ms = Some(NOW - SAFETY_STALE_AFTER_MS);
        let frame = assess(&jumped_source, NOW);
        assert_eq!(frame.gpu_temp_c.observed_at, NOW - SAFETY_STALE_AFTER_MS);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Stale);
        assert_eq!(frame.power_w.validity, SampleValidity::Stale);
        assert!(matches!(
            crate::safety::instant_status(&frame),
            (crate::safety::SafetyStatus::Critical(reason), false) if reason.contains("stale")
        ));

        let mapping = frame.to_sensory_mapping_at(NOW);
        let temp = mapping
            .stimuli
            .iter()
            .find(|stimulus| stimulus.signal == SignalId::GpuTempC)
            .unwrap();
        assert_eq!(temp.validity, SampleValidity::Stale);
        assert_eq!(temp.raw, Some(65.0));
        assert_eq!(temp.normalized, None);
    }

    #[test]
    fn mapping_wall_clock_regression_keeps_initially_valid_frame_valid() {
        let mut slightly_older_source = fixtures::healthy_real();
        slightly_older_source.source_unix_ms = Some(NOW - 100);
        let frame = assess(&slightly_older_source, NOW);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Valid);

        let mapping = frame.to_sensory_mapping_at(NOW - 1);
        let temp = mapping
            .stimuli
            .iter()
            .find(|stimulus| stimulus.signal == SignalId::GpuTempC)
            .unwrap();
        assert_eq!(temp.validity, SampleValidity::Valid);
        assert_eq!(temp.normalized, Some(0.65));
    }

    #[test]
    fn nvml_unavailable_is_missing_not_software_fallback() {
        let frame = assess_now(&fixtures::nvml_unavailable());
        assert_eq!(frame.source, TelemetrySource::NvmlUnavailable);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Missing);
        assert_eq!(frame.power_w.validity, SampleValidity::Missing);
        assert_eq!(frame.gpu_temp_c.value, None);
    }

    #[test]
    fn identity_and_degenerate_linear_normalization() {
        assert_eq!(Normalization::Identity.apply(42.0), 42.0);
        assert_eq!(Normalization::Identity.apply(-3.5), -3.5);
        let collapsed = Normalization::Linear {
            min: 10.0,
            max: 10.0,
        };
        assert_eq!(collapsed.apply(10.0), 0.0);
        let non_finite = Normalization::Linear {
            min: f32::NAN,
            max: 1.0,
        };
        assert_eq!(non_finite.apply(0.5), 0.0);
        let linear = Normalization::Linear {
            min: 0.0,
            max: 100.0,
        };
        assert_eq!(linear.apply(-10.0), 0.0);
        assert_eq!(linear.apply(150.0), 1.0);
        assert_eq!(linear.apply(25.0), 0.25);
    }

    #[test]
    fn signal_class_and_name_helpers() {
        assert!(SignalClass::Both.includes_runtime_input());
        assert!(SignalClass::Both.includes_safety());
        assert!(SignalClass::RuntimeInput.includes_runtime_input());
        assert!(!SignalClass::RuntimeInput.includes_safety());
        assert!(!SignalClass::SafetyOnly.includes_runtime_input());
        assert!(SignalClass::SafetyOnly.includes_safety());
        assert!(!SignalClass::ObservabilityOnly.includes_runtime_input());
        assert!(!SignalClass::ObservabilityOnly.includes_safety());
        for id in ALL_SIGNALS {
            assert_eq!(id.name(), signal_spec(id).name);
        }
        assert_eq!(SignalId::GpuTempC.name(), "gpu_temp_c");
    }

    #[test]
    fn engineering_range_is_inclusive_and_below_min_is_invalid() {
        let mut at_min = fixtures::healthy_real();
        at_min.gpu_temp_c = Some(0.0);
        let frame = assess_now(&at_min);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Valid);
        assert_eq!(frame.gpu_temp_c.value, Some(0.0));

        let mut at_max = fixtures::healthy_real();
        at_max.gpu_temp_c = Some(125.0);
        let frame = assess_now(&at_max);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Valid);
        assert_eq!(frame.gpu_temp_c.value, Some(125.0));
        assert_eq!(frame.gpu_temp_c.normalized(), Some(1.0));

        let mut below = fixtures::healthy_real();
        below.gpu_temp_c = Some(-1.0);
        let frame = assess_now(&below);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Invalid);
        assert_eq!(frame.gpu_temp_c.value, Some(-1.0));
        assert_eq!(frame.gpu_temp_c.normalized(), None);
    }

    #[test]
    fn future_missing_stays_missing_and_future_non_finite_is_invalid() {
        let mut missing = fixtures::healthy_real();
        missing.source_unix_ms = Some(NOW + 5_000);
        missing.gpu_temp_c = None;
        let frame = assess(&missing, NOW);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Missing);
        assert_eq!(frame.gpu_temp_c.value, None);

        let mut nan = fixtures::healthy_real();
        nan.source_unix_ms = Some(NOW + 5_000);
        nan.power_w = Some(f32::NAN);
        let frame = assess(&nan, NOW);
        assert_eq!(frame.power_w.validity, SampleValidity::Invalid);
        assert_eq!(frame.power_w.value, None);
    }

    #[test]
    fn infinity_is_invalid_with_none_value() {
        let mut raw = fixtures::healthy_real();
        raw.gpu_temp_c = Some(f32::INFINITY);
        let frame = assess_now(&raw);
        assert_eq!(frame.gpu_temp_c.validity, SampleValidity::Invalid);
        assert_eq!(frame.gpu_temp_c.value, None);
        assert_eq!(frame.gpu_temp_c.normalized(), None);
    }

    #[test]
    fn at_time_leaves_missing_alone_and_can_stale_a_valid_sample() {
        let dropout = assess_now(&fixtures::sensor_dropout());
        let later = dropout.power_w.at_time(NOW + 10_000);
        assert_eq!(later.validity, SampleValidity::Missing);
        assert_eq!(later.value, None);

        let healthy = assess_now(&fixtures::healthy_real());
        assert_eq!(healthy.gpu_temp_c.validity, SampleValidity::Valid);
        let stale = healthy.gpu_temp_c.at_time(NOW + SAFETY_STALE_AFTER_MS);
        assert_eq!(stale.validity, SampleValidity::Stale);
        assert_eq!(stale.value, Some(65.0));
        assert_eq!(stale.normalized(), None);
    }

    #[test]
    fn zero_cadence_is_floored_to_one() {
        let frame = assess_with_cadence(&fixtures::healthy_real(), NOW, 0);
        assert_eq!(frame.acquisition_cadence_ms, 1);
        let mapping = frame.to_sensory_mapping_at(NOW);
        assert_eq!(mapping.acquisition_cadence_ms, 1);
        assert!(mapping.stimuli.iter().all(|s| s.cadence_ms == 1));
    }

    #[test]
    fn unix_now_ms_is_epoch_based() {
        let now = unix_now_ms();
        assert!(
            now >= NOW,
            "unix_now_ms={now} should be at/after fixture NOW"
        );
    }
}
