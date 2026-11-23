# Telemetry contract

This is the normative validity / freshness / normalization / provenance
contract for `vahtisiru` ([GH#41](https://github.com/rmems/vahtisiru/issues/41)).

The frozen **CSV interchange** consumed by corinth-canal
(`timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w`) is a
separate contract in [`docs/telemetry_csv.md`](telemetry_csv.md) and
`vahtisiru::telemetry_csv` ([RM-629](https://linear.app/rpd-34/issue/RM-629)).
Do not treat CSV columns as [`TelemetrySample`] values: CSV fields are
required finite numbers, not `Option`.

Invalid, missing, stale, and simulated data is **never** inferred from magic
numeric values. Missing sensors stay `None`; they are never silently converted
into a legitimate `0.0`.

Acquisition (`src/gpu.rs`) is separate from validation / normalization / encoding
(`src/telemetry.rs`). Safety policy (`src/safety.rs`) consumes [`TelemetryFrame`]
samples and does not depend on corpus-ipc transport ([GH#40](https://github.com/rmems/vahtisiru/issues/40)).
Named relay states and hysteresis: [`docs/safety.md`](safety.md) (GH#42).

## Typed sample

```text
TelemetrySample<T> {
    value: Option<T>,       // engineering units; None if missing or non-finite
    observed_at: unix ms,   // source wall time, or receive time if source missing
    source: TelemetrySource, // Nvml | SoftwareFallback | NvmlUnavailable
    validity: SampleValidity, // Valid | Missing | Invalid | Stale
    unit: Unit,
}
```

## Sample clock and timestamp provenance (RM-1335)

Wall-clock source timestamps can jump, repeat, or arrive out of order.
Every assessed / emitted frame is stamped by a process-local
[`SampleClock`] (`src/time.rs`):

| Field | Meaning |
| --- | --- |
| `session_id` | Stable boot/session id (corpus-ipc `StimulusBatch.session_id`). A new `SampleClock` (process restart) is a new session. |
| `batch_id` | Strictly increasing sample sequence within that session (corpus-ipc `batch_id`; first frame is 0). Resets to 0 on restart. |
| `source_unix_ms` | Original producer wall time (`None` if the CSV cell / producer omitted it). |
| `received_at_unix_ms` / `acquired_at` | Unix-epoch milliseconds at receive/assessment. Held-frame expiry uses elapsed time from this receive instant; initial source-time validation remains separate. |
| `emitted_at_unix_ms` | Mapping-time instant (`to_sensory_mapping_at(now)`). Distinct from receive when a held frame is mapped later. |
| `timestamp_origin` | `LiveAcquire` \| `CsvSource` \| `Simulated` |
| `source_time_status` | `InOrder` \| `Duplicate` \| `Backward` \| `Future` \| `Missing` |

Duplicate, backward, missing, and very large source timestamps **do not
regress** `batch_id`. They are flagged on `source_time_status`. Future
source time (`source_unix_ms > now`, including `u64::MAX`) is also
[`SampleValidity::Invalid`] — the existing validity policy. Missing source
time uses receive time for sample age (so a just-received row is not
spuriously stale) and is flagged `Missing`.

Consumers must key frames by `(session_id, batch_id)`, never `batch_id`
alone. Live NVML, `--force-software-only`, and CSV/replay rows share this
contract by passing the same `SampleClock` into
`TelemetryFrame::from_raw_with_clock` / `assess_with_clock`.

Prometheus `telemetry_freshness_s` is computed from a monotonic
`Instant` captured at receive, not from source wall time.

Simulated data is [`TelemetrySource::SoftwareFallback`] and is used only for
`--force-software-only`. NVML/driver/device lookup failure is
[`TelemetrySource::NvmlUnavailable`] (all safety samples `Missing`) and
**fail-closes** safety — it is not treated as simulated idle. An NVML sample
that happens to read `0 °C` and `25 W` is still `source = Nvml` and may be a
legitimate idle GPU.

The supervisor takes its process lock before startup NVML queries. Startup
reads have a two-second wait limit; per-tick and post-actuation reads share a
900 ms budget per step, so a wedged driver is evaluated fail-closed within 1 s.
One shared gate admits only one blocking NVML operation at a time, including
power-limit actuation queries. A timed-out driver call may remain blocked in
its worker because NVML does not provide safe cancellation; the gate stays
occupied until that call returns. Later real reads report `NvmlUnavailable`
without launching more blocked workers. An unavailable tick is evaluated by
the safety machine immediately. Forced software-only telemetry keeps its
explicit `SoftwareFallback` provenance and never runs the probe below.

Driver health (`timeout -k 1s 1s nvidia-smi -L`) is a cached verdict, not a
per-tick command: see [the safety contract](safety.md) for its cadence.

## Pipeline

```text
RawTelemetry          (optional engineering values + source time + origin)
        │
        ▼  assess() / assess_with_clock() / TelemetryFrame::from_raw_with_clock()
TelemetryFrame        (per-signal TelemetrySample + session_id/batch_id + split times)
        │
        ├─ SafetyMachine::evaluate()      isolated; no IPC  → GH#42
        ├─ instant_status() / classify_frame()  instantaneous Ok/Warn/Critical
        ├─ to_sensory_mapping()            runtime-input / Both  → GH#40
        └─ to_observability_snapshot()      every signal, raw preserved
```

The supervisor initially validates each acquisition against its source time;
an already-old source reading is `Stale` and fails closed for safety. The
sample sequence still advances when
CSV/replay source times duplicate or go backward. If NVML is unavailable it emits `NvmlUnavailable` with missing safety samples
(fail closed), not `SoftwareFallback`. `SoftwareFallback` is reserved for
`--force-software-only`. `to_sensory_mapping_at(now)` takes `now` in
Unix-epoch milliseconds and re-evaluates freshness against
`received_at_unix_ms` for samples that passed initial validation, so a held
frame older than the per-signal stale threshold is `Stale` and drops its
normalized value. Mapping never upgrades an initially `Missing`, `Invalid`,
or `Stale` sample to `Valid`. If the relay wall clock moves backward after
receipt, elapsed receive time saturates to zero instead of manufacturing an
`Invalid` sample. Mapping carries `stale_after_ms` and the actual
`acquisition_cadence_ms` (`--step-interval-ms`). `SignalSpec.cadence_ms` is
the documented default (100 ms). For safety signals, `stale_after_ms` is
`2 × 10 × acquisition_cadence_ms`: two complete supervisor safety-evaluation
periods. Thus 50 ms and 500 ms acquisition intervals yield 1 s and 10 s
thresholds respectively. Future `observed_at > now` is `Invalid` during
initial source-timestamp validation; missing values remain `Missing`.

## Signal inventory

| Signal | Unit | Range | Origin | Class | Normalization | Cadence | Stale after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gpu_temp_c` | °C | 0…125 | measured | **both** (safety + runtime-input) | linear 0…100 → `[0, 1]` | 100 ms default | 2 safety-eval periods (2 s at default) |
| `power_w` | W | 0…2000 | measured | **both** | linear 0…350 → `[0, 1]` | 100 ms default | 2 safety-eval periods (2 s at default) |
| `gpu_clock_mhz` | MHz | 0…3000 | measured | runtime-input | linear 0…2500 → `[0, 1]` | 100 ms | 5 s |
| `mem_util_pct` | % | 0…100 | measured | runtime-input | linear 0…100 → `[0, 1]` | 100 ms | 5 s |
| `vram_temp_c` | °C | 0…125 | measured | observability-only | linear 0…100 → `[0, 1]` | 100 ms | 5 s |
| `mem_clock_mhz` | MHz | 0…12000 | measured | observability-only | linear 0…10000 → `[0, 1]` | 100 ms | 5 s |
| `fan_speed_pct` | % | 0…100 | measured | observability-only | linear 0…100 → `[0, 1]` | 100 ms | 5 s |
| `vddcr_gfx_v` | V | 0.4…1.5 | **derived** from power | observability-only | linear 0.5…1.2 → `[0, 1]` | 100 ms | 5 s |

Temperature and power are classified **both**. Configurable safety limits
(`SafetyPolicyConfig`) are separate from sensor sanity ranges and sensory
normalization. Defaults are 75/85 C and 85%/100% of the reported device default
power limit, with validated operator overrides. The 2000 W sanity ceiling permits
high-power devices; it is not a safety recommendation. The existing 0…350 W
sensory normalization remains unchanged and saturates at 1 above 350 W.
See [safety policy](safety.md).

`vddcr_gfx_v` is not an NVML voltage sensor. The identifier is historical
(AMD `VDDCR_GFX` rail naming). It is derived from board power for
observability and is omitted from the sensory mapping (no fabricated extra
signals).

`vram_temp_c` is not fabricated as `gpu_temp + 8`. `nvml-wrapper` 0.10 only
exposes `TemperatureSensor::Gpu`, so VRAM temperature is `Missing` until a
future adapter can read it.

## Safety vs sensory vs observability

- **Safety-critical** (`gpu_temp_c`, `power_w`): missing / invalid / stale fail
  closed (`SafetyStatus::Critical`). Software-fallback frames skip hardware
  thresholds because provenance says there is no real GPU to protect.
  `NvmlUnavailable` does **not** skip: missing safety samples are Critical.
- **Runtime-input candidates**: `gpu_temp_c`, `power_w`, `gpu_clock_mhz`,
  `mem_util_pct`. These are the only channels in `SensoryMapping` (a mapping
  hook, not a live Brainstem feed).
- **Observability-only**: `vram_temp_c`, `vddcr_gfx_v`, `mem_clock_mhz`,
  `fan_speed_pct`. Present in `ObservabilitySnapshot` with raw values even
  when invalid; never used as silent sensory-mapping zeros.

A legitimate zero (for example `mem_util_pct = 0.0` while `Valid`) is
distinct from missing (`value = None`, `validity = Missing`).

## Software fallback vs NVML unavailable

`--force-software-only` emits the documented idle estimates in
`telemetry::software_fallback` with `TelemetrySource::SoftwareFallback`. VRAM
temperature stays `None`. These numbers are estimates, not a second set of
magic flags.

When NVML/driver/device lookup fails *without* that flag, acquisition emits
`TelemetrySource::NvmlUnavailable` with every channel `Missing`. Safety
fail-closes. This is not software-only confirmation.

## corpus-ipc mapping surface (#41 owns types, #40 owns transport)

`TelemetryFrame::to_sensory_mapping_at(now)` produces `SensoryMapping` /
`MappedStimulus` with `session_id`, `batch_id`, source vs receive vs emit
timestamps, source-time status, acquisition source, validity (re-evaluated
at `now`), raw engineering value, normalized `[0, 1]` (only when `Valid`
at `now`), `stale_after_ms`, and the actual `cadence_ms`. The type name
`MappedStimulus` is historical (retired UDP `Stimuli` protocol); it is
**not** a neural stimulus and Vahtisiru does not run an SNN. `src/publish.rs`
(#40) maps this into published `corpus-ipc` `StimulusBatch` /
`IpcMessage::Stimuli` off the safety path. `telemetry` does not depend on
`corpus-ipc` and does not duplicate the wire schema.

## Fixtures

`telemetry::fixtures` provides deterministic bags for tests:

- `healthy_real` — NVML-like, including legitimate `mem_util_pct = 0.0`
- `software_fallback` — explicit simulated idle (`--force-software-only`)
- `nvml_unavailable` — all channels missing, fail-closed safety path
- `stale` — healthy values with `source_unix_ms` 10 s in the past
- `sensor_dropout` — `power_w = None` (not `0.0` / `NaN`)
- `non_finite` — `power_w = NaN`
- `out_of_range` — `gpu_temp_c = 200`
- `nvml_looks_like_old_magic` — NVML `0 °C` / `25 W`, **not** simulated
- `duplicate_source_timestamp` / `backward_source_timestamp` /
  `missing_source_timestamp` / `very_large_source_timestamp` — source-time
  anomalies (CSV/replay)
- `CSV_SOURCE_TIME_FIXTURE` / `raw_from_csv_row` — the same four anomalies
  as CSV rows, stamped through the shared `SampleClock`
