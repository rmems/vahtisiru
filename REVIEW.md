# Vahtisiru — Review Guide

## Running Tests

```bash
cargo test                    # all tests
cargo test --lib              # telemetry + telemetry_csv + gpu + cpu + safety + publish
cargo test --test software_only  # GPU-less integration harness (RM-1217)
cargo test --test safety_independence  # GH#42 learned-component independence, boundaries, trip events
cargo test telemetry          # typed contract / fixtures
cargo test gpu                # gpu acquisition + check_safety facade
cargo test safety              # SafetyMachine + named states (no GPU)
cargo test publish             # IPC isolation / StimulusBatch mapping / bounded queue policy
cargo test time               # sample clock + timestamp provenance
cargo test freshness          # freshness gauge basis / boundaries
cargo test telemetry_csv      # frozen CSV header / parse / replay (RM-629)
cargo test cpu                # cpu metrics tests
cargo test queue              # bounded queue capacity, policy, overflow counters
cargo test metrics            # metrics defaults, safety snapshot copy, queue exposition
cargo test test_safety        # safety-related tests
cargo test lock_guard         # single-instance lockfile tests
```

### Test inventory

| Binary | Test | What it covers |
|--------|------|----------------|
| lib.rs | `header_is_the_frozen_five_column_string` | CSV `HEADER` lock (RM-629) |
| lib.rs | `load_csv_accepts_canonical_header_and_parses_rows` | Canonical 5-column parse |
| lib.rs | `load_csv_rejects_bad_header` | Header mismatch fails closed |
| lib.rs | `load_csv_rejects_session_label_extension_as_header_mismatch` | Extra `session_label` column is not compatible |
| lib.rs | `load_csv_skips_malformed_short_rows` | Short rows + NaN skipped |
| lib.rs | `load_csv_skips_non_numeric_fields` | Non-numeric fields skipped |
| lib.rs | `load_csv_skips_non_finite_and_extra_columns` | Inf / extra columns skipped |
| lib.rs | `row_for_tick_wraps_around_and_rewrites_timestamp` | Replay wrap + `tick + 1` |
| lib.rs | `format_csv_round_trips_through_parse` | Producer emit → validate |
| lib.rs | `healthy_real_is_valid_nvml_with_legitimate_zero_util` | Healthy NVML fixture; 0% util ≠ missing |
| lib.rs | `software_fallback_is_explicit_source_not_magic_values` | Software fallback provenance |
| lib.rs | `nvml_matching_old_magic_is_not_simulated` | `0°C`/`25W` NVML is not simulated |
| lib.rs | `stale_samples_keep_raw_but_are_not_valid` | Stale keeps raw, normalized `None` |
| lib.rs | `sensor_dropout_is_missing_not_zero` | Dropout is `None`, not `0.0` |
| lib.rs | `non_finite_is_invalid_with_none_value` | NaN → invalid + `None` |
| lib.rs | `out_of_range_keeps_raw_and_is_invalid` | Out-of-range preserves raw |
| lib.rs | `normalization_is_deterministic` | Linear `[0, 1]` mapping |
| lib.rs | `sensory_mapping_omits_observability_only_without_filler` | corpus-ipc mapping surface |
| lib.rs | `observability_snapshot_preserves_raw_independently_of_normalization` | Raw vs normalized split |
| lib.rs | `missing_runtime_input_is_not_normalized_to_zero` | Missing ≠ normalized 0 |
| lib.rs | `inventory_classifies_every_gpu_signal` | Signal class/unit/range inventory |
| lib.rs | `test_raw_software_fallback_never_uses_silent_zero_for_missing_vram` | VRAM stays missing in fallback |
| lib.rs | `test_read_telemetry_force_software_only` | Forced software-only telemetry read |
| lib.rs | `test_is_gpu_healthy_does_not_panic` | GPU health check no-panic |
| lib.rs | `test_safety_ok_on_simulated_values` | Simulated telemetry skips safety |
| lib.rs | `test_safety_does_not_infer_simulation_from_old_magic_values` | No magic-value sim inference |
| lib.rs | `test_safety_warn_on_elevated_temp` | 75-85°C warning threshold |
| lib.rs | `test_safety_warn_on_elevated_power` | 300-350W warning threshold |
| lib.rs | `test_safety_critical_on_high_temp` | >85°C critical threshold |
| lib.rs | `test_safety_critical_on_high_power` | >350W critical threshold |
| lib.rs | `test_safety_critical_on_non_finite_telemetry` | NaN/Inf telemetry handling |
| lib.rs | `test_safety_critical_on_unknown_power_with_real_temperature` | Missing power with real temp |
| lib.rs | `test_check_safety_fail_closes_on_missing_temp_or_power` | Missing/None does not skip safety |
| lib.rs | `test_nvml_unavailable_fail_closes_safety` | Unavailable NVML is not simulated |
| lib.rs | `test_acquire_raw_force_software_is_fallback_not_unavailable` | force vs unavailable provenance |
| lib.rs | `future_observed_at_is_invalid_not_valid` | Future timestamps are Invalid |
| lib.rs | `mapping_carries_session_batch_and_split_timestamps` | Source vs receive vs emit |
| lib.rs | `csv_fixture_rows_share_clock_with_live_and_flag_source_anomalies` | CSV + live share SampleClock |
| lib.rs | `sample_clock_is_strictly_increasing_within_one_session` | Monotonic batch_id |
| lib.rs | `duplicate_source_timestamps_do_not_regress_the_sample_clock` | Duplicate source time |
| lib.rs | `backward_source_timestamps_do_not_regress_the_sample_clock` | Backward source time |
| lib.rs | `missing_source_timestamp_is_flagged_and_clock_still_advances` | Missing source time |
| lib.rs | `very_large_source_timestamp_is_future_and_does_not_regress_clock` | u64::MAX source time |
| lib.rs | `restart_resets_are_distinguishable_by_session_id` | New clock ⇒ new session |
| lib.rs | `freshness_seconds_uses_receive_time_not_source_time` | Freshness basis |
| lib.rs | `freshness_seconds_boundaries` | None / equal / backward now / max |
| lib.rs | `mapping_re_evaluates_stale_and_carries_thresholds` | Mapping-time freshness + stale_after |
| lib.rs | `mapping_reports_configured_acquisition_cadence` | Actual vs default cadence |
| lib.rs | `freshness_seconds_grows_when_export_time_advances` | Gauge age at scrape time |
| lib.rs | `test_safety_ok_on_normal_telemetry` | Normal readings pass |
| lib.rs | `test_safety_critical_on_stale_and_out_of_range` | Stale/OOR fail closed |
| lib.rs | `test_sensory_mapping_from_software_fallback_carries_provenance` | Mapping carries fallback source |
| lib.rs | `relay_metrics_default_values` | RelayMetrics default acquired_at |
| lib.rs | `record_safety_snapshot_copies_state_and_brake` | Shared metrics copy safety snapshot |
| lib.rs | `healthy_real_is_named_state` … `transitions_are_counted` | `SafetyMachine` GPU-less transitions (`src/safety.rs`) |
| lib.rs | `absent_brainstem_does_not_block_or_change_safety` … | IPC/Brainstem absence cannot stall safety (`src/publish.rs`) |
| lib.rs | `queue_capacity_is_validated` … `queue_metrics_exposition_shows_forced_overflow` | Bounded queue, policies, drop-reason cardinality (`src/publish.rs`) |
| lib.rs | `metrics_queue_snapshot_names_are_stable` | Queue Prometheus names under overflow (`src/cpu.rs`) |
| lib.rs | `mapping_to_stimulus_batch_carries_timestamp_source_and_validity` | GH#41 fields on canonical `StimulusBatch` |
| lib.rs | `missing_channel_is_masked_not_a_real_zero` | `valid_mask` vs legitimate zero |
| lib.rs | `stimulus_batch_round_trips_through_published_corpus_ipc_types` | serde round-trip of `IpcMessage::Stimuli` |
| lib.rs | `corpus_ipc_publisher_slow_consumer_does_not_block_safety` | Bounded policy-queue isolation |
| lib.rs | `software_only_emits_typed_corpus_ipc_frame_without_gpu` | Software-only UDP emit, no GPU |
| daemon.rs | `parses_ipc_flags` | `--ipc-endpoint` / `--ipc-disabled` / `--ipc-session-id` |
| daemon.rs | `dashboard_shows_only_valid_present_as_live_hardware` | Dashboard hides stale/invalid/missing |
| daemon.rs | `parses_custom_args_and_env_equiv` | CLI flag parsing |
| daemon.rs | `parses_defaults` | CLI defaults (including sensory-queue capacity/policy) |
| daemon.rs | `parses_sensory_queue_config` / `rejects_zero_queue_capacity` / `rejects_unknown_queue_policy` | Queue CLI validation |
| daemon.rs | `parses_force_software_only_false` | `--force-software-only=false` |
| daemon.rs | `lock_guard_created_and_removed` | Lockfile create/drop lifecycle |
| daemon.rs | `lock_guard_rejects_active_pid` | Second instance refused while PID alive |
| daemon.rs | `lock_guard_reclaims_stale_lock` | Stale lock (dead PID) reclaimed |
| tests/software_only.rs | `binary_help_and_version_exit_zero_without_gpu` | CLI `--help`/`-V` without NVML |
| tests/software_only.rs | `binary_software_only_starts_without_nvidia` | Forced software-only process start |
| tests/software_only.rs | `software_only_pipeline_evaluates_safety_then_best_effort_publish` | Acquire → evaluate → absent publish |
| tests/software_only.rs | `ipc_failure_cannot_stall_safety_loop_progress` | Hysteresis while IPC send fails |
| tests/software_only.rs | `missing_invalid_stale_and_simulated_frames_are_named_states` | Named fail-closed / simulated states |
| tests/software_only.rs | `slow_and_disconnected_publish_queues_do_not_block_critical_brake` | Disconnected queue ≠ safety pause |
| tests/software_only.rs | `acquire_without_force_is_not_simulated_on_ci` | No-GPU acquire is NvmlUnavailable, not sim |

As of RM-1143 (GH#39), `vahtisiru` no longer runs an in-process SNN or
exposes a UDP control surface — the `Stimuli`/`LearningReward`/`GetNeuroState`
tests and the `num-channels`/`num-lif`/`num-izh` CLI tests were removed along
with that code. See [`docs/ipc.md`](docs/ipc.md).

## Coverage update (RM-1217)

Post-GH#39 inventory of production branches that still needed deterministic,
GPU-less tests. Unit coverage already landed with GH#41/#42/#46; this change
locks the remaining edges and adds an integration harness under `tests/`.

### Gaps closed

- Telemetry: identity/degenerate normalization, inclusive engineering range,
  below-min / `+inf` / future-missing / `at_time` staleness, cadence floor of 1,
  `SignalClass` helpers, derived `vddcr_gfx_v`.
- Safety: exclusive critical thresholds, mixed-fault ranking (missing > invalid
  > stale; equal rank prefers `gpu_temp_c`), leftover-brake + simulated hold,
  `warn_from_frame` None for Ok/simulated.
- IPC isolation: hysteresis still reaches release while `FailingPublisher`
  errors; capacity-0 is rejected by validation; disconnected/slow
  `try_enqueue` cannot change `BrakeIntent::Apply`.
- CLI/lockfile: create/drop lifecycle, active-PID rejection, and stale-lock
  reclaim. Unparseable/empty lock fail-closed and `--step-interval-ms 0`
  rejection are runtime/clap behavior without dedicated named tests.
- Software-only integration crate: binary `--help`/`-V`, `--force-software-only`
  process start, acquire→evaluate→publish pipeline, no GPU.

### Gaps explicitly deferred

- `cpu::init_telemetry` / `cpu::run_metrics_collector`: bind a network port and run an indefinite loop; not safely unit-testable without an integration harness.
- `NvmlActuator` live `nvidia-smi` mutation: requires a real NVML-capable GPU; fail-closed-without-GPU is tested. Policy is covered by `SafetyMachine` + `FakeActuator`.
- Typed `corpus-ipc` round-trips and UDP emission are covered by the published
  `IpcMessage` tests (`stimulus_batch_round_trips_through_published_corpus_ipc_types`,
  `software_only_emits_typed_corpus_ipc_frame_without_gpu`). `AbsentPublisher`
  remains the disabled or unavailable fallback.
- The `main` async supervisor loop body (post-brake re-read, every-10-ticks cadence) is still exercised by the software-only binary smoke plus `evaluate_then_try_publish` / orderly-shutdown tests, not a full in-process tick harness.

## Coverage update (RM-342)

### Gaps closed (at the time, since partly superseded by RM-1143's UDP/SNN removal above)

- Software-only GPU telemetry fallback (`HardwareBridge::read_telemetry_force(true)`, `TelemetryFrame::to_sensory_mapping`, `is_gpu_healthy` no-panic).
- Single-instance lockfile lifecycle (`try_acquire_lock`, active PID rejection, stale lock reclamation, create/remove behavior).

### Gaps explicitly deferred

- `cpu::init_telemetry` / `cpu::run_metrics_collector`: bind a network port and run an indefinite loop; not safely unit-testable without an integration harness.
- `NvmlActuator` (`apply_emergency_brake` / `release_emergency_brake` / `detect_engaged_brake`, GH#46): the NVML/`nvidia-smi` backend requires a real NVML-capable GPU and can mutate board power limits, so its hardware path is intentionally not exercised in software-only CI. It fails closed with a typed `ActuatorError` when NVML is unavailable (regression-tested), and the brake/release *policy* it serves is unit-tested via `SafetyMachine` plus the in-memory `safety::FakeActuator`.
- The `main` async supervisor loop: covered by manual run instructions and end-to-end smoke tests rather than unit tests. Dashboard reading formatting is unit-tested. Hysteresis and IPC isolation are unit-tested via `SafetyMachine` / `evaluate_then_try_publish` without driving `main`, and the actuation boundary is exercised through the `SafetyActuator` trait with `FakeActuator`.

## Linting

```bash
cargo clippy --all-targets -- -D warnings   # lint (CI uses --all-features too)
cargo fmt --check                           # format check
cargo fmt                                   # auto-format
```

## Building

```bash
cargo build                  # debug build
cargo build --release        # release build
cargo build --all-features   # CI-equivalent build
```

## Running

```bash
cargo run -- --help                          # show CLI options
cargo run -- --force-software-only           # software-only mode (no GPU)
cargo run -- --metrics-ip 0.0.0.0            # custom Prometheus bind
cargo run -- --step-interval-ms 50           # faster tick interval
```

## CI Checks

The CI workflow (`.github/workflows/ci.yml`) reads `package.rust-version`
from `Cargo.toml` (currently 1.98.1) and installs **that** toolchain. It
asserts `rustc` and `rust-toolchain.toml` `channel` equal the declared MSRV.
Edition 2024 is independent of that pin (usable since rustc 1.85). None of the
jobs perform a real `cargo publish`. `cargo-semver-checks` is deferred until a
crates.io API baseline exists (GH#45).

**Build & Test**

1. `cargo fmt --check`
2. `cargo clippy --all-targets --all-features -- -D warnings`
3. `cargo build --all-features`
4. `cargo test --all-features`

**Exact MSRV** — `cargo test --locked --all-features` on the rustc version
exactly equal to `package.rust-version`.

**crates.io qualification** (software-only; no GPU, NVIDIA driver, or extra
`sudo` packages):

1. `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`
2. `cargo package --locked`
3. Packaged-manifest path-dependency check, then `cargo test --locked --all-features` from the unpacked `.crate` outside the repository
4. `cargo publish --dry-run --locked` (token-free; aborts before upload)

Bot reviewers: Codacy, CodeRabbit, Codex, Kilo, Devin, Gitar, Cursor Bugbot.
