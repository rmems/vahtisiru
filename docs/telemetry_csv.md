# Hardware telemetry CSV contract

Frozen interchange schema for hardware-telemetry CSVs that corinth-canal
ingests. Hosted here so **producers** can validate output before corinth sees
it ([RM-629](https://linear.app/rpd-34/issue/RM-629) /
[corinth-canal#160](https://github.com/rmems/corinth-canal/issues/160)).

This is a **one-way copy** of
`corinth-canal` `examples/support/telemetry_csv.rs` reader semantics.
There is no Cargo dependency in either direction. Corinth keeps its
example-support file as a reference consumer; the two copies may diverge.
Corinth's env-truth surface (`examples/support/config.rs`) is not copied.

The live NVML sample contract (`src/telemetry.rs`, [`telemetry.md`](telemetry.md))
is a different type: optional values, validity/provenance, and extra GPU
channels. Do not treat the two schemas as the same struct.

## Header (frozen)

```text
timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w
```

This string is [`telemetry_csv::HEADER`](../src/telemetry_csv.rs). After
trimming the header line, it must match **exactly**. Extra columns (for
example a trailing `session_label`) are a header mismatch, not a compatible
superset.

Do not add, remove, or rename columns.

## Columns

| Column | Parse | Notes |
| --- | --- | --- |
| `timestamp_ms` | `u64` | Unix-epoch milliseconds. Individual fields are not trimmed. |
| `gpu_temp_c` | finite `f32` | Die temperature, °C. |
| `gpu_power_w` | finite `f32` | Board power, watts. Live relay field is `power_w`. |
| `cpu_tctl_c` | finite `f32` | CPU Tctl, °C. Not currently acquired by this relay. |
| `cpu_package_power_w` | finite `f32` | CPU package power, watts. Not currently acquired by this relay. |

CSV fields are **required**. A missing sensor is a malformed row (skipped),
not a typed `None`. That is the opposite of the live [`TelemetrySample`]
contract, where dropout stays `None`.

`NaN`, `inf`, `-inf`, non-numeric text, a non-`u64` timestamp, and any
row whose comma-split length is not 5 are malformed.

## Reader / validator semantics

Implemented in [`src/telemetry_csv.rs`](../src/telemetry_csv.rs):

1. **Header.** Empty input fails. Trimmed header must equal `HEADER`.
2. **Blank lines.** Whitespace-only data lines are ignored (not skipped).
3. **Malformed data rows.** Short rows, extra columns, non-numeric fields,
   and non-finite floats are skipped and counted in `LoadResult.skipped`.
   They do not abort the load.
4. **Replay.** `row_for_tick(tick, rows)` uses `rows[tick % len]` and
   rewrites `timestamp_ms` to `tick + 1` so tick-indexed consumers join
   1-to-1 regardless of the CSV's absolute timestamps. An empty row list
   returns `None`. Synthetic fallback is **not** part of this contract.

Producers should call `parse_csv` / `load_csv` (or `format_csv` then
`parse_csv`) and require `LoadResult::has_usable_rows()` before publishing
a file for corinth ingest.

## Standalone validator (`vahtisiru-telemetry-csv`)

For producers and CI that cannot (or should not) link the full relay crate —
no NVML, NVIDIA hardware, privileged commands, or supervisor — the workspace
member [`telemetry-csv/`](../telemetry-csv/) packages the same semantics as a
zero-dependency crate, `vahtisiru-telemetry-csv`, plus a validator binary of
the same name ([RM-1905](https://linear.app/rpd-34/issue/RM-1905)).

```console
$ vahtisiru-telemetry-csv samples.csv
samples.csv: ok — 2 valid row(s), 0 malformed row(s) skipped

$ cat samples.csv | vahtisiru-telemetry-csv -        # stdin

$ vahtisiru-telemetry-csv --json broken.csv          # machine-readable
{"input":"broken.csv","status":"error","rows":0,"skipped":0,"error":"telemetry CSV 'broken.csv' header mismatch: expected '...', got '...'"}
```

Exit status is stable for CI use:

| Code | Meaning |
| --- | --- |
| `0` | Contract-valid input with at least one usable row |
| `1` | Contract failure: empty input, header mismatch, or no usable rows |
| `2` | I/O or usage error: unreadable file, bad arguments |

Malformed data rows stay permissive-skipped (not fatal) and are reported in
the `skipped` count — the validator does not tighten the reader contract.
This crate is the canonical standalone copy; `src/telemetry_csv.rs` in the
relay is kept in lockstep by `telemetry-csv/tests/parity.rs`.

## What this crate does not copy

- Corinth `TelemetrySource::{Synthetic, Csv}` and `synthetic_fallback`
- `csv_re4` source-label slugs
- Machine-local path / env resolution (`examples/support/config.rs`)

Those stay in corinth-canal.

## Producer follow-up

Point exporters at this document and `vahtisiru::telemetry_csv`:

- [rmems/gaming-telemetry#50](https://github.com/rmems/gaming-telemetry/issues/50) — CSV export currently appends `session_label`; that fails exact header match here
- [rmems/Theseus-Quarry#24](https://github.com/rmems/Theseus-Quarry/issues/24) — JSONL producer; any CSV projection for corinth must use this header
- [rmems/spikenaut-telemetry-etl#27](https://github.com/rmems/spikenaut-telemetry-etl/issues/27) — cleaning/validation between collectors and published datasets
