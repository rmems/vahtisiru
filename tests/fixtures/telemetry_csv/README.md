# Telemetry CSV conformance fixtures

Shared, vendored-safe fixture set for the **frozen** hardware-telemetry CSV
interchange contract hosted by this crate:

- Contract doc: [`docs/telemetry_csv.md`](../../../docs/telemetry_csv.md)
- Implementation: [`src/telemetry_csv.rs`](../../../src/telemetry_csv.rs)
  (`vahtisiru::telemetry_csv`)
- Origin: [RM-629](https://linear.app/rpd-34/issue/RM-629) /
  [corinth-canal#160](https://github.com/rmems/corinth-canal/issues/160)
- Consumer test: [`tests/telemetry_csv_conformance.rs`](../../telemetry_csv_conformance.rs)

These files exist so validator changes here and independent consumer
smoke-tests downstream can share one byte-level truth. Keep them small;
downstream producers may vendor this directory:

- [rmems/spikenaut-telemetry-etl#27](https://github.com/rmems/spikenaut-telemetry-etl/issues/27)
- [rmems/gaming-telemetry#50](https://github.com/rmems/gaming-telemetry/issues/50)
- [rmems/Theseus-Quarry#24](https://github.com/rmems/Theseus-Quarry/issues/24)

## Normative semantics (what a conforming reader MUST do)

1. **Header (frozen).** The first line, after trimming, must equal exactly:
   `timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w`.
   Extra columns (e.g. `session_label`) are a *header mismatch*, not a
   compatible superset. Do not add, remove, or rename columns.
2. **Empty input** is an error (`TelemetryCsvError::Empty`), not zero rows.
3. **Blank lines.** Whitespace-only data lines are ignored — not counted as
   skipped.
4. **Malformed data rows** are skipped and counted (`LoadResult.skipped`);
   they never abort the load. Malformed means: comma-split length ≠ 5,
   non-`u64` `timestamp_ms`, non-numeric fields, or non-finite floats
   (`NaN`, `inf`, `-inf`).
5. **Fields are required.** A missing sensor is a malformed row, not `None`
   (unlike the live `TelemetrySample` contract in `src/telemetry.rs`).
6. **Replay.** `row_for_tick(tick, rows)` returns `rows[tick % len]` with
   `timestamp_ms` rewritten to `tick + 1`; an empty row list returns `None`.
7. Individual CSV fields are **not** trimmed; only whole lines are.

## Fixture table

| File | Expected result |
| --- | --- |
| `canonical.csv` | `Ok`: 2 rows, `skipped = 0`, `has_usable_rows()` |
| `canonical_crlf.csv` | Same as `canonical.csv` — CRLF line endings accepted |
| `no_trailing_newline.csv` | Same as `canonical.csv` — missing final newline is fine |
| `header_extra_column.csv` | `Err(HeaderMismatch)` (the `session_label` producer case) |
| `header_mismatch.csv` | `Err(HeaderMismatch)` |
| `empty.csv` | `Err(Empty)` |
| `header_only.csv` | `Ok`: 0 rows, `skipped = 0`, `!has_usable_rows()` |
| `malformed_rows.csv` | `Ok`: 3 rows (`timestamp_ms` 1000, 4000, 7000), `skipped = 7` — short row, `NaN`, `inf`, `-inf`, non-`u64` timestamp, extra column, non-numeric field |
| `whitespace_lines.csv` | `Ok`: 2 rows, `skipped = 0` — whitespace-only lines are ignored |
