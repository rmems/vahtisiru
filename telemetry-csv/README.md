# vahtisiru-telemetry-csv

Standalone, software-only validator for the frozen hardware-telemetry CSV
contract that [corinth-canal](https://github.com/rmems/corinth-canal) ingests.
No NVML, NVIDIA hardware, privileged commands, or the `vahtisiru` relay
supervisor required — the crate has **zero dependencies**.

Contract documentation:
[`docs/telemetry_csv.md`](https://github.com/rmems/vahtisiru/blob/main/docs/telemetry_csv.md)
(RM-629 / RM-1905).

## Frozen header

```text
timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w
```

After trimming, the header line must match **exactly**. Extra columns are a
header mismatch, not a compatible superset.

## CLI

```console
$ vahtisiru-telemetry-csv samples.csv
samples.csv: ok — 2 valid row(s), 0 malformed row(s) skipped

$ vahtisiru-telemetry-csv --json broken.csv; echo $?
{"input":"broken.csv","status":"error","rows":0,"skipped":0,"error":"telemetry CSV 'broken.csv' header mismatch: expected '...', got '...'"}
1
```

Read from stdin with `-`:

```console
$ cat samples.csv | vahtisiru-telemetry-csv -
```

Exit status (stable for CI):

| Code | Meaning |
| --- | --- |
| `0` | Contract-valid input with ≥1 usable row |
| `1` | Contract failure: empty input, header mismatch, or no usable rows |
| `2` | I/O or usage error: unreadable file, bad arguments |

`--json` always prints a single-line object with `input`, `status`,
`rows`, `skipped`, and `error` fields.

## Library

```rust
use vahtisiru_telemetry_csv::{load_csv, parse_csv};

let result = load_csv(std::path::Path::new("samples.csv"))?;
assert!(result.has_usable_rows());
println!("{} rows, {} skipped", result.rows.len(), result.skipped);
# Ok::<(), vahtisiru_telemetry_csv::TelemetryCsvError>(())
```

Semantics: whitespace-only lines are ignored; malformed data rows (wrong
field count, non-numeric, `NaN`/±`inf`) are skipped and counted in
`LoadResult.skipped`; `row_for_tick` replays rows with `timestamp_ms`
rewritten to `tick + 1`.

## License

MIT OR Apache-2.0
