//! Drift guard between this crate and the copy kept inside the `vahtisiru`
//! relay crate (`src/telemetry_csv.rs`). The repo's convention is deliberate
//! one-way copies (corinth ↔ vahtisiru); this test fails if the two copies'
//! observable semantics diverge.
//!
//! Runs in-repo only: `tests/` is not packaged into the .crate, so the
//! `#[path]` reference to the sibling crate's source never reaches crates.io
//! consumers.

#[path = "../../src/telemetry_csv.rs"]
mod relay_copy;

use vahtisiru_telemetry_csv as contract;

const HEADER: &str = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w";

const CASES: &[&str] = &[
    "",
    "   \n\n",
    "t,gpu,gpuw,cpu,cpuw\n1000,60,250,70,120\n",
    // Extra header column (gaming-telemetry session_label extension).
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,session_label\n1000,60.5,250.0,70.0,120.0,kcd2\n",
    // Header only, with and without trailing newline.
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w",
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n",
    // Header with surrounding whitespace (trimmed, accepted).
    "  timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w  \n1000,60.5,250.0,70.0,120.0\n",
    // Canonical rows + CRLF.
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\r\n1000,60.5,250.0,70.0,120.0\r\n2000,61.0,252.5,70.5,121.5\r\n",
    // Malformed rows: short, extra column, non-numeric, non-finite, bad ts.
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
     1000,60.5,250.0,70.0,120.0\n\
     malformed,row\n\
     1200,60.5,250.0,70.0,120.0,extra\n\
     1300,abc,250.0,70.0,120.0\n\
     1400,inf,250.0,70.0,120.0\n\
     1500,-inf,250.0,70.0,120.0\n\
     1600,NaN,250.0,70.0,120.0\n\
     not_a_ts,61.0,252.5,70.5,121.5\n\
     2000,61.0,252.5,70.5,121.5\n",
    // Blank and whitespace-only lines ignored, not counted skipped.
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\n   \n1000,60.5,250.0,70.0,120.0\n\n",
    // Negative timestamp, float timestamp, huge u64.
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
     -1,60.5,250.0,70.0,120.0\n\
     1.5,60.5,250.0,70.0,120.0\n\
     18446744073709551615,60.5,250.0,70.0,120.0\n\
     18446744073709551616,60.5,250.0,70.0,120.0\n",
];

fn error_tag(err: &contract::TelemetryCsvError) -> &'static str {
    match err {
        contract::TelemetryCsvError::Io { .. } => "io",
        contract::TelemetryCsvError::Empty { .. } => "empty",
        contract::TelemetryCsvError::HeaderMismatch { .. } => "header_mismatch",
    }
}

fn relay_error_tag(err: &relay_copy::TelemetryCsvError) -> &'static str {
    match err {
        relay_copy::TelemetryCsvError::Io { .. } => "io",
        relay_copy::TelemetryCsvError::Empty { .. } => "empty",
        relay_copy::TelemetryCsvError::HeaderMismatch { .. } => "header_mismatch",
    }
}

fn row_eq(a: &contract::TelemetryCsvRow, b: &relay_copy::TelemetryCsvRow) -> bool {
    (
        a.timestamp_ms,
        a.gpu_temp_c,
        a.gpu_power_w,
        a.cpu_tctl_c,
        a.cpu_package_power_w,
    ) == (
        b.timestamp_ms,
        b.gpu_temp_c,
        b.gpu_power_w,
        b.cpu_tctl_c,
        b.cpu_package_power_w,
    )
}

fn rows_match(a: &[contract::TelemetryCsvRow], b: &[relay_copy::TelemetryCsvRow]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(a, b)| row_eq(a, b))
}

#[test]
fn header_constant_matches() {
    assert_eq!(contract::HEADER, relay_copy::HEADER);
    assert_eq!(contract::FIELD_COUNT, relay_copy::FIELD_COUNT);
    assert_eq!(contract::HEADER, HEADER);
}

#[test]
fn parse_csv_outcomes_match_relay_copy() {
    for (i, case) in CASES.iter().enumerate() {
        let ours = contract::parse_csv(case);
        let theirs = relay_copy::parse_csv(case);
        match (ours, theirs) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a.skipped, b.skipped, "case {i} skipped diverged");
                assert!(rows_match(&a.rows, &b.rows), "case {i} rows diverged");
            }
            (Err(a), Err(b)) => assert_eq!(
                error_tag(&a),
                relay_error_tag(&b),
                "case {i} error class diverged"
            ),
            (a, b) => panic!(
                "case {i}: ok/error divergence — ours {}, theirs {}",
                a.is_ok(),
                b.is_ok()
            ),
        }
    }
}

#[test]
fn row_values_match_relay_copy() {
    let csv = format!("{HEADER}\n1000,60.5,250.0,70.0,120.0\n2000,61.0,252.5,70.5,121.5\n");
    let ours = contract::parse_csv(&csv).unwrap();
    let theirs = relay_copy::parse_csv(&csv).unwrap();
    for (a, b) in ours.rows.iter().zip(theirs.rows.iter()) {
        assert_eq!(a.timestamp_ms, b.timestamp_ms);
        assert_eq!(a.gpu_temp_c, b.gpu_temp_c);
        assert_eq!(a.gpu_power_w, b.gpu_power_w);
        assert_eq!(a.cpu_tctl_c, b.cpu_tctl_c);
        assert_eq!(a.cpu_package_power_w, b.cpu_package_power_w);
    }
}

#[test]
fn row_for_tick_matches_relay_copy() {
    let csv = format!("{HEADER}\n111,10.0,100.0,20.0,200.0\n222,30.0,300.0,40.0,400.0\n");
    let ours = contract::parse_csv(&csv).unwrap();
    let theirs = relay_copy::parse_csv(&csv).unwrap();
    assert!(contract::row_for_tick(0, &[]).is_none());
    assert!(relay_copy::row_for_tick(0, &[]).is_none());
    for tick in 0..8 {
        let a = contract::row_for_tick(tick, &ours.rows).unwrap();
        let b = relay_copy::row_for_tick(tick, &theirs.rows).unwrap();
        assert_eq!(a.timestamp_ms, b.timestamp_ms, "tick {tick}");
        assert_eq!(a.gpu_temp_c, b.gpu_temp_c, "tick {tick}");
        assert_eq!(a.timestamp_ms, tick as u64 + 1, "tick {tick} rewrite");
    }
}

#[test]
fn format_csv_round_trips_through_both_parsers() {
    let rows = [contract::TelemetryCsvRow {
        timestamp_ms: 1000,
        gpu_temp_c: 60.5,
        gpu_power_w: 250.0,
        cpu_tctl_c: 70.0,
        cpu_package_power_w: 120.0,
    }];
    let rendered = contract::format_csv(&rows);
    assert!(contract::parse_csv(&rendered).unwrap().has_usable_rows());
    assert!(relay_copy::parse_csv(&rendered).unwrap().has_usable_rows());
}
