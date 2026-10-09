//! Shared conformance fixtures for the frozen telemetry CSV contract.
//!
//! Every fixture under `tests/fixtures/telemetry_csv/` is exercised through
//! the public `vahtisiru::telemetry_csv` API only — same surface downstream
//! producers validate against. The fixture README documents which semantics
//! are normative; this test pins them in bytes. See `docs/telemetry_csv.md`
//! and RM-629.

use std::path::{Path, PathBuf};

use vahtisiru::telemetry_csv::{
    HEADER, LoadResult, TelemetryCsvError, TelemetryCsvRow, format_csv, load_csv, parse_csv,
    row_for_tick,
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/telemetry_csv");

fn fixture_path(name: &str) -> PathBuf {
    Path::new(FIXTURES).join(name)
}

fn fixture_str(name: &str) -> &'static str {
    match name {
        "canonical.csv" => include_str!("fixtures/telemetry_csv/canonical.csv"),
        "canonical_crlf.csv" => include_str!("fixtures/telemetry_csv/canonical_crlf.csv"),
        "empty.csv" => include_str!("fixtures/telemetry_csv/empty.csv"),
        "header_extra_column.csv" => {
            include_str!("fixtures/telemetry_csv/header_extra_column.csv")
        }
        "header_mismatch.csv" => include_str!("fixtures/telemetry_csv/header_mismatch.csv"),
        "header_only.csv" => include_str!("fixtures/telemetry_csv/header_only.csv"),
        "malformed_rows.csv" => include_str!("fixtures/telemetry_csv/malformed_rows.csv"),
        "no_trailing_newline.csv" => {
            include_str!("fixtures/telemetry_csv/no_trailing_newline.csv")
        }
        "whitespace_lines.csv" => include_str!("fixtures/telemetry_csv/whitespace_lines.csv"),
        other => panic!("no embedded fixture named {other}"),
    }
}

fn canonical_rows() -> [TelemetryCsvRow; 2] {
    [
        TelemetryCsvRow {
            timestamp_ms: 1000,
            gpu_temp_c: 60.5,
            gpu_power_w: 250.0,
            cpu_tctl_c: 70.0,
            cpu_package_power_w: 120.0,
        },
        TelemetryCsvRow {
            timestamp_ms: 2000,
            gpu_temp_c: 61.0,
            gpu_power_w: 252.5,
            cpu_tctl_c: 70.5,
            cpu_package_power_w: 121.5,
        },
    ]
}

fn assert_rows_eq(actual: &[TelemetryCsvRow], expected: &[TelemetryCsvRow]) {
    assert_eq!(actual.len(), expected.len(), "accepted row count");
    for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.timestamp_ms, want.timestamp_ms,
            "row {idx} timestamp_ms"
        );
        assert!(
            (got.gpu_temp_c - want.gpu_temp_c).abs() < 1e-6,
            "row {idx} gpu_temp_c"
        );
        assert!(
            (got.gpu_power_w - want.gpu_power_w).abs() < 1e-6,
            "row {idx} gpu_power_w"
        );
        assert!(
            (got.cpu_tctl_c - want.cpu_tctl_c).abs() < 1e-6,
            "row {idx} cpu_tctl_c"
        );
        assert!(
            (got.cpu_package_power_w - want.cpu_package_power_w).abs() < 1e-6,
            "row {idx} cpu_package_power_w"
        );
    }
}

#[test]
fn canonical_fixture_loads_through_public_api() {
    let loaded: LoadResult = parse_csv(fixture_str("canonical.csv")).unwrap();
    assert!(loaded.has_usable_rows());
    assert_eq!(loaded.skipped, 0);
    assert_rows_eq(&loaded.rows, &canonical_rows());
}

#[test]
fn canonical_fixture_loads_from_disk() {
    let loaded = load_csv(&fixture_path("canonical.csv")).unwrap();
    assert!(loaded.has_usable_rows());
    assert_eq!(loaded.skipped, 0);
    assert_rows_eq(&loaded.rows, &canonical_rows());
}

#[test]
fn crlf_line_endings_are_accepted() {
    let loaded = parse_csv(fixture_str("canonical_crlf.csv")).unwrap();
    assert_eq!(loaded.skipped, 0);
    assert_rows_eq(&loaded.rows, &canonical_rows());
}

#[test]
fn missing_trailing_newline_is_accepted() {
    let contents = fixture_str("no_trailing_newline.csv");
    assert!(!contents.ends_with('\n'));
    let loaded = parse_csv(contents).unwrap();
    assert_eq!(loaded.skipped, 0);
    assert_rows_eq(&loaded.rows, &canonical_rows());
}

#[test]
fn header_mismatch_fixture_fails_with_header_mismatch() {
    let err = parse_csv(fixture_str("header_mismatch.csv")).unwrap_err();
    assert!(
        matches!(err, TelemetryCsvError::HeaderMismatch { .. }),
        "expected HeaderMismatch, got {err:?}"
    );
}

#[test]
fn extra_session_label_column_is_header_mismatch_not_superset() {
    // gaming-telemetry appends session_label today; the frozen schema fails
    // closed instead of silently widening (gaming-telemetry#50).
    let err = parse_csv(fixture_str("header_extra_column.csv")).unwrap_err();
    match err {
        TelemetryCsvError::HeaderMismatch { actual, .. } => {
            assert!(actual.contains("session_label"));
        }
        other => panic!("expected HeaderMismatch, got {other:?}"),
    }
}

#[test]
fn empty_fixture_is_an_error() {
    let err = parse_csv(fixture_str("empty.csv")).unwrap_err();
    assert!(
        matches!(err, TelemetryCsvError::Empty { .. }),
        "expected Empty, got {err:?}"
    );
}

#[test]
fn header_only_fixture_has_no_usable_rows() {
    let loaded = parse_csv(fixture_str("header_only.csv")).unwrap();
    assert!(loaded.rows.is_empty());
    assert_eq!(loaded.skipped, 0);
    assert!(!loaded.has_usable_rows());
}

#[test]
fn malformed_rows_fixture_skips_each_bad_class() {
    let loaded = parse_csv(fixture_str("malformed_rows.csv")).unwrap();
    // Skipped classes: short row, NaN, inf, -inf, non-u64 timestamp,
    // extra column, non-numeric field.
    assert_eq!(loaded.skipped, 7);
    let accepted: Vec<u64> = loaded.rows.iter().map(|r| r.timestamp_ms).collect();
    assert_eq!(accepted, vec![1000, 4000, 7000]);
}

#[test]
fn whitespace_only_lines_are_ignored_not_skipped() {
    let loaded = parse_csv(fixture_str("whitespace_lines.csv")).unwrap();
    assert_eq!(loaded.skipped, 0);
    assert_rows_eq(&loaded.rows, &canonical_rows());
}

#[test]
fn row_for_tick_wraps_and_rewrites_timestamp_from_fixture_rows() {
    let loaded = parse_csv(fixture_str("canonical.csv")).unwrap();
    let snap0 = row_for_tick(0, &loaded.rows).unwrap();
    let snap3 = row_for_tick(3, &loaded.rows).unwrap();
    assert!((snap0.gpu_temp_c - 60.5).abs() < 1e-6);
    assert!((snap3.gpu_temp_c - 61.0).abs() < 1e-6);
    assert_eq!(snap0.timestamp_ms, 1);
    assert_eq!(snap3.timestamp_ms, 4);
}

#[test]
fn row_for_tick_on_header_only_fixture_is_none() {
    let loaded = parse_csv(fixture_str("header_only.csv")).unwrap();
    assert!(row_for_tick(0, &loaded.rows).is_none());
}

#[test]
fn format_csv_round_trips_into_canonical_shape() {
    let rendered = format_csv(&canonical_rows());
    assert!(rendered.starts_with(HEADER));
    assert!(rendered.ends_with('\n'));
    let loaded = parse_csv(&rendered).unwrap();
    assert_eq!(loaded.skipped, 0);
    assert_rows_eq(&loaded.rows, &canonical_rows());
}
