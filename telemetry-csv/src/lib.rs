//! Canonical hardware-telemetry CSV interchange contract, as a standalone
//! crate ([RM-1905](https://linear.app/rpd-34/issue/RM-1905)).
//!
//! One-way copy of the reader/validator semantics from
//! `corinth-canal` `examples/support/telemetry_csv.rs`
//! ([RM-629](https://linear.app/rpd-34/issue/RM-629) /
//! [corinth-canal#160](https://github.com/rmems/corinth-canal/issues/160)).
//! There is **no** Cargo dependency in either direction; corinth keeps its
//! example-support copy as a reference consumer, and the copies may diverge.
//! Corinth's env-truth surface (`examples/support/config.rs`) is not copied
//! here.
//!
//! Schema is **frozen**. Do not add, remove, or rename columns.
//!
//! This is the interchange format corinth ingests. It is not the live NVML
//! sample contract (`vahtisiru::telemetry` in the relay crate): CSV fields
//! are required finite numbers (missing sensors are malformed rows, not
//! `None`), `gpu_power_w` is the CSV name for live `power_w`, and CPU Tctl /
//! package power are CSV columns the relay does not currently acquire.
//!
//! This crate has **no dependencies**: it requires no NVML, NVIDIA hardware,
//! privileged commands, or the relay supervisor. The companion
//! `vahtisiru-telemetry-csv` binary wraps the same semantics for CI and
//! non-Rust producers.
//!
//! Contract:
//! `docs/telemetry_csv.md` in <https://github.com/rmems/vahtisiru>.

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![doc(test(attr(deny(unused))))]

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

/// Frozen CSV header. Byte-for-byte match after trim; extra columns fail.
pub const HEADER: &str = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w";

/// Number of comma-separated fields in a valid data row.
pub const FIELD_COUNT: usize = 5;

/// One validated hardware-telemetry CSV row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelemetryCsvRow {
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// GPU die temperature (°C).
    pub gpu_temp_c: f32,
    /// GPU power usage (W).
    pub gpu_power_w: f32,
    /// CPU Tctl temperature (°C).
    pub cpu_tctl_c: f32,
    /// CPU package power usage (W).
    pub cpu_package_power_w: f32,
}

impl TelemetryCsvRow {
    /// Format this row as a single CSV data line (no trailing newline).
    #[must_use]
    pub fn to_csv_line(self) -> String {
        format!(
            "{},{},{},{},{}",
            self.timestamp_ms,
            self.gpu_temp_c,
            self.gpu_power_w,
            self.cpu_tctl_c,
            self.cpu_package_power_w
        )
    }
}

/// Outcome of parsing one physical line after the header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParseDataLine {
    /// Blank or whitespace-only. Not counted as skipped.
    Empty,
    /// Five fields, `u64` timestamp, four finite `f32` sensors.
    Ok(TelemetryCsvRow),
    /// Wrong field count, non-numeric, or non-finite. Counted as skipped.
    Malformed,
}

/// Header / I/O failure. Malformed *data* rows are skipped, not this error.
#[derive(Debug)]
pub enum TelemetryCsvError {
    /// The file could not be read.
    Io {
        /// Path to the CSV file.
        path: PathBuf,
        /// Underlying I/O error.
        source: io::Error,
    },
    /// No header line (empty input).
    Empty {
        /// Optional path to the CSV file.
        path: Option<PathBuf>,
    },
    /// Trimmed header is not exactly [`HEADER`].
    HeaderMismatch {
        /// Optional path to the CSV file.
        path: Option<PathBuf>,
        /// Actual header string encountered.
        actual: String,
    },
}

impl TelemetryCsvError {
    fn with_path(self, path: PathBuf) -> Self {
        match self {
            Self::Empty { .. } => Self::Empty { path: Some(path) },
            Self::HeaderMismatch { actual, .. } => Self::HeaderMismatch {
                path: Some(path),
                actual,
            },
            other => other,
        }
    }

    fn path_label(path: Option<&Path>) -> String {
        path.map(|p| format!(" '{}'", p.display()))
            .unwrap_or_default()
    }
}

impl fmt::Display for TelemetryCsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(
                f,
                "telemetry CSV '{}' could not be read: {source}",
                path.display()
            ),
            Self::Empty { path } => {
                write!(
                    f,
                    "telemetry CSV{} is empty",
                    Self::path_label(path.as_deref())
                )
            }
            Self::HeaderMismatch { path, actual } => write!(
                f,
                "telemetry CSV{} header mismatch: expected '{HEADER}', got '{actual}'",
                Self::path_label(path.as_deref())
            ),
        }
    }
}

impl std::error::Error for TelemetryCsvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Empty { .. } | Self::HeaderMismatch { .. } => None,
        }
    }
}

/// Rows accepted from a CSV plus how many malformed data lines were skipped.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadResult {
    /// Valid telemetry CSV rows parsed.
    pub rows: Vec<TelemetryCsvRow>,
    /// Number of malformed data rows skipped.
    pub skipped: usize,
}

impl LoadResult {
    /// True when the file is ingestible (canonical header and ≥1 valid row).
    #[must_use]
    pub fn has_usable_rows(&self) -> bool {
        !self.rows.is_empty()
    }
}

/// Fail fast on a missing or non-canonical header.
pub fn validate_header(header_line: Option<&str>) -> Result<(), TelemetryCsvError> {
    let header = header_line
        .ok_or(TelemetryCsvError::Empty { path: None })?
        .trim();
    if header != HEADER {
        return Err(TelemetryCsvError::HeaderMismatch {
            path: None,
            actual: header.to_string(),
        });
    }
    Ok(())
}

/// Parse one data line. The whole line is trimmed; individual fields are not.
#[must_use]
pub fn parse_data_line(raw_line: &str) -> ParseDataLine {
    let line = raw_line.trim();
    if line.is_empty() {
        return ParseDataLine::Empty;
    }
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() != FIELD_COUNT {
        return ParseDataLine::Malformed;
    }
    let Some(timestamp_ms) = fields[0].parse::<u64>().ok() else {
        return ParseDataLine::Malformed;
    };
    let Some(gpu_temp_c) = parse_finite_f32(fields[1]) else {
        return ParseDataLine::Malformed;
    };
    let Some(gpu_power_w) = parse_finite_f32(fields[2]) else {
        return ParseDataLine::Malformed;
    };
    let Some(cpu_tctl_c) = parse_finite_f32(fields[3]) else {
        return ParseDataLine::Malformed;
    };
    let Some(cpu_package_power_w) = parse_finite_f32(fields[4]) else {
        return ParseDataLine::Malformed;
    };
    ParseDataLine::Ok(TelemetryCsvRow {
        timestamp_ms,
        gpu_temp_c,
        gpu_power_w,
        cpu_tctl_c,
        cpu_package_power_w,
    })
}

fn parse_finite_f32(value: &str) -> Option<f32> {
    let parsed = value.parse::<f32>().ok()?;
    parsed.is_finite().then_some(parsed)
}

fn collect_rows<'a>(lines: impl Iterator<Item = &'a str>) -> LoadResult {
    let mut rows = Vec::new();
    let mut skipped = 0usize;
    for raw_line in lines {
        match parse_data_line(raw_line) {
            ParseDataLine::Empty => {}
            ParseDataLine::Ok(row) => rows.push(row),
            ParseDataLine::Malformed => skipped += 1,
        }
    }
    LoadResult { rows, skipped }
}

/// Parse an in-memory CSV: fail on header mismatch; skip malformed data rows.
pub fn parse_csv(contents: &str) -> Result<LoadResult, TelemetryCsvError> {
    let mut lines = contents.lines();
    validate_header(lines.next())?;
    Ok(collect_rows(lines))
}

/// Read a CSV from disk and apply [`parse_csv`].
pub fn load_csv(path: &Path) -> Result<LoadResult, TelemetryCsvError> {
    let contents = std::fs::read_to_string(path).map_err(|source| TelemetryCsvError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_csv(&contents).map_err(|err| err.with_path(path.to_path_buf()))
}

/// Replay wrap-around: `rows[tick % len]`, with `timestamp_ms` rewritten to
/// `tick + 1` so a 1-to-1 join against tick-indexed consumers stays stable
/// regardless of the CSV's absolute timestamps.
///
/// Empty `rows` returns [`None`]. Synthetic fallback is *not* part of this
/// contract (that lives in corinth's example-support env-truth surface).
#[must_use]
pub fn row_for_tick(tick: usize, rows: &[TelemetryCsvRow]) -> Option<TelemetryCsvRow> {
    if rows.is_empty() {
        return None;
    }
    let mut row = rows[tick % rows.len()];
    row.timestamp_ms = u64::try_from(tick).unwrap_or(u64::MAX).saturating_add(1);
    Some(row)
}

/// Serialize rows with the canonical header. Producers can round-trip this
/// through [`parse_csv`] before corinth ingests the file.
#[must_use]
pub fn format_csv(rows: &[TelemetryCsvRow]) -> String {
    let mut out = String::from(HEADER);
    out.push('\n');
    for row in rows {
        out.push_str(&row.to_csv_line());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scratch_dir() -> PathBuf {
        if let Some(dir) = std::env::var_os("CARGO_TARGET_TMPDIR") {
            return PathBuf::from(dir);
        }
        let dir = PathBuf::from("target").join("tmp-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_temp_csv(name: &str, contents: &str) -> PathBuf {
        let path = test_scratch_dir().join(format!(
            "vahtisiru_telemetry_csv_{}_{}.csv",
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn header_is_the_frozen_five_column_string() {
        assert_eq!(
            HEADER,
            "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w"
        );
        assert_eq!(HEADER.split(',').count(), FIELD_COUNT);
    }

    #[test]
    fn load_csv_accepts_canonical_header_and_parses_rows() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                   1000,60.5,250.0,70.0,120.0\n\
                   2000,61.0,252.5,70.5,121.5\n";
        let path = write_temp_csv("canonical", csv);
        let loaded = load_csv(&path).unwrap();
        assert_eq!(loaded.rows.len(), 2);
        assert_eq!(loaded.skipped, 0);
        assert_eq!(loaded.rows[0].timestamp_ms, 1000);
        assert!((loaded.rows[0].gpu_temp_c - 60.5).abs() < 1e-6);
        assert_eq!(loaded.rows[1].timestamp_ms, 2000);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_csv_rejects_bad_header() {
        let csv = "t,gpu,gpuw,cpu,cpuw\n1000,60,250,70,120\n";
        let path = write_temp_csv("bad_header", csv);
        let err = load_csv(&path).unwrap_err();
        assert!(err.to_string().contains("header mismatch"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_csv_rejects_session_label_extension_as_header_mismatch() {
        // gaming-telemetry currently appends session_label; that is not this
        // frozen schema. Exact-match header validation must fail closed.
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w,session_label\n\
                   1000,60.5,250.0,70.0,120.0,kcd2\n";
        let err = parse_csv(csv).unwrap_err();
        assert!(err.to_string().contains("header mismatch"));
    }

    #[test]
    fn load_csv_skips_malformed_short_rows() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                   1000,60.5,250.0,70.0,120.0\n\
                   malformed,row\n\
                   2000,NaN,250.0,70.0,120.0\n\
                   3000,61.0,252.5,70.5,121.5\n";
        let path = write_temp_csv("skip_bad", csv);
        let loaded = load_csv(&path).unwrap();
        assert_eq!(
            loaded.rows.len(),
            2,
            "expected only the two fully-valid rows"
        );
        assert_eq!(loaded.skipped, 2);
        assert_eq!(loaded.rows[0].timestamp_ms, 1000);
        assert_eq!(loaded.rows[1].timestamp_ms, 3000);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_csv_skips_non_numeric_fields() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                   1000,abc,250.0,70.0,120.0\n\
                   not_a_ts,60.5,250.0,70.0,120.0\n\
                   2000,61.0,252.5,70.5,121.5\n";
        let loaded = parse_csv(csv).unwrap();
        assert_eq!(loaded.rows.len(), 1);
        assert_eq!(loaded.skipped, 2);
        assert_eq!(loaded.rows[0].timestamp_ms, 2000);
    }

    #[test]
    fn load_csv_skips_non_finite_and_extra_columns() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                   1000,inf,250.0,70.0,120.0\n\
                   1100,-inf,250.0,70.0,120.0\n\
                   1200,60.5,250.0,70.0,120.0,extra\n\
                   1300,60.5,250.0,70.0,120.0\n";
        let loaded = parse_csv(csv).unwrap();
        assert_eq!(loaded.rows.len(), 1);
        assert_eq!(loaded.skipped, 3);
        assert_eq!(loaded.rows[0].timestamp_ms, 1300);
    }

    #[test]
    fn empty_lines_are_not_counted_as_skipped() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                   \n\
                   1000,60.5,250.0,70.0,120.0\n\
                   \n";
        let loaded = parse_csv(csv).unwrap();
        assert_eq!(loaded.rows.len(), 1);
        assert_eq!(loaded.skipped, 0);
    }

    #[test]
    fn empty_file_is_an_error() {
        let err = parse_csv("").unwrap_err();
        assert!(err.to_string().contains("is empty"));
    }

    #[test]
    fn header_only_is_valid_but_has_no_usable_rows() {
        let loaded = parse_csv(HEADER).unwrap();
        assert!(loaded.rows.is_empty());
        assert!(!loaded.has_usable_rows());
        assert_eq!(loaded.skipped, 0);
    }

    #[test]
    fn missing_file_is_io_error() {
        let path = test_scratch_dir().join("vahtisiru_telemetry_csv_does_not_exist.csv");
        let err = load_csv(&path).unwrap_err();
        assert!(err.to_string().contains("could not be read"));
    }

    #[test]
    fn row_for_tick_wraps_around_and_rewrites_timestamp() {
        let rows = vec![
            TelemetryCsvRow {
                timestamp_ms: 111,
                gpu_temp_c: 10.0,
                gpu_power_w: 100.0,
                cpu_tctl_c: 20.0,
                cpu_package_power_w: 200.0,
            },
            TelemetryCsvRow {
                timestamp_ms: 222,
                gpu_temp_c: 30.0,
                gpu_power_w: 300.0,
                cpu_tctl_c: 40.0,
                cpu_package_power_w: 400.0,
            },
        ];
        let snap0 = row_for_tick(0, &rows).unwrap();
        let snap3 = row_for_tick(3, &rows).unwrap();
        assert!((snap0.gpu_temp_c - 10.0).abs() < 1e-6);
        assert!((snap3.gpu_temp_c - 30.0).abs() < 1e-6);
        assert_eq!(snap0.timestamp_ms, 1);
        assert_eq!(snap3.timestamp_ms, 4);
    }

    #[test]
    fn row_for_tick_empty_is_none() {
        assert!(row_for_tick(0, &[]).is_none());
    }

    #[test]
    fn format_csv_round_trips_through_parse() {
        let rows = [TelemetryCsvRow {
            timestamp_ms: 1000,
            gpu_temp_c: 60.5,
            gpu_power_w: 250.0,
            cpu_tctl_c: 70.0,
            cpu_package_power_w: 120.0,
        }];
        let loaded = parse_csv(&format_csv(&rows)).unwrap();
        assert_eq!(loaded.skipped, 0);
        assert_eq!(loaded.rows[0].timestamp_ms, 1000);
        assert!((loaded.rows[0].gpu_temp_c - 60.5).abs() < 1e-6);
        assert!((loaded.rows[0].gpu_power_w - 250.0).abs() < 1e-6);
    }

    #[test]
    fn trimmed_header_whitespace_is_accepted() {
        let csv = "  timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w  \n\
                    1000,60.5,250.0,70.0,120.0\n";
        let loaded = parse_csv(csv).unwrap();
        assert_eq!(loaded.rows.len(), 1);
    }
}
