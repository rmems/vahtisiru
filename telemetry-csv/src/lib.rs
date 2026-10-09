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
