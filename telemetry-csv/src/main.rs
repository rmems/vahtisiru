//! Standalone validator CLI for the frozen hardware-telemetry CSV contract.
//!
//! Software-only: no NVML, GPU, privileged command, or relay supervisor.
//!
//! Usage: `vahtisiru-telemetry-csv [--json] <PATH>` where `PATH` is a CSV file
//! or `-` for stdin.
//!
//! Exit status (stable, for CI):
//! - `0`: contract-valid input with at least one usable row
//! - `1`: contract failure (empty input, header mismatch, or no usable rows)
//! - `2`: I/O or usage error (unreadable file, bad arguments)

use std::io::Read;
use std::process::ExitCode;

use vahtisiru_telemetry_csv::{HEADER, LoadResult, TelemetryCsvError, parse_csv};

const EXIT_CONTRACT: u8 = 1;
const EXIT_IO_OR_USAGE: u8 = 2;

const USAGE: &str = "\
Usage: vahtisiru-telemetry-csv [--json] <PATH>

Validate a hardware-telemetry CSV against the frozen contract header:
  timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w

Arguments:
  <PATH>    CSV file to validate, or '-' to read stdin.

Options:
  --json    Emit a single-line machine-readable result.
  -h, --help  Print this help.

Exit status: 0 = valid with usable rows; 1 = contract failure
(empty input, header mismatch, or no usable rows); 2 = I/O or usage error.

Malformed data rows are skipped (not fatal) and reported in the count.";

struct Args {
    json: bool,
    path: String,
}

fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut json = false;
    let mut path: Option<String> = None;
    for arg in argv {
        match arg.as_str() {
            "--json" => json = true,
            "-h" | "--help" => return Ok(None),
            _ if path.is_none() => path = Some(arg.clone()),
            _ => return Err(format!("unexpected argument '{arg}'")),
        }
    }
    let Some(path) = path else {
        return Err("missing PATH argument".to_string());
    };
    Ok(Some(Args { json, path }))
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

fn report_json(input: &str, result: &Result<LoadResult, ContractFailure>) {
    let (status, rows, skipped, error) = match result {
        Ok(load) => ("ok", load.rows.len(), load.skipped, String::new()),
        Err(failure) => ("error", 0, 0, failure.to_string()),
    };
    println!(
        "{{\"input\":\"{}\",\"status\":\"{status}\",\"rows\":{rows},\"skipped\":{skipped},\"error\":\"{}\"}}",
        json_escape(input),
        json_escape(&error)
    );
}

/// A contract-level failure: parses but is not publishable, or fails to parse.
enum ContractFailure {
    /// Header/I/O failure from the contract parser itself.
    Parse(TelemetryCsvError),
    /// Parsed cleanly but produced no usable rows.
    NoUsableRows,
}

impl std::fmt::Display for ContractFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(err) => write!(f, "{err}"),
            Self::NoUsableRows => write!(
                f,
                "no usable rows: contract requires at least one valid data row"
            ),
        }
    }
}

fn validate(contents: &str) -> Result<LoadResult, ContractFailure> {
    let load = parse_csv(contents).map_err(ContractFailure::Parse)?;
    if !load.has_usable_rows() {
        return Err(ContractFailure::NoUsableRows);
    }
    Ok(load)
}

fn read_input(path: &str) -> std::io::Result<String> {
    if path == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        return Ok(buf);
    }
    std::fs::read_to_string(path)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(Some(args)) => args,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(err) => {
            eprintln!("vahtisiru-telemetry-csv: {err}\n{USAGE}");
            return ExitCode::from(EXIT_IO_OR_USAGE);
        }
    };

    let input_label = if args.path == "-" {
        "<stdin>".to_string()
    } else {
        args.path.clone()
    };

    let contents = match read_input(&args.path) {
        Ok(contents) => contents,
        Err(err) => {
            if args.json {
                report_json(
                    &input_label,
                    &Err(ContractFailure::Parse(TelemetryCsvError::Io {
                        path: std::path::PathBuf::from(&args.path),
                        source: err,
                    })),
                );
            } else {
                eprintln!(
                    "vahtisiru-telemetry-csv: '{}' could not be read: {err}",
                    input_label
                );
            }
            return ExitCode::from(EXIT_IO_OR_USAGE);
        }
    };

    let result = validate(&contents);
    if args.json {
        report_json(&input_label, &result);
    } else {
        match &result {
            Ok(load) => println!(
                "{input_label}: ok — {} valid row(s), {} malformed row(s) skipped",
                load.rows.len(),
                load.skipped
            ),
            Err(failure) => {
                eprintln!("vahtisiru-telemetry-csv: {input_label}: {failure}");
                eprintln!("expected header: {HEADER}");
            }
        }
    }
    if result.is_ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(EXIT_CONTRACT)
    }
}
