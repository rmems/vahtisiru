//! End-to-end CLI conformance tests for `vahtisiru-telemetry-csv`.
//!
//! Software-only: drives the compiled binary against temp files and stdin.
//! Covers the canonical and malformed cases shared with the conformance
//! fixture work in RM-1904; when those fixtures land in the repository,
//! point these cases at the fixture paths instead of inline data.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_vahtisiru-telemetry-csv");

const HEADER: &str = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w";

fn scratch_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CARGO_TARGET_TMPDIR") {
        return PathBuf::from(dir);
    }
    let dir = std::env::temp_dir().join("vahtisiru-telemetry-csv-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_csv(name: &str, contents: &str) -> PathBuf {
    let path = scratch_dir().join(format!(
        "telemetry_csv_cli_{}_{}.csv",
        name,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&path, contents).unwrap();
    path
}

fn run(args: &[&str], stdin: Option<&str>) -> (i32, String, String) {
    let mut cmd = Command::new(BIN);
    cmd.args(args);
    let mut child = if stdin.is_some() {
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    } else {
        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };
    if let Some(input) = stdin {
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
    }
    let out = child.wait_with_output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn canonical_file_exits_zero_and_reports_counts() {
    let path = write_csv(
        "canonical",
        &format!("{HEADER}\n1000,60.5,250.0,70.0,120.0\n2000,61.0,252.5,70.5,121.5\n"),
    );
    let (code, stdout, _) = run(&[path.to_str().unwrap()], None);
    assert_eq!(code, 0);
    assert!(stdout.contains("2 valid row(s)"));
    assert!(stdout.contains("0 malformed row(s)"));
    assert!(stdout.contains(path.to_str().unwrap()));
}

#[test]
fn crlf_and_no_trailing_newline_are_accepted() {
    let path = write_csv("crlf", &format!("{HEADER}\r\n1000,60.5,250.0,70.0,120.0"));
    let (code, _, _) = run(&[path.to_str().unwrap()], None);
    assert_eq!(code, 0);
}

#[test]
fn malformed_rows_are_skipped_and_reported() {
    // Short row, extra column, non-numeric, non-finite: skipped, not fatal.
    let path = write_csv(
        "malformed",
        &format!(
            "{HEADER}\n\
             1000,60.5,250.0,70.0,120.0\n\
             short,row\n\
             1200,60.5,250.0,70.0,120.0,extra\n\
             1300,abc,250.0,70.0,120.0\n\
             1400,inf,250.0,70.0,120.0\n\
             1500,-inf,250.0,70.0,120.0\n\
             1600,NaN,250.0,70.0,120.0\n\
             2000,61.0,252.5,70.5,121.5\n"
        ),
    );
    let (code, stdout, _) = run(&[path.to_str().unwrap()], None);
    assert_eq!(code, 0);
    assert!(stdout.contains("2 valid row(s)"));
    assert!(stdout.contains("6 malformed row(s) skipped"));
}

#[test]
fn blank_and_whitespace_lines_are_not_counted() {
    let path = write_csv(
        "blank",
        &format!("{HEADER}\n\n   \n1000,60.5,250.0,70.0,120.0\n\n"),
    );
    let (code, stdout, _) = run(&[path.to_str().unwrap()], None);
    assert_eq!(code, 0);
    assert!(stdout.contains("0 malformed row(s)"));
}

#[test]
fn exact_header_rejection_exits_one() {
    let path = write_csv("bad_header", "t,gpu,gpuw,cpu,cpuw\n1000,60,250,70,120\n");
    let (code, _, stderr) = run(&[path.to_str().unwrap()], None);
    assert_eq!(code, 1);
    assert!(stderr.contains("header mismatch"));
    assert!(stderr.contains(path.to_str().unwrap()));
}

#[test]
fn extra_header_column_is_header_mismatch() {
    let path = write_csv(
        "session_label",
        &format!("{HEADER},session_label\n1000,60.5,250.0,70.0,120.0,kcd2\n"),
    );
    let (code, _, stderr) = run(&[path.to_str().unwrap()], None);
    assert_eq!(code, 1);
    assert!(stderr.contains("header mismatch"));
}

#[test]
fn header_only_input_exits_one() {
    let path = write_csv("header_only", HEADER);
    let (code, _, stderr) = run(&[path.to_str().unwrap()], None);
    assert_eq!(code, 1);
    assert!(stderr.contains("no usable rows"));
}

#[test]
fn all_malformed_input_exits_one() {
    let path = write_csv(
        "all_bad",
        &format!("{HEADER}\nbad,row\nalso,bad,fields,here,extra,cols\n"),
    );
    let (code, _, _) = run(&[path.to_str().unwrap()], None);
    assert_eq!(code, 1);
}

#[test]
fn empty_input_exits_one() {
    let path = write_csv("empty", "");
    let (code, _, stderr) = run(&[path.to_str().unwrap()], None);
    assert_eq!(code, 1);
    assert!(stderr.contains("is empty"));
}

#[test]
fn missing_file_exits_two() {
    let path = scratch_dir().join("telemetry_csv_cli_does_not_exist.csv");
    let (code, _, stderr) = run(&[path.to_str().unwrap()], None);
    assert_eq!(code, 2);
    assert!(stderr.contains("could not be read"));
}

#[test]
fn stdin_dash_reads_and_validates() {
    let (code, stdout, _) = run(
        &["-"],
        Some(&format!("{HEADER}\n1000,60.5,250.0,70.0,120.0\n")),
    );
    assert_eq!(code, 0);
    assert!(stdout.contains("<stdin>"));
}

#[test]
fn json_success_reports_counts() {
    let path = write_csv(
        "json_ok",
        &format!("{HEADER}\n1000,60.5,250.0,70.0,120.0\nbad,row\n"),
    );
    let (code, stdout, _) = run(&["--json", path.to_str().unwrap()], None);
    assert_eq!(code, 0);
    assert!(stdout.contains("\"status\":\"ok\""));
    assert!(stdout.contains("\"rows\":1"));
    assert!(stdout.contains("\"skipped\":1"));
    assert!(stdout.contains("\"error\":\"\""));
}

#[test]
fn json_error_is_still_single_line() {
    let (code, stdout, _) = run(
        &["--json", "-"],
        Some("wrong,header\n1000,60.5,250.0,70.0,120.0\n"),
    );
    assert_eq!(code, 1);
    assert_eq!(stdout.trim().lines().count(), 1);
    assert!(stdout.contains("\"status\":\"error\""));
    assert!(stdout.contains("header mismatch"));
}

#[test]
fn json_all_malformed_reports_skipped_count() {
    let path = write_csv(
        "json_all_bad",
        &format!("{HEADER}\nbad,row\nalso,bad,fields,here,extra,cols\n"),
    );
    let (code, stdout, _) = run(&["--json", path.to_str().unwrap()], None);
    assert_eq!(code, 1);
    assert!(stdout.contains("\"skipped\":2"));
    assert!(stdout.contains("no usable rows"));
}

#[test]
fn json_usage_error_still_emits_result() {
    let (code, stdout, _) = run(&["--json"], None);
    assert_eq!(code, 2);
    assert_eq!(stdout.trim().lines().count(), 1);
    assert!(stdout.contains("\"status\":\"error\""));
    assert!(stdout.contains("missing PATH argument"));
}

#[test]
fn no_arguments_exits_two_with_usage() {
    let (code, _, stderr) = run(&[], None);
    assert_eq!(code, 2);
    assert!(stderr.contains("Usage:"));
}

#[test]
fn help_exits_zero() {
    let (code, stdout, _) = run(&["--help"], None);
    assert_eq!(code, 0);
    assert!(stdout.contains(HEADER));
}
