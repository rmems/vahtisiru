//! GH#42: the deterministic safety path is an independent failure domain.
//!
//! These tests need no GPU, NVML, `sudo`, network peer, or Brainstem. They
//! prove that:
//!
//! 1. A learned / shadow-mode component (modelled here by [`ShadowAdvisor`])
//!    has no input into [`SafetyMachine`]: whether it is absent, panics,
//!    returns NaN/garbage, or recommends the opposite action, the
//!    deterministic snapshot sequence is bit-for-bit identical.
//! 2. Thresholds, freshness, and hysteresis behave exactly at their
//!    boundaries, and missing/stale/invalid telemetry fails closed.
//! 3. Every deterministic trip emits one structured, closed-vocabulary
//!    [`SafetyReason`] event when its fail-closed episode opens.
//! 4. A panicking sensory publisher cannot unwind through the safety loop.

use std::panic::{AssertUnwindSafe, catch_unwind};

use vahtisiru::publish::{
    AbsentPublisher, PUBLISHER_PANICKED, PublishError, SensoryPublisher, evaluate_then_try_publish,
};
use vahtisiru::safety::{
    ActuatorOutcome, BrakeIntent, SafetyMachine, SafetyPolicy, SafetyPolicyConfig, SafetyReason,
    SafetySnapshot, SafetyState,
};
use vahtisiru::telemetry::{
    RawTelemetry, SensoryMapping, TelemetryFrame, TelemetrySource, assess, fixtures,
};

const TEMP_WARN: f32 = 75.0;
const TEMP_CRIT: f32 = 85.0;
const POWER_WARN: f32 = 300.0;
const POWER_CRIT: f32 = 350.0;

fn policy() -> SafetyPolicy {
    SafetyPolicyConfig {
        temp_warn_c: TEMP_WARN,
        temp_critical_c: TEMP_CRIT,
        power_warn_w: Some(POWER_WARN),
        power_critical_w: Some(POWER_CRIT),
        ..Default::default()
    }
    .resolve(None)
    .unwrap()
}

fn machine() -> SafetyMachine {
    SafetyMachine::with_policy(policy())
}

fn raw(temp_c: Option<f32>, power_w: Option<f32>) -> RawTelemetry {
    RawTelemetry {
        gpu_temp_c: temp_c,
        power_w,
        ..fixtures::healthy_real()
    }
}

fn frame(temp_c: f32, power_w: f32) -> TelemetryFrame {
    assess(&raw(Some(temp_c), Some(power_w)), fixtures::NOW)
}

fn aged(temp_c: f32, power_w: f32, age_ms: u64) -> TelemetryFrame {
    let mut r = raw(Some(temp_c), Some(power_w));
    r.source_unix_ms = Some(fixtures::NOW - age_ms);
    assess(&r, fixtures::NOW)
}

// ── Shadow-mode learned component model ─────────────────────────────

/// What a learned/SNN layer might propose. It is *advice only*.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Proposal {
    Hold,
    Release,
    Apply,
    /// A scalar "risk" score; may be NaN/inf garbage.
    Risk(f32),
}

/// A shadow-mode learned component. The deterministic harness may consult it
/// for logging/comparison, but nothing it returns is an input to safety.
trait ShadowAdvisor {
    fn propose(&mut self, frame: &TelemetryFrame) -> Proposal;
}

struct PanickingAdvisor;
impl ShadowAdvisor for PanickingAdvisor {
    fn propose(&mut self, _frame: &TelemetryFrame) -> Proposal {
        panic!("learned component crashed");
    }
}

struct GarbageAdvisor(u32);
impl ShadowAdvisor for GarbageAdvisor {
    fn propose(&mut self, _frame: &TelemetryFrame) -> Proposal {
        self.0 += 1;
        match self.0 % 4 {
            0 => Proposal::Risk(f32::NAN),
            1 => Proposal::Risk(f32::INFINITY),
            2 => Proposal::Risk(f32::NEG_INFINITY),
            _ => Proposal::Risk(-1.0e30),
        }
    }
}

/// Always recommends the opposite of what deterministic policy would do.
struct AdversarialAdvisor;
impl ShadowAdvisor for AdversarialAdvisor {
    fn propose(&mut self, frame: &TelemetryFrame) -> Proposal {
        let hot = frame.gpu_temp_c.value.is_some_and(|t| t > TEMP_CRIT)
            || frame.power_w.value.is_some_and(|p| p > POWER_CRIT);
        if hot {
            Proposal::Release
        } else {
            Proposal::Apply
        }
    }
}

struct HoldAdvisor;
impl ShadowAdvisor for HoldAdvisor {
    fn propose(&mut self, _frame: &TelemetryFrame) -> Proposal {
        Proposal::Hold
    }
}

/// Supervisor-shaped step: deterministic safety first, then (optionally) the
/// shadow advisor, inside its own panic boundary. The advisor's output is
/// recorded for comparison only.
fn step(
    machine: &mut SafetyMachine,
    frame: &TelemetryFrame,
    advisor: Option<&mut (dyn ShadowAdvisor + 'static)>,
    shadow_log: &mut Vec<Option<Proposal>>,
) -> SafetySnapshot {
    let snap = machine.evaluate(frame);
    if let Some(advisor) = advisor {
        let proposal = catch_unwind(AssertUnwindSafe(|| advisor.propose(frame))).ok();
        shadow_log.push(proposal);
    }
    snap
}

/// Scenario that exercises warn, critical trip, apply, recovery, release,
/// missing/stale/invalid fail-closed and re-trip.
fn scenario() -> Vec<TelemetryFrame> {
    vec![
        frame(60.0, 200.0),                                      // healthy
        frame(TEMP_WARN, 200.0),                                 // warn (inclusive)
        frame(TEMP_CRIT + 5.0, 200.0),                           // thermal trip
        frame(60.0, 200.0),                                      // recovering 1
        frame(60.0, 200.0),                                      // recovering 2
        frame(60.0, 200.0),                                      // recovering 3 → release
        frame(60.0, 200.0),                                      // healthy after release
        assess(&raw(Some(60.0), Some(f32::NAN)), fixtures::NOW), // invalid power trip
        frame(60.0, POWER_CRIT + 1.0),                           // power critical, held
        assess(&raw(None, Some(200.0)), fixtures::NOW),          // temp missing, held
        aged(60.0, 200.0, 10_000),                               // stale, held
        assess(&fixtures::nvml_unavailable(), fixtures::NOW),
    ]
}

/// Drive the scenario with a fake actuator loop that always succeeds.
fn run(
    advisor: Option<&mut (dyn ShadowAdvisor + 'static)>,
) -> (Vec<SafetySnapshot>, Vec<Option<Proposal>>) {
    let mut m = machine();
    let mut log = Vec::new();
    let mut snaps = Vec::new();
    let mut advisor = advisor;
    for f in scenario() {
        let snap = step(&mut m, &f, advisor.as_deref_mut(), &mut log);
        let follow = match snap.intent {
            BrakeIntent::Apply => Some(m.record_actuator(ActuatorOutcome::Applied)),
            BrakeIntent::Release => Some(m.record_actuator(ActuatorOutcome::Released)),
            BrakeIntent::None => None,
        };
        snaps.push(snap);
        snaps.extend(follow);
    }
    (snaps, log)
}

#[test]
fn deterministic_path_is_identical_with_learned_component_absent_panicking_garbage_or_adversarial()
{
    let (baseline, none_log) = run(None);
    assert!(none_log.is_empty());

    // Sanity: the baseline actually trips and releases.
    assert!(baseline.iter().any(|s| s.intent == BrakeIntent::Apply));
    assert!(baseline.iter().any(|s| s.intent == BrakeIntent::Release));

    let mut panicking = PanickingAdvisor;
    let mut garbage = GarbageAdvisor(0);
    let mut adversarial = AdversarialAdvisor;
    let mut hold = HoldAdvisor;
    let advisors: [(&str, &mut (dyn ShadowAdvisor + 'static)); 4] = [
        ("panicking", &mut panicking),
        ("garbage", &mut garbage),
        ("adversarial", &mut adversarial),
        ("hold", &mut hold),
    ];
    for (name, advisor) in advisors {
        let (snaps, log) = run(Some(advisor));
        assert_eq!(
            snaps, baseline,
            "{name} learned component changed deterministic safety"
        );
        assert_eq!(log.len(), scenario().len());
        if name == "panicking" {
            assert!(log.iter().all(Option::is_none), "panic must be contained");
        }
        if name == "adversarial" {
            // It really did recommend the opposite on the critical frame.
            assert_eq!(log[2], Some(Proposal::Release));
        }
    }
}

/// Lexer state for [`strip_comments`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LexState {
    Code,
    LineComment,
    /// Nesting depth of `/* */` (always >= 1).
    BlockComment(usize),
    Str,
    Char,
}

impl LexState {
    /// Closing quote for a literal state.
    const fn quote(self) -> Option<char> {
        match self {
            Self::Str => Some('"'),
            Self::Char => Some('\''),
            _ => None,
        }
    }
}

/// Input chars, read position and output buffer for [`strip_comments`].
/// Every `lex_*` step consumes the current char (plus any second char of a
/// two-char token) and returns the next state.
struct Lexer {
    chars: Vec<char>,
    pos: usize,
    out: String,
}

impl Lexer {
    fn peek(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    /// Consume `n` chars, appending them to the output when `keep` is set.
    fn take(&mut self, n: usize, keep: bool) {
        for _ in 0..n {
            if let (true, Some(c)) = (keep, self.peek(0)) {
                self.out.push(c);
            }
            self.pos += 1;
        }
    }

    fn lex_code(&mut self) -> LexState {
        let (c, next, after) = (self.peek(0), self.peek(1), self.peek(2));
        match (c, next) {
            (Some('/'), Some('/')) => return self.skip(2, LexState::LineComment),
            (Some('/'), Some('*')) => return self.skip(2, LexState::BlockComment(1)),
            _ => {}
        }
        // A char literal ('x' or '\n'), not a lifetime like 'a or 'static.
        let is_char = c == Some('\'') && (next == Some('\\') || after == Some('\''));
        self.take(1, true);
        match c {
            Some('"') => LexState::Str,
            _ if is_char => LexState::Char,
            _ => LexState::Code,
        }
    }

    fn lex_line_comment(&mut self) -> LexState {
        let newline = self.peek(0) == Some('\n');
        self.take(1, newline);
        if newline {
            LexState::Code
        } else {
            LexState::LineComment
        }
    }

    fn lex_block_comment(&mut self, depth: usize) -> LexState {
        match (self.peek(0), self.peek(1)) {
            (Some('/'), Some('*')) => self.skip(2, LexState::BlockComment(depth + 1)),
            (Some('*'), Some('/')) if depth == 1 => self.skip(2, LexState::Code),
            (Some('*'), Some('/')) => self.skip(2, LexState::BlockComment(depth - 1)),
            _ => self.skip(1, LexState::BlockComment(depth)),
        }
    }

    /// Inside a string or char literal: keep everything, honour `\` escapes,
    /// and leave on the closing quote read from `state`.
    fn lex_literal(&mut self, state: LexState) -> LexState {
        let c = self.peek(0);
        if c == Some('\\') && self.peek(1).is_some() {
            self.take(2, true);
            return state;
        }
        self.take(1, true);
        if c == state.quote() {
            LexState::Code
        } else {
            state
        }
    }

    fn skip(&mut self, n: usize, next: LexState) -> LexState {
        self.take(n, false);
        next
    }
}

/// Remove `//` line comments (leading or trailing) and nested `/* */` block
/// comments, keeping string and char literal contents. Enough of a lexer for
/// the structural check; not a general Rust tokenizer (raw strings with `"#`
/// delimiters are treated as ordinary strings).
fn strip_comments(src: &str) -> String {
    let mut lexer = Lexer {
        chars: src.chars().collect(),
        pos: 0,
        out: String::with_capacity(src.len()),
    };
    let mut state = LexState::Code;
    while lexer.pos < lexer.chars.len() {
        state = match state {
            LexState::Code => lexer.lex_code(),
            LexState::LineComment => lexer.lex_line_comment(),
            LexState::BlockComment(depth) => lexer.lex_block_comment(depth),
            LexState::Str | LexState::Char => lexer.lex_literal(state),
        };
    }
    lexer.out
}

#[test]
fn strip_comments_removes_trailing_and_block_comments_only() {
    let src = "let a = 1; // tokio\n/* corpus_ipc /* nested */ */ let b = \"//x\";\nuse tokio;";
    let got = strip_comments(src);
    assert!(
        !got.contains("corpus_ipc") && !got.contains("nested"),
        "{got}"
    );
    assert!(
        got.contains("let b = \"//x\";") && got.contains("use tokio;"),
        "{got}"
    );
    assert_eq!(got.matches("tokio").count(), 1, "{got}");
    // A '"' char literal must not open a string; lifetimes are not chars.
    let got = strip_comments("let q = '\"'; // tokio\nfn f<'a>(x: &'a str) {} // corpus_ipc");
    assert_eq!(got, "let q = '\"'; \nfn f<'a>(x: &'a str) {} ");
}

#[test]
fn safety_module_has_no_dependency_on_learned_ipc_or_runtime_layers() {
    // Structural independence: the pure safety path must compile without
    // corpus-ipc, the publisher, NVML, tokio, or any Brainstem/SNN hook.
    for path in ["src/safety.rs", "src/safety/policy.rs"] {
        let src =
            std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
                .unwrap();
        let code = strip_comments(&src);
        for forbidden in [
            "corpus_ipc",
            "crate::publish",
            "crate::gpu",
            "crate::daemon",
            "nvml_wrapper",
            "tokio",
            "brainstem",
            "Brainstem",
            "neuromod",
        ] {
            assert!(
                !code.contains(forbidden),
                "{path} must not reference `{forbidden}` outside comments"
            );
        }
    }
}

// ── Exact threshold boundaries ──────────────────────────────────────

fn next_up(x: f32) -> f32 {
    f32::from_bits(x.to_bits() + 1)
}

fn next_down(x: f32) -> f32 {
    f32::from_bits(x.to_bits() - 1)
}

/// Healthy and warning never request the brake; every other state does.
fn assert_classified(temp: f32, power: f32, state: SafetyState, code: SafetyReason) {
    let snap = machine().evaluate(&frame(temp, power));
    let should_brake = !matches!(state, SafetyState::HealthyReal | SafetyState::Warning);
    assert_eq!(
        (snap.state, snap.reason_code, snap.desired_brake),
        (state, code, should_brake),
        "temp={temp} power={power}"
    );
}

#[test]
fn thermal_and_power_boundaries_are_exact() {
    use SafetyReason as R;
    use SafetyState as S;
    #[rustfmt::skip]
    let cases = [
        (next_down(TEMP_WARN), 200.0, S::HealthyReal, R::Healthy),
        (TEMP_WARN, 200.0, S::Warning, R::ThermalWarn), // warn is inclusive
        (TEMP_CRIT, 200.0, S::Warning, R::ThermalWarn), // critical is strict
        (next_up(TEMP_CRIT), 200.0, S::CriticalBraked, R::ThermalCritical),
        (60.0, next_down(POWER_WARN), S::HealthyReal, R::Healthy),
        (60.0, POWER_WARN, S::Warning, R::PowerWarn),
        (60.0, POWER_CRIT, S::Warning, R::PowerWarn),
        (60.0, next_up(POWER_CRIT), S::CriticalBraked, R::PowerCritical),
        // Critical on one channel beats warn on the other.
        (TEMP_WARN, POWER_CRIT + 1.0, S::CriticalBraked, R::PowerCritical),
        // Engineering-range edges: 0 °C / 0 W are valid readings, not missing.
        (0.0, 0.0, S::HealthyReal, R::Healthy),
        // 125 °C is in range and critical; beyond the range is invalid.
        (125.0, 200.0, S::CriticalBraked, R::ThermalCritical),
        (125.1, 200.0, S::TelemetryInvalid, R::TempInvalid),
        (-0.1, 200.0, S::TelemetryInvalid, R::TempInvalid),
        (60.0, 2_000.1, S::TelemetryInvalid, R::PowerInvalid),
    ];
    for (temp, power, state, code) in cases {
        assert_classified(temp, power, state, code);
    }
}

#[test]
fn non_finite_telemetry_fails_closed_and_never_passes_silently() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for (r, code) in [
            (raw(Some(bad), Some(200.0)), SafetyReason::TempInvalid),
            (raw(Some(60.0), Some(bad)), SafetyReason::PowerInvalid),
        ] {
            let snap = machine().evaluate(&assess(&r, fixtures::NOW));
            assert_fail_closed_apply(&snap, SafetyState::TelemetryInvalid, code);
            assert_eq!(snap.trip, Some(code), "{bad}");
        }
    }
}

/// Fail-closed snapshot: named state + code, brake desired, apply intent.
fn assert_fail_closed_apply(snap: &SafetySnapshot, state: SafetyState, code: SafetyReason) {
    assert_eq!(
        (
            snap.state,
            snap.reason_code,
            snap.desired_brake,
            snap.intent
        ),
        (state, code, true, BrakeIntent::Apply)
    );
}

// ── Stale / missing / future-dated telemetry ────────────────────────

#[test]
fn contract_staleness_boundary_is_exact_and_fails_closed() {
    let limit = vahtisiru::telemetry::SAFETY_STALE_AFTER_MS;
    let fresh = machine().evaluate(&aged(60.0, 200.0, limit - 1));
    assert_eq!(fresh.state, SafetyState::HealthyReal);
    let stale = machine().evaluate(&aged(60.0, 200.0, limit));
    assert_fail_closed_apply(&stale, SafetyState::TelemetryStale, SafetyReason::TempStale);
}

#[test]
fn policy_staleness_boundary_is_exact_and_fails_closed() {
    let config = SafetyPolicyConfig {
        power_warn_w: Some(POWER_WARN),
        power_critical_w: Some(POWER_CRIT),
        max_sample_age_ms: 50,
        max_acquisition_interval_ms: 50,
        ..Default::default()
    };
    let p = config.resolve(None).unwrap();
    let with_cadence = |age| {
        let mut f = aged(60.0, 200.0, age);
        f.acquisition_cadence_ms = 50;
        f
    };
    let ok = SafetyMachine::with_policy(p.clone()).evaluate(&with_cadence(49));
    assert_eq!(ok.state, SafetyState::HealthyReal);
    let stale = SafetyMachine::with_policy(p.clone()).evaluate(&with_cadence(50));
    assert_fail_closed_apply(
        &stale,
        SafetyState::TelemetryStale,
        SafetyReason::PolicyStale,
    );

    // Declared acquisition interval one ms over the policy is also stale.
    let mut slow = with_cadence(0);
    slow.acquisition_cadence_ms = 51;
    let snap = SafetyMachine::with_policy(p).evaluate(&slow);
    assert_fail_closed_apply(
        &snap,
        SafetyState::TelemetryStale,
        SafetyReason::PolicyStale,
    );
}

#[test]
fn missing_and_future_dated_samples_fail_closed_with_channel_specific_codes() {
    for (r, state, code) in [
        (
            raw(None, Some(200.0)),
            SafetyState::TelemetryMissing,
            SafetyReason::TempMissing,
        ),
        (
            raw(Some(60.0), None),
            SafetyState::TelemetryMissing,
            SafetyReason::PowerMissing,
        ),
        (
            fixtures::nvml_unavailable(),
            SafetyState::TelemetryMissing,
            SafetyReason::TempMissing,
        ),
        (
            RawTelemetry {
                source_unix_ms: Some(fixtures::NOW + 1),
                ..raw(Some(60.0), Some(200.0))
            },
            SafetyState::TelemetryInvalid,
            SafetyReason::TempInvalid,
        ),
    ] {
        assert_fail_closed_apply(&machine().evaluate(&assess(&r, fixtures::NOW)), state, code);
    }

    // Unknown device envelope: real telemetry cannot be authorized healthy.
    assert_fail_closed_apply(
        &SafetyMachine::new().evaluate(&frame(40.0, 50.0)),
        SafetyState::TelemetryMissing,
        SafetyReason::PowerPolicyUnavailable,
    );
}

#[test]
fn before_any_telemetry_the_machine_is_fail_closed() {
    let snap = machine().snapshot();
    assert_eq!(snap.state, SafetyState::TelemetryMissing);
    assert_eq!(snap.reason_code, SafetyReason::NoTelemetryYet);
    assert!(snap.desired_brake);
}

// ── Hysteresis ──────────────────────────────────────────────────────

fn braked_machine() -> SafetyMachine {
    let mut m = machine();
    assert_eq!(m.evaluate(&frame(90.0, 200.0)).intent, BrakeIntent::Apply);
    assert!(m.record_actuator(ActuatorOutcome::Applied).brake_engaged);
    m
}

#[test]
fn release_requires_exactly_the_configured_ok_streak() {
    let mut m = braked_machine();
    let ok = frame(60.0, 200.0);
    for n in 1..vahtisiru::safety::RELEASE_OK_STREAK {
        let snap = m.evaluate(&ok);
        assert_eq!(snap.state, SafetyState::Recovering);
        assert_eq!(snap.hysteresis_ok_count, n);
        assert_eq!(snap.intent, BrakeIntent::None, "released early at {n}");
    }
    assert_eq!(m.evaluate(&ok).intent, BrakeIntent::Release);
}

#[test]
fn any_non_ok_frame_during_recovery_resets_the_streak_and_holds_the_brake() {
    let interruptions = [
        frame(TEMP_WARN, 200.0),
        frame(60.0, POWER_WARN),
        frame(90.0, 200.0),
        assess(&raw(None, Some(200.0)), fixtures::NOW),
        assess(&raw(Some(f32::NAN), Some(200.0)), fixtures::NOW),
        aged(60.0, 200.0, 10_000),
        assess(&fixtures::software_fallback(), fixtures::NOW),
    ];
    let ok = frame(60.0, 200.0);
    for bad in interruptions {
        let mut m = braked_machine();
        let _ = m.evaluate(&ok);
        let _ = m.evaluate(&ok);
        let snap = m.evaluate(&bad);
        assert_eq!(snap.hysteresis_ok_count, 0, "{:?}", snap.reason_code);
        assert!(
            snap.desired_brake,
            "{:?} released the brake",
            snap.reason_code
        );
        assert_ne!(snap.intent, BrakeIntent::Release);
        // Streak restarts from zero.
        let _ = m.evaluate(&ok);
        let _ = m.evaluate(&ok);
        assert_eq!(m.evaluate(&ok).intent, BrakeIntent::Release);
    }
}

#[test]
fn simulated_telemetry_can_never_release_a_real_brake() {
    let mut m = braked_machine();
    let sim = assess(&fixtures::software_fallback(), fixtures::NOW);
    assert_eq!(sim.source, TelemetrySource::SoftwareFallback);
    for _ in 0..50 {
        let snap = m.evaluate(&sim);
        assert_eq!(snap.state, SafetyState::SimulatedSoftwareOnly);
        assert!(snap.desired_brake);
        assert_eq!(snap.intent, BrakeIntent::None);
    }
}

// ── Structured trip events ──────────────────────────────────────────

#[test]
fn every_trip_emits_exactly_one_rising_edge_event_with_reason_code() {
    let mut m = machine();
    assert_eq!(m.evaluate(&frame(60.0, 200.0)).trip, None);
    assert_eq!(m.evaluate(&frame(TEMP_WARN, 200.0)).trip, None);

    let trip = m.evaluate(&frame(90.0, 200.0));
    assert_eq!(trip.trip, Some(SafetyReason::ThermalCritical));
    assert_eq!(m.trips_total(), 1);

    // Held: no duplicate trip events while the brake is still desired,
    // including the actuator ack and different fail-closed reasons.
    let held = [
        m.record_actuator(ActuatorOutcome::Applied).trip,
        m.evaluate(&frame(60.0, POWER_CRIT + 5.0)).trip,
        m.evaluate(&aged(60.0, 200.0, 10_000)).trip,
    ];
    assert_eq!((held, m.trips_total()), ([None; 3], 1));

    // Recover and release.
    let ok = frame(60.0, 200.0);
    let _ = m.evaluate(&ok);
    let _ = m.evaluate(&ok);
    assert_eq!(m.evaluate(&ok).intent, BrakeIntent::Release);
    let released = m.record_actuator(ActuatorOutcome::Released);
    assert_eq!(
        (released.reason_code, released.trip, m.evaluate(&ok).trip),
        (SafetyReason::BrakeReleased, None, None)
    );

    // Re-trip on a different cause emits a new event with that cause.
    let again = m.evaluate(&assess(&raw(Some(60.0), None), fixtures::NOW));
    assert_eq!(again.trip, Some(SafetyReason::PowerMissing));
    assert_eq!(m.trips_total(), 2);
}

/// Blocks review (PR #85): the apply runs off the evaluation path, so frames
/// keep arriving before the actuator ack. A late ack after the frame dropped
/// into the warn band must not emit a second (mislabelled) trip.
#[test]
fn late_apply_ack_after_warn_frame_is_not_a_second_trip() {
    let mut m = machine();
    let steps = [
        m.evaluate(&frame(90.0, 200.0)),
        m.evaluate(&frame(80.0, 200.0)),
        m.record_actuator(ActuatorOutcome::Applied),
        m.evaluate(&frame(80.0, 200.0)),
    ];
    let trips: Vec<_> = steps.iter().map(|s| s.trip).collect();
    assert_eq!(
        (trips, m.trips_total()),
        (
            vec![Some(SafetyReason::ThermalCritical), None, None, None],
            1
        )
    );
    // Brake decisions are unchanged by the trip bookkeeping.
    let desired: Vec<_> = steps.iter().map(|s| (s.desired_brake, s.intent)).collect();
    assert_eq!(
        desired,
        vec![
            (true, BrakeIntent::Apply),
            (false, BrakeIntent::None),
            (true, BrakeIntent::None),
            (true, BrakeIntent::None),
        ]
    );
}

/// Blocks review (PR #85): a warn-band frame arriving while the release is in
/// flight (and again right after the Released ack) holds or re-applies the
/// brake, but the brake never came off in between, so it is not a new trip.
#[test]
fn warn_frame_while_release_in_flight_is_not_a_new_trip() {
    let mut m = braked_machine();
    assert_eq!(m.trips_total(), 1);
    let ok = frame(60.0, 200.0);
    let _ = m.evaluate(&ok);
    let _ = m.evaluate(&ok);
    let steps = [
        m.evaluate(&ok),
        m.evaluate(&frame(80.0, 200.0)),
        m.record_actuator(ActuatorOutcome::Released),
        m.evaluate(&frame(80.0, 200.0)),
    ];
    let trips: Vec<_> = steps.iter().map(|s| s.trip).collect();
    assert_eq!((trips, m.trips_total()), (vec![None; 4], 1));
    let decisions: Vec<_> = steps
        .iter()
        .map(|s| (s.state, s.desired_brake, s.intent))
        .collect();
    assert_eq!(
        decisions,
        vec![
            (SafetyState::Recovering, false, BrakeIntent::Release),
            (SafetyState::Warning, true, BrakeIntent::None),
            (SafetyState::HealthyReal, false, BrakeIntent::None),
            (SafetyState::Warning, true, BrakeIntent::Apply),
        ]
    );
    // A genuine re-trip after the brake really came off still emits.
    let _ = m.record_actuator(ActuatorOutcome::Applied);
    for _ in 0..3 {
        let _ = m.evaluate(&ok);
    }
    let _ = m.record_actuator(ActuatorOutcome::Released);
    assert_eq!(m.evaluate(&ok).state, SafetyState::HealthyReal);
    let again = m.evaluate(&frame(90.0, 200.0));
    assert_eq!(
        (again.trip, m.trips_total()),
        (Some(SafetyReason::ThermalCritical), 2)
    );
}

#[test]
fn leftover_brake_adoption_is_not_a_new_trip() {
    let mut m = machine();
    m.seed_brake_applied();
    assert_eq!(m.snapshot().reason_code, SafetyReason::LeftoverBrake);
    assert_eq!(m.evaluate(&frame(60.0, 200.0)).trip, None);
    assert_eq!(m.trips_total(), 0);
}

#[test]
fn actuator_failure_is_observable_and_does_not_suppress_the_trip() {
    let mut m = machine();
    let trip = m.evaluate(&frame(90.0, 200.0));
    assert_eq!(trip.trip, Some(SafetyReason::ThermalCritical));
    let failed = m.record_actuator(ActuatorOutcome::ApplyFailed("sudo denied".into()));
    // Apply must be retried; the failure is counted and named.
    assert_fail_closed_apply(
        &failed,
        SafetyState::ActuatorFailure,
        SafetyReason::ApplyFailed,
    );
    assert_eq!(
        (failed.actuator_failed, m.actuator_failures_total()),
        (true, 1)
    );
    // Next evaluation still classifies and still wants the brake.
    let next = m.evaluate(&frame(90.0, 200.0));
    assert_eq!(next.policy_state, SafetyState::CriticalBraked);
    assert_eq!(next.intent, BrakeIntent::Apply);
}

#[test]
fn reason_codes_are_a_closed_unique_vocabulary() {
    let labels: std::collections::BTreeSet<_> =
        SafetyReason::ALL.iter().map(|r| r.as_str()).collect();
    assert_eq!(labels.len(), SafetyReason::ALL.len());
    for l in labels {
        assert!(l.chars().all(|c| c.is_ascii_lowercase() || c == '_'), "{l}");
    }
}

// ── Sensory publisher failure isolation ─────────────────────────────

struct PanickingPublisher;
impl SensoryPublisher for PanickingPublisher {
    fn try_publish(&self, _mapping: &SensoryMapping) -> Result<(), PublishError> {
        panic!("publisher bug");
    }
}

#[test]
fn panicking_publisher_cannot_unwind_through_safety_evaluation() {
    let mut m = machine();
    let (snap, res) = evaluate_then_try_publish(&mut m, &frame(90.0, 200.0), &PanickingPublisher);
    assert_fail_closed_apply(
        &snap,
        SafetyState::CriticalBraked,
        SafetyReason::ThermalCritical,
    );
    assert_eq!(snap.trip, Some(SafetyReason::ThermalCritical));
    assert_eq!(
        res,
        Err(PublishError::SendFailed(PUBLISHER_PANICKED.to_string()))
    );
    // The machine is still usable and still deterministic afterwards.
    let (snap, _) = evaluate_then_try_publish(&mut m, &frame(90.0, 200.0), &AbsentPublisher);
    assert_eq!(snap.intent, BrakeIntent::Apply);
}
