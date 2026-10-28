use vahtisiru::safety::{
    ActuatorOutcome, BrakeIntent, SafetyMachine, SafetyPolicyConfig, SafetyState,
};
use vahtisiru::telemetry::{assess, fixtures};

fn frame(power: f32) -> vahtisiru::telemetry::TelemetryFrame {
    let mut raw = fixtures::healthy_real();
    raw.gpu_temp_c = Some(60.0);
    raw.power_w = Some(power);
    assess(&raw, fixtures::NOW)
}

#[test]
fn device_envelopes_change_actual_safety_decisions() {
    for (default_w, power, expected) in [
        (100.0, 80.0, SafetyState::HealthyReal),
        (100.0, 90.0, SafetyState::Warning),
        (100.0, 101.0, SafetyState::CriticalBraked),
        (600.0, 400.0, SafetyState::HealthyReal),
        (600.0, 520.0, SafetyState::Warning),
        (600.0, 601.0, SafetyState::CriticalBraked),
    ] {
        let policy = SafetyPolicyConfig::default()
            .resolve(Some(default_w))
            .unwrap();
        let mut machine = SafetyMachine::with_policy(policy);
        assert_eq!(machine.evaluate(&frame(power)).state, expected);
    }
}

#[test]
fn unknown_envelope_cannot_authorize_real_health_or_release() {
    let mut machine = SafetyMachine::new();
    assert_eq!(machine.evaluate(&frame(10.0)).intent, BrakeIntent::Apply);
    machine.seed_brake_applied();
    for _ in 0..10 {
        assert_eq!(machine.evaluate(&frame(10.0)).intent, BrakeIntent::None);
        assert!(machine.snapshot().desired_brake);
    }
    let simulated = assess(&fixtures::software_fallback(), fixtures::NOW);
    assert_eq!(
        machine.evaluate(&simulated).state,
        SafetyState::SimulatedSoftwareOnly
    );
    assert!(machine.snapshot().desired_brake);
}

#[test]
fn explicit_limits_work_without_capabilities_and_control_recovery() {
    let config = SafetyPolicyConfig {
        power_warn_w: Some(40.0),
        power_critical_w: Some(50.0),
        release_ok_streak: 5,
        ..Default::default()
    };
    let mut machine = SafetyMachine::with_policy(config.resolve(None).unwrap());
    assert_eq!(machine.evaluate(&frame(51.0)).intent, BrakeIntent::Apply);
    let _ = machine.record_actuator(ActuatorOutcome::Applied);
    for _ in 0..4 {
        assert_eq!(machine.evaluate(&frame(30.0)).intent, BrakeIntent::None);
    }
    assert_eq!(machine.evaluate(&frame(30.0)).intent, BrakeIntent::Release);
}

#[test]
fn malformed_or_device_exceeding_policy_is_rejected() {
    for (warn, critical) in [
        (0.0, 20.0),
        (20.0, 20.0),
        (30.0, 20.0),
        (f32::NAN, 20.0),
        (20.0, f32::INFINITY),
    ] {
        let c = SafetyPolicyConfig {
            power_warn_w: Some(warn),
            power_critical_w: Some(critical),
            ..Default::default()
        };
        assert!(c.resolve(None).is_err());
    }
    assert!(
        SafetyPolicyConfig {
            power_warn_w: Some(20.0),
            ..Default::default()
        }
        .resolve(None)
        .is_err()
    );
    assert!(
        SafetyPolicyConfig {
            power_warn_w: Some(90.0),
            power_critical_w: Some(110.0),
            ..Default::default()
        }
        .resolve(Some(100.0))
        .is_err()
    );
    assert!(
        SafetyPolicyConfig {
            temp_warn_c: 90.0,
            temp_critical_c: 80.0,
            ..Default::default()
        }
        .resolve(None)
        .is_err()
    );
    assert!(
        SafetyPolicyConfig {
            release_ok_streak: 0,
            ..Default::default()
        }
        .resolve(None)
        .is_err()
    );
    for cap in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(SafetyPolicyConfig::default().resolve(Some(cap)).is_err());
    }
}

#[test]
fn policy_can_tighten_freshness_and_acquisition_cadence() {
    let config = SafetyPolicyConfig {
        max_sample_age_ms: 50,
        max_acquisition_interval_ms: 20,
        ..Default::default()
    };
    let mut machine = SafetyMachine::with_policy(config.resolve(Some(100.0)).unwrap());
    assert_eq!(
        machine.evaluate(&frame(20.0)).state,
        SafetyState::TelemetryStale
    );
    let mut fresh = frame(20.0);
    fresh.acquisition_cadence_ms = 20;
    assert_eq!(machine.evaluate(&fresh).state, SafetyState::HealthyReal);
    fresh.gpu_temp_c.observed_at -= 50;
    assert_eq!(machine.evaluate(&fresh).state, SafetyState::TelemetryStale);
}

#[test]
fn binary_rejects_contradictory_policy_before_process_lock() {
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_vahtisiru"))
        .args([
            "--force-software-only",
            "--safety-temp-warn-c",
            "90",
            "--safety-temp-critical-c",
            "80",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("temperature limits must be finite"),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn binary_validates_environment_policy_and_cli_precedence() {
    let bin = env!("CARGO_BIN_EXE_vahtisiru");
    let result = std::process::Command::new(bin)
        .env("VAHTISIRU_SAFETY_POWER_WARN_W", "40")
        .args(["--force-software-only", "--safety-power-critical-w", "30"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("power limits must be finite"));
    // CLI must override the otherwise invalid environment temperature warning.
    // The second, deliberately invalid power pair stops before hardware/ports.
    let result = std::process::Command::new(bin)
        .env("VAHTISIRU_SAFETY_TEMP_WARN_C", "100")
        .args([
            "--force-software-only",
            "--safety-temp-warn-c",
            "70",
            "--safety-power-warn-w",
            "40",
            "--safety-power-critical-w",
            "30",
        ])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&result.stderr).contains("power limits must be finite"));
}

#[test]
fn explicit_temperature_and_power_override_the_derived_policy() {
    let config = SafetyPolicyConfig {
        temp_warn_c: 50.0,
        temp_critical_c: 65.0,
        power_warn_w: Some(40.0),
        power_critical_w: Some(50.0),
        ..Default::default()
    };
    let policy = config.resolve(Some(100.0)).unwrap();
    let mut machine = SafetyMachine::with_policy(policy);
    let mut sample = frame(20.0); // 60 C is warning under explicit 50/65 policy.
    assert_eq!(machine.evaluate(&sample).state, SafetyState::Warning);
    sample.gpu_temp_c.value = Some(66.0);
    assert_eq!(machine.evaluate(&sample).state, SafetyState::CriticalBraked);
    sample.gpu_temp_c.value = Some(40.0);
    sample.power_w.value = Some(51.0);
    assert_eq!(machine.evaluate(&sample).state, SafetyState::CriticalBraked);
}

#[test]
fn invalid_freshness_and_engineering_bounds_are_rejected() {
    for c in [
        SafetyPolicyConfig {
            max_sample_age_ms: 0,
            ..Default::default()
        },
        SafetyPolicyConfig {
            max_sample_age_ms: 2001,
            ..Default::default()
        },
        SafetyPolicyConfig {
            max_acquisition_interval_ms: 0,
            ..Default::default()
        },
        SafetyPolicyConfig {
            max_acquisition_interval_ms: 2001,
            ..Default::default()
        },
        SafetyPolicyConfig {
            temp_critical_c: 126.0,
            ..Default::default()
        },
        SafetyPolicyConfig {
            temp_warn_c: f32::NAN,
            ..Default::default()
        },
        SafetyPolicyConfig {
            power_warn_w: Some(100.0),
            power_critical_w: Some(2001.0),
            ..Default::default()
        },
    ] {
        assert!(c.resolve(None).is_err());
    }
    assert!(SafetyPolicyConfig::default().resolve(Some(2001.0)).is_err());
}

#[test]
fn binary_rejects_cadence_that_exceeds_policy() {
    for args in [
        vec!["--step-interval-ms", "101"],
        vec![
            "--step-interval-ms",
            "301",
            "--safety-max-acquisition-interval-ms",
            "300",
        ],
    ] {
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_vahtisiru"))
            .arg("--force-software-only")
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("step interval exceeds safety"));
    }
}
