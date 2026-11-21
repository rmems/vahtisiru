//! Miscellaneous step/display tests.

use super::*;

#[test]
fn dashboard_shows_only_valid_present_as_live_hardware() {
    let watts = |w: f32| format!("{w:.0}W");
    let celsius = |t: f32| format!("{t:.0}");
    let live = assess(&fixtures::healthy_real(), fixtures::NOW);
    let stale = assess(&fixtures::stale(), fixtures::NOW);
    let invalid = assess(&fixtures::out_of_range(), fixtures::NOW);
    let missing = assess(&fixtures::sensor_dropout(), fixtures::NOW);
    let shown = [
        format_live_reading(&live.power_w, watts),
        format_live_reading(&stale.power_w, watts),
        format_live_reading(&invalid.gpu_temp_c, celsius),
        format_live_reading(&missing.power_w, watts),
    ];
    assert_eq!(shown, ["200W", "stale", "inv", "n/a"]);
}
