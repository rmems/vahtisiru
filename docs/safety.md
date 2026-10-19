# Safety failure domain

This is the normative hardware-protection contract for `vahtisiru`
([GH#42](https://github.com/rmems/vahtisiru/issues/42)).

Safety evaluation is an **isolated failure domain**. It continues with
Brainstem completely absent, and it never waits on `corpus-ipc` send,
disconnect, or a slow consumer. Brainstem has **no** authority to override
Vahtisiru hard-safety policy — there is no IPC command that can inhibit the
brake.

Privileged `nvidia-smi` actuation stays in private `src/gpu.rs` (used only by
the `vahtisiru` executable). The state machine in `src/safety.rs` only
emits **intents**; hardware side effects go through [`SafetyActuator`]
(`NvmlActuator` in production, `FakeActuator` in tests). Apply/release is
best-effort and requires Linux, NVML, and passwordless `sudo -n nvidia-smi`
(see the crate README).

### GPU identity

The relay resolves a single GPU identity once at startup: it reads NVML index 0,
takes its immutable UUID, validates the UUID shape, and confirms it resolves
back through `device_by_uuid`. Every telemetry read, power-limit query, and
`nvidia-smi` command for the life of the process is scoped to that UUID:

```text
timeout -k 2 5s sudo -n nvidia-smi -i <UUID> -pl <watts>
```

Because the command pins `-i <UUID>`, a `sudo` rule restricted to a specific
`nvidia-smi` invocation must permit the `-i <UUID> -pl <watts>` form; the older
unqualified `nvidia-smi -pl` form no longer matches. If identity resolution
fails, the relay logs the reason and continues with **fail-closed** telemetry
(`NvmlUnavailable`) while **refusing** power-limit mutation — apply and release
return an `ActuatorError` without building or running any command. The relay
never tries another device and does not re-resolve mid-run; a restart binds a
newly attached GPU. This is a single-GPU (index 0) supervisor, not a
GPU-selection or multi-GPU tool.

## Ownership

| Guarantee | Owner |
| --- | --- |
| Telemetry acquisition, validity, freshness, provenance | Vahtisiru (`telemetry` + `gpu` acquire) |
| Hard-safety classification, hysteresis, brake intent | Vahtisiru (`safety`) |
| Power-limit apply/release | Vahtisiru (`gpu` actuator) |
| Safety/brake state and transition/error counters | Vahtisiru Prometheus (`:9000/metrics`) |
| Orderly SIGINT/SIGTERM shutdown (fail-closed) | Vahtisiru (`shutdown` + supervisor) |
| Sensory mapping types | Vahtisiru (`TelemetryFrame::to_sensory_mapping`) |
| Sensory transport to Brainstem | Vahtisiru `publish` → `corpus-ipc` `IpcMessage::Stimuli` (not required for safety) |
| SNN tick, neuromodulation, neural state | Brainstem |
| Reward / plasticity | Brainstem (never Vahtisiru) |

Vahtisiru does **not** run neural computation. Observing safety does **not**
require querying neural state.

## Named states

| `SafetyState` | When | Brake policy |
| --- | --- | --- |
| `healthy_real` | NVML, valid temp/power, below warn, brake released | none |
| `warning` | Valid configured warn-band | hold if already on; **re-apply if just released** |
| `critical_braked` | Valid temp or power exceeds configured critical limit | apply / hold |
| `recovering` | Brake on; consecutive **real** Ok streak | release after configured Ok streak (default 3) |
| `telemetry_missing` | Safety sample or power policy missing (`NvmlUnavailable`, dropout, Valid+`None`) | apply / hold (fail closed) |
| `telemetry_stale` | Safety sample older than stale threshold | apply / hold |
| `telemetry_invalid` | Non-finite or out of engineering range | apply / hold |
| `simulated_software_only` | `TelemetrySource::SoftwareFallback` | do not apply; **hold** if already on; reset hysteresis |
| `actuator_failure` | Last apply/release failed | overlay; retry the last intent; policy still evaluates |

`NvmlUnavailable` is **not** simulated. Missing safety samples fail closed.
`--force-software-only` is the only simulated path; simulated numbers never
drive thresholds.

When missing, invalid, and stale appear on different safety channels, the
worst fault wins: **missing > invalid > stale**. Equal rank prefers
`gpu_temp_c` (stable, matches the previous temp-then-power order).

## Hysteresis

1. Critical / missing / invalid / stale → `desired_brake = true`.
2. Warn while released (and not immediately post-release) does **not** apply.
3. Warn or critical **immediately after a successful release** re-applies
   (the restored default PL is not left in a still-hot/warn band).
4. Release requires the configured real `Ok` streak
   (`SafetyPolicyConfig.release_ok_streak`, default **3**) while the brake is
   engaged (~3 s at the default streak and 100 ms tick × every 10 ticks).
5. Simulated telemetry while braked holds the brake and resets the Ok streak.
6. Actuator failure does not freeze evaluation: the next frame still
   classifies and the intent is retried.

The supervisor evaluates the first acquired frame immediately, then on
the safety cadence (every 10 ticks) and again after an actuator task
completes, using a fresh frame. Failed apply/release is recorded and retried
on that cadence rather than every tick. `SafetyMachine::evaluate`
never publishes and never calls `nvidia-smi`. Pre-telemetry snapshots are
not dispatched as hardware commands: `--force-software-only` must be
classified first (hold, do not apply).

## Trip events and reason codes

Every `FrameAssessment` and `SafetySnapshot` carries a closed
`SafetyReason` code (`thermal_critical`, `power_critical`, `thermal_warn`,
`power_warn`, `temp_missing`, `power_missing`, `temp_invalid`,
`power_invalid`, `temp_stale`, `power_stale`, `policy_stale`,
`power_policy_unavailable`, `no_telemetry_yet`, `leftover_brake`,
`brake_released`, `apply_failed`, `release_failed`, `healthy`, `simulated`).
Free-text `last_reason` stays for logs only and is never a metric label.

`SafetySnapshot::trip` is `Some(code)` exactly on the step where policy newly
wants the brake (rising edge of `desired_brake`), and `None` while the brake
is merely held, on actuator feedback, and on leftover-brake adoption. The
supervisor logs `SAFETY TRIP reason=<code> state=<state>` and increments
`safety_trips_total{reason}`.

## Learned components have no input

`SafetyMachine::evaluate` takes only a `TelemetryFrame` and its immutable
`SafetyPolicy`. There is no parameter, trait, or IPC message through which a
learned / SNN / Brainstem component can suppress, delay, or override a trip.
`tests/safety_independence.rs` proves the snapshot sequence is identical when
a shadow advisor is absent, panics, returns NaN/inf garbage, or recommends the
opposite action, and statically checks that `src/safety*.rs` does not
reference `corpus_ipc`, `publish`, `gpu`, `daemon`, NVML, tokio, or Brainstem.
Sensory publication runs inside `try_publish_isolated`, so a panicking
publisher is contained and reported as `SendFailed("sensory publisher panicked")`.

## Configurable policy

`SafetyPolicyConfig::resolve(device_default_w)` validates finite ordered limits
and produces an immutable `SafetyPolicy`. Paired operator watt limits override
the derived 85%/100% device-default envelope and cannot exceed a known default.
Missing capabilities without explicit watts fail closed on real frames. Thermal
75/85 C defaults are explicit operator policy, not vendor-derived limits.

`SafetyMachine::new()` intentionally has no device envelope. Library consumers
resolve their capabilities/configuration and use `SafetyMachine::with_policy`.
`classify_frame_with_policy` and `instant_status_with_policy` use the same policy;
the variants without a policy cannot authorize healthy real telemetry.

Sample age can be tightened below 2000 ms, and the declared acquisition interval
is bounded (100 ms by default). Age is measured at the frame assessment time;
reassess held frames before evaluation. The daemon rejects configured ten-tick
intervals exceeding the sample-age limit. Actual scheduling and sensor reads
add elapsed time, so this is not a hard real-time response guarantee. The full
CLI/environment option table and valid ranges are in the README.

The 50% brake fraction stays fixed for restart compatibility. Apply requires
readable positive current/default power limits and refuses foreign sub-default
caps above or below the target; it never substitutes current for default.
Release rechecks that current still matches the expected relay target. Refusal
is an observable actuator error. Matching within 2 W is heuristic ownership:
external caps at the same target cannot be distinguished, and concurrent
operator writes can race the read/command sequence.

## Shutdown and restart

SIGINT (Ctrl-C) and SIGTERM enter a **controlled shutdown**. The process lock
at `/tmp/vahtisiru.lock` is released on the way out. Background metrics
collection and in-flight actuation are joined with a bounded timeout (they
must not hang indefinitely). Sensory UDP publication is best-effort and has no shutdown delivery guarantee. SIGKILL and power loss are **not** promised to
clean up.

**Fail-closed invariant:** shutdown never dispatches a new apply or release.
Exiting must not convert an unresolved unsafe or unverified state into a
full-power device by restoring the default GPU power limit.

| Situation at signal | Hardware action |
| --- | --- |
| Healthy, brake released | leave hardware unchanged |
| Warned, brake not engaged | do not apply; do not restore |
| Brake requested (not yet applied) | do not dispatch a new apply; next start evaluates immediately |
| In-flight apply | await (bounded); completing apply is fail-closed-friendly |
| Brake active | leave the throttle in place |
| Recovering / release pending but not dispatched | **abandon** the release; leave the brake on |
| In-flight release (already hysteresis-authorized) | await (bounded); this is the same policy that ran in-loop |
| Actuator failed | do not retry on the way out; log unresolved |
| Telemetry missing / stale / invalid | leave hardware unchanged (unverified) |
| Simulated / software-only with a real brake held | **must not** release; simulated numbers cannot authorize recovery |

Crash/restart recovery uses `classify_power_limit`:

- Current PL matches the relay's expected 50% target (2 W tolerance) → adopt
  as a leftover brake (`seed_brake_applied`) and release only after the
  configured real Ok streak (default 3).
- Current PL is below default but **not** that target → treat as an
  operator/device cap. Do **not** seed, do **not** auto-release.
- Limits unreadable, or at/above default → do not seed. The first acquired
  frame still fail-closes if telemetry is missing.

`--force-software-only` does not skip leftover-brake detection on the real
actuator, and it binds the same startup-resolved GPU UUID as the normal path so
a real persistent brake stays addressable under simulated telemetry. Simulated
frames hold an adopted brake and reset hysteresis. Leftover-brake detection and
release act on the UUID resolved at this start; if identity resolution failed,
power limits read as unreadable and no brake is adopted or released.

## IPC isolation

```text
TelemetryFrame
      │
      ▼  SafetyMachine::evaluate   ← never blocks, never IPC
SafetySnapshot (state, brake, intent)
      │
      ├─ spawn_blocking apply/release   (gpu, not on the eval path)
      └─ SensoryPublisher::try_publish   (best-effort, after eval)
             IsolatedPublishQueue.try_enqueue  (bounded; policy on full)
                   │
                   ▼  CorpusIpcPublisher worker (not awaited)
             IpcMessage::Stimuli JSON → UDP sendto
```

Production uses [`CorpusIpcPublisher`](../src/publish.rs) unless
`--ipc-disabled` (then [`AbsentPublisher`](../src/publish.rs)). The queue is
a bounded [`IsolatedPublishQueue`](../src/publish.rs) with a validated
capacity (`--sensory-queue-capacity`, default 32, range 1–4096) and an
explicit full-queue policy (`--sensory-queue-full-policy`, default
`drop-oldest`; also `reject-newest`); the worker is a detached UDP sender.
A full or disconnected queue is `SlowConsumer` / `Disconnected` (or a
successful enqueue that discarded the oldest frame) and **must not** be
awaited from the safety loop. Brainstem absence cannot stall evaluation.
**Safety evaluation continues** regardless of queue pressure.

## Prometheus

Exported without querying Brainstem:

| Metric | Kind | Meaning |
| --- | --- | --- |
| `safety_state{state=...}` | gauge 0/1 | one-hot reported state |
| `safety_state_id` | gauge 0–8 | stable numeric id |
| `safety_policy_state{state=...}` | gauge 0/1 | policy classification before ActuatorFailure overlay |
| `safety_brake_engaged` | gauge 0/1 | last successful apply still claimed |
| `safety_hysteresis_ok_count` | gauge | Ok streak while braked |
| `safety_transitions_total` | counter | reported-state changes |
| `safety_actuator_failures_total` | counter | apply/release errors |
| `safety_trips_total{reason}` | counter | deterministic trip events (rising edge of `desired_brake`); closed `SafetyReason` label set, all pre-registered at 0 |
| `telemetry_freshness_s` | gauge | sample age at scrape time (monotonic receive instant) |
| `sensory_queue_depth` | gauge | frames currently buffered |
| `sensory_queue_capacity` | gauge | configured finite capacity |
| `sensory_queue_enqueued_total` | counter | frames accepted into the queue |
| `sensory_queue_dropped_total{reason}` | counter | closed reason set: `reject_newest`, `drop_oldest`, `absent`, `disconnected`, `send_failed`, `mutex_contended` |
| `sensory_queue_full_policy{policy}` | gauge 0/1 | one-hot `drop_oldest` / `reject_newest` |
| `shutdown_total{reason}` | counter | orderly shutdown (`sigint` / `sigterm`) |
| `shutdown_unresolved_brake` | gauge 0/1 | brake still claimed or still desired at exit |
| `shutdown_unresolved_actuator` | gauge 0/1 | last apply/release still failed at exit |
| `shutdown_brake_left_engaged` | gauge 0/1 | hardware brake left in place (not restored) |

Drop `reason` is a closed vocabulary. Transport error strings and sensory
payloads are **never** used as labels, so series cardinality cannot grow
from input data.

Numeric ids: 0 `healthy_real`, 1 `warning`, 2 `critical_braked`,
3 `recovering`, 4 `telemetry_missing`, 5 `telemetry_stale`,
6 `telemetry_invalid`, 7 `simulated_software_only`, 8 `actuator_failure`.

Brake targets whose ±2 W matching band overlaps the device-default band are
refused as ambiguous (including default limits of 8 W or less with the 50%
strategy). They cannot be adopted as an already-engaged relay brake.
