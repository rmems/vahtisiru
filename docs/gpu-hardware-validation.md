# Trusted GPU hardware validation

The normal GitHub-hosted CI remains the merge and release qualification path.
It runs without an NVIDIA driver or GPU. This optional workflow adds evidence
from a physical NVIDIA device for a trusted commit; it is not a required PR
check and does not replace the hosted jobs.

## Read-only smoke

The `Trusted GPU Hardware Smoke` workflow is started manually from the GitHub
Actions page using the `main` branch. Its job has an explicit `main` ref guard
and runs only on a runner carrying all of these labels:

```text
self-hosted, linux, x64, gpu, nvidia
```

The runner needs the NVIDIA driver, `nvidia-smi`, a working NVML library, and
GNU coreutils (`timeout`). The production NVIDIA adapter uses `timeout` to bound
health and actuation commands. Keep the runner dedicated to this repository and
offline from untrusted workloads. Public pull requests do not trigger this
workflow. GitHub allows manual dispatch only to users with write access;
protect `main` and
limit that access to maintainers trusted to run code on the hardware host.
The workflow has only `contents: read` permission and checks out without
persistent credentials.

The `--gpu-hardware-smoke` mode first runs
`timeout -k 2s 3s nvidia-smi -L`, then uses the same NVML adapter as the
supervisor to:

1. Resolve NVML index 0 to a validated physical GPU UUID and round-trip it
   through NVML.
2. Read live temperature and board-power telemetry through that UUID.
3. Read current and default power limits through that UUID.
4. Print the UUID, readings, and read-only validation scope in the job log.

Missing driver/device access, invalid telemetry, an unresolved or
non-round-tripping UUID, or unreadable power limits fail the job explicitly.
This phase does not invoke power-limit actuation, `sudo`, or any `nvidia-smi`
write operation. No passwordless sudo rule is needed.

For an equivalent local run on a trusted Linux host with the repository's Rust
toolchain installed:

```sh
nvidia-smi -L
cargo run --locked --bin vahtisiru -- --gpu-hardware-smoke
```

## Actuation smoke (maintainer only)

The `Trusted GPU Hardware Actuation Smoke` workflow
(`.github/workflows/gpu-hardware-actuation-smoke.yml`) is the Phase 2 lane:
it **mutates the GPU power limit** on the trusted runner to prove the real
brake path applies and releases correctly. It is a separate, deliberate
maintainer action — never a PR check, never scheduled, and never a release
gate. Software-only CI remains authoritative for portable behaviour; this
lane only adds evidence that the production actuator works on one physical
device.

### Trust boundary

Same runner labels and ref guard as the read-only smoke, plus:

- Dispatch requires two explicit inputs: the physical GPU UUID that may be
  mutated and a ticked `confirm_mutation` acknowledgement. The job does not
  start without both and without `github.ref == refs/heads/main`.
- A `concurrency` group serialises actuation runs and never cancels one in
  progress, so cleanup is not interrupted by a second dispatch.
- The job has `contents: read` only and checks out without credentials. No
  secret is required or exposed; the sudoers rule is host configuration.

### Required host configuration

In addition to the read-only requirements, the runner user needs a
non-interactive sudoers rule that permits **only** the UUID-scoped
power-limit form the relay uses, for example:

```text
runner ALL=(root) NOPASSWD: /usr/bin/nvidia-smi -i GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx -pl [0-9]*
```

The smoke checks this rule with `sudo -n -l -- nvidia-smi -i <UUID> -pl <W>`
for both the apply and the release wattage **before** running anything.
`sudo -l <command>` exits non-zero when the rule does not match, and runs
nothing either way. A password prompt, a missing rule, or a rule that only
matches the older unscoped `nvidia-smi -pl` form is a refusal with no
mutation.

### What the run does

`vahtisiru --gpu-hardware-actuation-smoke --gpu-hardware-actuation-uuid <UUID>`
(the `actuation_smoke` module, driven through the same `GpuTarget`,
`NvmlActuator`, and `HardwareBridge` the supervisor uses):

0. **Process lock.** Takes the supervisor's single-instance lock
   (`/tmp/vahtisiru.lock`, plus the legacy-lock check) before any NVML call and
   holds it through cleanup. A running `vahtisiru` supervisor on the host makes
   the smoke refuse with no mutation: the two would share one brake target, and
   the smoke's release could lift a real brake mid-fault. **Stop the
   supervisor before dispatching.**
1. **Identity.** Resolves NVML index 0 and refuses unless its UUID equals the
   operator-supplied one.
2. **Pre-state.** Reads current and default limits through that UUID. Refuses
   unless both are readable, positive, and the current limit is at/above the
   default. A leftover relay brake (current ≈ 50 % of default) or a foreign
   sub-default cap is never adopted — the run names the recovery command and
   stops.
3. **Command plan.** Asks the actuator for the exact `nvidia-smi` argument
   vector it would run for both the brake target and the default, and refuses
   unless each pins `-i <UUID>` before a single `-pl <W>`. An unscoped
   `nvidia-smi -pl` plan cannot reach the hardware.
4. **Sudo.** The `sudo -l` check above, for both wattages.
5. **Apply.** Prints the operator recovery command, then calls the production
   `apply_emergency_brake`. Reads the limit back and requires it to be within
   ±2 W of the target.
6. **Healthy streak.** Acquires real NVML samples at the step interval and
   classifies each under the policy resolved against the device default
   (exactly as the supervisor does). Requires `--safety-release-ok-streak`
   consecutive real `Ok` samples (default 3) before releasing.
7. **Release.** Calls the production `release_emergency_brake`, reads the
   limit back, and requires it to be within ±2 W of the default.

The log records the trusted commit (`GITHUB_SHA`), UUID, pre-test
current/default limits, apply target, post-apply observation, each streak
sample, post-release observation, and the cleanup result. The final
`GPU actuation smoke PASSED` line is printed only after the post-release limit
has been **observed** at the default; a run that cannot observe that never
claims success. The workflow also prints the limits through `nvidia-smi`
before and (with `if: always()`) after the step, so a timed-out or cancelled
job still leaves evidence of the device state.

### Failure, cleanup, and operator recovery

Any failure after the brake is applied — unexpected post-apply limit, an
unhealthy sample during the streak, a release error, or a post-release limit
that is not the default — triggers one cleanup release through the same
production actuator, followed by a read-back. That actuator deliberately
refuses to restore a limit it does not recognise as the relay's own brake
(a foreign or unexpected wattage means something else set it), so in those
cases cleanup reports failure and hands the decision to the operator rather
than forcing the default. The result reports one of:

- `cleanup restored default` — the device was observed back at default.
- `CLEANUP FAILED` — the device could not be confirmed at default. The job
  fails and prints the recovery command.

If a run fails with `CLEANUP FAILED`, is cancelled, or times out, check the
`Record post-test power limits` step (or run `nvidia-smi -i <UUID>
--query-gpu=power.limit,power.default_limit --format=csv` on the host). If
`power.limit` is below `power.default_limit`, restore it with the same
UUID-scoped form the relay uses (the sudoers rule above already permits it):

```sh
sudo -n nvidia-smi -i <UUID> -pl <power.default_limit>
nvidia-smi -i <UUID> --query-gpu=power.limit,power.default_limit --format=csv
```

Do not re-run the actuation smoke until the read-back shows the default; the
pre-state check will refuse a leftover brake anyway. A reboot also restores
the driver default.

### Local invocation

Only on a trusted Linux host you own, with the sudoers rule above and no
`vahtisiru` supervisor running:

```sh
nvidia-smi -L
cargo run --release --locked --bin vahtisiru -- \
  --gpu-hardware-actuation-smoke \
  --gpu-hardware-actuation-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

`--force-software-only` is refused in this mode, and the two flags cannot be
combined with `--gpu-hardware-smoke`.

### Why software-only CI stays authoritative

Every refusal, observation, and cleanup branch of the actuation sequence is a
pure function of a backend trait and is exercised in `cargo test` with a
scripted fake (`src/actuation_smoke.rs`), with no GPU, `sudo`, or subprocess.
The hardware lane proves that one specific device, driver, and sudoers rule
behave as that logic assumes; it cannot prove portability, and a green
hardware run is not a substitute for the hosted jobs.
