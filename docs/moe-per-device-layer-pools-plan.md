# Plan B: Per-device layer-range expert pools (gfx906 expert-pool fork)

Status: reviewed. The independent review (ora-3) produced amendments A-E, all folded in below.
Decisions locked with the user:
- No new layer-range/split flag. Fixed layer ownership stays `-sm layer` + `-ts` (and `-ot` for exact
  per-layer pinning). The per-GPU cache range is `owned_layers(GPU i) INTERSECT offloaded_layers`.
- `-mcr` / `--moe-cache-range` is OPTIONAL but AUTHORITATIVE for cache membership. Any mismatch is a
  HARD ERROR with a clear logged message; never clip silently.
- Per-GPU slots `-mec N0,N1,...` (one value = all GPUs, legacy behavior).
- This is a capacity play, not a latency play.

Read together with `OPTIMIZATION-NOTES.md`, `README.md`, and `docs/multi-gpu.md`.

## 0. Status and verification

- Repo `/home/caf/DATA/Projects/gfx906-16gb-expert-pool`, branch `gfx906-expert-pool`, base commit
  `c25bdd9f67dc3ac1fb2cbcab21ad03f55c6c7450` (Plan A reverted; only `.gitignore` modified).
- Every anchor below was re-read against this tree. Pass numbering, buffer-usage enum, and the remap
  placement argument were verified in `ggml/src/ggml-backend.cpp`.

Anchor facts confirmed:

- `ggml_backend_sched_split_graph`: pass 1 at `:1142`, pass 2 at `:1179`, pass 3 at `:1259`, pass 4 at
  `:1320`, pass 5 at `:1352`.
- Pass 3 upgrade requires identical buft pointers: `sched->bufts[b] == sched->bufts[*node_backend_id]`
  (`:1298`). Per-device accelerator bufts are distinct pointers, so pass 3 cannot move a node across
  devices.
- `ggml_backend_sched_backend_from_buffer` at `:943-963`; the WEIGHTS-src branch in
  `ggml_backend_sched_backend_id_from_cur` is `:1022-1035`.
- Buffer usage enum `GGML_BACKEND_BUFFER_USAGE_{ANY,WEIGHTS,COMPUTE}` at
  `ggml/include/ggml-backend.h:50-52`; `set_usage`/`get_usage` at `:65-66`.
- `table_dev_buf` allocated at `ggml/src/ggml-backend.cpp:2654` with no `set_usage`, unlike `pool_buf`
  at `:2642`.
- `llama_expert_pool::table` is the DEVICE table (`table_dev`): registration returns it via
  `map_table` at `ggml-backend.cpp:2747-2748`, and the caller stores it at `src/llama-context.cpp:887-904`.
- `ggml_backend_sched_update_expert_pool` writes `ep.table_dev` at `:2190`; dispatch anchors on
  `ep.table_dev == node->src[0]` at `:1803` and calls with `split_backend` at `:1812`/`:1835`.

## 1. Overview and motivation

This is a capacity play, not a latency play. The single MI50 profile is VRAM-capped: 66 slots =
72.9% hit / 18.75 t/s / 0.888 GiB free; 80 slots = 77.1% / 20.32 t/s / 0.065 GiB free. The +1.4 t/s
from 66 to 80 slots is capacity, not code. The measured decode bottleneck is host round trips
(48.2 ms of 53.3 ms blocked), so the only mechanism by which a second GPU can help decode is
aggregate slot capacity -> higher hit rate -> fewer miss copies -> less blocked host time.

Plan B lets each GPU own a contiguous layer range and pool only that range's experts. Every pool
stays single-device; no demotion, no cross-device expert movement. The `n_accel > 1` guard at
`src/llama-context.cpp:709-724` exists because the old design pooled all experts on one backend while
layers ran elsewhere. Under Plan B a pool's owner is derived from its own layer, so pool and consumer
are co-located by construction; the guard's objection does not apply.

```
cache_range(GPU i) = owned_layers(GPU i) INTERSECT offloaded_layers
```

where `offloaded_layers` is what `--n-cpu-moe N` (layers `0..N-1`, N exclusive), `-cmoe`, or `-ot`
puts on the host. Every per-GPU cache range is a sub-range of the offloaded set.

## 2. Config / UX

### 2.1 Layer ownership and offload set

Ownership comes from `llama_model::load_tensors` / `get_layer_buft_list` (`src/llama-model.cpp:1591-1601`)
and `pimpl->dev_layer[il]` (`:1607-1614`), exposed as `model.dev_layer(il)`
(`src/llama-model.cpp:2248-2249`). The split denominator is `act_gpu_layers = min(n_gpu_layers,
n_layer_all + 1)`, which includes the output layer (`:1590`, `:1614`). For a 48-layer model with
`-ngl 99` the denominator is 49, not 48. This one-off is the source of most user confusion and is
called out in the examples.

Offload set comes from:

- `--n-cpu-moe N` -> layers `0..N-1` (`common/common.h:1213-1220`, `common/arg.cpp:2763-2772`);
- `-cmoe` -> all layers (`common/common.h:1209-1211`, `common/arg.cpp:2756-2762`);
- `-ot`/`--override-tensor` -> regex-selected tensors to `=CPU` (`common/arg.cpp:252-284`,
  `:2750-2755`). CPU buft name is `CPU` (`ggml/src/ggml-cpu/ggml-cpu.cpp:113`).

No new layer-range/split flag is introduced for ownership.

### 2.2 Flag grammar

| flag | grammar | meaning |
| --- | --- | --- |
| `-sm` / `--split-mode` | `layer` (required when `n_devices() > 1`) | fixed layer ownership |
| `-ts` / `--tensor-split` | `N0,N1,...` ratios | per-GPU layer ranges |
| `-ncmoe` / `--n-cpu-moe` | `N` | offload expert prefix `0..N-1` |
| `-cmoe` / `--cpu-moe` | - | offload all experts |
| `-ot` / `--override-tensor` | `pattern=CPU,...` | exact per-layer/tensor offload |
| `-mcr` / `--moe-cache-range` | `"lo0-hi0,lo1-hi1,..."` | optional, authoritative per-GPU cached-layer ranges |
| `-mec` / `--moe-expert-cache` | `N` or `N0,N1,...` | per-tensor slots; one value = all GPUs (legacy), else per GPU in device order |
| `-mpf` / `--moe-pool-fallback` | `PCT` | per-device warmth gate |

Semantics:

- `-mcr` entries are in `model.devices` order; entry `i` is GPU `i`. An empty entry (`-`) means that
  GPU caches nothing. When `-mcr` is given it is authoritative for cache membership. It may not be
  combined with `--moe-expert-cache 0`.
- `-mec` one value keeps today's behavior byte for byte. Multiple values must number exactly
  `n_devices()`; a mismatch is a hard error. A multi-value `-mec` at `n_devices() == 1` is a hard error.
- `-mcr` is ignored with a warning at `n_devices() == 1`; a multi-value `-mec` at `n_devices() == 1` is
  a hard error (single-GPU path is the legacy path, see 3.4).
- `--fit` may overwrite `-ts`/`-ngl` and synthesize its own per-layer overrides
  (`common/fit.cpp:564-586`); pass `-fit off` for the manual recipes below.

### 2.3 Worked examples

Example 1: 48-layer model, 2xMI50 16 GB, target ranges 0-9 / 10-19.

```
llama-server -m model.gguf \
  -fit off -sm layer -ngl 99 \
  -ts 10,39 \
  --n-cpu-moe 20 \
  -mcr "0-9,10-19" \
  -mec 40,40
```

Math: `act_gpu_layers = 49` (48 layers + output). `-ts 10,39` normalizes to cumulative `[10/49, 1.0]`.
`upper_bound` over `(il)/49` gives GPU0 = layers `0..9`, GPU1 = layers `10..48`. Offloaded = `0..19`.
Cache ranges: GPU0 `0..9` (10 layers), GPU1 `10..19` (10 layers). Exactly the intent.

Note: the commonly written `-ts 10,38` gives cumulative `[10/48, 1.0]`, so `il=10` maps to GPU0 and
GPU0 owns `0..10`; cache ranges become GPU0 `0..10`, GPU1 `11..19`. If you write `-mcr "0-9,10-19"`
with `-ts 10,38` the program hard-errors: layer 10 is owned by GPU0 but declared in GPU1's entry.
Use `-ts 10,39` (or `-ts 10,38` plus `-mcr "0-10,11-19"`).

Example 2: silent degeneration, `-ts 24,24`.

```
-fit off -sm layer -ngl 99 -ts 24,24 --n-cpu-moe 20 -mec 80
```

`-ts 24,24` -> cumulative `[0.5, 1.0]` -> GPU0 owns `0..24`, GPU1 owns `25..48`. Offloaded `0..19`
all fall inside GPU0. Result: every pool lands on GPU0, GPU1 hosts zero pools, and the design silently
degenerates to single-device. This is exactly the trap Plan B removes the guard for. `-mcr` makes it
explicit; `-mcr "0-19,"` declares GPU0 `0..19`, GPU1 empty. To actually distribute, offload layers on
both sides: `-cmoe` (GPU0 pools `0..24`, GPU1 pools `25..47`) or `-ot` covering layers in both ranges.

Example 3: N GPUs (4xMI50).

```
-fit off -sm layer -ngl 99 -ts 1,1,1,1 -cmoe -mec 40,40,40,40
```

Default memory-proportional split already divides layers equally for identical cards. `-cmoe` makes
every layer a candidate, so each GPU pools its own range. `-mcr` optional for explicit ranges.

### 2.4 Full hard-error list

When `-mcr` is given, `init_expert_pools` throws `std::runtime_error` (clear logged message, no silent
clipping) if any of:

1. Number of entries != `model.n_devices()`.
2. Malformed entry (not `lo-hi`, `lo > hi`, negative, or `hi >= n_layer_all`).
3. Entry `i` names layer `il` whose `model.dev_layer(il)` is not GPU `i` (in `model.devices` order).
   Covers "range crosses a split boundary" and "layer owned by another GPU".
4. Entry `i` names layer `il` that is not host-offloaded (no host-backed expert tensor). Covers `-mcr`
   naming a layer outside the offloaded set.
5. Two ranges overlap (within or across entries).
6. Two consecutive non-empty entries leave a gap (`hi_i + 1 != lo_{i+1}`).
7. The union of ranges does not equal the offloaded GPU-owned layer set (either an offloaded GPU-owned
   layer is omitted, or a declared layer is not in the set). To cache fewer layers, narrow the
   offloaded set with `-ot`/`--n-cpu-moe`; `-mcr` declares the partition of that set. LOCKED: this is a
   strict, authoritative partition (user decision: "hard error on log").
8. `-mcr` given while `--moe-expert-cache` is 0.

When `-mec` has multiple values: count != `n_devices()` is a hard error. Multiple values at
`n_devices() == 1` is a hard error.

## 3. Exact code-change list

All anchors are current lines. Rough total ~430-470 LOC across 10 files.

### 3.1 ggml core (the only ggml-core change)

A. Pin the remap (required invariant).

- `ggml/src/ggml-backend.cpp:2654`: after allocating `table_dev_buf`, add
  `ggml_backend_buffer_set_usage(table_dev_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);`.
  This makes the WEIGHTS-src branch (`:1022-1035`) place the remap `GET_ROWS` on the pool owner in
  pass 1 instead of relying on pass-2 expansion. Without it, pass 5 can create a scheduler copy of
  `table_dev` that the update at `:2190` never writes (stale table). +2 LOC.
- Debug check at the dispatch anchor `ggml/src/ggml-backend.cpp:1798-1824`: assert the remap split
  backend equals the pool backend, e.g. `GGML_ASSERT(split_backend == ep.backend);`. +5 LOC.
- Add `int backend_id` to `struct ggml_backend_sched_expert_pool` (`:780-814`) and set it in
  registration (`:2718-2726`), so the assert and stats can use it. +4 LOC.

Per-device stats:

- `ggml/include/ggml-backend.h:445-449`: declare
  `ggml_backend_sched_get_expert_pool_stats_by_backend(sched, backend_id, hits, misses, copy_bytes)`.
- `ggml/src/ggml-backend.cpp:2794-2822`: implement by filtering
  `ep.backend == sched->backends[backend_id]`. +30 LOC.
- Optional: per-device lines in `ggml_backend_sched_report_expert_pool_stats` (`:2048-2068`). +15 LOC.

### 3.2 Public API and cparams

- `include/llama.h:375-379`: add `const int32_t * expert_cache_slots_per_device;` (size
  `llama_max_devices()`, NULL = scalar for all) and `const char * moe_cache_range;` (NULL = derive).
  +8 LOC.
- `src/llama-context.cpp:4130-4131`: zero-init the two new fields. +2 LOC.
- `src/llama-cparams.h:24-29`: add `std::vector<int32_t> expert_cache_slots_per_device;` and
  `std::string moe_cache_range;`. +6 LOC.
- `src/llama-context.cpp:281-282`: copy the new params into cparams (vector from pointer + count,
  string copy). +10 LOC.

### 3.3 Common CLI

- `common/common.h:477-478`: add `std::vector<int32_t> expert_cache_slots_per_device;` and
  `std::string moe_cache_range;`. +3 LOC.
- `common/arg.cpp:2773-2784`: change `-mec` to parse a comma list; set the scalar to the single value
  when one value is given; keep `LLAMA_ARG_MOE_EXPERT_CACHE`. +20 LOC.
- `common/arg.cpp`: add `-mcr` / `--moe-cache-range` (string), with
  `set_env("LLAMA_ARG_MOE_CACHE_RANGE")`. +25 LOC.
- `common/common.cpp:1670-1671`: pass the vector pointer and range string into `llama_context_params`;
  the `spec_draft` zeroing must clear the vector too. +10 LOC.

### 3.4 init_expert_pools (`src/llama-context.cpp:594-926`)

B. Strict no-op at `n_devices() == 1` (required invariant).

Keep an explicit legacy branch when `model.n_devices() == 1`:

- retain the exact `n_accel > 1` guard semantics (`:709-724`), i.e. count every non-CPU backend
  including ACCEL (the context can register BLAS ACCEL backends at `:349-359`), and
- retain the "first non-CPU backend" mapping (`:726-740`), assigning every host-backed candidate to
  it, including candidates whose layer is CPU-owned. This preserves current partial-offload behavior.

The per-device path is used only for `n_devices() > 1`. This makes the equivalence gate (section 7)
provable. If the desired behavior is instead to skip CPU-owned candidates at `n_devices() == 1`, that
is a deliberate change and must be documented as such; the recommendation is to preserve.

Concrete changes:

1. Guard `:709-724`: wrap in the single-device branch; for `n_devices() > 1` replace with: split-mode
   check (hard-disable reason `unsupported-split-mode` if `split_mode() != LLAMA_SPLIT_MODE_LAYER`),
   mixed-arch check (section 4), and `GGML_ASSERT(!cparams.pipeline_parallel)`. +45 LOC.
2. Owner mapping helper near `:726`: `backend_id_for_dev(ggml_backend_dev_t)` scanning `backends` for
   `ggml_backend_get_device(...) == dev`. At `n_devices() == 1` return the legacy first-non-CPU id.
   +15 LOC.
3. Candidate struct `:774-780`: add `int owner_backend_id; bool conflict;`. +2 LOC.
4. Inventory `:784-816`: compute owner from `model.dev_layer(il)` (skip/log CPU-owned in multi-GPU
   path). D: on pointer dedup (`:807-813`), if a duplicate tensor maps to a different owner, set
   `conflict = true` and skip admission for that tensor entirely (host path), not just warn. +20 LOC.
5. Ledger `:742-772`: refactor into a per-device helper `device_budget(dev, &budget, &ceiling)` using
   the same formula; call once per owner device with candidates. `future_reserve` and `rail` are
   per-device (section 6). +50 LOC.
6. Alignment `:840-841`: per device via
   `ggml_backend_sched_get_buffer_type(sched, ggml_backend_sched_get_backend(sched, d))`. +5 LOC.
7. Grouping/planning `:842-858`: group candidate indices by `owner_backend_id`; per device call
   `ggml_backend_expert_pool_plan_weighted` with that device's budget and alignment; weights remain
   `profile_weights[candidate.layer]`. A device with no budget admits nothing and logs. +55 LOC.
8. Registration `:871-907`: pass `candidate.owner_backend_id` (not the global `backend_id`) at
   `:890-891`; track `actual_device_by_dev[d]`; store `backend_id` in the `llama_expert_pool` entry at
   `:904`. +20 LOC.
9. Budget check `:909-915`: per device. Keep global rollback via
   `ggml_backend_sched_clear_expert_pools` on hard error. +10 LOC.
10. Status `:674-683`, `:860-869`, `:917-925`: emit per-device line
    `expert pool device=<name> pools=<n> slots=<n> bytes=<n> budget=<n> ceiling=<n> allocated=<n>` plus
    the aggregate line with `devices=<n>`. Bypass the one-shot `expert_pool_status_reported` for
    per-device lines. +30 LOC.
11. `-mcr` parse/validate: parse `cparams.moe_cache_range`, resolve per-GPU sets, run the section 2.4
    hard-error checks against `model.dev_layer(il)` and the candidate offload set. +60 LOC.

### 3.5 Per-device fallback (E)

- `src/llama-context.cpp:2055-2111`: query per device via the new by-backend stats; keep per-device
  last hits/misses and below-counters; on trip set `disabled` on only that device's pools. Set the
  global `expert_pool_diagnostic_state->disabled` master only when every pool-owning device has
  tripped. +50 LOC.
- `src/llama-context.h:304-314`: add per-device last hits/misses vectors. +5 LOC.

### 3.6 Graph substitution

- `src/llama-graph.h:111-114`: add `int backend_id = -1; bool disabled = false;` to `llama_expert_pool`.
- `src/llama-graph.cpp:1569-1599`: after the global disabled check and `find(w)`, skip substitution
  when `ep.disabled`. +5 LOC.

### 3.7 Tests

- `tests/test-expert-pool.cpp`: planner grouping tests (two disjoint groups, two budgets, per-group
  floors); `-mcr` parser/validator unit tests on synthetic ownership/offload sets (no second GPU
  required). No new test file. +40 LOC.

## 4. Invariants

- Pool/consumer co-location: for every registered pool,
  `ggml_backend_get_device(ggml_backend_sched_get_backend(sched, owner_backend_id)) == model.dev_layer(candidate.layer)`.
  Debug-assert at registration.
- Remap pinned to owner (A): `table_dev_buf` usage is `WEIGHTS`; `split_backend == ep.backend` at the
  dispatch anchor `ggml-backend.cpp:1798-1824`.
- Single-device no-op (B): at `n_devices() == 1` the legacy guard and legacy "first accelerator"
  mapping are used verbatim.
- Cap/rail explicit (C): see section 6.
- No shared pool across devices (D): a tensor deduped across layers with different owners is not
  admitted.
- Fallback isolation (E): a single device trip never sets the global master unless all pool-owning
  devices tripped.
- `GGML_ASSERT(!cparams.pipeline_parallel)` before registration.
- Mixed-arch hard stop: all pool-owning devices must share
  `ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev))`; otherwise disable with reason
  `mixed-arch`.
- Split-mode: `n_devices() > 1` requires `LLAMA_SPLIT_MODE_LAYER`.
- Cache range subset: every per-GPU cache range is a sub-range of the offloaded set.

## 5. Failure modes

- Remap lands off-owner (A): stale table -> wrong experts served. Mitigated by `WEIGHTS` usage plus
  the dispatch assert. Highest-severity bug in this design.
- Single-GPU regression (B): changing the guard or skipping CPU-owned candidates alters partial-offload
  behavior; also an ACCEL backend can make the old guard trip. Mitigated by the legacy branch and the
  equivalence gate.
- Cap/rail misinterpretation (C): a global cap treated as per-device multiplies the budget by N.
  Mitigated by explicit logging.
- Shared tensor across devices (D): one pool cannot serve two devices. Mitigated by disabling
  admission for the conflicted tensor.
- Fallback masking (E): a hot device hides a cold one. Mitigated by per-device trips and the
  all-devices global master.
- Per-device budget accounting: `actual_device` includes `pool_buf + table_dev_buf`
  (`ggml-backend.cpp:2662-2669`); sum per device, not globally.
- Silent degeneration: `--n-cpu-moe` prefix inside GPU0's range puts all pools on GPU0 (Example 2).
  Mitigated by per-device status and optional `-mcr`.
- Spec/MTP: pools are forced off when speculative draft/MTP is selected (`common/common.cpp:1663-1674`)
  and for MTP contexts (`src/llama-context.cpp:695-699`). A draft model loaded after target init can
  still consume VRAM promised to pools; document that margins must account for it.
- Batch-1 imbalance: per-device hit rate is a weighted average over disjoint layer sets; a device
  owning hotter layers sees more misses. Measure per-device.
- Activation copies: `pipeline_parallel == false` and `n_copies == 1`, so each layer boundary copies
  activations between devices. This is the main cost that can offset the capacity gain.
- N > 16: `GGML_SCHED_MAX_BACKENDS = 16` (`ggml-backend.cpp:767-769`).

## 6. Per-device budget and cap/rail semantics (C)

Per device `d`:

```
free_d, total_d = ggml_backend_dev_memory(owner_dev_d)
used_d          = total_d - free_d
ceiling_d       = total_d - rail_bytes
budget_d        = min(pool_bound_d, ceiling_d - used_d - future_reserve)
```

- `LLAMA_MOE_POOL_RAIL_MIB` is per device (each device keeps its own rail free). Default 2879 MiB,
  floor 1024 MiB (`src/llama-context.cpp:639-648`). Aggregate reserve is `N * rail`. This is an
  explicit semantic change from single-device and must be logged.
- `LLAMA_MOE_POOL_CAP_MIB` is a global total across all pool-owning devices, not per-device. It is
  divided across devices in proportion to each device's rail-derived budget (integer division,
  remainder to earlier devices); each device's planner budget is `min(rail_budget_d, cap_share_d)`.
  Unset = no cap. It is never multiplied by N. Log the per-device shares and the global total.
- `future_reserve` (`180158464` bytes) is applied per device.
- Validation: `total_d != 0`, `free_d <= total_d`, `rail_bytes < total_d`, no overflow; invalid -> that
  device admits nothing and logs `invalid-device-memory`.
- Admission is per device: a device with zero candidates or zero budget simply has no pools; other
  devices are unaffected. Hard registration errors roll back all pools (existing all-or-none contract).

## 7. Validation gates, in order

1. Single-GPU no-op equivalence (runnable on this host, the code-level gate). Build the patched tree
   and run the existing single-GPU harness (`bench_expert_pool.sh`) at `-mec 66` and `-mec 80` with
   `--n-cpu-moe 48`. Compare against the pre-patch build: pool count, per-pool slot counts, per-pool
   device bytes, admission decision, hit rate, and output hash `1ed2a1c6edd70759`. Must be identical.
   This proves the legacy branch is a true no-op.
2. `-mec` scalar equivalence. `-mec 80` and `-mec 80,80` (single GPU) must behave identically; a
   multi-value `-mec` on one GPU must hard-error.
3. `-mcr` validator matrix. Unit-test all section 2.4 errors on synthetic ownership/offload sets: non-
   offloaded layer, wrong-owner layer, overlap, gap, incomplete coverage, malformed range, count
   mismatch, and `-mec 0`. No second GPU needed.
4. Deferred (needs 2 GPUs): per-device pools populated. Per-device status lines show `pools > 0` and
   `bytes > 0` on both devices; per-candidate owner log matches `dev_layer`.
5. Deferred: per-device hit stats. Each device reports a plausible non-zero hit rate; no device is
   silent.
6. Deferred: output-hash equality. Pooled (`-mec 80,80`) vs non-pooled (`-mec 0`) on the identical
   2-GPU config must match. The single-GPU hash `1ed2a1c6edd70759` is a secondary check; layer split
   should be bitwise identical on same-arch, so a mismatch is a red flag.
7. Deferred: per-device VRAM margin. Record `min free` per device under prefill and 128K context; no
   OOM on either card.
8. Deferred: activation-copy cost. Measure layer-split throughput with pools on vs off to isolate the
   `n_copies == 1` boundary-copy overhead.

## 8. Honest expected value and ranking

Capacity, not latency. The mechanism is aggregate slots -> hit rate -> fewer miss copies -> less
blocked host time. It does not reduce per-copy host round trips and does not overlap copies with
compute.

Expected magnitude: the single-card 66 -> 80 slot delta is +4.2 points hit and +1.4 t/s, VRAM-capped.
Two cards pooling disjoint layer sets with ~80 slots each should land near the per-layer 80-slot curve:
a low-single-digit aggregate hit-rate gain, roughly +1 to +2 t/s, before subtracting per-boundary
activation copies. Net could be neutral. It is the only way past one card's capacity.

Ranking:

1. Single-device more slots - config only, proven +1.4 t/s, but exhausted by VRAM (65 MB free at 80
   slots).
2. Intra-layer copy overlap - orthogonal latency mechanism, bounded roughly 5-8 ms/token, medium
   effort, device-local and testable here. (Measured flat and reverted; see
   `docs/moe-intra-layer-copy-overlap-plan.md`.)
3. Per-device layer-range pools - larger capacity bet, but uncertain net benefit from activation copies
   and unvalidatable end-to-end on a one-GPU host. Do not start until gate 1 passes and a second GPU
   is available.

## 9. Effort and risk

Effort: ~430-470 LOC across 10 files. The ggml-core delta is tiny (one `set_usage` line plus an assert
and a stats accessor); the bulk is `src/llama-context.cpp` (guard/branch, ledger, grouping, `-mcr`,
fallback, status) and the common CLI parsing.

Risk:

- Highest: the remap pinning (A). Get it wrong and pools serve stale experts silently. The `WEIGHTS`
  usage plus the dispatch assert are mandatory.
- High: single-GPU equivalence (B). The legacy branch and the gate must land in the same change.
- Medium: `-mcr` semantics (item 7, locked to strict) and cap/rail semantics (C) are behavioral
  contracts.
- Medium: 2-GPU validation is impossible on this host; gates 4-8 are deferred and must be scheduled
  with hardware.
- Low: shared-tensor conflict (D), fallback isolation (E), N > 16 cap.
- Non-goals preserved: no cross-device expert movement, no demotion, no kernel/remap/P2P change, no
  change to the single-GPU path.
