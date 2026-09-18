# Provenance

How this fork was assembled, what came from where, and what was deliberately left out.

## Base

Upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at
`972d2313bc0bf0a45f634f77d95c9fb03aeab12c` (tag `b11028`). `master` in this fork is a pristine
mirror of that commit.

The branch `gfx906-expert-pool` contains exactly two commits on top of it.

## Commit 1: the expert pool (donor work)

- Source: [`memoriaru/llama.cpp`](https://github.com/memoriaru/llama.cpp) branch `moe-expert-pool`
  at `555d1ec9daa4564cb06adb39c469acfdcd1d1e93`.
- Author: `zhanghewei <zhanghewei@dataeye.com>`. Note: this commit identity is not linked to any
  GitHub account (the API reports a null author login); the fork itself is hosted under the
  `memoriaru` account. Author name and email are preserved verbatim in the commit.
- Shared ancestor with this base: `9723942adc518b43c4b95dc4dce6906903eb5e09`.
- Donor net effect (`9723942..555d1ec`): 18 files, +1629/-4. It carries a long development history
  including three reverts, so it is landed here as one commit holding the net effect rather than as
  35 cherry-picks that would need conflicts resolved against states that were later undone.
- Carried here (13 files): `common/arg.cpp`, `common/common.cpp`, `common/common.h`,
  `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend.cpp`, `include/llama.h`,
  `src/llama-context.cpp`, `src/llama-context.h`, `src/llama-cparams.h`, `src/llama-graph.cpp`,
  `src/llama-graph.h`, `tests/CMakeLists.txt`, `tests/test-expert-pool.cpp`.
- Not carried from the donor (5 files): `docs/rfc-moe-expert-pool.md`,
  `docs/expert-pool-windows-test.md`, `docs/expert-pool-windows-test.en.md`,
  `examples/simple/simple.cpp`, `tools/llama-bench/llama-bench.cpp`. These are documentation and
  driver changes, not part of the pool mechanism this fork builds and tests.
- Kept unchanged by us even though unused after commit 2: the donor's `cparams.type_k` /
  `cparams.type_v` fields. Their only consumer was the donor's context-KV budget estimate, which
  commit 2 replaces. They are left in place to keep the second commit focused.

## Commit 2: what this fork changes

Subject: `moe: rail-based pool admission, weighted planning and telemetry for the expert pool`.

1. **Admission.** The donor budget reserves the full configured context KV plus a hard 3 GiB cap.
   On a 16 GB card at 128K context that admits 0 of 144 offloaded expert tensors, so the feature is
   inert exactly where it is needed. Replaced with a rail ledger:
   `budget = min(bound, dev_total - rail - used_now - future_reserve)`, where target KV is never
   subtracted twice, `LLAMA_MOE_POOL_RAIL_MIB` sets the rail (default 2879 MiB, floor 1024 MiB) and
   `LLAMA_MOE_POOL_CAP_MIB` is an optional absolute cap. `limited_by=slots|rail|cap` reports which
   constraint bound admission.
2. **Planning.** The donor assigns the same slot count to every tensor in model order. Replaced with
   a checked-byte uniform planner plus an optional weighted per-layer planner driven by
   `LLAMA_MOE_POOL_PROFILE`. Admission is all-or-none; every tensor keeps a floor at its own routing
   width; per-tensor slots are capped at `min(n_expert - 1, ceil(1.75 * requested))`. Equal weights
   reproduce the uniform result byte for byte.
3. **Allocation verification.** Each pool is checked against the actual returned buffer sizes, and
   any failure or overrun rolls the whole reservation back rather than leaving a partial set.
4. **Telemetry.** `status`, `first-use` and `runtime` markers plus `copy_bytes`, so the pool reports
   what it did and how many bytes it copied on misses.
5. **Speculative decoding.** Selecting a draft/MTP speculative type forces expert caching off before
   the target context is created, with a visible reason. The pool requires a measured draft reserve
   to be safe, and this fork does not implement one.

## Not carried from other sources

- Upstream PR `#27841` (gfx906 GCN MMQ config): merged upstream on 2026-09-12 and therefore already
  present in the base commit.
- Upstream PR `#28616` (branch-free HIP SWAR intrinsics): still open, and measured at +0.65% decode
  on this hardware with byte-identical output, which is within noise. Not carried.
- The `danielhanchen/llama.cpp` `qwen4exp/mtp` chain: the fork deliberately excludes MTP, so a
  fork-of-fork base would only make future upstream rebases harder.

## Build provenance

Each image records `source_commit` and `source_tree_sha256` in `/etc/llama.cpp-provenance`. Verified
pairs from the development machine:

| build | source_commit | source_tree_sha256 | image digest |
| --- | --- | --- | --- |
| fork, upstream `b11028` base | `972d2313bc0bf0a45f634f77d95c9fb03aeab12c` | `782ead5ff8588f9a5e52e8e1e592f2bb495c29cb07d2a1d7886a3d217fa3fe2a` | `sha256:b1fafa1e4b1ac880fa8c40e865304bacf197835efd53077b1dfdb901dcd92545` |
| earlier checkpoint, `d1a92352` base | `d1a92352cbd417fd840b4e765c0b82f5fe3d1d89` | `4349febeb5f83cfd4c308d65eda0a36325ebb6b1dac5882fcc5d0a9d3c269a60` | `sha256:7e5786d288668bfa56be4c1b65223d3ce49a56a6230c2bcedca41a27f855b799` |

The earlier checkpoint is the validated build the fork was compared against. Its base is 368 upstream
commits older, so model output is not bit-identical between the two; the fork measured 5.7% faster
cold and 6.3% faster warm at 66 slots with a 0.41 GiB lower VRAM peak.

## Tooling disclosure

The port, the budget and planner work, the telemetry and this documentation were developed with AI
agent assistance under human direction and review. Verify anything you depend on against the source
before trusting it.
