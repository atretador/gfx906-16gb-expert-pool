# gfx906-16gb-expert-pool

An unofficial fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds a persistent
expert pool (expert cache) for MoE models whose experts are offloaded with `--n-cpu-moe`, and makes
that cache actually usable on a 16 GB gfx906 card (Radeon Pro VII / MI50 / MI60).

Experimental. Locally validated on one card and one model. Not affiliated with, endorsed by, or
submitted to the ggml-org project.

## Why this fork exists

- Upstream llama.cpp has **no expert cache**. `--n-cpu-moe N` moves routed experts to the CPU, but
  every token then re-copies the experts it needs over PCIe. At 128K context on a 16 GB card that
  costs roughly 40% of decode throughput.
- A persistent expert pool exists in a separate fork (see Credits), but its admission budget reserves
  the **full configured context KV** plus a hard 3 GiB cap. On a 16 GB device at 128K that admits
  **0 of 144** offloaded expert tensors, so the feature silently does nothing.
- This fork keeps the pool mechanism and replaces the budget with a rail-based ledger, adds a
  weighted per-layer planner, adds telemetry, and makes the pool fail closed when speculative
  draft/MTP decoding is selected.

The pool is not the win by itself. The admission fix is what turns "0 tensors fit" into 66 slots.

## Credits

- Expert pool implementation (`--moe-expert-cache`, pool allocator, LRU slots, graph and expert-ID
  remap): originally written by **zhanghewei** in
  [`memoriaru/llama.cpp`](https://github.com/memoriaru/llama.cpp) branch `moe-expert-pool` at
  `555d1ec9daa4564cb06adb39c469acfdcd1d1e93`. Carried here as the first commit, authorship
  preserved. See PROVENANCE.md.
- llama.cpp and ggml: the ggml authors, MIT. Includes the upstream `qwen4exp` architecture.
- Validated model: [AtomicChat/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF),
  `AD-3.84bpw-IQ4_XS-M64`.
- ROCm/gfx906 build base: [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906).

## Build (gfx906 / ROCm)

`docker/Dockerfile` builds against a pinned ROCm/gfx906 base image and compiles the fork with
`-DGGML_HIP=ON -DGPU_TARGETS=gfx906`:

```sh
docker build -f docker/Dockerfile -t local/llama.cpp-gfx906:expert-pool .
```

The image records the source commit and a source-tree digest in `/etc/llama.cpp-provenance`, so a
built image can be tied back to an exact tree.

## Usage

```sh
llama-server -m <model.gguf> --n-cpu-moe 48 -ngl 99 -c 131072 \
  -ctk q8_0 -ctv q8_0 -fa on --moe-expert-cache 66
```

- `--moe-expert-cache N` / `-mec N`: requested pool slots **per offloaded expert weight tensor**
  (not a global count, not tokens). Each tensor gets its own pool. `0` disables the pool.
- `-ngl`/`--n-cpu-moe` decide how many layers are offloaded. The pool only applies to offloaded MoE
  tensors.
- Decode-shaped operations (few tokens per step) use the pool. Large prefill batches bypass it and
  run the stock host-copy path, which is intentional.

Environment knobs:

| Variable | Meaning |
| --- | --- |
| `LLAMA_MOE_POOL_RAIL_MIB` | VRAM held back from the pool for compute buffers and later allocations. Default 2879 MiB, hard floor 1024 MiB. Lower it to admit more slots. |
| `LLAMA_MOE_POOL_CAP_MIB` | Optional absolute cap on total pool bytes. Unset means no cap. `0` is rejected. |
| `LLAMA_MOE_POOL_PROFILE` | Optional text file with one `blk.<layer_index> <weight>` line per layer for weighted slot allocation. Missing or malformed entries fall back to weight 1.0. |
| `GGML_MOE_POOL_STATS` | Set to any value for per-pool hit/miss detail on top of the aggregate reports. |

Telemetry, one line per event type, visible at the default log level:

```
expert pool status=enabled reason=admitted requested_slots=66 actual_slots=66 pool_count=144 bytes=5292195840 cap=7793213440 ceiling=15015608320 limited_by=slots alloc=uniform profile=none slots_min=66 slots_med=66 slots_max=66 total_slots=9504
expert pool first-use=active reason=distinct-experts-within-slots
expert pool runtime reason=shutdown hits=... misses=... evictions=... hit_rate=... copy_bytes=...
```

`limited_by` says what actually constrained admission (`slots`, `rail` or `cap`), and `copy_bytes` is
the number of bytes copied on misses. Watch `copy_bytes`, not the hit-rate percentage: a slot moved
to a cheaper tensor raises the hit count while increasing bytes moved.

## Measured results

MI50 16 GB, gfx906/ROCm, `Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64`, 128K context, Q8_0 K/V,
`--n-cpu-moe 48`, expert-pool rail 2048 MiB, flash attention on, speculative decoding off, 18 threads,
batch 2048 / ubatch 512. Measured on commit `e815ead0d` (merge of upstream master `58367713a`,
2026-09-21), image `local/llama.cpp-gfx906:fork-128k-20260921`.

Each configuration loads the model and then issues two fixed 700-token generations at temperature 0 and
seed 42; the first is reported as cold (empty pool), the second as warm. Decode figures are the single-run
`timings.predicted_per_second` of each generation, not medians.

| configuration | decode cold / warm | VRAM peak | notes |
| --- | --- | --- | --- |
| `--moe-expert-cache 0` | 12.83 / 13.61 t/s | 9.79 GiB | stock CPU MoE path, 6.19 GiB free |
| `--moe-expert-cache 66` | 18.59 / 19.00 t/s | 14.93 GiB | shipping profile, 66 slots, 144 pools, 72.9% hits, 1.05 GiB free |
| `--moe-expert-cache 80` | 18.80 / 20.40 t/s | 15.84 GiB | 80 slots admitted, 77.1% hits, 0.15 GiB free |

Cache-off and cache-on outputs are not bit-identical (see below). Both pool configurations produced the
same output hash, and the pool path is deterministic across runs at a fixed seed.

### Previous build against this build

| configuration | previous build | current build |
| --- | --- | --- |
| `--moe-expert-cache 0` | 11.76 t/s | 12.83 / 13.61 t/s |
| `--moe-expert-cache 66` | 16.90 / 17.60 t/s, ~69% hits | 18.59 / 19.00 t/s, 72.9% hits |
| `--moe-expert-cache 80` | 16.39 t/s, 40 slots admitted, 61.8% hits | 18.80 / 20.40 t/s, 80 slots admitted, 77.1% hits |

- Previous build: image `local/llama.cpp-gfx906:fork-128k-20260918`, base commit `972d2313b`
  (upstream `b11028`), 2026-09-17.
- Current build: image `local/llama.cpp-gfx906:fork-128k-20260921`, merge commit `e815ead0d`
  (upstream master `58367713a`), 2026-09-21.

This is not a controlled A/B. The previous figures came from an earlier harness with a different fixed
prompt (about 78 tokens against 66 now), and the cache-off row moved too (11.76 to 12.83 t/s) even
though this fork does not touch that path. Part of the gain is upstream and harness drift, not the
pool. Only the current-build table above is reproducible with `bench_expert_pool.sh`.

The 80-slot row is not comparable in kind. The previous build admitted only 40 slots because its rail
and cap bound the request, so that figure measures a 40-slot pool against today's 80-slot pool.

Supporting measurements, with their limits:

- VRAM ceiling: 66 slots peaks at 14.93 GiB, 80 slots at 15.84 GiB with only 148 MB free. The 80-slot
  configuration completed this benchmark without an allocation failure, but at that headroom there is no
  margin for a longer context or a second client, and an earlier build showed microstutter at a
  comparable peak. Treat 66 slots as the validated profile and 80 as experimental. VRAM peak depends on
  context and generation length, so validate with your own workload.
- Cold versus warm is a small effect at this profile: once prefill has populated the pool, the first
  generation runs within about 2% of the warm figure (18.59 versus 19.00 t/s at 66 slots).
- The benefit is routing-locality dependent. A batch-1/ubatch-1 churn workload at 46.9% hits ran
  1.8x slower than cache-off, because every decode token drove 144 synchronous pool updates.
- A fixed-token perplexity comparison measured 1.0987 (cache off) versus 1.0480 (cache on). **This is
  not a quality improvement claim.** The two configurations run MoE arithmetic on different backends,
  and the sample was far too small to bound quality; treat the numbers as evidence of no measured
  degradation only.
- Enabling the pool is **not bit-identical** to cache-off: it moves MoE compute from the CPU backend
  to the accelerator, and those backends already differ in arithmetic for every quant type. Pool
  parity is bit-exact against the stock host-copy path when compared on the same backend.

Donor-reported numbers are not reproduced here and should not be read as MI50 results: the donor
measured +84% at 64 slots on an RTX 4090, a third party measured 2.8x slower at 16 slots and +54% at
160 slots on a PCIe 3.0 system, and an RX 9070 Vulkan report recorded regressions at 16/32/64 slots.

What this does not show: no contexts above 128K (admission can fall to zero), no vision workloads, no
MTP/speculative decoding, one model, one card.

### Reported on other models

Reported by the fork maintainer on different models and profiles. These do not use the protocol above
and are not reproduced here; the profiles differ in offload count, slot count, context and parallelism.

- [SC117/Ling-3.0-flash-abliterated-APEX-GGUF](https://huggingface.co/SC117/Ling-3.0-flash-abliterated-APEX-GGUF)
  compact: CPU layers 36 -> 41, expert cache 76, 200K context, Q8_0, about 22 tk/s -> 26 tk/s.
- [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) IQ4_NL_XL:
  CPU layers 22 -> 40, expert cache 124, 256K context, Q8_0, 33.7 tk/s -> 44.6 tk/s flat.
- Same model, expert cache 96, 303K context, Q8_0, parallel 2 (151552 per stream): single stream
  33.7 tk/s -> 44.6 tk/s flat, two-stream aggregate about 40 tk/s -> 60 tk/s.

## Updating from upstream

```sh
git fetch upstream
git rebase upstream/master
```

Conflicts are expected in `README.md` (this file replaces upstream's), `common/arg.cpp`,
`src/llama-context.cpp`, `src/llama-graph.cpp` and `ggml/src/ggml-backend.cpp`. After rebasing,
rebuild for gfx906 and re-run `test-expert-pool`.

## License

MIT, same as upstream llama.cpp.
