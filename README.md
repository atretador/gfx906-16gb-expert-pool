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

## Build

The expert pool is backend-agnostic: it lives in `ggml/src/ggml-backend.cpp` and
`src/llama-context.cpp` and uses only generic backend/buffer APIs, so it is not tied to ROCm. It has
been built and exercised on ROCm/HIP (gfx906) and on Vulkan (gfx906 + gfx802). CUDA uses the same
code path but has not been exercised here.

Common to every backend: build with CMake + Ninja, `-DCMAKE_BUILD_TYPE=Release`, and
`-DLLAMA_CURL=OFF` if you do not need the Hugging Face downloader. The `llama-server` and
`test-expert-pool` targets are what you need.

### ROCm / HIP (gfx906)

`docker/Dockerfile` builds against a pinned ROCm/gfx906 base image and compiles the fork with
`-DGGML_HIP=ON -DGPU_TARGETS=gfx906`:

```sh
docker build -f docker/Dockerfile -t local/llama.cpp-gfx906:expert-pool .
```

Native build:

```sh
cmake -B build -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx906 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target llama-server test-expert-pool
```

The image records the source commit and a source-tree digest in `/etc/llama.cpp-provenance`, so a
built image can be tied back to an exact tree.

### Vulkan

Needs a Vulkan loader, a vendor ICD, and `glslc` (the shader compiler).

```sh
cmake -B build-vulkan -G Ninja -DGGML_VULKAN=ON -DGGML_HIP=OFF -DGGML_CUDA=OFF \
  -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --parallel --target llama-server test-expert-pool llama-bench
```

`llama-server --list-devices` prints the Vulkan devices and their free VRAM. Vulkan device order is
independent of the ROCm order and is often reversed, so always check `--list-devices` before writing
`--tensor-split`.

### CUDA

```sh
cmake -B build-cuda -G Ninja -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --parallel --target llama-server test-expert-pool
```

Add `-DCMAKE_CUDA_ARCHITECTURES=native` (or an explicit list) if the default architecture set does
not match your cards. Multi-GPU CUDA has not been exercised by the maintainers.

## Usage

```sh
llama-server -m <model.gguf> --n-cpu-moe 48 -ngl 99 -c 131072 \
  -ctk q8_0 -ctv q8_0 -fa on --moe-expert-cache 66
```

- `--moe-expert-cache N` / `-mec N`: requested pool slots **per offloaded expert weight tensor**
  (not a global count, not tokens). Each tensor gets its own pool. `0` disables the pool. With more
  than one device, a comma-separated list (`-mec N0,N1,...`) sets a per-device count in device order;
  a single value applies to every device. See the multi-GPU section below.
- `--moe-pool-fallback PCT` / `-mpf PCT`: disable the pool for the rest of the context once its
  decode hit rate stays below `PCT` percent after a warm-up window. The stock host-weight path
  still streams the used experts, just without the pool readbacks and table uploads. `0` (default)
  never falls back.
- `-ngl`/`--n-cpu-moe` decide how many layers are offloaded. The pool only applies to offloaded MoE
  tensors.
- Decode-shaped operations (few tokens per step) use the pool. Large prefill batches bypass it and
  run the stock host-copy path, which is intentional.

### Balancing fixed layers and the cache

`--moe-expert-cache` and `--n-cpu-moe` compete for the same VRAM, so tune them together instead of
maximizing either. An expert layer that is not offloaded stays resident on the GPU (a "fixed" layer).

The two knobs drive different phases, so you can go all in on one or the other, or aim for a balance:

- Prompt processing (pp) is driven by the fixed layers: more fixed layers (lower `--n-cpu-moe`) means
  faster pp, because a fixed layer never copies and a large prefill batch bypasses the pool and streams
  CPU-resident experts over PCIe.
- Decode (tg) is driven by the expert cache allocation: more slots per tensor (larger
  `--moe-expert-cache`) means a higher hit rate and faster tg, and it needs the VRAM that fixed layers
  would otherwise occupy.

The data below is for Qwen3.6-35B-A3B (40 MoE layers, so `--n-cpu-moe 40` offloads all of them) with
only these two knobs moved (maintainer reports, not a controlled benchmark):

| `--n-cpu-moe` | fixed layers | expert cache | prompt processing |
| --- | --- | --- | --- |
| 18 | 22 | 0 | about 800 t/s |
| 27 | 13 | 20 | about 445 t/s |
| 30 | 10 | 40 | about 400 t/s |
| 40 | 0 | 85 | about 300 t/s |

At `--n-cpu-moe 27` with expert cache 20 the pool hit rate was only 45.7%, so `--moe-pool-fallback 50`
disabled the pool and decode continued on the host path: when the hit rate is that low the cache is not
paying for its VRAM at that balance.

These absolute numbers are specific to this model. On Qwen3.8-Flash-Next (48 layers) prompt processing
is much lower, closer to about 80 t/s, so read the trend rather than the values.

Environment knobs:

| Variable | Meaning |
| --- | --- |
| `LLAMA_MOE_POOL_RAIL_MIB` | VRAM held back from the pool for compute buffers and later allocations. Default 2879 MiB, hard floor 1024 MiB. Lower it to admit more slots. |
| `LLAMA_MOE_POOL_CAP_MIB` | Optional absolute cap on total pool bytes. Unset means no cap. `0` is rejected. |
| `LLAMA_MOE_POOL_PROFILE` | Optional text file with one `blk.<layer_index> <weight>` line per layer for weighted slot allocation. Missing or malformed entries fall back to weight 1.0. |
| `GGML_MOE_POOL_STATS` | Set to any value for per-pool hit/miss detail on top of the aggregate reports. |
| `GGML_MOE_POOL_REPORT_INTERVAL` | Graph executions between aggregate expert-pool hit/miss reports. Default 1024. Set to 0 to disable the periodic report. |
| `LLAMA_ARG_MOE_POOL_FALLBACK` | `--moe-pool-fallback PCT` / `-mpf PCT`: disable the pool for the rest of the context when its decode hit rate stays below `PCT` percent. Default 0 (never). |

Telemetry, one line per event type, visible at the default log level:

```
expert pool status=enabled reason=admitted requested_slots=66 actual_slots=66 pool_count=144 bytes=5292195840 cap=7793213440 ceiling=15015608320 limited_by=slots alloc=uniform profile=none slots_min=66 slots_med=66 slots_max=66 total_slots=9504
expert pool first-use=active reason=distinct-experts-within-slots
expert pool runtime reason=shutdown hits=... misses=... evictions=... hit_rate=... copy_bytes=...
```

`limited_by` says what actually constrained admission (`slots`, `rail` or `cap`), and `copy_bytes` is
the number of bytes copied on misses. Watch `copy_bytes`, not the hit-rate percentage: a slot moved
to a cheaper tensor raises the hit count while increasing bytes moved.

## Multi-GPU: per-device expert pools

The single-device pool keeps every tensor's cache on one accelerator. In multi-GPU mode each GPU
pools only the offloaded expert tensors of the layers it owns:

```
cache_range(gpu i) = layers_owned_by(gpu i)  INTERSECT  host_offloaded_layers
```

Each pool still lives entirely on one device (the device that owns its layer), and the pooled
`MUL_MAT_ID` plus its expert-id remap run on that same device. Experts never move between GPUs:
there is no demotion and no cross-device pool copy. Layer ownership comes from the normal layer
split (`--split-mode layer` with `--tensor-split`); the offloaded set comes from `--cpu-moe` /
`--n-cpu-moe` / `--override-tensor`.

This is a capacity feature. The extra GPU buys more aggregate cache slots, which raises the hit rate
and removes host round trips. It does not reduce the per-copy cost, and the pipeline scheduler stays
off, so layer boundaries still copy activations between devices.

### Required parameters

- `--split-mode layer` (`-sm layer`): required whenever more than one device is present. `row` and
  `tensor` are rejected for the pool (a layer's experts would be split across devices, so a single
  owner device is undefined) and the pool disables with `reason=unsupported-split-mode`.
- A host offload so there are pool candidates: `--cpu-moe` (`-cmoe`, all layers), `--n-cpu-moe N`
  (first N layers) or `--override-tensor "pattern=CPU"` (exact ranges). Without one the pool
  disables with `reason=no-offloaded-experts`.
- `--moe-expert-cache N` (`-mec N`), or the per-device list `-mec N0,N1,...` with exactly
  `n_devices` entries (a count mismatch is a hard error). A single value applies to every device.
- `--tensor-split` (`-ts`) when the GPUs differ in VRAM, so the layer split matches capacity.

The pooled layer set must span the split boundary. `--n-cpu-moe` is a prefix, so a prefix that falls
entirely inside the first GPU's layer range puts every pool on that one GPU and silently degenerates
to single-device. `--cpu-moe` always spans. Confirm with the per-device status lines below.

### Optional parameters

- `--moe-cache-range "lo-hi,lo-hi,..."` (`-mcr`): declare the per-device cached layer ranges
  explicitly, in device order. Authoritative: the ranges must partition the offloaded GPU-owned
  layers, and a `-` entry means that device caches nothing. Any mismatch is a hard error, never a
  silent clip.
- `--moe-pool-fallback PCT` (`-mpf`): per-device warmth gate. A device whose decode hit rate stays
  below `PCT` is disabled on its own; the global disable only happens when every pool-owning device
  has tripped, so a hot device cannot mask a cold one.

### Worked examples

Two GPUs, homogeneous, all experts offloaded (simplest working config):

```sh
llama-server -m <model.gguf> --host 127.0.0.1 --port 8080 \
  -ngl 99 -sm layer -ts 1,1 -c 8192 -fa on \
  -cmoe -mec 80
```

Two GPUs, offload a layer prefix and pool it across both cards. 48-layer model, boundary at layer 10.
The split denominator is `n_layer_all + 1` (the output layer is included), so the boundary is
`-ts 10,39`, not `10,38`:

```sh
llama-server -m <model.gguf> -ngl 99 -sm layer -ts 10,39 --n-cpu-moe 20 \
  -c 8192 -fa on -mec 40 -mcr "0-9,10-19"
```

`-ts 10,38` would make layer 10 GPU0-owned, so `-mcr "0-9,10-19"` would hard-error. Use `10,39`, or
declare the ranges the split actually produces.

Two GPUs, unequal VRAM, explicit per-device slot counts. GPU0 gets a small cache, GPU1 a large one:

```sh
llama-server -m <model.gguf> -ngl 99 -sm layer -ts 1,3 -cmoe \
  -c 8192 -fa on -mec 24,72
```

Three GPUs, different VRAM (10 GB + 8 GB + 8 GB). A single `-mec` value applies to every device:

```sh
llama-server -m <model.gguf> -ngl 99 -sm layer -ts 10,8,8 -cmoe \
  -c 8192 -fa on -mec 48
```

Same box with a per-device slot list instead, sized to each card:

```sh
llama-server -m <model.gguf> -ngl 99 -sm layer -ts 10,8,8 -cmoe \
  -c 8192 -fa on -mec 64,40,40
```

Three GPUs, homogeneous:

```sh
llama-server -m <model.gguf> -ngl 99 -sm layer -ts 1,1,1 -cmoe \
  -c 8192 -fa on -mec 64,64,64
```

Validated reference run (Vulkan, mixed pair, this fork). The Vulkan device order here is R9 380
first, so `-ts 1,15` gives the Pro VII the larger share. `-v` is needed for the per-device lines:

```sh
./build-vulkan/bin/llama-server -m Qwen3.6-35B-A3B-UD-IQ4_NL_XL.gguf \
  --host 127.0.0.1 --port 8899 -ngl 99 -c 2048 -b 512 -ub 128 \
  -sm layer -ts 1,15 --cpu-moe -mec 32 -mpf 0 --jinja -v
```

This produced 9 pools on Vulkan0 (R9 380) and 111 on Vulkan1 (Pro VII), with coherent output.

Notes for all examples:

- `-ts` is a ratio, not an exact layer count. Check the per-device status lines to see where the
  layers actually landed before relying on a `-mcr` partition.
- `-mcr` layer numbers are absolute layer indices; validate them against the split.
- Add `-v` to see the per-device pool lines; the server suppresses INFO by default.
- On 8 GB cards lower `LLAMA_MOE_POOL_RAIL_MIB` (for example to 1024); the 2879 MiB default is a large
  fraction of an 8 GB card.

### `--moe-cache-range` errors

`-mcr` is validated against real ownership and the offload set. All of the following are hard errors
with a logged message and a nonzero exit, never a silent clip:

- number of entries does not match `n_devices`;
- malformed entry, `lo > hi`, negative, or `hi` at or above the layer count;
- a layer in entry `i` is not owned by device `i` (for example `layer 3 in device 0's range is owned
  by device 1`);
- a layer is not host-offloaded;
- overlapping ranges, or a gap between two non-empty ranges;
- the union does not cover the offloaded GPU-owned layers;
- `-mcr` combined with an expert cache of `0`.

### Telemetry

The per-device lines are INFO level; run with `-v`, because the server suppresses INFO by default:

```
expert pool device=Vulkan0 pools=9 slots=288 bytes=165164544 budget=977567744 ceiling=1276116992 allocated=165164544
expert pool device=Vulkan1 pools=111 slots=3552 bytes=1951832576 budget=12819959808 ceiling=14144241664 allocated=1951832576
expert cache enabled: devices=2 pools=120 ... actual_device=2116997120 actual_host=122880 ...
```

`ceiling` is that device's total VRAM minus the rail, `budget` is what remains for the pool, and
`allocated` is that device's pool plus table bytes. The trailer `expert pool runtime ...` line is
aggregate across devices.

### Multi-GPU environment knobs

- `LLAMA_MOE_POOL_RAIL_MIB` is **per device**. Default 2879 MiB, hard floor 1024 MiB. On an 8 GB card
  the default rail is a large fraction of VRAM, so lower it (for example to 1024).
- `LLAMA_MOE_POOL_CAP_MIB` is a **global total** across all pool-owning devices, split between them
  in proportion to their rail budgets. It is never multiplied by the device count.

### Multi-GPU notes and limits

- The feature is a capacity play, not a latency play.
- `--split-mode row` / `tensor` are rejected; MoE tensor parallelism is not supported by the pool.
- Mixed architectures are only flagged (`reason=mixed-arch`) when the pool-owning devices report
  different backend registration names. Two different GPUs under one backend (two CUDA compute
  capabilities, or gfx802 + gfx906 under Vulkan) share a registration name and are **not** flagged.
  Different architectures compile different kernels, so cross-architecture bitwise output equality
  is not guaranteed; a same-architecture multi-GPU box keeps the bitwise property.
- At most `GGML_SCHED_MAX_BACKENDS` (16) backends exist, which bounds N.
- Speculative draft/MTP forces the pool off, unchanged from single-device.

### Multi-GPU validation status

- Single-device no-op gate: passed. Admission, slot counts, device bytes, hit counters and the output
  hash are identical to the pre-change build on the single-GPU path.
- Two-device functional gate: passed on Vulkan with a mixed pair (Radeon Pro VII / gfx906 and
  R9 380 / gfx802), `Qwen3.6-35B-A3B-IQ4_NL_XL` at 2048 context. Per-device status, exact per-device
  byte accounting, coherent and run-to-run deterministic output, no asserts. Two-device ownership was
  confirmed: 9 pools on the small card, 111 on the large one.
- Not done yet: a same-architecture multi-GPU run (needed for the bitwise pooled-versus-unpooled hash
  gate), per-device runtime hit-rate log lines, and CUDA multi-GPU at all.

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
`common/common.cpp`, `common/common.h`, `include/llama.h`, `src/llama-cparams.h`,
`src/llama-context.cpp`, `src/llama-context.h`, `src/llama-graph.cpp`, `src/llama-graph.h`,
`src/llama-moe-cache-range.h` (new file) and `ggml/src/ggml-backend.cpp`. After rebasing, rebuild and
re-run `test-expert-pool`.

## License

MIT, same as upstream llama.cpp.
