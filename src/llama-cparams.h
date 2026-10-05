#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    // number of expert slots kept in VRAM per offloaded MoE expert weight tensor (0 = off)
    int32_t  expert_cache_slots = 0;

    // per-device expert slots in device order (empty = use expert_cache_slots for all devices)
    std::vector<int32_t> expert_cache_slots_per_device;

    // per-device cached layer ranges from --moe-cache-range (empty = derive from ownership)
    std::string moe_cache_range;

    // warmth gate: disable the expert pool for the rest of the context when its decode hit
    // rate stays below this percent after a warm-up window (0 = never fall back)
    int32_t  expert_pool_fallback = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;
    bool training;           // set by llama_opt_init()

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    // cache element types (mirror of the memory module params, for size estimation)
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    llama_context * ctx_other;
};
