#pragma once

// parser and validator for --moe-cache-range (Plan B per-device layer-range expert pools).
// header-only and std-only so both src/llama-context.cpp and tests/test-expert-pool.cpp can
// use it without pulling in the rest of the context.

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// per-device cached layer ranges. lo[i] < 0 (and hi[i] < 0) means device i caches nothing.
struct llama_moe_cache_range {
    std::vector<int> lo;
    std::vector<int> hi;
};

inline bool llama_moe_cache_range_parse(const std::string & text, int n_device,
        llama_moe_cache_range & out, std::string & err) {
    out.lo.assign(n_device, -1);
    out.hi.assign(n_device, -1);

    // split on ',', keeping a trailing empty token: "0-19," declares two entries
    std::vector<std::string> tokens;
    size_t start = 0;
    while (true) {
        const size_t comma = text.find(',', start);
        const std::string tok = comma == std::string::npos ? text.substr(start) : text.substr(start, comma - start);
        tokens.push_back(tok);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    if ((int) tokens.size() != n_device) {
        err = "entry count " + std::to_string(tokens.size()) + " does not match device count " + std::to_string(n_device);
        return false;
    }

    auto trim = [](const std::string & s) {
        const size_t b = s.find_first_not_of(" \t");
        if (b == std::string::npos) {
            return std::string();
        }
        const size_t e = s.find_last_not_of(" \t");
        return s.substr(b, e - b + 1);
    };

    for (int d = 0; d < n_device; ++d) {
        const std::string tok = trim(tokens[d]);
        if (tok.empty() || tok == "-") {
            continue;
        }
        const size_t dash = tok.find('-');
        if (dash == std::string::npos || dash == 0 || dash == tok.size() - 1 ||
                tok.find('-', dash + 1) != std::string::npos) {
            err = "malformed range '" + tok + "' for device " + std::to_string(d) + " (expected lo-hi or -)";
            return false;
        }
        const std::string lo_s = tok.substr(0, dash);
        const std::string hi_s = tok.substr(dash + 1);
        for (const char * p = lo_s.c_str(); *p != '\0'; ++p) {
            if (*p < '0' || *p > '9') {
                err = "malformed range '" + tok + "' for device " + std::to_string(d);
                return false;
            }
        }
        for (const char * p = hi_s.c_str(); *p != '\0'; ++p) {
            if (*p < '0' || *p > '9') {
                err = "malformed range '" + tok + "' for device " + std::to_string(d);
                return false;
            }
        }
        // parse and bound-check before any narrowing cast: an out-of-range value must not
        // silently wrap into a plausible layer index
        errno = 0;
        char * end = nullptr;
        const long lo = std::strtol(lo_s.c_str(), &end, 10);
        if (errno == ERANGE || end == lo_s.c_str() || *end != '\0' || lo < 0 || lo > INT_MAX) {
            err = "range '" + tok + "' for device " + std::to_string(d) + " has an out-of-range lo";
            return false;
        }
        errno = 0;
        const long hi = std::strtol(hi_s.c_str(), &end, 10);
        if (errno == ERANGE || end == hi_s.c_str() || *end != '\0' || hi < 0 || hi > INT_MAX) {
            err = "range '" + tok + "' for device " + std::to_string(d) + " has an out-of-range hi";
            return false;
        }
        if (lo > hi) {
            err = "range '" + tok + "' for device " + std::to_string(d) + " has lo > hi";
            return false;
        }
        out.lo[d] = (int) lo;
        out.hi[d] = (int) hi;
    }
    return true;
}

// validate the parsed ranges against the layer ownership and offload sets. every mismatch
// is a hard error: the ranges must be a complete, non-overlapping partition of the
// offloaded GPU-owned layers, in device order. `layer_owner[il]` is the device index that
// owns layer il (-1 = CPU). `layer_offloaded[il]` is nonzero when layer il has a
// host-backed expert tensor.
inline bool llama_moe_cache_range_validate(const llama_moe_cache_range & ranges,
        const std::vector<int> & layer_owner,
        const std::vector<uint8_t> & layer_offloaded,
        int n_layer, int n_device, std::string & err) {
    if ((int) ranges.lo.size() != n_device || (int) ranges.hi.size() != n_device) {
        err = "range count does not match device count";
        return false;
    }
    if ((int) layer_owner.size() != n_layer || (int) layer_offloaded.size() != n_layer) {
        err = "layer sets have the wrong size";
        return false;
    }

    std::vector<uint8_t> seen(n_layer, 0);
    bool prev_nonempty = false;
    int prev_hi = -1;

    for (int d = 0; d < n_device; ++d) {
        const int lo = ranges.lo[d];
        const int hi = ranges.hi[d];
        if (lo < 0) {
            if (hi >= 0) {
                err = "device " + std::to_string(d) + " has a malformed empty range";
                return false;
            }
            prev_nonempty = false;
            continue;
        }
        if (hi >= n_layer) {
            err = "range " + std::to_string(lo) + "-" + std::to_string(hi) + " for device " +
                    std::to_string(d) + " exceeds the layer count " + std::to_string(n_layer);
            return false;
        }
        for (int il = lo; il <= hi; ++il) {
            if (seen[il]) {
                err = "layer " + std::to_string(il) + " is declared more than once";
                return false;
            }
            seen[il] = 1;
            if (layer_owner[il] != d) {
                err = "layer " + std::to_string(il) + " in device " + std::to_string(d) +
                        "'s range is owned by " + (layer_owner[il] < 0 ? std::string("CPU") : "device " + std::to_string(layer_owner[il]));
                return false;
            }
            if (!layer_offloaded[il]) {
                err = "layer " + std::to_string(il) + " in device " + std::to_string(d) +
                        "'s range is not host-offloaded";
                return false;
            }
        }
        if (prev_nonempty && prev_hi + 1 != lo) {
            err = "gap between device " + std::to_string(d - 1) + " and device " + std::to_string(d) +
                    " ranges (layer " + std::to_string(prev_hi + 1) + " is unassigned)";
            return false;
        }
        prev_hi = hi;
        prev_nonempty = true;
    }

    // the union must cover every offloaded GPU-owned layer
    for (int il = 0; il < n_layer; ++il) {
        if (layer_offloaded[il] && layer_owner[il] >= 0 && !seen[il]) {
            err = "offloaded GPU-owned layer " + std::to_string(il) + " is not covered by --moe-cache-range";
            return false;
        }
    }
    return true;
}
