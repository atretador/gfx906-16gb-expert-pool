# Build recipe for the gfx906-16gb-expert-pool fork.
#
# Build from the repository root so the whole source tree is the context:
#   docker build -f docker/Dockerfile -t local/llama.cpp-gfx906:expert-pool .
#
# The expert-pool feature is not upstream. This recipe builds the tree as it is,
# pinned to the upstream commit the branch is based on, and records both the
# base commit and a digest of the build context in /etc/llama.cpp-provenance.

FROM mixa3607/llama.cpp-gfx906@sha256:bf92e1ed078118f3f66c86c80f1c61537f4e407768567f6c14991643abc85bfb

ARG LLAMA_CPP_COMMIT=972d2313bc0bf0a45f634f77d95c9fb03aeab12c
LABEL org.opencontainers.image.source-commit="${LLAMA_CPP_COMMIT}" \
      org.opencontainers.image.description="MI50 (gfx906) expert-pool fork: upstream llama.cpp + MoE expert cache"

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

RUN test "$(git rev-parse HEAD)" = "${LLAMA_CPP_COMMIT}" \
    && tree_digest="$(find . -path './.git' -prune -o -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d ' ' -f1)" \
    && { printf 'source_commit=%s\n' "${LLAMA_CPP_COMMIT}"; printf 'source_tree_sha256=%s\n' "${tree_digest}"; } \
       | tee /etc/llama.cpp-provenance \
    && cmake -B build -G Ninja \
      -DGGML_HIP=ON \
      -DGPU_TARGETS=gfx906 \
      -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel 20 --target llama-server llama-bench test-expert-pool

WORKDIR /app
RUN cp /src/build/bin/llama-server /src/build/bin/llama-bench /app/

ENTRYPOINT ["/app/llama-server"]
