// SPDX-License-Identifier: Apache-2.0
// Shared host-side launch planning for the persistent stream engine.
#pragma once
#include <algorithm>
#include <initializer_list>
#include <ATen/xpu/XPUContext.h>
#include <torch/extension.h>
#include "mono_common.hpp"

namespace mono_decode_xpu::mono {
inline uint32_t resident_wgs(sycl::queue& queue, int64_t requested) {
    // grid_sync is a spinning grid-wide barrier: every work-group must be co-resident.
    // These kernels use >64 KiB SLM per work-group, allowing at most one per Xe core.
    // Oversubscribing the cores can deadlock the grid.
    const int64_t cores = queue.get_device().get_info<sycl::info::device::max_compute_units>() / 8;
    if (requested <= 0 || requested > cores) requested = cores;
    return static_cast<uint32_t>(requested);
}

inline int attention_chunks(sycl::queue& queue, const torch::Tensor& like,
                            int kv_heads, int row_packs, int capacity, int64_t hint) {
    if (hint > 0) return static_cast<int>(hint);
    int limit = 48;
    // The largest query pack makes the partial/merge buffers expensive on
    // B580. One chunk per Xe core improves that pack; keep the draft and
    // other devices on the existing scheduling policy.
    if (kv_heads == 4 && row_packs == 4 &&
        at::xpu::getDeviceProperties(like.get_device())->device_id == 0xE20B)
        limit = std::min(limit, static_cast<int>(resident_wgs(queue, 0)));
    return std::max(1, std::min(limit, (capacity + 255) / 256));
}

constexpr size_t STREAM_SLM = TAB * sizeof(uint16_t) + MSG * sizeof(float) + 16 * sizeof(float);
constexpr size_t NORM_SLM = 1056 * sizeof(float);

inline int64_t plan_streams(sycl::queue& queue, uint32_t workers,
                            std::initializer_list<Stream*> streams, size_t fixed_slm) {
    int64_t elements = ACC;
    for (auto* stream : streams) {
        if (stream->pair) stream_finalize(*stream, workers);
        int64_t first, last;
        stream_range(*stream, 0, workers, first, last);
        const int64_t required = (last - first) * 8 * 16;
        elements = std::max(elements, required);
        stream->ksplit = choose_ksplit(last - first, stream->m[0].k);
    }
    if (elements > ACC) {
        const size_t bytes = fixed_slm + size_t(elements) * sizeof(int32_t);
        const auto capacity = queue.get_device().get_info<sycl::info::device::local_mem_size>();
        TORCH_CHECK(bytes <= capacity, "persistent SLM requirement ", bytes,
                    " exceeds device capacity ", capacity, "; workers=", workers);
    }
    return elements;
}
}  // namespace mono_decode_xpu::mono
