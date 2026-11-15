// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
#include <vector>
#include <torch/extension.h>

namespace mono_decode_xpu {
// Owned by one generate and retained alongside captured graphs. There is no
// process-global counter or raw-pointer scratch cache in the fused attention.
struct AttentionMonoWorkspace {
    int rows, heads, kv_heads, capacity, tiles, chunks, groups, device;
    torch::Tensor packed_q, q, k, output, partial, metadata, counter;
};
std::shared_ptr<AttentionMonoWorkspace> attention_mono_workspace(
    torch::Tensor rawq, torch::Tensor rawk, torch::Tensor key);
std::vector<torch::Tensor> attention_mono(
    torch::Tensor rawq, torch::Tensor rawk, torch::Tensor value_rows,
    torch::Tensor weightq, torch::Tensor weightk, torch::Tensor freqs,
    torch::Tensor key, torch::Tensor value, torch::Tensor position,
    torch::Tensor seqlen, torch::Tensor mask, double epsq, double epsk,
    int64_t rotary_dim, const std::shared_ptr<AttentionMonoWorkspace>& workspace);
}  // namespace mono_decode_xpu
