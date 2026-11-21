// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatRayQuery.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"
#if !defined(__APPLE__)
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPUVulkan.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatVulkanContext.h"
#endif

namespace open3d {
namespace visualization {
namespace rendering {
namespace {

// Vulkan guarantees at least this storage-buffer range. The deliberately
// bounded experiment rejects requests that exceed it instead of truncating.
constexpr std::uint64_t kMaxBufferBytes = 128ull * 1024ull * 1024ull;

void CheckTensor(const core::Tensor& tensor,
                 core::Dtype dtype,
                 const core::SizeVector& tail,
                 const char* name) {
    const auto& shape = tensor.GetShape();
    if (tensor.GetDevice() != core::Device("CPU:0") ||
        tensor.GetDtype() != dtype || shape.size() != tail.size() + 1 ||
        !std::equal(tail.begin(), tail.end(), shape.begin() + 1)) {
        utility::LogError("GaussianSplatRayQuery: {} must be CPU {} [N,{}]",
                          name, dtype.ToString(), tail.ToString());
    }
    const auto bytes = static_cast<std::uint64_t>(tensor.NumElements()) *
                       static_cast<std::uint64_t>(dtype.ByteSize());
    if (bytes > kMaxBufferBytes) {
        utility::LogError("GaussianSplatRayQuery: {} exceeds 128 MiB", name);
    }
}

void CheckFinite(const core::Tensor& tensor, const char* name) {
    const float* values = tensor.GetDataPtr<float>();
    for (std::int64_t i = 0; i < tensor.NumElements(); ++i) {
        if (!std::isfinite(values[i])) {
            utility::LogError("GaussianSplatRayQuery: {} must be finite", name);
        }
    }
}

}  // namespace

struct GaussianSplatRayQuery::Impl {
    std::unique_ptr<GaussianSplatGpuContext> gpu;
    std::uint64_t generation = 0;
    std::array<std::uintptr_t, 5> buffers{};
    std::array<std::size_t, 5> capacities{};
    std::thread::id thread = std::this_thread::get_id();

    // Retain allocation capacity across queries. The buffer abstraction owns
    // destruction; this class never takes ownership of the global VkDevice.
    void EnsureBuffer(std::size_t index, std::size_t bytes) {
        bytes = std::max<std::size_t>(bytes, 16);
        if (bytes <= capacities[index]) return;
        buffers[index] = index == 3
                                 ? gpu->ResizeReadbackBuffer(buffers[index], bytes,
                                                             "gs.ray_query.hits")
                                 : gpu->ResizeBuffer(buffers[index], bytes,
                                                      "gs.ray_query");
        if (!buffers[index]) {
            utility::LogError("GaussianSplatRayQuery: GPU allocation failed");
        }
        capacities[index] = bytes;
    }
};

GaussianSplatRayQuery::GaussianSplatRayQuery() : impl_(new Impl) {}

GaussianSplatRayQuery::~GaussianSplatRayQuery() { Close(); }

void GaussianSplatRayQuery::Close() {
    if (!impl_) return;
#if !defined(__APPLE__)
    auto& vk = GaussianSplatVulkanContext::GetInstance();
    if (impl_->gpu && (!vk.IsValid() ||
                       impl_->generation != vk.GetGeneration())) {
        // OffscreenRenderer destruction can invalidate the shared device.
        // Its Vulkan allocations have already died with that device; avoid
        // invoking stale handles during a late Python finalizer.
        utility::LogWarning("GaussianSplatRayQuery must be closed before "
                            "destroying the renderer's Vulkan device");
        static_cast<void>(impl_->gpu.release());
    }
#endif
    impl_.reset();
}

core::Tensor GaussianSplatRayQuery::Query(const core::Tensor& geometry,
                                         const core::Tensor& spans,
                                         const core::Tensor& planes) {
    if (!impl_) utility::LogError("GaussianSplatRayQuery is closed");
    if (impl_->thread != std::this_thread::get_id()) {
        utility::LogError("GaussianSplatRayQuery must stay on its creating thread");
    }
    CheckTensor(geometry, core::Float32, {4, 4}, "geometry");
    CheckTensor(spans, core::UInt32, {8}, "spans");
    CheckTensor(planes, core::Float32, {4}, "planes");
    if (geometry.GetShape(0) != spans.GetShape(0)) {
        utility::LogError("GaussianSplatRayQuery: region counts differ");
    }
    const auto g = geometry.Contiguous();
    const auto s = spans.Contiguous();
    const auto p = planes.Contiguous();
    CheckFinite(g, "geometry");
    CheckFinite(p, "planes");
    const auto regions = static_cast<std::uint32_t>(g.GetShape(0));
    const auto* rows = s.GetDataPtr<std::uint32_t>();
    const auto* coefficients = g.GetDataPtr<float>();
    std::uint64_t candidate_count = 0;
    for (std::uint32_t r = 0; r < regions; ++r) {
        const auto* row = rows + 8 * r;
        const std::uint64_t count = static_cast<std::uint64_t>(row[1]) - row[0];
        if (row[0] != candidate_count || row[1] < row[0] || row[4] == 0 ||
            count % row[4] != 0 || row[7] != 0 ||
            static_cast<std::uint64_t>(row[5]) + row[6] >
                    static_cast<std::uint64_t>(p.GetShape(0)) ||
            static_cast<std::uint64_t>(row[2]) + count / row[4] >
                    std::numeric_limits<std::uint32_t>::max() ||
            static_cast<std::uint64_t>(row[3]) + row[4] >
                    std::numeric_limits<std::uint32_t>::max() ||
            !(coefficients[16 * r + 14] > 0.0f) ||
            !(coefficients[16 * r + 15] > 0.0f)) {
            utility::LogError("GaussianSplatRayQuery: invalid region {}", r);
        }
        candidate_count = row[1];
    }
    if (candidate_count * 4 * sizeof(float) > kMaxBufferBytes) {
        utility::LogError("GaussianSplatRayQuery: output exceeds 128 MiB; "
                          "split the query into explicit batches");
    }
    core::Tensor output = core::Tensor::Empty(
            {static_cast<std::int64_t>(candidate_count), 4}, core::Float32);
    if (candidate_count == 0) return output;

#if defined(__APPLE__)
    utility::LogError("GaussianSplatRayQuery currently requires Vulkan; "
                      "there is no CPU or Metal fallback");
#else
    auto& vk = GaussianSplatVulkanContext::GetInstance();
    if (impl_->gpu && (!vk.IsValid() ||
                       impl_->generation != vk.GetGeneration())) {
        utility::LogError("GaussianSplatRayQuery: shared Vulkan device was "
                          "destroyed; close and recreate the query");
    }
    if (!impl_->gpu) {
        if (!vk.Initialize()) {
            utility::LogError("GaussianSplatRayQuery: {}", vk.GetLastError());
        }
        if (vk.IsSoftwareDevice()) {
            utility::LogError("GaussianSplatRayQuery requires a hardware Vulkan "
                              "device; software fallback is disabled");
        }
        impl_->generation = vk.GetGeneration();
        impl_->gpu = CreateComputeGpuContextVulkan();
        if (!impl_->gpu || !impl_->gpu->EnsureProgramsLoaded()) {
            utility::LogError("GaussianSplatRayQuery: cannot load GPU programs");
        }
    }
    auto& gpu = *impl_->gpu;
    const std::array<std::size_t, 5> sizes{
            static_cast<std::size_t>(g.NumElements()) * sizeof(float),
            static_cast<std::size_t>(s.NumElements()) * sizeof(std::uint32_t),
            static_cast<std::size_t>(p.NumElements()) * sizeof(float),
            static_cast<std::size_t>(candidate_count) * 4 * sizeof(float), 16};
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        impl_->EnsureBuffer(i, sizes[i]);
    }
    gpu.UploadBuffer(impl_->buffers[0], g.GetDataPtr(), sizes[0], 0);
    gpu.UploadBuffer(impl_->buffers[1], s.GetDataPtr(), sizes[1], 0);
    if (sizes[2]) gpu.UploadBuffer(impl_->buffers[2], p.GetDataPtr(), sizes[2], 0);
    const std::array<std::uint32_t, 4> params{
            regions, static_cast<std::uint32_t>(candidate_count), 0, 0};
    gpu.UploadBuffer(impl_->buffers[4], params.data(), sizeof(params), 0);
    const auto groups = static_cast<std::uint32_t>((candidate_count + 127) / 128);
    const auto max_groups = gpu.GetMaxComputeWorkGroupCount();
    const auto groups_x = std::min(groups, max_groups);
    const auto groups_y = (groups + groups_x - 1) / groups_x;
    // Composite submission waits on its fence; the subsequent mapped readback
    // and Python wall timing therefore include all submitted GPU execution.
    GpuComputeFrame frame(gpu, GpuComputeFrame::kComposite);
    GpuComputePass(gpu, ComputeProgramId::kGsDexel, "gs.ray_query")
            .SSBO(0, impl_->buffers[0])
            .SSBO(1, impl_->buffers[1])
            .SSBO(2, impl_->buffers[2])
            .SSBO(3, impl_->buffers[3])
            .UBO(4, impl_->buffers[4])
            .Dispatch(groups_x, groups_y, 1);
    gpu.ReadbackBarrier();
    frame.End();
    if (!gpu.WasLastSubmitSuccessful() ||
        !gpu.DownloadBuffer(impl_->buffers[3], output.GetDataPtr(), sizes[3], 0)) {
        utility::LogError("GaussianSplatRayQuery: GPU execution/readback failed");
    }
#endif
    return output;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
