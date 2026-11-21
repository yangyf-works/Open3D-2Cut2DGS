// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CPU-side packing and GPU buffer-sizing implementation for the Gaussian splat
// compute pipeline (Vulkan and Metal backends share this code).

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatDataPacking.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <limits>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

// Integer ceiling division: ceil(value / divisor).
// Used locally to compute tile grid dimensions.
inline int CeilDiv(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

// Pack an Eigen 4x4 (column-major) into a 16-float array in column-major order
// matching std140 mat4 layout.
void PackMat4(float* dst, const Eigen::Matrix4f& m) {
    std::memcpy(dst, m.data(), 16 * sizeof(float));
}

// Shared coefficients are scene metadata: reject invalid values before
// either merging objects or uploading their common per-frame uniform.
void ValidateSurfaceLighting(const Eigen::Vector3f& lighting, float shininess) {
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(lighting[i]) || lighting[i] < 0.f) {
            utility::LogError("gaussian_splat_surface_lighting must contain "
                              "finite nonnegative ambient, diffuse and "
                              "specular coefficients.");
        }
    }
    if (!std::isfinite(shininess) || shininess <= 0.f) {
        utility::LogError("gaussian_splat_surface_shininess must be finite "
                          "and strictly positive.");
    }
}

// Reserve the uniform source layout once before packing or scene merging.
void ReservePackedAttrArrays(GaussianSplatPackedAttrs* out,
                             std::size_t splat_capacity) {
    out->positions.reserve(splat_capacity);
    out->rotations.reserve(splat_capacity);
    out->surface_data.reserve(kGaussianSplatSurfaceWords * splat_capacity);
}

void SetVisibilityMaskRange(std::vector<std::uint32_t>* visibility_mask,
                            std::uint32_t splat_start,
                            std::uint32_t splat_count,
                            bool visible) {
    const std::uint32_t splat_end = splat_start + splat_count;
    const std::uint32_t n_words = (splat_end + 31u) / 32u;
    visibility_mask->resize(n_words, 0u);
    if (!visible) {
        return;
    }

    for (std::uint32_t k = splat_start; k < splat_end; ++k) {
        (*visibility_mask)[k >> 5u] |= 1u << (k & 31u);
    }
}

// Radix-sort workgroup constants — must stay in sync with the radix shaders
// and the dispatch-args compute shader.
constexpr std::uint32_t kRadixWorkgroupSize = 256;
constexpr std::uint32_t kRadixSortBins = 256;
constexpr std::uint32_t kRadixTargetBlocksPerWG = 32;

}  // namespace

// ----- ComputeGaussianGpuBufferSizes -----------------------------------------

void ComputeGaussianGpuBufferSizes(const PackedGaussianScene& packed,
                                   GaussianGpuBufferSizes* out) {
    // Derive SSBO/UBO byte sizes from the packed-scene tile/splat counts.
    // All size formulas must stay in sync with the dispatch shaders.
    if (!out || !packed.valid) {
        return;
    }
    // ProjectedComposite (104 B): written by project, read only by composite.
    out->projected_composite_size =
            packed.splat_count * sizeof(ProjectedComposite);
    // steal_counter: one uint32 — atomic tile index for composite
    // work-stealing. Composite no longer needs per-tile counts/offsets arrays
    // (binary search).
    out->tile_scalar_size = sizeof(std::uint32_t);

    out->entries_capacity =
            std::max<std::uint32_t>(1u, packed.view_params.limits[0]);
    // Sort key/value ping-pong buffers: one uint32 per entry capacity.
    out->key_cap_size = std::max<std::size_t>(
            4u, static_cast<std::size_t>(out->entries_capacity) *
                        sizeof(std::uint32_t));

    out->radix_num_wg_cap = std::max(
            1u, (out->entries_capacity +
                 kRadixWorkgroupSize * kRadixTargetBlocksPerWG - 1u) /
                        (kRadixWorkgroupSize * kRadixTargetBlocksPerWG));
    out->histogram_buf_size = std::max<std::size_t>(
            4u, static_cast<std::size_t>(out->radix_num_wg_cap) *
                        kRadixSortBins * sizeof(std::uint32_t));

    // 8 slots: 4×histogram + 4×radix_scatter.
    out->dispatch_args_size = 8u * 3u * sizeof(std::uint32_t);
    out->radix_params_size = 4u * kGaussianRadixParamsStride;
}

// ----- PackGaussianViewParams ------------------------------------------------

PackedGaussianScene PackGaussianViewParams(
        const GaussianSplatPackedAttrs& attrs,
        const GaussianSplatRenderer::ViewRenderData& render_data,
        const GaussianSplatRenderer::RenderConfig& config) {
    // Build the 304-byte per-frame UBO from camera/viewport/scene state.
    // Called every frame; cost is a single memcpy-sized CPU write to the GPU.
    PackedGaussianScene packed;

    if (attrs.splat_count == 0 || render_data.viewport_size.x() <= 0 ||
        render_data.viewport_size.y() <= 0) {
        return packed;
    }

    ValidateSurfaceLighting(attrs.surface_lighting, attrs.surface_shininess);
    const std::uint32_t n = attrs.splat_count;
    const int w = render_data.viewport_size.x();
    const int h = render_data.viewport_size.y();
    const int tx = config.tile_size.x();
    const int ty = config.tile_size.y();
    const int tcx = CeilDiv(w, tx);
    const int tcy = CeilDiv(h, ty);

    auto& vp = packed.view_params;
    std::memset(&vp, 0, sizeof(vp));

    // entry_capacity sizes shared tile-entry buffers from avg_tiles_per_splat
    // (statistical estimate), decoupled from max_tiles_per_splat (hard cap
    // enforced in gaussian_project.comp). Writes are still bounds-checked
    // against entry_capacity in WriteSortEntries, so it won't overflow.
    const auto entry_budget =
            static_cast<std::uint64_t>(n) * config.avg_tiles_per_splat;
    const auto entry_capacity = static_cast<std::uint32_t>(std::min(
            entry_budget,
            static_cast<std::uint64_t>(config.max_tile_entries_total)));
    const std::uint32_t per_splat_cap = config.max_tiles_per_splat;

    // world_from_model: splat positions are in world space → identity.
    // render_data.model_matrix is the camera rig; do not use it here.
    PackMat4(vp.world_from_model, Eigen::Matrix4f::Identity());
    PackMat4(vp.view_from_world,
             render_data.view_matrix.matrix().cast<float>());
    PackMat4(vp.clip_from_view, render_data.projection_matrix.matrix());

    const std::array<float, 4> cam_pn{
            render_data.camera_position.x(), render_data.camera_position.y(),
            render_data.camera_position.z(),
            static_cast<float>(render_data.near_plane)};
    std::memcpy(vp.camera_position_and_near, cam_pn.data(), sizeof(cam_pn));

    const std::array<float, 4> vp_origin_size{
            static_cast<float>(render_data.viewport_origin.x()),
            static_cast<float>(render_data.viewport_origin.y()),
            static_cast<float>(w), static_cast<float>(h)};
    std::memcpy(vp.viewport_origin_and_size, vp_origin_size.data(),
                sizeof(vp_origin_size));

    const std::array<std::uint32_t, 4> scene_u{
            n, 0u,
            (attrs.antialias || config.antialias) ? 1u : 0u,
            render_data.screen_y_down ? 1u : 0u};
    std::memcpy(vp.scene, scene_u.data(), sizeof(scene_u));

    const std::array<std::uint32_t, 4> tiles_u{
            static_cast<std::uint32_t>(tx), static_cast<std::uint32_t>(ty),
            static_cast<std::uint32_t>(tcx), static_cast<std::uint32_t>(tcy)};
    std::memcpy(vp.tiles, tiles_u.data(), sizeof(tiles_u));

    vp.limits[0] = entry_capacity;
    vp.limits[1] = per_splat_cap;
    vp.limits[2] = config.max_tile_entries_total;
    // T = bits needed to hold the largest tile index =
    // floor(log2(max_index))+1. For a single tile (tcx*tcy == 1) max_tile_index
    // == 0, so T defaults to 1 (no zero-shift UB). Clamped to [1,31] so the
    // depth field gets >= 1 bit.
    const std::uint32_t max_tile_index =
            static_cast<std::uint32_t>(tcx * tcy) - 1u;
    std::uint32_t tile_key_bits = 0u;
    for (std::uint32_t v = max_tile_index; v > 0u; v >>= 1u) {
        ++tile_key_bits;
    }
    vp.limits[3] = std::clamp(tile_key_bits, 1u, 31u);

    const std::array<float, 4> depth_rng{
            static_cast<float>(render_data.near_plane),
            static_cast<float>(render_data.far_plane), 0.f, 0.f};
    std::memcpy(vp.depth_range_and_flags, depth_rng.data(), sizeof(depth_rng));
    std::memcpy(vp.surface_lighting, attrs.surface_lighting.data(),
                3 * sizeof(float));
    vp.surface_lighting[3] = attrs.surface_shininess;

    packed.splat_count = n;
    packed.tile_count = static_cast<std::uint32_t>(tcx * tcy);
    packed.pixel_count = static_cast<std::uint32_t>(w * h);
    packed.valid = true;
    return packed;
}

// ----- PackGaussianSplatAttrsDirect ------------------------------------------

void PackGaussianSplatAttrsDirect(const float* pts_ptr,
                                  std::size_t n,
                                  const float* scale_ptr,
                                  const float* rot_ptr,
                                  bool antialias,
                                  GaussianSplatPackedAttrs& out,
                                  const GaussianSplatClipData* clip_data,
                                  const GaussianSplatSurfaceData* surface) {
    const bool opaque = surface && surface->opaque;
    if (n && (!surface || !surface->colors)) {
        utility::LogError("Gaussian splats require surface_color.");
    }
    // Validate before modifying the cache.
    if (surface) {
        for (std::size_t i = 0; i < n; ++i) {
            if (surface->ids && surface->ids[i] < 0) {
                utility::LogError("Gaussian surface_id must be nonnegative.");
            }
            for (int j = 0; j < 3; ++j) {
                const float color = surface->colors[3 * i + j];
                if (!std::isfinite(color) || color < 0.f || color > 1.f) {
                    utility::LogError("Gaussian surface_color must be finite "
                                      "in [0,1].");
                }
            }
        }
    }
    // Validate every boundary before touching the output cache.
    if (clip_data) {
        for (std::size_t i = 0; i < n; ++i) {
            t::geometry::NormalizeGaussianClipData(clip_data[i]);
        }
    }
    // Geometry compression is unchanged; both modes store one unlit Float32
    // RGB triple. Every input retains its original index.
    out = GaussianSplatPackedAttrs{};
    out.antialias = antialias;
    if (n == 0) {
        out.visibility_mask.assign(0, 0u);
        return;
    }
    if (!pts_ptr || !scale_ptr || !rot_ptr) {
        return;
    }

    ReservePackedAttrArrays(&out, n);

    for (std::size_t i = 0; i < n; ++i) {
        out.positions.push_back({pts_ptr[i * 3 + 0], pts_ptr[i * 3 + 1],
                                 pts_ptr[i * 3 + 2], scale_ptr[i]});

        // Preserve Float32 components; the shader still normalizes the frame.
        out.rotations.push_back({
                rot_ptr[i * 4 + 0], rot_ptr[i * 4 + 1],
                rot_ptr[i * 4 + 2], rot_ptr[i * 4 + 3]});

        const auto clip = clip_data
                                  ? t::geometry::NormalizeGaussianClipData(clip_data[i])
                                  : GaussianSplatClipData{};
        // Keep validated clip values bit-exact in the raw-word SSBO. memcpy
        // avoids aliasing and numeric conversion; GLSL uses uintBitsToFloat.
        const std::array<float, 4> clip_values = {
                clip.clip1_angle, clip.clip1_split,
                clip.clip2_angle, clip.clip2_split};
        const auto clip_start = out.surface_data.size();
        out.surface_data.resize(clip_start + clip_values.size());
        std::memcpy(out.surface_data.data() + clip_start, clip_values.data(),
                    sizeof(clip_values));
        const auto metadata = out.surface_data.size();
        out.surface_data.resize(metadata + 5, 0u);
        out.surface_data[metadata] = surface->ids ? surface->ids[i] : 0u;
        out.surface_data[metadata + 1] = opaque ? 1u : 0u;
        std::memcpy(out.surface_data.data() + metadata + 2,
                    surface->colors + 3 * i, 3 * sizeof(float));
    }

    out.splat_count = static_cast<std::uint32_t>(out.positions.size());
    // Populate a default all-visible mask so the buffer is always present.
    // RebuildMergedGaussianData() overwrites this with per-object visibility.
    // Bit-packed: all-ones = all splats visible. ceil(n/32) words.
    out.visibility_mask.clear();
    SetVisibilityMaskRange(&out.visibility_mask, 0u, out.splat_count, true);
}

void MergeGaussianSplatPackedAttrs(
        const std::vector<GaussianSplatMergeItem>& items,
        GaussianSplatPackedAttrs* out,
        std::vector<std::uint32_t>* splat_starts) {
    if (!out) {
        return;
    }

    // Check every nonempty source before changing the output. Hidden objects
    // count too, so toggling visibility cannot silently change scene lighting.
    Eigen::Vector3f lighting(1.f, 0.f, 0.f);
    float shininess = 2.f;
    bool has_lighting = false;
    for (const auto& item : items) {
        if (!item.attrs || item.attrs->splat_count == 0) continue;
        ValidateSurfaceLighting(item.attrs->surface_lighting,
                                item.attrs->surface_shininess);
        if (has_lighting &&
            !(lighting.array() == item.attrs->surface_lighting.array()).all()) {
            utility::LogError("Gaussian objects in one scene must use identical "
                              "gaussian_splat_surface_lighting values, "
                              "including hidden objects.");
        }
        if (has_lighting && shininess != item.attrs->surface_shininess) {
            utility::LogError("Gaussian objects in one scene must use identical "
                              "gaussian_splat_surface_shininess values, "
                              "including hidden objects.");
        }
        lighting = item.attrs->surface_lighting;
        shininess = item.attrs->surface_shininess;
        has_lighting = true;
    }

    *out = GaussianSplatPackedAttrs{};
    out->surface_lighting = lighting;
    out->surface_shininess = shininess;
    if (splat_starts) {
        splat_starts->clear();
        splat_starts->reserve(items.size());
    }

    std::size_t merged_splat_capacity = 0u;
    for (const auto& item : items) {
        if (!item.attrs) {
            continue;
        }
        merged_splat_capacity += item.attrs->splat_count;
    }

    ReservePackedAttrArrays(out, merged_splat_capacity);

    std::uint32_t next_surface_id = 1u;
    for (const auto& item : items) {
        if (splat_starts) splat_starts->push_back(out->splat_count);
        if (!item.attrs) {
            continue;
        }
        const auto& src = *item.attrs;
        const std::uint32_t splat_count = src.splat_count;

        out->positions.insert(out->positions.end(), src.positions.begin(),
                              src.positions.end());
        out->rotations.insert(out->rotations.end(), src.rotations.begin(),
                              src.rotations.end());
        const std::size_t payload_start = out->surface_data.size();
        out->surface_data.insert(out->surface_data.end(), src.surface_data.begin(),
                               src.surface_data.end());
        // Namespace positive IDs per object and isolate every unknown sample.
        // A single global namespace prevents work/debug objects from mixing.
        std::unordered_map<std::uint32_t, std::uint32_t> surface_ids;
        for (std::uint32_t i = 0; i < splat_count; ++i) {
            const auto base = payload_start + i * kGaussianSplatSurfaceWords;
            if (!(out->surface_data[base + 5] & 1u)) continue;
            const auto source_id = out->surface_data[base + 4];
            auto found = surface_ids.find(source_id);
            if (source_id == 0u || found == surface_ids.end()) {
                if (next_surface_id == 0u) {
                    utility::LogError("Gaussian surface identity overflow.");
                }
                const auto id = next_surface_id++;
                out->surface_data[base + 4] = id;
                if (source_id) surface_ids[source_id] = id;
            } else {
                out->surface_data[base + 4] = found->second;
            }
        }

        SetVisibilityMaskRange(&out->visibility_mask, out->splat_count,
                               splat_count, item.visible);
        out->splat_count += splat_count;
        out->antialias = out->antialias || src.antialias;
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
