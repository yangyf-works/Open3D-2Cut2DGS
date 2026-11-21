// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CPU data structures and packing helpers shared by the Vulkan and Metal
// Gaussian splat compute backends.  The structures mirror the std140/std430
// layouts expected by the GLSL/MSL compute shaders.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "open3d/t/geometry/GaussianSplatClipping.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatRenderer.h"

namespace open3d {
namespace visualization {
namespace rendering {

// ----- GPU-compatible structure definitions ----------------------------------
// These must be tightly packed to match the std140/std430 GLSL layouts.

/// vec4 stored as 4 floats (std430 layout).
struct Std430Vec4 {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
};
static_assert(sizeof(Std430Vec4) == 16,
              "Float32 position/rotation arrays must match GLSL vec4 stride");

/// vec2 pairs avoid padding in the compact projected surfel buffer.
struct Std430Vec2 {
    float x = 0.f, y = 0.f;
};

/// Matches GaussianViewParams uniform block in the compute shaders (std140).
/// Named to match the GLSL block: `layout(std140, binding=0) uniform
/// GaussianViewParams`.
struct alignas(16) GaussianViewParams {
    // mat4 world_from_model (column-major, 4 vec4 columns)
    float world_from_model[16];
    // mat4 view_from_world
    float view_from_world[16];
    // mat4 clip_from_view
    float clip_from_view[16];
    // vec4 camera_position_and_near
    float camera_position_and_near[4];
    // vec4 viewport_origin_and_size
    float viewport_origin_and_size[4];
    // uvec4 scene  (x=splat_count, y=reserved (zero),
    // z=antialias_flag, w=screen_y_down_flag)
    std::uint32_t scene[4];
    // uvec4 tiles  (xy=tile_size, zw=tile_count)
    std::uint32_t tiles[4];
    // uvec4 limits (x=entry_capacity, y=max_tiles_per_splat,
    // z=max_tile_entries_total, w=tile_key_bits T = ceil(log2(tile_count)),
    // clamped to [1,31]: T bits for tile index, (32-T) bits for depth)
    std::uint32_t limits[4];
    // vec4 depth_range_and_flags (x=near, y=far, z=composite_over_base,
    // w=scene_depth_present)
    float depth_range_and_flags[4];
    // vec4 surface_lighting (ambient, diffuse, specular, shininess)
    float surface_lighting[4];
};
static_assert(sizeof(GaussianViewParams) == 304,
              "GaussianViewParams must be 304 bytes to match GLSL layout");
static_assert(offsetof(GaussianViewParams, surface_lighting) == 288,
              "Scene lighting must match the final std140 vec4");

/// Composite-pass surfel data, matching std430 vec2 alignment (binding 6).
/// Homogeneous pixel-space rows retain perspective-correct local coordinates;
/// clipping retains Float32 precision for evaluation; metadata follows it.
struct alignas(8) ProjectedComposite {
    alignas(8) Std430Vec2 row_x_xy, row_x_zw;
    alignas(8) Std430Vec2 row_y_xy, row_y_zw;
    alignas(8) Std430Vec2 row_w_xy, row_w_zw;
    alignas(8) Std430Vec2 clip1_angle_split, clip2_angle_split;
    std::uint32_t surface_id, surface_flags;
    alignas(8) Std430Vec2 surface_normal_xy, surface_normal_z_sigma;
    alignas(8) Std430Vec2 surface_color_rg, surface_color_b_reserved;
};
static_assert(sizeof(ProjectedComposite) == 104 && alignof(ProjectedComposite) == 8,
              "ProjectedComposite must match the compact GLSL layout");
static_assert(offsetof(ProjectedComposite, row_x_xy) == 0 &&
              offsetof(ProjectedComposite, row_x_zw) == 8 &&
              offsetof(ProjectedComposite, row_y_xy) == 16 &&
              offsetof(ProjectedComposite, row_y_zw) == 24 &&
              offsetof(ProjectedComposite, row_w_xy) == 32 &&
              offsetof(ProjectedComposite, row_w_zw) == 40 &&
              offsetof(ProjectedComposite, clip1_angle_split) == 48 &&
              offsetof(ProjectedComposite, clip2_angle_split) == 56 &&
              offsetof(ProjectedComposite, surface_id) == 64 &&
              offsetof(ProjectedComposite, surface_flags) == 68 &&
              offsetof(ProjectedComposite, surface_normal_xy) == 72 &&
              offsetof(ProjectedComposite, surface_normal_z_sigma) == 80 &&
              offsetof(ProjectedComposite, surface_color_rg) == 88 &&
              offsetof(ProjectedComposite, surface_color_b_reserved) == 96,
              "ProjectedComposite field offsets must match GLSL");

// Nine uint32 words: Float32 clip bits (0..3), identity/flags (4/5), RGB (6..8).
// Shared Phong coefficients are held once per scene in GaussianViewParams.
inline constexpr std::size_t kGaussianSplatSurfaceWords = 9;
inline constexpr float kGaussianSplatFixedAlpha = 0.99f;

/// Shared appearance plus optional reconstruction identity for n samples.
/// Colors are required for every nonempty input in either mode.
/// Zero/missing identity isolates a sample. Positive IDs are object-local.
struct GaussianSplatSurfaceData {
    bool opaque = false;
    const std::int32_t* ids = nullptr;
    const float* colors = nullptr;
};
using GaussianSplatClipData = t::geometry::GaussianSplatClipData;

/// Global counters written GPU-side (prefix-sum, scatter, dispatch-args) and
/// read CPU-side for diagnostics and error reporting.  Layout matches the
/// GaussianGlobalCounters SSBO block in the compute shaders (std430, binding
/// 10).
struct GaussianGpuCounters {
    std::uint32_t total_entries =
            0;                      ///< Raw total tile entries from prefix-sum.
    std::uint32_t error_flags = 0;  ///< GPU error bitmask (kGaussianGpuError*).
    std::uint32_t tile_count = 0;   ///< Total tile count for the frame.
    std::uint32_t splat_count = 0;  ///< Visible splat count for the frame.
};
static_assert(sizeof(GaussianGpuCounters) == 4 * sizeof(std::uint32_t),
              "GaussianGpuCounters must be 16 bytes");
static constexpr std::size_t kGaussianCounterCount =
        sizeof(GaussianGpuCounters) / sizeof(std::uint32_t);  // 4

// Index constants for code that accesses the counters buffer as a raw uint32
// array rather than through the struct.
static constexpr std::size_t kGaussianCounterTotalEntriesIndex = 0;
static constexpr std::size_t kGaussianCounterErrorFlagsIndex = 1;
static constexpr std::size_t kGaussianCounterTileCountIndex = 2;
static constexpr std::size_t kGaussianCounterSplatCountIndex = 3;

/// Set when the shared sort-entry buffer overflows (total entries > capacity).
inline constexpr std::uint32_t kGaussianGpuErrorTileEntryOverflow = 1u << 0;
/// Set when the radix-sort dispatch count is clamped to the entry capacity.
inline constexpr std::uint32_t kGaussianGpuErrorSortCountClamped = 1u << 1;
/// Set when a splat's tile footprint exceeded max_tiles_per_splat and the
/// splat was culled (dropped) rather than rendered.
inline constexpr std::uint32_t kGaussianGpuErrorMaxTilesPerSplatExceeded = 1u
                                                                           << 2;
inline constexpr std::uint32_t kGaussianGpuErrorKnownMask =
        kGaussianGpuErrorTileEntryOverflow | kGaussianGpuErrorSortCountClamped |
        kGaussianGpuErrorMaxTilesPerSplatExceeded;

static_assert(sizeof(Std430Vec2) == 8 && sizeof(Std430Vec4) == 16,
              "Geometry buffers must match GLSL vec2/vec4 array strides");
static_assert(offsetof(Std430Vec2, x) == 0 &&
              offsetof(Std430Vec2, y) == 4 &&
              offsetof(Std430Vec4, x) == 0 &&
              offsetof(Std430Vec4, y) == 4 &&
              offsetof(Std430Vec4, z) == 8 &&
              offsetof(Std430Vec4, w) == 12,
              "Geometry components must retain their FP32 shader offsets");

// ----- CPU-side representation -----------------------------------------------

/// GPU-ready per-splat packed attribute arrays, created once at scene-cache
/// time and uploaded to GPU SSBOs only when the scene content changes.  Stores
/// the same bit formats as the SSBO bindings consumed by gaussian_project.comp:
///   positions       — fp32 vec4 (x,y,z,isotropic scale), 16 B/splat
///   rotations       — Float32 wxyz, 16 B/splat
///   surface_data    — clips, identity and RGB, 36 B/splat
/// Position, scale, quaternion and canonical clips retain Float32 precision;
/// the shader normalizes the quaternion without an intermediate half format.
struct GaussianSplatPackedAttrs {
    /// Monotonic scene revision used by per-view GPU upload caches.
    std::uint64_t revision = 0;
    std::vector<Std430Vec4> positions;  ///< fp32 XYZ + scalar scale, 16 B/splat
    /// Float32 (w,x,y,z), stored without quantization, 16 B/splat.
    std::vector<Std430Vec4> rotations;
    std::vector<std::uint32_t> surface_data;  ///< 36 B/splat, clips in first 16 B
    /// Bit-packed visibility mask: bit k = 1 → render splat k, 0 → cull (via
    /// WriteInvalidProjection). Size = ceil(splat_count / 32) uint32 words.
    /// All bits set when all splats are visible.
    std::vector<std::uint32_t> visibility_mask;
    std::uint32_t splat_count = 0;
    bool antialias = false;
    /// CPU scene metadata, never replicated in per-GS GPU buffers.
    Eigen::Vector3f surface_lighting = Eigen::Vector3f(1.f, 0.f, 0.f);
    /// Shared specular exponent, packed into the existing per-frame UBO lane.
    float surface_shininess = 2.f;
};

/// One packed Gaussian-splat source to be merged into a scene-level packed
/// buffer.
struct GaussianSplatMergeItem {
    const GaussianSplatPackedAttrs* attrs = nullptr;
    bool visible = true;
};

/// Per-frame packed view-parameters for GPU upload.
/// Contains only the small per-frame data (UBO + scalar counters).  The large
/// per-splat attribute arrays live in GaussianSplatPackedAttrs (scene
/// lifetime).
struct PackedGaussianScene {
    GaussianViewParams view_params;
    std::uint32_t splat_count = 0;
    std::uint32_t tile_count = 0;
    std::uint32_t pixel_count = 0;
    bool valid = false;
};

// ----- GPU buffer sizing (absorbed from GaussianSplatBuffers) ----------------

/// UBO stride for radix-sort params shared by both GPU backends and the
/// radix-sort dispatch shaders.
inline constexpr std::uint32_t kGaussianRadixParamsStride = 256;

/// Byte sizes and capacities for Gaussian splat SSBOs/UBOs.
/// Must match allocation logic in gaussian_compute_dispatch_args.comp and the
/// radix shaders (tile cap, etc.).
struct GaussianGpuBufferSizes {
    /// Composite-pass projected data (104 B/splat, binding 6).
    std::size_t projected_composite_size = 0;
    /// steal_counter: one uint32 used by the composite work-stealing dispatch.
    /// (Previously: per-tile count/offset arrays of size tile_count×4 B each.)
    std::size_t tile_scalar_size = 0;
    /// Sort key/value ping-pong buffers (entry_capacity × 4 B each, ×2 arrays).
    std::size_t key_cap_size = 0;
    std::size_t histogram_buf_size = 0;
    /// 8 indirect dispatch entries (4×histogram + 4×scatter).
    std::size_t dispatch_args_size = 0;
    std::size_t radix_params_size = 0;
    std::uint32_t entries_capacity = 0;
    std::uint32_t radix_num_wg_cap = 0;
};

/// Compute GPU buffer sizes from a packed-scene frame description.
/// Called once per geometry pass to size intermediate SSBO allocations.
void ComputeGaussianGpuBufferSizes(const PackedGaussianScene& packed,
                                   GaussianGpuBufferSizes* out);

// ----- Packing helpers -------------------------------------------------------

/// Pack camera, viewport, and scene metadata into the view-params UBO.
/// Uses the splat count and antialias flag from `attrs`.
/// Per-frame GPU cost: glBufferSubData / MTLBuffer memcpy of 304 bytes.
/// Tile-entry buffer capacity (`limits.x`) is sized from
/// `config.avg_tiles_per_splat`; the per-splat hard cap (`limits.y`) enforced
/// in `gaussian_project.comp` comes from `config.max_tiles_per_splat`.
PackedGaussianScene PackGaussianViewParams(
        const GaussianSplatPackedAttrs& attrs,
        const GaussianSplatRenderer::ViewRenderData& render_data,
        const GaussianSplatRenderer::RenderConfig& config);

/// Pack Gaussian splat attributes from raw PointCloud data pointers into
/// GPU-ready format in one pass, preserving every input. Called once at
/// scene cache time (FilamentScene::CacheGaussianSplatData) to eliminate the
/// intermediate fp32 copy that was previously re-packed every scene-change
/// frame.
/// @param n                 Total number of splats in source
/// @param scale_ptr         Non-null when n>0: linear isotropic tangent scale (1 float/splat).
/// @param rot_ptr           Non-null when n>0: unit quaternion (w,x,y,z).
/// @param antialias         Stored as `out.antialias` metadata.
/// @param clip_data         Optional n boundaries; null means unclipped 2DGS.
/// @param surface           Required colors when n>0, optional IDs.
/// Quaternion and normalized clips retain Float32 precision during transfer.
/// Both modes use unlit RGB in [0,1]; scene metadata holds the common lighting.
/// Surface interpolation uses Gaussian weights without per-splat opacity.
void PackGaussianSplatAttrsDirect(
        const float* pts_ptr,
        std::size_t n,
        const float* scale_ptr,
        const float* rot_ptr,
        bool antialias,
        GaussianSplatPackedAttrs& out,
        const GaussianSplatClipData* clip_data = nullptr,
        const GaussianSplatSurfaceData* surface = nullptr);

/// Merge multiple per-object packed Gaussian-splat buffers into one scene-level
/// packed buffer. Both rendering modes use the same fixed per-splat layout.
/// Nonempty sources, including hidden objects, must use identical lighting.
/// Conflicting or invalid coefficients are rejected before replacing output.
/// @param items        Merge sources in output order.
/// @param out          Output packed buffer (overwritten).
/// @param splat_starts Optional output; receives per-item start index in merged
///                     splat arrays. `splat_starts[i]` corresponds to
///                     `items[i]`.
void MergeGaussianSplatPackedAttrs(
        const std::vector<GaussianSplatMergeItem>& items,
        GaussianSplatPackedAttrs* out,
        std::vector<std::uint32_t>* splat_starts = nullptr);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
