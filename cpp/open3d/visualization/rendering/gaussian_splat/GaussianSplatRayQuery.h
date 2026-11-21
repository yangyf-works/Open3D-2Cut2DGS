// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "open3d/core/Tensor.h"

namespace open3d {
namespace visualization {
namespace rendering {

/// Synchronous finite-surface ray query using the renderer's shared GLSL
/// intersection routine. All input/output tensors are explicitly CPU tensors;
/// upload, Vulkan execution and readback happen in Query(). This experimental
/// API does not infer material intervals or stop at a visible surface.
///
/// Calls must run on the rendering thread without concurrent renderer work.
/// Close the query before destroying an OffscreenRenderer or the GUI engine,
/// because their teardown destroys the shared Vulkan device.
class GaussianSplatRayQuery {
public:
    /// Resources must be initialized, as by gui::InitializeForPython().
    /// Vulkan initialization is lazy; no CPU/Metal fallback is performed.
    GaussianSplatRayQuery();
    ~GaussianSplatRayQuery();
    GaussianSplatRayQuery(const GaussianSplatRayQuery&) = delete;
    GaussianSplatRayQuery& operator=(const GaussianSplatRayQuery&) = delete;

    /// Query candidate rectangles using CPU Float32 [R,4,4] geometry,
    /// CPU UInt32 [R,8] spans, and CPU Float32 [P,4] clipping planes.
    /// Geometry rows 0..2 are projective/depth rows shared with composite;
    /// row 3 is (grid_origin_x, grid_origin_y, pitch, intersection_epsilon).
    /// Spans are (begin,end,start_i,start_j,width_j,plane_begin,plane_count,0),
    /// with contiguous non-overlapping candidate ranges starting at zero.
    /// Planes accept dot(plane.xy, local_uv) >= plane.z; w is reserved.
    /// Returns CPU Float32 [C,4] (depth,u,v,rho); missed depths are NaN.
    /// There is no per-ray layer cap. Oversized buffers fail explicitly.
    core::Tensor Query(const core::Tensor& geometry,
                       const core::Tensor& spans,
                       const core::Tensor& planes);

    /// Release this query's buffers and pipelines, retaining the shared device.
    /// This is idempotent. A closed query cannot be reused.
    void Close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
