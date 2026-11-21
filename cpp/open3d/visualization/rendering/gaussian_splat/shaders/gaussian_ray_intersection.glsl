// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef OPEN3D_GAUSSIAN_RAY_INTERSECTION_GLSL
#define OPEN3D_GAUSSIAN_RAY_INTERSECTION_GLSL

// Intersect a projected ray with a planar Gaussian in homogeneous local UV.
// Geometry is shared by the renderer and all-layer queries; support, visibility
// and opacity remain caller decisions so camera filtering cannot change solids.
bool GaussianRayLocalUV(vec2 pixel_center, vec3 row_x, vec3 row_y,
                        vec3 row_w, float epsilon, out vec2 local_uv) {
    vec3 plane_u = pixel_center.x * row_w - row_x;
    vec3 plane_v = pixel_center.y * row_w - row_y;
    vec3 local_h = cross(plane_u, plane_v);
    if (abs(local_h.z) < epsilon) return false;
    local_uv = local_h.xy / local_h.z;
    return !(any(isnan(local_uv)) || any(isinf(local_uv)));
}

// Preserve the renderer's strict outside test, including accepted boundaries.
bool GaussianRayOutsideClip(vec2 local_uv, vec3 clip_plane) {
    return dot(clip_plane.xy, local_uv) < clip_plane.z;
}

// Evaluate actual depth on the plane, not the Gaussian center depth.
float GaussianRayDepth(vec2 local_uv, float depth_u, float depth_v,
                       float depth_c) {
    return dot(vec3(depth_u, depth_v, depth_c), vec3(local_uv, 1.0));
}

#endif
