// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CPU normalization for the compact two-boundary surfel representation. These
// helpers preserve normalized local coordinates and never change the Gaussian.
#pragma once

#include <cmath>
#include <stdexcept>

namespace open3d {
namespace t {
namespace geometry {

struct GaussianSplatClipData {
    float clip1_angle = 4.f;
    float clip1_split = 0.f;
    float clip2_angle = 4.f;
    float clip2_split = 0.f;
};
static_assert(sizeof(GaussianSplatClipData) == 16,
              "Gaussian clipping requires exactly four FP32 values");

/// Normalize finite angles, retaining the exact inactive sentinel. Rounding
/// back to Float32 must not leave the excluded +pi endpoint in the result.
inline float NormalizeGaussianClipAngle(float angle) {
    if (!std::isfinite(angle)) {
        throw std::invalid_argument("Gaussian clip angles must be finite.");
    }
    if (angle == 4.f) return angle;
    constexpr double pi = 3.14159265358979323846;
    double value = std::remainder(static_cast<double>(angle), 2.0 * pi);
    float result = static_cast<float>(value);
    if (result >= static_cast<float>(pi)) result = -static_cast<float>(pi);
    return result;
}

/// Validate both boundaries before packing, including invisible Gaussians.
/// Inactive offsets are canonical zero; a second-only boundary is rejected.
inline GaussianSplatClipData NormalizeGaussianClipData(
        GaussianSplatClipData clip) {
    clip.clip1_angle = NormalizeGaussianClipAngle(clip.clip1_angle);
    clip.clip2_angle = NormalizeGaussianClipAngle(clip.clip2_angle);
    if (!std::isfinite(clip.clip1_split) || !std::isfinite(clip.clip2_split)) {
        throw std::invalid_argument("Gaussian clip splits must be finite.");
    }
    if (clip.clip1_angle == 4.f && clip.clip2_angle != 4.f) {
        throw std::invalid_argument("Gaussian second boundary requires the first boundary.");
    }
    if (clip.clip1_angle == 4.f) clip.clip1_split = 0.f;
    if (clip.clip2_angle == 4.f) clip.clip2_split = 0.f;
    return clip;
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
