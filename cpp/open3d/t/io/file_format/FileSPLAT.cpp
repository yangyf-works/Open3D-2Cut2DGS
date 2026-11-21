// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/io/PointCloudIO.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace io {

// The three-axis disk format cannot represent the strict isotropic-scale schema.
// Reject it before touching files or destination data; use PLY or NPZ instead.
bool ReadPointCloudFromSPLAT(const std::string&,
                             geometry::PointCloud&,
                             const ReadPointCloudOption&) {
    utility::LogError(
            "SPLAT is not supported for 2DGS: the format requires three scale "
            "components. Use PLY (scale_0) or NPZ (scale shape (N, 1)).");
}

bool WritePointCloudToSPLAT(const std::string&,
                            const geometry::PointCloud&,
                            const WritePointCloudOption&) {
    utility::LogError(
            "SPLAT is not supported for 2DGS: the format requires three scale "
            "components. Use PLY (scale_0) or NPZ (scale shape (N, 1)).");
}

}  // namespace io
}  // namespace t
}  // namespace open3d
