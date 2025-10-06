/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

class PoseLocalParameterization : public ceres::Manifold
{
public:
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool PlusJacobian(const double *x, double *jacobian) const;
    virtual bool Minus(const double *y, const double *x, double *y_minus_x) const;
    virtual bool MinusJacobian(const double *x, double *jacobian) const;
    virtual int AmbientSize() const { return 7; };
    virtual int TangentSize() const { return 6; };

    // Legacy method for backward compatibility
    virtual bool ComputeJacobian(const double *x, double *jacobian) const
    {
        return PlusJacobian(x, jacobian);
    }
};
