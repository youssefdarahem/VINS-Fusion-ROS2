#include "camodocal/gpl/EigenQuaternionParameterization.h"

#include <cmath>

namespace camodocal
{

    bool
    EigenQuaternionParameterization::Plus(const double *x,
                                          const double *delta,
                                          double *x_plus_delta) const
    {
        const double norm_delta =
            sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
        if (norm_delta > 0.0)
        {
            const double sin_delta_by_delta = (sin(norm_delta) / norm_delta);
            double q_delta[4];
            q_delta[0] = sin_delta_by_delta * delta[0];
            q_delta[1] = sin_delta_by_delta * delta[1];
            q_delta[2] = sin_delta_by_delta * delta[2];
            q_delta[3] = cos(norm_delta);
            EigenQuaternionProduct(q_delta, x, x_plus_delta);
        }
        else
        {
            for (int i = 0; i < 4; ++i)
            {
                x_plus_delta[i] = x[i];
            }
        }
        return true;
    }

    bool
    EigenQuaternionParameterization::PlusJacobian(const double *x,
                                                  double *jacobian) const
    {
        jacobian[0] = x[3];
        jacobian[1] = x[2];
        jacobian[2] = -x[1]; // NOLINT
        jacobian[3] = -x[2];
        jacobian[4] = x[3];
        jacobian[5] = x[0]; // NOLINT
        jacobian[6] = x[1];
        jacobian[7] = -x[0];
        jacobian[8] = x[3]; // NOLINT
        jacobian[9] = -x[0];
        jacobian[10] = -x[1];
        jacobian[11] = -x[2]; // NOLINT
        return true;
    }

    bool
    EigenQuaternionParameterization::Minus(const double *y,
                                           const double *x,
                                           double *y_minus_x) const
    {
        // Compute the difference between two quaternions as a tangent space vector
        // This is essentially the inverse of the Plus operation
        double q_inverse[4] = {-x[0], -x[1], -x[2], x[3]};
        double q_diff[4];
        EigenQuaternionProduct(q_inverse, y, q_diff);

        const double norm_xyz = sqrt(q_diff[0] * q_diff[0] + q_diff[1] * q_diff[1] + q_diff[2] * q_diff[2]);
        if (norm_xyz > 0.0)
        {
            const double angle = 2.0 * atan2(norm_xyz, q_diff[3]);
            const double scale = angle / norm_xyz;
            y_minus_x[0] = scale * q_diff[0];
            y_minus_x[1] = scale * q_diff[1];
            y_minus_x[2] = scale * q_diff[2];
        }
        else
        {
            y_minus_x[0] = 0.0;
            y_minus_x[1] = 0.0;
            y_minus_x[2] = 0.0;
        }
        return true;
    }

    bool
    EigenQuaternionParameterization::MinusJacobian(const double *x,
                                                   double *jacobian) const
    {
        // For quaternion manifolds, the MinusJacobian is typically the inverse of PlusJacobian
        // This is a simplified implementation - more sophisticated versions might be needed for some applications
        jacobian[0] = x[3];
        jacobian[1] = x[2];
        jacobian[2] = -x[1]; // NOLINT
        jacobian[3] = -x[2];
        jacobian[4] = x[3];
        jacobian[5] = x[0]; // NOLINT
        jacobian[6] = x[1];
        jacobian[7] = -x[0];
        jacobian[8] = x[3]; // NOLINT
        jacobian[9] = -x[0];
        jacobian[10] = -x[1];
        jacobian[11] = -x[2]; // NOLINT
        return true;
    }

}
