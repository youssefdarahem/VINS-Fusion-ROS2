# VINS Stability Configuration Changes

## Changes Made to lighthouse_mono_imu_config.yaml

### 1. GPU Acceleration (Disabled for Stability)

- `use_gpu_ceres: 0` (was 1) - GPU Ceres can cause instability

### 2. Extrinsic Calibration Strategy

- `estimate_extrinsic: 1` (was 2) - Use initial guess and optimize (more stable than fully online calibration)

### 3. Feature Tracking (More Conservative)

- `max_cnt: 150` (was 100) - More features for robustness
- `min_dist: 30` (was 20) - Better feature distribution
- `fisheye: 0` (was 1) - Disable fisheye processing if not needed

### 4. Optimization Parameters (More Time for Convergence)

- `max_solver_time: 0.08` (was 0.04) - Double the solver time
- `max_num_iterations: 8` (was 4) - More iterations for convergence

### 5. IMU Noise Parameters (More Conservative)

- `acc_n: 0.2` (was 0.08) - Higher accelerometer noise assumption
- `gyr_n: 0.05` (was 0.004) - Higher gyroscope noise assumption
- `acc_w: 0.02` (was 0.00004) - Higher accelerometer bias noise
- `gyr_w: 4.0e-5` (was 2.0e-6) - Higher gyroscope bias noise

### 6. Loop Closure (Disabled for Initial Testing)

- `loop_closure: 0` (was 1) - Disable to reduce complexity
- `fast_relocalization: 0` (was 1) - Disable for stability

## Why These Changes Help Numerical Stability:

1. **Conservative IMU parameters** - Higher noise assumptions make the system more robust to noisy IMU data
2. **More solver time** - Allows better convergence before timeout
3. **Disable complex features** - Loop closure and GPU acceleration can introduce instability
4. **Better extrinsic handling** - Initial guess + optimization is more stable than pure online estimation

## Test the Configuration:

Run VINS again with these settings. If still unstable, we can:

- Further increase IMU noise parameters
- Reduce feature count
- Adjust keyframe selection threshold
