//! # Flow Matching for Voice Synthesis
//!
//! This module implements Flow Matching, a cutting-edge alternative to diffusion models
//! that uses continuous normalizing flows and optimal transport theory for high-quality
//! voice synthesis with superior training stability and sampling efficiency.
//!
//! ## Key Features
//!
//! - **Optimal Transport**: Efficient straight paths between noise and data
//! - **Continuous Normalizing Flows**: Smooth, invertible transformations
//! - **Fast Sampling**: Direct path integration vs. iterative denoising
//! - **Training Stability**: More stable than score-based diffusion
//! - **Conditional Generation**: Speaker-conditioned synthesis
//!
//! ## Advantages over Diffusion Models
//!
//! - **Faster Training**: No variance schedule tuning required
//! - **Better Sample Quality**: Straighter paths = less error accumulation
//! - **Flexible ODE Solvers**: Euler, Heun, RK4, Dopri5 support
//! - **Theoretical Guarantees**: Optimal transport provides strong foundations
//!
//! ## References
//!
//! - Lipman et al. (2023). "Flow Matching for Generative Modeling"
//! - Liu et al. (2023). "Flow Straight and Fast: Learning to Generate and Transfer Data"
//! - Albergo & Vanden-Eijnden (2023). "Building Normalizing Flows with Stochastic Interpolants"

use crate::Error;
use candle_core::Device;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};

/// ODE solver type for flow integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OdeSolver {
    /// Euler method (first-order, fast)
    Euler,
    /// Heun's method (second-order)
    Heun,
    /// Classical Runge-Kutta (fourth-order)
    RK4,
    /// Dormand-Prince adaptive (fifth-order)
    Dopri5,
}

/// Flow matching method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowMatchingMethod {
    /// Conditional flow matching (optimal transport)
    ConditionalFlowMatching,
    /// Rectified flow
    RectifiedFlow,
    /// Stochastic interpolants
    StochasticInterpolants,
}

/// Configuration for flow matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMatchingConfig {
    /// Flow matching method
    pub method: FlowMatchingMethod,
    /// ODE solver for sampling
    pub ode_solver: OdeSolver,
    /// Number of integration steps
    pub num_steps: usize,
    /// Time discretization (0 = start, 1 = end)
    pub time_steps: Vec<f32>,
    /// Enable adaptive step size
    pub adaptive_steps: bool,
    /// Error tolerance for adaptive solver
    pub error_tolerance: f32,
    /// Enable velocity caching
    pub enable_caching: bool,
    /// Interpolation sigma (for stochastic interpolants)
    pub sigma: f32,
}

impl Default for FlowMatchingConfig {
    fn default() -> Self {
        Self {
            method: FlowMatchingMethod::ConditionalFlowMatching,
            ode_solver: OdeSolver::Heun,
            num_steps: 10,
            time_steps: Vec::new(), // Will be generated
            adaptive_steps: false,
            error_tolerance: 1e-3,
            enable_caching: true,
            sigma: 0.0,
        }
    }
}

impl FlowMatchingConfig {
    /// Generate time discretization
    fn generate_time_steps(&mut self) {
        if self.time_steps.is_empty() {
            self.time_steps = (0..=self.num_steps)
                .map(|i| i as f32 / self.num_steps as f32)
                .collect();
        }
    }
}

/// Flow matching model for voice synthesis
pub struct FlowMatchingModel {
    config: FlowMatchingConfig,
    device: Device,
    stats: FlowMatchingStats,
}

/// Statistics for flow matching
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlowMatchingStats {
    /// Total samples generated
    pub total_samples: usize,
    /// Average number of function evaluations
    pub avg_nfe: f32,
    /// Average synthesis time (milliseconds)
    pub avg_synthesis_time_ms: f32,
    /// Average quality score
    pub avg_quality: f32,
}

/// Result of flow matching synthesis
#[derive(Debug, Clone)]
pub struct FlowMatchingResult {
    /// Generated output
    pub output: Array2<f32>,
    /// Number of function evaluations
    pub nfe: usize,
    /// Synthesis time (milliseconds)
    pub synthesis_time_ms: f32,
    /// Quality estimate (0-1)
    pub quality_estimate: f32,
    /// Integration path (for visualization)
    pub trajectory: Vec<Array2<f32>>,
}

impl FlowMatchingModel {
    /// Create a new flow matching model
    pub fn new(mut config: FlowMatchingConfig, device: Device) -> Result<Self, Error> {
        info!(
            "Initializing flow matching model with method: {:?}",
            config.method
        );

        config.generate_time_steps();

        Ok(Self {
            config,
            device,
            stats: FlowMatchingStats::default(),
        })
    }

    /// Synthesize using flow matching
    pub fn synthesize(
        &mut self,
        speaker_embedding: &Array1<f32>,
        target_length: usize,
    ) -> Result<FlowMatchingResult, Error> {
        let start_time = std::time::Instant::now();

        info!(
            "Starting flow matching synthesis (length={})",
            target_length
        );

        // Initialize from noise
        let mut x = self.sample_prior(1, target_length)?;

        // Store trajectory for visualization
        let mut trajectory = vec![x.clone()];

        // Integrate ODE from t=0 to t=1
        let mut nfe = 0;
        let time_steps = self.config.time_steps.clone();

        // Dormand-Prince 4(5) adaptive solver takes a completely different path —
        // it controls its own step size rather than following the fixed time_steps grid.
        if let OdeSolver::Dopri5 = self.config.ode_solver {
            let result = self.integrate_dopri5(
                x,
                &time_steps,
                speaker_embedding,
                &mut nfe,
                trajectory,
                start_time,
            )?;
            return Ok(result);
        }

        // Fixed-step solvers: Euler, Heun, RK4
        for i in 0..time_steps.len() - 1 {
            let t = time_steps[i];
            let t_next = time_steps[i + 1];
            let dt = t_next - t;

            // Compute velocity field (first evaluation shared by all fixed-step solvers)
            let v = self.compute_velocity(&x, t, speaker_embedding)?;
            nfe += 1;

            x = match self.config.ode_solver {
                OdeSolver::Euler => {
                    // x_{t+dt} = x_t + dt * v_t
                    let mut x_next = x.clone();
                    for row in 0..x.dim().0 {
                        for col in 0..x.dim().1 {
                            x_next[[row, col]] += dt * v[[row, col]];
                        }
                    }
                    x_next
                }
                OdeSolver::Heun => {
                    // Predictor step
                    let mut x_pred = x.clone();
                    for row in 0..x.dim().0 {
                        for col in 0..x.dim().1 {
                            x_pred[[row, col]] += dt * v[[row, col]];
                        }
                    }

                    // Corrector step
                    let v_pred = self.compute_velocity(&x_pred, t_next, speaker_embedding)?;
                    nfe += 1;

                    let mut x_next = x.clone();
                    for row in 0..x.dim().0 {
                        for col in 0..x.dim().1 {
                            x_next[[row, col]] += dt * (v[[row, col]] + v_pred[[row, col]]) / 2.0;
                        }
                    }
                    x_next
                }
                OdeSolver::RK4 => {
                    // Classical Runge-Kutta 4th order
                    let k1 = v.clone();

                    let mut x_k2 = x.clone();
                    for row in 0..x.dim().0 {
                        for col in 0..x.dim().1 {
                            x_k2[[row, col]] += (dt / 2.0) * k1[[row, col]];
                        }
                    }
                    let k2 = self.compute_velocity(&x_k2, t + dt / 2.0, speaker_embedding)?;
                    nfe += 1;

                    let mut x_k3 = x.clone();
                    for row in 0..x.dim().0 {
                        for col in 0..x.dim().1 {
                            x_k3[[row, col]] += (dt / 2.0) * k2[[row, col]];
                        }
                    }
                    let k3 = self.compute_velocity(&x_k3, t + dt / 2.0, speaker_embedding)?;
                    nfe += 1;

                    let mut x_k4 = x.clone();
                    for row in 0..x.dim().0 {
                        for col in 0..x.dim().1 {
                            x_k4[[row, col]] += dt * k3[[row, col]];
                        }
                    }
                    let k4 = self.compute_velocity(&x_k4, t_next, speaker_embedding)?;
                    nfe += 1;

                    let mut x_next = x.clone();
                    for row in 0..x.dim().0 {
                        for col in 0..x.dim().1 {
                            x_next[[row, col]] += (dt / 6.0)
                                * (k1[[row, col]]
                                    + 2.0 * k2[[row, col]]
                                    + 2.0 * k3[[row, col]]
                                    + k4[[row, col]]);
                        }
                    }
                    x_next
                }
                // Dopri5 is handled above before this loop
                OdeSolver::Dopri5 => unreachable!(),
            };

            trajectory.push(x.clone());
        }

        let synthesis_time_ms = start_time.elapsed().as_millis() as f32;

        // Estimate quality
        let quality_estimate = self.estimate_quality(&x)?;

        // Update statistics
        self.stats.total_samples += 1;
        self.stats.avg_nfe = (self.stats.avg_nfe * (self.stats.total_samples - 1) as f32
            + nfe as f32)
            / self.stats.total_samples as f32;
        self.stats.avg_synthesis_time_ms = (self.stats.avg_synthesis_time_ms
            * (self.stats.total_samples - 1) as f32
            + synthesis_time_ms)
            / self.stats.total_samples as f32;
        self.stats.avg_quality = (self.stats.avg_quality * (self.stats.total_samples - 1) as f32
            + quality_estimate)
            / self.stats.total_samples as f32;

        info!(
            "Flow matching complete: {} NFE, {:.2}ms, quality={:.3}",
            nfe, synthesis_time_ms, quality_estimate
        );

        Ok(FlowMatchingResult {
            output: x,
            nfe,
            synthesis_time_ms,
            quality_estimate,
            trajectory,
        })
    }

    /// Dormand-Prince 4(5) adaptive ODE integration (DOPRI5 / RK45).
    ///
    /// Uses the full 7-stage Butcher tableau with embedded 4th/5th-order solutions
    /// for error-controlled adaptive step sizing from `t_start` to `t_end`.
    ///
    /// ## Butcher tableau (Dormand & Prince, 1980)
    ///
    /// Stage nodes (c):  0, 1/5, 3/10, 4/5, 8/9, 1, 1
    ///
    /// 5th-order weights (b):  35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
    /// 4th-order weights (b*): 5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40
    ///
    /// Step-size control: h_new = h * 0.9 * (1/err)^0.2
    #[allow(clippy::too_many_arguments)]
    fn integrate_dopri5(
        &mut self,
        mut x: Array2<f32>,
        time_steps: &[f32],
        condition: &Array1<f32>,
        nfe: &mut usize,
        mut trajectory: Vec<Array2<f32>>,
        start_time: std::time::Instant,
    ) -> Result<FlowMatchingResult, Error> {
        // ── Dormand-Prince Butcher coefficients ──────────────────────────────
        // Row 2
        const A21: f32 = 1.0 / 5.0;
        // Row 3
        const A31: f32 = 3.0 / 40.0;
        const A32: f32 = 9.0 / 40.0;
        // Row 4
        const A41: f32 = 44.0 / 45.0;
        const A42: f32 = -56.0 / 15.0;
        const A43: f32 = 32.0 / 9.0;
        // Row 5
        const A51: f32 = 19372.0 / 6561.0;
        const A52: f32 = -25360.0 / 2187.0;
        const A53: f32 = 64448.0 / 6561.0;
        const A54: f32 = -212.0 / 729.0;
        // Row 6
        const A61: f32 = 9017.0 / 3168.0;
        const A62: f32 = -355.0 / 33.0;
        const A63: f32 = 46732.0 / 5247.0;
        const A64: f32 = 49.0 / 176.0;
        const A65: f32 = -5103.0 / 18656.0;
        // Row 7  (= 5th-order solution weights b)
        const B1: f32 = 35.0 / 384.0;
        // B2 = 0
        const B3: f32 = 500.0 / 1113.0;
        const B4: f32 = 125.0 / 192.0;
        const B5: f32 = -2187.0 / 6784.0;
        const B6: f32 = 11.0 / 84.0;
        // B7 = 0  (FSAL: k7 of accepted step becomes k1 of next — not used here for simplicity)

        // 4th-order embedded weights (b*)
        const BS1: f32 = 5179.0 / 57600.0;
        // BS2 = 0
        const BS3: f32 = 7571.0 / 16695.0;
        const BS4: f32 = 393.0 / 640.0;
        const BS5: f32 = -92097.0 / 339200.0;
        const BS6: f32 = 187.0 / 2100.0;
        const BS7: f32 = 1.0 / 40.0;

        // Error coefficients  e_i = b_i - b*_i
        const E1: f32 = B1 - BS1;
        // E2 = 0
        const E3: f32 = B3 - BS3;
        const E4: f32 = B4 - BS4;
        const E5: f32 = B5 - BS5;
        const E6: f32 = B6 - BS6;
        const E7: f32 = -BS7; // b7 = 0

        // Node offsets  (c2..c6; c1=0, c7=1 not needed explicitly)
        const C2: f32 = 1.0 / 5.0;
        const C3: f32 = 3.0 / 10.0;
        const C4: f32 = 4.0 / 5.0;
        const C5: f32 = 8.0 / 9.0;
        // C6 = 1.0, C7 = 1.0

        let t_start = *time_steps.first().unwrap_or(&0.0_f32);
        let t_end = *time_steps.last().unwrap_or(&1.0_f32);
        let tol = self.config.error_tolerance;
        let (rows, cols) = x.dim();
        let n_elem = (rows * cols) as f32;

        // Initial step size: span / num_steps
        let span = t_end - t_start;
        let mut h = span / self.config.num_steps as f32;
        let h_min = h * 1e-4;
        let h_max = h * 10.0_f32;

        let mut t = t_start;

        debug!(
            "Dopri5 adaptive integration: t=[{}, {}], h0={:.4e}, tol={:.2e}",
            t_start, t_end, h, tol
        );

        // Safety limit: prevent runaway loops on degenerate vector fields
        const MAX_NFE: usize = 10_000;

        while t < t_end - 1e-10 {
            // Clamp final step to avoid overshooting t_end
            if t + h > t_end {
                h = t_end - t;
            }

            // ── Stage 1 ──────────────────────────────────────────────────────
            let k1 = self.compute_velocity(&x, t, condition)?;
            *nfe += 1;

            // ── Stage 2 ──────────────────────────────────────────────────────
            let mut x_s = Array2::zeros((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    x_s[[r, c]] = x[[r, c]] + h * A21 * k1[[r, c]];
                }
            }
            let k2 = self.compute_velocity(&x_s, t + C2 * h, condition)?;
            *nfe += 1;

            // ── Stage 3 ──────────────────────────────────────────────────────
            for r in 0..rows {
                for c in 0..cols {
                    x_s[[r, c]] = x[[r, c]] + h * (A31 * k1[[r, c]] + A32 * k2[[r, c]]);
                }
            }
            let k3 = self.compute_velocity(&x_s, t + C3 * h, condition)?;
            *nfe += 1;

            // ── Stage 4 ──────────────────────────────────────────────────────
            for r in 0..rows {
                for c in 0..cols {
                    x_s[[r, c]] =
                        x[[r, c]] + h * (A41 * k1[[r, c]] + A42 * k2[[r, c]] + A43 * k3[[r, c]]);
                }
            }
            let k4 = self.compute_velocity(&x_s, t + C4 * h, condition)?;
            *nfe += 1;

            // ── Stage 5 ──────────────────────────────────────────────────────
            for r in 0..rows {
                for c in 0..cols {
                    x_s[[r, c]] = x[[r, c]]
                        + h * (A51 * k1[[r, c]]
                            + A52 * k2[[r, c]]
                            + A53 * k3[[r, c]]
                            + A54 * k4[[r, c]]);
                }
            }
            let k5 = self.compute_velocity(&x_s, t + C5 * h, condition)?;
            *nfe += 1;

            // ── Stage 6 ──────────────────────────────────────────────────────
            for r in 0..rows {
                for c in 0..cols {
                    x_s[[r, c]] = x[[r, c]]
                        + h * (A61 * k1[[r, c]]
                            + A62 * k2[[r, c]]
                            + A63 * k3[[r, c]]
                            + A64 * k4[[r, c]]
                            + A65 * k5[[r, c]]);
                }
            }
            let k6 = self.compute_velocity(&x_s, t + h, condition)?;
            *nfe += 1;

            // ── 5th-order solution (advancing state) ─────────────────────────
            let mut x5 = Array2::zeros((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    x5[[r, c]] = x[[r, c]]
                        + h * (B1 * k1[[r, c]]
                            + B3 * k3[[r, c]]
                            + B4 * k4[[r, c]]
                            + B5 * k5[[r, c]]
                            + B6 * k6[[r, c]]);
                }
            }

            // ── Stage 7 (needed for 4th-order error estimate only) ────────────
            let k7 = self.compute_velocity(&x5, t + h, condition)?;
            *nfe += 1;

            // ── Error estimate: RMS of (x5 - x4) / tol ───────────────────────
            // x5 - x4 = h * sum_i e_i * k_i   (per element)
            let mut err_sum_sq = 0.0_f32;
            for r in 0..rows {
                for c in 0..cols {
                    let diff = h
                        * (E1 * k1[[r, c]]
                            + E3 * k3[[r, c]]
                            + E4 * k4[[r, c]]
                            + E5 * k5[[r, c]]
                            + E6 * k6[[r, c]]
                            + E7 * k7[[r, c]]);
                    // Scale relative to max(|x5|, 1) for mixed absolute/relative control
                    let scale = x5[[r, c]].abs().max(1.0) * tol;
                    err_sum_sq += (diff / scale).powi(2);
                }
            }
            let err = (err_sum_sq / n_elem).sqrt();

            if err <= 1.0 {
                // ── Accept step ───────────────────────────────────────────────
                x = x5;
                t += h;
                trajectory.push(x.clone());
                debug!("Dopri5 accept: t={:.4}, h={:.4e}, err={:.3e}", t, h, err);
            }
            // ── Adjust step size (PI controller exponent 0.2 = 1/5) ──────────
            let factor = if err < 1e-10 {
                5.0_f32 // Maximum growth
            } else {
                0.9 * (1.0_f32 / err).powf(0.2)
            };
            h = (h * factor).clamp(h_min, h_max);

            if *nfe >= MAX_NFE {
                debug!("Dopri5: NFE safety limit reached at t={:.4}", t);
                break;
            }
        }

        let synthesis_time_ms = start_time.elapsed().as_millis() as f32;
        let quality_estimate = self.estimate_quality(&x)?;

        self.stats.total_samples += 1;
        self.stats.avg_nfe = (self.stats.avg_nfe * (self.stats.total_samples - 1) as f32
            + *nfe as f32)
            / self.stats.total_samples as f32;
        self.stats.avg_synthesis_time_ms = (self.stats.avg_synthesis_time_ms
            * (self.stats.total_samples - 1) as f32
            + synthesis_time_ms)
            / self.stats.total_samples as f32;
        self.stats.avg_quality = (self.stats.avg_quality * (self.stats.total_samples - 1) as f32
            + quality_estimate)
            / self.stats.total_samples as f32;

        info!(
            "Dopri5 complete: {} NFE, {:.2}ms, quality={:.3}",
            nfe, synthesis_time_ms, quality_estimate
        );

        Ok(FlowMatchingResult {
            output: x,
            nfe: *nfe,
            synthesis_time_ms,
            quality_estimate,
            trajectory,
        })
    }

    /// Compute velocity field at time t
    fn compute_velocity(
        &self,
        x_t: &Array2<f32>,
        t: f32,
        condition: &Array1<f32>,
    ) -> Result<Array2<f32>, Error> {
        let (batch_size, length) = x_t.dim();

        match self.config.method {
            FlowMatchingMethod::ConditionalFlowMatching => {
                // Conditional flow: v_t(x) = (x_1 - x_t) / (1 - t)
                // where x_1 is the target conditioned on speaker embedding
                let x_target = self.generate_target(condition, length)?;

                let denominator = (1.0 - t).max(1e-6);
                let mut velocity = Array2::zeros((batch_size, length));

                for i in 0..batch_size {
                    for j in 0..length {
                        velocity[[i, j]] = (x_target[[i, j]] - x_t[[i, j]]) / denominator;
                    }
                }

                Ok(velocity)
            }
            FlowMatchingMethod::RectifiedFlow => {
                // Rectified flow: v_t(x) = x_1 - x_0
                // Straight path from noise to data
                let x_target = self.generate_target(condition, length)?;
                let x_source = self.sample_prior(batch_size, length)?;

                let mut velocity = Array2::zeros((batch_size, length));
                for i in 0..batch_size {
                    for j in 0..length {
                        velocity[[i, j]] = x_target[[i, j]] - x_source[[i, j]];
                    }
                }

                Ok(velocity)
            }
            FlowMatchingMethod::StochasticInterpolants => {
                // Stochastic interpolant: x_t = (1-t) * x_0 + t * x_1 + sigma * noise
                let x_target = self.generate_target(condition, length)?;

                let mut velocity = Array2::zeros((batch_size, length));
                for i in 0..batch_size {
                    for j in 0..length {
                        // Simple velocity: derivative of linear interpolation
                        velocity[[i, j]] = x_target[[i, j]] - x_t[[i, j]];

                        // Add stochastic component
                        if self.config.sigma > 0.0 {
                            let mut rng = scirs2_core::random::thread_rng();
                            let noise: f32 = rng.random_range(-1.0..1.0);
                            velocity[[i, j]] += self.config.sigma * noise;
                        }
                    }
                }

                Ok(velocity)
            }
        }
    }

    /// Sample from prior distribution (Gaussian noise)
    fn sample_prior(&self, batch_size: usize, length: usize) -> Result<Array2<f32>, Error> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut noise = Array2::zeros((batch_size, length));

        for i in 0..batch_size {
            for j in 0..length {
                noise[[i, j]] = rng.random_range(-1.0..1.0);
            }
        }

        Ok(noise)
    }

    /// Generate target conditioned on speaker embedding
    fn generate_target(
        &self,
        condition: &Array1<f32>,
        length: usize,
    ) -> Result<Array2<f32>, Error> {
        // Placeholder: In practice, use learned conditional model
        let mut target = Array2::zeros((1, length));

        for j in 0..length {
            let cond_idx = j % condition.len();
            target[[0, j]] = condition[cond_idx] * 0.5;
        }

        Ok(target)
    }

    /// Estimate synthesis quality
    fn estimate_quality(&self, output: &Array2<f32>) -> Result<f32, Error> {
        // Simple quality metric based on smoothness and range
        let mut smoothness = 0.0;

        for i in 0..output.dim().0 {
            for j in 1..output.dim().1 {
                smoothness += (output[[i, j]] - output[[i, j - 1]]).abs();
            }
        }

        smoothness = 1.0 - (smoothness / (output.len() as f32)).min(1.0);

        // Check if values are in reasonable range
        let mut in_range = 0;
        for value in output.iter() {
            if value.abs() <= 1.0 {
                in_range += 1;
            }
        }
        let range_score = in_range as f32 / output.len() as f32;

        let quality = (0.7 * smoothness + 0.3 * range_score).clamp(0.0, 1.0);

        Ok(quality)
    }

    /// Get statistics
    pub fn get_stats(&self) -> &FlowMatchingStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = FlowMatchingStats::default();
    }
}

/// Builder for flow matching model
pub struct FlowMatchingBuilder {
    config: FlowMatchingConfig,
    device: Option<Device>,
}

impl FlowMatchingBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: FlowMatchingConfig::default(),
            device: None,
        }
    }

    /// Set flow matching method
    pub fn method(mut self, method: FlowMatchingMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set ODE solver
    pub fn ode_solver(mut self, solver: OdeSolver) -> Self {
        self.config.ode_solver = solver;
        self
    }

    /// Set number of integration steps
    pub fn num_steps(mut self, steps: usize) -> Self {
        self.config.num_steps = steps;
        self
    }

    /// Enable adaptive step size
    pub fn adaptive_steps(mut self, enable: bool) -> Self {
        self.config.adaptive_steps = enable;
        self
    }

    /// Set device
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Build the model
    pub fn build(self) -> Result<FlowMatchingModel, Error> {
        let device = self.device.unwrap_or(Device::Cpu);
        FlowMatchingModel::new(self.config, device)
    }
}

impl Default for FlowMatchingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_matching_creation() {
        let model = FlowMatchingBuilder::new().build();
        assert!(model.is_ok());
    }

    #[test]
    fn test_flow_matching_synthesis() {
        let mut model = FlowMatchingBuilder::new().num_steps(5).build().unwrap();

        let speaker_embedding = Array1::from_vec(vec![0.5; 256]);
        let result = model.synthesize(&speaker_embedding, 100);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.output.dim(), (1, 100));
        assert!(result.nfe > 0);
        assert!(result.synthesis_time_ms >= 0.0);
        assert!(result.quality_estimate >= 0.0 && result.quality_estimate <= 1.0);
    }

    #[test]
    fn test_different_ode_solvers() {
        for solver in [
            OdeSolver::Euler,
            OdeSolver::Heun,
            OdeSolver::RK4,
            OdeSolver::Dopri5,
        ] {
            let mut model = FlowMatchingBuilder::new()
                .ode_solver(solver)
                .num_steps(3)
                .build()
                .unwrap();

            let speaker_embedding = Array1::from_vec(vec![0.5; 256]);
            let result = model.synthesize(&speaker_embedding, 50);

            assert!(result.is_ok(), "Failed with solver: {:?}", solver);
        }
    }

    #[test]
    fn test_dopri5_solver() {
        let mut model = FlowMatchingBuilder::new()
            .ode_solver(OdeSolver::Dopri5)
            .num_steps(5)
            .build()
            .unwrap();

        let speaker_embedding = Array1::from_vec(vec![0.5; 256]);
        let result = model.synthesize(&speaker_embedding, 50).unwrap();

        assert_eq!(result.output.dim(), (1, 50));
        assert!(result.nfe > 0);
        assert!(
            result.quality_estimate >= 0.0 && result.quality_estimate <= 1.0,
            "Quality out of [0,1]: {}",
            result.quality_estimate
        );

        // Dopri5 should use more NFE than Euler (7 stage evaluations per step)
        let mut euler_model = FlowMatchingBuilder::new()
            .ode_solver(OdeSolver::Euler)
            .num_steps(5)
            .build()
            .unwrap();
        let result_euler = euler_model.synthesize(&speaker_embedding, 50).unwrap();
        assert!(
            result.nfe >= result_euler.nfe,
            "Dopri5 NFE ({}) should be >= Euler NFE ({})",
            result.nfe,
            result_euler.nfe
        );
    }

    #[test]
    fn test_different_flow_methods() {
        for method in [
            FlowMatchingMethod::ConditionalFlowMatching,
            FlowMatchingMethod::RectifiedFlow,
            FlowMatchingMethod::StochasticInterpolants,
        ] {
            let mut model = FlowMatchingBuilder::new()
                .method(method)
                .num_steps(3)
                .build()
                .unwrap();

            let speaker_embedding = Array1::from_vec(vec![0.5; 256]);
            let result = model.synthesize(&speaker_embedding, 50);

            assert!(result.is_ok(), "Failed with method: {:?}", method);
        }
    }

    #[test]
    fn test_trajectory_tracking() {
        let mut model = FlowMatchingBuilder::new().num_steps(5).build().unwrap();

        let speaker_embedding = Array1::from_vec(vec![0.5; 256]);
        let result = model.synthesize(&speaker_embedding, 50).unwrap();

        // Trajectory should have num_steps + 1 points
        assert_eq!(result.trajectory.len(), 6);
    }

    #[test]
    fn test_statistics_tracking() {
        let mut model = FlowMatchingBuilder::new().num_steps(3).build().unwrap();

        let speaker_embedding = Array1::from_vec(vec![0.5; 256]);

        for _ in 0..5 {
            let _ = model.synthesize(&speaker_embedding, 50);
        }

        let stats = model.get_stats();
        assert_eq!(stats.total_samples, 5);
        assert!(stats.avg_nfe > 0.0);
        assert!(stats.avg_synthesis_time_ms >= 0.0);
    }

    #[test]
    fn test_stats_reset() {
        let mut model = FlowMatchingBuilder::new().build().unwrap();

        let speaker_embedding = Array1::from_vec(vec![0.5; 256]);
        let _ = model.synthesize(&speaker_embedding, 50);

        assert_eq!(model.get_stats().total_samples, 1);

        model.reset_stats();
        assert_eq!(model.get_stats().total_samples, 0);
    }

    #[test]
    fn test_rk4_higher_accuracy() {
        // RK4 should use more function evaluations per step
        let mut model_euler = FlowMatchingBuilder::new()
            .ode_solver(OdeSolver::Euler)
            .num_steps(5)
            .build()
            .unwrap();

        let mut model_rk4 = FlowMatchingBuilder::new()
            .ode_solver(OdeSolver::RK4)
            .num_steps(5)
            .build()
            .unwrap();

        let speaker_embedding = Array1::from_vec(vec![0.5; 256]);

        let result_euler = model_euler.synthesize(&speaker_embedding, 50).unwrap();
        let result_rk4 = model_rk4.synthesize(&speaker_embedding, 50).unwrap();

        // RK4 should have more function evaluations
        assert!(result_rk4.nfe > result_euler.nfe);
    }

    #[test]
    fn test_quality_estimation() {
        let model = FlowMatchingBuilder::new().build().unwrap();

        // Smooth output should have higher quality
        let smooth = Array2::from_shape_fn((1, 100), |(_i, j)| (j as f32 / 100.0).sin());
        let quality_smooth = model.estimate_quality(&smooth).unwrap();

        // Noisy output should have lower quality
        let mut rng = scirs2_core::random::thread_rng();
        let noisy = Array2::from_shape_fn((1, 100), |(_i, _j)| rng.random_range(-5.0..5.0));
        let quality_noisy = model.estimate_quality(&noisy).unwrap();

        assert!(quality_smooth > quality_noisy);
    }

    #[test]
    fn test_conditional_generation() {
        let mut model = FlowMatchingBuilder::new().num_steps(3).build().unwrap();

        // Different speaker embeddings should produce different outputs
        let embedding1 = Array1::from_vec(vec![0.3; 256]);
        let embedding2 = Array1::from_vec(vec![0.7; 256]);

        let result1 = model.synthesize(&embedding1, 50).unwrap();
        let result2 = model.synthesize(&embedding2, 50).unwrap();

        // Outputs should be different
        let mut difference = 0.0;
        for i in 0..result1.output.dim().0 {
            for j in 0..result1.output.dim().1 {
                difference += (result1.output[[i, j]] - result2.output[[i, j]]).abs();
            }
        }

        assert!(difference > 0.0);
    }
}
