//! SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) implementation
//!
//! SI-SDR is a scale-invariant metric that measures the quality of
//! source separation or signal enhancement. It's particularly useful
//! for evaluating vocoders and audio enhancement algorithms.

use crate::{Result, VocoderError};
use scirs2_core::ndarray::{s, Array1};

/// Calculate SI-SDR between reference and estimated signals
pub fn calculate_si_sdr(reference: &Array1<f32>, estimated: &Array1<f32>) -> Result<f32> {
    // Ensure signals have the same length
    let min_len = reference.len().min(estimated.len());
    if min_len == 0 {
        return Err(VocoderError::InputError(
            "Empty signals provided".to_string(),
        ));
    }

    let ref_signal = reference.slice(s![..min_len]);
    let est_signal = estimated.slice(s![..min_len]);

    // Calculate the optimal scaling factor (projection)
    let alpha = calculate_scaling_factor(&ref_signal.to_owned(), &est_signal.to_owned());

    // Scale the estimated signal
    let scaled_estimate = est_signal.mapv(|x| x * alpha);

    // Calculate signal and noise power
    let signal_power = calculate_power(&ref_signal.to_owned());
    let noise_power = calculate_noise_power(&ref_signal.to_owned(), &scaled_estimate);

    // Calculate SI-SDR in dB
    if noise_power > 1e-20 && signal_power > 1e-20 {
        Ok(10.0 * (signal_power / noise_power).log10())
    } else if noise_power <= 1e-20 {
        Ok(f32::INFINITY) // Perfect reconstruction
    } else {
        Ok(f32::NEG_INFINITY) // Very poor reconstruction
    }
}

/// Calculate optimal scaling factor using projection
fn calculate_scaling_factor(reference: &Array1<f32>, estimated: &Array1<f32>) -> f32 {
    let dot_product: f32 = reference
        .iter()
        .zip(estimated.iter())
        .map(|(&r, &e)| r * e)
        .sum();
    let estimated_energy: f32 = estimated.iter().map(|&x| x * x).sum();

    if estimated_energy > 1e-20 {
        dot_product / estimated_energy
    } else {
        0.0
    }
}

/// Calculate signal power
fn calculate_power(signal: &Array1<f32>) -> f32 {
    signal.iter().map(|&x| x * x).sum()
}

/// Calculate noise power (difference between reference and scaled estimate)
fn calculate_noise_power(reference: &Array1<f32>, scaled_estimate: &Array1<f32>) -> f32 {
    reference
        .iter()
        .zip(scaled_estimate.iter())
        .map(|(&r, &e)| (r - e).powi(2))
        .sum()
}

/// Calculate SDR (Signal-to-Distortion Ratio) without scale invariance
pub fn calculate_sdr(reference: &Array1<f32>, estimated: &Array1<f32>) -> Result<f32> {
    let min_len = reference.len().min(estimated.len());
    if min_len == 0 {
        return Err(VocoderError::InputError(
            "Empty signals provided".to_string(),
        ));
    }

    let ref_signal = reference.slice(s![..min_len]);
    let est_signal = estimated.slice(s![..min_len]);

    let signal_power = calculate_power(&ref_signal.to_owned());
    let noise_power = calculate_noise_power(&ref_signal.to_owned(), &est_signal.to_owned());

    if noise_power > 1e-20 && signal_power > 1e-20 {
        Ok(10.0 * (signal_power / noise_power).log10())
    } else if noise_power <= 1e-20 {
        Ok(f32::INFINITY)
    } else {
        Ok(f32::NEG_INFINITY)
    }
}

/// Calculate Scale-Invariant Signal-to-Interference Ratio (SI-SIR)
pub fn calculate_si_sir(
    reference: &Array1<f32>,
    estimated: &Array1<f32>,
    interference: &Array1<f32>,
) -> Result<f32> {
    let min_len = reference.len().min(estimated.len()).min(interference.len());
    if min_len == 0 {
        return Err(VocoderError::InputError(
            "Empty signals provided".to_string(),
        ));
    }

    let ref_signal = reference.slice(s![..min_len]);
    let est_signal = estimated.slice(s![..min_len]);
    let int_signal = interference.slice(s![..min_len]);

    // Calculate scaling factor for target signal
    let _alpha_target = calculate_scaling_factor(&ref_signal.to_owned(), &est_signal.to_owned());

    // Calculate scaling factor for interference
    let alpha_interference =
        calculate_scaling_factor(&int_signal.to_owned(), &est_signal.to_owned());

    // Calculate interference component in estimate
    let interference_component = int_signal.mapv(|x| x * alpha_interference);

    // Calculate target signal power and interference power
    let target_power = calculate_power(&ref_signal.to_owned());
    let interference_power = calculate_power(&interference_component);

    if interference_power > 1e-20 && target_power > 1e-20 {
        Ok(10.0 * (target_power / interference_power).log10())
    } else if interference_power <= 1e-20 {
        Ok(f32::INFINITY)
    } else {
        Ok(f32::NEG_INFINITY)
    }
}

/// Calculate multiple source separation metrics
#[derive(Debug, Clone)]
pub struct SeparationMetrics {
    /// Scale-Invariant Signal-to-Distortion Ratio (dB)
    pub si_sdr: f32,

    /// Scale-Invariant Signal-to-Interference Ratio (dB)
    pub si_sir: Option<f32>,

    /// Signal-to-Distortion Ratio (dB, not scale-invariant)
    pub sdr: f32,

    /// Signal-to-Artifacts Ratio (dB)
    pub sar: f32,
}

/// Calculate comprehensive separation metrics
pub fn calculate_separation_metrics(
    reference: &Array1<f32>,
    estimated: &Array1<f32>,
    interference: Option<&Array1<f32>>,
) -> Result<SeparationMetrics> {
    let si_sdr = calculate_si_sdr(reference, estimated)?;
    let sdr = calculate_sdr(reference, estimated)?;

    let si_sir = if let Some(interference) = interference {
        Some(calculate_si_sir(reference, estimated, interference)?)
    } else {
        None
    };

    // Calculate SAR (simplified version)
    let sar = calculate_sar(reference, estimated)?;

    Ok(SeparationMetrics {
        si_sdr,
        si_sir,
        sdr,
        sar,
    })
}

/// Calculate Signal-to-Artifacts Ratio
fn calculate_sar(reference: &Array1<f32>, estimated: &Array1<f32>) -> Result<f32> {
    let min_len = reference.len().min(estimated.len());
    let ref_signal = reference.slice(s![..min_len]);
    let est_signal = estimated.slice(s![..min_len]);

    // Calculate optimal scaling
    let alpha = calculate_scaling_factor(&ref_signal.to_owned(), &est_signal.to_owned());
    let scaled_target = ref_signal.mapv(|x| x * alpha);

    // Artifacts = estimate - scaled_target
    let artifact_power: f32 = est_signal
        .iter()
        .zip(scaled_target.iter())
        .map(|(&e, &s)| (e - s).powi(2))
        .sum();

    let target_power = calculate_power(&scaled_target);

    if artifact_power > 1e-20 && target_power > 1e-20 {
        Ok(10.0 * (target_power / artifact_power).log10())
    } else if artifact_power <= 1e-20 {
        Ok(f32::INFINITY)
    } else {
        Ok(f32::NEG_INFINITY)
    }
}

/// Batch calculation of SI-SDR for multiple signal pairs
pub fn calculate_si_sdr_batch(
    references: &[Array1<f32>],
    estimates: &[Array1<f32>],
) -> Result<Vec<f32>> {
    if references.len() != estimates.len() {
        return Err(VocoderError::InputError(
            "Number of reference and estimated signals must match".to_string(),
        ));
    }

    let mut results = Vec::with_capacity(references.len());
    for (ref_signal, est_signal) in references.iter().zip(estimates.iter()) {
        let si_sdr = calculate_si_sdr(ref_signal, est_signal)?;
        results.push(si_sdr);
    }

    Ok(results)
}

/// Calculate improvement in SI-SDR
pub fn calculate_si_sdr_improvement(
    reference: &Array1<f32>,
    degraded: &Array1<f32>,
    enhanced: &Array1<f32>,
) -> Result<f32> {
    let si_sdr_degraded = calculate_si_sdr(reference, degraded)?;
    let si_sdr_enhanced = calculate_si_sdr(reference, enhanced)?;

    Ok(si_sdr_enhanced - si_sdr_degraded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_perfect_reconstruction() {
        let signal = Array1::from_vec(vec![1.0, -0.5, 0.8, -0.3, 0.2]);
        let si_sdr = calculate_si_sdr(&signal, &signal).unwrap();

        // Perfect reconstruction should have very high SI-SDR
        assert!(si_sdr > 100.0 || si_sdr == f32::INFINITY);
    }

    #[test]
    fn test_scaled_signal() {
        let reference = Array1::from_vec(vec![1.0, -0.5, 0.8, -0.3, 0.2]);
        let scaled = reference.mapv(|x| x * 2.0); // Scale by 2

        let si_sdr = calculate_si_sdr(&reference, &scaled).unwrap();

        // Scale-invariant metric should still be perfect for scaled signal
        assert!(si_sdr > 100.0 || si_sdr == f32::INFINITY);
    }

    #[test]
    fn test_noisy_signal() {
        let clean = Array1::from_vec(vec![1.0, -0.5, 0.8, -0.3, 0.2]);
        let noisy = Array1::from_vec(vec![1.1, -0.4, 0.7, -0.4, 0.1]);

        let si_sdr = calculate_si_sdr(&clean, &noisy).unwrap();

        // Noisy signal should have finite, reasonable SI-SDR
        assert!(si_sdr.is_finite());
        assert!(si_sdr < 50.0); // Should be significantly less than perfect
    }

    #[test]
    fn test_scaling_factor_calculation() {
        let reference = Array1::from_vec(vec![2.0, 4.0, 6.0]);
        let estimated = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let alpha = calculate_scaling_factor(&reference, &estimated);

        // Should be 2.0 since estimated = reference / 2
        assert!((alpha - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sdr_vs_si_sdr() {
        let reference = Array1::from_vec(vec![1.0, -0.5, 0.8]);
        let scaled = reference.mapv(|x| x * 0.5);

        let sdr = calculate_sdr(&reference, &scaled).unwrap();
        let si_sdr = calculate_si_sdr(&reference, &scaled).unwrap();

        // SDR should be affected by scaling, SI-SDR should not
        assert!(sdr < si_sdr || si_sdr == f32::INFINITY);
    }

    #[test]
    fn test_empty_signals() {
        let empty = Array1::from_vec(vec![]);
        let signal = Array1::from_vec(vec![1.0, 2.0]);

        let result = calculate_si_sdr(&empty, &signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_separation_metrics() {
        let reference = Array1::from_vec(vec![1.0, -0.5, 0.8, -0.3]);
        let estimated = Array1::from_vec(vec![0.9, -0.4, 0.7, -0.4]);
        let interference = Array1::from_vec(vec![0.1, 0.2, -0.1, 0.05]);

        let metrics =
            calculate_separation_metrics(&reference, &estimated, Some(&interference)).unwrap();

        assert!(metrics.si_sdr.is_finite());
        assert!(metrics.sdr.is_finite());
        assert!(metrics.sar.is_finite());
        assert!(metrics.si_sir.is_some());
        assert!(metrics.si_sir.unwrap().is_finite());
    }

    #[test]
    fn test_si_sdr_improvement() {
        let reference = Array1::from_vec(vec![1.0, -0.5, 0.8, -0.3]);
        let degraded = Array1::from_vec(vec![0.8, -0.3, 0.6, -0.1]); // Worse quality
        let enhanced = Array1::from_vec(vec![0.98, -0.48, 0.78, -0.28]); // Better quality

        let improvement = calculate_si_sdr_improvement(&reference, &degraded, &enhanced).unwrap();

        // Enhancement should improve SI-SDR
        assert!(improvement > 0.0);
    }

    #[test]
    fn test_batch_calculation() {
        let references = vec![
            Array1::from_vec(vec![1.0, -0.5]),
            Array1::from_vec(vec![0.8, 0.3]),
        ];
        let estimates = vec![
            Array1::from_vec(vec![0.9, -0.4]),
            Array1::from_vec(vec![0.7, 0.4]),
        ];

        let results = calculate_si_sdr_batch(&references, &estimates).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].is_finite());
        assert!(results[1].is_finite());
    }
}
