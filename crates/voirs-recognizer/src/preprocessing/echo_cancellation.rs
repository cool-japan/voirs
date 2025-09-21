//! Echo cancellation module
//!
//! This module implements acoustic echo cancellation (AEC) for removing
//! echo and feedback from audio signals using adaptive filtering techniques.

use crate::RecognitionError;
use std::collections::VecDeque;
use voirs_sdk::AudioBuffer;

/// Echo cancellation configuration
#[derive(Debug, Clone)]
pub struct EchoCancellationConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Adaptive filter length
    pub filter_length: usize,
    /// Adaptation rate (learning rate)
    pub adaptation_rate: f32,
    /// Non-linear processing threshold
    pub nlp_threshold: f32,
}

impl Default for EchoCancellationConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            filter_length: 1024,
            adaptation_rate: 0.01,
            nlp_threshold: 0.5,
        }
    }
}

/// Echo cancellation processing statistics
#[derive(Debug, Clone)]
pub struct EchoCancellationStats {
    /// Echo reduction in dB
    pub echo_reduction_db: f32,
    /// Residual echo level in dB
    pub residual_echo_db: f32,
    /// Adaptive filter convergence (0.0 to 1.0)
    pub convergence: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
}

/// Echo cancellation processing result
#[derive(Debug, Clone)]
pub struct EchoCancellationResult {
    /// Enhanced audio buffer
    pub enhanced_audio: AudioBuffer,
    /// Processing statistics
    pub stats: EchoCancellationStats,
}

/// Adaptive echo cancellation processor
#[derive(Debug)]
pub struct EchoCancellationProcessor {
    /// Configuration
    config: EchoCancellationConfig,
    /// Adaptive filter coefficients
    filter_coeffs: Vec<f32>,
    /// Reference signal delay line
    reference_delay: VecDeque<f32>,
    /// Far-end signal buffer
    far_end_buffer: VecDeque<f32>,
    /// Echo estimate buffer
    echo_estimate: Vec<f32>,
    /// Power estimation for adaptation control
    power_estimate: f32,
    /// Convergence tracking
    convergence_tracker: f32,
    /// Step size adaptation
    step_size: f32,
}

impl EchoCancellationProcessor {
    /// Create a new echo cancellation processor
    pub fn new(config: EchoCancellationConfig) -> Result<Self, RecognitionError> {
        Ok(Self {
            config: config.clone(),
            filter_coeffs: vec![0.0; config.filter_length],
            reference_delay: VecDeque::with_capacity(config.filter_length),
            far_end_buffer: VecDeque::with_capacity(config.filter_length),
            echo_estimate: vec![0.0; config.filter_length],
            power_estimate: 0.0,
            convergence_tracker: 0.0,
            step_size: config.adaptation_rate,
        })
    }

    /// Process audio buffer with echo cancellation
    ///
    /// # Arguments
    ///
    /// * `near_end` - Near-end signal (microphone input with echo)
    /// * `far_end` - Far-end signal (speaker output causing echo) - optional
    pub async fn process(
        &mut self,
        near_end: &AudioBuffer,
    ) -> Result<EchoCancellationResult, RecognitionError> {
        let start_time = std::time::Instant::now();

        let samples = near_end.samples();
        let mut enhanced_samples = Vec::with_capacity(samples.len());

        // Process each sample
        for &sample in samples {
            let enhanced_sample = self.process_sample(sample, None);
            enhanced_samples.push(enhanced_sample);
        }

        let enhanced_audio = AudioBuffer::mono(enhanced_samples, near_end.sample_rate());

        // Calculate statistics
        let echo_reduction_db = self.calculate_echo_reduction(near_end, &enhanced_audio);
        let residual_echo_db = self.calculate_residual_echo(&enhanced_audio);
        let convergence = self.convergence_tracker;
        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        let stats = EchoCancellationStats {
            echo_reduction_db,
            residual_echo_db,
            convergence,
            processing_time_ms,
        };

        Ok(EchoCancellationResult {
            enhanced_audio,
            stats,
        })
    }

    /// Process audio buffer with far-end reference
    pub async fn process_with_reference(
        &mut self,
        near_end: &AudioBuffer,
        far_end: &AudioBuffer,
    ) -> Result<EchoCancellationResult, RecognitionError> {
        let start_time = std::time::Instant::now();

        let near_samples = near_end.samples();
        let far_samples = far_end.samples();
        let mut enhanced_samples = Vec::with_capacity(near_samples.len());

        // Process each sample pair
        for (i, &near_sample) in near_samples.iter().enumerate() {
            let far_sample = if i < far_samples.len() {
                Some(far_samples[i])
            } else {
                None
            };

            let enhanced_sample = self.process_sample(near_sample, far_sample);
            enhanced_samples.push(enhanced_sample);
        }

        let enhanced_audio = AudioBuffer::mono(enhanced_samples, near_end.sample_rate());

        // Calculate statistics
        let echo_reduction_db = self.calculate_echo_reduction(near_end, &enhanced_audio);
        let residual_echo_db = self.calculate_residual_echo(&enhanced_audio);
        let convergence = self.convergence_tracker;
        let processing_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        let stats = EchoCancellationStats {
            echo_reduction_db,
            residual_echo_db,
            convergence,
            processing_time_ms,
        };

        Ok(EchoCancellationResult {
            enhanced_audio,
            stats,
        })
    }

    /// Process a single sample with adaptive filtering
    fn process_sample(&mut self, near_sample: f32, far_sample: Option<f32>) -> f32 {
        // Add far-end sample to reference delay line
        if let Some(far) = far_sample {
            self.far_end_buffer.push_back(far);
        } else {
            // If no far-end reference, use a simple noise reduction approach
            self.far_end_buffer.push_back(0.0);
        }

        // Maintain buffer size
        if self.far_end_buffer.len() > self.config.filter_length {
            self.far_end_buffer.pop_front();
        }

        // Calculate echo estimate using adaptive filter
        let echo_estimate = self.calculate_echo_estimate();

        // Subtract echo estimate from near-end signal
        let error_signal = near_sample - echo_estimate;

        // Update adaptive filter coefficients using NLMS algorithm
        self.update_filter_coefficients(error_signal);

        // Apply non-linear processing if needed
        let enhanced_sample = self.apply_nonlinear_processing(error_signal);

        enhanced_sample
    }

    /// Calculate echo estimate using current filter coefficients
    fn calculate_echo_estimate(&self) -> f32 {
        let mut estimate = 0.0;

        for (i, &coeff) in self.filter_coeffs.iter().enumerate() {
            if i < self.far_end_buffer.len() {
                estimate += coeff * self.far_end_buffer[self.far_end_buffer.len() - 1 - i];
            }
        }

        estimate
    }

    /// Update adaptive filter coefficients using NLMS algorithm
    fn update_filter_coefficients(&mut self, error: f32) {
        // Calculate input power
        let input_power = self.far_end_buffer.iter().map(|&x| x * x).sum::<f32>();

        // Update power estimate with smoothing
        self.power_estimate = 0.9 * self.power_estimate + 0.1 * input_power;

        // Calculate normalized step size
        let epsilon = 1e-10; // Small constant to prevent division by zero
        let normalized_step = self.step_size / (self.power_estimate + epsilon);

        // Update filter coefficients
        for (i, coeff) in self.filter_coeffs.iter_mut().enumerate() {
            if i < self.far_end_buffer.len() {
                let input_sample = self.far_end_buffer[self.far_end_buffer.len() - 1 - i];
                *coeff += normalized_step * error * input_sample;
            }
        }

        // Update convergence tracker
        let filter_norm = self
            .filter_coeffs
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        self.convergence_tracker = 0.99 * self.convergence_tracker + 0.01 * filter_norm;
    }

    /// Apply non-linear processing to suppress residual echo
    fn apply_nonlinear_processing(&self, signal: f32) -> f32 {
        let abs_signal = signal.abs();

        if abs_signal < self.config.nlp_threshold {
            // Apply attenuation to weak signals (likely residual echo)
            signal * 0.5
        } else {
            // Pass strong signals (likely speech) unchanged
            signal
        }
    }

    /// Calculate echo reduction in dB
    fn calculate_echo_reduction(&self, original: &AudioBuffer, enhanced: &AudioBuffer) -> f32 {
        let original_power = original.samples().iter().map(|&x| x * x).sum::<f32>()
            / original.samples().len() as f32;
        let enhanced_power = enhanced.samples().iter().map(|&x| x * x).sum::<f32>()
            / enhanced.samples().len() as f32;

        if original_power > 0.0 && enhanced_power > 0.0 {
            10.0 * (original_power / enhanced_power).log10()
        } else {
            0.0
        }
    }

    /// Calculate residual echo level in dB
    fn calculate_residual_echo(&self, audio: &AudioBuffer) -> f32 {
        let power =
            audio.samples().iter().map(|&x| x * x).sum::<f32>() / audio.samples().len() as f32;

        if power > 0.0 {
            10.0 * power.log10()
        } else {
            -60.0
        }
    }

    /// Reset processor state
    pub fn reset(&mut self) -> Result<(), RecognitionError> {
        self.filter_coeffs.fill(0.0);
        self.reference_delay.clear();
        self.far_end_buffer.clear();
        self.echo_estimate.fill(0.0);
        self.power_estimate = 0.0;
        self.convergence_tracker = 0.0;
        self.step_size = self.config.adaptation_rate;
        Ok(())
    }

    /// Get current convergence status
    pub fn get_convergence(&self) -> f32 {
        self.convergence_tracker
    }

    /// Set adaptation rate
    pub fn set_adaptation_rate(&mut self, rate: f32) {
        self.config.adaptation_rate = rate;
        self.step_size = rate;
    }

    /// Set non-linear processing threshold
    pub fn set_nlp_threshold(&mut self, threshold: f32) {
        self.config.nlp_threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo_cancellation_processor_creation() {
        let config = EchoCancellationConfig::default();
        let processor = EchoCancellationProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_echo_cancellation_basic() {
        let config = EchoCancellationConfig::default();
        let mut processor = EchoCancellationProcessor::new(config).unwrap();

        let samples = vec![0.1f32; 1024];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.stats.processing_time_ms > 0.0);
        assert!(result.enhanced_audio.samples().len() == audio.samples().len());
    }

    #[tokio::test]
    async fn test_echo_cancellation_with_reference() {
        let config = EchoCancellationConfig::default();
        let mut processor = EchoCancellationProcessor::new(config).unwrap();

        let near_samples = vec![0.1f32; 1024];
        let far_samples = vec![0.05f32; 1024];
        let near_audio = AudioBuffer::mono(near_samples, 16000);
        let far_audio = AudioBuffer::mono(far_samples, 16000);

        let result = processor
            .process_with_reference(&near_audio, &far_audio)
            .await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.stats.echo_reduction_db >= 0.0);
    }

    #[tokio::test]
    async fn test_echo_cancellation_adaptation() {
        let config = EchoCancellationConfig {
            adaptation_rate: 0.1,
            filter_length: 128,
            ..Default::default()
        };
        let mut processor = EchoCancellationProcessor::new(config).unwrap();

        // Process multiple frames with both near-end and far-end signals to allow adaptation
        for _ in 0..10 {
            let near_samples = vec![0.1f32; 128];
            let far_samples = vec![0.05f32; 128]; // Provide far-end reference
            let near_audio = AudioBuffer::mono(near_samples, 16000);
            let far_audio = AudioBuffer::mono(far_samples, 16000);
            let _ = processor
                .process_with_reference(&near_audio, &far_audio)
                .await
                .unwrap();
        }

        assert!(processor.get_convergence() > 0.0);
    }

    #[tokio::test]
    async fn test_echo_cancellation_reset() {
        let config = EchoCancellationConfig::default();
        let mut processor = EchoCancellationProcessor::new(config).unwrap();

        // Process some audio first
        let samples = vec![0.1f32; 1024];
        let audio = AudioBuffer::mono(samples, 16000);
        let _ = processor.process(&audio).await.unwrap();

        // Reset and check state
        let result = processor.reset();
        assert!(result.is_ok());
        assert!((processor.get_convergence() - 0.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_echo_cancellation_parameter_adjustment() {
        let config = EchoCancellationConfig::default();
        let mut processor = EchoCancellationProcessor::new(config).unwrap();

        processor.set_adaptation_rate(0.05);
        processor.set_nlp_threshold(0.3);

        let samples = vec![0.1f32; 1024];
        let audio = AudioBuffer::mono(samples, 16000);

        let result = processor.process(&audio).await;
        assert!(result.is_ok());
    }
}
