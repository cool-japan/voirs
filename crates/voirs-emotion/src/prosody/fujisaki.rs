//! Fujisaki Model for F0 Contour Generation
//!
//! Implements the Fujisaki model for generating natural intonation patterns.
//! The model represents F0 contours as the sum of:
//! - Base frequency (Fb)
//! - Phrase components (slow variations)
//! - Accent components (rapid fluctuations)
//!
//! Reference: Fujisaki, H. (1983). "Dynamic characteristics of voice fundamental
//! frequency in speech and singing." The production of speech, 39-55.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Fujisaki model for F0 contour generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FujisakiModel {
    /// Base frequency (Hz)
    base_frequency: f32,
    /// Phrase command parameters
    phrase_commands: Vec<PhraseCommand>,
    /// Accent commands
    accent_commands: Vec<AccentCommand>,
    /// Time constant for phrase component (alpha)
    alpha: f32,
    /// Time constant for accent component (beta)
    beta: f32,
}

/// Phrase command for slow F0 variations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseCommand {
    /// Amplitude of phrase command
    pub amplitude: f32,
    /// Onset time (seconds)
    pub onset: f32,
    /// Offset time (seconds)
    pub offset: f32,
}

/// Accent command for rapid F0 fluctuations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccentCommand {
    /// Amplitude of accent command
    pub amplitude: f32,
    /// Onset time (seconds)
    pub onset: f32,
}

/// Configuration for emotion-specific Fujisaki parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionFujisakiConfig {
    /// Base frequency multiplier for emotion
    pub base_freq_multiplier: f32,
    /// Phrase command amplitude scaling
    pub phrase_amplitude_scale: f32,
    /// Accent command amplitude scaling
    pub accent_amplitude_scale: f32,
    /// Alpha parameter modification
    pub alpha_multiplier: f32,
    /// Beta parameter modification
    pub beta_multiplier: f32,
}

impl FujisakiModel {
    /// Create a new Fujisaki model with default parameters
    pub fn new(base_frequency: f32) -> Self {
        Self {
            base_frequency,
            phrase_commands: Vec::new(),
            accent_commands: Vec::new(),
            alpha: 3.0, // Typical value for phrase component
            beta: 20.0, // Typical value for accent component
        }
    }

    /// Add a phrase command to the model
    pub fn add_phrase_command(&mut self, amplitude: f32, onset: f32, offset: f32) {
        self.phrase_commands.push(PhraseCommand {
            amplitude,
            onset,
            offset,
        });
    }

    /// Add an accent command to the model
    pub fn add_accent_command(&mut self, amplitude: f32, onset: f32) {
        self.accent_commands
            .push(AccentCommand { amplitude, onset });
    }

    /// Set time constants
    pub fn set_time_constants(&mut self, alpha: f32, beta: f32) {
        self.alpha = alpha;
        self.beta = beta;
    }

    /// Generate F0 contour for given time points
    ///
    /// # Arguments
    ///
    /// * `time_points` - Time points in seconds
    ///
    /// # Returns
    ///
    /// Vector of F0 values (Hz) at each time point
    pub fn generate_f0_contour(&self, time_points: &[f32]) -> Vec<f32> {
        time_points.iter().map(|&t| self.compute_f0(t)).collect()
    }

    /// Compute F0 at a specific time point
    fn compute_f0(&self, t: f32) -> f32 {
        // Logarithmic F0 = ln(Fb) + Fp(t) + Fa(t)
        let log_fb = self.base_frequency.ln();
        let fp = self.phrase_component(t);
        let fa = self.accent_component(t);

        (log_fb + fp + fa).exp()
    }

    /// Compute phrase component at time t
    fn phrase_component(&self, t: f32) -> f32 {
        let mut fp = 0.0;

        for cmd in &self.phrase_commands {
            if t >= cmd.onset && t <= cmd.offset {
                // Rising portion
                let t1 = t - cmd.onset;
                fp += cmd.amplitude * (1.0 - (1.0 + self.alpha * t1) * (-self.alpha * t1).exp());
            } else if t > cmd.offset {
                // Falling portion
                let t1 = cmd.offset - cmd.onset;
                let t2 = t - cmd.offset;
                let peak =
                    cmd.amplitude * (1.0 - (1.0 + self.alpha * t1) * (-self.alpha * t1).exp());
                fp += peak * (-self.alpha * t2).exp();
            }
        }

        fp
    }

    /// Compute accent component at time t
    fn accent_component(&self, t: f32) -> f32 {
        let mut fa = 0.0;

        for cmd in &self.accent_commands {
            if t >= cmd.onset {
                let t_diff = t - cmd.onset;
                // Accent impulse response
                fa += cmd.amplitude * self.beta * t_diff * (-self.beta * t_diff).exp();
            }
        }

        fa
    }

    /// Apply emotion-specific modifications to the model
    pub fn apply_emotion_config(&mut self, config: &EmotionFujisakiConfig) {
        self.base_frequency *= config.base_freq_multiplier;
        self.alpha *= config.alpha_multiplier;
        self.beta *= config.beta_multiplier;

        for cmd in &mut self.phrase_commands {
            cmd.amplitude *= config.phrase_amplitude_scale;
        }

        for cmd in &mut self.accent_commands {
            cmd.amplitude *= config.accent_amplitude_scale;
        }
    }
}

impl EmotionFujisakiConfig {
    /// Create configuration for happy emotion
    pub fn happy() -> Self {
        Self {
            base_freq_multiplier: 1.2,   // Higher pitch
            phrase_amplitude_scale: 1.3, // More phrase variation
            accent_amplitude_scale: 1.4, // Stronger accents
            alpha_multiplier: 0.9,       // Slightly slower phrase changes
            beta_multiplier: 1.1,        // Slightly faster accents
        }
    }

    /// Create configuration for sad emotion
    pub fn sad() -> Self {
        Self {
            base_freq_multiplier: 0.85,  // Lower pitch
            phrase_amplitude_scale: 0.7, // Less phrase variation
            accent_amplitude_scale: 0.6, // Weaker accents
            alpha_multiplier: 1.2,       // Slower phrase changes
            beta_multiplier: 0.8,        // Slower accents
        }
    }

    /// Create configuration for angry emotion
    pub fn angry() -> Self {
        Self {
            base_freq_multiplier: 1.3,   // Much higher pitch
            phrase_amplitude_scale: 1.5, // Strong phrase variation
            accent_amplitude_scale: 1.8, // Very strong accents
            alpha_multiplier: 0.7,       // Faster phrase changes
            beta_multiplier: 1.3,        // Much faster accents
        }
    }

    /// Create configuration for calm emotion
    pub fn calm() -> Self {
        Self {
            base_freq_multiplier: 0.95,  // Slightly lower pitch
            phrase_amplitude_scale: 0.8, // Reduced phrase variation
            accent_amplitude_scale: 0.7, // Softer accents
            alpha_multiplier: 1.3,       // Slower, smoother phrase changes
            beta_multiplier: 0.9,        // Gentler accents
        }
    }

    /// Create configuration for excited emotion
    pub fn excited() -> Self {
        Self {
            base_freq_multiplier: 1.4,   // Very high pitch
            phrase_amplitude_scale: 1.6, // Large phrase variation
            accent_amplitude_scale: 2.0, // Maximum accent strength
            alpha_multiplier: 0.6,       // Very fast phrase changes
            beta_multiplier: 1.4,        // Very fast accents
        }
    }

    /// Create configuration for fearful emotion
    pub fn fearful() -> Self {
        Self {
            base_freq_multiplier: 1.25,  // Elevated pitch
            phrase_amplitude_scale: 1.4, // Irregular phrase variation
            accent_amplitude_scale: 1.3, // Somewhat strong accents
            alpha_multiplier: 0.8,       // Faster, less controlled changes
            beta_multiplier: 1.2,        // Faster accents
        }
    }
}

/// Builder for creating Fujisaki models from linguistic input
pub struct FujisakiModelBuilder {
    base_frequency: f32,
    phrase_commands: Vec<PhraseCommand>,
    accent_commands: Vec<AccentCommand>,
    alpha: f32,
    beta: f32,
}

impl FujisakiModelBuilder {
    /// Create a new builder with default base frequency
    pub fn new(base_frequency: f32) -> Self {
        Self {
            base_frequency,
            phrase_commands: Vec::new(),
            accent_commands: Vec::new(),
            alpha: 3.0,
            beta: 20.0,
        }
    }

    /// Add phrase boundary with automatic command generation
    pub fn add_phrase_boundary(mut self, time: f32, duration: f32, prominence: f32) -> Self {
        self.phrase_commands.push(PhraseCommand {
            amplitude: prominence * 0.3, // Scale prominence to reasonable amplitude
            onset: time,
            offset: time + duration,
        });
        self
    }

    /// Add accent with automatic command generation
    pub fn add_accent(mut self, time: f32, prominence: f32) -> Self {
        self.accent_commands.push(AccentCommand {
            amplitude: prominence * 0.4, // Scale prominence to reasonable amplitude
            onset: time,
        });
        self
    }

    /// Set custom time constants
    pub fn with_time_constants(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self
    }

    /// Build the Fujisaki model
    pub fn build(self) -> FujisakiModel {
        FujisakiModel {
            base_frequency: self.base_frequency,
            phrase_commands: self.phrase_commands,
            accent_commands: self.accent_commands,
            alpha: self.alpha,
            beta: self.beta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fujisaki_basic() {
        let mut model = FujisakiModel::new(100.0);
        model.add_phrase_command(0.2, 0.0, 1.0);
        model.add_accent_command(0.3, 0.5);

        let time_points = vec![0.0, 0.5, 1.0, 1.5];
        let f0_contour = model.generate_f0_contour(&time_points);

        assert_eq!(f0_contour.len(), 4);
        assert!(f0_contour[0] > 0.0);
        assert!(f0_contour.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_phrase_component() {
        let mut model = FujisakiModel::new(100.0);
        model.add_phrase_command(0.3, 0.0, 1.0);

        let f0_at_start = model.compute_f0(0.0);
        let f0_at_mid = model.compute_f0(0.5);
        let f0_at_end = model.compute_f0(1.0);

        // F0 should rise during phrase command
        assert!(f0_at_mid > f0_at_start);
        assert!(f0_at_end >= f0_at_mid);
    }

    #[test]
    fn test_accent_component() {
        let mut model = FujisakiModel::new(100.0);
        model.add_accent_command(0.5, 0.5);

        let f0_before = model.compute_f0(0.4);
        let f0_peak = model.compute_f0(0.55);
        let f0_after = model.compute_f0(0.8);

        // F0 should peak shortly after accent onset
        assert!(f0_peak > f0_before);
        assert!(f0_peak > f0_after);
    }

    #[test]
    fn test_emotion_modifications() {
        let mut model = FujisakiModel::new(100.0);
        model.add_phrase_command(0.2, 0.0, 1.0);
        model.add_accent_command(0.3, 0.5);

        let original_base = model.base_frequency;

        model.apply_emotion_config(&EmotionFujisakiConfig::happy());

        assert!(model.base_frequency > original_base);
    }

    #[test]
    fn test_builder_pattern() {
        let model = FujisakiModelBuilder::new(120.0)
            .add_phrase_boundary(0.0, 1.5, 1.0)
            .add_accent(0.5, 0.8)
            .add_accent(1.0, 0.6)
            .with_time_constants(3.5, 22.0)
            .build();

        assert_eq!(model.phrase_commands.len(), 1);
        assert_eq!(model.accent_commands.len(), 2);
        assert!((model.alpha - 3.5).abs() < 1e-5);
    }

    #[test]
    fn test_emotion_configs() {
        let happy = EmotionFujisakiConfig::happy();
        let sad = EmotionFujisakiConfig::sad();
        let angry = EmotionFujisakiConfig::angry();

        // Happy should have higher pitch than sad
        assert!(happy.base_freq_multiplier > sad.base_freq_multiplier);

        // Angry should have strongest accents
        assert!(angry.accent_amplitude_scale > happy.accent_amplitude_scale);
        assert!(angry.accent_amplitude_scale > sad.accent_amplitude_scale);
    }

    #[test]
    fn test_f0_contour_generation() {
        let model = FujisakiModelBuilder::new(100.0)
            .add_phrase_boundary(0.0, 2.0, 1.0)
            .add_accent(0.5, 1.0)
            .add_accent(1.5, 0.8)
            .build();

        let time_points: Vec<f32> = (0..100).map(|i| i as f32 * 0.02).collect();
        let f0_contour = model.generate_f0_contour(&time_points);

        assert_eq!(f0_contour.len(), 100);

        // All F0 values should be positive and finite
        assert!(f0_contour.iter().all(|&f| f > 0.0 && f.is_finite()));

        // F0 should vary (not all the same)
        let min_f0 = f0_contour.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_f0 = f0_contour.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_f0 > min_f0);
    }
}
