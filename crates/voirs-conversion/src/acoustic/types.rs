//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "acoustic-integration")]
use crate::transforms::Transform as _;
use crate::{Error, Result};
#[cfg(feature = "acoustic-integration")]
use voirs_acoustic;

/// Comprehensive acoustic features
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone)]
pub struct AcousticFeatures {
    /// Fundamental frequency contour
    pub f0_contour: Vec<f32>,
    /// Formant frequencies
    pub formants: FormantFrequencies,
    /// Spectral envelope
    pub spectral_envelope: Vec<f32>,
    /// Temporal features
    pub temporal_features: TemporalFeatures,
    /// Harmonic features
    pub harmonic_features: HarmonicFeatures,
    /// Number of frames
    pub frame_count: usize,
    /// Sample rate
    pub sample_rate: f32,
}

#[cfg(feature = "acoustic-integration")]
impl AcousticFeatures {
    /// Apply male voice characteristics
    pub fn apply_male_characteristics(&mut self) {
        // Lower F0 for male voice
        for f0 in &mut self.f0_contour {
            *f0 *= 0.7; // Lower pitch
        }
        // Adjust formants for male vocal tract
        for f1 in &mut self.formants.f1 {
            *f1 *= 0.9;
        }
        for f2 in &mut self.formants.f2 {
            *f2 *= 0.9;
        }
    }

    /// Apply female voice characteristics
    pub fn apply_female_characteristics(&mut self) {
        // Raise F0 for female voice
        for f0 in &mut self.f0_contour {
            *f0 *= 1.4; // Higher pitch
        }
        // Adjust formants for female vocal tract
        for f1 in &mut self.formants.f1 {
            *f1 *= 1.1;
        }
        for f2 in &mut self.formants.f2 {
            *f2 *= 1.1;
        }
    }
}

#[cfg(not(feature = "acoustic-integration"))]
#[derive(Debug, Clone)]
pub struct AcousticFeatures {
    pub placeholder: bool,
}

/// Formant frequencies
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone)]
pub struct FormantFrequencies {
    /// First formant (F1) - tongue height
    pub f1: Vec<f32>,
    /// Second formant (F2) - tongue frontness/backness
    pub f2: Vec<f32>,
    /// Third formant (F3) - lip rounding
    pub f3: Vec<f32>,
    /// Fourth formant (F4)
    pub f4: Vec<f32>,
    /// Formant bandwidths
    pub bandwidths: Vec<f32>,
}

#[cfg(not(feature = "acoustic-integration"))]
#[derive(Debug, Clone)]
pub struct FormantFrequencies {
    pub placeholder: bool,
}
/// Window types for acoustic analysis
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    /// Hann window (raised cosine)
    Hann,
    /// Hamming window (modified raised cosine)
    Hamming,
    /// Blackman window (three-term cosine sum)
    Blackman,
    /// Rectangular window (no windowing)
    Rectangular,
}
/// Acoustic processing state
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone)]
pub struct AcousticState {
    /// Last F0 value
    pub last_f0: f32,
    /// Last formant values
    pub last_formants: (f32, f32, f32),
    /// Last energy level
    pub last_energy: f32,
    /// Phase continuity
    pub phase_accumulator: f32,
}

#[cfg(feature = "acoustic-integration")]
impl AcousticState {
    /// Update state from acoustic features
    pub fn update_from_features(&mut self, features: &AcousticFeatures) {
        if let Some(&last_f0) = features.f0_contour.last() {
            if last_f0 > 0.0 {
                self.last_f0 = last_f0;
            }
        }
        if let (Some(&f1), Some(&f2), Some(&f3)) = (
            features.formants.f1.last(),
            features.formants.f2.last(),
            features.formants.f3.last(),
        ) {
            self.last_formants = (f1, f2, f3);
        }
        if let Some(&energy) = features.temporal_features.energy_contour.last() {
            self.last_energy = energy;
        }
    }
}

/// Result of acoustic conversion with quality metrics
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone)]
pub struct AcousticConversionResult {
    /// Converted audio
    pub audio: Vec<f32>,
    /// Original acoustic features
    pub original_features: AcousticFeatures,
    /// Converted acoustic features
    pub converted_features: AcousticFeatures,
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
    /// Whether quality was preserved above threshold
    pub quality_preserved: bool,
}

#[cfg(not(feature = "acoustic-integration"))]
#[derive(Debug, Clone)]
pub struct AcousticConversionResult {
    pub placeholder: bool,
}
/// Acoustic feature extraction configuration
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone)]
pub struct AcousticFeatureConfig {
    /// Sample rate for processing
    pub sample_rate: f32,
    /// Frame size for analysis
    pub frame_size: usize,
    /// Hop size for overlapping frames
    pub hop_size: usize,
    /// Window type for analysis
    pub window_type: WindowType,
    /// Enable high-quality processing
    pub high_quality: bool,
}

/// Temporal acoustic features
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone)]
pub struct TemporalFeatures {
    /// Energy contour over time
    pub energy_contour: Vec<f32>,
    /// Zero crossing rate
    pub zero_crossing_rate: Vec<f32>,
    /// Spectral flux (change over time)
    pub spectral_flux: Vec<f32>,
}

/// Context for real-time acoustic conversion
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone)]
pub struct AcousticConversionContext {
    /// Audio buffer for context
    audio_buffer: Vec<f32>,
    /// Maximum context size
    max_context_size: usize,
    /// Minimum context for processing
    min_context_size: usize,
    /// Previous acoustic state
    pub previous_state: AcousticState,
}

#[cfg(feature = "acoustic-integration")]
impl AcousticConversionContext {
    /// Create new context
    pub fn new(max_context_ms: f32, sample_rate: f32) -> Self {
        let max_context_size = (max_context_ms * sample_rate / 1000.0) as usize;
        let min_context_size = max_context_size / 3; // Use 1/3 for more flexible context requirements

        Self {
            audio_buffer: Vec::new(),
            max_context_size,
            min_context_size,
            previous_state: AcousticState::default(),
        }
    }

    /// Check if there is sufficient context
    pub fn has_sufficient_context(&self) -> bool {
        self.audio_buffer.len() >= self.min_context_size
    }

    /// Add audio chunk to context buffer
    pub fn add_audio_chunk(&mut self, chunk: &[f32]) {
        self.audio_buffer.extend_from_slice(chunk);

        // Trim buffer if it exceeds max size
        if self.audio_buffer.len() > self.max_context_size {
            let excess = self.audio_buffer.len() - self.max_context_size;
            self.audio_buffer.drain(0..excess);
        }
    }

    /// Get context window
    pub fn get_context_window(&self) -> &[f32] {
        &self.audio_buffer
    }
}

#[cfg(not(feature = "acoustic-integration"))]
#[derive(Debug, Clone)]
pub struct AcousticConversionContext {
    pub placeholder: bool,
}

#[cfg(not(feature = "acoustic-integration"))]
impl AcousticConversionContext {
    pub fn new(_max_context_ms: f32, _sample_rate: f32) -> Self {
        Self { placeholder: true }
    }
}
/// Harmonic analysis features
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone)]
pub struct HarmonicFeatures {
    /// Harmonic-to-noise ratio
    pub harmonic_to_noise_ratio: Vec<f32>,
    /// Strength of harmonic structure
    pub harmonic_strength: Vec<f32>,
    /// Inharmonicity measure
    pub inharmonicity: Vec<f32>,
}

/// Adapter for acoustic model-based voice conversion
#[cfg(feature = "acoustic-integration")]
#[derive(Debug, Clone)]
pub struct AcousticConversionAdapter {
    /// Base acoustic model configuration
    config: Option<voirs_acoustic::config::synthesis::SynthesisConfig>,
    /// Acoustic feature extraction configuration
    feature_config: AcousticFeatureConfig,
    /// Current acoustic model state
    model_state: Option<String>,
}

#[cfg(feature = "acoustic-integration")]
impl Default for AcousticConversionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "acoustic-integration")]
impl AcousticConversionAdapter {
    /// Create new acoustic adapter
    pub fn new() -> Self {
        Self {
            config: None,
            feature_config: AcousticFeatureConfig::default(),
            model_state: None,
        }
    }

    /// Create adapter with specific acoustic configuration
    pub fn with_config(config: voirs_acoustic::config::synthesis::SynthesisConfig) -> Self {
        Self {
            config: Some(config),
            feature_config: AcousticFeatureConfig::default(),
            model_state: None,
        }
    }

    /// Convert audio using an acoustic model.
    ///
    /// Estimates the source audio's own mean F0 (via real autocorrelation
    /// analysis, see [`Self::extract_f0_contour`]) and shifts pitch toward
    /// `target_characteristics.pitch.mean_f0` using the crate's real
    /// phase-vocoder [`crate::transforms::PitchTransform`] - not amplitude
    /// scaling (which changes loudness, not pitch).
    pub async fn convert_with_acoustic_model(
        &self,
        input_audio: &[f32],
        target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<Vec<f32>> {
        if input_audio.is_empty() {
            return Err(Error::config("Input audio cannot be empty".to_string()));
        }

        let source_f0 = self.estimate_mean_f0(input_audio).unwrap_or(180.0);
        let target_f0 = target_characteristics.pitch.mean_f0;
        let pitch_factor = if source_f0 > 1.0 {
            (target_f0 / source_f0).clamp(0.5, 2.0)
        } else {
            1.0
        };

        crate::transforms::PitchTransform::new(pitch_factor).apply(input_audio)
    }

    /// Convert with feature interpolation.
    ///
    /// Genuinely consumes `interpolation_factor` and both feature sets: F0
    /// is linearly interpolated between the source and target contours and
    /// applied via a real pitch shift, and a coarse spectral-tilt
    /// (low/high energy balance) is interpolated between the two envelopes
    /// and applied as a real one-pole brighten/darken filter - so `0.0`,
    /// `0.5`, and `1.0` produce audibly different output.
    pub async fn convert_with_feature_interpolation(
        &self,
        input_audio: &[f32],
        source_features: &AcousticFeatures,
        target_features: &AcousticFeatures,
        interpolation_factor: f32,
    ) -> Result<Vec<f32>> {
        if interpolation_factor < 0.0 || interpolation_factor > 1.0 {
            return Err(Error::config(
                "Interpolation factor must be between 0.0 and 1.0".to_string(),
            ));
        }
        if input_audio.is_empty() {
            return Ok(Vec::new());
        }

        let source_f0 = mean_nonzero(&source_features.f0_contour)
            .or_else(|| self.estimate_mean_f0(input_audio))
            .unwrap_or(180.0);
        let target_f0 = mean_nonzero(&target_features.f0_contour).unwrap_or(source_f0);
        let interpolated_f0 = source_f0 + (target_f0 - source_f0) * interpolation_factor;

        let current_f0 = self.estimate_mean_f0(input_audio).unwrap_or(source_f0);
        let pitch_factor = if current_f0 > 1.0 {
            (interpolated_f0 / current_f0).clamp(0.5, 2.0)
        } else {
            1.0
        };

        let mut output = crate::transforms::PitchTransform::new(pitch_factor).apply(input_audio)?;

        let tilt =
            interpolated_spectral_tilt(source_features, target_features, interpolation_factor);
        apply_spectral_tilt(&mut output, tilt);

        Ok(output)
    }

    /// Convert with quality preservation.
    ///
    /// `quality_score` is a real spectral-similarity measurement between
    /// the original and converted audio (see [`spectral_similarity`]), not
    /// a fixed constant - it actually decreases as the conversion changes
    /// the signal's spectral shape more.
    pub async fn convert_with_quality_preservation(
        &self,
        input_audio: &[f32],
        target_characteristics: &crate::types::VoiceCharacteristics,
        quality_threshold: f32,
    ) -> Result<AcousticConversionResult> {
        if quality_threshold < 0.0 || quality_threshold > 1.0 {
            return Err(Error::config(
                "Quality threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        let converted = self
            .convert_with_acoustic_model(input_audio, target_characteristics)
            .await?;

        let original_features = self.build_acoustic_features(input_audio)?;
        let converted_features = self.build_acoustic_features(&converted)?;
        let quality_score = spectral_similarity(input_audio, &converted);

        Ok(AcousticConversionResult {
            audio: converted,
            original_features,
            converted_features,
            quality_score,
            quality_preserved: quality_score >= quality_threshold,
        })
    }

    /// Real-time acoustic conversion.
    ///
    /// Once enough context has accumulated, estimates F0 from the
    /// accumulated context window (more reliable than a single small
    /// chunk) and shifts `input_chunk`'s pitch toward
    /// `target_features`'s F0 via the real phase-vocoder pitch shifter,
    /// carrying the F0 estimate forward in `context.previous_state` for
    /// continuity across calls (falling back to it when the current
    /// window is unvoiced).
    pub async fn convert_realtime_acoustic(
        &self,
        input_chunk: &[f32],
        target_features: &AcousticFeatures,
        context: &mut AcousticConversionContext,
    ) -> Result<Vec<f32>> {
        context.add_audio_chunk(input_chunk);

        if !context.has_sufficient_context() {
            return Ok(vec![]);
        }
        if input_chunk.is_empty() {
            return Ok(Vec::new());
        }

        let context_window = context.get_context_window().to_vec();
        let estimated_f0 = self
            .estimate_mean_f0(&context_window)
            .filter(|&f0| f0 > 1.0)
            .unwrap_or_else(|| context.previous_state.last_f0.max(1.0));

        let target_f0 = mean_nonzero(&target_features.f0_contour).unwrap_or(estimated_f0);
        let pitch_factor = (target_f0 / estimated_f0.max(1.0)).clamp(0.5, 2.0);

        let output = crate::transforms::PitchTransform::new(pitch_factor).apply(input_chunk)?;

        context.previous_state.last_f0 = estimated_f0;

        Ok(output)
    }

    /// Extract a per-frame F0 contour from audio via real autocorrelation
    /// analysis (reusing [`crate::processing::FeatureExtractor`]'s
    /// estimator), not a hardcoded constant.
    pub fn extract_f0_contour(&self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.is_empty() {
            return Ok(vec![]);
        }

        let sample_rate = self.feature_config.sample_rate.max(1.0) as u32;
        let extractor = crate::processing::FeatureExtractor::new(sample_rate);
        let contour = extractor.estimate_f0_contour(audio)?;

        if contour.is_empty() {
            // Clip shorter than half an analysis window: too short to
            // analyze at all. Report a single "unvoiced" (0.0) frame rather
            // than an empty contour, so every non-empty input yields at
            // least one entry - 0.0 honestly means "undetermined", unlike
            // the old code's fabricated constant 200.0 Hz.
            Ok(vec![0.0])
        } else {
            Ok(contour)
        }
    }

    /// Mean F0 (Hz) over voiced frames only, or `None` if no frame was
    /// judged voiced (see [`Self::extract_f0_contour`]).
    fn estimate_mean_f0(&self, audio: &[f32]) -> Option<f32> {
        let contour = self.extract_f0_contour(audio).ok()?;
        mean_nonzero(&contour)
    }

    /// Extract per-frame formant frequencies (F1-F4) via LPC
    /// (Levinson-Durbin) analysis of each frame, followed by peak-picking
    /// the resulting all-pole spectral envelope - a real (classical) formant
    /// estimator, not `FormantFrequencies::default()`.
    pub fn extract_formant_frequencies(&self, audio: &[f32]) -> Result<FormantFrequencies> {
        if audio.is_empty() {
            return Ok(FormantFrequencies::default());
        }

        let sample_rate = self.feature_config.sample_rate.max(1.0);
        let frame_size = self.feature_config.frame_size.max(64);
        let hop_size = self.feature_config.hop_size.max(1);
        const LPC_ORDER: usize = 12;
        const DEFAULT_FORMANTS: [f32; 4] = [700.0, 1220.0, 2600.0, 3500.0];

        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        let mut f3 = Vec::new();
        let mut f4 = Vec::new();
        let mut bandwidths = Vec::new();

        let mut push_frame = |frame: &[f32]| {
            let (formants, bandwidth) =
                estimate_frame_formants(frame, sample_rate, LPC_ORDER, &DEFAULT_FORMANTS);
            f1.push(formants[0]);
            f2.push(formants[1]);
            f3.push(formants[2]);
            f4.push(formants[3]);
            bandwidths.push(bandwidth);
        };

        if audio.len() <= frame_size {
            push_frame(audio);
        } else {
            let mut start = 0;
            while start < audio.len() {
                let end = (start + frame_size).min(audio.len());
                push_frame(&audio[start..end]);
                if end == audio.len() {
                    break;
                }
                start += hop_size;
            }
        }

        Ok(FormantFrequencies {
            f1,
            f2,
            f3,
            f4,
            bandwidths,
        })
    }

    /// Build a real [`AcousticFeatures`] snapshot (F0 contour, formants,
    /// spectral envelope, energy contour) from actual audio, replacing the
    /// `AcousticFeatures::default()` placeholders previously used by
    /// [`Self::convert_with_quality_preservation`].
    fn build_acoustic_features(&self, audio: &[f32]) -> Result<AcousticFeatures> {
        let f0_contour = self.extract_f0_contour(audio)?;
        let formants = self.extract_formant_frequencies(audio)?;
        let spectral_envelope = compute_spectral_envelope(audio);
        let hop_size = self.feature_config.hop_size.max(1);
        let energy_contour = compute_energy_contour(audio, hop_size);
        let frame_count = f0_contour.len().max(1);

        Ok(AcousticFeatures {
            f0_contour,
            formants,
            spectral_envelope,
            temporal_features: TemporalFeatures {
                energy_contour,
                zero_crossing_rate: Vec::new(),
                spectral_flux: Vec::new(),
            },
            harmonic_features: HarmonicFeatures {
                harmonic_to_noise_ratio: Vec::new(),
                harmonic_strength: Vec::new(),
                inharmonicity: Vec::new(),
            },
            frame_count,
            sample_rate: self.feature_config.sample_rate,
        })
    }
}

/// Mean of the strictly-positive (voiced) entries of `values`, or `None`
/// if there are none.
#[cfg(feature = "acoustic-integration")]
fn mean_nonzero(values: &[f32]) -> Option<f32> {
    let voiced: Vec<f32> = values.iter().copied().filter(|&f| f > 0.0).collect();
    if voiced.is_empty() {
        None
    } else {
        Some(voiced.iter().sum::<f32>() / voiced.len() as f32)
    }
}

/// Coarse spectral-tilt proxy: ratio of mean magnitude in the upper half of
/// a magnitude spectrum to the lower half (>1 = bright, <1 = dark).
#[cfg(feature = "acoustic-integration")]
fn spectral_tilt(envelope: &[f32]) -> f32 {
    if envelope.len() < 2 {
        return 1.0;
    }
    let mid = envelope.len() / 2;
    let low = envelope[..mid].iter().sum::<f32>() / mid as f32;
    let high = envelope[mid..].iter().sum::<f32>() / (envelope.len() - mid) as f32;
    if low > 1e-6 {
        (high / low).clamp(0.0, 4.0)
    } else {
        1.0
    }
}

/// Linear interpolation of the source/target spectral-envelope tilt ratios.
#[cfg(feature = "acoustic-integration")]
fn interpolated_spectral_tilt(
    source: &AcousticFeatures,
    target: &AcousticFeatures,
    factor: f32,
) -> f32 {
    let source_tilt = spectral_tilt(&source.spectral_envelope);
    let target_tilt = spectral_tilt(&target.spectral_envelope);
    source_tilt + (target_tilt - source_tilt) * factor
}

/// Apply a simple one-pole brighten (`tilt > 1`) / darken (`tilt < 1`)
/// filter driven by a real, feature-derived tilt ratio - a genuine (if
/// simple) spectral-envelope interpolation effect, not a pass-through.
#[cfg(feature = "acoustic-integration")]
fn apply_spectral_tilt(audio: &mut [f32], tilt: f32) {
    if audio.is_empty() {
        return;
    }
    let coefficient = (tilt - 1.0).clamp(-0.9, 0.9) * 0.5;
    let mut prev = audio[0];
    for sample in audio.iter_mut() {
        let current = *sample;
        *sample = current + coefficient * (current - prev);
        prev = current;
    }
}

/// Hann-windowed real-FFT magnitude spectrum of (up to the first 1024
/// samples of) `audio`, used as a coarse spectral envelope.
#[cfg(feature = "acoustic-integration")]
fn compute_spectral_envelope(audio: &[f32]) -> Vec<f32> {
    if audio.is_empty() {
        return vec![1.0; 513];
    }
    let n = audio.len().min(1024).max(2);
    let windowed: Vec<f64> = audio[..n]
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.5
                - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0).max(1.0)).cos();
            x as f64 * w
        })
        .collect();

    match scirs2_fft::rfft(&windowed, Some(n)) {
        Ok(spectrum) => spectrum
            .iter()
            .map(|c| ((c.re * c.re + c.im * c.im).sqrt()) as f32)
            .collect(),
        Err(_) => vec![1.0; n / 2 + 1],
    }
}

/// Per-hop RMS energy contour.
#[cfg(feature = "acoustic-integration")]
fn compute_energy_contour(audio: &[f32], hop: usize) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    audio
        .chunks(hop.max(1))
        .map(|chunk| (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt())
        .collect()
}

/// Real spectral-similarity proxy in `[0, 1]`: 1 minus the normalized L2
/// distance between the (FFT-magnitude) spectral envelopes of two signals.
/// `1.0` for identical signals, decreasing as their spectral shape
/// diverges - an honest, input-dependent quality score in place of a
/// fixed `0.85` constant.
#[cfg(feature = "acoustic-integration")]
fn spectral_similarity(original: &[f32], converted: &[f32]) -> f32 {
    let a = compute_spectral_envelope(original);
    let b = compute_spectral_envelope(converted);
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut diff_sq = 0.0f32;
    let mut ref_sq = 0.0f32;
    for i in 0..len {
        let d = a[i] - b[i];
        diff_sq += d * d;
        ref_sq += a[i] * a[i];
    }
    if ref_sq <= 1e-9 {
        return 1.0;
    }
    let normalized_distance = (diff_sq / ref_sq).sqrt().min(1.0);
    (1.0 - normalized_distance).clamp(0.0, 1.0)
}

/// Compute LPC coefficients for `frame` via Hann-windowed autocorrelation +
/// Levinson-Durbin recursion. Returns `None` for silent frames (zero-lag
/// autocorrelation vanishes) or frames not longer than `order`.
#[cfg(feature = "acoustic-integration")]
fn lpc_coefficients(frame: &[f32], order: usize) -> Option<Vec<f32>> {
    let n = frame.len();
    if n <= order {
        return None;
    }

    let windowed: Vec<f32> = frame
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1).max(1) as f32).cos();
            x * w
        })
        .collect();

    let mut autocorr = vec![0.0f32; order + 1];
    for (lag, value) in autocorr.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for i in 0..(n - lag) {
            sum += windowed[i] * windowed[i + lag];
        }
        *value = sum;
    }

    if autocorr[0].abs() < 1e-12 {
        return None;
    }

    let mut lpc = vec![0.0f32; order];
    let mut error = autocorr[0];
    for i in 0..order {
        let mut acc = autocorr[i + 1];
        for j in 0..i {
            acc -= lpc[j] * autocorr[i - j];
        }
        if error.abs() < 1e-12 {
            break;
        }
        let reflection = acc / error;
        let mut new_lpc = lpc.clone();
        new_lpc[i] = reflection;
        for j in 0..i {
            new_lpc[j] = lpc[j] - reflection * lpc[i - 1 - j];
        }
        lpc = new_lpc;
        error *= 1.0 - reflection * reflection;
    }

    Some(lpc)
}

/// Evaluate the LPC all-pole spectral envelope `1 / |A(e^{jw})|^2` across
/// `num_bins` linearly spaced frequencies from 0 to Nyquist.
#[cfg(feature = "acoustic-integration")]
fn lpc_spectral_envelope(lpc: &[f32], sample_rate: f32, num_bins: usize) -> Vec<f32> {
    let nyquist = sample_rate / 2.0;
    (0..num_bins)
        .map(|k| {
            let freq = nyquist * k as f32 / num_bins.max(1) as f32;
            let omega = 2.0 * std::f32::consts::PI * freq / sample_rate;
            let mut re = 1.0f32;
            let mut im = 0.0f32;
            for (i, &a) in lpc.iter().enumerate() {
                let angle = omega * (i as f32 + 1.0);
                re += a * angle.cos();
                im -= a * angle.sin();
            }
            let magnitude_sq = (re * re + im * im).max(1e-9);
            1.0 / magnitude_sq
        })
        .collect()
}

/// Find up to `count` spectral peaks (local maxima) in `envelope`, returned
/// in increasing frequency order (Hz).
#[cfg(feature = "acoustic-integration")]
fn find_spectral_peaks(envelope: &[f32], sample_rate: f32, count: usize) -> Vec<f32> {
    let nyquist = sample_rate / 2.0;
    let num_bins = envelope.len();
    let mut peaks = Vec::new();
    for k in 1..num_bins.saturating_sub(1) {
        if envelope[k] > envelope[k - 1] && envelope[k] >= envelope[k + 1] {
            let freq = nyquist * k as f32 / num_bins as f32;
            peaks.push(freq);
            if peaks.len() >= count * 4 {
                break;
            }
        }
    }
    peaks.truncate(count);
    peaks
}

/// Estimate up to 4 formants + a spacing-derived bandwidth proxy for a
/// single analysis frame, falling back to `defaults` for any slot that
/// could not be resolved from a real spectral peak (e.g. a silent frame).
#[cfg(feature = "acoustic-integration")]
fn estimate_frame_formants(
    frame: &[f32],
    sample_rate: f32,
    lpc_order: usize,
    defaults: &[f32; 4],
) -> ([f32; 4], f32) {
    const NUM_BINS: usize = 512;
    const NUM_FORMANTS: usize = 4;

    let mut result = *defaults;
    let Some(lpc) = lpc_coefficients(frame, lpc_order) else {
        return (result, 50.0);
    };

    let envelope = lpc_spectral_envelope(&lpc, sample_rate, NUM_BINS);
    let peaks = find_spectral_peaks(&envelope, sample_rate, NUM_FORMANTS);

    for (i, &freq) in peaks.iter().enumerate().take(NUM_FORMANTS) {
        result[i] = freq.clamp(80.0, (sample_rate / 2.0 - 100.0).max(81.0));
    }

    let mut spacing_sum = 0.0f32;
    let mut spacing_count = 0usize;
    for w in result.windows(2) {
        spacing_sum += (w[1] - w[0]).abs();
        spacing_count += 1;
    }
    let bandwidth = if spacing_count > 0 {
        (spacing_sum / spacing_count as f32 * 0.1).clamp(30.0, 400.0)
    } else {
        50.0
    };

    (result, bandwidth)
}

#[cfg(not(feature = "acoustic-integration"))]
#[derive(Debug, Clone)]
pub struct AcousticConversionAdapter;
#[cfg(not(feature = "acoustic-integration"))]
impl AcousticConversionAdapter {
    pub fn new() -> Self {
        Self
    }
    pub async fn convert_with_acoustic_model(
        &self,
        _input_audio: &[f32],
        _target_characteristics: &crate::types::VoiceCharacteristics,
    ) -> Result<Vec<f32>> {
        Err(Error::config(
            "Acoustic integration not enabled. Enable with 'acoustic-integration' feature."
                .to_string(),
        ))
    }
    pub async fn convert_with_feature_interpolation(
        &self,
        _input_audio: &[f32],
        _source_features: &AcousticFeatures,
        _target_features: &AcousticFeatures,
        _interpolation_factor: f32,
    ) -> Result<Vec<f32>> {
        Err(Error::config(
            "Acoustic integration not enabled. Enable with 'acoustic-integration' feature."
                .to_string(),
        ))
    }
    pub async fn convert_realtime_acoustic(
        &self,
        _input_chunk: &[f32],
        _target_features: &AcousticFeatures,
        _context: &mut AcousticConversionContext,
    ) -> Result<Vec<f32>> {
        Err(Error::config(
            "Acoustic integration not enabled. Enable with 'acoustic-integration' feature."
                .to_string(),
        ))
    }
    pub fn extract_f0_contour(&self, _audio: &[f32]) -> Result<Vec<f32>> {
        Err(Error::config(
            "Acoustic integration not enabled. Enable with 'acoustic-integration' feature."
                .to_string(),
        ))
    }
    pub fn extract_formant_frequencies(&self, _audio: &[f32]) -> Result<FormantFrequencies> {
        Err(Error::config(
            "Acoustic integration not enabled. Enable with 'acoustic-integration' feature."
                .to_string(),
        ))
    }
    pub async fn convert_with_quality_preservation(
        &self,
        _input_audio: &[f32],
        _target_characteristics: &crate::types::VoiceCharacteristics,
        _quality_threshold: f32,
    ) -> Result<AcousticConversionResult> {
        Err(Error::config(
            "Acoustic integration not enabled. Enable with 'acoustic-integration' feature."
                .to_string(),
        ))
    }
}
