//! Basic forced alignment implementation
//!
//! This module provides a basic forced alignment implementation that can align
//! phonemes or text with audio using dynamic time warping and acoustic models.

use crate::traits::*;
use crate::RecognitionError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use voirs_sdk::{AudioBuffer, LanguageCode, Phoneme};

/// Basic forced alignment model implementation
pub struct ForcedAlignModel {
    /// Model configuration
    config: ForcedAlignConfig,
    /// Model state
    state: Arc<RwLock<ForcedAlignState>>,
    /// Supported languages
    supported_languages: Vec<LanguageCode>,
    /// Model metadata
    metadata: PhonemeRecognizerMetadata,
}

/// Forced alignment configuration
#[derive(Debug, Clone)]
pub struct ForcedAlignConfig {
    /// Path to the acoustic model
    pub model_path: String,
    /// Path to the pronunciation dictionary
    pub dictionary_path: Option<String>,
    /// Frame shift in milliseconds
    pub frame_shift_ms: f32,
    /// Frame length in milliseconds
    pub frame_length_ms: f32,
    /// Beam width for alignment
    pub beam_width: usize,
    /// Minimum phoneme duration in milliseconds
    pub min_phoneme_duration_ms: f32,
    /// Maximum phoneme duration in milliseconds
    pub max_phoneme_duration_ms: f32,
    /// Use GPU if available
    pub use_gpu: bool,
    /// Number of threads
    pub num_threads: usize,
    /// Confidence threshold for accepting alignments
    pub confidence_threshold: f32,
}

impl Default for ForcedAlignConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            dictionary_path: None,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            beam_width: 100,
            min_phoneme_duration_ms: 20.0,
            max_phoneme_duration_ms: 1000.0,
            use_gpu: false,
            num_threads: num_cpus::get(),
            confidence_threshold: 0.3,
        }
    }
}

/// Internal state for forced alignment model
struct ForcedAlignState {
    /// Whether the model is loaded
    loaded: bool,
    /// Model path
    model_path: String,
    /// Dictionary path
    dictionary_path: Option<String>,
    /// Pronunciation dictionary
    dictionary: HashMap<String, Vec<String>>,
    /// Loading time
    load_time: Option<Duration>,
    /// Alignment count
    alignment_count: usize,
    /// Total alignment time
    total_alignment_time: Duration,
}

impl ForcedAlignState {
    fn new(model_path: String, dictionary_path: Option<String>) -> Self {
        Self {
            loaded: false,
            model_path,
            dictionary_path,
            dictionary: HashMap::new(),
            load_time: None,
            alignment_count: 0,
            total_alignment_time: Duration::ZERO,
        }
    }
}

impl ForcedAlignModel {
    /// Create a new forced alignment model
    pub async fn new(
        model_path: String,
        dictionary_path: Option<String>,
    ) -> Result<Self, RecognitionError> {
        // Validate model file exists
        if !Path::new(&model_path).exists() {
            return Err(RecognitionError::ModelLoadError {
                message: format!("Model file not found: {}", model_path),
                source: None,
            });
        }

        // Validate dictionary file if provided
        if let Some(ref dict_path) = dictionary_path {
            if !Path::new(dict_path).exists() {
                return Err(RecognitionError::ModelLoadError {
                    message: format!("Dictionary file not found: {}", dict_path),
                    source: None,
                });
            }
        }

        let config = ForcedAlignConfig {
            model_path: model_path.clone(),
            dictionary_path: dictionary_path.clone(),
            ..Default::default()
        };

        // Basic forced alignment supports common languages
        let supported_languages = vec![
            LanguageCode::EnUs,
            LanguageCode::EnGb,
            LanguageCode::DeDe,
            LanguageCode::FrFr,
            LanguageCode::EsEs,
        ];

        let metadata = PhonemeRecognizerMetadata {
            name: "Basic Forced Alignment".to_string(),
            version: "1.0.0".to_string(),
            description: "Basic forced alignment using dynamic time warping".to_string(),
            supported_languages: supported_languages.clone(),
            alignment_methods: vec![AlignmentMethod::Forced, AlignmentMethod::Hybrid],
            alignment_accuracy: 0.85,
            supported_features: vec![
                PhonemeRecognitionFeature::WordAlignment,
                PhonemeRecognitionFeature::CustomPronunciation,
                PhonemeRecognitionFeature::ConfidenceScoring,
                PhonemeRecognitionFeature::PronunciationAssessment,
            ],
        };

        let state = Arc::new(RwLock::new(ForcedAlignState::new(
            model_path,
            dictionary_path,
        )));

        Ok(Self {
            config,
            state,
            supported_languages,
            metadata,
        })
    }

    /// Create with custom configuration
    pub async fn with_config(config: ForcedAlignConfig) -> Result<Self, RecognitionError> {
        Self::new(config.model_path.clone(), config.dictionary_path.clone()).await
    }

    /// Load the model if not already loaded
    async fn ensure_loaded(&self) -> Result<(), RecognitionError> {
        let mut state = self.state.write().await;

        if !state.loaded {
            let start_time = Instant::now();

            tracing::info!("Loading forced alignment model: {}", state.model_path);

            // Verify model file exists and is readable
            if !std::path::Path::new(&state.model_path).exists() {
                return Err(RecognitionError::ModelLoadError {
                    message: format!("Model file not found: {}", state.model_path),
                    source: None,
                });
            }

            let model_metadata = std::fs::metadata(&state.model_path).map_err(|e| {
                RecognitionError::ModelLoadError {
                    message: format!("Failed to read model file metadata: {}", e),
                    source: Some(Box::new(e)),
                }
            })?;

            tracing::info!(
                "Model file size: {:.2} MB",
                model_metadata.len() as f64 / (1024.0 * 1024.0)
            );

            // Load pronunciation dictionary if provided
            if let Some(ref dict_path) = state.dictionary_path {
                tracing::info!("Loading pronunciation dictionary: {}", dict_path);
                state.dictionary = self.load_dictionary(dict_path).await?;
            } else {
                // Load default dictionary
                state.dictionary = self.load_default_dictionary().await?;
            }

            state.loaded = true;
            state.load_time = Some(start_time.elapsed());

            tracing::info!(
                "Forced alignment model loaded in {:?}",
                state.load_time.expect("load_time was just set to Some")
            );
        }

        Ok(())
    }

    /// Load pronunciation dictionary from file (CMU Dict format)
    async fn load_dictionary(
        &self,
        dict_path: &str,
    ) -> Result<HashMap<String, Vec<String>>, RecognitionError> {
        use std::io::{BufRead, BufReader};

        if dict_path.is_empty() {
            return self.load_default_dictionary().await;
        }

        let file =
            std::fs::File::open(dict_path).map_err(|e| RecognitionError::ModelLoadError {
                message: format!("Failed to open dictionary file: {}", e),
                source: Some(Box::new(e)),
            })?;

        let reader = BufReader::new(file);
        let mut dictionary = HashMap::new();
        let mut line_count = 0;

        for line_result in reader.lines() {
            let line = line_result.map_err(|e| RecognitionError::ModelLoadError {
                message: format!("Failed to read dictionary line: {}", e),
                source: Some(Box::new(e)),
            })?;

            line_count += 1;

            // Skip comments and empty lines
            if line.trim().is_empty() || line.starts_with(";;;") {
                continue;
            }

            // Parse CMU Dict format: WORD  P1 P2 P3 ...
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            // Extract word (may have variant marker like WORD(2))
            let word_part = parts[0];
            let word = if let Some(paren_pos) = word_part.find('(') {
                word_part[..paren_pos].to_uppercase()
            } else {
                word_part.to_uppercase()
            };

            // Extract phonemes (skip stress markers if present)
            let phonemes: Vec<String> = parts[1..]
                .iter()
                .map(|p| {
                    // Remove stress markers (0, 1, 2) from vowels
                    p.chars().filter(|c| !c.is_numeric()).collect::<String>()
                })
                .collect();

            // Only insert if we don't have this word yet (use first variant)
            dictionary.entry(word).or_insert(phonemes);
        }

        tracing::info!(
            "Loaded {} words from pronunciation dictionary (processed {} lines)",
            dictionary.len(),
            line_count
        );

        if dictionary.is_empty() {
            return Err(RecognitionError::ModelLoadError {
                message: "Dictionary file contains no valid entries".to_string(),
                source: None,
            });
        }

        Ok(dictionary)
    }

    /// Load default pronunciation dictionary
    async fn load_default_dictionary(
        &self,
    ) -> Result<HashMap<String, Vec<String>>, RecognitionError> {
        let mut dictionary = HashMap::new();

        // Add common English words with ARPAbet phonemes
        // Vowels
        dictionary.insert(
            "HELLO".to_string(),
            vec!["HH".into(), "AH".into(), "L".into(), "OW".into()],
        );
        dictionary.insert(
            "WORLD".to_string(),
            vec!["W".into(), "ER".into(), "L".into(), "D".into()],
        );
        dictionary.insert(
            "TEST".to_string(),
            vec!["T".into(), "EH".into(), "S".into(), "T".into()],
        );
        dictionary.insert(
            "SPEECH".to_string(),
            vec!["S".into(), "P".into(), "IY".into(), "CH".into()],
        );
        dictionary.insert(
            "VOICE".to_string(),
            vec!["V".into(), "OY".into(), "S".into()],
        );
        dictionary.insert(
            "RECOGNITION".to_string(),
            vec![
                "R".into(),
                "EH".into(),
                "K".into(),
                "AH".into(),
                "G".into(),
                "N".into(),
                "IH".into(),
                "SH".into(),
                "AH".into(),
                "N".into(),
            ],
        );
        dictionary.insert(
            "PHONEME".to_string(),
            vec!["F".into(), "OW".into(), "N".into(), "IY".into(), "M".into()],
        );
        dictionary.insert(
            "ALIGN".to_string(),
            vec!["AH".into(), "L".into(), "AY".into(), "N".into()],
        );
        dictionary.insert(
            "FORCED".to_string(),
            vec!["F".into(), "AO".into(), "R".into(), "S".into(), "T".into()],
        );
        dictionary.insert(
            "AUDIO".to_string(),
            vec!["AO".into(), "D".into(), "IY".into(), "OW".into()],
        );

        // Common words
        dictionary.insert("THE".to_string(), vec!["DH".into(), "AH".into()]);
        dictionary.insert("A".to_string(), vec!["AH".into()]);
        dictionary.insert("IS".to_string(), vec!["IH".into(), "Z".into()]);
        dictionary.insert("TO".to_string(), vec!["T".into(), "UW".into()]);
        dictionary.insert("AND".to_string(), vec!["AH".into(), "N".into(), "D".into()]);
        dictionary.insert("OF".to_string(), vec!["AH".into(), "V".into()]);
        dictionary.insert("IN".to_string(), vec!["IH".into(), "N".into()]);
        dictionary.insert("FOR".to_string(), vec!["F".into(), "AO".into(), "R".into()]);
        dictionary.insert(
            "WITH".to_string(),
            vec!["W".into(), "IH".into(), "DH".into()],
        );
        dictionary.insert("ON".to_string(), vec!["AA".into(), "N".into()]);

        // Numbers
        dictionary.insert("ONE".to_string(), vec!["W".into(), "AH".into(), "N".into()]);
        dictionary.insert("TWO".to_string(), vec!["T".into(), "UW".into()]);
        dictionary.insert(
            "THREE".to_string(),
            vec!["TH".into(), "R".into(), "IY".into()],
        );
        dictionary.insert(
            "FOUR".to_string(),
            vec!["F".into(), "AO".into(), "R".into()],
        );
        dictionary.insert(
            "FIVE".to_string(),
            vec!["F".into(), "AY".into(), "V".into()],
        );

        tracing::info!(
            "Loaded default pronunciation dictionary with {} words",
            dictionary.len()
        );

        Ok(dictionary)
    }

    /// Perform forced alignment using dynamic time warping
    async fn align_with_dtw(
        &self,
        audio: &AudioBuffer,
        phonemes: &[Phoneme],
        config: Option<&PhonemeRecognitionConfig>,
    ) -> Result<PhonemeAlignment, RecognitionError> {
        self.ensure_loaded().await?;

        let start_time = Instant::now();

        // Extract acoustic features
        let features = self.extract_features(audio).await?;

        // Perform DTW alignment
        let alignment = self.dtw_align(&features, phonemes, config).await?;

        // Update statistics
        let mut state = self.state.write().await;
        state.alignment_count += 1;
        state.total_alignment_time += start_time.elapsed();

        Ok(alignment)
    }

    /// Extract acoustic features from audio
    async fn extract_features(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        let samples = audio.samples();
        let frame_length =
            (self.config.frame_length_ms / 1000.0 * audio.sample_rate() as f32) as usize;
        let frame_shift =
            (self.config.frame_shift_ms / 1000.0 * audio.sample_rate() as f32) as usize;

        let mut features = Vec::new();
        let mut start = 0;

        while start + frame_length <= samples.len() {
            let frame = &samples[start..start + frame_length];
            let feature_vector = self.compute_mfcc(frame, audio.sample_rate())?;
            features.push(feature_vector);
            start += frame_shift;
        }

        if features.is_empty() {
            return Err(RecognitionError::PhonemeRecognitionError {
                message: "No features extracted from audio".to_string(),
                source: None,
            });
        }

        Ok(features)
    }

    /// Compute MFCC features for a frame using real DSP
    fn compute_mfcc(&self, frame: &[f32], sample_rate: u32) -> Result<Vec<f32>, RecognitionError> {
        use scirs2_core::ndarray::*;
        use scirs2_fft::rfft;

        const NUM_MEL_FILTERS: usize = 26;
        const NUM_MFCC: usize = 13;

        // Apply Hamming window
        let windowed: Vec<f32> = frame
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let window = 0.54
                    - 0.46
                        * (2.0 * std::f32::consts::PI * i as f32 / (frame.len() - 1) as f32).cos();
                sample * window
            })
            .collect();

        // Compute power spectrum using SciRS2-FFT
        let fft_size = frame.len().next_power_of_two();
        let mut fft_input = vec![0.0f64; fft_size];
        for (i, &val) in windowed.iter().enumerate() {
            fft_input[i] = val as f64;
        }

        let spectrum = rfft(&fft_input, Some(fft_size)).map_err(|e| {
            RecognitionError::PhonemeRecognitionError {
                message: format!("FFT failed: {:?}", e),
                source: None,
            }
        })?;

        // Compute power spectrum
        let power_spectrum: Vec<f32> = spectrum
            .iter()
            .map(|c| {
                let power = c.re * c.re + c.im * c.im;
                power as f32
            })
            .collect();

        // Apply Mel filterbank
        let mel_energies = self.apply_mel_filterbank(&power_spectrum, sample_rate, fft_size)?;

        // Compute log energies
        let log_mel_energies: Vec<f32> =
            mel_energies.iter().map(|&e| (e.max(1e-10)).ln()).collect();

        // Apply DCT to get MFCC coefficients
        let mfcc = self.apply_dct(&log_mel_energies, NUM_MFCC)?;

        Ok(mfcc)
    }

    /// Apply Mel filterbank to power spectrum
    fn apply_mel_filterbank(
        &self,
        power_spectrum: &[f32],
        sample_rate: u32,
        fft_size: usize,
    ) -> Result<Vec<f32>, RecognitionError> {
        const NUM_MEL_FILTERS: usize = 26;
        const MIN_FREQ_HZ: f32 = 0.0;
        let max_freq_hz = (sample_rate as f32 / 2.0).min(8000.0);

        // Convert Hz to Mel scale
        let hz_to_mel = |freq: f32| 2595.0 * (1.0 + freq / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let min_mel = hz_to_mel(MIN_FREQ_HZ);
        let max_mel = hz_to_mel(max_freq_hz);

        // Create Mel filter bank
        let mut mel_filters = vec![vec![0.0; power_spectrum.len()]; NUM_MEL_FILTERS];

        // Mel filter center points
        let mel_points: Vec<f32> = (0..NUM_MEL_FILTERS + 2)
            .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (NUM_MEL_FILTERS + 1) as f32)
            .collect();

        let hz_points: Vec<f32> = mel_points.iter().map(|&mel| mel_to_hz(mel)).collect();

        // Convert Hz points to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((fft_size as f32 + 1.0) * hz / sample_rate as f32).floor() as usize)
            .collect();

        // Build triangular filters
        for i in 0..NUM_MEL_FILTERS {
            let start = bin_points[i];
            let center = bin_points[i + 1];
            let end = bin_points[i + 2];

            // Rising slope
            for j in start..center {
                if center > start {
                    mel_filters[i][j] = (j - start) as f32 / (center - start) as f32;
                }
            }

            // Falling slope
            for j in center..end {
                if end > center && j < power_spectrum.len() {
                    mel_filters[i][j] = (end - j) as f32 / (end - center) as f32;
                }
            }
        }

        // Apply filters to power spectrum
        let mut mel_energies = vec![0.0; NUM_MEL_FILTERS];
        for i in 0..NUM_MEL_FILTERS {
            mel_energies[i] = power_spectrum
                .iter()
                .zip(mel_filters[i].iter())
                .map(|(&power, &filter)| power * filter)
                .sum();
        }

        Ok(mel_energies)
    }

    /// Apply Discrete Cosine Transform (DCT)
    fn apply_dct(&self, input: &[f32], num_coeffs: usize) -> Result<Vec<f32>, RecognitionError> {
        let n = input.len();
        let mut output = vec![0.0; num_coeffs];

        for k in 0..num_coeffs {
            let mut sum = 0.0;
            for (i, &val) in input.iter().enumerate() {
                sum +=
                    val * ((std::f32::consts::PI * k as f32 * (i as f32 + 0.5)) / n as f32).cos();
            }
            output[k] = sum;
        }

        Ok(output)
    }

    /// Perform DTW alignment between features and phonemes
    async fn dtw_align(
        &self,
        features: &[Vec<f32>],
        phonemes: &[Phoneme],
        _config: Option<&PhonemeRecognitionConfig>,
    ) -> Result<PhonemeAlignment, RecognitionError> {
        let frame_duration = self.config.frame_shift_ms / 1000.0;
        let total_duration = features.len() as f32 * frame_duration;

        if phonemes.is_empty() {
            return Ok(PhonemeAlignment {
                phonemes: Vec::new(),
                total_duration,
                alignment_confidence: 0.0,
                word_alignments: Vec::new(),
            });
        }

        // Perform real DTW alignment
        let (path, _total_cost) = self.compute_dtw(features, phonemes)?;
        // Same projection the DTW used, reused for the confidence posteriors.
        let templates = self.phoneme_templates_mfcc(phonemes)?;

        // Convert DTW path to phoneme alignment
        let mut aligned_phonemes = Vec::new();
        let mut word_alignments = Vec::new();

        // Group frames by phoneme
        let mut phoneme_frames: Vec<Vec<usize>> = vec![Vec::new(); phonemes.len()];
        for (frame_idx, phoneme_idx) in &path {
            if *phoneme_idx < phoneme_frames.len() {
                phoneme_frames[*phoneme_idx].push(*frame_idx);
            }
        }

        // Create aligned phonemes with timing information
        for (i, phoneme) in phonemes.iter().enumerate() {
            if phoneme_frames[i].is_empty() {
                continue; // Skip phonemes with no aligned frames
            }

            let start_frame = *phoneme_frames[i]
                .first()
                .expect("phoneme_frames[i] is non-empty (checked above)");
            let end_frame = *phoneme_frames[i]
                .last()
                .expect("phoneme_frames[i] is non-empty (checked above)");

            let start_time = start_frame as f32 * frame_duration;
            let end_time = (end_frame + 1) as f32 * frame_duration;

            // Real per-phoneme confidence: how much better this phoneme explains the
            // frames it was given than any of the other phonemes in the sequence would.
            let confidence = self.phoneme_posterior(features, &templates, &phoneme_frames[i], i);

            aligned_phonemes.push(AlignedPhoneme {
                phoneme: phoneme.clone(),
                start_time,
                end_time,
                confidence,
            });
        }

        // Create word alignments (all phonemes as one word for now)
        if !aligned_phonemes.is_empty() {
            let word_alignment = WordAlignment {
                word: "aligned_sequence".to_string(),
                start_time: aligned_phonemes[0].start_time,
                end_time: aligned_phonemes
                    .last()
                    .expect("aligned_phonemes is non-empty (checked above)")
                    .end_time,
                phonemes: aligned_phonemes.clone(),
                confidence: aligned_phonemes.iter().map(|p| p.confidence).sum::<f32>()
                    / aligned_phonemes.len() as f32,
            };
            word_alignments.push(word_alignment);
        }

        let overall_confidence = if !aligned_phonemes.is_empty() {
            aligned_phonemes.iter().map(|p| p.confidence).sum::<f32>()
                / aligned_phonemes.len() as f32
        } else {
            0.0
        };

        Ok(PhonemeAlignment {
            phonemes: aligned_phonemes,
            total_duration,
            alignment_confidence: overall_confidence,
            word_alignments,
        })
    }

    /// Compute DTW alignment path between features and phonemes
    fn compute_dtw(
        &self,
        features: &[Vec<f32>],
        phonemes: &[Phoneme],
    ) -> Result<(Vec<(usize, usize)>, f32), RecognitionError> {
        let n = features.len();
        let m = phonemes.len();

        if n == 0 || m == 0 {
            return Ok((Vec::new(), 0.0));
        }

        // Project each phoneme template into the same cepstral space as the features,
        // once, instead of rebuilding it for all n*m matrix cells.
        let templates = self.phoneme_templates_mfcc(phonemes)?;

        // Initialize DTW cost matrix
        let mut cost_matrix = vec![vec![f32::INFINITY; m + 1]; n + 1];
        cost_matrix[0][0] = 0.0;

        // Fill cost matrix using dynamic programming
        for i in 1..=n {
            for j in 1..=m {
                // Calculate local cost (distance between feature and phoneme)
                let local_cost = Self::local_cost(&features[i - 1], &templates[j - 1]);

                // Find minimum cost path
                let min_prev_cost = cost_matrix[i - 1][j]
                    .min(cost_matrix[i][j - 1])
                    .min(cost_matrix[i - 1][j - 1]);

                cost_matrix[i][j] = local_cost + min_prev_cost;
            }
        }

        // Backtrack to find optimal path
        let mut path = Vec::new();
        let mut i = n;
        let mut j = m;

        while i > 0 && j > 0 {
            path.push((i - 1, j - 1));

            // Find which direction we came from
            let diag = cost_matrix[i - 1][j - 1];
            let left = cost_matrix[i][j - 1];
            let up = cost_matrix[i - 1][j];

            if diag <= left && diag <= up {
                i -= 1;
                j -= 1;
            } else if left <= up {
                j -= 1;
            } else {
                i -= 1;
            }
        }

        path.reverse();

        let final_cost = cost_matrix[n][m];

        Ok((path, final_cost))
    }

    /// Project every phoneme's spectral template into MFCC space.
    ///
    /// # Errors
    /// Propagates any DCT failure from [`Self::apply_dct`].
    fn phoneme_templates_mfcc(
        &self,
        phonemes: &[Phoneme],
    ) -> Result<Vec<Vec<f32>>, RecognitionError> {
        // Distinct symbols dominate; cache so a repeated phoneme costs one lookup.
        let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
        let mut templates = Vec::with_capacity(phonemes.len());
        for phoneme in phonemes {
            let key = phoneme.symbol.to_uppercase();
            if let Some(cached) = cache.get(&key) {
                templates.push(cached.clone());
                continue;
            }
            let template = self.template_mfcc(phoneme)?;
            cache.insert(key, template.clone());
            templates.push(template);
        }
        Ok(templates)
    }

    /// Convert a phoneme's spectral template into the same cepstral space as the
    /// features extracted from audio.
    ///
    /// [`Self::create_phoneme_template`] describes a *log-mel spectral energy profile*,
    /// while [`Self::compute_mfcc`] produces *cepstral* coefficients — the DCT of the
    /// log mel energies. Comparing the two directly is meaningless, so the template is
    /// pushed through the identical tail of the feature pipeline: spread across the
    /// filterbank's `NUM_MEL_FILTERS` bands, log-compressed with the same floor, then
    /// DCT-transformed to the same number of coefficients.
    ///
    /// # Errors
    /// Propagates any DCT failure from [`Self::apply_dct`].
    fn template_mfcc(&self, phoneme: &Phoneme) -> Result<Vec<f32>, RecognitionError> {
        /// Must match `apply_mel_filterbank`.
        const NUM_MEL_FILTERS: usize = 26;
        /// Must match `compute_mfcc`.
        const NUM_MFCC: usize = 13;
        /// Must match the floor used on real mel energies in `compute_mfcc`.
        const ENERGY_FLOOR: f32 = 1e-10;

        let profile = self.create_phoneme_template(phoneme);
        if profile.is_empty() {
            return Ok(vec![0.0; NUM_MFCC]);
        }

        // Stretch the coarse profile across the filterbank by linear interpolation, so
        // that bin k of the profile lands on the mel band covering the same frequency.
        let mut mel_energies = Vec::with_capacity(NUM_MEL_FILTERS);
        #[allow(clippy::cast_precision_loss)]
        let last = (profile.len() - 1) as f32;
        for band in 0..NUM_MEL_FILTERS {
            #[allow(clippy::cast_precision_loss)]
            let position = if NUM_MEL_FILTERS > 1 {
                band as f32 * last / (NUM_MEL_FILTERS - 1) as f32
            } else {
                0.0
            };
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let lower = position.floor() as usize;
            let upper = (lower + 1).min(profile.len() - 1);
            #[allow(clippy::cast_precision_loss)]
            let fraction = position - lower as f32;
            let value = profile[lower] * (1.0 - fraction) + profile[upper] * fraction;
            mel_energies.push(value.max(0.0));
        }

        let log_mel: Vec<f32> = mel_energies
            .iter()
            .map(|&e| e.max(ENERGY_FLOOR).ln())
            .collect();

        self.apply_dct(&log_mel, NUM_MFCC)
    }

    /// Local cost between a real feature frame and a phoneme template, both in MFCC
    /// space.
    ///
    /// The zeroth coefficient is dropped because it encodes the frame's overall loudness
    /// rather than its spectral shape, and the remaining coefficients are compared with
    /// a cosine distance, which is invariant to the absolute scale of either vector.
    /// The result lies in `[0, 2]`, where 0 means the frame's spectral shape matches the
    /// template exactly.
    fn local_cost(feature: &[f32], template: &[f32]) -> f32 {
        const EPSILON: f32 = 1e-8;
        /// Neutral cost used when a vector carries no shape information at all.
        const NEUTRAL: f32 = 1.0;

        fn shape(v: &[f32]) -> &[f32] {
            if v.len() > 1 {
                &v[1..]
            } else {
                v
            }
        }
        let a = shape(feature);
        let b = shape(template);
        let len = a.len().min(b.len());
        if len == 0 {
            return NEUTRAL;
        }

        let dot: f32 = a[..len].iter().zip(&b[..len]).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < EPSILON || norm_b < EPSILON {
            return NEUTRAL;
        }

        (1.0 - (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)).clamp(0.0, 2.0)
    }

    /// Create an acoustically-motivated MFCC-like template for a phoneme.
    ///
    /// The template vector represents a log-mel spectral energy distribution across
    /// `num_coeffs` frequency bins ordered from low (bin 0) to high frequency.
    /// Templates are derived from established formant frequency knowledge:
    ///
    /// - Vowels: concentrated energy at F1 and F2 formant positions
    /// - Fricatives: concentrated energy in high-frequency bins (bins 3+)
    /// - Stops: brief burst energy with relative silence in preceding bins
    /// - Nasals: low-frequency resonance with anti-formant notch around 1kHz (bin 1)
    /// - Silence/SIL/SP: near-zero across all bins
    /// - Default: flat (white-noise-like) profile
    ///
    /// The final template is L2-normalized so that cosine similarity comparisons
    /// are meaningful without amplitude biasing.
    fn create_phoneme_template(&self, phoneme: &Phoneme) -> Vec<f32> {
        let num_coeffs = 13;
        let symbol = phoneme.symbol.to_uppercase();
        let symbol = symbol.as_str();

        // Bin layout (num_coeffs = 13):
        //   bin 0  ~  0–500 Hz   (F1 low vowels)
        //   bin 1  ~  500–1000 Hz (F1 high vowels / nasal resonance)
        //   bin 2  ~  1000–1500 Hz
        //   bin 3  ~  1500–2000 Hz (F2 front vowels)
        //   bin 4  ~  2000–2500 Hz (F2 front vowels high)
        //   bins 5-12 ~  2500 Hz and above (fricative / stop burst region)
        //
        // We fill the template as a sum of Gaussian "bumps" centred at relevant bins
        // to produce smooth, band-limited energy profiles.

        let gaussian = |bin: f32, centre: f32, width: f32, amplitude: f32| -> f32 {
            amplitude * (-(bin - centre).powi(2) / (2.0 * width * width)).exp()
        };

        let mut template = vec![0.0f32; num_coeffs];

        match symbol {
            // ── Vowels ───────────────────────────────────────────────────────────────
            // /a/ (AA, AH, AW): broad F1≈800 Hz (bin 1), F2≈1200 Hz (bin 2)
            "A" | "AA" | "AH" | "AW" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 1.0, 0.8, 3.0)   // F1 broad ~800 Hz
                        + gaussian(b, 2.0, 0.6, 2.0); // F2 ~1200 Hz
                }
            }
            // /i/ (IY, IH): low F1≈350 Hz (bin 0), high F2≈2300 Hz (bin 4)
            "I" | "IY" | "IH" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 0.0, 0.5, 2.5)   // F1 low ~350 Hz
                        + gaussian(b, 4.0, 0.7, 3.0); // F2 high ~2300 Hz
                }
            }
            // /u/ (UW, UH): low F1≈350 Hz (bin 0), low F2≈800 Hz (bin 1)
            "U" | "UW" | "UH" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 0.0, 0.5, 2.5)   // F1 low ~350 Hz
                        + gaussian(b, 1.0, 0.5, 2.0); // F2 low ~800 Hz
                }
            }
            // /e/ (EY, EH, AE): mid F1≈550 Hz (bin 1), high F2≈1800 Hz (bin 3)
            "E" | "EY" | "EH" | "AE" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 0.8, 0.6, 2.0)   // F1 mid ~550 Hz
                        + gaussian(b, 3.0, 0.7, 2.5); // F2 ~1800 Hz
                }
            }
            // /o/ (OW, AO, OY): mid F1≈500 Hz (bin 1), low F2≈900 Hz (bin 1-2)
            "O" | "OW" | "AO" | "OY" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 0.9, 0.6, 2.5)   // F1 ~500 Hz
                        + gaussian(b, 1.5, 0.5, 2.0); // F2 low ~900 Hz
                }
            }
            // AY, ER, IX: treat as mid vowels
            "AY" | "ER" | "IX" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 1.0, 0.7, 2.0) + gaussian(b, 2.5, 0.7, 1.5);
                }
            }

            // ── Fricatives ───────────────────────────────────────────────────────────
            // /s/, /z/: very high-frequency noise (bins 7–12)
            "S" | "Z" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 8.0, 1.5, 3.5);
                }
            }
            // /sh/, /zh/ (SH, ZH): somewhat lower high-frequency noise (bins 5–10)
            "SH" | "ZH" | "CH" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 6.5, 1.5, 3.0);
                }
            }
            // /f/, /v/, /th/ (F, V, TH, DH): diffuse high-frequency noise
            "F" | "V" | "TH" | "DH" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 7.0, 2.0, 2.5);
                }
            }
            // /h/ (HH): aspirate — broad high-frequency energy
            "HH" | "H" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 5.0, 2.5, 2.0);
                }
            }

            // ── Stops ────────────────────────────────────────────────────────────────
            // Voiced stops (/b/, /d/, /g/): low-frequency murmur + broad burst
            "B" | "D" | "G" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    // Preceding silence region (bins 0-1 suppressed) + burst (bins 5+)
                    template[i] = gaussian(b, 0.3, 0.3, 0.5)   // low murmur bar
                        + gaussian(b, 5.5, 1.5, 2.5); // burst
                }
            }
            // Voiceless stops (/p/, /t/, /k/): sharper high-frequency burst
            "P" | "T" | "K" | "KK" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 6.0, 1.5, 3.0); // burst only
                }
            }

            // ── Nasals ───────────────────────────────────────────────────────────────
            // /m/, /n/, /ng/ (M, N, NG): low-frequency resonance + anti-formant at ~1kHz (bin 2)
            "M" | "N" | "NG" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    // Strong low-frequency resonance
                    let resonance = gaussian(b, 0.5, 0.5, 3.0);
                    // Anti-formant notch near 1kHz (bin 2): subtract Gaussian dip
                    let notch = gaussian(b, 2.0, 0.4, 1.5);
                    template[i] = (resonance - notch).max(0.0);
                }
            }

            // ── Approximants / semivowels ─────────────────────────────────────────
            "W" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 0.5, 0.7, 2.5) + gaussian(b, 1.2, 0.5, 1.5);
                }
            }
            "L" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 1.5, 0.8, 2.0) + gaussian(b, 3.0, 0.6, 1.0);
                }
            }
            "R" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 1.0, 0.7, 2.0) + gaussian(b, 2.5, 0.7, 1.5);
                }
            }
            "Y" => {
                for i in 0..num_coeffs {
                    let b = i as f32;
                    template[i] = gaussian(b, 0.5, 0.6, 1.5) + gaussian(b, 4.0, 0.7, 2.0);
                }
            }

            // ── Silence / pause ──────────────────────────────────────────────────────
            "SIL" | "SP" | "SPN" | "<SIL>" | "<SP>" | "PAU" | "" => {
                // Near-zero template; tiny epsilon avoids divide-by-zero in normalisation
                for val in template.iter_mut() {
                    *val = 1e-6;
                }
                // Return early; normalisation would collapse near-zero to 1/sqrt(n)
                return template;
            }

            // ── Default (unknown phoneme) ─────────────────────────────────────────
            _ => {
                // Flat white-noise-like profile: equal energy across all bins
                for val in template.iter_mut() {
                    *val = 1.0;
                }
            }
        }

        // L2-normalise so that cosine comparisons are amplitude-independent
        let l2_norm: f32 = template.iter().map(|x| x * x).sum::<f32>().sqrt();
        if l2_norm > 1e-8 {
            for val in template.iter_mut() {
                *val /= l2_norm;
            }
        }

        template
    }

    /// Real posterior probability that `phoneme_index` explains the frames the DTW
    /// path assigned to it.
    ///
    /// The aligner already scores every (frame, phoneme) pair with
    /// [`Self::compute_local_cost`]. Treating those costs as negative log-likelihoods,
    /// a softmax over the *whole* phoneme sequence turns them into a distribution over
    /// which phoneme each frame really looks like:
    ///
    /// ```text
    /// p(j | frame) = exp(-(cost_j - min_k cost_k) / T) / Σ_i exp(-(cost_i - min_k cost_k) / T)
    /// ```
    ///
    /// The temperature `T` is the mean deviation of that frame's costs from their
    /// minimum, which makes the score invariant to the absolute scale of the features —
    /// the previous implementation divided by a global cost and saturated at its clamp
    /// floor for every phoneme of every utterance.
    ///
    /// The returned confidence is the mean posterior across the phoneme's frames, so it
    /// is high only when the phoneme fits its frames distinctly better than the
    /// alternatives, and approaches `1 / n` when it fits no better than the rest.
    fn phoneme_posterior(
        &self,
        features: &[Vec<f32>],
        templates: &[Vec<f32>],
        frames: &[usize],
        phoneme_index: usize,
    ) -> f32 {
        const EPSILON: f32 = 1e-8;

        if frames.is_empty() || templates.is_empty() || phoneme_index >= templates.len() {
            return 0.0;
        }
        if templates.len() == 1 {
            // With a single candidate there is nothing to discriminate against.
            return 1.0;
        }

        let mut total = 0.0_f32;
        let mut counted = 0_u32;

        for &frame_index in frames {
            let Some(feature) = features.get(frame_index) else {
                continue;
            };

            let costs: Vec<f32> = templates
                .iter()
                .map(|template| Self::local_cost(feature, template))
                .collect();

            let min_cost = costs.iter().copied().fold(f32::INFINITY, f32::min);
            if !min_cost.is_finite() {
                continue;
            }

            #[allow(clippy::cast_precision_loss)]
            let count = costs.len() as f32;
            let temperature = costs.iter().map(|c| c - min_cost).sum::<f32>() / count;
            if temperature <= EPSILON {
                // Every phoneme fits this frame equally well: the frame carries no
                // discriminating evidence, so the posterior is uniform.
                total += 1.0 / count;
                counted += 1;
                continue;
            }

            let weights: Vec<f32> = costs
                .iter()
                .map(|cost| (-(cost - min_cost) / temperature).exp())
                .collect();
            let normaliser: f32 = weights.iter().sum();
            if normaliser <= EPSILON {
                continue;
            }

            total += weights[phoneme_index] / normaliser;
            counted += 1;
        }

        if counted == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = total / counted as f32;
        mean.clamp(0.0, 1.0)
    }

    /// Convert text to phonemes using dictionary
    async fn text_to_phonemes(
        &self,
        text: &str,
        _language: LanguageCode,
    ) -> Result<Vec<Phoneme>, RecognitionError> {
        self.ensure_loaded().await?;

        let state = self.state.read().await;
        let uppercase_text = text.to_uppercase();
        let words: Vec<&str> = uppercase_text.split_whitespace().collect();
        let mut phonemes = Vec::new();

        for word in words {
            if let Some(word_phonemes) = state.dictionary.get(word) {
                for phoneme_str in word_phonemes {
                    phonemes.push(Phoneme {
                        symbol: phoneme_str.clone(),
                        ipa_symbol: phoneme_str.clone(),
                        stress: 0,
                        syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
                        duration_ms: None,
                        confidence: 1.0,
                    });
                }
            } else {
                // Handle unknown words with a simple fallback
                tracing::warn!("Unknown word in dictionary: {}", word);
                for char in word.chars() {
                    phonemes.push(Phoneme {
                        symbol: char.to_string(),
                        ipa_symbol: char.to_string(),
                        stress: 0,
                        syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
                        duration_ms: None,
                        confidence: 0.5,
                    });
                }
            }
        }

        Ok(phonemes)
    }

    /// Get model statistics
    pub async fn get_stats(&self) -> ForcedAlignStats {
        let state = self.state.read().await;
        ForcedAlignStats {
            alignment_count: state.alignment_count,
            total_alignment_time: state.total_alignment_time,
            average_alignment_time: if state.alignment_count > 0 {
                state.total_alignment_time / state.alignment_count as u32
            } else {
                Duration::ZERO
            },
            load_time: state.load_time,
            dictionary_size: state.dictionary.len(),
        }
    }

    /// Add word to dictionary
    pub async fn add_word_to_dictionary(
        &self,
        word: String,
        phonemes: Vec<String>,
    ) -> Result<(), RecognitionError> {
        let mut state = self.state.write().await;
        state.dictionary.insert(word.to_uppercase(), phonemes);
        Ok(())
    }

    /// Get pronunciation for a word
    pub async fn get_pronunciation(
        &self,
        word: &str,
    ) -> Result<Option<Vec<String>>, RecognitionError> {
        self.ensure_loaded().await?;
        let state = self.state.read().await;
        Ok(state.dictionary.get(&word.to_uppercase()).cloned())
    }
}

/// Forced alignment model statistics
#[derive(Debug, Clone)]
pub struct ForcedAlignStats {
    /// Total number of alignments performed
    pub alignment_count: usize,
    /// Total alignment time
    pub total_alignment_time: Duration,
    /// Average alignment time
    pub average_alignment_time: Duration,
    /// Model load time
    pub load_time: Option<Duration>,
    /// Dictionary size
    pub dictionary_size: usize,
}

#[async_trait]
impl PhonemeRecognizer for ForcedAlignModel {
    /// Reference-free phoneme recognition is not supported by this model.
    ///
    /// [`ForcedAlignModel`] is a *forced* aligner: it decides **where** a phoneme
    /// sequence you already know occurs, using real MFCC features and DTW. Deciding
    /// **which** phonemes were spoken, with no reference, requires a trained acoustic
    /// classifier, and this crate ships none — the spectral templates used for the
    /// alignment cost are hand-derived formant profiles, far too coarse to label
    /// phonemes on their own.
    ///
    /// Rather than emit labels with no evidence behind them, this method fails closed.
    /// Use [`PhonemeRecognizer::align_phonemes`] or [`PhonemeRecognizer::align_text`]
    /// with a reference, or transcribe with an ASR model first and align its output.
    async fn recognize_phonemes(
        &self,
        _audio: &AudioBuffer,
        _config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<Vec<Phoneme>> {
        Err(RecognitionError::FeatureNotSupported {
            feature: "reference-free phoneme recognition: ForcedAlignModel aligns a known \
                      phoneme sequence to audio and has no trained phoneme classifier. Call \
                      align_phonemes() or align_text() with a reference, or transcribe with an \
                      ASR model first."
                .to_string(),
        }
        .into())
    }

    async fn align_phonemes(
        &self,
        audio: &AudioBuffer,
        expected: &[Phoneme],
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment> {
        self.align_with_dtw(audio, expected, config)
            .await
            .map_err(|e| e.into())
    }

    async fn align_text(
        &self,
        audio: &AudioBuffer,
        text: &str,
        config: Option<&PhonemeRecognitionConfig>,
    ) -> RecognitionResult<PhonemeAlignment> {
        let language = config.map(|c| c.language).unwrap_or(LanguageCode::EnUs);
        let phonemes = self.text_to_phonemes(text, language).await.map_err(|e| {
            RecognitionError::PhonemeRecognitionError {
                message: format!("Text to phoneme conversion failed: {}", e),
                source: Some(Box::new(e)),
            }
        })?;

        self.align_phonemes(audio, &phonemes, config).await
    }

    fn metadata(&self) -> PhonemeRecognizerMetadata {
        self.metadata.clone()
    }

    fn supports_feature(&self, feature: PhonemeRecognitionFeature) -> bool {
        self.metadata.supported_features.contains(&feature)
    }
}

impl Clone for ForcedAlignModel {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            state: self.state.clone(),
            supported_languages: self.supported_languages.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use voirs_sdk::AudioBuffer;

    fn create_mock_model_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "mock acoustic model data").unwrap();
        file
    }

    #[tokio::test]
    async fn test_forced_align_model_creation() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        assert_eq!(model.metadata.name, "Basic Forced Alignment");
        assert!(model.supported_languages.contains(&LanguageCode::EnUs));
    }

    #[tokio::test]
    async fn test_forced_align_missing_file() {
        let result = ForcedAlignModel::new("nonexistent.bin".to_string(), None).await;
        assert!(result.is_err());
    }

    /// Regression test for the removed round-robin "recognition": the model used to
    /// return `AH, L, OW, W, ER` cycling on position, with a hardcoded confidence, for
    /// any audio at all. It must now refuse instead of labelling without evidence.
    #[tokio::test]
    async fn test_phoneme_recognition_fails_closed() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1); // 0.1 second of audio

        let err = model
            .recognize_phonemes(&audio, None)
            .await
            .expect_err("reference-free recognition must not invent phoneme labels");
        let rendered = err.to_string();
        assert!(
            rendered.contains("reference-free phoneme recognition"),
            "unexpected error: {rendered}"
        );
        assert!(
            rendered.contains("align_text"),
            "the error should point at the supported path: {rendered}"
        );
    }

    #[tokio::test]
    async fn test_phoneme_alignment() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1);

        let phonemes = vec![
            Phoneme {
                symbol: "H".to_string(),
                ipa_symbol: "H".to_string(),
                stress: 0,
                syllable_position: voirs_sdk::types::SyllablePosition::Onset,
                duration_ms: None,
                confidence: 1.0,
            },
            Phoneme {
                symbol: "AH".to_string(),
                ipa_symbol: "AH".to_string(),
                stress: 0,
                syllable_position: voirs_sdk::types::SyllablePosition::Nucleus,
                duration_ms: None,
                confidence: 1.0,
            },
        ];

        let result = model.align_phonemes(&audio, &phonemes, None).await.unwrap();
        assert_eq!(result.phonemes.len(), 2);
        assert!(result.alignment_confidence > 0.0);
        assert!(result.total_duration > 0.0);
    }

    #[tokio::test]
    async fn test_text_alignment() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1);

        let result = model.align_text(&audio, "HELLO", None).await.unwrap();
        assert!(!result.phonemes.is_empty());
        assert!(result.alignment_confidence > 0.0);
    }

    #[tokio::test]
    async fn test_dictionary_operations() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();

        // Test getting existing pronunciation
        let pronunciation = model.get_pronunciation("HELLO").await.unwrap();
        assert!(pronunciation.is_some());

        // Test adding new word
        let new_phonemes = vec![
            "T".to_string(),
            "EH".to_string(),
            "S".to_string(),
            "T".to_string(),
        ];
        model
            .add_word_to_dictionary("TESTING".to_string(), new_phonemes.clone())
            .await
            .unwrap();

        let retrieved = model.get_pronunciation("TESTING").await.unwrap();
        assert_eq!(retrieved, Some(new_phonemes));
    }

    #[tokio::test]
    async fn test_features() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();

        assert!(model.supports_feature(PhonemeRecognitionFeature::WordAlignment));
        assert!(model.supports_feature(PhonemeRecognitionFeature::CustomPronunciation));
        assert!(model.supports_feature(PhonemeRecognitionFeature::ConfidenceScoring));
        assert!(model.supports_feature(PhonemeRecognitionFeature::PronunciationAssessment));
    }

    #[tokio::test]
    async fn test_statistics() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();

        let model = ForcedAlignModel::new(model_path, None).await.unwrap();
        let audio = AudioBuffer::new(vec![0.1; 1600], 16000, 1);

        // Initial stats
        let stats = model.get_stats().await;
        assert_eq!(stats.alignment_count, 0);

        // After alignment
        let phonemes = vec![Phoneme {
            symbol: "H".to_string(),
            ipa_symbol: "H".to_string(),
            stress: 0,
            syllable_position: voirs_sdk::types::SyllablePosition::Onset,
            duration_ms: None,
            confidence: 1.0,
        }];

        let _result = model.align_phonemes(&audio, &phonemes, None).await.unwrap();
        let stats = model.get_stats().await;
        assert_eq!(stats.alignment_count, 1);
        assert!(stats.total_alignment_time > Duration::ZERO);
    }

    // ── Template energy distribution tests ──────────────────────────────────────

    /// Helper: create a minimal Phoneme with just a symbol
    fn make_phoneme(symbol: &str) -> Phoneme {
        Phoneme {
            symbol: symbol.to_string(),
            ipa_symbol: symbol.to_string(),
            stress: 0,
            syllable_position: voirs_sdk::types::SyllablePosition::Unknown,
            duration_ms: None,
            confidence: 1.0,
        }
    }

    /// Helper: split a template into low-frequency half and high-frequency half,
    /// and return (low_energy, high_energy).
    fn split_energy(template: &[f32]) -> (f32, f32) {
        let mid = template.len() / 2;
        let low: f32 = template[..mid].iter().map(|x| x * x).sum();
        let high: f32 = template[mid..].iter().map(|x| x * x).sum();
        (low, high)
    }

    #[test]
    fn test_vowel_template_energy_distribution() {
        // Build a model using a dummy config (we only need create_phoneme_template,
        // which is a pure function depending only on self.config)
        use std::sync::Arc;
        use tokio::sync::RwLock;
        let model = ForcedAlignModel {
            config: ForcedAlignConfig::default(),
            state: Arc::new(RwLock::new(ForcedAlignState::new(String::new(), None))),
            supported_languages: vec![],
            metadata: PhonemeRecognizerMetadata {
                name: String::new(),
                version: String::new(),
                description: String::new(),
                supported_languages: vec![],
                alignment_methods: vec![],
                alignment_accuracy: 0.0,
                supported_features: vec![],
            },
        };

        let a_template = model.create_phoneme_template(&make_phoneme("AA"));
        let s_template = model.create_phoneme_template(&make_phoneme("S"));

        let (a_low, _a_high) = split_energy(&a_template);
        let (s_low, _s_high) = split_energy(&s_template);

        // Vowel /a/ must have more low-frequency energy than fricative /s/
        assert!(
            a_low > s_low,
            "/a/ low-freq energy ({a_low:.4}) must exceed /s/ low-freq energy ({s_low:.4})"
        );
    }

    #[test]
    fn test_fricative_template_energy_distribution() {
        use std::sync::Arc;
        use tokio::sync::RwLock;
        let model = ForcedAlignModel {
            config: ForcedAlignConfig::default(),
            state: Arc::new(RwLock::new(ForcedAlignState::new(String::new(), None))),
            supported_languages: vec![],
            metadata: PhonemeRecognizerMetadata {
                name: String::new(),
                version: String::new(),
                description: String::new(),
                supported_languages: vec![],
                alignment_methods: vec![],
                alignment_accuracy: 0.0,
                supported_features: vec![],
            },
        };

        let s_template = model.create_phoneme_template(&make_phoneme("S"));
        let a_template = model.create_phoneme_template(&make_phoneme("AA"));

        let (_s_low, s_high) = split_energy(&s_template);
        let (_a_low, a_high) = split_energy(&a_template);

        // Fricative /s/ must have more high-frequency energy than vowel /a/
        assert!(
            s_high > a_high,
            "/s/ high-freq energy ({s_high:.4}) must exceed /a/ high-freq energy ({a_high:.4})"
        );
    }

    #[test]
    fn test_silence_template_near_zero() {
        use std::sync::Arc;
        use tokio::sync::RwLock;
        let model = ForcedAlignModel {
            config: ForcedAlignConfig::default(),
            state: Arc::new(RwLock::new(ForcedAlignState::new(String::new(), None))),
            supported_languages: vec![],
            metadata: PhonemeRecognizerMetadata {
                name: String::new(),
                version: String::new(),
                description: String::new(),
                supported_languages: vec![],
                alignment_methods: vec![],
                alignment_accuracy: 0.0,
                supported_features: vec![],
            },
        };

        for sil_sym in &["SIL", "SP", "SPN", "<SIL>", "PAU"] {
            let template = model.create_phoneme_template(&make_phoneme(sil_sym));
            let l2_norm: f32 = template.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                l2_norm < 1e-3,
                "SIL-like symbol '{sil_sym}' template L2 norm ({l2_norm:.2e}) must be near zero"
            );
        }
    }

    /// Build audio whose first half is a low-frequency tone (vowel-like: energy in the
    /// low mel bands) and whose second half is a high-frequency buzz (fricative-like).
    fn vowel_then_fricative(sample_rate: u32) -> AudioBuffer {
        let half = sample_rate as usize / 2;
        let mut samples = Vec::with_capacity(half * 2);
        for i in 0..half {
            let t = i as f32 / sample_rate as f32;
            // 200 Hz: energy concentrated in the lowest mel bands.
            samples.push((t * 200.0 * std::f32::consts::TAU).sin() * 0.8);
        }
        for i in 0..half {
            let t = i as f32 / sample_rate as f32;
            // 6 kHz: energy concentrated in the highest mel bands.
            samples.push((t * 6000.0 * std::f32::consts::TAU).sin() * 0.8);
        }
        AudioBuffer::new(samples, sample_rate, 1)
    }

    /// Regression test for the confidence that used to be a saturated constant.
    ///
    /// Confidence is now a real posterior over the phoneme sequence, computed from the
    /// same MFCC features and templates the DTW itself uses, so a phoneme really does
    /// score higher on the frames that look like it.
    #[tokio::test]
    async fn posterior_confidence_is_bounded_and_discriminating() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();
        let model = ForcedAlignModel::new(model_path, None).await.unwrap();

        // Real MFCC frames, extracted by the model's own pipeline from real audio.
        let audio = vowel_then_fricative(16000);
        let features = model
            .extract_features(&audio)
            .await
            .expect("MFCC extraction from real audio");
        assert!(
            features.len() >= 4,
            "need several frames: {}",
            features.len()
        );

        // Frames from the first (low-frequency) and last (high-frequency) quarter.
        let vowel_frame = features.len() / 4;
        let fricative_frame = features.len() * 3 / 4;

        let phonemes = vec![make_phoneme("AA"), make_phoneme("S")];
        let templates = model
            .phoneme_templates_mfcc(&phonemes)
            .expect("templates project into MFCC space");

        let vowel_on_vowel = model.phoneme_posterior(&features, &templates, &[vowel_frame], 0);
        let fricative_on_vowel = model.phoneme_posterior(&features, &templates, &[vowel_frame], 1);
        let fricative_on_fricative =
            model.phoneme_posterior(&features, &templates, &[fricative_frame], 1);
        let vowel_on_fricative =
            model.phoneme_posterior(&features, &templates, &[fricative_frame], 0);

        // Posteriors over one frame form a distribution.
        assert!(
            (vowel_on_vowel + fricative_on_vowel - 1.0).abs() < 1e-4,
            "{vowel_on_vowel} + {fricative_on_vowel} must be 1"
        );
        // The vowel template must explain the low-frequency frame better than /s/ does,
        // and /s/ must explain the high-frequency frame better than the vowel does.
        assert!(
            vowel_on_vowel > fricative_on_vowel,
            "vowel template on a 200 Hz frame: {vowel_on_vowel} vs {fricative_on_vowel}"
        );
        assert!(
            fricative_on_fricative > vowel_on_fricative,
            "/s/ template on a 6 kHz frame: {fricative_on_fricative} vs {vowel_on_fricative}"
        );

        // Degenerate inputs are reported as zero confidence, never as a fixed floor.
        assert_eq!(model.phoneme_posterior(&features, &templates, &[], 0), 0.0);
        assert_eq!(model.phoneme_posterior(&features, &[], &[0], 0), 0.0);
        assert_eq!(model.phoneme_posterior(&features, &templates, &[0], 9), 0.0);
        // Out-of-range frame indices are skipped, leaving nothing measured.
        assert_eq!(
            model.phoneme_posterior(&features, &templates, &[usize::MAX], 0),
            0.0
        );

        // A single candidate cannot be discriminated against anything.
        assert_eq!(
            model.phoneme_posterior(&features, &templates[..1], &[0], 0),
            1.0
        );

        // Every value stays inside [0, 1] across a longer sequence.
        let many: Vec<Phoneme> = ["AA", "S", "M", "T", "IY"]
            .iter()
            .map(|s| make_phoneme(s))
            .collect();
        let many_templates = model
            .phoneme_templates_mfcc(&many)
            .expect("templates project into MFCC space");
        for index in 0..many.len() {
            let confidence =
                model.phoneme_posterior(&features, &many_templates, &[0, 1, 2, 3], index);
            assert!(
                (0.0..=1.0).contains(&confidence),
                "confidence {confidence} out of range for index {index}"
            );
        }

        // A frame carrying no spectral shape at all yields a uniform posterior rather
        // than an invented number.
        let featureless = vec![vec![0.0f32; 13]];
        #[allow(clippy::cast_precision_loss)]
        let expected = 1.0 / many.len() as f32;
        let uniform = model.phoneme_posterior(&featureless, &many_templates, &[0], 0);
        assert!(
            (uniform - expected).abs() < 1e-3,
            "featureless frames must give a uniform posterior: {uniform} vs {expected}"
        );
    }

    /// The aligner's output must follow the audio: aligning the *correct* phoneme order
    /// to structured speech must score better than aligning the reversed order.
    #[tokio::test]
    async fn alignment_confidence_follows_the_audio() {
        let model_file = create_mock_model_file();
        let model_path = model_file.path().to_string_lossy().to_string();
        let model = ForcedAlignModel::new(model_path, None).await.unwrap();

        let audio = vowel_then_fricative(16000);
        let correct = vec![make_phoneme("AA"), make_phoneme("S")];
        let reversed = vec![make_phoneme("S"), make_phoneme("AA")];

        let correct_alignment = model
            .align_phonemes(&audio, &correct, None)
            .await
            .expect("alignment in the correct order");
        let reversed_alignment = model
            .align_phonemes(&audio, &reversed, None)
            .await
            .expect("alignment in the reversed order");

        assert_eq!(correct_alignment.phonemes.len(), 2);
        assert_eq!(reversed_alignment.phonemes.len(), 2);
        assert!(
            correct_alignment.alignment_confidence > reversed_alignment.alignment_confidence,
            "the vowel-then-fricative order must beat the reverse: {} vs {}",
            correct_alignment.alignment_confidence,
            reversed_alignment.alignment_confidence
        );

        // Timings come from the real DTW path, not from an even subdivision.
        let first = &correct_alignment.phonemes[0];
        let second = &correct_alignment.phonemes[1];
        assert!(first.start_time < second.start_time);
        assert!(first.end_time <= second.end_time);
        for phoneme in &correct_alignment.phonemes {
            assert!((0.0..=1.0).contains(&phoneme.confidence));
        }
    }
}
