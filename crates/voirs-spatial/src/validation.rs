//! Perceptual Validation Testing Suite
//!
//! This module provides comprehensive testing for spatial audio perception,
//! including localization accuracy, distance perception, immersion quality,
//! and HRTF validation across different populations and use cases.

use crate::hrtf::HrtfProcessor;
use crate::position::{Listener, SoundSource};
use crate::types::{AudioChannel, Position3D};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Perceptual validation test suite
pub struct PerceptualTestSuite {
    /// Test configurations
    configs: Vec<ValidationTestConfig>,
    /// Results storage
    results: Vec<ValidationTestResult>,
    /// Test subjects data
    subjects: Vec<TestSubject>,
    /// HRTF processor for testing
    hrtf_processor: HrtfProcessor,
}

/// Validation test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationTestConfig {
    /// Test name
    pub name: String,
    /// Test type
    pub test_type: ValidationTestType,
    /// Test parameters
    pub parameters: TestParameters,
    /// Success criteria
    pub success_criteria: SuccessCriteria,
    /// Number of trials
    pub trial_count: u32,
    /// Test duration
    pub duration: Duration,
}

/// Types of validation tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationTestType {
    /// Sound localization accuracy test
    LocalizationAccuracy,
    /// Distance perception test
    DistancePerception,
    /// Elevation perception test
    ElevationPerception,
    /// Front/back discrimination test
    FrontBackDiscrimination,
    /// Immersion quality assessment
    ImmersionQuality,
    /// HRTF validation across populations
    HrtfValidation,
    /// Motion tracking accuracy
    MotionTracking,
    /// Doppler effect accuracy
    DopplerAccuracy,
    /// Room acoustics validation
    RoomAcoustics,
    /// Binaural rendering quality
    BinauralQuality,
}

/// Test parameters for different validation tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestParameters {
    /// Source positions to test
    pub source_positions: Vec<Position3D>,
    /// Listener positions
    pub listener_positions: Vec<Position3D>,
    /// Audio frequencies to test
    pub test_frequencies: Vec<f32>,
    /// Sound levels (dB)
    pub sound_levels: Vec<f32>,
    /// Environment parameters
    pub environment: EnvironmentParameters,
    /// Test-specific parameters
    pub specific_params: HashMap<String, f32>,
}

/// Environment parameters for testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentParameters {
    /// Room size (length, width, height)
    pub room_size: (f32, f32, f32),
    /// Reverberation time (RT60)
    pub reverberation_time: f32,
    /// Background noise level
    pub noise_level: f32,
    /// Temperature (affects air absorption)
    pub temperature: f32,
    /// Humidity (affects air absorption)
    pub humidity: f32,
}

/// Success criteria for tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    /// Minimum accuracy percentage
    pub min_accuracy: f32,
    /// Maximum error tolerance
    pub max_error: f32,
    /// Minimum Mean Opinion Score (MOS)
    pub min_mos: f32,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f32,
}

/// Test subject information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSubject {
    /// Unique subject ID
    pub id: String,
    /// Age
    pub age: u32,
    /// Gender
    pub gender: Gender,
    /// Hearing ability
    pub hearing_ability: HearingAbility,
    /// Head measurements
    pub head_measurements: HeadMeasurements,
    /// Previous VR/AR experience level
    pub experience_level: ExperienceLevel,
    /// Audio expertise level
    pub audio_expertise: AudioExpertise,
}

/// Gender enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Gender {
    /// Male gender
    Male,
    /// Female gender
    Female,
    /// Other gender identity
    Other,
    /// Prefer not to specify gender
    PreferNotToSay,
}

/// Hearing ability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HearingAbility {
    /// Normal hearing ability
    Normal,
    /// Mild hearing loss
    MildLoss,
    /// Moderate hearing loss
    ModerateLoss,
    /// Severe hearing loss
    SevereLoss,
    /// Complete hearing loss
    Deaf,
}

/// Head measurements for HRTF validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadMeasurements {
    /// Head width (cm)
    pub head_width: f32,
    /// Head depth (cm)
    pub head_depth: f32,
    /// Inter-aural distance (cm)
    pub interaural_distance: f32,
    /// Shoulder width (cm)
    pub shoulder_width: f32,
    /// Pinna measurements
    pub pinna: PinnaMeasurements,
}

/// Pinna (ear) measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinnaMeasurements {
    /// Pinna height (cm)
    pub height: f32,
    /// Pinna width (cm)
    pub width: f32,
    /// Concha depth (cm)
    pub concha_depth: f32,
}

/// Experience levels with VR/AR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExperienceLevel {
    /// No prior experience with VR/AR
    Novice,
    /// Limited experience with VR/AR
    Beginner,
    /// Some experience with VR/AR
    Intermediate,
    /// Substantial experience with VR/AR
    Advanced,
    /// Extensive professional experience with VR/AR
    Expert,
}

/// Audio expertise levels
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AudioExpertise {
    /// General consumer with basic audio knowledge
    General,
    /// Enthusiast with high-end audio equipment and trained ear
    Audiophile,
    /// Professional music producer with studio experience
    MusicProducer,
    /// Professional audio engineer with technical expertise
    AudioEngineer,
    /// Academic or industry researcher in audio technology
    Researcher,
}

/// Validation test result  
#[derive(Debug, Clone)]
pub struct ValidationTestResult {
    /// Test configuration used
    pub test_config: ValidationTestConfig,
    /// Subject who performed the test
    pub subject: TestSubject,
    /// Test outcomes
    pub outcomes: Vec<TestOutcome>,
    /// Overall statistics
    pub statistics: TestStatistics,
    /// Subjective ratings
    pub subjective_ratings: SubjectiveRatings,
    /// Test timestamp
    pub timestamp: Instant,
}

/// Individual test outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestOutcome {
    /// Trial number
    pub trial_number: u32,
    /// Presented stimulus
    pub stimulus: StimulusData,
    /// Subject response
    pub response: ResponseData,
    /// Accuracy metrics
    pub accuracy: AccuracyMetrics,
    /// Response time
    pub response_time: Duration,
}

/// Stimulus data for a test trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StimulusData {
    /// Sound source position
    pub source_position: Position3D,
    /// Listener position
    pub listener_position: Position3D,
    /// Audio frequency (Hz)
    pub frequency: f32,
    /// Sound level (dB)
    pub level: f32,
    /// Duration (seconds)
    pub duration: f32,
    /// Additional stimulus properties
    pub properties: HashMap<String, f32>,
}

/// Subject response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseData {
    /// Perceived source position
    pub perceived_position: Position3D,
    /// Confidence rating (1-7 scale)
    pub confidence: u32,
    /// Additional response data
    pub additional_data: HashMap<String, f32>,
}

/// Accuracy metrics for a trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Angular error (degrees)
    pub angular_error: f32,
    /// Distance error (meters)
    pub distance_error: f32,
    /// Elevation error (degrees)
    pub elevation_error: f32,
    /// Front/back confusion (binary)
    pub front_back_confusion: bool,
    /// Overall accuracy score (0-1)
    pub overall_accuracy: f32,
}

/// Test statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestStatistics {
    /// Total trials completed
    pub total_trials: u32,
    /// Mean accuracy
    pub mean_accuracy: f32,
    /// Standard deviation of accuracy
    pub accuracy_std_dev: f32,
    /// Mean angular error (degrees)
    pub mean_angular_error: f32,
    /// Mean distance error (meters)
    pub mean_distance_error: f32,
    /// Front/back confusion rate
    pub front_back_confusion_rate: f32,
    /// Response time statistics
    pub response_time_stats: ResponseTimeStats,
}

/// Response time statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeStats {
    /// Mean response time
    pub mean: Duration,
    /// Median response time
    pub median: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// 95th percentile
    pub p95: Duration,
}

/// Subjective quality ratings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectiveRatings {
    /// Overall quality (1-5 MOS scale)
    pub overall_quality: f32,
    /// Localization naturalness (1-5)
    pub localization_naturalness: f32,
    /// Immersion level (1-5)
    pub immersion_level: f32,
    /// Comfort level (1-5)
    pub comfort_level: f32,
    /// Presence feeling (1-5)
    pub presence: f32,
    /// Any reported artifacts or issues
    pub artifacts: Vec<String>,
}

/// Perceptual validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Test suite summary
    pub summary: ValidationSummary,
    /// Results by test type
    pub results_by_type: HashMap<ValidationTestType, Vec<ValidationTestResult>>,
    /// Population analysis
    pub population_analysis: PopulationAnalysis,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Report timestamp
    pub generated_at: Instant,
}

/// Summary of validation test suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total subjects tested
    pub total_subjects: u32,
    /// Total trials completed
    pub total_trials: u32,
    /// Overall pass rate
    pub overall_pass_rate: f32,
    /// Mean accuracy across all tests
    pub mean_accuracy: f32,
    /// Mean MOS score
    pub mean_mos: f32,
    /// Tests that passed success criteria
    pub passing_tests: Vec<String>,
    /// Tests that failed success criteria
    pub failing_tests: Vec<String>,
}

/// Population-based analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationAnalysis {
    /// Results by age group
    pub by_age_group: HashMap<String, PopulationStats>,
    /// Results by gender
    pub by_gender: HashMap<Gender, PopulationStats>,
    /// Results by hearing ability
    pub by_hearing_ability: HashMap<HearingAbility, PopulationStats>,
    /// Results by experience level
    pub by_experience_level: HashMap<ExperienceLevel, PopulationStats>,
}

/// Statistics for a population group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStats {
    /// Number of subjects
    pub subject_count: u32,
    /// Mean accuracy
    pub mean_accuracy: f32,
    /// Standard deviation
    pub accuracy_std_dev: f32,
    /// Mean MOS score
    pub mean_mos: f32,
    /// Pass rate
    pub pass_rate: f32,
}

impl PerceptualTestSuite {
    /// Create new perceptual test suite
    pub fn new(hrtf_processor: HrtfProcessor) -> Self {
        Self {
            configs: Vec::new(),
            results: Vec::new(),
            subjects: Vec::new(),
            hrtf_processor,
        }
    }

    /// Add test configuration
    pub fn add_test_config(&mut self, config: ValidationTestConfig) {
        self.configs.push(config);
    }

    /// Add test subject
    pub fn add_subject(&mut self, subject: TestSubject) {
        self.subjects.push(subject);
    }

    /// Run all validation tests
    pub async fn run_all_tests(&mut self) -> Result<ValidationReport> {
        tracing::info!("Starting perceptual validation test suite");

        for config in self.configs.clone() {
            for subject in self.subjects.clone() {
                let result = self.run_test(&config, &subject).await?;
                self.results.push(result);
            }
        }

        let report = self.generate_report().await?;
        tracing::info!("Completed perceptual validation test suite");
        Ok(report)
    }

    /// Run a specific test for a subject
    pub async fn run_test(
        &self,
        config: &ValidationTestConfig,
        subject: &TestSubject,
    ) -> Result<ValidationTestResult> {
        tracing::info!(
            "Running test '{}' for subject '{}'",
            config.name,
            subject.id
        );

        let mut outcomes = Vec::new();

        for trial_num in 0..config.trial_count {
            let outcome = self.run_trial(config, subject, trial_num).await?;
            outcomes.push(outcome);
        }

        let statistics = self.calculate_statistics(&outcomes);
        let subjective_ratings = self.collect_subjective_ratings(config, subject).await?;

        Ok(ValidationTestResult {
            test_config: config.clone(),
            subject: subject.clone(),
            outcomes,
            statistics,
            subjective_ratings,
            timestamp: Instant::now(),
        })
    }

    /// Run a single trial
    async fn run_trial(
        &self,
        config: &ValidationTestConfig,
        subject: &TestSubject,
        trial_num: u32,
    ) -> Result<TestOutcome> {
        // Select test parameters for this trial
        let stimulus = self.generate_stimulus(config, trial_num)?;

        let start_time = Instant::now();

        // Simulate spatial audio processing
        let processed_audio = self.process_spatial_audio(&stimulus, subject).await?;

        // Simulate subject response (in real implementation, this would be user input)
        let response = self.simulate_subject_response(&stimulus, &processed_audio, subject)?;

        let response_time = start_time.elapsed();

        // Calculate accuracy metrics
        let accuracy = self.calculate_accuracy(&stimulus, &response)?;

        Ok(TestOutcome {
            trial_number: trial_num,
            stimulus,
            response,
            accuracy,
            response_time,
        })
    }

    /// Generate stimulus for a trial
    fn generate_stimulus(
        &self,
        config: &ValidationTestConfig,
        trial_num: u32,
    ) -> Result<StimulusData> {
        let params = &config.parameters;

        // Select position based on trial number and test type
        let source_position = if params.source_positions.is_empty() {
            self.generate_random_position(config.test_type)?
        } else {
            params.source_positions[trial_num as usize % params.source_positions.len()]
        };

        let listener_position = if params.listener_positions.is_empty() {
            Position3D::new(0.0, 1.7, 0.0) // Default head height
        } else {
            params.listener_positions[trial_num as usize % params.listener_positions.len()]
        };

        let frequency = if params.test_frequencies.is_empty() {
            1000.0 // Default 1kHz
        } else {
            params.test_frequencies[trial_num as usize % params.test_frequencies.len()]
        };

        let level = if params.sound_levels.is_empty() {
            70.0 // Default 70dB
        } else {
            params.sound_levels[trial_num as usize % params.sound_levels.len()]
        };

        Ok(StimulusData {
            source_position,
            listener_position,
            frequency,
            level,
            duration: 2.0, // 2 second duration
            properties: HashMap::new(),
        })
    }

    /// Generate random position based on test type
    fn generate_random_position(&self, test_type: ValidationTestType) -> Result<Position3D> {
        use fastrand;

        match test_type {
            ValidationTestType::LocalizationAccuracy => {
                // Random positions on sphere around listener
                let azimuth = fastrand::f32() * 2.0 * std::f32::consts::PI;
                let elevation = (fastrand::f32() - 0.5) * std::f32::consts::PI;
                let distance = 2.0 + fastrand::f32() * 3.0; // 2-5 meters

                Ok(Position3D::new(
                    distance * elevation.cos() * azimuth.cos(),
                    distance * elevation.sin(),
                    distance * elevation.cos() * azimuth.sin(),
                ))
            }
            ValidationTestType::DistancePerception => {
                // Fixed angle, varying distance
                let distance = 0.5 + fastrand::f32() * 19.5; // 0.5-20 meters
                Ok(Position3D::new(distance, 1.7, 0.0))
            }
            ValidationTestType::ElevationPerception => {
                // Fixed horizontal plane, varying elevation
                let elevation = (fastrand::f32() - 0.5) * std::f32::consts::PI * 0.8; // ±72 degrees
                let distance = 3.0;
                Ok(Position3D::new(
                    0.0,
                    distance * elevation.sin(),
                    distance * elevation.cos(),
                ))
            }
            ValidationTestType::FrontBackDiscrimination => {
                // Front or back positions
                let is_front = fastrand::bool();
                let angle_offset = (fastrand::f32() - 0.5) * 0.7; // ±20 degrees
                let base_angle = if is_front { 0.0 } else { std::f32::consts::PI };
                let angle = base_angle + angle_offset;
                let distance = 2.0;

                Ok(Position3D::new(
                    distance * angle.sin(),
                    1.7,
                    distance * angle.cos(),
                ))
            }
            _ => {
                // Default random position
                Ok(Position3D::new(
                    (fastrand::f32() - 0.5) * 10.0,
                    fastrand::f32() * 3.0,
                    (fastrand::f32() - 0.5) * 10.0,
                ))
            }
        }
    }

    /// Process spatial audio for stimulus
    async fn process_spatial_audio(
        &self,
        stimulus: &StimulusData,
        subject: &TestSubject,
    ) -> Result<Vec<f32>> {
        // Create listener and source
        let mut listener = Listener::new();
        listener.set_position(stimulus.listener_position);
        let source = SoundSource::new_point("test_source".to_string(), stimulus.source_position);

        // Generate test signal
        let sample_rate = 44100;
        let duration_samples = (stimulus.duration * sample_rate as f32) as usize;
        let mut audio_signal = vec![0.0f32; duration_samples];

        // Generate pure tone
        for (i, sample) in audio_signal.iter_mut().enumerate().take(duration_samples) {
            let t = i as f32 / sample_rate as f32;
            *sample = (2.0 * std::f32::consts::PI * stimulus.frequency * t).sin() * 0.1;
            // Reduce amplitude
        }

        // Apply HRTF processing (simplified)
        // In a real implementation, this would use the full HRTF processor
        // For now, just return the original signal
        let processed_signal = audio_signal.clone();

        Ok(processed_signal)
    }

    /// Simulate subject response (in real test, this would be user input)
    fn simulate_subject_response(
        &self,
        stimulus: &StimulusData,
        _processed_audio: &[f32],
        subject: &TestSubject,
    ) -> Result<ResponseData> {
        use fastrand;

        // Simulate human localization accuracy based on subject characteristics
        let base_accuracy = match subject.audio_expertise {
            AudioExpertise::General => 0.7,
            AudioExpertise::Audiophile => 0.8,
            AudioExpertise::MusicProducer => 0.85,
            AudioExpertise::AudioEngineer => 0.9,
            AudioExpertise::Researcher => 0.95,
        };

        // Add noise based on hearing ability
        let hearing_factor = match subject.hearing_ability {
            HearingAbility::Normal => 1.0,
            HearingAbility::MildLoss => 0.9,
            HearingAbility::ModerateLoss => 0.7,
            HearingAbility::SevereLoss => 0.5,
            HearingAbility::Deaf => 0.1,
        };

        let accuracy = base_accuracy * hearing_factor;

        // Add random error
        let error_scale = (1.0 - accuracy) * 2.0; // Higher error for lower accuracy
        let position_error = Position3D::new(
            (fastrand::f32() - 0.5) * error_scale,
            (fastrand::f32() - 0.5) * error_scale,
            (fastrand::f32() - 0.5) * error_scale,
        );

        let perceived_position = Position3D::new(
            stimulus.source_position.x + position_error.x,
            stimulus.source_position.y + position_error.y,
            stimulus.source_position.z + position_error.z,
        );

        // Confidence correlates with accuracy
        let confidence = ((accuracy * 5.0) as u32).clamp(1, 7);

        Ok(ResponseData {
            perceived_position,
            confidence,
            additional_data: HashMap::new(),
        })
    }

    /// Calculate accuracy metrics
    fn calculate_accuracy(
        &self,
        stimulus: &StimulusData,
        response: &ResponseData,
    ) -> Result<AccuracyMetrics> {
        let true_pos = &stimulus.source_position;
        let perceived_pos = &response.perceived_position;

        // Calculate angular error
        let true_vec = Position3D::new(true_pos.x, 0.0, true_pos.z).normalized();
        let perceived_vec = Position3D::new(perceived_pos.x, 0.0, perceived_pos.z).normalized();
        let angular_error = true_vec.dot(&perceived_vec).acos() * 180.0 / std::f32::consts::PI;

        // Calculate distance error
        let true_distance = true_pos.magnitude();
        let perceived_distance = perceived_pos.magnitude();
        let distance_error = (true_distance - perceived_distance).abs();

        // Calculate elevation error
        let true_elevation =
            (true_pos.y / true_pos.magnitude()).asin() * 180.0 / std::f32::consts::PI;
        let perceived_elevation =
            (perceived_pos.y / perceived_pos.magnitude()).asin() * 180.0 / std::f32::consts::PI;
        let elevation_error = (true_elevation - perceived_elevation).abs();

        // Front/back confusion check
        let front_back_confusion = (true_pos.z > 0.0) != (perceived_pos.z > 0.0);

        // Overall accuracy score (inverse of normalized error)
        let overall_accuracy =
            1.0 / (1.0 + angular_error / 180.0 + distance_error / 10.0 + elevation_error / 90.0);

        Ok(AccuracyMetrics {
            angular_error,
            distance_error,
            elevation_error,
            front_back_confusion,
            overall_accuracy,
        })
    }

    /// Calculate statistics for test outcomes
    fn calculate_statistics(&self, outcomes: &[TestOutcome]) -> TestStatistics {
        if outcomes.is_empty() {
            return TestStatistics {
                total_trials: 0,
                mean_accuracy: 0.0,
                accuracy_std_dev: 0.0,
                mean_angular_error: 0.0,
                mean_distance_error: 0.0,
                front_back_confusion_rate: 0.0,
                response_time_stats: ResponseTimeStats {
                    mean: Duration::from_secs(0),
                    median: Duration::from_secs(0),
                    std_dev: Duration::from_secs(0),
                    p95: Duration::from_secs(0),
                },
            };
        }

        let total_trials = outcomes.len() as u32;

        // Accuracy statistics
        let accuracies: Vec<f32> = outcomes
            .iter()
            .map(|o| o.accuracy.overall_accuracy)
            .collect();
        let mean_accuracy = accuracies.iter().sum::<f32>() / accuracies.len() as f32;
        let accuracy_variance = accuracies
            .iter()
            .map(|&x| (x - mean_accuracy).powi(2))
            .sum::<f32>()
            / accuracies.len() as f32;
        let accuracy_std_dev = accuracy_variance.sqrt();

        // Error statistics
        let angular_errors: Vec<f32> = outcomes.iter().map(|o| o.accuracy.angular_error).collect();
        let mean_angular_error = angular_errors.iter().sum::<f32>() / angular_errors.len() as f32;

        let distance_errors: Vec<f32> =
            outcomes.iter().map(|o| o.accuracy.distance_error).collect();
        let mean_distance_error =
            distance_errors.iter().sum::<f32>() / distance_errors.len() as f32;

        // Front/back confusion rate
        let confusion_count = outcomes
            .iter()
            .filter(|o| o.accuracy.front_back_confusion)
            .count();
        let front_back_confusion_rate = confusion_count as f32 / total_trials as f32;

        // Response time statistics
        let mut response_times: Vec<Duration> = outcomes.iter().map(|o| o.response_time).collect();
        response_times.sort();

        let mean_response_time = Duration::from_nanos(
            (response_times.iter().map(|d| d.as_nanos()).sum::<u128>()
                / response_times.len() as u128) as u64,
        );
        let median_response_time = response_times[response_times.len() / 2];
        let p95_index = (response_times.len() as f32 * 0.95) as usize;
        let p95_response_time = response_times[p95_index.min(response_times.len() - 1)];

        // Response time standard deviation
        let mean_nanos = mean_response_time.as_nanos() as f64;
        let variance = response_times
            .iter()
            .map(|d| (d.as_nanos() as f64 - mean_nanos).powi(2))
            .sum::<f64>()
            / response_times.len() as f64;
        let std_dev_response_time = Duration::from_nanos(variance.sqrt() as u64);

        TestStatistics {
            total_trials,
            mean_accuracy,
            accuracy_std_dev,
            mean_angular_error,
            mean_distance_error,
            front_back_confusion_rate,
            response_time_stats: ResponseTimeStats {
                mean: mean_response_time,
                median: median_response_time,
                std_dev: std_dev_response_time,
                p95: p95_response_time,
            },
        }
    }

    /// Collect subjective ratings (simulated for testing)
    async fn collect_subjective_ratings(
        &self,
        _config: &ValidationTestConfig,
        subject: &TestSubject,
    ) -> Result<SubjectiveRatings> {
        use fastrand;

        // Simulate ratings based on subject characteristics
        let base_quality = match subject.audio_expertise {
            AudioExpertise::General => 3.0,
            AudioExpertise::Audiophile => 3.5,
            AudioExpertise::MusicProducer => 4.0,
            AudioExpertise::AudioEngineer => 4.2,
            AudioExpertise::Researcher => 4.5,
        };

        let noise = (fastrand::f32() - 0.5) * 0.5; // ±0.25
        let quality = (base_quality + noise).clamp(1.0, 5.0);

        Ok(SubjectiveRatings {
            overall_quality: quality,
            localization_naturalness: quality + (fastrand::f32() - 0.5) * 0.3,
            immersion_level: quality + (fastrand::f32() - 0.5) * 0.4,
            comfort_level: quality + (fastrand::f32() - 0.5) * 0.2,
            presence: quality + (fastrand::f32() - 0.5) * 0.6,
            artifacts: Vec::new(),
        })
    }

    /// Generate validation report
    async fn generate_report(&self) -> Result<ValidationReport> {
        let mut results_by_type: HashMap<ValidationTestType, Vec<ValidationTestResult>> =
            HashMap::new();

        for result in &self.results {
            results_by_type
                .entry(result.test_config.test_type)
                .or_default()
                .push(result.clone());
        }

        let summary = self.generate_summary(&results_by_type);
        let population_analysis = self.generate_population_analysis(&self.results);
        let recommendations = self.generate_recommendations(&summary, &population_analysis);

        Ok(ValidationReport {
            summary,
            results_by_type,
            population_analysis,
            recommendations,
            generated_at: Instant::now(),
        })
    }

    /// Generate summary statistics
    fn generate_summary(
        &self,
        results_by_type: &HashMap<ValidationTestType, Vec<ValidationTestResult>>,
    ) -> ValidationSummary {
        let total_subjects = self.subjects.len() as u32;
        let total_trials: u32 = self.results.iter().map(|r| r.statistics.total_trials).sum();

        let mean_accuracy = if self.results.is_empty() {
            0.0
        } else {
            self.results
                .iter()
                .map(|r| r.statistics.mean_accuracy)
                .sum::<f32>()
                / self.results.len() as f32
        };

        let mean_mos = if self.results.is_empty() {
            0.0
        } else {
            self.results
                .iter()
                .map(|r| r.subjective_ratings.overall_quality)
                .sum::<f32>()
                / self.results.len() as f32
        };

        // Count passing/failing tests based on criteria
        let mut passing_tests = Vec::new();
        let mut failing_tests = Vec::new();

        for (test_type, results) in results_by_type {
            let test_name = format!("{test_type:?}");
            let pass_rate = results
                .iter()
                .filter(|r| {
                    let criteria = &r.test_config.success_criteria;
                    r.statistics.mean_accuracy >= criteria.min_accuracy
                        && r.subjective_ratings.overall_quality >= criteria.min_mos
                })
                .count() as f32
                / results.len() as f32;

            if pass_rate >= 0.8 {
                // 80% pass rate threshold
                passing_tests.push(test_name);
            } else {
                failing_tests.push(test_name);
            }
        }

        let overall_pass_rate =
            passing_tests.len() as f32 / (passing_tests.len() + failing_tests.len()) as f32;

        ValidationSummary {
            total_subjects,
            total_trials,
            overall_pass_rate,
            mean_accuracy,
            mean_mos,
            passing_tests,
            failing_tests,
        }
    }

    /// Generate population analysis
    fn generate_population_analysis(&self, results: &[ValidationTestResult]) -> PopulationAnalysis {
        let mut by_age_group = HashMap::new();
        let mut by_gender = HashMap::new();
        let mut by_hearing_ability = HashMap::new();
        let mut by_experience_level = HashMap::new();

        // Group results by demographics
        for result in results {
            let subject = &result.subject;

            // Age groups
            let age_group = if subject.age < 25 {
                "18-24"
            } else if subject.age < 35 {
                "25-34"
            } else if subject.age < 45 {
                "35-44"
            } else if subject.age < 55 {
                "45-54"
            } else {
                "55+"
            }
            .to_string();

            self.update_population_stats(&mut by_age_group, age_group, result);
            self.update_population_stats(&mut by_gender, subject.gender, result);
            self.update_population_stats(&mut by_hearing_ability, subject.hearing_ability, result);
            self.update_population_stats(
                &mut by_experience_level,
                subject.experience_level,
                result,
            );
        }

        PopulationAnalysis {
            by_age_group,
            by_gender,
            by_hearing_ability,
            by_experience_level,
        }
    }

    /// Update population statistics
    fn update_population_stats<K: Clone + std::hash::Hash + Eq>(
        &self,
        map: &mut HashMap<K, PopulationStats>,
        key: K,
        result: &ValidationTestResult,
    ) {
        let stats = map.entry(key).or_insert_with(|| PopulationStats {
            subject_count: 0,
            mean_accuracy: 0.0,
            accuracy_std_dev: 0.0,
            mean_mos: 0.0,
            pass_rate: 0.0,
        });

        stats.subject_count += 1;

        // Update running averages (simplified)
        let n = stats.subject_count as f32;
        stats.mean_accuracy =
            (stats.mean_accuracy * (n - 1.0) + result.statistics.mean_accuracy) / n;
        stats.mean_mos =
            (stats.mean_mos * (n - 1.0) + result.subjective_ratings.overall_quality) / n;

        // Calculate pass rate
        let passed =
            result.statistics.mean_accuracy >= result.test_config.success_criteria.min_accuracy;
        stats.pass_rate = (stats.pass_rate * (n - 1.0) + if passed { 1.0 } else { 0.0 }) / n;
    }

    /// Generate recommendations based on results
    fn generate_recommendations(
        &self,
        summary: &ValidationSummary,
        population_analysis: &PopulationAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if summary.overall_pass_rate < 0.8 {
            recommendations.push(
                "Overall pass rate is below 80%. Consider improving core spatial audio algorithms."
                    .to_string(),
            );
        }

        if summary.mean_accuracy < 0.85 {
            recommendations.push(
                "Mean accuracy is below 85%. Focus on improving localization algorithms."
                    .to_string(),
            );
        }

        if summary.mean_mos < 4.0 {
            recommendations.push(
                "Mean Opinion Score is below 4.0. Improve perceptual quality of spatial rendering."
                    .to_string(),
            );
        }

        // Check for demographic-specific issues
        for (hearing_ability, stats) in &population_analysis.by_hearing_ability {
            if matches!(
                hearing_ability,
                HearingAbility::MildLoss | HearingAbility::ModerateLoss
            ) && stats.mean_accuracy < 0.7
            {
                recommendations.push(format!(
                    "Users with {:?} show lower accuracy ({}%). Consider accessibility improvements.",
                    hearing_ability, (stats.mean_accuracy * 100.0) as u32
                ));
            }
        }

        recommendations
    }
}

/// Create standard validation test configurations
pub fn create_standard_test_configs() -> Vec<ValidationTestConfig> {
    vec![
        // Localization accuracy test
        ValidationTestConfig {
            name: "Localization Accuracy Test".to_string(),
            test_type: ValidationTestType::LocalizationAccuracy,
            parameters: TestParameters {
                source_positions: vec![], // Will be generated randomly
                listener_positions: vec![Position3D::new(0.0, 1.7, 0.0)],
                test_frequencies: vec![250.0, 500.0, 1000.0, 2000.0, 4000.0],
                sound_levels: vec![60.0, 70.0, 80.0],
                environment: EnvironmentParameters {
                    room_size: (10.0, 10.0, 3.0),
                    reverberation_time: 0.3,
                    noise_level: 40.0,
                    temperature: 20.0,
                    humidity: 50.0,
                },
                specific_params: HashMap::new(),
            },
            success_criteria: SuccessCriteria {
                min_accuracy: 0.85,
                max_error: 15.0, // degrees
                min_mos: 4.0,
                max_latency_ms: 20.0,
            },
            trial_count: 50,
            duration: Duration::from_secs(2),
        },
        // Distance perception test
        ValidationTestConfig {
            name: "Distance Perception Test".to_string(),
            test_type: ValidationTestType::DistancePerception,
            parameters: TestParameters {
                source_positions: (0..20)
                    .map(|i| {
                        let distance = 0.5 + (i as f32) * 0.975; // 0.5m to 20m
                        Position3D::new(distance, 1.7, 0.0)
                    })
                    .collect(),
                listener_positions: vec![Position3D::new(0.0, 1.7, 0.0)],
                test_frequencies: vec![1000.0],
                sound_levels: vec![70.0],
                environment: EnvironmentParameters {
                    room_size: (30.0, 30.0, 5.0),
                    reverberation_time: 0.5,
                    noise_level: 35.0,
                    temperature: 20.0,
                    humidity: 50.0,
                },
                specific_params: HashMap::new(),
            },
            success_criteria: SuccessCriteria {
                min_accuracy: 0.80,
                max_error: 1.0, // meters
                min_mos: 3.8,
                max_latency_ms: 20.0,
            },
            trial_count: 20,
            duration: Duration::from_secs(3),
        },
        // Front/back discrimination test
        ValidationTestConfig {
            name: "Front/Back Discrimination Test".to_string(),
            test_type: ValidationTestType::FrontBackDiscrimination,
            parameters: TestParameters {
                source_positions: vec![], // Will be generated
                listener_positions: vec![Position3D::new(0.0, 1.7, 0.0)],
                test_frequencies: vec![1000.0, 2000.0, 4000.0],
                sound_levels: vec![70.0],
                environment: EnvironmentParameters {
                    room_size: (8.0, 8.0, 3.0),
                    reverberation_time: 0.2,
                    noise_level: 30.0,
                    temperature: 20.0,
                    humidity: 50.0,
                },
                specific_params: HashMap::new(),
            },
            success_criteria: SuccessCriteria {
                min_accuracy: 0.95, // High requirement for front/back
                max_error: 5.0,     // degrees
                min_mos: 4.2,
                max_latency_ms: 20.0,
            },
            trial_count: 40,
            duration: Duration::from_secs(2),
        },
    ]
}

/// Create diverse test subject pool
pub fn create_test_subjects() -> Vec<TestSubject> {
    vec![
        TestSubject {
            id: "subject_001".to_string(),
            age: 25,
            gender: Gender::Male,
            hearing_ability: HearingAbility::Normal,
            head_measurements: HeadMeasurements {
                head_width: 15.5,
                head_depth: 19.0,
                interaural_distance: 17.5,
                shoulder_width: 45.0,
                pinna: PinnaMeasurements {
                    height: 6.2,
                    width: 3.5,
                    concha_depth: 1.2,
                },
            },
            experience_level: ExperienceLevel::Intermediate,
            audio_expertise: AudioExpertise::General,
        },
        TestSubject {
            id: "subject_002".to_string(),
            age: 32,
            gender: Gender::Female,
            hearing_ability: HearingAbility::Normal,
            head_measurements: HeadMeasurements {
                head_width: 14.2,
                head_depth: 17.8,
                interaural_distance: 16.5,
                shoulder_width: 38.0,
                pinna: PinnaMeasurements {
                    height: 5.8,
                    width: 3.2,
                    concha_depth: 1.0,
                },
            },
            experience_level: ExperienceLevel::Advanced,
            audio_expertise: AudioExpertise::Audiophile,
        },
        TestSubject {
            id: "subject_003".to_string(),
            age: 45,
            gender: Gender::Male,
            hearing_ability: HearingAbility::MildLoss,
            head_measurements: HeadMeasurements {
                head_width: 16.0,
                head_depth: 19.5,
                interaural_distance: 18.0,
                shoulder_width: 48.0,
                pinna: PinnaMeasurements {
                    height: 6.5,
                    width: 3.8,
                    concha_depth: 1.3,
                },
            },
            experience_level: ExperienceLevel::Novice,
            audio_expertise: AudioExpertise::General,
        },
        TestSubject {
            id: "subject_004".to_string(),
            age: 28,
            gender: Gender::Female,
            hearing_ability: HearingAbility::Normal,
            head_measurements: HeadMeasurements {
                head_width: 14.0,
                head_depth: 17.5,
                interaural_distance: 16.2,
                shoulder_width: 36.0,
                pinna: PinnaMeasurements {
                    height: 5.6,
                    width: 3.0,
                    concha_depth: 0.9,
                },
            },
            experience_level: ExperienceLevel::Expert,
            audio_expertise: AudioExpertise::AudioEngineer,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hrtf::HrtfProcessor;

    #[tokio::test]
    async fn test_perceptual_test_suite() {
        let hrtf_processor = HrtfProcessor::new_default().await.unwrap();
        let mut suite = PerceptualTestSuite::new(hrtf_processor);

        // Add test configurations
        let configs = create_standard_test_configs();
        for config in configs {
            suite.add_test_config(config);
        }

        // Add test subjects
        let subjects = create_test_subjects();
        for subject in subjects {
            suite.add_subject(subject);
        }

        // Run a single test (to avoid long test times)
        if let (Some(config), Some(subject)) = (suite.configs.first(), suite.subjects.first()) {
            let result = suite.run_test(config, subject).await.unwrap();
            assert!(result.outcomes.len() > 0);
            assert!(result.statistics.mean_accuracy >= 0.0);
        }
    }

    #[tokio::test]
    async fn test_stimulus_generation() {
        let hrtf_processor = HrtfProcessor::new_default().await.unwrap();
        let suite = PerceptualTestSuite::new(hrtf_processor);

        let config = &create_standard_test_configs()[0];
        let stimulus = suite.generate_stimulus(config, 0).unwrap();

        assert!(stimulus.frequency > 0.0);
        assert!(stimulus.level > 0.0);
        assert!(stimulus.duration > 0.0);
    }

    #[tokio::test]
    async fn test_accuracy_calculation() {
        let hrtf_processor = HrtfProcessor::new_default().await.unwrap();
        let suite = PerceptualTestSuite::new(hrtf_processor);

        let stimulus = StimulusData {
            source_position: Position3D::new(2.0, 0.0, 0.0),
            listener_position: Position3D::new(0.0, 1.7, 0.0),
            frequency: 1000.0,
            level: 70.0,
            duration: 2.0,
            properties: HashMap::new(),
        };

        let response = ResponseData {
            perceived_position: Position3D::new(2.1, 0.1, 0.0),
            confidence: 5,
            additional_data: HashMap::new(),
        };

        let accuracy = suite.calculate_accuracy(&stimulus, &response).unwrap();
        assert!(accuracy.angular_error < 10.0); // Small error for close positions
        assert!(accuracy.distance_error < 0.5);
        assert!(accuracy.overall_accuracy > 0.8);
    }

    #[test]
    fn test_standard_configs_creation() {
        let configs = create_standard_test_configs();
        assert_eq!(configs.len(), 3);

        let localization_test = configs
            .iter()
            .find(|c| c.test_type == ValidationTestType::LocalizationAccuracy)
            .unwrap();
        assert!(localization_test.trial_count > 0);
        assert!(localization_test.success_criteria.min_accuracy > 0.0);
    }

    #[test]
    fn test_subjects_creation() {
        let subjects = create_test_subjects();
        assert_eq!(subjects.len(), 4);

        // Check diversity
        let genders: std::collections::HashSet<_> = subjects.iter().map(|s| s.gender).collect();
        assert!(genders.len() >= 2); // At least 2 different genders

        let hearing_abilities: std::collections::HashSet<_> =
            subjects.iter().map(|s| s.hearing_ability).collect();
        assert!(hearing_abilities.len() >= 2); // At least 2 different hearing abilities
    }
}
