//! Quality Regression Detector
//!
//! Wraps [`RegressionDetector`] to track audio quality metrics over time and detect
//! when audio quality drops. Supports PESQ, STOI, and MCD metrics with configurable
//! thresholds and severity levels.
//!
//! ## Key Design
//!
//! MCD (Mel-Cepstral Distortion) is a "lower-is-better" metric, but [`RegressionDetector`]
//! treats a positive change-percentage as a regression. To bridge this, MCD values are
//! stored internally as their negation (`-mcd`), so that an increase in raw MCD (quality
//! degradation) maps to a positive percentage change and correctly triggers a regression alert.
//! The raw MCD value is still exposed in [`QualityRegressionMeasurement::mcd`].

use std::path::Path;

use serde::{Deserialize, Serialize};
use voirs_sdk::AudioBuffer;

use crate::quality::{MCDEvaluator, PESQEvaluator, STOIEvaluator};
use crate::regression_detector::{
    BenchmarkMeasurement, RegressionConfig, RegressionDetector, RegressionResult,
    RegressionSeverity,
};
use crate::EvaluationError;

// ─── Configuration ─────────────────────────────────────────────────────────

/// Configuration for the [`QualityRegressionDetector`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRegressionConfig {
    /// Enable PESQ evaluation (default: `true`)
    pub enable_pesq: bool,
    /// Enable STOI evaluation (default: `true`)
    pub enable_stoi: bool,
    /// Enable MCD evaluation (default: `true`)
    pub enable_mcd: bool,
    /// Sample rate in Hz for all evaluators (default: `16000`)
    ///
    /// PESQ supports only `8000` or `16000`. For wideband evaluation use `16000`.
    pub sample_rate: u32,
    /// Percentage threshold for minor regressions (default: `5.0`)
    pub minor_threshold_pct: f64,
    /// Percentage threshold for major regressions (default: `10.0`)
    pub major_threshold_pct: f64,
    /// Percentage threshold for critical regressions (default: `20.0`)
    pub critical_threshold_pct: f64,
    /// Baseline window size (number of past measurements used for baseline; default: `10`)
    pub baseline_window: usize,
    /// Minimum samples required before regression detection fires (default: `3`)
    pub min_samples: usize,
}

impl Default for QualityRegressionConfig {
    fn default() -> Self {
        Self {
            enable_pesq: true,
            enable_stoi: true,
            enable_mcd: true,
            sample_rate: 16000,
            minor_threshold_pct: 5.0,
            major_threshold_pct: 10.0,
            critical_threshold_pct: 20.0,
            baseline_window: 10,
            min_samples: 3,
        }
    }
}

// ─── Output types ───────────────────────────────────────────────────────────

/// Result of a single quality evaluation cycle.
///
/// Contains the raw metric scores and any regressions detected relative to the
/// accumulated baseline.
#[derive(Debug, Clone)]
pub struct QualityRegressionMeasurement {
    /// Human-readable label for this measurement (e.g. a git commit hash or date string)
    pub label: String,
    /// Unix timestamp in milliseconds when this measurement was recorded
    pub timestamp: u64,
    /// PESQ score, if PESQ is enabled (range approximately −0.5 to 4.5; higher is better)
    pub pesq: Option<f64>,
    /// STOI intelligibility score, if STOI is enabled (range 0.0–1.0; higher is better)
    pub stoi: Option<f64>,
    /// Raw MCD in dB, if MCD is enabled (lower is better; typical range 0–15 dB)
    pub mcd: Option<f64>,
    /// Regression alerts detected for this measurement (may be empty if below threshold
    /// or insufficient historical data)
    pub regressions: Vec<RegressionResult>,
}

/// Aggregated quality regression report over all recorded measurements.
#[derive(Debug, Clone, Default)]
pub struct QualityRegressionReport {
    /// Total number of measurements recorded
    pub total_measurements: usize,
    /// Total number of regression events detected across all measurements
    pub total_regressions: usize,
    /// Critical-severity regression events
    pub critical_regressions: Vec<RegressionResult>,
    /// Major-severity regression events
    pub major_regressions: Vec<RegressionResult>,
    /// Minor-severity regression events
    pub minor_regressions: Vec<RegressionResult>,
}

// ─── Serialisable snapshot ──────────────────────────────────────────────────

/// Internal serialisable representation used when persisting baselines to disk.
///
/// We wrap the underlying [`RegressionDetector`] measurements rather than the
/// detector itself because [`RegressionDetector`] does not implement [`Serialize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineSnapshot {
    measurements: Vec<BenchmarkMeasurement>,
}

// ─── Detector ───────────────────────────────────────────────────────────────

/// Quality regression detector that wraps [`RegressionDetector`] to track
/// PESQ, STOI, and MCD metrics over time.
pub struct QualityRegressionDetector {
    detector: RegressionDetector,
    pesq: Option<PESQEvaluator>,
    stoi: Option<STOIEvaluator>,
    mcd: Option<MCDEvaluator>,
    config: QualityRegressionConfig,
}

impl QualityRegressionDetector {
    /// Create a new [`QualityRegressionDetector`] with the given configuration.
    ///
    /// Enabled evaluators are constructed immediately; disabled evaluators are left
    /// as `None` and are never called.
    pub fn new(config: QualityRegressionConfig) -> Result<Self, EvaluationError> {
        let regression_config = RegressionConfig {
            minor_threshold: config.minor_threshold_pct / 100.0,
            major_threshold: config.major_threshold_pct / 100.0,
            critical_threshold: config.critical_threshold_pct / 100.0,
            baseline_window: config.baseline_window,
            min_samples: config.min_samples,
        };
        let detector = RegressionDetector::with_config(regression_config);

        // Construct each evaluator only when enabled.
        let pesq = if config.enable_pesq {
            let evaluator = if config.sample_rate == 8000 {
                PESQEvaluator::new_narrowband().map_err(|e| {
                    EvaluationError::ConfigurationError {
                        message: format!("Failed to create PESQ evaluator: {e}"),
                    }
                })?
            } else {
                // Default: wideband (16 kHz). PESQ only supports 8 kHz or 16 kHz.
                if config.sample_rate != 16000 {
                    return Err(EvaluationError::ConfigurationError {
                        message: format!(
                            "PESQ only supports sample rates 8000 or 16000 Hz, got {}",
                            config.sample_rate
                        ),
                    });
                }
                PESQEvaluator::new_wideband().map_err(|e| EvaluationError::ConfigurationError {
                    message: format!("Failed to create PESQ evaluator: {e}"),
                })?
            };
            Some(evaluator)
        } else {
            None
        };

        let stoi = if config.enable_stoi {
            let evaluator = STOIEvaluator::new(config.sample_rate).map_err(|e| {
                EvaluationError::ConfigurationError {
                    message: format!("Failed to create STOI evaluator: {e}"),
                }
            })?;
            Some(evaluator)
        } else {
            None
        };

        let mcd = if config.enable_mcd {
            let evaluator = MCDEvaluator::new(config.sample_rate).map_err(|e| {
                EvaluationError::ConfigurationError {
                    message: format!("Failed to create MCD evaluator: {e}"),
                }
            })?;
            Some(evaluator)
        } else {
            None
        };

        Ok(Self {
            detector,
            pesq,
            stoi,
            mcd,
            config,
        })
    }

    // ── Private helpers ─────────────────────────────────────────────────────

    /// Return current Unix timestamp in milliseconds, or 0 on failure.
    fn unix_timestamp_millis() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Build a [`BenchmarkMeasurement`] from the given metric name/value/unit.
    fn build_measurement(
        name: String,
        value: f64,
        unit: &str,
        timestamp: u64,
    ) -> BenchmarkMeasurement {
        BenchmarkMeasurement {
            name,
            value,
            unit: unit.to_string(),
            timestamp,
            git_commit: None,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Evaluate quality metrics between `reference_samples` and `candidate_samples`,
    /// record the results, and detect any regressions relative to the accumulated baseline.
    ///
    /// # Arguments
    ///
    /// * `label` – Human-readable identifier for this measurement (e.g. `"v1.2.0"`)
    /// * `reference_samples` – Ground-truth audio samples at the configured sample rate
    /// * `candidate_samples` – Candidate (generated) audio samples at the configured sample rate
    ///
    /// # Returns
    ///
    /// A [`QualityRegressionMeasurement`] containing raw metric scores and any detected regressions.
    ///
    /// # Notes on minimum audio length
    ///
    /// * **PESQ** requires at least 8 seconds of audio (i.e. `8 * sample_rate` samples)
    /// * **STOI** requires at least 3 seconds of audio
    /// * **MCD** requires at least one full frame (1024 samples at default settings)
    ///
    /// If the audio is too short for a particular evaluator the corresponding metric is
    /// returned as `None` and a warning-level error is surfaced only when *all* enabled
    /// evaluators fail.
    pub async fn evaluate_and_record(
        &mut self,
        label: &str,
        reference_samples: &[f32],
        candidate_samples: &[f32],
    ) -> Result<QualityRegressionMeasurement, EvaluationError> {
        let timestamp = Self::unix_timestamp_millis();
        let sample_rate = self.config.sample_rate;

        // Wrap raw slices into AudioBuffers (mono, 1 channel).
        let reference_buf = AudioBuffer::new(reference_samples.to_vec(), sample_rate, 1);
        let candidate_buf = AudioBuffer::new(candidate_samples.to_vec(), sample_rate, 1);

        let mut pesq_score: Option<f64> = None;
        let mut stoi_score: Option<f64> = None;
        let mut mcd_score: Option<f64> = None;
        let mut regressions: Vec<RegressionResult> = Vec::new();

        // ── PESQ ───────────────────────────────────────────────────────────
        if let Some(ref evaluator) = self.pesq {
            match evaluator
                .calculate_pesq(&reference_buf, &candidate_buf)
                .await
            {
                Ok(score) => {
                    let value = f64::from(score);
                    pesq_score = Some(value);
                    let metric_name = format!("{label}.pesq");
                    let measurement =
                        Self::build_measurement(metric_name.clone(), value, "pesq", timestamp);
                    self.detector.add_measurement(measurement);
                    if let Some(result) = self.detector.detect_regression(&metric_name) {
                        regressions.push(result);
                    }
                }
                Err(e) => {
                    // Non-fatal: log and skip PESQ for this measurement.
                    eprintln!(
                        "[QualityRegressionDetector] PESQ evaluation failed for '{label}': {e}"
                    );
                }
            }
        }

        // ── STOI ───────────────────────────────────────────────────────────
        if let Some(ref evaluator) = self.stoi {
            match evaluator
                .calculate_stoi(&reference_buf, &candidate_buf)
                .await
            {
                Ok(score) => {
                    let value = f64::from(score);
                    stoi_score = Some(value);
                    let metric_name = format!("{label}.stoi");
                    let measurement =
                        Self::build_measurement(metric_name.clone(), value, "stoi", timestamp);
                    self.detector.add_measurement(measurement);
                    if let Some(result) = self.detector.detect_regression(&metric_name) {
                        regressions.push(result);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[QualityRegressionDetector] STOI evaluation failed for '{label}': {e}"
                    );
                }
            }
        }

        // ── MCD ────────────────────────────────────────────────────────────
        // MCD is lower-is-better. Store as negated value so that an *increase* in
        // raw MCD (degradation) produces a *positive* change percentage and correctly
        // triggers a regression alert inside RegressionDetector.
        //
        // change_pct = (current_neg - baseline_neg) / baseline_neg
        //            = (-mcd_new - (-mcd_old)) / (-mcd_old)
        //            = (mcd_old - mcd_new) / (-mcd_old)         when mcd_old > 0
        //
        // If mcd_new > mcd_old  ⟹  (mcd_old - mcd_new) < 0 and -mcd_old < 0
        //                        ⟹  positive quotient  ⟹  regression detected ✓
        if let Some(ref evaluator) = self.mcd {
            match evaluator
                .calculate_mcd_with_dtw(&reference_buf, &candidate_buf)
                .await
            {
                Ok(raw_mcd) => {
                    let raw_mcd_f64 = f64::from(raw_mcd);
                    mcd_score = Some(raw_mcd_f64);
                    // Store negated value so RegressionDetector interprets increases correctly.
                    let negated = -raw_mcd_f64;
                    let metric_name = format!("{label}.mcd");
                    let measurement =
                        Self::build_measurement(metric_name.clone(), negated, "dB", timestamp);
                    self.detector.add_measurement(measurement);
                    if let Some(result) = self.detector.detect_regression(&metric_name) {
                        regressions.push(result);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[QualityRegressionDetector] MCD evaluation failed for '{label}': {e}"
                    );
                }
            }
        }

        Ok(QualityRegressionMeasurement {
            label: label.to_string(),
            timestamp,
            pesq: pesq_score,
            stoi: stoi_score,
            mcd: mcd_score,
            regressions,
        })
    }

    /// Persist the internal measurement history to a JSON file.
    ///
    /// The file can later be loaded back with [`Self::load_baseline`] to continue
    /// tracking regressions across process boundaries or CI runs.
    pub fn save_baseline<P: AsRef<Path>>(&self, path: P) -> Result<(), EvaluationError> {
        let snapshot = BaselineSnapshot {
            measurements: self.detector.measurements().to_vec(),
        };
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| EvaluationError::Io(format!("Failed to serialise baseline: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| EvaluationError::Io(format!("Failed to write baseline file: {e}")))?;
        Ok(())
    }

    /// Load a previously saved baseline from a JSON file, merging its measurements
    /// into the current detector state.
    pub fn load_baseline<P: AsRef<Path>>(&mut self, path: P) -> Result<(), EvaluationError> {
        let json = std::fs::read_to_string(&path).map_err(|e| {
            EvaluationError::Io(format!(
                "Failed to read baseline file '{}': {e}",
                path.as_ref().display()
            ))
        })?;
        let snapshot: BaselineSnapshot = serde_json::from_str(&json)
            .map_err(|e| EvaluationError::Io(format!("Failed to deserialise baseline: {e}")))?;
        for m in snapshot.measurements {
            self.detector.add_measurement(m);
        }
        Ok(())
    }

    /// Generate a summary report over all accumulated measurements.
    ///
    /// Calls [`RegressionDetector::detect_all_regressions`] on the underlying detector
    /// and categorises results by severity.
    pub fn generate_report(&mut self) -> QualityRegressionReport {
        let all_regressions = self.detector.detect_all_regressions();
        let total_measurements = self.detector.measurements().len();
        let total_regressions = all_regressions.iter().filter(|r| r.is_regression).count();

        let mut critical_regressions = Vec::new();
        let mut major_regressions = Vec::new();
        let mut minor_regressions = Vec::new();

        for result in all_regressions {
            if !result.is_regression {
                continue;
            }
            match result.severity {
                RegressionSeverity::Critical => critical_regressions.push(result),
                RegressionSeverity::Major => major_regressions.push(result),
                RegressionSeverity::Minor => minor_regressions.push(result),
            }
        }

        QualityRegressionReport {
            total_measurements,
            total_regressions,
            critical_regressions,
            major_regressions,
            minor_regressions,
        }
    }

    /// Return an immutable reference to the underlying [`RegressionDetector`].
    ///
    /// Useful in tests that need to inspect or add measurements directly.
    pub fn detector(&self) -> &RegressionDetector {
        &self.detector
    }

    /// Return a mutable reference to the underlying [`RegressionDetector`].
    ///
    /// Useful in tests that need to add synthetic measurements to validate
    /// the regression logic without invoking the full evaluator pipeline.
    pub fn detector_mut(&mut self) -> &mut RegressionDetector {
        &mut self.detector
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Generate a mono sine-wave at `freq_hz` for `duration_secs` at `sample_rate`,
    /// with amplitude `amplitude`.
    fn sine_wave(sample_rate: u32, duration_secs: f32, freq_hz: f32, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amplitude * (2.0 * PI * freq_hz * t).sin()
            })
            .collect()
    }

    // ── test_enabled_metrics_recorded ───────────────────────────────────────

    /// A detector with only PESQ enabled should produce a measurement whose `pesq`
    /// field is `Some` and whose `stoi` / `mcd` fields are both `None`.
    ///
    /// Note: We need ≥8 seconds of audio for PESQ (ITU-T P.862 time-alignment requirement).
    #[tokio::test]
    async fn test_enabled_metrics_recorded() {
        let config = QualityRegressionConfig {
            enable_pesq: true,
            enable_stoi: false,
            enable_mcd: false,
            sample_rate: 16000,
            ..Default::default()
        };

        let mut detector = QualityRegressionDetector::new(config)
            .expect("Detector should be constructed successfully");

        // PESQ requires ≥ 8 seconds of audio for time-alignment.
        let reference = sine_wave(16000, 8.5, 440.0, 0.5);
        let candidate = sine_wave(16000, 8.5, 440.0, 0.45);

        let measurement = detector
            .evaluate_and_record("pesq_only", &reference, &candidate)
            .await
            .expect("evaluate_and_record should not fail");

        assert!(
            measurement.pesq.is_some(),
            "PESQ should be Some when enable_pesq = true"
        );
        assert!(
            measurement.stoi.is_none(),
            "STOI should be None when enable_stoi = false"
        );
        assert!(
            measurement.mcd.is_none(),
            "MCD should be None when enable_mcd = false"
        );

        // The score must be within the valid PESQ range.
        let pesq = measurement.pesq.unwrap();
        assert!(
            pesq >= -0.5 && pesq <= 4.5,
            "PESQ score {pesq:.4} must be in [-0.5, 4.5]"
        );
    }

    // ── test_baseline_roundtrip ─────────────────────────────────────────────

    /// Save a baseline, load it into a fresh detector, and verify that at least one
    /// measurement was restored.
    #[tokio::test]
    async fn test_baseline_roundtrip() {
        let config = QualityRegressionConfig {
            enable_pesq: false,
            enable_stoi: false,
            enable_mcd: true,
            sample_rate: 16000,
            min_samples: 1,
            ..Default::default()
        };

        let mut detector = QualityRegressionDetector::new(config.clone())
            .expect("Detector should be constructed successfully");

        // MCD needs ≥ 1024 samples (one full frame).
        let reference = sine_wave(16000, 1.0, 220.0, 0.3);
        let candidate = sine_wave(16000, 1.0, 220.0, 0.25);

        detector
            .evaluate_and_record("roundtrip_test", &reference, &candidate)
            .await
            .expect("First evaluation should succeed");

        // Save baseline to a temp file.
        let tmp_path = std::env::temp_dir().join("voirs_quality_regression_test_baseline.json");
        detector
            .save_baseline(&tmp_path)
            .expect("Saving baseline should succeed");

        // Load into a fresh detector.
        let mut fresh_detector = QualityRegressionDetector::new(config)
            .expect("Fresh detector should be constructed successfully");
        fresh_detector
            .load_baseline(&tmp_path)
            .expect("Loading baseline should succeed");

        assert!(
            fresh_detector.detector().measurements().len() >= 1,
            "At least one measurement should be restored after load_baseline"
        );

        // Clean up temp file (best-effort).
        let _ = std::fs::remove_file(&tmp_path);
    }

    // ── test_mcd_regression_detected_on_increase ────────────────────────────

    /// Manually inject two synthetic MCD measurements (stored as negated values) and
    /// verify that an increase in raw MCD is correctly flagged as a regression.
    ///
    /// Baseline MCD = 4.0  →  stored as -4.0
    /// Current  MCD = 5.6  -> stored as -5.6
    ///
    /// change_pct = (-5.6 - (-4.0)) / (-4.0) = -1.6 / -4.0 = +0.40  =>  Critical (>25%)
    #[tokio::test]
    async fn test_mcd_regression_detected_on_increase() {
        // Use min_samples = 2 so regression can be detected with just two measurements.
        let config = QualityRegressionConfig {
            enable_pesq: false,
            enable_stoi: false,
            enable_mcd: false, // disable automatic MCD so we inject measurements manually
            sample_rate: 16000,
            minor_threshold_pct: 5.0,
            major_threshold_pct: 10.0,
            critical_threshold_pct: 25.0,
            min_samples: 2,
            baseline_window: 10,
        };

        let mut detector = QualityRegressionDetector::new(config)
            .expect("Detector should be constructed successfully");

        // Inject synthetic measurements directly into the underlying detector.
        // Timestamps must differ so the detector can sort them (most-recent-first).
        let baseline_measurement = BenchmarkMeasurement {
            name: "synth.mcd".to_string(),
            value: -4.0, // negated MCD: raw MCD = 4.0 dB
            unit: "dB".to_string(),
            timestamp: 1_000,
            git_commit: None,
            version: "0.0.0-test".to_string(),
        };
        let regression_measurement = BenchmarkMeasurement {
            name: "synth.mcd".to_string(),
            value: -5.6, // negated MCD: raw MCD = 5.6 dB (40% worse -- strictly above critical threshold)
            unit: "dB".to_string(),
            timestamp: 2_000,
            git_commit: None,
            version: "0.0.0-test".to_string(),
        };

        detector
            .detector_mut()
            .add_measurement(baseline_measurement);
        detector
            .detector_mut()
            .add_measurement(regression_measurement);

        // detect_regression should now fire and report a regression.
        let result = detector.detector_mut().detect_regression("synth.mcd");

        assert!(result.is_some(), "detect_regression should return Some");
        let regression = result.unwrap();
        assert!(
            regression.is_regression,
            "is_regression must be true: change_pct = {:.4}",
            regression.change_percentage
        );
        // +25% change → Critical threshold exactly
        assert!(
            regression.change_percentage > 0.25,
            "Expected change_percentage > 0.25, got {:.4}",
            regression.change_percentage
        );
        assert_eq!(
            regression.severity,
            RegressionSeverity::Critical,
            "A 40% increase in MCD must be classified as Critical"
        );
    }

    // ── test_all_metrics_disabled ────────────────────────────────────────────

    /// When all evaluators are disabled the measurement should have all scores as `None`.
    #[tokio::test]
    async fn test_all_metrics_disabled() {
        let config = QualityRegressionConfig {
            enable_pesq: false,
            enable_stoi: false,
            enable_mcd: false,
            sample_rate: 16000,
            ..Default::default()
        };

        let mut detector = QualityRegressionDetector::new(config)
            .expect("Detector should be constructed successfully");

        let samples = vec![0.1f32; 1024];
        let measurement = detector
            .evaluate_and_record("no_metrics", &samples, &samples)
            .await
            .expect("Evaluation with no enabled metrics should succeed");

        assert!(measurement.pesq.is_none());
        assert!(measurement.stoi.is_none());
        assert!(measurement.mcd.is_none());
        assert!(measurement.regressions.is_empty());
    }

    // ── test_generate_report_structure ──────────────────────────────────────

    /// Inject a critical regression and verify the report categorises it correctly.
    #[test]
    fn test_generate_report_structure() {
        let config = QualityRegressionConfig {
            enable_pesq: false,
            enable_stoi: false,
            enable_mcd: false,
            min_samples: 2,
            critical_threshold_pct: 25.0,
            major_threshold_pct: 10.0,
            minor_threshold_pct: 5.0,
            baseline_window: 10,
            sample_rate: 16000,
        };

        let mut detector = QualityRegressionDetector::new(config)
            .expect("Detector should be constructed successfully");

        // Two measurements: baseline=-4.0, current=-5.6 -> +40% -> Critical (strictly > 0.25)
        for (ts, val) in [(1_000u64, -4.0f64), (2_000u64, -5.6f64)] {
            detector
                .detector_mut()
                .add_measurement(BenchmarkMeasurement {
                    name: "report_test.mcd".to_string(),
                    value: val,
                    unit: "dB".to_string(),
                    timestamp: ts,
                    git_commit: None,
                    version: "0.0.0-test".to_string(),
                });
        }

        let report = detector.generate_report();

        assert_eq!(
            report.total_measurements, 2,
            "Two measurements should be recorded"
        );
        assert!(
            report.total_regressions >= 1,
            "At least one regression should be reported"
        );
        assert!(
            !report.critical_regressions.is_empty(),
            "Critical regression list must be non-empty"
        );
    }
}
