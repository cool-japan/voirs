//! A/B Testing and Quality Comparison Example
//!
//! This example demonstrates comprehensive A/B testing and quality evaluation:
//! 1. Automated A/B test setup and execution
//! 2. Objective quality metrics (PESQ, STOI, MCD)
//! 3. Subjective evaluation framework simulation
//! 4. Statistical significance testing
//! 5. Performance vs quality trade-off analysis
//! 6. Comparative analysis across models and configurations
//!
//! ## Running this example:
//! ```bash
//! cargo run --example ab_testing_quality_comparison
//! ```
//!
//! ## Key Features:
//! - Multi-model comparison (A/B/C testing)
//! - Objective quality metrics computation
//! - Statistical significance analysis
//! - Performance benchmarking integration
//! - Quality score aggregation and ranking
//! - Automated test report generation

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    pub test_name: String,
    pub test_cases: Vec<TestCase>,
    pub quality_metrics: Vec<QualityMetric>,
    pub statistical_config: StatisticalConfig,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub name: String,
    pub description: String,
    pub model_config: ModelConfig,
    pub test_texts: Vec<String>,
    pub reference_audio_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_name: String,
    pub voice_id: String,
    pub quality_preset: QualityPreset,
    pub custom_parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityPreset {
    Fast,
    Balanced,
    HighQuality,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityMetric {
    PESQ,              // Perceptual Evaluation of Speech Quality
    STOI,              // Short-Time Objective Intelligibility
    MCD,               // Mel-Cepstral Distortion
    SpectralSNR,       // Spectral Signal-to-Noise Ratio
    F0RMSE,            // Fundamental Frequency Root Mean Square Error
    VoicingSimilarity, // Voicing decision accuracy
    RTF,               // Real-Time Factor (performance)
    SubjectiveScore,   // Simulated human evaluation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    pub confidence_level: f64, // e.g., 0.95 for 95%
    pub minimum_sample_size: usize,
    pub effect_size_threshold: f64,
    pub test_type: StatisticalTest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalTest {
    TTest,          // Two-sample t-test
    MannWhitney,    // Mann-Whitney U test (non-parametric)
    ANOVA,          // Analysis of variance (for multiple groups)
    WilcoxonSigned, // Wilcoxon signed-rank test
}

#[derive(Debug, Serialize)]
pub struct ABTestReport {
    pub test_id: String,
    pub config: ABTestConfig,
    pub results: Vec<TestCaseResult>,
    pub comparative_analysis: ComparativeAnalysis,
    pub statistical_results: StatisticalResults,
    pub recommendations: Vec<Recommendation>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub execution_time: Duration,
}

#[derive(Debug, Serialize)]
pub struct TestCaseResult {
    pub case_name: String,
    pub quality_scores: HashMap<String, f64>, // metric_name -> score
    pub performance_metrics: PerformanceMetrics,
    pub sample_count: usize,
    pub audio_samples: Vec<GeneratedSample>,
}

#[derive(Debug, Serialize)]
pub struct GeneratedSample {
    pub text: String,
    pub audio_path: PathBuf,
    pub generation_time: Duration,
    pub quality_scores: HashMap<String, f64>,
}

#[derive(Debug, Serialize)]
pub struct PerformanceMetrics {
    pub avg_rtf: f64,
    pub peak_rtf: f64,
    pub avg_latency_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput_samples_per_sec: f64,
}

#[derive(Debug, Serialize)]
pub struct ComparativeAnalysis {
    pub rankings: Vec<ModelRanking>,
    pub pairwise_comparisons: Vec<PairwiseComparison>,
    pub quality_vs_performance: QualityPerformanceAnalysis,
}

#[derive(Debug, Serialize)]
pub struct ModelRanking {
    pub metric: String,
    pub ranked_models: Vec<RankedModel>,
}

#[derive(Debug, Serialize)]
pub struct RankedModel {
    pub model_name: String,
    pub score: f64,
    pub rank: usize,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Serialize)]
pub struct PairwiseComparison {
    pub model_a: String,
    pub model_b: String,
    pub metric: String,
    pub p_value: f64,
    pub effect_size: f64,
    pub significant: bool,
    pub winner: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct QualityPerformanceAnalysis {
    pub pareto_optimal_configs: Vec<String>,
    pub quality_performance_scores: Vec<QualityPerformanceScore>,
    pub efficiency_rankings: Vec<EfficiencyRanking>,
}

#[derive(Debug, Serialize)]
pub struct QualityPerformanceScore {
    pub model_name: String,
    pub quality_score: f64,
    pub performance_score: f64,
    pub combined_score: f64,
}

#[derive(Debug, Serialize)]
pub struct EfficiencyRanking {
    pub model_name: String,
    pub efficiency_score: f64, // quality per unit of computational cost
    pub rank: usize,
}

#[derive(Debug, Serialize)]
pub struct StatisticalResults {
    pub overall_significance: bool,
    pub significant_comparisons: usize,
    pub total_comparisons: usize,
    pub power_analysis: PowerAnalysis,
}

#[derive(Debug, Serialize)]
pub struct PowerAnalysis {
    pub achieved_power: f64,
    pub recommended_sample_size: usize,
    pub current_sample_size: usize,
}

#[derive(Debug, Serialize)]
pub struct Recommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub description: String,
    pub supporting_evidence: Vec<String>,
}

#[derive(Debug, Serialize)]
pub enum RecommendationCategory {
    ModelSelection,
    ConfigurationOptimization,
    PerformanceTuning,
    QualityImprovement,
    DeploymentStrategy,
}

#[derive(Debug, Serialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

pub struct ABTestSuite {
    config: ABTestConfig,
}

impl ABTestSuite {
    pub fn new(config: ABTestConfig) -> Self {
        Self { config }
    }

    /// Run the complete A/B testing suite
    pub async fn run_tests(&mut self) -> Result<ABTestReport> {
        let test_id = Uuid::new_v4().to_string();
        let start_time = Instant::now();

        info!("🧪 Starting A/B Testing Suite: {}", self.config.test_name);
        info!("📋 Test ID: {}", test_id);

        // Create output directory
        tokio::fs::create_dir_all(&self.config.output_dir)
            .await
            .context("Failed to create output directory")?;

        let mut results = Vec::new();

        // Execute each test case
        for test_case in &self.config.test_cases.clone() {
            info!("🔬 Running test case: {}", test_case.name);

            let result = self.run_test_case(test_case).await?;
            results.push(result);
        }

        // Perform comparative analysis
        let comparative_analysis = self.perform_comparative_analysis(&results)?;

        // Run statistical tests
        let statistical_results = self.perform_statistical_analysis(&results)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(&results, &comparative_analysis)?;

        let execution_time = start_time.elapsed();

        let report = ABTestReport {
            test_id,
            config: self.config.clone(),
            results,
            comparative_analysis,
            statistical_results,
            recommendations,
            timestamp: chrono::Utc::now(),
            execution_time,
        };

        // Save report
        self.save_report(&report).await?;

        info!("✅ A/B Testing completed in {:?}", execution_time);
        Ok(report)
    }

    async fn run_test_case(&self, test_case: &TestCase) -> Result<TestCaseResult> {
        let mut audio_samples = Vec::new();
        let mut all_quality_scores: HashMap<String, Vec<f64>> = HashMap::new();
        let mut performance_metrics_samples = Vec::new();

        info!("  📝 Processing {} test texts", test_case.test_texts.len());

        for (idx, text) in test_case.test_texts.iter().enumerate() {
            info!(
                "    🎵 Synthesizing text {}/{}: \"{}\"",
                idx + 1,
                test_case.test_texts.len(),
                text.chars().take(50).collect::<String>()
            );

            let sample_result = self.synthesize_and_evaluate(test_case, text, idx).await?;

            // Collect quality scores for aggregation
            for (metric, score) in &sample_result.quality_scores {
                all_quality_scores
                    .entry(metric.clone())
                    .or_default()
                    .push(*score);
            }

            performance_metrics_samples.push(sample_result.generation_time);
            audio_samples.push(sample_result);
        }

        // Aggregate quality scores (mean, with confidence intervals)
        let quality_scores = all_quality_scores
            .into_iter()
            .map(|(metric, scores)| {
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                (metric, mean)
            })
            .collect();

        // Calculate performance metrics
        let performance_metrics =
            self.calculate_performance_metrics(&performance_metrics_samples, &audio_samples)?;

        Ok(TestCaseResult {
            case_name: test_case.name.clone(),
            quality_scores,
            performance_metrics,
            sample_count: audio_samples.len(),
            audio_samples,
        })
    }

    async fn synthesize_and_evaluate(
        &self,
        test_case: &TestCase,
        text: &str,
        idx: usize,
    ) -> Result<GeneratedSample> {
        let start_time = Instant::now();

        // In a real implementation, this would use the actual VoiRS synthesis
        // For demo purposes, we simulate the synthesis process
        info!(
            "      🔧 Simulating synthesis with {} model...",
            test_case.model_config.model_name
        );

        // Simulate synthesis time based on quality preset
        let synthesis_delay = match test_case.model_config.quality_preset {
            QualityPreset::Fast => Duration::from_millis(100),
            QualityPreset::Balanced => Duration::from_millis(300),
            QualityPreset::HighQuality => Duration::from_millis(800),
            QualityPreset::Custom => Duration::from_millis(500),
        };

        tokio::time::sleep(synthesis_delay).await;

        let generation_time = start_time.elapsed();

        // Generate output path
        let audio_path = self.config.output_dir.join(format!(
            "{}_{}_sample_{}.wav",
            test_case.name.replace(" ", "_"),
            test_case.model_config.model_name.replace(" ", "_"),
            idx
        ));

        // Simulate quality evaluation
        let quality_scores = self
            .evaluate_sample_quality(test_case, text, &audio_path)
            .await?;

        Ok(GeneratedSample {
            text: text.to_string(),
            audio_path,
            generation_time,
            quality_scores,
        })
    }

    async fn evaluate_sample_quality(
        &self,
        test_case: &TestCase,
        _text: &str,
        _audio_path: &PathBuf,
    ) -> Result<HashMap<String, f64>> {
        let mut scores = HashMap::new();

        // Simulate quality metric computation
        // In real implementation, this would use actual quality measurement libraries
        for metric in &self.config.quality_metrics {
            let score = self.simulate_quality_metric(metric, &test_case.model_config)?;
            scores.insert(format!("{:?}", metric), score);
        }

        Ok(scores)
    }

    fn simulate_quality_metric(&self, metric: &QualityMetric, config: &ModelConfig) -> Result<f64> {
        // Simulate realistic quality scores based on model configuration
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let base_score = match metric {
            QualityMetric::PESQ => {
                // PESQ typically ranges from 1.0 to 4.5
                match config.quality_preset {
                    QualityPreset::Fast => 2.5 + rng.r#gen::<f64>() * 0.5,
                    QualityPreset::Balanced => 3.2 + rng.r#gen::<f64>() * 0.4,
                    QualityPreset::HighQuality => 3.8 + rng.r#gen::<f64>() * 0.3,
                    QualityPreset::Custom => 3.0 + rng.r#gen::<f64>() * 0.8,
                }
            }
            QualityMetric::STOI => {
                // STOI ranges from 0.0 to 1.0
                match config.quality_preset {
                    QualityPreset::Fast => 0.75 + rng.r#gen::<f64>() * 0.15,
                    QualityPreset::Balanced => 0.85 + rng.r#gen::<f64>() * 0.10,
                    QualityPreset::HighQuality => 0.92 + rng.r#gen::<f64>() * 0.06,
                    QualityPreset::Custom => 0.80 + rng.r#gen::<f64>() * 0.15,
                }
            }
            QualityMetric::MCD => {
                // MCD (lower is better), typically 2-20 dB
                match config.quality_preset {
                    QualityPreset::Fast => 8.0 + rng.r#gen::<f64>() * 4.0,
                    QualityPreset::Balanced => 5.5 + rng.r#gen::<f64>() * 2.0,
                    QualityPreset::HighQuality => 3.2 + rng.r#gen::<f64>() * 1.5,
                    QualityPreset::Custom => 6.0 + rng.r#gen::<f64>() * 3.0,
                }
            }
            QualityMetric::RTF => {
                // Real-Time Factor (lower is better for performance)
                match config.quality_preset {
                    QualityPreset::Fast => 0.1 + rng.r#gen::<f64>() * 0.05,
                    QualityPreset::Balanced => 0.3 + rng.r#gen::<f64>() * 0.1,
                    QualityPreset::HighQuality => 0.8 + rng.r#gen::<f64>() * 0.2,
                    QualityPreset::Custom => 0.4 + rng.r#gen::<f64>() * 0.3,
                }
            }
            QualityMetric::SubjectiveScore => {
                // Subjective Mean Opinion Score (1-5)
                match config.quality_preset {
                    QualityPreset::Fast => 3.2 + rng.r#gen::<f64>() * 0.6,
                    QualityPreset::Balanced => 3.8 + rng.r#gen::<f64>() * 0.4,
                    QualityPreset::HighQuality => 4.3 + rng.r#gen::<f64>() * 0.3,
                    QualityPreset::Custom => 3.5 + rng.r#gen::<f64>() * 0.8,
                }
            }
            _ => {
                // Generic normalized score for other metrics
                match config.quality_preset {
                    QualityPreset::Fast => 0.6 + rng.r#gen::<f64>() * 0.2,
                    QualityPreset::Balanced => 0.75 + rng.r#gen::<f64>() * 0.15,
                    QualityPreset::HighQuality => 0.85 + rng.r#gen::<f64>() * 0.1,
                    QualityPreset::Custom => 0.7 + rng.r#gen::<f64>() * 0.25,
                }
            }
        };

        Ok(base_score)
    }

    fn calculate_performance_metrics(
        &self,
        generation_times: &[Duration],
        _samples: &[GeneratedSample],
    ) -> Result<PerformanceMetrics> {
        let times_ms: Vec<f64> = generation_times
            .iter()
            .map(|d| d.as_millis() as f64)
            .collect();

        let avg_latency_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let avg_rtf = avg_latency_ms / 1000.0; // Simplified RTF calculation
        let peak_rtf = times_ms.iter().fold(0.0f64, |acc, &x| acc.max(x / 1000.0));

        Ok(PerformanceMetrics {
            avg_rtf,
            peak_rtf,
            avg_latency_ms,
            memory_usage_mb: 150.0,                        // Simulated
            throughput_samples_per_sec: 22050.0 / avg_rtf, // Approximate
        })
    }

    fn perform_comparative_analysis(
        &self,
        results: &[TestCaseResult],
    ) -> Result<ComparativeAnalysis> {
        let mut rankings = Vec::new();
        let mut pairwise_comparisons = Vec::new();

        // Generate rankings for each metric
        let all_metrics: std::collections::HashSet<String> = results
            .iter()
            .flat_map(|r| r.quality_scores.keys().cloned())
            .collect();

        let all_metrics_vec: Vec<String> = all_metrics.iter().cloned().collect();
        for metric in &all_metrics_vec {
            let mut ranked_models: Vec<RankedModel> = results
                .iter()
                .filter_map(|result| {
                    result.quality_scores.get(metric).map(|&score| {
                        RankedModel {
                            model_name: result.case_name.clone(),
                            score,
                            rank: 0, // Will be set below
                            confidence_interval: (score * 0.95, score * 1.05), // Simplified CI
                        }
                    })
                })
                .collect();

            // Sort by score (higher is better for most metrics, except MCD and RTF)
            let reverse_sort = matches!(metric.as_str(), "MCD" | "RTF");
            ranked_models.sort_by(|a, b| {
                if reverse_sort {
                    a.score.partial_cmp(&b.score).unwrap()
                } else {
                    b.score.partial_cmp(&a.score).unwrap()
                }
            });

            // Assign ranks
            for (i, model) in ranked_models.iter_mut().enumerate() {
                model.rank = i + 1;
            }

            rankings.push(ModelRanking {
                metric: metric.clone(),
                ranked_models,
            });
        }

        // Generate pairwise comparisons
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                let model_a = &results[i];
                let model_b = &results[j];

                for metric in &all_metrics_vec {
                    if let (Some(&score_a), Some(&score_b)) = (
                        model_a.quality_scores.get(metric),
                        model_b.quality_scores.get(metric),
                    ) {
                        // Simulate statistical test
                        let p_value = self.simulate_statistical_test(score_a, score_b)?;
                        let effect_size = ((score_a - score_b) / ((score_a + score_b) / 2.0)).abs();
                        let significant =
                            p_value < (1.0 - self.config.statistical_config.confidence_level);

                        let winner = if significant {
                            if score_a > score_b {
                                Some(model_a.case_name.clone())
                            } else {
                                Some(model_b.case_name.clone())
                            }
                        } else {
                            None
                        };

                        pairwise_comparisons.push(PairwiseComparison {
                            model_a: model_a.case_name.clone(),
                            model_b: model_b.case_name.clone(),
                            metric: metric.clone(),
                            p_value,
                            effect_size,
                            significant,
                            winner,
                        });
                    }
                }
            }
        }

        // Quality vs Performance Analysis
        let quality_performance = self.analyze_quality_performance(results)?;

        Ok(ComparativeAnalysis {
            rankings,
            pairwise_comparisons,
            quality_vs_performance: quality_performance,
        })
    }

    fn simulate_statistical_test(&self, score_a: f64, score_b: f64) -> Result<f64> {
        // Simplified p-value simulation based on score difference
        let difference = (score_a - score_b).abs();
        let relative_diff = difference / ((score_a + score_b) / 2.0);

        // Simulate p-value: larger differences -> smaller p-values
        let p_value = (1.0 - relative_diff).max(0.001).min(0.999);
        Ok(p_value)
    }

    fn analyze_quality_performance(
        &self,
        results: &[TestCaseResult],
    ) -> Result<QualityPerformanceAnalysis> {
        let mut quality_performance_scores = Vec::new();

        for result in results {
            // Calculate composite quality score (higher is better)
            let quality_metrics = ["PESQ", "STOI", "SubjectiveScore"];
            let quality_score = quality_metrics
                .iter()
                .filter_map(|&metric| result.quality_scores.get(metric))
                .sum::<f64>()
                / quality_metrics.len() as f64;

            // Calculate performance score (inverse RTF, so higher is better)
            let performance_score = 1.0 / result.performance_metrics.avg_rtf.max(0.01);

            // Combined score (weighted average)
            let combined_score = 0.7 * quality_score + 0.3 * performance_score;

            quality_performance_scores.push(QualityPerformanceScore {
                model_name: result.case_name.clone(),
                quality_score,
                performance_score,
                combined_score,
            });
        }

        // Find Pareto optimal configurations
        let mut pareto_optimal_configs = Vec::new();
        for i in 0..quality_performance_scores.len() {
            let mut is_dominated = false;
            for j in 0..quality_performance_scores.len() {
                if i != j {
                    let a = &quality_performance_scores[i];
                    let b = &quality_performance_scores[j];
                    // b dominates a if b is better or equal in both dimensions and strictly better in at least one
                    if b.quality_score >= a.quality_score
                        && b.performance_score >= a.performance_score
                        && (b.quality_score > a.quality_score
                            || b.performance_score > a.performance_score)
                    {
                        is_dominated = true;
                        break;
                    }
                }
            }
            if !is_dominated {
                pareto_optimal_configs.push(quality_performance_scores[i].model_name.clone());
            }
        }

        // Efficiency rankings (quality per unit of computational cost)
        let mut efficiency_rankings: Vec<EfficiencyRanking> = results
            .iter()
            .map(|result| {
                let quality = result.quality_scores.values().sum::<f64>()
                    / result.quality_scores.len() as f64;
                let efficiency = quality / result.performance_metrics.avg_rtf.max(0.01);
                EfficiencyRanking {
                    model_name: result.case_name.clone(),
                    efficiency_score: efficiency,
                    rank: 0, // Will be set below
                }
            })
            .collect();

        efficiency_rankings
            .sort_by(|a, b| b.efficiency_score.partial_cmp(&a.efficiency_score).unwrap());
        for (i, ranking) in efficiency_rankings.iter_mut().enumerate() {
            ranking.rank = i + 1;
        }

        Ok(QualityPerformanceAnalysis {
            pareto_optimal_configs,
            quality_performance_scores,
            efficiency_rankings,
        })
    }

    fn perform_statistical_analysis(
        &self,
        results: &[TestCaseResult],
    ) -> Result<StatisticalResults> {
        let total_comparisons = results.len() * (results.len() - 1) / 2;
        let significant_comparisons = (total_comparisons as f64 * 0.3) as usize; // Simulated

        let power_analysis = PowerAnalysis {
            achieved_power: 0.85, // Simulated
            recommended_sample_size: self.config.statistical_config.minimum_sample_size * 2,
            current_sample_size: results.iter().map(|r| r.sample_count).max().unwrap_or(0),
        };

        Ok(StatisticalResults {
            overall_significance: significant_comparisons > 0,
            significant_comparisons,
            total_comparisons,
            power_analysis,
        })
    }

    fn generate_recommendations(
        &self,
        results: &[TestCaseResult],
        analysis: &ComparativeAnalysis,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // Model Selection Recommendations
        if let Some(best_overall) = analysis.quality_vs_performance.efficiency_rankings.first() {
            recommendations.push(Recommendation {
                category: RecommendationCategory::ModelSelection,
                priority: RecommendationPriority::High,
                description: format!(
                    "Consider using '{}' as the primary model due to its superior efficiency score of {:.2}",
                    best_overall.model_name, best_overall.efficiency_score
                ),
                supporting_evidence: vec![
                    "Highest efficiency ranking in quality-per-computational-cost analysis".to_string(),
                    "Strong performance across multiple quality metrics".to_string(),
                ],
            });
        }

        // Pareto Optimal Recommendations
        if !analysis
            .quality_vs_performance
            .pareto_optimal_configs
            .is_empty()
        {
            recommendations.push(Recommendation {
                category: RecommendationCategory::DeploymentStrategy,
                priority: RecommendationPriority::Medium,
                description: format!(
                    "The following configurations are Pareto optimal: {}. Consider using different configs for different use cases.",
                    analysis.quality_vs_performance.pareto_optimal_configs.join(", ")
                ),
                supporting_evidence: vec![
                    "These configurations offer the best trade-offs between quality and performance".to_string(),
                    "No other configuration dominates these in both quality and performance".to_string(),
                ],
            });
        }

        // Quality Improvement Recommendations
        let avg_quality: f64 = results
            .iter()
            .map(|r| r.quality_scores.values().sum::<f64>() / r.quality_scores.len() as f64)
            .sum::<f64>()
            / results.len() as f64;

        if avg_quality < 3.5 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::QualityImprovement,
                priority: RecommendationPriority::Critical,
                description: "Overall quality scores are below acceptable threshold. Consider model fine-tuning or higher quality presets.".to_string(),
                supporting_evidence: vec![
                    format!("Average quality score: {:.2} (target: >3.5)", avg_quality),
                    "Multiple quality metrics indicate suboptimal performance".to_string(),
                ],
            });
        }

        Ok(recommendations)
    }

    async fn save_report(&self, report: &ABTestReport) -> Result<()> {
        // Save JSON report
        let json_path = self
            .config
            .output_dir
            .join(format!("ab_test_report_{}.json", report.test_id));
        let json_content = serde_json::to_string_pretty(report)?;
        tokio::fs::write(&json_path, json_content).await?;
        info!("📄 JSON report saved to: {}", json_path.display());

        // Save human-readable summary
        let summary_path = self
            .config
            .output_dir
            .join(format!("ab_test_summary_{}.txt", report.test_id));
        let summary = self.generate_text_summary(report)?;
        tokio::fs::write(&summary_path, summary).await?;
        info!("📄 Summary report saved to: {}", summary_path.display());

        Ok(())
    }

    fn generate_text_summary(&self, report: &ABTestReport) -> Result<String> {
        let mut summary = String::new();

        summary.push_str(&format!("🧪 VoiRS A/B Testing Report\n"));
        summary.push_str(&"=".repeat(60));
        summary.push_str(&format!("\nTest: {}\n", report.config.test_name));
        summary.push_str(&format!("ID: {}\n", report.test_id));
        summary.push_str(&format!(
            "Timestamp: {}\n",
            report.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        summary.push_str(&format!("Duration: {:?}\n\n", report.execution_time));

        summary.push_str("📊 Test Results Summary:\n");
        summary.push_str(&"-".repeat(30));
        summary.push_str(&format!("\n"));

        for result in &report.results {
            summary.push_str(&format!("\n🔬 {}\n", result.case_name));
            summary.push_str(&format!("  Samples: {}\n", result.sample_count));
            summary.push_str(&format!(
                "  Avg RTF: {:.3}x\n",
                result.performance_metrics.avg_rtf
            ));

            for (metric, score) in &result.quality_scores {
                summary.push_str(&format!("  {}: {:.3}\n", metric, score));
            }
        }

        summary.push_str(&format!("\n🏆 Rankings:\n"));
        summary.push_str(&"-".repeat(20));
        summary.push_str(&format!("\n"));

        for ranking in &report.comparative_analysis.rankings {
            summary.push_str(&format!("\n📈 {} Rankings:\n", ranking.metric));
            for (i, model) in ranking.ranked_models.iter().enumerate() {
                summary.push_str(&format!(
                    "  {}. {} ({:.3})\n",
                    i + 1,
                    model.model_name,
                    model.score
                ));
            }
        }

        summary.push_str(&format!("\n💡 Recommendations:\n"));
        summary.push_str(&"-".repeat(25));
        summary.push_str(&format!("\n"));

        for rec in &report.recommendations {
            let priority_emoji = match rec.priority {
                RecommendationPriority::Critical => "🚨",
                RecommendationPriority::High => "⚡",
                RecommendationPriority::Medium => "💡",
                RecommendationPriority::Low => "💭",
            };
            summary.push_str(&format!("\n{} {}\n", priority_emoji, rec.description));
            for evidence in &rec.supporting_evidence {
                summary.push_str(&format!("   • {}\n", evidence));
            }
        }

        Ok(summary)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🧪 VoiRS A/B Testing and Quality Comparison");
    println!("{}", "=".repeat(60));

    // Configure test cases
    let test_config = ABTestConfig {
        test_name: "Voice Model Quality Comparison".to_string(),
        test_cases: vec![
            TestCase {
                name: "Fast Model".to_string(),
                description: "High-speed synthesis optimized for low latency".to_string(),
                model_config: ModelConfig {
                    model_name: "voirs-fast".to_string(),
                    voice_id: "neural-voice-1".to_string(),
                    quality_preset: QualityPreset::Fast,
                    custom_parameters: HashMap::new(),
                },
                test_texts: vec![
                    "Hello, this is a test of the VoiRS speech synthesis system.".to_string(),
                    "The quick brown fox jumps over the lazy dog.".to_string(),
                    "VoiRS provides high-quality neural voice synthesis capabilities.".to_string(),
                    "This longer sentence tests the model's ability to handle extended utterances with proper prosody and intonation patterns.".to_string(),
                ],
                reference_audio_dir: None,
            },
            TestCase {
                name: "Balanced Model".to_string(),
                description: "Balanced synthesis providing good quality-performance trade-off".to_string(),
                model_config: ModelConfig {
                    model_name: "voirs-balanced".to_string(),
                    voice_id: "neural-voice-1".to_string(),
                    quality_preset: QualityPreset::Balanced,
                    custom_parameters: HashMap::new(),
                },
                test_texts: vec![
                    "Hello, this is a test of the VoiRS speech synthesis system.".to_string(),
                    "The quick brown fox jumps over the lazy dog.".to_string(),
                    "VoiRS provides high-quality neural voice synthesis capabilities.".to_string(),
                    "This longer sentence tests the model's ability to handle extended utterances with proper prosody and intonation patterns.".to_string(),
                ],
                reference_audio_dir: None,
            },
            TestCase {
                name: "High Quality Model".to_string(),
                description: "Premium synthesis optimized for maximum quality".to_string(),
                model_config: ModelConfig {
                    model_name: "voirs-premium".to_string(),
                    voice_id: "neural-voice-1".to_string(),
                    quality_preset: QualityPreset::HighQuality,
                    custom_parameters: HashMap::new(),
                },
                test_texts: vec![
                    "Hello, this is a test of the VoiRS speech synthesis system.".to_string(),
                    "The quick brown fox jumps over the lazy dog.".to_string(),
                    "VoiRS provides high-quality neural voice synthesis capabilities.".to_string(),
                    "This longer sentence tests the model's ability to handle extended utterances with proper prosody and intonation patterns.".to_string(),
                ],
                reference_audio_dir: None,
            },
        ],
        quality_metrics: vec![
            QualityMetric::PESQ,
            QualityMetric::STOI,
            QualityMetric::MCD,
            QualityMetric::RTF,
            QualityMetric::SubjectiveScore,
        ],
        statistical_config: StatisticalConfig {
            confidence_level: 0.95,
            minimum_sample_size: 10,
            effect_size_threshold: 0.2,
            test_type: StatisticalTest::TTest,
        },
        output_dir: PathBuf::from("/tmp/voirs_ab_test_results"),
    };

    // Run A/B testing
    let mut test_suite = ABTestSuite::new(test_config);

    match test_suite.run_tests().await {
        Ok(report) => {
            println!("\n✅ A/B Testing completed successfully!");
            println!("📊 Test Results:");
            println!("   • {} test cases executed", report.results.len());
            println!(
                "   • {} total comparisons",
                report.statistical_results.total_comparisons
            );
            println!(
                "   • {} significant differences found",
                report.statistical_results.significant_comparisons
            );
            println!(
                "   • {} recommendations generated",
                report.recommendations.len()
            );

            // Print top recommendations
            if !report.recommendations.is_empty() {
                println!("\n💡 Top Recommendations:");
                for (i, rec) in report.recommendations.iter().take(3).enumerate() {
                    let priority_symbol = match rec.priority {
                        RecommendationPriority::Critical => "🚨",
                        RecommendationPriority::High => "⚡",
                        RecommendationPriority::Medium => "💡",
                        RecommendationPriority::Low => "💭",
                    };
                    println!("   {}. {} {}", i + 1, priority_symbol, rec.description);
                }
            }

            // Print efficiency rankings
            if !report
                .comparative_analysis
                .quality_vs_performance
                .efficiency_rankings
                .is_empty()
            {
                println!("\n🏆 Efficiency Rankings:");
                for (i, ranking) in report
                    .comparative_analysis
                    .quality_vs_performance
                    .efficiency_rankings
                    .iter()
                    .take(3)
                    .enumerate()
                {
                    println!(
                        "   {}. {} (efficiency: {:.2})",
                        i + 1,
                        ranking.model_name,
                        ranking.efficiency_score
                    );
                }
            }

            println!("\n📄 Detailed reports saved to: /tmp/voirs_ab_test_results/");
        }
        Err(e) => {
            eprintln!("❌ A/B Testing failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
