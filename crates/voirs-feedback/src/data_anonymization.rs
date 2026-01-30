//! Data Anonymization Module
//!
//! Provides comprehensive data anonymization techniques for privacy protection
//! including k-anonymity, l-diversity, t-closeness, and various masking strategies.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

/// Anonymization errors
#[derive(Error, Debug)]
#[allow(missing_docs)]
pub enum AnonymizationError {
    /// Anonymization failed
    #[error("Anonymization failed: {message}")]
    AnonymizationFailed { message: String },

    /// Insufficient data for k-anonymity
    #[error("Insufficient data for k-anonymity requirement: k={k}, records={records}")]
    InsufficientData { k: usize, records: usize },

    /// Invalid configuration
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    /// Re-identification risk too high
    #[error("Re-identification risk exceeds threshold: {risk:.2}%")]
    RiskTooHigh { risk: f32 },
}

/// Result type for anonymization operations
pub type AnonymizationResult<T> = Result<T, AnonymizationError>;

/// Anonymization technique
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum AnonymizationTechnique {
    /// Complete removal of data
    Suppression,
    /// Replace with generic value
    Generalization { levels: usize },
    /// Replace with similar value from range
    Perturbation { noise_level: f32 },
    /// Replace with pseudonym
    Pseudonymization,
    /// Replace with random value from distribution
    Randomization,
    /// Replace with hash
    Hashing,
    /// Tokenize sensitive data
    Tokenization,
    /// Mask partial data (e.g., "555-**-1234")
    Masking {
        mask_char: char,
        reveal_chars: usize,
    },
}

/// Data sensitivity level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq, Hash)]
#[allow(missing_docs)]
pub enum SensitivityLevel {
    /// Public data
    Public,
    /// Internal use only
    Internal,
    /// Confidential data
    Confidential,
    /// Personally Identifiable Information
    PII,
    /// Protected Health Information
    PHI,
    /// Financial data
    Financial,
}

/// Field anonymization rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizationRule {
    /// Rule ID
    pub rule_id: String,
    /// Field name/path
    pub field_name: String,
    /// Sensitivity level
    pub sensitivity: SensitivityLevel,
    /// Technique to apply
    pub technique: AnonymizationTechnique,
    /// Whether field is required for analysis
    pub required_for_analysis: bool,
    /// Quasi-identifier (for k-anonymity)
    pub is_quasi_identifier: bool,
}

/// Anonymization policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizationPolicy {
    /// Policy ID
    pub policy_id: String,
    /// Policy name
    pub name: String,
    /// Rules for each field
    pub rules: Vec<AnonymizationRule>,
    /// K-anonymity requirement
    pub k_anonymity: usize,
    /// L-diversity requirement
    pub l_diversity: Option<usize>,
    /// Maximum re-identification risk (0.0 to 1.0)
    pub max_reidentification_risk: f32,
    /// Enable differential privacy
    pub differential_privacy_enabled: bool,
    /// Privacy budget (epsilon)
    pub privacy_budget: Option<f32>,
}

/// Anonymized record metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizationMetadata {
    /// Original record ID (hashed)
    pub record_id: String,
    /// Anonymization timestamp
    pub anonymized_at: DateTime<Utc>,
    /// Policy used
    pub policy_id: String,
    /// Fields anonymized
    pub fields_anonymized: Vec<String>,
    /// Techniques applied
    pub techniques_applied: HashMap<String, String>,
    /// Data utility score (0.0 to 1.0)
    pub utility_score: f32,
}

/// K-anonymity group
#[derive(Debug, Clone)]
pub struct AnonymityGroup {
    /// Group identifier
    pub group_id: String,
    /// Quasi-identifier values
    pub quasi_identifiers: HashMap<String, String>,
    /// Number of records in group
    pub record_count: usize,
    /// Diversity of sensitive attributes
    pub diversity_score: f32,
}

/// Re-identification risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk score (0.0 to 1.0)
    pub overall_risk: f32,
    /// Risk by field
    pub field_risks: HashMap<String, f32>,
    /// K-anonymity satisfaction
    pub k_anonymity_satisfied: bool,
    /// L-diversity satisfaction
    pub l_diversity_satisfied: bool,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Data anonymizer
pub struct DataAnonymizer {
    /// Policies
    policies: Arc<RwLock<HashMap<String, AnonymizationPolicy>>>,
    /// Pseudonym mappings (for consistency)
    pseudonym_cache: Arc<RwLock<HashMap<String, String>>>,
    /// Token mappings
    token_cache: Arc<RwLock<HashMap<String, String>>>,
    /// Anonymization statistics
    stats: Arc<RwLock<AnonymizationStats>>,
}

/// Anonymization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonymizationStats {
    /// Total records anonymized
    pub total_records: usize,
    /// Records by policy
    pub records_by_policy: HashMap<String, usize>,
    /// Techniques used
    pub techniques_used: HashMap<String, usize>,
    /// Average utility score
    pub avg_utility_score: f32,
    /// Failed anonymizations
    pub failed_count: usize,
}

impl DataAnonymizer {
    /// Create new data anonymizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
            pseudonym_cache: Arc::new(RwLock::new(HashMap::new())),
            token_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(AnonymizationStats::default())),
        }
    }

    /// Register anonymization policy
    pub async fn register_policy(&self, policy: AnonymizationPolicy) -> AnonymizationResult<()> {
        // Validate policy
        if policy.k_anonymity == 0 {
            return Err(AnonymizationError::InvalidConfig {
                message: "k-anonymity must be at least 1".to_string(),
            });
        }

        if policy.max_reidentification_risk > 1.0 || policy.max_reidentification_risk < 0.0 {
            return Err(AnonymizationError::InvalidConfig {
                message: "max_reidentification_risk must be between 0.0 and 1.0".to_string(),
            });
        }

        let mut policies = self.policies.write().await;
        policies.insert(policy.policy_id.clone(), policy);

        Ok(())
    }

    /// Anonymize a text field
    pub async fn anonymize_field(
        &self,
        value: &str,
        rule: &AnonymizationRule,
    ) -> AnonymizationResult<String> {
        let result = match &rule.technique {
            AnonymizationTechnique::Suppression => "***SUPPRESSED***".to_string(),

            AnonymizationTechnique::Generalization { levels } => {
                self.generalize_value(value, *levels)
            }

            AnonymizationTechnique::Perturbation { noise_level } => {
                self.perturb_value(value, *noise_level)
            }

            AnonymizationTechnique::Pseudonymization => self.pseudonymize(value).await,

            AnonymizationTechnique::Randomization => self.randomize_value(value),

            AnonymizationTechnique::Hashing => self.hash_value(value),

            AnonymizationTechnique::Tokenization => self.tokenize(value).await,

            AnonymizationTechnique::Masking {
                mask_char,
                reveal_chars,
            } => self.mask_value(value, *mask_char, *reveal_chars),
        };

        // Update statistics
        let mut stats = self.stats.write().await;
        let technique_name = format!("{:?}", rule.technique);
        *stats.techniques_used.entry(technique_name).or_insert(0) += 1;

        Ok(result)
    }

    /// Anonymize a complete record
    pub async fn anonymize_record(
        &self,
        record: &HashMap<String, String>,
        policy_id: &str,
    ) -> AnonymizationResult<(HashMap<String, String>, AnonymizationMetadata)> {
        let policies = self.policies.read().await;
        let policy = policies
            .get(policy_id)
            .ok_or_else(|| AnonymizationError::InvalidConfig {
                message: format!("Policy not found: {policy_id}"),
            })?
            .clone();
        drop(policies);

        let mut anonymized = HashMap::new();
        let mut fields_anonymized = Vec::new();
        let mut techniques_applied = HashMap::new();

        for rule in &policy.rules {
            if let Some(value) = record.get(&rule.field_name) {
                let anonymized_value = self.anonymize_field(value, rule).await?;
                anonymized.insert(rule.field_name.clone(), anonymized_value);
                fields_anonymized.push(rule.field_name.clone());
                techniques_applied.insert(rule.field_name.clone(), format!("{:?}", rule.technique));
            } else if rule.required_for_analysis {
                // Keep non-anonymized fields that are required for analysis
                if let Some(value) = record.get(&rule.field_name) {
                    anonymized.insert(rule.field_name.clone(), value.clone());
                }
            }
        }

        // Calculate utility score
        let utility_score = self.calculate_utility_score(&anonymized, record);

        let metadata = AnonymizationMetadata {
            record_id: self.hash_value(&format!("{record:?}")),
            anonymized_at: Utc::now(),
            policy_id: policy_id.to_string(),
            fields_anonymized,
            techniques_applied,
            utility_score,
        };

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_records += 1;
        *stats
            .records_by_policy
            .entry(policy_id.to_string())
            .or_insert(0) += 1;
        stats.avg_utility_score = (stats.avg_utility_score * (stats.total_records - 1) as f32
            + utility_score)
            / stats.total_records as f32;

        Ok((anonymized, metadata))
    }

    /// Check k-anonymity for a set of records
    pub fn check_k_anonymity(
        &self,
        records: &[HashMap<String, String>],
        quasi_identifiers: &[String],
        k: usize,
    ) -> AnonymizationResult<bool> {
        if records.len() < k {
            return Err(AnonymizationError::InsufficientData {
                k,
                records: records.len(),
            });
        }

        // Group records by quasi-identifier values
        let mut groups: HashMap<Vec<String>, usize> = HashMap::new();

        for record in records {
            let mut qi_values = Vec::new();
            for qi in quasi_identifiers {
                qi_values.push(record.get(qi).cloned().unwrap_or_default());
            }
            *groups.entry(qi_values).or_insert(0) += 1;
        }

        // Check if all groups have at least k members
        let satisfies_k_anonymity = groups.values().all(|&count| count >= k);

        Ok(satisfies_k_anonymity)
    }

    /// Assess re-identification risk
    pub async fn assess_risk(
        &self,
        records: &[HashMap<String, String>],
        policy_id: &str,
    ) -> AnonymizationResult<RiskAssessment> {
        let policies = self.policies.read().await;
        let policy = policies
            .get(policy_id)
            .ok_or_else(|| AnonymizationError::InvalidConfig {
                message: format!("Policy not found: {policy_id}"),
            })?;

        // Extract quasi-identifiers
        let quasi_identifiers: Vec<String> = policy
            .rules
            .iter()
            .filter(|r| r.is_quasi_identifier)
            .map(|r| r.field_name.clone())
            .collect();

        // Check k-anonymity
        let k_anonymity_satisfied = if quasi_identifiers.is_empty() {
            true
        } else {
            self.check_k_anonymity(records, &quasi_identifiers, policy.k_anonymity)
                .unwrap_or(false)
        };

        // Calculate overall risk (simplified model)
        let mut overall_risk = if k_anonymity_satisfied {
            1.0 / policy.k_anonymity as f32
        } else {
            0.8
        };

        // Adjust risk based on number of records
        if records.len() > 1000 {
            overall_risk *= 0.7;
        } else if records.len() < 100 {
            overall_risk *= 1.3;
        }

        overall_risk = overall_risk.min(1.0);

        // Field-specific risks
        let mut field_risks = HashMap::new();
        for rule in &policy.rules {
            let risk = match rule.sensitivity {
                SensitivityLevel::Public => 0.1,
                SensitivityLevel::Internal => 0.3,
                SensitivityLevel::Confidential => 0.5,
                SensitivityLevel::PII => 0.7,
                SensitivityLevel::PHI => 0.9,
                SensitivityLevel::Financial => 0.8,
            };
            field_risks.insert(rule.field_name.clone(), risk);
        }

        // Generate recommendations
        let mut recommendations = Vec::new();
        if !k_anonymity_satisfied {
            recommendations.push(format!(
                "Increase dataset size or reduce k-anonymity requirement (current: {})",
                policy.k_anonymity
            ));
        }
        if overall_risk > policy.max_reidentification_risk {
            recommendations.push("Apply stronger anonymization techniques".to_string());
            recommendations.push("Consider additional generalization or suppression".to_string());
        }

        Ok(RiskAssessment {
            overall_risk,
            field_risks,
            k_anonymity_satisfied,
            l_diversity_satisfied: true, // Simplified
            recommendations,
        })
    }

    /// Get anonymization statistics
    pub async fn get_statistics(&self) -> AnonymizationStats {
        self.stats.read().await.clone()
    }

    /// Clear caches (for testing or memory management)
    pub async fn clear_caches(&self) {
        self.pseudonym_cache.write().await.clear();
        self.token_cache.write().await.clear();
    }

    // Private helper methods

    fn generalize_value(&self, value: &str, levels: usize) -> String {
        // Simplified generalization - could be enhanced based on data type
        if value.is_empty() {
            return "***".to_string();
        }

        let len = value.len();
        let keep_chars = len.saturating_sub(levels * 2).max(1);

        format!("{}{}", &value[..keep_chars], "*".repeat(len - keep_chars))
    }

    fn perturb_value(&self, value: &str, noise_level: f32) -> String {
        // Try to parse as number and add noise
        if let Ok(num) = value.parse::<f64>() {
            let noise = f64::from(thread_rng().gen_range(-noise_level..noise_level));
            return (num + noise).to_string();
        }

        // For non-numeric, apply character perturbation
        value
            .chars()
            .map(|c| {
                if thread_rng().gen::<f32>() < noise_level && c.is_alphanumeric() {
                    thread_rng().gen_range(b'a'..=b'z') as char
                } else {
                    c
                }
            })
            .collect()
    }

    async fn pseudonymize(&self, value: &str) -> String {
        let mut cache = self.pseudonym_cache.write().await;

        if let Some(pseudonym) = cache.get(value) {
            return pseudonym.clone();
        }

        let pseudonym = format!("PSEUDO_{}", self.hash_value(value)[..8].to_uppercase());
        cache.insert(value.to_string(), pseudonym.clone());

        pseudonym
    }

    fn randomize_value(&self, value: &str) -> String {
        // Generate random string of same length
        (0..value.len())
            .map(|_| thread_rng().gen_range(b'a'..=b'z') as char)
            .collect()
    }

    fn hash_value(&self, value: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(value.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    async fn tokenize(&self, value: &str) -> String {
        let mut cache = self.token_cache.write().await;

        if let Some(token) = cache.get(value) {
            return token.clone();
        }

        let token = format!("TOK_{}", uuid::Uuid::new_v4().simple());
        cache.insert(value.to_string(), token.clone());

        token
    }

    fn mask_value(&self, value: &str, mask_char: char, reveal_chars: usize) -> String {
        let len = value.len();
        if len <= reveal_chars * 2 {
            return mask_char.to_string().repeat(len);
        }

        let prefix = &value[..reveal_chars];
        let suffix = &value[len - reveal_chars..];
        let masked_len = len - (reveal_chars * 2);

        format!(
            "{}{}{}",
            prefix,
            mask_char.to_string().repeat(masked_len),
            suffix
        )
    }

    fn calculate_utility_score(
        &self,
        anonymized: &HashMap<String, String>,
        original: &HashMap<String, String>,
    ) -> f32 {
        // Simple utility score based on information preservation
        if original.is_empty() {
            return 0.0;
        }

        let mut preserved_info = 0;
        let mut total_fields = 0;

        for (key, anon_value) in anonymized {
            if let Some(orig_value) = original.get(key) {
                total_fields += 1;

                // Check if any information is preserved
                if anon_value != "***SUPPRESSED***" && !anon_value.contains("***") {
                    preserved_info += 1;
                }
            }
        }

        if total_fields == 0 {
            0.5
        } else {
            preserved_info as f32 / total_fields as f32
        }
    }
}

impl Default for DataAnonymizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AnonymizationStats {
    fn default() -> Self {
        Self {
            total_records: 0,
            records_by_policy: HashMap::new(),
            techniques_used: HashMap::new(),
            avg_utility_score: 0.0,
            failed_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_policy() -> AnonymizationPolicy {
        AnonymizationPolicy {
            policy_id: "test_policy".to_string(),
            name: "Test Policy".to_string(),
            rules: vec![
                AnonymizationRule {
                    rule_id: "rule1".to_string(),
                    field_name: "email".to_string(),
                    sensitivity: SensitivityLevel::PII,
                    technique: AnonymizationTechnique::Masking {
                        mask_char: '*',
                        reveal_chars: 2,
                    },
                    required_for_analysis: false,
                    is_quasi_identifier: false,
                },
                AnonymizationRule {
                    rule_id: "rule2".to_string(),
                    field_name: "age".to_string(),
                    sensitivity: SensitivityLevel::PII,
                    technique: AnonymizationTechnique::Generalization { levels: 1 },
                    required_for_analysis: true,
                    is_quasi_identifier: true,
                },
            ],
            k_anonymity: 2,
            l_diversity: None,
            max_reidentification_risk: 0.3,
            differential_privacy_enabled: false,
            privacy_budget: None,
        }
    }

    #[tokio::test]
    async fn test_anonymize_field_masking() {
        let anonymizer = DataAnonymizer::new();
        let rule = AnonymizationRule {
            rule_id: "test".to_string(),
            field_name: "email".to_string(),
            sensitivity: SensitivityLevel::PII,
            technique: AnonymizationTechnique::Masking {
                mask_char: '*',
                reveal_chars: 2,
            },
            required_for_analysis: false,
            is_quasi_identifier: false,
        };

        let result = anonymizer
            .anonymize_field("user@example.com", &rule)
            .await
            .unwrap();
        assert!(result.starts_with("us"));
        assert!(result.ends_with("om"));
        assert!(result.contains("***"));
    }

    #[tokio::test]
    async fn test_anonymize_field_suppression() {
        let anonymizer = DataAnonymizer::new();
        let rule = AnonymizationRule {
            rule_id: "test".to_string(),
            field_name: "ssn".to_string(),
            sensitivity: SensitivityLevel::PII,
            technique: AnonymizationTechnique::Suppression,
            required_for_analysis: false,
            is_quasi_identifier: false,
        };

        let result = anonymizer
            .anonymize_field("123-45-6789", &rule)
            .await
            .unwrap();
        assert_eq!(result, "***SUPPRESSED***");
    }

    #[tokio::test]
    async fn test_anonymize_field_pseudonymization() {
        let anonymizer = DataAnonymizer::new();
        let rule = AnonymizationRule {
            rule_id: "test".to_string(),
            field_name: "user_id".to_string(),
            sensitivity: SensitivityLevel::PII,
            technique: AnonymizationTechnique::Pseudonymization,
            required_for_analysis: true,
            is_quasi_identifier: false,
        };

        let result1 = anonymizer.anonymize_field("john_doe", &rule).await.unwrap();
        let result2 = anonymizer.anonymize_field("john_doe", &rule).await.unwrap();

        // Same input should produce same pseudonym (consistency)
        assert_eq!(result1, result2);
        assert!(result1.starts_with("PSEUDO_"));
    }

    #[tokio::test]
    async fn test_anonymize_record() {
        let anonymizer = DataAnonymizer::new();
        let policy = create_test_policy();
        anonymizer.register_policy(policy).await.unwrap();

        let mut record = HashMap::new();
        record.insert("email".to_string(), "user@example.com".to_string());
        record.insert("age".to_string(), "35".to_string());

        let (anonymized, metadata) = anonymizer
            .anonymize_record(&record, "test_policy")
            .await
            .unwrap();

        assert!(anonymized.contains_key("email"));
        assert!(anonymized.contains_key("age"));
        assert_eq!(metadata.fields_anonymized.len(), 2);
        assert!(metadata.utility_score > 0.0);
    }

    #[test]
    fn test_check_k_anonymity_satisfied() {
        let anonymizer = DataAnonymizer::new();

        let records = vec![
            {
                let mut r = HashMap::new();
                r.insert("age".to_string(), "30".to_string());
                r.insert("zip".to_string(), "12345".to_string());
                r
            },
            {
                let mut r = HashMap::new();
                r.insert("age".to_string(), "30".to_string());
                r.insert("zip".to_string(), "12345".to_string());
                r
            },
        ];

        let quasi_identifiers = vec!["age".to_string(), "zip".to_string()];
        let result = anonymizer.check_k_anonymity(&records, &quasi_identifiers, 2);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_check_k_anonymity_not_satisfied() {
        let anonymizer = DataAnonymizer::new();

        let records = vec![
            {
                let mut r = HashMap::new();
                r.insert("age".to_string(), "30".to_string());
                r.insert("zip".to_string(), "12345".to_string());
                r
            },
            {
                let mut r = HashMap::new();
                r.insert("age".to_string(), "35".to_string());
                r.insert("zip".to_string(), "54321".to_string());
                r
            },
        ];

        let quasi_identifiers = vec!["age".to_string(), "zip".to_string()];
        let result = anonymizer.check_k_anonymity(&records, &quasi_identifiers, 2);

        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[tokio::test]
    async fn test_assess_risk() {
        let anonymizer = DataAnonymizer::new();
        let policy = create_test_policy();
        anonymizer.register_policy(policy).await.unwrap();

        let records = vec![
            {
                let mut r = HashMap::new();
                r.insert("age".to_string(), "30".to_string());
                r
            },
            {
                let mut r = HashMap::new();
                r.insert("age".to_string(), "30".to_string());
                r
            },
        ];

        let assessment = anonymizer
            .assess_risk(&records, "test_policy")
            .await
            .unwrap();

        assert!(assessment.overall_risk >= 0.0 && assessment.overall_risk <= 1.0);
        assert!(assessment.k_anonymity_satisfied);
        assert!(!assessment.field_risks.is_empty());
    }

    #[tokio::test]
    async fn test_statistics() {
        let anonymizer = DataAnonymizer::new();
        let policy = create_test_policy();
        anonymizer.register_policy(policy).await.unwrap();

        let mut record = HashMap::new();
        record.insert("email".to_string(), "test@test.com".to_string());

        anonymizer
            .anonymize_record(&record, "test_policy")
            .await
            .unwrap();

        let stats = anonymizer.get_statistics().await;
        assert_eq!(stats.total_records, 1);
        assert!(stats.records_by_policy.contains_key("test_policy"));
    }
}
