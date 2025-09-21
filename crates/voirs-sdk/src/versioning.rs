//! Semantic versioning compliance and API stability guarantees for VoiRS
//!
//! This module provides version management, compatibility checking, and API stability
//! features to ensure reliable long-term support and migration paths.

use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

/// VoiRS version manager for API stability and compatibility
pub struct VersionManager {
    /// Current library version
    current_version: Version,
    /// API compatibility matrix
    compatibility_matrix: CompatibilityMatrix,
    /// Deprecation tracker
    deprecation_tracker: DeprecationTracker,
    /// API stability policies
    stability_policies: StabilityPolicies,
}

impl VersionManager {
    /// Create new version manager
    pub fn new() -> Result<Self, VersioningError> {
        let current_version = Version::parse(env!("CARGO_PKG_VERSION"))?;

        Ok(Self {
            current_version,
            compatibility_matrix: CompatibilityMatrix::new(),
            deprecation_tracker: DeprecationTracker::new(),
            stability_policies: StabilityPolicies::default(),
        })
    }

    /// Get current library version
    pub fn current_version(&self) -> &Version {
        &self.current_version
    }

    /// Check if a version requirement is compatible
    pub fn is_compatible(&self, requirement: &VersionReq) -> bool {
        requirement.matches(&self.current_version)
    }

    /// Check compatibility with specific version
    pub fn check_compatibility(&self, other_version: &Version) -> CompatibilityResult {
        self.compatibility_matrix
            .check_compatibility(&self.current_version, other_version)
    }

    /// Get migration path from one version to another
    pub fn get_migration_path(&self, from: &Version, to: &Version) -> MigrationPath {
        if from == to {
            return MigrationPath {
                from: from.clone(),
                to: to.clone(),
                steps: vec![],
                breaking_changes: vec![],
                recommended_actions: vec![],
                estimated_effort: MigrationEffort::None,
            };
        }

        let mut path = MigrationPath {
            from: from.clone(),
            to: to.clone(),
            steps: vec![],
            breaking_changes: vec![],
            recommended_actions: vec![],
            estimated_effort: MigrationEffort::Low,
        };

        // Analyze version differences
        if to.major > from.major {
            path.breaking_changes.push(BreakingChange {
                change_type: ChangeType::MajorVersionBump,
                description: format!("Major version change from {} to {}", from.major, to.major),
                affected_apis: vec!["All public APIs".to_string()],
                migration_notes: "Review all API usage for breaking changes".to_string(),
                introduced_in: to.clone(),
            });
            path.estimated_effort = MigrationEffort::High;
        }

        if to.minor > from.minor {
            path.steps.push(MigrationStep {
                step_type: StepType::FeatureUpdate,
                description: format!(
                    "Minor version update from {}.{} to {}.{}",
                    from.major, from.minor, to.major, to.minor
                ),
                required: false,
                validation: "Check for new features and deprecations".to_string(),
            });

            if path.estimated_effort == MigrationEffort::None {
                path.estimated_effort = MigrationEffort::Low;
            }
        }

        if to.patch > from.patch {
            path.steps.push(MigrationStep {
                step_type: StepType::BugFix,
                description: "Patch version update - bug fixes and improvements".to_string(),
                required: true,
                validation: "Test existing functionality".to_string(),
            });
        }

        // Add deprecation warnings
        for deprecation in self.deprecation_tracker.get_deprecations_between(from, to) {
            path.recommended_actions.push(format!(
                "Update deprecated API: {} (deprecated in {}, will be removed in {})",
                deprecation.api_name,
                deprecation.deprecated_since,
                deprecation
                    .removal_version
                    .as_ref()
                    .unwrap_or(&"future version".to_string())
            ));
        }

        path
    }

    /// Register API deprecation
    pub fn deprecate_api(
        &mut self,
        api_name: &str,
        reason: &str,
        removal_version: Option<Version>,
    ) {
        let deprecation = DeprecatedApi {
            api_name: api_name.to_string(),
            reason: reason.to_string(),
            deprecated_since: self.current_version.clone(),
            removal_version: removal_version.map(|v| v.to_string()),
            replacement: None,
            migration_guide: None,
        };

        self.deprecation_tracker.add_deprecation(deprecation);
    }

    /// Get all current deprecations
    pub fn get_current_deprecations(&self) -> Vec<&DeprecatedApi> {
        self.deprecation_tracker.get_current_deprecations()
    }

    /// Check if API is deprecated
    pub fn is_api_deprecated(&self, api_name: &str) -> bool {
        self.deprecation_tracker.is_deprecated(api_name)
    }

    /// Validate version bump compatibility
    pub fn validate_version_bump(&self, new_version: &Version) -> VersionBumpValidation {
        let mut validation = VersionBumpValidation {
            is_valid: true,
            issues: vec![],
            warnings: vec![],
            recommendations: vec![],
        };

        // Check semver compliance
        if new_version <= &self.current_version {
            validation.is_valid = false;
            validation.issues.push(format!(
                "New version {} must be greater than current version {}",
                new_version, self.current_version
            ));
        }

        // Validate version increment rules
        let major_diff = new_version.major.saturating_sub(self.current_version.major);
        let minor_diff = new_version.minor.saturating_sub(self.current_version.minor);
        let patch_diff = new_version.patch.saturating_sub(self.current_version.patch);

        if major_diff > 1 {
            validation.warnings.push(
                "Skipping major versions is unusual - consider incremental releases".to_string(),
            );
        }

        if major_diff > 0 && (new_version.minor > 0 || new_version.patch > 0) {
            validation
                .recommendations
                .push("Consider starting major versions with .0.0".to_string());
        }

        if minor_diff > 0 && major_diff == 0 && new_version.patch > 0 {
            validation
                .recommendations
                .push("Consider starting minor versions with .0 patch level".to_string());
        }

        // Check for breaking changes in non-major version bumps
        if major_diff == 0 {
            let pending_breaking_changes = self.get_pending_breaking_changes();
            if !pending_breaking_changes.is_empty() {
                validation.is_valid = false;
                validation.issues.push(
                    "Breaking changes detected but major version not incremented".to_string(),
                );
            }
        }

        validation
    }

    /// Get stability guarantees for current version
    pub fn get_stability_guarantees(&self) -> StabilityGuarantees {
        let api_level = if self.current_version.major == 0 {
            ApiStabilityLevel::Experimental
        } else {
            ApiStabilityLevel::Stable
        };

        StabilityGuarantees {
            api_level,
            public_api_breaking_changes: self.stability_policies.public_api_policy.clone(),
            internal_api_changes: self.stability_policies.internal_api_policy.clone(),
            deprecation_timeline: self.stability_policies.deprecation_timeline.clone(),
            lts_support: self.stability_policies.lts_support.clone(),
        }
    }

    fn get_pending_breaking_changes(&self) -> Vec<String> {
        // In a real implementation, this would check for actual breaking changes
        // For now, return empty as this is a placeholder
        vec![]
    }
}

impl Default for VersionManager {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Compatibility matrix for version checking
#[derive(Debug, Clone)]
pub struct CompatibilityMatrix {
    /// Known incompatible version pairs
    incompatible_pairs: HashMap<String, Vec<String>>,
    /// Compatibility rules
    rules: Vec<CompatibilityRule>,
}

impl CompatibilityMatrix {
    pub fn new() -> Self {
        let mut matrix = Self {
            incompatible_pairs: HashMap::new(),
            rules: vec![
                // Standard semver rules
                CompatibilityRule {
                    name: "major_version_breaking".to_string(),
                    description: "Major version changes indicate breaking changes".to_string(),
                    rule_type: RuleType::Breaking,
                    condition: "major_version_different".to_string(),
                },
                CompatibilityRule {
                    name: "minor_version_additive".to_string(),
                    description: "Minor version changes are backward compatible".to_string(),
                    rule_type: RuleType::Compatible,
                    condition: "minor_version_higher".to_string(),
                },
                CompatibilityRule {
                    name: "patch_version_fixes".to_string(),
                    description: "Patch version changes are fully compatible".to_string(),
                    rule_type: RuleType::Compatible,
                    condition: "patch_version_different".to_string(),
                },
            ],
        };

        // Add known incompatible version pairs (if any)
        matrix
            .incompatible_pairs
            .insert("0.1.0".to_string(), vec!["0.2.0".to_string()]);

        matrix
    }

    pub fn check_compatibility(&self, current: &Version, other: &Version) -> CompatibilityResult {
        if current == other {
            return CompatibilityResult {
                compatible: true,
                level: CompatibilityLevel::Identical,
                issues: vec![],
                recommendations: vec![],
            };
        }

        let mut result = CompatibilityResult {
            compatible: true,
            level: CompatibilityLevel::FullyCompatible,
            issues: vec![],
            recommendations: vec![],
        };

        // Check for known incompatibilities
        if let Some(incompatible_versions) = self.incompatible_pairs.get(&current.to_string()) {
            if incompatible_versions.contains(&other.to_string()) {
                result.compatible = false;
                result.level = CompatibilityLevel::Incompatible;
                result
                    .issues
                    .push("Known incompatible version pair".to_string());
                return result;
            }
        }

        // Apply compatibility rules
        if other.major != current.major {
            result.level = if other.major > current.major {
                CompatibilityLevel::MajorUpgrade
            } else {
                CompatibilityLevel::Incompatible
            };

            if other.major < current.major {
                result.compatible = false;
                result
                    .issues
                    .push("Downgrading major version may cause compatibility issues".to_string());
            } else {
                result
                    .recommendations
                    .push("Review breaking changes before upgrading".to_string());
            }
        } else if other.minor != current.minor {
            result.level = if other.minor > current.minor {
                CompatibilityLevel::MinorUpgrade
            } else {
                CompatibilityLevel::BackwardCompatible
            };

            if other.minor < current.minor {
                result
                    .recommendations
                    .push("Downgrading minor version may remove features".to_string());
            }
        } else if other.patch != current.patch {
            result.level = if other.patch > current.patch {
                CompatibilityLevel::PatchUpgrade
            } else {
                CompatibilityLevel::BackwardCompatible
            };
        }

        result
    }
}

/// Deprecation tracker for API lifecycle management
#[derive(Debug, Clone)]
pub struct DeprecationTracker {
    deprecations: Vec<DeprecatedApi>,
}

impl DeprecationTracker {
    pub fn new() -> Self {
        Self {
            deprecations: vec![
                // Example deprecations - in real usage these would be loaded from configuration
                DeprecatedApi {
                    api_name: "old_synthesis_method".to_string(),
                    reason: "Replaced with more efficient implementation".to_string(),
                    deprecated_since: Version::parse("0.1.0").unwrap(),
                    removal_version: Some("1.0.0".to_string()),
                    replacement: Some("new_synthesis_method".to_string()),
                    migration_guide: Some(
                        "Replace calls to old_synthesis_method() with new_synthesis_method()"
                            .to_string(),
                    ),
                },
            ],
        }
    }

    pub fn add_deprecation(&mut self, deprecation: DeprecatedApi) {
        self.deprecations.push(deprecation);
    }

    pub fn get_current_deprecations(&self) -> Vec<&DeprecatedApi> {
        self.deprecations.iter().collect()
    }

    pub fn is_deprecated(&self, api_name: &str) -> bool {
        self.deprecations.iter().any(|d| d.api_name == api_name)
    }

    pub fn get_deprecations_between(&self, from: &Version, to: &Version) -> Vec<&DeprecatedApi> {
        self.deprecations
            .iter()
            .filter(|d| d.deprecated_since > *from && d.deprecated_since <= *to)
            .collect()
    }
}

/// API stability policies
#[derive(Debug, Clone)]
pub struct StabilityPolicies {
    pub public_api_policy: String,
    pub internal_api_policy: String,
    pub deprecation_timeline: String,
    pub lts_support: String,
}

impl Default for StabilityPolicies {
    fn default() -> Self {
        Self {
            public_api_policy: "Public APIs are stable within major versions".to_string(),
            internal_api_policy: "Internal APIs may change in minor versions".to_string(),
            deprecation_timeline: "APIs deprecated for at least one major version before removal"
                .to_string(),
            lts_support: "LTS versions supported for 2 years".to_string(),
        }
    }
}

// Data structures for version management

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityResult {
    pub compatible: bool,
    pub level: CompatibilityLevel,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompatibilityLevel {
    Identical,
    FullyCompatible,
    BackwardCompatible,
    PatchUpgrade,
    MinorUpgrade,
    MajorUpgrade,
    Incompatible,
}

#[derive(Debug, Clone)]
pub struct CompatibilityRule {
    pub name: String,
    pub description: String,
    pub rule_type: RuleType,
    pub condition: String,
}

#[derive(Debug, Clone)]
pub enum RuleType {
    Compatible,
    Breaking,
    Warning,
}

#[derive(Debug, Clone)]
pub struct MigrationPath {
    pub from: Version,
    pub to: Version,
    pub steps: Vec<MigrationStep>,
    pub breaking_changes: Vec<BreakingChange>,
    pub recommended_actions: Vec<String>,
    pub estimated_effort: MigrationEffort,
}

#[derive(Debug, Clone)]
pub struct MigrationStep {
    pub step_type: StepType,
    pub description: String,
    pub required: bool,
    pub validation: String,
}

#[derive(Debug, Clone)]
pub enum StepType {
    BugFix,
    FeatureUpdate,
    BreakingChange,
    Configuration,
    DataMigration,
}

#[derive(Debug, Clone)]
pub struct BreakingChange {
    pub change_type: ChangeType,
    pub description: String,
    pub affected_apis: Vec<String>,
    pub migration_notes: String,
    pub introduced_in: Version,
}

#[derive(Debug, Clone)]
pub enum ChangeType {
    MajorVersionBump,
    ApiRemoval,
    ApiSignatureChange,
    BehaviorChange,
    ConfigurationChange,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MigrationEffort {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for MigrationEffort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MigrationEffort::None => write!(f, "No migration needed"),
            MigrationEffort::Low => write!(f, "Low effort (< 1 day)"),
            MigrationEffort::Medium => write!(f, "Medium effort (1-3 days)"),
            MigrationEffort::High => write!(f, "High effort (> 3 days)"),
            MigrationEffort::Critical => write!(f, "Critical effort (weeks)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeprecatedApi {
    pub api_name: String,
    pub reason: String,
    pub deprecated_since: Version,
    pub removal_version: Option<String>,
    pub replacement: Option<String>,
    pub migration_guide: Option<String>,
}

#[derive(Debug, Clone)]
pub struct VersionBumpValidation {
    pub is_valid: bool,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StabilityGuarantees {
    pub api_level: ApiStabilityLevel,
    pub public_api_breaking_changes: String,
    pub internal_api_changes: String,
    pub deprecation_timeline: String,
    pub lts_support: String,
}

#[derive(Debug, Clone)]
pub enum ApiStabilityLevel {
    Experimental,
    Beta,
    Stable,
    LTS,
}

impl fmt::Display for ApiStabilityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiStabilityLevel::Experimental => write!(f, "Experimental (no stability guarantees)"),
            ApiStabilityLevel::Beta => write!(f, "Beta (limited stability)"),
            ApiStabilityLevel::Stable => write!(f, "Stable (semver compliant)"),
            ApiStabilityLevel::LTS => write!(f, "Long Term Support"),
        }
    }
}

/// Versioning-related errors
#[derive(Error, Debug)]
pub enum VersioningError {
    #[error("Version parse error: {0}")]
    ParseError(#[from] semver::Error),
    #[error("Compatibility check failed: {0}")]
    CompatibilityError(String),
    #[error("Invalid version bump: {0}")]
    InvalidVersionBump(String),
    #[error("Migration error: {0}")]
    MigrationError(String),
}

/// Utility functions for version management
pub mod utils {
    use super::*;

    /// Check if version is pre-release
    pub fn is_prerelease(version: &Version) -> bool {
        !version.pre.is_empty()
    }

    /// Check if version is stable (>= 1.0.0)
    pub fn is_stable(version: &Version) -> bool {
        version.major >= 1
    }

    /// Get next major version
    pub fn next_major(version: &Version) -> Version {
        Version::new(version.major + 1, 0, 0)
    }

    /// Get next minor version
    pub fn next_minor(version: &Version) -> Version {
        Version::new(version.major, version.minor + 1, 0)
    }

    /// Get next patch version
    pub fn next_patch(version: &Version) -> Version {
        Version::new(version.major, version.minor, version.patch + 1)
    }

    /// Format version with stability indicator
    pub fn format_version_with_stability(version: &Version) -> String {
        if is_prerelease(version) {
            format!("{} (pre-release)", version)
        } else if is_stable(version) {
            format!("{} (stable)", version)
        } else {
            format!("{} (experimental)", version)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_manager_creation() {
        let manager = VersionManager::new();
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert!(!manager.current_version().to_string().is_empty());
    }

    #[test]
    fn test_compatibility_checking() {
        let manager = VersionManager::new().unwrap();

        // Same version should be compatible
        let same_req = VersionReq::parse("=0.1.0").unwrap();
        // Note: This might fail if the actual version is different
        // In a real test, we'd use the actual current version

        // Test with current version
        let current_str = manager.current_version().to_string();
        let current_req = VersionReq::parse(&format!("={}", current_str)).unwrap();
        assert!(manager.is_compatible(&current_req));

        // Compatible range should work - including pre-release versions
        let compatible_req = VersionReq::parse(">=0.1.0-alpha.0, <2.0.0").unwrap();
        assert!(manager.is_compatible(&compatible_req));
    }

    #[test]
    fn test_migration_path() {
        let manager = VersionManager::new().unwrap();

        let from = Version::parse("0.1.0").unwrap();
        let to = Version::parse("0.2.0").unwrap();

        let path = manager.get_migration_path(&from, &to);
        assert_eq!(path.from, from);
        assert_eq!(path.to, to);
        assert!(!path.steps.is_empty() || !path.breaking_changes.is_empty());
    }

    #[test]
    fn test_api_deprecation() {
        let mut manager = VersionManager::new().unwrap();

        manager.deprecate_api("old_function", "Use new_function instead", None);
        assert!(manager.is_api_deprecated("old_function"));
        assert!(!manager.is_api_deprecated("new_function"));

        let deprecations = manager.get_current_deprecations();
        assert!(deprecations.iter().any(|d| d.api_name == "old_function"));
    }

    #[test]
    fn test_version_bump_validation() {
        let manager = VersionManager::new().unwrap();
        let current = manager.current_version();

        // Valid patch bump
        let next_patch = Version::new(current.major, current.minor, current.patch + 1);
        let validation = manager.validate_version_bump(&next_patch);
        assert!(validation.is_valid);

        // Invalid downgrade
        if current.patch > 0 {
            let downgrade = Version::new(current.major, current.minor, current.patch - 1);
            let validation = manager.validate_version_bump(&downgrade);
            assert!(!validation.is_valid);
        }
    }

    #[test]
    fn test_compatibility_matrix() {
        let matrix = CompatibilityMatrix::new();

        let v1 = Version::parse("1.0.0").unwrap();
        let v2 = Version::parse("1.1.0").unwrap();
        let v3 = Version::parse("2.0.0").unwrap();

        // Minor version upgrade should be compatible
        let result = matrix.check_compatibility(&v1, &v2);
        assert!(result.compatible);
        assert_eq!(result.level, CompatibilityLevel::MinorUpgrade);

        // Major version upgrade should be flagged
        let result = matrix.check_compatibility(&v1, &v3);
        // Should still be marked as compatible but with major upgrade level
        assert_eq!(result.level, CompatibilityLevel::MajorUpgrade);
    }

    #[test]
    fn test_stability_guarantees() {
        let manager = VersionManager::new().unwrap();
        let guarantees = manager.get_stability_guarantees();

        // Verify stability level based on version
        if manager.current_version().major == 0 {
            assert!(matches!(
                guarantees.api_level,
                ApiStabilityLevel::Experimental
            ));
        } else {
            assert!(matches!(guarantees.api_level, ApiStabilityLevel::Stable));
        }
    }

    #[test]
    fn test_utility_functions() {
        use super::utils::*;

        let stable_version = Version::parse("1.2.3").unwrap();
        let experimental_version = Version::parse("0.1.0").unwrap();
        let prerelease_version = Version::parse("1.0.0-alpha.1").unwrap();

        assert!(is_stable(&stable_version));
        assert!(!is_stable(&experimental_version));
        assert!(is_prerelease(&prerelease_version));

        assert_eq!(
            next_major(&stable_version),
            Version::parse("2.0.0").unwrap()
        );
        assert_eq!(
            next_minor(&stable_version),
            Version::parse("1.3.0").unwrap()
        );
        assert_eq!(
            next_patch(&stable_version),
            Version::parse("1.2.4").unwrap()
        );
    }
}
