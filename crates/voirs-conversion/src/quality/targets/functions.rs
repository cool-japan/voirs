//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::{
    adaptive_controller::{AdaptiveQualityController, QualityTrend, QualityTrigger},
    artifact_detection::{ArtifactDetector, ArtifactType, DetectedArtifacts},
    metrics::{ObjectiveQualityMetrics, QualityMetricsSystem},
};
use crate::{Error, Result};
use std::collections::HashMap;
use tracing::{debug, info};
