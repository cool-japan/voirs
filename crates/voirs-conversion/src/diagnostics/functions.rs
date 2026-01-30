//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use crate::{
    config::ConversionConfig,
    quality::ArtifactDetector,
    types::{
        ConversionRequest, ConversionResult, ConversionType, DetectedArtifacts,
        ObjectiveQualityMetrics, VoiceCharacteristics,
    },
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, trace, warn};
