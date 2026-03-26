//! # RSession - Trait Implementations
//!
//! This module contains trait implementations for `RSession`.
//!
//! ## Implemented Traits
//!
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::rsession::RSession;
use tracing::debug;

impl Drop for RSession {
    fn drop(&mut self) {
        if self.working_dir.exists() {
            let _ = std::fs::remove_dir_all(&self.working_dir);
        }
        debug!("Cleaned up R session: {}", self.session_id);
    }
}
