//! # PooledObject - Trait Implementations
//!
//! This module contains trait implementations for `PooledObject`.
//!
//! ## Implemented Traits
//!
//! - `Drop`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl<T: Send + Sync + 'static> Drop for PooledObject<T> {
    fn drop(&mut self) {
        if let Some(object) = self.object.take() {
            let pool = self.pool.clone();
            let deallocation_count = self.deallocation_count.clone();
            tokio::spawn(async move {
                let mut pool_guard = pool.write().await;
                pool_guard.push_back(object);
                let mut dealloc_count = deallocation_count.write().await;
                *dealloc_count += 1;
            });
        }
    }
}
