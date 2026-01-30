//! # LocationContext - Trait Implementations
//!
//! This module contains trait implementations for `LocationContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for LocationContext {
    fn default() -> Self {
        LocationContext {
            ip_address: None,
            country: None,
            region: None,
            city: None,
            timezone: None,
            isp: None,
            asn: None,
            is_vpn: None,
            is_proxy: None,
        }
    }
}
