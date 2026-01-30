// Copyright (c) 2024 VoiRS Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Utility modules for voice cloning system
//!
//! This module contains helper utilities for safe and efficient operations.

pub mod lock_helpers;

pub use lock_helpers::{recover_from_poison, MutexExt, RwLockExt};
