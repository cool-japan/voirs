// Copyright (c) 2024 VoiRS Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lock Helper Utilities for Safe Lock Operations
//!
//! This module provides safe wrappers for RwLock and Mutex operations that return
//! proper Result types instead of panicking on lock errors.

use crate::Error;
use std::sync::{Arc, LockResult, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::{Mutex, MutexGuard};

/// Extension trait for safe RwLock operations
pub trait RwLockExt<T> {
    /// Safely acquire a read lock, returning a Result instead of panicking
    fn safe_read(&self) -> Result<RwLockReadGuard<'_, T>, Error>;

    /// Safely acquire a write lock, returning a Result instead of panicking
    fn safe_write(&self) -> Result<RwLockWriteGuard<'_, T>, Error>;

    /// Try to acquire a read lock without blocking
    fn try_safe_read(&self) -> Result<RwLockReadGuard<'_, T>, Error>;

    /// Try to acquire a write lock without blocking
    fn try_safe_write(&self) -> Result<RwLockWriteGuard<'_, T>, Error>;
}

/// Extension trait for safe Mutex operations
pub trait MutexExt<T> {
    /// Safely acquire a mutex lock, returning a Result instead of panicking
    fn safe_lock(&self) -> Result<MutexGuard<'_, T>, Error>;

    /// Try to acquire a mutex lock without blocking
    fn try_safe_lock(&self) -> Result<MutexGuard<'_, T>, Error>;
}

impl<T> RwLockExt<T> for RwLock<T> {
    fn safe_read(&self) -> Result<RwLockReadGuard<'_, T>, Error> {
        self.read()
            .map_err(|e| Error::LockError(format!("Failed to acquire read lock: {}", e)))
    }

    fn safe_write(&self) -> Result<RwLockWriteGuard<'_, T>, Error> {
        self.write()
            .map_err(|e| Error::LockError(format!("Failed to acquire write lock: {}", e)))
    }

    fn try_safe_read(&self) -> Result<RwLockReadGuard<'_, T>, Error> {
        self.try_read().map_err(|e| match e {
            std::sync::TryLockError::Poisoned(p) => {
                Error::LockError(format!("Read lock poisoned: {}", p))
            }
            std::sync::TryLockError::WouldBlock => {
                Error::LockError("Read lock would block".to_string())
            }
        })
    }

    fn try_safe_write(&self) -> Result<RwLockWriteGuard<'_, T>, Error> {
        self.try_write().map_err(|e| match e {
            std::sync::TryLockError::Poisoned(p) => {
                Error::LockError(format!("Write lock poisoned: {}", p))
            }
            std::sync::TryLockError::WouldBlock => {
                Error::LockError("Write lock would block".to_string())
            }
        })
    }
}

impl<T> MutexExt<T> for Mutex<T> {
    fn safe_lock(&self) -> Result<MutexGuard<'_, T>, Error> {
        self.lock()
            .map_err(|e| Error::LockError(format!("Failed to acquire mutex lock: {}", e)))
    }

    fn try_safe_lock(&self) -> Result<MutexGuard<'_, T>, Error> {
        self.try_lock().map_err(|e| match e {
            std::sync::TryLockError::Poisoned(p) => {
                Error::LockError(format!("Mutex lock poisoned: {}", p))
            }
            std::sync::TryLockError::WouldBlock => {
                Error::LockError("Mutex lock would block".to_string())
            }
        })
    }
}

impl<T> RwLockExt<T> for Arc<RwLock<T>> {
    fn safe_read(&self) -> Result<RwLockReadGuard<'_, T>, Error> {
        self.as_ref().safe_read()
    }

    fn safe_write(&self) -> Result<RwLockWriteGuard<'_, T>, Error> {
        self.as_ref().safe_write()
    }

    fn try_safe_read(&self) -> Result<RwLockReadGuard<'_, T>, Error> {
        self.as_ref().try_safe_read()
    }

    fn try_safe_write(&self) -> Result<RwLockWriteGuard<'_, T>, Error> {
        self.as_ref().try_safe_write()
    }
}

impl<T> MutexExt<T> for Arc<Mutex<T>> {
    fn safe_lock(&self) -> Result<MutexGuard<'_, T>, Error> {
        self.as_ref().safe_lock()
    }

    fn try_safe_lock(&self) -> Result<MutexGuard<'_, T>, Error> {
        self.as_ref().try_safe_lock()
    }
}

/// Helper function to handle poisoned lock recovery
pub fn recover_from_poison<T, F>(result: LockResult<T>, recovery_fn: F) -> Result<T, Error>
where
    F: FnOnce(&PoisonError<T>) -> Result<T, Error>,
{
    result.or_else(|poison_error| {
        tracing::warn!("Lock was poisoned, attempting recovery");
        recovery_fn(&poison_error)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_rwlock_safe_read() {
        let lock = RwLock::new(42);
        let guard = lock.safe_read().unwrap();
        assert_eq!(*guard, 42);
    }

    #[test]
    fn test_rwlock_safe_write() {
        let lock = RwLock::new(42);
        {
            let mut guard = lock.safe_write().unwrap();
            *guard = 100;
        }
        let guard = lock.safe_read().unwrap();
        assert_eq!(*guard, 100);
    }

    #[test]
    fn test_mutex_safe_lock() {
        let mutex = Mutex::new(vec![1, 2, 3]);
        let guard = mutex.safe_lock().unwrap();
        assert_eq!(guard.len(), 3);
    }

    #[test]
    fn test_arc_rwlock_safe_operations() {
        let lock = Arc::new(RwLock::new(String::from("test")));
        {
            let guard = lock.safe_read().unwrap();
            assert_eq!(&*guard, "test");
        }
        {
            let mut guard = lock.safe_write().unwrap();
            *guard = String::from("modified");
        }
        let guard = lock.safe_read().unwrap();
        assert_eq!(&*guard, "modified");
    }

    #[test]
    fn test_arc_mutex_safe_lock() {
        let mutex = Arc::new(Mutex::new(0));
        let mut guard = mutex.safe_lock().unwrap();
        *guard += 1;
        assert_eq!(*guard, 1);
    }

    #[test]
    fn test_try_safe_read() {
        let lock = RwLock::new(42);
        let _write_guard = lock.write().expect("lock should not be poisoned"); // Hold write lock

        // This should fail because write lock is held
        let result = lock.try_safe_read();
        assert!(result.is_err());
        if let Err(Error::LockError(msg)) = result {
            assert!(msg.contains("would block") || msg.contains("Poisoned"));
        }
    }

    #[test]
    fn test_try_safe_write() {
        let lock = RwLock::new(42);
        let _read_guard = lock.read().expect("lock should not be poisoned"); // Hold read lock

        // This should fail because read lock is held
        let result = lock.try_safe_write();
        assert!(result.is_err());
        if let Err(Error::LockError(msg)) = result {
            assert!(msg.contains("would block") || msg.contains("Poisoned"));
        }
    }
}
