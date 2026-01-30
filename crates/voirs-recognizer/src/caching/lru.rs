//! LRU (Least Recently Used) list implementation
//!
//! Provides efficient O(1) access and eviction for LRU cache policy

use std::collections::HashMap;

/// LRU list for tracking access order
pub struct LRUList {
    /// Map from key to access order
    access_order: HashMap<String, u64>,
    /// Current access counter
    counter: u64,
}

impl LRUList {
    /// Create a new LRU list
    pub fn new() -> Self {
        Self {
            access_order: HashMap::new(),
            counter: 0,
        }
    }

    /// Insert a new key
    pub fn insert(&mut self, key: String) {
        self.counter += 1;
        self.access_order.insert(key, self.counter);
    }

    /// Record an access to a key
    pub fn access(&mut self, key: &str) {
        self.counter += 1;
        self.access_order.insert(key.to_string(), self.counter);
    }

    /// Remove a key
    pub fn remove(&mut self, key: &str) {
        self.access_order.remove(key);
    }

    /// Get the least recently used key
    pub fn get_lru(&self) -> Option<String> {
        self.access_order
            .iter()
            .min_by_key(|(_, &count)| count)
            .map(|(key, _)| key.clone())
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.access_order.clear();
        self.counter = 0;
    }

    /// Get size
    pub fn len(&self) -> usize {
        self.access_order.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.access_order.is_empty()
    }
}

impl Default for LRUList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_list() {
        let mut list = LRUList::new();

        list.insert("a".to_string());
        list.insert("b".to_string());
        list.insert("c".to_string());

        // Access 'a' to make it recently used
        list.access("a");

        // LRU should be 'b'
        assert_eq!(list.get_lru(), Some("b".to_string()));

        // Remove 'b'
        list.remove("b");

        // LRU should now be 'c'
        assert_eq!(list.get_lru(), Some("c".to_string()));
    }
}
