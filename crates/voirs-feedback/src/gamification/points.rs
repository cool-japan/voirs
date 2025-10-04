//! Point system with multi-currency support and marketplace
//!
//! This module provides a comprehensive point system including:
//! - Multi-currency point economy (Experience, Achievement, Social, Premium, Event, Skill points)
//! - Point marketplace for rewards and upgrades
//! - Point transfer and exchange mechanisms
//! - Bonus events and multipliers
//! - Point history and analytics

use crate::traits::{FocusArea, UserProgress};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Point system manager
#[derive(Debug, Clone)]
pub struct PointSystem {
    /// User point balances
    user_balances: HashMap<Uuid, PointBalance>,
    /// Transaction history
    transactions: HashMap<Uuid, Vec<PointTransaction>>,
    /// Marketplace items
    marketplace_items: HashMap<Uuid, MarketplaceItem>,
    /// Active bonus events
    bonus_events: Vec<BonusEvent>,
    /// Exchange rates between currencies
    exchange_rates: HashMap<(PointCurrency, PointCurrency), f32>,
}

impl PointSystem {
    /// Create a new point system
    pub fn new() -> Self {
        let mut system = Self {
            user_balances: HashMap::new(),
            transactions: HashMap::new(),
            marketplace_items: HashMap::new(),
            bonus_events: Vec::new(),
            exchange_rates: HashMap::new(),
        };

        system.initialize_marketplace();
        system.setup_exchange_rates();
        system
    }

    /// Initialize marketplace with default items
    fn initialize_marketplace(&mut self) {
        let items = vec![
            MarketplaceItem {
                id: Uuid::new_v4(),
                name: "Extra Practice Session".to_string(),
                description: "Unlock an additional practice session for today".to_string(),
                category: ItemCategory::PowerUp,
                cost: vec![PointCost {
                    currency: PointCurrency::Experience,
                    amount: 50,
                }],
                benefits: vec![Benefit::ExtraSession],
                availability: ItemAvailability::Always,
                purchase_limit: Some(3),
                icon_url: None,
            },
            MarketplaceItem {
                id: Uuid::new_v4(),
                name: "Pronunciation Booster".to_string(),
                description: "Get detailed pronunciation feedback for 24 hours".to_string(),
                category: ItemCategory::Enhancement,
                cost: vec![PointCost {
                    currency: PointCurrency::Skill,
                    amount: 100,
                }],
                benefits: vec![Benefit::EnhancedFeedback],
                availability: ItemAvailability::Always,
                purchase_limit: None,
                icon_url: None,
            },
            MarketplaceItem {
                id: Uuid::new_v4(),
                name: "Theme Pack: Neon".to_string(),
                description: "Unlock the stylish neon theme for your interface".to_string(),
                category: ItemCategory::Cosmetic,
                cost: vec![PointCost {
                    currency: PointCurrency::Achievement,
                    amount: 200,
                }],
                benefits: vec![Benefit::Theme("neon".to_string())],
                availability: ItemAvailability::Always,
                purchase_limit: Some(1),
                icon_url: None,
            },
            MarketplaceItem {
                id: Uuid::new_v4(),
                name: "Streak Saver".to_string(),
                description: "Protect your streak from one missed day".to_string(),
                category: ItemCategory::Insurance,
                cost: vec![PointCost {
                    currency: PointCurrency::Premium,
                    amount: 25,
                }],
                benefits: vec![Benefit::StreakProtection],
                availability: ItemAvailability::Always,
                purchase_limit: Some(5),
                icon_url: None,
            },
            MarketplaceItem {
                id: Uuid::new_v4(),
                name: "Exclusive Badge: Golden Voice".to_string(),
                description: "Show off with this limited-edition golden voice badge".to_string(),
                category: ItemCategory::Collectible,
                cost: vec![
                    PointCost {
                        currency: PointCurrency::Achievement,
                        amount: 500,
                    },
                    PointCost {
                        currency: PointCurrency::Social,
                        amount: 300,
                    },
                ],
                benefits: vec![Benefit::Badge("golden_voice".to_string())],
                availability: ItemAvailability::Limited { remaining: 100 },
                purchase_limit: Some(1),
                icon_url: None,
            },
        ];

        for item in items {
            self.marketplace_items.insert(item.id, item);
        }
    }

    /// Setup exchange rates between currencies
    fn setup_exchange_rates(&mut self) {
        // Experience can be converted to other currencies at lower rates
        self.exchange_rates
            .insert((PointCurrency::Experience, PointCurrency::Achievement), 0.5);
        self.exchange_rates
            .insert((PointCurrency::Experience, PointCurrency::Social), 0.7);
        self.exchange_rates
            .insert((PointCurrency::Experience, PointCurrency::Skill), 0.8);

        // Achievement points are valuable
        self.exchange_rates
            .insert((PointCurrency::Achievement, PointCurrency::Experience), 2.0);
        self.exchange_rates
            .insert((PointCurrency::Achievement, PointCurrency::Social), 1.5);

        // Social points facilitate community engagement
        self.exchange_rates
            .insert((PointCurrency::Social, PointCurrency::Experience), 1.3);
        self.exchange_rates
            .insert((PointCurrency::Social, PointCurrency::Skill), 1.1);

        // Premium points cannot be earned, only purchased
        // Event points are temporary and cannot be exchanged
    }

    /// Award points to user
    pub fn award_points(
        &mut self,
        user_id: Uuid,
        currency: PointCurrency,
        amount: u32,
        reason: String,
    ) -> PointTransaction {
        let multiplier = self.get_active_multiplier(currency);
        let final_amount = (amount as f32 * multiplier) as u32;

        // Update balance
        let balance = self
            .user_balances
            .entry(user_id)
            .or_insert_with(PointBalance::default);
        match currency {
            PointCurrency::Experience => balance.experience += final_amount,
            PointCurrency::Achievement => balance.achievement += final_amount,
            PointCurrency::Social => balance.social += final_amount,
            PointCurrency::Premium => balance.premium += final_amount,
            PointCurrency::Event => balance.event += final_amount,
            PointCurrency::Skill => balance.skill += final_amount,
        }

        // Record transaction
        let transaction = PointTransaction {
            id: Uuid::new_v4(),
            user_id,
            transaction_type: TransactionType::Award,
            currency,
            amount: final_amount,
            reason,
            timestamp: Utc::now(),
            related_item: None,
        };

        self.transactions
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(transaction.clone());

        transaction
    }

    /// Spend points
    pub fn spend_points(
        &mut self,
        user_id: Uuid,
        currency: PointCurrency,
        amount: u32,
        reason: String,
    ) -> Result<PointTransaction, String> {
        let balance = self
            .user_balances
            .get_mut(&user_id)
            .ok_or("User not found")?;

        let current_amount = match currency {
            PointCurrency::Experience => balance.experience,
            PointCurrency::Achievement => balance.achievement,
            PointCurrency::Social => balance.social,
            PointCurrency::Premium => balance.premium,
            PointCurrency::Event => balance.event,
            PointCurrency::Skill => balance.skill,
        };

        if current_amount < amount {
            return Err("Insufficient points".to_string());
        }

        // Deduct points
        match currency {
            PointCurrency::Experience => balance.experience -= amount,
            PointCurrency::Achievement => balance.achievement -= amount,
            PointCurrency::Social => balance.social -= amount,
            PointCurrency::Premium => balance.premium -= amount,
            PointCurrency::Event => balance.event -= amount,
            PointCurrency::Skill => balance.skill -= amount,
        }

        // Record transaction
        let transaction = PointTransaction {
            id: Uuid::new_v4(),
            user_id,
            transaction_type: TransactionType::Spend,
            currency,
            amount,
            reason,
            timestamp: Utc::now(),
            related_item: None,
        };

        self.transactions
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(transaction.clone());

        Ok(transaction)
    }

    /// Exchange points between currencies
    pub fn exchange_points(
        &mut self,
        user_id: Uuid,
        from_currency: PointCurrency,
        to_currency: PointCurrency,
        amount: u32,
    ) -> Result<PointTransaction, String> {
        if from_currency == to_currency {
            return Err("Cannot exchange same currency".to_string());
        }

        // Check if exchange is allowed
        let exchange_rate = self
            .exchange_rates
            .get(&(from_currency, to_currency))
            .ok_or("Exchange not supported")?;

        // Calculate conversion
        let received_amount = (amount as f32 * exchange_rate) as u32;

        // Spend from source currency
        self.spend_points(
            user_id,
            from_currency,
            amount,
            format!("Exchange to {:?}", to_currency),
        )?;

        // Award to target currency
        let transaction = self.award_points(
            user_id,
            to_currency,
            received_amount,
            format!("Exchange from {:?}", from_currency),
        );

        Ok(transaction)
    }

    /// Purchase marketplace item
    pub fn purchase_item(
        &mut self,
        user_id: Uuid,
        item_id: Uuid,
    ) -> Result<PurchaseResult, String> {
        let item = self
            .marketplace_items
            .get(&item_id)
            .ok_or("Item not found")?
            .clone();

        // Check availability
        match &item.availability {
            ItemAvailability::Limited { remaining } => {
                if *remaining == 0 {
                    return Err("Item sold out".to_string());
                }
            }
            ItemAvailability::TimeLimited { expires_at } => {
                if Utc::now() > *expires_at {
                    return Err("Item expired".to_string());
                }
            }
            ItemAvailability::Always => {}
        }

        // Check purchase limit
        if let Some(limit) = item.purchase_limit {
            let user_purchases = self.get_user_purchase_count(user_id, item_id);
            if user_purchases >= limit {
                return Err("Purchase limit reached".to_string());
            }
        }

        // Check if user has enough points
        let balance = self.user_balances.get(&user_id).ok_or("User not found")?;
        for cost in &item.cost {
            let current_amount = match cost.currency {
                PointCurrency::Experience => balance.experience,
                PointCurrency::Achievement => balance.achievement,
                PointCurrency::Social => balance.social,
                PointCurrency::Premium => balance.premium,
                PointCurrency::Event => balance.event,
                PointCurrency::Skill => balance.skill,
            };

            if current_amount < cost.amount {
                return Err(format!("Insufficient {:?} points", cost.currency));
            }
        }

        // Deduct points
        let mut transactions = Vec::new();
        for cost in &item.cost {
            let transaction = self.spend_points(
                user_id,
                cost.currency,
                cost.amount,
                format!("Purchase: {}", item.name),
            )?;
            transactions.push(transaction);
        }

        // Update item availability
        if let Some(marketplace_item) = self.marketplace_items.get_mut(&item_id) {
            if let ItemAvailability::Limited { remaining } = &mut marketplace_item.availability {
                *remaining -= 1;
            }
        }

        Ok(PurchaseResult {
            item,
            transactions,
            purchased_at: Utc::now(),
        })
    }

    /// Transfer points between users
    pub fn transfer_points(
        &mut self,
        from_user: Uuid,
        to_user: Uuid,
        currency: PointCurrency,
        amount: u32,
        message: Option<String>,
    ) -> Result<(PointTransaction, PointTransaction), String> {
        // Check if transfers are allowed for this currency
        match currency {
            PointCurrency::Premium | PointCurrency::Event => {
                return Err("Cannot transfer this currency".to_string());
            }
            _ => {}
        }

        // Check if sender has enough points
        let sender_balance = self
            .user_balances
            .get(&from_user)
            .ok_or("Sender not found")?;
        let current_amount = match currency {
            PointCurrency::Experience => sender_balance.experience,
            PointCurrency::Achievement => sender_balance.achievement,
            PointCurrency::Social => sender_balance.social,
            PointCurrency::Premium => sender_balance.premium,
            PointCurrency::Event => sender_balance.event,
            PointCurrency::Skill => sender_balance.skill,
        };

        if current_amount < amount {
            return Err("Insufficient points to transfer".to_string());
        }

        // Perform transfer
        let transfer_message = message.unwrap_or_else(|| "Point transfer".to_string());

        let debit_transaction = self.spend_points(
            from_user,
            currency,
            amount,
            format!("Transfer to {}: {}", to_user, transfer_message),
        )?;

        let credit_transaction = self.award_points(
            to_user,
            currency,
            amount,
            format!("Transfer from {}: {}", from_user, transfer_message),
        );

        Ok((debit_transaction, credit_transaction))
    }

    /// Start bonus event
    pub fn start_bonus_event(&mut self, event: BonusEvent) {
        self.bonus_events.push(event);
    }

    /// Get user's point balance
    pub fn get_balance(&self, user_id: Uuid) -> PointBalance {
        self.user_balances
            .get(&user_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get user's transaction history
    pub fn get_transaction_history(
        &self,
        user_id: Uuid,
        limit: Option<usize>,
    ) -> Vec<&PointTransaction> {
        self.transactions
            .get(&user_id)
            .map(|transactions| {
                let mut sorted_transactions: Vec<&PointTransaction> = transactions.iter().collect();
                sorted_transactions.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
                if let Some(limit) = limit {
                    sorted_transactions.into_iter().take(limit).collect()
                } else {
                    sorted_transactions
                }
            })
            .unwrap_or_default()
    }

    /// Get marketplace items
    pub fn get_marketplace_items(&self, category: Option<ItemCategory>) -> Vec<&MarketplaceItem> {
        self.marketplace_items
            .values()
            .filter(|item| category.is_none() || category.as_ref() == Some(&item.category))
            .collect()
    }

    /// Get point earning statistics
    pub fn get_earning_stats(&self, user_id: Uuid) -> PointEarningStats {
        let empty_vec = Vec::new();
        let transactions = self.transactions.get(&user_id).unwrap_or(&empty_vec);

        let mut stats = PointEarningStats {
            total_earned: HashMap::new(),
            total_spent: HashMap::new(),
            earning_rate: HashMap::new(),
            top_earning_activities: Vec::new(),
        };

        // Calculate totals
        for transaction in transactions {
            match transaction.transaction_type {
                TransactionType::Award => {
                    *stats.total_earned.entry(transaction.currency).or_insert(0) +=
                        transaction.amount;
                }
                TransactionType::Spend => {
                    *stats.total_spent.entry(transaction.currency).or_insert(0) +=
                        transaction.amount;
                }
            }
        }

        // Calculate earning rates based on actual activity period
        for (currency, total) in &stats.total_earned {
            let days_active = self.calculate_days_active(user_id, *currency, transactions);
            let earning_rate = if days_active > 0.0 {
                *total as f32 / days_active
            } else {
                0.0
            };
            stats.earning_rate.insert(*currency, earning_rate);
        }

        stats
    }

    /// Helper methods
    fn get_active_multiplier(&self, currency: PointCurrency) -> f32 {
        for event in &self.bonus_events {
            if event.is_active() && event.affected_currencies.contains(&currency) {
                return event.multiplier;
            }
        }
        1.0
    }

    /// Calculate actual days active for a user and currency
    fn calculate_days_active(
        &self,
        _user_id: Uuid,
        currency: PointCurrency,
        transactions: &[PointTransaction],
    ) -> f32 {
        let currency_transactions: Vec<&PointTransaction> = transactions
            .iter()
            .filter(|t| {
                t.currency == currency && matches!(t.transaction_type, TransactionType::Award)
            })
            .collect();

        if currency_transactions.is_empty() {
            return 0.0;
        }

        // Find the earliest and latest award transactions for this currency
        let earliest = currency_transactions
            .iter()
            .map(|t| t.timestamp)
            .min()
            .unwrap();

        let latest = currency_transactions
            .iter()
            .map(|t| t.timestamp)
            .max()
            .unwrap();

        // Calculate the span of days
        let duration = latest.signed_duration_since(earliest);
        let days_span = duration.num_days() as f32;

        // If all transactions happened on the same day, return 1 day
        // Otherwise, add 1 to include both the first and last day
        if days_span <= 0.0 {
            1.0
        } else {
            days_span + 1.0
        }
    }

    fn get_user_purchase_count(&self, user_id: Uuid, item_id: Uuid) -> u32 {
        self.transactions
            .get(&user_id)
            .map(|transactions| {
                transactions
                    .iter()
                    .filter(|t| t.related_item == Some(item_id))
                    .count() as u32
            })
            .unwrap_or(0)
    }
}

impl Default for PointSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Point currencies in the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PointCurrency {
    /// General experience points
    Experience,
    /// Achievement-specific points
    Achievement,
    /// Social interaction points
    Social,
    /// Premium currency (purchased)
    Premium,
    /// Event-specific points
    Event,
    /// Skill-specific points
    Skill,
}

/// User's point balance across all currencies
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PointBalance {
    /// Experience points
    pub experience: u32,
    /// Achievement points
    pub achievement: u32,
    /// Social points
    pub social: u32,
    /// Premium points
    pub premium: u32,
    /// Event points
    pub event: u32,
    /// Skill points
    pub skill: u32,
}

impl PointBalance {
    /// Get balance for specific currency
    pub fn get_currency_balance(&self, currency: PointCurrency) -> u32 {
        match currency {
            PointCurrency::Experience => self.experience,
            PointCurrency::Achievement => self.achievement,
            PointCurrency::Social => self.social,
            PointCurrency::Premium => self.premium,
            PointCurrency::Event => self.event,
            PointCurrency::Skill => self.skill,
        }
    }

    /// Get total points across all currencies
    pub fn get_total_points(&self) -> u32 {
        self.experience + self.achievement + self.social + self.premium + self.event + self.skill
    }
}

/// Point transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointTransaction {
    /// Transaction ID
    pub id: Uuid,
    /// User ID
    pub user_id: Uuid,
    /// Transaction type
    pub transaction_type: TransactionType,
    /// Currency
    pub currency: PointCurrency,
    /// Amount
    pub amount: u32,
    /// Reason for transaction
    pub reason: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Related marketplace item (if applicable)
    pub related_item: Option<Uuid>,
}

/// Transaction types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransactionType {
    /// Description
    Award,
    /// Description
    Spend,
}

/// Marketplace item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceItem {
    /// Item ID
    pub id: Uuid,
    /// Item name
    pub name: String,
    /// Item description
    pub description: String,
    /// Item category
    pub category: ItemCategory,
    /// Cost in various currencies
    pub cost: Vec<PointCost>,
    /// Benefits provided
    pub benefits: Vec<Benefit>,
    /// Availability status
    pub availability: ItemAvailability,
    /// Purchase limit per user
    pub purchase_limit: Option<u32>,
    /// Icon URL
    pub icon_url: Option<String>,
}

/// Point cost for marketplace items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCost {
    /// Currency type
    pub currency: PointCurrency,
    /// Amount required
    pub amount: u32,
}

/// Item categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ItemCategory {
    /// Description
    PowerUp,
    /// Description
    Enhancement,
    /// Description
    Cosmetic,
    /// Description
    Insurance,
    /// Description
    Collectible,
}

/// Benefits provided by items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Benefit {
    /// Description
    ExtraSession,
    /// Description
    EnhancedFeedback,
    /// Description
    Theme(String),
    /// Description
    StreakProtection,
    /// Description
    Badge(String),
    /// Description
    DoublePoints,
    /// Description
    SkipCooldown,
}

/// Item availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ItemAvailability {
    /// Description
    Always,
    /// Description
    /// Description
    Limited { remaining: u32 },
    /// Description
    /// Description
    TimeLimited { expires_at: DateTime<Utc> },
}

/// Purchase result
#[derive(Debug, Clone)]
pub struct PurchaseResult {
    /// Purchased item
    pub item: MarketplaceItem,
    /// Transaction records
    pub transactions: Vec<PointTransaction>,
    /// Purchase timestamp
    pub purchased_at: DateTime<Utc>,
}

/// Bonus event for point multipliers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonusEvent {
    /// Event ID
    pub id: Uuid,
    /// Event name
    pub name: String,
    /// Event description
    pub description: String,
    /// Point multiplier
    pub multiplier: f32,
    /// Affected currencies
    pub affected_currencies: Vec<PointCurrency>,
    /// Start time
    pub starts_at: DateTime<Utc>,
    /// End time
    pub ends_at: DateTime<Utc>,
    /// Event conditions
    pub conditions: Option<EventConditions>,
}

impl BonusEvent {
    /// Check if event is currently active
    pub fn is_active(&self) -> bool {
        let now = Utc::now();
        now >= self.starts_at && now <= self.ends_at
    }
}

/// Event conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventConditions {
    /// Minimum sessions required
    pub min_sessions: Option<u32>,
    /// Required focus areas
    pub required_focus_areas: Vec<FocusArea>,
    /// Minimum accuracy threshold
    pub min_accuracy: Option<f32>,
}

/// Point earning statistics
#[derive(Debug, Clone)]
pub struct PointEarningStats {
    /// Total points earned by currency
    pub total_earned: HashMap<PointCurrency, u32>,
    /// Total points spent by currency
    pub total_spent: HashMap<PointCurrency, u32>,
    /// Average earning rate per day
    pub earning_rate: HashMap<PointCurrency, f32>,
    /// Top earning activities
    pub top_earning_activities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_system_creation() {
        let system = PointSystem::new();
        assert!(!system.marketplace_items.is_empty());
        assert!(!system.exchange_rates.is_empty());
    }

    #[test]
    fn test_point_awarding() {
        let mut system = PointSystem::new();
        let user_id = Uuid::new_v4();

        let transaction = system.award_points(
            user_id,
            PointCurrency::Experience,
            100,
            "Test session completion".to_string(),
        );

        assert_eq!(transaction.amount, 100);
        assert_eq!(transaction.currency, PointCurrency::Experience);

        let balance = system.get_balance(user_id);
        assert_eq!(balance.experience, 100);
    }

    #[test]
    fn test_point_spending() {
        let mut system = PointSystem::new();
        let user_id = Uuid::new_v4();

        // First award some points
        system.award_points(user_id, PointCurrency::Experience, 100, "Setup".to_string());

        // Then spend some
        let result = system.spend_points(
            user_id,
            PointCurrency::Experience,
            50,
            "Test purchase".to_string(),
        );
        assert!(result.is_ok());

        let balance = system.get_balance(user_id);
        assert_eq!(balance.experience, 50);
    }

    #[test]
    fn test_insufficient_points() {
        let mut system = PointSystem::new();
        let user_id = Uuid::new_v4();

        // Try to spend points without having any
        let result =
            system.spend_points(user_id, PointCurrency::Experience, 50, "Test".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_point_exchange() {
        let mut system = PointSystem::new();
        let user_id = Uuid::new_v4();

        // Award experience points
        system.award_points(user_id, PointCurrency::Experience, 100, "Setup".to_string());

        // Exchange to achievement points
        let result = system.exchange_points(
            user_id,
            PointCurrency::Experience,
            PointCurrency::Achievement,
            100,
        );

        assert!(result.is_ok());

        let balance = system.get_balance(user_id);
        assert_eq!(balance.experience, 0);
        assert_eq!(balance.achievement, 50); // 100 * 0.5 exchange rate
    }

    #[test]
    fn test_marketplace_purchase() {
        let mut system = PointSystem::new();
        let user_id = Uuid::new_v4();

        // Award enough points
        system.award_points(user_id, PointCurrency::Experience, 100, "Setup".to_string());

        // Find an item to purchase
        let items = system.get_marketplace_items(None);
        let item = items.iter().find(|item| {
            item.cost
                .iter()
                .any(|cost| cost.currency == PointCurrency::Experience && cost.amount <= 100)
        });

        if let Some(item) = item {
            let result = system.purchase_item(user_id, item.id);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_point_transfer() {
        let mut system = PointSystem::new();
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();

        // Award points to user1
        system.award_points(user1, PointCurrency::Social, 100, "Setup".to_string());

        // Transfer to user2
        let result = system.transfer_points(
            user1,
            user2,
            PointCurrency::Social,
            50,
            Some("Gift".to_string()),
        );

        assert!(result.is_ok());

        let balance1 = system.get_balance(user1);
        let balance2 = system.get_balance(user2);

        assert_eq!(balance1.social, 50);
        assert_eq!(balance2.social, 50);
    }

    #[test]
    fn test_bonus_event() {
        let mut system = PointSystem::new();
        let user_id = Uuid::new_v4();

        // Create bonus event
        let event = BonusEvent {
            id: Uuid::new_v4(),
            name: "Double XP Weekend".to_string(),
            description: "Earn double experience points".to_string(),
            multiplier: 2.0,
            affected_currencies: vec![PointCurrency::Experience],
            starts_at: Utc::now() - chrono::Duration::hours(1),
            ends_at: Utc::now() + chrono::Duration::hours(1),
            conditions: None,
        };

        system.start_bonus_event(event);

        // Award points during event
        let transaction =
            system.award_points(user_id, PointCurrency::Experience, 100, "Test".to_string());

        assert_eq!(transaction.amount, 200); // 100 * 2.0 multiplier

        let balance = system.get_balance(user_id);
        assert_eq!(balance.experience, 200);
    }
}
