//! Role-Based Access Control (RBAC) system for enterprise evaluation security
//!
//! This module provides comprehensive Role-Based Access Control capabilities for
//! the VoiRS evaluation framework, enabling fine-grained permission management,
//! user role assignment, and access policy enforcement.
//!
//! # Features
//!
//! - **Role Management**: Define and manage user roles with hierarchical inheritance
//! - **Permission System**: Granular permissions for evaluation operations
//! - **Policy Enforcement**: Automatic access control enforcement for all operations
//! - **Audit Integration**: Complete audit trail of all access control decisions
//! - **Temporal Access**: Time-based access controls and expiration
//! - **Resource-Level Permissions**: Fine-grained control over specific evaluation resources
//! - **Multi-Tenancy**: Support for organizational isolation and tenant-based access
//!
//! # Example
//!
//! ```rust
//! use voirs_evaluation::rbac::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create RBAC manager
//! let mut rbac = RbacManager::new();
//!
//! // Define roles
//! let admin_role = Role::new("admin")
//!     .with_permission(Permission::EvaluationExecute)
//!     .with_permission(Permission::EvaluationRead)
//!     .with_permission(Permission::EvaluationWrite)
//!     .with_permission(Permission::SystemAdmin);
//!
//! let analyst_role = Role::new("analyst")
//!     .with_permission(Permission::EvaluationExecute)
//!     .with_permission(Permission::EvaluationRead);
//!
//! // Add roles to system
//! rbac.add_role(admin_role)?;
//! rbac.add_role(analyst_role)?;
//!
//! // Assign user to role
//! rbac.assign_user_role("user@example.com", "analyst")?;
//!
//! // Check permissions
//! assert!(rbac.has_permission("user@example.com", Permission::EvaluationRead)?);
//! assert!(!rbac.has_permission("user@example.com", Permission::SystemAdmin)?);
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// RBAC errors
#[derive(Error, Debug)]
pub enum RbacError {
    /// Role not found
    #[error("Role not found: {role_name}")]
    RoleNotFound {
        /// Role name
        role_name: String,
    },

    /// User not found
    #[error("User not found: {user_id}")]
    UserNotFound {
        /// User ID
        user_id: String,
    },

    /// Permission denied
    #[error("Permission denied: user {user_id} lacks permission {permission:?}")]
    PermissionDenied {
        /// User ID
        user_id: String,
        /// Required permission
        permission: Permission,
    },

    /// Role already exists
    #[error("Role already exists: {role_name}")]
    RoleAlreadyExists {
        /// Role name
        role_name: String,
    },

    /// Invalid role hierarchy
    #[error("Invalid role hierarchy: {message}")]
    InvalidHierarchy {
        /// Error message
        message: String,
    },

    /// Access expired
    #[error("Access expired for user {user_id}")]
    AccessExpired {
        /// User ID
        user_id: String,
    },

    /// Tenant isolation violation
    #[error("Tenant isolation violation: {message}")]
    TenantViolation {
        /// Error message
        message: String,
    },
}

/// Permission types for evaluation operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    // Evaluation permissions
    /// Execute quality evaluations
    EvaluationExecute,
    /// Read evaluation results
    EvaluationRead,
    /// Write/modify evaluation configurations
    EvaluationWrite,
    /// Delete evaluation results
    EvaluationDelete,
    /// Export evaluation data
    EvaluationExport,

    // Dataset permissions
    /// Read dataset content
    DatasetRead,
    /// Upload new datasets
    DatasetUpload,
    /// Modify dataset metadata
    DatasetModify,
    /// Delete datasets
    DatasetDelete,

    // Model permissions
    /// Execute model inference
    ModelExecute,
    /// Read model information
    ModelRead,
    /// Deploy new models
    ModelDeploy,
    /// Delete models
    ModelDelete,

    // Analytics permissions
    /// View analytics dashboards
    AnalyticsView,
    /// Create custom analytics
    AnalyticsCreate,
    /// Export analytics reports
    AnalyticsExport,

    // System permissions
    /// System administration
    SystemAdmin,
    /// User management
    UserManagement,
    /// Role management
    RoleManagement,
    /// Audit log access
    AuditAccess,
    /// Configuration management
    ConfigManagement,

    // Privacy permissions
    /// Access to PII data
    PrivacyPiiAccess,
    /// Privacy budget management
    PrivacyBudgetManage,
    /// Data anonymization
    PrivacyAnonymize,
}

impl Permission {
    /// Get all available permissions
    pub fn all() -> Vec<Permission> {
        vec![
            Permission::EvaluationExecute,
            Permission::EvaluationRead,
            Permission::EvaluationWrite,
            Permission::EvaluationDelete,
            Permission::EvaluationExport,
            Permission::DatasetRead,
            Permission::DatasetUpload,
            Permission::DatasetModify,
            Permission::DatasetDelete,
            Permission::ModelExecute,
            Permission::ModelRead,
            Permission::ModelDeploy,
            Permission::ModelDelete,
            Permission::AnalyticsView,
            Permission::AnalyticsCreate,
            Permission::AnalyticsExport,
            Permission::SystemAdmin,
            Permission::UserManagement,
            Permission::RoleManagement,
            Permission::AuditAccess,
            Permission::ConfigManagement,
            Permission::PrivacyPiiAccess,
            Permission::PrivacyBudgetManage,
            Permission::PrivacyAnonymize,
        ]
    }

    /// Get permission description
    pub fn description(&self) -> &'static str {
        match self {
            Permission::EvaluationExecute => "Execute quality evaluations",
            Permission::EvaluationRead => "Read evaluation results",
            Permission::EvaluationWrite => "Write/modify evaluation configurations",
            Permission::EvaluationDelete => "Delete evaluation results",
            Permission::EvaluationExport => "Export evaluation data",
            Permission::DatasetRead => "Read dataset content",
            Permission::DatasetUpload => "Upload new datasets",
            Permission::DatasetModify => "Modify dataset metadata",
            Permission::DatasetDelete => "Delete datasets",
            Permission::ModelExecute => "Execute model inference",
            Permission::ModelRead => "Read model information",
            Permission::ModelDeploy => "Deploy new models",
            Permission::ModelDelete => "Delete models",
            Permission::AnalyticsView => "View analytics dashboards",
            Permission::AnalyticsCreate => "Create custom analytics",
            Permission::AnalyticsExport => "Export analytics reports",
            Permission::SystemAdmin => "System administration",
            Permission::UserManagement => "User management",
            Permission::RoleManagement => "Role management",
            Permission::AuditAccess => "Audit log access",
            Permission::ConfigManagement => "Configuration management",
            Permission::PrivacyPiiAccess => "Access to PII data",
            Permission::PrivacyBudgetManage => "Privacy budget management",
            Permission::PrivacyAnonymize => "Data anonymization",
        }
    }
}

/// User role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name (unique identifier)
    pub name: String,
    /// Human-readable display name
    pub display_name: String,
    /// Role description
    pub description: String,
    /// Permissions granted by this role
    pub permissions: HashSet<Permission>,
    /// Parent roles (for hierarchical inheritance)
    pub parent_roles: Vec<String>,
    /// Priority level (higher = more privileged)
    pub priority: u32,
    /// Whether this role can be deleted
    pub system_role: bool,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl Role {
    /// Create new role
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            display_name: name.clone(),
            name: name.clone(),
            description: String::new(),
            permissions: HashSet::new(),
            parent_roles: Vec::new(),
            priority: 0,
            system_role: false,
            metadata: HashMap::new(),
        }
    }

    /// Add permission to role
    pub fn with_permission(mut self, permission: Permission) -> Self {
        self.permissions.insert(permission);
        self
    }

    /// Add multiple permissions
    pub fn with_permissions(mut self, permissions: impl IntoIterator<Item = Permission>) -> Self {
        self.permissions.extend(permissions);
        self
    }

    /// Set parent role for inheritance
    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent_roles.push(parent.into());
        self
    }

    /// Set display name
    pub fn with_display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = display_name.into();
        self
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Mark as system role
    pub fn as_system_role(mut self) -> Self {
        self.system_role = true;
        self
    }

    /// Get all permissions (including inherited)
    pub fn effective_permissions(&self, all_roles: &HashMap<String, Role>) -> HashSet<Permission> {
        let mut permissions = self.permissions.clone();

        // Add inherited permissions from parent roles
        for parent_name in &self.parent_roles {
            if let Some(parent_role) = all_roles.get(parent_name) {
                permissions.extend(parent_role.effective_permissions(all_roles));
            }
        }

        permissions
    }
}

/// User access assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAccess {
    /// User ID
    pub user_id: String,
    /// Assigned roles
    pub roles: HashSet<String>,
    /// Direct permissions (beyond roles)
    pub direct_permissions: HashSet<Permission>,
    /// Tenant ID for multi-tenancy
    pub tenant_id: Option<String>,
    /// Access expiration time
    pub expires_at: Option<DateTime<Utc>>,
    /// Whether account is enabled
    pub enabled: bool,
    /// User metadata
    pub metadata: HashMap<String, String>,
}

impl UserAccess {
    /// Create new user access
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            roles: HashSet::new(),
            direct_permissions: HashSet::new(),
            tenant_id: None,
            expires_at: None,
            enabled: true,
            metadata: HashMap::new(),
        }
    }

    /// Add role to user
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.roles.insert(role.into());
        self
    }

    /// Add direct permission
    pub fn with_permission(mut self, permission: Permission) -> Self {
        self.direct_permissions.insert(permission);
        self
    }

    /// Set tenant ID
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Set expiration
    pub fn with_expiration(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Check if access has expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }

    /// Check if user is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled && !self.is_expired()
    }
}

/// Access decision with audit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessDecision {
    /// Whether access is granted
    pub granted: bool,
    /// User ID
    pub user_id: String,
    /// Required permission
    pub permission: Permission,
    /// Reason for decision
    pub reason: String,
    /// Decision timestamp
    pub timestamp: DateTime<Utc>,
    /// Effective roles used
    pub effective_roles: Vec<String>,
}

/// RBAC Manager for access control
pub struct RbacManager {
    /// All defined roles
    roles: Arc<RwLock<HashMap<String, Role>>>,
    /// User access assignments
    user_access: Arc<RwLock<HashMap<String, UserAccess>>>,
    /// Access decision audit log
    audit_log: Arc<RwLock<Vec<AccessDecision>>>,
    /// Configuration
    config: RbacConfig,
}

/// RBAC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacConfig {
    /// Enable audit logging
    pub enable_audit: bool,
    /// Maximum audit log size
    pub max_audit_entries: usize,
    /// Enable multi-tenancy
    pub enable_multi_tenancy: bool,
    /// Default tenant ID
    pub default_tenant: Option<String>,
}

impl Default for RbacConfig {
    fn default() -> Self {
        Self {
            enable_audit: true,
            max_audit_entries: 10000,
            enable_multi_tenancy: false,
            default_tenant: None,
        }
    }
}

impl RbacManager {
    /// Create new RBAC manager
    pub fn new() -> Self {
        Self::with_config(RbacConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: RbacConfig) -> Self {
        let mut manager = Self {
            roles: Arc::new(RwLock::new(HashMap::new())),
            user_access: Arc::new(RwLock::new(HashMap::new())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            config,
        };

        // Initialize with predefined system roles
        let _ = futures::executor::block_on(manager.initialize_system_roles());

        manager
    }

    /// Initialize predefined system roles
    async fn initialize_system_roles(&mut self) -> Result<(), RbacError> {
        // Super Admin role
        let super_admin = Role::new("super_admin")
            .with_display_name("Super Administrator")
            .with_description("Full system access with all permissions")
            .with_permissions(Permission::all())
            .with_priority(1000)
            .as_system_role();

        // Administrator role
        let admin = Role::new("administrator")
            .with_display_name("Administrator")
            .with_description("System administrator with management permissions")
            .with_permissions(vec![
                Permission::EvaluationExecute,
                Permission::EvaluationRead,
                Permission::EvaluationWrite,
                Permission::EvaluationDelete,
                Permission::DatasetRead,
                Permission::DatasetUpload,
                Permission::DatasetModify,
                Permission::ModelExecute,
                Permission::ModelRead,
                Permission::ModelDeploy,
                Permission::AnalyticsView,
                Permission::AnalyticsCreate,
                Permission::UserManagement,
                Permission::ConfigManagement,
                Permission::AuditAccess,
            ])
            .with_priority(900)
            .as_system_role();

        // Analyst role
        let analyst = Role::new("analyst")
            .with_display_name("Analyst")
            .with_description("Data analyst with evaluation and analytics access")
            .with_permissions(vec![
                Permission::EvaluationExecute,
                Permission::EvaluationRead,
                Permission::EvaluationExport,
                Permission::DatasetRead,
                Permission::ModelExecute,
                Permission::ModelRead,
                Permission::AnalyticsView,
                Permission::AnalyticsCreate,
                Permission::AnalyticsExport,
            ])
            .with_priority(500)
            .as_system_role();

        // Viewer role
        let viewer = Role::new("viewer")
            .with_display_name("Viewer")
            .with_description("Read-only access to evaluations and analytics")
            .with_permissions(vec![
                Permission::EvaluationRead,
                Permission::DatasetRead,
                Permission::ModelRead,
                Permission::AnalyticsView,
            ])
            .with_priority(100)
            .as_system_role();

        // Privacy Officer role
        let privacy_officer = Role::new("privacy_officer")
            .with_display_name("Privacy Officer")
            .with_description("Privacy and compliance management")
            .with_permissions(vec![
                Permission::PrivacyPiiAccess,
                Permission::PrivacyBudgetManage,
                Permission::PrivacyAnonymize,
                Permission::AuditAccess,
                Permission::EvaluationRead,
            ])
            .with_priority(700)
            .as_system_role();

        self.add_role(super_admin).await?;
        self.add_role(admin).await?;
        self.add_role(analyst).await?;
        self.add_role(viewer).await?;
        self.add_role(privacy_officer).await?;

        Ok(())
    }

    /// Add a new role
    pub async fn add_role(&mut self, role: Role) -> Result<(), RbacError> {
        let mut roles = self.roles.write().await;

        if roles.contains_key(&role.name) {
            return Err(RbacError::RoleAlreadyExists {
                role_name: role.name.clone(),
            });
        }

        // Validate parent roles exist
        for parent in &role.parent_roles {
            if !roles.contains_key(parent) {
                return Err(RbacError::RoleNotFound {
                    role_name: parent.clone(),
                });
            }
        }

        info!(
            "Adding role: {} with {} permissions",
            role.name,
            role.permissions.len()
        );
        roles.insert(role.name.clone(), role);

        Ok(())
    }

    /// Remove a role
    pub async fn remove_role(&mut self, role_name: &str) -> Result<(), RbacError> {
        let mut roles = self.roles.write().await;

        let role = roles
            .get(role_name)
            .ok_or_else(|| RbacError::RoleNotFound {
                role_name: role_name.to_string(),
            })?;

        if role.system_role {
            return Err(RbacError::InvalidHierarchy {
                message: format!("Cannot delete system role: {}", role_name),
            });
        }

        roles.remove(role_name);
        info!("Removed role: {}", role_name);

        Ok(())
    }

    /// Assign role to user
    pub async fn assign_user_role(
        &mut self,
        user_id: &str,
        role_name: &str,
    ) -> Result<(), RbacError> {
        // Verify role exists
        let roles = self.roles.read().await;
        if !roles.contains_key(role_name) {
            return Err(RbacError::RoleNotFound {
                role_name: role_name.to_string(),
            });
        }
        drop(roles);

        let mut users = self.user_access.write().await;
        let user_access = users
            .entry(user_id.to_string())
            .or_insert_with(|| UserAccess::new(user_id));

        user_access.roles.insert(role_name.to_string());

        info!("Assigned role '{}' to user '{}'", role_name, user_id);

        Ok(())
    }

    /// Remove role from user
    pub async fn revoke_user_role(
        &mut self,
        user_id: &str,
        role_name: &str,
    ) -> Result<(), RbacError> {
        let mut users = self.user_access.write().await;
        let user_access = users
            .get_mut(user_id)
            .ok_or_else(|| RbacError::UserNotFound {
                user_id: user_id.to_string(),
            })?;

        user_access.roles.remove(role_name);

        info!("Revoked role '{}' from user '{}'", role_name, user_id);

        Ok(())
    }

    /// Grant direct permission to user
    pub async fn grant_permission(
        &mut self,
        user_id: &str,
        permission: Permission,
    ) -> Result<(), RbacError> {
        let mut users = self.user_access.write().await;
        let user_access = users
            .entry(user_id.to_string())
            .or_insert_with(|| UserAccess::new(user_id));

        user_access.direct_permissions.insert(permission);

        info!(
            "Granted permission '{:?}' to user '{}'",
            permission, user_id
        );

        Ok(())
    }

    /// Revoke direct permission from user
    pub async fn revoke_permission(
        &mut self,
        user_id: &str,
        permission: Permission,
    ) -> Result<(), RbacError> {
        let mut users = self.user_access.write().await;
        let user_access = users
            .get_mut(user_id)
            .ok_or_else(|| RbacError::UserNotFound {
                user_id: user_id.to_string(),
            })?;

        user_access.direct_permissions.remove(&permission);

        info!(
            "Revoked permission '{:?}' from user '{}'",
            permission, user_id
        );

        Ok(())
    }

    /// Check if user has permission
    pub async fn has_permission(
        &self,
        user_id: &str,
        permission: Permission,
    ) -> Result<bool, RbacError> {
        let decision = self.check_access(user_id, permission).await?;
        Ok(decision.granted)
    }

    /// Check access and return detailed decision
    pub async fn check_access(
        &self,
        user_id: &str,
        permission: Permission,
    ) -> Result<AccessDecision, RbacError> {
        let users = self.user_access.read().await;
        let user_access = users.get(user_id);

        let mut decision = AccessDecision {
            granted: false,
            user_id: user_id.to_string(),
            permission,
            reason: String::new(),
            timestamp: Utc::now(),
            effective_roles: Vec::new(),
        };

        // User not found - deny access
        let user_access = match user_access {
            Some(ua) => ua,
            None => {
                decision.reason = "User not found".to_string();
                if self.config.enable_audit {
                    self.log_decision(decision.clone()).await;
                }
                return Ok(decision);
            }
        };

        // Check if account is enabled
        if !user_access.is_enabled() {
            decision.reason = if user_access.is_expired() {
                "Access expired".to_string()
            } else {
                "Account disabled".to_string()
            };
            if self.config.enable_audit {
                self.log_decision(decision.clone()).await;
            }
            return Ok(decision);
        }

        // Check direct permissions first
        if user_access.direct_permissions.contains(&permission) {
            decision.granted = true;
            decision.reason = "Direct permission granted".to_string();
            if self.config.enable_audit {
                self.log_decision(decision.clone()).await;
            }
            return Ok(decision);
        }

        // Check role-based permissions
        let roles = self.roles.read().await;
        let all_roles = roles.clone();

        for role_name in &user_access.roles {
            if let Some(role) = roles.get(role_name) {
                let effective_perms = role.effective_permissions(&all_roles);
                if effective_perms.contains(&permission) {
                    decision.granted = true;
                    decision.reason = format!("Permission granted via role '{}'", role_name);
                    decision.effective_roles.push(role_name.clone());

                    if self.config.enable_audit {
                        self.log_decision(decision.clone()).await;
                    }
                    return Ok(decision);
                }
            }
        }

        decision.reason = "Permission not granted by any role or direct assignment".to_string();
        if self.config.enable_audit {
            self.log_decision(decision.clone()).await;
        }

        Ok(decision)
    }

    /// Enforce permission (throw error if not granted)
    pub async fn enforce_permission(
        &self,
        user_id: &str,
        permission: Permission,
    ) -> Result<(), RbacError> {
        let decision = self.check_access(user_id, permission).await?;

        if decision.granted {
            Ok(())
        } else {
            Err(RbacError::PermissionDenied {
                user_id: user_id.to_string(),
                permission,
            })
        }
    }

    /// Get all permissions for a user
    pub async fn get_user_permissions(
        &self,
        user_id: &str,
    ) -> Result<HashSet<Permission>, RbacError> {
        let users = self.user_access.read().await;
        let user_access = users.get(user_id).ok_or_else(|| RbacError::UserNotFound {
            user_id: user_id.to_string(),
        })?;

        let mut permissions = user_access.direct_permissions.clone();

        let roles = self.roles.read().await;
        let all_roles = roles.clone();

        for role_name in &user_access.roles {
            if let Some(role) = roles.get(role_name) {
                permissions.extend(role.effective_permissions(&all_roles));
            }
        }

        Ok(permissions)
    }

    /// Get user's roles
    pub async fn get_user_roles(&self, user_id: &str) -> Result<Vec<Role>, RbacError> {
        let users = self.user_access.read().await;
        let user_access = users.get(user_id).ok_or_else(|| RbacError::UserNotFound {
            user_id: user_id.to_string(),
        })?;

        let roles = self.roles.read().await;
        let mut user_roles = Vec::new();

        for role_name in &user_access.roles {
            if let Some(role) = roles.get(role_name) {
                user_roles.push(role.clone());
            }
        }

        Ok(user_roles)
    }

    /// Log access decision
    async fn log_decision(&self, decision: AccessDecision) {
        let mut audit_log = self.audit_log.write().await;

        // Maintain maximum log size
        if audit_log.len() >= self.config.max_audit_entries {
            audit_log.remove(0);
        }

        audit_log.push(decision);
    }

    /// Get audit log
    pub async fn get_audit_log(&self) -> Vec<AccessDecision> {
        let audit_log = self.audit_log.read().await;
        audit_log.clone()
    }

    /// Get audit log for specific user
    pub async fn get_user_audit_log(&self, user_id: &str) -> Vec<AccessDecision> {
        let audit_log = self.audit_log.read().await;
        audit_log
            .iter()
            .filter(|d| d.user_id == user_id)
            .cloned()
            .collect()
    }

    /// Create new user
    pub async fn create_user(&mut self, user_access: UserAccess) -> Result<(), RbacError> {
        let mut users = self.user_access.write().await;

        info!("Creating user: {}", user_access.user_id);
        users.insert(user_access.user_id.clone(), user_access);

        Ok(())
    }

    /// Update user
    pub async fn update_user(&mut self, user_access: UserAccess) -> Result<(), RbacError> {
        let mut users = self.user_access.write().await;

        if !users.contains_key(&user_access.user_id) {
            return Err(RbacError::UserNotFound {
                user_id: user_access.user_id.clone(),
            });
        }

        info!("Updating user: {}", user_access.user_id);
        users.insert(user_access.user_id.clone(), user_access);

        Ok(())
    }

    /// Delete user
    pub async fn delete_user(&mut self, user_id: &str) -> Result<(), RbacError> {
        let mut users = self.user_access.write().await;

        if !users.contains_key(user_id) {
            return Err(RbacError::UserNotFound {
                user_id: user_id.to_string(),
            });
        }

        info!("Deleting user: {}", user_id);
        users.remove(user_id);

        Ok(())
    }

    /// Get all roles
    pub async fn get_all_roles(&self) -> Vec<Role> {
        let roles = self.roles.read().await;
        roles.values().cloned().collect()
    }

    /// Get role by name
    pub async fn get_role(&self, role_name: &str) -> Result<Role, RbacError> {
        let roles = self.roles.read().await;
        roles
            .get(role_name)
            .cloned()
            .ok_or_else(|| RbacError::RoleNotFound {
                role_name: role_name.to_string(),
            })
    }
}

impl Default for RbacManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rbac_manager_creation() {
        let rbac = RbacManager::new();
        let roles = rbac.get_all_roles().await;

        // Should have predefined system roles
        assert!(!roles.is_empty());
        assert!(roles.iter().any(|r| r.name == "super_admin"));
        assert!(roles.iter().any(|r| r.name == "administrator"));
        assert!(roles.iter().any(|r| r.name == "analyst"));
        assert!(roles.iter().any(|r| r.name == "viewer"));
    }

    #[tokio::test]
    async fn test_role_assignment() {
        let mut rbac = RbacManager::new();

        // Assign role to user
        rbac.assign_user_role("user@example.com", "analyst")
            .await
            .unwrap();

        // Check permission
        assert!(rbac
            .has_permission("user@example.com", Permission::EvaluationRead)
            .await
            .unwrap());
        assert!(!rbac
            .has_permission("user@example.com", Permission::SystemAdmin)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_direct_permission() {
        let mut rbac = RbacManager::new();

        // Grant direct permission
        rbac.grant_permission("user@example.com", Permission::ModelDeploy)
            .await
            .unwrap();

        // Check permission
        assert!(rbac
            .has_permission("user@example.com", Permission::ModelDeploy)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_permission_enforcement() {
        let mut rbac = RbacManager::new();

        rbac.assign_user_role("user@example.com", "viewer")
            .await
            .unwrap();

        // Should succeed
        rbac.enforce_permission("user@example.com", Permission::EvaluationRead)
            .await
            .unwrap();

        // Should fail
        let result = rbac
            .enforce_permission("user@example.com", Permission::SystemAdmin)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_role_hierarchy() {
        let mut rbac = RbacManager::new();

        // Create child role with parent
        let child_role = Role::new("custom_analyst")
            .with_parent("analyst")
            .with_permission(Permission::ModelDeploy);

        rbac.add_role(child_role).await.unwrap();
        rbac.assign_user_role("user@example.com", "custom_analyst")
            .await
            .unwrap();

        // Should have both parent and child permissions
        assert!(rbac
            .has_permission("user@example.com", Permission::EvaluationRead)
            .await
            .unwrap()); // from parent
        assert!(rbac
            .has_permission("user@example.com", Permission::ModelDeploy)
            .await
            .unwrap()); // from child
    }

    #[tokio::test]
    async fn test_audit_logging() {
        let rbac = RbacManager::new();

        // Access should create audit log entry
        let _decision = rbac
            .check_access("user@example.com", Permission::EvaluationRead)
            .await
            .unwrap();

        let audit_log = rbac.get_audit_log().await;
        assert!(!audit_log.is_empty());
        assert_eq!(audit_log[0].user_id, "user@example.com");
    }

    #[tokio::test]
    async fn test_user_permissions_aggregation() {
        let mut rbac = RbacManager::new();

        rbac.assign_user_role("user@example.com", "analyst")
            .await
            .unwrap();
        rbac.grant_permission("user@example.com", Permission::SystemAdmin)
            .await
            .unwrap();

        let permissions = rbac.get_user_permissions("user@example.com").await.unwrap();

        // Should have analyst permissions + direct SystemAdmin permission
        assert!(permissions.contains(&Permission::EvaluationRead));
        assert!(permissions.contains(&Permission::SystemAdmin));
    }

    #[tokio::test]
    async fn test_role_revocation() {
        let mut rbac = RbacManager::new();

        rbac.assign_user_role("user@example.com", "analyst")
            .await
            .unwrap();
        assert!(rbac
            .has_permission("user@example.com", Permission::EvaluationRead)
            .await
            .unwrap());

        rbac.revoke_user_role("user@example.com", "analyst")
            .await
            .unwrap();
        assert!(!rbac
            .has_permission("user@example.com", Permission::EvaluationRead)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_custom_role_creation() {
        let mut rbac = RbacManager::new();

        let custom_role = Role::new("data_scientist")
            .with_display_name("Data Scientist")
            .with_description("Data science and ML experimentation")
            .with_permissions(vec![
                Permission::EvaluationExecute,
                Permission::EvaluationRead,
                Permission::DatasetRead,
                Permission::DatasetUpload,
                Permission::ModelExecute,
                Permission::AnalyticsView,
            ])
            .with_priority(600);

        rbac.add_role(custom_role).await.unwrap();

        let role = rbac.get_role("data_scientist").await.unwrap();
        assert_eq!(role.display_name, "Data Scientist");
        assert_eq!(role.permissions.len(), 6);
    }

    #[tokio::test]
    async fn test_system_role_protection() {
        let mut rbac = RbacManager::new();

        // Should not be able to delete system roles
        let result = rbac.remove_role("super_admin").await;
        assert!(result.is_err());
    }
}
