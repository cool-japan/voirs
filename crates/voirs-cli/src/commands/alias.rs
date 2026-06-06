//! Command alias management system.
//!
//! This module allows users to create shortcuts for frequently used commands,
//! improving productivity and reducing typing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use voirs_sdk::Result;

/// Alias definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alias {
    /// Alias name
    pub name: String,
    /// Command to execute
    pub command: String,
    /// Description of what this alias does
    pub description: Option<String>,
}

/// Alias manager
pub struct AliasManager {
    /// Path to aliases file
    aliases_path: PathBuf,
}

impl AliasManager {
    /// Create a new alias manager
    pub fn new() -> Result<Self> {
        let aliases_path = Self::get_aliases_path()?;
        Ok(Self { aliases_path })
    }

    /// Get the path to the aliases file
    fn get_aliases_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir().ok_or_else(|| {
            voirs_sdk::VoirsError::config_error("Could not find config directory")
        })?;

        let voirs_dir = config_dir.join("voirs");
        fs::create_dir_all(&voirs_dir).map_err(|e| voirs_sdk::VoirsError::IoError {
            path: voirs_dir.clone(),
            operation: voirs_sdk::error::IoOperation::Write,
            source: e,
        })?;

        Ok(voirs_dir.join("aliases.json"))
    }

    /// Load aliases from file
    pub fn load(&self) -> Result<HashMap<String, Alias>> {
        if !self.aliases_path.exists() {
            return Ok(HashMap::new());
        }

        let content =
            fs::read_to_string(&self.aliases_path).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: self.aliases_path.clone(),
                operation: voirs_sdk::error::IoOperation::Read,
                source: e,
            })?;

        let aliases: HashMap<String, Alias> = serde_json::from_str(&content).unwrap_or_default();
        Ok(aliases)
    }

    /// Save aliases to file
    fn save(&self, aliases: &HashMap<String, Alias>) -> Result<()> {
        let content = serde_json::to_string_pretty(aliases).map_err(|e| {
            voirs_sdk::VoirsError::config_error(format!("Failed to serialize aliases: {}", e))
        })?;

        fs::write(&self.aliases_path, content).map_err(|e| voirs_sdk::VoirsError::IoError {
            path: self.aliases_path.clone(),
            operation: voirs_sdk::error::IoOperation::Write,
            source: e,
        })?;

        Ok(())
    }

    /// Add or update an alias
    pub fn add_alias(
        &self,
        name: String,
        command: String,
        description: Option<String>,
    ) -> Result<()> {
        // Validate alias name (alphanumeric and hyphens only)
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(voirs_sdk::VoirsError::config_error(
                "Alias name must contain only alphanumeric characters, hyphens, and underscores",
            ));
        }

        // Prevent shadowing built-in commands
        if is_builtin_command(&name) {
            return Err(voirs_sdk::VoirsError::config_error(format!(
                "Cannot create alias '{}': this is a built-in command",
                name
            )));
        }

        let mut aliases = self.load()?;
        aliases.insert(
            name.clone(),
            Alias {
                name,
                command,
                description,
            },
        );
        self.save(&aliases)?;

        Ok(())
    }

    /// Remove an alias
    pub fn remove_alias(&self, name: &str) -> Result<bool> {
        let mut aliases = self.load()?;
        let removed = aliases.remove(name).is_some();
        if removed {
            self.save(&aliases)?;
        }
        Ok(removed)
    }

    /// Get an alias by name
    pub fn get_alias(&self, name: &str) -> Result<Option<Alias>> {
        let aliases = self.load()?;
        Ok(aliases.get(name).cloned())
    }

    /// List all aliases
    pub fn list_aliases(&self) -> Result<Vec<Alias>> {
        let aliases = self.load()?;
        let mut alias_list: Vec<Alias> = aliases.into_values().collect();
        alias_list.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(alias_list)
    }

    /// Clear all aliases
    pub fn clear(&self) -> Result<()> {
        if self.aliases_path.exists() {
            fs::remove_file(&self.aliases_path).map_err(|e| voirs_sdk::VoirsError::IoError {
                path: self.aliases_path.clone(),
                operation: voirs_sdk::error::IoOperation::Delete,
                source: e,
            })?;
        }
        Ok(())
    }
}

/// Check if a name is a built-in command
fn is_builtin_command(name: &str) -> bool {
    matches!(
        name,
        "synthesize"
            | "synthesize-file"
            | "list-voices"
            | "voice-info"
            | "download-voice"
            | "preview-voice"
            | "compare-voices"
            | "test"
            | "interactive"
            | "batch"
            | "server"
            | "train"
            | "config"
            | "capabilities"
            | "history"
            | "alias"
            | "export"
            | "import"
            | "help"
            | "version"
    )
}

/// Run alias command
pub async fn run_alias(subcommand: AliasSubcommand) -> Result<()> {
    let manager = AliasManager::new()?;

    match subcommand {
        AliasSubcommand::Add {
            name,
            command,
            description,
        } => {
            manager.add_alias(name.clone(), command.clone(), description)?;
            println!("✅ Alias '{}' created", name);
            println!("   Command: voirs {}", command);
            Ok(())
        }

        AliasSubcommand::Remove { name } => {
            if manager.remove_alias(&name)? {
                println!("✅ Alias '{}' removed", name);
            } else {
                println!("⚠️  Alias '{}' not found", name);
            }
            Ok(())
        }

        AliasSubcommand::List => {
            let aliases = manager.list_aliases()?;
            if aliases.is_empty() {
                println!("No aliases defined yet.");
                println!();
                println!("Create an alias with:");
                println!("  voirs alias add <name> <command>");
            } else {
                println!("📋 Defined aliases:");
                println!();
                for alias in aliases {
                    println!("  {} → voirs {}", alias.name, alias.command);
                    if let Some(desc) = alias.description {
                        println!("      {}", desc);
                    }
                }
            }
            Ok(())
        }

        AliasSubcommand::Show { name } => {
            if let Some(alias) = manager.get_alias(&name)? {
                println!("Alias: {}", alias.name);
                println!("Command: voirs {}", alias.command);
                if let Some(desc) = alias.description {
                    println!("Description: {}", desc);
                }
            } else {
                println!("❌ Alias '{}' not found", name);
            }
            Ok(())
        }

        AliasSubcommand::Clear => {
            manager.clear()?;
            println!("✅ All aliases cleared");
            Ok(())
        }
    }
}

/// Alias subcommands
#[derive(Debug, Clone)]
pub enum AliasSubcommand {
    /// Add a new alias
    Add {
        /// Alias name
        name: String,
        /// Command to execute
        command: String,
        /// Optional description
        description: Option<String>,
    },
    /// Remove an alias
    Remove {
        /// Alias name to remove
        name: String,
    },
    /// List all aliases
    List,
    /// Show details of a specific alias
    Show {
        /// Alias name to show
        name: String,
    },
    /// Clear all aliases
    Clear,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alias_creation() {
        let alias = Alias {
            name: "quick".to_string(),
            command: "synthesize --quality medium".to_string(),
            description: Some("Quick synthesis".to_string()),
        };

        assert_eq!(alias.name, "quick");
        assert_eq!(alias.command, "synthesize --quality medium");
    }

    #[test]
    fn test_builtin_command_check() {
        assert!(is_builtin_command("synthesize"));
        assert!(is_builtin_command("list-voices"));
        assert!(!is_builtin_command("my-custom-alias"));
    }

    #[test]
    fn test_alias_serialization() {
        let mut aliases = HashMap::new();
        aliases.insert(
            "quick".to_string(),
            Alias {
                name: "quick".to_string(),
                command: "synthesize --quality medium".to_string(),
                description: None,
            },
        );

        let json = serde_json::to_string(&aliases).unwrap();
        let deserialized: HashMap<String, Alias> = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.len(), 1);
        assert!(deserialized.contains_key("quick"));
    }
}
