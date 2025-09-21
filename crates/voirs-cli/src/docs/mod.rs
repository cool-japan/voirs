//! Documentation generation for VoiRS CLI
//!
//! This module provides functionality for generating comprehensive documentation
//! including man pages, help text, and user guides.

pub mod man;

use crate::error::Result;

/// Documentation generator for VoiRS CLI
pub struct DocumentationGenerator {
    /// The command name for documentation
    command_name: String,
    /// Version information
    version: String,
}

impl DocumentationGenerator {
    /// Create a new documentation generator
    pub fn new(command_name: String, version: String) -> Self {
        Self {
            command_name,
            version,
        }
    }

    /// Generate all documentation formats
    pub fn generate_all(&self, output_dir: &std::path::Path) -> Result<()> {
        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Generate man pages
        self.generate_man_pages(output_dir)?;

        Ok(())
    }

    /// Generate man pages for all commands
    pub fn generate_man_pages(&self, output_dir: &std::path::Path) -> Result<()> {
        let man_dir = output_dir.join("man");
        std::fs::create_dir_all(&man_dir)?;

        // Generate main man page
        man::generate_main_man_page(&self.command_name, &self.version, &man_dir)?;

        // Generate subcommand man pages
        man::generate_subcommand_man_pages(&self.command_name, &self.version, &man_dir)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_documentation_generator_creation() {
        let generator = DocumentationGenerator::new("voirs".to_string(), "1.0.0".to_string());
        assert_eq!(generator.command_name, "voirs");
        assert_eq!(generator.version, "1.0.0");
    }

    #[test]
    fn test_generate_all_creates_directories() {
        let temp_dir = TempDir::new().unwrap();
        let generator = DocumentationGenerator::new("voirs".to_string(), "1.0.0".to_string());

        generator.generate_all(temp_dir.path()).unwrap();

        assert!(temp_dir.path().join("man").exists());
    }
}