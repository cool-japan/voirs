//! Batch processing templates for common workflows.
//!
//! This module provides pre-configured templates for common batch processing
//! use cases like audiobooks, podcasts, and voice-overs.

use serde::{Deserialize, Serialize};
use voirs_sdk::QualityLevel;

/// Batch processing template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTemplate {
    /// Template name
    pub name: String,
    /// Description of what this template is for
    pub description: String,
    /// Quality level
    pub quality: QualityLevel,
    /// Speaking rate
    pub rate: f32,
    /// Pitch adjustment
    pub pitch: f32,
    /// Volume adjustment
    pub volume: f32,
    /// Whether to enhance audio
    pub enhance: bool,
    /// Output format
    pub output_format: String,
    /// Additional template-specific settings
    pub settings: TemplateSettings,
}

/// Template-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateSettings {
    /// Chapter markers for audiobooks
    pub chapter_markers: bool,
    /// Add intro/outro
    pub add_bookends: bool,
    /// Pause duration between chapters (ms)
    pub chapter_pause_ms: Option<u64>,
    /// Normalize audio levels
    pub normalize: bool,
    /// Target loudness in LUFS
    pub target_loudness: Option<f32>,
}

impl Default for TemplateSettings {
    fn default() -> Self {
        Self {
            chapter_markers: false,
            add_bookends: false,
            chapter_pause_ms: None,
            normalize: false,
            target_loudness: None,
        }
    }
}

/// Get a template by name
pub fn get_template(name: &str) -> Option<BatchTemplate> {
    match name.to_lowercase().as_str() {
        "audiobook" => Some(audiobook_template()),
        "podcast" => Some(podcast_template()),
        "voiceover" => Some(voiceover_template()),
        "narration" => Some(narration_template()),
        "news" => Some(news_template()),
        "education" => Some(education_template()),
        "fast" => Some(fast_template()),
        "high-quality" => Some(high_quality_template()),
        _ => None,
    }
}

/// List all available templates
pub fn list_templates() -> Vec<BatchTemplate> {
    vec![
        audiobook_template(),
        podcast_template(),
        voiceover_template(),
        narration_template(),
        news_template(),
        education_template(),
        fast_template(),
        high_quality_template(),
    ]
}

/// Audiobook template - optimized for long-form narrative content
fn audiobook_template() -> BatchTemplate {
    BatchTemplate {
        name: "audiobook".to_string(),
        description: "Optimized for audiobook production with chapter support".to_string(),
        quality: QualityLevel::High,
        rate: 0.95, // Slightly slower for better comprehension
        pitch: 0.0,
        volume: -2.0, // Slightly quieter for comfort
        enhance: true,
        output_format: "mp3".to_string(),
        settings: TemplateSettings {
            chapter_markers: true,
            add_bookends: true,
            chapter_pause_ms: Some(2000),
            normalize: true,
            target_loudness: Some(-16.0),
        },
    }
}

/// Podcast template - optimized for conversational content
fn podcast_template() -> BatchTemplate {
    BatchTemplate {
        name: "podcast".to_string(),
        description: "Optimized for podcast episodes with consistent audio levels".to_string(),
        quality: QualityLevel::High,
        rate: 1.0,
        pitch: 0.0,
        volume: 0.0,
        enhance: true,
        output_format: "mp3".to_string(),
        settings: TemplateSettings {
            chapter_markers: false,
            add_bookends: true,
            chapter_pause_ms: Some(1000),
            normalize: true,
            target_loudness: Some(-16.0),
        },
    }
}

/// Voice-over template - optimized for video narration
fn voiceover_template() -> BatchTemplate {
    BatchTemplate {
        name: "voiceover".to_string(),
        description: "Optimized for video voice-overs with precise timing".to_string(),
        quality: QualityLevel::High,
        rate: 1.05, // Slightly faster for dynamic feel
        pitch: 0.0,
        volume: 0.0,
        enhance: true,
        output_format: "wav".to_string(), // Lossless for video editing
        settings: TemplateSettings {
            chapter_markers: false,
            add_bookends: false,
            chapter_pause_ms: None,
            normalize: true,
            target_loudness: Some(-14.0), // Louder for video
        },
    }
}

/// Narration template - optimized for documentary-style content
fn narration_template() -> BatchTemplate {
    BatchTemplate {
        name: "narration".to_string(),
        description: "Optimized for documentary and educational narration".to_string(),
        quality: QualityLevel::High,
        rate: 0.98,
        pitch: -0.5, // Slightly deeper for authority
        volume: 0.0,
        enhance: true,
        output_format: "flac".to_string(),
        settings: TemplateSettings {
            chapter_markers: false,
            add_bookends: false,
            chapter_pause_ms: Some(1500),
            normalize: true,
            target_loudness: Some(-16.0),
        },
    }
}

/// News template - optimized for news reading
fn news_template() -> BatchTemplate {
    BatchTemplate {
        name: "news".to_string(),
        description: "Optimized for news broadcasts with clear delivery".to_string(),
        quality: QualityLevel::High,
        rate: 1.1, // Slightly faster for news delivery
        pitch: 0.0,
        volume: 0.0,
        enhance: true,
        output_format: "mp3".to_string(),
        settings: TemplateSettings {
            chapter_markers: false,
            add_bookends: false,
            chapter_pause_ms: Some(500),
            normalize: true,
            target_loudness: Some(-14.0),
        },
    }
}

/// Education template - optimized for educational content
fn education_template() -> BatchTemplate {
    BatchTemplate {
        name: "education".to_string(),
        description: "Optimized for educational materials with clear pronunciation".to_string(),
        quality: QualityLevel::High,
        rate: 0.92, // Slower for better understanding
        pitch: 0.0,
        volume: 0.0,
        enhance: true,
        output_format: "mp3".to_string(),
        settings: TemplateSettings {
            chapter_markers: true,
            add_bookends: false,
            chapter_pause_ms: Some(2000),
            normalize: true,
            target_loudness: Some(-16.0),
        },
    }
}

/// Fast template - optimized for quick processing
fn fast_template() -> BatchTemplate {
    BatchTemplate {
        name: "fast".to_string(),
        description: "Fast processing with medium quality for quick turnaround".to_string(),
        quality: QualityLevel::Medium,
        rate: 1.0,
        pitch: 0.0,
        volume: 0.0,
        enhance: false,
        output_format: "mp3".to_string(),
        settings: TemplateSettings::default(),
    }
}

/// High-quality template - maximum quality settings
fn high_quality_template() -> BatchTemplate {
    BatchTemplate {
        name: "high-quality".to_string(),
        description: "Maximum quality for professional production".to_string(),
        quality: QualityLevel::Ultra,
        rate: 1.0,
        pitch: 0.0,
        volume: 0.0,
        enhance: true,
        output_format: "flac".to_string(),
        settings: TemplateSettings {
            chapter_markers: false,
            add_bookends: false,
            chapter_pause_ms: None,
            normalize: true,
            target_loudness: Some(-16.0),
        },
    }
}

/// Display template information
pub fn display_template(template: &BatchTemplate) {
    println!("📋 Template: {}", template.name);
    println!("   {}", template.description);
    println!();
    println!("Settings:");
    println!("  Quality: {:?}", template.quality);
    println!("  Rate: {}x", template.rate);
    println!("  Pitch: {} semitones", template.pitch);
    println!("  Volume: {} dB", template.volume);
    println!(
        "  Enhancement: {}",
        if template.enhance {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("  Output format: {}", template.output_format);
    println!();
    if template.settings.chapter_markers {
        println!("  📚 Chapter markers: enabled");
    }
    if template.settings.add_bookends {
        println!("  🎬 Intro/outro: enabled");
    }
    if let Some(pause) = template.settings.chapter_pause_ms {
        println!("  ⏸️  Chapter pause: {}ms", pause);
    }
    if template.settings.normalize {
        println!("  📊 Audio normalization: enabled");
        if let Some(lufs) = template.settings.target_loudness {
            println!("     Target: {} LUFS", lufs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_template() {
        let template = get_template("audiobook");
        assert!(template.is_some());
        let template = template.unwrap();
        assert_eq!(template.name, "audiobook");
        assert!(template.settings.chapter_markers);
    }

    #[test]
    fn test_list_templates() {
        let templates = list_templates();
        assert_eq!(templates.len(), 8);
        assert!(templates.iter().any(|t| t.name == "audiobook"));
        assert!(templates.iter().any(|t| t.name == "podcast"));
    }

    #[test]
    fn test_audiobook_template() {
        let template = audiobook_template();
        assert_eq!(template.rate, 0.95);
        assert_eq!(template.quality, QualityLevel::High);
        assert!(template.settings.chapter_markers);
    }

    #[test]
    fn test_fast_template() {
        let template = fast_template();
        assert_eq!(template.quality, QualityLevel::Medium);
        assert!(!template.enhance);
    }
}
