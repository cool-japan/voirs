//! Voice search functionality.

use crate::error::{CliError, Result};
use std::collections::HashMap;
use voirs_sdk::config::AppConfig;
use voirs_sdk::types::VoiceConfig;
use voirs_sdk::{QualityLevel, Result as VoirsResult, VoirsPipeline};

/// Voice search functionality
pub struct VoiceSearch {
    /// Available voices
    voices: Vec<VoiceConfig>,
}

impl VoiceSearch {
    /// Create a new voice search instance
    pub async fn new(config: &AppConfig) -> Result<Self> {
        let pipeline = VoirsPipeline::builder().build().await?;
        let voices = pipeline.list_voices().await?;

        Ok(Self { voices })
    }

    /// Search voices by query string
    pub fn search(&self, query: &str) -> Vec<VoiceSearchResult> {
        let query_lower = query.to_lowercase();
        let terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut results = Vec::new();

        for voice in &self.voices {
            let score = self.calculate_relevance_score(voice, &terms);
            if score > 0.0 {
                results.push(VoiceSearchResult {
                    voice: voice.clone(),
                    relevance_score: score,
                    match_reasons: self.get_match_reasons(voice, &terms),
                });
            }
        }

        // Sort by relevance score (highest first)
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

        results
    }

    /// Search voices by specific criteria
    pub fn search_by_criteria(&self, criteria: &VoiceSearchCriteria) -> Vec<VoiceSearchResult> {
        let mut results = Vec::new();

        for voice in &self.voices {
            let mut score = 1.0;
            let mut reasons = Vec::new();
            let mut matches = true;

            // Language filter
            if let Some(ref language) = criteria.language {
                if voice.language.as_str().to_lowercase() != language.to_lowercase() {
                    matches = false;
                } else {
                    reasons.push("Matching language".to_string());
                }
            }

            // Gender filter
            if let Some(ref gender) = criteria.gender {
                if let Some(voice_gender) = &voice.characteristics.gender {
                    if format!("{:?}", voice_gender).to_lowercase() != gender.to_lowercase() {
                        matches = false;
                    } else {
                        reasons.push("Matching gender".to_string());
                        score += 0.5;
                    }
                } else if criteria.require_gender {
                    matches = false;
                }
            }

            // Age filter
            if let Some(ref age) = criteria.age {
                if let Some(voice_age) = &voice.characteristics.age {
                    if format!("{:?}", voice_age).to_lowercase() != age.to_lowercase() {
                        matches = false;
                    } else {
                        reasons.push("Matching age".to_string());
                        score += 0.3;
                    }
                } else if criteria.require_age {
                    matches = false;
                }
            }

            // Style filter
            if let Some(ref style) = criteria.style {
                if format!("{:?}", voice.characteristics.style).to_lowercase()
                    != style.to_lowercase()
                {
                    matches = false;
                } else {
                    reasons.push("Matching style".to_string());
                    score += 0.4;
                }
            }

            // Quality filter
            if let Some(ref quality) = criteria.min_quality {
                let voice_quality_score = match voice.characteristics.quality {
                    QualityLevel::Low => 1,
                    QualityLevel::Medium => 2,
                    QualityLevel::High => 3,
                    QualityLevel::Ultra => 4,
                };
                let min_quality_score = match quality.as_str() {
                    "low" => 1,
                    "medium" => 2,
                    "high" => 3,
                    "ultra" => 4,
                    _ => 1,
                };

                if voice_quality_score < min_quality_score {
                    matches = false;
                } else {
                    if voice_quality_score >= min_quality_score {
                        reasons.push("Meets quality requirements".to_string());
                        score += 0.2;
                    }
                }
            }

            // Emotion support filter
            if criteria.emotion_support {
                if voice.characteristics.emotion_support {
                    reasons.push("Supports emotions".to_string());
                    score += 0.3;
                } else if criteria.require_emotion_support {
                    matches = false;
                }
            }

            if matches {
                results.push(VoiceSearchResult {
                    voice: voice.clone(),
                    relevance_score: score,
                    match_reasons: reasons,
                });
            }
        }

        // Sort by relevance score
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

        results
    }

    /// Get voice recommendations based on text content
    pub fn recommend_for_text(&self, text: &str) -> Vec<VoiceSearchResult> {
        let mut criteria = VoiceSearchCriteria::default();

        // Analyze text to suggest appropriate voice characteristics
        let word_count = text.split_whitespace().count();
        let has_questions = text.contains('?');
        let has_exclamations = text.contains('!');
        let is_formal = self.is_formal_text(text);

        // Suggest style based on content
        if has_exclamations || text.to_uppercase() == text {
            criteria.style = Some("Energetic".to_string());
        } else if has_questions || is_formal {
            criteria.style = Some("Professional".to_string());
        } else if word_count > 100 {
            criteria.style = Some("Narrative".to_string());
        }

        // Suggest quality based on text length
        if word_count > 50 {
            criteria.min_quality = Some("high".to_string());
        } else {
            criteria.min_quality = Some("medium".to_string());
        }

        self.search_by_criteria(&criteria)
    }

    /// Calculate relevance score for a voice given search terms
    fn calculate_relevance_score(&self, voice: &VoiceConfig, terms: &[&str]) -> f32 {
        let mut score = 0.0;

        for term in terms {
            // Check voice ID (highest weight)
            if voice.id.to_lowercase().contains(term) {
                score += 3.0;
            }

            // Check voice name
            if voice.name.to_lowercase().contains(term) {
                score += 2.5;
            }

            // Check language
            if voice.language.as_str().to_lowercase().contains(term) {
                score += 2.0;
            }

            // Check description in metadata
            if let Some(description) = voice.metadata.get("description") {
                if description.to_lowercase().contains(term) {
                    score += 1.5;
                }
            }

            // Check characteristics
            if let Some(gender) = &voice.characteristics.gender {
                if format!("{:?}", gender).to_lowercase().contains(term) {
                    score += 1.0;
                }
            }

            if let Some(age) = &voice.characteristics.age {
                if format!("{:?}", age).to_lowercase().contains(term) {
                    score += 1.0;
                }
            }

            if format!("{:?}", voice.characteristics.style)
                .to_lowercase()
                .contains(term)
            {
                score += 1.0;
            }

            // Check metadata tags
            for (key, value) in &voice.metadata {
                if key.to_lowercase().contains(term) || value.to_lowercase().contains(term) {
                    score += 0.5;
                }
            }
        }

        score
    }

    /// Get reasons why a voice matched the search terms
    fn get_match_reasons(&self, voice: &VoiceConfig, terms: &[&str]) -> Vec<String> {
        let mut reasons = Vec::new();

        for term in terms {
            if voice.id.to_lowercase().contains(term) {
                reasons.push(format!("ID contains '{}'", term));
            }
            if voice.name.to_lowercase().contains(term) {
                reasons.push(format!("Name contains '{}'", term));
            }
            if voice.language.as_str().to_lowercase().contains(term) {
                reasons.push(format!("Language matches '{}'", term));
            }
            if let Some(description) = voice.metadata.get("description") {
                if description.to_lowercase().contains(term) {
                    reasons.push(format!("Description contains '{}'", term));
                }
            }
        }

        if reasons.is_empty() {
            reasons.push("General match".to_string());
        }

        reasons
    }

    /// Check if text appears to be formal
    fn is_formal_text(&self, text: &str) -> bool {
        let formal_indicators = [
            "please",
            "thank you",
            "regarding",
            "furthermore",
            "however",
            "therefore",
            "moreover",
            "nevertheless",
            "consequently",
            "accordingly",
        ];

        let text_lower = text.to_lowercase();
        formal_indicators
            .iter()
            .any(|&indicator| text_lower.contains(indicator))
    }

    /// Get voice statistics
    pub fn get_statistics(&self) -> VoiceStatistics {
        let mut stats = VoiceStatistics::default();

        stats.total_voices = self.voices.len();

        // Language distribution
        for voice in &self.voices {
            let lang = voice.language.as_str();
            *stats.languages.entry(lang.to_string()).or_insert(0) += 1;
        }

        // Gender distribution
        for voice in &self.voices {
            if let Some(gender) = &voice.characteristics.gender {
                let gender_str = format!("{:?}", gender);
                *stats.genders.entry(gender_str).or_insert(0) += 1;
            } else {
                *stats.genders.entry("Unknown".to_string()).or_insert(0) += 1;
            }
        }

        // Quality distribution
        for voice in &self.voices {
            let quality_str = format!("{:?}", voice.characteristics.quality);
            *stats.qualities.entry(quality_str).or_insert(0) += 1;
        }

        // Emotion support
        stats.emotion_support_count = self
            .voices
            .iter()
            .filter(|v| v.characteristics.emotion_support)
            .count();

        stats
    }
}

/// Voice search criteria
#[derive(Debug, Default)]
pub struct VoiceSearchCriteria {
    pub language: Option<String>,
    pub gender: Option<String>,
    pub age: Option<String>,
    pub style: Option<String>,
    pub min_quality: Option<String>,
    pub emotion_support: bool,
    pub require_gender: bool,
    pub require_age: bool,
    pub require_emotion_support: bool,
}

/// Voice search result
#[derive(Debug, Clone)]
pub struct VoiceSearchResult {
    pub voice: VoiceConfig,
    pub relevance_score: f32,
    pub match_reasons: Vec<String>,
}

/// Voice statistics
#[derive(Debug, Default)]
pub struct VoiceStatistics {
    pub total_voices: usize,
    pub languages: HashMap<String, usize>,
    pub genders: HashMap<String, usize>,
    pub qualities: HashMap<String, usize>,
    pub emotion_support_count: usize,
}

/// Run voice search command
pub async fn run_voice_search(
    query: Option<&str>,
    language: Option<&str>,
    gender: Option<&str>,
    age: Option<&str>,
    style: Option<&str>,
    min_quality: Option<&str>,
    emotion_support: bool,
    show_stats: bool,
    config: &AppConfig,
) -> Result<()> {
    let search = VoiceSearch::new(config).await?;

    if show_stats {
        print_voice_statistics(&search.get_statistics());
        return Ok(());
    }

    let results = if let Some(query) = query {
        search.search(query)
    } else {
        let criteria = VoiceSearchCriteria {
            language: language.map(|s| s.to_string()),
            gender: gender.map(|s| s.to_string()),
            age: age.map(|s| s.to_string()),
            style: style.map(|s| s.to_string()),
            min_quality: min_quality.map(|s| s.to_string()),
            emotion_support,
            require_gender: gender.is_some(),
            require_age: age.is_some(),
            require_emotion_support: emotion_support,
        };
        search.search_by_criteria(&criteria)
    };

    if results.is_empty() {
        println!("No voices found matching your criteria.");
        if let Some(query) = query {
            println!("Try using broader search terms or check the available voices with 'voirs voices list'.");
        }
        return Ok(());
    }

    println!("Found {} voice(s):", results.len());
    println!();

    for (i, result) in results.iter().enumerate().take(10) {
        // Limit to top 10 results
        print_voice_search_result(i + 1, result);
        if i < results.len() - 1 {
            println!("---");
        }
    }

    if results.len() > 10 {
        println!();
        println!(
            "... and {} more results. Use more specific search criteria to narrow down.",
            results.len() - 10
        );
    }

    Ok(())
}

/// Print voice search result
fn print_voice_search_result(index: usize, result: &VoiceSearchResult) {
    println!("{}. {} ({})", index, result.voice.name, result.voice.id);
    println!("   Language: {}", result.voice.language.as_str());

    if let Some(gender) = &result.voice.characteristics.gender {
        println!("   Gender: {:?}", gender);
    }

    if let Some(age) = &result.voice.characteristics.age {
        println!("   Age: {:?}", age);
    }

    println!("   Style: {:?}", result.voice.characteristics.style);
    println!("   Quality: {:?}", result.voice.characteristics.quality);

    if result.voice.characteristics.emotion_support {
        println!("   ✓ Emotion support");
    }

    println!("   Relevance: {:.1}", result.relevance_score);
    println!("   Matches: {}", result.match_reasons.join(", "));

    if let Some(description) = result.voice.metadata.get("description") {
        if description.len() <= 100 {
            println!("   Description: {}", description);
        } else {
            println!("   Description: {}...", &description[..97]);
        }
    }
}

/// Print voice statistics
fn print_voice_statistics(stats: &VoiceStatistics) {
    println!("Voice Database Statistics");
    println!("========================");
    println!("Total voices: {}", stats.total_voices);
    println!();

    println!("By Language:");
    let mut lang_vec: Vec<_> = stats.languages.iter().collect();
    lang_vec.sort_by_key(|(_, count)| **count);
    lang_vec.reverse();
    for (lang, count) in lang_vec {
        println!("  {}: {}", lang, count);
    }
    println!();

    println!("By Gender:");
    let mut gender_vec: Vec<_> = stats.genders.iter().collect();
    gender_vec.sort_by_key(|(_, count)| **count);
    gender_vec.reverse();
    for (gender, count) in gender_vec {
        println!("  {}: {}", gender, count);
    }
    println!();

    println!("By Quality:");
    let mut quality_vec: Vec<_> = stats.qualities.iter().collect();
    quality_vec.sort_by_key(|(_, count)| **count);
    quality_vec.reverse();
    for (quality, count) in quality_vec {
        println!("  {}: {}", quality, count);
    }
    println!();

    println!(
        "Emotion Support: {} voices ({:.1}%)",
        stats.emotion_support_count,
        (stats.emotion_support_count as f32 / stats.total_voices as f32) * 100.0
    );
}
