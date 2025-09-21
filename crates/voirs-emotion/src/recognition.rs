//! # Emotion Recognition System
//!
//! Automatic emotion detection from input text using lexical analysis,
//! sentiment analysis, and pattern matching to determine appropriate
//! emotional expressions for voice synthesis.

use crate::{Emotion, EmotionDimensions, EmotionIntensity, EmotionVector, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for emotion recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionRecognitionConfig {
    /// Minimum confidence threshold for emotion detection
    pub confidence_threshold: f32,
    /// Whether to use context-aware analysis
    pub context_aware: bool,
    /// Maximum text length to analyze (for performance)
    pub max_text_length: usize,
    /// Weight for sentiment analysis in final decision
    pub sentiment_weight: f32,
    /// Weight for lexical analysis in final decision
    pub lexical_weight: f32,
    /// Weight for context analysis in final decision
    pub context_weight: f32,
}

impl Default for EmotionRecognitionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.3,
            context_aware: true,
            max_text_length: 10000,
            sentiment_weight: 0.4,
            lexical_weight: 0.4,
            context_weight: 0.2,
        }
    }
}

/// Result of emotion recognition analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionRecognitionResult {
    /// Primary detected emotion
    pub primary_emotion: Emotion,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Alternative emotions with their confidence scores
    pub alternatives: Vec<(Emotion, f32)>,
    /// Emotional dimensions detected
    pub dimensions: EmotionDimensions,
    /// Suggested intensity
    pub intensity: EmotionIntensity,
    /// Recognition method used
    pub method: RecognitionMethod,
    /// Analysis metadata
    pub metadata: RecognitionMetadata,
}

/// Emotion recognition methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecognitionMethod {
    /// Lexical analysis using emotion keywords
    Lexical,
    /// Sentiment-based analysis
    Sentiment,
    /// Context-aware analysis
    Context,
    /// Combined analysis using multiple methods
    Combined,
}

/// Metadata from recognition analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionMetadata {
    /// Text length analyzed
    pub text_length: usize,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Number of emotion keywords found
    pub keyword_count: usize,
    /// Sentiment polarity (-1.0 to 1.0)
    pub sentiment_polarity: f32,
    /// Sentiment magnitude (0.0 to 1.0)
    pub sentiment_magnitude: f32,
}

/// Emotion keyword mapping
#[derive(Debug, Clone)]
struct EmotionKeyword {
    emotion: Emotion,
    weight: f32,
    intensity_modifier: f32,
}

/// Main emotion recognition engine
#[derive(Debug)]
pub struct EmotionRecognizer {
    config: EmotionRecognitionConfig,
    emotion_keywords: HashMap<String, Vec<EmotionKeyword>>,
    sentiment_analyzer: SentimentAnalyzer,
}

impl EmotionRecognizer {
    /// Create a new emotion recognizer with default configuration
    pub fn new() -> Self {
        Self::with_config(EmotionRecognitionConfig::default())
    }

    /// Create a new emotion recognizer with custom configuration
    pub fn with_config(config: EmotionRecognitionConfig) -> Self {
        let mut recognizer = Self {
            config,
            emotion_keywords: HashMap::new(),
            sentiment_analyzer: SentimentAnalyzer::new(),
        };
        recognizer.initialize_emotion_keywords();
        recognizer
    }

    /// Recognize emotions from input text
    pub fn recognize(&self, text: &str) -> Result<EmotionRecognitionResult> {
        let start_time = std::time::Instant::now();

        // Truncate text if too long
        let text = if text.len() > self.config.max_text_length {
            &text[..self.config.max_text_length]
        } else {
            text
        };

        let text_length = text.len();
        let text_lower = text.to_lowercase();

        // Perform different types of analysis
        let lexical_result = self.analyze_lexical(&text_lower)?;
        let sentiment_result = self.sentiment_analyzer.analyze(&text_lower)?;
        let context_result = if self.config.context_aware {
            self.analyze_context(&text_lower)?
        } else {
            ContextAnalysis::default()
        };

        // Combine results
        let combined_result =
            self.combine_analyses(&lexical_result, &sentiment_result, &context_result)?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        let metadata = RecognitionMetadata {
            text_length,
            processing_time_ms: processing_time,
            keyword_count: lexical_result.keywords_found,
            sentiment_polarity: sentiment_result.polarity,
            sentiment_magnitude: sentiment_result.magnitude,
        };

        Ok(EmotionRecognitionResult {
            primary_emotion: combined_result.emotion,
            confidence: combined_result.confidence,
            alternatives: combined_result.alternatives,
            dimensions: combined_result.dimensions,
            intensity: combined_result.intensity,
            method: RecognitionMethod::Combined,
            metadata,
        })
    }

    /// Recognize emotion using only lexical analysis
    pub fn recognize_lexical(&self, text: &str) -> Result<EmotionRecognitionResult> {
        let start_time = std::time::Instant::now();
        let text_lower = text.to_lowercase();
        let lexical_result = self.analyze_lexical(&text_lower)?;

        let processing_time = start_time.elapsed().as_millis() as u64;
        let metadata = RecognitionMetadata {
            text_length: text.len(),
            processing_time_ms: processing_time,
            keyword_count: lexical_result.keywords_found,
            sentiment_polarity: 0.0,
            sentiment_magnitude: 0.0,
        };

        Ok(EmotionRecognitionResult {
            primary_emotion: lexical_result.emotion,
            confidence: lexical_result.confidence,
            alternatives: lexical_result.alternatives,
            dimensions: lexical_result.dimensions,
            intensity: lexical_result.intensity,
            method: RecognitionMethod::Lexical,
            metadata,
        })
    }

    /// Update configuration
    pub fn update_config(&mut self, config: EmotionRecognitionConfig) {
        self.config = config;
    }

    /// Add custom emotion keywords
    pub fn add_emotion_keyword(
        &mut self,
        keyword: &str,
        emotion: Emotion,
        weight: f32,
        intensity_modifier: f32,
    ) {
        let emotion_keyword = EmotionKeyword {
            emotion,
            weight,
            intensity_modifier,
        };

        self.emotion_keywords
            .entry(keyword.to_lowercase())
            .or_default()
            .push(emotion_keyword);
    }

    /// Initialize built-in emotion keywords
    fn initialize_emotion_keywords(&mut self) {
        // Happy/Joy keywords
        let happy_keywords = [
            ("happy", 1.0),
            ("joy", 1.0),
            ("joyful", 0.9),
            ("cheerful", 0.8),
            ("delighted", 0.9),
            ("elated", 0.9),
            ("ecstatic", 1.0),
            ("thrilled", 0.8),
            ("excited", 0.7),
            ("pleased", 0.6),
            ("content", 0.5),
            ("glad", 0.7),
            ("wonderful", 0.6),
            ("amazing", 0.5),
            ("fantastic", 0.6),
            ("excellent", 0.5),
            ("great", 0.4),
            ("good", 0.3),
            ("smile", 0.5),
            ("laugh", 0.6),
            ("celebration", 0.7),
            ("party", 0.5),
            ("birthday", 0.4),
            ("success", 0.5),
        ];

        for (keyword, weight) in happy_keywords {
            self.add_emotion_keyword(keyword, Emotion::Happy, weight, weight);
        }

        // Sad keywords
        let sad_keywords = [
            ("sad", 1.0),
            ("depressed", 0.9),
            ("melancholy", 0.8),
            ("sorrowful", 0.9),
            ("grief", 0.9),
            ("mourn", 0.8),
            ("weep", 0.8),
            ("cry", 0.7),
            ("tears", 0.6),
            ("lonely", 0.7),
            ("blue", 0.4),
            ("down", 0.5),
            ("heartbroken", 0.9),
            ("devastated", 0.8),
            ("disappointed", 0.6),
            ("loss", 0.6),
            ("death", 0.7),
            ("funeral", 0.8),
            ("tragedy", 0.8),
        ];

        for (keyword, weight) in sad_keywords {
            self.add_emotion_keyword(keyword, Emotion::Sad, weight, weight);
        }

        // Angry keywords
        let angry_keywords = [
            ("angry", 1.0),
            ("mad", 0.9),
            ("furious", 1.0),
            ("rage", 1.0),
            ("enraged", 0.9),
            ("livid", 0.9),
            ("irritated", 0.6),
            ("annoyed", 0.5),
            ("frustrated", 0.7),
            ("outraged", 0.8),
            ("hate", 0.8),
            ("disgusted", 0.7),
            ("resentful", 0.7),
            ("bitter", 0.6),
            ("hostile", 0.8),
            ("aggressive", 0.8),
            ("fight", 0.6),
            ("attack", 0.7),
            ("war", 0.6),
            ("violence", 0.7),
        ];

        for (keyword, weight) in angry_keywords {
            self.add_emotion_keyword(keyword, Emotion::Angry, weight, weight);
        }

        // Fear keywords
        let fear_keywords = [
            ("fear", 1.0),
            ("afraid", 0.9),
            ("scared", 0.8),
            ("terrified", 1.0),
            ("frightened", 0.9),
            ("anxious", 0.7),
            ("worried", 0.6),
            ("nervous", 0.6),
            ("panic", 0.9),
            ("dread", 0.8),
            ("horror", 0.9),
            ("terror", 1.0),
            ("phobia", 0.8),
            ("danger", 0.6),
            ("threat", 0.7),
            ("risk", 0.4),
            ("emergency", 0.6),
            ("alarm", 0.6),
            ("warning", 0.5),
            ("caution", 0.4),
        ];

        for (keyword, weight) in fear_keywords {
            self.add_emotion_keyword(keyword, Emotion::Fear, weight, weight);
        }

        // Surprise keywords
        let surprise_keywords = [
            ("surprise", 1.0),
            ("surprised", 0.9),
            ("shocked", 0.8),
            ("stunned", 0.8),
            ("amazed", 0.7),
            ("astonished", 0.8),
            ("bewildered", 0.7),
            ("confused", 0.5),
            ("unexpected", 0.6),
            ("sudden", 0.5),
            ("wow", 0.6),
            ("omg", 0.7),
            ("unbelievable", 0.7),
            ("incredible", 0.6),
            ("remarkable", 0.5),
        ];

        for (keyword, weight) in surprise_keywords {
            self.add_emotion_keyword(keyword, Emotion::Surprise, weight, weight);
        }

        // Disgust keywords
        let disgust_keywords = [
            ("disgust", 1.0),
            ("disgusted", 0.9),
            ("revolting", 0.8),
            ("repulsive", 0.8),
            ("sick", 0.6),
            ("nauseous", 0.7),
            ("gross", 0.7),
            ("awful", 0.6),
            ("terrible", 0.5),
            ("horrible", 0.6),
            ("nasty", 0.7),
            ("vile", 0.8),
            ("filthy", 0.6),
            ("rotten", 0.6),
            ("stinking", 0.6),
        ];

        for (keyword, weight) in disgust_keywords {
            self.add_emotion_keyword(keyword, Emotion::Disgust, weight, weight);
        }

        // Contempt keywords
        let contempt_keywords = [
            ("contempt", 1.0),
            ("scorn", 0.9),
            ("disdain", 0.8),
            ("arrogant", 0.7),
            ("superior", 0.5),
            ("condescending", 0.8),
            ("dismissive", 0.7),
            ("sneering", 0.8),
            ("mocking", 0.7),
            ("ridiculous", 0.6),
            ("pathetic", 0.7),
            ("worthless", 0.8),
        ];

        for (keyword, weight) in contempt_keywords {
            self.add_emotion_keyword(keyword, Emotion::Angry, weight, weight);
        }
    }

    /// Perform lexical analysis on text
    fn analyze_lexical(&self, text: &str) -> Result<LexicalAnalysis> {
        let mut emotion_scores: HashMap<Emotion, f32> = HashMap::new();
        let mut keywords_found = 0;
        let words: Vec<&str> = text.split_whitespace().collect();

        // Analyze each word and phrase
        for window_size in 1..=3 {
            for window in words.windows(window_size) {
                let phrase = window.join(" ");

                if let Some(keywords) = self.emotion_keywords.get(&phrase) {
                    keywords_found += 1;
                    for keyword in keywords {
                        let score = emotion_scores.entry(keyword.emotion.clone()).or_insert(0.0);
                        *score += keyword.weight;
                    }
                }
            }
        }

        // Find the dominant emotion
        let (primary_emotion, confidence) = emotion_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(emotion, score)| (emotion.clone(), *score / words.len() as f32))
            .unwrap_or((Emotion::Neutral, 0.0));

        // Create alternatives list
        let mut alternatives: Vec<(Emotion, f32)> = emotion_scores
            .into_iter()
            .filter(|(emotion, _)| *emotion != primary_emotion)
            .map(|(emotion, score)| (emotion, score / words.len() as f32))
            .collect();
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        alternatives.truncate(3);

        // Map to dimensions
        let dimensions = self.emotion_to_dimensions(primary_emotion.clone(), confidence);
        let intensity = EmotionIntensity::from_confidence(confidence);

        Ok(LexicalAnalysis {
            emotion: primary_emotion,
            confidence: confidence.min(1.0),
            alternatives,
            dimensions,
            intensity,
            keywords_found,
        })
    }

    /// Analyze context patterns in text
    fn analyze_context(&self, text: &str) -> Result<ContextAnalysis> {
        let mut context_score = 0.0;
        let mut context_emotion = Emotion::Neutral;

        // Question patterns often indicate curiosity or confusion
        if text.contains('?') {
            context_score += 0.3;
            context_emotion = Emotion::Surprise;
        }

        // Exclamation patterns often indicate strong emotions
        let exclamation_count = text.matches('!').count();
        if exclamation_count > 0 {
            context_score += (exclamation_count as f32 * 0.2).min(0.8);
            // Keep existing emotion if detected, otherwise excited
            if context_emotion == Emotion::Neutral {
                context_emotion = Emotion::Happy;
            }
        }

        // ALL CAPS indicates strong emotion
        let caps_ratio =
            text.chars().filter(|c| c.is_uppercase()).count() as f32 / text.len() as f32;
        if caps_ratio > 0.3 {
            context_score += caps_ratio;
            context_emotion = Emotion::Angry; // Often anger or excitement
        }

        // Repetition patterns (!!!, ???, ...)
        if text.contains("...") {
            context_score += 0.4;
            context_emotion = Emotion::Sad; // Often contemplative or sad
        }

        Ok(ContextAnalysis {
            emotion: context_emotion,
            confidence: context_score.min(1.0),
            patterns_detected: vec![], // Could be expanded
            dimensions: EmotionDimensions::neutral(),
        })
    }

    /// Combine different analysis results
    fn combine_analyses(
        &self,
        lexical: &LexicalAnalysis,
        sentiment: &SentimentAnalysis,
        context: &ContextAnalysis,
    ) -> Result<CombinedAnalysis> {
        let lexical_weight = self.config.lexical_weight;
        let sentiment_weight = self.config.sentiment_weight;
        let context_weight = self.config.context_weight;

        // Calculate weighted confidence
        let total_confidence = (lexical.confidence * lexical_weight)
            + (sentiment.confidence * sentiment_weight)
            + (context.confidence * context_weight);

        // Determine primary emotion based on highest weighted confidence
        let primary_emotion = if lexical.confidence * lexical_weight
            >= sentiment.confidence * sentiment_weight
            && lexical.confidence * lexical_weight >= context.confidence * context_weight
        {
            lexical.emotion.clone()
        } else if sentiment.confidence * sentiment_weight >= context.confidence * context_weight {
            sentiment.emotion.clone()
        } else {
            context.emotion.clone()
        };

        // Combine dimensions
        let combined_dimensions = EmotionDimensions {
            valence: (lexical.dimensions.valence * lexical_weight
                + sentiment.dimensions.valence * sentiment_weight
                + context.dimensions.valence * context_weight)
                / (lexical_weight + sentiment_weight + context_weight),
            arousal: (lexical.dimensions.arousal * lexical_weight
                + sentiment.dimensions.arousal * sentiment_weight
                + context.dimensions.arousal * context_weight)
                / (lexical_weight + sentiment_weight + context_weight),
            dominance: (lexical.dimensions.dominance * lexical_weight
                + sentiment.dimensions.dominance * sentiment_weight
                + context.dimensions.dominance * context_weight)
                / (lexical_weight + sentiment_weight + context_weight),
        };

        // Combine alternatives
        let mut combined_alternatives = lexical.alternatives.clone();
        combined_alternatives.push((sentiment.emotion.clone(), sentiment.confidence));
        combined_alternatives.push((context.emotion.clone(), context.confidence));
        combined_alternatives
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        combined_alternatives.truncate(3);

        let intensity = EmotionIntensity::from_confidence(total_confidence);

        Ok(CombinedAnalysis {
            emotion: primary_emotion,
            confidence: total_confidence.min(1.0),
            alternatives: combined_alternatives,
            dimensions: combined_dimensions,
            intensity,
        })
    }

    /// Map emotion to dimensional values
    fn emotion_to_dimensions(&self, emotion: Emotion, intensity: f32) -> EmotionDimensions {
        let base_dims = match emotion {
            Emotion::Happy => EmotionDimensions {
                valence: 0.8,
                arousal: 0.6,
                dominance: 0.2,
            },
            Emotion::Sad => EmotionDimensions {
                valence: -0.8,
                arousal: -0.4,
                dominance: -0.6,
            },
            Emotion::Angry => EmotionDimensions {
                valence: -0.6,
                arousal: 0.8,
                dominance: 0.4,
            },
            Emotion::Fear => EmotionDimensions {
                valence: -0.7,
                arousal: 0.6,
                dominance: -0.8,
            },
            Emotion::Surprise => EmotionDimensions {
                valence: 0.2,
                arousal: 0.8,
                dominance: -0.2,
            },
            Emotion::Disgust => EmotionDimensions {
                valence: -0.8,
                arousal: 0.2,
                dominance: 0.0,
            },
            Emotion::Excited => EmotionDimensions {
                valence: 0.6,
                arousal: 0.8,
                dominance: 0.3,
            },
            Emotion::Calm => EmotionDimensions {
                valence: 0.3,
                arousal: -0.6,
                dominance: 0.2,
            },
            Emotion::Tender => EmotionDimensions {
                valence: 0.5,
                arousal: -0.2,
                dominance: -0.3,
            },
            Emotion::Confident => EmotionDimensions {
                valence: 0.4,
                arousal: 0.2,
                dominance: 0.7,
            },
            Emotion::Melancholic => EmotionDimensions {
                valence: -0.6,
                arousal: -0.3,
                dominance: -0.4,
            },
            Emotion::Custom(_) => EmotionDimensions {
                valence: 0.0,
                arousal: 0.0,
                dominance: 0.0,
            },
            Emotion::Neutral => EmotionDimensions {
                valence: 0.0,
                arousal: 0.0,
                dominance: 0.0,
            },
        };

        // Scale by intensity
        EmotionDimensions {
            valence: base_dims.valence * intensity,
            arousal: base_dims.arousal * intensity,
            dominance: base_dims.dominance * intensity,
        }
    }
}

impl Default for EmotionRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of lexical analysis
#[derive(Debug)]
struct LexicalAnalysis {
    emotion: Emotion,
    confidence: f32,
    alternatives: Vec<(Emotion, f32)>,
    dimensions: EmotionDimensions,
    intensity: EmotionIntensity,
    keywords_found: usize,
}

/// Result of sentiment analysis
#[derive(Debug)]
struct SentimentAnalysis {
    emotion: Emotion,
    confidence: f32,
    polarity: f32,
    magnitude: f32,
    dimensions: EmotionDimensions,
}

/// Result of context analysis
#[derive(Debug)]
struct ContextAnalysis {
    emotion: Emotion,
    confidence: f32,
    patterns_detected: Vec<String>,
    dimensions: EmotionDimensions,
}

impl Default for ContextAnalysis {
    fn default() -> Self {
        Self {
            emotion: Emotion::Neutral,
            confidence: 0.0,
            patterns_detected: vec![],
            dimensions: EmotionDimensions::neutral(),
        }
    }
}

/// Combined analysis result
#[derive(Debug)]
struct CombinedAnalysis {
    emotion: Emotion,
    confidence: f32,
    alternatives: Vec<(Emotion, f32)>,
    dimensions: EmotionDimensions,
    intensity: EmotionIntensity,
}

/// Simple sentiment analyzer
#[derive(Debug)]
struct SentimentAnalyzer {
    positive_words: Vec<&'static str>,
    negative_words: Vec<&'static str>,
}

impl SentimentAnalyzer {
    fn new() -> Self {
        Self {
            positive_words: vec![
                "good",
                "great",
                "excellent",
                "amazing",
                "wonderful",
                "fantastic",
                "perfect",
                "awesome",
                "brilliant",
                "outstanding",
                "superb",
                "magnificent",
                "love",
                "like",
                "enjoy",
                "appreciate",
                "adore",
                "treasure",
                "beautiful",
                "lovely",
                "gorgeous",
                "stunning",
                "attractive",
                "success",
                "win",
                "victory",
                "triumph",
                "achieve",
                "accomplish",
            ],
            negative_words: vec![
                "bad",
                "terrible",
                "awful",
                "horrible",
                "disgusting",
                "revolting",
                "hate",
                "dislike",
                "despise",
                "loathe",
                "detest",
                "abhor",
                "fail",
                "failure",
                "lose",
                "defeat",
                "disappoint",
                "reject",
                "ugly",
                "hideous",
                "repulsive",
                "gross",
                "nasty",
                "vile",
                "problem",
                "issue",
                "trouble",
                "difficulty",
                "struggle",
                "crisis",
            ],
        }
    }

    fn analyze(&self, text: &str) -> Result<SentimentAnalysis> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut positive_score = 0.0;
        let mut negative_score = 0.0;

        for word in words.iter() {
            let word_lower = word.to_lowercase();
            if self.positive_words.contains(&word_lower.as_str()) {
                positive_score += 1.0;
            } else if self.negative_words.contains(&word_lower.as_str()) {
                negative_score += 1.0;
            }
        }

        let total_sentiment_words = positive_score + negative_score;
        let polarity = if total_sentiment_words > 0.0 {
            (positive_score - negative_score) / total_sentiment_words
        } else {
            0.0
        };

        let magnitude = if words.is_empty() {
            0.0
        } else {
            total_sentiment_words / words.len() as f32
        };

        let (emotion, confidence) = if polarity > 0.3 {
            (Emotion::Happy, magnitude)
        } else if polarity < -0.3 {
            (Emotion::Sad, magnitude)
        } else {
            (Emotion::Neutral, 0.0)
        };

        let dimensions = EmotionDimensions {
            valence: polarity,
            arousal: magnitude,
            dominance: 0.0,
        };

        Ok(SentimentAnalysis {
            emotion,
            confidence: confidence.min(1.0),
            polarity,
            magnitude,
            dimensions,
        })
    }
}

/// Extension trait for EmotionIntensity
impl EmotionIntensity {
    /// Create intensity from confidence score
    pub fn from_confidence(confidence: f32) -> Self {
        match confidence {
            c if c >= 0.8 => EmotionIntensity::VERY_HIGH,
            c if c >= 0.6 => EmotionIntensity::HIGH,
            c if c >= 0.4 => EmotionIntensity::MEDIUM,
            c if c >= 0.2 => EmotionIntensity::LOW,
            _ => EmotionIntensity::VERY_LOW,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_recognizer_creation() {
        let recognizer = EmotionRecognizer::new();
        assert!(!recognizer.emotion_keywords.is_empty());
    }

    #[test]
    fn test_happy_text_recognition() {
        let recognizer = EmotionRecognizer::new();
        let result = recognizer
            .recognize("I am so happy and joyful today!")
            .unwrap();

        assert_eq!(result.primary_emotion, Emotion::Happy);
        assert!(result.confidence > 0.1);
        assert!(result.metadata.keyword_count > 0);
    }

    #[test]
    fn test_sad_text_recognition() {
        let recognizer = EmotionRecognizer::new();
        let result = recognizer.recognize("I feel so sad and depressed").unwrap();

        assert_eq!(result.primary_emotion, Emotion::Sad);
        assert!(result.confidence > 0.1);
    }

    #[test]
    fn test_angry_text_recognition() {
        let recognizer = EmotionRecognizer::new();
        let result = recognizer.recognize("I am furious and angry!").unwrap();

        assert_eq!(result.primary_emotion, Emotion::Angry);
        assert!(result.confidence > 0.1);
    }

    #[test]
    fn test_neutral_text_recognition() {
        let recognizer = EmotionRecognizer::new();
        let result = recognizer.recognize("The weather is normal today").unwrap();

        // Should default to neutral with low confidence
        assert!(result.confidence < 0.5);
    }

    #[test]
    fn test_lexical_only_recognition() {
        let recognizer = EmotionRecognizer::new();
        let result = recognizer
            .recognize_lexical("excited and thrilled")
            .unwrap();

        assert_eq!(result.primary_emotion, Emotion::Happy);
        assert_eq!(result.method, RecognitionMethod::Lexical);
    }

    #[test]
    fn test_custom_keywords() {
        let mut recognizer = EmotionRecognizer::new();
        recognizer.add_emotion_keyword("blissful", Emotion::Happy, 1.0, 1.0);

        let result = recognizer.recognize("I feel blissful").unwrap();
        assert_eq!(result.primary_emotion, Emotion::Happy);
    }

    #[test]
    fn test_config_update() {
        let mut recognizer = EmotionRecognizer::new();
        let mut config = EmotionRecognitionConfig::default();
        config.confidence_threshold = 0.5;

        recognizer.update_config(config);
        assert_eq!(recognizer.config.confidence_threshold, 0.5);
    }

    #[test]
    fn test_alternatives_generation() {
        let recognizer = EmotionRecognizer::new();
        let result = recognizer
            .recognize("I am happy but also excited and thrilled")
            .unwrap();

        assert!(!result.alternatives.is_empty());
        assert!(result.alternatives.len() <= 3);
    }

    #[test]
    fn test_intensity_mapping() {
        assert_eq!(
            EmotionIntensity::from_confidence(0.9),
            EmotionIntensity::VERY_HIGH
        );
        assert_eq!(
            EmotionIntensity::from_confidence(0.7),
            EmotionIntensity::HIGH
        );
        assert_eq!(
            EmotionIntensity::from_confidence(0.5),
            EmotionIntensity::MEDIUM
        );
        assert_eq!(
            EmotionIntensity::from_confidence(0.3),
            EmotionIntensity::LOW
        );
        assert_eq!(
            EmotionIntensity::from_confidence(0.1),
            EmotionIntensity::VERY_LOW
        );
    }

    #[test]
    fn test_punctuation_context() {
        let recognizer = EmotionRecognizer::new();

        // Test exclamation marks
        let result1 = recognizer.recognize("Hello!").unwrap();
        assert!(result1.confidence > 0.0);

        // Test questions
        let result2 = recognizer.recognize("How are you?").unwrap();
        assert!(result2.confidence > 0.0);

        // Test ellipsis
        let result3 = recognizer.recognize("Well...").unwrap();
        assert!(result3.confidence > 0.0);
    }

    #[test]
    fn test_performance_metadata() {
        let recognizer = EmotionRecognizer::new();
        let result = recognizer.recognize("This is a test message").unwrap();

        assert!(result.metadata.processing_time_ms < 1000);
        assert_eq!(result.metadata.text_length, 22);
    }

    #[test]
    fn test_long_text_handling() {
        let recognizer = EmotionRecognizer::new();
        let long_text = "happy ".repeat(1000);
        let result = recognizer.recognize(&long_text).unwrap();

        assert_eq!(result.primary_emotion, Emotion::Happy);
        assert!(result.confidence > 0.1);
    }
}
