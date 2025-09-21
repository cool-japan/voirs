//! Semantic analysis functionality for context-aware preprocessing.

use crate::{LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Semantic context information for text
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticContext {
    /// Detected topics with confidence scores
    pub topics: HashMap<String, f32>,
    /// Sentiment polarity (-1.0 to 1.0)
    pub sentiment_polarity: f32,
    /// Formality level (0.0 to 1.0, higher = more formal)
    pub formality_level: f32,
    /// Technical complexity score (0.0 to 1.0)
    pub technical_complexity: f32,
    /// Domain/field classification
    pub domain: Option<String>,
    /// Emotional tone indicators
    pub emotion_indicators: Vec<String>,
    /// Linguistic register (formal, informal, technical, etc.)
    pub register: String,
}

impl Default for SemanticContext {
    fn default() -> Self {
        Self {
            topics: HashMap::new(),
            sentiment_polarity: 0.0,
            formality_level: 0.5,
            technical_complexity: 0.0,
            domain: None,
            emotion_indicators: Vec::new(),
            register: "neutral".to_string(),
        }
    }
}

/// Trait for semantic analysis functionality
pub trait SemanticAnalysis: Send + Sync {
    /// Analyze semantic context of text
    fn analyze_context(&self, text: &str) -> Result<SemanticContext>;

    /// Detect topics in text
    fn detect_topics(&self, text: &str) -> Result<HashMap<String, f32>>;

    /// Analyze sentiment
    fn analyze_sentiment(&self, text: &str) -> Result<f32>;

    /// Assess formality level
    fn assess_formality(&self, text: &str) -> Result<f32>;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<LanguageCode>;
}

/// Basic semantic analyzer
#[derive(Debug, Clone)]
pub struct BasicSemanticAnalyzer {
    /// Topic models
    pub topic_models: HashMap<String, TopicModel>,
    /// Sentiment lexicons
    pub sentiment_lexicons: HashMap<LanguageCode, SentimentLexicon>,
    /// Formality indicators
    pub formality_indicators: HashMap<LanguageCode, FormalityIndicators>,
}

/// Simple topic model
#[derive(Debug, Clone)]
pub struct TopicModel {
    /// Topic keywords with weights
    pub keywords: HashMap<String, f32>,
    /// Topic name
    pub name: String,
    /// Topic confidence threshold
    pub threshold: f32,
}

/// Sentiment lexicon
#[derive(Debug, Clone)]
pub struct SentimentLexicon {
    /// Word sentiment scores (-1.0 to 1.0)
    pub word_scores: HashMap<String, f32>,
    /// Negation patterns
    pub negation_patterns: Vec<String>,
    /// Intensifier patterns with multipliers
    pub intensifier_patterns: Vec<(String, f32)>,
}

/// Formality indicators
#[derive(Debug, Clone)]
pub struct FormalityIndicators {
    /// Formal words/phrases with scores
    pub formal_indicators: HashMap<String, f32>,
    /// Informal words/phrases with scores
    pub informal_indicators: HashMap<String, f32>,
    /// Technical terms
    pub technical_terms: HashSet<String>,
}

impl SemanticAnalysis for BasicSemanticAnalyzer {
    fn analyze_context(&self, text: &str) -> Result<SemanticContext> {
        let words: Vec<&str> = text.split_whitespace().collect();

        // Topic detection
        let topics = self.detect_topics(text)?;

        // Sentiment analysis
        let sentiment_polarity = self.analyze_sentiment(text)?;

        // Formality assessment
        let formality_level = self.assess_formality(text)?;

        // Technical complexity assessment
        let technical_complexity = self.assess_technical_complexity(&words);

        // Domain classification
        let domain = self.classify_domain(&topics);

        // Emotion indicators
        let emotion_indicators = self.detect_emotion_indicators(&words);

        // Register determination
        let register = self.determine_register(formality_level, technical_complexity);

        Ok(SemanticContext {
            topics,
            sentiment_polarity,
            formality_level,
            technical_complexity,
            domain,
            emotion_indicators,
            register,
        })
    }

    fn detect_topics(&self, text: &str) -> Result<HashMap<String, f32>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut topic_scores = HashMap::new();

        for (topic_name, topic_model) in &self.topic_models {
            let mut score = 0.0;
            let mut word_count = 0;

            for word in &words {
                if let Some(word_score) = topic_model.keywords.get(&word.to_lowercase()) {
                    score += word_score;
                    word_count += 1;
                }
            }

            if word_count > 0 {
                let normalized_score = score / words.len() as f32;
                if normalized_score >= topic_model.threshold {
                    topic_scores.insert(topic_name.clone(), normalized_score);
                }
            }
        }

        Ok(topic_scores)
    }

    fn analyze_sentiment(&self, text: &str) -> Result<f32> {
        let words: Vec<&str> = text.split_whitespace().collect();

        // Use English lexicon as default
        let lexicon = self.sentiment_lexicons.get(&LanguageCode::EnUs);

        if let Some(lexicon) = lexicon {
            let mut total_score = 0.0;
            let mut scored_words = 0;
            let mut negation_active = false;

            for word in &words {
                let word_lower = word.to_lowercase();

                // Check for negation
                if lexicon
                    .negation_patterns
                    .iter()
                    .any(|pattern| word_lower.contains(pattern))
                {
                    negation_active = true;
                    continue;
                }

                // Get sentiment score
                if let Some(score) = lexicon.word_scores.get(&word_lower) {
                    let final_score = if negation_active { -score } else { *score };
                    total_score += final_score;
                    scored_words += 1;
                    negation_active = false; // Reset negation after applying
                }
            }

            if scored_words > 0 {
                Ok(total_score / scored_words as f32)
            } else {
                Ok(0.0)
            }
        } else {
            Ok(0.0) // Neutral if no lexicon available
        }
    }

    fn assess_formality(&self, text: &str) -> Result<f32> {
        let words: Vec<&str> = text.split_whitespace().collect();

        // Use English indicators as default
        let indicators = self.formality_indicators.get(&LanguageCode::EnUs);

        if let Some(indicators) = indicators {
            let mut formal_score = 0.0;
            let mut informal_score = 0.0;

            for word in &words {
                let word_lower = word.to_lowercase();

                if let Some(score) = indicators.formal_indicators.get(&word_lower) {
                    formal_score += score;
                }

                if let Some(score) = indicators.informal_indicators.get(&word_lower) {
                    informal_score += score;
                }
            }

            let total_score = formal_score + informal_score;
            if total_score > 0.0 {
                Ok(formal_score / total_score)
            } else {
                Ok(0.5) // Neutral
            }
        } else {
            Ok(0.5) // Neutral if no indicators available
        }
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        self.sentiment_lexicons.keys().copied().collect()
    }
}

impl BasicSemanticAnalyzer {
    /// Create a new basic semantic analyzer with comprehensive defaults
    pub fn new() -> Self {
        let mut analyzer = Self {
            topic_models: HashMap::new(),
            sentiment_lexicons: HashMap::new(),
            formality_indicators: HashMap::new(),
        };

        // Initialize with comprehensive defaults
        analyzer.initialize_default_topic_models();
        analyzer.initialize_default_sentiment_lexicons();
        analyzer.initialize_default_formality_indicators();

        analyzer
    }

    /// Add a topic model
    pub fn add_topic_model(&mut self, name: String, model: TopicModel) {
        self.topic_models.insert(name, model);
    }

    /// Add a sentiment lexicon for a language
    pub fn add_sentiment_lexicon(&mut self, language: LanguageCode, lexicon: SentimentLexicon) {
        self.sentiment_lexicons.insert(language, lexicon);
    }

    /// Add formality indicators for a language
    pub fn add_formality_indicators(
        &mut self,
        language: LanguageCode,
        indicators: FormalityIndicators,
    ) {
        self.formality_indicators.insert(language, indicators);
    }

    /// Initialize comprehensive default topic models
    fn initialize_default_topic_models(&mut self) {
        // Technology topic model
        let mut tech_model = TopicModel::new("technology".to_string(), 0.1);
        let tech_keywords = vec![
            ("computer", 0.9),
            ("software", 0.8),
            ("algorithm", 0.8),
            ("programming", 0.9),
            ("code", 0.7),
            ("development", 0.7),
            ("system", 0.6),
            ("network", 0.7),
            ("database", 0.8),
            ("server", 0.7),
            ("application", 0.6),
            ("digital", 0.6),
            ("internet", 0.7),
            ("web", 0.6),
            ("mobile", 0.6),
            ("cloud", 0.7),
            ("artificial", 0.8),
            ("intelligence", 0.8),
            ("machine", 0.7),
            ("learning", 0.8),
            ("data", 0.6),
            ("analytics", 0.7),
            ("security", 0.7),
            ("encryption", 0.8),
        ];
        for (word, weight) in tech_keywords {
            tech_model.add_keyword(word.to_string(), weight);
        }
        self.add_topic_model("technology".to_string(), tech_model);

        // Business topic model
        let mut business_model = TopicModel::new("business".to_string(), 0.1);
        let business_keywords = vec![
            ("company", 0.8),
            ("market", 0.8),
            ("customer", 0.7),
            ("sales", 0.8),
            ("revenue", 0.9),
            ("profit", 0.9),
            ("investment", 0.8),
            ("strategy", 0.7),
            ("management", 0.7),
            ("finance", 0.8),
            ("marketing", 0.8),
            ("brand", 0.7),
            ("product", 0.6),
            ("service", 0.6),
            ("business", 0.9),
            ("enterprise", 0.8),
            ("corporate", 0.8),
            ("commercial", 0.7),
            ("economic", 0.7),
            ("financial", 0.8),
            ("industry", 0.7),
            ("sector", 0.7),
            ("competition", 0.7),
            ("growth", 0.7),
        ];
        for (word, weight) in business_keywords {
            business_model.add_keyword(word.to_string(), weight);
        }
        self.add_topic_model("business".to_string(), business_model);

        // Health topic model
        let mut health_model = TopicModel::new("health".to_string(), 0.1);
        let health_keywords = vec![
            ("medical", 0.9),
            ("health", 0.9),
            ("doctor", 0.8),
            ("patient", 0.8),
            ("treatment", 0.8),
            ("medicine", 0.8),
            ("hospital", 0.8),
            ("clinic", 0.7),
            ("diagnosis", 0.8),
            ("therapy", 0.7),
            ("surgery", 0.8),
            ("disease", 0.7),
            ("symptoms", 0.7),
            ("prevention", 0.7),
            ("wellness", 0.7),
            ("fitness", 0.6),
            ("nutrition", 0.7),
            ("pharmaceutical", 0.8),
            ("research", 0.6),
            ("clinical", 0.8),
            ("healthcare", 0.9),
            ("nursing", 0.7),
            ("emergency", 0.7),
            ("recovery", 0.6),
        ];
        for (word, weight) in health_keywords {
            health_model.add_keyword(word.to_string(), weight);
        }
        self.add_topic_model("health".to_string(), health_model);

        // Education topic model
        let mut education_model = TopicModel::new("education".to_string(), 0.1);
        let education_keywords = vec![
            ("school", 0.8),
            ("university", 0.8),
            ("student", 0.8),
            ("teacher", 0.8),
            ("learning", 0.9),
            ("education", 0.9),
            ("academic", 0.8),
            ("study", 0.7),
            ("research", 0.7),
            ("curriculum", 0.8),
            ("course", 0.7),
            ("degree", 0.7),
            ("knowledge", 0.7),
            ("scholarship", 0.8),
            ("tuition", 0.7),
            ("exam", 0.6),
            ("grade", 0.6),
            ("class", 0.6),
            ("lecture", 0.7),
            ("professor", 0.8),
            ("college", 0.8),
            ("training", 0.7),
            ("skill", 0.6),
            ("instruction", 0.7),
        ];
        for (word, weight) in education_keywords {
            education_model.add_keyword(word.to_string(), weight);
        }
        self.add_topic_model("education".to_string(), education_model);

        // Science topic model
        let mut science_model = TopicModel::new("science".to_string(), 0.1);
        let science_keywords = vec![
            ("science", 0.9),
            ("research", 0.8),
            ("experiment", 0.8),
            ("theory", 0.7),
            ("hypothesis", 0.8),
            ("analysis", 0.7),
            ("method", 0.6),
            ("result", 0.6),
            ("conclusion", 0.7),
            ("discovery", 0.8),
            ("innovation", 0.7),
            ("laboratory", 0.8),
            ("physics", 0.8),
            ("chemistry", 0.8),
            ("biology", 0.8),
            ("mathematics", 0.8),
            ("engineering", 0.8),
            ("technology", 0.7),
            ("scientific", 0.8),
            ("academic", 0.7),
            ("publication", 0.7),
            ("journal", 0.7),
            ("peer", 0.6),
            ("review", 0.6),
        ];
        for (word, weight) in science_keywords {
            science_model.add_keyword(word.to_string(), weight);
        }
        self.add_topic_model("science".to_string(), science_model);
    }

    /// Initialize comprehensive default sentiment lexicons
    fn initialize_default_sentiment_lexicons(&mut self) {
        // English sentiment lexicon
        let mut en_lexicon = SentimentLexicon::new();

        // Positive words
        let positive_words = vec![
            ("excellent", 0.9),
            ("amazing", 0.9),
            ("wonderful", 0.8),
            ("fantastic", 0.9),
            ("great", 0.7),
            ("good", 0.6),
            ("nice", 0.5),
            ("beautiful", 0.7),
            ("perfect", 0.9),
            ("outstanding", 0.9),
            ("superb", 0.8),
            ("brilliant", 0.8),
            ("awesome", 0.8),
            ("terrific", 0.8),
            ("marvelous", 0.8),
            ("spectacular", 0.8),
            ("love", 0.8),
            ("like", 0.4),
            ("enjoy", 0.6),
            ("happy", 0.7),
            ("pleased", 0.6),
            ("satisfied", 0.6),
            ("delighted", 0.8),
            ("thrilled", 0.8),
            ("excited", 0.7),
            ("positive", 0.6),
            ("successful", 0.7),
            ("effective", 0.5),
            ("impressive", 0.7),
            ("remarkable", 0.7),
            ("exceptional", 0.8),
            ("superior", 0.7),
        ];
        for (word, score) in positive_words {
            en_lexicon.add_word(word.to_string(), score);
        }

        // Negative words
        let negative_words = vec![
            ("terrible", -0.9),
            ("awful", -0.9),
            ("horrible", -0.9),
            ("disgusting", -0.9),
            ("bad", -0.6),
            ("poor", -0.5),
            ("worst", -0.9),
            ("hate", -0.8),
            ("dislike", -0.5),
            ("disappointing", -0.7),
            ("frustrated", -0.7),
            ("angry", -0.7),
            ("sad", -0.6),
            ("upset", -0.6),
            ("annoyed", -0.5),
            ("irritated", -0.5),
            ("terrible", -0.9),
            ("dreadful", -0.8),
            ("appalling", -0.9),
            ("shocking", -0.7),
            ("unacceptable", -0.8),
            ("inadequate", -0.6),
            ("insufficient", -0.5),
            ("useless", -0.8),
            ("worthless", -0.8),
            ("pathetic", -0.8),
            ("ridiculous", -0.6),
            ("absurd", -0.6),
            ("stupid", -0.7),
            ("foolish", -0.6),
            ("nonsense", -0.6),
            ("wrong", -0.4),
        ];
        for (word, score) in negative_words {
            en_lexicon.add_word(word.to_string(), score);
        }

        self.add_sentiment_lexicon(LanguageCode::EnUs, en_lexicon);
    }

    /// Initialize comprehensive default formality indicators
    fn initialize_default_formality_indicators(&mut self) {
        // English formality indicators
        let mut en_indicators = FormalityIndicators::new();

        // Formal indicators
        let formal_words = vec![
            ("therefore", 0.9),
            ("furthermore", 0.9),
            ("however", 0.8),
            ("nevertheless", 0.9),
            ("consequently", 0.9),
            ("accordingly", 0.8),
            ("subsequently", 0.8),
            ("moreover", 0.8),
            ("additionally", 0.8),
            ("likewise", 0.7),
            ("nonetheless", 0.8),
            ("whereas", 0.8),
            ("regarding", 0.7),
            ("concerning", 0.7),
            ("pursuant", 0.9),
            ("henceforth", 0.9),
            ("heretofore", 0.9),
            ("notwithstanding", 0.9),
            ("aforementioned", 0.9),
            ("herewith", 0.8),
            ("kindly", 0.7),
            ("please", 0.5),
            ("respectfully", 0.8),
            ("sincerely", 0.8),
            ("cordially", 0.8),
            ("gratefully", 0.7),
            ("appreciate", 0.6),
            ("acknowledge", 0.7),
            ("endeavor", 0.8),
            ("utilize", 0.7),
            ("implement", 0.6),
            ("establish", 0.6),
        ];
        for (word, score) in formal_words {
            en_indicators.add_formal_indicator(word.to_string(), score);
        }

        // Informal indicators
        let informal_words = vec![
            ("yeah", 0.9),
            ("yep", 0.8),
            ("nope", 0.8),
            ("gonna", 0.9),
            ("wanna", 0.9),
            ("gotta", 0.9),
            ("kinda", 0.8),
            ("sorta", 0.8),
            ("dunno", 0.9),
            ("ain't", 0.9),
            ("can't", 0.4),
            ("won't", 0.4),
            ("don't", 0.4),
            ("isn't", 0.4),
            ("wasn't", 0.4),
            ("weren't", 0.4),
            ("cool", 0.6),
            ("awesome", 0.6),
            ("sweet", 0.7),
            ("neat", 0.6),
            ("stuff", 0.6),
            ("things", 0.4),
            ("guys", 0.7),
            ("folks", 0.6),
            ("ok", 0.7),
            ("okay", 0.6),
            ("alright", 0.7),
            ("sure", 0.5),
            ("totally", 0.7),
            ("really", 0.4),
            ("pretty", 0.4),
            ("super", 0.6),
        ];
        for (word, score) in informal_words {
            en_indicators.add_informal_indicator(word.to_string(), score);
        }

        // Technical terms
        let technical_terms = vec![
            "algorithm",
            "methodology",
            "implementation",
            "optimization",
            "configuration",
            "specification",
            "architecture",
            "framework",
            "paradigm",
            "protocol",
            "interface",
            "abstraction",
            "encapsulation",
            "polymorphism",
            "inheritance",
            "instantiation",
            "initialization",
            "synchronization",
            "asynchronous",
            "concurrent",
            "distributed",
            "scalable",
            "modular",
            "extensible",
            "maintainable",
        ];
        for term in technical_terms {
            en_indicators.add_technical_term(term.to_string());
        }

        self.add_formality_indicators(LanguageCode::EnUs, en_indicators);
    }

    /// Assess technical complexity based on word complexity
    fn assess_technical_complexity(&self, words: &[&str]) -> f32 {
        let mut _technical_count = 0;
        let mut total_complexity = 0.0;

        for word in words {
            let word_len = word.len();

            // Simple heuristics for technical complexity
            if word_len > 10 {
                _technical_count += 1;
                total_complexity += 0.3;
            }

            if word.contains('_') || word.contains('-') {
                _technical_count += 1;
                total_complexity += 0.2;
            }

            if word.chars().any(|c| c.is_uppercase()) && word.len() > 3 {
                _technical_count += 1;
                total_complexity += 0.1;
            }
        }

        if words.is_empty() {
            0.0
        } else {
            total_complexity / words.len() as f32
        }
    }

    /// Classify domain based on topic scores
    fn classify_domain(&self, topics: &HashMap<String, f32>) -> Option<String> {
        topics
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(domain, _)| domain.clone())
    }

    /// Detect emotion indicators in text
    fn detect_emotion_indicators(&self, words: &[&str]) -> Vec<String> {
        let emotion_words = [
            "happy",
            "sad",
            "angry",
            "excited",
            "worried",
            "surprised",
            "disappointed",
            "frustrated",
            "delighted",
            "anxious",
        ];

        words
            .iter()
            .filter_map(|word| {
                let word_lower = word.to_lowercase();
                if emotion_words.contains(&word_lower.as_str()) {
                    Some(word_lower)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Determine linguistic register
    fn determine_register(&self, formality_level: f32, technical_complexity: f32) -> String {
        match (formality_level, technical_complexity) {
            (f, t) if f > 0.7 && t > 0.5 => "academic".to_string(),
            (f, _) if f > 0.7 => "formal".to_string(),
            (f, _) if f < 0.3 => "informal".to_string(),
            (_, t) if t > 0.6 => "technical".to_string(),
            _ => "neutral".to_string(),
        }
    }
}

impl Default for BasicSemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TopicModel {
    /// Create a new topic model
    pub fn new(name: String, threshold: f32) -> Self {
        Self {
            keywords: HashMap::new(),
            name,
            threshold,
        }
    }

    /// Add a keyword with weight
    pub fn add_keyword(&mut self, keyword: String, weight: f32) {
        self.keywords.insert(keyword, weight);
    }
}

impl SentimentLexicon {
    /// Create a new sentiment lexicon
    pub fn new() -> Self {
        Self {
            word_scores: HashMap::new(),
            negation_patterns: vec!["not".to_string(), "no".to_string(), "never".to_string()],
            intensifier_patterns: vec![
                ("very".to_string(), 1.5),
                ("extremely".to_string(), 2.0),
                ("quite".to_string(), 1.2),
            ],
        }
    }

    /// Add a word with sentiment score
    pub fn add_word(&mut self, word: String, score: f32) {
        self.word_scores.insert(word, score.clamp(-1.0, 1.0));
    }
}

impl Default for SentimentLexicon {
    fn default() -> Self {
        Self::new()
    }
}

impl FormalityIndicators {
    /// Create new formality indicators
    pub fn new() -> Self {
        Self {
            formal_indicators: HashMap::new(),
            informal_indicators: HashMap::new(),
            technical_terms: HashSet::new(),
        }
    }

    /// Add a formal indicator
    pub fn add_formal_indicator(&mut self, word: String, score: f32) {
        self.formal_indicators.insert(word, score);
    }

    /// Add an informal indicator
    pub fn add_informal_indicator(&mut self, word: String, score: f32) {
        self.informal_indicators.insert(word, score);
    }

    /// Add a technical term
    pub fn add_technical_term(&mut self, term: String) {
        self.technical_terms.insert(term);
    }
}

impl Default for FormalityIndicators {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_context_default() {
        let context = SemanticContext::default();
        assert_eq!(context.sentiment_polarity, 0.0);
        assert_eq!(context.formality_level, 0.5);
        assert_eq!(context.technical_complexity, 0.0);
        assert_eq!(context.register, "neutral");
    }

    #[test]
    fn test_topic_model() {
        let mut model = TopicModel::new("technology".to_string(), 0.1);
        model.add_keyword("computer".to_string(), 0.8);
        model.add_keyword("software".to_string(), 0.7);

        assert_eq!(model.keywords.len(), 2);
        assert_eq!(model.keywords.get("computer"), Some(&0.8));
    }

    #[test]
    fn test_sentiment_lexicon() {
        let mut lexicon = SentimentLexicon::new();
        lexicon.add_word("happy".to_string(), 0.8);
        lexicon.add_word("sad".to_string(), -0.6);

        assert_eq!(lexicon.word_scores.get("happy"), Some(&0.8));
        assert_eq!(lexicon.word_scores.get("sad"), Some(&-0.6));
    }

    #[test]
    fn test_formality_indicators() {
        let mut indicators = FormalityIndicators::new();
        indicators.add_formal_indicator("therefore".to_string(), 0.8);
        indicators.add_informal_indicator("yeah".to_string(), 0.9);
        indicators.add_technical_term("algorithm".to_string());

        assert_eq!(indicators.formal_indicators.len(), 1);
        assert_eq!(indicators.informal_indicators.len(), 1);
        assert!(indicators.technical_terms.contains("algorithm"));
    }

    #[test]
    fn test_basic_semantic_analyzer() {
        let analyzer = BasicSemanticAnalyzer::new();

        // Test with empty text
        let context = analyzer.analyze_context("").unwrap();
        assert_eq!(context.topics.len(), 0);
        assert_eq!(context.sentiment_polarity, 0.0);
    }

    #[test]
    fn test_technical_complexity() {
        let analyzer = BasicSemanticAnalyzer::new();
        let words = vec!["very_complex_function", "algorithm", "data"];
        let complexity = analyzer.assess_technical_complexity(&words);
        assert!(complexity > 0.0);
    }

    #[test]
    fn test_emotion_indicators() {
        let analyzer = BasicSemanticAnalyzer::new();
        let words = vec!["I", "am", "very", "happy", "today"];
        let emotions = analyzer.detect_emotion_indicators(&words);
        assert!(emotions.contains(&"happy".to_string()));
    }

    #[test]
    fn test_register_determination() {
        let analyzer = BasicSemanticAnalyzer::new();

        assert_eq!(analyzer.determine_register(0.8, 0.6), "academic");
        assert_eq!(analyzer.determine_register(0.8, 0.2), "formal");
        assert_eq!(analyzer.determine_register(0.2, 0.1), "informal");
        assert_eq!(analyzer.determine_register(0.5, 0.7), "technical");
        assert_eq!(analyzer.determine_register(0.5, 0.3), "neutral");
    }
}
