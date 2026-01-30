//! LLM-based Musical Understanding
//!
//! This module provides advanced musical understanding capabilities using Large Language Models
//! for semantic analysis, musical context interpretation, and intelligent musical decision-making.

use crate::score::{MusicalNote, MusicalScore};
use crate::types::{Articulation, Dynamics, Expression, VoiceCharacteristics};
use scirs2_core::random::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LLM-based musical understanding engine
///
/// Provides semantic analysis and intelligent musical interpretation using
/// large language models for enhanced musical understanding.
#[derive(Debug)]
pub struct LlmMusicalUnderstanding {
    /// Model configuration
    config: LlmConfig,
    /// Musical context cache
    context_cache: HashMap<String, MusicalContext>,
    /// Semantic embeddings for musical concepts
    semantic_embeddings: HashMap<String, Vec<f32>>,
}

/// Configuration for LLM-based understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Model type (e.g., "gpt-4", "claude-3", "local-llama")
    pub model_type: String,
    /// Maximum context length
    pub max_context_length: usize,
    /// Temperature for sampling (0.0-1.0)
    pub temperature: f32,
    /// Top-p sampling threshold
    pub top_p: f32,
    /// Enable caching for repeated queries
    pub enable_caching: bool,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_type: "gpt-4".to_string(),
            max_context_length: 8192,
            temperature: 0.7,
            top_p: 0.9,
            enable_caching: true,
            embedding_dim: 768,
        }
    }
}

/// Musical context derived from LLM analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalContext {
    /// Emotional context
    pub emotion: String,
    /// Musical style context
    pub style: String,
    /// Narrative or lyrical content
    pub narrative: String,
    /// Suggested tempo markings
    pub tempo_markings: Vec<String>,
    /// Dynamic suggestions
    pub dynamic_suggestions: Vec<DynamicSuggestion>,
    /// Articulation recommendations
    pub articulation_recommendations: Vec<ArticulationRecommendation>,
    /// Phrasing boundaries
    pub phrasing_boundaries: Vec<PhrasingBoundary>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Dynamic suggestion from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicSuggestion {
    /// Beat position
    pub beat: f32,
    /// Suggested dynamics
    pub dynamics: Dynamics,
    /// Reason for suggestion
    pub reason: String,
}

/// Articulation recommendation from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticulationRecommendation {
    /// Note index
    pub note_index: usize,
    /// Suggested articulation
    pub articulation: Articulation,
    /// Reason for recommendation
    pub reason: String,
}

/// Phrasing boundary detected by LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhrasingBoundary {
    /// Beat position
    pub beat: f32,
    /// Boundary type (e.g., "breath", "caesura", "phrase_end")
    pub boundary_type: String,
    /// Suggested breath duration
    pub breath_duration: Option<f32>,
}

/// Musical prompt for LLM analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusicalPrompt {
    /// Score representation
    pub score: String,
    /// Lyrics (if available)
    pub lyrics: Option<String>,
    /// Musical style hint
    pub style_hint: Option<String>,
    /// Emotional context hint
    pub emotion_hint: Option<String>,
    /// Performance instructions
    pub performance_instructions: Option<String>,
}

/// LLM response for musical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Musical context analysis
    pub context: MusicalContext,
    /// Voice characteristics suggestions
    pub voice_suggestions: Option<VoiceCharacteristics>,
    /// Expression mappings
    pub expression_mappings: HashMap<String, Expression>,
    /// Additional commentary
    pub commentary: String,
}

impl LlmMusicalUnderstanding {
    /// Create a new LLM musical understanding engine
    pub fn new(config: LlmConfig) -> Self {
        Self {
            config,
            context_cache: HashMap::new(),
            semantic_embeddings: HashMap::new(),
        }
    }

    /// Analyze musical score for semantic understanding
    ///
    /// # Arguments
    /// * `score` - Musical score to analyze
    /// * `lyrics` - Optional lyrics for context
    ///
    /// # Returns
    /// Musical context derived from LLM analysis
    pub async fn analyze_score(
        &mut self,
        score: &MusicalScore,
        lyrics: Option<&str>,
    ) -> crate::Result<MusicalContext> {
        // Convert score to text representation for LLM
        let score_text = self.score_to_text(score);

        // Create cache key
        let cache_key = format!("{:?}_{:?}", score_text, lyrics);

        // Check cache if enabled
        if self.config.enable_caching {
            if let Some(cached_context) = self.context_cache.get(&cache_key) {
                return Ok(cached_context.clone());
            }
        }

        // Create musical prompt
        let prompt = MusicalPrompt {
            score: score_text.clone(),
            lyrics: lyrics.map(String::from),
            style_hint: None,
            emotion_hint: None,
            performance_instructions: None,
        };

        // Analyze with LLM (simulated for now - in production, this would call actual LLM API)
        let context = self.simulate_llm_analysis(&prompt)?;

        // Cache the result
        if self.config.enable_caching {
            self.context_cache.insert(cache_key, context.clone());
        }

        Ok(context)
    }

    /// Batch analyze multiple scores for efficient processing
    ///
    /// # Arguments
    /// * `scores` - List of musical scores to analyze
    ///
    /// # Returns
    /// List of musical contexts for each score
    pub async fn batch_analyze_scores(
        &mut self,
        scores: &[(MusicalScore, Option<String>)],
    ) -> crate::Result<Vec<MusicalContext>> {
        let mut results = Vec::with_capacity(scores.len());

        for (score, lyrics) in scores {
            let context = self
                .analyze_score(score, lyrics.as_ref().map(|s| s.as_str()))
                .await?;
            results.push(context);
        }

        Ok(results)
    }

    /// Create advanced musical prompt with custom instructions
    ///
    /// # Arguments
    /// * `score` - Musical score
    /// * `prompt_config` - Custom prompt configuration
    ///
    /// # Returns
    /// Customized musical prompt for LLM
    pub fn create_advanced_prompt(
        &self,
        score: &MusicalScore,
        prompt_config: &AdvancedPromptConfig,
    ) -> MusicalPrompt {
        let score_text = self.score_to_text(score);

        MusicalPrompt {
            score: score_text,
            lyrics: prompt_config.lyrics.clone(),
            style_hint: Some(prompt_config.style.clone()),
            emotion_hint: Some(prompt_config.emotion.clone()),
            performance_instructions: Some(prompt_config.instructions.clone()),
        }
    }

    /// Clear cache to free memory
    pub fn clear_cache(&mut self) {
        self.context_cache.clear();
        self.semantic_embeddings.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        CacheStats {
            context_cache_size: self.context_cache.len(),
            embedding_cache_size: self.semantic_embeddings.len(),
            total_memory_estimate: self.estimate_memory_usage(),
        }
    }

    /// Estimate memory usage of caches
    fn estimate_memory_usage(&self) -> usize {
        let context_size = self.context_cache.len() * std::mem::size_of::<MusicalContext>();
        let embedding_size = self.semantic_embeddings.len()
            * (std::mem::size_of::<String>() + self.config.embedding_dim * 4);
        context_size + embedding_size
    }

    /// Generate semantic embeddings for musical concepts
    ///
    /// # Arguments
    /// * `concept` - Musical concept to embed (e.g., "joyful melody", "dramatic climax")
    ///
    /// # Returns
    /// Vector embedding representing the concept
    pub fn generate_embedding(&mut self, concept: &str) -> Vec<f32> {
        // Check cache
        if let Some(embedding) = self.semantic_embeddings.get(concept) {
            return embedding.clone();
        }

        // Generate embedding (simulated - in production, use actual embedding model)
        let embedding = self.simulate_embedding_generation(concept);

        // Cache the embedding
        self.semantic_embeddings
            .insert(concept.to_string(), embedding.clone());

        embedding
    }

    /// Find similar musical concepts using embeddings
    ///
    /// # Arguments
    /// * `concept` - Source musical concept
    /// * `candidates` - List of candidate concepts to compare
    ///
    /// # Returns
    /// List of concepts sorted by similarity (most similar first)
    pub fn find_similar_concepts(
        &mut self,
        concept: &str,
        candidates: &[&str],
    ) -> Vec<(String, f32)> {
        let source_embedding = self.generate_embedding(concept);

        let mut similarities: Vec<(String, f32)> = candidates
            .iter()
            .map(|&candidate| {
                let candidate_embedding = self.generate_embedding(candidate);
                let similarity = self.cosine_similarity(&source_embedding, &candidate_embedding);
                (candidate.to_string(), similarity)
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        similarities
    }

    /// Interpret natural language musical instructions
    ///
    /// # Arguments
    /// * `instruction` - Natural language instruction (e.g., "make it more dramatic")
    /// * `current_score` - Current musical score
    ///
    /// # Returns
    /// Modified musical parameters based on instruction
    pub async fn interpret_instruction(
        &self,
        instruction: &str,
        current_score: &MusicalScore,
    ) -> crate::Result<InstructionInterpretation> {
        // Simulate LLM interpretation (in production, use actual LLM)
        let interpretation =
            self.simulate_instruction_interpretation(instruction, current_score)?;

        Ok(interpretation)
    }

    /// Generate musical commentary and analysis
    ///
    /// # Arguments
    /// * `score` - Musical score to analyze
    ///
    /// # Returns
    /// Detailed textual analysis of the score
    pub async fn generate_commentary(&self, score: &MusicalScore) -> crate::Result<String> {
        let score_text = self.score_to_text(score);

        // Simulate LLM commentary generation
        let time_signature = format!(
            "{}/{}",
            score.time_signature.numerator, score.time_signature.denominator
        );
        let key_root = format!("{:?}", score.key_signature.root);
        let key_mode = match score.key_signature.mode {
            crate::score::Mode::Major => "major",
            crate::score::Mode::Minor => "minor",
            _ => "other",
        };
        let character = self.infer_character(&score_text);
        let phrasing_style = self.infer_phrasing_style(&score_text);

        let commentary = format!(
            "Musical Analysis:\n\
            The composition features {} notes.\n\
            Tempo: {} BPM in {} time.\n\
            Key: {} {}.\n\
            The melodic contour suggests a {} character with {} phrasing.",
            score.notes.len(),
            score.tempo,
            time_signature,
            key_root,
            key_mode,
            character,
            phrasing_style,
        );

        Ok(commentary)
    }

    // ===== Internal Helper Methods =====

    /// Convert musical score to text representation
    fn score_to_text(&self, score: &MusicalScore) -> String {
        format!(
            "Tempo: {} BPM\nTime: {}/{}\nKey: {:?} {:?}\nNotes: {}",
            score.tempo,
            score.time_signature.numerator,
            score.time_signature.denominator,
            score.key_signature.root,
            score.key_signature.mode,
            score.notes.len()
        )
    }

    /// Simulate LLM analysis (placeholder for actual LLM API call)
    fn simulate_llm_analysis(&self, prompt: &MusicalPrompt) -> crate::Result<MusicalContext> {
        // In production, this would make an actual API call to LLM
        // For now, use rule-based heuristics

        let emotion = if let Some(lyrics) = &prompt.lyrics {
            if lyrics.to_lowercase().contains("love") || lyrics.to_lowercase().contains("heart") {
                "romantic"
            } else if lyrics.to_lowercase().contains("sad") || lyrics.to_lowercase().contains("cry")
            {
                "melancholic"
            } else {
                "neutral"
            }
        } else {
            "instrumental"
        }
        .to_string();

        Ok(MusicalContext {
            emotion,
            style: "contemporary".to_string(),
            narrative: "A flowing melody with expressive phrasing".to_string(),
            tempo_markings: vec!["moderato".to_string(), "espressivo".to_string()],
            dynamic_suggestions: vec![
                DynamicSuggestion {
                    beat: 0.0,
                    dynamics: Dynamics::MezzoForte,
                    reason: "Opening phrase - moderate intensity".to_string(),
                },
                DynamicSuggestion {
                    beat: 8.0,
                    dynamics: Dynamics::Forte,
                    reason: "Climactic moment - increase energy".to_string(),
                },
            ],
            articulation_recommendations: vec![],
            phrasing_boundaries: vec![PhrasingBoundary {
                beat: 4.0,
                boundary_type: "breath".to_string(),
                breath_duration: Some(0.3),
            }],
            confidence: 0.85,
        })
    }

    /// Simulate embedding generation (placeholder for actual embedding model)
    fn simulate_embedding_generation(&self, concept: &str) -> Vec<f32> {
        // In production, use actual embedding model (e.g., sentence-transformers)
        // For now, generate deterministic pseudo-random embeddings based on concept hash

        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();

        // Use concept hash as seed for deterministic embeddings
        let seed: u64 = concept.bytes().map(|b| b as u64).sum();
        let mut seeded_rng = scirs2_core::random::StdRng::seed_from_u64(seed);

        (0..self.config.embedding_dim)
            .map(|_| seeded_rng.gen_range(-1.0..1.0))
            .collect()
    }

    /// Calculate cosine similarity between two embeddings
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a * magnitude_b)
    }

    /// Simulate instruction interpretation
    fn simulate_instruction_interpretation(
        &self,
        instruction: &str,
        _score: &MusicalScore,
    ) -> crate::Result<InstructionInterpretation> {
        let instruction_lower = instruction.to_lowercase();

        let mut tempo_change = 0.0;
        let mut dynamics_change = 0.0;
        let mut expression_change = Expression::Neutral;

        if instruction_lower.contains("faster") || instruction_lower.contains("quicker") {
            tempo_change = 10.0;
        } else if instruction_lower.contains("slower") {
            tempo_change = -10.0;
        }

        if instruction_lower.contains("louder") || instruction_lower.contains("dramatic") {
            dynamics_change = 0.2;
            expression_change = Expression::Dramatic;
        } else if instruction_lower.contains("softer") || instruction_lower.contains("gentle") {
            dynamics_change = -0.2;
            expression_change = Expression::Calm;
        }

        Ok(InstructionInterpretation {
            tempo_change,
            dynamics_change,
            expression_change,
            articulation_changes: vec![],
            interpretation: format!("Interpreted: {}", instruction),
            confidence: 0.8,
        })
    }

    /// Infer musical character from score text
    fn infer_character(&self, score_text: &str) -> &'static str {
        if score_text.contains("Forte") || score_text.contains("Allegro") {
            "energetic"
        } else if score_text.contains("Piano") || score_text.contains("Adagio") {
            "gentle"
        } else {
            "balanced"
        }
    }

    /// Infer phrasing style from score text
    fn infer_phrasing_style(&self, _score_text: &str) -> &'static str {
        "legato"
    }
}

impl Default for LlmMusicalUnderstanding {
    fn default() -> Self {
        Self::new(LlmConfig::default())
    }
}

/// Interpretation of natural language instructions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionInterpretation {
    /// Tempo change in BPM
    pub tempo_change: f32,
    /// Dynamics change (-1.0 to 1.0)
    pub dynamics_change: f32,
    /// Expression change
    pub expression_change: Expression,
    /// Articulation changes
    pub articulation_changes: Vec<(usize, Articulation)>,
    /// Textual interpretation
    pub interpretation: String,
    /// Confidence score
    pub confidence: f32,
}

/// Advanced prompt configuration for LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedPromptConfig {
    /// Lyrics text
    pub lyrics: Option<String>,
    /// Musical style description
    pub style: String,
    /// Emotional context
    pub emotion: String,
    /// Performance instructions
    pub instructions: String,
    /// Additional context
    pub context: HashMap<String, String>,
}

impl Default for AdvancedPromptConfig {
    fn default() -> Self {
        Self {
            lyrics: None,
            style: "contemporary".to_string(),
            emotion: "neutral".to_string(),
            instructions: "Perform naturally with appropriate expression".to_string(),
            context: HashMap::new(),
        }
    }
}

/// Cache statistics for LLM understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cached contexts
    pub context_cache_size: usize,
    /// Number of cached embeddings
    pub embedding_cache_size: usize,
    /// Estimated memory usage in bytes
    pub total_memory_estimate: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{KeySignature, Mode, Note, TimeSignature};

    #[test]
    fn test_llm_understanding_creation() {
        let config = LlmConfig::default();
        let llm = LlmMusicalUnderstanding::new(config);

        assert_eq!(llm.config.model_type, "gpt-4");
        assert_eq!(llm.config.embedding_dim, 768);
    }

    #[test]
    fn test_embedding_generation() {
        let mut llm = LlmMusicalUnderstanding::default();

        let embedding1 = llm.generate_embedding("joyful melody");
        let embedding2 = llm.generate_embedding("joyful melody");

        // Same concept should generate same embedding
        assert_eq!(embedding1.len(), embedding2.len());
        assert_eq!(embedding1, embedding2);
    }

    #[test]
    fn test_cosine_similarity() {
        let llm = LlmMusicalUnderstanding::default();

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        // Identical vectors
        assert!((llm.cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);

        // Orthogonal vectors
        assert!((llm.cosine_similarity(&a, &c) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_find_similar_concepts() {
        let mut llm = LlmMusicalUnderstanding::default();

        let candidates = vec!["happy tune", "sad song", "energetic rhythm"];
        let similar = llm.find_similar_concepts("joyful melody", &candidates);

        assert_eq!(similar.len(), 3);
        // Verify all similarities are between -1 and 1
        for (_, similarity) in &similar {
            assert!(*similarity >= -1.0 && *similarity <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_score_analysis() {
        let mut llm = LlmMusicalUnderstanding::default();

        let score = MusicalScore {
            title: "Test Score".to_string(),
            composer: "Test Composer".to_string(),
            tempo: 120.0,
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            key_signature: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            notes: vec![],
            lyrics: None,
            metadata: std::collections::HashMap::new(),
            duration: std::time::Duration::from_secs(0),
            sections: vec![],
            markers: vec![],
            breath_marks: vec![],
            dynamics: vec![],
            expressions: vec![],
        };

        let context = llm.analyze_score(&score, Some("I love you")).await.unwrap();

        assert!(context.confidence > 0.0);
        assert!(!context.emotion.is_empty());
    }

    #[tokio::test]
    async fn test_instruction_interpretation() {
        let llm = LlmMusicalUnderstanding::default();

        let score = MusicalScore {
            title: "Test Score".to_string(),
            composer: "Test Composer".to_string(),
            tempo: 120.0,
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            key_signature: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            notes: vec![],
            lyrics: None,
            metadata: std::collections::HashMap::new(),
            duration: std::time::Duration::from_secs(0),
            sections: vec![],
            markers: vec![],
            breath_marks: vec![],
            dynamics: vec![],
            expressions: vec![],
        };

        let interpretation = llm
            .interpret_instruction("make it louder and more dramatic", &score)
            .await
            .unwrap();

        assert!(interpretation.dynamics_change > 0.0);
        assert!(matches!(
            interpretation.expression_change,
            Expression::Dramatic
        ));
    }

    #[tokio::test]
    async fn test_commentary_generation() {
        let llm = LlmMusicalUnderstanding::default();

        let score = MusicalScore {
            title: "Test Score".to_string(),
            composer: "Test Composer".to_string(),
            tempo: 120.0,
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            key_signature: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            notes: vec![],
            lyrics: None,
            metadata: std::collections::HashMap::new(),
            duration: std::time::Duration::from_secs(0),
            sections: vec![],
            markers: vec![],
            breath_marks: vec![],
            dynamics: vec![],
            expressions: vec![],
        };

        let commentary = llm.generate_commentary(&score).await.unwrap();

        assert!(!commentary.is_empty());
        assert!(commentary.contains("Musical Analysis"));
    }

    #[test]
    fn test_config_defaults() {
        let config = LlmConfig::default();

        assert_eq!(config.model_type, "gpt-4");
        assert_eq!(config.max_context_length, 8192);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
        assert!(config.enable_caching);
    }

    #[tokio::test]
    async fn test_batch_analysis() {
        let mut llm = LlmMusicalUnderstanding::default();

        let score1 = MusicalScore {
            title: "Song 1".to_string(),
            composer: "Composer 1".to_string(),
            tempo: 120.0,
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            key_signature: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            notes: vec![],
            lyrics: None,
            metadata: std::collections::HashMap::new(),
            duration: std::time::Duration::from_secs(0),
            sections: vec![],
            markers: vec![],
            breath_marks: vec![],
            dynamics: vec![],
            expressions: vec![],
        };

        let score2 = score1.clone();
        let scores = vec![
            (score1, Some("Love song".to_string())),
            (score2, Some("Happy song".to_string())),
        ];

        let results = llm.batch_analyze_scores(&scores).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.confidence > 0.0));
    }

    #[test]
    fn test_cache_management() {
        let mut llm = LlmMusicalUnderstanding::default();

        // Generate some embeddings to populate cache
        llm.generate_embedding("test1");
        llm.generate_embedding("test2");

        let stats = llm.get_cache_stats();
        assert_eq!(stats.embedding_cache_size, 2);
        assert!(stats.total_memory_estimate > 0);

        // Clear cache
        llm.clear_cache();
        let stats_after = llm.get_cache_stats();
        assert_eq!(stats_after.embedding_cache_size, 0);
    }

    #[test]
    fn test_advanced_prompt_config() {
        let config = AdvancedPromptConfig::default();
        assert_eq!(config.style, "contemporary");
        assert_eq!(config.emotion, "neutral");

        let llm = LlmMusicalUnderstanding::default();
        let score = MusicalScore {
            title: "Test".to_string(),
            composer: "Test".to_string(),
            tempo: 120.0,
            time_signature: TimeSignature {
                numerator: 4,
                denominator: 4,
            },
            key_signature: KeySignature {
                root: Note::C,
                mode: Mode::Major,
                accidentals: 0,
            },
            notes: vec![],
            lyrics: None,
            metadata: std::collections::HashMap::new(),
            duration: std::time::Duration::from_secs(0),
            sections: vec![],
            markers: vec![],
            breath_marks: vec![],
            dynamics: vec![],
            expressions: vec![],
        };

        let prompt = llm.create_advanced_prompt(&score, &config);
        assert!(prompt.style_hint.is_some());
        assert!(prompt.emotion_hint.is_some());
    }
}
