//! Predefined emotion presets and emotion library

use crate::{
    types::{Emotion, EmotionIntensity, EmotionParameters, EmotionVector},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Predefined emotion preset
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionPreset {
    /// Preset name
    pub name: String,
    /// Description of the emotion
    pub description: String,
    /// Emotion parameters
    pub parameters: EmotionParameters,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Intensity level recommendation
    pub recommended_intensity: EmotionIntensity,
}

impl EmotionPreset {
    /// Create a new emotion preset
    pub fn new(name: String, description: String, parameters: EmotionParameters) -> Self {
        Self {
            name,
            description,
            parameters,
            tags: Vec::new(),
            recommended_intensity: EmotionIntensity::MEDIUM,
        }
    }

    /// Add tags to the preset
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set recommended intensity
    pub fn with_intensity(mut self, intensity: EmotionIntensity) -> Self {
        self.recommended_intensity = intensity;
        self
    }

    /// Check if preset has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Get parameters with applied intensity scaling
    pub fn get_scaled_parameters(&self, intensity_scale: f32) -> EmotionParameters {
        let mut params = self.parameters.clone();

        // Scale emotion vector intensities
        for emotion_intensity in params.emotion_vector.emotions.values_mut() {
            *emotion_intensity = EmotionIntensity::new(emotion_intensity.value() * intensity_scale);
        }

        // Scale prosody parameters
        let base_scale = intensity_scale.clamp(0.0, 1.0);
        params.pitch_shift = 1.0 + (params.pitch_shift - 1.0) * base_scale;
        params.tempo_scale = 1.0 + (params.tempo_scale - 1.0) * base_scale;
        params.energy_scale = 1.0 + (params.energy_scale - 1.0) * base_scale;
        params.breathiness *= base_scale;
        params.roughness *= base_scale;

        params
    }
}

/// Library of emotion presets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionPresetLibrary {
    /// All presets in the library
    presets: HashMap<String, EmotionPreset>,
    /// Categories for organization
    categories: HashMap<String, Vec<String>>,
}

impl EmotionPresetLibrary {
    /// Create a new preset library with default presets
    ///
    /// This is the recommended way to create a preset library as it comes
    /// with a comprehensive set of predefined emotion presets.
    pub fn new() -> Self {
        let mut library = Self::empty();
        library.add_default_presets();
        library
    }

    /// Create an empty preset library with no presets
    ///
    /// Use this when you want to build a completely custom preset library
    /// without any predefined emotions.
    pub fn empty() -> Self {
        Self {
            presets: HashMap::new(),
            categories: HashMap::new(),
        }
    }

    /// Create library with default presets
    ///
    /// **Deprecated**: Use `new()` instead, which now includes default presets
    #[deprecated(since = "0.1.0", note = "Use `new()` instead")]
    pub fn with_defaults() -> Self {
        Self::new()
    }

    /// Add a preset to the library
    pub fn add_preset(&mut self, preset: EmotionPreset) {
        self.presets.insert(preset.name.clone(), preset);
    }

    /// Get a preset by name
    pub fn get_preset(&self, name: &str) -> Option<&EmotionPreset> {
        self.presets.get(name)
    }

    /// Get preset parameters with intensity scaling
    pub fn get_preset_parameters(
        &self,
        name: &str,
        intensity: Option<f32>,
    ) -> Option<EmotionParameters> {
        if let Some(preset) = self.get_preset(name) {
            let scale = intensity.unwrap_or(preset.recommended_intensity.value());
            Some(preset.get_scaled_parameters(scale))
        } else {
            None
        }
    }

    /// Remove a preset
    pub fn remove_preset(&mut self, name: &str) -> Option<EmotionPreset> {
        self.presets.remove(name)
    }

    /// List all preset names
    pub fn list_presets(&self) -> Vec<String> {
        self.presets.keys().cloned().collect()
    }

    /// Find presets by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<&EmotionPreset> {
        self.presets
            .values()
            .filter(|preset| preset.has_tag(tag))
            .collect()
    }

    /// Find presets by emotion type
    pub fn find_by_emotion(&self, emotion: Emotion) -> Vec<&EmotionPreset> {
        self.presets
            .values()
            .filter(|preset| {
                preset
                    .parameters
                    .emotion_vector
                    .emotions
                    .contains_key(&emotion)
            })
            .collect()
    }

    /// Add presets to a category
    pub fn add_category(&mut self, category: String, preset_names: Vec<String>) {
        self.categories.insert(category, preset_names);
    }

    /// Get presets in a category
    pub fn get_category(&self, category: &str) -> Vec<&EmotionPreset> {
        if let Some(names) = self.categories.get(category) {
            names
                .iter()
                .filter_map(|name| self.presets.get(name))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// List all categories
    pub fn list_categories(&self) -> Vec<String> {
        self.categories.keys().cloned().collect()
    }

    /// Add default emotion presets
    fn add_default_presets(&mut self) {
        // Basic emotions
        self.add_basic_emotions();

        // Complex emotions
        self.add_complex_emotions();

        // Situational emotions
        self.add_situational_emotions();

        // Voice acting presets
        self.add_voice_acting_presets();

        // Extended emotions
        self.add_extended_emotions();

        // Professional presets
        self.add_professional_presets();

        // Age-related presets
        self.add_age_related_presets();

        // Intensity variations
        self.add_intensity_variations();

        // Cultural variations
        self.add_cultural_variations();

        // Setup categories
        self.setup_default_categories();
    }

    /// Add fundamental emotion presets based on Ekman's basic emotions.
    ///
    /// Creates presets for the primary emotions that are universally recognized
    /// across cultures: happiness, sadness, anger, fear, and calm (a composite emotion
    /// for emotional baseline). Each preset includes culturally neutral prosody
    /// parameters and voice quality settings.
    ///
    /// # Presets Added
    /// - **happy**: Joyful, upbeat with increased pitch/tempo/energy
    /// - **sad**: Melancholy with breathiness, reduced pitch/tempo/energy  
    /// - **angry**: Intense with roughness, elevated pitch/tempo/energy
    /// - **fear**: Anxious with tremor, high pitch/tempo, moderate energy
    /// - **calm**: Peaceful baseline with slight prosody reduction and smoothness
    fn add_basic_emotions(&mut self) {
        // Happy
        let mut happy_vector = EmotionVector::new();
        happy_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        let happy_params = EmotionParameters::new(happy_vector).with_prosody(1.2, 1.1, 1.3);
        self.add_preset(
            EmotionPreset::new(
                "happy".to_string(),
                "Joyful and upbeat emotion".to_string(),
                happy_params,
            )
            .with_tags(vec!["basic".to_string(), "positive".to_string()]),
        );

        // Sad
        let mut sad_vector = EmotionVector::new();
        sad_vector.add_emotion(Emotion::Sad, EmotionIntensity::HIGH);
        let sad_params = EmotionParameters::new(sad_vector)
            .with_prosody(0.8, 0.7, 0.6)
            .with_custom_param("breathiness".to_string(), 0.4);
        self.add_preset(
            EmotionPreset::new(
                "sad".to_string(),
                "Melancholy and sorrowful emotion".to_string(),
                sad_params,
            )
            .with_tags(vec!["basic".to_string(), "negative".to_string()]),
        );

        // Angry
        let mut angry_vector = EmotionVector::new();
        angry_vector.add_emotion(Emotion::Angry, EmotionIntensity::HIGH);
        let angry_params = EmotionParameters::new(angry_vector)
            .with_prosody(1.4, 1.3, 1.5)
            .with_custom_param("roughness".to_string(), 0.5);
        self.add_preset(
            EmotionPreset::new(
                "angry".to_string(),
                "Intense anger and frustration".to_string(),
                angry_params,
            )
            .with_tags(vec![
                "basic".to_string(),
                "negative".to_string(),
                "intense".to_string(),
            ]),
        );

        // Fear
        let mut fear_vector = EmotionVector::new();
        fear_vector.add_emotion(Emotion::Fear, EmotionIntensity::HIGH);
        let fear_params = EmotionParameters::new(fear_vector)
            .with_prosody(1.5, 1.4, 1.2)
            .with_custom_param("tremor".to_string(), 0.3);
        self.add_preset(
            EmotionPreset::new(
                "fear".to_string(),
                "Anxious and fearful emotion".to_string(),
                fear_params,
            )
            .with_tags(vec![
                "basic".to_string(),
                "negative".to_string(),
                "intense".to_string(),
            ]),
        );

        // Calm
        let mut calm_vector = EmotionVector::new();
        calm_vector.add_emotion(Emotion::Calm, EmotionIntensity::HIGH);
        let calm_params = EmotionParameters::new(calm_vector)
            .with_prosody(0.95, 0.9, 0.85)
            .with_custom_param("smoothness".to_string(), 0.4);
        self.add_preset(
            EmotionPreset::new(
                "calm".to_string(),
                "Peaceful and relaxed emotion".to_string(),
                calm_params,
            )
            .with_tags(vec![
                "basic".to_string(),
                "positive".to_string(),
                "relaxed".to_string(),
            ]),
        );
    }

    fn add_complex_emotions(&mut self) {
        // Bittersweet (happy + sad)
        let mut bittersweet_vector = EmotionVector::new();
        bittersweet_vector.add_emotion(Emotion::Happy, EmotionIntensity::MEDIUM);
        bittersweet_vector.add_emotion(Emotion::Sad, EmotionIntensity::MEDIUM);
        let bittersweet_params =
            EmotionParameters::new(bittersweet_vector).with_prosody(1.0, 0.9, 0.9);
        self.add_preset(
            EmotionPreset::new(
                "bittersweet".to_string(),
                "Mixed feelings of joy and sorrow".to_string(),
                bittersweet_params,
            )
            .with_tags(vec!["complex".to_string(), "mixed".to_string()]),
        );

        // Triumphant (happy + confident)
        let mut triumphant_vector = EmotionVector::new();
        triumphant_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        triumphant_vector.add_emotion(Emotion::Confident, EmotionIntensity::HIGH);
        let triumphant_params =
            EmotionParameters::new(triumphant_vector).with_prosody(1.3, 1.2, 1.4);
        self.add_preset(
            EmotionPreset::new(
                "triumphant".to_string(),
                "Victorious and proud emotion".to_string(),
                triumphant_params,
            )
            .with_tags(vec![
                "complex".to_string(),
                "positive".to_string(),
                "intense".to_string(),
            ]),
        );

        // Melancholic nostalgia
        let mut nostalgic_vector = EmotionVector::new();
        nostalgic_vector.add_emotion(Emotion::Melancholic, EmotionIntensity::MEDIUM);
        nostalgic_vector.add_emotion(Emotion::Tender, EmotionIntensity::LOW);
        let nostalgic_params = EmotionParameters::new(nostalgic_vector)
            .with_prosody(0.9, 0.8, 0.8)
            .with_custom_param("warmth".to_string(), 0.3);
        self.add_preset(
            EmotionPreset::new(
                "nostalgic".to_string(),
                "Wistful longing for the past".to_string(),
                nostalgic_params,
            )
            .with_tags(vec!["complex".to_string(), "reflective".to_string()]),
        );
    }

    fn add_situational_emotions(&mut self) {
        // Excited anticipation
        let mut excited_vector = EmotionVector::new();
        excited_vector.add_emotion(Emotion::Excited, EmotionIntensity::HIGH);
        excited_vector.add_emotion(Emotion::Happy, EmotionIntensity::MEDIUM);
        let excited_params = EmotionParameters::new(excited_vector).with_prosody(1.4, 1.5, 1.6);
        self.add_preset(
            EmotionPreset::new(
                "excited".to_string(),
                "High energy and enthusiasm".to_string(),
                excited_params,
            )
            .with_tags(vec![
                "situational".to_string(),
                "positive".to_string(),
                "energetic".to_string(),
            ]),
        );

        // Gentle comfort
        let mut gentle_vector = EmotionVector::new();
        gentle_vector.add_emotion(Emotion::Tender, EmotionIntensity::HIGH);
        gentle_vector.add_emotion(Emotion::Calm, EmotionIntensity::MEDIUM);
        let gentle_params = EmotionParameters::new(gentle_vector)
            .with_prosody(0.9, 0.8, 0.7)
            .with_custom_param("softness".to_string(), 0.5);
        self.add_preset(
            EmotionPreset::new(
                "gentle".to_string(),
                "Soft and comforting emotion".to_string(),
                gentle_params,
            )
            .with_tags(vec![
                "situational".to_string(),
                "positive".to_string(),
                "soft".to_string(),
            ]),
        );

        // Mysterious
        let mut mysterious_vector = EmotionVector::new();
        mysterious_vector.add_emotion(Emotion::Neutral, EmotionIntensity::MEDIUM);
        let mysterious_params = EmotionParameters::new(mysterious_vector)
            .with_prosody(0.8, 0.7, 0.6)
            .with_custom_param("whisper".to_string(), 0.3);
        self.add_preset(
            EmotionPreset::new(
                "mysterious".to_string(),
                "Enigmatic and secretive tone".to_string(),
                mysterious_params,
            )
            .with_tags(vec!["situational".to_string(), "atmospheric".to_string()]),
        );
    }

    fn add_voice_acting_presets(&mut self) {
        // Narrator
        let mut narrator_vector = EmotionVector::new();
        narrator_vector.add_emotion(Emotion::Neutral, EmotionIntensity::HIGH);
        narrator_vector.add_emotion(Emotion::Confident, EmotionIntensity::LOW);
        let narrator_params = EmotionParameters::new(narrator_vector).with_prosody(1.0, 0.95, 1.0);
        self.add_preset(
            EmotionPreset::new(
                "narrator".to_string(),
                "Professional storytelling voice".to_string(),
                narrator_params,
            )
            .with_tags(vec!["voice_acting".to_string(), "professional".to_string()]),
        );

        // Villain
        let mut villain_vector = EmotionVector::new();
        villain_vector.add_emotion(Emotion::Angry, EmotionIntensity::MEDIUM);
        villain_vector.add_emotion(Emotion::Confident, EmotionIntensity::HIGH);
        let villain_params = EmotionParameters::new(villain_vector)
            .with_prosody(0.8, 0.9, 1.2)
            .with_custom_param("menace".to_string(), 0.4);
        self.add_preset(
            EmotionPreset::new(
                "villain".to_string(),
                "Menacing antagonist voice".to_string(),
                villain_params,
            )
            .with_tags(vec!["voice_acting".to_string(), "character".to_string()]),
        );

        // Child-like
        let mut childlike_vector = EmotionVector::new();
        childlike_vector.add_emotion(Emotion::Excited, EmotionIntensity::MEDIUM);
        childlike_vector.add_emotion(Emotion::Happy, EmotionIntensity::MEDIUM);
        let childlike_params = EmotionParameters::new(childlike_vector)
            .with_prosody(1.3, 1.2, 1.1)
            .with_custom_param("innocence".to_string(), 0.6);
        self.add_preset(
            EmotionPreset::new(
                "childlike".to_string(),
                "Innocent and playful voice".to_string(),
                childlike_params,
            )
            .with_tags(vec![
                "voice_acting".to_string(),
                "character".to_string(),
                "youthful".to_string(),
            ]),
        );
    }

    /// Add extended emotion presets for nuanced emotional expression.
    ///
    /// Creates presets for more complex emotions that go beyond basic categories,
    /// including psychological states, social emotions, and compound feelings.
    /// These presets enable more sophisticated emotional expression in synthesis.
    ///
    /// # Presets Added
    /// - **anxiety**: Fear + sadness combination with tension effects
    /// - **disgust**: Strong distaste with nasal voice quality modifications  
    /// - **contempt**: Angry + confident blend with smugness parameters
    /// - **surprise**: Sudden reaction with gasping effects and high arousal
    /// - **pride**: Confident + happy with chest resonance enhancement
    /// - **shame**: Sad + fear with hesitation and reduced energy
    fn add_extended_emotions(&mut self) {
        // Anxiety (fear + worried)
        let mut anxiety_vector = EmotionVector::new();
        anxiety_vector.add_emotion(Emotion::Fear, EmotionIntensity::MEDIUM);
        anxiety_vector.add_emotion(Emotion::Sad, EmotionIntensity::LOW);
        let anxiety_params = EmotionParameters::new(anxiety_vector)
            .with_prosody(1.2, 1.1, 0.8)
            .with_custom_param("tension".to_string(), 0.4);
        self.add_preset(
            EmotionPreset::new(
                "anxiety".to_string(),
                "Worried and apprehensive emotion".to_string(),
                anxiety_params,
            )
            .with_tags(vec!["extended".to_string(), "negative".to_string()]),
        );

        // Disgust
        let mut disgust_vector = EmotionVector::new();
        disgust_vector.add_emotion(Emotion::Disgust, EmotionIntensity::HIGH);
        let disgust_params = EmotionParameters::new(disgust_vector)
            .with_prosody(0.7, 0.8, 1.1)
            .with_custom_param("nasal".to_string(), 0.3);
        self.add_preset(
            EmotionPreset::new(
                "disgust".to_string(),
                "Strong distaste and revulsion".to_string(),
                disgust_params,
            )
            .with_tags(vec!["extended".to_string(), "negative".to_string()]),
        );

        // Contempt (using angry + confident combination)
        let mut contempt_vector = EmotionVector::new();
        contempt_vector.add_emotion(Emotion::Angry, EmotionIntensity::MEDIUM);
        contempt_vector.add_emotion(Emotion::Confident, EmotionIntensity::HIGH);
        let contempt_params = EmotionParameters::new(contempt_vector)
            .with_prosody(0.85, 0.9, 1.2)
            .with_custom_param("smugness".to_string(), 0.4);
        self.add_preset(
            EmotionPreset::new(
                "contempt".to_string(),
                "Disdainful and superior attitude".to_string(),
                contempt_params,
            )
            .with_tags(vec!["extended".to_string(), "negative".to_string()]),
        );

        // Surprise
        let mut surprise_vector = EmotionVector::new();
        surprise_vector.add_emotion(Emotion::Surprise, EmotionIntensity::HIGH);
        let surprise_params = EmotionParameters::new(surprise_vector)
            .with_prosody(1.3, 1.4, 1.3)
            .with_custom_param("gasp".to_string(), 0.2);
        self.add_preset(
            EmotionPreset::new(
                "surprise".to_string(),
                "Sudden unexpected reaction".to_string(),
                surprise_params,
            )
            .with_tags(vec!["extended".to_string(), "reactive".to_string()]),
        );

        // Pride
        let mut pride_vector = EmotionVector::new();
        pride_vector.add_emotion(Emotion::Confident, EmotionIntensity::HIGH);
        pride_vector.add_emotion(Emotion::Happy, EmotionIntensity::MEDIUM);
        let pride_params = EmotionParameters::new(pride_vector)
            .with_prosody(1.1, 1.0, 1.3)
            .with_custom_param("chest_resonance".to_string(), 0.3);
        self.add_preset(
            EmotionPreset::new(
                "pride".to_string(),
                "Dignified self-satisfaction".to_string(),
                pride_params,
            )
            .with_tags(vec!["extended".to_string(), "positive".to_string()]),
        );

        // Shame
        let mut shame_vector = EmotionVector::new();
        shame_vector.add_emotion(Emotion::Sad, EmotionIntensity::MEDIUM);
        shame_vector.add_emotion(Emotion::Fear, EmotionIntensity::LOW);
        let shame_params = EmotionParameters::new(shame_vector)
            .with_prosody(0.8, 0.7, 0.5)
            .with_custom_param("hesitation".to_string(), 0.4);
        self.add_preset(
            EmotionPreset::new(
                "shame".to_string(),
                "Embarrassed and self-conscious".to_string(),
                shame_params,
            )
            .with_tags(vec!["extended".to_string(), "negative".to_string()]),
        );
    }

    /// Add professional role-based emotion presets.
    ///
    /// Creates presets designed for specific professional contexts where emotional
    /// expression follows established patterns and expectations. These presets help
    /// create appropriate voices for different professional scenarios.
    ///
    /// # Presets Added
    /// - **teacher**: Educational tone with confidence + calm, enhanced clarity
    /// - **customer_service**: Helpful demeanor with happiness + calm, high politeness
    /// - **news_anchor**: Authoritative neutral + confident with professional delivery
    /// - **doctor**: Medical professional with calm + confident, reassuring tone
    ///
    /// # Usage Context
    /// These presets are particularly useful for:
    /// - Training and educational content
    /// - Customer service applications
    /// - News and media synthesis
    /// - Healthcare and medical applications
    fn add_professional_presets(&mut self) {
        // Teacher
        let mut teacher_vector = EmotionVector::new();
        teacher_vector.add_emotion(Emotion::Confident, EmotionIntensity::MEDIUM);
        teacher_vector.add_emotion(Emotion::Calm, EmotionIntensity::MEDIUM);
        let teacher_params = EmotionParameters::new(teacher_vector)
            .with_prosody(1.05, 0.95, 1.1)
            .with_custom_param("clarity".to_string(), 0.5);
        self.add_preset(
            EmotionPreset::new(
                "teacher".to_string(),
                "Educational and instructive tone".to_string(),
                teacher_params,
            )
            .with_tags(vec!["professional".to_string(), "educational".to_string()]),
        );

        // Customer Service
        let mut service_vector = EmotionVector::new();
        service_vector.add_emotion(Emotion::Happy, EmotionIntensity::MEDIUM);
        service_vector.add_emotion(Emotion::Calm, EmotionIntensity::HIGH);
        let service_params = EmotionParameters::new(service_vector)
            .with_prosody(1.1, 1.0, 1.0)
            .with_custom_param("politeness".to_string(), 0.6);
        self.add_preset(
            EmotionPreset::new(
                "customer_service".to_string(),
                "Helpful and accommodating tone".to_string(),
                service_params,
            )
            .with_tags(vec!["professional".to_string(), "service".to_string()]),
        );

        // News Anchor
        let mut anchor_vector = EmotionVector::new();
        anchor_vector.add_emotion(Emotion::Neutral, EmotionIntensity::HIGH);
        anchor_vector.add_emotion(Emotion::Confident, EmotionIntensity::MEDIUM);
        let anchor_params = EmotionParameters::new(anchor_vector)
            .with_prosody(1.0, 0.9, 1.0)
            .with_custom_param("authority".to_string(), 0.4);
        self.add_preset(
            EmotionPreset::new(
                "news_anchor".to_string(),
                "Professional news delivery".to_string(),
                anchor_params,
            )
            .with_tags(vec!["professional".to_string(), "media".to_string()]),
        );

        // Doctor
        let mut doctor_vector = EmotionVector::new();
        doctor_vector.add_emotion(Emotion::Calm, EmotionIntensity::HIGH);
        doctor_vector.add_emotion(Emotion::Confident, EmotionIntensity::MEDIUM);
        let doctor_params = EmotionParameters::new(doctor_vector)
            .with_prosody(0.95, 0.9, 0.9)
            .with_custom_param("reassurance".to_string(), 0.4);
        self.add_preset(
            EmotionPreset::new(
                "doctor".to_string(),
                "Medical professional demeanor".to_string(),
                doctor_params,
            )
            .with_tags(vec!["professional".to_string(), "medical".to_string()]),
        );
    }

    fn add_age_related_presets(&mut self) {
        // Elderly
        let mut elderly_vector = EmotionVector::new();
        elderly_vector.add_emotion(Emotion::Calm, EmotionIntensity::HIGH);
        elderly_vector.add_emotion(Emotion::Melancholic, EmotionIntensity::LOW);
        let elderly_params = EmotionParameters::new(elderly_vector)
            .with_prosody(0.85, 0.8, 0.9)
            .with_custom_param("wisdom".to_string(), 0.5);
        self.add_preset(
            EmotionPreset::new(
                "elderly".to_string(),
                "Aged and experienced voice".to_string(),
                elderly_params,
            )
            .with_tags(vec!["age_related".to_string(), "mature".to_string()]),
        );

        // Teenager
        let mut teen_vector = EmotionVector::new();
        teen_vector.add_emotion(Emotion::Excited, EmotionIntensity::MEDIUM);
        teen_vector.add_emotion(Emotion::Happy, EmotionIntensity::LOW);
        let teen_params = EmotionParameters::new(teen_vector)
            .with_prosody(1.2, 1.1, 1.2)
            .with_custom_param("enthusiasm".to_string(), 0.4);
        self.add_preset(
            EmotionPreset::new(
                "teenager".to_string(),
                "Youthful and energetic voice".to_string(),
                teen_params,
            )
            .with_tags(vec!["age_related".to_string(), "youthful".to_string()]),
        );

        // Toddler
        let mut toddler_vector = EmotionVector::new();
        toddler_vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        toddler_vector.add_emotion(Emotion::Excited, EmotionIntensity::MEDIUM);
        let toddler_params = EmotionParameters::new(toddler_vector)
            .with_prosody(1.5, 1.3, 1.1)
            .with_custom_param("playfulness".to_string(), 0.7);
        self.add_preset(
            EmotionPreset::new(
                "toddler".to_string(),
                "Very young child's voice".to_string(),
                toddler_params,
            )
            .with_tags(vec!["age_related".to_string(), "childish".to_string()]),
        );
    }

    fn add_intensity_variations(&mut self) {
        // Subtle happiness
        let mut subtle_happy_vector = EmotionVector::new();
        subtle_happy_vector.add_emotion(Emotion::Happy, EmotionIntensity::LOW);
        let subtle_happy_params = EmotionParameters::new(subtle_happy_vector)
            .with_prosody(1.05, 1.02, 1.1)
            .with_custom_param("subtle_warmth".to_string(), 0.2);
        self.add_preset(
            EmotionPreset::new(
                "subtle_happy".to_string(),
                "Gently positive and content".to_string(),
                subtle_happy_params,
            )
            .with_tags(vec![
                "intensity_variation".to_string(),
                "subtle".to_string(),
            ]),
        );

        // Overwhelming fear
        let mut overwhelming_fear_vector = EmotionVector::new();
        overwhelming_fear_vector.add_emotion(Emotion::Fear, EmotionIntensity::new(0.95));
        let overwhelming_fear_params = EmotionParameters::new(overwhelming_fear_vector)
            .with_prosody(1.8, 1.6, 1.4)
            .with_custom_param("panic".to_string(), 0.7);
        self.add_preset(
            EmotionPreset::new(
                "overwhelming_fear".to_string(),
                "Intense panic and terror".to_string(),
                overwhelming_fear_params,
            )
            .with_tags(vec![
                "intensity_variation".to_string(),
                "extreme".to_string(),
            ]),
        );

        // Mild sadness
        let mut mild_sad_vector = EmotionVector::new();
        mild_sad_vector.add_emotion(Emotion::Sad, EmotionIntensity::LOW);
        let mild_sad_params = EmotionParameters::new(mild_sad_vector)
            .with_prosody(0.95, 0.92, 0.9)
            .with_custom_param("wistfulness".to_string(), 0.2);
        self.add_preset(
            EmotionPreset::new(
                "mild_sadness".to_string(),
                "Slightly melancholy but controlled".to_string(),
                mild_sad_params,
            )
            .with_tags(vec!["intensity_variation".to_string(), "mild".to_string()]),
        );
    }

    fn add_cultural_variations(&mut self) {
        // Respectful (Japanese-inspired)
        let mut respectful_vector = EmotionVector::new();
        respectful_vector.add_emotion(Emotion::Calm, EmotionIntensity::HIGH);
        respectful_vector.add_emotion(Emotion::Neutral, EmotionIntensity::MEDIUM);
        let respectful_params = EmotionParameters::new(respectful_vector)
            .with_prosody(0.9, 0.85, 0.8)
            .with_custom_param("humility".to_string(), 0.5);
        self.add_preset(
            EmotionPreset::new(
                "respectful".to_string(),
                "Deferential and humble tone".to_string(),
                respectful_params,
            )
            .with_tags(vec!["cultural".to_string(), "polite".to_string()]),
        );

        // Passionate (Latin-inspired)
        let mut passionate_vector = EmotionVector::new();
        passionate_vector.add_emotion(Emotion::Excited, EmotionIntensity::HIGH);
        passionate_vector.add_emotion(Emotion::Happy, EmotionIntensity::MEDIUM);
        let passionate_params = EmotionParameters::new(passionate_vector)
            .with_prosody(1.3, 1.2, 1.5)
            .with_custom_param("fire".to_string(), 0.6);
        self.add_preset(
            EmotionPreset::new(
                "passionate".to_string(),
                "Fiery and emotionally intense".to_string(),
                passionate_params,
            )
            .with_tags(vec!["cultural".to_string(), "expressive".to_string()]),
        );

        // Reserved (Nordic-inspired)
        let mut reserved_vector = EmotionVector::new();
        reserved_vector.add_emotion(Emotion::Calm, EmotionIntensity::HIGH);
        reserved_vector.add_emotion(Emotion::Neutral, EmotionIntensity::HIGH);
        let reserved_params = EmotionParameters::new(reserved_vector)
            .with_prosody(0.95, 0.9, 0.85)
            .with_custom_param("stoicism".to_string(), 0.4);
        self.add_preset(
            EmotionPreset::new(
                "reserved".to_string(),
                "Controlled and understated".to_string(),
                reserved_params,
            )
            .with_tags(vec!["cultural".to_string(), "restrained".to_string()]),
        );
    }

    fn setup_default_categories(&mut self) {
        self.add_category(
            "basic_emotions".to_string(),
            vec![
                "happy".to_string(),
                "sad".to_string(),
                "angry".to_string(),
                "fear".to_string(),
                "calm".to_string(),
            ],
        );

        self.add_category(
            "complex_emotions".to_string(),
            vec![
                "bittersweet".to_string(),
                "triumphant".to_string(),
                "nostalgic".to_string(),
            ],
        );

        self.add_category(
            "situational".to_string(),
            vec![
                "excited".to_string(),
                "gentle".to_string(),
                "mysterious".to_string(),
            ],
        );

        self.add_category(
            "voice_acting".to_string(),
            vec![
                "narrator".to_string(),
                "villain".to_string(),
                "childlike".to_string(),
            ],
        );

        self.add_category(
            "extended_emotions".to_string(),
            vec![
                "anxiety".to_string(),
                "disgust".to_string(),
                "contempt".to_string(),
                "surprise".to_string(),
                "pride".to_string(),
                "shame".to_string(),
            ],
        );

        self.add_category(
            "professional".to_string(),
            vec![
                "teacher".to_string(),
                "customer_service".to_string(),
                "news_anchor".to_string(),
                "doctor".to_string(),
            ],
        );

        self.add_category(
            "age_related".to_string(),
            vec![
                "elderly".to_string(),
                "teenager".to_string(),
                "toddler".to_string(),
            ],
        );

        self.add_category(
            "intensity_variations".to_string(),
            vec![
                "subtle_happy".to_string(),
                "overwhelming_fear".to_string(),
                "mild_sadness".to_string(),
            ],
        );

        self.add_category(
            "cultural_variations".to_string(),
            vec![
                "respectful".to_string(),
                "passionate".to_string(),
                "reserved".to_string(),
            ],
        );

        // Update existing categories with new presets
        self.add_category(
            "positive".to_string(),
            vec![
                "happy".to_string(),
                "excited".to_string(),
                "triumphant".to_string(),
                "gentle".to_string(),
                "pride".to_string(),
                "subtle_happy".to_string(),
                "passionate".to_string(),
            ],
        );

        self.add_category(
            "negative".to_string(),
            vec![
                "sad".to_string(),
                "angry".to_string(),
                "fear".to_string(),
                "anxiety".to_string(),
                "disgust".to_string(),
                "contempt".to_string(),
                "shame".to_string(),
                "overwhelming_fear".to_string(),
                "mild_sadness".to_string(),
            ],
        );

        // Add specialty categories
        self.add_category(
            "youthful".to_string(),
            vec![
                "teenager".to_string(),
                "toddler".to_string(),
                "childlike".to_string(),
            ],
        );

        self.add_category(
            "mature".to_string(),
            vec![
                "elderly".to_string(),
                "doctor".to_string(),
                "narrator".to_string(),
            ],
        );

        self.add_category(
            "expressive".to_string(),
            vec![
                "passionate".to_string(),
                "excited".to_string(),
                "triumphant".to_string(),
                "overwhelming_fear".to_string(),
            ],
        );

        self.add_category(
            "subtle".to_string(),
            vec![
                "subtle_happy".to_string(),
                "mild_sadness".to_string(),
                "reserved".to_string(),
                "respectful".to_string(),
            ],
        );
    }

    /// Export library to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Error::from)
    }

    /// Import library from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Error::from)
    }

    /// Save library to file
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json).map_err(Error::from)
    }

    /// Load library from file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path).map_err(Error::from)?;
        Self::from_json(&json)
    }

    /// Get number of presets
    pub fn len(&self) -> usize {
        self.presets.len()
    }

    /// Check if library is empty
    pub fn is_empty(&self) -> bool {
        self.presets.is_empty()
    }
}

impl Default for EmotionPresetLibrary {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_preset_creation() {
        let mut vector = EmotionVector::new();
        vector.add_emotion(Emotion::Happy, EmotionIntensity::HIGH);
        let params = EmotionParameters::new(vector);

        let preset = EmotionPreset::new(
            "test_happy".to_string(),
            "Test happy emotion".to_string(),
            params,
        );

        assert_eq!(preset.name, "test_happy");
        assert_eq!(preset.description, "Test happy emotion");
    }

    #[test]
    fn test_preset_library() {
        let library = EmotionPresetLibrary::with_defaults();

        assert!(library.len() > 0);
        assert!(library.get_preset("happy").is_some());
        assert!(library.get_preset("sad").is_some());
        assert!(library.get_preset("nonexistent").is_none());

        // Test new extended emotion presets
        assert!(library.get_preset("anxiety").is_some());
        assert!(library.get_preset("disgust").is_some());
        assert!(library.get_preset("contempt").is_some());
        assert!(library.get_preset("surprise").is_some());
        assert!(library.get_preset("pride").is_some());
        assert!(library.get_preset("shame").is_some());

        // Test professional presets
        assert!(library.get_preset("teacher").is_some());
        assert!(library.get_preset("customer_service").is_some());
        assert!(library.get_preset("news_anchor").is_some());
        assert!(library.get_preset("doctor").is_some());

        // Test age-related presets
        assert!(library.get_preset("elderly").is_some());
        assert!(library.get_preset("teenager").is_some());
        assert!(library.get_preset("toddler").is_some());

        // Test intensity variations
        assert!(library.get_preset("subtle_happy").is_some());
        assert!(library.get_preset("overwhelming_fear").is_some());
        assert!(library.get_preset("mild_sadness").is_some());

        // Test cultural variations
        assert!(library.get_preset("respectful").is_some());
        assert!(library.get_preset("passionate").is_some());
        assert!(library.get_preset("reserved").is_some());
    }

    #[test]
    fn test_preset_scaling() {
        let library = EmotionPresetLibrary::with_defaults();
        let preset = library.get_preset("happy").unwrap();

        let scaled = preset.get_scaled_parameters(0.5);
        let original_intensity = preset
            .parameters
            .emotion_vector
            .emotions
            .get(&Emotion::Happy)
            .unwrap()
            .value();
        let scaled_intensity = scaled
            .emotion_vector
            .emotions
            .get(&Emotion::Happy)
            .unwrap()
            .value();

        assert!(scaled_intensity < original_intensity);
    }

    #[test]
    fn test_category_management() {
        let library = EmotionPresetLibrary::with_defaults();

        let positive_emotions = library.get_category("positive");
        assert!(!positive_emotions.is_empty());

        let basic_emotions = library.get_category("basic_emotions");
        assert!(basic_emotions.len() >= 5);

        // Test new categories
        let extended_emotions = library.get_category("extended_emotions");
        assert_eq!(extended_emotions.len(), 6);

        let professional = library.get_category("professional");
        assert_eq!(professional.len(), 4);

        let age_related = library.get_category("age_related");
        assert_eq!(age_related.len(), 3);

        let intensity_variations = library.get_category("intensity_variations");
        assert_eq!(intensity_variations.len(), 3);

        let cultural_variations = library.get_category("cultural_variations");
        assert_eq!(cultural_variations.len(), 3);

        // Test specialty categories
        let youthful = library.get_category("youthful");
        assert_eq!(youthful.len(), 3);

        let mature = library.get_category("mature");
        assert_eq!(mature.len(), 3);

        let expressive = library.get_category("expressive");
        assert_eq!(expressive.len(), 4);

        let subtle = library.get_category("subtle");
        assert_eq!(subtle.len(), 4);
    }

    #[test]
    fn test_preset_search() {
        let library = EmotionPresetLibrary::with_defaults();

        let basic_presets = library.find_by_tag("basic");
        assert!(!basic_presets.is_empty());

        let happy_presets = library.find_by_emotion(Emotion::Happy);
        assert!(!happy_presets.is_empty());
    }

    #[test]
    fn test_library_serialization() {
        let library = EmotionPresetLibrary::with_defaults();

        let json = library.to_json().unwrap();
        assert!(!json.is_empty());

        let loaded_library = EmotionPresetLibrary::from_json(&json).unwrap();
        assert_eq!(library.len(), loaded_library.len());
    }
}
