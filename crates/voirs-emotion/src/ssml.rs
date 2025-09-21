//! SSML (Speech Synthesis Markup Language) extensions for emotion control

use crate::{
    types::{Emotion, EmotionIntensity, EmotionParameters, EmotionVector},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SSML emotion extension attributes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionSSMLAttributes {
    /// Primary emotion name
    pub name: String,
    /// Emotion intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Duration of emotion effect
    pub duration: Option<String>,
    /// Transition style
    pub transition: Option<String>,
    /// Additional emotion properties
    pub properties: HashMap<String, String>,
}

impl EmotionSSMLAttributes {
    /// Create new SSML emotion attributes
    pub fn new(name: String, intensity: f32) -> Self {
        Self {
            name,
            intensity: intensity.clamp(0.0, 1.0),
            duration: None,
            transition: None,
            properties: HashMap::new(),
        }
    }

    /// Set duration
    pub fn with_duration(mut self, duration: String) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set transition style
    pub fn with_transition(mut self, transition: String) -> Self {
        self.transition = Some(transition);
        self
    }

    /// Add property
    pub fn with_property(mut self, key: String, value: String) -> Self {
        self.properties.insert(key, value);
        self
    }

    /// Convert to emotion parameters
    pub fn to_emotion_parameters(&self) -> Result<EmotionParameters> {
        let emotion = Emotion::from_str(&self.name);
        let intensity = EmotionIntensity::new(self.intensity);

        let mut emotion_vector = EmotionVector::new();
        emotion_vector.add_emotion(emotion, intensity);

        let mut params = EmotionParameters::new(emotion_vector);

        // Parse duration if provided
        if let Some(duration_str) = &self.duration {
            if let Ok(duration_ms) = parse_duration(duration_str) {
                params.duration_ms = Some(duration_ms);
            }
        }

        // Apply properties
        for (key, value) in &self.properties {
            match key.as_str() {
                "pitch-shift" => {
                    if let Ok(shift) = value.parse::<f32>() {
                        params.pitch_shift = shift;
                    }
                }
                "tempo-scale" => {
                    if let Ok(scale) = value.parse::<f32>() {
                        params.tempo_scale = scale;
                    }
                }
                "energy-scale" => {
                    if let Ok(scale) = value.parse::<f32>() {
                        params.energy_scale = scale;
                    }
                }
                "breathiness" => {
                    if let Ok(breathiness) = value.parse::<f32>() {
                        params.breathiness = breathiness;
                    }
                }
                "roughness" => {
                    if let Ok(roughness) = value.parse::<f32>() {
                        params.roughness = roughness;
                    }
                }
                _ => {
                    // Store as custom parameter
                    if let Ok(float_value) = value.parse::<f32>() {
                        params.custom_params.insert(key.clone(), float_value);
                    }
                }
            }
        }

        Ok(params)
    }
}

/// SSML emotion processor for parsing and generating emotion markup
#[derive(Debug, Clone)]
pub struct EmotionSSMLProcessor {
    /// Namespace for emotion extensions
    pub namespace: String,
    /// Default emotion attributes
    pub default_attributes: EmotionSSMLAttributes,
    /// Custom emotion mappings
    pub custom_mappings: HashMap<String, EmotionParameters>,
}

impl EmotionSSMLProcessor {
    /// Create new SSML emotion processor
    pub fn new() -> Self {
        Self {
            namespace: "emotion".to_string(),
            default_attributes: EmotionSSMLAttributes::new("neutral".to_string(), 0.5),
            custom_mappings: HashMap::new(),
        }
    }

    /// Parse emotion attributes from SSML tag
    pub fn parse_emotion_tag(&self, tag_content: &str) -> Result<EmotionSSMLAttributes> {
        let mut attributes = EmotionSSMLAttributes::new("neutral".to_string(), 0.5);

        // Simple attribute parsing (in a real implementation, use a proper XML parser)
        for part in tag_content.split_whitespace() {
            if let Some((key, value)) = part.split_once('=') {
                let value = value.trim_matches('"').trim_matches('\'');

                match key {
                    "name" => attributes.name = value.to_string(),
                    "intensity" => {
                        if let Ok(intensity) = value.parse::<f32>() {
                            attributes.intensity = intensity.clamp(0.0, 1.0);
                        }
                    }
                    "duration" => attributes.duration = Some(value.to_string()),
                    "transition" => attributes.transition = Some(value.to_string()),
                    _ => {
                        attributes
                            .properties
                            .insert(key.to_string(), value.to_string());
                    }
                }
            }
        }

        Ok(attributes)
    }

    /// Generate SSML emotion tag
    pub fn generate_emotion_tag(&self, attributes: &EmotionSSMLAttributes) -> String {
        let mut tag = format!(
            "<{}:emotion name=\"{}\" intensity=\"{}\"",
            self.namespace, attributes.name, attributes.intensity
        );

        if let Some(duration) = &attributes.duration {
            tag.push_str(&format!(" duration=\"{duration}\""));
        }

        if let Some(transition) = &attributes.transition {
            tag.push_str(&format!(" transition=\"{transition}\""));
        }

        for (key, value) in &attributes.properties {
            tag.push_str(&format!(" {}=\"{}\"", key, value));
        }

        tag.push('>');
        tag
    }

    /// Generate closing emotion tag
    pub fn generate_emotion_close_tag(&self) -> String {
        format!("</{}>", self.namespace)
    }

    /// Process SSML text with emotion tags
    pub fn process_ssml_text(&self, ssml_text: &str) -> Result<Vec<EmotionSegment>> {
        let mut segments = Vec::new();
        let mut current_emotion = self.default_attributes.clone();
        let mut current_text = String::new();

        // Simple parsing (in a real implementation, use a proper XML parser)
        let mut chars = ssml_text.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '<' {
                // Save current text segment if not empty
                if !current_text.is_empty() {
                    let params = current_emotion.to_emotion_parameters()?;
                    segments.push(EmotionSegment {
                        text: current_text.clone(),
                        emotion_params: params,
                    });
                    current_text.clear();
                }

                // Parse tag
                let mut tag = String::new();
                while let Some(tag_ch) = chars.next() {
                    if tag_ch == '>' {
                        break;
                    }
                    tag.push(tag_ch);
                }

                // Check if it's an emotion tag
                if tag.starts_with(&format!("{}:emotion", self.namespace)) {
                    let attributes_str = tag
                        .strip_prefix(&format!("{}:emotion", self.namespace))
                        .unwrap_or("")
                        .trim();
                    current_emotion = self.parse_emotion_tag(attributes_str)?;
                } else if tag == format!("/{}", self.namespace) {
                    current_emotion = self.default_attributes.clone();
                }
                // Other SSML tags are ignored for emotion processing
            } else {
                current_text.push(ch);
            }
        }

        // Add final segment if there's remaining text
        if !current_text.is_empty() {
            let params = current_emotion.to_emotion_parameters()?;
            segments.push(EmotionSegment {
                text: current_text,
                emotion_params: params,
            });
        }

        Ok(segments)
    }

    /// Generate SSML with emotion tags from segments
    pub fn generate_ssml_from_segments(&self, segments: &[EmotionSegment]) -> Result<String> {
        let mut ssml = String::new();

        for segment in segments {
            // Convert emotion parameters back to attributes
            let attributes = self.emotion_parameters_to_attributes(&segment.emotion_params)?;

            // Generate opening tag
            ssml.push_str(&self.generate_emotion_tag(&attributes));

            // Add text content
            ssml.push_str(&segment.text);

            // Generate closing tag
            ssml.push_str(&self.generate_emotion_close_tag());
        }

        Ok(ssml)
    }

    /// Convert emotion parameters to SSML attributes
    fn emotion_parameters_to_attributes(
        &self,
        params: &EmotionParameters,
    ) -> Result<EmotionSSMLAttributes> {
        // Get dominant emotion
        let (emotion, intensity) = params
            .emotion_vector
            .dominant_emotion()
            .unwrap_or((Emotion::Neutral, EmotionIntensity::MEDIUM));

        let mut attributes =
            EmotionSSMLAttributes::new(emotion.as_str().to_string(), intensity.value());

        // Add duration if available
        if let Some(duration_ms) = params.duration_ms {
            attributes.duration = Some(format!("{}ms", duration_ms));
        }

        // Add prosody properties
        if (params.pitch_shift - 1.0).abs() > 0.01 {
            attributes
                .properties
                .insert("pitch-shift".to_string(), params.pitch_shift.to_string());
        }
        if (params.tempo_scale - 1.0).abs() > 0.01 {
            attributes
                .properties
                .insert("tempo-scale".to_string(), params.tempo_scale.to_string());
        }
        if (params.energy_scale - 1.0).abs() > 0.01 {
            attributes
                .properties
                .insert("energy-scale".to_string(), params.energy_scale.to_string());
        }

        // Add voice quality properties
        if params.breathiness.abs() > 0.01 {
            attributes
                .properties
                .insert("breathiness".to_string(), params.breathiness.to_string());
        }
        if params.roughness.abs() > 0.01 {
            attributes
                .properties
                .insert("roughness".to_string(), params.roughness.to_string());
        }

        // Add custom parameters
        for (key, value) in &params.custom_params {
            attributes.properties.insert(key.clone(), value.to_string());
        }

        Ok(attributes)
    }

    /// Add custom emotion mapping
    pub fn add_custom_mapping(&mut self, name: String, params: EmotionParameters) {
        self.custom_mappings.insert(name, params);
    }

    /// Get custom emotion mapping
    pub fn get_custom_mapping(&self, name: &str) -> Option<&EmotionParameters> {
        self.custom_mappings.get(name)
    }

    /// Set namespace for emotion tags
    pub fn set_namespace(&mut self, namespace: String) {
        self.namespace = namespace;
    }

    /// Set default emotion attributes
    pub fn set_default_attributes(&mut self, attributes: EmotionSSMLAttributes) {
        self.default_attributes = attributes;
    }
}

impl Default for EmotionSSMLProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Text segment with associated emotion parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionSegment {
    /// Text content
    pub text: String,
    /// Emotion parameters for this segment
    pub emotion_params: EmotionParameters,
}

impl EmotionSegment {
    /// Create new emotion segment
    pub fn new(text: String, emotion_params: EmotionParameters) -> Self {
        Self {
            text,
            emotion_params,
        }
    }

    /// Create neutral emotion segment
    pub fn neutral(text: String) -> Self {
        Self::new(text, EmotionParameters::neutral())
    }

    /// Get text length
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Check if segment is empty
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// Parse duration string (e.g., "1000ms", "2s", "0.5s")
fn parse_duration(duration_str: &str) -> Result<u64> {
    let duration_str = duration_str.trim().to_lowercase();

    if duration_str.ends_with("ms") {
        let value_str = duration_str.strip_suffix("ms").ok_or_else(|| {
            Error::Validation(format!("Invalid duration format: {}", duration_str))
        })?;
        value_str
            .parse::<u64>()
            .map_err(|_| Error::Validation(format!("Invalid duration: {}", duration_str)))
    } else if duration_str.ends_with("s") {
        let value_str = duration_str.strip_suffix("s").ok_or_else(|| {
            Error::Validation(format!("Invalid duration format: {}", duration_str))
        })?;
        let seconds: f64 = value_str
            .parse()
            .map_err(|_| Error::Validation(format!("Invalid duration: {}", duration_str)))?;
        Ok((seconds * 1000.0) as u64)
    } else {
        // Assume milliseconds if no unit
        duration_str
            .parse::<u64>()
            .map_err(|_| Error::Validation(format!("Invalid duration: {}", duration_str)))
    }
}

/// SSML emotion extension builder
pub struct EmotionSSMLBuilder {
    processor: EmotionSSMLProcessor,
    segments: Vec<EmotionSegment>,
}

impl EmotionSSMLBuilder {
    /// Create new SSML builder
    pub fn new() -> Self {
        Self {
            processor: EmotionSSMLProcessor::new(),
            segments: Vec::new(),
        }
    }

    /// Add text with emotion
    pub fn add_emotion_text(mut self, text: String, emotion: Emotion, intensity: f32) -> Self {
        let mut emotion_vector = EmotionVector::new();
        emotion_vector.add_emotion(emotion, EmotionIntensity::new(intensity));
        let params = EmotionParameters::new(emotion_vector);

        self.segments.push(EmotionSegment::new(text, params));
        self
    }

    /// Add text with emotion parameters
    pub fn add_parametric_text(mut self, text: String, params: EmotionParameters) -> Self {
        self.segments.push(EmotionSegment::new(text, params));
        self
    }

    /// Add neutral text
    pub fn add_neutral_text(mut self, text: String) -> Self {
        self.segments.push(EmotionSegment::neutral(text));
        self
    }

    /// Build SSML string
    pub fn build(self) -> Result<String> {
        self.processor.generate_ssml_from_segments(&self.segments)
    }

    /// Set namespace
    pub fn namespace(mut self, namespace: String) -> Self {
        self.processor.set_namespace(namespace);
        self
    }
}

impl Default for EmotionSSMLBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotion_attributes() {
        let attrs = EmotionSSMLAttributes::new("happy".to_string(), 0.8)
            .with_duration("1000ms".to_string())
            .with_property("pitch-shift".to_string(), "1.2".to_string());

        assert_eq!(attrs.name, "happy");
        assert_eq!(attrs.intensity, 0.8);
        assert_eq!(attrs.duration, Some("1000ms".to_string()));
        assert_eq!(
            attrs.properties.get("pitch-shift"),
            Some(&"1.2".to_string())
        );
    }

    #[test]
    fn test_attributes_to_parameters() {
        let attrs = EmotionSSMLAttributes::new("happy".to_string(), 0.7)
            .with_property("pitch-shift".to_string(), "1.2".to_string());

        let params = attrs.to_emotion_parameters().unwrap();
        assert_eq!(params.pitch_shift, 1.2);
        assert!(params.emotion_vector.emotions.contains_key(&Emotion::Happy));
    }

    #[test]
    fn test_ssml_tag_generation() {
        let processor = EmotionSSMLProcessor::new();
        let attrs = EmotionSSMLAttributes::new("happy".to_string(), 0.8);

        let tag = processor.generate_emotion_tag(&attrs);
        assert!(tag.contains("name=\"happy\""));
        assert!(tag.contains("intensity=\"0.8\""));
    }

    #[test]
    fn test_ssml_parsing() {
        let processor = EmotionSSMLProcessor::new();
        let attrs_str = "name=\"happy\" intensity=\"0.8\" duration=\"1000ms\"";

        let attrs = processor.parse_emotion_tag(attrs_str).unwrap();
        assert_eq!(attrs.name, "happy");
        assert_eq!(attrs.intensity, 0.8);
        assert_eq!(attrs.duration, Some("1000ms".to_string()));
    }

    #[test]
    fn test_duration_parsing() {
        assert_eq!(parse_duration("1000ms").unwrap(), 1000);
        assert_eq!(parse_duration("2s").unwrap(), 2000);
        assert_eq!(parse_duration("0.5s").unwrap(), 500);
        assert_eq!(parse_duration("500").unwrap(), 500);
    }

    #[test]
    fn test_ssml_builder() {
        let ssml = EmotionSSMLBuilder::new()
            .add_emotion_text("Hello".to_string(), Emotion::Happy, 0.8)
            .add_neutral_text(" world".to_string())
            .add_emotion_text("!".to_string(), Emotion::Excited, 0.9)
            .build()
            .unwrap();

        assert!(ssml.contains("Hello"));
        assert!(ssml.contains("world"));
        assert!(ssml.contains("!"));
    }

    #[test]
    fn test_segment_processing() {
        let processor = EmotionSSMLProcessor::new();

        let segments = vec![
            EmotionSegment::new("Hello".to_string(), EmotionParameters::neutral()),
            EmotionSegment::neutral(" world!".to_string()),
        ];

        let ssml = processor.generate_ssml_from_segments(&segments).unwrap();
        assert!(ssml.contains("Hello"));
        assert!(ssml.contains("world!"));
    }
}
