//! Named entity recognition functionality for context-aware preprocessing.

use crate::{LanguageCode, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Named entity types for recognition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// Person name
    Person,
    /// Organization name
    Organization,
    /// Location name
    Location,
    /// Date/time expression
    DateTime,
    /// Money/currency amount
    Money,
    /// Percentage
    Percentage,
    /// Technical term
    Technical,
    /// Brand name
    Brand,
    /// Other/miscellaneous
    Other,
}

/// Trait for named entity recognition functionality
pub trait NamedEntityRecognition: Send + Sync {
    /// Recognize entities in text and return (entity, type, start, end) tuples
    fn recognize_entities(&self, text: &str) -> Result<Vec<(String, EntityType, usize, usize)>>;

    /// Check if a word is a named entity
    fn is_named_entity(&self, word: &str) -> Result<Option<EntityType>>;

    /// Get supported languages
    fn supported_languages(&self) -> Vec<LanguageCode>;
}

/// Simple named entity recognizer
#[derive(Debug, Clone)]
pub struct SimpleNamedEntityRecognizer {
    /// Entity dictionaries
    pub entity_dicts: HashMap<EntityType, HashSet<String>>,
    /// Pattern-based rules
    pub pattern_rules: Vec<(regex::Regex, EntityType)>,
    /// Contextual rules
    pub contextual_rules: Vec<ContextualNerRule>,
}

/// Contextual NER rule
#[derive(Debug, Clone)]
pub struct ContextualNerRule {
    /// Entity pattern
    pub entity_pattern: String,
    /// Context pattern
    pub context_pattern: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Rule confidence
    pub confidence: f32,
}

impl NamedEntityRecognition for SimpleNamedEntityRecognizer {
    fn recognize_entities(&self, text: &str) -> Result<Vec<(String, EntityType, usize, usize)>> {
        let mut entities = Vec::new();

        // First pass: pattern-based recognition on full text
        entities.extend(self.recognize_pattern_entities(text)?);

        // Second pass: dictionary-based recognition with multi-word support
        entities.extend(self.recognize_dictionary_entities(text)?);

        // Third pass: contextual recognition
        entities.extend(self.recognize_contextual_entities(text)?);

        // Remove duplicates and resolve conflicts
        Ok(self.resolve_entity_conflicts(entities))
    }

    fn is_named_entity(&self, word: &str) -> Result<Option<EntityType>> {
        for (entity_type, entity_dict) in &self.entity_dicts {
            if entity_dict.contains(word)
                || entity_dict
                    .iter()
                    .any(|entity| entity.to_lowercase() == word.to_lowercase())
            {
                return Ok(Some(entity_type.clone()));
            }
        }

        for (pattern, entity_type) in &self.pattern_rules {
            if pattern.is_match(word) {
                return Ok(Some(entity_type.clone()));
            }
        }

        Ok(None)
    }

    fn supported_languages(&self) -> Vec<LanguageCode> {
        vec![LanguageCode::EnUs] // Default implementation
    }
}

impl SimpleNamedEntityRecognizer {
    /// Create a new simple named entity recognizer
    pub fn new() -> Self {
        let mut entity_dicts = HashMap::new();
        let mut pattern_rules = Vec::new();

        // Initialize comprehensive entity dictionaries
        entity_dicts.insert(EntityType::Person, Self::default_person_names());
        entity_dicts.insert(EntityType::Organization, Self::default_organizations());
        entity_dicts.insert(EntityType::Location, Self::default_locations());
        entity_dicts.insert(EntityType::Technical, Self::default_technical_terms());
        entity_dicts.insert(EntityType::Brand, Self::default_brands());

        // Add comprehensive patterns
        Self::add_default_patterns(&mut pattern_rules);

        Self {
            entity_dicts,
            pattern_rules,
            contextual_rules: Self::default_contextual_rules(),
        }
    }

    /// Create default person names dictionary
    fn default_person_names() -> HashSet<String> {
        let names = vec![
            // Common English names
            "John",
            "Jane",
            "Michael",
            "Sarah",
            "David",
            "Lisa",
            "Chris",
            "Emma",
            "Alex",
            "Maria",
            "James",
            "Anna",
            "Robert",
            "Emily",
            "William",
            "Jessica",
            "Thomas",
            "Ashley",
            "Daniel",
            "Amanda",
            "Matthew",
            "Stephanie",
            "Andrew",
            "Jennifer",
            // Common titles
            "Dr",
            "Mr",
            "Mrs",
            "Ms",
            "Prof",
            "President",
            "CEO",
            "Director",
        ];
        names.into_iter().map(String::from).collect()
    }

    /// Create default organizations dictionary
    fn default_organizations() -> HashSet<String> {
        let orgs = vec![
            // Tech companies
            "Google",
            "Microsoft",
            "Apple",
            "Amazon",
            "Meta",
            "Tesla",
            "Netflix",
            "Adobe",
            "Intel",
            "NVIDIA",
            "IBM",
            "Oracle",
            "Salesforce",
            "Uber",
            "Airbnb",
            // Universities
            "MIT",
            "Stanford",
            "Harvard",
            "Berkeley",
            "UCLA",
            "Yale",
            "Princeton",
            // Government/Organizations
            "NASA",
            "FBI",
            "CIA",
            "NATO",
            "UN",
            "WHO",
            "UNESCO",
        ];
        orgs.into_iter().map(String::from).collect()
    }

    /// Create default locations dictionary
    fn default_locations() -> HashSet<String> {
        let locations = vec![
            // Major cities
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Phoenix",
            "Philadelphia",
            "San Antonio",
            "San Diego",
            "Dallas",
            "San Jose",
            "Austin",
            "Jacksonville",
            "London",
            "Paris",
            "Tokyo",
            "Berlin",
            "Rome",
            "Madrid",
            "Amsterdam",
            "Sydney",
            "Melbourne",
            "Toronto",
            "Vancouver",
            "Montreal",
            // Countries
            "USA",
            "Canada",
            "Mexico",
            "UK",
            "France",
            "Germany",
            "Spain",
            "Italy",
            "Japan",
            "China",
            "India",
            "Australia",
            "Brazil",
            "Argentina",
            // States/Provinces
            "California",
            "Texas",
            "Florida",
            "New York",
            "Pennsylvania",
            "Illinois",
            "Ohio",
            "Georgia",
            "North Carolina",
            "Michigan",
        ];
        locations.into_iter().map(String::from).collect()
    }

    /// Create default technical terms dictionary
    fn default_technical_terms() -> HashSet<String> {
        let terms = vec![
            // Programming
            "API",
            "SDK",
            "JSON",
            "XML",
            "HTTP",
            "HTTPS",
            "REST",
            "GraphQL",
            "JavaScript",
            "Python",
            "Java",
            "C++",
            "Rust",
            "Go",
            "TypeScript",
            // Tech terms
            "AI",
            "ML",
            "GPU",
            "CPU",
            "RAM",
            "SSD",
            "HDD",
            "USB",
            "WiFi",
            "Bluetooth",
            "IoT",
            "VR",
            "AR",
            "5G",
            "4G",
            "LTE",
            "DNS",
            "VPN",
            "SSL",
            "TLS",
            // Science
            "DNA",
            "RNA",
            "CO2",
            "H2O",
            "pH",
            "MHz",
            "GHz",
            "KB",
            "MB",
            "GB",
            "TB",
        ];
        terms.into_iter().map(String::from).collect()
    }

    /// Create default brands dictionary
    fn default_brands() -> HashSet<String> {
        let brands = vec![
            // Tech brands
            "iPhone",
            "iPad",
            "MacBook",
            "Windows",
            "Android",
            "Chrome",
            "Firefox",
            "PlayStation",
            "Xbox",
            "Nintendo",
            "Samsung",
            "Sony",
            "LG",
            "Dell",
            "HP",
            // Car brands
            "Toyota",
            "Honda",
            "Ford",
            "BMW",
            "Mercedes",
            "Audi",
            "Volkswagen",
            "Tesla",
            "Nissan",
            "Hyundai",
            "Kia",
            "Mazda",
            "Subaru",
            // Other brands
            "Coca-Cola",
            "Pepsi",
            "McDonald's",
            "Starbucks",
            "Nike",
            "Adidas",
            "Disney",
            "Warner",
            "Universal",
            "Paramount",
        ];
        brands.into_iter().map(String::from).collect()
    }

    /// Add default pattern rules
    fn add_default_patterns(pattern_rules: &mut Vec<(regex::Regex, EntityType)>) {
        let patterns = vec![
            // DateTime patterns
            (r"\b\d{1,2}/\d{1,2}/\d{4}\b", EntityType::DateTime),
            (r"\b\d{4}-\d{2}-\d{2}\b", EntityType::DateTime),
            (
                r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)\b",
                EntityType::DateTime,
            ),
            (
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                EntityType::DateTime,
            ),
            // Money patterns
            (r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b", EntityType::Money),
            (
                r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|CAD|AUD)\b",
                EntityType::Money,
            ),
            (
                r"\b(?:USD|EUR|GBP|JPY|CAD|AUD)\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b",
                EntityType::Money,
            ),
            // Percentage patterns
            (r"\d+(?:\.\d+)?%", EntityType::Percentage),
            (r"\d+(?:\.\d+)?\s*percent", EntityType::Percentage),
            // Technical patterns
            (
                r"\d+(?:\.\d+)?\s*(?:GB|MB|KB|TB|Hz|MHz|GHz|Mbps|Gbps)",
                EntityType::Technical,
            ),
            (
                r"(?i)(?:HTTP|HTTPS|FTP|SSH)://[^\s]+",
                EntityType::Technical,
            ),
            (
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                EntityType::Technical,
            ),
            // Organization patterns
            (
                r"\b[A-Z][a-z]+\s+(?:Inc|Corp|LLC|Ltd|Co|Company|Corporation|Limited)\b",
                EntityType::Organization,
            ),
            (
                r"\b(?:University|College|Institute|Academy)\s+of\s+[A-Z][a-z]+\b",
                EntityType::Organization,
            ),
        ];

        for (pattern_str, entity_type) in patterns {
            if let Ok(regex) = regex::Regex::new(pattern_str) {
                pattern_rules.push((regex, entity_type));
            }
        }
    }

    /// Create default contextual rules
    fn default_contextual_rules() -> Vec<ContextualNerRule> {
        vec![
            ContextualNerRule {
                entity_pattern: r"\b[A-Z][a-z]+\b".to_string(),
                context_pattern: r"\b(?:Mr|Mrs|Ms|Dr|Prof)\s+".to_string(),
                entity_type: EntityType::Person,
                confidence: 0.9,
            },
            ContextualNerRule {
                entity_pattern: r"\b[A-Z][a-z]+\b".to_string(),
                context_pattern: r"\bin\s+".to_string(),
                entity_type: EntityType::Location,
                confidence: 0.7,
            },
            ContextualNerRule {
                entity_pattern: r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b".to_string(),
                context_pattern: r"\bworks\s+at\s+".to_string(),
                entity_type: EntityType::Organization,
                confidence: 0.8,
            },
        ]
    }

    /// Add a named entity to the dictionary
    pub fn add_entity(&mut self, entity_type: EntityType, entity: String) {
        self.entity_dicts
            .entry(entity_type)
            .or_default()
            .insert(entity);
    }

    /// Add multiple entities at once
    pub fn add_entities(&mut self, entity_type: EntityType, entities: Vec<String>) {
        let dict = self.entity_dicts.entry(entity_type).or_default();

        for entity in entities {
            dict.insert(entity);
        }
    }

    /// Add a pattern-based rule
    pub fn add_pattern_rule(&mut self, pattern: &str, entity_type: EntityType) -> Result<()> {
        match regex::Regex::new(pattern) {
            Ok(regex) => {
                self.pattern_rules.push((regex, entity_type));
                Ok(())
            }
            Err(e) => Err(crate::G2pError::ConfigError(format!(
                "Invalid regex pattern: {e}"
            ))),
        }
    }

    /// Add a contextual NER rule
    pub fn add_contextual_rule(&mut self, rule: ContextualNerRule) {
        self.contextual_rules.push(rule);
    }

    /// Load entities from a file or resource
    pub fn load_entity_dictionary(&mut self, entity_type: EntityType, entities: Vec<String>) {
        self.add_entities(entity_type, entities);
    }

    /// Get statistics about the entity dictionary
    pub fn get_statistics(&self) -> HashMap<EntityType, usize> {
        self.entity_dicts
            .iter()
            .map(|(entity_type, dict)| (entity_type.clone(), dict.len()))
            .collect()
    }

    /// Recognize entities using pattern matching
    fn recognize_pattern_entities(
        &self,
        text: &str,
    ) -> Result<Vec<(String, EntityType, usize, usize)>> {
        let mut entities = Vec::new();

        for (pattern, entity_type) in &self.pattern_rules {
            for mat in pattern.find_iter(text) {
                let entity_text = mat.as_str().to_string();
                let start = self.char_to_word_index(text, mat.start());
                let end = self.char_to_word_index(text, mat.end() - 1) + 1;
                entities.push((entity_text, entity_type.clone(), start, end));
            }
        }

        Ok(entities)
    }

    /// Recognize entities from dictionaries with multi-word support
    fn recognize_dictionary_entities(
        &self,
        text: &str,
    ) -> Result<Vec<(String, EntityType, usize, usize)>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut entities = Vec::new();

        // Try different n-gram sizes (3-1 words) - longer first
        for n in (1..=3).rev() {
            for i in 0..words.len() {
                if i + n > words.len() {
                    break;
                }

                let phrase = words[i..i + n].join(" ");
                let clean_phrase = phrase.trim_matches(|c: char| c.is_ascii_punctuation());

                // Check against all entity dictionaries (case-insensitive)
                for (entity_type, entity_dict) in &self.entity_dicts {
                    // Check both original case and lowercase
                    if entity_dict.contains(clean_phrase)
                        || entity_dict
                            .iter()
                            .any(|entity| entity.to_lowercase() == clean_phrase.to_lowercase())
                    {
                        entities.push((clean_phrase.to_string(), entity_type.clone(), i, i + n));
                    }
                }
            }
        }

        Ok(entities)
    }

    /// Recognize entities using contextual rules
    fn recognize_contextual_entities(
        &self,
        text: &str,
    ) -> Result<Vec<(String, EntityType, usize, usize)>> {
        let mut entities = Vec::new();

        for rule in &self.contextual_rules {
            // Create combined pattern that looks for context + entity
            let combined_pattern = format!("{}({})", rule.context_pattern, rule.entity_pattern);
            if let Ok(combined_regex) = regex::Regex::new(&combined_pattern) {
                for captures in combined_regex.captures_iter(text) {
                    if let Some(entity_match) = captures.get(1) {
                        let entity_text = entity_match.as_str().to_string();
                        let start = self.char_to_word_index(text, entity_match.start());
                        let end = self.char_to_word_index(text, entity_match.end() - 1) + 1;
                        entities.push((entity_text, rule.entity_type.clone(), start, end));
                    }
                }
            }
        }

        Ok(entities)
    }

    /// Convert character index to word index
    fn char_to_word_index(&self, text: &str, char_index: usize) -> usize {
        let mut current_pos = 0;
        let mut word_index = 0;

        for word in text.split_whitespace() {
            // Find the start of this word in the original text
            if let Some(word_start) = text[current_pos..].find(word) {
                let absolute_start = current_pos + word_start;
                let absolute_end = absolute_start + word.len();

                // Check if the char_index falls within this word
                if char_index >= absolute_start && char_index < absolute_end {
                    return word_index;
                }

                current_pos = absolute_end;
                word_index += 1;
            } else {
                break;
            }
        }

        // If we didn't find the character within any word, return the last word index
        word_index.saturating_sub(1)
    }

    /// Resolve conflicts between overlapping entities
    fn resolve_entity_conflicts(
        &self,
        mut entities: Vec<(String, EntityType, usize, usize)>,
    ) -> Vec<(String, EntityType, usize, usize)> {
        // Sort by start position and length (longer entities first)
        entities.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| (b.3 - b.2).cmp(&(a.3 - a.2))));

        let mut resolved = Vec::new();
        let mut used_positions = std::collections::HashSet::new();

        for entity in entities {
            let range: std::collections::HashSet<usize> = (entity.2..entity.3).collect();

            // Check if this entity overlaps with any already selected entity
            if range.is_disjoint(&used_positions) {
                // Add positions to used set
                used_positions.extend(range);
                resolved.push(entity);
            }
        }

        resolved
    }

    /// Create a language-specific recognizer
    pub fn new_for_language(language: LanguageCode) -> Self {
        let mut recognizer = Self::new();

        // Add language-specific entities and patterns
        match language {
            LanguageCode::EnUs => {
                // Already initialized with English defaults
            }
            LanguageCode::Es => {
                recognizer.add_spanish_entities();
            }
            LanguageCode::Fr => {
                recognizer.add_french_entities();
            }
            LanguageCode::De => {
                recognizer.add_german_entities();
            }
            _ => {
                // Use English defaults for other languages
            }
        }

        recognizer
    }

    /// Add Spanish-specific entities
    fn add_spanish_entities(&mut self) {
        let spanish_names = vec![
            "José",
            "María",
            "Antonio",
            "Carmen",
            "Manuel",
            "Dolores",
            "Francisco",
            "Isabel",
            "Juan",
            "Ana",
            "Carlos",
            "Teresa",
            "Luis",
            "Pilar",
            "Miguel",
            "Rosa",
        ];
        self.add_entities(
            EntityType::Person,
            spanish_names.into_iter().map(String::from).collect(),
        );

        let spanish_cities = vec![
            "Madrid",
            "Barcelona",
            "Valencia",
            "Sevilla",
            "Zaragoza",
            "Málaga",
            "Murcia",
            "Palma",
            "Las Palmas",
            "Bilbao",
            "Alicante",
            "Córdoba",
            "Valladolid",
            "Vigo",
            "Gijón",
        ];
        self.add_entities(
            EntityType::Location,
            spanish_cities.into_iter().map(String::from).collect(),
        );
    }

    /// Add French-specific entities
    fn add_french_entities(&mut self) {
        let french_names = vec![
            "Pierre",
            "Marie",
            "Jean",
            "Françoise",
            "Michel",
            "Monique",
            "Philippe",
            "Catherine",
            "Alain",
            "Nathalie",
            "Bernard",
            "Isabelle",
            "Christophe",
            "Sylvie",
            "Nicolas",
            "Martine",
        ];
        self.add_entities(
            EntityType::Person,
            french_names.into_iter().map(String::from).collect(),
        );

        let french_cities = vec![
            "Paris",
            "Marseille",
            "Lyon",
            "Toulouse",
            "Nice",
            "Nantes",
            "Strasbourg",
            "Montpellier",
            "Bordeaux",
            "Lille",
            "Rennes",
            "Reims",
            "Le Havre",
            "Saint-Étienne",
            "Toulon",
        ];
        self.add_entities(
            EntityType::Location,
            french_cities.into_iter().map(String::from).collect(),
        );
    }

    /// Add German-specific entities
    fn add_german_entities(&mut self) {
        let german_names = vec![
            "Hans",
            "Anna",
            "Karl",
            "Maria",
            "Heinrich",
            "Elisabeth",
            "Friedrich",
            "Margarete",
            "Wilhelm",
            "Emma",
            "Otto",
            "Martha",
            "Ernst",
            "Johanna",
            "Paul",
            "Bertha",
        ];
        self.add_entities(
            EntityType::Person,
            german_names.into_iter().map(String::from).collect(),
        );

        let german_cities = vec![
            "Berlin",
            "Hamburg",
            "München",
            "Köln",
            "Frankfurt",
            "Stuttgart",
            "Düsseldorf",
            "Dortmund",
            "Essen",
            "Leipzig",
            "Bremen",
            "Dresden",
            "Hannover",
            "Nürnberg",
            "Duisburg",
        ];
        self.add_entities(
            EntityType::Location,
            german_cities.into_iter().map(String::from).collect(),
        );
    }

    /// Get entity confidence score
    pub fn get_entity_confidence(&self, entity: &str, entity_type: &EntityType) -> f32 {
        // Check if it's a dictionary match (high confidence)
        if let Some(dict) = self.entity_dicts.get(entity_type) {
            if dict.contains(entity) {
                return 0.95;
            }
        }

        // Check if it matches patterns (medium confidence)
        for (pattern, pattern_type) in &self.pattern_rules {
            if pattern_type == entity_type && pattern.is_match(entity) {
                return 0.85;
            }
        }

        // Check contextual rules (variable confidence)
        for rule in &self.contextual_rules {
            if rule.entity_type == *entity_type {
                if let Ok(entity_regex) = regex::Regex::new(&rule.entity_pattern) {
                    if entity_regex.is_match(entity) {
                        return rule.confidence;
                    }
                }
            }
        }

        0.0 // No match found
    }
}

impl Default for SimpleNamedEntityRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_entity_recognition() {
        let mut recognizer = SimpleNamedEntityRecognizer::new();

        // Add some test entities
        recognizer.add_entity(EntityType::Person, "John".to_string());
        recognizer.add_entity(EntityType::Location, "Paris".to_string());

        // Test recognition
        assert_eq!(
            recognizer.is_named_entity("John").unwrap(),
            Some(EntityType::Person)
        );
        assert_eq!(
            recognizer.is_named_entity("Paris").unwrap(),
            Some(EntityType::Location)
        );
        assert_eq!(recognizer.is_named_entity("unknown").unwrap(), None);
    }

    #[test]
    fn test_pattern_recognition() {
        let recognizer = SimpleNamedEntityRecognizer::new();

        // Test money pattern
        assert_eq!(
            recognizer.is_named_entity("$100.50").unwrap(),
            Some(EntityType::Money)
        );

        // Test percentage pattern
        assert_eq!(
            recognizer.is_named_entity("25%").unwrap(),
            Some(EntityType::Percentage)
        );

        // Test date pattern
        assert_eq!(
            recognizer.is_named_entity("12/25/2023").unwrap(),
            Some(EntityType::DateTime)
        );
    }

    #[test]
    fn test_entity_text_recognition() {
        let mut recognizer = SimpleNamedEntityRecognizer::new();
        recognizer.add_entity(EntityType::Person, "Alice".to_string());
        recognizer.add_entity(EntityType::Location, "Tokyo".to_string());

        let entities = recognizer
            .recognize_entities("Alice visited Tokyo yesterday")
            .unwrap();

        assert_eq!(entities.len(), 2);
        assert!(entities.contains(&("Alice".to_string(), EntityType::Person, 0, 1)));
        assert!(entities.contains(&("Tokyo".to_string(), EntityType::Location, 2, 3)));
    }

    #[test]
    fn test_add_pattern_rule() {
        let mut recognizer = SimpleNamedEntityRecognizer::new();

        // Test adding a valid pattern
        assert!(recognizer
            .add_pattern_rule(r"^\d{3}-\d{3}-\d{4}$", EntityType::Other)
            .is_ok());

        // Test adding an invalid pattern
        assert!(recognizer
            .add_pattern_rule(r"[", EntityType::Other)
            .is_err());
    }

    #[test]
    fn test_statistics() {
        let mut recognizer = SimpleNamedEntityRecognizer::new();
        recognizer.add_entities(
            EntityType::Person,
            vec!["John".to_string(), "Jane".to_string()],
        );
        recognizer.add_entity(EntityType::Location, "Paris".to_string());

        let stats = recognizer.get_statistics();
        // Note: The recognizer now comes with default entities, so we check for at least the added ones
        assert!(stats.get(&EntityType::Person).unwrap() >= &2);
        assert!(stats.get(&EntityType::Location).unwrap() >= &1);
    }

    #[test]
    fn test_enhanced_entity_recognition() {
        let recognizer = SimpleNamedEntityRecognizer::new();

        let text = "Dr. Smith visited New York on 12/25/2023 and spent $1,500.50 on gifts.";
        let entities = recognizer.recognize_entities(text).unwrap();

        // Should find: Dr. Smith (person), New York (location), 12/25/2023 (datetime), $1,500.50 (money)
        assert!(entities.len() >= 3); // At least datetime and money should be found

        // Check for specific entity types
        let entity_types: Vec<EntityType> =
            entities.iter().map(|(_, et, _, _)| et.clone()).collect();
        assert!(entity_types.contains(&EntityType::DateTime));
        assert!(entity_types.contains(&EntityType::Money));
    }

    #[test]
    fn test_multi_word_entities() {
        let mut recognizer = SimpleNamedEntityRecognizer::new();
        recognizer.add_entity(EntityType::Location, "New York".to_string());
        recognizer.add_entity(EntityType::Organization, "Google Inc".to_string());

        let text = "I work at Google Inc in New York City";
        let entities = recognizer.recognize_entities(text).unwrap();

        // Should find multi-word entities
        let entity_texts: Vec<String> = entities
            .iter()
            .map(|(text, _, _, _)| text.clone())
            .collect();
        assert!(entity_texts.contains(&"New York".to_string()));
        assert!(entity_texts.contains(&"Google Inc".to_string()));
    }

    #[test]
    fn test_confidence_scoring() {
        let mut recognizer = SimpleNamedEntityRecognizer::new();
        recognizer.add_entity(EntityType::Person, "Alice".to_string());

        // Dictionary match should have high confidence
        let confidence = recognizer.get_entity_confidence("Alice", &EntityType::Person);
        assert!(confidence >= 0.9);

        // Pattern match should have medium confidence
        let confidence = recognizer.get_entity_confidence("$100", &EntityType::Money);
        assert!((0.8..0.9).contains(&confidence));

        // No match should have zero confidence
        let confidence = recognizer.get_entity_confidence("unknown", &EntityType::Person);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_language_specific_recognizer() {
        let spanish_recognizer = SimpleNamedEntityRecognizer::new_for_language(LanguageCode::Es);

        // Should recognize Spanish names
        assert_eq!(
            spanish_recognizer.is_named_entity("José").unwrap(),
            Some(EntityType::Person)
        );
        assert_eq!(
            spanish_recognizer.is_named_entity("Madrid").unwrap(),
            Some(EntityType::Location)
        );
    }

    #[test]
    fn test_advanced_patterns() {
        let recognizer = SimpleNamedEntityRecognizer::new();

        // Test email pattern
        assert_eq!(
            recognizer.is_named_entity("user@example.com").unwrap(),
            Some(EntityType::Technical)
        );

        // Test URL pattern
        assert_eq!(
            recognizer.is_named_entity("https://example.com").unwrap(),
            Some(EntityType::Technical)
        );

        // Test technical units
        assert_eq!(
            recognizer.is_named_entity("16GB").unwrap(),
            Some(EntityType::Technical)
        );
    }

    #[test]
    fn test_contextual_recognition() {
        let recognizer = SimpleNamedEntityRecognizer::new();

        let text = "Dr. Johnson works at Microsoft Corporation.";
        let entities = recognizer.recognize_entities(text).unwrap();

        // Should find entities based on context
        let entity_types: Vec<EntityType> =
            entities.iter().map(|(_, et, _, _)| et.clone()).collect();
        // At least should find Microsoft as organization due to default entities
        assert!(!entity_types.is_empty());
    }

    #[test]
    fn test_conflict_resolution() {
        let mut recognizer = SimpleNamedEntityRecognizer::new();
        recognizer.add_entity(EntityType::Person, "Apple".to_string());
        // Apple is also in default brands

        let text = "Apple is a great company.";
        let entities = recognizer.recognize_entities(text).unwrap();

        // Should only have one entity (conflict resolved)
        let apple_entities: Vec<_> = entities
            .iter()
            .filter(|(text, _, _, _)| text == "Apple")
            .collect();
        assert_eq!(apple_entities.len(), 1);
    }
}
