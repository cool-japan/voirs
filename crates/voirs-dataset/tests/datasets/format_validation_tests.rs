//! Dataset format validation tests
//!
//! Tests for validating dataset format consistency and detecting corruption.

use std::collections::HashMap;
use voirs_dataset::*;

#[tokio::test]
async fn test_audio_format_validation() {
    // Test that audio formats are properly validated
    let mut dataset = MemoryDataset::new("audio-format-test".to_string());

    // Create samples with different audio characteristics
    let valid_audio = AudioData::new(vec![0.1; 22050], 22050, 1); // 1 second, mono
    let stereo_audio = AudioData::new(vec![0.1; 44100], 22050, 2); // 1 second, stereo
    let high_sr_audio = AudioData::new(vec![0.1; 48000], 48000, 1); // 1 second, 48kHz

    dataset.add_item(DatasetSample::new(
        "valid-001".to_string(),
        "Valid audio sample".to_string(),
        valid_audio,
        LanguageCode::EnUs,
    ));

    dataset.add_item(DatasetSample::new(
        "stereo-001".to_string(),
        "Stereo audio sample".to_string(),
        stereo_audio,
        LanguageCode::EnUs,
    ));

    dataset.add_item(DatasetSample::new(
        "highsr-001".to_string(),
        "High sample rate audio".to_string(),
        high_sr_audio,
        LanguageCode::EnUs,
    ));

    // Validate format consistency
    let format_report = validate_audio_formats(&dataset);

    assert!(format_report.sample_rates.len() > 1); // Multiple sample rates detected
    assert!(format_report.channel_counts.len() > 1); // Multiple channel counts detected
    assert!(format_report.inconsistent_formats); // Should flag format inconsistency

    // Check individual sample properties
    let sample1 = dataset.get_item(0).unwrap();
    assert_eq!(sample1.audio.sample_rate(), 22050);
    assert_eq!(sample1.audio.channels(), 1);

    let sample2 = dataset.get_item(1).unwrap();
    assert_eq!(sample2.audio.sample_rate(), 22050);
    assert_eq!(sample2.audio.channels(), 2);

    let sample3 = dataset.get_item(2).unwrap();
    assert_eq!(sample3.audio.sample_rate(), 48000);
    assert_eq!(sample3.audio.channels(), 1);
}

#[tokio::test]
async fn test_audio_corruption_detection() {
    // Test detection of corrupted or problematic audio
    let mut dataset = MemoryDataset::new("corruption-test".to_string());

    // Create samples with various audio issues
    let empty_audio = AudioData::new(vec![], 22050, 1); // Empty audio
    let silent_audio = AudioData::new(vec![0.0; 22050], 22050, 1); // Silent audio
    let clipped_audio = create_clipped_audio(); // Clipped audio
    let nan_audio = create_nan_audio(); // Audio with NaN values

    dataset.add_item(DatasetSample::new(
        "empty-001".to_string(),
        "Empty audio sample".to_string(),
        empty_audio,
        LanguageCode::EnUs,
    ));

    dataset.add_item(DatasetSample::new(
        "silent-001".to_string(),
        "Silent audio sample".to_string(),
        silent_audio,
        LanguageCode::EnUs,
    ));

    dataset.add_item(DatasetSample::new(
        "clipped-001".to_string(),
        "Clipped audio sample".to_string(),
        clipped_audio,
        LanguageCode::EnUs,
    ));

    dataset.add_item(DatasetSample::new(
        "nan-001".to_string(),
        "NaN audio sample".to_string(),
        nan_audio,
        LanguageCode::EnUs,
    ));

    // Run corruption detection
    let corruption_report = detect_audio_corruption(&dataset);

    assert!(corruption_report.empty_audio_count > 0);
    assert!(corruption_report.silent_audio_count > 0);
    assert!(corruption_report.clipped_audio_count > 0);
    assert!(corruption_report.invalid_samples_count > 0);
    assert!(!corruption_report.is_clean);

    // Validate the dataset and check for errors
    let validation_report = dataset.validate().unwrap();
    assert!(!validation_report.is_valid);
    assert!(!validation_report.errors.is_empty());
}

#[tokio::test]
async fn test_text_format_validation() {
    // Test text format consistency and validation
    let mut dataset = MemoryDataset::new("text-format-test".to_string());

    // Create samples with various text characteristics
    let samples = vec![
        ("normal-001", "This is normal text.", LanguageCode::EnUs),
        (
            "unicode-001",
            "This has unicode: café, naïve, résumé",
            LanguageCode::EnUs,
        ),
        (
            "japanese-001",
            "これは日本語のテキストです。",
            LanguageCode::Ja,
        ),
        ("mixed-001", "Mixed: English and 日本語", LanguageCode::EnUs),
        (
            "special-001",
            "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
            LanguageCode::EnUs,
        ),
        ("numeric-001", "Numbers: 123 456.789", LanguageCode::EnUs),
        (
            "whitespace-001",
            "   Extra    whitespace   ",
            LanguageCode::EnUs,
        ),
        (
            "newline-001",
            "Text with\nnewlines\nand\ttabs",
            LanguageCode::EnUs,
        ),
    ];

    for (id, text, lang) in samples {
        let audio = AudioData::new(vec![0.1; 22050], 22050, 1);
        dataset.add_item(DatasetSample::new(
            id.to_string(),
            text.to_string(),
            audio,
            lang,
        ));
    }

    // Validate text formats
    let text_report = validate_text_formats(&dataset);

    assert!(text_report.unicode_characters_detected);
    assert!(text_report.special_characters_detected);
    assert!(text_report.whitespace_issues_detected);
    assert!(text_report.mixed_scripts_detected);

    // Check character set consistency
    assert!(text_report.character_sets.len() > 1); // Multiple character sets

    // Verify specific text properties
    let unicode_sample = dataset.get_item(1).unwrap();
    assert!(unicode_sample.text.contains("café"));

    let japanese_sample = dataset.get_item(2).unwrap();
    assert!(japanese_sample.text.contains("日本語"));
}

#[tokio::test]
async fn test_metadata_consistency() {
    // Test that metadata is consistent across samples
    let mut dataset = MemoryDataset::new("metadata-test".to_string());

    // Create samples with different metadata structures
    let mut sample1 = create_basic_sample("meta-001", "Sample one");
    sample1.metadata.insert(
        "version".to_string(),
        serde_json::Value::String("1.0".to_string()),
    );
    sample1.metadata.insert(
        "quality".to_string(),
        serde_json::Value::Number(serde_json::Number::from(85)),
    );

    let mut sample2 = create_basic_sample("meta-002", "Sample two");
    sample2.metadata.insert(
        "version".to_string(),
        serde_json::Value::String("1.1".to_string()),
    );
    sample2.metadata.insert(
        "quality".to_string(),
        serde_json::Value::Number(serde_json::Number::from(92)),
    );
    sample2.metadata.insert(
        "source".to_string(),
        serde_json::Value::String("recording".to_string()),
    );

    let mut sample3 = create_basic_sample("meta-003", "Sample three");
    sample3.metadata.insert(
        "quality".to_string(),
        serde_json::Value::String("high".to_string()),
    ); // Different type!

    dataset.add_item(sample1);
    dataset.add_item(sample2);
    dataset.add_item(sample3);

    // Validate metadata consistency
    let metadata_report = validate_metadata_consistency(&dataset);

    assert!(!metadata_report.consistent_schema);
    assert!(!metadata_report.type_mismatches.is_empty());
    assert!(!metadata_report.missing_fields.is_empty());

    // Check specific inconsistencies
    assert!(metadata_report.type_mismatches.contains_key("quality"));
    assert!(metadata_report
        .missing_fields
        .contains(&"source".to_string()));
}

#[tokio::test]
async fn test_id_uniqueness_validation() {
    // Test that sample IDs are unique
    let mut dataset = MemoryDataset::new("id-test".to_string());

    // Add samples with some duplicate IDs
    dataset.add_item(create_basic_sample("unique-001", "First sample"));
    dataset.add_item(create_basic_sample("unique-002", "Second sample"));
    dataset.add_item(create_basic_sample("unique-001", "Duplicate ID sample")); // Duplicate!
    dataset.add_item(create_basic_sample("unique-003", "Third sample"));
    dataset.add_item(create_basic_sample("unique-002", "Another duplicate")); // Another duplicate!

    // Validate ID uniqueness
    let id_report = validate_id_uniqueness(&dataset);

    assert!(!id_report.all_unique);
    assert_eq!(id_report.duplicate_ids.len(), 2); // Two duplicate IDs
    assert!(id_report.duplicate_ids.contains(&"unique-001".to_string()));
    assert!(id_report.duplicate_ids.contains(&"unique-002".to_string()));
    assert_eq!(id_report.total_samples, 5);
    assert_eq!(id_report.unique_samples, 3);
}

// Helper functions and structs

fn create_basic_sample(id: &str, text: &str) -> DatasetSample {
    let audio = AudioData::new(vec![0.1; 22050], 22050, 1);
    DatasetSample::new(id.to_string(), text.to_string(), audio, LanguageCode::EnUs)
}

fn create_clipped_audio() -> AudioData {
    // Create audio with clipped samples
    let mut samples = vec![0.1; 22050];
    for i in (0..samples.len()).step_by(100) {
        samples[i] = 1.0; // Clipped positive
        if i + 50 < samples.len() {
            samples[i + 50] = -1.0; // Clipped negative
        }
    }
    AudioData::new(samples, 22050, 1)
}

fn create_nan_audio() -> AudioData {
    // Create audio with NaN values
    let mut samples = vec![0.1; 22050];
    samples[1000] = f32::NAN;
    samples[2000] = f32::INFINITY;
    samples[3000] = f32::NEG_INFINITY;
    AudioData::new(samples, 22050, 1)
}

// Report structures for validation results

#[derive(Debug)]
struct AudioFormatReport {
    sample_rates: Vec<u32>,
    channel_counts: Vec<u32>,
    inconsistent_formats: bool,
}

#[derive(Debug)]
struct AudioCorruptionReport {
    empty_audio_count: usize,
    silent_audio_count: usize,
    clipped_audio_count: usize,
    invalid_samples_count: usize,
    is_clean: bool,
}

#[derive(Debug)]
struct TextFormatReport {
    unicode_characters_detected: bool,
    special_characters_detected: bool,
    whitespace_issues_detected: bool,
    mixed_scripts_detected: bool,
    character_sets: Vec<String>,
}

#[derive(Debug)]
struct MetadataConsistencyReport {
    consistent_schema: bool,
    type_mismatches: HashMap<String, Vec<String>>,
    missing_fields: Vec<String>,
}

#[derive(Debug)]
struct IdUniquenessReport {
    all_unique: bool,
    duplicate_ids: Vec<String>,
    total_samples: usize,
    unique_samples: usize,
}

// Validation functions

fn validate_audio_formats(dataset: &MemoryDataset) -> AudioFormatReport {
    let mut sample_rates = Vec::new();
    let mut channel_counts = Vec::new();

    for i in 0..dataset.len() {
        let sample = dataset.get_item(i).unwrap();
        let sr = sample.audio.sample_rate();
        let ch = sample.audio.channels();

        if !sample_rates.contains(&sr) {
            sample_rates.push(sr);
        }
        if !channel_counts.contains(&ch) {
            channel_counts.push(ch);
        }
    }

    sample_rates.sort();
    channel_counts.sort();

    AudioFormatReport {
        inconsistent_formats: sample_rates.len() > 1 || channel_counts.len() > 1,
        sample_rates,
        channel_counts,
    }
}

fn detect_audio_corruption(dataset: &MemoryDataset) -> AudioCorruptionReport {
    let mut empty_count = 0;
    let mut silent_count = 0;
    let mut clipped_count = 0;
    let mut invalid_count = 0;

    for i in 0..dataset.len() {
        let sample = dataset.get_item(i).unwrap();
        let samples = sample.audio.samples();

        // Check for empty audio
        if samples.is_empty() {
            empty_count += 1;
            continue;
        }

        // Check for silent audio
        let max_amplitude = samples.iter().fold(0.0f32, |max, &s| max.max(s.abs()));
        if max_amplitude < 0.001 {
            silent_count += 1;
        }

        // Check for clipping
        let clipped_samples = samples.iter().filter(|&&s| s.abs() >= 0.999).count();
        if clipped_samples > samples.len() / 100 {
            // More than 1% clipped
            clipped_count += 1;
        }

        // Check for invalid values (NaN, infinity)
        let invalid_samples = samples.iter().filter(|&&s| !s.is_finite()).count();
        if invalid_samples > 0 {
            invalid_count += 1;
        }
    }

    AudioCorruptionReport {
        empty_audio_count: empty_count,
        silent_audio_count: silent_count,
        clipped_audio_count: clipped_count,
        invalid_samples_count: invalid_count,
        is_clean: empty_count == 0 && silent_count == 0 && clipped_count == 0 && invalid_count == 0,
    }
}

fn validate_text_formats(dataset: &MemoryDataset) -> TextFormatReport {
    let mut unicode_detected = false;
    let mut special_detected = false;
    let mut whitespace_issues = false;
    let mut mixed_scripts = false;
    let mut character_sets = Vec::new();

    for i in 0..dataset.len() {
        let sample = dataset.get_item(i).unwrap();
        let text = &sample.text;

        // Check for Unicode characters
        if text.chars().any(|c| c as u32 > 127) {
            unicode_detected = true;
        }

        // Check for special characters
        if text
            .chars()
            .any(|c| "!@#$%^&*()_+-=[]{}|;':\",./<>?".contains(c))
        {
            special_detected = true;
        }

        // Check for whitespace issues
        if text.starts_with(' ') || text.ends_with(' ') || text.contains("  ") {
            whitespace_issues = true;
        }

        // Check for mixed scripts (simplified detection)
        let has_latin = text.chars().any(|c| c.is_alphabetic() && (c as u32) < 256);
        let has_cjk = text.chars().any(|c| {
            let code = c as u32;
            (0x4E00..=0x9FFF).contains(&code) || // CJK Unified Ideographs
            (0x3040..=0x309F).contains(&code) || // Hiragana
            (0x30A0..=0x30FF).contains(&code) // Katakana
        });

        if has_latin && has_cjk {
            mixed_scripts = true;
        }

        // Determine character set
        let charset = if text.is_ascii() {
            "ASCII"
        } else if has_cjk {
            "CJK"
        } else {
            "Unicode"
        };

        if !character_sets.contains(&charset.to_string()) {
            character_sets.push(charset.to_string());
        }
    }

    TextFormatReport {
        unicode_characters_detected: unicode_detected,
        special_characters_detected: special_detected,
        whitespace_issues_detected: whitespace_issues,
        mixed_scripts_detected: mixed_scripts,
        character_sets,
    }
}

fn validate_metadata_consistency(dataset: &MemoryDataset) -> MetadataConsistencyReport {
    let mut all_keys = std::collections::HashSet::new();
    let mut key_types: HashMap<String, std::collections::HashSet<String>> = HashMap::new();

    // Collect all keys and their types
    for i in 0..dataset.len() {
        let sample = dataset.get_item(i).unwrap();
        for (key, value) in &sample.metadata {
            all_keys.insert(key.clone());

            let value_type = match value {
                serde_json::Value::String(_) => "string",
                serde_json::Value::Number(_) => "number",
                serde_json::Value::Bool(_) => "boolean",
                serde_json::Value::Array(_) => "array",
                serde_json::Value::Object(_) => "object",
                serde_json::Value::Null => "null",
            };

            key_types
                .entry(key.clone())
                .or_default()
                .insert(value_type.to_string());
        }
    }

    // Check for type mismatches
    let mut type_mismatches = HashMap::new();
    for (key, types) in &key_types {
        if types.len() > 1 {
            type_mismatches.insert(key.clone(), types.iter().cloned().collect());
        }
    }

    // Check for missing fields
    let mut missing_fields = Vec::new();
    for key in &all_keys {
        let mut missing_count = 0;
        for i in 0..dataset.len() {
            let sample = dataset.get_item(i).unwrap();
            if !sample.metadata.contains_key(key) {
                missing_count += 1;
            }
        }
        if missing_count > 0 && missing_count < dataset.len() {
            missing_fields.push(key.clone());
        }
    }

    MetadataConsistencyReport {
        consistent_schema: type_mismatches.is_empty() && missing_fields.is_empty(),
        type_mismatches,
        missing_fields,
    }
}

fn validate_id_uniqueness(dataset: &MemoryDataset) -> IdUniquenessReport {
    let mut id_counts: HashMap<String, usize> = HashMap::new();

    for i in 0..dataset.len() {
        let sample = dataset.get_item(i).unwrap();
        *id_counts.entry(sample.id.clone()).or_insert(0) += 1;
    }

    let duplicate_ids: Vec<String> = id_counts
        .iter()
        .filter(|(_, &count)| count > 1)
        .map(|(id, _)| id.clone())
        .collect();

    IdUniquenessReport {
        all_unique: duplicate_ids.is_empty(),
        duplicate_ids,
        total_samples: dataset.len(),
        unique_samples: id_counts.len(),
    }
}
