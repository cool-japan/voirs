//! Integration tests for voirs-dataset
//!
//! These tests verify end-to-end functionality, multi-format compatibility,
//! and performance characteristics of the dataset management system.

mod datasets;

use std::time::Instant;
use tempfile::TempDir;
use tokio::fs;
use voirs_dataset::datasets::dummy::{AudioType, DummyConfig, DummyDataset, TextType};
use voirs_dataset::datasets::ljspeech::LjSpeechDataset;
use voirs_dataset::traits::{Dataset, DatasetSample};

/// Test end-to-end dummy dataset workflow
#[tokio::test]
async fn test_dummy_dataset_workflow() {
    // Create dataset with various configurations
    let configs = [
        DummyConfig {
            num_samples: 50,
            audio_type: AudioType::SineWave,
            text_type: TextType::Lorem,
            ..Default::default()
        },
        DummyConfig {
            num_samples: 30,
            audio_type: AudioType::WhiteNoise,
            text_type: TextType::Phonetic,
            seed: Some(12345),
            ..Default::default()
        },
        DummyConfig {
            num_samples: 20,
            audio_type: AudioType::Mixed,
            text_type: TextType::RandomWords,
            min_duration: 2.0,
            max_duration: 8.0,
            ..Default::default()
        },
    ];

    for (i, config) in configs.iter().enumerate() {
        let dataset = DummyDataset::with_config(config.clone());

        // Test basic properties
        assert_eq!(dataset.len(), config.num_samples);
        assert!(!dataset.is_empty());

        // Test metadata
        let metadata = dataset.metadata();
        assert_eq!(metadata.name, "DummyDataset");
        assert_eq!(metadata.total_samples, config.num_samples);
        assert!(metadata.total_duration > 0.0);

        // Test sample access
        for j in 0..std::cmp::min(5, config.num_samples) {
            let sample = dataset.get(j).await.unwrap();
            assert!(!sample.id().is_empty());
            assert!(!sample.text().is_empty());
            assert!(sample.duration() > 0.0);
            assert_eq!(sample.language(), config.language.as_str());
        }

        // Test statistics
        let stats = dataset.statistics().await.unwrap();
        assert_eq!(stats.total_items, config.num_samples);
        assert!(stats.total_duration > 0.0);
        assert!(stats.average_duration > 0.0);
        assert!(!stats.language_distribution.is_empty());
        assert!(!stats.speaker_distribution.is_empty());

        // Test validation
        let report = dataset.validate().await.unwrap();
        assert!(report.is_valid);
        assert_eq!(report.items_validated, config.num_samples);
        assert!(report.errors.is_empty());

        println!(
            "Workflow test {} passed with {} samples",
            i + 1,
            config.num_samples
        );
    }
}

/// Test audio generation varieties
#[tokio::test]
async fn test_audio_generation_types() {
    let audio_types = vec![
        AudioType::SineWave,
        AudioType::WhiteNoise,
        AudioType::PinkNoise,
        AudioType::Silence,
        AudioType::Mixed,
    ];

    for audio_type in audio_types {
        let config = DummyConfig {
            num_samples: 10,
            audio_type,
            min_duration: 1.0,
            max_duration: 3.0,
            ..Default::default()
        };

        let dataset = DummyDataset::with_config(config);
        assert_eq!(dataset.len(), 10);

        // Test first sample
        let sample = dataset.get(0).await.unwrap();
        assert!(sample.duration() >= 1.0 && sample.duration() <= 3.0);
        assert_eq!(sample.audio.sample_rate(), 22050);
        assert_eq!(sample.audio.channels(), 1);

        // Verify audio content based on type
        let samples = sample.audio.samples();
        match audio_type {
            AudioType::Silence => {
                assert!(samples.iter().all(|&s| s == 0.0));
            }
            AudioType::SineWave
            | AudioType::WhiteNoise
            | AudioType::PinkNoise
            | AudioType::Mixed => {
                assert!(samples.iter().any(|&s| s != 0.0));
            }
        }

        println!("Audio type {audio_type:?} test passed");
    }
}

/// Test text generation varieties
#[tokio::test]
async fn test_text_generation_types() {
    let text_types = vec![
        TextType::Lorem,
        TextType::Phonetic,
        TextType::Numbers,
        TextType::RandomWords,
    ];

    for text_type in text_types {
        let config = DummyConfig {
            num_samples: 15,
            text_type,
            seed: Some(42), // Use seed for consistent testing
            ..Default::default()
        };

        let dataset = DummyDataset::with_config(config);
        assert_eq!(dataset.len(), 15);

        // Test all samples have appropriate text
        for i in 0..15 {
            let sample = dataset.get(i).await.unwrap();
            let text = sample.text();
            assert!(!text.is_empty());

            match text_type {
                TextType::Lorem => {
                    let text_lower = text.to_lowercase();
                    // Check for any lorem ipsum words (more comprehensive check)
                    let lorem_words = vec![
                        "lorem",
                        "ipsum",
                        "dolor",
                        "sit",
                        "amet",
                        "consectetur",
                        "adipiscing",
                        "elit",
                        "sed",
                        "do",
                        "eiusmod",
                        "tempor",
                        "incididunt",
                        "ut",
                        "labore",
                        "et",
                        "dolore",
                        "magna",
                        "aliqua",
                        "enim",
                        "ad",
                        "minim",
                        "veniam",
                        "quis",
                        "nostrud",
                        "exercitation",
                        "ullamco",
                        "laboris",
                        "nisi",
                        "aliquip",
                        "ex",
                        "ea",
                        "commodo",
                        "consequat",
                        "duis",
                        "aute",
                        "irure",
                        "in",
                        "reprehenderit",
                        "voluptate",
                        "velit",
                        "esse",
                        "cillum",
                        "fugiat",
                        "nulla",
                        "pariatur",
                        "excepteur",
                        "sint",
                        "occaecat",
                        "cupidatat",
                        "non",
                        "proident",
                        "sunt",
                        "culpa",
                        "qui",
                        "officia",
                        "deserunt",
                        "mollit",
                        "anim",
                        "id",
                        "est",
                        "laborum",
                    ];
                    let has_lorem_word = lorem_words.iter().any(|word| text_lower.contains(word));
                    assert!(
                        has_lorem_word,
                        "Text '{text}' should contain at least one lorem ipsum word"
                    );
                }
                TextType::Phonetic => {
                    // Phonetic samples should be complete sentences from the predefined list
                    let phonetic_samples = [
                        "The quick brown fox jumps over the lazy dog.",
                        "Pack my box with five dozen liquor jugs.",
                        "How vexingly quick daft zebras jump!",
                        "Bright vixens jump; dozy fowl quack.",
                        "Sphinx of black quartz, judge my vow.",
                        "Two driven jocks help fax my big quiz.",
                        "Quick zephyrs blow, vexing daft Jim.",
                        "The five boxing wizards jump quickly.",
                        "Jackdaws love my big sphinx of quartz.",
                        "Mr. Jock, TV quiz PhD., bags few lynx.",
                    ];
                    assert!(
                        phonetic_samples.contains(&text),
                        "Text '{text}' should be one of the predefined phonetic samples"
                    );
                    assert!(text.ends_with('.') || text.ends_with('!'));
                }
                TextType::Numbers => {
                    let text_lower = text.to_lowercase();
                    assert!(
                        text_lower.contains("number")
                            || text_lower.contains("sample")
                            || text_lower.contains("item")
                            || text_lower.contains("clip")
                    );
                }
                TextType::RandomWords => {
                    assert!(text.len() > 3); // Should have multiple words
                }
            }
        }

        println!("Text type {text_type:?} test passed");
    }
}

/// Test dataset reproducibility
#[tokio::test]
async fn test_dataset_reproducibility() {
    let seed = 98765;
    let config = DummyConfig {
        num_samples: 25,
        seed: Some(seed),
        audio_type: AudioType::Mixed,
        text_type: TextType::RandomWords,
        ..Default::default()
    };

    // Create two identical datasets
    let dataset1 = DummyDataset::with_config(config.clone());
    let dataset2 = DummyDataset::with_config(config);

    assert_eq!(dataset1.len(), dataset2.len());

    // Compare all samples
    for i in 0..dataset1.len() {
        let sample1 = dataset1.get(i).await.unwrap();
        let sample2 = dataset2.get(i).await.unwrap();

        assert_eq!(sample1.id(), sample2.id());
        assert_eq!(sample1.text(), sample2.text());
        assert_eq!(sample1.duration(), sample2.duration());
        assert_eq!(sample1.language(), sample2.language());

        // Audio samples should be identical
        assert_eq!(sample1.audio.samples().len(), sample2.audio.samples().len());
        for (s1, s2) in sample1
            .audio
            .samples()
            .iter()
            .zip(sample2.audio.samples().iter())
        {
            assert!((s1 - s2).abs() < f32::EPSILON);
        }
    }

    println!(
        "Reproducibility test passed with {} samples",
        dataset1.len()
    );
}

/// Test large dataset performance
#[tokio::test]
async fn test_large_dataset_performance() {
    let config = DummyConfig {
        num_samples: 1000,
        audio_type: AudioType::SineWave,
        text_type: TextType::Numbers,
        min_duration: 0.5,
        max_duration: 2.0,
        ..Default::default()
    };

    // Measure creation time
    let start = Instant::now();
    let dataset = DummyDataset::with_config(config);
    let creation_time = start.elapsed();

    assert_eq!(dataset.len(), 1000);
    println!("Created 1000-sample dataset in {creation_time:?}");

    // Measure access time for random samples
    let start = Instant::now();
    let sample_indices = vec![0, 100, 200, 500, 750, 999];
    for &index in &sample_indices {
        let sample = dataset.get(index).await.unwrap();
        assert!(!sample.id().is_empty());
        assert!(sample.duration() > 0.0);
    }
    let access_time = start.elapsed();

    println!(
        "Accessed {} samples in {:?}",
        sample_indices.len(),
        access_time
    );

    // Measure statistics calculation time
    let start = Instant::now();
    let stats = dataset.statistics().await.unwrap();
    let stats_time = start.elapsed();

    assert_eq!(stats.total_items, 1000);
    assert!(stats.total_duration > 0.0);
    println!("Calculated statistics in {stats_time:?}");

    // Measure validation time
    let start = Instant::now();
    let report = dataset.validate().await.unwrap();
    let validation_time = start.elapsed();

    assert!(report.is_valid);
    assert_eq!(report.items_validated, 1000);
    println!("Validated dataset in {validation_time:?}");

    // Performance assertions (reasonable thresholds)
    assert!(
        creation_time.as_millis() < 5000,
        "Dataset creation too slow: {creation_time:?}"
    );
    assert!(
        access_time.as_millis() < 100,
        "Sample access too slow: {access_time:?}"
    );
    assert!(
        stats_time.as_millis() < 200,
        "Statistics calculation too slow: {stats_time:?}"
    );
    assert!(
        validation_time.as_millis() < 500,
        "Validation too slow: {validation_time:?}"
    );
}

/// Test different audio formats and configurations
#[tokio::test]
async fn test_audio_format_compatibility() {
    let sample_rates = vec![8000, 16000, 22050, 44100, 48000];
    let channel_configs = vec![1, 2];

    for &sample_rate in &sample_rates {
        for &channels in &channel_configs {
            let config = DummyConfig {
                num_samples: 5,
                sample_rate,
                channels,
                audio_type: AudioType::SineWave,
                min_duration: 1.0,
                max_duration: 2.0,
                ..Default::default()
            };

            let dataset = DummyDataset::with_config(config);
            let sample = dataset.get(0).await.unwrap();

            assert_eq!(sample.audio.sample_rate(), sample_rate);
            assert_eq!(sample.audio.channels(), channels);

            // Verify sample count matches expected duration
            let expected_samples =
                (sample.duration() * sample_rate as f32 * channels as f32) as usize;
            let actual_samples = sample.audio.samples().len();
            let tolerance = channels as usize; // Allow some tolerance for rounding

            assert!(
                (actual_samples as i32 - expected_samples as i32).abs() <= tolerance as i32,
                "Sample count mismatch: expected ~{}, got {} (sr: {}, ch: {}, dur: {})",
                expected_samples,
                actual_samples,
                sample_rate,
                channels,
                sample.duration()
            );

            println!("Audio format test passed: {sample_rate}Hz, {channels} channel(s)");
        }
    }
}

/// Test LJSpeech dataset structure (without actual data)
#[tokio::test]
async fn test_ljspeech_structure() {
    // Test error handling when dataset doesn't exist
    let temp_dir = TempDir::new().unwrap();
    let non_existent_path = temp_dir.path().join("non_existent");

    let result = LjSpeechDataset::load(&non_existent_path).await;
    assert!(result.is_err());

    // Test error handling when metadata.csv doesn't exist
    let empty_dir = temp_dir.path().join("empty_ljspeech");
    fs::create_dir_all(&empty_dir).await.unwrap();

    let result = LjSpeechDataset::load(&empty_dir).await;
    assert!(result.is_err());

    println!("LJSpeech structure validation test passed");
}

/// Test memory efficiency with multiple datasets
#[tokio::test]
async fn test_memory_efficiency() {
    let mut datasets = Vec::new();

    // Create multiple small datasets
    for i in 0..10 {
        let config = DummyConfig {
            num_samples: 50,
            seed: Some(i as u64),
            audio_type: AudioType::SineWave,
            min_duration: 0.5,
            max_duration: 1.5,
            ..Default::default()
        };

        let dataset = DummyDataset::with_config(config);
        datasets.push(dataset);
    }

    // Test that all datasets are accessible
    for (i, dataset) in datasets.iter().enumerate() {
        assert_eq!(dataset.len(), 50);

        let sample = dataset.get(0).await.unwrap();
        assert!(!sample.id().is_empty());
        assert!(sample.duration() > 0.0);

        println!("Dataset {} accessible with {} samples", i, dataset.len());
    }

    // Test concurrent access
    let first_dataset = &datasets[0];
    let last_dataset = &datasets[datasets.len() - 1];

    let (sample1, sample2) = tokio::join!(first_dataset.get(10), last_dataset.get(10));

    assert!(sample1.is_ok());
    assert!(sample2.is_ok());

    println!(
        "Memory efficiency test passed with {} datasets",
        datasets.len()
    );
}

/// Test error handling and edge cases
#[tokio::test]
async fn test_error_handling() {
    let dataset = DummyDataset::small(); // 10 samples

    // Test out-of-bounds access
    let result = dataset.get(100).await;
    assert!(result.is_err());

    // Test edge case: zero-length dataset
    let empty_config = DummyConfig {
        num_samples: 0,
        ..Default::default()
    };
    let empty_dataset = DummyDataset::with_config(empty_config);
    assert_eq!(empty_dataset.len(), 0);
    assert!(empty_dataset.is_empty());

    // Accessing empty dataset should fail
    let result = empty_dataset.get(0).await;
    assert!(result.is_err());

    // Statistics on empty dataset should work
    let stats = empty_dataset.statistics().await.unwrap();
    assert_eq!(stats.total_items, 0);
    assert_eq!(stats.total_duration, 0.0);

    // Validation on empty dataset should work
    let report = empty_dataset.validate().await.unwrap();
    assert!(report.is_valid);
    assert_eq!(report.items_validated, 0);

    println!("Error handling test passed");
}

/// Test dataset iterator functionality
#[tokio::test]
async fn test_dataset_iteration() {
    let dataset = DummyDataset::small(); // 10 samples
    let iterator = dataset.iter();

    let indices: Vec<usize> = iterator.collect();
    assert_eq!(indices.len(), 10);
    assert_eq!(indices, (0..10).collect::<Vec<_>>());

    // Test that we can use the iterator to access all samples
    let mut sample_count = 0;
    for index in dataset.iter() {
        let sample = dataset.get(index).await.unwrap();
        assert!(!sample.id().is_empty());
        sample_count += 1;
    }
    assert_eq!(sample_count, 10);

    println!("Dataset iteration test passed");
}

/// Test batch access functionality
#[tokio::test]
async fn test_batch_access() {
    let dataset = DummyDataset::with_config(DummyConfig {
        num_samples: 20,
        seed: Some(777),
        ..Default::default()
    });

    // Test batch access
    let indices = vec![0, 5, 10, 15, 19];
    let samples = dataset.get_batch(&indices).await.unwrap();

    assert_eq!(samples.len(), indices.len());

    for (i, sample) in samples.iter().enumerate() {
        let expected_sample = dataset.get(indices[i]).await.unwrap();
        assert_eq!(sample.id(), expected_sample.id());
        assert_eq!(sample.text(), expected_sample.text());
        assert_eq!(sample.duration(), expected_sample.duration());
    }

    // Test empty batch
    let empty_samples = dataset.get_batch(&[]).await.unwrap();
    assert!(empty_samples.is_empty());

    // Test batch with out-of-bounds index
    let result = dataset.get_batch(&[0, 100]).await;
    assert!(result.is_err());

    println!("Batch access test passed");
}

/// Test random sample access
#[tokio::test]
async fn test_random_access() {
    let dataset = DummyDataset::with_config(DummyConfig {
        num_samples: 100,
        seed: Some(555),
        ..Default::default()
    });

    // Get multiple random samples
    let mut random_samples = Vec::new();
    for _ in 0..10 {
        let sample = dataset.get_random().await.unwrap();
        random_samples.push(sample);
    }

    assert_eq!(random_samples.len(), 10);

    // Verify all samples are valid
    for sample in &random_samples {
        assert!(!sample.id().is_empty());
        assert!(!sample.text().is_empty());
        assert!(sample.duration() > 0.0);
    }

    // Test random access on empty dataset
    let empty_dataset = DummyDataset::with_config(DummyConfig {
        num_samples: 0,
        ..Default::default()
    });

    let result = empty_dataset.get_random().await;
    assert!(result.is_err());

    println!("Random access test passed");
}
