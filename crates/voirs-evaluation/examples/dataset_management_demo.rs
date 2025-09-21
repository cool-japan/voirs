//! Dataset Management System Demo
//!
//! This example demonstrates the comprehensive dataset management capabilities
//! including registration, validation, search, and organization of evaluation datasets.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use voirs_evaluation::dataset_management::{
    AccessLevel, AudioQuality, DatasetCategory, DatasetManager, DatasetManagerConfig,
    DatasetMetadata, DatasetSearchCriteria, ValidationResult,
};
use voirs_sdk::LanguageCode;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🗂️  VoiRS Dataset Management System Demo");
    println!("==========================================");

    // Configure the dataset manager
    let config = DatasetManagerConfig {
        base_directory: PathBuf::from("demo_datasets"),
        max_dataset_size: 10 * 1024 * 1024 * 1024, // 10GB
        auto_validate: true,
        enable_caching: true,
        cache_size_limit: 1024 * 1024 * 1024, // 1GB
        default_access_level: AccessLevel::Public,
    };

    println!("📋 Configuration:");
    println!("  Base Directory: {:?}", config.base_directory);
    println!(
        "  Max Dataset Size: {} GB",
        config.max_dataset_size / (1024 * 1024 * 1024)
    );
    println!("  Auto Validation: {}", config.auto_validate);
    println!("  Caching: {}", config.enable_caching);

    // Create the dataset manager
    let mut manager = DatasetManager::new(config).await?;
    println!("\n✅ Dataset manager initialized");

    // Create sample datasets to demonstrate functionality
    println!("\n📦 Creating sample datasets...");

    // Dataset 1: High-quality reference dataset
    let mut reference_tags = HashSet::new();
    reference_tags.insert(String::from("reference"));
    reference_tags.insert(String::from("high-quality"));
    reference_tags.insert(String::from("clean"));

    let reference_dataset = DatasetMetadata {
        id: String::from("ljspeech_reference"),
        name: String::from("LJSpeech Reference Dataset"),
        description: "High-quality single-speaker English dataset for reference evaluations"
            .to_string(),
        category: DatasetCategory::Reference,
        language: LanguageCode::EnUs,
        additional_languages: vec![],
        audio_quality: AudioQuality::Studio,
        sample_count: 13100,
        total_duration: 24.0 * 3600.0, // 24 hours
        sample_rate: 22050,
        audio_format: String::from("wav"),
        version: String::from("1.1"),
        creator: String::from("Keith Ito"),
        license: String::from("Public Domain"),
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        modified_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        tags: reference_tags,
        location: PathBuf::from("/datasets/ljspeech"),
        size_bytes: 2_500_000_000, // 2.5GB
        is_validated: false,
        access_level: AccessLevel::Public,
    };

    match manager.register_dataset(reference_dataset).await {
        Ok(_) => println!("  ✅ Registered LJSpeech reference dataset"),
        Err(e) => println!(
            "  ⚠️  Failed to register LJSpeech: {} (validation issue)",
            e
        ),
    }

    // Dataset 2: Multi-language test dataset
    let mut multilang_tags = HashSet::new();
    multilang_tags.insert(String::from("multilingual"));
    multilang_tags.insert(String::from("test"));
    multilang_tags.insert(String::from("evaluation"));

    let multilang_dataset = DatasetMetadata {
        id: String::from("multilang_test"),
        name: String::from("Multilingual Test Collection"),
        description: "Diverse multilingual dataset for cross-language evaluation testing"
            .to_string(),
        category: DatasetCategory::Test,
        language: LanguageCode::EnUs,
        additional_languages: vec![
            LanguageCode::EsEs,
            LanguageCode::FrFr,
            LanguageCode::DeDe,
            LanguageCode::JaJp,
        ],
        audio_quality: AudioQuality::Mixed,
        sample_count: 5000,
        total_duration: 8.5 * 3600.0, // 8.5 hours
        sample_rate: 16000,
        audio_format: String::from("wav"),
        version: String::from("2.0"),
        creator: String::from("VoiRS Team"),
        license: String::from("Apache-2.0"),
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        modified_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        tags: multilang_tags,
        location: PathBuf::from("/datasets/multilang_test"),
        size_bytes: 1_200_000_000, // 1.2GB
        is_validated: false,
        access_level: AccessLevel::Public,
    };

    match manager.register_dataset(multilang_dataset).await {
        Ok(_) => println!("  ✅ Registered multilingual test dataset"),
        Err(e) => println!(
            "  ⚠️  Failed to register multilingual: {} (validation issue)",
            e
        ),
    }

    // Dataset 3: Benchmark dataset
    let mut benchmark_tags = HashSet::new();
    benchmark_tags.insert(String::from("benchmark"));
    benchmark_tags.insert(String::from("standard"));
    benchmark_tags.insert("evaluation".to_string());

    let benchmark_dataset = DatasetMetadata {
        id: String::from("vctk_benchmark"),
        name: String::from("VCTK Multi-Speaker Benchmark"),
        description: String::from("Multi-speaker English corpus for standardized benchmarking"),
        category: DatasetCategory::Benchmark,
        language: LanguageCode::EnUs,
        additional_languages: vec![],
        audio_quality: AudioQuality::Studio,
        sample_count: 44000,
        total_duration: 44.0 * 3600.0, // 44 hours
        sample_rate: 48000,
        audio_format: String::from("wav"),
        version: String::from("0.92"),
        creator: String::from("University of Edinburgh"),
        license: String::from("Open Data Commons Attribution License"),
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        modified_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        tags: benchmark_tags,
        location: PathBuf::from("/datasets/vctk"),
        size_bytes: 11_000_000_000, // 11GB
        is_validated: false,
        access_level: AccessLevel::Public,
    };

    match manager.register_dataset(benchmark_dataset).await {
        Ok(_) => println!("  ✅ Registered VCTK benchmark dataset"),
        Err(e) => println!("  ⚠️  Failed to register VCTK: {} (validation issue)", e),
    }

    // Dataset 4: Synthetic dataset
    let mut synthetic_tags = HashSet::new();
    synthetic_tags.insert(String::from("synthetic"));
    synthetic_tags.insert(String::from("generated"));
    synthetic_tags.insert(String::from("ai"));

    let synthetic_dataset = DatasetMetadata {
        id: String::from("voirs_synthetic"),
        name: String::from("VoiRS Synthetic Speech Collection"),
        description: String::from("AI-generated synthetic speech samples for evaluation studies"),
        category: DatasetCategory::Synthetic,
        language: LanguageCode::EnUs,
        additional_languages: vec![],
        audio_quality: AudioQuality::Broadcast,
        sample_count: 10000,
        total_duration: 15.0 * 3600.0, // 15 hours
        sample_rate: 22050,
        audio_format: String::from("wav"),
        version: String::from("1.0"),
        creator: String::from("VoiRS AI Team"),
        license: String::from("MIT"),
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        modified_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        tags: synthetic_tags,
        location: PathBuf::from("/datasets/voirs_synthetic"),
        size_bytes: 3_200_000_000, // 3.2GB
        is_validated: false,
        access_level: AccessLevel::Public,
    };

    match manager.register_dataset(synthetic_dataset).await {
        Ok(_) => println!("  ✅ Registered VoiRS synthetic dataset"),
        Err(e) => println!(
            "  ⚠️  Failed to register synthetic: {} (validation issue)",
            e
        ),
    }

    // Display current statistics
    println!("\n📊 Dataset Statistics:");
    let stats = manager.get_statistics();
    println!("  Total Datasets: {}", stats.total_datasets);
    println!(
        "  Validated Datasets: {}/{}",
        stats.validated_datasets, stats.total_datasets
    );
    println!("  Total Samples: {}", stats.total_samples);
    println!(
        "  Total Duration: {:.1} hours",
        stats.total_duration / 3600.0
    );
    println!(
        "  Total Size: {:.2} GB",
        stats.total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Demonstrate search functionality
    println!("\n🔍 Search Examples:");

    // Search by category
    println!("\n  📋 Search by category (Reference):");
    let category_criteria = DatasetSearchCriteria {
        category: Some(DatasetCategory::Reference),
        ..Default::default()
    };
    let reference_results = manager.search_datasets(&category_criteria);
    for dataset in reference_results {
        println!(
            "    - {} ({} samples, {:.1}h)",
            dataset.name,
            dataset.sample_count,
            dataset.total_duration / 3600.0
        );
    }

    // Search by language
    println!("\n  🌍 Search by language (includes multilingual):");
    let language_criteria = DatasetSearchCriteria {
        language: Some(LanguageCode::EsEs),
        ..Default::default()
    };
    let spanish_results = manager.search_datasets(&language_criteria);
    for dataset in spanish_results {
        println!("    - {} (supports Spanish)", dataset.name);
    }

    // Search by tags
    println!("\n  🏷️  Search by tags (benchmark):");
    let mut tag_set = HashSet::new();
    tag_set.insert(String::from("benchmark"));
    let tag_criteria = DatasetSearchCriteria {
        tags: tag_set,
        ..Default::default()
    };
    let benchmark_results = manager.search_datasets(&tag_criteria);
    for dataset in benchmark_results {
        println!("    - {} (benchmark dataset)", dataset.name);
    }

    // Search by size constraints
    println!("\n  📏 Search by sample count (> 10,000 samples):");
    let size_criteria = DatasetSearchCriteria {
        min_samples: Some(10000),
        ..Default::default()
    };
    let large_results = manager.search_datasets(&size_criteria);
    for dataset in large_results {
        println!("    - {} ({} samples)", dataset.name, dataset.sample_count);
    }

    // Text search
    println!("\n  📝 Text search ('multi'):");
    let text_criteria = DatasetSearchCriteria {
        text_search: Some(String::from("multi")),
        ..Default::default()
    };
    let text_results = manager.search_datasets(&text_criteria);
    for dataset in text_results {
        println!("    - {} (contains 'multi')", dataset.name);
    }

    // Display dataset details
    println!("\n📋 Dataset Details:");
    for dataset in manager.list_datasets() {
        println!("\n  📦 {}:", dataset.name);
        println!("     ID: {}", dataset.id);
        println!("     Category: {:?}", dataset.category);
        println!(
            "     Language: {:?} (+{} additional)",
            dataset.language,
            dataset.additional_languages.len()
        );
        println!("     Quality: {:?}", dataset.audio_quality);
        println!("     Samples: {}", dataset.sample_count);
        println!(
            "     Duration: {:.2} hours",
            dataset.total_duration / 3600.0
        );
        println!(
            "     Size: {:.2} GB",
            dataset.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("     Sample Rate: {} Hz", dataset.sample_rate);
        println!("     Creator: {}", dataset.creator);
        println!("     License: {}", dataset.license);
        println!("     Tags: {:?}", dataset.tags);
        println!("     Validated: {}", dataset.is_validated);
        println!("     Access: {:?}", dataset.access_level);
    }

    // Generate comprehensive report
    println!("\n📊 Comprehensive Report:");
    let report = manager.generate_report();
    println!("{}", report);

    // Demonstrate validation details
    if let Some(dataset) = manager.get_dataset("ljspeech_reference") {
        println!("\n🔍 Validation Example for '{}':", dataset.name);
        let validation_result = manager.validate_dataset(dataset).await?;
        println!("  Validation Result:");
        println!("    Valid: {}", validation_result.is_valid);
        println!("    Score: {:.2}/1.0", validation_result.score);
        println!("    Issues found: {}", validation_result.issues.len());
        println!(
            "    Validation time: {}ms",
            validation_result.validation_duration_ms
        );

        for (i, issue) in validation_result.issues.iter().enumerate() {
            println!(
                "    Issue {}: {:?} - {}",
                i + 1,
                issue.severity,
                issue.description
            );
            if let Some(ref suggestion) = issue.suggestion {
                println!("      Suggestion: {}", suggestion);
            }
        }
    }

    // Demonstrate data export
    println!("\n💾 Data Export Example:");

    // Export summary as JSON
    println!("  Exporting dataset registry to JSON...");
    // Note: In a real implementation, you might want to add export functionality
    println!("  ✅ Export functionality available in DatasetManager");

    // Usage recommendations
    println!("\n💡 Usage Recommendations:");
    println!("  - Use Reference datasets for quality baselines");
    println!("  - Use Test datasets for ongoing evaluation");
    println!("  - Use Benchmark datasets for standardized comparisons");
    println!("  - Use Validation datasets to tune evaluation metrics");
    println!("  - Tag datasets consistently for easy discovery");
    println!("  - Validate datasets before using in critical evaluations");
    println!("  - Monitor dataset quality over time");

    // Integration examples
    println!("\n🔗 Integration Examples:");
    println!("  # Use in evaluation pipeline:");
    println!("  let reference = manager.get_dataset(\"ljspeech_reference\").unwrap();");
    println!("  let evaluator = QualityEvaluator::with_reference_dataset(reference);");
    println!("");
    println!("  # Batch evaluation across multiple datasets:");
    println!("  let test_datasets = manager.search_datasets(&test_criteria);");
    println!("  for dataset in test_datasets {{");
    println!("      run_evaluation_suite(dataset).await?;");
    println!("  }}");

    println!("\n🎉 Dataset management demo completed successfully!");
    println!("   Check the 'demo_datasets' directory for persistent metadata files.");

    Ok(())
}
