use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityExample {
    pub id: String,
    pub title: String,
    pub description: String,
    pub author: String,
    pub author_email: Option<String>,
    pub category: ExampleCategory,
    pub difficulty: DifficultyLevel,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub version: String,
    pub code_url: String,
    pub demo_url: Option<String>,
    pub documentation_url: Option<String>,
    pub dependencies: Vec<String>,
    pub license: String,
    pub upvotes: u32,
    pub downloads: u32,
    pub featured: bool,
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ExampleCategory {
    Synthesis,
    VoiceCloning,
    EmotionControl,
    SpatialAudio,
    RealTime,
    GameIntegration,
    WebIntegration,
    MobileIntegration,
    IoTIntegration,
    ProductionDeployment,
    PerformanceOptimization,
    CustomModels,
    AudioProcessing,
    CreativeApplications,
    Educational,
    Accessibility,
    Research,
    Experimental,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityGallery {
    pub examples: HashMap<String, CommunityExample>,
    pub categories: HashMap<ExampleCategory, Vec<String>>,
    pub featured_examples: Vec<String>,
    pub trending_examples: Vec<String>,
    pub recent_examples: Vec<String>,
    pub most_downloaded: Vec<String>,
    pub verified_authors: Vec<String>,
}

impl CommunityGallery {
    pub fn new() -> Self {
        Self {
            examples: HashMap::new(),
            categories: HashMap::new(),
            featured_examples: Vec::new(),
            trending_examples: Vec::new(),
            recent_examples: Vec::new(),
            most_downloaded: Vec::new(),
            verified_authors: Vec::new(),
        }
    }

    pub async fn load_from_directory(directory: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut gallery = Self::new();

        if !directory.exists() {
            return Ok(gallery);
        }

        let mut entries = fs::read_dir(directory).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().extension() == Some(std::ffi::OsStr::new("json")) {
                let content = fs::read_to_string(entry.path()).await?;
                let example: CommunityExample = serde_json::from_str(&content)?;
                gallery.add_example(example);
            }
        }

        gallery.build_indices();
        Ok(gallery)
    }

    pub fn add_example(&mut self, example: CommunityExample) {
        let id = example.id.clone();
        let category = example.category.clone();

        self.examples.insert(id.clone(), example);

        self.categories
            .entry(category)
            .or_insert_with(Vec::new)
            .push(id);
    }

    pub fn build_indices(&mut self) {
        let mut examples: Vec<_> = self.examples.values().collect();

        examples.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        self.recent_examples = examples.iter().take(10).map(|e| e.id.clone()).collect();

        examples.sort_by(|a, b| b.downloads.cmp(&a.downloads));
        self.most_downloaded = examples.iter().take(10).map(|e| e.id.clone()).collect();

        self.featured_examples = examples
            .iter()
            .filter(|e| e.featured)
            .map(|e| e.id.clone())
            .collect();
    }

    pub fn get_by_category(&self, category: &ExampleCategory) -> Vec<&CommunityExample> {
        if let Some(ids) = self.categories.get(category) {
            ids.iter().filter_map(|id| self.examples.get(id)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn search(&self, query: &str) -> Vec<&CommunityExample> {
        let query = query.to_lowercase();
        self.examples
            .values()
            .filter(|example| {
                example.title.to_lowercase().contains(&query)
                    || example.description.to_lowercase().contains(&query)
                    || example
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query))
            })
            .collect()
    }

    pub fn get_trending(&self) -> Vec<&CommunityExample> {
        self.trending_examples
            .iter()
            .filter_map(|id| self.examples.get(id))
            .collect()
    }

    pub fn get_featured(&self) -> Vec<&CommunityExample> {
        self.featured_examples
            .iter()
            .filter_map(|id| self.examples.get(id))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionSubmission {
    pub title: String,
    pub description: String,
    pub author: String,
    pub author_email: String,
    pub category: ExampleCategory,
    pub difficulty: DifficultyLevel,
    pub tags: Vec<String>,
    pub code_repository: String,
    pub demo_url: Option<String>,
    pub documentation: String,
    pub license: String,
    pub dependencies: Vec<String>,
    pub testing_instructions: String,
    pub example_code: String,
}

impl ContributionSubmission {
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.title.trim().is_empty() {
            errors.push("Title is required".to_string());
        }

        if self.description.len() < 50 {
            errors.push("Description must be at least 50 characters".to_string());
        }

        if self.author.trim().is_empty() {
            errors.push("Author name is required".to_string());
        }

        if !self.author_email.contains('@') {
            errors.push("Valid email address is required".to_string());
        }

        if self.code_repository.is_empty() {
            errors.push("Code repository URL is required".to_string());
        }

        if self.example_code.is_empty() {
            errors.push("Example code is required".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn to_community_example(&self) -> CommunityExample {
        CommunityExample {
            id: format!("contrib_{}", uuid::Uuid::new_v4().to_string()),
            title: self.title.clone(),
            description: self.description.clone(),
            author: self.author.clone(),
            author_email: Some(self.author_email.clone()),
            category: self.category.clone(),
            difficulty: self.difficulty.clone(),
            tags: self.tags.clone(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            version: "1.0.0".to_string(),
            code_url: self.code_repository.clone(),
            demo_url: self.demo_url.clone(),
            documentation_url: None,
            dependencies: self.dependencies.clone(),
            license: self.license.clone(),
            upvotes: 0,
            downloads: 0,
            featured: false,
            verified: false,
        }
    }
}

pub struct CommunityManager {
    gallery: CommunityGallery,
    moderation_queue: Vec<ContributionSubmission>,
}

impl CommunityManager {
    pub async fn new(gallery_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let gallery = CommunityGallery::load_from_directory(gallery_path).await?;

        Ok(Self {
            gallery,
            moderation_queue: Vec::new(),
        })
    }

    pub async fn submit_contribution(
        &mut self,
        submission: ContributionSubmission,
    ) -> Result<String, Vec<String>> {
        submission.validate()?;

        self.moderation_queue.push(submission.clone());

        let submission_id = format!("submission_{}", uuid::Uuid::new_v4().to_string());

        println!("✅ Contribution submitted successfully!");
        println!("   Submission ID: {}", submission_id);
        println!("   Title: {}", submission.title);
        println!("   Author: {}", submission.author);
        println!("   Status: Pending moderation");

        Ok(submission_id)
    }

    pub fn approve_contribution(&mut self, submission_index: usize) -> Result<String, String> {
        if submission_index >= self.moderation_queue.len() {
            return Err("Invalid submission index".to_string());
        }

        let submission = self.moderation_queue.remove(submission_index);
        let example = submission.to_community_example();
        let example_id = example.id.clone();

        self.gallery.add_example(example);
        self.gallery.build_indices();

        Ok(example_id)
    }

    pub fn get_pending_submissions(&self) -> &[ContributionSubmission] {
        &self.moderation_queue
    }

    pub fn get_gallery(&self) -> &CommunityGallery {
        &self.gallery
    }

    pub async fn export_gallery(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.gallery)?;
        fs::write(path, json).await?;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 VoiRS Community Contributions Gallery");
    println!("========================================");

    let gallery_path = Path::new("./community_gallery");

    let mut community = CommunityManager::new(gallery_path).await?;

    println!("\n📊 Gallery Statistics:");
    {
        let gallery = community.get_gallery();
        println!("   Total Examples: {}", gallery.examples.len());
        println!("   Categories: {}", gallery.categories.len());
        println!("   Featured Examples: {}", gallery.featured_examples.len());
        println!("   Recent Examples: {}", gallery.recent_examples.len());
    }

    let submission = ContributionSubmission {
        title: "Real-time Voice Chat with Emotion Analysis".to_string(),
        description: "A comprehensive example demonstrating real-time voice synthesis with emotion detection and adaptive response generation. Perfect for building interactive voice assistants or gaming applications with dynamic character voices.".to_string(),
        author: "John Developer".to_string(),
        author_email: "john@example.com".to_string(),
        category: ExampleCategory::RealTime,
        difficulty: DifficultyLevel::Advanced,
        tags: vec![
            "real-time".to_string(),
            "emotion".to_string(),
            "chat".to_string(),
            "interactive".to_string(),
        ],
        code_repository: "https://github.com/johndeveloper/voirs-realtime-chat".to_string(),
        demo_url: Some("https://voice-chat-demo.example.com".to_string()),
        documentation: "Complete setup and usage documentation with examples".to_string(),
        license: "MIT".to_string(),
        dependencies: vec!["tokio".to_string(), "serde".to_string()],
        testing_instructions: "Run `cargo test` and `cargo run --example chat_demo`".to_string(),
        example_code: r#"
use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    let synthesizer = VoirsSynthesizer::new()
        .with_voice("neural_voice_1")
        .with_emotion_analysis(true)
        .build().await?;
    
    let input = "Hello, how are you feeling today?";
    let audio = synthesizer.synthesize_with_emotion(input, EmotionState::Friendly).await?;
    
    Ok(())
}
"#.to_string(),
    };

    println!("\n📝 Submitting Community Contribution...");
    match community.submit_contribution(submission).await {
        Ok(submission_id) => {
            println!("✅ Submission successful: {}", submission_id);

            println!(
                "\n🔍 Pending Submissions: {}",
                community.get_pending_submissions().len()
            );

            if !community.get_pending_submissions().is_empty() {
                println!("\n✅ Approving first submission...");
                match community.approve_contribution(0) {
                    Ok(example_id) => {
                        println!("✅ Contribution approved: {}", example_id);

                        {
                            let updated_gallery = community.get_gallery();
                            println!(
                                "📊 Updated Gallery: {} examples",
                                updated_gallery.examples.len()
                            );

                            if let Some(example) = updated_gallery.examples.get(&example_id) {
                                println!("📖 New Example: {} by {}", example.title, example.author);
                            }
                        }
                    }
                    Err(e) => println!("❌ Approval failed: {}", e),
                }
            }
        }
        Err(errors) => {
            println!("❌ Submission failed:");
            for error in errors {
                println!("   • {}", error);
            }
        }
    }

    println!("\n🔍 Searching Examples...");
    {
        let gallery = community.get_gallery();
        let search_results = gallery.search("real-time");
        println!(
            "   Found {} examples matching 'real-time'",
            search_results.len()
        );

        for example in search_results {
            println!("   📖 {} ({})", example.title, example.author);
        }

        println!("\n⭐ Featured Examples:");
        let featured = gallery.get_featured();
        if featured.is_empty() {
            println!("   No featured examples yet");
        } else {
            for example in featured {
                println!("   🌟 {} by {}", example.title, example.author);
            }
        }
    }

    println!("\n💾 Exporting gallery...");
    let export_path = Path::new("/tmp/community_gallery_export.json");
    community.export_gallery(export_path).await?;
    println!("✅ Gallery exported to: {}", export_path.display());

    println!("\n🎉 Community Contributions Gallery Demo Complete!");
    println!("\nTo contribute your own example:");
    println!("1. Create your VoiRS example with comprehensive documentation");
    println!("2. Submit through the community portal or API");
    println!("3. Wait for moderation and approval");
    println!("4. Share your example with the VoiRS community!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contribution_validation() {
        let valid_submission = ContributionSubmission {
            title: "Test Example".to_string(),
            description: "This is a comprehensive test example with detailed description and usage instructions".to_string(),
            author: "Test Author".to_string(),
            author_email: "test@example.com".to_string(),
            category: ExampleCategory::Synthesis,
            difficulty: DifficultyLevel::Beginner,
            tags: vec!["test".to_string()],
            code_repository: "https://github.com/test/example".to_string(),
            demo_url: None,
            documentation: "Test documentation".to_string(),
            license: "MIT".to_string(),
            dependencies: vec![],
            testing_instructions: "Run tests".to_string(),
            example_code: "fn main() {}".to_string(),
        };

        assert!(valid_submission.validate().is_ok());
    }

    #[test]
    fn test_gallery_search() {
        let mut gallery = CommunityGallery::new();

        let example = CommunityExample {
            id: "test1".to_string(),
            title: "Real-time Synthesis".to_string(),
            description: "A real-time example".to_string(),
            author: "Test".to_string(),
            author_email: None,
            category: ExampleCategory::RealTime,
            difficulty: DifficultyLevel::Intermediate,
            tags: vec!["real-time".to_string(), "synthesis".to_string()],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            version: "1.0.0".to_string(),
            code_url: "https://github.com/test".to_string(),
            demo_url: None,
            documentation_url: None,
            dependencies: vec![],
            license: "MIT".to_string(),
            upvotes: 5,
            downloads: 100,
            featured: false,
            verified: true,
        };

        gallery.add_example(example);

        let results = gallery.search("real-time");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Real-time Synthesis");
    }

    #[test]
    fn test_category_filtering() {
        let mut gallery = CommunityGallery::new();

        let example = CommunityExample {
            id: "test1".to_string(),
            title: "Synthesis Example".to_string(),
            description: "A synthesis example".to_string(),
            author: "Test".to_string(),
            author_email: None,
            category: ExampleCategory::Synthesis,
            difficulty: DifficultyLevel::Beginner,
            tags: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            version: "1.0.0".to_string(),
            code_url: "https://github.com/test".to_string(),
            demo_url: None,
            documentation_url: None,
            dependencies: vec![],
            license: "MIT".to_string(),
            upvotes: 0,
            downloads: 0,
            featured: false,
            verified: false,
        };

        gallery.add_example(example);

        let synthesis_examples = gallery.get_by_category(&ExampleCategory::Synthesis);
        assert_eq!(synthesis_examples.len(), 1);

        let realtime_examples = gallery.get_by_category(&ExampleCategory::RealTime);
        assert_eq!(realtime_examples.len(), 0);
    }
}
