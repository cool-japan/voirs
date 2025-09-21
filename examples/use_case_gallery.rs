use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealWorldUseCase {
    pub id: String,
    pub title: String,
    pub description: String,
    pub industry: Industry,
    pub use_case_type: UseCaseType,
    pub company: Option<String>,
    pub company_size: CompanySize,
    pub region: String,
    pub deployment_scale: DeploymentScale,
    pub technical_requirements: TechnicalRequirements,
    pub business_metrics: BusinessMetrics,
    pub implementation_details: ImplementationDetails,
    pub lessons_learned: Vec<String>,
    pub roi_impact: String,
    pub challenges_faced: Vec<String>,
    pub solutions_implemented: Vec<String>,
    pub future_plans: String,
    pub contact_info: Option<ContactInfo>,
    pub case_study_url: Option<String>,
    pub demo_video_url: Option<String>,
    pub testimonial: Option<String>,
    pub created_at: DateTime<Utc>,
    pub featured: bool,
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Industry {
    Healthcare,
    Education,
    Entertainment,
    Gaming,
    Automotive,
    Finance,
    Retail,
    Manufacturing,
    Telecommunications,
    Government,
    Media,
    Accessibility,
    Research,
    SocialMedia,
    CustomerService,
    SmartHome,
    IoT,
    Broadcasting,
    Podcasting,
    AudioBooks,
    VirtualReality,
    AugmentedReality,
    CallCenters,
    LanguageLearning,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum UseCaseType {
    VoiceAssistant,
    ContentGeneration,
    VoiceCloning,
    RealTimeSynthesis,
    BatchProcessing,
    MultilingualTTS,
    EmotionalSynthesis,
    PersonalizedVoices,
    AccessibilityTool,
    InteractiveMedia,
    VoiceOver,
    GameCharacters,
    VirtualAnchor,
    AudioBookNarration,
    PodcastGeneration,
    CallCenterAutomation,
    LanguageLearning,
    VoiceCommerce,
    SmartHomeTTS,
    VehicleAssistant,
    MedicalAssistant,
    EducationalContent,
    BroadcastAutomation,
    SocialMediaContent,
    CustomerSupport,
    VoiceMail,
    AnnouncementSystem,
    NavigationAssistant,
    NewsReading,
    StorytellingAI,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CompanySize {
    Startup,       // 1-50 employees
    SmallBusiness, // 51-200 employees
    MidMarket,     // 201-1000 employees
    Enterprise,    // 1000+ employees
    Individual,    // Personal/hobby project
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeploymentScale {
    Development, // < 1K requests/month
    Small,       // 1K-10K requests/month
    Medium,      // 10K-100K requests/month
    Large,       // 100K-1M requests/month
    Enterprise,  // 1M+ requests/month
    Global,      // Multi-region deployment
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalRequirements {
    pub latency_requirements: LatencyRequirements,
    pub quality_requirements: QualityRequirements,
    pub platform_targets: Vec<PlatformTarget>,
    pub integration_points: Vec<String>,
    pub scalability_needs: ScalabilityNeeds,
    pub security_requirements: Vec<String>,
    pub compliance_requirements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRequirements {
    pub max_acceptable_latency_ms: u32,
    pub target_latency_ms: u32,
    pub real_time_required: bool,
    pub streaming_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    pub minimum_mos_score: f32,
    pub naturalness_priority: bool,
    pub pronunciation_accuracy: bool,
    pub emotion_support_required: bool,
    pub multilingual_support: bool,
    pub voice_consistency: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PlatformTarget {
    Web,
    iOS,
    Android,
    Desktop,
    Server,
    EdgeDevice,
    VR,
    AR,
    SmartSpeaker,
    Automotive,
    IoT,
    Embedded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityNeeds {
    pub concurrent_users: u32,
    pub requests_per_second: u32,
    pub geographical_distribution: bool,
    pub auto_scaling_required: bool,
    pub load_balancing_needed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessMetrics {
    pub implementation_timeline_months: u32,
    pub development_cost_usd: Option<u32>,
    pub monthly_operational_cost_usd: Option<u32>,
    pub user_satisfaction_score: Option<f32>,
    pub performance_improvement_percentage: Option<f32>,
    pub cost_savings_percentage: Option<f32>,
    pub user_engagement_increase: Option<f32>,
    pub revenue_impact_usd: Option<u32>,
    pub time_to_market_reduction_months: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationDetails {
    pub architecture_overview: String,
    pub key_components_used: Vec<String>,
    pub integration_patterns: Vec<String>,
    pub deployment_strategy: String,
    pub monitoring_approach: String,
    pub backup_and_recovery: String,
    pub security_measures: Vec<String>,
    pub testing_strategy: String,
    pub maintenance_approach: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactInfo {
    pub name: String,
    pub title: String,
    pub email: String,
    pub linkedin: Option<String>,
    pub company_website: Option<String>,
}

pub struct UseCaseGallery {
    pub use_cases: HashMap<String, RealWorldUseCase>,
    pub industry_index: HashMap<Industry, Vec<String>>,
    pub use_case_type_index: HashMap<UseCaseType, Vec<String>>,
    pub company_size_index: HashMap<CompanySize, Vec<String>>,
    pub deployment_scale_index: HashMap<DeploymentScale, Vec<String>>,
    pub featured_cases: Vec<String>,
    pub success_stories: Vec<String>,
}

impl UseCaseGallery {
    pub fn new() -> Self {
        Self {
            use_cases: HashMap::new(),
            industry_index: HashMap::new(),
            use_case_type_index: HashMap::new(),
            company_size_index: HashMap::new(),
            deployment_scale_index: HashMap::new(),
            featured_cases: Vec::new(),
            success_stories: Vec::new(),
        }
    }

    pub fn add_use_case(&mut self, use_case: RealWorldUseCase) {
        let id = use_case.id.clone();

        self.industry_index
            .entry(use_case.industry.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        self.use_case_type_index
            .entry(use_case.use_case_type.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        self.company_size_index
            .entry(use_case.company_size.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        self.deployment_scale_index
            .entry(use_case.deployment_scale.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        if use_case.featured {
            self.featured_cases.push(id.clone());
        }

        if use_case
            .business_metrics
            .user_satisfaction_score
            .unwrap_or(0.0)
            >= 4.0
            && use_case
                .business_metrics
                .performance_improvement_percentage
                .unwrap_or(0.0)
                >= 20.0
        {
            self.success_stories.push(id.clone());
        }

        self.use_cases.insert(id, use_case);
    }

    pub fn get_by_industry(&self, industry: &Industry) -> Vec<&RealWorldUseCase> {
        if let Some(ids) = self.industry_index.get(industry) {
            ids.iter().filter_map(|id| self.use_cases.get(id)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_by_use_case_type(&self, use_case_type: &UseCaseType) -> Vec<&RealWorldUseCase> {
        if let Some(ids) = self.use_case_type_index.get(use_case_type) {
            ids.iter().filter_map(|id| self.use_cases.get(id)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_by_company_size(&self, company_size: &CompanySize) -> Vec<&RealWorldUseCase> {
        if let Some(ids) = self.company_size_index.get(company_size) {
            ids.iter().filter_map(|id| self.use_cases.get(id)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_success_stories(&self) -> Vec<&RealWorldUseCase> {
        self.success_stories
            .iter()
            .filter_map(|id| self.use_cases.get(id))
            .collect()
    }

    pub fn get_featured_cases(&self) -> Vec<&RealWorldUseCase> {
        self.featured_cases
            .iter()
            .filter_map(|id| self.use_cases.get(id))
            .collect()
    }

    pub fn search(&self, query: &str) -> Vec<&RealWorldUseCase> {
        let query = query.to_lowercase();
        self.use_cases
            .values()
            .filter(|use_case| {
                use_case.title.to_lowercase().contains(&query)
                    || use_case.description.to_lowercase().contains(&query)
                    || use_case
                        .implementation_details
                        .architecture_overview
                        .to_lowercase()
                        .contains(&query)
                    || use_case
                        .company
                        .as_ref()
                        .map_or(false, |c| c.to_lowercase().contains(&query))
            })
            .collect()
    }

    pub fn get_statistics(&self) -> UseCaseStatistics {
        let total_cases = self.use_cases.len();

        let avg_satisfaction = self
            .use_cases
            .values()
            .filter_map(|uc| uc.business_metrics.user_satisfaction_score)
            .collect::<Vec<_>>()
            .iter()
            .sum::<f32>()
            / self.use_cases.len() as f32;

        let avg_roi = self
            .use_cases
            .values()
            .filter_map(|uc| uc.business_metrics.performance_improvement_percentage)
            .collect::<Vec<_>>()
            .iter()
            .sum::<f32>()
            / self.use_cases.len() as f32;

        UseCaseStatistics {
            total_cases,
            industries_covered: self.industry_index.len(),
            use_case_types_covered: self.use_case_type_index.len(),
            featured_cases: self.featured_cases.len(),
            success_stories: self.success_stories.len(),
            average_satisfaction_score: avg_satisfaction,
            average_roi_improvement: avg_roi,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UseCaseStatistics {
    pub total_cases: usize,
    pub industries_covered: usize,
    pub use_case_types_covered: usize,
    pub featured_cases: usize,
    pub success_stories: usize,
    pub average_satisfaction_score: f32,
    pub average_roi_improvement: f32,
}

fn create_sample_use_cases() -> Vec<RealWorldUseCase> {
    vec![
        RealWorldUseCase {
            id: "healthcare_patient_communication".to_string(),
            title: "AI-Powered Patient Communication System".to_string(),
            description: "Large hospital network implemented VoiRS for multilingual patient communication, medication reminders, and appointment notifications with personalized voices for better patient engagement.".to_string(),
            industry: Industry::Healthcare,
            use_case_type: UseCaseType::VoiceAssistant,
            company: Some("Metropolitan Health Network".to_string()),
            company_size: CompanySize::Enterprise,
            region: "North America".to_string(),
            deployment_scale: DeploymentScale::Large,
            technical_requirements: TechnicalRequirements {
                latency_requirements: LatencyRequirements {
                    max_acceptable_latency_ms: 2000,
                    target_latency_ms: 800,
                    real_time_required: false,
                    streaming_required: false,
                },
                quality_requirements: QualityRequirements {
                    minimum_mos_score: 4.2,
                    naturalness_priority: true,
                    pronunciation_accuracy: true,
                    emotion_support_required: true,
                    multilingual_support: true,
                    voice_consistency: true,
                },
                platform_targets: vec![PlatformTarget::Server, PlatformTarget::Web, PlatformTarget::iOS, PlatformTarget::Android],
                integration_points: vec!["Electronic Health Records".to_string(), "Appointment System".to_string(), "Pharmacy Management".to_string()],
                scalability_needs: ScalabilityNeeds {
                    concurrent_users: 5000,
                    requests_per_second: 200,
                    geographical_distribution: true,
                    auto_scaling_required: true,
                    load_balancing_needed: true,
                },
                security_requirements: vec!["HIPAA Compliance".to_string(), "End-to-end Encryption".to_string(), "Audit Logging".to_string()],
                compliance_requirements: vec!["HIPAA".to_string(), "SOC 2".to_string()],
            },
            business_metrics: BusinessMetrics {
                implementation_timeline_months: 8,
                development_cost_usd: Some(850000),
                monthly_operational_cost_usd: Some(45000),
                user_satisfaction_score: Some(4.6),
                performance_improvement_percentage: Some(35.0),
                cost_savings_percentage: Some(22.0),
                user_engagement_increase: Some(45.0),
                revenue_impact_usd: Some(2500000),
                time_to_market_reduction_months: Some(3),
            },
            implementation_details: ImplementationDetails {
                architecture_overview: "Microservices architecture with Kubernetes orchestration, Redis caching, and PostgreSQL for data persistence".to_string(),
                key_components_used: vec!["voirs-cloning".to_string(), "voirs-emotion".to_string(), "voirs-conversion".to_string()],
                integration_patterns: vec!["REST API".to_string(), "Webhook callbacks".to_string(), "Message queues".to_string()],
                deployment_strategy: "Blue-green deployment with rolling updates".to_string(),
                monitoring_approach: "Prometheus + Grafana with custom healthcare metrics".to_string(),
                backup_and_recovery: "Multi-region backup with 4-hour RTO".to_string(),
                security_measures: vec!["TLS 1.3".to_string(), "OAuth 2.0".to_string(), "Data encryption at rest".to_string()],
                testing_strategy: "Automated testing with patient privacy simulation".to_string(),
                maintenance_approach: "DevOps with 24/7 monitoring and on-call support".to_string(),
            },
            lessons_learned: vec![
                "Patient voice preferences vary significantly by demographic".to_string(),
                "Multilingual support requires cultural context, not just translation".to_string(),
                "HIPAA compliance affects caching strategies significantly".to_string(),
                "Voice quality consistency is critical for patient trust".to_string(),
            ],
            roi_impact: "35% improvement in patient communication effectiveness, 22% reduction in communication-related staff costs, 45% increase in appointment adherence".to_string(),
            challenges_faced: vec![
                "HIPAA compliance complexity".to_string(),
                "Multi-language medical terminology accuracy".to_string(),
                "Integration with legacy hospital systems".to_string(),
                "Staff training and adoption".to_string(),
            ],
            solutions_implemented: vec![
                "Custom medical dictionary for accurate pronunciation".to_string(),
                "Gradual rollout with pilot departments".to_string(),
                "Comprehensive staff training program".to_string(),
                "Legacy system API wrappers".to_string(),
            ],
            future_plans: "Expansion to telehealth platform and AI-powered symptom assessment with voice analysis".to_string(),
            contact_info: Some(ContactInfo {
                name: "Dr. Sarah Chen".to_string(),
                title: "Chief Technology Officer".to_string(),
                email: "s.chen@metrohealth.example.com".to_string(),
                linkedin: Some("https://linkedin.com/in/sarahchen-cto".to_string()),
                company_website: Some("https://metrohealth.example.com".to_string()),
            }),
            case_study_url: Some("https://metrohealth.example.com/case-studies/ai-voice-communication".to_string()),
            demo_video_url: Some("https://vimeo.com/example/healthcare-demo".to_string()),
            testimonial: Some("VoiRS transformed our patient communication. The multilingual support and emotional context awareness have significantly improved patient satisfaction scores.".to_string()),
            created_at: Utc::now(),
            featured: true,
            verified: true,
        },

        RealWorldUseCase {
            id: "gaming_character_voices".to_string(),
            title: "Dynamic NPC Voice Generation for AAA Game".to_string(),
            description: "Major game studio implemented VoiRS for generating dynamic NPC dialogue with emotion-aware character voices, reducing voice acting costs by 60% while maintaining AAA quality.".to_string(),
            industry: Industry::Gaming,
            use_case_type: UseCaseType::GameCharacters,
            company: Some("Stellar Games Studio".to_string()),
            company_size: CompanySize::MidMarket,
            region: "Europe".to_string(),
            deployment_scale: DeploymentScale::Medium,
            technical_requirements: TechnicalRequirements {
                latency_requirements: LatencyRequirements {
                    max_acceptable_latency_ms: 100,
                    target_latency_ms: 50,
                    real_time_required: true,
                    streaming_required: true,
                },
                quality_requirements: QualityRequirements {
                    minimum_mos_score: 4.4,
                    naturalness_priority: true,
                    pronunciation_accuracy: true,
                    emotion_support_required: true,
                    multilingual_support: true,
                    voice_consistency: true,
                },
                platform_targets: vec![PlatformTarget::Desktop, PlatformTarget::Server],
                integration_points: vec!["Unity Engine".to_string(), "Game State Manager".to_string(), "Audio Pipeline".to_string()],
                scalability_needs: ScalabilityNeeds {
                    concurrent_users: 10000,
                    requests_per_second: 500,
                    geographical_distribution: false,
                    auto_scaling_required: false,
                    load_balancing_needed: true,
                },
                security_requirements: vec!["DRM Protection".to_string(), "Anti-piracy measures".to_string()],
                compliance_requirements: vec!["ESRB Guidelines".to_string()],
            },
            business_metrics: BusinessMetrics {
                implementation_timeline_months: 6,
                development_cost_usd: Some(420000),
                monthly_operational_cost_usd: Some(18000),
                user_satisfaction_score: Some(4.7),
                performance_improvement_percentage: Some(40.0),
                cost_savings_percentage: Some(60.0),
                user_engagement_increase: Some(25.0),
                revenue_impact_usd: Some(1800000),
                time_to_market_reduction_months: Some(4),
            },
            implementation_details: ImplementationDetails {
                architecture_overview: "Real-time synthesis integrated directly into game engine with local GPU acceleration".to_string(),
                key_components_used: vec!["voirs-emotion".to_string(), "voirs-cloning".to_string(), "voirs-spatial".to_string()],
                integration_patterns: vec!["Native Unity plugin".to_string(), "Direct memory access".to_string(), "GPU compute shaders".to_string()],
                deployment_strategy: "Embedded in game client with cloud fallback".to_string(),
                monitoring_approach: "Game telemetry integration with custom voice metrics".to_string(),
                backup_and_recovery: "Local voice model caching with automatic updates".to_string(),
                security_measures: vec!["Model encryption".to_string(), "Runtime obfuscation".to_string()],
                testing_strategy: "Automated game scenario testing with voice validation".to_string(),
                maintenance_approach: "Game update pipeline with voice model updates".to_string(),
            },
            lessons_learned: vec![
                "Character voice consistency across dialogue sessions is crucial".to_string(),
                "Emotion transitions need careful tuning for believable characters".to_string(),
                "GPU memory management is critical for real-time performance".to_string(),
                "Player voice preference settings significantly impact engagement".to_string(),
            ],
            roi_impact: "60% reduction in voice acting costs, 40% faster content iteration, 25% increase in player engagement with story content".to_string(),
            challenges_faced: vec![
                "Real-time performance constraints".to_string(),
                "Character voice consistency across sessions".to_string(),
                "GPU memory limitations".to_string(),
                "Integration with existing audio pipeline".to_string(),
            ],
            solutions_implemented: vec![
                "GPU-optimized model quantization".to_string(),
                "Character voice state persistence".to_string(),
                "Streaming model loading system".to_string(),
                "Custom Unity audio mixer integration".to_string(),
            ],
            future_plans: "Player voice cloning for personalized NPCs and multiplayer voice morphing features".to_string(),
            contact_info: Some(ContactInfo {
                name: "Marcus Thompson".to_string(),
                title: "Lead Audio Engineer".to_string(),
                email: "m.thompson@stellargames.example.com".to_string(),
                linkedin: Some("https://linkedin.com/in/marcusthompson-audio".to_string()),
                company_website: Some("https://stellargames.example.com".to_string()),
            }),
            case_study_url: Some("https://stellargames.example.com/blog/ai-voice-revolution".to_string()),
            demo_video_url: Some("https://youtube.com/watch?v=gaming-demo-example".to_string()),
            testimonial: Some("VoiRS allowed us to create more dynamic and personalized character interactions than ever before. Our players love the responsive dialogue system.".to_string()),
            created_at: Utc::now(),
            featured: true,
            verified: true,
        },
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏭 VoiRS Real-World Use Case Gallery");
    println!("====================================");

    let mut gallery = UseCaseGallery::new();

    println!("\n📚 Loading sample use cases...");
    let sample_cases = create_sample_use_cases();
    for use_case in sample_cases {
        gallery.add_use_case(use_case);
    }

    let stats = gallery.get_statistics();
    println!("\n📊 Gallery Statistics:");
    println!("   Total Use Cases: {}", stats.total_cases);
    println!("   Industries Covered: {}", stats.industries_covered);
    println!("   Use Case Types: {}", stats.use_case_types_covered);
    println!("   Featured Cases: {}", stats.featured_cases);
    println!("   Success Stories: {}", stats.success_stories);
    println!(
        "   Average Satisfaction: {:.1}/5.0",
        stats.average_satisfaction_score
    );
    println!(
        "   Average ROI Improvement: {:.1}%",
        stats.average_roi_improvement
    );

    println!("\n🏥 Healthcare Use Cases:");
    let healthcare_cases = gallery.get_by_industry(&Industry::Healthcare);
    for case in healthcare_cases {
        println!(
            "   📋 {} ({})",
            case.title,
            case.company.as_ref().unwrap_or(&"Anonymous".to_string())
        );
        println!(
            "      💰 ROI: {:.1}% improvement",
            case.business_metrics
                .performance_improvement_percentage
                .unwrap_or(0.0)
        );
        println!(
            "      ⭐ Satisfaction: {:.1}/5.0",
            case.business_metrics.user_satisfaction_score.unwrap_or(0.0)
        );
        println!("      🎯 Scale: {:?}", case.deployment_scale);
    }

    println!("\n🎮 Gaming Use Cases:");
    let gaming_cases = gallery.get_by_industry(&Industry::Gaming);
    for case in gaming_cases {
        println!(
            "   🎮 {} ({})",
            case.title,
            case.company.as_ref().unwrap_or(&"Anonymous".to_string())
        );
        println!(
            "      💰 Cost Savings: {:.1}%",
            case.business_metrics.cost_savings_percentage.unwrap_or(0.0)
        );
        println!(
            "      🚀 Engagement Increase: {:.1}%",
            case.business_metrics
                .user_engagement_increase
                .unwrap_or(0.0)
        );
        println!(
            "      ⚡ Latency Requirement: {}ms",
            case.technical_requirements
                .latency_requirements
                .target_latency_ms
        );
    }

    println!("\n⭐ Success Stories:");
    let success_stories = gallery.get_success_stories();
    for story in success_stories {
        println!("   🌟 {}", story.title);
        println!(
            "      💬 \"{}\"",
            story
                .testimonial
                .as_ref()
                .unwrap_or(&"Great results with VoiRS implementation.".to_string())
        );
        println!("      📈 Impact: {}", story.roi_impact);
        if let Some(contact) = &story.contact_info {
            println!("      👤 Contact: {} ({})", contact.name, contact.title);
        }
    }

    println!("\n🔍 Search Example: 'real-time'");
    let search_results = gallery.search("real-time");
    for result in search_results {
        println!("   🔎 {} - {:?}", result.title, result.use_case_type);
        println!(
            "      ⚡ Latency: {}ms target",
            result
                .technical_requirements
                .latency_requirements
                .target_latency_ms
        );
    }

    println!("\n🏢 Enterprise Deployments:");
    let enterprise_cases = gallery.get_by_company_size(&CompanySize::Enterprise);
    for case in enterprise_cases {
        println!("   🏢 {} - {:?}", case.title, case.industry);
        println!(
            "      📊 Scale: {:?} ({} concurrent users)",
            case.deployment_scale,
            case.technical_requirements
                .scalability_needs
                .concurrent_users
        );
        println!(
            "      💵 Monthly Cost: ${}",
            case.business_metrics
                .monthly_operational_cost_usd
                .unwrap_or(0)
        );
    }

    println!("\n📋 Lessons Learned Summary:");
    let all_lessons: Vec<String> = gallery
        .use_cases
        .values()
        .flat_map(|case| case.lessons_learned.iter().cloned())
        .collect();

    let mut lesson_counts: HashMap<String, usize> = HashMap::new();
    for lesson in all_lessons {
        *lesson_counts.entry(lesson).or_insert(0) += 1;
    }

    let mut sorted_lessons: Vec<_> = lesson_counts.into_iter().collect();
    sorted_lessons.sort_by(|a, b| b.1.cmp(&a.1));

    for (lesson, count) in sorted_lessons.iter().take(5) {
        println!("   📝 {} (mentioned {} times)", lesson, count);
    }

    println!("\n💾 Exporting use case gallery...");
    let export_data = serde_json::to_string_pretty(&gallery.use_cases)?;
    let export_path = Path::new("/tmp/use_case_gallery_export.json");
    fs::write(export_path, export_data).await?;
    println!("✅ Use case gallery exported to: {}", export_path.display());

    println!("\n🎉 Use Case Gallery Demo Complete!");
    println!("\nKey Insights:");
    println!("• Healthcare and Gaming show highest ROI potential");
    println!("• Real-time requirements are critical for user engagement");
    println!("• Multi-language support is essential for global deployments");
    println!("• Voice consistency and quality directly impact user satisfaction");
    println!("• Integration complexity varies significantly by industry");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_use_case_gallery_creation() {
        let gallery = UseCaseGallery::new();
        assert_eq!(gallery.use_cases.len(), 0);
        assert_eq!(gallery.featured_cases.len(), 0);
    }

    #[test]
    fn test_use_case_filtering() {
        let mut gallery = UseCaseGallery::new();
        let sample_cases = create_sample_use_cases();

        for case in sample_cases {
            gallery.add_use_case(case);
        }

        let healthcare_cases = gallery.get_by_industry(&Industry::Healthcare);
        assert!(!healthcare_cases.is_empty());

        let gaming_cases = gallery.get_by_industry(&Industry::Gaming);
        assert!(!gaming_cases.is_empty());

        let enterprise_cases = gallery.get_by_company_size(&CompanySize::Enterprise);
        assert!(!enterprise_cases.is_empty());
    }

    #[test]
    fn test_statistics_calculation() {
        let mut gallery = UseCaseGallery::new();
        let sample_cases = create_sample_use_cases();

        for case in sample_cases {
            gallery.add_use_case(case);
        }

        let stats = gallery.get_statistics();
        assert!(stats.total_cases > 0);
        assert!(stats.average_satisfaction_score > 0.0);
        assert!(stats.average_roi_improvement > 0.0);
    }

    #[test]
    fn test_search_functionality() {
        let mut gallery = UseCaseGallery::new();
        let sample_cases = create_sample_use_cases();

        for case in sample_cases {
            gallery.add_use_case(case);
        }

        let search_results = gallery.search("healthcare");
        assert!(!search_results.is_empty());

        let gaming_results = gallery.search("gaming");
        assert!(!gaming_results.is_empty());

        let empty_results = gallery.search("nonexistent");
        assert!(empty_results.is_empty());
    }
}
