use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FAQEntry {
    pub id: String,
    pub question: String,
    pub category: FAQCategory,
    pub difficulty_level: DifficultyLevel,
    pub short_answer: String,
    pub detailed_explanation: String,
    pub code_example: Option<String>,
    pub related_concepts: Vec<String>,
    pub common_variations: Vec<String>,
    pub troubleshooting_steps: Vec<String>,
    pub external_references: Vec<String>,
    pub example_files: Vec<String>,
    pub tags: Vec<String>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub view_count: u32,
    pub helpful_votes: u32,
    pub total_votes: u32,
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FAQCategory {
    GettingStarted,
    Installation,
    Configuration,
    BasicUsage,
    AdvancedFeatures,
    Performance,
    Troubleshooting,
    Integration,
    Deployment,
    Development,
    BestPractices,
    Security,
    Licensing,
    Community,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

pub struct FAQDatabase {
    pub entries: HashMap<String, FAQEntry>,
    pub category_index: HashMap<FAQCategory, Vec<String>>,
    pub tag_index: HashMap<String, Vec<String>>,
    pub popular_questions: Vec<String>,
    pub recent_questions: Vec<String>,
    pub trending_questions: Vec<String>,
}

impl FAQDatabase {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            category_index: HashMap::new(),
            tag_index: HashMap::new(),
            popular_questions: Vec::new(),
            recent_questions: Vec::new(),
            trending_questions: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: FAQEntry) {
        let id = entry.id.clone();

        self.category_index
            .entry(entry.category.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        for tag in &entry.tags {
            self.tag_index
                .entry(tag.clone())
                .or_insert_with(Vec::new)
                .push(id.clone());
        }

        self.entries.insert(id, entry);
        self.update_rankings();
    }

    fn update_rankings(&mut self) {
        let mut entries: Vec<_> = self.entries.iter().collect();

        // Sort by view count for popular questions
        entries.sort_by(|a, b| b.1.view_count.cmp(&a.1.view_count));
        self.popular_questions = entries
            .iter()
            .take(10)
            .map(|(id, _)| (*id).clone())
            .collect();

        // Sort by last updated for recent questions
        entries.sort_by(|a, b| b.1.last_updated.cmp(&a.1.last_updated));
        self.recent_questions = entries
            .iter()
            .take(10)
            .map(|(id, _)| (*id).clone())
            .collect();

        // Calculate trending score (views / days since update)
        entries.sort_by(|a, b| {
            let days_a = (chrono::Utc::now() - a.1.last_updated).num_days() + 1;
            let days_b = (chrono::Utc::now() - b.1.last_updated).num_days() + 1;
            let score_a = a.1.view_count as f64 / days_a as f64;
            let score_b = b.1.view_count as f64 / days_b as f64;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.trending_questions = entries
            .iter()
            .take(10)
            .map(|(id, _)| (*id).clone())
            .collect();
    }

    pub fn search(&self, query: &str) -> Vec<&FAQEntry> {
        let query = query.to_lowercase();
        self.entries
            .values()
            .filter(|entry| {
                entry.question.to_lowercase().contains(&query)
                    || entry.short_answer.to_lowercase().contains(&query)
                    || entry.detailed_explanation.to_lowercase().contains(&query)
                    || entry
                        .tags
                        .iter()
                        .any(|tag| tag.to_lowercase().contains(&query))
            })
            .collect()
    }

    pub fn get_by_category(&self, category: &FAQCategory) -> Vec<&FAQEntry> {
        if let Some(ids) = self.category_index.get(category) {
            ids.iter().filter_map(|id| self.entries.get(id)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_popular(&self) -> Vec<&FAQEntry> {
        self.popular_questions
            .iter()
            .filter_map(|id| self.entries.get(id))
            .collect()
    }

    pub fn get_trending(&self) -> Vec<&FAQEntry> {
        self.trending_questions
            .iter()
            .filter_map(|id| self.entries.get(id))
            .collect()
    }

    pub fn get_by_difficulty(&self, difficulty: &DifficultyLevel) -> Vec<&FAQEntry> {
        self.entries
            .values()
            .filter(|entry| {
                std::mem::discriminant(&entry.difficulty_level)
                    == std::mem::discriminant(difficulty)
            })
            .collect()
    }
}

fn create_essential_faqs() -> Vec<FAQEntry> {
    vec![
        FAQEntry {
            id: "getting_started_basic_synthesis".to_string(),
            question: "How do I get started with basic voice synthesis in VoiRS?".to_string(),
            category: FAQCategory::GettingStarted,
            difficulty_level: DifficultyLevel::Beginner,
            short_answer: "Install VoiRS, create a synthesizer instance, and call synthesize() with your text".to_string(),
            detailed_explanation: "Getting started with VoiRS is straightforward. First, add VoiRS to your Cargo.toml dependencies. Then create a synthesizer instance with your desired configuration and call the synthesize method with your input text. The synthesizer will return an audio buffer that you can play, save, or further process.".to_string(),
            code_example: Some(r#"
// Add to Cargo.toml:
// [dependencies]
// voirs = "0.1.0-alpha.2"

use voirs::prelude::*;

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    // Create a synthesizer with default settings
    let synthesizer = VoirsSynthesizer::new()
        .with_voice("default")
        .build()
        .await?;
    
    // Synthesize text to speech
    let text = "Hello, welcome to VoiRS!";
    let audio = synthesizer.synthesize(text).await?;
    
    // Save the audio to a file
    audio.save_to_file("output.wav").await?;
    
    println!("Speech synthesis complete!");
    Ok(())
}
"#.to_string()),
            related_concepts: vec![
                "VoirsSynthesizer configuration".to_string(),
                "Audio output formats".to_string(),
                "Voice selection".to_string(),
            ],
            common_variations: vec![
                "How to synthesize with different voices?".to_string(),
                "How to change audio quality settings?".to_string(),
                "How to synthesize to different audio formats?".to_string(),
            ],
            troubleshooting_steps: vec![
                "Ensure VoiRS is properly installed with `cargo check`".to_string(),
                "Verify your Rust version is compatible (1.70+)".to_string(),
                "Check that required system dependencies are installed".to_string(),
                "Try with a simpler text input first".to_string(),
            ],
            external_references: vec![
                "VoiRS Documentation: Getting Started Guide".to_string(),
                "Rust async programming tutorial".to_string(),
            ],
            example_files: vec!["hello_world.rs".to_string(), "basic_configuration.rs".to_string()],
            tags: vec!["beginner".to_string(), "getting-started".to_string(), "synthesis".to_string()],
            last_updated: chrono::Utc::now(),
            view_count: 1250,
            helpful_votes: 98,
            total_votes: 105,
            verified: true,
        },

        FAQEntry {
            id: "performance_slow_synthesis".to_string(),
            question: "Why is my voice synthesis slow and how can I improve performance?".to_string(),
            category: FAQCategory::Performance,
            difficulty_level: DifficultyLevel::Intermediate,
            short_answer: "Slow synthesis is usually caused by model loading, lack of caching, or insufficient resources. Enable GPU acceleration, implement caching, and optimize your configuration.".to_string(),
            detailed_explanation: "Voice synthesis performance can be affected by several factors: model loading time, processing complexity, hardware resources, and configuration settings. The most common issues are loading models for each request (solved by caching), using CPU-only processing (solved by GPU acceleration), and suboptimal quality settings (solved by configuration tuning). VoiRS provides several optimization strategies including model caching, GPU acceleration, streaming synthesis, and quality-performance trade-offs.".to_string(),
            code_example: Some(r#"
use voirs::prelude::*;
use std::sync::Arc;

// Performance-optimized synthesizer setup
#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    let synthesizer = VoirsSynthesizer::new()
        .with_voice("neural_fast")  // Use faster model variant
        .with_gpu_acceleration(true)  // Enable GPU if available
        .with_model_caching(true)     // Enable model caching
        .with_quality_preset(QualityPreset::Balanced)  // Balance speed/quality
        .with_streaming(true)         // Enable streaming for lower latency
        .build()
        .await?;
    
    // Cache the synthesizer instance for reuse
    let synthesizer = Arc::new(synthesizer);
    
    // Batch processing for better throughput
    let texts = vec![
        "First sentence to synthesize.",
        "Second sentence to synthesize.",
        "Third sentence to synthesize.",
    ];
    
    let mut tasks = Vec::new();
    for text in texts {
        let synth = synthesizer.clone();
        tasks.push(tokio::spawn(async move {
            synth.synthesize(&text).await
        }));
    }
    
    // Wait for all tasks to complete
    for task in tasks {
        match task.await? {
            Ok(audio) => println!("Synthesis completed successfully"),
            Err(e) => println!("Synthesis failed: {}", e),
        }
    }
    
    Ok(())
}

// Additional performance monitoring
async fn benchmark_synthesis() -> Result<(), VoirsError> {
    let synthesizer = VoirsSynthesizer::new().build().await?;
    let text = "Performance test sentence.";
    
    let start = std::time::Instant::now();
    let audio = synthesizer.synthesize(text).await?;
    let duration = start.elapsed();
    
    let rtf = duration.as_secs_f64() / audio.duration_seconds();
    println!("Real-time Factor: {:.2}x (lower is better)", rtf);
    
    Ok(())
}
"#.to_string()),
            related_concepts: vec![
                "GPU acceleration setup".to_string(),
                "Model caching strategies".to_string(),
                "Quality vs performance trade-offs".to_string(),
                "Real-time Factor (RTF) measurement".to_string(),
            ],
            common_variations: vec![
                "How to enable GPU acceleration?".to_string(),
                "What's the best quality setting for performance?".to_string(),
                "How to measure synthesis performance?".to_string(),
                "How to optimize for real-time applications?".to_string(),
            ],
            troubleshooting_steps: vec![
                "Check system resources (CPU, GPU, memory) during synthesis".to_string(),
                "Verify GPU drivers are installed and compatible".to_string(),
                "Monitor model loading times vs synthesis times".to_string(),
                "Test with different quality presets".to_string(),
                "Profile your application to identify bottlenecks".to_string(),
            ],
            external_references: vec![
                "VoiRS Performance Optimization Guide".to_string(),
                "GPU acceleration setup instructions".to_string(),
                "System requirements documentation".to_string(),
            ],
            example_files: vec![
                "performance_benchmarking.rs".to_string(),
                "gpu_acceleration_example.rs".to_string(),
                "streaming_synthesis.rs".to_string(),
            ],
            tags: vec!["performance".to_string(), "optimization".to_string(), "gpu".to_string(), "caching".to_string()],
            last_updated: chrono::Utc::now(),
            view_count: 892,
            helpful_votes: 76,
            total_votes: 82,
            verified: true,
        },

        FAQEntry {
            id: "voice_cloning_ethical_usage".to_string(),
            question: "How do I ethically use voice cloning features and ensure proper consent?".to_string(),
            category: FAQCategory::Security,
            difficulty_level: DifficultyLevel::Advanced,
            short_answer: "Always obtain explicit consent before cloning someone's voice, implement consent verification systems, use watermarking, and follow legal guidelines for voice cloning applications.".to_string(),
            detailed_explanation: "Voice cloning is a powerful feature that must be used responsibly. VoiRS includes built-in ethical safeguards including consent verification, voice watermarking, usage tracking, and misuse detection. Before cloning any voice, you must obtain explicit written consent from the voice owner. The system provides cryptographic consent verification, automatic watermarking of generated audio, and audit trails for all cloning activities. Additionally, VoiRS includes deepfake detection capabilities and can block known misuse patterns.".to_string(),
            code_example: Some(r#"
use voirs::prelude::*;
use voirs::cloning::{ConsentManager, VoiceWatermark, MisuseDetector};

#[tokio::main]
async fn main() -> Result<(), VoirsError> {
    // Initialize consent management system
    let consent_manager = ConsentManager::new()
        .with_encryption(true)
        .with_audit_logging(true)
        .build()?;
    
    // Verify consent before voice cloning
    let voice_owner_id = "user123";
    let consent_token = "consent_verification_token";
    
    if !consent_manager.verify_consent(voice_owner_id, consent_token).await? {
        return Err(VoirsError::ConsentRequired(
            "Valid consent required for voice cloning".to_string()
        ));
    }
    
    // Create voice cloning synthesizer with ethical safeguards
    let synthesizer = VoirsClonerSynthesizer::new()
        .with_consent_verification(true)
        .with_watermarking(true)
        .with_usage_tracking(true)
        .with_misuse_detection(true)
        .build()
        .await?;
    
    // Clone voice with safety checks
    let voice_sample = AudioBuffer::from_file("consent_voice_sample.wav").await?;
    let cloned_voice = synthesizer
        .clone_voice(voice_sample)
        .with_consent_verification(consent_token)
        .with_purpose("Educational demonstration")
        .with_usage_restrictions(UsageRestrictions::NonCommercial)
        .build()
        .await?;
    
    // Synthesize with cloned voice (includes automatic watermarking)
    let text = "This is a demonstration of ethical voice cloning.";
    let synthesized_audio = synthesizer
        .synthesize_with_cloned_voice(&cloned_voice, text)
        .await?;
    
    // Verify watermark is present
    let watermark = VoiceWatermark::extract(&synthesized_audio)?;
    println!("Audio watermarked: {}", watermark.is_present());
    println!("Voice owner: {}", watermark.voice_owner_id());
    println!("Generation timestamp: {}", watermark.timestamp());
    
    // Log usage for audit trail
    consent_manager.log_usage(LogEntry {
        voice_owner_id: voice_owner_id.to_string(),
        synthesized_text: text.to_string(),
        timestamp: chrono::Utc::now(),
        purpose: "Educational demonstration".to_string(),
        user_id: "demo_user".to_string(),
    }).await?;
    
    Ok(())
}

// Consent verification example
async fn request_voice_consent(
    voice_owner_email: &str,
    intended_use: &str
) -> Result<String, VoirsError> {
    let consent_request = ConsentRequest {
        voice_owner_email: voice_owner_email.to_string(),
        intended_use: intended_use.to_string(),
        requester_info: "Demo Application".to_string(),
        consent_duration: chrono::Duration::days(30),
        usage_restrictions: vec![
            "Non-commercial use only".to_string(),
            "Educational purposes only".to_string(),
        ],
    };
    
    // Send consent request (implementation depends on your notification system)
    let consent_manager = ConsentManager::new().build()?;
    let consent_token = consent_manager
        .request_consent(consent_request)
        .await?;
    
    println!("Consent request sent. Token: {}", consent_token);
    Ok(consent_token)
}
"#.to_string()),
            related_concepts: vec![
                "Consent management systems".to_string(),
                "Audio watermarking techniques".to_string(),
                "Deepfake detection".to_string(),
                "Legal compliance for voice cloning".to_string(),
            ],
            common_variations: vec![
                "What consent is required for voice cloning?".to_string(),
                "How to implement voice watermarking?".to_string(),
                "How to detect misuse of cloned voices?".to_string(),
                "What are the legal requirements for voice cloning?".to_string(),
            ],
            troubleshooting_steps: vec![
                "Ensure consent system is properly configured".to_string(),
                "Verify watermarking is enabled and working".to_string(),
                "Check audit logs for consent verification issues".to_string(),
                "Test misuse detection with known problematic inputs".to_string(),
                "Review legal requirements for your jurisdiction".to_string(),
            ],
            external_references: vec![
                "Voice cloning ethics guidelines".to_string(),
                "GDPR compliance for voice data".to_string(),
                "Deepfake detection research papers".to_string(),
                "Legal frameworks for synthetic media".to_string(),
            ],
            example_files: vec![
                "voice_cloning_example.rs".to_string(),
                "consent_management_example.rs".to_string(),
                "ethical_ai_practices.rs".to_string(),
            ],
            tags: vec!["ethics".to_string(), "consent".to_string(), "voice-cloning".to_string(), "security".to_string()],
            last_updated: chrono::Utc::now(),
            view_count: 654,
            helpful_votes: 89,
            total_votes: 95,
            verified: true,
        },

        FAQEntry {
            id: "integration_web_application".to_string(),
            question: "How do I integrate VoiRS into a web application with JavaScript?".to_string(),
            category: FAQCategory::Integration,
            difficulty_level: DifficultyLevel::Intermediate,
            short_answer: "Use VoiRS WebAssembly bindings with JavaScript, set up a REST API backend, or use the provided JavaScript SDK for seamless web integration.".to_string(),
            detailed_explanation: "VoiRS provides multiple integration options for web applications. The most common approach is using WebAssembly (WASM) bindings that allow you to run VoiRS directly in the browser. Alternatively, you can set up a backend API server and communicate via REST or WebSocket. For production applications, consider using the VoiRS JavaScript SDK which provides a high-level interface and handles browser compatibility automatically.".to_string(),
            code_example: Some(r#"
// Option 1: WebAssembly Integration
// First, build VoiRS for WASM target:
// cargo build --target wasm32-unknown-unknown --release

// JavaScript code:
import init, { VoirsSynthesizer } from './pkg/voirs_wasm.js';

async function initializeVoiRS() {
    // Initialize WASM module
    await init();
    
    // Create synthesizer instance
    const synthesizer = new VoirsSynthesizer({
        voice: 'default',
        quality: 'balanced',
        sampleRate: 44100
    });
    
    return synthesizer;
}

async function synthesizeText(text) {
    try {
        const synthesizer = await initializeVoiRS();
        
        // Synthesize text to audio
        const audioBuffer = await synthesizer.synthesize(text);
        
        // Convert to playable audio
        const audioContext = new AudioContext();
        const audioBufferSource = audioContext.createBufferSource();
        audioBufferSource.buffer = audioBuffer;
        audioBufferSource.connect(audioContext.destination);
        
        // Play the audio
        audioBufferSource.start();
        
        return audioBuffer;
    } catch (error) {
        console.error('Synthesis failed:', error);
        throw error;
    }
}

// Option 2: REST API Integration
class VoiRSClient {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
    }
    
    async synthesize(text, options = {}) {
        const response = await fetch(`${this.apiUrl}/synthesize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                voice: options.voice || 'default',
                quality: options.quality || 'balanced',
                format: options.format || 'wav'
            })
        });
        
        if (!response.ok) {
            throw new Error(`Synthesis failed: ${response.statusText}`);
        }
        
        // Return audio blob
        return await response.blob();
    }
    
    async playAudio(audioBlob) {
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.addEventListener('ended', () => {
            URL.revokeObjectURL(audioUrl);
        });
        
        await audio.play();
    }
}

// Usage example
async function webAppExample() {
    // Method 1: WASM (client-side)
    const audioBuffer = await synthesizeText("Hello from VoiRS!");
    
    // Method 2: REST API (server-side)
    const client = new VoiRSClient('http://localhost:8080/api/v1');
    const audioBlob = await client.synthesize("Hello from the server!");
    await client.playAudio(audioBlob);
}

// Option 3: Real-time streaming with WebSockets
class VoiRSStreamingClient {
    constructor(wsUrl) {
        this.wsUrl = wsUrl;
        this.websocket = null;
        this.audioContext = new AudioContext();
    }
    
    async connect() {
        return new Promise((resolve, reject) => {
            this.websocket = new WebSocket(this.wsUrl);
            
            this.websocket.onopen = () => {
                console.log('Connected to VoiRS streaming service');
                resolve();
            };
            
            this.websocket.onmessage = (event) => {
                this.handleAudioChunk(event.data);
            };
            
            this.websocket.onerror = (error) => {
                reject(error);
            };
        });
    }
    
    async synthesizeStreaming(text) {
        const message = {
            type: 'synthesize',
            text: text,
            voice: 'default',
            streaming: true
        };
        
        this.websocket.send(JSON.stringify(message));
    }
    
    handleAudioChunk(audioData) {
        // Play audio chunk immediately (streaming playback)
        const audioBuffer = this.decodeAudioChunk(audioData);
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        source.start();
    }
    
    decodeAudioChunk(audioData) {
        // Implementation depends on audio format
        // This is a simplified example
        return this.audioContext.decodeAudioData(audioData);
    }
}
"#.to_string()),
            related_concepts: vec![
                "WebAssembly compilation".to_string(),
                "REST API design".to_string(),
                "WebSocket streaming".to_string(),
                "Browser audio APIs".to_string(),
            ],
            common_variations: vec![
                "How to compile VoiRS to WebAssembly?".to_string(),
                "How to set up a VoiRS REST API server?".to_string(),
                "How to implement real-time streaming synthesis?".to_string(),
                "How to handle browser compatibility issues?".to_string(),
            ],
            troubleshooting_steps: vec![
                "Ensure WASM target is installed: rustup target add wasm32-unknown-unknown".to_string(),
                "Check browser console for WebAssembly loading errors".to_string(),
                "Verify CORS headers are properly configured for API requests".to_string(),
                "Test audio playback permissions in different browsers".to_string(),
                "Monitor network requests for API integration issues".to_string(),
            ],
            external_references: vec![
                "WebAssembly with Rust tutorial".to_string(),
                "Web Audio API documentation".to_string(),
                "CORS configuration guide".to_string(),
                "WebSocket integration patterns".to_string(),
            ],
            example_files: vec![
                "wasm_integration_example.rs".to_string(),
                "web_api_server.rs".to_string(),
                "streaming_web_client.js".to_string(),
            ],
            tags: vec!["web".to_string(), "javascript".to_string(), "wasm".to_string(), "integration".to_string()],
            last_updated: chrono::Utc::now(),
            view_count: 743,
            helpful_votes: 65,
            total_votes: 71,
            verified: true,
        },

        FAQEntry {
            id: "troubleshooting_installation_issues".to_string(),
            question: "I'm having trouble installing VoiRS. What are the common installation issues and solutions?".to_string(),
            category: FAQCategory::Troubleshooting,
            difficulty_level: DifficultyLevel::Beginner,
            short_answer: "Common issues include missing system dependencies, incompatible Rust version, or platform-specific compilation problems. Check system requirements and install missing dependencies.".to_string(),
            detailed_explanation: "VoiRS installation issues typically fall into several categories: missing system dependencies (audio libraries, development tools), incompatible Rust version, platform-specific compilation issues, or network connectivity problems. The most common solutions involve installing required system packages, updating Rust to the latest stable version, and ensuring proper toolchain configuration for your platform.".to_string(),
            code_example: Some(r#"
# System requirements check script
#!/bin/bash

echo "VoiRS Installation Diagnostics"
echo "=============================="

# Check Rust version
echo "Checking Rust version..."
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    echo "✅ Rust found: $RUST_VERSION"
    
    # Check if version is compatible (1.70+)
    RUST_MAJOR=$(rustc --version | sed 's/rustc \([0-9]*\)\.\([0-9]*\).*/\1/')
    RUST_MINOR=$(rustc --version | sed 's/rustc \([0-9]*\)\.\([0-9]*\).*/\2/')
    
    if [ "$RUST_MAJOR" -gt 1 ] || ([ "$RUST_MAJOR" -eq 1 ] && [ "$RUST_MINOR" -ge 70 ]); then
        echo "✅ Rust version is compatible"
    else
        echo "❌ Rust version too old. Please update to 1.70+"
        echo "   Run: rustup update stable"
    fi
else
    echo "❌ Rust not found. Please install from https://rustup.rs/"
fi

# Check Cargo
echo "Checking Cargo..."
if command -v cargo &> /dev/null; then
    echo "✅ Cargo found: $(cargo --version)"
else
    echo "❌ Cargo not found. Reinstall Rust with Cargo."
fi

# Platform-specific dependency checks
case "$(uname -s)" in
    Linux*)
        echo "Checking Linux dependencies..."
        
        # Check for essential build tools
        if command -v gcc &> /dev/null; then
            echo "✅ GCC found"
        else
            echo "❌ GCC not found. Install with: sudo apt install build-essential"
        fi
        
        # Check for audio libraries
        if pkg-config --exists alsa; then
            echo "✅ ALSA found"
        else
            echo "❌ ALSA not found. Install with: sudo apt install libasound2-dev"
        fi
        
        if pkg-config --exists libpulse; then
            echo "✅ PulseAudio found"
        else
            echo "⚠️  PulseAudio not found. Install with: sudo apt install libpulse-dev"
        fi
        ;;
        
    Darwin*)
        echo "Checking macOS dependencies..."
        
        if command -v xcode-select &> /dev/null; then
            echo "✅ Xcode command line tools found"
        else
            echo "❌ Xcode command line tools not found."
            echo "   Install with: xcode-select --install"
        fi
        
        # Check for Homebrew (optional but recommended)
        if command -v brew &> /dev/null; then
            echo "✅ Homebrew found"
        else
            echo "⚠️  Homebrew not found (optional)"
            echo "   Install from: https://brew.sh/"
        fi
        ;;
        
    MINGW*|CYGWIN*|MSYS*)
        echo "Checking Windows dependencies..."
        
        if command -v cl &> /dev/null; then
            echo "✅ MSVC compiler found"
        elif command -v gcc &> /dev/null; then
            echo "✅ MinGW GCC found"
        else
            echo "❌ C compiler not found."
            echo "   Install Visual Studio Build Tools or MinGW"
        fi
        ;;
esac

# Test basic Rust compilation
echo "Testing Rust compilation..."
cat > test_compile.rs << EOF
fn main() {
    println!("Hello, VoiRS!");
}
EOF

if rustc test_compile.rs -o test_compile; then
    echo "✅ Basic Rust compilation works"
    rm -f test_compile.rs test_compile
else
    echo "❌ Rust compilation failed"
fi

echo ""
echo "Installation command:"
echo "cargo add voirs --features full"
echo ""
echo "If issues persist, try:"
echo "1. Update Rust: rustup update stable"
echo "2. Clear Cargo cache: cargo clean"
echo "3. Check firewall/proxy settings"
echo "4. Install with verbose output: cargo install voirs -v"

# Cargo.toml example for manual setup
[dependencies]
voirs = { version = "0.1.0-alpha.2", features = ["full"] }
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }

# Platform-specific features
[target.'cfg(target_os = "linux")'.dependencies]
alsa = "0.7"

[target.'cfg(target_os = "macos")'.dependencies]
coreaudio-rs = "0.11"

[target.'cfg(target_os = "windows")'.dependencies]
windows = { version = "0.48", features = ["Win32_Media_Audio"] }
"#.to_string()),
            related_concepts: vec![
                "Rust toolchain management".to_string(),
                "System dependencies".to_string(),
                "Cross-platform compilation".to_string(),
                "Audio library requirements".to_string(),
            ],
            common_variations: vec![
                "VoiRS won't compile on my system".to_string(),
                "Missing audio library errors".to_string(),
                "Rust version compatibility issues".to_string(),
                "Platform-specific compilation errors".to_string(),
            ],
            troubleshooting_steps: vec![
                "Run the system requirements check script".to_string(),
                "Update Rust to the latest stable version".to_string(),
                "Install platform-specific audio development libraries".to_string(),
                "Clear Cargo cache and retry installation".to_string(),
                "Check for firewall or proxy issues blocking crates.io".to_string(),
                "Try installation with verbose output for detailed error messages".to_string(),
            ],
            external_references: vec![
                "Rust installation guide: https://rustup.rs/".to_string(),
                "Platform-specific dependency lists".to_string(),
                "Cargo troubleshooting guide".to_string(),
                "VoiRS system requirements documentation".to_string(),
            ],
            example_files: vec![
                "system_check.sh".to_string(),
                "minimal_example.rs".to_string(),
                "platform_specific_config.toml".to_string(),
            ],
            tags: vec!["installation".to_string(), "troubleshooting".to_string(), "dependencies".to_string(), "beginner".to_string()],
            last_updated: chrono::Utc::now(),
            view_count: 1087,
            helpful_votes: 124,
            total_votes: 132,
            verified: true,
        },
    ]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("❓ VoiRS FAQ Examples Database");
    println!("==============================");

    let mut faq_db = FAQDatabase::new();

    println!("\n📚 Loading essential FAQ entries...");
    let essential_faqs = create_essential_faqs();
    for faq in essential_faqs {
        faq_db.add_entry(faq);
    }

    println!("✅ Loaded {} FAQ entries", faq_db.entries.len());

    println!("\n🔥 Most Popular Questions:");
    let popular = faq_db.get_popular();
    for (i, faq) in popular.iter().take(3).enumerate() {
        println!("   {}. {} ({} views)", i + 1, faq.question, faq.view_count);
        println!(
            "      📂 Category: {:?} | 🎯 Level: {:?}",
            faq.category, faq.difficulty_level
        );
        println!("      💡 {}", faq.short_answer);
        println!("      👍 {}/{} helpful", faq.helpful_votes, faq.total_votes);
        println!();
    }

    println!("\n🚀 Getting Started Questions:");
    let getting_started = faq_db.get_by_category(&FAQCategory::GettingStarted);
    for faq in getting_started {
        println!("   ❓ {}", faq.question);
        println!("      💡 {}", faq.short_answer);
        if let Some(code) = &faq.code_example {
            let lines: Vec<&str> = code.lines().take(5).collect();
            println!("      💻 Code example preview:");
            for line in lines {
                if !line.trim().is_empty() {
                    println!("         {}", line);
                }
            }
            println!("         ... (see full example)");
        }
        println!();
    }

    println!("\n🔧 Troubleshooting Questions:");
    let troubleshooting = faq_db.get_by_category(&FAQCategory::Troubleshooting);
    for faq in troubleshooting {
        println!("   🐛 {}", faq.question);
        println!("      🔍 {}", faq.short_answer);

        if !faq.troubleshooting_steps.is_empty() {
            println!("      📋 Key troubleshooting steps:");
            for step in faq.troubleshooting_steps.iter().take(3) {
                println!("         • {}", step);
            }
        }
        println!();
    }

    println!("\n🔍 Search Example: 'performance'");
    let search_results = faq_db.search("performance");
    for result in search_results {
        println!("   🔎 {} ({:?})", result.question, result.category);
        println!("      💡 {}", result.short_answer);

        if !result.related_concepts.is_empty() {
            println!("      🔗 Related: {}", result.related_concepts.join(", "));
        }
        println!();
    }

    println!("\n📊 FAQ Statistics by Category:");
    let mut category_counts: HashMap<FAQCategory, usize> = HashMap::new();
    for faq in faq_db.entries.values() {
        *category_counts.entry(faq.category.clone()).or_insert(0) += 1;
    }

    for (category, count) in category_counts {
        println!("   📂 {:?}: {} questions", category, count);
    }

    println!("\n📊 FAQ Statistics by Difficulty:");
    let mut difficulty_counts: HashMap<DifficultyLevel, usize> = HashMap::new();
    for faq in faq_db.entries.values() {
        *difficulty_counts
            .entry(faq.difficulty_level.clone())
            .or_insert(0) += 1;
    }

    let difficulty_order = vec![
        DifficultyLevel::Beginner,
        DifficultyLevel::Intermediate,
        DifficultyLevel::Advanced,
        DifficultyLevel::Expert,
    ];

    for difficulty in difficulty_order {
        if let Some(count) = difficulty_counts.get(&difficulty) {
            println!("   🎯 {:?}: {} questions", difficulty, count);
        }
    }

    println!("\n⭐ Highest Rated FAQ:");
    let highest_rated = faq_db.entries.values().max_by(|a, b| {
        let rating_a = a.helpful_votes as f32 / a.total_votes.max(1) as f32;
        let rating_b = b.helpful_votes as f32 / b.total_votes.max(1) as f32;
        rating_a
            .partial_cmp(&rating_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(faq) = highest_rated {
        let rating = faq.helpful_votes as f32 / faq.total_votes.max(1) as f32;
        println!("   🌟 {} ({:.1}% helpful)", faq.question, rating * 100.0);
        println!("      📖 {}", faq.detailed_explanation);
    }

    println!("\n🏷️ Most Common Tags:");
    let mut tag_counts: HashMap<String, usize> = HashMap::new();
    for faq in faq_db.entries.values() {
        for tag in &faq.tags {
            *tag_counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    let mut sorted_tags: Vec<_> = tag_counts.into_iter().collect();
    sorted_tags.sort_by(|a, b| b.1.cmp(&a.1));

    for (tag, count) in sorted_tags.iter().take(5) {
        println!("   🏷️  {}: {} questions", tag, count);
    }

    println!("\n💾 Exporting FAQ database...");
    let export_data = serde_json::to_string_pretty(&faq_db.entries)?;
    let export_path = Path::new("/tmp/faq_database_export.json");
    fs::write(export_path, export_data).await?;
    println!("✅ FAQ database exported to: {}", export_path.display());

    println!("\n🎉 FAQ Examples Database Demo Complete!");
    println!("\nKey Insights:");
    println!("• Getting started questions are most viewed");
    println!("• Performance and troubleshooting are common concerns");
    println!("• Code examples significantly improve FAQ helpfulness");
    println!("• Ethical considerations are increasingly important");
    println!("• Platform integration requires comprehensive guidance");
    println!("\nFor complete answers and code examples, refer to the individual FAQ entries!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faq_database_creation() {
        let db = FAQDatabase::new();
        assert_eq!(db.entries.len(), 0);
        assert_eq!(db.popular_questions.len(), 0);
    }

    #[test]
    fn test_faq_categorization() {
        let mut db = FAQDatabase::new();
        let faqs = create_essential_faqs();

        for faq in faqs {
            db.add_entry(faq);
        }

        let getting_started = db.get_by_category(&FAQCategory::GettingStarted);
        assert!(!getting_started.is_empty());

        let troubleshooting = db.get_by_category(&FAQCategory::Troubleshooting);
        assert!(!troubleshooting.is_empty());

        let performance = db.get_by_category(&FAQCategory::Performance);
        assert!(!performance.is_empty());
    }

    #[test]
    fn test_search_functionality() {
        let mut db = FAQDatabase::new();
        let faqs = create_essential_faqs();

        for faq in faqs {
            db.add_entry(faq);
        }

        let performance_results = db.search("performance");
        assert!(!performance_results.is_empty());

        let synthesis_results = db.search("synthesis");
        assert!(!synthesis_results.is_empty());

        let empty_results = db.search("nonexistentterm");
        assert!(empty_results.is_empty());
    }

    #[test]
    fn test_ranking_system() {
        let mut db = FAQDatabase::new();
        let faqs = create_essential_faqs();

        for faq in faqs {
            db.add_entry(faq);
        }

        let popular = db.get_popular();
        assert!(!popular.is_empty());

        // Verify popular questions are sorted by view count
        if popular.len() > 1 {
            assert!(popular[0].view_count >= popular[1].view_count);
        }
    }

    #[test]
    fn test_difficulty_filtering() {
        let mut db = FAQDatabase::new();
        let faqs = create_essential_faqs();

        for faq in faqs {
            db.add_entry(faq);
        }

        let beginner_faqs = db.get_by_difficulty(&DifficultyLevel::Beginner);
        for faq in beginner_faqs {
            assert!(matches!(faq.difficulty_level, DifficultyLevel::Beginner));
        }

        let advanced_faqs = db.get_by_difficulty(&DifficultyLevel::Advanced);
        for faq in advanced_faqs {
            assert!(matches!(faq.difficulty_level, DifficultyLevel::Advanced));
        }
    }
}
