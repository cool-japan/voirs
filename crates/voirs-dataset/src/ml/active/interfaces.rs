//! Annotation interfaces for active learning
//!
//! This module provides different annotation interfaces including
//! web-based, command-line, and API-based interfaces for human annotation.

use crate::{DatasetError, DatasetSample, QualityMetrics, Result};
use std::collections::HashMap;
use std::io::{self, Write};

use super::config::HumanLoopConfig;
use super::types::{AnnotationInterface, AnnotationResult, AudioIssue};

/// Web-based annotation interface
pub struct WebAnnotationInterface {
    config: HumanLoopConfig,
    is_running: bool,
}

impl WebAnnotationInterface {
    /// Create a new web annotation interface
    pub fn new(config: &HumanLoopConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            is_running: false,
        })
    }

    /// Get server port
    pub fn port(&self) -> u16 {
        match &self.config.interface_type {
            super::config::AnnotationInterfaceType::Web { port, .. } => *port,
            _ => 8080,
        }
    }

    /// Check if audio playback is enabled
    pub fn has_audio_playback(&self) -> bool {
        match &self.config.interface_type {
            super::config::AnnotationInterfaceType::Web {
                enable_audio_playback,
                ..
            } => *enable_audio_playback,
            _ => false,
        }
    }

    /// Check if spectrograms are shown
    pub fn shows_spectrograms(&self) -> bool {
        match &self.config.interface_type {
            super::config::AnnotationInterfaceType::Web {
                show_spectrograms, ..
            } => *show_spectrograms,
            _ => false,
        }
    }

    async fn start_server(&mut self) -> Result<()> {
        tracing::info!("Starting web annotation server on port {}", self.port());

        // In a real implementation, this would start an HTTP server
        // For now, we'll simulate it
        self.is_running = true;

        tracing::info!(
            "Web annotation interface available at http://localhost:{}",
            self.port()
        );
        Ok(())
    }

    async fn stop_server(&mut self) -> Result<()> {
        tracing::info!("Stopping web annotation server");
        self.is_running = false;
        Ok(())
    }

    async fn serve_sample(&self, sample: &DatasetSample) -> Result<AnnotationResult> {
        // Simulate web-based annotation
        tracing::info!("Serving sample {} for web annotation", sample.id);

        // In a real implementation, this would:
        // 1. Generate HTML page with audio player and form
        // 2. Wait for user submission
        // 3. Parse form data into AnnotationResult

        // For simulation, create a mock annotation
        let mut result = AnnotationResult::new(
            sample.id.to_string(),
            "web_annotator".to_string(),
            QualityMetrics {
                overall_quality: Some(0.8),
                snr: Some(25.0),
                ..Default::default()
            },
        );

        // Simulate some annotation decisions
        if sample.text.len() < 10 {
            result.add_audio_issue(AudioIssue::SpeechQuality);
        }

        if sample.audio.duration() < 0.5 {
            result.add_audio_issue(AudioIssue::LowVolume);
        }

        result.confidence = 0.9;

        Ok(result)
    }
}

#[async_trait::async_trait]
impl AnnotationInterface for WebAnnotationInterface {
    async fn start(&mut self) -> Result<()> {
        self.start_server().await
    }

    async fn stop(&mut self) -> Result<()> {
        self.stop_server().await
    }

    async fn present_sample(&mut self, sample: &DatasetSample) -> Result<AnnotationResult> {
        if !self.is_running {
            return Err(DatasetError::Configuration(
                "Web interface not started".to_string(),
            ));
        }

        self.serve_sample(sample).await
    }

    async fn show_feedback(&mut self, feedback: &str) -> Result<()> {
        tracing::info!("Web feedback: {}", feedback);
        // In a real implementation, this would update the web UI
        Ok(())
    }

    async fn get_statistics(&self) -> Result<HashMap<String, f32>> {
        let mut stats = HashMap::new();
        stats.insert("interface_type".to_string(), 1.0); // Web = 1
        stats.insert(
            "is_running".to_string(),
            if self.is_running { 1.0 } else { 0.0 },
        );
        stats.insert("port".to_string(), self.port() as f32);
        stats.insert(
            "audio_playback".to_string(),
            if self.has_audio_playback() { 1.0 } else { 0.0 },
        );
        Ok(stats)
    }
}

/// Command-line annotation interface
pub struct CLIAnnotationInterface {
    config: HumanLoopConfig,
    is_active: bool,
}

impl CLIAnnotationInterface {
    /// Create a new CLI annotation interface
    pub fn new(config: &HumanLoopConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            is_active: false,
        })
    }

    /// Check if colors are enabled
    pub fn uses_colors(&self) -> bool {
        match &self.config.interface_type {
            super::config::AnnotationInterfaceType::CLI { use_colors, .. } => *use_colors,
            _ => false,
        }
    }

    /// Check if progress bars are shown
    pub fn shows_progress(&self) -> bool {
        match &self.config.interface_type {
            super::config::AnnotationInterfaceType::CLI { show_progress, .. } => *show_progress,
            _ => false,
        }
    }

    /// Check if auto-play is enabled
    pub fn auto_plays_audio(&self) -> bool {
        match &self.config.interface_type {
            super::config::AnnotationInterfaceType::CLI {
                auto_play_audio, ..
            } => *auto_play_audio,
            _ => false,
        }
    }

    async fn interactive_annotation(&self, sample: &DatasetSample) -> Result<AnnotationResult> {
        println!("\n{}", "=".repeat(60));
        println!("🎵 Annotating Sample: {}", sample.id);
        println!("{}", "=".repeat(60));

        // Display sample information
        println!("📝 Text: {}", sample.text);
        println!("⏱️  Duration: {:.2}s", sample.audio.duration());
        println!("🔊 Sample Rate: {}Hz", sample.audio.sample_rate());

        if let Some(speaker) = sample.speaker_id() {
            println!("👤 Speaker: {speaker}");
        }

        // Simulate audio playback
        if self.auto_plays_audio() {
            println!("🔊 Playing audio... (simulated)");
            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
        }

        println!("\nPlease provide annotations:");

        // Get overall quality
        print!("Overall quality (0.0-1.0): ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let overall_quality: f32 = input.trim().parse().unwrap_or(0.5);

        // Get SNR estimate
        print!("SNR estimate (dB): ");
        io::stdout().flush().unwrap();
        input.clear();
        io::stdin().read_line(&mut input).unwrap();
        let snr: f32 = input.trim().parse().unwrap_or(20.0);

        // Check for issues
        println!("\nAny audio issues? (y/n): ");
        input.clear();
        io::stdin().read_line(&mut input).unwrap();
        let mut issues = vec![];

        if input.trim().to_lowercase() == "y" {
            println!("Select issues (comma-separated numbers):");
            println!("1. Noise");
            println!("2. Clipping");
            println!("3. Low Volume");
            println!("4. High Volume");
            println!("5. Distortion");
            println!("6. Echo");
            println!("7. Reverb");
            println!("8. Artifacts");
            println!("9. Speech Quality");

            input.clear();
            io::stdin().read_line(&mut input).unwrap();

            for choice in input.trim().split(',') {
                match choice.trim().parse::<u32>() {
                    Ok(1) => issues.push(AudioIssue::Noise),
                    Ok(2) => issues.push(AudioIssue::Clipping),
                    Ok(3) => issues.push(AudioIssue::LowVolume),
                    Ok(4) => issues.push(AudioIssue::HighVolume),
                    Ok(5) => issues.push(AudioIssue::Distortion),
                    Ok(6) => issues.push(AudioIssue::Echo),
                    Ok(7) => issues.push(AudioIssue::Reverb),
                    Ok(8) => issues.push(AudioIssue::Artifacts),
                    Ok(9) => issues.push(AudioIssue::SpeechQuality),
                    _ => {}
                }
            }
        }

        // Get text corrections
        print!("Any text corrections? (leave empty if none): ");
        io::stdout().flush().unwrap();
        input.clear();
        io::stdin().read_line(&mut input).unwrap();
        let text_corrections = if input.trim().is_empty() {
            None
        } else {
            Some(input.trim().to_string())
        };

        // Get confidence
        print!("Confidence in annotation (0.0-1.0): ");
        io::stdout().flush().unwrap();
        input.clear();
        io::stdin().read_line(&mut input).unwrap();
        let confidence: f32 = input.trim().parse().unwrap_or(0.8);

        // Create annotation result
        let mut result = AnnotationResult::new(
            sample.id.to_string(),
            "cli_annotator".to_string(),
            QualityMetrics {
                overall_quality: Some(overall_quality),
                snr: Some(snr),
                ..Default::default()
            },
        );

        for issue in issues {
            result.add_audio_issue(issue);
        }

        if let Some(correction) = text_corrections {
            result.set_text_correction(correction);
        }

        result.confidence = confidence;

        println!("✅ Annotation completed!");

        Ok(result)
    }

    async fn quick_annotation(&self, sample: &DatasetSample) -> Result<AnnotationResult> {
        // Simplified annotation for batch processing
        println!("Quick annotation for: {} ({})", sample.id, sample.text);

        // Use heuristics for quick annotation
        let duration = sample.audio.duration();
        let text_length = sample.text.len();

        let overall_quality = if duration > 0.5 && text_length > 10 {
            0.8
        } else {
            0.6
        };

        let snr = if duration > 1.0 { 25.0 } else { 20.0 };

        let mut result = AnnotationResult::new(
            sample.id.to_string(),
            "cli_quick_annotator".to_string(),
            QualityMetrics {
                overall_quality: Some(overall_quality),
                snr: Some(snr),
                ..Default::default()
            },
        );

        // Add automatic issue detection
        if duration < 0.3 {
            result.add_audio_issue(AudioIssue::LowVolume);
        }

        if text_length < 5 {
            result.add_audio_issue(AudioIssue::SpeechQuality);
        }

        result.confidence = 0.7; // Lower confidence for quick annotation

        Ok(result)
    }
}

#[async_trait::async_trait]
impl AnnotationInterface for CLIAnnotationInterface {
    async fn start(&mut self) -> Result<()> {
        println!("🚀 Starting CLI annotation interface");
        self.is_active = true;
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        println!("⏹️  Stopping CLI annotation interface");
        self.is_active = false;
        Ok(())
    }

    async fn present_sample(&mut self, sample: &DatasetSample) -> Result<AnnotationResult> {
        if !self.is_active {
            return Err(DatasetError::Configuration(
                "CLI interface not started".to_string(),
            ));
        }

        // Check if we should do interactive or quick annotation
        println!("Annotation mode: (i)nteractive or (q)uick? [q]: ");
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim().to_lowercase().as_str() {
            "i" | "interactive" => self.interactive_annotation(sample).await,
            _ => self.quick_annotation(sample).await,
        }
    }

    async fn show_feedback(&mut self, feedback: &str) -> Result<()> {
        if self.uses_colors() {
            println!("💬 \x1b[32mFeedback:\x1b[0m {feedback}");
        } else {
            println!("Feedback: {feedback}");
        }
        Ok(())
    }

    async fn get_statistics(&self) -> Result<HashMap<String, f32>> {
        let mut stats = HashMap::new();
        stats.insert("interface_type".to_string(), 2.0); // CLI = 2
        stats.insert(
            "is_active".to_string(),
            if self.is_active { 1.0 } else { 0.0 },
        );
        stats.insert(
            "uses_colors".to_string(),
            if self.uses_colors() { 1.0 } else { 0.0 },
        );
        stats.insert(
            "shows_progress".to_string(),
            if self.shows_progress() { 1.0 } else { 0.0 },
        );
        stats.insert(
            "auto_play".to_string(),
            if self.auto_plays_audio() { 1.0 } else { 0.0 },
        );
        Ok(stats)
    }
}

/// API-based annotation interface
pub struct APIAnnotationInterface {
    config: HumanLoopConfig,
    endpoint: String,
    auth_token: Option<String>,
    timeout: u64,
    is_connected: bool,
}

impl APIAnnotationInterface {
    /// Create a new API annotation interface
    pub fn new(config: &HumanLoopConfig) -> Result<Self> {
        let (endpoint, auth_token, timeout) = match &config.interface_type {
            super::config::AnnotationInterfaceType::API {
                endpoint,
                auth_token,
                timeout,
            } => (endpoint.clone(), auth_token.clone(), *timeout),
            _ => {
                return Err(DatasetError::Configuration(
                    "Invalid interface type for API annotation".to_string(),
                ));
            }
        };

        Ok(Self {
            config: config.clone(),
            endpoint,
            auth_token,
            timeout,
            is_connected: false,
        })
    }

    /// Get API endpoint
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Check if authentication is configured
    pub fn has_auth(&self) -> bool {
        self.auth_token.is_some()
    }

    /// Get timeout duration
    pub fn timeout_seconds(&self) -> u64 {
        self.timeout
    }

    async fn connect_to_api(&mut self) -> Result<()> {
        tracing::info!("Connecting to annotation API: {}", self.endpoint);

        // In a real implementation, this would:
        // 1. Test connection to the API endpoint
        // 2. Authenticate if needed
        // 3. Verify API compatibility

        // Simulate connection
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        self.is_connected = true;

        tracing::info!("Successfully connected to annotation API");
        Ok(())
    }

    async fn disconnect_from_api(&mut self) -> Result<()> {
        tracing::info!("Disconnecting from annotation API");
        self.is_connected = false;
        Ok(())
    }

    async fn submit_sample_to_api(&self, sample: &DatasetSample) -> Result<AnnotationResult> {
        if !self.is_connected {
            return Err(DatasetError::Configuration(
                "Not connected to API".to_string(),
            ));
        }

        tracing::info!("Submitting sample {} to API for annotation", sample.id);

        // In a real implementation, this would:
        // 1. Serialize sample data to JSON
        // 2. Make HTTP POST request to annotation endpoint
        // 3. Wait for annotator response or timeout
        // 4. Parse response into AnnotationResult

        // Simulate API call with timeout
        let api_timeout = tokio::time::Duration::from_secs(self.timeout);

        match tokio::time::timeout(api_timeout, self.simulate_api_annotation(sample)).await {
            Ok(result) => result,
            Err(_) => Err(DatasetError::Configuration(
                "API annotation timeout".to_string(),
            )),
        }
    }

    async fn simulate_api_annotation(&self, sample: &DatasetSample) -> Result<AnnotationResult> {
        // Simulate API processing time
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Create mock annotation result
        let mut result = AnnotationResult::new(
            sample.id.to_string(),
            "api_annotator".to_string(),
            QualityMetrics {
                overall_quality: Some(0.75),
                snr: Some(22.0),
                ..Default::default()
            },
        );

        // Simulate some basic quality checks
        if sample.audio.duration() < 0.5 {
            result.add_audio_issue(AudioIssue::LowVolume);
        }

        if sample.text.len() < 5 {
            result.add_audio_issue(AudioIssue::SpeechQuality);
        }

        // Add API-specific metadata
        result.set_notes(format!("Annotated via API: {}", self.endpoint));
        result.confidence = 0.85;

        Ok(result)
    }
}

#[async_trait::async_trait]
impl AnnotationInterface for APIAnnotationInterface {
    async fn start(&mut self) -> Result<()> {
        self.connect_to_api().await
    }

    async fn stop(&mut self) -> Result<()> {
        self.disconnect_from_api().await
    }

    async fn present_sample(&mut self, sample: &DatasetSample) -> Result<AnnotationResult> {
        self.submit_sample_to_api(sample).await
    }

    async fn show_feedback(&mut self, feedback: &str) -> Result<()> {
        tracing::info!("API feedback: {}", feedback);
        // In a real implementation, this might send feedback to the API
        Ok(())
    }

    async fn get_statistics(&self) -> Result<HashMap<String, f32>> {
        let mut stats = HashMap::new();
        stats.insert("interface_type".to_string(), 3.0); // API = 3
        stats.insert(
            "is_connected".to_string(),
            if self.is_connected { 1.0 } else { 0.0 },
        );
        stats.insert(
            "has_auth".to_string(),
            if self.has_auth() { 1.0 } else { 0.0 },
        );
        stats.insert("timeout".to_string(), self.timeout_seconds() as f32);
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::active::AnnotationInterfaceType;
    use crate::{AudioData, LanguageCode, SpeakerInfo};

    fn create_test_sample() -> DatasetSample {
        let samples = vec![0.0; 16000]; // 1 second at 16kHz
        let audio = AudioData::new(samples, 16000, 1);
        DatasetSample {
            id: "test_sample".to_string(),
            audio,
            text: "Test annotation text".to_string(),
            speaker: Some(SpeakerInfo {
                id: "speaker1".to_string(),
                name: Some("Test Speaker".to_string()),
                gender: None,
                age: None,
                accent: None,
                metadata: std::collections::HashMap::new(),
            }),
            language: LanguageCode::EnUs,
            quality: QualityMetrics::default(),
            phonemes: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_web_annotation_interface() {
        let config = HumanLoopConfig {
            interface_type: AnnotationInterfaceType::Web {
                port: 8080,
                enable_audio_playback: true,
                show_spectrograms: true,
                custom_css: None,
            },
            ..Default::default()
        };

        let mut interface = WebAnnotationInterface::new(&config).unwrap();

        assert!(!interface.is_running);
        assert_eq!(interface.port(), 8080);
        assert!(interface.has_audio_playback());
        assert!(interface.shows_spectrograms());

        interface.start().await.unwrap();
        assert!(interface.is_running);

        let sample = create_test_sample();
        let result = interface.present_sample(&sample).await.unwrap();
        assert_eq!(result.sample_id, "test_sample");
        assert_eq!(result.annotator_id, "web_annotator");

        interface.stop().await.unwrap();
        assert!(!interface.is_running);
    }

    #[tokio::test]
    async fn test_cli_annotation_interface() {
        let config = HumanLoopConfig {
            interface_type: AnnotationInterfaceType::CLI {
                use_colors: true,
                show_progress: true,
                auto_play_audio: false,
            },
            ..Default::default()
        };

        let mut interface = CLIAnnotationInterface::new(&config).unwrap();

        assert!(!interface.is_active);
        assert!(interface.uses_colors());
        assert!(interface.shows_progress());
        assert!(!interface.auto_plays_audio());

        interface.start().await.unwrap();
        assert!(interface.is_active);

        let stats = interface.get_statistics().await.unwrap();
        assert_eq!(stats.get("interface_type"), Some(&2.0));
        assert_eq!(stats.get("is_active"), Some(&1.0));

        interface.stop().await.unwrap();
        assert!(!interface.is_active);
    }

    #[tokio::test]
    async fn test_api_annotation_interface() {
        let config = HumanLoopConfig {
            interface_type: AnnotationInterfaceType::API {
                endpoint: "https://api.example.com/annotate".to_string(),
                auth_token: Some("test_token".to_string()),
                timeout: 30,
            },
            ..Default::default()
        };

        let mut interface = APIAnnotationInterface::new(&config).unwrap();

        assert!(!interface.is_connected);
        assert_eq!(interface.endpoint(), "https://api.example.com/annotate");
        assert!(interface.has_auth());
        assert_eq!(interface.timeout_seconds(), 30);

        interface.start().await.unwrap();
        assert!(interface.is_connected);

        let sample = create_test_sample();
        let result = interface.present_sample(&sample).await.unwrap();
        assert_eq!(result.sample_id, "test_sample");
        assert_eq!(result.annotator_id, "api_annotator");
        assert!(result.notes.is_some());

        interface.stop().await.unwrap();
        assert!(!interface.is_connected);
    }
}
