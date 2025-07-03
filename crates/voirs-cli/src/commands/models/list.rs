//! Model listing command implementation.

use std::collections::HashMap;
use voirs::config::AppConfig;
use voirs::error::Result;
use crate::model_types::{ModelInfo, ModelType};
use crate::GlobalOptions;
use hf_hub::{api::sync::Api, Repo, RepoType};
use reqwest::Client;
use serde_json::Value;

/// Run list models command
pub async fn run_list_models(
    backend: Option<&str>,
    detailed: bool,
    config: &AppConfig,
    global: &GlobalOptions,
) -> Result<()> {
    if !global.quiet {
        println!("Available TTS Models:");
        println!("====================");
    }
    
    // Get available models from various sources
    let acoustic_models = get_acoustic_models(backend).await?;
    let vocoder_models = get_vocoder_models(backend).await?;
    
    // Display acoustic models
    if !global.quiet {
        println!("\nAcoustic Models:");
        println!("----------------");
    }
    
    for model in acoustic_models {
        display_model_info(&model, detailed, global);
    }
    
    // Display vocoder models  
    if !global.quiet {
        println!("\nVocoder Models:");
        println!("---------------");
    }
    
    for model in vocoder_models {
        display_model_info(&model, detailed, global);
    }
    
    Ok(())
}

/// Get available acoustic models
async fn get_acoustic_models(backend: Option<&str>) -> Result<Vec<ModelInfo>> {
    let mut models = Vec::new();
    
    // Get local models first
    models.extend(get_local_models(ModelType::Acoustic).await?);
    
    // Add popular HuggingFace models for TTS
    let popular_models = get_popular_tts_models().await?;
    models.extend(popular_models.into_iter().filter(|m| m.model_type == ModelType::Acoustic));
    
    // Filter by backend if specified
    if let Some(backend_filter) = backend {
        models.retain(|m| m.supported_backends.iter().any(|b| b.contains(backend_filter)));
    }
    
    Ok(models)
}

/// Get available vocoder models
async fn get_vocoder_models(backend: Option<&str>) -> Result<Vec<ModelInfo>> {
    let mut models = Vec::new();
    
    // Get local models first
    models.extend(get_local_models(ModelType::Vocoder).await?);
    
    // Add popular HuggingFace models for vocoders
    let popular_models = get_popular_tts_models().await?;
    models.extend(popular_models.into_iter().filter(|m| m.model_type == ModelType::Vocoder));
    
    // Filter by backend if specified
    if let Some(backend_filter) = backend {
        models.retain(|m| m.supported_backends.iter().any(|b| b.contains(backend_filter)));
    }
    
    Ok(models)
}

/// Get locally installed models
async fn get_local_models(model_type: ModelType) -> Result<Vec<ModelInfo>> {
    let mut models = Vec::new();
    
    // Get models directory from config
    let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let models_dir = std::path::PathBuf::from(home_dir).join(".voirs").join("models");
    
    if !models_dir.exists() {
        return Ok(models);
    }
    
    // Scan for model directories
    for entry in std::fs::read_dir(&models_dir)? {
        let entry = entry?;
        let model_dir = entry.path();
        
        if model_dir.is_dir() {
            if let Ok(model_info) = load_local_model_info(&model_dir, model_type.clone()).await {
                models.push(model_info);
            }
        }
    }
    
    Ok(models)
}

/// Load model info from local installation
async fn load_local_model_info(
    model_dir: &std::path::Path,
    expected_type: ModelType,
) -> Result<ModelInfo> {
    let config_path = model_dir.join(".voirs-model.json");
    
    if config_path.exists() {
        // Load from VoiRS model config
        let config_content = std::fs::read_to_string(&config_path)?;
        let config: Value = serde_json::from_str(&config_content)?;
        
        Ok(ModelInfo {
            id: config["model_id"].as_str().unwrap_or("unknown").to_string(),
            name: config["name"].as_str().unwrap_or("Unknown Model").to_string(),
            model_type: expected_type,
            language: "unknown".to_string(),
            description: config["description"].as_str().unwrap_or("").to_string(),
            version: "local".to_string(),
            size_mb: config["total_size_mb"].as_f64().unwrap_or(0.0),
            sample_rate: 22050,
            quality_score: 4.0,
            supported_backends: vec!["pytorch".to_string()],
            is_installed: true,
            installation_path: Some(model_dir.to_string_lossy().to_string()),
            metadata: HashMap::new(),
        })
    } else {
        // Fallback: infer from directory structure
        let model_id = model_dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Calculate directory size
        let size_mb = calculate_directory_size(model_dir)? as f64 / (1024.0 * 1024.0);
        
        Ok(ModelInfo {
            id: model_id.clone(),
            name: model_id.clone(),
            model_type: expected_type,
            language: "unknown".to_string(),
            description: "Locally installed model".to_string(),
            version: "local".to_string(),
            size_mb,
            sample_rate: 22050,
            quality_score: 3.5,
            supported_backends: vec!["pytorch".to_string()],
            is_installed: true,
            installation_path: Some(model_dir.to_string_lossy().to_string()),
            metadata: HashMap::new(),
        })
    }
}

/// Calculate directory size in bytes
fn calculate_directory_size(dir: &std::path::Path) -> Result<u64> {
    let mut total_size = 0;
    
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                total_size += std::fs::metadata(&path)?.len();
            } else if path.is_dir() {
                total_size += calculate_directory_size(&path)?;
            }
        }
    }
    
    Ok(total_size)
}

/// Get popular TTS models from HuggingFace Hub
async fn get_popular_tts_models() -> Result<Vec<ModelInfo>> {
    // For now, return a curated list of popular TTS models
    // In a real implementation, this would query the HF Hub API
    Ok(vec![
        ModelInfo {
            id: "microsoft/speecht5_tts".to_string(),
            name: "SpeechT5 TTS".to_string(),
            model_type: ModelType::Acoustic,
            language: "en".to_string(),
            description: "Microsoft's SpeechT5 text-to-speech model".to_string(),
            version: "1.0.0".to_string(),
            size_mb: 238.5,
            sample_rate: 16000,
            quality_score: 4.3,
            supported_backends: vec!["pytorch".to_string(), "transformers".to_string()],
            is_installed: false,
            installation_path: None,
            metadata: HashMap::new(),
        },
        ModelInfo {
            id: "facebook/fastspeech2-en-ljspeech".to_string(),
            name: "FastSpeech2 English (LJSpeech)".to_string(),
            model_type: ModelType::Acoustic,
            language: "en".to_string(),
            description: "FastSpeech2 model trained on LJSpeech dataset".to_string(),
            version: "1.0.0".to_string(),
            size_mb: 43.2,
            sample_rate: 22050,
            quality_score: 4.1,
            supported_backends: vec!["pytorch".to_string(), "onnx".to_string()],
            is_installed: false,
            installation_path: None,
            metadata: HashMap::new(),
        },
        ModelInfo {
            id: "facebook/hifi-gan".to_string(),
            name: "HiFi-GAN Universal Vocoder".to_string(),
            model_type: ModelType::Vocoder,
            language: "multilingual".to_string(),
            description: "High-fidelity neural vocoder for speech synthesis".to_string(),
            version: "1.0.0".to_string(),
            size_mb: 55.8,
            sample_rate: 22050,
            quality_score: 4.5,
            supported_backends: vec!["pytorch".to_string(), "onnx".to_string()],
            is_installed: false,
            installation_path: None,
            metadata: HashMap::new(),
        },
        ModelInfo {
            id: "nvidia/waveglow".to_string(),
            name: "WaveGlow Vocoder".to_string(),
            model_type: ModelType::Vocoder,
            language: "multilingual".to_string(),
            description: "Flow-based generative model for speech synthesis".to_string(),
            version: "1.0.0".to_string(),
            size_mb: 128.7,
            sample_rate: 22050,
            quality_score: 4.4,
            supported_backends: vec!["pytorch".to_string()],
            is_installed: false,
            installation_path: None,
            metadata: HashMap::new(),
        },
    ])
}

/// Display model information
fn display_model_info(model: &ModelInfo, detailed: bool, global: &GlobalOptions) {
    if global.quiet {
        return;
    }
    
    let status = if model.is_installed { "✓" } else { "✗" };
    
    if detailed {
        println!("\n{} {} ({})", status, model.name, model.id);
        println!("  Language: {}", model.language);
        println!("  Version: {}", model.version);
        println!("  Size: {:.1} MB", model.size_mb);
        println!("  Quality: {:.1}/5.0", model.quality_score);
        println!("  Sample Rate: {} Hz", model.sample_rate);
        println!("  Backends: {}", model.supported_backends.join(", "));
        println!("  Description: {}", model.description);
        if let Some(path) = &model.installation_path {
            println!("  Installed at: {}", path);
        }
    } else {
        println!("{} {} ({}) - {:.1}MB - Quality: {:.1}/5.0", 
                 status, model.name, model.id, model.size_mb, model.quality_score);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voirs::config::AppConfig;
    use crate::GlobalOptions;
    
    #[tokio::test]
    async fn test_get_acoustic_models() {
        let models = get_acoustic_models(None).await.unwrap();
        assert!(!models.is_empty());
        assert!(models.iter().all(|m| m.model_type == ModelType::Acoustic));
    }
    
    #[tokio::test]
    async fn test_get_vocoder_models() {
        let models = get_vocoder_models(None).await.unwrap();
        assert!(!models.is_empty());
        assert!(models.iter().all(|m| m.model_type == ModelType::Vocoder));
    }
}