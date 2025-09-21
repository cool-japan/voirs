# Audio Analysis

VoiRS Recognizer provides comprehensive audio analysis capabilities for quality assessment, speaker characterization, prosody analysis, and voice activity detection. This guide covers all the analysis features and their practical applications.

## Overview

The audio analysis module offers:
- **Quality Metrics**: SNR, THD, dynamic range, frequency response
- **Speaker Analysis**: Gender, age, emotion, accent detection
- **Prosody Analysis**: Pitch, rhythm, stress patterns, intonation
- **Voice Activity Detection**: Speech segments, silence detection
- **Spectral Analysis**: MFCC, spectral features, formants

## Basic Audio Analysis

### Simple Quality Assessment

```rust
use voirs_recognizer::prelude::*;

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    // Load audio file
    let audio = load_audio("speech.wav", &AudioLoadConfig::default()).await?;
    
    // Create analyzer with default configuration
    let analyzer_config = AudioAnalysisConfig::default();
    let analyzer = AudioAnalyzerImpl::new(analyzer_config).await?;
    
    // Perform analysis
    let analysis = analyzer.analyze(&audio, Some(&analyzer_config)).await?;
    
    // Display quality metrics
    println!("Audio Quality Metrics:");
    for (metric, value) in &analysis.quality_metrics {
        println!("  {}: {:.3}", metric, value);
    }
    
    Ok(())
}
```

### Comprehensive Analysis

```rust
// Configure for complete analysis
let comprehensive_config = AudioAnalysisConfig {
    quality_metrics: true,
    prosody_analysis: true,
    speaker_analysis: true,
    spectral_analysis: true,
    vad_enabled: true,
    formant_analysis: true,
    ..Default::default()
};

let analyzer = AudioAnalyzerImpl::new(comprehensive_config).await?;
let analysis = analyzer.analyze(&audio, Some(&comprehensive_config)).await?;

// Access all analysis results
println!("Complete Audio Analysis:");
print_quality_metrics(&analysis.quality_metrics);
print_speaker_analysis(&analysis.speaker_characteristics);
print_prosody_analysis(&analysis.prosody);
print_spectral_analysis(&analysis.spectral_features);
print_vad_results(&analysis.vad_segments);
```

## Quality Metrics

### Signal Quality Assessment

```rust
// Extract detailed quality metrics
let quality_metrics = &analysis.quality_metrics;

// Signal-to-Noise Ratio
if let Some(snr) = quality_metrics.get("snr") {
    println!("SNR: {:.2} dB", snr);
    match snr {
        x if *x > 20.0 => println!("  Quality: Excellent"),
        x if *x > 15.0 => println!("  Quality: Good"),
        x if *x > 10.0 => println!("  Quality: Fair"),
        _ => println!("  Quality: Poor"),
    }
}

// Total Harmonic Distortion
if let Some(thd) = quality_metrics.get("thd") {
    println!("THD: {:.3}%", thd * 100.0);
    if *thd > 0.1 {
        println!("  Warning: High distortion detected");
    }
}

// Dynamic Range
if let Some(dynamic_range) = quality_metrics.get("dynamic_range") {
    println!("Dynamic Range: {:.1} dB", dynamic_range);
    if *dynamic_range < 20.0 {
        println!("  Warning: Low dynamic range (compressed audio)");
    }
}

// Clipping Detection
if let Some(clipping) = quality_metrics.get("clipping_percentage") {
    println!("Clipping: {:.2}%", clipping * 100.0);
    if *clipping > 0.01 {
        println!("  Warning: Audio clipping detected");
    }
}
```

### Frequency Response Analysis

```rust
// Analyze frequency characteristics
if let Some(spectral_features) = &analysis.spectral_features {
    println!("Spectral Analysis:");
    println!("  Spectral Centroid: {:.1} Hz", spectral_features.centroid);
    println!("  Spectral Bandwidth: {:.1} Hz", spectral_features.bandwidth);
    println!("  Spectral Rolloff: {:.1} Hz", spectral_features.rolloff);
    println!("  Zero Crossing Rate: {:.3}", spectral_features.zcr);
    
    // Frequency band analysis
    let bands = &spectral_features.frequency_bands;
    println!("  Frequency Bands:");
    println!("    Low (80-250 Hz): {:.2} dB", bands.low);
    println!("    Mid (250-4000 Hz): {:.2} dB", bands.mid);
    println!("    High (4000-8000 Hz): {:.2} dB", bands.high);
    
    // Check for frequency balance
    let balance = bands.mid - bands.low;
    if balance > 10.0 {
        println!("  Note: Mid-frequency emphasis detected");
    } else if balance < -10.0 {
        println!("  Note: Low-frequency emphasis detected");
    }
}
```

## Speaker Analysis

### Gender and Age Detection

```rust
if let Some(speaker_info) = &analysis.speaker_characteristics {
    println!("Speaker Characteristics:");
    
    // Gender classification
    println!("  Gender: {:?} (confidence: {:.2})", 
             speaker_info.gender, 
             speaker_info.gender_confidence);
    
    // Age estimation
    println!("  Age Range: {:?}", speaker_info.age_range);
    if let Some(age_confidence) = speaker_info.age_confidence {
        println!("  Age Confidence: {:.2}", age_confidence);
    }
    
    // Vocal characteristics
    if let Some(vocal_tract_length) = speaker_info.vocal_tract_length {
        println!("  Vocal Tract Length: {:.1} cm", vocal_tract_length);
    }
    
    if let Some(pitch_range) = &speaker_info.pitch_range {
        println!("  Pitch Range: {:.1} - {:.1} Hz", 
                 pitch_range.min, 
                 pitch_range.max);
    }
}
```

### Emotion Recognition

```rust
// Analyze emotional content
if let Some(emotion) = &analysis.speaker_characteristics.as_ref()?.emotion {
    println!("Emotion Analysis:");
    println!("  Primary Emotion: {:?}", emotion.primary);
    println!("  Confidence: {:.2}", emotion.confidence);
    
    // Emotional dimensions
    println!("  Valence: {:.2} ({})", 
             emotion.valence,
             if emotion.valence > 0.0 { "Positive" } else { "Negative" });
    println!("  Arousal: {:.2} ({})", 
             emotion.arousal,
             if emotion.arousal > 0.0 { "High Energy" } else { "Low Energy" });
    println!("  Dominance: {:.2} ({})", 
             emotion.dominance,
             if emotion.dominance > 0.0 { "Confident" } else { "Submissive" });
    
    // Emotional state interpretation
    match (emotion.valence > 0.0, emotion.arousal > 0.0) {
        (true, true) => println!("  Emotional State: Excited/Happy"),
        (true, false) => println!("  Emotional State: Calm/Content"),
        (false, true) => println!("  Emotional State: Angry/Stressed"),
        (false, false) => println!("  Emotional State: Sad/Depressed"),
    }
}
```

### Accent and Dialect Detection

```rust
// Detect accent and dialect
if let Some(accent_info) = &analysis.speaker_characteristics.as_ref()?.accent {
    println!("Accent Analysis:");
    println!("  Detected Accent: {:?}", accent_info.accent_type);
    println!("  Confidence: {:.2}", accent_info.confidence);
    println!("  Region: {:?}", accent_info.region);
    
    // Accent-specific features
    if let Some(features) = &accent_info.features {
        println!("  Accent Features:");
        for (feature, strength) in features {
            println!("    {}: {:.2}", feature, strength);
        }
    }
}
```

## Prosody Analysis

### Pitch Analysis

```rust
if let Some(prosody) = &analysis.prosody {
    println!("Prosody Analysis:");
    
    // Pitch characteristics
    let pitch = &prosody.pitch;
    println!("  Pitch Statistics:");
    println!("    Mean F0: {:.1} Hz", pitch.mean_f0);
    println!("    F0 Range: {:.1} - {:.1} Hz", pitch.min_f0, pitch.max_f0);
    println!("    F0 Std Dev: {:.1} Hz", pitch.std_f0);
    println!("    Pitch Variability: {:.2}", pitch.variability);
    
    // Pitch contour analysis
    if let Some(contour) = &pitch.contour {
        println!("  Pitch Contour:");
        println!("    Rising Trend: {:.2}%", contour.rising_percentage * 100.0);
        println!("    Falling Trend: {:.2}%", contour.falling_percentage * 100.0);
        println!("    Stable Regions: {:.2}%", contour.stable_percentage * 100.0);
    }
    
    // Identify pitch patterns
    if pitch.variability > 0.3 {
        println!("  Note: High pitch variability (expressive speech)");
    } else if pitch.variability < 0.1 {
        println!("  Note: Low pitch variability (monotone speech)");
    }
}
```

### Rhythm and Timing

```rust
// Analyze speech rhythm
if let Some(rhythm) = &prosody.rhythm {
    println!("  Rhythm Analysis:");
    println!("    Speaking Rate: {:.1} words/min", rhythm.speaking_rate);
    println!("    Syllable Rate: {:.1} syllables/sec", rhythm.syllable_rate);
    println!("    Pause Frequency: {:.1} pauses/min", rhythm.pause_frequency);
    println!("    Average Pause Duration: {:.2}s", rhythm.average_pause_duration);
    
    // Rhythm quality assessment
    match rhythm.speaking_rate {
        x if x > 180.0 => println!("    Speech Rate: Fast"),
        x if x > 150.0 => println!("    Speech Rate: Normal"),
        x if x > 120.0 => println!("    Speech Rate: Slow"),
        _ => println!("    Speech Rate: Very Slow"),
    }
    
    // Pause pattern analysis
    if rhythm.pause_frequency > 40.0 {
        println!("    Note: Frequent pauses detected");
    }
    if rhythm.average_pause_duration > 0.5 {
        println!("    Note: Long pauses detected");
    }
}
```

### Stress and Intonation

```rust
// Analyze stress patterns
if let Some(stress) = &prosody.stress {
    println!("  Stress Analysis:");
    println!("    Stress Pattern: {:?}", stress.pattern);
    println!("    Emphasis Strength: {:.2}", stress.emphasis_strength);
    println!("    Stress Regularity: {:.2}", stress.regularity);
    
    // Stress position analysis
    if let Some(positions) = &stress.positions {
        println!("    Stressed Positions:");
        for (time, strength) in positions {
            println!("      {:.2}s: {:.2}", time, strength);
        }
    }
}

// Analyze intonation patterns
if let Some(intonation) = &prosody.intonation {
    println!("  Intonation Analysis:");
    println!("    Contour Type: {:?}", intonation.contour_type);
    println!("    Final Boundary: {:?}", intonation.final_boundary);
    println!("    Question Likelihood: {:.2}", intonation.question_likelihood);
    println!("    Statement Likelihood: {:.2}", intonation.statement_likelihood);
    
    // Intonation interpretation
    if intonation.question_likelihood > 0.7 {
        println!("    Interpretation: Question intonation");
    } else if intonation.statement_likelihood > 0.7 {
        println!("    Interpretation: Statement intonation");
    } else {
        println!("    Interpretation: Neutral intonation");
    }
}
```

## Voice Activity Detection

### Speech Segment Analysis

```rust
// Analyze voice activity
if let Some(vad_segments) = &analysis.vad_segments {
    println!("Voice Activity Detection:");
    println!("  Total Segments: {}", vad_segments.len());
    
    let mut total_speech_time = 0.0;
    let mut total_silence_time = 0.0;
    
    for (i, segment) in vad_segments.iter().enumerate() {
        let duration = segment.end - segment.start;
        
        println!("  Segment {}: {:.2}s - {:.2}s ({:.2}s) - {:?}", 
                 i + 1,
                 segment.start,
                 segment.end,
                 duration,
                 segment.activity_type);
        
        match segment.activity_type {
            VoiceActivityType::Speech => total_speech_time += duration,
            VoiceActivityType::Silence => total_silence_time += duration,
            _ => {}
        }
    }
    
    let total_time = total_speech_time + total_silence_time;
    println!("  Speech Time: {:.2}s ({:.1}%)", 
             total_speech_time, 
             (total_speech_time / total_time) * 100.0);
    println!("  Silence Time: {:.2}s ({:.1}%)", 
             total_silence_time, 
             (total_silence_time / total_time) * 100.0);
    
    // Speech density analysis
    let speech_density = total_speech_time / total_time;
    match speech_density {
        x if x > 0.8 => println!("  Speech Density: High (continuous speech)"),
        x if x > 0.6 => println!("  Speech Density: Medium (normal conversation)"),
        x if x > 0.4 => println!("  Speech Density: Low (slow speech/many pauses)"),
        _ => println!("  Speech Density: Very Low (sparse speech)"),
    }
}
```

### Advanced VAD Features

```rust
// Configure advanced VAD
let advanced_vad_config = AudioAnalysisConfig {
    vad_enabled: true,
    vad_sensitivity: 0.5,
    vad_min_speech_duration: 0.1,
    vad_min_silence_duration: 0.2,
    vad_energy_threshold: 0.01,
    vad_frequency_analysis: true,
    ..Default::default()
};

let analyzer = AudioAnalyzerImpl::new(advanced_vad_config).await?;
let analysis = analyzer.analyze(&audio, Some(&advanced_vad_config)).await?;

// Analyze VAD quality
if let Some(vad_quality) = &analysis.vad_quality {
    println!("VAD Quality Assessment:");
    println!("  Accuracy: {:.2}", vad_quality.accuracy);
    println!("  Precision: {:.2}", vad_quality.precision);
    println!("  Recall: {:.2}", vad_quality.recall);
    println!("  F1 Score: {:.2}", vad_quality.f1_score);
}
```

## Spectral Analysis

### MFCC Features

```rust
// Extract MFCC features
if let Some(mfcc) = &analysis.spectral_features.as_ref()?.mfcc {
    println!("MFCC Analysis:");
    println!("  Coefficients: {} dimensions", mfcc.coefficients.len());
    
    // Display first few coefficients
    for (i, coeff) in mfcc.coefficients.iter().take(5).enumerate() {
        println!("  C{}: {:.3}", i, coeff);
    }
    
    // MFCC statistics
    if let Some(stats) = &mfcc.statistics {
        println!("  MFCC Statistics:");
        println!("    Mean: {:.3}", stats.mean);
        println!("    Std Dev: {:.3}", stats.std_dev);
        println!("    Range: {:.3} - {:.3}", stats.min, stats.max);
    }
}
```

### Formant Analysis

```rust
// Analyze formants
if let Some(formants) = &analysis.spectral_features.as_ref()?.formants {
    println!("Formant Analysis:");
    
    for (i, formant) in formants.iter().enumerate() {
        println!("  F{}: {:.1} Hz (bandwidth: {:.1} Hz)", 
                 i + 1,
                 formant.frequency,
                 formant.bandwidth);
    }
    
    // Vowel identification based on formants
    if formants.len() >= 2 {
        let f1 = formants[0].frequency;
        let f2 = formants[1].frequency;
        
        let vowel_estimate = estimate_vowel(f1, f2);
        println!("  Estimated Vowel: {}", vowel_estimate);
    }
}

fn estimate_vowel(f1: f32, f2: f32) -> &'static str {
    match (f1, f2) {
        (f1, f2) if f1 < 400.0 && f2 > 2000.0 => "i (as in 'see')",
        (f1, f2) if f1 < 500.0 && f2 < 1200.0 => "u (as in 'too')",
        (f1, f2) if f1 > 700.0 && f2 < 1500.0 => "a (as in 'father')",
        (f1, f2) if f1 < 600.0 && f2 > 1500.0 => "e (as in 'set')",
        (f1, f2) if f1 > 500.0 && f2 > 1500.0 => "o (as in 'caught')",
        _ => "Unknown vowel",
    }
}
```

## Performance Monitoring

### Real-time Analysis

```rust
// Monitor analysis performance
use std::time::Instant;

let start_time = Instant::now();
let analysis = analyzer.analyze(&audio, Some(&config)).await?;
let analysis_time = start_time.elapsed();

println!("Analysis Performance:");
println!("  Processing Time: {:?}", analysis_time);
println!("  Audio Duration: {:.2}s", audio.duration());
println!("  Real-time Factor: {:.3}", 
         analysis_time.as_secs_f32() / audio.duration());

// Memory usage
let memory_usage = get_memory_usage(); // Custom function
println!("  Memory Usage: {:.1} MB", memory_usage / 1024.0 / 1024.0);
```

### Batch Analysis

```rust
// Analyze multiple files efficiently
async fn batch_audio_analysis(
    files: Vec<&str>,
    config: &AudioAnalysisConfig
) -> Result<Vec<AudioAnalysisResult>, RecognitionError> {
    let analyzer = AudioAnalyzerImpl::new(config.clone()).await?;
    let mut results = Vec::new();
    
    for file in files {
        let audio = load_audio(file, &AudioLoadConfig::default()).await?;
        let analysis = analyzer.analyze(&audio, Some(config)).await?;
        results.push(analysis);
    }
    
    Ok(results)
}

// Generate batch report
fn generate_batch_report(results: &[AudioAnalysisResult]) {
    println!("Batch Analysis Report:");
    println!("  Files Analyzed: {}", results.len());
    
    // Average quality metrics
    let avg_snr = results.iter()
        .filter_map(|r| r.quality_metrics.get("snr"))
        .sum::<f32>() / results.len() as f32;
    
    println!("  Average SNR: {:.2} dB", avg_snr);
    
    // Quality distribution
    let high_quality = results.iter()
        .filter(|r| r.quality_metrics.get("snr").unwrap_or(&0.0) > &15.0)
        .count();
    
    println!("  High Quality Files: {} ({:.1}%)", 
             high_quality, 
             (high_quality as f32 / results.len() as f32) * 100.0);
}
```

## Custom Analysis

### Custom Metrics

```rust
// Implement custom analysis metrics
pub struct CustomAnalyzer {
    config: AudioAnalysisConfig,
}

impl CustomAnalyzer {
    pub fn new(config: AudioAnalysisConfig) -> Self {
        Self { config }
    }
    
    pub async fn analyze_custom(&self, audio: &AudioBuffer) -> Result<CustomAnalysisResult, RecognitionError> {
        // Custom analysis implementation
        let silence_ratio = self.calculate_silence_ratio(audio)?;
        let energy_distribution = self.calculate_energy_distribution(audio)?;
        let spectral_stability = self.calculate_spectral_stability(audio)?;
        
        Ok(CustomAnalysisResult {
            silence_ratio,
            energy_distribution,
            spectral_stability,
        })
    }
    
    fn calculate_silence_ratio(&self, audio: &AudioBuffer) -> Result<f32, RecognitionError> {
        let threshold = 0.01;
        let silence_samples = audio.samples()
            .iter()
            .filter(|&&sample| sample.abs() < threshold)
            .count();
        
        Ok(silence_samples as f32 / audio.len() as f32)
    }
    
    fn calculate_energy_distribution(&self, audio: &AudioBuffer) -> Result<Vec<f32>, RecognitionError> {
        // Divide audio into segments and calculate energy for each
        let segment_size = 1024;
        let mut energy_dist = Vec::new();
        
        for chunk in audio.samples().chunks(segment_size) {
            let energy: f32 = chunk.iter().map(|&s| s * s).sum();
            energy_dist.push(energy / chunk.len() as f32);
        }
        
        Ok(energy_dist)
    }
    
    fn calculate_spectral_stability(&self, audio: &AudioBuffer) -> Result<f32, RecognitionError> {
        // Analyze spectral stability over time
        // Implementation would involve FFT analysis
        Ok(0.85) // Placeholder
    }
}
```

## Best Practices

### 1. Quality Thresholds

```rust
// Set quality thresholds for different applications
pub struct QualityThresholds {
    pub min_snr: f32,
    pub max_thd: f32,
    pub min_dynamic_range: f32,
    pub max_clipping: f32,
}

impl QualityThresholds {
    pub fn for_broadcast() -> Self {
        Self {
            min_snr: 20.0,
            max_thd: 0.05,
            min_dynamic_range: 30.0,
            max_clipping: 0.001,
        }
    }
    
    pub fn for_voice_call() -> Self {
        Self {
            min_snr: 15.0,
            max_thd: 0.1,
            min_dynamic_range: 15.0,
            max_clipping: 0.01,
        }
    }
    
    pub fn for_speech_recognition() -> Self {
        Self {
            min_snr: 10.0,
            max_thd: 0.15,
            min_dynamic_range: 12.0,
            max_clipping: 0.05,
        }
    }
}

// Validate audio quality
fn validate_audio_quality(analysis: &AudioAnalysisResult, thresholds: &QualityThresholds) -> bool {
    let snr = analysis.quality_metrics.get("snr").unwrap_or(&0.0);
    let thd = analysis.quality_metrics.get("thd").unwrap_or(&1.0);
    let dynamic_range = analysis.quality_metrics.get("dynamic_range").unwrap_or(&0.0);
    let clipping = analysis.quality_metrics.get("clipping_percentage").unwrap_or(&1.0);
    
    *snr >= thresholds.min_snr &&
    *thd <= thresholds.max_thd &&
    *dynamic_range >= thresholds.min_dynamic_range &&
    *clipping <= thresholds.max_clipping
}
```

### 2. Analysis Optimization

```rust
// Optimize analysis for different use cases
pub enum AnalysisProfile {
    Quick,      // Fast analysis with essential metrics
    Standard,   // Balanced analysis
    Detailed,   // Comprehensive analysis
    Custom(AudioAnalysisConfig),
}

impl AnalysisProfile {
    pub fn to_config(&self) -> AudioAnalysisConfig {
        match self {
            AnalysisProfile::Quick => AudioAnalysisConfig {
                quality_metrics: true,
                prosody_analysis: false,
                speaker_analysis: false,
                spectral_analysis: false,
                vad_enabled: true,
                ..Default::default()
            },
            AnalysisProfile::Standard => AudioAnalysisConfig {
                quality_metrics: true,
                prosody_analysis: true,
                speaker_analysis: true,
                spectral_analysis: false,
                vad_enabled: true,
                ..Default::default()
            },
            AnalysisProfile::Detailed => AudioAnalysisConfig {
                quality_metrics: true,
                prosody_analysis: true,
                speaker_analysis: true,
                spectral_analysis: true,
                vad_enabled: true,
                formant_analysis: true,
                ..Default::default()
            },
            AnalysisProfile::Custom(config) => config.clone(),
        }
    }
}
```

## Next Steps

- Learn about [Phoneme Alignment](./phoneme-alignment.md) for linguistic analysis
- Explore [Real-time Streaming](./streaming.md) for live audio analysis
- Check out [Performance Optimization](./performance.md) for production deployment
- Review [Multi-language Support](./multi-language.md) for international applications