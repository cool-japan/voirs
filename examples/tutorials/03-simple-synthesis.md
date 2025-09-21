# Tutorial 3: Simple Voice Synthesis

**Duration**: 25-30 minutes  
**Level**: Beginner  
**Prerequisites**: Tutorials 1-2 completed

## Overview

Now that you understand VoiRS basics and configuration, it's time to explore different synthesis techniques. This tutorial covers various ways to generate speech and handle different types of content.

## What You'll Learn

- Text preprocessing and normalization
- Handling different content types (numbers, dates, abbreviations)
- Batch synthesis for multiple texts
- Audio manipulation and post-processing
- Performance monitoring and optimization

## Text Preprocessing

### Basic Text Normalization

```rust
use voirs_sdk::{VoirsSdk, TextProcessor, SynthesisConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    let config = SynthesisConfig::default();
    
    // Raw text that needs preprocessing
    let raw_text = "Dr. Smith said the temp. was 98.6°F at 3:30 PM on 12/25/2023.";
    
    // Preprocess text for better synthesis
    let processor = TextProcessor::new();
    let processed_text = processor
        .expand_abbreviations()
        .normalize_numbers()
        .normalize_dates()
        .normalize_times()
        .process(raw_text)?;
    
    println!("Original:  {}", raw_text);
    println!("Processed: {}", processed_text);
    // Output: "Doctor Smith said the temperature was ninety-eight point six degrees Fahrenheit at three thirty P M on December twenty-fifth, two thousand twenty-three."
    
    let audio = sdk.synthesize(&processed_text, &config).await?;
    std::fs::write("preprocessed_speech.wav", audio.data)?;
    
    Ok(())
}
```

### Custom Text Processing

```rust
use regex::Regex;
use voirs_sdk::TextProcessor;

struct CustomTextProcessor {
    email_regex: Regex,
    url_regex: Regex,
    phone_regex: Regex,
}

impl CustomTextProcessor {
    fn new() -> Self {
        Self {
            email_regex: Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap(),
            url_regex: Regex::new(r"https?://[^\s]+").unwrap(),
            phone_regex: Regex::new(r"\b\d{3}-\d{3}-\d{4}\b").unwrap(),
        }
    }
    
    fn process(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        // Replace emails
        result = self.email_regex.replace_all(&result, |caps: &regex::Captures| {
            let email = &caps[0];
            let parts: Vec<&str> = email.split('@').collect();
            format!("{} at {}", parts[0], parts[1].replace('.', " dot "))
        }).to_string();
        
        // Replace URLs
        result = self.url_regex.replace_all(&result, "web address").to_string();
        
        // Replace phone numbers
        result = self.phone_regex.replace_all(&result, |caps: &regex::Captures| {
            let phone = &caps[0];
            let parts: Vec<&str> = phone.split('-').collect();
            format!("{} {} {}", parts[0], parts[1], parts[2])
        }).to_string();
        
        result
    }
}

// Usage
let processor = CustomTextProcessor::new();
let text = "Contact us at support@example.com or visit https://example.com or call 555-123-4567";
let processed = processor.process(text);
println!("Processed: {}", processed);
// Output: "Contact us at support at example dot com or visit web address or call 555 123 4567"
```

## Handling Different Content Types

### Numbers and Mathematics

```rust
use voirs_sdk::{NumberProcessor, MathProcessor};

async fn synthesize_mathematical_content() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    let config = SynthesisConfig::default();
    
    let math_expressions = vec![
        "The equation is: x² + 2x - 8 = 0",
        "Calculate 15% of $1,250.75",
        "The ratio is 3:2 or 1.5:1",
        "Temperature: -15°C to 25°C",
        "Coordinates: (42.3601° N, 71.0589° W)"
    ];
    
    let number_processor = NumberProcessor::new()
        .with_currency_symbols()
        .with_mathematical_notation()
        .with_coordinates();
    
    for (i, expression) in math_expressions.iter().enumerate() {
        let processed = number_processor.process(expression)?;
        let audio = sdk.synthesize(&processed, &config).await?;
        
        let filename = format!("math_example_{}.wav", i + 1);
        std::fs::write(filename, audio.data)?;
        
        println!("✅ Generated: {} -> {}", expression, processed);
    }
    
    Ok(())
}
```

### Dates and Times

```rust
use voirs_sdk::DateTimeProcessor;
use chrono::{DateTime, Utc};

async fn synthesize_temporal_content() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    let config = SynthesisConfig::default();
    
    let datetime_processor = DateTimeProcessor::new()
        .with_relative_dates(true)
        .with_24hour_format(false)  // Use 12-hour format
        .with_timezone_names(true);
    
    let temporal_texts = vec![
        "Meeting scheduled for 2024-03-15 at 14:30 UTC",
        "The deadline is next Friday, 3/22/24",
        "Born on December 25th, 1990",
        "Event starts at 9:45 AM EST",
        "Duration: 2h 30m 45s"
    ];
    
    for text in temporal_texts {
        let processed = datetime_processor.process(text)?;
        let audio = sdk.synthesize(&processed, &config).await?;
        
        println!("Original:  {}", text);
        println!("Processed: {}", processed);
        println!("---");
    }
    
    Ok(())
}
```

## Batch Synthesis

### Sequential Processing

```rust
use voirs_sdk::{VoirsSdk, SynthesisConfig, BatchProcessor};
use std::collections::HashMap;

async fn batch_synthesis_sequential() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    let config = SynthesisConfig::default();
    
    let texts = vec![
        ("intro", "Welcome to our presentation"),
        ("section1", "First, let's discuss the background"),
        ("section2", "Next, we'll explore the methodology"),
        ("section3", "Finally, we'll review the results"),
        ("outro", "Thank you for your attention")
    ];
    
    let mut results = HashMap::new();
    
    for (id, text) in texts {
        println!("Synthesizing: {}", id);
        
        let start = std::time::Instant::now();
        let audio = sdk.synthesize(text, &config).await?;
        let duration = start.elapsed();
        
        let filename = format!("{}.wav", id);
        std::fs::write(&filename, &audio.data)?;
        
        results.insert(id, (filename, audio.duration_seconds(), duration));
        
        println!("  ✅ Generated {} ({:.2}s audio, {:.2}s processing)", 
                 id, audio.duration_seconds(), duration.as_secs_f64());
    }
    
    // Summary
    let total_audio_time: f64 = results.values().map(|(_, audio_dur, _)| audio_dur).sum();
    let total_processing_time: f64 = results.values().map(|(_, _, proc_dur)| proc_dur.as_secs_f64()).sum();
    
    println!("\n📊 Batch Summary:");
    println!("Total audio generated: {:.2}s", total_audio_time);
    println!("Total processing time: {:.2}s", total_processing_time);
    println!("Real-time factor: {:.2}x", total_audio_time / total_processing_time);
    
    Ok(())
}
```

### Parallel Processing

```rust
use tokio::task;
use futures::future::join_all;

async fn batch_synthesis_parallel() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    let config = SynthesisConfig::default();
    
    let texts = vec![
        "This is the first sentence to synthesize.",
        "Here's the second sentence for parallel processing.",
        "The third sentence demonstrates concurrent synthesis.",
        "Fourth sentence shows efficiency improvements.",
        "Finally, the fifth sentence completes our batch."
    ];
    
    println!("Starting parallel synthesis of {} texts...", texts.len());
    
    let start_time = std::time::Instant::now();
    
    // Create tasks for parallel execution
    let tasks: Vec<_> = texts.into_iter().enumerate().map(|(i, text)| {
        let sdk = sdk.clone();
        let config = config.clone();
        
        task::spawn(async move {
            let audio = sdk.synthesize(text, &config).await?;
            let filename = format!("parallel_{}.wav", i + 1);
            std::fs::write(&filename, &audio.data)?;
            Ok::<_, anyhow::Error>((i + 1, filename, audio.duration_seconds()))
        })
    }).collect();
    
    // Wait for all tasks to complete
    let results = join_all(tasks).await;
    
    let total_time = start_time.elapsed();
    
    // Process results
    for result in results {
        match result? {
            Ok((id, filename, duration)) => {
                println!("✅ Generated {} ({:.2}s)", filename, duration);
            }
            Err(e) => {
                eprintln!("❌ Error generating file {}: {}", id, e);
            }
        }
    }
    
    println!("\n⚡ Parallel processing completed in {:.2}s", total_time.as_secs_f64());
    
    Ok(())
}
```

## Audio Post-Processing

### Basic Audio Manipulation

```rust
use voirs_sdk::{AudioProcessor, AudioEffects, AudioData};

fn post_process_audio(audio: AudioData) -> anyhow::Result<AudioData> {
    let processor = AudioProcessor::new(audio.sample_rate);
    
    let processed = processor
        .normalize(0.8)                    // Normalize to 80% of max amplitude
        .apply_fade_in(0.1)               // 100ms fade-in
        .apply_fade_out(0.2)              // 200ms fade-out
        .remove_silence(0.02, 0.5)        // Remove silence > 500ms, threshold 2%
        .apply_high_pass_filter(80.0)     // Remove low-frequency noise
        .apply_low_pass_filter(8000.0)    // Remove high-frequency artifacts
        .process(audio)?;
    
    Ok(processed)
}

async fn synthesis_with_post_processing() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    let config = SynthesisConfig::default();
    
    let text = "This audio will be post-processed for better quality.";
    let raw_audio = sdk.synthesize(text, &config).await?;
    
    // Save raw audio
    std::fs::write("raw_audio.wav", &raw_audio.data)?;
    
    // Apply post-processing
    let processed_audio = post_process_audio(raw_audio)?;
    std::fs::write("processed_audio.wav", &processed_audio.data)?;
    
    println!("✅ Generated both raw and processed versions");
    
    Ok(())
}
```

### Audio Concatenation

```rust
use voirs_sdk::AudioConcatenator;

async fn create_audio_sequence() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    let config = SynthesisConfig::default();
    
    let segments = vec![
        ("intro", "Welcome to this audio sequence."),
        ("pause", ""), // Empty text creates a pause
        ("content", "Here is the main content of our presentation."),
        ("pause", ""),
        ("outro", "Thank you for listening.")
    ];
    
    let mut concatenator = AudioConcatenator::new();
    
    for (name, text) in segments {
        if text.is_empty() {
            // Add a 1-second pause
            concatenator.add_silence(1.0);
            println!("Added 1s pause");
        } else {
            let audio = sdk.synthesize(text, &config).await?;
            concatenator.add_audio(audio);
            println!("Added segment: {}", name);
        }
    }
    
    let final_audio = concatenator.build()?;
    std::fs::write("audio_sequence.wav", final_audio.data)?;
    
    println!("✅ Created audio sequence ({:.2}s total)", final_audio.duration_seconds());
    
    Ok(())
}
```

## Performance Monitoring

### Synthesis Metrics

```rust
use voirs_sdk::{SynthesisMetrics, PerformanceMonitor};
use std::time::Instant;

struct SynthesisTimer {
    start: Instant,
    character_count: usize,
}

impl SynthesisTimer {
    fn new(text: &str) -> Self {
        Self {
            start: Instant::now(),
            character_count: text.chars().count(),
        }
    }
    
    fn finish(self, audio_duration: f64) -> SynthesisMetrics {
        let processing_time = self.start.elapsed().as_secs_f64();
        
        SynthesisMetrics {
            text_length: self.character_count,
            audio_duration_seconds: audio_duration,
            processing_time_seconds: processing_time,
            real_time_factor: audio_duration / processing_time,
            characters_per_second: self.character_count as f64 / processing_time,
        }
    }
}

async fn monitored_synthesis() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    let config = SynthesisConfig::default();
    
    let texts = vec![
        "Short text for testing.",
        "This is a medium-length text that should take a moderate amount of time to synthesize and will help us understand the relationship between text length and processing time.",
        "This is a very long text that contains multiple sentences and should demonstrate how VoiRS handles longer inputs. The purpose is to measure performance characteristics across different text lengths and see how processing time scales with input size. We expect longer texts to take proportionally more time, but the relationship might not be perfectly linear due to various optimizations and batch processing techniques used internally by the synthesis engine."
    ];
    
    let mut all_metrics = Vec::new();
    
    for (i, text) in texts.iter().enumerate() {
        println!("Processing text {} ({} chars)...", i + 1, text.chars().count());
        
        let timer = SynthesisTimer::new(text);
        let audio = sdk.synthesize(text, &config).await?;
        let metrics = timer.finish(audio.duration_seconds());
        
        println!("  📊 Metrics:");
        println!("    Processing time: {:.3}s", metrics.processing_time_seconds);
        println!("    Audio duration: {:.3}s", metrics.audio_duration_seconds);
        println!("    Real-time factor: {:.2}x", metrics.real_time_factor);
        println!("    Characters/sec: {:.1}", metrics.characters_per_second);
        
        all_metrics.push(metrics);
        
        let filename = format!("monitored_synthesis_{}.wav", i + 1);
        std::fs::write(filename, audio.data)?;
    }
    
    // Calculate averages
    let avg_rtf: f64 = all_metrics.iter().map(|m| m.real_time_factor).sum::<f64>() / all_metrics.len() as f64;
    let avg_cps: f64 = all_metrics.iter().map(|m| m.characters_per_second).sum::<f64>() / all_metrics.len() as f64;
    
    println!("\n📈 Overall Performance:");
    println!("Average real-time factor: {:.2}x", avg_rtf);
    println!("Average characters per second: {:.1}", avg_cps);
    
    Ok(())
}
```

## Complete Example: News Article Synthesis

```rust
use voirs_sdk::{VoirsSdk, SynthesisConfig, TextProcessor, AudioConcatenator};
use serde_json::Value;

async fn synthesize_news_article() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    let config = SynthesisConfig::builder()
        .voice_id("news-reader")
        .sample_rate(22050)
        .quality(0.85)
        .speaking_rate(0.95)  // Slightly slower for news
        .build()?;
    
    // Simulated news article structure
    let article = NewsArticle {
        headline: "Breaking: New Voice Synthesis Technology Announced",
        byline: "By Jane Smith, Tech Reporter",
        date: "March 15, 2024",
        paragraphs: vec![
            "A breakthrough in voice synthesis technology was announced today by researchers at the VoiRS laboratory.",
            "The new system, called VoiRS 2.0, promises to deliver more natural-sounding speech with improved emotional expression.",
            "According to Dr. Sarah Johnson, lead researcher on the project, the technology uses advanced neural networks to better understand context and intonation.",
            "The system will be available for commercial use starting next quarter, with applications in accessibility, entertainment, and education."
        ]
    };
    
    let processor = TextProcessor::new()
        .with_abbreviation_expansion()
        .with_number_normalization();
    
    let mut concatenator = AudioConcatenator::new();
    
    // Synthesize headline with emphasis
    let headline_config = SynthesisConfig::builder()
        .from_config(&config)
        .emphasis_level(1.2)
        .build()?;
    
    let headline_text = processor.process(&article.headline)?;
    let headline_audio = sdk.synthesize(&headline_text, &headline_config).await?;
    concatenator.add_audio(headline_audio);
    concatenator.add_silence(1.0);
    
    // Synthesize byline and date
    let byline_text = format!("{}, {}", article.byline, article.date);
    let byline_processed = processor.process(&byline_text)?;
    let byline_audio = sdk.synthesize(&byline_processed, &config).await?;
    concatenator.add_audio(byline_audio);
    concatenator.add_silence(1.5);
    
    // Synthesize article body
    for (i, paragraph) in article.paragraphs.iter().enumerate() {
        let processed_paragraph = processor.process(paragraph)?;
        let paragraph_audio = sdk.synthesize(&processed_paragraph, &config).await?;
        concatenator.add_audio(paragraph_audio);
        
        // Add pause between paragraphs
        if i < article.paragraphs.len() - 1 {
            concatenator.add_silence(0.8);
        }
    }
    
    let final_audio = concatenator.build()?;
    std::fs::write("news_article.wav", final_audio.data)?;
    
    println!("✅ Generated news article audio ({:.2}s)", final_audio.duration_seconds());
    
    Ok(())
}

struct NewsArticle {
    headline: String,
    byline: String,
    date: String,
    paragraphs: Vec<String>,
}
```

## Best Practices

1. **Preprocess text**: Always normalize text for better synthesis quality
2. **Monitor performance**: Track processing times and real-time factors
3. **Use appropriate configurations**: Match config to content type
4. **Handle errors gracefully**: Implement robust error handling
5. **Consider batch processing**: Use parallel processing for multiple texts

## Common Issues

- **Poor pronunciation**: Use phonetic spelling or custom dictionary
- **Unnatural pauses**: Adjust punctuation and sentence structure
- **Inconsistent quality**: Ensure consistent preprocessing across texts
- **Performance bottlenecks**: Profile and optimize critical paths

## Next Steps

In the next tutorial, we'll explore voice cloning - how to create custom voices that sound like specific people.

Continue to [Tutorial 4: Voice Cloning Basics](./04-voice-cloning.md) →

## Additional Resources

- [Text Processing Examples](../text_processing_examples.rs)
- [Batch Processing Guide](../batch_synthesis.rs)
- [Performance Optimization](../performance_optimization_techniques.rs)

---

**Estimated completion time**: 25-30 minutes  
**Difficulty**: ⭐⭐☆☆☆  
**Next tutorial**: [Voice Cloning Basics](./04-voice-cloning.md)