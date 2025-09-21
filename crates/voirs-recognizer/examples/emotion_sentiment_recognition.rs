//! Emotion and Sentiment Recognition Example
//!
//! This example demonstrates how to detect emotions and sentiment from speech
//! using VoiRS Recognizer's audio analysis and speech recognition capabilities.
//!
//! Usage:
//! ```bash
//! cargo run --example emotion_sentiment_recognition --features="whisper-pure"
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};
use voirs_recognizer::asr::{whisper::WhisperConfig, FallbackConfig, WhisperModelSize};
use voirs_recognizer::prelude::*;
use voirs_recognizer::{PerformanceValidator, RecognitionError};

// Emotion classification types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmotionType {
    Joy,
    Sadness,
    Anger,
    Fear,
    Disgust,
    Surprise,
    Neutral,
}

impl EmotionType {
    fn from_features(
        pitch_mean: f32,
        pitch_variance: f32,
        energy: f32,
        speaking_rate: f32,
    ) -> Self {
        // Simple rule-based emotion classification based on prosodic features
        match (pitch_mean, pitch_variance, energy, speaking_rate) {
            // Joy: Higher pitch, higher energy, faster speech
            (p, _, e, r) if p > 180.0 && e > 0.15 && r > 4.0 => EmotionType::Joy,

            // Sadness: Lower pitch, lower energy, slower speech
            (p, _, e, r) if p < 140.0 && e < 0.08 && r < 2.5 => EmotionType::Sadness,

            // Anger: Higher pitch variance, higher energy, faster speech
            (_, v, e, r) if v > 800.0 && e > 0.20 && r > 4.5 => EmotionType::Anger,

            // Fear: Higher pitch, higher variance, moderate energy
            (p, v, e, _) if p > 200.0 && v > 600.0 && e > 0.10 => EmotionType::Fear,

            // Disgust: Lower pitch, lower energy, slower speech
            (p, _, e, r) if p < 150.0 && e < 0.10 && r < 3.0 => EmotionType::Disgust,

            // Surprise: Higher pitch, higher variance, higher energy
            (p, v, e, _) if p > 190.0 && v > 700.0 && e > 0.18 => EmotionType::Surprise,

            // Default to neutral
            _ => EmotionType::Neutral,
        }
    }

    fn to_string(&self) -> &'static str {
        match self {
            EmotionType::Joy => "Joy",
            EmotionType::Sadness => "Sadness",
            EmotionType::Anger => "Anger",
            EmotionType::Fear => "Fear",
            EmotionType::Disgust => "Disgust",
            EmotionType::Surprise => "Surprise",
            EmotionType::Neutral => "Neutral",
        }
    }
}

// Sentiment classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SentimentType {
    Positive,
    Negative,
    Neutral,
}

impl SentimentType {
    fn from_emotion_and_text(emotion: EmotionType, text: &str) -> Self {
        // Combine emotion and text analysis for sentiment
        let emotion_sentiment = match emotion {
            EmotionType::Joy | EmotionType::Surprise => SentimentType::Positive,
            EmotionType::Sadness
            | EmotionType::Anger
            | EmotionType::Fear
            | EmotionType::Disgust => SentimentType::Negative,
            EmotionType::Neutral => SentimentType::Neutral,
        };

        // Simple text-based sentiment indicators
        let positive_words = [
            "good",
            "great",
            "excellent",
            "happy",
            "love",
            "wonderful",
            "amazing",
            "fantastic",
        ];
        let negative_words = [
            "bad",
            "terrible",
            "awful",
            "hate",
            "horrible",
            "disgusting",
            "angry",
            "sad",
        ];

        let text_lower = text.to_lowercase();
        let positive_count = positive_words
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count();
        let negative_count = negative_words
            .iter()
            .filter(|&&word| text_lower.contains(word))
            .count();

        let text_sentiment = if positive_count > negative_count {
            SentimentType::Positive
        } else if negative_count > positive_count {
            SentimentType::Negative
        } else {
            SentimentType::Neutral
        };

        // Combine emotion and text sentiment (weighted average)
        match (emotion_sentiment, text_sentiment) {
            (SentimentType::Positive, SentimentType::Positive) => SentimentType::Positive,
            (SentimentType::Negative, SentimentType::Negative) => SentimentType::Negative,
            (SentimentType::Positive, SentimentType::Neutral) => SentimentType::Positive,
            (SentimentType::Negative, SentimentType::Neutral) => SentimentType::Negative,
            (SentimentType::Neutral, s) => s,
            _ => SentimentType::Neutral,
        }
    }

    fn to_string(&self) -> &'static str {
        match self {
            SentimentType::Positive => "Positive",
            SentimentType::Negative => "Negative",
            SentimentType::Neutral => "Neutral",
        }
    }
}

// Combined emotion and sentiment analysis result
#[derive(Debug, Clone)]
pub struct EmotionSentimentResult {
    pub emotion: EmotionType,
    pub sentiment: SentimentType,
    pub confidence: f32,
    pub prosodic_features: HashMap<String, f32>,
    pub text_features: HashMap<String, f32>,
}

// Emotion and sentiment analyzer
pub struct EmotionSentimentAnalyzer {
    asr: IntelligentASRFallback,
    audio_analyzer: AudioAnalyzerImpl,
}

impl EmotionSentimentAnalyzer {
    pub async fn new() -> Result<Self, RecognitionError> {
        // Configure ASR for emotion analysis
        let fallback_config = FallbackConfig {
            quality_threshold: 0.6,
            max_processing_time_seconds: 30.0,
            max_retries: 3,
            ..Default::default()
        };

        let asr = IntelligentASRFallback::new(fallback_config).await?;

        // Configure audio analyzer for prosodic features
        let analyzer_config = AudioAnalysisConfig {
            emotional_analysis: true,
            frame_size: 1024,
            hop_size: 512,
            ..Default::default()
        };

        let audio_analyzer = AudioAnalyzerImpl::new(analyzer_config).await?;

        Ok(Self {
            asr,
            audio_analyzer,
        })
    }

    pub async fn analyze(
        &mut self,
        audio: &AudioBuffer,
    ) -> Result<EmotionSentimentResult, RecognitionError> {
        // Step 1: Perform speech recognition
        let recognition_result = self.asr.transcribe(audio, None).await?;

        // Step 2: Analyze audio for prosodic features
        let audio_analysis = self
            .audio_analyzer
            .analyze(audio, Some(&AudioAnalysisConfig::default()))
            .await?;

        // Step 3: Extract prosodic features
        let mut prosodic_features = HashMap::new();

        let pitch_mean = audio_analysis.prosody.pitch.mean_f0;
        let pitch_variance = audio_analysis.prosody.pitch.f0_std;
        let energy = audio_analysis
            .quality_metrics
            .get("energy")
            .copied()
            .unwrap_or(0.0);
        let speaking_rate = audio_analysis.prosody.rhythm.speaking_rate;

        prosodic_features.insert("pitch_mean".to_string(), pitch_mean);
        prosodic_features.insert("pitch_variance".to_string(), pitch_variance);
        prosodic_features.insert("energy".to_string(), energy);
        prosodic_features.insert("speaking_rate".to_string(), speaking_rate);

        // Add additional prosodic features
        prosodic_features.insert(
            "pitch_range".to_string(),
            audio_analysis.prosody.pitch.f0_range,
        );
        prosodic_features.insert("pitch_std".to_string(), audio_analysis.prosody.pitch.f0_std);
        prosodic_features.insert(
            "pitch_mean".to_string(),
            audio_analysis.prosody.pitch.mean_f0,
        );

        // Step 4: Extract text features
        let mut text_features = HashMap::new();
        let text = &recognition_result.transcript.text;

        text_features.insert(
            "word_count".to_string(),
            text.split_whitespace().count() as f32,
        );
        text_features.insert("char_count".to_string(), text.len() as f32);
        text_features.insert(
            "avg_word_length".to_string(),
            if text.split_whitespace().count() > 0 {
                text.chars().filter(|c| !c.is_whitespace()).count() as f32
                    / text.split_whitespace().count() as f32
            } else {
                0.0
            },
        );

        // Step 5: Classify emotion
        let emotion = EmotionType::from_features(pitch_mean, pitch_variance, energy, speaking_rate);

        // Step 6: Classify sentiment
        let sentiment = SentimentType::from_emotion_and_text(emotion, text);

        // Step 7: Calculate confidence based on feature consistency
        let confidence =
            self.calculate_confidence(&prosodic_features, &text_features, emotion, sentiment);

        Ok(EmotionSentimentResult {
            emotion,
            sentiment,
            confidence,
            prosodic_features,
            text_features,
        })
    }

    fn calculate_confidence(
        &self,
        prosodic_features: &HashMap<String, f32>,
        text_features: &HashMap<String, f32>,
        emotion: EmotionType,
        sentiment: SentimentType,
    ) -> f32 {
        let mut confidence_score = 0.0;
        let mut feature_count = 0;

        // Confidence based on prosodic feature strength
        if let Some(energy) = prosodic_features.get("energy") {
            confidence_score += energy * 2.0; // Higher energy = higher confidence
            feature_count += 1;
        }

        if let Some(pitch_variance) = prosodic_features.get("pitch_variance") {
            confidence_score += (pitch_variance / 1000.0).min(1.0); // Normalize variance
            feature_count += 1;
        }

        // Confidence based on text length (more text = higher confidence)
        if let Some(word_count) = text_features.get("word_count") {
            confidence_score += (word_count / 10.0).min(1.0);
            feature_count += 1;
        }

        // Consistency bonus (emotion and sentiment agree)
        let consistency_bonus = match (emotion, sentiment) {
            (EmotionType::Joy, SentimentType::Positive) => 0.2,
            (EmotionType::Sadness, SentimentType::Negative) => 0.2,
            (EmotionType::Anger, SentimentType::Negative) => 0.2,
            (EmotionType::Fear, SentimentType::Negative) => 0.2,
            (EmotionType::Disgust, SentimentType::Negative) => 0.2,
            (EmotionType::Surprise, SentimentType::Positive) => 0.2,
            _ => 0.0,
        };

        confidence_score += consistency_bonus;
        feature_count += 1;

        // Normalize and clamp confidence
        (confidence_score / feature_count as f32).min(1.0).max(0.0)
    }
}

#[tokio::main]
async fn main() -> Result<(), RecognitionError> {
    println!("😊 Emotion and Sentiment Recognition Demo");
    println!("=========================================\n");

    // Step 1: Initialize emotion and sentiment analyzer
    println!("🧠 Initializing emotion and sentiment analyzer...");
    let mut analyzer = EmotionSentimentAnalyzer::new().await?;
    println!("✅ Analyzer initialized with prosodic and text analysis");

    // Step 2: Create diverse emotional audio samples
    println!("\n🎭 Creating diverse emotional audio samples...");
    let emotional_samples = create_emotional_audio_samples();

    println!(
        "✅ Created {} emotional audio samples:",
        emotional_samples.len()
    );
    for (i, (emotion, _, _)) in emotional_samples.iter().enumerate() {
        println!("   {}. {}", i + 1, emotion);
    }

    // Step 3: Performance monitoring setup
    println!("\n📊 Setting up performance monitoring...");
    let validator = PerformanceValidator::new().with_verbose(false);
    println!("✅ Performance monitoring ready");

    // Step 4: Analyze emotions and sentiments
    println!("\n🔍 Analyzing emotions and sentiments...");
    println!("┌─────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Sample            │ Detected    │ Sentiment │ Confidence │ Pitch   │ Energy  │ Rate     │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────────┤");

    let mut analysis_results = Vec::new();
    let mut total_processing_time = Duration::ZERO;

    for (expected_emotion, audio, text_content) in emotional_samples {
        let analysis_start = Instant::now();
        let result = analyzer.analyze(&audio).await?;
        let processing_time = analysis_start.elapsed();

        total_processing_time += processing_time;
        analysis_results.push((expected_emotion.clone(), result.clone(), processing_time));

        // Display results
        let pitch_mean = result.prosodic_features.get("pitch_mean").unwrap_or(&0.0);
        let energy = result.prosodic_features.get("energy").unwrap_or(&0.0);
        let speaking_rate = result
            .prosodic_features
            .get("speaking_rate")
            .unwrap_or(&0.0);

        println!(
            "│ {:15} │ {:9} │ {:7} │ {:8.2} │ {:5.0}Hz │ {:5.3} │ {:6.1}/s │",
            expected_emotion,
            result.emotion.to_string(),
            result.sentiment.to_string(),
            result.confidence,
            pitch_mean,
            energy,
            speaking_rate
        );
    }

    println!("└─────────────────────────────────────────────────────────────────────────────────────────┘");

    // Step 5: Accuracy analysis
    println!("\n📈 Accuracy Analysis:");

    let mut correct_emotions = 0;
    let mut total_samples = analysis_results.len();
    let mut emotion_confusion_matrix = HashMap::new();

    for (expected, result, _) in &analysis_results {
        let expected_emotion = emotion_from_string(expected);
        let detected_emotion = result.emotion;

        if expected_emotion == detected_emotion {
            correct_emotions += 1;
        }

        // Build confusion matrix
        let key = format!(
            "{} -> {}",
            expected_emotion.to_string(),
            detected_emotion.to_string()
        );
        *emotion_confusion_matrix.entry(key).or_insert(0) += 1;
    }

    let emotion_accuracy = (correct_emotions as f32 / total_samples as f32) * 100.0;

    println!("   Emotion Recognition:");
    println!(
        "   • Correct predictions: {}/{}",
        correct_emotions, total_samples
    );
    println!("   • Accuracy: {:.1}%", emotion_accuracy);

    // Sentiment accuracy
    let mut correct_sentiments = 0;
    for (expected, result, _) in &analysis_results {
        let expected_sentiment = sentiment_from_emotion_string(expected);
        let detected_sentiment = result.sentiment;

        if expected_sentiment == detected_sentiment {
            correct_sentiments += 1;
        }
    }

    let sentiment_accuracy = (correct_sentiments as f32 / total_samples as f32) * 100.0;

    println!("   Sentiment Analysis:");
    println!(
        "   • Correct predictions: {}/{}",
        correct_sentiments, total_samples
    );
    println!("   • Accuracy: {:.1}%", sentiment_accuracy);

    // Step 6: Performance analysis
    println!("\n⚡ Performance Analysis:");

    let avg_processing_time = total_processing_time / total_samples as u32;
    let total_audio_duration: f32 = analysis_results
        .iter()
        .map(|(_, _, _)| 2.0) // Each sample is ~2 seconds
        .sum();
    let overall_rtf = total_processing_time.as_secs_f32() / total_audio_duration;

    println!("   Processing Performance:");
    println!(
        "   • Average processing time: {:.0}ms",
        avg_processing_time.as_secs_f32() * 1000.0
    );
    println!("   • Total audio duration: {:.1}s", total_audio_duration);
    println!("   • Overall RTF: {:.3}", overall_rtf);
    println!(
        "   • RTF Status: {}",
        if overall_rtf < 0.3 {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Memory usage
    let (memory_usage, memory_passed) = validator.estimate_memory_usage()?;
    println!(
        "   • Memory usage: {:.1} MB ({})",
        memory_usage as f64 / (1024.0 * 1024.0),
        if memory_passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );

    // Step 7: Feature importance analysis
    println!("\n🔍 Feature Importance Analysis:");

    let mut prosodic_feature_stats = HashMap::new();
    let mut text_feature_stats = HashMap::new();

    // Collect feature statistics
    for (_, result, _) in &analysis_results {
        for (feature, value) in &result.prosodic_features {
            let entry = prosodic_feature_stats
                .entry(feature.clone())
                .or_insert(Vec::new());
            entry.push(*value);
        }

        for (feature, value) in &result.text_features {
            let entry = text_feature_stats
                .entry(feature.clone())
                .or_insert(Vec::new());
            entry.push(*value);
        }
    }

    println!("   Prosodic Features:");
    for (feature, values) in prosodic_feature_stats {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        println!(
            "   • {}: mean={:.2}, variance={:.2}",
            feature, mean, variance
        );
    }

    println!("   Text Features:");
    for (feature, values) in text_feature_stats {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        println!(
            "   • {}: mean={:.2}, variance={:.2}",
            feature, mean, variance
        );
    }

    // Step 8: Confidence analysis
    println!("\n🎯 Confidence Analysis:");

    let confidences: Vec<f32> = analysis_results
        .iter()
        .map(|(_, result, _)| result.confidence)
        .collect();
    let avg_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
    let max_confidence = confidences.iter().copied().fold(0.0f32, f32::max);
    let min_confidence = confidences.iter().copied().fold(1.0f32, f32::min);

    println!("   • Average confidence: {:.3}", avg_confidence);
    println!("   • Maximum confidence: {:.3}", max_confidence);
    println!("   • Minimum confidence: {:.3}", min_confidence);

    // High confidence predictions
    let high_confidence_count = confidences.iter().filter(|&&c| c > 0.7).count();
    println!(
        "   • High confidence (>0.7): {}/{} ({:.1}%)",
        high_confidence_count,
        confidences.len(),
        (high_confidence_count as f32 / confidences.len() as f32) * 100.0
    );

    // Step 9: Emotion distribution analysis
    println!("\n📊 Emotion Distribution Analysis:");

    let mut emotion_counts = HashMap::new();
    let mut sentiment_counts = HashMap::new();

    for (_, result, _) in &analysis_results {
        *emotion_counts
            .entry(result.emotion.to_string())
            .or_insert(0) += 1;
        *sentiment_counts
            .entry(result.sentiment.to_string())
            .or_insert(0) += 1;
    }

    println!("   Detected Emotions:");
    for (emotion, count) in emotion_counts {
        println!(
            "   • {}: {} ({:.1}%)",
            emotion,
            count,
            (count as f32 / total_samples as f32) * 100.0
        );
    }

    println!("   Detected Sentiments:");
    for (sentiment, count) in sentiment_counts {
        println!(
            "   • {}: {} ({:.1}%)",
            sentiment,
            count,
            (count as f32 / total_samples as f32) * 100.0
        );
    }

    // Step 10: Recommendations and insights
    println!("\n💡 Insights and Recommendations:");

    if emotion_accuracy > 80.0 {
        println!("   ✅ Excellent Emotion Recognition:");
        println!("   • Prosodic features are highly discriminative");
        println!("   • Current model is production-ready");
    } else if emotion_accuracy > 60.0 {
        println!("   ✅ Good Emotion Recognition:");
        println!("   • Consider adding more training data");
        println!("   • Fine-tune feature extraction parameters");
    } else {
        println!("   ⚠️ Emotion Recognition Needs Improvement:");
        println!("   • Increase feature diversity");
        println!("   • Add machine learning classification");
        println!("   • Collect more labeled training data");
    }

    if sentiment_accuracy > 85.0 {
        println!("   ✅ Excellent Sentiment Analysis:");
        println!("   • Text and prosodic features complement well");
        println!("   • Multi-modal approach is effective");
    } else {
        println!("   ⚠️ Sentiment Analysis Improvements:");
        println!("   • Enhance text sentiment lexicon");
        println!("   • Balance prosodic and text feature weights");
        println!("   • Consider domain-specific adaptations");
    }

    if avg_confidence > 0.6 {
        println!("   ✅ Good Confidence Calibration:");
        println!("   • Confidence scores are meaningful");
        println!("   • Can be used for uncertainty estimation");
    } else {
        println!("   ⚠️ Confidence Calibration Issues:");
        println!("   • Review confidence calculation method");
        println!("   • Consider probabilistic approaches");
    }

    println!("\n✅ Emotion and sentiment recognition demo completed!");
    println!("🎯 Key achievements:");
    println!("   • Emotion accuracy: {:.1}%", emotion_accuracy);
    println!("   • Sentiment accuracy: {:.1}%", sentiment_accuracy);
    println!("   • Average confidence: {:.3}", avg_confidence);
    println!("   • Processing RTF: {:.3}", overall_rtf);

    println!("\n🚀 Next steps:");
    println!("   • Integrate with machine learning models");
    println!("   • Add real-time emotion monitoring");
    println!("   • Implement emotion-based response adaptation");
    println!("   • Create emotion-aware conversational AI");

    Ok(())
}

fn create_emotional_audio_samples() -> Vec<(String, AudioBuffer, String)> {
    let mut samples = Vec::new();
    let sample_rate = 16000;

    // Joy: High pitch, high energy, fast speech
    let joy_audio = create_emotional_audio(sample_rate, 200.0, 0.18, 4.5, "I am so happy today");
    samples.push((
        "Joy".to_string(),
        joy_audio,
        "I am so happy today".to_string(),
    ));

    // Sadness: Low pitch, low energy, slow speech
    let sadness_audio = create_emotional_audio(sample_rate, 130.0, 0.06, 2.0, "I feel so sad");
    samples.push((
        "Sadness".to_string(),
        sadness_audio,
        "I feel so sad".to_string(),
    ));

    // Anger: High variance, high energy, fast speech
    let anger_audio = create_emotional_audio(sample_rate, 180.0, 0.25, 5.0, "I am very angry");
    samples.push((
        "Anger".to_string(),
        anger_audio,
        "I am very angry".to_string(),
    ));

    // Fear: High pitch, high variance, moderate energy
    let fear_audio = create_emotional_audio(sample_rate, 210.0, 0.12, 3.8, "I am scared");
    samples.push(("Fear".to_string(), fear_audio, "I am scared".to_string()));

    // Surprise: High pitch, high variance, high energy
    let surprise_audio = create_emotional_audio(sample_rate, 195.0, 0.20, 4.2, "Oh wow amazing");
    samples.push((
        "Surprise".to_string(),
        surprise_audio,
        "Oh wow amazing".to_string(),
    ));

    // Neutral: Moderate everything
    let neutral_audio = create_emotional_audio(sample_rate, 160.0, 0.10, 3.0, "This is neutral");
    samples.push((
        "Neutral".to_string(),
        neutral_audio,
        "This is neutral".to_string(),
    ));

    samples
}

fn create_emotional_audio(
    sample_rate: u32,
    base_pitch: f32,
    amplitude: f32,
    rate: f32,
    _text: &str,
) -> AudioBuffer {
    let duration = 2.0; // 2 seconds
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // Create pitch variation based on emotion
        let pitch_variation = match base_pitch {
            p if p > 190.0 => 30.0 * (3.0 * t).sin(), // High variance for surprise/fear
            p if p > 170.0 => 20.0 * (2.0 * t).sin(), // Moderate variance for joy/anger
            _ => 10.0 * (1.0 * t).sin(),              // Low variance for sadness/neutral
        };

        let current_pitch = base_pitch + pitch_variation;

        // Create amplitude envelope based on emotion
        let envelope = match amplitude {
            a if a > 0.2 => 0.8 + 0.2 * (6.0 * t).sin(), // High energy emotions
            a if a > 0.15 => 0.7 + 0.3 * (4.0 * t).sin(), // Moderate energy
            _ => 0.5 + 0.5 * (2.0 * t).sin(),            // Low energy emotions
        };

        // Generate speech-like signal with formants
        let f1 = current_pitch;
        let f2 = current_pitch * 2.2;
        let f3 = current_pitch * 3.8;

        let speech_signal = amplitude
            * envelope
            * (0.6 * (2.0 * std::f32::consts::PI * f1 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * f2 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * f3 * t).sin());

        // Add some noise for realism
        let noise = 0.005 * (i as f32 * 0.01).sin();

        samples.push(speech_signal + noise);
    }

    AudioBuffer::mono(samples, sample_rate)
}

fn emotion_from_string(emotion_str: &str) -> EmotionType {
    match emotion_str {
        "Joy" => EmotionType::Joy,
        "Sadness" => EmotionType::Sadness,
        "Anger" => EmotionType::Anger,
        "Fear" => EmotionType::Fear,
        "Disgust" => EmotionType::Disgust,
        "Surprise" => EmotionType::Surprise,
        _ => EmotionType::Neutral,
    }
}

fn sentiment_from_emotion_string(emotion_str: &str) -> SentimentType {
    match emotion_str {
        "Joy" | "Surprise" => SentimentType::Positive,
        "Sadness" | "Anger" | "Fear" | "Disgust" => SentimentType::Negative,
        _ => SentimentType::Neutral,
    }
}
