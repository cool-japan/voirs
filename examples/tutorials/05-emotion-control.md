# Tutorial 5: Emotion Control

**Duration**: 30-35 minutes  
**Level**: Intermediate  
**Prerequisites**: Tutorials 1-4 completed

## Overview

Emotion control is one of VoiRS's standout features, allowing you to add natural emotional expression to synthesized speech. This tutorial covers both basic emotional synthesis and advanced techniques for creating emotionally intelligent voice systems.

## What You'll Learn

- Understanding emotional expression in synthesis
- Basic emotion control and intensity management
- Advanced emotional transitions and blending
- Context-aware emotional synthesis
- Emotion recognition and response systems
- Real-time emotional adaptation

## Emotion Fundamentals

### Understanding Emotion Models

```rust
use voirs_emotion::{EmotionProcessor, Emotion, EmotionVector, EmotionIntensity};

fn explore_emotion_models() -> anyhow::Result<()> {
    // Basic discrete emotions
    let discrete_emotions = vec![
        Emotion::Happy,
        Emotion::Sad,
        Emotion::Angry,
        Emotion::Excited,
        Emotion::Calm,
        Emotion::Confident,
        Emotion::Surprised,
        Emotion::Fearful,
    ];
    
    println!("🎭 Discrete Emotion Model:");
    for emotion in discrete_emotions {
        println!("  {} - {}", emotion, emotion.description());
    }
    
    // Advanced dimensional model (Valence-Arousal-Dominance)
    let dimensional_emotion = EmotionVector::new()
        .valence(0.8)      // Positive (0.0 = negative, 1.0 = positive)
        .arousal(0.6)      // Medium energy (0.0 = low, 1.0 = high)
        .dominance(0.7);   // Confident (0.0 = submissive, 1.0 = dominant)
    
    println!("\n🎯 Dimensional Emotion Model:");
    println!("  Valence: {:.2} (positivity)", dimensional_emotion.valence());
    println!("  Arousal: {:.2} (energy level)", dimensional_emotion.arousal());
    println!("  Dominance: {:.2} (confidence)", dimensional_emotion.dominance());
    
    // Convert between models
    let equivalent_discrete = dimensional_emotion.to_discrete_emotion();
    println!("  Equivalent discrete emotion: {}", equivalent_discrete);
    
    Ok(())
}
```

### Basic Emotion Synthesis

```rust
use voirs_sdk::{VoirsSdk, SynthesisConfig};
use voirs_emotion::{EmotionProcessor, Emotion, EmotionIntensity};

#[tokio::main]
async fn basic_emotion_synthesis() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    
    // Text to synthesize with different emotions
    let text = "Hello! Welcome to our voice synthesis demonstration.";
    
    // Define different emotional expressions
    let emotional_variants = vec![
        ("neutral", Emotion::Neutral, EmotionIntensity::MEDIUM),
        ("happy", Emotion::Happy, EmotionIntensity::HIGH),
        ("excited", Emotion::Excited, EmotionIntensity::VERY_HIGH),
        ("calm", Emotion::Calm, EmotionIntensity::MEDIUM),
        ("confident", Emotion::Confident, EmotionIntensity::HIGH),
        ("sad", Emotion::Sad, EmotionIntensity::MEDIUM),
    ];
    
    for (name, emotion, intensity) in emotional_variants {
        println!("🎬 Synthesizing with {} emotion (intensity: {:?})", name, intensity);
        
        // Configure emotion processor
        let emotion_processor = EmotionProcessor::builder()
            .emotion(emotion)
            .intensity(intensity)
            .enable_natural_variation(true)  // Add slight variations for naturalness
            .build()?;
        
        // Configure synthesis with emotion
        let config = SynthesisConfig::builder()
            .voice_id("default")
            .emotion_processor(emotion_processor)
            .sample_rate(22050)
            .build()?;
        
        // Synthesize
        let audio = sdk.synthesize(text, &config).await?;
        
        // Save output
        let filename = format!("emotion_{}_{}.wav", name, intensity.as_string());
        std::fs::write(&filename, audio.data)?;
        
        println!("  ✅ Saved: {} ({:.2}s)", filename, audio.duration_seconds());
    }
    
    Ok(())
}
```

## Advanced Emotion Control

### Emotion Intensity and Fine-tuning

```rust
use voirs_emotion::{EmotionProcessor, EmotionIntensity, EmotionModulation};

async fn advanced_emotion_control() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    
    // Fine-grained emotion control
    let emotion_config = EmotionProcessor::builder()
        .emotion(Emotion::Happy)
        .intensity(EmotionIntensity::custom(0.75))  // Custom intensity level
        .pitch_modulation(1.15)                     // 15% higher pitch
        .pace_modulation(1.1)                       // 10% faster speaking
        .volume_modulation(1.05)                    // 5% louder
        .breath_pattern(BreathPattern::Excited)     // Excited breathing
        .vocal_effort(VocalEffort::Energetic)       // Energetic delivery
        .build()?;
    
    // Apply micro-expressions for naturalness
    let micro_expressions = EmotionModulation::builder()
        .add_micro_pause(0.1, 0.05)               // Brief pauses
        .add_pitch_variation(0.95, 1.05)          // Natural pitch range
        .add_pace_variation(0.98, 1.02)           // Slight pace changes
        .build()?;
    
    emotion_config.apply_modulation(micro_expressions);
    
    let text = "This advanced emotion control creates incredibly natural and expressive speech!";
    
    let config = SynthesisConfig::builder()
        .emotion_processor(emotion_config)
        .build()?;
    
    let audio = sdk.synthesize(text, &config).await?;
    std::fs::write("advanced_emotion_control.wav", audio.data)?;
    
    println!("✅ Advanced emotion control demo saved");
    
    Ok(())
}
```

### Emotional Transitions and Blending

```rust
use voirs_emotion::{EmotionTransition, TransitionCurve, EmotionBlender};
use std::time::Duration;

async fn emotion_transitions() -> anyhow::Result<()> {
    let sdk = VoirsSdk::new().await?;
    
    // Define an emotional journey
    let emotional_script = vec![
        (0.0, "Hello everyone!", Emotion::Neutral),
        (2.0, "I'm excited to announce some great news!", Emotion::Excited),
        (4.5, "Unfortunately, there have been some challenges.", Emotion::Sad),
        (7.0, "But I'm confident we can overcome them together.", Emotion::Confident),
        (9.0, "Thank you for your patience and support.", Emotion::Grateful),
    ];
    
    let mut audio_segments = Vec::new();
    let mut transition_builder = EmotionTransition::builder();
    
    for (i, (timestamp, text, target_emotion)) in emotional_script.iter().enumerate() {
        // Configure transition if not the first segment
        if i > 0 {
            let previous_emotion = emotional_script[i-1].2;
            
            let transition = transition_builder
                .from_emotion(previous_emotion)
                .to_emotion(*target_emotion)
                .duration(Duration::from_millis(500))  // 500ms transition
                .curve(TransitionCurve::Smooth)         // Smooth curve
                .preserve_naturalness(true)             // Keep natural flow
                .build()?;
            
            let emotion_processor = EmotionProcessor::builder()
                .transition(transition)
                .build()?;
                
            let config = SynthesisConfig::builder()
                .emotion_processor(emotion_processor)
                .build()?;
        } else {
            // First segment - no transition needed
            let emotion_processor = EmotionProcessor::builder()
                .emotion(*target_emotion)
                .intensity(EmotionIntensity::MEDIUM)
                .build()?;
                
            let config = SynthesisConfig::builder()
                .emotion_processor(emotion_processor)
                .build()?;
        }
        
        let audio = sdk.synthesize(text, &config).await?;
        audio_segments.push(audio);
        
        println!("✅ Synthesized segment {}: {} ({})", i+1, text, target_emotion);
    }
    
    // Combine all segments into a cohesive narrative
    let final_audio = combine_audio_segments(audio_segments).await?;
    std::fs::write("emotional_journey.wav", final_audio.data)?;
    
    println!("🎭 Complete emotional journey saved as 'emotional_journey.wav'");
    
    Ok(())
}

async fn combine_audio_segments(segments: Vec<AudioData>) -> anyhow::Result<AudioData> {
    use voirs_sdk::AudioConcatenator;
    
    let mut concatenator = AudioConcatenator::new();
    
    for (i, segment) in segments.into_iter().enumerate() {
        concatenator.add_audio(segment);
        
        // Add small pause between segments except the last one
        if i < segments.len() - 1 {
            concatenator.add_silence(0.3);  // 300ms pause
        }
    }
    
    concatenator.build()
}
```

## Context-Aware Emotional Synthesis

### Emotion Recognition from Text

```rust
use voirs_emotion::{EmotionAnalyzer, TextualCue, EmotionContext};

struct ContextualEmotionSynthesizer {
    sdk: VoirsSdk,
    emotion_analyzer: EmotionAnalyzer,
}

impl ContextualEmotionSynthesizer {
    async fn new() -> anyhow::Result<Self> {
        Ok(Self {
            sdk: VoirsSdk::new().await?,
            emotion_analyzer: EmotionAnalyzer::new()?,
        })
    }
    
    async fn synthesize_with_context(&self, text: &str, context: Option<EmotionContext>) -> anyhow::Result<AudioData> {
        // Analyze text for emotional cues
        let detected_emotions = self.emotion_analyzer
            .analyze_text(text)?
            .detect_sentiment()?
            .identify_keywords()?
            .assess_intensity()?
            .build();
        
        println!("🔍 Detected emotions in text:");
        for emotion_cue in &detected_emotions.cues {
            println!("  {} (confidence: {:.2}): {}", 
                emotion_cue.emotion, 
                emotion_cue.confidence, 
                emotion_cue.trigger_phrase);
        }
        
        // Apply context if provided
        let final_emotion = if let Some(ctx) = context {
            self.blend_context_with_detection(detected_emotions, ctx)?
        } else {
            detected_emotions.primary_emotion()
        };
        
        // Configure synthesis with detected/contextual emotion
        let emotion_processor = EmotionProcessor::builder()
            .emotion(final_emotion.emotion)
            .intensity(final_emotion.intensity)
            .confidence_weight(final_emotion.confidence)
            .enable_adaptive_expression(true)
            .build()?;
        
        let config = SynthesisConfig::builder()
            .emotion_processor(emotion_processor)
            .build()?;
        
        self.sdk.synthesize(text, &config).await
    }
    
    fn blend_context_with_detection(&self, detected: EmotionAnalysis, context: EmotionContext) -> anyhow::Result<WeightedEmotion> {
        let context_weight = context.strength();
        let detection_weight = 1.0 - context_weight;
        
        // Blend detected emotion with contextual emotion
        let blended = EmotionBlender::new()
            .add_emotion(detected.primary_emotion(), detection_weight)
            .add_emotion(context.base_emotion(), context_weight)
            .normalize()
            .build()?;
        
        Ok(blended)
    }
}

async fn contextual_synthesis_demo() -> anyhow::Result<()> {
    let synthesizer = ContextualEmotionSynthesizer::new().await?;
    
    // Test sentences with different emotional contexts
    let test_cases = vec![
        (
            "The project is completed ahead of schedule.",
            Some(EmotionContext::workplace_announcement()),
            "workplace_good_news.wav"
        ),
        (
            "I can't believe this happened to me.",
            Some(EmotionContext::personal_disappointment()),
            "personal_disappointment.wav"
        ),
        (
            "Welcome to our annual celebration!",
            Some(EmotionContext::festive_event()),
            "celebration_welcome.wav"
        ),
        (
            "Please consider all options carefully.",
            Some(EmotionContext::formal_advice()),
            "formal_advice.wav"
        ),
    ];
    
    for (text, context, filename) in test_cases {
        println!("🎯 Processing: {}", text);
        
        let audio = synthesizer.synthesize_with_context(text, context).await?;
        std::fs::write(filename, audio.data)?;
        
        println!("  ✅ Saved: {}\n", filename);
    }
    
    Ok(())
}
```

## Real-time Emotional Adaptation

### Streaming Emotional Synthesis

```rust
use voirs_emotion::{RealTimeEmotionProcessor, EmotionFeedback};
use tokio::sync::mpsc;

struct RealTimeEmotionalSynthesizer {
    sdk: VoirsSdk,
    emotion_processor: RealTimeEmotionProcessor,
    feedback_receiver: mpsc::Receiver<EmotionFeedback>,
}

impl RealTimeEmotionalSynthesizer {
    async fn new() -> anyhow::Result<(Self, mpsc::Sender<EmotionFeedback>)> {
        let (feedback_sender, feedback_receiver) = mpsc::channel(100);
        
        let synthesizer = Self {
            sdk: VoirsSdk::new().await?,
            emotion_processor: RealTimeEmotionProcessor::new()?,
            feedback_receiver,
        };
        
        Ok((synthesizer, feedback_sender))
    }
    
    async fn start_synthesis_loop(&mut self) -> anyhow::Result<()> {
        let mut current_emotion = Emotion::Neutral;
        let mut current_intensity = EmotionIntensity::MEDIUM;
        
        println!("🎙️ Starting real-time emotional synthesis...");
        println!("Send text via stdin, type 'quit' to exit");
        
        let mut stdin_lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();
        
        loop {
            tokio::select! {
                // Handle text input
                line_result = stdin_lines.next_line() => {
                    match line_result? {
                        Some(text) if text.trim() == "quit" => break,
                        Some(text) => {
                            self.process_text_with_current_emotion(&text, current_emotion, current_intensity).await?;
                        }
                        None => break,
                    }
                }
                
                // Handle emotion feedback
                feedback = self.feedback_receiver.recv() => {
                    if let Some(feedback) = feedback {
                        current_emotion = self.adapt_emotion(current_emotion, feedback.emotion);
                        current_intensity = self.adapt_intensity(current_intensity, feedback.intensity_delta);
                        
                        println!("🔄 Emotion adapted to: {} (intensity: {:?})", 
                            current_emotion, current_intensity);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn process_text_with_current_emotion(
        &self,
        text: &str,
        emotion: Emotion,
        intensity: EmotionIntensity
    ) -> anyhow::Result<()> {
        let emotion_processor = EmotionProcessor::builder()
            .emotion(emotion)
            .intensity(intensity)
            .enable_real_time_adaptation(true)
            .build()?;
        
        let config = SynthesisConfig::builder()
            .emotion_processor(emotion_processor)
            .enable_streaming(true)
            .build()?;
        
        // Stream synthesis for real-time output
        let mut stream = self.sdk.synthesize_streaming(text, &config).await?;
        
        while let Some(chunk) = stream.next().await {
            // In a real application, you would send this to an audio output device
            // For this example, we'll just show progress
            print!("🔊");
            std::io::stdout().flush()?;
        }
        
        println!(" ✅ Synthesis complete");
        
        Ok(())
    }
    
    fn adapt_emotion(&self, current: Emotion, feedback: Emotion) -> Emotion {
        // Simple emotion adaptation - in a real system this would be more sophisticated
        match (current, feedback) {
            (Emotion::Neutral, new_emotion) => new_emotion,
            (current_emotion, Emotion::Neutral) => current_emotion,
            (_, feedback_emotion) => {
                // Blend current emotion with feedback
                EmotionBlender::new()
                    .add_emotion(current_emotion, 0.7)
                    .add_emotion(feedback_emotion, 0.3)
                    .blend()
                    .primary_emotion()
            }
        }
    }
    
    fn adapt_intensity(&self, current: EmotionIntensity, delta: f32) -> EmotionIntensity {
        let current_value = current.as_float();
        let new_value = (current_value + delta).clamp(0.0, 1.0);
        EmotionIntensity::custom(new_value)
    }
}

async fn real_time_synthesis_demo() -> anyhow::Result<()> {
    let (mut synthesizer, feedback_sender) = RealTimeEmotionalSynthesizer::new().await?;
    
    // Spawn a task to simulate feedback
    let _feedback_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(5)).await;
        let _ = feedback_sender.send(EmotionFeedback {
            emotion: Emotion::Excited,
            intensity_delta: 0.2,
            timestamp: std::time::Instant::now(),
        }).await;
        
        tokio::time::sleep(Duration::from_secs(10)).await;
        let _ = feedback_sender.send(EmotionFeedback {
            emotion: Emotion::Calm,
            intensity_delta: -0.3,
            timestamp: std::time::Instant::now(),
        }).await;
    });
    
    synthesizer.start_synthesis_loop().await?;
    
    Ok(())
}
```

## Emotion Evaluation and Quality Assessment

### Perceptual Emotion Evaluation

```rust
use voirs_evaluation::{EmotionEvaluator, PerceptualMetrics};

async fn evaluate_emotional_synthesis() -> anyhow::Result<()> {
    let evaluator = EmotionEvaluator::new().await?;
    
    // Test different emotional expressions
    let emotion_test_cases = vec![
        ("happy", Emotion::Happy, "I'm so excited about this opportunity!"),
        ("sad", Emotion::Sad, "I'm disappointed by the recent developments."),
        ("angry", Emotion::Angry, "This is completely unacceptable behavior!"),
        ("calm", Emotion::Calm, "Let's take a moment to think about this calmly."),
        ("confident", Emotion::Confident, "I know we can achieve our goals together."),
    ];
    
    let mut evaluation_results = Vec::new();
    
    for (emotion_name, emotion, text) in emotion_test_cases {
        println!("🧪 Evaluating {} emotion synthesis...", emotion_name);
        
        // Synthesize with target emotion
        let emotion_processor = EmotionProcessor::builder()
            .emotion(emotion)
            .intensity(EmotionIntensity::HIGH)
            .build()?;
        
        let config = SynthesisConfig::builder()
            .emotion_processor(emotion_processor)
            .build()?;
        
        let sdk = VoirsSdk::new().await?;
        let audio = sdk.synthesize(text, &config).await?;
        
        // Evaluate emotional authenticity
        let metrics = evaluator
            .evaluate_emotion_accuracy(&audio, emotion).await?
            .evaluate_naturalness(&audio).await?
            .evaluate_intensity_match(&audio, EmotionIntensity::HIGH).await?
            .evaluate_consistency(&audio).await?
            .build();
        
        println!("  📊 Evaluation Results:");
        println!("    Emotion accuracy: {:.3}", metrics.emotion_accuracy);
        println!("    Naturalness: {:.3}", metrics.naturalness);
        println!("    Intensity match: {:.3}", metrics.intensity_match);
        println!("    Consistency: {:.3}", metrics.consistency);
        println!("    Overall score: {:.3}", metrics.overall_score());
        
        evaluation_results.push((emotion_name, metrics));
        
        // Save audio for manual evaluation
        let filename = format!("emotion_eval_{}.wav", emotion_name);
        std::fs::write(&filename, audio.data)?;
        println!("    💾 Saved: {}", filename);
        println!();
    }
    
    // Generate evaluation report
    generate_emotion_evaluation_report(evaluation_results)?;
    
    Ok(())
}

fn generate_emotion_evaluation_report(results: Vec<(&str, PerceptualMetrics)>) -> anyhow::Result<()> {
    println!("📋 EMOTION SYNTHESIS EVALUATION REPORT");
    println!("=" .repeat(50));
    
    println!("{:<12} {:>12} {:>12} {:>12} {:>12} {:>12}", 
        "Emotion", "Accuracy", "Naturalness", "Intensity", "Consistency", "Overall");
    println!("-".repeat(80));
    
    let mut total_score = 0.0;
    for (emotion_name, metrics) in &results {
        println!("{:<12} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
            emotion_name,
            metrics.emotion_accuracy,
            metrics.naturalness,
            metrics.intensity_match,
            metrics.consistency,
            metrics.overall_score()
        );
        total_score += metrics.overall_score();
    }
    
    println!("-".repeat(80));
    println!("{:<12} {:>60.3}", "AVERAGE", total_score / results.len() as f64);
    
    // Identify best and worst performing emotions
    let best = results.iter().max_by(|a, b| a.1.overall_score().partial_cmp(&b.1.overall_score()).unwrap()).unwrap();
    let worst = results.iter().min_by(|a, b| a.1.overall_score().partial_cmp(&b.1.overall_score()).unwrap()).unwrap();
    
    println!("\n🏆 Best performing emotion: {} (score: {:.3})", best.0, best.1.overall_score());
    println!("⚠️  Needs improvement: {} (score: {:.3})", worst.0, worst.1.overall_score());
    
    // Save detailed report
    let report_json = serde_json::to_string_pretty(&results)?;
    std::fs::write("emotion_evaluation_report.json", report_json)?;
    
    println!("\n💾 Detailed report saved to: emotion_evaluation_report.json");
    
    Ok(())
}
```

## Complete Example: Emotional Chatbot

```rust
use voirs_emotion::{EmotionProcessor, ConversationContext, EmotionHistory};

struct EmotionalChatbot {
    sdk: VoirsSdk,
    conversation_context: ConversationContext,
    emotion_history: EmotionHistory,
    base_personality: Emotion,
}

impl EmotionalChatbot {
    async fn new(personality: Emotion) -> anyhow::Result<Self> {
        Ok(Self {
            sdk: VoirsSdk::new().await?,
            conversation_context: ConversationContext::new(),
            emotion_history: EmotionHistory::new(),
            base_personality: personality,
        })
    }
    
    async fn respond(&mut self, user_message: &str) -> anyhow::Result<AudioData> {
        // Analyze user's emotional state from message
        let user_emotion = self.analyze_user_emotion(user_message)?;
        
        // Update conversation context
        self.conversation_context.add_user_input(user_message, user_emotion);
        
        // Generate appropriate response text (simplified for this example)
        let response_text = self.generate_response_text(user_message, user_emotion)?;
        
        // Determine appropriate emotional response
        let response_emotion = self.determine_response_emotion(user_emotion)?;
        
        // Update emotion history
        self.emotion_history.add_emotion(response_emotion, std::time::Instant::now());
        
        // Configure emotional synthesis
        let emotion_processor = EmotionProcessor::builder()
            .emotion(response_emotion.emotion)
            .intensity(response_emotion.intensity)
            .personality_bias(self.base_personality)
            .conversation_context(&self.conversation_context)
            .emotion_history(&self.emotion_history)
            .enable_empathetic_response(true)
            .build()?;
        
        let config = SynthesisConfig::builder()
            .emotion_processor(emotion_processor)
            .voice_id("friendly-assistant")
            .build()?;
        
        // Synthesize emotional response
        let audio = self.sdk.synthesize(&response_text, &config).await?;
        
        // Update conversation context with our response
        self.conversation_context.add_bot_response(&response_text, response_emotion);
        
        Ok(audio)
    }
    
    fn analyze_user_emotion(&self, message: &str) -> anyhow::Result<DetectedEmotion> {
        // Simplified emotion detection - in reality this would be more sophisticated
        let message_lower = message.to_lowercase();
        
        if message_lower.contains("happy") || message_lower.contains("great") || message_lower.contains("awesome") {
            Ok(DetectedEmotion::new(Emotion::Happy, EmotionIntensity::HIGH, 0.8))
        } else if message_lower.contains("sad") || message_lower.contains("disappointed") {
            Ok(DetectedEmotion::new(Emotion::Sad, EmotionIntensity::MEDIUM, 0.7))
        } else if message_lower.contains("angry") || message_lower.contains("frustrated") {
            Ok(DetectedEmotion::new(Emotion::Angry, EmotionIntensity::MEDIUM, 0.6))
        } else if message_lower.contains("help") || message_lower.contains("confused") {
            Ok(DetectedEmotion::new(Emotion::Concerned, EmotionIntensity::MEDIUM, 0.5))
        } else {
            Ok(DetectedEmotion::new(Emotion::Neutral, EmotionIntensity::MEDIUM, 0.3))
        }
    }
    
    fn generate_response_text(&self, user_message: &str, user_emotion: DetectedEmotion) -> anyhow::Result<String> {
        // Simplified response generation based on detected emotion
        let response = match user_emotion.emotion {
            Emotion::Happy => "That's wonderful to hear! I'm so glad you're feeling positive.",
            Emotion::Sad => "I'm sorry you're feeling down. Is there anything I can do to help?",
            Emotion::Angry => "I understand your frustration. Let's see how we can address this together.",
            Emotion::Concerned => "I'm here to help. Let me guide you through this step by step.",
            _ => "Thank you for sharing that with me. How else can I assist you today?",
        };
        
        Ok(response.to_string())
    }
    
    fn determine_response_emotion(&self, user_emotion: DetectedEmotion) -> anyhow::Result<EmotionalResponse> {
        // Empathetic response strategy
        let response_emotion = match user_emotion.emotion {
            Emotion::Happy => EmotionalResponse::new(Emotion::Happy, EmotionIntensity::MEDIUM),
            Emotion::Sad => EmotionalResponse::new(Emotion::Compassionate, EmotionIntensity::MEDIUM),
            Emotion::Angry => EmotionalResponse::new(Emotion::Calm, EmotionIntensity::HIGH),
            Emotion::Concerned => EmotionalResponse::new(Emotion::Helpful, EmotionIntensity::MEDIUM),
            _ => EmotionalResponse::new(self.base_personality, EmotionIntensity::MEDIUM),
        };
        
        Ok(response_emotion)
    }
}

async fn emotional_chatbot_demo() -> anyhow::Result<()> {
    let mut chatbot = EmotionalChatbot::new(Emotion::Friendly).await?;
    
    // Simulate conversation
    let conversation = vec![
        "Hi there! I'm having a great day!",
        "Actually, I'm feeling a bit frustrated about work.",
        "Thank you for understanding. Can you help me with something?",
        "That's exactly what I needed to hear. You're very helpful!",
    ];
    
    for (i, user_message) in conversation.iter().enumerate() {
        println!("👤 User: {}", user_message);
        
        let response_audio = chatbot.respond(user_message).await?;
        
        // Save the response (in reality, you'd play it)
        let filename = format!("chatbot_response_{}.wav", i + 1);
        std::fs::write(&filename, response_audio.data)?;
        
        println!("🤖 Bot: [Audio response saved as {}]", filename);
        println!("    Conversation context: {:?}", chatbot.conversation_context.current_mood());
        println!();
        
        // Small delay to simulate natural conversation flow
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    
    // Generate conversation summary
    let summary = chatbot.conversation_context.generate_summary()?;
    println!("📊 Conversation Summary:");
    println!("  Duration: {:.1}s", summary.duration.as_secs_f64());
    println!("  User emotions detected: {:?}", summary.user_emotions);
    println!("  Bot emotions used: {:?}", summary.bot_emotions);
    println!("  Overall conversation mood: {:?}", summary.overall_mood);
    
    Ok(())
}
```

## Best Practices for Emotional Synthesis

1. **Match Emotion to Content**: Ensure emotional expression aligns with text meaning
2. **Use Appropriate Intensity**: Don't over-dramatize unless specifically needed
3. **Consider Context**: Take into account the situation and audience
4. **Test Perceptually**: Evaluate emotional authenticity with human listeners
5. **Enable Natural Variation**: Use micro-expressions for more natural speech
6. **Respect Cultural Differences**: Emotional expression varies across cultures

## Common Issues and Solutions

### Issue: "Emotions sound artificial"
**Solution**: Enable natural variation and use micro-expressions:
```rust
.enable_natural_variation(true)
.add_micro_expressions(true)
.emotion_stability(0.8)  // Allow some variation
```

### Issue: "Emotional transitions are jarring"
**Solution**: Use smooth transitions between emotions:
```rust
.transition_duration(Duration::from_millis(500))
.transition_curve(TransitionCurve::Smooth)
```

### Issue: "Intensity is too extreme"
**Solution**: Use custom intensity levels and test different values:
```rust
.intensity(EmotionIntensity::custom(0.6))  // More subtle
```

## Next Steps

In the next tutorial, we'll explore real-time processing capabilities for interactive applications and live synthesis.

Continue to [Tutorial 6: Real-time Processing](./06-realtime-processing.md) →

## Additional Resources

- [Emotion Control Examples](../emotion_control_example.rs)
- [Emotional Chatbot Example](../conversational_ai_example.rs)
- [Emotion Evaluation Guide](../emotion_evaluation.rs)

---

**Estimated completion time**: 30-35 minutes  
**Difficulty**: ⭐⭐⭐☆☆  
**Next tutorial**: [Real-time Processing](./06-realtime-processing.md)