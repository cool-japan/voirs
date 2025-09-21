//! Emotion detection implementation
//!
//! Provides the main emotion recognition engine with multi-dimensional analysis,
//! sentiment polarity detection, and temporal tracking.

use super::{
    EmotionConfig, EmotionDetection, EmotionRecognizer, EmotionStats, EmotionType,
    MultiDimensionalEmotion, SentimentAnalysis, SentimentPolarity, EmotionFeatureExtractor,
    EmotionTracker, EmotionModel
};
use crate::RecognitionError;
use voirs_sdk::AudioBuffer;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Main emotion detector implementation
pub struct EmotionDetectorImpl {
    /// Configuration
    config: EmotionConfig,
    /// Feature extractor
    feature_extractor: EmotionFeatureExtractor,
    /// Emotion model
    model: Arc<dyn EmotionModel + Send + Sync>,
    /// Emotion tracker
    tracker: Arc<Mutex<EmotionTracker>>,
    /// Statistics
    stats: Arc<Mutex<EmotionStats>>,
    /// Analysis history for temporal smoothing
    emotion_history: Arc<Mutex<Vec<EmotionDetection>>>,
    /// Sentiment history for temporal smoothing
    sentiment_history: Arc<Mutex<Vec<SentimentAnalysis>>>,
}

impl EmotionDetectorImpl {
    /// Create new emotion detector
    pub async fn new(
        config: EmotionConfig,
        model: Arc<dyn EmotionModel + Send + Sync>,
    ) -> Result<Self, RecognitionError> {
        let feature_extractor = EmotionFeatureExtractor::new();
        let tracker = Arc::new(Mutex::new(EmotionTracker::new()));
        let stats = Arc::new(Mutex::new(EmotionStats::default()));
        
        Ok(Self {
            config,
            feature_extractor,
            model,
            tracker,
            stats,
            emotion_history: Arc::new(Mutex::new(Vec::new())),
            sentiment_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Extract audio features for emotion recognition
    async fn extract_features(&self, audio: &AudioBuffer) -> Result<HashMap<String, f32>, RecognitionError> {
        self.feature_extractor.extract_emotion_features(audio).await
    }

    /// Apply temporal smoothing to emotion detection
    fn apply_temporal_smoothing(&self, current_emotion: &EmotionDetection) -> EmotionDetection {
        if self.config.temporal_smoothing <= 0.0 {
            return current_emotion.clone();
        }

        let history = self.emotion_history.lock().unwrap();
        
        // Get recent emotions for the same type
        let recent_emotions: Vec<_> = history.iter()
            .filter(|e| e.emotion == current_emotion.emotion)
            .filter(|e| current_emotion.timestamp.duration_since(e.timestamp) < Duration::from_secs(10))
            .collect();

        if recent_emotions.is_empty() {
            return current_emotion.clone();
        }

        // Apply exponential smoothing
        let alpha = self.config.temporal_smoothing;
        let avg_confidence = recent_emotions.iter()
            .map(|e| e.confidence)
            .sum::<f32>() / recent_emotions.len() as f32;
        let avg_intensity = recent_emotions.iter()
            .map(|e| e.intensity)
            .sum::<f32>() / recent_emotions.len() as f32;

        let mut smoothed = current_emotion.clone();
        smoothed.confidence = alpha * current_emotion.confidence + (1.0 - alpha) * avg_confidence;
        smoothed.intensity = alpha * current_emotion.intensity + (1.0 - alpha) * avg_intensity;
        
        smoothed
    }

    /// Apply temporal smoothing to sentiment analysis
    fn apply_sentiment_smoothing(&self, current_sentiment: &SentimentAnalysis) -> SentimentAnalysis {
        if self.config.temporal_smoothing <= 0.0 {
            return current_sentiment.clone();
        }

        let history = self.sentiment_history.lock().unwrap();
        
        let recent_sentiments: Vec<_> = history.iter()
            .filter(|s| current_sentiment.timestamp.duration_since(s.timestamp) < Duration::from_secs(10))
            .collect();

        if recent_sentiments.is_empty() {
            return current_sentiment.clone();
        }

        let alpha = self.config.temporal_smoothing;
        let avg_valence = recent_sentiments.iter()
            .map(|s| s.valence)
            .sum::<f32>() / recent_sentiments.len() as f32;
        let avg_arousal = recent_sentiments.iter()
            .map(|s| s.arousal)
            .sum::<f32>() / recent_sentiments.len() as f32;
        let avg_dominance = recent_sentiments.iter()
            .map(|s| s.dominance)
            .sum::<f32>() / recent_sentiments.len() as f32;

        let mut smoothed = current_sentiment.clone();
        smoothed.valence = alpha * current_sentiment.valence + (1.0 - alpha) * avg_valence;
        smoothed.arousal = alpha * current_sentiment.arousal + (1.0 - alpha) * avg_arousal;
        smoothed.dominance = alpha * current_sentiment.dominance + (1.0 - alpha) * avg_dominance;
        
        // Recalculate polarity based on smoothed valence
        smoothed.polarity = if smoothed.valence > 0.1 {
            SentimentPolarity::Positive
        } else if smoothed.valence < -0.1 {
            SentimentPolarity::Negative
        } else {
            SentimentPolarity::Neutral
        };

        smoothed
    }

    /// Update emotion statistics
    fn update_stats(&self, emotion: &EmotionDetection, sentiment: Option<&SentimentAnalysis>) {
        let mut stats = self.stats.lock().unwrap();
        
        stats.total_analyses += 1;
        stats.last_analysis = Some(emotion.timestamp);
        
        // Update emotion distribution
        *stats.emotion_distribution.entry(emotion.emotion).or_insert(0) += 1;
        
        // Update average confidence
        let emotion_count = stats.emotion_distribution[&emotion.emotion] as f32;
        let current_avg = stats.avg_confidence.get(&emotion.emotion).copied().unwrap_or(0.0);
        let new_avg = (current_avg * (emotion_count - 1.0) + emotion.confidence) / emotion_count;
        stats.avg_confidence.insert(emotion.emotion, new_avg);
        
        // Update sentiment distribution
        if let Some(sent) = sentiment {
            *stats.sentiment_distribution.entry(sent.polarity).or_insert(0) += 1;
        }
    }

    /// Clean up old history entries
    fn cleanup_history(&self) {
        let cutoff = Instant::now() - Duration::from_secs(300); // Keep 5 minutes
        
        {
            let mut emotion_history = self.emotion_history.lock().unwrap();
            emotion_history.retain(|e| e.timestamp > cutoff);
            if emotion_history.len() > 100 {
                emotion_history.drain(0..emotion_history.len() - 100);
            }
        }
        
        {
            let mut sentiment_history = self.sentiment_history.lock().unwrap();
            sentiment_history.retain(|s| s.timestamp > cutoff);
            if sentiment_history.len() > 100 {
                sentiment_history.drain(0..sentiment_history.len() - 100);
            }
        }
    }
}

#[async_trait]
impl EmotionRecognizer for EmotionDetectorImpl {
    async fn detect_emotions(&mut self, audio: &AudioBuffer) -> Result<Vec<EmotionDetection>, RecognitionError> {
        // Extract features
        let features = self.extract_features(audio).await?;
        
        // Run emotion detection through model
        let emotion_scores = self.model.predict_emotions(&features).await?;
        
        let mut detections = Vec::new();
        
        for (emotion, score) in emotion_scores {
            if score < self.config.min_confidence {
                continue;
            }
            
            let detection = EmotionDetection {
                emotion,
                confidence: score,
                intensity: score, // Simple mapping, could be more sophisticated
                timestamp: Instant::now(),
                start_time: Duration::ZERO,
                end_time: Duration::from_secs_f32(audio.samples().len() as f32 / audio.sample_rate() as f32),
                emotion_scores: emotion_scores.clone(),
            };
            
            // Apply temporal smoothing if enabled
            let smoothed_detection = if self.config.enable_tracking {
                self.apply_temporal_smoothing(&detection)
            } else {
                detection
            };
            
            detections.push(smoothed_detection.clone());
            
            // Add to history
            {
                let mut history = self.emotion_history.lock().unwrap();
                history.push(smoothed_detection);
            }
        }
        
        // Update statistics
        for detection in &detections {
            self.update_stats(detection, None);
        }
        
        // Update tracker
        if self.config.enable_tracking && !detections.is_empty() {
            let mut tracker = self.tracker.lock().unwrap();
            tracker.update(&detections[0]).await;
        }
        
        // Clean up old history
        self.cleanup_history();
        
        Ok(detections)
    }

    async fn analyze_sentiment(&mut self, audio: &AudioBuffer) -> Result<SentimentAnalysis, RecognitionError> {
        // Extract features
        let features = self.extract_features(audio).await?;
        
        // Run sentiment analysis through model
        let sentiment_scores = self.model.predict_sentiment(&features).await?;
        
        let valence = sentiment_scores.get("valence").copied().unwrap_or(0.0);
        let arousal = sentiment_scores.get("arousal").copied().unwrap_or(0.0);
        let dominance = sentiment_scores.get("dominance").copied().unwrap_or(0.0);
        
        // Determine polarity from valence
        let polarity = if valence > 0.1 {
            SentimentPolarity::Positive
        } else if valence < -0.1 {
            SentimentPolarity::Negative
        } else {
            SentimentPolarity::Neutral
        };
        
        // Calculate confidence based on absolute valence
        let confidence = valence.abs().min(1.0);
        
        let sentiment = SentimentAnalysis {
            polarity,
            confidence,
            valence,
            arousal,
            dominance,
            timestamp: Instant::now(),
        };
        
        // Apply temporal smoothing if enabled
        let smoothed_sentiment = if self.config.enable_tracking {
            self.apply_sentiment_smoothing(&sentiment)
        } else {
            sentiment
        };
        
        // Add to history
        {
            let mut history = self.sentiment_history.lock().unwrap();
            history.push(smoothed_sentiment.clone());
        }
        
        // Update statistics
        self.update_stats(&EmotionDetection {
            emotion: EmotionType::Neutral,
            confidence: smoothed_sentiment.confidence,
            intensity: smoothed_sentiment.arousal,
            timestamp: smoothed_sentiment.timestamp,
            start_time: Duration::ZERO,
            end_time: Duration::from_secs_f32(audio.samples().len() as f32 / audio.sample_rate() as f32),
            emotion_scores: HashMap::new(),
        }, Some(&smoothed_sentiment));
        
        Ok(smoothed_sentiment)
    }

    async fn analyze_multi_dimensional(&mut self, audio: &AudioBuffer) -> Result<MultiDimensionalEmotion, RecognitionError> {
        // Get emotion detections
        let emotions = self.detect_emotions(audio).await?;
        
        if emotions.is_empty() {
            return Err(RecognitionError::AudioAnalysisError {
                message: "No emotions detected in audio".to_string(),
                source: None,
            });
        }
        
        let primary_emotion = emotions[0].clone();
        let secondary_emotions = emotions[1..].to_vec();
        
        // Get sentiment analysis
        let sentiment = if self.config.sentiment_analysis {
            self.analyze_sentiment(audio).await?
        } else {
            SentimentAnalysis {
                polarity: SentimentPolarity::Neutral,
                confidence: 0.0,
                valence: 0.0,
                arousal: 0.0,
                dominance: 0.0,
                timestamp: Instant::now(),
            }
        };
        
        // Detect stress and fatigue
        let stress_level = if self.config.stress_detection {
            self.detect_stress(audio).await?
        } else {
            0.0
        };
        
        let fatigue_level = if self.config.fatigue_detection {
            self.detect_fatigue(audio).await?
        } else {
            0.0
        };
        
        // Calculate complexity score based on number of emotions and their intensities
        let complexity_score = if emotions.len() == 1 {
            0.2 // Simple emotional state
        } else {
            let intensity_variance = emotions.iter()
                .map(|e| e.intensity)
                .fold(0.0, |acc, intensity| {
                    let mean = emotions.iter().map(|e| e.intensity).sum::<f32>() / emotions.len() as f32;
                    acc + (intensity - mean).powi(2)
                }) / emotions.len() as f32;
            (0.2 + intensity_variance * 2.0).min(1.0)
        };
        
        Ok(MultiDimensionalEmotion {
            primary_emotion,
            secondary_emotions,
            sentiment,
            stress_level,
            fatigue_level,
            complexity_score,
        })
    }

    async fn detect_stress(&mut self, audio: &AudioBuffer) -> Result<f32, RecognitionError> {
        let features = self.extract_features(audio).await?;
        
        // Stress indicators from audio features
        let voice_tremor = features.get("jitter").copied().unwrap_or(0.0);
        let pitch_variance = features.get("pitch_variance").copied().unwrap_or(0.0);
        let speaking_rate = features.get("speaking_rate").copied().unwrap_or(1.0);
        let pause_frequency = features.get("pause_frequency").copied().unwrap_or(0.0);
        
        // Simple stress calculation (would use ML model in production)
        let stress_score = (voice_tremor * 0.3 + 
                           pitch_variance * 0.3 + 
                           (speaking_rate - 1.0).abs() * 0.2 + 
                           pause_frequency * 0.2).min(1.0);
        
        Ok(stress_score)
    }

    async fn detect_fatigue(&mut self, audio: &AudioBuffer) -> Result<f32, RecognitionError> {
        let features = self.extract_features(audio).await?;
        
        // Fatigue indicators from audio features
        let energy_level = features.get("energy").copied().unwrap_or(0.5);
        let articulation_clarity = features.get("articulation").copied().unwrap_or(1.0);
        let speaking_rate = features.get("speaking_rate").copied().unwrap_or(1.0);
        let pitch_range = features.get("pitch_range").copied().unwrap_or(1.0);
        
        // Simple fatigue calculation (would use ML model in production)
        let fatigue_score = ((1.0 - energy_level) * 0.4 + 
                            (1.0 - articulation_clarity) * 0.3 + 
                            (1.0 - speaking_rate.min(1.0)) * 0.2 + 
                            (1.0 - pitch_range) * 0.1).min(1.0);
        
        Ok(fatigue_score)
    }

    async fn update_config(&mut self, config: EmotionConfig) -> Result<(), RecognitionError> {
        self.config = config;
        Ok(())
    }

    fn get_config(&self) -> &EmotionConfig {
        &self.config
    }

    fn get_supported_emotions(&self) -> Vec<EmotionType> {
        EmotionType::all()
    }
}

impl EmotionDetectorImpl {
    /// Get emotion statistics
    pub fn get_stats(&self) -> EmotionStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }
    
    /// Get emotion tracker
    pub fn get_tracker(&self) -> Arc<Mutex<EmotionTracker>> {
        Arc::clone(&self.tracker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::emotion::models::MockEmotionModel;

    #[tokio::test]
    async fn test_emotion_detector_creation() {
        let config = EmotionConfig::default();
        let model = Arc::new(MockEmotionModel::new());
        
        let detector = EmotionDetectorImpl::new(config, model).await;
        assert!(detector.is_ok());
        
        let detector = detector.unwrap();
        assert_eq!(detector.get_supported_emotions().len(), 12);
    }

    #[tokio::test]
    async fn test_emotion_detection() {
        let config = EmotionConfig::default();
        let model = Arc::new(MockEmotionModel::new());
        
        let mut detector = EmotionDetectorImpl::new(config, model).await.unwrap();
        
        // Create test audio
        let samples = vec![0.1f32; 16000]; // 1 second at 16kHz
        let audio = AudioBuffer::mono(samples, 16000);
        
        let emotions = detector.detect_emotions(&audio).await;
        assert!(emotions.is_ok());
    }

    #[tokio::test]
    async fn test_sentiment_analysis() {
        let config = EmotionConfig::default();
        let model = Arc::new(MockEmotionModel::new());
        
        let mut detector = EmotionDetectorImpl::new(config, model).await.unwrap();
        
        // Create test audio
        let samples = vec![0.1f32; 16000];
        let audio = AudioBuffer::mono(samples, 16000);
        
        let sentiment = detector.analyze_sentiment(&audio).await;
        assert!(sentiment.is_ok());
        
        let sentiment = sentiment.unwrap();
        assert!(sentiment.confidence >= 0.0 && sentiment.confidence <= 1.0);
        assert!(sentiment.valence >= -1.0 && sentiment.valence <= 1.0);
    }

    #[tokio::test]
    async fn test_stress_detection() {
        let config = EmotionConfig::default();
        let model = Arc::new(MockEmotionModel::new());
        
        let mut detector = EmotionDetectorImpl::new(config, model).await.unwrap();
        
        // Create test audio
        let samples = vec![0.1f32; 16000];
        let audio = AudioBuffer::mono(samples, 16000);
        
        let stress = detector.detect_stress(&audio).await;
        assert!(stress.is_ok());
        
        let stress_level = stress.unwrap();
        assert!(stress_level >= 0.0 && stress_level <= 1.0);
    }

    #[tokio::test]
    async fn test_config_update() {
        let config = EmotionConfig::default();
        let model = Arc::new(MockEmotionModel::new());
        
        let mut detector = EmotionDetectorImpl::new(config, model).await.unwrap();
        
        let mut new_config = EmotionConfig::default();
        new_config.min_confidence = 0.8;
        
        let result = detector.update_config(new_config).await;
        assert!(result.is_ok());
        
        assert_eq!(detector.get_config().min_confidence, 0.8);
    }
}