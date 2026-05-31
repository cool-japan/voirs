use crate::traits::{AgeRange, Gender, SpeakerCharacteristics};
use crate::RecognitionError;
use std::collections::HashMap;
use voirs_sdk::AudioBuffer;

use super::analyzer::SpeakerAnalyzer;

/// Speaker segment with timing and identification
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerSegment {
    /// Speaker ID (assigned during diarization)
    pub speaker_id: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Speaker characteristics for this segment
    pub characteristics: SpeakerCharacteristics,
    /// Confidence score for speaker assignment
    pub confidence: f32,
}

/// Complete diarization result
#[derive(Debug, Clone)]
pub struct SpeakerDiarizationResult {
    /// List of speaker segments
    pub segments: Vec<SpeakerSegment>,
    /// Number of unique speakers detected
    pub num_speakers: usize,
    /// Overall diarization confidence
    pub overall_confidence: f32,
    /// Speaker embeddings for each unique speaker
    pub speaker_embeddings: HashMap<String, SpeakerEmbedding>,
}

/// Speaker embedding representation
#[derive(Debug, Clone)]
pub struct SpeakerEmbedding {
    /// Feature vector representing the speaker
    pub features: Vec<f32>,
    /// Number of segments used to create this embedding
    pub segment_count: usize,
    /// Average confidence of segments
    pub average_confidence: f32,
}

/// Speaker change detection result
#[derive(Debug, Clone)]
pub struct SpeakerChangePoint {
    /// Time of speaker change in seconds
    pub time: f32,
    /// Confidence of the change detection
    pub confidence: f32,
    /// Previous speaker characteristics
    pub previous_speaker: Option<SpeakerCharacteristics>,
    /// New speaker characteristics
    pub new_speaker: Option<SpeakerCharacteristics>,
}

/// Speaker diarization analyzer for multi-speaker identification
pub struct SpeakerDiarizer {
    /// Base speaker analyzer
    pub(super) speaker_analyzer: SpeakerAnalyzer,
    /// Window size for speaker embedding extraction (seconds)
    pub(super) window_size: f32,
    /// Overlap between windows (seconds)
    pub(super) window_overlap: f32,
    /// Similarity threshold for speaker clustering
    pub(super) similarity_threshold: f32,
    /// Minimum segment duration (seconds)
    pub(super) min_segment_duration: f32,
}

/// Window embedding for speaker diarization
#[derive(Debug, Clone)]
struct WindowEmbedding {
    /// Feature vector for the window
    features: Vec<f32>,
    /// Start time of the window
    start_time: f32,
    /// End time of the window
    end_time: f32,
    /// Speaker characteristics for this window
    characteristics: SpeakerCharacteristics,
    /// Confidence score for this window
    confidence: f32,
}

/// Speaker cluster for grouping similar embeddings
#[derive(Debug, Clone)]
pub(super) struct SpeakerCluster {
    /// Unique identifier for the cluster/speaker
    pub(super) id: String,
    /// Centroid (average) feature vector
    pub(super) centroid: Vec<f32>,
    /// Indices of embeddings belonging to this cluster
    pub(super) member_indices: Vec<usize>,
}

impl SpeakerDiarizer {
    /// Create a new speaker diarizer
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying speaker analyzer cannot be created.
    pub async fn new() -> Result<Self, RecognitionError> {
        let speaker_analyzer = SpeakerAnalyzer::new().await?;

        Ok(Self {
            speaker_analyzer,
            window_size: 3.0,          // 3 second windows
            window_overlap: 1.5,       // 50% overlap
            similarity_threshold: 0.8, // High similarity threshold
            min_segment_duration: 0.5, // Minimum 0.5 second segments
        })
    }

    /// Create a speaker diarizer with custom parameters
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of analysis windows in seconds
    /// * `window_overlap` - Overlap between windows in seconds
    /// * `similarity_threshold` - Threshold for speaker clustering (0.0-1.0)
    /// * `min_segment_duration` - Minimum segment duration in seconds
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying speaker analyzer cannot be created.
    pub async fn with_config(
        window_size: f32,
        window_overlap: f32,
        similarity_threshold: f32,
        min_segment_duration: f32,
    ) -> Result<Self, RecognitionError> {
        let speaker_analyzer = SpeakerAnalyzer::new().await?;

        Ok(Self {
            speaker_analyzer,
            window_size,
            window_overlap,
            similarity_threshold,
            min_segment_duration,
        })
    }

    /// Perform complete speaker diarization on audio
    ///
    /// # Arguments
    ///
    /// * `audio` - Input audio buffer
    ///
    /// # Returns
    ///
    /// Complete diarization result with speaker segments and embeddings
    ///
    /// # Errors
    ///
    /// Returns an error if speaker analysis fails or if audio is too short.
    pub async fn diarize(
        &self,
        audio: &AudioBuffer,
    ) -> Result<SpeakerDiarizationResult, RecognitionError> {
        // Step 1: Extract speaker embeddings from overlapping windows
        let embeddings = self.extract_speaker_embeddings(audio).await?;

        // Step 2: Cluster embeddings to identify speakers
        let clusters = self.cluster_speakers(&embeddings)?;

        // Step 3: Create speaker segments
        let segments = self
            .create_speaker_segments(&embeddings, &clusters, audio)
            .await?;

        // Step 4: Merge adjacent segments from same speaker
        let merged_segments = self.merge_adjacent_segments(segments);

        // Step 5: Calculate overall statistics
        let num_speakers = clusters.len();
        let overall_confidence = Self::calculate_overall_confidence(&merged_segments);

        // Step 6: Create speaker embeddings map
        let speaker_embeddings = Self::create_speaker_embeddings_map(&merged_segments, &clusters);

        Ok(SpeakerDiarizationResult {
            segments: merged_segments,
            num_speakers,
            overall_confidence,
            speaker_embeddings,
        })
    }

    /// Detect speaker change points in audio
    ///
    /// # Arguments
    ///
    /// * `audio` - Input audio buffer
    ///
    /// # Returns
    ///
    /// List of detected speaker change points
    ///
    /// # Errors
    ///
    /// Returns an error if speaker analysis fails.
    pub async fn detect_speaker_changes(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Vec<SpeakerChangePoint>, RecognitionError> {
        let embeddings = self.extract_speaker_embeddings(audio).await?;
        let mut change_points = Vec::new();

        if embeddings.len() < 2 {
            return Ok(change_points);
        }

        for i in 1..embeddings.len() {
            let prev_embedding = &embeddings[i - 1];
            let curr_embedding = &embeddings[i];

            let similarity = Self::calculate_embedding_similarity(
                &prev_embedding.features,
                &curr_embedding.features,
            );

            // Detect change if similarity is below threshold
            if similarity < self.similarity_threshold {
                let change_time = curr_embedding.start_time;
                let confidence = 1.0 - similarity; // Higher confidence for lower similarity

                change_points.push(SpeakerChangePoint {
                    time: change_time,
                    confidence,
                    previous_speaker: Some(prev_embedding.characteristics.clone()),
                    new_speaker: Some(curr_embedding.characteristics.clone()),
                });
            }
        }

        Ok(change_points)
    }

    /// Extract speaker embeddings from overlapping windows
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    async fn extract_speaker_embeddings(
        &self,
        audio: &AudioBuffer,
    ) -> Result<Vec<WindowEmbedding>, RecognitionError> {
        let sample_rate = audio.sample_rate() as f32;
        let samples = audio.samples();
        let window_samples = (self.window_size * sample_rate) as usize;
        let hop_samples = ((self.window_size - self.window_overlap) * sample_rate) as usize;

        let mut embeddings = Vec::new();
        let mut pos = 0;

        while pos + window_samples <= samples.len() {
            let window_audio = AudioBuffer::new(
                samples[pos..pos + window_samples].to_vec(),
                audio.sample_rate(),
                audio.channels(),
            );

            // Analyze speaker characteristics for this window
            let characteristics = self.speaker_analyzer.analyze_speaker(&window_audio).await?;

            // Create feature vector from characteristics
            let features = Self::extract_features_from_characteristics(&characteristics);

            let start_time = pos as f32 / sample_rate;
            let end_time = (pos + window_samples) as f32 / sample_rate;

            embeddings.push(WindowEmbedding {
                features,
                start_time,
                end_time,
                characteristics,
                confidence: 0.8, // Base confidence, will be refined during clustering
            });

            pos += hop_samples;
        }

        Ok(embeddings)
    }

    /// Extract numerical features from speaker characteristics
    pub(super) fn extract_features_from_characteristics(
        characteristics: &SpeakerCharacteristics,
    ) -> Vec<f32> {
        let mut features = Vec::new();

        // F0 features
        features.push(characteristics.voice_characteristics.f0_range.0); // Min F0
        features.push(characteristics.voice_characteristics.f0_range.1); // Max F0
        features.push(
            (characteristics.voice_characteristics.f0_range.0
                + characteristics.voice_characteristics.f0_range.1)
                / 2.0,
        ); // Mean F0

        // Formant features
        features.extend_from_slice(&characteristics.voice_characteristics.formants);

        // Voice quality features
        features.push(characteristics.voice_characteristics.voice_quality.jitter);
        features.push(characteristics.voice_characteristics.voice_quality.shimmer);
        features.push(characteristics.voice_characteristics.voice_quality.hnr);

        // Gender and age encoding (one-hot-ish)
        match characteristics.gender {
            Some(Gender::Male) => {
                features.push(1.0);
                features.push(0.0);
            }
            Some(Gender::Female) => {
                features.push(0.0);
                features.push(1.0);
            }
            Some(Gender::Other) | None => {
                features.push(0.5);
                features.push(0.5);
            }
        }

        // Age encoding
        match characteristics.age_range {
            Some(AgeRange::Child) => features.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]),
            Some(AgeRange::Teen) => features.extend_from_slice(&[0.0, 1.0, 0.0, 0.0]),
            Some(AgeRange::Adult) => features.extend_from_slice(&[0.0, 0.0, 1.0, 0.0]),
            Some(AgeRange::Senior) => features.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]),
            None => features.extend_from_slice(&[0.25, 0.25, 0.25, 0.25]),
        }

        // Normalize features to [0, 1] range
        Self::normalize_features(features)
    }

    /// Normalize feature vector
    fn normalize_features(mut features: Vec<f32>) -> Vec<f32> {
        // Simple min-max normalization
        let min_val = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() > f32::EPSILON {
            for feature in &mut features {
                *feature = (*feature - min_val) / (max_val - min_val);
            }
        }

        features
    }

    /// Cluster speaker embeddings using simple k-means-like approach
    #[allow(clippy::unnecessary_wraps)]
    fn cluster_speakers(
        &self,
        embeddings: &[WindowEmbedding],
    ) -> Result<Vec<SpeakerCluster>, RecognitionError> {
        if embeddings.is_empty() {
            return Ok(Vec::new());
        }

        if embeddings.len() == 1 {
            return Ok(vec![SpeakerCluster {
                id: "Speaker_1".to_string(),
                centroid: embeddings[0].features.clone(),
                member_indices: vec![0],
            }]);
        }

        // Start with first embedding as first cluster
        let mut clusters = vec![SpeakerCluster {
            id: "Speaker_1".to_string(),
            centroid: embeddings[0].features.clone(),
            member_indices: vec![0],
        }];

        // Process remaining embeddings
        for (i, embedding) in embeddings.iter().enumerate().skip(1) {
            let mut best_similarity = 0.0;
            let mut best_cluster_idx = None;

            // Find most similar cluster
            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                let similarity =
                    Self::calculate_embedding_similarity(&embedding.features, &cluster.centroid);
                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_cluster_idx = Some(cluster_idx);
                }
            }

            // Assign to cluster if similarity is above threshold, otherwise create new cluster
            if let Some(cluster_idx) = best_cluster_idx {
                if best_similarity >= self.similarity_threshold {
                    clusters[cluster_idx].member_indices.push(i);
                    // Update centroid
                    Self::update_cluster_centroid(&mut clusters[cluster_idx], embeddings);
                } else {
                    // Create new cluster
                    clusters.push(SpeakerCluster {
                        id: format!("Speaker_{}", clusters.len() + 1),
                        centroid: embedding.features.clone(),
                        member_indices: vec![i],
                    });
                }
            }
        }

        Ok(clusters)
    }

    /// Update cluster centroid based on member embeddings
    #[allow(clippy::cast_precision_loss)]
    fn update_cluster_centroid(cluster: &mut SpeakerCluster, embeddings: &[WindowEmbedding]) {
        if cluster.member_indices.is_empty() {
            return;
        }

        let feature_dim = cluster.centroid.len();
        let mut new_centroid = vec![0.0; feature_dim];

        for &member_idx in &cluster.member_indices {
            for (i, &feature) in embeddings[member_idx].features.iter().enumerate() {
                new_centroid[i] += feature;
            }
        }

        let count = cluster.member_indices.len() as f32;
        for feature in &mut new_centroid {
            *feature /= count;
        }

        cluster.centroid = new_centroid;
    }

    /// Calculate similarity between two embedding vectors
    pub(super) fn calculate_embedding_similarity(embedding1: &[f32], embedding2: &[f32]) -> f32 {
        if embedding1.len() != embedding2.len() {
            return 0.0;
        }

        // Cosine similarity
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for (&e1, &e2) in embedding1.iter().zip(embedding2.iter()) {
            dot_product += e1 * e2;
            norm1 += e1 * e1;
            norm2 += e2 * e2;
        }

        let norm_product = (norm1 * norm2).sqrt();
        if norm_product > f32::EPSILON {
            dot_product / norm_product
        } else {
            0.0
        }
    }

    /// Create speaker segments from embeddings and clusters
    async fn create_speaker_segments(
        &self,
        embeddings: &[WindowEmbedding],
        clusters: &[SpeakerCluster],
        _audio: &AudioBuffer,
    ) -> Result<Vec<SpeakerSegment>, RecognitionError> {
        let mut segments = Vec::new();

        for cluster in clusters {
            for &embedding_idx in &cluster.member_indices {
                let embedding = &embeddings[embedding_idx];

                segments.push(SpeakerSegment {
                    speaker_id: cluster.id.clone(),
                    start_time: embedding.start_time,
                    end_time: embedding.end_time,
                    characteristics: embedding.characteristics.clone(),
                    confidence: embedding.confidence,
                });
            }
        }

        // Sort segments by start time
        segments.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(segments)
    }

    /// Merge adjacent segments from the same speaker
    pub(super) fn merge_adjacent_segments(
        &self,
        segments: Vec<SpeakerSegment>,
    ) -> Vec<SpeakerSegment> {
        if segments.is_empty() {
            return segments;
        }

        let mut merged = Vec::new();
        let mut current_segment = segments[0].clone();

        for segment in segments.into_iter().skip(1) {
            if segment.speaker_id == current_segment.speaker_id
                && (segment.start_time - current_segment.end_time).abs() < self.min_segment_duration
            {
                // Merge segments
                current_segment.end_time = segment.end_time;
                current_segment.confidence =
                    (current_segment.confidence + segment.confidence) / 2.0;
            } else {
                // Different speaker or gap too large
                if current_segment.end_time - current_segment.start_time
                    >= self.min_segment_duration
                {
                    merged.push(current_segment);
                }
                current_segment = segment;
            }
        }

        // Add the last segment
        if current_segment.end_time - current_segment.start_time >= self.min_segment_duration {
            merged.push(current_segment);
        }

        merged
    }

    /// Calculate overall diarization confidence
    #[allow(clippy::cast_precision_loss)]
    fn calculate_overall_confidence(segments: &[SpeakerSegment]) -> f32 {
        if segments.is_empty() {
            return 0.0;
        }

        let total_confidence: f32 = segments.iter().map(|s| s.confidence).sum();
        total_confidence / segments.len() as f32
    }

    /// Create speaker embeddings map for unique speakers
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn create_speaker_embeddings_map(
        segments: &[SpeakerSegment],
        clusters: &[SpeakerCluster],
    ) -> HashMap<String, SpeakerEmbedding> {
        let mut embeddings_map = HashMap::new();

        for cluster in clusters {
            let speaker_segments: Vec<&SpeakerSegment> = segments
                .iter()
                .filter(|s| s.speaker_id == cluster.id)
                .collect();

            if !speaker_segments.is_empty() {
                let segment_count = speaker_segments.len();
                let average_confidence = speaker_segments.iter().map(|s| s.confidence).sum::<f32>()
                    / segment_count as f32;

                embeddings_map.insert(
                    cluster.id.clone(),
                    SpeakerEmbedding {
                        features: cluster.centroid.clone(),
                        segment_count,
                        average_confidence,
                    },
                );
            }
        }

        embeddings_map
    }
}
