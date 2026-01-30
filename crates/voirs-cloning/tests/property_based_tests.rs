//! Property-based tests for voirs-cloning using proptest
//!
//! These tests use property-based testing to validate edge cases and invariants
//! across a wide range of inputs.

use proptest::prelude::*;
use voirs_cloning::embedding::{EmbeddingMetadata, SpeakerEmbedding, VoiceQuality};

/// Generate a valid embedding vector with dimension between 64 and 1024
fn arb_embedding_vector() -> impl Strategy<Value = Vec<f32>> {
    (64usize..=1024).prop_flat_map(|dim| {
        prop::collection::vec(
            (-10.0f32..10.0f32).prop_map(|x| {
                if x.is_nan() || x.is_infinite() {
                    0.0
                } else {
                    x
                }
            }),
            dim,
        )
    })
}

/// Generate a valid embedding
fn arb_speaker_embedding() -> impl Strategy<Value = SpeakerEmbedding> {
    arb_embedding_vector().prop_map(SpeakerEmbedding::new)
}

/// Generate a pair of embeddings with the same dimension
fn arb_embedding_pair() -> impl Strategy<Value = (SpeakerEmbedding, SpeakerEmbedding)> {
    (64usize..=1024).prop_flat_map(|dim| {
        let vec1 = prop::collection::vec(-10.0f32..10.0f32, dim);
        let vec2 = prop::collection::vec(-10.0f32..10.0f32, dim);
        (vec1, vec2).prop_map(|(v1, v2)| (SpeakerEmbedding::new(v1), SpeakerEmbedding::new(v2)))
    })
}

proptest! {
    /// Property: Similarity of an embedding with itself should always be 1.0 (or very close)
    #[test]
    fn prop_similarity_reflexive(embedding in arb_speaker_embedding()) {
        let sim = embedding.similarity(&embedding);
        prop_assert!(sim >= 0.99 && sim <= 1.01, "Self-similarity should be ~1.0, got {}", sim);
    }

    /// Property: Similarity should be symmetric: sim(a, b) == sim(b, a)
    #[test]
    fn prop_similarity_symmetric((emb1, emb2) in arb_embedding_pair()) {
        let sim_ab = emb1.similarity(&emb2);
        let sim_ba = emb2.similarity(&emb1);
        prop_assert!(
            (sim_ab - sim_ba).abs() < 1e-5,
            "Similarity should be symmetric: {} vs {}",
            sim_ab,
            sim_ba
        );
    }

    /// Property: Similarity should always be in range [-1, 1] (cosine similarity can be negative)
    #[test]
    fn prop_similarity_bounded((emb1, emb2) in arb_embedding_pair()) {
        let sim = emb1.similarity(&emb2);
        prop_assert!(
            sim >= -1.01 && sim <= 1.01 || sim.is_nan(),
            "Cosine similarity should be in [-1,1], got {}",
            sim
        );
    }

    /// Property: Distance should be non-negative
    #[test]
    fn prop_distance_non_negative((emb1, emb2) in arb_embedding_pair()) {
        let dist = emb1.distance(&emb2);
        prop_assert!(dist >= 0.0, "Distance should be non-negative, got {}", dist);
    }

    /// Property: Distance of an embedding to itself should be 0
    #[test]
    fn prop_distance_reflexive(embedding in arb_speaker_embedding()) {
        let dist = embedding.distance(&embedding);
        prop_assert!(dist < 1e-5, "Self-distance should be ~0, got {}", dist);
    }

    /// Property: Distance should be symmetric: dist(a, b) == dist(b, a)
    #[test]
    fn prop_distance_symmetric((emb1, emb2) in arb_embedding_pair()) {
        let dist_ab = emb1.distance(&emb2);
        let dist_ba = emb2.distance(&emb1);
        prop_assert!(
            (dist_ab - dist_ba).abs() < 1e-5,
            "Distance should be symmetric: {} vs {}",
            dist_ab,
            dist_ba
        );
    }

    /// Property: Triangle inequality: dist(a, c) <= dist(a, b) + dist(b, c)
    #[test]
    fn prop_distance_triangle_inequality(
        emb_a in arb_speaker_embedding(),
        emb_b in arb_speaker_embedding(),
        emb_c in arb_speaker_embedding()
    ) {
        // Ensure all embeddings have the same dimension
        if emb_a.dimension == emb_b.dimension && emb_b.dimension == emb_c.dimension {
            let dist_ac = emb_a.distance(&emb_c);
            let dist_ab = emb_a.distance(&emb_b);
            let dist_bc = emb_b.distance(&emb_c);

            prop_assert!(
                dist_ac <= dist_ab + dist_bc + 1e-5,
                "Triangle inequality violated: {} > {} + {}",
                dist_ac,
                dist_ab,
                dist_bc
            );
        }
    }

    /// Property: Normalized embedding should have norm ~1.0
    #[test]
    fn prop_normalize_unit_norm(vector in arb_embedding_vector()) {
        let mut embedding = SpeakerEmbedding::new(vector);
        embedding.normalize();

        // Calculate norm manually
        let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!(
            (norm - 1.0).abs() < 1e-5,
            "Normalized embedding should have norm ~1.0, got {}",
            norm
        );
    }

    /// Property: Normalizing twice should give the same result
    #[test]
    fn prop_normalize_idempotent(vector in arb_embedding_vector()) {
        let mut emb1 = SpeakerEmbedding::new(vector.clone());
        let mut emb2 = SpeakerEmbedding::new(vector);

        emb1.normalize();
        emb2.normalize();
        emb2.normalize(); // Normalize twice

        let diff: f32 = emb1.vector
            .iter()
            .zip(&emb2.vector)
            .map(|(a, b)| (a - b).abs())
            .sum();

        prop_assert!(
            diff < 1e-5,
            "Double normalization should be idempotent, diff: {}",
            diff
        );
    }

    /// Property: is_valid should detect NaN and infinite values
    #[test]
    fn prop_is_valid_detects_invalid(
        dim in 64usize..=1024,
        invalid_idx in 0usize..64
    ) {
        // Create valid embedding
        let mut vector = vec![1.0; dim];

        // Make one value invalid
        if invalid_idx < dim {
            vector[invalid_idx] = f32::NAN;
            let embedding = SpeakerEmbedding::new(vector.clone());
            prop_assert!(!embedding.is_valid(), "Should detect NaN");

            vector[invalid_idx] = f32::INFINITY;
            let embedding = SpeakerEmbedding::new(vector);
            prop_assert!(!embedding.is_valid(), "Should detect Infinity");
        }
    }

    /// Property: quality_score should be in [0, 1]
    #[test]
    fn prop_quality_score_bounded(
        embedding in arb_speaker_embedding(),
        confidence in 0.0f32..=1.0f32
    ) {
        let mut emb = embedding;
        emb.confidence = confidence;
        let score = emb.quality_score();
        prop_assert!(
            score >= 0.0 && score <= 1.0,
            "Quality score should be in [0,1], got {}",
            score
        );
    }

    /// Property: Embeddings with different dimensions should return 0 similarity
    #[test]
    fn prop_different_dimensions_zero_similarity(
        dim1 in 64usize..=512,
        dim2 in 513usize..=1024
    ) {
        let emb1 = SpeakerEmbedding::new(vec![1.0; dim1]);
        let emb2 = SpeakerEmbedding::new(vec![1.0; dim2]);

        let sim = emb1.similarity(&emb2);
        prop_assert_eq!(sim, 0.0, "Different dimensions should give 0 similarity");
    }

    /// Property: Embeddings with different dimensions should return infinity distance
    #[test]
    fn prop_different_dimensions_infinite_distance(
        dim1 in 64usize..=512,
        dim2 in 513usize..=1024
    ) {
        let emb1 = SpeakerEmbedding::new(vec![1.0; dim1]);
        let emb2 = SpeakerEmbedding::new(vec![1.0; dim2]);

        let dist = emb1.distance(&emb2);
        prop_assert_eq!(dist, f32::INFINITY, "Different dimensions should give infinite distance");
    }

    /// Property: Scaling an embedding shouldn't change similarity (cosine similarity is scale-invariant)
    #[test]
    fn prop_similarity_scale_invariant(
        (emb1, emb2) in arb_embedding_pair(),
        scale in 0.1f32..10.0f32
    ) {
        let sim_before = emb1.similarity(&emb2);

        // Scale emb1
        let scaled_vector: Vec<f32> = emb1.vector.iter().map(|x| x * scale).collect();
        let emb1_scaled = SpeakerEmbedding::new(scaled_vector);

        let sim_after = emb1_scaled.similarity(&emb2);

        prop_assert!(
            (sim_before - sim_after).abs() < 0.01,
            "Similarity should be scale-invariant: {} vs {}",
            sim_before,
            sim_after
        );
    }
}

proptest! {
    #[test]
    fn prop_voice_quality_bounded(
        f0_mean in 50.0f32..500.0f32,
        f0_std in 0.0f32..50.0f32,
        spectral_centroid in 1000.0f32..5000.0f32,
        spectral_bandwidth in 500.0f32..3000.0f32,
        jitter in 0.0f32..0.1f32,
        shimmer in 0.0f32..0.1f32,
        energy_mean in 0.1f32..1.0f32,
        energy_std in 0.0f32..0.5f32
    ) {
        let quality = VoiceQuality {
            f0_mean,
            f0_std,
            spectral_centroid,
            spectral_bandwidth,
            jitter,
            shimmer,
            energy_mean,
            energy_std,
        };

        let score = quality.overall_quality();
        prop_assert!(
            score >= 0.0 && score <= 1.0,
            "Overall quality should be in [0,1], got {}",
            score
        );
    }

    #[test]
    fn prop_voice_quality_monotonic_jitter(
        f0_mean in 50.0f32..500.0f32,
        f0_std in 0.0f32..50.0f32,
        spectral_centroid in 1000.0f32..5000.0f32,
        spectral_bandwidth in 500.0f32..3000.0f32,
        jitter1 in 0.0f32..0.05f32,
        jitter2 in 0.05f32..0.1f32,
        shimmer in 0.0f32..0.05f32,
        energy_mean in 0.5f32..1.0f32,
        energy_std in 0.0f32..0.1f32
    ) {
        let quality1 = VoiceQuality {
            f0_mean,
            f0_std,
            spectral_centroid,
            spectral_bandwidth,
            jitter: jitter1,
            shimmer,
            energy_mean,
            energy_std,
        };

        let quality2 = VoiceQuality {
            f0_mean,
            f0_std,
            spectral_centroid,
            spectral_bandwidth,
            jitter: jitter2,
            shimmer,
            energy_mean,
            energy_std,
        };

        let score1 = quality1.overall_quality();
        let score2 = quality2.overall_quality();

        prop_assert!(
            score1 >= score2,
            "Lower jitter should give higher quality: {} vs {}",
            score1,
            score2
        );
    }
}

// Property tests for SIMD operations
mod simd_property_tests {
    use super::*;
    use voirs_cloning::embedding::simd_ops::*;

    proptest! {
        /// Property: SIMD cosine similarity should match manual calculation
        #[test]
        fn prop_simd_cosine_similarity_correct(
            dim in 8usize..=256,
            seed1 in 0u64..10000,
            seed2 in 0u64..10000
        ) {
            // Use fastrand with seed for reproducible random vectors
            fastrand::seed(seed1);
            let a: Vec<f32> = (0..dim).map(|_| fastrand::f32() * 2.0 - 1.0).collect();

            fastrand::seed(seed2);
            let b: Vec<f32> = (0..dim).map(|_| fastrand::f32() * 2.0 - 1.0).collect();

            let simd_result = simd_cosine_similarity(&a, &b);

            // Manual calculation
            let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let manual_result = dot / (norm_a * norm_b);

            prop_assert!(
                (simd_result - manual_result).abs() < 1e-5,
                "SIMD result {} should match manual calculation {}",
                simd_result,
                manual_result
            );
        }

        /// Property: SIMD Euclidean distance should match manual calculation
        #[test]
        fn prop_simd_euclidean_distance_correct(
            dim in 8usize..=256,
            seed1 in 0u64..10000,
            seed2 in 0u64..10000
        ) {
            fastrand::seed(seed1);
            let a: Vec<f32> = (0..dim).map(|_| fastrand::f32() * 2.0 - 1.0).collect();

            fastrand::seed(seed2);
            let b: Vec<f32> = (0..dim).map(|_| fastrand::f32() * 2.0 - 1.0).collect();

            let simd_result = simd_euclidean_distance(&a, &b);

            // Manual calculation
            let manual_result: f32 = a.iter()
                .zip(&b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt();

            prop_assert!(
                (simd_result - manual_result).abs() < 1e-5,
                "SIMD result {} should match manual calculation {}",
                simd_result,
                manual_result
            );
        }

        /// Property: SIMD normalized vector should have norm ~1.0
        #[test]
        fn prop_simd_normalize_unit_norm(
            dim in 8usize..=256,
            seed in 0u64..10000
        ) {
            fastrand::seed(seed);
            let mut vec: Vec<f32> = (0..dim).map(|_| fastrand::f32() * 2.0 - 1.0).collect();

            simd_normalize_inplace(&mut vec);

            let norm = simd_l2_norm(&vec);
            prop_assert!(
                (norm - 1.0).abs() < 1e-5,
                "Normalized vector should have norm ~1.0, got {}",
                norm
            );
        }

        /// Property: SIMD dot product should be commutative
        #[test]
        fn prop_simd_dot_product_commutative(
            dim in 8usize..=256,
            seed1 in 0u64..10000,
            seed2 in 0u64..10000
        ) {
            fastrand::seed(seed1);
            let a: Vec<f32> = (0..dim).map(|_| fastrand::f32() * 2.0 - 1.0).collect();

            fastrand::seed(seed2);
            let b: Vec<f32> = (0..dim).map(|_| fastrand::f32() * 2.0 - 1.0).collect();

            let dot_ab = simd_dot_product(&a, &b);
            let dot_ba = simd_dot_product(&b, &a);

            prop_assert!(
                (dot_ab - dot_ba).abs() < 1e-5,
                "Dot product should be commutative: {} vs {}",
                dot_ab,
                dot_ba
            );
        }

        /// Property: SIMD weighted average with equal weights should equal simple average
        #[test]
        fn prop_simd_weighted_average_equal_weights(
            dim in 8usize..=64,
            n_embeddings in 2usize..=5,
            seed in 0u64..10000
        ) {
            fastrand::seed(seed);

            let embeddings: Vec<Vec<f32>> = (0..n_embeddings)
                .map(|_| (0..dim).map(|_| fastrand::f32() * 2.0 - 1.0).collect())
                .collect();

            let weight = 1.0 / n_embeddings as f32;
            let weights = vec![weight; n_embeddings];

            let result = simd_weighted_average(&embeddings, &weights);

            // Manual average
            let mut manual_avg = vec![0.0f32; dim];
            for emb in &embeddings {
                for (i, &val) in emb.iter().enumerate() {
                    manual_avg[i] += val / n_embeddings as f32;
                }
            }

            for (simd_val, &manual_val) in result.iter().zip(&manual_avg) {
                prop_assert!(
                    (simd_val - manual_val).abs() < 1e-4,
                    "SIMD weighted average should match manual: {} vs {}",
                    simd_val,
                    manual_val
                );
            }
        }
    }
}
