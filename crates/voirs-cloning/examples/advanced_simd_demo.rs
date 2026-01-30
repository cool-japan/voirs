//! Advanced SIMD Operations Demo
//!
//! This example demonstrates the SIMD-optimized embedding operations.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example advanced_simd_demo
//! ```

use voirs_cloning::{
    embedding::{
        simd_cosine_similarity, simd_dot_product, simd_euclidean_distance, simd_l2_norm,
        simd_normalize_inplace, simd_weighted_average,
    },
    SpeakerEmbedding, VoiceSample,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VoiRS SIMD-Optimized Operations Demo ===\n");

    // Step 1: Speaker Embeddings
    println!("Step 1: Speaker Embedding Operations...");

    let embedding1 = SpeakerEmbedding::new(vec![0.1; 256]);
    let embedding2 = SpeakerEmbedding::new(vec![0.11; 256]);

    println!("   ✓ Created speaker embeddings (256 dimensions)");

    // Step 2: SIMD-Optimized Similarity
    println!("\nStep 2: SIMD-Optimized Similarity...");

    let similarity = embedding1.similarity(&embedding2);
    let distance = embedding1.distance(&embedding2);

    println!("   ✓ Cosine similarity: {:.4} (using SIMD)", similarity);
    println!("   ✓ Euclidean distance: {:.4} (using SIMD)", distance);

    // Step 3: Normalization
    println!("\nStep 3: L2 Normalization...");

    let mut embedding = SpeakerEmbedding::new(vec![3.0, 4.0, 0.0]);
    println!("   - Before: {:?}", embedding.vector);

    embedding.normalize();
    println!("   - After: {:?}", embedding.vector);
    println!("   ✓ Normalized using SIMD");

    // Step 4: Direct SIMD API
    println!("\nStep 4: Direct SIMD API...");

    let vec_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let vec_b = vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1];

    let dot = simd_dot_product(&vec_a, &vec_b);
    let norm_a = simd_l2_norm(&vec_a);
    let sim = simd_cosine_similarity(&vec_a, &vec_b);

    println!("   ✓ Dot product: {:.4}", dot);
    println!("   ✓ L2 norm: {:.4}", norm_a);
    println!("   ✓ Cosine similarity: {:.4}", sim);

    // Step 5: Weighted Average
    println!("\nStep 5: Weighted Average...");

    let embeddings = vec![vec![1.0; 64], vec![0.5; 64], vec![0.0; 64]];

    let weights = vec![0.5, 0.3, 0.2];
    let weighted_avg = simd_weighted_average(&embeddings, &weights);

    println!("   ✓ Weighted average computed (64 dimensions)");
    println!("     Weights: {:?}", weights);

    // Step 6: Performance Test
    println!("\nStep 6: Performance Test...");

    let large_vec1 = vec![0.1; 1024];
    let large_vec2 = vec![0.11; 1024];

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = simd_cosine_similarity(&large_vec1, &large_vec2);
    }
    let duration = start.elapsed();

    println!("   ✓ 1000 operations on 1024-dim vectors");
    println!("     Total: {:?}", duration);
    println!("     Per op: {:?}", duration / 1000);

    println!("\n🎉 SIMD operations demo completed!");

    Ok(())
}
