//! Regression tests for the single-builder invariant.
//!
//! The SDK once shipped two different structs both named `VoirsPipelineBuilder`:
//! the crate-root one always produced stub components, while the `pipeline::` one
//! wired real components. Which one a caller got depended entirely on the import
//! path they happened to use. These tests pin the invariant that there is now
//! exactly ONE builder type reachable from every path.

use voirs_sdk::prelude::*;

/// Every import path must name the same type. This is a compile-time assertion:
/// the assignments only typecheck if all paths resolve to one struct.
#[test]
fn all_import_paths_name_the_same_builder() {
    fn _same_type(b: voirs_sdk::VoirsPipelineBuilder) {
        let a: voirs_sdk::builder::VoirsPipelineBuilder = b;
        let c: voirs_sdk::pipeline::VoirsPipelineBuilder = a;
        let d: voirs_sdk::prelude::VoirsPipelineBuilder = c;
        let _e: voirs_sdk::builder::builder_impl::VoirsPipelineBuilder = d;
    }

    fn _builder_method_returns_same_type() {
        let _b: voirs_sdk::VoirsPipelineBuilder = voirs_sdk::VoirsPipeline::builder();
    }

    // The type names must also agree at runtime.
    assert_eq!(
        std::any::type_name::<voirs_sdk::VoirsPipelineBuilder>(),
        std::any::type_name::<voirs_sdk::pipeline::VoirsPipelineBuilder>(),
    );
    assert_eq!(
        std::any::type_name::<voirs_sdk::builder::VoirsPipelineBuilder>(),
        std::any::type_name::<voirs_sdk::prelude::VoirsPipelineBuilder>(),
    );
}

/// The default build path must not silently produce stub components.
///
/// Without real model weights and without an explicit `with_test_mode(true)`,
/// building must fail with a typed error rather than returning a pipeline that
/// emits a canned tone.
#[tokio::test]
async fn default_build_fails_closed_without_model_weights() {
    let cache_dir = tempfile::tempdir().expect("temp dir");

    let result = voirs_sdk::VoirsPipelineBuilder::new()
        .with_validation(false)
        .with_auto_download(false)
        .with_cache_dir(cache_dir.path())
        .build()
        .await;

    assert!(
        result.is_err(),
        "the default builder must not succeed without real model weights"
    );
}

/// The same must hold via `VoirsPipeline::builder()` and via the prelude, since
/// they are the same builder.
#[tokio::test]
async fn every_entry_point_fails_closed_identically() {
    let cache_dir = tempfile::tempdir().expect("temp dir");

    let via_pipeline = VoirsPipeline::builder()
        .with_validation(false)
        .with_auto_download(false)
        .with_cache_dir(cache_dir.path())
        .build()
        .await;

    let via_prelude = VoirsPipelineBuilder::new()
        .with_validation(false)
        .with_auto_download(false)
        .with_cache_dir(cache_dir.path())
        .build()
        .await;

    assert!(via_pipeline.is_err());
    assert!(via_prelude.is_err());
}

/// Explicit test mode is the ONLY way to obtain stub components, and it must be
/// opt-in rather than inferred from the build profile.
#[tokio::test]
async fn stub_components_require_explicit_opt_in() {
    let cache_dir = tempfile::tempdir().expect("temp dir");

    let pipeline = VoirsPipelineBuilder::new()
        .with_validation(false)
        .with_cache_dir(cache_dir.path())
        .with_test_mode(true)
        .build()
        .await
        .expect("explicit test mode must build");

    // Stub synthesis still has to depend on its input.
    let short = pipeline.synthesize("Hi").await.expect("synthesis");
    let long = pipeline
        .synthesize("Hi there, this is a much longer sentence to synthesize.")
        .await
        .expect("synthesis");

    assert!(!short.is_empty());
    assert!(
        long.len() > short.len(),
        "output must vary with input ({} vs {} samples)",
        long.len(),
        short.len()
    );
}
