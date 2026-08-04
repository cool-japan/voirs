use super::*;

#[test]
fn test_truncate_path() {
    let path = PathBuf::from("/very/long/path/to/some/directory/file.txt");
    let truncated = truncate_path(&path, 20);
    assert!(truncated.len() <= 20);
    assert!(truncated.starts_with("..."));
}

#[test]
fn test_lr_schedulers() {
    // Test step scheduler
    let lr_step = apply_lr_scheduler("step", 0.001, 100, 100, 0.1, 1000, 0);
    assert!((lr_step - 0.0001).abs() < 1e-6); // Should be 0.001 * 0.1^1

    // Test exponential scheduler
    let lr_exp = apply_lr_scheduler("exponential", 0.001, 10, 100, 0.95, 1000, 0);
    assert!((lr_exp - (0.001 * 0.95_f64.powf(10.0))).abs() < 1e-9);

    // Test cosine scheduler
    let lr_cos = apply_lr_scheduler("cosine", 0.001, 500, 100, 0.1, 1000, 0);
    assert!(lr_cos > 0.0 && lr_cos <= 0.001);

    // Test onecycle scheduler
    let lr_one = apply_lr_scheduler("onecycle", 0.001, 250, 100, 0.1, 1000, 0);
    assert!(lr_one > 0.001); // Should be in increasing phase
}

/// The `plateau` scheduler must stay flat immediately after an
/// improvement (epochs_since_improvement == 0) and only start decaying
/// once a full `step_size`-epoch window has passed without one -- a real
/// decay driven by validation stagnation, not the "act like none" no-op
/// it used to be.
#[test]
fn test_plateau_scheduler_decays_only_after_stagnation_window() {
    let flat = apply_lr_scheduler("plateau", 0.001, 500, 10, 0.5, 1000, 0);
    assert_eq!(flat, 0.001, "no stagnation yet, LR must stay at initial_lr");

    let still_flat = apply_lr_scheduler("plateau", 0.001, 500, 10, 0.5, 1000, 9);
    assert_eq!(
        still_flat, 0.001,
        "fewer than step_size stagnant epochs, LR must stay at initial_lr"
    );

    let one_decay = apply_lr_scheduler("plateau", 0.001, 500, 10, 0.5, 1000, 10);
    assert!(
        (one_decay - 0.0005).abs() < 1e-9,
        "exactly one stagnation window elapsed, LR must be halved once, got {one_decay}"
    );

    let two_decays = apply_lr_scheduler("plateau", 0.001, 500, 10, 0.5, 1000, 20);
    assert!(
        (two_decays - 0.00025).abs() < 1e-9,
        "two stagnation windows elapsed, LR must be halved twice, got {two_decays}"
    );
}

/// An unknown scheduler name (should never reach this defensive fallback
/// through the real CLI path, since `resolve_training_config` validates
/// up front) must still behave safely rather than panicking.
#[test]
fn test_unknown_scheduler_name_falls_back_to_initial_lr() {
    let lr = apply_lr_scheduler("not-a-real-scheduler", 0.001, 50, 10, 0.5, 1000, 0);
    assert_eq!(lr, 0.001);
}

/// Regression test for the fake-loss-fallback finding: the abort threshold
/// must actually gate on real accumulated failures, not let a run continue
/// forever while quietly substituting fabricated numbers.
#[test]
fn test_should_abort_on_errors_threshold() {
    assert!(!should_abort_on_errors(0, 10), "no failures yet");
    assert!(
        !should_abort_on_errors(5, 10),
        "exactly half must not abort"
    );
    assert!(
        should_abort_on_errors(6, 10),
        "a majority of failures must abort"
    );
    assert!(
        should_abort_on_errors(1, 1),
        "a single-batch epoch must abort on its first failure"
    );
    assert!(
        !should_abort_on_errors(0, 1),
        "a single-batch epoch with zero failures must not abort"
    );
}

/// Regression test for the hardcoded-summary-loss finding: when validation
/// never ran for a checkpoint, the persisted metadata must say so honestly
/// (`null`) instead of substituting a fabricated `0.0` that looks like a
/// perfect (measured) validation loss.
#[tokio::test]
async fn test_save_checkpoint_records_missing_validation_honestly() {
    let temp = tempfile::tempdir().expect("failed to create temp dir");
    let varmap = VarMap::new();

    save_checkpoint(
        temp.path(),
        "no_val_ckpt",
        3,
        0.42,
        None,
        "DiffWave",
        &varmap,
    )
    .await
    .expect("save_checkpoint should succeed");

    let metadata_path = temp.path().join("no_val_ckpt.json");
    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&metadata_path).unwrap())
            .expect("metadata should be valid JSON");

    assert!(
        metadata["val_loss"].is_null(),
        "no validation ran this epoch, val_loss must be null, not fabricated: {metadata}"
    );
    assert_eq!(metadata["train_loss"], 0.42);
    assert_eq!(metadata["model_type"], "DiffWave");
    assert!(
        temp.path().join("no_val_ckpt.safetensors").exists(),
        "a real (if empty) safetensors file should still be written"
    );
}

/// Foundational check for resume (defect 2): a checkpoint written by
/// `save_checkpoint` must be loadable back into a *fresh* `VarMap` (i.e. one
/// simulating a freshly-constructed, not-yet-trained model, exactly the
/// scenario `--resume` needs) via candle's own `VarMap::load`. This exercises
/// the hand-built SafeTensors header/offset encoding end-to-end: if the
/// offsets or header shape encoding were ever wrong, this is where it would
/// show up, independent of any specific model architecture.
#[tokio::test]
async fn test_save_checkpoint_round_trips_through_varmap_load() {
    let temp = tempfile::tempdir().expect("failed to create temp dir");
    let device = Device::Cpu;

    // Source VarMap with several differently-shaped tensors set to known,
    // non-default values (a silently-zeroed or mismatched load would be
    // caught by the value assertions below).
    let mut source = VarMap::new();
    source
        .get(
            (2, 3),
            "layer_a.weight",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .expect("failed to register layer_a.weight");
    source
        .get(
            5,
            "layer_b.bias",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .expect("failed to register layer_b.bias");
    source
        .get(
            (4, 4, 2),
            "layer_c.weight",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .expect("failed to register layer_c.weight");

    let a_vals: Vec<f32> = (1..=6).map(|v| v as f32).collect();
    let b_vals: Vec<f32> = (1..=5).map(|v| v as f32 * 10.0).collect();
    let c_vals: Vec<f32> = (0..32).map(|v| v as f32 * 0.5).collect();

    source
        .set_one(
            "layer_a.weight",
            Tensor::from_slice(&a_vals, (2, 3), &device).expect("tensor"),
        )
        .expect("failed to set layer_a.weight");
    source
        .set_one(
            "layer_b.bias",
            Tensor::from_slice(&b_vals, 5, &device).expect("tensor"),
        )
        .expect("failed to set layer_b.bias");
    source
        .set_one(
            "layer_c.weight",
            Tensor::from_slice(&c_vals, (4, 4, 2), &device).expect("tensor"),
        )
        .expect("failed to set layer_c.weight");

    save_checkpoint(
        temp.path(),
        "roundtrip",
        3,
        0.1,
        Some(0.05),
        "Test",
        &source,
    )
    .await
    .expect("save_checkpoint should succeed");

    // Fresh target VarMap simulating a newly-constructed, untrained model
    // immediately before a `--resume` load.
    let mut target = VarMap::new();
    target
        .get(
            (2, 3),
            "layer_a.weight",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .expect("failed to register layer_a.weight");
    target
        .get(
            5,
            "layer_b.bias",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .expect("failed to register layer_b.bias");
    target
        .get(
            (4, 4, 2),
            "layer_c.weight",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .expect("failed to register layer_c.weight");

    target
        .load(temp.path().join("roundtrip.safetensors"))
        .expect("a checkpoint written by save_checkpoint must be loadable by VarMap::load");

    let get_vals = |vm: &VarMap, name: &str| -> Vec<f32> {
        let data = vm.data().lock().expect("lock should not be poisoned");
        data.get(name)
            .expect("tensor should exist after load")
            .as_tensor()
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to_vec1")
    };

    assert_eq!(get_vals(&target, "layer_a.weight"), a_vals);
    assert_eq!(get_vals(&target, "layer_b.bias"), b_vals);
    assert_eq!(get_vals(&target, "layer_c.weight"), c_vals);
}

/// Companion case: when validation *did* run, the real measured value must
/// be persisted (and the correct model type recorded, not a hardcoded
/// "DiffWave" regardless of caller).
#[tokio::test]
async fn test_save_checkpoint_records_real_validation_loss_and_model_type() {
    let temp = tempfile::tempdir().expect("failed to create temp dir");
    let varmap = VarMap::new();

    save_checkpoint(
        temp.path(),
        "val_ckpt",
        7,
        0.31,
        Some(0.19),
        "HiFiGan",
        &varmap,
    )
    .await
    .expect("save_checkpoint should succeed");

    let metadata_path = temp.path().join("val_ckpt.json");
    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&metadata_path).unwrap())
            .expect("metadata should be valid JSON");

    assert_eq!(metadata["val_loss"], 0.19);
    assert_eq!(metadata["train_loss"], 0.31);
    assert_eq!(metadata["model_type"], "HiFiGan");
}

/// Behavioral test for the real gradient-clipping mechanism (defects 4/5):
/// build a tiny but *real* autograd graph with a deliberately huge
/// residual so the true gradient norm is large and easy to reason about,
/// then verify `clip_gradients` (a) reports the real pre-clip norm, and
/// (b) actually rescales the gradients so their post-clip norm matches
/// `grad_clip`, rather than being a no-op dressed up with a fake number.
#[test]
fn test_clip_gradients_reports_real_norm_and_rescales_when_exceeded() {
    let device = Device::Cpu;
    // w starts at 10.0; loss = (w * x - y)^2 with x=1, y=0 gives a huge,
    // easy-to-predict gradient: d(loss)/dw = 2*(w*x-y)*x = 20.0 at w=10.
    let w = Var::new(10.0f32, &device).expect("failed to create Var");
    let vars = vec![w.clone()];

    let x = Tensor::new(1.0f32, &device).expect("tensor");
    let y = Tensor::new(0.0f32, &device).expect("tensor");
    let pred = (w.as_tensor() * &x).expect("mul");
    let diff = (pred - &y).expect("sub");
    let loss = diff.sqr().expect("sqr");

    let mut grads = loss.backward().expect("backward should succeed");
    let grad_clip = 1e-3;
    let norm = clip_gradients(&mut grads, &vars, grad_clip).expect("clip_gradients should succeed");

    assert!(
        (norm - 20.0).abs() < 1e-3,
        "real pre-clip gradient norm should be ~20.0 (2*w for w=10), got {norm}"
    );

    // The gradient stored for `w` must now have been rescaled down to
    // (approximately) grad_clip -- not left untouched, and not zeroed.
    let clipped_grad = grads
        .get(w.as_tensor())
        .expect("gradient for w should still be present after clipping");
    let clipped_norm = clipped_grad
        .sqr()
        .expect("sqr")
        .sum_all()
        .expect("sum_all")
        .to_vec0::<f32>()
        .expect("to_vec0")
        .sqrt() as f64;
    assert!(
        (clipped_norm - grad_clip).abs() < 1e-6,
        "post-clip gradient norm should equal grad_clip ({grad_clip}), got {clipped_norm}"
    );
}

/// When `grad_clip <= 0.0` (disabled) or the real norm is already under
/// the threshold, gradients must pass through unchanged -- clipping must
/// never silently alter updates the user didn't ask to clip.
#[test]
fn test_clip_gradients_is_noop_when_disabled_or_under_threshold() {
    let device = Device::Cpu;
    let w = Var::new(10.0f32, &device).expect("failed to create Var");
    let vars = vec![w.clone()];
    let x = Tensor::new(1.0f32, &device).expect("tensor");
    let y = Tensor::new(0.0f32, &device).expect("tensor");

    let make_loss = || {
        let pred = (w.as_tensor() * &x).expect("mul");
        let diff = (pred - &y).expect("sub");
        diff.sqr().expect("sqr")
    };

    // Disabled (grad_clip <= 0.0): the real norm is still reported, but
    // no rescaling happens.
    let loss = make_loss();
    let mut grads = loss.backward().expect("backward");
    let norm = clip_gradients(&mut grads, &vars, 0.0).expect("clip_gradients");
    assert!((norm - 20.0).abs() < 1e-3);
    let unclipped = grads
        .get(w.as_tensor())
        .expect("gradient present")
        .to_vec0::<f32>()
        .expect("to_vec0");
    assert!(
        (unclipped - 20.0).abs() < 1e-3,
        "grad_clip disabled must leave the gradient untouched, got {unclipped}"
    );

    // Under threshold: grad_clip larger than the real norm, no rescaling.
    let loss = make_loss();
    let mut grads = loss.backward().expect("backward");
    let norm = clip_gradients(&mut grads, &vars, 1000.0).expect("clip_gradients");
    assert!((norm - 20.0).abs() < 1e-3);
    let unclipped = grads
        .get(w.as_tensor())
        .expect("gradient present")
        .to_vec0::<f32>()
        .expect("to_vec0");
    assert!(
        (unclipped - 20.0).abs() < 1e-3,
        "grad_clip above the real norm must leave the gradient untouched, got {unclipped}"
    );
}

/// Resume must fail closed (typed error, no silent from-scratch fallback)
/// when the checkpoint file doesn't exist.
#[test]
fn test_resume_from_checkpoint_fails_closed_on_missing_file() {
    let temp = tempfile::tempdir().expect("failed to create temp dir");
    let mut varmap = VarMap::new();
    let missing = temp.path().join("does-not-exist.safetensors");

    let result = resume_from_checkpoint(&mut varmap, &missing, "DiffWave", true);
    assert!(result.is_err(), "a missing checkpoint must be rejected");
}

/// Resume must fail closed when the checkpoint's tensors don't match the
/// (freshly-constructed) model's VarMap -- e.g. a shape mismatch -- rather
/// than silently loading a partial/incompatible set of weights.
#[tokio::test]
async fn test_resume_from_checkpoint_fails_closed_on_shape_mismatch() {
    let temp = tempfile::tempdir().expect("failed to create temp dir");
    let device = Device::Cpu;

    // Source checkpoint has a (2, 3) tensor named "w".
    let mut source = VarMap::new();
    source
        .get(
            (2, 3),
            "w",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .expect("register w");
    save_checkpoint(temp.path(), "ckpt", 0, 0.1, None, "Test", &source)
        .await
        .expect("save_checkpoint should succeed");

    // Target VarMap registers "w" with an incompatible shape (3, 3).
    let mut target = VarMap::new();
    target
        .get(
            (3, 3),
            "w",
            candle_nn::Init::Const(0.0),
            DType::F32,
            &device,
        )
        .expect("register w");

    let checkpoint_path = temp.path().join("ckpt.safetensors");
    let result = resume_from_checkpoint(&mut target, &checkpoint_path, "Test", true);
    assert!(
        result.is_err(),
        "a shape-mismatched checkpoint must be rejected, not partially loaded"
    );
}

/// End-to-end resume: weights come back correctly, and the epoch to
/// resume at is recovered from the checkpoint's JSON sidecar (the epoch
/// *after* the one the checkpoint recorded).
#[tokio::test]
async fn test_resume_from_checkpoint_restores_weights_and_next_epoch() {
    let temp = tempfile::tempdir().expect("failed to create temp dir");
    let device = Device::Cpu;

    let mut source = VarMap::new();
    source
        .get(3, "w", candle_nn::Init::Const(0.0), DType::F32, &device)
        .expect("register w");
    source
        .set_one(
            "w",
            Tensor::from_slice(&[1.0f32, 2.0, 3.0], 3, &device).expect("tensor"),
        )
        .expect("set w");

    // Checkpoint recorded as having completed epoch 4 (0-indexed).
    save_checkpoint(temp.path(), "ckpt", 4, 0.1, Some(0.05), "Test", &source)
        .await
        .expect("save_checkpoint should succeed");

    let mut target = VarMap::new();
    target
        .get(3, "w", candle_nn::Init::Const(0.0), DType::F32, &device)
        .expect("register w");

    let checkpoint_path = temp.path().join("ckpt.safetensors");
    let start_epoch = resume_from_checkpoint(&mut target, &checkpoint_path, "Test", true)
        .expect("resume should succeed");

    assert_eq!(
        start_epoch, 5,
        "resume must continue at the epoch after the one the checkpoint recorded"
    );

    let restored: Vec<f32> = {
        let data = target.data().lock().expect("lock should not be poisoned");
        data.get("w")
            .expect("w should exist")
            .as_tensor()
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to_vec1")
    };
    assert_eq!(restored, vec![1.0, 2.0, 3.0]);
}

/// When the checkpoint has no JSON sidecar (or it's unreadable), resume
/// must still succeed (weights are independent of the sidecar) but
/// honestly restart the epoch counter at 0 rather than guessing.
#[tokio::test]
async fn test_resume_from_checkpoint_restarts_epoch_when_sidecar_missing() {
    let temp = tempfile::tempdir().expect("failed to create temp dir");
    let device = Device::Cpu;

    let mut source = VarMap::new();
    source
        .get(2, "w", candle_nn::Init::Const(0.0), DType::F32, &device)
        .expect("register w");

    save_checkpoint(temp.path(), "ckpt", 9, 0.1, None, "Test", &source)
        .await
        .expect("save_checkpoint should succeed");

    // Delete the sidecar JSON to simulate a checkpoint moved/copied
    // without it.
    std::fs::remove_file(temp.path().join("ckpt.json")).expect("failed to remove sidecar");

    let mut target = VarMap::new();
    target
        .get(2, "w", candle_nn::Init::Const(0.0), DType::F32, &device)
        .expect("register w");

    let checkpoint_path = temp.path().join("ckpt.safetensors");
    let start_epoch = resume_from_checkpoint(&mut target, &checkpoint_path, "Test", true)
        .expect("resume should succeed even without a sidecar");

    assert_eq!(
        start_epoch, 0,
        "without a sidecar, the epoch counter must honestly restart at 0"
    );
}

/// Critical verification for resume (defect 2): a `Tensor` handle obtained
/// from a `VarBuilder`-backed `VarMap` *before* `VarMap::load` runs must
/// observe the loaded values afterward. This is exactly the real resume
/// ordering: a model's layers grab `Tensor` clones from `vb.get(...)` at
/// *construction* time (e.g. `candle_nn::Linear` stores the `Tensor` it got
/// back, not a `Var`), and only then does `resume_from_checkpoint` call
/// `varmap.load(...)`. If a `Tensor` handle obtained before the load did NOT
/// see the update, resume would silently load real weights into storage the
/// model never reads from again -- a no-op dressed up as success that every
/// other resume test above would still pass, since they only inspect
/// `varmap.data()` directly rather than a held-before-load `Tensor`.
#[tokio::test]
async fn test_varmap_load_is_visible_through_a_tensor_handle_grabbed_before_load() {
    let temp = tempfile::tempdir().expect("failed to create temp dir");
    let device = Device::Cpu;

    // Simulate model construction: grab a `Tensor` handle from the
    // `VarBuilder` *before* any resume load happens.
    let mut vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
    let held: Tensor = vb.get(3, "w").expect("vb.get should succeed");

    // Sanity check: the freshly-initialized value is not yet the
    // checkpoint's value, so the assertion below is actually meaningful.
    let initial: Vec<f32> = held.flatten_all().unwrap().to_vec1().unwrap();
    assert_ne!(initial, vec![7.0f32, 8.0, 9.0]);

    // Write a checkpoint with known values via a separate source VarMap.
    let mut source = VarMap::new();
    source
        .get(3, "w", candle_nn::Init::Const(0.0), DType::F32, &device)
        .expect("register w");
    source
        .set_one(
            "w",
            Tensor::from_slice(&[7.0f32, 8.0, 9.0], 3, &device).expect("tensor"),
        )
        .expect("set w");
    save_checkpoint(temp.path(), "ckpt", 0, 0.1, None, "Test", &source)
        .await
        .expect("save_checkpoint should succeed");

    // Resume: load into `vm` -- the SAME VarMap `held` was grabbed from --
    // *after* `held` was already obtained, exactly like real model resume.
    let checkpoint_path = temp.path().join("ckpt.safetensors");
    resume_from_checkpoint(&mut vm, &checkpoint_path, "Test", true).expect("resume should succeed");

    // The pre-grabbed Tensor handle must observe the loaded values.
    let after: Vec<f32> = held.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(
        after,
        vec![7.0, 8.0, 9.0],
        "a Tensor handle obtained before VarMap::load must see the loaded \
         values afterward -- otherwise resume silently loads weights the \
         model never reads"
    );
}
