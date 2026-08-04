//! Parameter initialization, accounting and checkpoint loading for VITS2 modules.
//!
//! Every VITS2 sub-module (generator, text encoder, ...) keeps its learnable
//! parameters in a [`candle_nn::VarMap`]. This module provides the three
//! operations that all of them share:
//!
//! * [`seed_varmap`] — deterministic, reproducible initialization of every
//!   registered variable. Values are drawn from a per-tensor RNG seeded from the
//!   caller supplied seed **and the tensor name**, so the result does not depend
//!   on `HashMap` iteration order and is bit-identical across runs and platforms.
//! * [`count_parameters`] — the *real* number of scalars registered in the
//!   `VarMap`. No architecture-derived estimates: what is reported is what the
//!   model actually holds.
//! * [`load_safetensors_into_varmap`] — real checkpoint loading from a
//!   SafeTensors file, including PyTorch `weight_norm` (`weight_g` / `weight_v`)
//!   fusion and trailing-singleton shape adaptation (so that `Conv1d(k=1)`
//!   checkpoints can populate `Linear` layers).

use crate::{Result, VocoderError};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use std::collections::HashMap;
use std::path::Path;

/// 64-bit FNV-1a hash.
///
/// Used to derive a stable per-tensor RNG seed from the tensor name. Chosen over
/// `std::collections::hash_map::DefaultHasher` because the latter is explicitly
/// not guaranteed to be stable across Rust releases, which would silently break
/// initialization reproducibility.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Classification of a parameter tensor, used to pick its initializer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    /// Bias vectors: initialized to zero.
    Bias,
    /// Normalization scales (`*norm*.weight`): initialized to one.
    NormScale,
    /// Lookup tables (embeddings, relative-position embeddings): fan-in is the
    /// embedding width, i.e. the last dimension.
    Lookup,
    /// Everything else (conv / linear kernels): fan-in is the product of all
    /// dimensions except the first, matching PyTorch's
    /// `_calculate_fan_in_and_fan_out` for `Conv*d` / `Linear` weights.
    Kernel,
}

/// Decide how a variable should be initialized based on its name.
fn classify(name: &str) -> ParamKind {
    if name.ends_with(".bias") || name == "bias" || name.ends_with(".beta") {
        ParamKind::Bias
    } else if name.contains("norm") && (name.ends_with(".weight") || name.ends_with(".gamma")) {
        ParamKind::NormScale
    } else if name.contains("emb_rel")
        || name.contains("emb.")
        || name.ends_with("embedding.weight")
    {
        ParamKind::Lookup
    } else {
        ParamKind::Kernel
    }
}

/// Deterministically (re-)initialize every variable registered in `varmap`.
///
/// `Var::set` copies into the variable's existing storage, so layers that
/// already captured the tensor (every `Conv1d`, `Linear`, ... built from the
/// matching [`candle_nn::VarBuilder`]) observe the new values immediately.
///
/// The distribution is PyTorch's default for convolution and linear layers:
/// `U(-1/sqrt(fan_in), 1/sqrt(fan_in))` for kernels and lookup tables, zeros for
/// biases and ones for normalization scales.
///
/// # Errors
/// Returns [`VocoderError::ModelError`] if the `VarMap` mutex is poisoned, and
/// [`VocoderError::CandleError`] if a tensor could not be created or assigned.
pub fn seed_varmap(varmap: &VarMap, seed: u64, device: &Device) -> Result<()> {
    let data = varmap
        .data()
        .lock()
        .map_err(|e| VocoderError::ModelError(format!("VarMap mutex poisoned: {e}")))?;

    for (name, var) in data.iter() {
        let dims = var.shape().dims().to_vec();
        let elem_count: usize = dims.iter().product();
        if elem_count == 0 {
            continue;
        }

        let values: Vec<f32> = match classify(name) {
            ParamKind::Bias => vec![0.0; elem_count],
            ParamKind::NormScale => vec![1.0; elem_count],
            kind => {
                let fan_in = match kind {
                    ParamKind::Lookup => dims.last().copied().unwrap_or(1),
                    _ => {
                        if dims.len() > 1 {
                            dims[1..].iter().product::<usize>().max(1)
                        } else {
                            elem_count.max(1)
                        }
                    }
                };
                let bound = (1.0f64 / fan_in as f64).sqrt() as f32;
                let mut rng = fastrand::Rng::with_seed(seed ^ fnv1a64(name.as_bytes()));
                (0..elem_count)
                    .map(|_| (rng.f32() * 2.0 - 1.0) * bound)
                    .collect()
            }
        };

        let tensor = Tensor::from_vec(values, dims, device)?;
        var.set(&tensor)?;
    }

    Ok(())
}

/// Count the parameters actually registered in `varmap`.
///
/// This is a true count of allocated scalars, not an architecture-derived
/// estimate.
///
/// # Errors
/// Returns [`VocoderError::ModelError`] if the `VarMap` mutex is poisoned.
pub fn count_parameters(varmap: &VarMap) -> Result<u64> {
    let data = varmap
        .data()
        .lock()
        .map_err(|e| VocoderError::ModelError(format!("VarMap mutex poisoned: {e}")))?;
    Ok(data.values().map(|v| v.elem_count() as u64).sum())
}

/// Summary of a checkpoint load.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WeightLoadReport {
    /// Number of tensors written into the `VarMap`.
    pub loaded: usize,
    /// Checkpoint tensors that had no counterpart in the model.
    pub unmapped: usize,
    /// Checkpoint tensors whose shape did not match the model parameter.
    pub shape_mismatches: usize,
    /// Checkpoint tensors with an unsupported dtype.
    pub unsupported_dtype: usize,
    /// Model parameters that no checkpoint tensor supplied.
    ///
    /// Non-zero means part of the network kept its pseudo-random initialization,
    /// so the model is **not** fully pretrained.
    pub missing_parameters: usize,
    /// Total number of parameters registered in the model.
    pub total_parameters: usize,
}

impl WeightLoadReport {
    /// Whether every model parameter was supplied by the checkpoint.
    pub fn is_complete(&self) -> bool {
        self.missing_parameters == 0
    }
}

/// Decode a SafeTensors view into `f32` values.
fn decode_tensor(view: &safetensors::tensor::TensorView<'_>) -> Option<Vec<f32>> {
    let raw = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => Some(
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        safetensors::Dtype::F16 => Some(
            raw.chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
        ),
        safetensors::Dtype::BF16 => Some(
            raw.chunks_exact(2)
                .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
        ),
        _ => None,
    }
}

/// A decoded checkpoint entry.
struct RawTensor {
    values: Vec<f32>,
    shape: Vec<usize>,
}

/// Fuse a PyTorch `weight_norm` pair into a plain weight tensor.
///
/// `weight_v` has the full kernel shape, `weight_g` holds one scale per output
/// channel (dimension 0). The effective weight is `g * v / ||v||` where the norm
/// is taken over every dimension except dimension 0 — exactly what
/// `torch.nn.utils.weight_norm(dim=0)` computes.
fn fuse_weight_norm(g: &RawTensor, v: &RawTensor) -> Option<RawTensor> {
    let out_channels = *v.shape.first()?;
    if out_channels == 0 || !v.values.len().is_multiple_of(out_channels) {
        return None;
    }
    if g.values.len() != out_channels {
        return None;
    }
    let stride = v.values.len() / out_channels;
    let mut fused = Vec::with_capacity(v.values.len());
    for (channel, scale) in g.values.iter().enumerate().take(out_channels) {
        let slice = &v.values[channel * stride..(channel + 1) * stride];
        let norm = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        let factor = if norm > f32::EPSILON {
            scale / norm
        } else {
            0.0
        };
        fused.extend(slice.iter().map(|x| x * factor));
    }
    Some(RawTensor {
        values: fused,
        shape: v.shape.clone(),
    })
}

/// Adapt a checkpoint shape to the target shape when they only differ by
/// trailing singleton dimensions.
///
/// This lets a `Conv1d(kernel_size = 1)` checkpoint tensor `[out, in, 1]`
/// populate a `Linear` weight `[out, in]` (and vice versa), which is exactly the
/// difference between the reference VITS attention implementation (1x1
/// convolutions) and this one (linear projections).
fn adapt_shape(source: &[usize], target: &[usize]) -> Option<Vec<usize>> {
    let squeeze = |dims: &[usize]| -> Vec<usize> {
        let mut d: Vec<usize> = dims.to_vec();
        while d.len() > 1 && *d.last().unwrap_or(&0) == 1 {
            d.pop();
        }
        d
    };
    if source == target {
        return Some(target.to_vec());
    }
    if squeeze(source) == squeeze(target)
        && source.iter().product::<usize>() == target.iter().product::<usize>()
    {
        Some(target.to_vec())
    } else {
        None
    }
}

/// Load a SafeTensors checkpoint into `varmap`.
///
/// `map_name` translates a checkpoint tensor name into the internal parameter
/// name; returning `None` skips the tensor.
///
/// The loader performs real work: it decodes F32/F16/BF16 payloads, fuses
/// `weight_g`/`weight_v` weight-norm pairs, adapts trailing singleton
/// dimensions, and writes matching tensors into the `VarMap` so that all layers
/// built from it immediately use the loaded values.
///
/// # Errors
/// Returns [`VocoderError::ModelError`] if the file cannot be read or parsed, or
/// if not a single tensor could be matched to a model parameter (fail-closed: a
/// checkpoint that does not fit the model must never be silently ignored).
pub fn load_safetensors_into_varmap<F>(
    varmap: &mut VarMap,
    path: &Path,
    device: &Device,
    map_name: F,
) -> Result<WeightLoadReport>
where
    F: Fn(&str) -> Option<String>,
{
    load_safetensors_into_varmap_with_mode(varmap, path, device, map_name, LoadMode::Strict)
}

/// How thoroughly a checkpoint must cover the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadMode {
    /// Every model parameter must be supplied by the checkpoint (default).
    ///
    /// A partially-matching checkpoint is rejected, because the unmatched layers
    /// would silently keep their pseudo-random initialization while
    /// `is_pretrained()` reported success.
    #[default]
    Strict,
    /// Accept a partial load; the caller is responsible for interpreting
    /// [`WeightLoadReport::missing_parameters`].
    ///
    /// Intended for adapting checkpoints layer by layer, never for inference.
    Partial,
}

/// Load a SafeTensors checkpoint into `varmap` with an explicit coverage policy.
///
/// See [`load_safetensors_into_varmap`]; this variant additionally lets the
/// caller accept a partial load.
///
/// # Errors
/// Returns [`VocoderError::ModelError`] if the file cannot be read or parsed, if
/// no tensor matched a model parameter, or — under [`LoadMode::Strict`] — if any
/// model parameter was left unset by the checkpoint.
pub fn load_safetensors_into_varmap_with_mode<F>(
    varmap: &mut VarMap,
    path: &Path,
    device: &Device,
    map_name: F,
    mode: LoadMode,
) -> Result<WeightLoadReport>
where
    F: Fn(&str) -> Option<String>,
{
    let data = std::fs::read(path).map_err(|e| {
        VocoderError::ModelError(format!("Failed to read VITS2 weights file {path:?}: {e}"))
    })?;
    let st = safetensors::SafeTensors::deserialize(&data).map_err(|e| {
        VocoderError::ModelError(format!("Failed to parse SafeTensors file {path:?}: {e}"))
    })?;

    let mut report = WeightLoadReport::default();

    // Decode everything first so that weight-norm pairs can be fused.
    let mut decoded: HashMap<String, RawTensor> = HashMap::new();
    for (name, view) in st.tensors() {
        match decode_tensor(&view) {
            Some(values) => {
                decoded.insert(
                    name,
                    RawTensor {
                        values,
                        shape: view.shape().to_vec(),
                    },
                );
            }
            None => {
                tracing::warn!("VITS2: skipping tensor {name} — unsupported dtype");
                report.unsupported_dtype += 1;
            }
        }
    }

    // Fuse weight_g / weight_v pairs into plain `.weight` entries.
    let wn_bases: Vec<String> = decoded
        .keys()
        .filter_map(|k| k.strip_suffix(".weight_v").map(|b| b.to_string()))
        .collect();
    for base in wn_bases {
        let v_name = format!("{base}.weight_v");
        let g_name = format!("{base}.weight_g");
        let fused = match (decoded.get(&g_name), decoded.get(&v_name)) {
            (Some(g), Some(v)) => fuse_weight_norm(g, v),
            _ => None,
        };
        decoded.remove(&v_name);
        decoded.remove(&g_name);
        if let Some(fused) = fused {
            decoded.insert(format!("{base}.weight"), fused);
        } else {
            tracing::warn!("VITS2: could not fuse weight-norm pair for {base}");
            report.unmapped += 1;
        }
    }

    // Snapshot the target shapes so the mutex is not held while writing.
    let targets: HashMap<String, Vec<usize>> = {
        let map_data = varmap
            .data()
            .lock()
            .map_err(|e| VocoderError::ModelError(format!("VarMap mutex poisoned: {e}")))?;
        map_data
            .iter()
            .map(|(k, v)| (k.clone(), v.shape().dims().to_vec()))
            .collect()
    };

    // Deterministic ordering keeps log output and the report reproducible.
    let mut entries: Vec<(String, RawTensor)> = decoded.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    report.total_parameters = targets.len();
    let mut filled: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (external_name, raw) in entries {
        let Some(internal_name) = map_name(&external_name) else {
            report.unmapped += 1;
            continue;
        };
        let Some(target_shape) = targets.get(&internal_name) else {
            tracing::debug!(
                "VITS2: '{external_name}' → '{internal_name}' is not a model parameter"
            );
            report.unmapped += 1;
            continue;
        };
        let Some(shape) = adapt_shape(&raw.shape, target_shape) else {
            tracing::warn!(
                "VITS2: shape mismatch for '{internal_name}' (from '{external_name}'): \
                 checkpoint {:?} vs model {:?}",
                raw.shape,
                target_shape
            );
            report.shape_mismatches += 1;
            continue;
        };

        let tensor = Tensor::from_vec(raw.values, shape, device)?;
        match varmap.set_one(&internal_name, &tensor) {
            Ok(()) => {
                report.loaded += 1;
                filled.insert(internal_name);
            }
            Err(e) => {
                tracing::warn!("VITS2: failed to set '{internal_name}': {e}");
                report.shape_mismatches += 1;
            }
        }
    }

    let mut missing: Vec<&String> = targets.keys().filter(|k| !filled.contains(*k)).collect();
    missing.sort();
    report.missing_parameters = missing.len();

    if report.loaded == 0 {
        return Err(VocoderError::ModelError(format!(
            "No VITS2 weights could be loaded from {path:?} \
             ({} unmapped, {} shape mismatches, {} unsupported dtype) — \
             verify that the checkpoint matches this model configuration",
            report.unmapped, report.shape_mismatches, report.unsupported_dtype
        )));
    }

    if mode == LoadMode::Strict && report.missing_parameters > 0 {
        let sample: Vec<&str> = missing.iter().take(5).map(|s| s.as_str()).collect();
        return Err(VocoderError::ModelError(format!(
            "Incomplete VITS2 checkpoint {path:?}: {} of {} model parameters were not supplied \
             (e.g. {sample:?}); those layers would keep their pseudo-random initialization. \
             Use a matching checkpoint, or load_weights_partial if a partial load is intended",
            report.missing_parameters, report.total_parameters
        )));
    }

    tracing::info!(
        "VITS2 checkpoint {path:?}: {} loaded, {} unmapped, {} shape mismatches, {} missing",
        report.loaded,
        report.unmapped,
        report.shape_mismatches,
        report.missing_parameters
    );

    Ok(report)
}

/// Strip checkpoint container prefixes such as `net_g.`, `module.` or `dec.`.
///
/// Repeats until no known prefix remains so that deeply nested names such as
/// `module.net_g.dec.ups.0.weight` reduce to `ups.0.weight`.
pub fn strip_checkpoint_prefixes(name: &str) -> &str {
    // NOTE: longer prefixes must come first, otherwise e.g. `decoder.` would be
    // truncated to `oder.` by the shorter `dec.` entry.
    const PREFIXES: [&str; 9] = [
        "module.",
        "model.",
        "net_g.",
        "generator.",
        "vocoder.",
        "decoder.",
        "dec.",
        "text_encoder.",
        "enc_p.",
    ];
    let mut current = name;
    'outer: loop {
        for prefix in PREFIXES {
            if let Some(rest) = current.strip_prefix(prefix) {
                current = rest;
                continue 'outer;
            }
        }
        return current;
    }
}

/// Rewrite the `prefix.{index}.` head of a parameter name.
///
/// Returns `Some((index, remainder))` when `name` starts with `prefix` followed
/// by a numeric index and a dot.
pub fn split_indexed_prefix<'a>(name: &'a str, prefix: &str) -> Option<(usize, &'a str)> {
    let rest = name.strip_prefix(prefix)?;
    let rest = rest.strip_prefix('.')?;
    let dot = rest.find('.')?;
    let index: usize = rest[..dot].parse().ok()?;
    Some((index, &rest[dot + 1..]))
}

/// Translate PyTorch layer-norm parameter names (`gamma`/`beta`) to the Candle
/// convention (`weight`/`bias`).
pub fn normalize_layernorm_suffix(name: &str) -> String {
    if let Some(base) = name.strip_suffix(".gamma") {
        format!("{base}.weight")
    } else if let Some(base) = name.strip_suffix(".beta") {
        format!("{base}.bias")
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarBuilder;

    fn build_varmap() -> (VarMap, Device) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _ = candle_nn::linear(4, 8, vb.pp("proj")).expect("linear");
        let _ = candle_nn::layer_norm(8, 1e-5, vb.pp("norm")).expect("layer norm");
        (varmap, device)
    }

    #[test]
    fn test_seed_varmap_is_deterministic_and_order_independent() {
        let (varmap_a, device) = build_varmap();
        let (varmap_b, _) = build_varmap();
        seed_varmap(&varmap_a, 1234, &device).expect("seed a");
        seed_varmap(&varmap_b, 1234, &device).expect("seed b");

        let a = varmap_a.data().lock().expect("lock a");
        let b = varmap_b.data().lock().expect("lock b");
        assert_eq!(a.len(), b.len());
        for (name, var) in a.iter() {
            let other = b.get(name).expect("same key set");
            let va = var
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("v");
            let vb = other
                .flatten_all()
                .expect("flat")
                .to_vec1::<f32>()
                .expect("v");
            assert_eq!(va, vb, "parameter {name} differs between runs");
        }
    }

    #[test]
    fn test_seed_varmap_respects_parameter_kind() {
        let (varmap, device) = build_varmap();
        seed_varmap(&varmap, 7, &device).expect("seed");
        let data = varmap.data().lock().expect("lock");

        let bias = data
            .get("proj.bias")
            .expect("bias")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        assert!(bias.iter().all(|&x| x == 0.0));

        let norm_weight = data
            .get("norm.weight")
            .expect("norm weight")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        assert!(norm_weight.iter().all(|&x| x == 1.0));

        let weight = data
            .get("proj.weight")
            .expect("weight")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        // fan_in = 4 → bound = 0.5
        assert!(weight.iter().all(|&x| x.abs() <= 0.5));
        assert!(weight.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_different_seeds_give_different_weights() {
        let (varmap_a, device) = build_varmap();
        let (varmap_b, _) = build_varmap();
        seed_varmap(&varmap_a, 1, &device).expect("seed a");
        seed_varmap(&varmap_b, 2, &device).expect("seed b");

        let a = varmap_a.data().lock().expect("lock");
        let b = varmap_b.data().lock().expect("lock");
        let wa = a
            .get("proj.weight")
            .expect("w")
            .flatten_all()
            .expect("f")
            .to_vec1::<f32>()
            .expect("v");
        let wb = b
            .get("proj.weight")
            .expect("w")
            .flatten_all()
            .expect("f")
            .to_vec1::<f32>()
            .expect("v");
        assert_ne!(wa, wb);
    }

    #[test]
    fn test_count_parameters_is_real() {
        let (varmap, _) = build_varmap();
        // linear: 8*4 weight + 8 bias; layer norm: 8 weight + 8 bias
        assert_eq!(count_parameters(&varmap).expect("count"), 32 + 8 + 8 + 8);
    }

    #[test]
    fn test_fuse_weight_norm_matches_definition() {
        // v = [[3, 4]], g = [10] → ||v|| = 5 → w = 10 * v / 5 = [6, 8]
        let v = RawTensor {
            values: vec![3.0, 4.0],
            shape: vec![1, 2],
        };
        let g = RawTensor {
            values: vec![10.0],
            shape: vec![1, 1],
        };
        let fused = fuse_weight_norm(&g, &v).expect("fused");
        assert_eq!(fused.shape, vec![1, 2]);
        assert!((fused.values[0] - 6.0).abs() < 1e-5);
        assert!((fused.values[1] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_adapt_shape_trailing_singletons() {
        assert_eq!(adapt_shape(&[8, 4, 1], &[8, 4]), Some(vec![8, 4]));
        assert_eq!(adapt_shape(&[8, 4], &[8, 4, 1]), Some(vec![8, 4, 1]));
        assert_eq!(adapt_shape(&[8, 4], &[8, 4]), Some(vec![8, 4]));
        assert_eq!(adapt_shape(&[8, 5], &[8, 4]), None);
    }

    #[test]
    fn test_strip_checkpoint_prefixes() {
        assert_eq!(
            strip_checkpoint_prefixes("module.net_g.dec.ups.0.weight"),
            "ups.0.weight"
        );
        assert_eq!(strip_checkpoint_prefixes("ups.0.weight"), "ups.0.weight");
        // `decoder.` must not be truncated to `oder.` by the shorter `dec.` rule.
        assert_eq!(
            strip_checkpoint_prefixes("decoder.conv_pre.weight"),
            "conv_pre.weight"
        );
        assert_eq!(strip_checkpoint_prefixes("enc_p.emb.weight"), "emb.weight");
    }

    #[test]
    fn test_split_indexed_prefix() {
        assert_eq!(
            split_indexed_prefix("resblocks.5.convs1.2.weight", "resblocks"),
            Some((5, "convs1.2.weight"))
        );
        assert_eq!(split_indexed_prefix("conv_pre.weight", "resblocks"), None);
    }

    #[test]
    fn test_normalize_layernorm_suffix() {
        assert_eq!(normalize_layernorm_suffix("norm1.gamma"), "norm1.weight");
        assert_eq!(normalize_layernorm_suffix("norm1.beta"), "norm1.bias");
        assert_eq!(normalize_layernorm_suffix("norm1.weight"), "norm1.weight");
    }

    #[test]
    fn test_load_safetensors_round_trip() {
        let (varmap_src, device) = build_varmap();
        seed_varmap(&varmap_src, 99, &device).expect("seed");

        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("weights.safetensors");
        varmap_src.save(&path).expect("save");

        let (mut varmap_dst, _) = build_varmap();
        seed_varmap(&varmap_dst, 1, &device).expect("seed dst");
        let report =
            load_safetensors_into_varmap(&mut varmap_dst, &path, &device, |n| Some(n.to_string()))
                .expect("load");
        assert_eq!(report.loaded, 4);
        assert_eq!(report.shape_mismatches, 0);
        assert_eq!(report.missing_parameters, 0);
        assert_eq!(report.total_parameters, 4);
        assert!(report.is_complete());

        let src = varmap_src.data().lock().expect("lock");
        let dst = varmap_dst.data().lock().expect("lock");
        for (name, var) in src.iter() {
            let a = var.flatten_all().expect("f").to_vec1::<f32>().expect("v");
            let b = dst
                .get(name)
                .expect("key")
                .flatten_all()
                .expect("f")
                .to_vec1::<f32>()
                .expect("v");
            assert_eq!(a, b, "{name} not restored");
        }
    }

    #[test]
    fn test_load_safetensors_fails_closed_when_nothing_matches() {
        let (varmap_src, device) = build_varmap();
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("weights.safetensors");
        varmap_src.save(&path).expect("save");

        let (mut varmap_dst, _) = build_varmap();
        let err = load_safetensors_into_varmap(&mut varmap_dst, &path, &device, |_| None)
            .expect_err("must fail closed");
        assert!(err.to_string().contains("No VITS2 weights"));
    }

    #[test]
    fn test_strict_mode_rejects_partial_coverage() {
        let (varmap_src, device) = build_varmap();
        seed_varmap(&varmap_src, 5, &device).expect("seed");
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("weights.safetensors");
        varmap_src.save(&path).expect("save");

        // Only the layer-norm parameters are mapped; the linear layer would keep
        // its random initialization.
        let partial_map = |n: &str| {
            if n.starts_with("norm.") {
                Some(n.to_string())
            } else {
                None
            }
        };

        let (mut strict_target, _) = build_varmap();
        let err = load_safetensors_into_varmap(&mut strict_target, &path, &device, partial_map)
            .expect_err("strict mode must reject a partial checkpoint");
        assert!(err.to_string().contains("Incomplete VITS2 checkpoint"));

        // Partial mode accepts it but reports exactly what is missing.
        let (mut partial_target, _) = build_varmap();
        let report = load_safetensors_into_varmap_with_mode(
            &mut partial_target,
            &path,
            &device,
            partial_map,
            LoadMode::Partial,
        )
        .expect("partial load");
        assert_eq!(report.loaded, 2);
        assert_eq!(report.missing_parameters, 2);
        assert_eq!(report.total_parameters, 4);
        assert!(!report.is_complete());
    }

    #[test]
    fn test_load_safetensors_missing_file_errors() {
        let (mut varmap, device) = build_varmap();
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("does-not-exist.safetensors");
        let err =
            load_safetensors_into_varmap(&mut varmap, &path, &device, |n| Some(n.to_string()))
                .expect_err("must fail");
        assert!(err.to_string().contains("Failed to read"));
    }
}
