//! Relative multi-head attention for the VITS2 text encoder.
//!
//! Real scaled dot-product attention over the actual query / key / value
//! contents, with two independent position signals:
//!
//! * a learned relative-position bias (Shaw et al. 2018, the form VITS uses),
//!   **added** to the content score rather than replacing it, and
//! * optional rotary position embeddings (RoPE) applied to the query and key.
//!
//! Also hosts the small row-major <-> tensor conversion helpers shared by the
//! text-encoder modules.

use crate::{Result, VocoderError};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Upper bound applied to
/// [`TextEncoderConfig::window_size`](super::text_encoder::TextEncoderConfig::window_size)
/// when building the learned relative-position embeddings.
pub const MAX_RELATIVE_POSITION: u32 = 32;

/// Attention module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub n_heads: u32,
    /// Hidden dimension
    pub hidden_channels: u32,
    /// Dimension per head
    pub head_dim: u32,
    /// Window size for relative position encoding (clipping distance)
    pub window_size: Option<u32>,
    /// Use relative position encoding
    pub relative_attention: bool,
    /// Maximum relative position (upper bound for the window size)
    pub max_relative_position: u32,
    /// Dropout probability
    pub p_dropout: f32,
    /// Apply rotary position embeddings (RoPE) to the query and key projections
    ///
    /// Requires an even `head_dim`. Can be combined with the learned relative
    /// bias; both are position signals applied to different parts of the score.
    #[serde(default)]
    pub use_rope: bool,
}

/// Relative multi-head attention.
///
/// Computes real scaled dot-product attention over the supplied query / key /
/// value contents:
///
/// ```text
/// score[h,i,j] = ( q[h,i]·k[h,j] + q[h,i]·r_k[clip(j-i)] ) / sqrt(head_dim)
/// attn         = softmax_j(score + additive_mask)
/// out[h,i]     = sum_j attn[h,i,j] * v[h,j] + sum_m weight[h,i,m] * r_v[m]
/// ```
///
/// where `r_k` / `r_v` are learnable relative-position embeddings and
/// `weight[h,i,m]` accumulates the attention mass that falls on relative offset
/// `m`. The relative term is *added* to the content score, never substituted
/// for it.
#[derive(Debug, Clone)]
pub struct RelativeMultiHeadAttention {
    /// Configuration
    pub config: AttentionConfig,
    /// Query projection
    q_proj: Linear,
    /// Key projection
    k_proj: Linear,
    /// Value projection
    v_proj: Linear,
    /// Output projection
    out_proj: Linear,
    /// Learnable relative key embeddings `[1, 2 * window + 1, head_dim]`
    emb_rel_k: Option<Tensor>,
    /// Learnable relative value embeddings `[1, 2 * window + 1, head_dim]`
    emb_rel_v: Option<Tensor>,
    /// Effective relative window size
    window: usize,
    /// Apply rotary position embeddings to the query and key projections
    use_rope: bool,
    /// `1 / sqrt(head_dim)`
    scale: f64,
    /// Compute device
    device: Device,
}

impl RelativeMultiHeadAttention {
    /// Create a new attention module whose parameters are registered under `vb`.
    ///
    /// # Errors
    /// Returns [`VocoderError::ModelError`] for an inconsistent configuration and
    /// [`VocoderError::CandleError`] when the tensors cannot be created.
    pub fn new(config: AttentionConfig, vb: VarBuilder) -> Result<Self> {
        if config.n_heads == 0 {
            return Err(VocoderError::ModelError(
                "Attention requires at least one head".to_string(),
            ));
        }
        if config.hidden_channels == 0 {
            return Err(VocoderError::ModelError(
                "Attention hidden channels must be greater than 0".to_string(),
            ));
        }
        if config.head_dim == 0 || config.head_dim * config.n_heads != config.hidden_channels {
            return Err(VocoderError::ModelError(format!(
                "Attention head_dim ({}) * n_heads ({}) must equal hidden_channels ({})",
                config.head_dim, config.n_heads, config.hidden_channels
            )));
        }

        let hidden = config.hidden_channels as usize;
        let head_dim = config.head_dim as usize;
        let device = vb.device().clone();

        let q_proj = candle_nn::linear(hidden, hidden, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden, hidden, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden, hidden, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(hidden, hidden, vb.pp("out_proj"))?;

        let window = config
            .window_size
            .map(|w| (w as usize).min(config.max_relative_position.max(1) as usize))
            .unwrap_or(0);

        if config.use_rope && !head_dim.is_multiple_of(2) {
            return Err(VocoderError::ModelError(format!(
                "Rotary position embeddings require an even head_dim (got {head_dim})"
            )));
        }

        let (emb_rel_k, emb_rel_v) = if config.relative_attention && window > 0 {
            let span = 2 * window + 1;
            let init = candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0 / (head_dim as f64).sqrt(),
            };
            (
                Some(vb.get_with_hints((1, span, head_dim), "emb_rel_k", init)?),
                Some(vb.get_with_hints((1, span, head_dim), "emb_rel_v", init)?),
            )
        } else {
            (None, None)
        };

        let use_rope = config.use_rope;
        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            emb_rel_k,
            emb_rel_v,
            window,
            use_rope,
            scale: 1.0 / (head_dim as f64).sqrt(),
            device,
        })
    }

    /// Whether relative position embeddings are active.
    pub fn uses_relative_positions(&self) -> bool {
        self.emb_rel_k.is_some()
    }

    /// Whether rotary position embeddings are active.
    pub fn uses_rope(&self) -> bool {
        self.use_rope
    }

    /// Build the `[seq_len, seq_len]` relative-offset index matrix.
    ///
    /// Entry `(i, j)` is `clamp(j - i, -window, window) + window`.
    fn relative_indices(&self, seq_len: usize) -> Result<Tensor> {
        let window = self.window as i64;
        let mut indices = Vec::with_capacity(seq_len * seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                let offset = (j as i64 - i as i64).clamp(-window, window) + window;
                indices.push(offset as u32);
            }
        }
        Ok(Tensor::from_vec(indices, (seq_len, seq_len), &self.device)?)
    }

    /// Attention over `[batch, seq_len, hidden]` tensors.
    ///
    /// `mask`, when given, is a `[query_len, key_len]` (or broadcastable) tensor
    /// where `1.0` means "attend" and `0.0` means "block".
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] on shape mismatches and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn forward_tensor(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, q_len, hidden) = query.dims3()?;
        let (k_batch, k_len, k_hidden) = key.dims3()?;
        let (v_batch, v_len, v_hidden) = value.dims3()?;

        if hidden != self.config.hidden_channels as usize {
            return Err(VocoderError::VocodingError(format!(
                "Attention expects hidden dimension {}, got {hidden}",
                self.config.hidden_channels
            )));
        }
        if k_hidden != hidden || v_hidden != hidden {
            return Err(VocoderError::VocodingError(
                "Query, key and value must share the same hidden dimension".to_string(),
            ));
        }
        if batch != k_batch || batch != v_batch {
            return Err(VocoderError::VocodingError(
                "Query, key and value must share the same batch size".to_string(),
            ));
        }
        if k_len != v_len {
            return Err(VocoderError::VocodingError(
                "Key and value must have the same sequence length".to_string(),
            ));
        }
        if self.uses_relative_positions() && q_len != k_len {
            return Err(VocoderError::VocodingError(
                "Relative attention requires equal query and key lengths".to_string(),
            ));
        }

        let heads = self.config.n_heads as usize;
        let head_dim = self.config.head_dim as usize;

        let split = |t: &Tensor, len: usize| -> Result<Tensor> {
            Ok(t.reshape((batch, len, heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };

        let q = split(&self.q_proj.forward(query)?, q_len)?;
        let k = split(&self.k_proj.forward(key)?, k_len)?;
        let v = split(&self.v_proj.forward(value)?, k_len)?;

        // Rotary position embeddings rotate the content term only; the learned
        // relative bias below keeps operating on the un-rotated query.
        let (q_content, k_content) = if self.use_rope {
            let (cos_q, sin_q) = rope_tables(q_len, head_dim, &self.device)?;
            let rotated_q = apply_rope(&q, &cos_q, &sin_q)?;
            let rotated_k = if k_len == q_len {
                apply_rope(&k, &cos_q, &sin_q)?
            } else {
                let (cos_k, sin_k) = rope_tables(k_len, head_dim, &self.device)?;
                apply_rope(&k, &cos_k, &sin_k)?
            };
            (rotated_q, rotated_k)
        } else {
            (q.clone(), k.clone())
        };

        // Content scores: [batch, heads, q_len, k_len]
        let mut scores = q_content
            .matmul(&k_content.transpose(2, 3)?.contiguous()?)?
            .affine(self.scale, 0.0)?;

        // Relative position bias, added to (not substituted for) the content score.
        let rel_indices = if self.uses_relative_positions() {
            let idx = self
                .relative_indices(q_len)?
                .reshape((1, 1, q_len, k_len))?
                .expand((batch, heads, q_len, k_len))?
                .contiguous()?;
            if let Some(emb_rel_k) = &self.emb_rel_k {
                let span = 2 * self.window + 1;
                let rel_k = emb_rel_k
                    .reshape((span, head_dim))?
                    .t()?
                    .reshape((1, 1, head_dim, span))?
                    .contiguous()?;
                let compact = q.broadcast_matmul(&rel_k)?.affine(self.scale, 0.0)?;
                let rel_scores = compact.contiguous()?.gather(&idx, 3)?;
                scores = (scores + rel_scores)?;
            }
            Some(idx)
        } else {
            None
        };

        if let Some(mask) = mask {
            // 1.0 -> keep (adds 0), 0.0 -> block (adds -1e9).
            let additive = mask.affine(1.0e9, -1.0e9)?;
            scores = scores.broadcast_add(&additive)?;
        }

        let attn = candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?;
        let mut out = attn.matmul(&v)?;

        if let (Some(emb_rel_v), Some(idx)) = (&self.emb_rel_v, rel_indices.as_ref()) {
            let span = 2 * self.window + 1;
            let zeros = Tensor::zeros((batch, heads, q_len, span), DType::F32, &self.device)?;
            let weights = zeros.scatter_add(idx, &attn.contiguous()?, 3)?;
            let rel_v = emb_rel_v.reshape((1, 1, span, head_dim))?;
            out = (out + weights.broadcast_matmul(&rel_v)?)?;
        }

        let merged = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, q_len, hidden))?;
        Ok(self.out_proj.forward(&merged)?)
    }

    /// Attention over row-major `[seq_len][hidden]` buffers.
    ///
    /// `mask[i][j] == true` means position `i` may attend to position `j`.
    ///
    /// # Errors
    /// Returns [`VocoderError::VocodingError`] for empty or ragged input and
    /// [`VocoderError::CandleError`] on tensor failures.
    pub fn forward(
        &self,
        query: &[Vec<f32>],
        key: &[Vec<f32>],
        value: &[Vec<f32>],
        mask: Option<&[Vec<bool>]>,
    ) -> Result<Vec<Vec<f32>>> {
        if query.is_empty() || key.is_empty() || value.is_empty() {
            return Err(VocoderError::VocodingError(
                "Empty input tensors".to_string(),
            ));
        }

        let q = rows_to_tensor(query, &self.device)?;
        let k = rows_to_tensor(key, &self.device)?;
        let v = rows_to_tensor(value, &self.device)?;
        let mask_tensor = match mask {
            Some(mask) => Some(mask_to_tensor(mask, query.len(), key.len(), &self.device)?),
            None => None,
        };

        let out = self.forward_tensor(&q, &k, &v, mask_tensor.as_ref())?;
        tensor_to_rows(&out)
    }

    /// Number of scalars held by this module.
    pub fn num_parameters(&self) -> u64 {
        let linear = |l: &Linear| {
            l.weight().elem_count() as u64 + l.bias().map_or(0, |b| b.elem_count() as u64)
        };
        linear(&self.q_proj)
            + linear(&self.k_proj)
            + linear(&self.v_proj)
            + linear(&self.out_proj)
            + self.emb_rel_k.as_ref().map_or(0, |t| t.elem_count() as u64)
            + self.emb_rel_v.as_ref().map_or(0, |t| t.elem_count() as u64)
    }
}

/// Build the rotary-position cosine/sine tables `[seq_len, head_dim / 2]`.
///
/// Uses the standard RoPE frequency schedule `theta_i = 10000^(-2i/head_dim)`.
fn rope_tables(seq_len: usize, head_dim: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * half);
    let mut sin = Vec::with_capacity(seq_len * half);
    for pos in 0..seq_len {
        for i in 0..half {
            let theta = 10000.0_f64.powf(-2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * theta;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq_len, half), device)?,
        Tensor::from_vec(sin, (seq_len, half), device)?,
    ))
}

/// Apply rotary position embeddings to a `[batch, heads, seq_len, head_dim]` tensor.
///
/// Adjacent feature pairs are rotated by the position-dependent angle:
/// `x'[2i] = x[2i] * cos - x[2i+1] * sin`, `x'[2i+1] = x[2i] * sin + x[2i+1] * cos`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, heads, seq_len, head_dim) = x.dims4()?;
    let half = head_dim / 2;
    let pairs = x.reshape((batch, heads, seq_len, half, 2))?;
    let x0 = pairs.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = pairs.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = cos.reshape((1, 1, seq_len, half))?;
    let sin = sin.reshape((1, 1, seq_len, half))?;
    let out0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let out1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    Ok(Tensor::stack(&[out0, out1], 4)?.reshape((batch, heads, seq_len, head_dim))?)
}

/// Convert row-major `[seq_len][hidden]` data into a `[1, seq_len, hidden]` tensor.
pub(crate) fn rows_to_tensor(rows: &[Vec<f32>], device: &Device) -> Result<Tensor> {
    let seq_len = rows.len();
    if seq_len == 0 {
        return Err(VocoderError::VocodingError("Empty input".to_string()));
    }
    let hidden = rows[0].len();
    if hidden == 0 {
        return Err(VocoderError::VocodingError(
            "Input rows must not be empty".to_string(),
        ));
    }
    if rows.iter().any(|r| r.len() != hidden) {
        return Err(VocoderError::VocodingError(
            "All input rows must have the same length".to_string(),
        ));
    }
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Ok(Tensor::from_vec(flat, (1, seq_len, hidden), device)?)
}

/// Convert a `[1, seq_len, hidden]` tensor back into row-major data.
pub(crate) fn tensor_to_rows(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    let (_, seq_len, hidden) = tensor.dims3()?;
    let flat = tensor.flatten_all()?.to_vec1::<f32>()?;
    Ok((0..seq_len)
        .map(|i| flat[i * hidden..(i + 1) * hidden].to_vec())
        .collect())
}

/// Convert a boolean attention mask into a `[q_len, k_len]` 1.0/0.0 tensor.
pub(crate) fn mask_to_tensor(
    mask: &[Vec<bool>],
    q_len: usize,
    k_len: usize,
    device: &Device,
) -> Result<Tensor> {
    if mask.len() != q_len || mask.iter().any(|row| row.len() != k_len) {
        return Err(VocoderError::VocodingError(format!(
            "Attention mask must be {q_len}x{k_len}"
        )));
    }
    let flat: Vec<f32> = mask
        .iter()
        .flat_map(|row| row.iter().map(|&keep| if keep { 1.0 } else { 0.0 }))
        .collect();
    Ok(Tensor::from_vec(flat, (q_len, k_len), device)?)
}
