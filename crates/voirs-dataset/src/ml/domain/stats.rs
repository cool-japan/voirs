//! Real two-sample test statistics and distribution distances.
//!
//! These routines back the domain-shift detectors in [`super::adapter`]. They
//! are implemented from first principles over `&[f32]` slices using only the
//! standard library (no external statistics crates), with all intermediate math
//! performed in `f64` for numerical stability.
//!
//! Provided statistics:
//! * [`ks_statistic`] — two-sample Kolmogorov–Smirnov `D`.
//! * [`mann_whitney_u`] — Mann–Whitney `U` with tie-corrected mid-ranks.
//! * [`chi_square_statistic`] / [`chi_square_cdf`] — binned homogeneity test.
//! * [`anderson_darling_2sample`] — standardized Scholz–Stephens `k`-sample (k=2).
//! * [`wasserstein1`] — 1-D earth-mover distance (L1 between empirical CDFs).
//! * [`jensen_shannon_divergence`] — base-2 JS divergence of shared-support histograms.
//! * [`kl_divergence`] — epsilon-smoothed empirical KL divergence.
//! * [`mmd_rbf`] — unbiased Maximum Mean Discrepancy with a median-heuristic RBF kernel.

use std::cmp::Ordering;

/// Largest per-group sample count used by [`mmd_rbf`]; larger inputs are
/// deterministically strided down to bound the `O(n^2)` kernel evaluation.
const MMD_MAX: usize = 1024;

/// Collect the finite values of `data` as `f64`.
fn finite_f64(data: &[f32]) -> Vec<f64> {
    data.iter()
        .map(|&x| x as f64)
        .filter(|x| x.is_finite())
        .collect()
}

/// Collect the finite values of `data` as `f64`, sorted ascending.
fn sorted_f64(data: &[f32]) -> Vec<f64> {
    let mut v = finite_f64(data);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    v
}

/// Two-sample Kolmogorov–Smirnov statistic `D = sup_x |F_a(x) − F_b(x)|`.
///
/// The supremum is reached at one of the pooled sample points, so the empirical
/// CDFs are compared after each jump while merging the two sorted sequences.
/// Returns `0.0` for empty inputs. Range: `[0, 1]`.
pub(crate) fn ks_statistic(a: &[f32], b: &[f32]) -> f64 {
    let xa = sorted_f64(a);
    let xb = sorted_f64(b);
    if xa.is_empty() || xb.is_empty() {
        return 0.0;
    }

    let na = xa.len() as f64;
    let nb = xb.len() as f64;
    let mut i = 0usize;
    let mut j = 0usize;
    let mut d = 0.0f64;

    while i < xa.len() && j < xb.len() {
        // Smallest not-yet-consumed pooled value.
        let x = if xa[i] <= xb[j] { xa[i] } else { xb[j] };
        while i < xa.len() && xa[i] <= x {
            i += 1;
        }
        while j < xb.len() && xb[j] <= x {
            j += 1;
        }
        let fa = i as f64 / na;
        let fb = j as f64 / nb;
        d = d.max((fa - fb).abs());
    }

    d
}

/// Mann–Whitney `U` statistic for sample `a` against `b`, using average
/// (mid-)ranks so ties are handled correctly.
///
/// For two samples drawn from the same distribution `U ≈ n·m / 2`; the statistic
/// approaches `0` (or `n·m`) as the samples separate. Returns `0.0` for empty
/// inputs.
pub(crate) fn mann_whitney_u(a: &[f32], b: &[f32]) -> f64 {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 {
        return 0.0;
    }

    // Pool with a group label (0 = a, 1 = b) and sort by value.
    let mut pooled: Vec<(f64, u8)> = Vec::with_capacity(na + nb);
    for &x in a {
        let v = x as f64;
        if v.is_finite() {
            pooled.push((v, 0));
        }
    }
    for &x in b {
        let v = x as f64;
        if v.is_finite() {
            pooled.push((v, 1));
        }
    }
    pooled.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(Ordering::Equal));

    let total = pooled.len();
    let mut rank_sum_a = 0.0f64;
    let mut i = 0usize;
    while i < total {
        let v = pooled[i].0;
        // End of the tie block (exclusive); slice is sorted so this partitions.
        let j = pooled.partition_point(|p| p.0 <= v);
        // 1-based ranks i+1..=j share the average rank (i + 1 + j) / 2.
        let avg_rank = ((i + 1 + j) as f64) / 2.0;
        let a_in_block = pooled[i..j].iter().filter(|p| p.1 == 0).count() as f64;
        rank_sum_a += avg_rank * a_in_block;
        i = j;
    }

    let na_f = na as f64;
    rank_sum_a - na_f * (na_f + 1.0) / 2.0
}

/// Bin both samples on a shared support `[min, max]` and return the raw bin
/// counts `(counts_a, counts_b)`. Returns `None` for empty inputs or `bins == 0`.
///
/// When every pooled value is identical the histograms collapse to a single
/// populated bin, which makes downstream divergences vanish as they should.
fn binned_counts_pair(a: &[f32], b: &[f32], bins: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    if a.is_empty() || b.is_empty() || bins == 0 {
        return None;
    }

    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &x in a.iter().chain(b.iter()) {
        let v = x as f64;
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }

    let mut ca = vec![0.0f64; bins];
    let mut cb = vec![0.0f64; bins];

    if (hi - lo) <= f64::EPSILON {
        // Degenerate support: all mass in one bin for both samples.
        ca[0] = a.len() as f64;
        cb[0] = b.len() as f64;
        return Some((ca, cb));
    }

    let width = (hi - lo) / bins as f64;
    let index_of = |v: f64| -> usize { (((v - lo) / width) as usize).min(bins - 1) };

    for &x in a {
        let v = x as f64;
        if v.is_finite() {
            ca[index_of(v)] += 1.0;
        }
    }
    for &x in b {
        let v = x as f64;
        if v.is_finite() {
            cb[index_of(v)] += 1.0;
        }
    }

    Some((ca, cb))
}

/// A reasonable bin count for histogram-based statistics: `sqrt(min(n, m))`
/// clamped to `[4, 32]`.
pub(crate) fn default_bins(a: &[f32], b: &[f32]) -> usize {
    let n = a.len().min(b.len());
    ((n as f64).sqrt().round() as usize).clamp(4, 32)
}

/// Pearson chi-square statistic for the 2×`bins` homogeneity contingency table
/// together with its degrees of freedom (`bins_used − 1`).
///
/// Expected counts use the standard `row_total · col_total / N` rule; empty
/// columns are skipped and do not contribute degrees of freedom. For samples
/// from the same distribution the statistic is small; it grows as they diverge.
pub(crate) fn chi_square_statistic(a: &[f32], b: &[f32], bins: usize) -> (f64, usize) {
    let Some((ca, cb)) = binned_counts_pair(a, b, bins) else {
        return (0.0, 0);
    };

    let na: f64 = ca.iter().sum();
    let nb: f64 = cb.iter().sum();
    let total = na + nb;
    if total <= 0.0 {
        return (0.0, 0);
    }

    let mut chi2 = 0.0f64;
    let mut used = 0usize;
    for (oa, ob) in ca.iter().zip(cb.iter()) {
        let col = oa + ob;
        if col <= 0.0 {
            continue;
        }
        used += 1;
        let e_a = na * col / total;
        let e_b = nb * col / total;
        if e_a > 0.0 {
            chi2 += (oa - e_a).powi(2) / e_a;
        }
        if e_b > 0.0 {
            chi2 += (ob - e_b).powi(2) / e_b;
        }
    }

    (chi2, used.saturating_sub(1))
}

/// Natural logarithm of the gamma function (Lanczos approximation, g = 7).
fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients (g = 7, n = 9).
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        // Reflection formula for the left half-plane.
        let pi = std::f64::consts::PI;
        pi.ln() - (pi * x).sin().abs().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Regularized lower incomplete gamma `P(s, x) = γ(s, x) / Γ(s)`.
///
/// Series expansion for `x < s + 1`, Lentz continued fraction otherwise
/// (Numerical Recipes `gammp`). Result is clamped to `[0, 1]`.
fn reg_lower_gamma(s: f64, x: f64) -> f64 {
    if s <= 0.0 || x <= 0.0 {
        return 0.0;
    }

    if x < s + 1.0 {
        // Series representation.
        let mut ap = s;
        let mut del = 1.0 / s;
        let mut sum = del;
        for _ in 0..512 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        (sum * (-x + s * x.ln() - ln_gamma(s)).exp()).clamp(0.0, 1.0)
    } else {
        // Continued fraction for the upper tail Q, then P = 1 − Q.
        let tiny = 1e-300;
        let mut b = x + 1.0 - s;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for n in 1..512 {
            let an = -(n as f64) * (n as f64 - s);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        let q = (-x + s * x.ln() - ln_gamma(s)).exp() * h;
        (1.0 - q).clamp(0.0, 1.0)
    }
}

/// Cumulative distribution function of the chi-square distribution with `dof`
/// degrees of freedom evaluated at `chi2` (i.e. `1 − p_value`).
pub(crate) fn chi_square_cdf(chi2: f64, dof: usize) -> f64 {
    if dof == 0 || chi2 <= 0.0 {
        return 0.0;
    }
    reg_lower_gamma(dof as f64 / 2.0, chi2 / 2.0)
}

/// Standardized two-sample Anderson–Darling statistic (Scholz & Stephens 1987,
/// `k = 2`, mid-rank version that tolerates ties).
///
/// The midrank `A²_akN` is computed and then standardized to mean `0` / variance
/// `1` under the null hypothesis using the published variance polynomial, so the
/// value is comparable across sample sizes. For samples that are too small to
/// standardize (`N < 8`) the centered raw statistic `A²_akN − (k−1)` is returned.
/// Returns `0.0` when the pooled sample has no variation.
pub(crate) fn anderson_darling_2sample(a: &[f32], b: &[f32]) -> f64 {
    let xa = sorted_f64(a);
    let xb = sorted_f64(b);
    let n1 = xa.len();
    let n2 = xb.len();
    if n1 == 0 || n2 == 0 {
        return 0.0;
    }

    let n_total = n1 + n2;
    let n = n_total as f64;

    let mut pooled = Vec::with_capacity(n_total);
    pooled.extend_from_slice(&xa);
    pooled.extend_from_slice(&xb);
    pooled.sort_by(|p, q| p.partial_cmp(q).unwrap_or(Ordering::Equal));

    let mut distinct = pooled.clone();
    distinct.dedup();
    if distinct.len() < 2 {
        return 0.0;
    }

    let samples: [&[f64]; 2] = [&xa, &xb];
    let ns = [n1 as f64, n2 as f64];

    let mut a2akn = 0.0f64;
    for (sample, &ni) in samples.iter().zip(ns.iter()) {
        let mut acc = 0.0f64;
        for &zs in &distinct {
            let cnt_lt = pooled.partition_point(|&v| v < zs) as f64;
            let cnt_le = pooled.partition_point(|&v| v <= zs) as f64;
            let lj = cnt_le - cnt_lt;
            let bj = cnt_lt + lj / 2.0;

            let s_lt = sample.partition_point(|&v| v < zs) as f64;
            let s_le = sample.partition_point(|&v| v <= zs) as f64;
            let fij = s_le - s_lt;
            let mij = s_le - fij / 2.0;

            let denom = bj * (n - bj) - n * lj / 4.0;
            if denom.abs() < 1e-12 {
                continue;
            }
            acc += lj / n * (n * mij - bj * ni).powi(2) / denom;
        }
        a2akn += acc / ni;
    }
    a2akn *= (n - 1.0) / n;

    let k = 2.0f64;
    let big_h = 1.0 / ns[0] + 1.0 / ns[1];

    if n_total < 8 {
        return a2akn - (k - 1.0);
    }

    // Harmonic prefix sums: harm[j] = sum_{m=1}^{j} 1/m.
    let mut harm = vec![0.0f64; n_total];
    for j in 1..n_total {
        harm[j] = harm[j - 1] + 1.0 / j as f64;
    }
    let h = harm[n_total - 1];
    let mut g = 0.0f64;
    for l in 1..=(n_total - 2) {
        g += (h - harm[l]) / (n_total - l) as f64;
    }

    let a_coef = (4.0 * g - 6.0) * (k - 1.0) + (10.0 - 6.0 * g) * big_h;
    let b_coef = (2.0 * g - 4.0) * k * k + 8.0 * h * k + (2.0 * g - 14.0 * h - 4.0) * big_h
        - 8.0 * h
        + 4.0 * g
        - 6.0;
    let c_coef = (6.0 * h + 2.0 * g - 2.0) * k * k
        + (4.0 * h - 4.0 * g + 6.0) * k
        + (2.0 * h - 6.0) * big_h
        + 4.0 * h;
    let d_coef = (2.0 * h + 6.0) * k * k - 4.0 * h * k;

    let sigma_sq = (a_coef * n * n * n + b_coef * n * n + c_coef * n + d_coef)
        / ((n - 1.0) * (n - 2.0) * (n - 3.0));
    if sigma_sq <= 0.0 || !sigma_sq.is_finite() {
        return a2akn - (k - 1.0);
    }

    (a2akn - (k - 1.0)) / sigma_sq.sqrt()
}

/// 1-D Wasserstein-1 (earth-mover) distance: the `L1` area between the two
/// empirical CDFs, `∫ |F_a(x) − F_b(x)| dx`.
///
/// Handles unequal sample sizes. For two constant samples offset by `s` the
/// result is exactly `|s|`. Returns `0.0` for empty inputs.
pub(crate) fn wasserstein1(a: &[f32], b: &[f32]) -> f64 {
    let xa = sorted_f64(a);
    let xb = sorted_f64(b);
    if xa.is_empty() || xb.is_empty() {
        return 0.0;
    }

    let mut support = Vec::with_capacity(xa.len() + xb.len());
    support.extend_from_slice(&xa);
    support.extend_from_slice(&xb);
    support.sort_by(|p, q| p.partial_cmp(q).unwrap_or(Ordering::Equal));
    support.dedup();
    if support.len() < 2 {
        return 0.0;
    }

    let na = xa.len() as f64;
    let nb = xb.len() as f64;
    let mut i = 0usize;
    let mut j = 0usize;
    let mut w = 0.0f64;

    for window in support.windows(2) {
        let x = window[0];
        let dx = window[1] - window[0];
        while i < xa.len() && xa[i] <= x {
            i += 1;
        }
        while j < xb.len() && xb[j] <= x {
            j += 1;
        }
        let fa = i as f64 / na;
        let fb = j as f64 / nb;
        w += (fa - fb).abs() * dx;
    }

    w
}

/// Jensen–Shannon divergence (base 2) between shared-support histograms.
///
/// `JS = ½·KL(P‖M) + ½·KL(Q‖M)` with `M = ½(P + Q)`; using `log2` bounds the
/// result to `[0, 1]`. Equals `0` for identical distributions. Returns `0.0`
/// for empty inputs.
pub(crate) fn jensen_shannon_divergence(a: &[f32], b: &[f32], bins: usize) -> f64 {
    let Some((ca, cb)) = binned_counts_pair(a, b, bins) else {
        return 0.0;
    };
    let sa: f64 = ca.iter().sum();
    let sb: f64 = cb.iter().sum();
    if sa <= 0.0 || sb <= 0.0 {
        return 0.0;
    }

    let mut js = 0.0f64;
    for (oa, ob) in ca.iter().zip(cb.iter()) {
        let p = oa / sa;
        let q = ob / sb;
        let m = 0.5 * (p + q);
        if p > 0.0 && m > 0.0 {
            js += 0.5 * p * (p / m).log2();
        }
        if q > 0.0 && m > 0.0 {
            js += 0.5 * q * (q / m).log2();
        }
    }

    js.clamp(0.0, 1.0)
}

/// Epsilon-smoothed empirical Kullback–Leibler divergence `KL(P‖Q)` in nats.
///
/// Both shared-support histograms have `eps` added to every bin before
/// renormalization, which keeps the divergence finite even where `Q` would
/// otherwise be zero. Equals `0` for identical distributions; grows without an
/// upper bound as they diverge. Returns `0.0` for empty inputs.
pub(crate) fn kl_divergence(a: &[f32], b: &[f32], bins: usize, eps: f64) -> f64 {
    let Some((ca, cb)) = binned_counts_pair(a, b, bins) else {
        return 0.0;
    };
    let eps = eps.max(0.0);
    let ta: f64 = ca.iter().map(|c| c + eps).sum();
    let tb: f64 = cb.iter().map(|c| c + eps).sum();
    if ta <= 0.0 || tb <= 0.0 {
        return 0.0;
    }

    let mut kl = 0.0f64;
    for (oa, ob) in ca.iter().zip(cb.iter()) {
        let p = (oa + eps) / ta;
        let q = (ob + eps) / tb;
        if p > 0.0 && q > 0.0 {
            kl += p * (p / q).ln();
        }
    }

    kl.max(0.0)
}

/// Deterministically stride `data` down to at most `max` elements.
fn subsample(data: &[f64], max: usize) -> Vec<f64> {
    if data.len() <= max || max == 0 {
        return data.to_vec();
    }
    let step = data.len().div_ceil(max);
    data.iter().step_by(step).copied().collect()
}

/// Median of all pairwise absolute differences within `pooled` — the classic
/// RBF bandwidth heuristic. Returns `0.0` when there is no spread.
fn median_pairwise_bandwidth(pooled: &[f64]) -> f64 {
    let n = pooled.len();
    if n < 2 {
        return 0.0;
    }
    let mut diffs = Vec::with_capacity(n * (n - 1) / 2);
    for (i, &x) in pooled.iter().enumerate() {
        for &y in pooled.iter().skip(i + 1) {
            diffs.push((x - y).abs());
        }
    }
    if diffs.is_empty() {
        return 0.0;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = diffs.len() / 2;
    if diffs.len() % 2 == 0 {
        0.5 * (diffs[mid - 1] + diffs[mid])
    } else {
        diffs[mid]
    }
}

/// Unbiased Maximum Mean Discrepancy (squared) with a Gaussian/RBF kernel.
///
/// The bandwidth is set by the median heuristic over the pooled sample, giving
/// `k(x, y) = exp(−‖x − y‖² / (2σ²))`. The estimator
/// `MMD²_u = E_xx[k] + E_yy[k] − 2·E_xy[k]` (diagonal excluded) is `≈ 0` for
/// equal distributions and positive otherwise; tiny negative values from the
/// unbiased correction are possible. Large inputs are strided to [`MMD_MAX`]
/// per group to bound the cost. Returns `0.0` for degenerate inputs.
pub(crate) fn mmd_rbf(a: &[f32], b: &[f32]) -> f64 {
    let xa = subsample(&finite_f64(a), MMD_MAX);
    let xb = subsample(&finite_f64(b), MMD_MAX);
    let n = xa.len();
    let m = xb.len();
    if n < 2 || m < 2 {
        return 0.0;
    }

    let mut pooled = xa.clone();
    pooled.extend_from_slice(&xb);
    let sigma = median_pairwise_bandwidth(&pooled);
    if sigma <= 0.0 {
        return 0.0;
    }
    let gamma = 1.0 / (2.0 * sigma * sigma);
    let kernel = |x: f64, y: f64| (-gamma * (x - y) * (x - y)).exp();

    let mut sxx = 0.0f64;
    for (i, &xi) in xa.iter().enumerate() {
        for (j, &xj) in xa.iter().enumerate() {
            if i != j {
                sxx += kernel(xi, xj);
            }
        }
    }
    let mut syy = 0.0f64;
    for (i, &yi) in xb.iter().enumerate() {
        for (j, &yj) in xb.iter().enumerate() {
            if i != j {
                syy += kernel(yi, yj);
            }
        }
    }
    let mut sxy = 0.0f64;
    for &xi in &xa {
        for &yj in &xb {
            sxy += kernel(xi, yj);
        }
    }

    sxx / ((n * (n - 1)) as f64) + syy / ((m * (m - 1)) as f64) - 2.0 * sxy / ((n * m) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic LCG (Knuth MMIX constants) used to synthesize
    /// reproducible Gaussian samples without pulling in an RNG dependency.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Top 53 bits -> [0, 1).
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        fn next_gauss(&mut self) -> f64 {
            let u1 = self.next_unit().max(1e-12);
            let u2 = self.next_unit();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }

        fn gaussian_sample(&mut self, n: usize, mean: f64, std: f64) -> Vec<f32> {
            (0..n)
                .map(|_| (mean + std * self.next_gauss()) as f32)
                .collect()
        }
    }

    // ---------- Kolmogorov–Smirnov ----------

    #[test]
    fn test_ks_identical_is_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(ks_statistic(&a, &a).abs() < 1e-12);
    }

    #[test]
    fn test_ks_disjoint_is_one() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((ks_statistic(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_ks_shift_increases_statistic() {
        let mut rng = Lcg::new(1);
        let a = rng.gaussian_sample(400, 0.0, 1.0);
        let same = rng.gaussian_sample(400, 0.0, 1.0);
        let shifted = rng.gaussian_sample(400, 3.0, 1.0);
        assert!(ks_statistic(&a, &shifted) > ks_statistic(&a, &same));
    }

    // ---------- Mann–Whitney U ----------

    #[test]
    fn test_mann_whitney_identical_is_half_nm() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let u = mann_whitney_u(&a, &b);
        let expected = (a.len() * b.len()) as f64 / 2.0;
        assert!((u - expected).abs() < 1e-9, "u = {u}, expected {expected}");
    }

    #[test]
    fn test_mann_whitney_disjoint_is_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 11.0, 12.0];
        // All of `a` ranks below all of `b` -> U_a = 0.
        assert!(mann_whitney_u(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn test_mann_whitney_shift_moves_away_from_center() {
        let mut rng = Lcg::new(2);
        let a = rng.gaussian_sample(300, 0.0, 1.0);
        let same = rng.gaussian_sample(300, 0.0, 1.0);
        let shifted = rng.gaussian_sample(300, 2.5, 1.0);
        let center = (a.len() * a.len()) as f64 / 2.0;
        let dev_same = (mann_whitney_u(&a, &same) - center).abs();
        let dev_shift = (mann_whitney_u(&a, &shifted) - center).abs();
        assert!(dev_shift > dev_same);
    }

    // ---------- Chi-square ----------

    #[test]
    fn test_chi_square_cdf_known_value() {
        // Chi-square with 2 dof has CDF 1 − e^{−x/2}; at x = 2 that is 1 − 1/e.
        let cdf = chi_square_cdf(2.0, 2);
        let expected = 1.0 - (-1.0f64).exp();
        assert!((cdf - expected).abs() < 1e-6, "cdf = {cdf}");
    }

    #[test]
    fn test_chi_square_shift_increases_statistic() {
        let mut rng = Lcg::new(3);
        let a = rng.gaussian_sample(600, 0.0, 1.0);
        let same = rng.gaussian_sample(600, 0.0, 1.0);
        let shifted = rng.gaussian_sample(600, 3.0, 1.0);
        let (chi_same, _) = chi_square_statistic(&a, &same, 12);
        let (chi_shift, _) = chi_square_statistic(&a, &shifted, 12);
        assert!(chi_shift > chi_same, "{chi_shift} !> {chi_same}");
    }

    // ---------- Anderson–Darling ----------

    #[test]
    fn test_anderson_darling_shift_increases_and_is_finite() {
        let mut rng = Lcg::new(4);
        let a = rng.gaussian_sample(300, 0.0, 1.0);
        let same = rng.gaussian_sample(300, 0.0, 1.0);
        let shifted = rng.gaussian_sample(300, 2.0, 1.0);
        let ad_same = anderson_darling_2sample(&a, &same);
        let ad_shift = anderson_darling_2sample(&a, &shifted);
        assert!(ad_same.is_finite() && ad_shift.is_finite());
        assert!(ad_shift > ad_same, "{ad_shift} !> {ad_same}");
    }

    // ---------- Wasserstein-1 ----------

    #[test]
    fn test_wasserstein_shifted_constants_equals_shift() {
        let a = vec![2.0f32; 8];
        let b = vec![5.0f32; 8];
        assert!((wasserstein1(&a, &b) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_wasserstein_identical_is_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!(wasserstein1(&a, &a).abs() < 1e-12);
    }

    #[test]
    fn test_wasserstein_unequal_sizes() {
        // F_a steps 0 -> 1 at 0; F_b steps to 1 at 4 (two values 4 each).
        let a = vec![0.0f32];
        let b = vec![4.0f32, 4.0f32];
        assert!((wasserstein1(&a, &b) - 4.0).abs() < 1e-9);
    }

    // ---------- Jensen–Shannon ----------

    #[test]
    fn test_js_identical_is_zero() {
        let mut rng = Lcg::new(5);
        let a = rng.gaussian_sample(500, 0.0, 1.0);
        assert!(jensen_shannon_divergence(&a, &a, 16).abs() < 1e-9);
    }

    #[test]
    fn test_js_shift_is_positive_and_bounded() {
        let mut rng = Lcg::new(6);
        let a = rng.gaussian_sample(500, 0.0, 1.0);
        let shifted = rng.gaussian_sample(500, 5.0, 1.0);
        let js = jensen_shannon_divergence(&a, &shifted, 16);
        assert!(js > 0.1, "js = {js}");
        assert!((0.0..=1.0).contains(&js));
    }

    // ---------- KL divergence ----------

    #[test]
    fn test_kl_identical_is_near_zero() {
        let mut rng = Lcg::new(7);
        let a = rng.gaussian_sample(500, 0.0, 1.0);
        let kl = kl_divergence(&a, &a, 16, 1e-9);
        assert!(kl < 1e-6, "kl = {kl}");
    }

    #[test]
    fn test_kl_shift_is_positive() {
        let mut rng = Lcg::new(8);
        let a = rng.gaussian_sample(500, 0.0, 1.0);
        let same = rng.gaussian_sample(500, 0.0, 1.0);
        let shifted = rng.gaussian_sample(500, 4.0, 1.0);
        let kl_same = kl_divergence(&a, &same, 16, 1e-9);
        let kl_shift = kl_divergence(&a, &shifted, 16, 1e-9);
        assert!(kl_shift > kl_same, "{kl_shift} !> {kl_same}");
    }

    // ---------- MMD ----------

    #[test]
    fn test_mmd_same_distribution_is_near_zero() {
        // Two independent draws from the same distribution: the unbiased MMD^2
        // estimate is ~0 (it may be slightly negative by construction) and
        // negligible next to a genuinely shifted comparison.
        let mut rng = Lcg::new(9);
        let a = rng.gaussian_sample(200, 0.0, 1.0);
        let b = rng.gaussian_sample(200, 0.0, 1.0);
        let shifted = rng.gaussian_sample(200, 3.0, 1.0);

        let mmd_same = mmd_rbf(&a, &b);
        let mmd_shift = mmd_rbf(&a, &shifted);

        assert!(mmd_same.abs() < 0.05, "mmd_same = {mmd_same}");
        assert!(
            mmd_same.abs() < 0.1 * mmd_shift,
            "mmd_same {mmd_same} should be negligible vs mmd_shift {mmd_shift}"
        );
    }

    #[test]
    fn test_mmd_shift_is_positive() {
        let mut rng = Lcg::new(10);
        let a = rng.gaussian_sample(200, 0.0, 1.0);
        let same = rng.gaussian_sample(200, 0.0, 1.0);
        let shifted = rng.gaussian_sample(200, 3.0, 1.0);
        let mmd_same = mmd_rbf(&a, &same);
        let mmd_shift = mmd_rbf(&a, &shifted);
        assert!(mmd_shift > mmd_same, "{mmd_shift} !> {mmd_same}");
        assert!(mmd_shift > 0.05, "mmd_shift = {mmd_shift}");
    }

    // ---------- Gamma helpers ----------

    #[test]
    fn test_ln_gamma_known_values() {
        // Γ(5) = 24, Γ(1/2) = sqrt(pi).
        assert!((ln_gamma(5.0) - 24.0f64.ln()).abs() < 1e-9);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-9);
    }

    #[test]
    fn test_scale_change_detected_by_divergences() {
        // A pure variance change (same mean) should still register on the
        // distribution-aware statistics.
        let mut rng = Lcg::new(11);
        let a = rng.gaussian_sample(600, 0.0, 1.0);
        let scaled = rng.gaussian_sample(600, 0.0, 4.0);
        assert!(ks_statistic(&a, &scaled) > 0.1);
        assert!(jensen_shannon_divergence(&a, &scaled, 20) > 0.02);
        assert!(mmd_rbf(&a, &scaled) > 0.01);
    }
}
