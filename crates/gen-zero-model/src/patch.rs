//! Differential patching for [`ReflexPlugin`]: ship a small delta between two
//! trained checkpoints instead of a full weight archive, and hot-swap a
//! running plugin's weights without reloading it from disk.
//!
//! A [`ReflexPatch`] is content-addressed at both ends: `base_sha256` is the
//! [`ReflexPlugin::sha256`] the patch expects to start from, `target_sha256`
//! is the hash it promises to produce. [`ReflexPlugin::apply_patch`] and
//! [`ReflexPlugin::apply_patch_in_place`] both fail closed on every mismatch
//! (wrong base, unknown head, wrong-length delta, non-finite delta, or a
//! result that doesn't hash to `target_sha256`) rather than silently shipping
//! a plugin that isn't the one the patch author verified.
//!
//! Two apply paths, two different trade-offs:
//! - [`ReflexPlugin::apply_patch`] clones every weight buffer first, adds the
//!   deltas onto the clones, and only returns the new plugin once its hash is
//!   verified. `self` is never touched, on success or failure.
//! - [`ReflexPlugin::apply_patch_in_place`] adds the deltas directly onto
//!   `self`'s own buffers: no clone of the weight data at all. Every failure
//!   mode that can be checked without first mutating (wrong base hash,
//!   unknown head, wrong-length delta, non-finite delta) is checked before
//!   any buffer is touched, so `self` is left untouched for all of those. The
//!   one check that cannot happen before mutation is the final hash
//!   comparison (you cannot hash a result you have not yet computed): if a
//!   patch has well-formed, finite, correctly-shaped deltas that simply don't
//!   hash to the promised `target_sha256` (a mismatched or tampered patch,
//!   not a corrupt one), `apply_patch_in_place` returns `Err` with `self`
//!   already mutated into that (structurally valid, but unverified) state.
//!   The error message says so; callers that need a hard rollback guarantee
//!   should use `apply_patch` instead.
//!
//! Exactness: `base + (target - base)` is not always bit-identical to
//! `target` in f32 (it is only guaranteed when the two values are within a
//! factor of two of each other, e.g. not across a sign change, or when the
//! difference overflows). [`ReflexPatch::diff`] therefore also records, per
//! array, a sparse list of [`ExactFixup`]s: the exact target value for every
//! element whose SIMD add does not round to the target bit pattern. Apply
//! writes those values after the add, so a patch reproduces its target for
//! every pair of same-shape checkpoints, not just nearby ones.

use crate::error::ModelError;
use crate::reflex::{ReflexHead, ReflexOperator, ReflexPlugin};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const PATCH_MAGIC: &[u8; 8] = b"GZRPTCH1";
/// Archive format tag written into every patch header. v2 added the
/// [`ExactFixup`] lists; a v1 archive is rejected rather than applied
/// without them.
pub const PATCH_FORMAT: &str = "gen_zero.reflex_patch.rust.v2";

const OPERATOR_ARRAY_NAMES: [&str; 7] = [
    "w_down", "w_ctx", "w_gate", "b_gate", "w_cand", "b_cand", "w_up",
];

// ---------------------------------------------------------------------------
// SIMD primitives
// ---------------------------------------------------------------------------

/// `dst[i] += delta[i]` for every lane, 8-wide via AVX2 where available at
/// runtime, falling back to a portable scalar loop otherwise (non-x86_64
/// targets, or an x86_64 CPU without AVX2). The caller must pre-validate
/// `dst.len() == delta.len()`; this is an internal primitive, not a public
/// entry point, so it only debug-asserts the invariant instead of returning
/// `Result` (mirroring `matvec`'s convention elsewhere in this crate).
fn simd_add_assign(dst: &mut [f32], delta: &[f32]) {
    debug_assert_eq!(dst.len(), delta.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { add_assign_avx2(dst, delta) };
            return;
        }
    }
    scalar_add_assign(dst, delta);
}

fn scalar_add_assign(dst: &mut [f32], delta: &[f32]) {
    for (d, &v) in dst.iter_mut().zip(delta.iter()) {
        *d += v;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_assign_avx2(dst: &mut [f32], delta: &[f32]) {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    let n = dst.len();
    let mut i = 0;
    while i + 8 <= n {
        let a = _mm256_loadu_ps(dst.as_ptr().add(i));
        let b = _mm256_loadu_ps(delta.as_ptr().add(i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_add_ps(a, b));
        i += 8;
    }
    scalar_add_assign(&mut dst[i..], &delta[i..]);
}

/// `target[i] - base[i]` for every lane, same dispatch strategy as
/// [`simd_add_assign`]. Used only by [`ReflexPatch::diff`].
fn simd_sub(target: &[f32], base: &[f32]) -> Vec<f32> {
    debug_assert_eq!(target.len(), base.len());
    let mut out = target.to_vec();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { sub_assign_avx2(&mut out, base) };
            return out;
        }
    }
    for (o, &b) in out.iter_mut().zip(base.iter()) {
        *o -= b;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sub_assign_avx2(out: &mut [f32], base: &[f32]) {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps};
    let n = out.len();
    let mut i = 0;
    while i + 8 <= n {
        let a = _mm256_loadu_ps(out.as_ptr().add(i));
        let b = _mm256_loadu_ps(base.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_sub_ps(a, b));
        i += 8;
    }
    for j in i..n {
        out[j] -= base[j];
    }
}

/// Shape- and finiteness-check a delta before it is ever applied to a live
/// buffer. Called before *every* mutation in `apply_patch_in_place`, so a
/// corrupt or tampered delta is rejected with `self` still untouched.
fn validate_delta(label: &str, current: &[f32], delta: &[f32]) -> Result<(), ModelError> {
    if current.len() != delta.len() {
        return Err(ModelError::ReflexPatch(format!(
            "delta {label} has {} elements, expected {} to match the current array",
            delta.len(),
            current.len()
        )));
    }
    if !delta.iter().all(|v| v.is_finite()) {
        return Err(ModelError::ReflexPatch(format!(
            "delta {label} contains a NaN or Inf value"
        )));
    }
    Ok(())
}

/// `base + delta`, allocating a fresh `Vec`. Used by the allocating
/// `apply_patch` path; the length check here is the "corrupt delta" guard
/// for that path (finiteness of the *result* is checked by the
/// `ReflexOperator::new` / `ReflexHead::new` reconstruction that follows).
fn checked_add(
    label: &str,
    base: &[f32],
    delta: &[f32],
    fixups: &[ExactFixup],
) -> Result<Vec<f32>, ModelError> {
    if base.len() != delta.len() {
        return Err(ModelError::ReflexPatch(format!(
            "delta {label} has {} elements, expected {} to match the base array",
            delta.len(),
            base.len()
        )));
    }
    validate_fixups(label, base.len(), fixups)?;
    let mut out = base.to_vec();
    simd_add_assign(&mut out, delta);
    write_fixups(&mut out, fixups);
    Ok(out)
}

/// `target - base` plus the fix-ups that make `base + delta` reproduce
/// `target` bit for bit. A difference that overflows to Inf is stored as a
/// zero delta with a fix-up, so the delta array itself is always finite.
fn checked_sub(
    label: &str,
    base: &[f32],
    target: &[f32],
) -> Result<(Vec<f32>, Vec<ExactFixup>), ModelError> {
    if base.len() != target.len() {
        return Err(ModelError::ReflexPatch(format!(
            "base/target {label} length mismatch: {} vs {}",
            base.len(),
            target.len()
        )));
    }
    if u32::try_from(base.len()).is_err() {
        return Err(ModelError::ReflexPatch(format!(
            "{label} has more than u32::MAX elements"
        )));
    }
    let mut delta = simd_sub(target, base);
    let mut fixups = Vec::new();
    for (i, d) in delta.iter_mut().enumerate() {
        if !d.is_finite() {
            *d = 0.0;
        }
        if (base[i] + *d).to_bits() != target[i].to_bits() {
            fixups.push(ExactFixup {
                index: i as u32,
                value: target[i],
            });
        }
    }
    Ok((delta, fixups))
}

/// Fix-ups must name in-range, strictly increasing indices and finite
/// values; anything else is a corrupt or tampered patch.
fn validate_fixups(label: &str, len: usize, fixups: &[ExactFixup]) -> Result<(), ModelError> {
    let mut prev: Option<u32> = None;
    for f in fixups {
        if f.index as usize >= len {
            return Err(ModelError::ReflexPatch(format!(
                "fixup for {label} has index {} out of range for {len} elements",
                f.index
            )));
        }
        if prev.is_some_and(|p| f.index <= p) {
            return Err(ModelError::ReflexPatch(format!(
                "fixups for {label} are not strictly increasing at index {}",
                f.index
            )));
        }
        if !f.value.is_finite() {
            return Err(ModelError::ReflexPatch(format!(
                "fixup for {label} at index {} is NaN or Inf",
                f.index
            )));
        }
        prev = Some(f.index);
    }
    Ok(())
}

/// Callers must run [`validate_fixups`] first; indices are then in range.
fn write_fixups(arr: &mut [f32], fixups: &[ExactFixup]) {
    for f in fixups {
        arr[f.index as usize] = f.value;
    }
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn hex_decode(s: &str) -> Result<[u8; 32], ModelError> {
    if s.len() != 64 {
        return Err(ModelError::ReflexPatch(format!(
            "hash hex string has length {}, expected 64",
            s.len()
        )));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
            .map_err(|e| ModelError::ReflexPatch(format!("invalid hash hex at byte {i}: {e}")))?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ReflexPatch
// ---------------------------------------------------------------------------

/// A delta between two [`ReflexPlugin`] weight states, identified by the
/// SHA-256 of each end (see [`ReflexPlugin::sha256`]).
///
/// `head_deltas` is keyed by head name and need not cover every head in the
/// plugin: a head absent from `head_deltas` is left unchanged. A patch that
/// names a head the base plugin does not have fails closed at apply time.
#[derive(Debug, Clone, PartialEq)]
pub struct ReflexPatch {
    pub base_sha256: [u8; 32],
    pub target_sha256: [u8; 32],
    /// Free-form provenance: training run id, timestamp, notes. Not
    /// authenticated by either hash; do not use it for anything
    /// security-relevant.
    pub metadata: String,
    pub delta_w_down: Vec<f32>,
    pub delta_w_ctx: Vec<f32>,
    pub delta_w_gate: Vec<f32>,
    pub delta_b_gate: Vec<f32>,
    pub delta_w_cand: Vec<f32>,
    pub delta_b_cand: Vec<f32>,
    pub delta_w_up: Vec<f32>,
    /// Exact values for operator elements the add cannot reproduce, in
    /// [`OPERATOR_ARRAY_NAMES`] order. Empty when every add is exact.
    pub operator_fixups: [Vec<ExactFixup>; 7],
    pub head_deltas: BTreeMap<String, ReflexHeadDelta>,
}

/// The delta for one [`ReflexHead`]'s `weight` and `bias` arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct ReflexHeadDelta {
    pub delta_weight: Vec<f32>,
    pub delta_bias: Vec<f32>,
    pub weight_fixups: Vec<ExactFixup>,
    pub bias_fixups: Vec<ExactFixup>,
}

/// After the delta add, element `index` is overwritten with `value`. Used
/// only where `base + delta` does not round to the target bit pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactFixup {
    pub index: u32,
    pub value: f32,
}

impl ReflexPatch {
    /// Build a patch as `target - base`, element-wise, over every operator
    /// weight array and every head both plugins share.
    ///
    /// `base` and `target` must have identical head name sets and identical
    /// array shapes throughout (they are meant to be two checkpoints of the
    /// *same* plugin at different training steps); a structural mismatch
    /// fails closed instead of diffing a subset.
    ///
    /// `target - base` alone does not always add back to `target` bit for
    /// bit in f32 (sign changes, very different magnitudes, overflow), so
    /// every element where it does not also gets an [`ExactFixup`] holding
    /// the exact target value. The result reproduces `target_sha256` for any
    /// pair of same-shape checkpoints.
    pub fn diff(
        base: &ReflexPlugin,
        target: &ReflexPlugin,
        metadata: impl Into<String>,
    ) -> Result<Self, ModelError> {
        let base_ops = base.operator().weight_arrays();
        let target_ops = target.operator().weight_arrays();

        let (delta_w_down, fx_w_down) = checked_sub("w_down", base_ops[0], target_ops[0])?;
        let (delta_w_ctx, fx_w_ctx) = checked_sub("w_ctx", base_ops[1], target_ops[1])?;
        let (delta_w_gate, fx_w_gate) = checked_sub("w_gate", base_ops[2], target_ops[2])?;
        let (delta_b_gate, fx_b_gate) = checked_sub("b_gate", base_ops[3], target_ops[3])?;
        let (delta_w_cand, fx_w_cand) = checked_sub("w_cand", base_ops[4], target_ops[4])?;
        let (delta_b_cand, fx_b_cand) = checked_sub("b_cand", base_ops[5], target_ops[5])?;
        let (delta_w_up, fx_w_up) = checked_sub("w_up", base_ops[6], target_ops[6])?;

        let base_names: Vec<&str> = base.heads().map(|h| h.name.as_str()).collect();
        let target_names: Vec<&str> = target.heads().map(|h| h.name.as_str()).collect();
        if base_names != target_names {
            return Err(ModelError::ReflexPatch(
                "base and target plugins have different head sets; cannot diff".into(),
            ));
        }

        let mut head_deltas = BTreeMap::new();
        for base_head in base.heads() {
            let target_head = target
                .heads()
                .find(|h| h.name == base_head.name)
                .expect("head name sets were just checked equal");
            let (delta_weight, weight_fixups) = checked_sub(
                &format!("head {:?} weight", base_head.name),
                &base_head.weight,
                &target_head.weight,
            )?;
            let (delta_bias, bias_fixups) = checked_sub(
                &format!("head {:?} bias", base_head.name),
                &base_head.bias,
                &target_head.bias,
            )?;
            head_deltas.insert(
                base_head.name.clone(),
                ReflexHeadDelta {
                    delta_weight,
                    delta_bias,
                    weight_fixups,
                    bias_fixups,
                },
            );
        }

        Ok(Self {
            base_sha256: base.sha256()?,
            target_sha256: target.sha256()?,
            metadata: metadata.into(),
            delta_w_down,
            delta_w_ctx,
            delta_w_gate,
            delta_b_gate,
            delta_w_cand,
            delta_b_cand,
            delta_w_up,
            operator_fixups: [
                fx_w_down, fx_w_ctx, fx_w_gate, fx_b_gate, fx_w_cand, fx_b_cand, fx_w_up,
            ],
            head_deltas,
        })
    }

    /// Total number of [`ExactFixup`]s across every array.
    pub fn fixup_count(&self) -> usize {
        self.operator_fixups.iter().map(Vec::len).sum::<usize>()
            + self
                .head_deltas
                .values()
                .map(|d| d.weight_fixups.len() + d.bias_fixups.len())
                .sum::<usize>()
    }

    /// Serialize to a compact binary format: an 8-byte magic, an 8-byte
    /// little-endian JSON header length, the JSON header (hashes as hex,
    /// metadata, and every array's declared length), zero-padding to a
    /// 4-byte boundary, then every delta array as tightly packed
    /// little-endian `f32`, in the same fixed operator order
    /// [`ReflexPlugin::to_bytes`] uses, followed by each head's weight delta
    /// then bias delta in head-name order, then every fix-up list in the same
    /// order (operator arrays, then each head's weight and bias), each as
    /// its little-endian `u32` indices followed by its `f32` values.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ModelError> {
        let header = ReflexPatchHeader {
            format: PATCH_FORMAT.to_string(),
            base_sha256: hex_encode(&self.base_sha256),
            target_sha256: hex_encode(&self.target_sha256),
            metadata: self.metadata.clone(),
            operator_lens: [
                self.delta_w_down.len(),
                self.delta_w_ctx.len(),
                self.delta_w_gate.len(),
                self.delta_b_gate.len(),
                self.delta_w_cand.len(),
                self.delta_b_cand.len(),
                self.delta_w_up.len(),
            ],
            operator_fixup_lens: std::array::from_fn(|i| self.operator_fixups[i].len()),
            head_deltas: self
                .head_deltas
                .iter()
                .map(|(name, d)| ReflexPatchHeadEntry {
                    name: name.clone(),
                    weight_len: d.delta_weight.len(),
                    bias_len: d.delta_bias.len(),
                    weight_fixups: d.weight_fixups.len(),
                    bias_fixups: d.bias_fixups.len(),
                })
                .collect(),
        };
        let header_bytes = serde_json::to_vec(&header)
            .map_err(|e| ModelError::ReflexPatch(format!("encoding header: {e}")))?;

        let mut out = Vec::new();
        out.extend_from_slice(PATCH_MAGIC);
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        let pad = (4 - out.len() % 4) % 4;
        out.resize(out.len() + pad, 0);

        for arr in [
            &self.delta_w_down,
            &self.delta_w_ctx,
            &self.delta_w_gate,
            &self.delta_b_gate,
            &self.delta_w_cand,
            &self.delta_b_cand,
            &self.delta_w_up,
        ] {
            out.extend_from_slice(bytemuck::cast_slice(arr));
        }
        // `head_deltas` is a `BTreeMap`, so this iterates in the same sorted
        // key order the header above was built in.
        for d in self.head_deltas.values() {
            out.extend_from_slice(bytemuck::cast_slice(&d.delta_weight));
            out.extend_from_slice(bytemuck::cast_slice(&d.delta_bias));
        }
        let head_fixups = self
            .head_deltas
            .values()
            .flat_map(|d| [&d.weight_fixups, &d.bias_fixups]);
        for fixups in self.operator_fixups.iter().chain(head_fixups) {
            for f in fixups {
                out.extend_from_slice(&f.index.to_le_bytes());
            }
            for f in fixups {
                out.extend_from_slice(&f.value.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// Parse the binary format written by [`Self::to_bytes`]. Fails closed
    /// (never panics) on a truncated, misaligned, malformed-header, or
    /// trailing-garbage archive, mirroring [`ReflexPlugin::from_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        if bytes.len() < PATCH_MAGIC.len() + 8 || &bytes[..PATCH_MAGIC.len()] != PATCH_MAGIC {
            return Err(ModelError::ReflexPatch(
                "bad magic: not a gen-zero reflex patch archive".into(),
            ));
        }
        let mut cursor = PATCH_MAGIC.len();
        let header_len =
            u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().expect("8-byte slice"))
                as usize;
        cursor += 8;
        let header_end = cursor.checked_add(header_len).ok_or_else(|| {
            ModelError::ReflexPatch("header length overflows archive size".into())
        })?;
        if header_end > bytes.len() {
            return Err(ModelError::ReflexPatch(
                "archive truncated: header length exceeds buffer".into(),
            ));
        }
        let header: ReflexPatchHeader = serde_json::from_slice(&bytes[cursor..header_end])
            .map_err(|e| ModelError::ReflexPatch(format!("decoding header: {e}")))?;
        if header.format != PATCH_FORMAT {
            return Err(ModelError::ReflexPatch(format!(
                "archive format {:?} != {PATCH_FORMAT:?}",
                header.format
            )));
        }
        cursor = header_end + (4 - header_end % 4) % 4;

        // `n` 4-byte words starting at `cursor`, bounds- and alignment-checked
        // as `T` (every array in the payload is `f32` or `u32`).
        fn words<T: bytemuck::Pod>(
            bytes: &[u8],
            cursor: &mut usize,
            n: usize,
        ) -> Result<Vec<T>, ModelError> {
            let nbytes = n
                .checked_mul(4)
                .ok_or_else(|| ModelError::ReflexPatch("array length overflow".into()))?;
            let end = cursor
                .checked_add(nbytes)
                .ok_or_else(|| ModelError::ReflexPatch("array extends past archive size".into()))?;
            if end > bytes.len() {
                return Err(ModelError::ReflexPatch(
                    "archive truncated: payload shorter than declared shapes".into(),
                ));
            }
            let slice: &[T] = bytemuck::try_cast_slice(&bytes[*cursor..end]).map_err(|e| {
                ModelError::ReflexPatch(format!("misaligned array in archive: {e}"))
            })?;
            *cursor = end;
            Ok(slice.to_vec())
        }
        let mut take = |n: usize| words::<f32>(bytes, &mut cursor, n);

        let [dw_down, dw_ctx, dw_gate, db_gate, dw_cand, db_cand, dw_up] = header.operator_lens;
        let delta_w_down = take(dw_down)?;
        let delta_w_ctx = take(dw_ctx)?;
        let delta_w_gate = take(dw_gate)?;
        let delta_b_gate = take(db_gate)?;
        let delta_w_cand = take(dw_cand)?;
        let delta_b_cand = take(db_cand)?;
        let delta_w_up = take(dw_up)?;

        let mut head_arrays = Vec::with_capacity(header.head_deltas.len());
        for entry in &header.head_deltas {
            head_arrays.push((take(entry.weight_len)?, take(entry.bias_len)?));
        }

        // Fix-up lists: `n` u32 indices then `n` f32 values.
        let mut take_fixups = |n: usize| -> Result<Vec<ExactFixup>, ModelError> {
            let indices = words::<u32>(bytes, &mut cursor, n)?;
            let values = words::<f32>(bytes, &mut cursor, n)?;
            Ok(indices
                .into_iter()
                .zip(values)
                .map(|(index, value)| ExactFixup { index, value })
                .collect())
        };
        let mut operator_fixups: [Vec<ExactFixup>; 7] = Default::default();
        for (slot, &n) in operator_fixups.iter_mut().zip(&header.operator_fixup_lens) {
            *slot = take_fixups(n)?;
        }

        let mut head_deltas = BTreeMap::new();
        for (entry, (delta_weight, delta_bias)) in header.head_deltas.iter().zip(head_arrays) {
            let weight_fixups = take_fixups(entry.weight_fixups)?;
            let bias_fixups = take_fixups(entry.bias_fixups)?;
            if head_deltas
                .insert(
                    entry.name.clone(),
                    ReflexHeadDelta {
                        delta_weight,
                        delta_bias,
                        weight_fixups,
                        bias_fixups,
                    },
                )
                .is_some()
            {
                return Err(ModelError::ReflexPatch(format!(
                    "duplicate head {:?} in patch archive",
                    entry.name
                )));
            }
        }

        if cursor != bytes.len() {
            return Err(ModelError::ReflexPatch(format!(
                "archive has {} unexpected trailing bytes",
                bytes.len() - cursor
            )));
        }

        Ok(Self {
            base_sha256: hex_decode(&header.base_sha256)?,
            target_sha256: hex_decode(&header.target_sha256)?,
            metadata: header.metadata,
            delta_w_down,
            delta_w_ctx,
            delta_w_gate,
            delta_b_gate,
            delta_w_cand,
            delta_b_cand,
            delta_w_up,
            operator_fixups,
            head_deltas,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReflexPatchHeader {
    format: String,
    base_sha256: String,
    target_sha256: String,
    metadata: String,
    operator_lens: [usize; 7],
    operator_fixup_lens: [usize; 7],
    head_deltas: Vec<ReflexPatchHeadEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReflexPatchHeadEntry {
    name: String,
    weight_len: usize,
    bias_len: usize,
    weight_fixups: usize,
    bias_fixups: usize,
}

// ---------------------------------------------------------------------------
// ReflexPlugin::apply_patch / apply_patch_in_place
// ---------------------------------------------------------------------------

impl ReflexPlugin {
    /// Clone every weight buffer, add `patch`'s deltas onto the clones, and
    /// return the result as a new, independently verified `ReflexPlugin`.
    /// `self` is never mutated, on success or on error. See the module docs
    /// for how this differs from [`Self::apply_patch_in_place`].
    pub fn apply_patch(&self, patch: &ReflexPatch) -> Result<ReflexPlugin, ModelError> {
        let base_hash = self.sha256()?;
        if base_hash != patch.base_sha256 {
            return Err(ModelError::ReflexPatch(format!(
                "base hash mismatch: plugin is {}, patch expects {}",
                hex_encode(&base_hash),
                hex_encode(&patch.base_sha256)
            )));
        }
        for name in patch.head_deltas.keys() {
            if !self.heads().any(|h| &h.name == name) {
                return Err(ModelError::ReflexPatch(format!(
                    "patch references unknown head {name:?}"
                )));
            }
        }

        let cfg = self.operator().config();
        let ops = self.operator().weight_arrays();
        let new_operator = ReflexOperator::new(
            cfg,
            checked_add(
                "w_down",
                ops[0],
                &patch.delta_w_down,
                &patch.operator_fixups[0],
            )?,
            checked_add(
                "w_ctx",
                ops[1],
                &patch.delta_w_ctx,
                &patch.operator_fixups[1],
            )?,
            checked_add(
                "w_gate",
                ops[2],
                &patch.delta_w_gate,
                &patch.operator_fixups[2],
            )?,
            checked_add(
                "b_gate",
                ops[3],
                &patch.delta_b_gate,
                &patch.operator_fixups[3],
            )?,
            checked_add(
                "w_cand",
                ops[4],
                &patch.delta_w_cand,
                &patch.operator_fixups[4],
            )?,
            checked_add(
                "b_cand",
                ops[5],
                &patch.delta_b_cand,
                &patch.operator_fixups[5],
            )?,
            checked_add("w_up", ops[6], &patch.delta_w_up, &patch.operator_fixups[6])?,
        )?;

        let mut new_heads: Vec<ReflexHead> = Vec::new();
        for head in self.heads() {
            let (weight, bias) = match patch.head_deltas.get(&head.name) {
                Some(d) => (
                    checked_add(
                        &format!("head {:?} weight", head.name),
                        &head.weight,
                        &d.delta_weight,
                        &d.weight_fixups,
                    )?,
                    checked_add(
                        &format!("head {:?} bias", head.name),
                        &head.bias,
                        &d.delta_bias,
                        &d.bias_fixups,
                    )?,
                ),
                None => (head.weight.clone(), head.bias.clone()),
            };
            new_heads.push(ReflexHead::new(
                head.name.clone(),
                cfg.input_dim,
                weight,
                bias,
                head.candidates.clone(),
            )?);
        }

        let new_plugin = ReflexPlugin::new(
            self.task().to_string(),
            new_operator,
            new_heads,
            self.default_head().map(|s| s.to_string()),
        )?;

        let final_hash = new_plugin.sha256()?;
        if final_hash != patch.target_sha256 {
            return Err(ModelError::ReflexPatch(format!(
                "target hash mismatch after applying patch: got {}, expected {}",
                hex_encode(&final_hash),
                hex_encode(&patch.target_sha256)
            )));
        }
        Ok(new_plugin)
    }

    /// Add `patch`'s deltas directly onto `self`'s own weight buffers: no
    /// clone of the weight data. Every check that does not require having
    /// already computed the result (base hash, unknown head, wrong-length
    /// delta, non-finite delta) runs before any buffer is touched, so `self`
    /// is left untouched by all of those failure modes. See the module docs
    /// for the one failure mode (a well-formed patch whose target hash
    /// doesn't match) that leaves `self` mutated on error.
    pub fn apply_patch_in_place(&mut self, patch: &ReflexPatch) -> Result<(), ModelError> {
        let base_hash = self.sha256()?;
        if base_hash != patch.base_sha256 {
            return Err(ModelError::ReflexPatch(format!(
                "base hash mismatch: plugin is {}, patch expects {}",
                hex_encode(&base_hash),
                hex_encode(&patch.base_sha256)
            )));
        }
        for name in patch.head_deltas.keys() {
            if !self.heads().any(|h| &h.name == name) {
                return Err(ModelError::ReflexPatch(format!(
                    "patch references unknown head {name:?}"
                )));
            }
        }

        let operator_deltas: [&[f32]; 7] = [
            &patch.delta_w_down,
            &patch.delta_w_ctx,
            &patch.delta_w_gate,
            &patch.delta_b_gate,
            &patch.delta_w_cand,
            &patch.delta_b_cand,
            &patch.delta_w_up,
        ];
        {
            let current = self.operator().weight_arrays();
            for (((name, cur), delta), fixups) in OPERATOR_ARRAY_NAMES
                .into_iter()
                .zip(current)
                .zip(operator_deltas)
                .zip(&patch.operator_fixups)
            {
                validate_delta(name, cur, delta)?;
                validate_fixups(name, cur.len(), fixups)?;
            }
        }
        for (name, d) in &patch.head_deltas {
            let head = self
                .heads()
                .find(|h| &h.name == name)
                .expect("head presence already checked above");
            validate_delta(
                &format!("head {name:?} weight"),
                &head.weight,
                &d.delta_weight,
            )?;
            validate_delta(&format!("head {name:?} bias"), &head.bias, &d.delta_bias)?;
            validate_fixups(
                &format!("head {name:?} weight"),
                head.weight.len(),
                &d.weight_fixups,
            )?;
            validate_fixups(
                &format!("head {name:?} bias"),
                head.bias.len(),
                &d.bias_fixups,
            )?;
        }

        // Every precondition checked above guarantees equal lengths and
        // finite deltas from here on, so none of this can panic.
        {
            let operator = self.operator_mut();
            let current = operator.weight_arrays_mut();
            for ((arr, delta), fixups) in current
                .into_iter()
                .zip(operator_deltas)
                .zip(&patch.operator_fixups)
            {
                simd_add_assign(arr, delta);
                write_fixups(arr, fixups);
            }
        }
        for (name, d) in &patch.head_deltas {
            let head = self
                .head_mut(name)
                .expect("head presence already checked above");
            simd_add_assign(&mut head.weight, &d.delta_weight);
            write_fixups(&mut head.weight, &d.weight_fixups);
            simd_add_assign(&mut head.bias, &d.delta_bias);
            write_fixups(&mut head.bias, &d.bias_fixups);
        }

        // Defense in depth: finite + finite can still overflow to Inf. This
        // is the one check that cannot run before mutation, since it checks
        // the result of the mutation itself.
        for (name, arr) in OPERATOR_ARRAY_NAMES
            .into_iter()
            .zip(self.operator().weight_arrays())
        {
            if !arr.iter().all(|v| v.is_finite()) {
                return Err(ModelError::NumericalInstability(format!(
                    "reflex patch produced non-finite operator weight {name}"
                )));
            }
        }
        for head in self.heads() {
            if !head.weight.iter().all(|v| v.is_finite())
                || !head.bias.iter().all(|v| v.is_finite())
            {
                return Err(ModelError::NumericalInstability(format!(
                    "reflex patch produced non-finite weights in head {:?}",
                    head.name
                )));
            }
        }

        let final_hash = self.sha256()?;
        if final_hash != patch.target_sha256 {
            return Err(ModelError::ReflexPatch(format!(
                "target hash mismatch after in-place patch: got {}, expected {}. \
                 Weights are finite and correctly shaped but do not match the patch's \
                 declared target; self has been mutated in place and must be discarded.",
                hex_encode(&final_hash),
                hex_encode(&patch.target_sha256)
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reflex::ReflexOperatorConfig;

    /// Deterministic pseudo-random f32 generator (xorshift32), matching the
    /// one in `reflex.rs`'s own test module (not reused directly: that one
    /// is private to `reflex.rs`).
    struct Xorshift32(u32);
    impl Xorshift32 {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    fn random_operator(input_dim: usize, rank: usize, seed: u32) -> ReflexOperator {
        let mut rng = Xorshift32(seed);
        let config = ReflexOperatorConfig {
            input_dim,
            lora_rank: rank,
            alpha: 0.5,
            steps: 4,
            epsilon: 1e-6,
        };
        ReflexOperator::new(
            config,
            rng.vec(input_dim * rank),
            rng.vec(input_dim * rank),
            rng.vec(2 * rank * rank),
            rng.vec(rank),
            rng.vec(2 * rank * rank),
            rng.vec(rank),
            rng.vec(rank * input_dim),
        )
        .unwrap()
    }

    fn random_plugin(seed: u32) -> ReflexPlugin {
        let op = random_operator(32, 6, seed);
        let mut rng = Xorshift32(seed ^ 0xABCD);
        let head_a = ReflexHead::new(
            "a",
            32,
            rng.vec(3 * 32),
            rng.vec(3),
            vec!["x".into(), "y".into(), "z".into()],
        )
        .unwrap();
        let head_b = ReflexHead::new(
            "b",
            32,
            rng.vec(2 * 32),
            rng.vec(2),
            vec!["p".into(), "q".into()],
        )
        .unwrap();
        ReflexPlugin::new("multi", op, vec![head_a, head_b], Some("a".into())).unwrap()
    }

    /// Elementwise `base + delta`, computed with a plain scalar loop
    /// independent of `simd_add_assign`, so a test that compares against
    /// this is a real bit-exactness check, not a tautology.
    fn reference_add(base: &[f32], delta: &[f32]) -> Vec<f32> {
        base.iter()
            .zip(delta.iter())
            .map(|(&b, &d)| b + d)
            .collect()
    }

    /// Build a target plugin as `base + delta` for every array, using
    /// `reference_add`, plus a `ReflexPatch` carrying exactly those deltas
    /// with correct base/target hashes. Deltas are drawn from the same
    /// magnitude range as the base weights, so this exercises the common,
    /// well-behaved case (this is a single-add construction, not a
    /// diff-then-add round trip, so it carries no floating-point round-trip
    /// risk regardless of magnitude).
    fn base_target_and_patch(seed: u32) -> (ReflexPlugin, ReflexPlugin, ReflexPatch) {
        let base = random_plugin(seed);
        let mut rng = Xorshift32(seed ^ 0x1234);

        let ops = base.operator().weight_arrays();
        let delta_w_down = rng.vec(ops[0].len());
        let delta_w_ctx = rng.vec(ops[1].len());
        let delta_w_gate = rng.vec(ops[2].len());
        let delta_b_gate = rng.vec(ops[3].len());
        let delta_w_cand = rng.vec(ops[4].len());
        let delta_b_cand = rng.vec(ops[5].len());
        let delta_w_up = rng.vec(ops[6].len());

        let target_operator = ReflexOperator::new(
            base.operator().config(),
            reference_add(ops[0], &delta_w_down),
            reference_add(ops[1], &delta_w_ctx),
            reference_add(ops[2], &delta_w_gate),
            reference_add(ops[3], &delta_b_gate),
            reference_add(ops[4], &delta_w_cand),
            reference_add(ops[5], &delta_b_cand),
            reference_add(ops[6], &delta_w_up),
        )
        .unwrap();

        let mut head_deltas = BTreeMap::new();
        let mut target_heads = Vec::new();
        for head in base.heads() {
            let delta_weight = rng.vec(head.weight.len());
            let delta_bias = rng.vec(head.bias.len());
            target_heads.push(
                ReflexHead::new(
                    head.name.clone(),
                    base.input_dim(),
                    reference_add(&head.weight, &delta_weight),
                    reference_add(&head.bias, &delta_bias),
                    head.candidates.clone(),
                )
                .unwrap(),
            );
            head_deltas.insert(
                head.name.clone(),
                ReflexHeadDelta {
                    delta_weight,
                    delta_bias,
                    weight_fixups: Vec::new(),
                    bias_fixups: Vec::new(),
                },
            );
        }

        let target = ReflexPlugin::new(
            base.task().to_string(),
            target_operator,
            target_heads,
            base.default_head().map(|s| s.to_string()),
        )
        .unwrap();

        let patch = ReflexPatch {
            base_sha256: base.sha256().unwrap(),
            target_sha256: target.sha256().unwrap(),
            metadata: "test patch".into(),
            delta_w_down,
            delta_w_ctx,
            delta_w_gate,
            delta_b_gate,
            delta_w_cand,
            delta_b_cand,
            delta_w_up,
            operator_fixups: Default::default(),
            head_deltas,
        };

        (base, target, patch)
    }

    // -- creation / application cycle: W_base + delta == W_target --------

    #[test]
    fn apply_patch_matches_independently_built_target_bit_exact() {
        let (base, target, patch) = base_target_and_patch(11);
        let patched = base.apply_patch(&patch).unwrap();

        assert_eq!(
            patched.operator().weight_arrays(),
            target.operator().weight_arrays()
        );
        for (ph, th) in patched.heads().zip(target.heads()) {
            assert_eq!(ph.weight, th.weight, "head {:?} weight", ph.name);
            assert_eq!(ph.bias, th.bias, "head {:?} bias", ph.name);
        }
        assert_eq!(patched.sha256().unwrap(), target.sha256().unwrap());
    }

    #[test]
    fn apply_patch_output_matches_target_output_bit_exact() {
        let (base, target, patch) = base_target_and_patch(22);
        let patched = base.apply_patch(&patch).unwrap();

        let mut rng = Xorshift32(999);
        let x = rng.vec(32);
        let (logits_patched, _) = patched.forward(&x, Some("a")).unwrap();
        let (logits_target, _) = target.forward(&x, Some("a")).unwrap();
        assert_eq!(logits_patched, logits_target);
    }

    #[test]
    fn apply_patch_in_place_matches_apply_patch_and_target() {
        let (base, target, patch) = base_target_and_patch(33);
        let mut in_place = base.clone();
        in_place.apply_patch_in_place(&patch).unwrap();

        assert_eq!(in_place.sha256().unwrap(), target.sha256().unwrap());
        assert_eq!(
            in_place.operator().weight_arrays(),
            target.operator().weight_arrays()
        );
        for (ih, th) in in_place.heads().zip(target.heads()) {
            assert_eq!(ih.weight, th.weight);
            assert_eq!(ih.bias, th.bias);
        }
    }

    #[test]
    fn apply_patch_leaves_base_untouched() {
        let (base, _target, patch) = base_target_and_patch(44);
        let base_hash_before = base.sha256().unwrap();
        let _patched = base.apply_patch(&patch).unwrap();
        assert_eq!(base.sha256().unwrap(), base_hash_before);
    }

    // -- fail-closed: wrong base hash -------------------------------------

    #[test]
    fn apply_patch_fails_closed_on_wrong_base_hash() {
        let (base, _target, mut patch) = base_target_and_patch(55);
        patch.base_sha256[0] ^= 0xFF;
        let err = base.apply_patch(&patch).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
    }

    #[test]
    fn apply_patch_in_place_fails_closed_on_wrong_base_hash_and_leaves_self_untouched() {
        let (mut base, _target, mut patch) = base_target_and_patch(56);
        patch.base_sha256[0] ^= 0xFF;
        let hash_before = base.sha256().unwrap();
        let err = base.apply_patch_in_place(&patch).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
        assert_eq!(base.sha256().unwrap(), hash_before);
    }

    // -- fail-closed: tampered target hash --------------------------------

    #[test]
    fn apply_patch_fails_closed_on_tampered_target_hash() {
        let (base, _target, mut patch) = base_target_and_patch(66);
        patch.target_sha256[0] ^= 0xFF;
        let err = base.apply_patch(&patch).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
    }

    #[test]
    fn apply_patch_in_place_fails_closed_on_tampered_target_hash() {
        let (mut base, _target, mut patch) = base_target_and_patch(67);
        patch.target_sha256[0] ^= 0xFF;
        // This is the one documented case where a failure leaves `self`
        // mutated: the deltas are well-formed, so the error only surfaces
        // once the (now-wrong) target hash is compared.
        let err = base.apply_patch_in_place(&patch).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
    }

    // -- fail-closed: corrupt (wrong-length) delta ------------------------

    #[test]
    fn apply_patch_fails_closed_on_corrupt_delta_length() {
        let (base, _target, mut patch) = base_target_and_patch(77);
        patch.delta_w_down.pop();
        let err = base.apply_patch(&patch).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
    }

    #[test]
    fn apply_patch_in_place_fails_closed_on_corrupt_delta_length_and_leaves_self_untouched() {
        let (mut base, _target, mut patch) = base_target_and_patch(78);
        patch.delta_w_up.pop();
        let hash_before = base.sha256().unwrap();
        let err = base.apply_patch_in_place(&patch).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
        assert_eq!(base.sha256().unwrap(), hash_before);
    }

    // -- fail-closed: NaN in a delta ---------------------------------------

    #[test]
    fn apply_patch_fails_closed_on_nan_delta() {
        let (base, _target, mut patch) = base_target_and_patch(88);
        patch.delta_w_gate[0] = f32::NAN;
        assert!(base.apply_patch(&patch).is_err());
    }

    #[test]
    fn apply_patch_in_place_fails_closed_on_nan_delta_and_leaves_self_untouched() {
        let (mut base, _target, mut patch) = base_target_and_patch(89);
        let head_name = patch.head_deltas.keys().next().unwrap().clone();
        patch.head_deltas.get_mut(&head_name).unwrap().delta_weight[0] = f32::NAN;
        let hash_before = base.sha256().unwrap();
        let err = base.apply_patch_in_place(&patch).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
        assert_eq!(base.sha256().unwrap(), hash_before);
    }

    // -- fail-closed: patch names a head the plugin doesn't have ----------

    #[test]
    fn apply_patch_fails_closed_on_unknown_head() {
        let (base, _target, mut patch) = base_target_and_patch(90);
        patch.head_deltas.insert(
            "does-not-exist".into(),
            ReflexHeadDelta {
                delta_weight: vec![0.0; 32],
                delta_bias: vec![0.0; 1],
                weight_fixups: Vec::new(),
                bias_fixups: Vec::new(),
            },
        );
        let err = base.apply_patch(&patch).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
    }

    #[test]
    fn apply_patch_in_place_fails_closed_on_unknown_head_and_leaves_self_untouched() {
        let (mut base, _target, mut patch) = base_target_and_patch(91);
        patch.head_deltas.insert(
            "does-not-exist".into(),
            ReflexHeadDelta {
                delta_weight: vec![0.0; 32],
                delta_bias: vec![0.0; 1],
                weight_fixups: Vec::new(),
                bias_fixups: Vec::new(),
            },
        );
        let hash_before = base.sha256().unwrap();
        let err = base.apply_patch_in_place(&patch).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
        assert_eq!(base.sha256().unwrap(), hash_before);
    }

    // -- ReflexPatch::diff round trip --------------------------------------

    #[test]
    fn diff_then_apply_recovers_target() {
        let base = random_plugin(101);
        let mut rng = Xorshift32(202);
        // Small, base-magnitude-comparable perturbation: well inside the
        // regime where float subtraction and addition are exact inverses
        // (see the caveat on `ReflexPatch::diff`).
        let ops = base.operator().weight_arrays();
        let bumped_w_up: Vec<f32> = ops[6].iter().map(|&v| v + rng.next_f32() * 0.01).collect();
        let target_operator = ReflexOperator::new(
            base.operator().config(),
            ops[0].to_vec(),
            ops[1].to_vec(),
            ops[2].to_vec(),
            ops[3].to_vec(),
            ops[4].to_vec(),
            ops[5].to_vec(),
            bumped_w_up,
        )
        .unwrap();
        let target_heads: Vec<ReflexHead> = base
            .heads()
            .map(|h| {
                ReflexHead::new(
                    h.name.clone(),
                    base.input_dim(),
                    h.weight.clone(),
                    h.bias.clone(),
                    h.candidates.clone(),
                )
                .unwrap()
            })
            .collect();
        let target = ReflexPlugin::new(
            base.task().to_string(),
            target_operator,
            target_heads,
            base.default_head().map(|s| s.to_string()),
        )
        .unwrap();

        let patch = ReflexPatch::diff(&base, &target, "diff test").unwrap();
        let patched = base.apply_patch(&patch).unwrap();
        assert_eq!(patched.sha256().unwrap(), target.sha256().unwrap());
    }

    /// Two independently initialized checkpoints: weights differ in sign and
    /// magnitude, so plain `base + (target - base)` is not bit-exact for a
    /// large share of elements. The fix-ups must close every gap, on both
    /// apply paths and through the binary format.
    #[test]
    fn diff_of_independent_checkpoints_round_trips_exactly() {
        let base = random_plugin(7);
        let target = random_plugin(8);
        let patch = ReflexPatch::diff(&base, &target, "independent").unwrap();
        assert!(
            patch.fixup_count() > 0,
            "fixture must exercise inexact adds, got 0 fix-ups"
        );

        let patched = base.apply_patch(&patch).unwrap();
        assert_eq!(patched.sha256().unwrap(), target.sha256().unwrap());

        let mut in_place = random_plugin(7);
        in_place.apply_patch_in_place(&patch).unwrap();
        assert_eq!(in_place.sha256().unwrap(), target.sha256().unwrap());

        let loaded = ReflexPatch::from_bytes(&patch.to_bytes().unwrap()).unwrap();
        assert_eq!(loaded, patch);
        assert_eq!(
            base.apply_patch(&loaded).unwrap().sha256().unwrap(),
            target.sha256().unwrap()
        );
    }

    #[test]
    fn checked_sub_fixes_sign_crossing_negative_zero_and_overflow() {
        let base = [1.0e-3_f32, 0.25, 3.0e38, 1.0];
        let target = [-7.0e-9_f32, -0.0, -3.0e38, 1.5];
        let (delta, fixups) = checked_sub("t", &base, &target).unwrap();
        assert!(delta.iter().all(|d| d.is_finite()));
        let mut out = base.to_vec();
        simd_add_assign(&mut out, &delta);
        write_fixups(&mut out, &fixups);
        let bits = |a: &[f32]| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&out), bits(&target));
        // 1.0 -> 1.5 is exact, so it must not carry a fix-up.
        assert!(fixups.iter().all(|f| f.index != 3));
    }

    fn patch_with_one_fixup(fixup: ExactFixup) -> (ReflexPlugin, ReflexPatch) {
        let (base, _target, mut patch) = base_target_and_patch(404);
        patch.operator_fixups[0] = vec![fixup];
        (base, patch)
    }

    #[test]
    fn tampered_fixups_fail_closed_on_both_apply_paths() {
        let bad = [
            vec![ExactFixup {
                index: u32::MAX,
                value: 0.0,
            }],
            vec![ExactFixup {
                index: 0,
                value: f32::NAN,
            }],
        ];
        for fixups in bad {
            let (base, mut patch) = patch_with_one_fixup(fixups[0]);
            patch.operator_fixups[0] = fixups;
            let err = base.apply_patch(&patch).unwrap_err();
            assert!(matches!(err, ModelError::ReflexPatch(_)), "{err:?}");

            let mut in_place = base.clone();
            let hash_before = in_place.sha256().unwrap();
            let err = in_place.apply_patch_in_place(&patch).unwrap_err();
            assert!(matches!(err, ModelError::ReflexPatch(_)), "{err:?}");
            assert_eq!(in_place.sha256().unwrap(), hash_before);
        }

        let (base, mut patch) = patch_with_one_fixup(ExactFixup {
            index: 2,
            value: 0.0,
        });
        patch.operator_fixups[0].push(ExactFixup {
            index: 1,
            value: 0.0,
        });
        let err = base.apply_patch(&patch).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)), "{err:?}");
    }

    // -- binary round trip --------------------------------------------------

    #[test]
    fn patch_binary_round_trip_applies_identically() {
        let (base, target, patch) = base_target_and_patch(303);
        let bytes = patch.to_bytes().unwrap();
        let loaded = ReflexPatch::from_bytes(&bytes).unwrap();
        assert_eq!(loaded, patch);

        let patched = base.apply_patch(&loaded).unwrap();
        assert_eq!(patched.sha256().unwrap(), target.sha256().unwrap());
    }

    #[test]
    fn patch_from_bytes_rejects_bad_magic() {
        let (_base, _target, patch) = base_target_and_patch(304);
        let mut bytes = patch.to_bytes().unwrap();
        bytes[0] ^= 0xFF;
        let err = ReflexPatch::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
    }

    #[test]
    fn patch_from_bytes_rejects_truncated_payload() {
        let (_base, _target, patch) = base_target_and_patch(305);
        let bytes = patch.to_bytes().unwrap();
        let truncated = &bytes[..bytes.len() - 4];
        let err = ReflexPatch::from_bytes(truncated).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
    }

    #[test]
    fn patch_from_bytes_rejects_trailing_garbage() {
        let (_base, _target, patch) = base_target_and_patch(306);
        let mut bytes = patch.to_bytes().unwrap();
        bytes.extend_from_slice(&[0u8; 4]);
        let err = ReflexPatch::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, ModelError::ReflexPatch(_)));
    }
}
