//! Text to `(MixedCurvatureCoord, 256-bit HDC fingerprint)` projection.
//!
//! [`TextEmbeddingProjector`] is a lexical hashing embedding, not a learned
//! semantic one. Two texts land close when they share words, word pairs and
//! character trigrams; paraphrases with no shared n-grams do not. It is
//! deterministic, needs no model file and no corpus statistics (there is no
//! IDF: every n-gram kind has the same weight).
//!
//! The pipeline:
//!
//! 1. Normalize: lowercase, split on every character that is not alphanumeric.
//! 2. Features: word unigrams, adjacent word bigrams, and character trigrams of
//!    each word padded with `^` and `$`. Each feature is hashed with keyed
//!    BLAKE3 (the key is derived from [`PROJECTOR_VERSION`]) and weighted
//!    `1 + ln(tf)`.
//! 3. Fingerprint: Charikar SimHash. Every feature hash seeds 256 pseudo-random
//!    signs; bit `i` is set when the weighted sum of sign `i` is positive. The
//!    expected Hamming distance of two fingerprints is `256 * theta / pi`,
//!    `theta` the angle between the two weighted feature vectors.
//! 4. Coordinate: a 16-row random projection of the same feature vector with
//!    entries uniform in `[-1, 1]`, divided by the vector's norm and scaled to
//!    unit variance (`u`). Then
//!    - `H^4`: `u[0..4]` read as the space part of a point on the hyperboloid
//!      of curvature `-c`, mapped to the Poincare ball by stereographic
//!      projection, `x = u / (1 + sqrt(1 + c |u|^2))`, so `sqrt(c) |x| < 1`;
//!    - `S^3`: `u[4..8]` as a direction. The chart stores the unit direction;
//!      the graph scales it to the sphere radius `R` in
//!      [`MixedCurvatureCoord::to_point`];
//!    - `R^8`: `tanh(u[8..16])`, bounded in `(-1, 1)`.
//!
//! The hyperbolic depth of a projected point is a hash artifact: it carries no
//! hierarchy, so the band it implies means nothing.
//!
//! Every failure is an error: blank text, text with no alphanumeric token, text
//! over [`MAX_PAYLOAD_BYTES`], and a coordinate the chart refuses (for example a
//! spherical block of norm ~0, or a hyperbolic block past the boundary floor of
//! a very steep curvature). Nothing is clamped or replaced.

use crate::error::LodError;
use crate::manifold::MixedCurvatureCoord;
use crate::node::MAX_PAYLOAD_BYTES;

/// Identity of the projection. A text projected under another version lands
/// somewhere else, so stored nodes and queries must share it.
pub const PROJECTOR_VERSION: &str = "gen-zero-lod/lexical-ngram-simhash/v1";

/// Bits in an HDC fingerprint.
pub const HDC_BITS: usize = 256;

/// Random-projection streams per feature: 4 words of SimHash signs, then one
/// per coordinate (4 hyperbolic, 4 spherical, 8 Euclidean).
const SIGN_WORDS: u64 = 4;
const COORD_ROWS: usize = 16;

/// Feature kinds, hashed as a prefix so a word and a trigram never collide.
const UNIGRAM: u8 = 1;
const BIGRAM: u8 = 2;
const TRIGRAM: u8 = 3;

/// Deterministic text projector for one ball curvature.
#[derive(Clone, Debug)]
pub struct TextEmbeddingProjector {
    curvature: f32,
    hasher: blake3::Hasher,
}

impl TextEmbeddingProjector {
    /// A projector onto the Poincare ball of curvature `-curvature`. The
    /// curvature must be finite and positive.
    pub fn new(curvature: f32) -> Result<Self, LodError> {
        if !(curvature.is_finite() && curvature > 0.0) {
            return Err(crate::manifold::Reject::DomainViolation.into());
        }
        let key = blake3::hash(PROJECTOR_VERSION.as_bytes());
        Ok(Self {
            curvature,
            hasher: blake3::Hasher::new_keyed(key.as_bytes()),
        })
    }

    /// The ball curvature this projector targets.
    pub fn curvature(&self) -> f32 {
        self.curvature
    }

    /// Project `text` to a chart coordinate and a 256-bit SimHash fingerprint.
    /// See the module docs for the construction and the refusals.
    pub fn project_text(&self, text: &str) -> Result<(MixedCurvatureCoord, [u64; 4]), LodError> {
        let features = self.features(text)?;
        let mut signs = [0.0_f64; HDC_BITS];
        let mut rows = [0.0_f64; COORD_ROWS];
        let mut norm_sq = 0.0_f64;
        // `features` is sorted by hash, so the float sums run in one fixed order.
        for &(hash, weight) in &features {
            for word in 0..SIGN_WORDS {
                let bits = mix(hash, word);
                for bit in 0..64 {
                    let sign = if bits >> bit & 1 == 1 {
                        weight
                    } else {
                        -weight
                    };
                    signs[word as usize * 64 + bit] += sign;
                }
            }
            for (row, acc) in rows.iter_mut().enumerate() {
                *acc += weight * uniform(mix(hash, SIGN_WORDS + row as u64));
            }
            norm_sq += weight * weight;
        }

        let mut fingerprint = [0_u64; 4];
        for (i, &s) in signs.iter().enumerate() {
            if s > 0.0 {
                fingerprint[i / 64] |= 1 << (i % 64);
            }
        }

        // A uniform [-1, 1] entry has variance 1/3.
        let scale = 3.0_f64.sqrt() / norm_sq.sqrt();
        let u = rows.map(|r| r * scale);
        let c = f64::from(self.curvature);
        let h_norm_sq: f64 = u[0..4].iter().map(|v| v * v).sum();
        let denom = 1.0 + (1.0 + c * h_norm_sq).sqrt();
        let hyperbolic = [0, 1, 2, 3].map(|i| (u[i] / denom) as f32);
        let spherical = [4, 5, 6, 7].map(|i| u[i] as f32);
        let euclidean = [8, 9, 10, 11, 12, 13, 14, 15].map(|i| u[i].tanh() as f32);
        let coord =
            MixedCurvatureCoord::with_curvature(hyperbolic, spherical, euclidean, self.curvature)?;
        Ok((coord, fingerprint))
    }

    /// Weighted features of `text`, sorted by hash, one entry per distinct hash.
    fn features(&self, text: &str) -> Result<Vec<(u64, f64)>, LodError> {
        if text.len() > MAX_PAYLOAD_BYTES {
            return Err(LodError::PayloadTooLarge {
                len: text.len(),
                max: MAX_PAYLOAD_BYTES,
            });
        }
        if text.trim().is_empty() {
            return Err(LodError::EmptyInput("text is blank".into()));
        }
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return Err(LodError::EmptyInput(
                "text has no alphanumeric token".into(),
            ));
        }

        let mut hashes = Vec::with_capacity(tokens.len() * 8);
        for (i, token) in tokens.iter().enumerate() {
            hashes.push(self.hash(UNIGRAM, &[token.as_bytes()]));
            if let Some(next) = tokens.get(i + 1) {
                hashes.push(self.hash(BIGRAM, &[token.as_bytes(), &[0], next.as_bytes()]));
            }
            let padded: Vec<char> = std::iter::once('^')
                .chain(token.chars())
                .chain(std::iter::once('$'))
                .collect();
            for window in padded.windows(3) {
                let trigram: String = window.iter().collect();
                hashes.push(self.hash(TRIGRAM, &[trigram.as_bytes()]));
            }
        }
        hashes.sort_unstable();

        let mut features: Vec<(u64, f64)> = Vec::with_capacity(hashes.len());
        let mut i = 0;
        while i < hashes.len() {
            let run = hashes[i..].iter().take_while(|&&h| h == hashes[i]).count();
            features.push((hashes[i], 1.0 + (run as f64).ln()));
            i += run;
        }
        Ok(features)
    }

    fn hash(&self, kind: u8, parts: &[&[u8]]) -> u64 {
        let mut hasher = self.hasher.clone();
        hasher.update(&[kind]);
        for part in parts {
            hasher.update(part);
        }
        let digest = hasher.finalize();
        let mut word = [0_u8; 8];
        word.copy_from_slice(&digest.as_bytes()[..8]);
        u64::from_le_bytes(word)
    }
}

/// Lowercased maximal runs of alphanumeric characters.
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current.extend(ch.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// SplitMix64 finalizer of `hash` on stream `stream`.
fn mix(hash: u64, stream: u64) -> u64 {
    let mut z = hash.wrapping_add((stream + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The top 53 bits of `bits` as a uniform value in `[-1, 1)`.
fn uniform(bits: u64) -> f64 {
    (bits >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::GeometryParams;
    use crate::node::hdc_hamming_distance_256;

    fn project(text: &str) -> (MixedCurvatureCoord, [u64; 4]) {
        TextEmbeddingProjector::new(1.0)
            .unwrap()
            .project_text(text)
            .unwrap()
    }

    fn distance(a: &MixedCurvatureCoord, b: &MixedCurvatureCoord) -> f32 {
        a.product_distance_with_params(b, [1.0; 3], 1.0, 1.0)
            .unwrap()
    }

    #[test]
    fn projection_is_deterministic_and_ignores_case_and_spacing() {
        let (a, fa) = project("Water boils at 100 degrees Celsius.");
        let (b, fb) = project("Water boils at 100 degrees Celsius.");
        assert_eq!(fa, fb);
        assert_eq!(a.hyperbolic, b.hyperbolic);
        assert_eq!(a.spherical, b.spherical);
        assert_eq!(a.euclidean, b.euclidean);
        let (c, fc) = project("  water   BOILS at 100 degrees celsius ");
        assert_eq!(fa, fc);
        assert_eq!(a.euclidean, c.euclidean);
        // A fresh projector built from the same version gives the same bits.
        let again = TextEmbeddingProjector::new(1.0).unwrap();
        assert_eq!(
            again.project_text("water boils").unwrap().1,
            project("water boils").1
        );
    }

    #[test]
    fn projection_refuses_empty_punctuation_and_oversized_text() {
        let p = TextEmbeddingProjector::new(1.0).unwrap();
        for text in ["", "   ", "\n\t", "!!! ... ---"] {
            assert!(
                matches!(p.project_text(text), Err(LodError::EmptyInput(_))),
                "{text:?}"
            );
        }
        let big = "a ".repeat(MAX_PAYLOAD_BYTES / 2 + 1);
        assert!(matches!(
            p.project_text(&big),
            Err(LodError::PayloadTooLarge { .. })
        ));
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(TextEmbeddingProjector::new(bad).is_err());
        }
    }

    #[test]
    fn shared_ngrams_bring_texts_closer_in_both_fingerprint_and_manifold() {
        let (base, fb) = project("the reactor coolant pump failed during the night shift");
        let (near, fnear) = project("reactor coolant pump failed on the night shift");
        let (far, ffar) = project("quarterly marketing budget for the new espresso brand");
        let h_near = hdc_hamming_distance_256(&fb, &fnear);
        let h_far = hdc_hamming_distance_256(&fb, &ffar);
        assert!(h_near < h_far, "hamming near {h_near} far {h_far}");
        assert!(h_near < 64, "near pair shares most n-grams: {h_near}");
        // Unrelated texts sit near the random-hyperplane mean of 128 bits.
        assert!((80..=176).contains(&h_far), "far pair {h_far}");
        let d_near = distance(&base, &near);
        let d_far = distance(&base, &far);
        assert!(d_near < d_far, "geodesic near {d_near} far {d_far}");
    }

    #[test]
    fn projection_stays_inside_the_ball_of_every_curvature() {
        let texts = [
            "a",
            "x y z",
            "水在一百度沸腾",
            "The quick brown fox jumps over the lazy dog, again and again and again.",
        ];
        for c in [0.25_f32, 1.0, 4.0, 25.0] {
            let p = TextEmbeddingProjector::new(c).unwrap();
            let geometry = GeometryParams {
                curvature: f64::from(c),
                ..GeometryParams::UNIT
            };
            for text in texts {
                let (coord, _) = p.project_text(text).unwrap();
                let norm_sq: f32 = coord.hyperbolic.iter().map(|v| v * v).sum();
                assert!(c * norm_sq < 1.0, "c {c} text {text:?}");
                assert!(coord.euclidean.iter().all(|v| v.abs() < 1.0));
                let s_norm: f32 = coord.spherical.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!((s_norm - 1.0).abs() < 1e-5);
                crate::node::normalized_depth(&coord.hyperbolic, geometry.curvature).unwrap();
            }
        }
    }

    #[test]
    fn unicode_text_without_spaces_still_shares_trigrams() {
        let (a, fa) = project("水在一百度沸腾");
        let (b, fb) = project("水在一百度沸腾了");
        let (_, fc) = project("股票市场今天下跌");
        assert!(hdc_hamming_distance_256(&fa, &fb) < hdc_hamming_distance_256(&fa, &fc));
        assert!(distance(&a, &b).is_finite());
    }
}
