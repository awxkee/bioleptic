/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::header::EntropyCoder;
use crate::multichannel::{ChannelCore, ChannelMeta, compress_multi_with, decompress_multi_with};
use crate::{BiolepticError, BiolepticHeader, CompressionMethod, DataType, arans, cmodel};
use flate2::Compression;
use flate2::write::DeflateEncoder;
use osclet::{BorderMode, DaubechiesFamily, Osclet, SymletFamily};
use std::io::Write;

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug, Default)]
pub enum CutoffLevel {
    #[default]
    Low,
    Medium,
    High,
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug)]
#[repr(u8)]
#[derive(Default)]
pub enum QuantizationScale {
    S6 = 6,
    S7 = 7,
    S8 = 8,
    S9 = 9,
    S10 = 10,
    #[default]
    S11 = 11,
    S12 = 12,
}

impl QuantizationScale {
    /// Returns the scale as a raw `u8` shift amount.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns the multiplier applied to DWT coefficients: `1 << scale`.
    pub fn multiplier(self) -> f32 {
        (1u32 << self.as_u8()) as f32
    }
}

impl TryFrom<u8> for QuantizationScale {
    type Error = BiolepticError;

    fn try_from(value: u8) -> Result<Self, BiolepticError> {
        match value {
            6 => Ok(Self::S6),
            7 => Ok(Self::S7),
            8 => Ok(Self::S8),
            9 => Ok(Self::S9),
            10 => Ok(Self::S10),
            11 => Ok(Self::S11),
            12 => Ok(Self::S12),
            _ => Err(BiolepticError::InvalidQuantizationScale(value)),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct CompressionOptions {
    pub method: CompressionMethod,
    pub scale: QuantizationScale,
    pub cutoff_level: CutoffLevel,
    pub entropy_coder: Option<EntropyCoder>,
    /// Explicit quantization multiplier applied to DWT coefficients. When
    /// `None`, the power-of-two `scale.multiplier()` is used. Setting this lets
    /// you hit a precise PRD target instead of being pinned to the
    /// power-of-two grid. Larger = finer = lower PRD, more bits.
    pub quant_multiplier: Option<f32>,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        CompressionOptions {
            method: CompressionMethod::Cdf97,
            scale: QuantizationScale::S11,
            cutoff_level: CutoffLevel::default(),
            entropy_coder: None,
            quant_multiplier: None,
        }
    }
}

impl CompressionOptions {
    pub fn from_method(method: CompressionMethod) -> Self {
        CompressionOptions {
            method,
            ..Default::default()
        }
    }

    /// Sets the wavelet transform.
    #[must_use]
    pub fn with_method(mut self, method: CompressionMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets the quantization scale (also selects the detail-threshold table).
    #[must_use]
    pub fn with_scale(mut self, scale: QuantizationScale) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the detail-coefficient cutoff level.
    #[must_use]
    pub fn with_cutoff_level(mut self, cutoff_level: CutoffLevel) -> Self {
        self.cutoff_level = cutoff_level;
        self
    }

    /// Pins the payload entropy coder to a specific choice.
    #[must_use]
    pub fn with_entropy_coder(mut self, entropy_coder: EntropyCoder) -> Self {
        self.entropy_coder = Some(entropy_coder);
        self
    }

    /// Lets the compressor pick the entropy coder automatically (the default).
    #[must_use]
    pub fn with_auto_entropy_coder(mut self) -> Self {
        self.entropy_coder = None;
        self
    }

    /// Sets an explicit quantization multiplier, overriding `scale`'s power of
    /// two. Pair with a scale whose threshold is 0 (S6/S7) and `CutoffLevel::Low`
    /// for a pure uniform quantizer (the high-fidelity optimum).
    #[must_use]
    pub fn with_multiplier(mut self, multiplier: f32) -> Self {
        self.quant_multiplier = Some(multiplier);
        self
    }
}

fn threshold(details: &mut [i16], scale: QuantizationScale, cutoff_level: CutoffLevel) {
    let mut threshold = match scale {
        QuantizationScale::S6 => 0,
        QuantizationScale::S7 => 0,
        QuantizationScale::S8 => 1,
        QuantizationScale::S9 => 1,
        QuantizationScale::S10 => 2,
        QuantizationScale::S11 => 2,
        QuantizationScale::S12 => 3,
    };
    match cutoff_level {
        CutoffLevel::Low => {}
        CutoffLevel::Medium => {
            threshold *= 3;
        }
        CutoffLevel::High => {
            threshold *= 7;
        }
    }
    for det in details.iter_mut() {
        if det.unsigned_abs() < threshold {
            *det = 0;
        }
    }
}

fn compute_max_levels(signal_len: usize, filter_length: usize) -> usize {
    if signal_len < filter_length {
        return 1;
    }
    // floor(log2(signal_len / filter_length))
    let ratio = signal_len / filter_length;
    let max = usize::BITS as usize - ratio.leading_zeros() as usize - 1;
    max.clamp(1, 8)
}

/// Selects the number of DWT levels for a signal length + method. Shared by the
/// single-channel and multi-channel paths so they always agree (equal-length
/// channels therefore decompose to identical subband geometry).
pub(crate) fn pick_levels(signal_len: usize, method: CompressionMethod) -> usize {
    if signal_len < 20 {
        1
    } else if signal_len < 40 {
        2
    } else if signal_len < 60 {
        3
    } else if signal_len < 80 {
        4
    } else {
        let dwt_worker = match method {
            CompressionMethod::Cdf53 => Osclet::make_cdf53_f32(),
            CompressionMethod::Cdf97 => Osclet::make_cdf97_f32(),
            CompressionMethod::Db4 => {
                Osclet::make_daubechies_f32(DaubechiesFamily::Db4, BorderMode::Wrap)
            }
            CompressionMethod::Sym4 => {
                Osclet::make_symlet_f32(SymletFamily::Sym4, BorderMode::Wrap)
            }
        };
        compute_max_levels(signal_len, dwt_worker.filter_length())
    }
}

/// Per-channel compression result without the framing header. The single-channel
/// wrapper turns this into a `BiolepticHeader` + payload; the multi-channel
/// container stores the numeric fields per channel and shares the rest.
pub(crate) struct CoreOut {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub quant_multiplier: f32,
    pub scale: u8,
    pub entropy_coder: u8,
    pub payload: Vec<u8>,
}

/// Core transform + quantize + entropy-code, given a precomputed `level`.
/// Contains everything the old `compress` did except building the header.
pub(crate) fn compress_core(
    data: &[f32],
    options: CompressionOptions,
    level: usize,
) -> Result<CoreOut, BiolepticError> {
    let mut v_min = f32::INFINITY;
    let mut v_max = f32::NEG_INFINITY;
    let mut working_data = vec![0.; data.len()];
    for (dst, &src) in working_data.iter_mut().zip(data.iter()) {
        #[allow(clippy::if_same_then_else)]
        let val = if src.is_finite() {
            src
        } else if src.is_nan() {
            0.
        } else if src.is_sign_negative() {
            0.
        } else {
            1.
        };
        v_min = val.min(v_min);
        v_max = val.max(v_max);
        *dst = val;
    }
    let mut v_sum = 0.;
    let range = v_max - v_min;
    let mut v_mean = 0.;
    if range > 1e-5 {
        let range_scale = 1. / range;
        let diff = v_min;
        for dst in working_data.iter_mut() {
            let q = (*dst - diff) * range_scale;
            v_sum += q;
            *dst = q;
        }
        v_mean = v_sum / data.len() as f32;
        for dst in working_data.iter_mut() {
            *dst -= v_mean;
        }
    } else {
        working_data.fill(0.);
    }

    let dwt_worker = match options.method {
        CompressionMethod::Cdf53 => Osclet::make_cdf53_f32(),
        CompressionMethod::Cdf97 => Osclet::make_cdf97_f32(),
        CompressionMethod::Db4 => {
            Osclet::make_daubechies_f32(DaubechiesFamily::Db4, BorderMode::Wrap)
        }
        CompressionMethod::Sym4 => Osclet::make_symlet_f32(SymletFamily::Sym4, BorderMode::Wrap),
    };

    if working_data.len() < dwt_worker.filter_length() {
        let target_len = dwt_worker.filter_length();
        let current_len = working_data.len();
        let needed = target_len - current_len;
        // Wrap-extend: mirror the existing samples cyclically to avoid
        // zero-padding artifacts at the filter boundary.
        let extension = (0..needed)
            .map(|i| working_data[i % current_len])
            .collect::<Vec<f32>>();
        working_data.extend_from_slice(&extension);
    }

    let dwt = dwt_worker
        .multi_dwt(&working_data, level)
        .map_err(|x| BiolepticError::UnderlyingDwtError(x.to_string()))?;

    if dwt.levels.is_empty() {
        return Err(BiolepticError::UnderlyingDwtError(
            "Internal DWT returned zero levels, what shouldn't happen".to_string(),
        ));
    }

    let last_dwt_level = match dwt.levels.last() {
        None => {
            return Err(BiolepticError::UnderlyingDwtError(
                "Internal DWT returned zero levels, what shouldn't happen".to_string(),
            ));
        }
        Some(v) => v,
    };

    let scale_multiplier = options
        .quant_multiplier
        .unwrap_or_else(|| options.scale.multiplier());

    let mut approximation = last_dwt_level
        .approximations
        .iter()
        .map(|&x| {
            (x * scale_multiplier)
                .min(i16::MAX as f32)
                .max(i16::MIN as f32) as i16
        })
        .collect::<Vec<i16>>();

    let mut details = dwt
        .levels
        .iter()
        .map(|x| {
            x.details
                .iter()
                .map(|&x| {
                    (x * scale_multiplier)
                        .min(i16::MAX as f32)
                        .max(i16::MIN as f32) as i16
                })
                .collect::<Vec<i16>>()
        })
        .collect::<Vec<Vec<i16>>>();

    let mut total_details_length = 0usize;

    for level_details in details.iter_mut() {
        threshold(level_details, options.scale, options.cutoff_level);
        total_details_length += level_details.len();
    }

    let entropy_coder = options.entropy_coder.unwrap_or(EntropyCoder::Cmodel);

    let payload = match entropy_coder {
        EntropyCoder::Cmodel => {
            // Coefficient-aware coder: subband order is [approximation, detail0..L-1].
            let mut subbands: Vec<&[i16]> = Vec::with_capacity(1 + details.len());
            subbands.push(approximation.as_slice());
            for level_details in details.iter() {
                subbands.push(level_details.as_slice());
            }
            cmodel::encode_coeffs(&subbands)
        }
        EntropyCoder::Deflate | EntropyCoder::Arans => {
            // Legacy: concatenate to one i16 stream, then byte-code.
            approximation
                .try_reserve_exact(total_details_length)
                .map_err(|_| BiolepticError::OutOfMemoryError(total_details_length))?;
            for level_details in details.iter() {
                approximation.extend_from_slice(level_details);
            }
            let approximation_bytes = approximation
                .into_iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>();
            match entropy_coder {
                EntropyCoder::Deflate => {
                    let mut e = DeflateEncoder::new(Vec::new(), Compression::default());
                    e.write_all(&approximation_bytes)
                        .map_err(|x| BiolepticError::UnderlyingCompressorError(x.to_string()))?;
                    e.finish()
                        .map_err(|x| BiolepticError::UnderlyingCompressorError(x.to_string()))?
                }
                EntropyCoder::Arans => arans::encode_stream(&approximation_bytes),
                EntropyCoder::Cmodel => unreachable!(),
            }
        }
    };

    Ok(CoreOut {
        min: v_min,
        max: v_max,
        mean: v_mean,
        quant_multiplier: scale_multiplier,
        scale: options.scale.as_u8(),
        entropy_coder: entropy_coder.into(),
        payload,
    })
}

/// Compresses a slice of `f32` samples into a single-channel Bioleptic stream.
///
/// Non-finite values (`NaN`, `±inf`) are substituted before processing:
/// `NaN` and `-inf` become `0.0`, `+inf` becomes `1.0`. The signal is then
/// mean-centered and range-normalized, transformed with a multi-level DWT,
/// quantized to `i16`, thresholded, and entropy-coded.
pub fn compress(data: &[f32], options: CompressionOptions) -> Result<Vec<u8>, BiolepticError> {
    if data.is_empty() {
        return Err(BiolepticError::UnsupportedCompressorConfiguration(
            "Can't compress empty data".to_string(),
        ));
    }
    if data.len() > i32::MAX as usize {
        return Err(BiolepticError::UnsupportedCompressorConfiguration(format!(
            "Can't compress data bigger than {}, but data was {}",
            i32::MAX,
            data.len()
        )));
    }

    let level = pick_levels(data.len(), options.method);
    let core = compress_core(data, options, level)?;

    let header = BiolepticHeader::new(
        DataType::Float32,
        options.method,
        level as u8,
        options.scale,
        data.len() as u32,
        core.min,
        core.max,
        core.mean,
        core.payload.len() as u32,
        EntropyCoder::try_from(core.entropy_coder)?,
        core.quant_multiplier,
    );

    let mut header_bytes = header.to_bytes().to_vec();
    header_bytes.extend_from_slice(&core.payload);

    Ok(header_bytes)
}

/// Compresses at the coarsest multiplier whose reconstruction PRD is `<= target_prd`.
///
/// Sweeps the quantization multiplier geometrically and returns the first
/// (coarsest, fewest-bits) encoding that meets the fidelity target. For a pure
/// uniform quantizer, pass `options` with `scale = S7` and `CutoffLevel::Low`
/// so the detail threshold is disabled.
pub fn compress_to_prd(
    data: &[f32],
    target_prd: f64,
    options: CompressionOptions,
) -> Result<Vec<u8>, BiolepticError> {
    fn prd(orig: &[f32], rec: &[f32]) -> f64 {
        let n = orig.len() as f64;
        let mean = orig.iter().map(|&x| x as f64).sum::<f64>() / n;
        let num: f64 = orig
            .iter()
            .zip(rec.iter())
            .map(|(&x, &y)| {
                let d = x as f64 - y as f64;
                d * d
            })
            .sum();
        let den: f64 = orig
            .iter()
            .map(|&x| {
                let c = x as f64 - mean;
                c * c
            })
            .sum();
        if den == 0.0 {
            0.0
        } else {
            (num / den).sqrt() * 100.0
        }
    }

    // The multiplier acts on normalized coefficients; this geometric range spans
    // roughly PRD 2% .. 0.05% for typical biosignals.
    const STEPS: usize = 48;
    let (lo, hi) = (32.0f32, 8192.0f32);
    for k in 0..STEPS {
        let t = k as f32 / (STEPS - 1) as f32;
        let m = lo * (hi / lo).powf(t); // ascending => coarsest acceptable first
        let encoded = compress(data, options.with_multiplier(m))?;
        let decoded = crate::decompressor::decompress(&encoded)?;
        if prd(data, &decoded) <= target_prd {
            return Ok(encoded);
        }
    }
    Err(BiolepticError::UnsupportedCompressorConfiguration(format!(
        "could not meet PRD target {target_prd}% within the multiplier sweep"
    )))
}

/// Adapter implementing the container's per-channel codec over `compress_core` /
/// `decompress_core`.
struct BiolepticCore {
    options: CompressionOptions,
}

impl ChannelCore for BiolepticCore {
    type Err = BiolepticError;

    fn encode(
        &self,
        data: &[f32],
        _method: [u8; 4],
        levels: u8,
    ) -> Result<(ChannelMeta, Vec<u8>), BiolepticError> {
        let core = compress_core(data, self.options, levels as usize)?;
        let meta = ChannelMeta {
            min: core.min,
            max: core.max,
            mean: core.mean,
            quant_multiplier: core.quant_multiplier,
            scale: core.scale,
            entropy_coder: core.entropy_coder,
            compressed_size: 0, // container fills this from the payload length
        };
        Ok((meta, core.payload))
    }

    fn decode(
        &self,
        meta: &ChannelMeta,
        payload: &[u8],
        method: [u8; 4],
        levels: u8,
        signal_length: u32,
    ) -> Result<Vec<f32>, BiolepticError> {
        let method = CompressionMethod::try_from(u32::from_le_bytes(method))?;
        crate::decompressor::decompress_core(
            payload,
            method,
            levels as usize,
            signal_length as usize,
            meta.min,
            meta.max,
            meta.mean,
            meta.quant_multiplier,
            meta.scale,
            EntropyCoder::try_from(meta.entropy_coder)?,
        )
    }
}

/// Compresses several equal-length channels into one `BILX` container.
///
/// All channels must share the same length.
pub fn compress_multi(
    channels: &[&[f32]],
    options: CompressionOptions,
) -> Result<Vec<u8>, BiolepticError> {
    if channels.is_empty() {
        return Err(BiolepticError::UnsupportedCompressorConfiguration(
            "Need at least one channel".to_string(),
        ));
    }
    let len = channels[0].len();
    if len == 0 {
        return Err(BiolepticError::UnsupportedCompressorConfiguration(
            "Can't compress empty channels".to_string(),
        ));
    }
    if len > i32::MAX as usize {
        return Err(BiolepticError::UnsupportedCompressorConfiguration(format!(
            "Can't compress data bigger than {}, but data was {}",
            i32::MAX,
            len
        )));
    }
    if channels.len() > u16::MAX as usize {
        return Err(BiolepticError::UnsupportedCompressorConfiguration(format!(
            "At most {} channels are supported, got {}",
            u16::MAX,
            channels.len()
        )));
    }

    let levels = pick_levels(len, options.method) as u8;
    let method_tag: u32 = options.method.into();
    let core = BiolepticCore { options };

    // data_type byte: 0 == f32 (the only supported type today).
    compress_multi_with(&core, channels, method_tag.to_le_bytes(), levels, 0)
        .map_err(|e| BiolepticError::UnsupportedCompressorConfiguration(e.to_string()))
}

/// Decompresses a `BILX` multi-channel container into one `Vec<f32>` per channel.
pub fn decompress_multi(bytes: &[u8]) -> Result<Vec<Vec<f32>>, BiolepticError> {
    let core = BiolepticCore {
        options: CompressionOptions::default(),
    };
    decompress_multi_with(&core, bytes)
        .map_err(|e| BiolepticError::DecompressionError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompressor::decompress;

    /// Generates a synthetic PPG-like signal.
    /// Models the systolic peak, dicrotic notch, and diastolic peak.
    pub(crate) fn generate_ppg(samples: usize, sample_rate: f32, heart_rate_bpm: f32) -> Vec<f32> {
        let rr_interval = 60.0 / heart_rate_bpm;
        let mut signal = vec![0.0f32; samples];

        for i in 0..samples {
            let t = i as f32 / sample_rate;
            let phase = (t / rr_interval).fract();

            // systolic rise — fast gaussian peak at ~25% of cycle
            let systolic = 1.0 * gaussian(phase, 0.25, 0.06);

            // dicrotic notch — small dip at ~45% of cycle
            let notch = -0.08 * gaussian(phase, 0.45, 0.02);

            // diastolic peak — smaller secondary bump at ~55% of cycle
            let diastolic = 0.15 * gaussian(phase, 0.55, 0.04);

            // slow baseline variation simulating respiration (~0.3 Hz)
            let baseline = 0.03 * (2.0 * std::f32::consts::PI * 0.3 * t).sin();

            // noise
            let noise = 0.005 * pseudo_noise(i);

            signal[i] = (systolic + notch + diastolic + baseline + noise) * 3500.0;
        }

        signal
    }

    #[inline]
    fn gaussian(x: f32, mean: f32, std: f32) -> f32 {
        (-(x - mean).powi(2) / (2.0 * std.powi(2))).exp()
    }

    /// Deterministic pseudo-noise via LCG, avoids rand dependency
    #[inline]
    fn pseudo_noise(i: usize) -> f32 {
        let x = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
        // map to [-1, 1]
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    pub(crate) fn prd(original: &[f32], reconstructed: &[f32]) -> f64 {
        assert_eq!(original.len(), reconstructed.len());
        let n = original.len() as f64;

        // mean of original
        let mean = original.iter().map(|&x| x as f64).sum::<f64>() / n;

        // numerator: squared error
        let num = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(&x, &y)| {
                let diff = x as f64 - y as f64;
                diff * diff
            })
            .sum::<f64>();

        // denominator: signal energy around mean
        let den = original
            .iter()
            .map(|&x| {
                let centered = x as f64 - mean;
                centered * centered
            })
            .sum::<f64>();

        if den == 0.0 {
            return 0.0;
        }

        (num / den).sqrt() * 100.0
    }

    #[test]
    fn test_coding() {
        let r_means = generate_ppg(500000, 120., 90.);
        let raw_bytes = r_means.len() * size_of::<f32>();
        for coder in [
            EntropyCoder::Deflate,
            EntropyCoder::Arans,
            EntropyCoder::Cmodel,
        ] {
            let encoded = compress(
                &r_means,
                CompressionOptions::from_method(CompressionMethod::Cdf97)
                    .with_entropy_coder(coder)
                    .with_multiplier(1200.),
            )
            .unwrap();
            let compressed_bytes = encoded.len();
            let decompressed = decompress(&encoded).unwrap();
            let cr = raw_bytes as f32 / compressed_bytes as f32;
            let prd_val = prd(&r_means, &decompressed);
            assert!(prd_val < 0.5, "got PRD {prd_val}");
            println!(
                "n={:5}  raw={:8}  compressed={:8}  cr={:6.2}:1  PRD={:.4}%",
                r_means.len(),
                raw_bytes,
                compressed_bytes,
                cr,
                prd_val
            );
        }
    }

    #[test]
    fn test_coding_small() {
        for coder in [
            EntropyCoder::Deflate,
            EntropyCoder::Arans,
            EntropyCoder::Cmodel,
        ] {
            let r_means = [1., 2., 3., 4., 5., 6.];
            let raw_bytes = r_means.len() * size_of::<f32>();
            let encoded = compress(
                &r_means,
                CompressionOptions::from_method(CompressionMethod::Cdf53).with_entropy_coder(coder),
            )
            .unwrap();
            let compressed_bytes = encoded.len();
            let decompressed = decompress(&encoded).unwrap();
            assert_eq!(decompressed.len(), r_means.len());
            let cr = raw_bytes as f32 / compressed_bytes as f32;
            let prd_val = prd(&r_means, &decompressed);
            assert!(prd_val < 0.5, "got PRD {prd_val}");
            println!(
                "n={:5}  raw={:8}  compressed={:8}  cr={:6.2}:1  PRD={:.4}%",
                r_means.len(),
                raw_bytes,
                compressed_bytes,
                cr,
                prd_val
            );
        }
    }

    // #[derive(Clone, Copy, Debug)]
    // enum SampleFmt {
    //     I16le,
    //     F32le,
    // }
    //
    // /// Load one channel of a raw (headerless) `.bin` as `f32`.
    // ///
    // /// * `channels` — interleave factor (1 = single lead, 2 = MIT-BIH, 12 = PTB-XL…)
    // /// * `channel`  — which 0-based channel to extract
    // /// * `skip`     — header bytes to drop before the samples (0 for raw dumps)
    // fn load_channel(
    //     path: &str,
    //     fmt: SampleFmt,
    //     channels: usize,
    //     channel: usize,
    //     skip: usize,
    // ) -> io::Result<Vec<f32>> {
    //     let raw = fs::read(path)?;
    //     let body = &raw[skip.min(raw.len())..];
    //
    //     let all: Vec<f32> = match fmt {
    //         SampleFmt::I16le => body
    //             .chunks_exact(2)
    //             .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
    //             .collect(),
    //         SampleFmt::F32le => body
    //             .chunks_exact(4)
    //             .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    //             .collect(),
    //     };
    //
    //     // De-interleave: keep every `channels`-th sample starting at `channel`.
    //     Ok(all
    //         .into_iter()
    //         .skip(channel)
    //         .step_by(channels.max(1))
    //         .collect())
    // }
    //
    // /// When you don't know the format, print the first few samples under each
    // /// interpretation. ECG ADC values are typically small signed ints (|v| < ~5000);
    // /// f32 dumps look like sane physical magnitudes. The size hints also help:
    // /// a file divisible by 4 *might* be f32; one only divisible by 2 is i16.
    // fn sniff(path: &str) -> io::Result<()> {
    //     let raw = fs::read(path)?;
    //     println!(
    //         "file: {} bytes  (÷2={}, ÷4={})",
    //         raw.len(),
    //         raw.len() % 2 == 0,
    //         raw.len() % 4 == 0
    //     );
    //     let as_i16: Vec<i16> = raw
    //         .chunks_exact(2)
    //         .take(8)
    //         .map(|b| i16::from_le_bytes([b[0], b[1]]))
    //         .collect();
    //     let as_f32: Vec<f32> = raw
    //         .chunks_exact(4)
    //         .take(8)
    //         .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    //         .collect();
    //     println!("  as i16le: {:?}", as_i16);
    //     println!("  as f32le: {:?}", as_f32);
    //     Ok(())
    // }
    //
    // /// Compress one window and report ratio + distortion.
    // fn test_window(seg: &[f32], opts: CompressionOptions) {
    //     let raw_bytes = seg.len() * std::mem::size_of::<f32>();
    //     let encoded = compress(seg, opts).expect("compress");
    //     let decoded = decompress(&encoded).expect("decompress");
    //     let cr = raw_bytes as f32 / encoded.len() as f32;
    //     println!(
    //         "  n={:6}  raw={:8}  comp={:7}  CR={:6.2}:1  PRD={:.3}%",
    //         seg.len(),
    //         raw_bytes,
    //         encoded.len(),
    //         cr,
    //         prd(seg, &decoded)
    //     );
    // }
    //
    // /// Sweep the whole signal in fixed blocks and report the aggregate — a far more
    // /// honest figure than a single hand-picked window, since CR and PRD both depend
    // /// on block size (smaller blocks pay more fixed header per block).
    // fn sweep_blocks(sig: &[f32], block: usize, opts: CompressionOptions) {
    //     let (mut raw_total, mut comp_total, mut prd_sum, mut nblocks) =
    //         (0usize, 0usize, 0f64, 0usize);
    //     for chunk in sig.chunks(block) {
    //         if chunk.len() < 16 {
    //             continue;
    //         } // skip a tiny tail block
    //         let encoded = compress(chunk, opts).expect("compress");
    //         let decoded = decompress(&encoded).expect("decompress");
    //         raw_total += chunk.len() * 4;
    //         comp_total += encoded.len();
    //         prd_sum += prd(chunk, &decoded);
    //         nblocks += 1;
    //     }
    //     println!(
    //         "  block={:5}: {} blocks  CR={:.2}:1  meanPRD={:.3}%",
    //         block,
    //         nblocks,
    //         raw_total as f32 / comp_total as f32,
    //         prd_sum / nblocks as f64
    //     );
    // }
    //
    // #[test]
    // fn test_coding2() {
    //     sniff("./assets/ecg_aVR.bin").unwrap();
    //     let r_means = load_channel("./assets/ecg_aVR.bin", SampleFmt::I16le, 7, 2, 0).unwrap();
    //
    //     let raw_bytes = r_means.len() * size_of::<f32>();
    //
    //     for coder in [
    //         EntropyCoder::Deflate,
    //         EntropyCoder::Arans,
    //         EntropyCoder::Cmodel,
    //     ] {
    //         let encoded = compress(
    //             &r_means,
    //             CompressionOptions::from_method(CompressionMethod::Cdf97).with_entropy_coder(coder),
    //         )
    //         .unwrap();
    //         let compressed_bytes = encoded.len();
    //         let decompressed = decompress(&encoded).unwrap();
    //         let cr = raw_bytes as f32 / compressed_bytes as f32;
    //         let prd_val = prd(&r_means, &decompressed);
    //         assert!(prd_val < 0.5, "got PRD {prd_val}");
    //         println!(
    //             "Entropy {:?} n={:5}  raw={:8}  compressed={:8}  cr={:6.2}:1  PRD={:.4}%",
    //             coder,
    //             r_means.len(),
    //             raw_bytes,
    //             compressed_bytes,
    //             cr,
    //             0.
    //         );
    //     }
    // }

    #[test]
    fn test_compress_to_prd() {
        let sig = generate_ppg(20000, 120., 90.);
        let encoded = compress_to_prd(
            &sig,
            0.31,
            CompressionOptions::from_method(CompressionMethod::Cdf97)
                .with_scale(QuantizationScale::S7)
                .with_entropy_coder(EntropyCoder::Cmodel),
        )
        .unwrap();
        let decoded = decompress(&encoded).unwrap();
        let p = prd(&sig, &decoded);
        println!("compress_to_prd(0.31) -> PRD={p:.4}%");
        assert!(p <= 0.31 + 1e-6, "target missed: PRD {p}");
    }

    #[test]
    fn test_multichannel_roundtrip() {
        // Two correlated leads + one independent; equal length.
        let a = generate_ppg(8000, 120., 90.);
        let b: Vec<f32> = a.iter().map(|&x| x * 0.8 + 100.0).collect();
        let c = generate_ppg(8000, 120., 75.);
        let channels: Vec<&[f32]> = vec![&a, &b, &c];

        let opts = CompressionOptions::from_method(CompressionMethod::Cdf97)
            .with_scale(QuantizationScale::S7)
            .with_multiplier(472.0)
            .with_entropy_coder(EntropyCoder::Cmodel);

        let encoded = compress_multi(&channels, opts).unwrap();
        let decoded = decompress_multi(&encoded).unwrap();
        assert_eq!(decoded.len(), 3);
        for (orig, rec) in channels.iter().zip(decoded.iter()) {
            assert_eq!(orig.len(), rec.len());
            let p = prd(orig, rec);
            assert!(p < 1.0, "channel PRD {p}");
        }

        // Each channel must match its own single-channel encode/decode bit-for-bit
        // in reconstruction (proves the core split didn't change the math).
        for ch in &channels {
            let single = decompress(&compress(ch, opts).unwrap()).unwrap();
            let multi_idx = channels.iter().position(|c| c == ch).unwrap();
            assert_eq!(single, decoded[multi_idx]);
        }
    }

    #[test]
    fn test_multichannel_length_mismatch_errors() {
        let a = vec![1.0f32; 100];
        let b = vec![2.0f32; 99];
        let channels: Vec<&[f32]> = vec![&a, &b];
        let res = compress_multi(&channels, CompressionOptions::default());
        assert!(res.is_err(), "mismatched lengths must error, not pad");
    }
}
