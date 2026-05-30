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
    use std::{fs, io};

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
        // #[rustfmt::skip]
        // let r_means = [217.23157, 215.67552, 215.06216, 215.06221, 215.37572, 213.01607, 213.58386, 214.25183, 214.9871, 215.82896, 216.06363, 213.49269, 214.26846, 215.38809, 216.10449, 216.7581, 217.44806, 218.23366, 218.68704, 219.03677, 219.49788, 219.90564, 219.1889, 217.77367, 216.98332, 216.88615, 213.8439, 214.54115, 215.04988, 217.35072, 214.40901, 215.14978, 217.70839, 218.36266, 218.99208, 217.64491, 217.30862, 217.44238, 217.78906, 218.38847, 218.92596, 219.17801, 219.36043, 219.54408, 219.74702, 220.14243, 220.71124, 221.24063, 221.80835, 222.33897, 222.77641, 223.14272, 223.5221, 223.83257, 224.11247, 223.25172, 221.01003, 219.79028, 219.35567, 219.30293, 219.39572, 219.65808, 220.13982, 220.36964, 220.46214, 220.42639, 220.69572, 221.14003, 221.68759, 222.18318, 222.52472, 222.88013, 223.21518, 223.54077, 223.76608, 223.85362, 223.90585, 223.94562, 222.97073, 220.73985, 219.44333, 218.93787, 218.88582, 218.95444, 219.24374, 219.46704, 219.61621, 219.64369, 219.63622, 219.7927, 220.05626, 220.50317, 220.90195, 221.24147, 221.4977, 221.79135, 222.05875, 222.38885, 222.71448, 222.94823, 223.10927, 223.2588, 223.54808, 222.86108, 220.66046, 219.29869, 218.78043, 218.78537, 218.89235, 219.08115, 219.54941, 219.88058, 219.98447, 219.99211, 220.06358, 220.27615, 220.66753, 221.12798, 221.58208, 222.01543, 222.41112, 222.70154, 222.83594, 223.26054, 223.03955, 223.4678, 223.77551, 224.11346, 223.45522, 221.09393, 219.58519, 219.06276, 219.15833, 219.53085, 219.96498, 220.39095, 220.42624, 220.36208, 220.35495, 220.51996, 220.72617, 221.09756, 221.45454, 221.83559, 222.1312, 222.39728, 222.55342, 222.60048, 222.76472, 222.88567, 223.19852, 222.72998, 220.77544, 219.3764, 218.81155, 218.86603, 219.11823, 219.61452, 220.04689, 220.16678, 220.17804, 220.13068, 220.14015, 220.40698, 220.76636, 221.21057, 221.54918, 221.86246, 222.13618, 222.36798, 222.45055, 222.57278, 222.662, 222.91347, 222.8032, 220.95314, 219.19891, 218.28836, 218.17749, 218.32068, 218.58577, 218.928, 219.1817, 219.2994, 219.28897, 219.25613, 219.27203, 219.75539, 220.08246, 220.44008, 220.82315, 221.056, 221.19098, 221.37898, 221.51997, 221.67493, 221.93684, 222.18825, 221.73181, 219.89319, 218.4242, 217.79993, 217.80731, 218.0267, 218.2905, 218.78299, 218.90704, 218.99844, 218.99672, 219.02867, 219.20068, 219.5211, 219.84291, 220.17386, 220.3959, 220.76634, 220.99942, 221.05164, 221.15906, 221.41646, 221.67738, 221.52618, 219.59038, 217.9517, 217.19768, 217.2373, 217.44408, 217.84799, 218.34537, 218.53671, 218.49055, 218.41135, 218.39099, 218.6021, 219.04373, 219.46092, 219.84595, 220.14851, 220.27122, 220.3849, 220.61403, 220.88216, 220.9089, 219.58481, 217.66103, 216.6833, 216.50801, 216.65797, 217.01437, 217.49915, 217.76329, 217.90422, 217.89687, 217.89635, 218.13141, 218.42831, 218.85193, 219.23645, 219.5056, 219.71562, 219.89828, 219.98514, 220.08214, 220.1671, 220.22606, 219.4255, 217.54854, 216.22035, 215.7154, 215.78159, 215.87956, 216.20837, 216.62822, 216.7427, 216.82932, 216.90123, 217.1534, 217.65562, 218.0896, 218.48296, 218.83907, 219.09482, 219.29463, 219.53578, 219.69652, 219.90388, 220.04433, 220.39833, 220.70824, 220.66061, 218.84424, 217.02527, 216.08195, 215.92885, 216.0738, 216.36356, 216.76161, 217.01762, 217.13591, 217.10376, 217.05737, 217.4181, 217.77473, 218.21591, 218.63762, 218.98605, 219.1768, 219.38693, 219.58162, 219.71043, 219.83496, 220.11148, 220.33395, 220.16035, 218.34648, 216.63904, 215.85318, 215.84613, 216.06877, 216.54868, 216.95581, 217.04912, 217.01826, 216.9776, 217.06743, 217.1982, 217.59746, 218.0523, 218.42844, 218.80223, 218.9875, 219.12991, 219.23734, 219.42659, 219.53473, 219.72142, 219.99309, 219.4722, 217.46686, 215.88124, 215.27585, 215.32677, 215.44994, 215.833, 216.2022, 216.35533, 216.40877, 216.44916, 216.5663, 216.84596, 217.33192, 217.80113, 218.13649, 218.46883, 218.73096, 218.93182, 219.0285, 219.13051, 219.31607, 219.55096, 219.80554, 219.91533, 218.81061, 216.8593, 215.54688, 215.14645, 215.25441, 215.4788, 215.77348, 216.253, 216.40582, 216.5342, 216.54486, 216.546, 216.96738, 217.40909, 217.86398, 218.24644, 218.5973, 218.79604, 219.0449, 219.23349, 219.4031, 219.62523, 219.79926, 220.02594, 220.08167, 218.92802, 216.79398, 215.40588, 214.93466, 215.08887, 215.31465, 215.73573, 216.16199, 216.17747, 216.20137, 216.26755, 216.44345, 216.86665, 217.29053, 217.74956, 218.0601, 218.39294, 218.61531, 218.73672, 218.91371, 219.09958, 219.25717, 219.46635, 219.44705, 218.30608, 216.36978, 215.08072, 214.69511, 214.81325, 215.14325, 215.54984, 215.87064, 215.99632, 216.04463, 216.10963, 216.30641, 216.69844, 217.14786, 217.59824, 218.01825, 218.34299, 218.67249, 218.892, 219.04784, 219.14037, 219.37871, 219.6786, 219.84334, 219.576, 217.588, 215.7561, 214.86876, 214.82724, 214.93819, 215.33649, 215.72943, 215.91922, 216.05222, 216.10667, 216.1265, 216.31447, 216.93874, 217.40681, 217.93001, 218.3367, 218.51799, 218.68413, 218.8698, 218.95381, 219.0229, 219.3221, 219.49614, 219.63904, 219.01274, 216.8848, 215.27109, 214.71306, 214.85292, 215.12947, 215.3864, 215.77805, 215.89615, 215.8512, 215.8085, 216.07538, 216.39, 216.85008, 217.21846, 217.5636, 217.87506, 218.09186, 218.23326, 218.45515, 218.53038, 218.57791, 218.64462, 218.16667, 216.4564, 215.17282, 214.67448, 214.79704, 215.34294, 215.87994, 216.46555, 216.71075, 216.6889, 216.77473, 216.91537, 217.16136, 217.44131, 217.84276, 218.11934, 218.33278, 218.64095, 218.85408, 219.06296, 219.21573, 219.45311, 219.79245, 219.74673, 218.30644, 216.31789, 215.10428, 214.84395, 214.92365, 215.41937, 215.83913, 216.1197, 216.15314, 216.0943, 216.32544, 216.6479, 217.05124, 217.50603, 217.98302, 218.40112, 218.76729, 219.1298, 219.44519, 219.589, 219.7291, 219.85246, 219.94162, 219.76802, 218.05513, 216.07217, 215.05603, 214.89972, 215.16853, 215.51639, 215.98448, 216.3333, 216.38063, 216.39073, 216.3941, 216.53299, 216.94846, 217.41597, 217.8949, 218.34222, 218.52328, 218.67334, 218.70253, 219.01466, 219.35692, 219.23128, 217.45343, 215.66554, 214.5813, 214.39716, 214.73192, 215.24515, 215.8138, 216.21048, 216.44815, 216.68152, 217.1879, 217.7236, 218.22191, 218.62633, 218.9734, 219.30748, 219.64557, 219.9129, 220.0613, 220.15657, 220.25642, 220.2229, 218.83502, 216.59001, 215.3116, 214.9465, 215.09433, 215.25528, 215.48595, 215.92743, 216.14503, 216.1216, 216.27934, 216.29839, 216.51965, 216.9021, 217.40778, 217.90646, 218.32263, 218.5464, 218.7362, 218.76576, 218.75551, 219.0442, 219.21616, 219.2481, 217.7512, 215.71558, 214.53189, 214.2775, 214.51791, 214.9263, 215.7454, 216.19997, 216.35718, 216.25626, 216.21625, 216.57458, 217.10562, 217.60071, 217.96089, 218.22548, 218.32884, 218.63568, 218.99042, 219.18329, 219.4001, 219.58119, 219.66211, 219.59914, 218.12685, 216.15988, 215.12859, 214.87244, 215.03032, 215.24692, 215.61844, 215.92097, 216.01454, 216.05508, 216.17557, 216.33432, 216.47516, 216.89563, 217.36488, 217.79109, 218.32004, 218.5092, 218.59369, 218.6679, 218.86407, 219.13483, 218.61017, 216.71393, 215.12965, 214.35434, 214.34787, 214.5813, 214.89534, 215.36107, 215.58098, 215.57138, 215.62811, 215.81523, 215.90395, 216.47083, 216.89426, 217.3432, 217.56631, 217.77385, 218.10321, 218.3486, 218.51048, 218.6327, 218.95013, 219.08565, 218.11003, 215.94525, 214.51213, 214.01747, 214.12004, 214.4262, 214.76045, 215.04613, 215.20912, 215.22298, 215.26955, 215.45847, 215.87096, 216.35637, 216.86798, 217.34134, 217.75096, 218.06375, 218.34708, 218.51122, 218.64015, 218.76587, 219.08302, 219.38145, 218.99927, 216.77081, 214.94348, 214.08725, 214.03291, 214.26811, 214.82536, 215.16843, 215.38669, 215.45483, 215.40935, 215.42444, 215.81918, 216.27289, 216.73586, 217.17809, 217.5315, 217.88918, 218.25198, 218.54163, 218.71718, 218.9897, 219.37666, 219.06448, 217.14076, 215.29684, 214.47209, 214.4146, 214.71848, 215.24242, 215.97957, 216.45853, 216.5662, 216.60587, 216.61818, 216.83769, 217.3945, 218.14543, 218.90924, 219.53522, 219.91841, 220.25786, 220.71677, 221.18188, 221.87793, 222.57726, 222.36487, 219.74907, 217.2226, 216.3908, 216.3825, 216.56276, 216.90265, 217.30664, 217.50764, 217.45969, 217.25983, 217.17564, 217.50157, 217.89525, 218.48558, 219.0974, 219.57178, 219.92203, 220.07452, 220.15257, 220.30045, 220.56145, 220.75293, 219.63234, 217.26286, 215.74673, 215.2489, 215.33214, 215.69528, 216.16196, 216.45988, 216.52577, 216.60663, 216.73314, 217.00848, 217.38051, 217.76826, 218.18674, 218.68034, 219.24467, 219.735, 220.13924, 220.35513, 220.37068, 220.55302, 220.4004, 218.38045, 216.41351, 215.55334, 215.49432, 215.71344, 216.25377, 216.89012, 217.19226, 217.12395, 216.93665, 216.96402, 217.1057, 217.58395, 218.13213, 218.70404, 219.05888, 219.28717, 219.33247, 219.51799, 219.7559, 219.63174, 217.9322, 216.03024, 215.10384, 215.09836, 215.36058, 215.96898, 216.56735, 216.8037, 216.6511, 216.45811, 216.44054, 216.66315, 217.18929, 217.77661, 218.29749, 218.69019, 218.91891, 219.1439, 219.38432, 219.68289, 219.9682, 219.16977, 216.73401, 214.94919, 214.41412, 214.48508, 214.81949, 215.34843, 215.69531, 215.86128, 215.80925, 215.74467, 215.98009, 216.31964, 216.74962, 217.18459, 217.48027, 217.86372, 218.29948, 218.60446, 218.64597, 218.74814, 219.0301, 218.67128, 216.61752, 214.81252, 213.98344, 213.97852, 214.26434, 214.71274, 215.14258, 215.34352, 215.37488, 215.3943, 215.37656, 215.57062, 216.00829, 216.35953, 216.71722, 217.09706, 217.40245, 217.60976, 217.65462, 217.72641, 217.90605, 218.11322, 217.97342, 216.59756, 214.59233, 213.44296, 213.40314, 213.67021, 214.13968, 214.69418, 214.82227, 214.84431, 214.74512, 214.85446, 215.04375, 215.45044, 215.9312, 216.39438, 216.67825, 216.92116, 217.12106, 217.33032, 217.3549, 217.57948, 217.65475, 216.78497, 214.72824, 213.22467, 212.6435, 212.94276, 213.44888, 213.8859, 214.35518, 214.51442, 214.33917, 214.30916, 214.56247, 214.97379, 215.37782, 215.73405, 216.17361, 216.65253, 217.05821, 217.49405, 217.75105, 217.94827, 218.20201, 218.45287, 217.90768, 215.56418, 213.58084, 212.64856, 212.60564, 212.82504, 213.22412, 213.87086, 214.15395, 214.36357, 214.29602, 214.34435, 214.43924, 214.86873, 215.34541, 215.8743, 216.24986, 216.60739, 216.9209, 217.29678, 217.66463, 218.03055, 218.30565, 218.53583, 218.69083, 216.98033, 214.58827, 213.14763, 212.79099, 213.20195, 213.80234, 214.51382, 215.12952, 215.45514, 215.76665, 215.77473, 216.00961, 216.39124, 216.58372, 216.98111, 217.40387, 217.7318, 217.89542, 218.00226, 218.08012, 218.23886, 218.47847, 218.5678, 216.93951, 214.71075, 213.39197, 213.1386, 213.39084, 213.73325, 214.35214, 214.87921, 214.99208, 214.99263, 215.01883, 215.17484, 215.7148, 216.40338, 217.15091, 217.831, 218.07684, 218.3509, 218.57571, 218.60918, 218.73956, 219.11192, 218.44336, 215.97237, 214.12369, 213.34193, 213.45197, 213.6406, 214.013, 214.55365, 214.74692, 214.83917, 214.82649, 215.01466, 215.43947, 215.96814, 216.48251, 217.08344, 217.61932, 218.04553, 218.30286, 218.32317, 218.4547, 218.53894, 217.41449, 215.07648, 213.57423, 213.04523, 213.2356, 213.76125, 214.36975, 214.92432, 215.11555, 215.21973, 215.19377, 215.21631, 215.27661, 215.81174, 216.24321, 216.6037, 216.95587, 217.28223, 217.48192, 217.73457, 218.04654, 217.23466, 215.05878, 213.46959, 212.94131, 213.06946, 213.6374, 214.34065, 214.88522, 215.10052, 215.06485, 214.98967, 215.14647, 215.62448, 216.13347, 216.68097, 217.02242, 217.2107, 217.34117, 217.48662, 217.56705, 217.68927, 216.77914, 214.72563, 213.15747, 212.73668, 212.93474, 213.31241, 213.86928, 214.17697, 214.316, 214.36052, 214.36461, 214.8135, 215.21945, 215.75043, 216.26656, 216.7682, 217.18106, 217.43948, 217.55591, 217.66281, 217.88007, 217.82887, 216.43567, 214.23206, 212.83939, 212.5072, 212.74591, 213.26486, 213.79448, 214.34674, 214.52264, 214.66518, 214.67342, 214.71948, 215.06049, 215.47314, 215.89767, 216.31342, 216.58028, 216.89935, 217.17844, 217.2967, 217.52597, 217.90202, 217.65623, 215.56197, 213.6268, 212.76959, 212.85541, 213.21796, 213.79982, 214.29001, 214.45282, 214.50874, 214.48888, 214.62839, 215.01038, 215.55055, 215.99321, 216.45026, 216.70567, 216.84732, 216.93294, 217.10689, 217.20322, 217.23425, 216.02275, 214.08156, 212.75214, 212.27351, 212.4129, 212.75337, 213.24617, 213.64502, 213.77472, 213.7352, 213.78838, 214.12813, 214.68396, 215.273, 215.76064, 216.09222, 216.34009, 216.51973, 216.49364, 216.50674, 216.6811, 216.74384, 215.87358, 213.69458, 211.96443, 211.31888, 211.5174, 211.86824, 212.244, 212.85063, 213.26659, 213.37997, 213.45668, 213.5235, 213.96072, 214.48764, 214.95181, 215.41936, 215.72415, 216.0552, 216.30557, 216.44832, 216.42918, 216.555, 216.71608, 216.79868, 215.67947, 213.50854, 211.9471, 211.55182, 211.89925, 212.23659, 212.89806, 213.30302, 213.471, 213.5724, 213.54796, 213.67032, 214.27122, 214.604, 214.9701, 215.3189, 215.59798, 215.8125, 215.88339, 215.92435, 215.98886, 216.22307, 216.29196, 215.10646, 213.00806, 211.59381, 211.36147, 211.74251, 212.4867, 213.22153, 213.69772, 213.7765, 213.75241, 213.93721, 214.0738, 214.33215, 214.84842, 215.33652, 215.62793, 215.88475, 216.05727, 216.15688, 216.11607, 216.38693, 216.55385, 216.4945, 215.02866, 212.89595, 211.55139, 211.40843, 211.67259, 212.12495, 212.69133, 213.12955, 213.25548, 213.27585, 213.37161, 213.51268, 214.02158, 214.58266, 214.9664, 215.35043, 215.5561, 215.73666, 215.91939, 216.02505, 216.1625, 216.32759, 216.39801, 214.92068, 212.77306, 211.42393, 211.1652, 211.55338, 212.21303, 212.87976, 213.21635, 213.2815, 213.26659, 213.28732, 213.47595, 213.97557, 214.55336, 215.01035, 215.3768, 215.62445, 215.96472, 216.0441, 216.09668, 216.35927, 216.43079, 215.55875, 213.42082, 211.83078, 211.36502, 211.61249, 212.2003, 212.90312, 213.41432, 213.45839, 213.43776, 213.34343, 213.58508, 213.99628, 214.4596, 215.04861, 215.48616, 215.72589, 215.89096, 216.15758, 216.19449, 216.20709, 216.36119, 216.16322, 214.07742, 212.06316, 211.15099, 211.30917, 211.80623, 212.22304, 212.81055, 213.10425, 213.09525, 212.92604, 212.9111, 213.21109, 213.74232, 214.438, 215.06381, 215.446, 215.9422, 216.27017, 216.40253, 216.55153, 216.65889, 216.7126, 215.50906, 213.29736, 211.88551, 211.69049, 212.17026, 212.89902, 213.69531, 214.42668, 214.7016, 214.73206, 214.75557, 214.90582, 215.54068, 216.34903, 217.11385, 217.68567, 218.09059, 218.26892, 218.35988, 218.4909, 218.53294, 218.14217, 215.98405, 213.85739, 212.62311, 212.48302, 212.84332, 213.35489, 213.89925, 214.15839, 214.15009, 214.05931, 214.08876, 214.19604, 214.45966, 214.84366, 215.30838, 215.74825, 216.05025, 216.21748, 216.3455, 216.40773, 216.38564, 214.71994, 212.5685, 211.31381, 211.18263, 211.5171, 212.11746, 212.85368, 213.1368, 213.20374, 213.29697, 213.34552, 213.45288, 214.06236, 214.62492, 215.1074, 215.5005, 215.85591, 216.05608, 216.11603, 216.23077, 216.40361, 215.67448, 213.4481, 211.61209, 210.9858, 211.14023, 211.59898, 212.2879, 212.80135, 212.98804, 213.08397, 213.08554, 213.19308, 213.54794, 214.10355, 214.58174, 214.95757, 215.23648, 215.56236, 215.60248, 215.78995, 216.14307, 216.30031, 215.93044, 213.86325, 211.79054, 210.87839, 211.06001, 211.62877, 212.35876, 212.93958, 213.01509, 212.85745, 212.69443, 212.79784, 213.09476, 213.56248, 214.11163, 214.73657, 215.11371, 215.40132, 215.52893, 215.54814, 215.6572, 215.64754, 214.14842, 212.12729, 210.99896, 210.94218, 211.3142, 212.03642, 212.79816, 213.117, 213.1041, 213.00323, 213.03508, 213.3948, 213.78496, 214.3094, 214.63185, 215.06622, 215.2594, 215.44772, 215.51976, 215.61115, 215.6644, 215.15015, 213.11108, 211.30258, 210.65158, 210.83511, 211.64883, 212.44044, 212.95285, 213.05682, 213.05508, 213.13297, 213.28725, 213.5349, 214.01341, 214.60289, 215.27374, 215.85481, 216.37952, 216.65, 216.84744, 217.19029, 217.68977, 217.00537, 214.57964, 212.74269, 212.08127, 212.36568, 213.01706, 213.62718, 213.95126, 213.91399, 213.72635, 213.68082, 213.74901, 213.80899, 214.27959, 214.60774, 215.1082, 215.53989, 215.8613, 216.03294, 216.11397, 216.19481, 215.81548, 213.8966, 211.97395, 211.10764, 211.20285, 211.60583, 212.11142, 212.6939, 212.79091, 212.66457, 212.65955, 212.88356, 213.29475, 213.78444, 214.36424, 214.84511, 215.22461, 215.45792, 215.54965, 215.74991, 215.78601, 215.89066, 214.79306, 212.63821, 211.08067, 210.73991, 211.02051, 211.43347, 212.21793, 212.7644, 212.94392, 213.07553, 213.20642, 213.30397, 213.47406, 213.95369, 214.53119, 215.019, 215.26512, 215.41278, 215.56723, 215.7908, 215.91621, 216.12032, 215.90657, 214.23183, 212.26085, 211.19682, 211.31822, 211.71541, 212.37634, 213.08124, 213.31396, 213.18947, 212.97557, 213.3471, 213.75981, 214.15536, 214.66011, 215.0554, 215.26677, 215.38611, 215.42317, 215.61575, 215.80167, 215.98183, 215.15164, 213.22475, 211.80116, 211.3877, 211.55624, 212.19312, 212.84216, 213.3405, 213.42455, 213.33461, 213.34465, 213.50626, 213.89445, 214.35983, 214.94444, 215.29431, 215.47624, 215.71591, 215.80096, 215.72952, 215.67517, 215.66031, 214.50659, 212.47365, 211.12692, 210.96269, 211.36882, 212.11392, 212.82922, 213.29718, 213.33788, 213.17506, 213.1927, 213.58226, 214.05647, 214.43053, 214.8883, 215.34625, 215.57408, 215.80048, 216.04678, 216.19453, 216.17284, 216.19052, 215.521, 213.55078, 211.79085, 211.18398, 211.3899, 212.2156, 212.96013, 213.4528, 213.45859, 213.38588, 213.36272, 213.50081, 213.69525, 214.21426, 214.6841, 215.07181, 215.44948, 215.69508, 215.89032, 216.09805, 216.23099, 215.54297, 213.51727, 211.92776, 211.38264, 211.75653, 212.32101, 213.084, 213.48824, 213.40817, 213.2399, 213.11888, 213.31331, 213.93027, 214.41042, 214.8579, 215.26617, 215.72173, 215.87572, 215.89507, 215.9724, 216.1234, 215.93011, 213.92476, 211.92305, 211.00386, 210.93312, 211.42198, 212.19528, 212.84201, 213.01712, 213.0841, 213.13988, 213.19247, 213.35124, 213.93332, 214.43002, 214.9554, 215.26274, 215.38358, 215.59848, 215.80661, 215.77351, 216.05676, 215.4949, 213.38275, 211.70717, 211.20721, 211.46504, 212.17871, 213.01907, 213.47592, 213.60428, 213.68826, 213.61746, 213.63968, 213.94446, 214.50264, 214.87512, 215.24338, 215.47177, 215.60121, 215.65881, 215.64343, 215.58603, 215.6737, 214.89401, 212.87073, 211.54332, 211.3215, 211.3646, 212.09169, 212.87729, 213.18977, 213.17885, 213.18596, 213.23207, 213.36272, 213.88194, 214.26935, 214.79764, 215.16249, 215.5263, 215.77509, 215.74422, 215.74557, 215.88956, 216.02542, 214.75827, 212.68988, 211.48405, 211.40103, 211.8389, 212.547, 213.12561, 213.56212, 213.66809, 213.614, 213.75116, 213.99622, 214.41066, 214.87154, 215.19453, 215.77162, 216.12619, 216.29262, 216.45201, 216.54097, 216.74478, 216.90047, 216.34352, 214.10278, 212.32466, 211.63463, 211.75746, 212.24788, 212.84108, 213.24597, 213.28539, 213.18141, 212.83104, 212.80525, 213.17484, 213.74649, 214.26825, 214.68394, 214.9829, 215.28578, 215.57768, 215.83319, 216.0099, 216.02505, 216.05144, 215.85016, 213.87292, 212.0897, 211.41048, 211.65907, 212.09444, 212.89154, 213.51909, 213.74068, 213.73083, 213.56163, 213.65573, 213.72006, 214.35735, 214.98355, 215.4497, 215.67407, 215.81717, 215.84581, 216.12706, 216.34422, 216.5038, 216.68549, 215.05461, 212.85397, 211.64767, 211.64067, 212.32214, 213.04276, 213.66693, 213.93121, 213.89598, 213.88522, 213.8948, 214.16718, 214.47227, 214.75443, 215.25781, 215.80739, 216.1392, 216.32053, 216.41815, 216.35014, 216.51805, 216.70303, 216.4757, 214.55478, 212.52542, 211.6379, 211.69203, 212.3804, 213.1474, 213.71878, 214.03024, 213.95293, 213.65459, 213.73071, 214.21126, 214.82153, 215.32802, 215.54343, 215.92996, 216.33904, 216.60136, 216.77232, 216.93451, 216.896, 216.79274, 216.61974, 214.97643, 212.9983, 212.07816];
        let raw_bytes = r_means.len() * size_of::<f32>();
        let compressed_bytes = compress_to_prd(
            &r_means,
            0.35,
            CompressionOptions::from_method(CompressionMethod::Cdf97)
                .with_entropy_coder(EntropyCoder::Cmodel)
                .with_multiplier(1200.),
        )
        .unwrap()
        .len();
        let cr = raw_bytes as f32 / compressed_bytes as f32;
        println!(
            "n={:5}  raw={:8}  compressed={:8}  cr={:6.2}:1",
            r_means.len(),
            raw_bytes,
            compressed_bytes,
            cr,
        );
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

    #[derive(Clone, Copy, Debug)]
    enum SampleFmt {
        I16le,
        F32le,
    }

    /// Load one channel of a raw (headerless) `.bin` as `f32`.
    ///
    /// * `channels` — interleave factor (1 = single lead, 2 = MIT-BIH, 12 = PTB-XL…)
    /// * `channel`  — which 0-based channel to extract
    /// * `skip`     — header bytes to drop before the samples (0 for raw dumps)
    fn load_channel(
        path: &str,
        fmt: SampleFmt,
        channels: usize,
        channel: usize,
        skip: usize,
    ) -> io::Result<Vec<f32>> {
        let raw = fs::read(path)?;
        let body = &raw[skip.min(raw.len())..];

        let all: Vec<f32> = match fmt {
            SampleFmt::I16le => body
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
                .collect(),
            SampleFmt::F32le => body
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        };

        // De-interleave: keep every `channels`-th sample starting at `channel`.
        Ok(all
            .into_iter()
            .skip(channel)
            .step_by(channels.max(1))
            .collect())
    }

    /// When you don't know the format, print the first few samples under each
    /// interpretation. ECG ADC values are typically small signed ints (|v| < ~5000);
    /// f32 dumps look like sane physical magnitudes. The size hints also help:
    /// a file divisible by 4 *might* be f32; one only divisible by 2 is i16.
    fn sniff(path: &str) -> io::Result<()> {
        let raw = fs::read(path)?;
        println!(
            "file: {} bytes  (÷2={}, ÷4={})",
            raw.len(),
            raw.len() % 2 == 0,
            raw.len() % 4 == 0
        );
        let as_i16: Vec<i16> = raw
            .chunks_exact(2)
            .take(8)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let as_f32: Vec<f32> = raw
            .chunks_exact(4)
            .take(8)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        println!("  as i16le: {:?}", as_i16);
        println!("  as f32le: {:?}", as_f32);
        Ok(())
    }

    /// Compress one window and report ratio + distortion.
    fn test_window(seg: &[f32], opts: CompressionOptions) {
        let raw_bytes = seg.len() * std::mem::size_of::<f32>();
        let encoded = compress(seg, opts).expect("compress");
        let decoded = decompress(&encoded).expect("decompress");
        let cr = raw_bytes as f32 / encoded.len() as f32;
        println!(
            "  n={:6}  raw={:8}  comp={:7}  CR={:6.2}:1  PRD={:.3}%",
            seg.len(),
            raw_bytes,
            encoded.len(),
            cr,
            prd(seg, &decoded)
        );
    }

    /// Sweep the whole signal in fixed blocks and report the aggregate — a far more
    /// honest figure than a single hand-picked window, since CR and PRD both depend
    /// on block size (smaller blocks pay more fixed header per block).
    fn sweep_blocks(sig: &[f32], block: usize, opts: CompressionOptions) {
        let (mut raw_total, mut comp_total, mut prd_sum, mut nblocks) =
            (0usize, 0usize, 0f64, 0usize);
        for chunk in sig.chunks(block) {
            if chunk.len() < 16 {
                continue;
            } // skip a tiny tail block
            let encoded = compress(chunk, opts).expect("compress");
            let decoded = decompress(&encoded).expect("decompress");
            raw_total += chunk.len() * 4;
            comp_total += encoded.len();
            prd_sum += prd(chunk, &decoded);
            nblocks += 1;
        }
        println!(
            "  block={:5}: {} blocks  CR={:.2}:1  meanPRD={:.3}%",
            block,
            nblocks,
            raw_total as f32 / comp_total as f32,
            prd_sum / nblocks as f64
        );
    }

    #[test]
    fn test_coding2() {
        sniff("./assets/ecg_aVR.bin").unwrap();
        let r_means = load_channel("./assets/ecg_aVR.bin", SampleFmt::I16le, 7, 2, 0).unwrap();

        let raw_bytes = r_means.len() * size_of::<f32>();

        for coder in [
            EntropyCoder::Deflate,
            EntropyCoder::Arans,
            EntropyCoder::Cmodel,
        ] {
            let encoded = compress(
                &r_means,
                CompressionOptions::from_method(CompressionMethod::Cdf97).with_entropy_coder(coder),
            )
            .unwrap();
            let compressed_bytes = encoded.len();
            let decompressed = decompress(&encoded).unwrap();
            let cr = raw_bytes as f32 / compressed_bytes as f32;
            let prd_val = prd(&r_means, &decompressed);
            assert!(prd_val < 0.5, "got PRD {prd_val}");
            println!(
                "Entropy {:?} n={:5}  raw={:8}  compressed={:8}  cr={:6.2}:1  PRD={:.4}%",
                coder,
                r_means.len(),
                raw_bytes,
                compressed_bytes,
                cr,
                0.
            );
        }
    }

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
