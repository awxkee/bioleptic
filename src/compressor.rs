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
use crate::{BiolepticError, BiolepticHeader, CompressionMethod, DataType};
use osclet::{BorderMode, DaubechiesFamily, Osclet, SymletFamily};
use std::io::Cursor;

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug, Default)]
pub enum CutoffLevel {
    #[default]
    Low,
    Medium,
    High,
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug)]
#[repr(u8)]
pub enum QuantizationScale {
    S6 = 6,
    S7 = 7,
    S8 = 8,
    S9 = 9,
    S10 = 10,
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

impl Default for QuantizationScale {
    fn default() -> Self {
        Self::S11
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

#[derive(Copy, Clone, Hash, Debug)]
pub struct CompressionOptions {
    pub method: CompressionMethod,
    pub scale: QuantizationScale,
    pub cutoff_level: CutoffLevel,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        CompressionOptions {
            method: CompressionMethod::Cdf97,
            scale: QuantizationScale::S11,
            cutoff_level: CutoffLevel::default(),
        }
    }
}

impl CompressionOptions {
    pub fn from_method(method: CompressionMethod) -> Self {
        let mut q = Self::default();
        q.method = method;
        q
    }
}

fn threshold(details: &mut [i16], scale: QuantizationScale, cutoff_level: CutoffLevel) {
    let mut threshold = match scale {
        QuantizationScale::S6 => 1,
        QuantizationScale::S7 => 1,
        QuantizationScale::S8 => 2,
        QuantizationScale::S9 => 2,
        QuantizationScale::S10 => 3,
        QuantizationScale::S11 => 3,
        QuantizationScale::S12 => 4,
    };
    match cutoff_level {
        CutoffLevel::Low => {}
        CutoffLevel::Medium => {
            threshold = threshold * 3;
        }
        CutoffLevel::High => {
            threshold = threshold * 7;
        }
    }
    for det in details.iter_mut() {
        if det.unsigned_abs() < threshold {
            *det = 0;
        }
    }
}

/// Compresses a slice of `f32` samples into a Bioleptic-encoded byte vector.
///
/// Non-finite values (`NaN`, `±inf`) are substituted before processing:
/// `NaN` and `-inf` become `0.0`, `+inf` becomes `1.0`. The signal is then
/// mean-centered and range-normalized, transformed with a multi-level DWT,
/// quantized to `i16`, thresholded, and entropy-coded with zstd.
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
    let mut v_min = f32::INFINITY;
    let mut v_max = f32::NEG_INFINITY;
    let mut working_data = vec![0.; data.len()];
    for (dst, &src) in working_data.iter_mut().zip(data.iter()) {
        let val = if src.is_finite() {
            src
        } else {
            if src.is_nan() {
                0.
            } else {
                if src.is_sign_negative() { 0. } else { 1. }
            }
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
            *dst = *dst - v_mean;
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

    let level = if data.len() < 20 {
        1
    } else if data.len() < 40 {
        2
    } else if data.len() < 60 {
        3
    } else if data.len() < 80 {
        4
    } else {
        5
    };

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

    let scale_multiplier = options.scale.multiplier();

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

    approximation
        .try_reserve_exact(total_details_length)
        .map_err(|_| BiolepticError::OutOfMemoryError(total_details_length))?;

    for level_details in details.iter() {
        approximation.extend_from_slice(&level_details);
    }

    let approximation_bytes = approximation
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();

    let compressed_data = zstd::encode_all(Cursor::new(approximation_bytes), 0)
        .map_err(|x| BiolepticError::UnderlyingCompressorError(x.to_string()))?;

    let header = BiolepticHeader::new(
        DataType::Float32,
        options.method,
        level as u8,
        options.scale,
        data.len() as u32,
        v_min,
        v_max,
        v_mean,
        compressed_data.len() as u32,
    );

    let mut header_bytes = header.to_bytes().to_vec();
    header_bytes.extend_from_slice(&compressed_data);

    Ok(header_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompressor::decompress;

    /// Generates a synthetic PPG-like signal.
    /// Models the systolic peak, dicrotic notch, and diastolic peak.
    pub fn generate_ppg(samples: usize, sample_rate: f32, heart_rate_bpm: f32) -> Vec<f32> {
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

    pub fn prd(original: &[f32], reconstructed: &[f32]) -> f64 {
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
        let encoded = compress(
            &r_means,
            CompressionOptions::from_method(CompressionMethod::Sym4),
        )
        .unwrap();
        println!("{:?}", encoded.len());
        let decompressed = decompress(&encoded).unwrap();
        println!("{:?}", decompressed.len());
        let prd = prd(&r_means, &decompressed);
        assert!(prd < 0.5);
        println!("PRD {}", prd);
    }
}
