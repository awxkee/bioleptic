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
use crate::mla::fmla;
use crate::{
    BIOLEPTIC_HEADER_SIZE, BiolepticError, BiolepticHeader, CompressionMethod, arans, cmodel,
};
use flate2::read::DeflateDecoder;
use osclet::{BorderMode, DaubechiesFamily, DwtSize, MultiLevelDwtRef, Osclet, SymletFamily};
use std::io::Read;

/// Core decode: entropy-decode + dequantize + inverse DWT + denormalize, given
/// the shared transform parameters and the per-channel normalization/fidelity
/// numbers explicitly. Shared by single-channel `decompress` and the
/// multi-channel container.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decompress_core(
    payload: &[u8],
    compression_method: CompressionMethod,
    dwt_levels: usize,
    signal_length: usize,
    min: f32,
    max: f32,
    mean: f32,
    quant_multiplier: f32,
    scale: u8,
    entropy_coder: EntropyCoder,
) -> Result<Vec<f32>, BiolepticError> {
    if signal_length > i32::MAX as usize {
        return Err(BiolepticError::DecompressionError(format!(
            "Can't decompress data bigger than {}, but data was {}",
            i32::MAX,
            signal_length,
        )));
    }

    if dwt_levels > 10 {
        return Err(BiolepticError::DecompressionError(format!(
            "Max supported level is 10 but it was {}",
            dwt_levels,
        )));
    }
    if dwt_levels == 0 {
        return Err(BiolepticError::DecompressionError(format!(
            "Min supported level is 1 but it was {}",
            dwt_levels,
        )));
    }

    if !(6..=12).contains(&scale) {
        return Err(BiolepticError::DecompressionError(format!(
            "Supported scales only [6, 12] but it was {scale}"
        )));
    }

    let dwt_worker = match compression_method {
        CompressionMethod::Cdf53 => Osclet::make_cdf53_f32(),
        CompressionMethod::Cdf97 => Osclet::make_cdf97_f32(),
        CompressionMethod::Db4 => {
            Osclet::make_daubechies_f32(DaubechiesFamily::Db4, BorderMode::Wrap)
        }
        CompressionMethod::Sym4 => Osclet::make_symlet_f32(SymletFamily::Sym4, BorderMode::Wrap),
    };

    let mut levels_length: Vec<DwtSize> = vec![DwtSize::new(0); dwt_levels];
    levels_length[0] = dwt_worker.dwt_size(signal_length);

    for i in 1..dwt_levels {
        let level_size = dwt_worker.dwt_size(levels_length[i - 1].approx_length);
        levels_length[i] = level_size;
    }

    let rcp_scale = 1. / quant_multiplier;

    // Entropy-decode the payload into the quantized coefficient stream, laid out
    // as [approximation, detail0, detail1, ..., detail_{L-1}].
    let quantized_data: Vec<i16> = match entropy_coder {
        EntropyCoder::Cmodel => {
            let mut lengths = Vec::with_capacity(dwt_levels + 1);
            lengths.push(levels_length[dwt_levels - 1].approx_length);
            for level in levels_length.iter() {
                lengths.push(level.details_length);
            }
            let subbands = cmodel::decode_coeffs(payload, &lengths)
                .map_err(|x| BiolepticError::DecompressionError(x.to_string()))?;
            let total: usize = lengths.iter().sum();
            let mut q = Vec::with_capacity(total);
            for sb in subbands {
                q.extend_from_slice(&sb);
            }
            q
        }
        EntropyCoder::Deflate | EntropyCoder::Arans => {
            let decoded_data = match entropy_coder {
                EntropyCoder::Deflate => {
                    let mut decoder = DeflateDecoder::new(payload);
                    let mut decoded_data = Vec::new();
                    let total_out = decoder.total_out() as usize;
                    decoded_data
                        .try_reserve_exact(total_out)
                        .map_err(|_| BiolepticError::OutOfMemoryError(total_out))?;
                    decoded_data.resize(total_out, 0);
                    decoder
                        .read_to_end(&mut decoded_data)
                        .map_err(|x| BiolepticError::DecompressionError(x.to_string()))?;
                    decoded_data
                }
                EntropyCoder::Arans => arans::decode_stream(payload)
                    .map_err(|x| BiolepticError::DecompressionError(x.to_string()))?,
                EntropyCoder::Cmodel => unreachable!(),
            };
            decoded_data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|x| i16::from_le_bytes([x[0], x[1]]))
                .collect::<Vec<i16>>()
        }
    };

    let approx_length = levels_length[dwt_levels - 1].approx_length;

    let mut details = vec![];
    let mut details_start = approx_length;
    for level in 0..dwt_levels {
        let detail_level: Vec<f32> = quantized_data
            [details_start..details_start + levels_length[level].details_length]
            .iter()
            .map(|&x| x as f32 * rcp_scale)
            .collect();
        details.push(detail_level);
        details_start += levels_length[level].details_length;
    }

    let approximations: Vec<f32> = quantized_data[..approx_length]
        .iter()
        .map(|&x| x as f32 * rcp_scale)
        .collect();

    let mut iwdt = dwt_worker
        .multi_idwt(&MultiLevelDwtRef {
            approximations: &approximations,
            details: details.iter().map(|x| x.as_slice()).collect(),
        })
        .map_err(|x| BiolepticError::UnderlyingDwtError(x.to_string()))?;

    let range = max - min;
    for v in iwdt.iter_mut() {
        *v = fmla(*v + mean, range, min);
    }
    // DWT might produce a different size for odd-sized data, so truncate/pad.
    if iwdt.len() != signal_length {
        iwdt.resize(signal_length, 0.);
    }

    Ok(iwdt)
}

/// Decompresses a single-channel Bioleptic-encoded byte slice back into `f32`.
///
/// Reads and validates the header, then delegates to [`decompress_core`].
pub fn decompress(bytes: &[u8]) -> Result<Vec<f32>, BiolepticError> {
    let header = BiolepticHeader::from_bytes(bytes)?;

    let compressed_size = header.compressed_size as usize;

    // `from_bytes` guarantees bytes.len() >= BIOLEPTIC_HEADER_SIZE.
    let data_remainder_size = bytes.len() - BIOLEPTIC_HEADER_SIZE;
    if data_remainder_size < compressed_size {
        return Err(BiolepticError::DecompressionError(format!(
            "Minimum data size is {}, but it was {}",
            BIOLEPTIC_HEADER_SIZE + compressed_size,
            bytes.len(),
        )));
    }

    let compressed_data = &bytes[BIOLEPTIC_HEADER_SIZE..BIOLEPTIC_HEADER_SIZE + compressed_size];

    decompress_core(
        compressed_data,
        header.compression_method()?,
        header.levels as usize,
        header.signal_length as usize,
        header.min_f32(),
        header.max_f32(),
        header.mean_f32(),
        header.quant_multiplier(),
        header.scale,
        EntropyCoder::try_from(header.entropy_coder)?,
    )
}
