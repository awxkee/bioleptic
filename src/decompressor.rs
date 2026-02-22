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
use crate::{BIOLEPTIC_HEADER_SIZE, BiolepticError, BiolepticHeader, CompressionMethod};
use osclet::{BorderMode, DaubechiesFamily, DwtSize, MultiLevelDwtRef, Osclet, SymletFamily};
use std::io::Cursor;

/// Decompresses a Bioleptic-encoded byte slice back into `f32` samples.
///
/// Reads and validates the header, entropy-decodes the payload with zstd,
/// dequantizes coefficients, reconstructs the signal via inverse multi-level
/// DWT, then reverses the mean-centering and range normalization applied
/// during compression.
pub fn decompress(bytes: &[u8]) -> Result<Vec<f32>, BiolepticError> {
    let header = BiolepticHeader::from_bytes(bytes)?;

    let signal_length = header.signal_length as usize;

    if signal_length > i32::MAX as usize {
        return Err(BiolepticError::DecompressionError(format!(
            "Can't decompress data bigger than {}, but data was {}",
            i32::MAX,
            signal_length,
        )));
    }

    let dwt_levels = header.levels as usize;

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

    let compression_method = header.compression_method()?;

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

    let compressed_size = header.compressed_size as usize;

    let data_remainder_size = bytes.len() - BIOLEPTIC_HEADER_SIZE;
    if data_remainder_size < compressed_size {
        return Err(BiolepticError::DecompressionError(format!(
            "Minimum data size is {}, but it was {}",
            BIOLEPTIC_HEADER_SIZE + compressed_size,
            bytes.len(),
        )));
    }

    let compressed_data = &bytes[BIOLEPTIC_HEADER_SIZE..BIOLEPTIC_HEADER_SIZE + compressed_size];

    let decoded_data = zstd::decode_all(Cursor::new(&compressed_data)).unwrap();

    let quantized_data = decoded_data
        .chunks_exact(2)
        .map(|x| i16::from_le_bytes([x[0], x[1]]))
        .collect::<Vec<i16>>();

    let scale = header.scale;
    if scale < 6 || scale > 12 {
        return Err(BiolepticError::DecompressionError(format!(
            "Supported scales only [6, 12] but it was {scale}"
        )));
    }

    let rcp_scale = 1. / (1 << scale) as f32;

    let mut details = vec![];
    let mut details_start = levels_length[dwt_levels - 1].approx_length;
    for level in 0..dwt_levels {
        let detail_level: Vec<f32> = quantized_data
            [details_start..details_start + levels_length[level].details_length]
            .iter()
            .map(|&x| x as f32 * rcp_scale)
            .collect();
        details.push(detail_level);
        details_start += levels_length[level].details_length;
    }

    let mut iwdt = dwt_worker
        .multi_idwt(&MultiLevelDwtRef {
            approximations: &quantized_data[..levels_length[dwt_levels - 1].approx_length]
                .iter()
                .map(|&x| x as f32 * rcp_scale)
                .collect::<Vec<f32>>(),
            details: details.iter().map(|x| x.as_slice()).collect(),
        })
        .map_err(|x| BiolepticError::UnderlyingDwtError(x.to_string()))?;

    let range = header.max_f32() - header.min_f32();
    let v_min = header.min_f32();
    let v_mean = header.mean_f32();

    for v in iwdt.iter_mut() {
        *v = (*v + v_mean) * range + v_min;
    }

    Ok(iwdt)
}
