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
use bioleptic::{
    CompressionMethod, CompressionOptions, CutoffLevel, QuantizationScale, compress, decompress,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct BiolpCompressionOptions {
    inner: CompressionOptions,
}

#[wasm_bindgen]
#[derive(Copy, Clone)]
pub enum BiolpCompressionMethod {
    Cdf97,
    Cdf53,
    Sym4,
    Db4,
}

#[wasm_bindgen]
#[derive(Copy, Clone)]
pub enum BiolpCutoffLevel {
    Low,
    Medium,
    High,
}

#[wasm_bindgen]
#[derive(Copy, Clone)]
pub enum BiolpQuantizationScale {
    S6 = 6,
    S7 = 7,
    S8 = 8,
    S9 = 9,
    S10 = 10,
    S11 = 11,
    S12 = 12,
}

impl From<BiolpQuantizationScale> for QuantizationScale {
    fn from(s: BiolpQuantizationScale) -> Self {
        match s {
            BiolpQuantizationScale::S6 => QuantizationScale::S6,
            BiolpQuantizationScale::S7 => QuantizationScale::S7,
            BiolpQuantizationScale::S8 => QuantizationScale::S8,
            BiolpQuantizationScale::S9 => QuantizationScale::S9,
            BiolpQuantizationScale::S10 => QuantizationScale::S10,
            BiolpQuantizationScale::S11 => QuantizationScale::S11,
            BiolpQuantizationScale::S12 => QuantizationScale::S12,
        }
    }
}

#[wasm_bindgen]
impl BiolpCompressionOptions {
    #[wasm_bindgen(constructor)]
    pub fn new(
        method: BiolpCompressionMethod,
        scale: BiolpQuantizationScale,
        cutoff: BiolpCutoffLevel,
    ) -> Result<BiolpCompressionOptions, JsError> {
        let method = match method {
            BiolpCompressionMethod::Cdf97 => CompressionMethod::Cdf97,
            BiolpCompressionMethod::Cdf53 => CompressionMethod::Cdf53,
            BiolpCompressionMethod::Sym4 => CompressionMethod::Sym4,
            BiolpCompressionMethod::Db4 => CompressionMethod::Db4,
        };
        let cutoff = match cutoff {
            BiolpCutoffLevel::Low => CutoffLevel::Low,
            BiolpCutoffLevel::Medium => CutoffLevel::Medium,
            BiolpCutoffLevel::High => CutoffLevel::High,
        };
        let scale = QuantizationScale::from(scale);
        Ok(Self {
            inner: CompressionOptions {
                method,
                scale,
                cutoff_level: cutoff,
            },
        })
    }
}

/// Compress a Float32Array into a Uint8Array.
#[wasm_bindgen]
pub fn compress_signal(
    data: &[f32],
    options: Option<BiolpCompressionOptions>,
) -> Result<Vec<u8>, JsError> {
    let opts = options.map(|o| o.inner).unwrap_or_default();
    compress(data, opts).map_err(|e| JsError::new(&e.to_string()))
}

/// Decompress a Uint8Array back into a Float32Array.
#[wasm_bindgen]
pub fn decompress_signal(data: &[u8]) -> Result<Vec<f32>, JsError> {
    decompress(data).map_err(|e| JsError::new(&e.to_string()))
}
