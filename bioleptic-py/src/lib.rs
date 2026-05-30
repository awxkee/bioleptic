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

use ::bioleptic::{
    CompressionMethod, CompressionOptions, CutoffLevel, EntropyCoder, QuantizationScale, compress,
    decompress,
};
use numpy::{IntoPyArray, Ix1, PyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(from_py_object, name = "CompressionOptions")]
#[derive(Clone)]
pub struct BiolpCompressionOptions {
    inner: CompressionOptions,
}

#[pymethods]
impl BiolpCompressionOptions {
    /// Build options from string-y Python arguments.
    ///
    /// * `method` — wavelet transform: `"cdf97"`, `"cdf53"`, `"db4"`, `"sym4"`.
    /// * `scale`  — quantization shift (DWT coeffs scaled by `1 << scale`); 6..=12.
    /// * `cutoff` — detail-threshold aggressiveness: `"low"`, `"medium"`, `"high"`.
    /// * `coder`  — payload entropy coder: `"deflate"`, `"arans"`, `"cmodel"`, `"auto"`.
    /// * `quant_multiplier` — explicit coefficient multiplier; overrides `scale`.
    #[new]
    #[pyo3(signature = (method = "cdf97", scale = 11, cutoff = "low", coder = None, quant_multiplier = None))]
    fn new(
        method: &str,
        scale: u8,
        cutoff: &str,
        coder: Option<&str>,
        quant_multiplier: Option<f32>,
    ) -> PyResult<Self> {
        let method = match method {
            "cdf97" => CompressionMethod::Cdf97,
            "cdf53" => CompressionMethod::Cdf53,
            "sym4" => CompressionMethod::Sym4,
            "db4" => CompressionMethod::Db4,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown method {other:?}, expected 'cdf97' or 'cdf53' or 'db4' or 'sym4'"
                )));
            }
        };
        let cutoff: CutoffLevel = match cutoff {
            "low" => CutoffLevel::Low,
            "medium" => CutoffLevel::Medium,
            "high" => CutoffLevel::High,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown cutoff level {other:?}, expected 'low' or 'medium' or 'high'"
                )));
            }
        };

        let scale =
            QuantizationScale::try_from(scale).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let entropy_coder: Option<EntropyCoder> = match coder {
            None | Some("auto") => None,
            Some("deflate") => Some(EntropyCoder::Deflate),
            Some("arans") => Some(EntropyCoder::Arans),
            Some("cmodel") => Some(EntropyCoder::Cmodel),
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "Unknown coder {other:?}, expected 'auto', 'deflate', 'arans'"
                )));
            }
        };

        Ok(Self {
            inner: CompressionOptions {
                method,
                scale,
                entropy_coder,
                cutoff_level: cutoff,
                quant_multiplier,
            },
        })
    }
}

/// Compress a 1-D float32 **or** float64 NumPy array into Bioleptic `bytes`.
///
/// float64 input is downcast to float32 before compression (the codec works in
/// f32 internally and always reconstructs f32 on decompress).
#[pyfunction]
#[pyo3(signature = (data, options = None))]
fn compress_signal<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    options: Option<BiolpCompressionOptions>,
) -> PyResult<Bound<'py, PyBytes>> {
    let opts = options.map(|o| o.inner).unwrap_or_default();

    let result = if let Ok(arr) = data.cast::<PyArray1<f32>>() {
        let ro = arr.readonly();
        compress(ro.as_slice()?, opts)
    } else if let Ok(arr) = data.cast::<PyArray1<f64>>() {
        let ro = arr.readonly();
        let buf: Vec<f32> = ro.as_slice()?.iter().map(|&x| x as f32).collect::<Vec<_>>();
        compress(&buf, opts)
    } else {
        return Err(PyValueError::new_err(
            "expected a 1-D float32 or float64 NumPy array",
        ));
    };

    let bytes = result.map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &bytes))
}

/// Decompress a Bioleptic bytes object into a 1-D float32 NumPy array.
#[pyfunction]
fn decompress_signal<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyArray<f32, Ix1>>> {
    let signal = decompress(data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let pyarray = signal.into_pyarray(py);
    Ok(pyarray)
}

#[pymodule]
fn bioleptic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BiolpCompressionOptions>()?;
    m.add_function(wrap_pyfunction!(compress_signal, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_signal, m)?)?;
    Ok(())
}
