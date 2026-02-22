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
    CompressionMethod, CompressionOptions, CutoffLevel, QuantizationScale, compress, decompress,
};
use numpy::{IntoPyArray, Ix1, PyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyCompressionOptions {
    inner: CompressionOptions,
}

#[pymethods]
impl PyCompressionOptions {
    #[new]
    #[pyo3(signature = (method = "cdf97", scale = 11))]
    fn new(method: &str, scale: u8) -> PyResult<Self> {
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
        let scale =
            QuantizationScale::try_from(scale).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: CompressionOptions {
                method,
                scale,
                cutoff_level: CutoffLevel::default(),
            },
        })
    }
}

/// Compress a 1-D float32 NumPy array into a Bioleptic-encoded bytes object.
#[pyfunction]
#[pyo3(signature = (data, options = None))]
fn compress_signal<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    options: Option<PyCompressionOptions>,
) -> PyResult<Bound<'py, PyBytes>> {
    let opts = options.map(|o| o.inner).unwrap_or_default();
    let slice = data.as_slice()?;
    let bytes = compress(slice, opts).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &bytes).into())
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
    m.add_class::<PyCompressionOptions>()?;
    m.add_function(wrap_pyfunction!(compress_signal, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_signal, m)?)?;
    Ok(())
}
