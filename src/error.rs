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
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum BiolepticError {
    InvalidCompressionMethod([u8; 4]),
    InvalidMagic([u8; 4]),
    InvalidDataType([u8; 2]),
    InvalidVersion([u8; 2]),
    UnderlyingDwtError(String),
    UnderlyingCompressorError(String),
    OutOfMemoryError(usize),
    InvalidHeader,
    UnsupportedCompressorConfiguration(String),
    DecompressionError(String),
    InvalidQuantizationScale(u8),
}

impl Display for BiolepticError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BiolepticError::InvalidCompressionMethod(v) => {
                let s = std::str::from_utf8(v).unwrap_or("????");
                f.write_fmt(format_args!("Invalid compression method '{}'", s))
            }
            BiolepticError::InvalidMagic(v) => {
                let s = std::str::from_utf8(v).unwrap_or("????");
                f.write_fmt(format_args!("Magic should be 'BILP' but it was '{}'", s))
            }
            BiolepticError::InvalidDataType(v) => {
                let s = std::str::from_utf8(v).unwrap_or("??");
                f.write_fmt(format_args!("Invalid compression method '{}'", s))
            }
            BiolepticError::InvalidVersion(v) => {
                let s = std::str::from_utf8(v).unwrap_or("??");
                f.write_fmt(format_args!("Invalid header version '{}'", s))
            }
            BiolepticError::OutOfMemoryError(s) => f.write_fmt(format_args!(
                "Out of memory, can't allocate additional {}",
                s
            )),
            BiolepticError::UnderlyingDwtError(e) => f.write_fmt(format_args!("{}", e)),
            BiolepticError::InvalidHeader => f.write_str("Header is invalid"),
            BiolepticError::UnderlyingCompressorError(e) => f.write_fmt(format_args!("{}", e)),
            BiolepticError::UnsupportedCompressorConfiguration(s) => f.write_fmt(format_args!(
                "Unsupported compression configuration '{}'",
                s
            )),
            BiolepticError::DecompressionError(s) => {
                f.write_fmt(format_args!("Can't decompress data, reason: {}", s))
            }
            BiolepticError::InvalidQuantizationScale(s) => f.write_fmt(format_args!(
                "Only scaled 6..12 is supported, but it was {s}"
            )),
        }
    }
}
