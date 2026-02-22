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
use crate::compressor::QuantizationScale;
use crate::error::BiolepticError;

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug)]
pub enum CompressionMethod {
    Cdf53,
    Cdf97,
    Db4,
    Sym4,
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug)]
pub enum DataType {
    Float32,
}

const CDF53: u32 = u32::from_le_bytes(*b"cf53");
const CDF97: u32 = u32::from_le_bytes(*b"cf97");
const DB4: u32 = u32::from_le_bytes(*b"db04");
const SYM4: u32 = u32::from_le_bytes(*b"sym4");

impl TryFrom<u32> for CompressionMethod {
    type Error = BiolepticError;
    fn try_from(value: u32) -> Result<Self, BiolepticError> {
        match value {
            CDF97 => Ok(CompressionMethod::Cdf97),
            CDF53 => Ok(CompressionMethod::Cdf53),
            DB4 => Ok(CompressionMethod::Db4),
            SYM4 => Ok(CompressionMethod::Sym4),
            _ => Err(BiolepticError::InvalidCompressionMethod(
                value.to_ne_bytes(),
            )),
        }
    }
}

impl Into<u32> for CompressionMethod {
    fn into(self) -> u32 {
        match self {
            CompressionMethod::Cdf53 => CDF53,
            CompressionMethod::Cdf97 => CDF97,
            CompressionMethod::Db4 => DB4,
            CompressionMethod::Sym4 => SYM4,
        }
    }
}

const FLOAT_32: u16 = u16::from_le_bytes(*b"f3");

impl TryFrom<u16> for DataType {
    type Error = BiolepticError;

    fn try_from(value: u16) -> Result<Self, BiolepticError> {
        match value {
            FLOAT_32 => Ok(DataType::Float32),
            _ => Err(BiolepticError::InvalidDataType(value.to_ne_bytes())),
        }
    }
}

impl Into<u16> for DataType {
    fn into(self) -> u16 {
        match self {
            DataType::Float32 => FLOAT_32,
        }
    }
}

pub const BIOLEPTIC_MAGIC: [u8; 4] = *b"BILP";

/// Current format version.
pub const BIOLEPTIC_VERSION: u16 = u16::from_le_bytes([1, 0]);

/// Fixed size of the header in bytes.
pub const BIOLEPTIC_HEADER_SIZE: usize = size_of::<BiolepticHeader>();

#[repr(C, packed)]
pub struct BiolepticHeader {
    /// Magic bytes identifying the format: `b"BILP"`.
    pub magic: [u8; 4],
    /// Format version for forward compatibility.
    pub version: u16,
    /// Data type of the original signal encoded as a 2-byte tag (e.g. `b"f3"` for `f32`).
    pub data_type: u16,
    /// Compression method encoded as a 4-byte tag (e.g. `b"cf97"` for CDF 9/7).
    pub compression_method: [u8; 4],
    /// Number of DWT decomposition levels applied during compression.
    pub levels: u8,
    /// Quantization scale factor — DWT coefficients are multiplied by `1 << scale`
    /// before being cast to `i16`.
    pub scale: u8,
    /// Reserved for future use — must be zero.
    pub reserved0: [u8; 2],
    /// Number of samples in the original signal before compression.
    pub signal_length: u32,
    /// Minimum value of the signal after non-finite substitution, stored as `f32` bits
    /// via `f32::to_bits()`. Used together with `max` to reverse range normalization
    /// during decompression.
    pub min: u32,
    /// Maximum value of the signal after non-finite substitution, stored as `f32` bits
    /// via `f32::to_bits()`. Used together with `min` to reverse range normalization
    /// during decompression.
    pub max: u32,
    /// Arithmetic mean of the signal (computed after non-finite substitution), stored
    /// as `f32` bits via `f32::to_bits()`. Subtracted from the signal before range
    /// normalization; must be added back during decompression.
    pub mean: u32,
    /// Byte length of the compressed payload that follows the header. Allows readers
    /// to allocate exactly the right buffer and detect truncated streams without having
    /// to rely on EOF.
    pub compressed_size: u32,
    /// Reserved for future use — must be zero.
    pub reserved1: [u8; 16],
}

impl BiolepticHeader {
    /// Creates a new header with the current magic and version.
    pub fn new(
        data_type: DataType,
        compression_method: CompressionMethod,
        levels: u8,
        scale: QuantizationScale,
        signal_length: u32,
        min: f32,
        max: f32,
        mean: f32,
        compressed_size: u32,
    ) -> Self {
        let compression_method_impl: u32 = compression_method.into();
        Self {
            magic: BIOLEPTIC_MAGIC,
            version: BIOLEPTIC_VERSION,
            data_type: data_type.into(),
            compression_method: compression_method_impl.to_le_bytes(),
            levels,
            scale: scale.as_u8(),
            reserved0: [0; 2],
            signal_length,
            min: min.to_bits(),
            max: max.to_bits(),
            mean: mean.to_bits(),
            reserved1: [0; 16],
            compressed_size,
        }
    }

    /// Serializes the header to bytes in little-endian order.
    pub fn to_bytes(&self) -> [u8; BIOLEPTIC_HEADER_SIZE] {
        let mut buf = [0u8; BIOLEPTIC_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.data_type.to_le_bytes());
        buf[8..12].copy_from_slice(&self.compression_method);
        buf[12] = self.levels;
        buf[13] = self.scale;
        buf[14..16].copy_from_slice(&self.reserved0); // reserved0
        buf[16..20].copy_from_slice(&self.signal_length.to_le_bytes());
        buf[20..24].copy_from_slice(&self.min.to_le_bytes());
        buf[24..28].copy_from_slice(&self.max.to_le_bytes());
        buf[28..32].copy_from_slice(&self.mean.to_le_bytes());
        buf[32..36].copy_from_slice(&self.compressed_size.to_le_bytes());
        buf[36..52].copy_from_slice(&self.reserved1);
        buf
    }

    /// Deserializes a header from bytes, validating magic and version.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, BiolepticError> {
        if buf.len() < BIOLEPTIC_HEADER_SIZE {
            return Err(BiolepticError::InvalidHeader);
        }

        let buf = &buf[..BIOLEPTIC_HEADER_SIZE];

        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if magic != BIOLEPTIC_MAGIC {
            return Err(BiolepticError::InvalidMagic(magic));
        }

        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        if version != BIOLEPTIC_VERSION {
            return Err(BiolepticError::InvalidVersion(version.to_ne_bytes()));
        }

        let data_type = u16::from_le_bytes(buf[6..8].try_into().unwrap());
        let compression_method = u32::from_le_bytes(buf[8..12].try_into().unwrap());

        // validate enums
        let v_data_type = DataType::try_from(data_type)?;
        let _ = CompressionMethod::try_from(compression_method)?;
        let _ = QuantizationScale::try_from(buf[13]);

        let f_min = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        let f_max = u32::from_le_bytes(buf[24..28].try_into().unwrap());
        let f_mean = u32::from_le_bytes(buf[28..32].try_into().unwrap());

        match v_data_type {
            DataType::Float32 => {
                if !f32::from_bits(f_min).is_finite() {
                    return Err(BiolepticError::InvalidHeader);
                }
                if !f32::from_bits(f_max).is_finite() {
                    return Err(BiolepticError::InvalidHeader);
                }
                if !f32::from_bits(f_mean).is_finite() {
                    return Err(BiolepticError::InvalidHeader);
                }
            }
        }

        Ok(Self {
            magic,
            version,
            data_type,
            compression_method: buf[8..12].try_into().unwrap(),
            levels: buf[12],
            scale: buf[13],
            reserved0: buf[14..16].try_into().unwrap(),
            signal_length: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
            min: f_min,
            max: f_max,
            mean: f_mean,
            compressed_size: u32::from_le_bytes(buf[32..36].try_into().unwrap()),
            reserved1: buf[36..52].try_into().unwrap(),
        })
    }

    /// Returns the normalization min as `f32`.
    pub fn min_f32(&self) -> f32 {
        f32::from_bits(self.min)
    }

    /// Returns the normalization max as `f32`.
    pub fn max_f32(&self) -> f32 {
        f32::from_bits(self.max)
    }

    /// Returns the signal mean as `f32`.
    pub fn mean_f32(&self) -> f32 {
        f32::from_bits(self.mean)
    }

    /// Returns the compression method as an enum.
    pub fn compression_method(&self) -> Result<CompressionMethod, BiolepticError> {
        CompressionMethod::try_from(u32::from_le_bytes(self.compression_method))
    }

    /// Returns the data type as an enum.
    pub fn data_type(&self) -> Result<DataType, BiolepticError> {
        DataType::try_from(self.data_type)
    }
}

impl std::fmt::Debug for BiolepticHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BiolepticHeader")
            .field("magic", &std::str::from_utf8(&self.magic).unwrap_or("????"))
            .field("version", &{ self.version })
            .field("data_type", &self.data_type())
            .field("compression_method", &self.compression_method())
            .field("levels", &self.levels)
            .field("scale", &self.scale)
            .field("signal_length", &{ self.signal_length })
            .field("min", &self.min_f32())
            .field("max", &self.max_f32())
            .field("mean", &self.mean_f32())
            .finish()
    }
}
