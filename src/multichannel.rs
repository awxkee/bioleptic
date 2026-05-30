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
use crate::FORMAT_MAJOR;

pub const BILX_MAGIC: [u8; 4] = *b"BILX";
pub const BILX_MAJOR: u8 = 1;
pub const BILX_MINOR: u8 = 0;
pub const BILX_VERSION: u16 = u16::from_ne_bytes([BILX_MAJOR, BILX_MINOR]);
pub(crate) const CHANNEL_META_SIZE: usize = 24;

#[derive(Debug)]
pub(crate) enum MultiError {
    NoChannels,
    LengthMismatch {
        channel: usize,
        expected: usize,
        got: usize,
    },
    Truncated,
    BadMagic([u8; 4]),
    BadVersion(u16),
    UnsupportedTransform(u8),
    UnsupportedFlag(u8),
    Core(String),
}
impl std::fmt::Display for MultiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MultiError::NoChannels => f.write_str("at least one channel is required"),
            MultiError::LengthMismatch {
                channel,
                expected,
                got,
            } => write!(
                f,
                "channel {channel} length {got} != shared length {expected}"
            ),
            MultiError::Truncated => f.write_str("container truncated"),
            MultiError::BadMagic(m) => write!(f, "bad magic {m:?}"),
            MultiError::BadVersion(v) => write!(f, "unsupported container version {v}"),
            MultiError::UnsupportedTransform(t) => {
                write!(f, "unsupported inter-channel transform {t}")
            }
            MultiError::UnsupportedFlag(x) => write!(f, "unsupported flags {x:#x}"),
            MultiError::Core(s) => write!(f, "channel codec error: {s}"),
        }
    }
}
impl std::error::Error for MultiError {}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ChannelMeta {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub quant_multiplier: f32,
    pub scale: u8,
    pub entropy_coder: u8,
    pub compressed_size: u32,
}
impl ChannelMeta {
    fn to_bytes(self) -> [u8; CHANNEL_META_SIZE] {
        let mut b = [0u8; CHANNEL_META_SIZE];
        b[0..4].copy_from_slice(&self.min.to_bits().to_le_bytes());
        b[4..8].copy_from_slice(&self.max.to_bits().to_le_bytes());
        b[8..12].copy_from_slice(&self.mean.to_bits().to_le_bytes());
        b[12..16].copy_from_slice(&self.quant_multiplier.to_bits().to_le_bytes());
        b[16] = self.scale;
        b[17] = self.entropy_coder;
        // b[18..20] reserved (per-channel flags, future)
        b[20..24].copy_from_slice(&self.compressed_size.to_le_bytes());
        b
    }
    fn from_bytes(b: &[u8]) -> Self {
        let rd = |s: &[u8]| f32::from_bits(u32::from_le_bytes(s.try_into().unwrap()));
        ChannelMeta {
            min: rd(&b[0..4]),
            max: rd(&b[4..8]),
            mean: rd(&b[8..12]),
            quant_multiplier: rd(&b[12..16]),
            scale: b[16],
            entropy_coder: b[17],
            compressed_size: u32::from_le_bytes(b[20..24].try_into().unwrap()),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ContainerHeader {
    pub n_channels: u16,
    pub signal_length: u32,
    pub compression_method: [u8; 4],
    pub levels: u8,
    pub inter_channel_transform: u8,
    pub flags: u8,
    pub data_type: u8,
    pub transform_data_len: u16,
    pub reserved: [u8; 8],
}

pub(crate) const CONTAINER_HEADER_SIZE: usize = size_of::<ContainerHeader>();

impl ContainerHeader {
    fn to_bytes(self) -> [u8; CONTAINER_HEADER_SIZE] {
        let mut b = [0u8; CONTAINER_HEADER_SIZE];
        b[0..4].copy_from_slice(&BILX_MAGIC);
        b[4..6].copy_from_slice(&BILX_VERSION.to_le_bytes());
        b[6..8].copy_from_slice(&self.n_channels.to_le_bytes());
        b[8..12].copy_from_slice(&self.signal_length.to_le_bytes());
        b[12..16].copy_from_slice(&self.compression_method);
        b[16] = self.levels;
        b[17] = self.inter_channel_transform;
        b[18] = self.flags;
        b[19] = self.data_type;
        b[20..22].copy_from_slice(&self.transform_data_len.to_le_bytes());
        // b[22..28] reserved
        b
    }
    fn from_bytes(b: &[u8]) -> Result<Self, MultiError> {
        if b.len() < CONTAINER_HEADER_SIZE {
            return Err(MultiError::Truncated);
        }
        let magic: [u8; 4] = b[0..4].try_into().unwrap();
        if magic != BILX_MAGIC {
            return Err(MultiError::BadMagic(magic));
        }
        let (major, minor) = (b[4], b[5]);
        if major != FORMAT_MAJOR {
            return Err(MultiError::BadVersion(u16::from_le_bytes([major, minor])));
        }
        Ok(ContainerHeader {
            n_channels: u16::from_le_bytes(b[6..8].try_into().unwrap()),
            signal_length: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            compression_method: b[12..16].try_into().unwrap(),
            levels: b[16],
            inter_channel_transform: b[17],
            flags: b[18],
            data_type: b[19],
            transform_data_len: u16::from_le_bytes(b[20..22].try_into().unwrap()),
            reserved: [0; 8],
        })
    }
}

/// The per-channel codec, decoupled from the container so the framing can be
/// validated without the wavelet backend. The real crate implements this over
/// `compressor::compress_core` / `decompressor::decompress_core`.
pub(crate) trait ChannelCore {
    type Err: std::fmt::Display;
    fn encode(
        &self,
        data: &[f32],
        method: [u8; 4],
        levels: u8,
    ) -> Result<(ChannelMeta, Vec<u8>), Self::Err>;
    fn decode(
        &self,
        meta: &ChannelMeta,
        payload: &[u8],
        method: [u8; 4],
        levels: u8,
        signal_length: u32,
    ) -> Result<Vec<f32>, Self::Err>;
}

fn assemble(header: &ContainerHeader, metas: &[ChannelMeta], payloads: &[Vec<u8>]) -> Vec<u8> {
    let body: usize = payloads.iter().map(|p| p.len()).sum();
    let mut out =
        Vec::with_capacity(CONTAINER_HEADER_SIZE + metas.len() * CHANNEL_META_SIZE + body);
    out.extend_from_slice(&header.to_bytes());
    // transform_data would go here for modes >= 1 (transform_data_len == 0 for mode 0)
    for m in metas {
        out.extend_from_slice(&m.to_bytes());
    }
    for p in payloads {
        out.extend_from_slice(p);
    }
    out
}

/// Parse into header + metas + borrowed payload slices.
#[allow(clippy::type_complexity)]
fn parse(bytes: &[u8]) -> Result<(ContainerHeader, Vec<ChannelMeta>, Vec<&[u8]>), MultiError> {
    let header = ContainerHeader::from_bytes(bytes)?;
    if header.inter_channel_transform != 0 {
        return Err(MultiError::UnsupportedTransform(
            header.inter_channel_transform,
        ));
    }
    if header.flags != 0 {
        return Err(MultiError::UnsupportedFlag(header.flags));
    }
    let n = header.n_channels as usize;
    if n == 0 {
        return Err(MultiError::NoChannels);
    }
    let mut pos = CONTAINER_HEADER_SIZE
        .checked_add(header.transform_data_len as usize)
        .ok_or(MultiError::Truncated)?;
    let metas_end = pos
        .checked_add(
            n.checked_mul(CHANNEL_META_SIZE)
                .ok_or(MultiError::Truncated)?,
        )
        .ok_or(MultiError::Truncated)?;
    if bytes.len() < metas_end {
        return Err(MultiError::Truncated);
    }
    let mut metas = Vec::with_capacity(n);
    for _ in 0..n {
        metas.push(ChannelMeta::from_bytes(
            &bytes[pos..pos + CHANNEL_META_SIZE],
        ));
        pos += CHANNEL_META_SIZE;
    }
    let mut payloads = Vec::with_capacity(n);
    for m in &metas {
        let cs = m.compressed_size as usize;
        let end = pos.checked_add(cs).ok_or(MultiError::Truncated)?;
        if end > bytes.len() {
            return Err(MultiError::Truncated);
        }
        payloads.push(&bytes[pos..end]);
        pos = end;
    }
    Ok((header, metas, payloads))
}

/// Compress equal-length channels. Errors (does not pad) on length mismatch.
pub(crate) fn compress_multi_with<C: ChannelCore>(
    core: &C,
    channels: &[&[f32]],
    method: [u8; 4],
    levels: u8,
    data_type: u8,
) -> Result<Vec<u8>, MultiError> {
    if channels.is_empty() {
        return Err(MultiError::NoChannels);
    }
    let len = channels[0].len();
    for (i, c) in channels.iter().enumerate() {
        if c.len() != len {
            return Err(MultiError::LengthMismatch {
                channel: i,
                expected: len,
                got: c.len(),
            });
        }
    }
    let mut metas = Vec::with_capacity(channels.len());
    let mut payloads = Vec::with_capacity(channels.len());
    for c in channels {
        let (mut meta, payload) = core
            .encode(c, method, levels)
            .map_err(|e| MultiError::Core(e.to_string()))?;
        meta.compressed_size = payload.len() as u32; // container owns the boundary
        metas.push(meta);
        payloads.push(payload);
    }
    let header = ContainerHeader {
        n_channels: channels.len() as u16,
        signal_length: len as u32,
        compression_method: method,
        levels,
        inter_channel_transform: 0,
        flags: 0,
        data_type,
        transform_data_len: 0,
        reserved: [0; 8],
    };
    Ok(assemble(&header, &metas, &payloads))
}

/// Decompress all channels back to `Vec<Vec<f32>>` in stored order.
pub(crate) fn decompress_multi_with<C: ChannelCore>(
    core: &C,
    bytes: &[u8],
) -> Result<Vec<Vec<f32>>, MultiError> {
    let (header, metas, payloads) = parse(bytes)?;
    let mut out = Vec::with_capacity(metas.len());
    for (meta, payload) in metas.iter().zip(payloads.iter()) {
        out.push(
            core.decode(
                meta,
                payload,
                header.compression_method,
                header.levels,
                header.signal_length,
            )
            .map_err(|e| MultiError::Core(e.to_string()))?,
        );
    }
    Ok(out)
}
