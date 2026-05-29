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
use std::fmt;
use std::fmt::Display;

const PROB_BITS: u32 = 12;
const PROB_SCALE: u32 = 1 << PROB_BITS;
const STATE_LO: u32 = 1 << 16;
const P_INIT: u16 = 2048;
const RATE: u32 = 4;

const MAX_COEFF: usize = 1 << 28; // 268M coeffs

fn pick_ctx_bits(len: usize) -> u32 {
    if len < 1 << 13 { 3 } else { 8 }
}

#[inline(always)]
fn update(p: &mut u16, bit: u8) {
    if bit == 0 {
        *p += ((PROB_SCALE - *p as u32) >> RATE) as u16;
    } else {
        *p -= (*p as u32 >> RATE) as u16;
    }
    if *p < 1 {
        *p = 1;
    }
    if *p > (PROB_SCALE - 1) as u16 {
        *p = (PROB_SCALE - 1) as u16;
    }
}

#[inline(always)]
fn enc_put(x: &mut u32, freq: u32, start: u32, rev: &mut Vec<u8>) {
    let upper = ((STATE_LO >> PROB_BITS) << 16).wrapping_mul(freq);
    while *x >= upper {
        rev.push((*x >> 8) as u8);
        rev.push(*x as u8);
        *x >>= 16;
    }
    *x = (*x / freq) * PROB_SCALE + (*x % freq) + start;
}

fn encode_plane(data: &[u8], ctx_bits: u32) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let nctx = 1usize << ctx_bits;
    let shift = 8 - ctx_bits;
    let mut model = vec![P_INIT; nctx * 256];
    let mut recs: Vec<(u32, u32)> = Vec::with_capacity(data.len() * 8);
    let mut prev = 0usize;
    for &byte in data {
        let base = (prev >> shift) * 256;
        let mut node = 1usize;
        for i in (0..8).rev() {
            let bit = (byte >> i) & 1;
            let idx = base + node;
            let p0 = model[idx] as u32;
            let (f, s) = if bit == 0 {
                (p0, 0)
            } else {
                (PROB_SCALE - p0, p0)
            };
            recs.push((f, s));
            update(&mut model[idx], bit);
            node = (node << 1) | bit as usize;
        }
        prev = byte as usize;
    }
    let mut x = STATE_LO;
    let mut rev = Vec::with_capacity(data.len());
    for &(f, s) in recs.iter().rev() {
        enc_put(&mut x, f, s, &mut rev);
    }
    rev.push((x >> 24) as u8);
    rev.push((x >> 16) as u8);
    rev.push((x >> 8) as u8);
    rev.push(x as u8);
    rev.reverse();
    rev
}

fn decode_plane(payload: &[u8], n: usize, ctx_bits: u32) -> Result<Vec<u8>, AransError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if payload.len() < 4 {
        return Err(AransError::Truncated);
    }
    let mut pp = 0usize;
    let mut x = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    pp += 4;
    let mask = PROB_SCALE - 1;
    let nctx = 1usize << ctx_bits;
    let shift = 8 - ctx_bits;
    let mut model = vec![P_INIT; nctx * 256];

    let mut out = Vec::new();
    out.try_reserve(n).map_err(|_| AransError::BadLength)?;

    let mut prev = 0usize;
    for _ in 0..n {
        let base = (prev >> shift) * 256;
        let mut node = 1usize;
        for _ in 0..8 {
            let idx = base + node;
            let p0 = model[idx] as u32;
            let slot = x & mask;
            let (bit, f, s) = if slot < p0 {
                (0u8, p0, 0)
            } else {
                (1u8, PROB_SCALE - p0, p0)
            };
            x = f * (x >> PROB_BITS) + slot - s;
            while x < STATE_LO {
                // need two bytes for a 16-bit renorm chunk
                if pp + 2 > payload.len() {
                    return Err(AransError::Truncated);
                }
                let lo = payload[pp] as u32;
                let hi = payload[pp + 1] as u32;
                pp += 2;
                x = (x << 16) | (lo | (hi << 8));
            }
            update(&mut model[idx], bit);
            node = (node << 1) | bit as usize;
        }
        out.push((node & 0xFF) as u8);
        prev = node & 0xFF;
    }
    Ok(out)
}

fn wv(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn rv(b: &[u8], p: &mut usize) -> Result<u64, AransError> {
    let mut v = 0u64;
    let mut s = 0u32;
    loop {
        if *p >= b.len() {
            return Err(AransError::Truncated);
        }
        let c = b[*p];
        *p += 1;
        v |= ((c & 0x7F) as u64) << s;
        if c & 0x80 == 0 {
            break;
        }
        s += 7;
        if s >= 64 {
            return Err(AransError::BadLength); // varint too long / would overflow
        }
    }
    Ok(v)
}

#[derive(Debug)]
pub(crate) enum AransError {
    /// Stream ended before a declared field/payload was fully read.
    Truncated,
    /// A declared length or context value is inconsistent or implausibly large.
    BadLength,
}

impl Display for AransError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AransError::Truncated => f.write_str("Stream truncated"),
            AransError::BadLength => f.write_str("Inconsistent or oversized length in stream"),
        }
    }
}
impl std::error::Error for AransError {}

/// Drop-in replacement for `lzrans::encode` on the i16 coefficient byte stream.
pub fn encode_stream(data: &[u8]) -> Vec<u8> {
    if data.len() <= 5 {
        return data.to_vec();
    }
    let ncoeff = data.len() / 2;
    let mut lo = Vec::with_capacity(ncoeff);
    let mut hi = Vec::with_capacity(ncoeff);
    for c in data.chunks_exact(2) {
        lo.push(c[0]);
        hi.push(c[1]);
    }
    let ctx = pick_ctx_bits(ncoeff);
    let elo = encode_plane(&lo, ctx);
    let ehi = encode_plane(&hi, ctx);
    let mut out = Vec::with_capacity(elo.len() + ehi.len() + 16);
    out.push((data.len() & 1) as u8); // parity flag: odd trailing byte present?
    out.push(ctx as u8);
    wv(ncoeff as u64, &mut out);
    wv(elo.len() as u64, &mut out);
    out.extend_from_slice(&elo);
    out.extend_from_slice(&ehi);
    if data.len() & 1 == 1 {
        out.push(data[data.len() - 1]); // odd tail byte verbatim
    }
    out
}

pub fn decode_stream(stream: &[u8]) -> Result<Vec<u8>, AransError> {
    // Mirror encode: empty -> empty, and the <=5 byte verbatim passthrough.
    if stream.is_empty() {
        return Ok(Vec::new());
    }
    if stream.len() <= 5 {
        return Ok(stream.to_vec());
    }

    let mut p = 0usize;
    let odd = stream[p] & 1;
    p += 1;
    let ctx = stream[p] as u32;
    p += 1;
    if ctx > 8 {
        return Err(AransError::BadLength); // ctx_bits must be 0..=8
    }

    let ncoeff = rv(stream, &mut p)? as usize;
    if ncoeff > MAX_COEFF {
        return Err(AransError::BadLength);
    }
    let elo_len = rv(stream, &mut p)? as usize;

    // Layout: [ p .. p+elo_len ) = lo plane, [ .. len-tail ) = hi plane, [last] = odd tail.
    let tail = (odd == 1) as usize;
    let elo_end = p.checked_add(elo_len).ok_or(AransError::BadLength)?;
    let hi_end = stream
        .len()
        .checked_sub(tail)
        .ok_or(AransError::Truncated)?;
    if elo_end > hi_end {
        return Err(AransError::Truncated);
    }

    let lo = decode_plane(&stream[p..elo_end], ncoeff, ctx)?;
    let hi = decode_plane(&stream[elo_end..hi_end], ncoeff, ctx)?;
    if lo.len() != ncoeff || hi.len() != ncoeff {
        return Err(AransError::BadLength);
    }

    let total = ncoeff
        .checked_mul(2)
        .and_then(|v| v.checked_add(tail))
        .ok_or(AransError::BadLength)?;
    let mut out = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|_| AransError::BadLength)?;
    out.resize(total, 0);
    for ((dst, lo), hi) in out
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(lo[..ncoeff].iter())
        .zip(hi[..ncoeff].iter())
    {
        dst[0] = *lo;
        dst[1] = *hi;
    }
    if odd == 1 {
        out[total - 1] = stream[stream.len() - 1];
    }
    Ok(out)
}
