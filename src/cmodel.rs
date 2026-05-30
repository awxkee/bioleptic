/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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

//! Replaces the byte-plane bit-tree with a significance-map + sign +
//! per-subband magnitude model, coded on the same adaptive binary rANS
//! machinery as `arans.rs` (identical constants and `update` rule).
//!
//! Payload layout:  [nsub:u8][nbits per subband: u8 * nsub][rANS bytes]
//! Subband lengths are NOT stored: the decoder derives them from the DWT
//! geometry (same as the existing decompressor).

const PROB_BITS: u32 = 12;
const PROB_SCALE: u32 = 1 << PROB_BITS;
const STATE_LO: u32 = 1 << 16;
const P_INIT: u16 = 2048;
const RATE: u32 = 4;

#[derive(Debug)]
pub(crate) enum CModelError {
    Truncated,
    BadLength,
}
impl std::fmt::Display for CModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CModelError::Truncated => f.write_str("Stream truncated"),
            CModelError::BadLength => f.write_str("Inconsistent length in coefficient stream"),
        }
    }
}
impl std::error::Error for CModelError {}

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

// ---- generic context-indexed binary rANS ----
struct BinEnc {
    model: Vec<u16>,
    recs: Vec<(u32, u32)>,
}
impl BinEnc {
    fn new(nctx: usize) -> Self {
        BinEnc {
            model: vec![P_INIT; nctx],
            recs: Vec::new(),
        }
    }
    #[inline(always)]
    fn bit(&mut self, ctx: usize, bit: u8) {
        let p0 = self.model[ctx] as u32;
        let (f, s) = if bit == 0 {
            (p0, 0)
        } else {
            (PROB_SCALE - p0, p0)
        };
        self.recs.push((f, s));
        update(&mut self.model[ctx], bit);
    }
    fn finish(self) -> Vec<u8> {
        let mut x = STATE_LO;
        let mut rev = Vec::with_capacity(self.recs.len());
        for &(f, s) in self.recs.iter().rev() {
            enc_put(&mut x, f, s, &mut rev);
        }
        rev.push((x >> 24) as u8);
        rev.push((x >> 16) as u8);
        rev.push((x >> 8) as u8);
        rev.push(x as u8);
        rev.reverse();
        rev
    }
}

struct BinDec<'a> {
    model: Vec<u16>,
    x: u32,
    payload: &'a [u8],
    pp: usize,
}
impl<'a> BinDec<'a> {
    fn new(nctx: usize, payload: &'a [u8]) -> Result<Self, CModelError> {
        if payload.len() < 4 {
            return Err(CModelError::Truncated);
        }
        let x = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        Ok(BinDec {
            model: vec![P_INIT; nctx],
            x,
            payload,
            pp: 4,
        })
    }
    #[inline(always)]
    fn bit(&mut self, ctx: usize) -> Result<u8, CModelError> {
        let p0 = self.model[ctx] as u32;
        let mask = PROB_SCALE - 1;
        let slot = self.x & mask;
        let (bit, f, s) = if slot < p0 {
            (0u8, p0, 0)
        } else {
            (1u8, PROB_SCALE - p0, p0)
        };
        self.x = f * (self.x >> PROB_BITS) + slot - s;
        while self.x < STATE_LO {
            if self.pp + 2 > self.payload.len() {
                return Err(CModelError::Truncated);
            }
            let lo = self.payload[self.pp] as u32;
            let hi = self.payload[self.pp + 1] as u32;
            self.pp += 2;
            self.x = (self.x << 16) | (lo | (hi << 8));
        }
        update(&mut self.model[ctx], bit);
        Ok(bit)
    }
}

// ---- context layout per subband: [sig:2][sign:1][mag tree:1<<nbits] ----
struct Layout {
    off: Vec<usize>, // base offset of each subband's context block
    nbits: Vec<u32>,
    total: usize,
}
fn layout(nbits: &[u32]) -> Layout {
    let mut off = Vec::with_capacity(nbits.len());
    let mut base = 0usize;
    for &b in nbits {
        off.push(base);
        base += 3 + (1usize << b); // 2 sig + 1 sign + (1<<b) mag-tree nodes
    }
    Layout {
        off,
        nbits: nbits.to_vec(),
        total: base,
    }
}

#[inline(always)]
fn ctx_sig(l: &Layout, sb: usize, prev_sig: u8) -> usize {
    l.off[sb] + prev_sig as usize
}

#[inline(always)]
fn ctx_sign(l: &Layout, sb: usize) -> usize {
    l.off[sb] + 2
}

#[inline(always)]
fn ctx_mag(l: &Layout, sb: usize, node: usize) -> usize {
    l.off[sb] + 3 + node
}

fn nbits_for(maxmag: i32) -> u32 {
    let bl = if maxmag <= 0 {
        0
    } else {
        32 - (maxmag as u32).leading_zeros()
    };
    bl.max(1)
}

/// Encode subbands (each a slice of quantized i16 coefficients) in order.
pub(crate) fn encode_coeffs(subbands: &[&[i16]]) -> Vec<u8> {
    let nsub = subbands.len();
    let nbits: Vec<u32> = subbands
        .iter()
        .map(|sb| {
            let mx = sb.iter().map(|&v| (v as i32).abs()).max().unwrap_or(0);
            nbits_for(mx)
        })
        .collect();
    let l = layout(&nbits);
    let mut enc = BinEnc::new(l.total.max(1));
    for (sb, (coeffs, &b)) in subbands
        .iter()
        .zip(l.nbits[..subbands.len()].iter())
        .enumerate()
    {
        let mut prev_sig = 0u8;
        for &v in coeffs.iter() {
            let v = v as i32;
            let sig = (v != 0) as u8;
            enc.bit(ctx_sig(&l, sb, prev_sig), sig);
            if sig == 1 {
                enc.bit(ctx_sign(&l, sb), if v > 0 { 0 } else { 1 });
                let mag = v.unsigned_abs() - 1; // |v|>=1
                let mut node = 1usize;
                let mut i = b as i32 - 1;
                while i >= 0 {
                    let bit = ((mag >> i) & 1) as u8;
                    enc.bit(ctx_mag(&l, sb, node), bit);
                    node = (node << 1) | bit as usize;
                    i -= 1;
                }
            }
            prev_sig = sig;
        }
    }
    let rans = enc.finish();
    let mut out = Vec::with_capacity(1 + nsub + rans.len());
    out.push(nsub as u8);
    for &b in &nbits {
        out.push(b as u8);
    }
    out.extend_from_slice(&rans);
    out
}

/// Decode into per-subband Vecs given the known subband lengths.
pub(crate) fn decode_coeffs(
    payload: &[u8],
    lengths: &[usize],
) -> Result<Vec<Vec<i16>>, CModelError> {
    if payload.is_empty() {
        return Err(CModelError::Truncated);
    }
    let nsub = payload[0] as usize;
    if nsub != lengths.len() {
        return Err(CModelError::BadLength);
    }
    if payload.len() < 1 + nsub {
        return Err(CModelError::Truncated);
    }
    let nbits: Vec<u32> = payload[1..1 + nsub].iter().map(|&b| b as u32).collect();
    for &b in &nbits {
        if b > 16 {
            return Err(CModelError::BadLength);
        }
    }
    let l = layout(&nbits);
    let mut dec = BinDec::new(l.total.max(1), &payload[1 + nsub..])?;
    let mut out: Vec<Vec<i16>> = Vec::with_capacity(nsub);
    for (sb, &b) in l.nbits[..nsub].iter().enumerate() {
        let mut col = vec![0i16; lengths[sb]];
        let mut prev_sig = 0u8;
        for slot in col.iter_mut() {
            let sig = dec.bit(ctx_sig(&l, sb, prev_sig))?;
            if sig == 1 {
                let neg = dec.bit(ctx_sign(&l, sb))?;
                let mut node = 1usize;
                let mut mag = 0u32;
                for _ in 0..b {
                    let bit = dec.bit(ctx_mag(&l, sb, node))?;
                    mag = (mag << 1) | bit as u32;
                    node = (node << 1) | bit as usize;
                }
                let m = (mag + 1) as i32;
                *slot = (if neg == 1 { -m } else { m }) as i16;
            }
            prev_sig = sig;
        }
        out.push(col);
    }
    Ok(out)
}
