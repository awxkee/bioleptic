# Bioleptic

**Fast, lossy wavelet compression for physiological signals — ECG, PPG, and accelerometry — with Rust, Python, and JavaScript/WebAssembly bindings.**

[![Crates.io](https://img.shields.io/crates/v/bioleptic.svg)](https://crates.io/crates/bioleptic)
[![PyPI](https://img.shields.io/pypi/v/bioleptic.svg)](https://pypi.org/project/bioleptic/)
[![npm](https://img.shields.io/npm/v/bioleptic-js.svg)](https://www.npmjs.com/package/bioleptic-js)

Bioleptic is a biosignal compression library that shrinks physiological time-series data — electrocardiogram (ECG), photoplethysmogram (PPG), and accelerometer streams — at high compression ratios with low reconstruction error (PRD). It combines a multi-level discrete wavelet transform (DWT) with scalar quantization and adaptive entropy coding, and runs natively in Rust or from Python (NumPy) and the browser (WebAssembly). That makes it a good fit for wearables, remote patient monitoring, medical-device storage, and large physiological datasets where bandwidth and disk are tight.

## Features

- **Wavelet-based** — CDF 9/7, CDF 5/3, Daubechies-4, and Symlet-4 transforms
- **Tunable rate–distortion** — pick a quantization scale (or an explicit multiplier) to trade size against fidelity for a target PRD
- **Multiple entropy coders** — Deflate, an adaptive binary rANS, and a significance-map coefficient model
- **Float32 and Float64 input** — f64 is downcast to f32 internally
- **Cross-platform** — one core, with Rust, Python/NumPy, and JavaScript/WASM frontends
- **Lightweight & dependency-light** — small footprint, suitable for embedded and edge use

## Algorithm

1. Non-finite substitution (`NaN` → 0, `±inf` → 0/1)
2. Mean-centering + range normalization
3. Multi-level DWT (CDF 9/7, CDF 5/3, Daubechies-4, or Symlet-4; depth scales with signal length)
4. Quantization to `i16` with a configurable scale
5. Detail-coefficient thresholding
6. Entropy coding (Deflate, adaptive rANS, or a significance-map coefficient model)

## Installation & Usage

### Rust

```bash
cargo add bioleptic
```

```rust
let compressed = compress(&signal, CompressionOptions::default())?;
let recovered  = decompress(&compressed)?;
```

### Python

```bash
pip install bioleptic-py
```

```python
from bioleptic import compress_signal, decompress_signal, CompressionOptions

compressed = compress_signal(signal, CompressionOptions("cdf97", 11, "low"))
recovered  = decompress_signal(compressed)
```

### JavaScript

```bash
npm install bioleptic-js
```

```js
import {
    CompressionMethod,
    CompressionOptions,
    CutoffLevel,
    QuantizationScale,
    compressSignal,
    decompressSignal
} from "bioleptic-js";

const signal = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

const options = new CompressionOptions(
    CompressionMethod.Cdf97,
    QuantizationScale.S11,
    CutoffLevel.Low,
);

const compressed = compressSignal(signal, options);  // Uint8Array
const recovered = decompressSignal(compressed);       // Float32Array
console.info("Recovered signal", recovered);
```

## License

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.