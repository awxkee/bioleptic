# Bioleptic

Wavelet-based signal compression for physiological data (ECG, PPG, accelerometry).

## Algorithm

1. Non-finite substitution (`NaN` → 0, `±inf` → 0/1)
2. Mean-centering + range normalization
3. Multi-level DWT (CDF 5/3 or CDF 9/7, 5 levels)
4. Quantization to `i16` with configurable scale
5. Detail coefficient thresholding
6. Entropy coding with deflate

### Rust

```rust
let compressed = compress(&signal, CompressionOptions::default())?;
let recovered  = decompress(&compressed)?;
```

### Python

```bash
pip install bioleptic-py
```

```python
from bioleptic import compress_signal, decompress_signal, BiolpCompressionOptions

compressed = compress_signal(signal, BiolpCompressionOptions("cdf97", 11, "low"))
recovered  = decompress_signal(compressed)
```

### JavaScript

```bash
npm install bioleptic-js
```

```js
import {
    BiolpCompressionMethod,
    BiolpCompressionOptions, BiolpCutoffLevel,
    BiolpQuantizationScale,
    compress_signal,
    decompress_signal
} from "bioleptic-js";

const signal = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

const options = new BiolpCompressionOptions(
    BiolpCompressionMethod.Cdf97,
    BiolpQuantizationScale.S11,
    BiolpCutoffLevel.Low,
);

const compressed = compress_signal(signal, options);  // Uint8Array
const recovered = decompress_signal(compressed);     // Float32Array
console.info("Recovered signal", recovered);
```

----

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.