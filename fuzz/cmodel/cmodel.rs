#![no_main]

mod cmodel_enc;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let (prefix, i16_slice, _suffix) = unsafe { data.align_to::<i16>() };
    if !prefix.is_empty() {
        return;
    }

    if i16_slice.is_empty() {
        return;
    }

    // Try all channel counts from 1 to 7
    for num_channels in 1usize..=7 {
        let chunk_size = (i16_slice.len() / num_channels).max(1);
        let subbands: Vec<&[i16]> = (0..num_channels)
            .map(|i| {
                let start = (i * chunk_size).min(i16_slice.len());
                let end = ((i + 1) * chunk_size).min(i16_slice.len());
                &i16_slice[start..end]
            })
            .filter(|s| !s.is_empty())
            .collect();

        if subbands.is_empty() {
            continue;
        }

        let lengths: Vec<usize> = subbands.iter().map(|s| s.len()).collect();
        let encoded = cmodel_enc::encode_coeffs(&subbands);

        match cmodel_enc::decode_coeffs(&encoded, &lengths) {
            Ok(decoded) => assert_eq!(subbands, decoded),
            Err(_) => {}
        }
    }
});
