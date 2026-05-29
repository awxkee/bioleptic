#![no_main]

mod arans_enc;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let encoded = arans_enc::encode_stream(data);
    match arans_enc::decode_stream(&encoded) {
        Ok(v) => assert_eq!(data, v),
        Err(_) => {}
    }
});
