use std::str;
use std::fmt;

pub fn decode_utf8(mut bytes: &[u8]) -> String {
    let mut output = String::new();

    loop {
        match str::from_utf8(bytes) {
            Ok(s) => {
                // The entire rest of the string was valid UTF-8, we are done
                output.push_str(s);
                return output;
            }
            Err(e) => {
                let (good, bad) = bytes.split_at(e.valid_up_to());

                if !good.is_empty() {
                    let s = unsafe {
                        // This is safe because we have already validated this
                        // UTF-8 data via the call to `str::from_utf8`; there's
                        // no need to check it a second time
                        str::from_utf8_unchecked(good)
                    };
                    output.push_str(s);
                }

                if bad.is_empty() {
                    //  No more data left
                    return output;
                }

                // Do whatever type of recovery you need to here
                output.push_str(&fmt::format(format_args!("<byte:{:2x}>", bad[0])));

                // Skip the bad byte and try again
                bytes = &bad[1..];
            }
        }
    }
}
