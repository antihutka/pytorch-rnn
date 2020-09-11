extern crate json;

use json::{Array, object};
use std::str;
use std::fs::File;
use std::io::Write;

pub fn write_json(tokens: Vec<Vec<u8>>, outpath: &str) {
	let mut jtoks = Array::new();
	for token in tokens {
		match str::from_utf8(&token) {
			Ok(t) => {
				jtoks.push(t.into());
			}
			Err(_) => {
				jtoks.push(token.into())
			}
		}
	}
	let jobj = object!{"idx_to_token" => jtoks};
	let mut file = File::create(outpath).unwrap();
	let cont = jobj.dump().into_bytes();
	file.write_all(&cont).unwrap();
}
