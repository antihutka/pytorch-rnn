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

pub fn read_json(outpath: &str) -> Vec<Vec<u8>> {
	let cont = std::fs::read_to_string(outpath).unwrap();
	let parsed = json::parse(&cont).unwrap();
	let mut tokens = Vec::new();
	for token in parsed["idx_to_token"].members() {
		if token.is_string() {
			let t = token.as_str().unwrap();
			tokens.push(t.as_bytes().to_owned());
		} else {
			let mut tbytes = Vec::new();
			for byte in token.members() {
				tbytes.push(byte.as_u8().unwrap());
			}
			tokens.push(tbytes)
		}
	}
	return tokens;
}
