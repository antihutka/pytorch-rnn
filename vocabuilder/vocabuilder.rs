extern crate clap;
extern crate memmap;
extern crate radix_trie;

mod utfdec;

use utfdec::decode_utf8;
use clap::{Arg, App};
use memmap::MmapOptions;
use memmap::Mmap;
use std::fs::File;
use std::cmp::max;
use std::collections::HashMap;
use std::collections::HashSet;
use radix_trie::Trie;

fn get_basic_tokens(input: &Mmap) -> HashSet<u8> {
	let mut bt = HashSet::new();
	for x in input.iter() {
		if !bt.contains(x) {
			bt.insert(*x);
		}
	}
	return bt
}

fn print_token_counts(tokens : &Vec<Vec<u8>>, token_counts : &[usize]) {
	let mut tc : Vec<(usize, &usize)> = token_counts.iter().enumerate().collect();
	tc.sort_by(|a, b| b.1.cmp(a.1));
	for (idx, cnt) in tc[0..25].iter() {
		print!("{:?} -> {},  ", decode_utf8(&tokens[*idx]), cnt);
	}
	println!("");
}

fn print_pair_counts(tokens: &Vec<Vec<u8>>, pair_counts: &Vec<Vec<usize>>) {
	let mut cnts = Vec::new();
	for (idx1, iv) in pair_counts.iter().enumerate() {
		for (idx2, cnt) in iv.iter().enumerate() {
			cnts.push((idx1, idx2, cnt));
		}
	}
	cnts.sort_by(|a, b| b.2.cmp(a.2));
	//println!("{:?}", cnts);
	for (idx1, idx2, cnt) in cnts[0..25].iter() {
		let mtoken = [&tokens[*idx1][..], &tokens[*idx2][..]].concat();
		print!("{:?} -> {},  ", decode_utf8(&mtoken), cnt);
	}
	println!();
}

fn do_bpe(input_data: &[u8], basic_tokens: HashSet<u8>) {
	let mut tokens = Vec::new();
	for x in basic_tokens {
		tokens.push(vec![x]);
	}
	let mut trie = Trie::<&[u8], usize>::new();
	let mut max_token_len = 0;
	for (idx, tok) in tokens.iter().enumerate() {
		trie.insert(tok, idx);
		max_token_len = max(max_token_len, tok.len());
	}
	println!("Max token length = {}", max_token_len);
	let mut remain = input_data;
	let mut token_counts = vec![0; tokens.len()];
	let mut pair_counts = vec![vec![0; tokens.len()]; tokens.len()];
	let mut prev_token = std::usize::MAX;
	while remain.len() > 0 {
		let v = *trie.get_ancestor_value(&remain[..max_token_len]).unwrap();
		let k = &tokens[v];
		token_counts[v] += 1;
		remain = &remain[k.len()..];
		if prev_token != std::usize::MAX {
			pair_counts[prev_token][v] += 1;
		}
		prev_token = v;
	}
	//println!("Token counts: {:?}", token_counts);
	print_token_counts(&tokens, &token_counts);
	//println!("Pair counts: {:?}", pair_counts);
	print_pair_counts(&tokens, &pair_counts);
}

fn main() {
	println!("Hello, world");
	let matches = App::new("Vocabuilder")
		.version("0.0.1")
		.arg(Arg::with_name("input").short("i").long("input").takes_value(true))
		.get_matches();
	let input_file = matches.value_of("input").unwrap();
	println!("input file {}", input_file);
	let file = File::open(input_file).unwrap();
	let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
	let input_length = mmap.len();
	println!("input length {}", input_length);
	println!("scanning basic tokens");
	let basic_tokens = get_basic_tokens(&mmap);
	println!("done, {} tokens found", basic_tokens.len());
	do_bpe(&mmap[0..input_length-1], basic_tokens);
}
