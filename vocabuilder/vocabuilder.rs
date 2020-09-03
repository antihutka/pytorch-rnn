extern crate clap;
extern crate memmap;
extern crate radix_trie;
extern crate regex;

mod utfdec;

use utfdec::decode_utf8;
use clap::{Arg, App};
use memmap::MmapOptions;
use memmap::Mmap;
use std::fs::File;
use std::cmp::{min,max};
use std::collections::HashSet;
use radix_trie::Trie;
use regex::bytes::Regex;

struct VocabuilderOptions {
	min_merge: usize,
	min_keep: usize,
}

fn get_basic_tokens(input: &Mmap) -> HashSet<u8> {
	let mut bt = HashSet::new();
	for x in input.iter() {
		if !bt.contains(x) {
			bt.insert(*x);
		}
	}
	return bt
}

fn sorted_token_counts(token_counts: &[usize]) -> Vec<(usize, usize)> {
	let mut tc : Vec<(usize, usize)> = token_counts.iter().cloned().enumerate().collect();
	tc.sort_by(|a, b| b.1.cmp(&a.1));
	return tc
}

fn print_token_counts(tokens : &Vec<Vec<u8>>, sorted_tokens: &Vec<(usize,usize)>) {
	for (idx, cnt) in sorted_tokens[0..25].iter() {
		print!("{:?} -> {},  ", decode_utf8(&tokens[*idx]), cnt);
	}
	println!("");
}

fn sorted_pair_counts(pair_counts: &Vec<Vec<usize>>) -> Vec<(usize, usize, usize)> {
	let mut cnts = Vec::new();
	for (idx1, iv) in pair_counts.iter().enumerate() {
		for (idx2, cnt) in iv.iter().enumerate() {
			cnts.push((idx1, idx2, *cnt));
		}
	}
	cnts.sort_by(|a, b| b.2.cmp(&a.2));
	return cnts;
}

fn concat_tokens(tokens: &Vec<Vec<u8>>, idx1: usize, idx2: usize) -> Vec<u8> {
	return [&tokens[idx1][..], &tokens[idx2][..]].concat();
}

fn print_pair_counts(tokens: &Vec<Vec<u8>>, sorted_pairs: &Vec<(usize, usize, usize)>) {
	for (idx1, idx2, cnt) in sorted_pairs[0..25].iter() {
		let mtoken = concat_tokens(tokens, *idx1, *idx2);
		print!("{:?} -> {},  ", decode_utf8(&mtoken), cnt);
	}
	println!();
}

fn filter_token_pairs(tokens: &Vec<Vec<u8>>, pairs_in: Vec<(usize, usize, usize)>) -> Vec<(usize, usize, usize)> {
	let re = Regex::new(r"\n|[^A-Za-z0-9_][A-Za-z0-9_]").unwrap();
	let mut pairs_out = Vec::new();
	for (t1, t2, cnt) in pairs_in {
		let merged = concat_tokens(tokens, t1, t2);
		if re.is_match(&merged) {
			//println!("Not allowing token {:?}", decode_utf8(&merged));
			continue;
		}
		pairs_out.push((t1,t2,cnt));
	}
	return pairs_out;
}

fn count_tokens_pairs(input_data: &[u8], tokens: &Vec<Vec<u8>>) -> (usize, Vec<usize>, Vec<Vec<usize>>) {
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
	let mut total_tokens = 0;
	while remain.len() > 0 {
		let v = *trie.get_ancestor_value(&remain[..min(max_token_len,remain.len())]).unwrap();
		let k = &tokens[v];
		token_counts[v] += 1;
		remain = &remain[k.len()..];
		if prev_token != std::usize::MAX {
			pair_counts[prev_token][v] += 1;
		}
		prev_token = v;
		total_tokens += 1;
	}
	return (total_tokens, token_counts, pair_counts)
}

fn do_bpe(input_data: &[u8], basic_tokens: HashSet<u8>, opts: &VocabuilderOptions) {
	let mut merged_tokens = HashSet::<Vec<u8>>::new();
	loop {
		let mut tokens = Vec::new();
		for x in basic_tokens.iter() {
			tokens.push(vec![*x]);
		}
		for x in merged_tokens.iter() {
			tokens.push(x.clone());
		}
		let (total_tokens, token_counts, pair_counts) = count_tokens_pairs(input_data, &tokens);
		println!("Tokenized {} bytes to {} tokens, {:.4} tpb", input_data.len(), total_tokens, input_data.len() as f64 / total_tokens as f64);
		let sorted_tokens = sorted_token_counts(&token_counts);
		print_token_counts(&tokens, &sorted_tokens);
		let sorted_pairs = filter_token_pairs(&tokens, sorted_pair_counts(&pair_counts));
		print_pair_counts(&tokens, &sorted_pairs);
		
		let mut changed = false;
		if sorted_pairs[0].2 >= opts.min_merge {
			for sp in sorted_pairs.iter().take(10) {
				let (t1, t2, tc) = *sp;
				if tc < opts.min_merge {
					break;
				}
				let merged = concat_tokens(&tokens, t1, t2);
				println!("Merging top pair {:?} + {:?} -> {:?}", decode_utf8(&tokens[t1]), decode_utf8(&tokens[t2]), decode_utf8(&merged));
				merged_tokens.insert(merged);
			}
			changed = true;
		}
		for st in sorted_tokens.iter() {
			if st.1 <= opts.min_keep && merged_tokens.contains(&tokens[st.0]) {
				println!("Unmerging token {:?} ({})", decode_utf8(&tokens[st.0]), st.1);
				merged_tokens.remove(&tokens[st.0]);
				changed = true;
			}
		}
		if !changed {
			return;
		}
	}
}

fn main() {
	println!("Hello, world");
	let matches = App::new("Vocabuilder")
		.version("0.0.1")
		.arg(Arg::with_name("input").short("i").long("input").takes_value(true))
		.arg(Arg::with_name("min_merge").long("min-merge").takes_value(true))
		.arg(Arg::with_name("min_keep").long("min-keep").takes_value(true))
		.get_matches();
	let input_file = matches.value_of("input").unwrap();
	let min_merge = matches.value_of("min_merge").unwrap_or("1000").parse().unwrap();
	let min_keep = matches.value_of("min_keep").unwrap_or("500").parse().unwrap();
	let vopts = VocabuilderOptions{min_merge, min_keep};
	println!("input file {}", input_file);
	let file = File::open(input_file).unwrap();
	let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
	let input_length = mmap.len();
	println!("input length {}", input_length);
	println!("scanning basic tokens");
	let basic_tokens = get_basic_tokens(&mmap);
	println!("done, {} tokens found", basic_tokens.len());
	do_bpe(&mmap[0..input_length-1], basic_tokens, &vopts);
}
