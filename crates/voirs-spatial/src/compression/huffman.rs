//! Canonical Huffman entropy coding for spatial-audio byte streams.
//!
//! This module implements a self-contained, dependency-free, deterministic
//! **canonical Huffman** coder operating on raw bytes. It replaces the previous
//! run-length placeholder used by [`crate::compression`] for the entropy-coding
//! stage and provides an exact, lossless round-trip ([`decode`] inverts
//! [`encode`] for every possible input).
//!
//! # Algorithm
//!
//! ## Encoding
//! 1. Build a 256-entry byte-frequency histogram of the input.
//! 2. Construct a Huffman tree with a deterministic tie-break (lowest combined
//!    frequency first; ties broken by an insertion sequence number) and read off
//!    the *code length* (depth) for every symbol that occurs.
//! 3. Convert those lengths to a **canonical** Huffman code. Canonical codes can
//!    be reconstructed from the per-symbol lengths alone, so the bitstream itself
//!    never needs to carry an explicit code table — only the 256 lengths.
//! 4. Bit-pack the per-symbol codes (MSB-first) into the payload, preceded by a
//!    32-bit original-length field so the decoder knows exactly how many symbols
//!    to emit (and can therefore ignore trailing pad bits).
//!
//! ## Decoding
//! 1. Read the format marker and original length.
//! 2. Rebuild the canonical code lengths from the compact header.
//! 3. Recompute the identical canonical codes and walk the bitstream MSB-first,
//!    emitting one byte per decoded code until `original_len` bytes are produced.
//!
//! # Header / container format
//!
//! The first output byte is a **format marker**:
//!
//! * [`FORMAT_RAW`] – the remaining bytes are the original input verbatim. Used
//!   whenever Huffman coding would not shrink the data (incompressible / tiny
//!   inputs and the empty input), guaranteeing the coder *never inflates* beyond
//!   `input.len() + 1`.
//! * [`FORMAT_SINGLE`] – the input consists of a single distinct byte value. The
//!   next byte is that value and the following 4 bytes are the little-endian
//!   repeat count. (A degenerate alphabet of size one has no meaningful Huffman
//!   tree, so it is stored explicitly with length-1 semantics.)
//! * [`FORMAT_HUFFMAN`] – a canonical-Huffman block laid out as:
//!     - 4 bytes little-endian: original (decoded) length in bytes.
//!     - 1 byte: number of present symbols minus one (`0..=255`).
//!     - For each present symbol: 1 byte symbol value + 1 byte code length.
//!     - Bit-packed payload (MSB-first), padded to a whole byte with zero bits.
//!
//! The sparse `(symbol, length)` table is compact for the skewed distributions
//! typical of quantized audio while still allowing a full 256-symbol alphabet.

/// Marker: the payload is stored uncompressed (raw bytes follow).
const FORMAT_RAW: u8 = 0;
/// Marker: a single distinct byte repeated; value + repeat count follow.
const FORMAT_SINGLE: u8 = 1;
/// Marker: a canonical-Huffman-coded block follows.
const FORMAT_HUFFMAN: u8 = 2;

/// Number of distinct byte values.
const ALPHABET: usize = 256;

/// Encode `data` with canonical Huffman coding.
///
/// The returned buffer always begins with a one-byte format marker and is
/// guaranteed never to exceed `data.len() + 1` bytes (the coder falls back to a
/// raw block when compression would not help), so the transform can never
/// inflate the stream by more than the single marker byte.
pub fn encode(data: &[u8]) -> Vec<u8> {
    // Empty input: a bare RAW marker round-trips to an empty vector.
    if data.is_empty() {
        return vec![FORMAT_RAW];
    }

    // Histogram of byte frequencies.
    let mut freq = [0u64; ALPHABET];
    for &byte in data {
        freq[byte as usize] += 1;
    }
    let distinct = freq.iter().filter(|&&f| f > 0).count();

    // Single distinct symbol: a degenerate alphabet of size one (length-1 case)
    // is stored as value + repeat count. The explicit form is a fixed 6 bytes,
    // so for very short runs (<= 4 bytes) a raw block is smaller; fall back to
    // raw there to honour the never-inflate guarantee.
    if distinct == 1 {
        let symbol = freq.iter().position(|&f| f > 0).unwrap_or(0) as u8;
        // Single form: marker + symbol + 4-byte count = 6 bytes.
        // Raw form: marker + data = data.len() + 1 bytes.
        // The single form is no larger than raw once data.len() >= 5.
        if data.len() >= 5 {
            let mut out = Vec::with_capacity(6);
            out.push(FORMAT_SINGLE);
            out.push(symbol);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            return out;
        }
        let mut raw = Vec::with_capacity(data.len() + 1);
        raw.push(FORMAT_RAW);
        raw.extend_from_slice(data);
        return raw;
    }

    // Derive canonical code lengths from a Huffman tree.
    let lengths = build_code_lengths(&freq);
    let codes = canonical_codes(&lengths);

    // Bit-pack the payload MSB-first.
    let mut writer = BitWriter::new();
    for &byte in data {
        let idx = byte as usize;
        writer.write_code(codes[idx], lengths[idx]);
    }
    let payload = writer.finish();

    // Assemble the Huffman container.
    let mut out = Vec::with_capacity(payload.len() + 2 * distinct + 6);
    out.push(FORMAT_HUFFMAN);
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.push((distinct - 1) as u8);
    for symbol in 0..ALPHABET {
        if lengths[symbol] > 0 {
            out.push(symbol as u8);
            out.push(lengths[symbol]);
        }
    }
    out.extend_from_slice(&payload);

    // Never inflate: if the Huffman encoding is not smaller than a raw block
    // (raw block = 1 marker byte + the original `data.len()` bytes), fall back
    // to storing the original bytes verbatim.
    if out.len() > data.len() {
        let mut raw = Vec::with_capacity(data.len() + 1);
        raw.push(FORMAT_RAW);
        raw.extend_from_slice(data);
        raw
    } else {
        out
    }
}

/// Decode a buffer produced by [`encode`] back to the exact original bytes.
///
/// Returns `Err` with a descriptive message if the buffer is malformed
/// (truncated header, inconsistent lengths, or an undecodable bitstream).
pub fn decode(data: &[u8]) -> Result<Vec<u8>, String> {
    let (&marker, rest) = data
        .split_first()
        .ok_or_else(|| "huffman: empty buffer".to_string())?;

    match marker {
        FORMAT_RAW => Ok(rest.to_vec()),
        FORMAT_SINGLE => {
            if rest.len() < 5 {
                return Err("huffman: truncated single-symbol header".to_string());
            }
            let symbol = rest[0];
            let count = u32::from_le_bytes([rest[1], rest[2], rest[3], rest[4]]) as usize;
            Ok(vec![symbol; count])
        }
        FORMAT_HUFFMAN => decode_huffman(rest),
        other => Err(format!("huffman: unknown format marker {other}")),
    }
}

/// Decode the body of a [`FORMAT_HUFFMAN`] block (marker already consumed).
fn decode_huffman(body: &[u8]) -> Result<Vec<u8>, String> {
    if body.len() < 5 {
        return Err("huffman: truncated header".to_string());
    }
    let original_len = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let table_count = body[4] as usize + 1;

    let table_bytes = table_count * 2;
    let header_end = 5 + table_bytes;
    if body.len() < header_end {
        return Err("huffman: truncated symbol table".to_string());
    }

    // Rebuild per-symbol code lengths from the sparse table.
    let mut lengths = [0u8; ALPHABET];
    for i in 0..table_count {
        let symbol = body[5 + i * 2] as usize;
        let length = body[5 + i * 2 + 1];
        if length == 0 || length > 32 {
            return Err(format!("huffman: invalid code length {length}"));
        }
        lengths[symbol] = length;
    }

    let codes = canonical_codes(&lengths);

    // Build a (length, code) -> symbol lookup keyed by code length.
    let max_len = lengths.iter().copied().max().unwrap_or(0) as usize;
    let mut by_len: Vec<Vec<(u32, u8)>> = vec![Vec::new(); max_len + 1];
    for symbol in 0..ALPHABET {
        let len = lengths[symbol] as usize;
        if len > 0 {
            by_len[len].push((codes[symbol], symbol as u8));
        }
    }

    // Walk the bitstream MSB-first, accumulating bits until a code matches.
    let mut reader = BitReader::new(&body[header_end..]);
    let mut output = Vec::with_capacity(original_len);
    let mut acc: u32 = 0;
    let mut acc_len: usize = 0;

    while output.len() < original_len {
        let bit = reader
            .read_bit()
            .ok_or_else(|| "huffman: bitstream underrun".to_string())?;
        acc = (acc << 1) | bit as u32;
        acc_len += 1;
        if acc_len > max_len {
            return Err("huffman: code length overflow".to_string());
        }
        if let Some(&(_, symbol)) = by_len[acc_len].iter().find(|&&(code, _)| code == acc) {
            output.push(symbol);
            acc = 0;
            acc_len = 0;
        }
    }

    Ok(output)
}

/// Build canonical Huffman code lengths for every byte that occurs in `freq`.
///
/// Symbols with zero frequency receive length 0 (absent). The tree is built with
/// a deterministic tie-break so that identical inputs always yield identical
/// lengths regardless of platform or hash ordering.
fn build_code_lengths(freq: &[u64; ALPHABET]) -> [u8; ALPHABET] {
    // Each tree node carries its frequency, an insertion sequence number used as
    // a stable tie-breaker, and either a leaf symbol or two child indices.
    struct Node {
        freq: u64,
        seq: u64,
        symbol: Option<u8>,
        left: usize,
        right: usize,
    }

    let mut nodes: Vec<Node> = Vec::new();
    let mut heap: Vec<usize> = Vec::new();
    let mut seq: u64 = 0;

    for (symbol, &f) in freq.iter().enumerate() {
        if f > 0 {
            let idx = nodes.len();
            nodes.push(Node {
                freq: f,
                seq,
                symbol: Some(symbol as u8),
                left: usize::MAX,
                right: usize::MAX,
            });
            heap.push(idx);
            seq += 1;
        }
    }

    // Deterministic ordering: smaller frequency first, then smaller sequence.
    let order = |nodes: &[Node], a: usize, b: usize| -> std::cmp::Ordering {
        nodes[a]
            .freq
            .cmp(&nodes[b].freq)
            .then(nodes[a].seq.cmp(&nodes[b].seq))
    };

    // Repeatedly merge the two lowest-weight nodes until a single root remains.
    while heap.len() > 1 {
        heap.sort_by(|&a, &b| order(&nodes, a, b));
        let left = heap.remove(0);
        let right = heap.remove(0);
        let idx = nodes.len();
        let combined_freq = nodes[left].freq + nodes[right].freq;
        nodes.push(Node {
            freq: combined_freq,
            seq,
            symbol: None,
            left,
            right,
        });
        seq += 1;
        heap.push(idx);
    }

    // Walk the tree to record the depth (= code length) of each leaf.
    let mut lengths = [0u8; ALPHABET];
    if let Some(&root) = heap.first() {
        // Iterative depth-first traversal to avoid recursion limits.
        let mut stack = vec![(root, 0u8)];
        while let Some((idx, depth)) = stack.pop() {
            let node = &nodes[idx];
            match node.symbol {
                Some(symbol) => {
                    // A one-node tree never reaches here (handled as the
                    // single-symbol case), so depth >= 1 for all leaves.
                    lengths[symbol as usize] = depth.max(1);
                }
                None => {
                    stack.push((node.left, depth + 1));
                    stack.push((node.right, depth + 1));
                }
            }
        }
    }

    lengths
}

/// Assign canonical Huffman codes given per-symbol code `lengths`.
///
/// Symbols are ordered by `(length, symbol)`; codes are assigned in increasing
/// numerical order, left-shifting by the length delta between groups. This is
/// the canonical scheme reproducible from lengths alone, so both encoder and
/// decoder derive an identical table.
fn canonical_codes(lengths: &[u8; ALPHABET]) -> [u32; ALPHABET] {
    // Count how many symbols share each length.
    let max_len = lengths.iter().copied().max().unwrap_or(0) as usize;
    let mut bl_count = vec![0u32; max_len + 1];
    for &len in lengths.iter() {
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }

    // First code value for each length (canonical recurrence).
    let mut next_code = vec![0u32; max_len + 2];
    let mut code = 0u32;
    for bits in 1..=max_len {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign in ascending symbol order within each length group, which the
    // ascending symbol loop satisfies automatically.
    let mut codes = [0u32; ALPHABET];
    for symbol in 0..ALPHABET {
        let len = lengths[symbol] as usize;
        if len > 0 {
            codes[symbol] = next_code[len];
            next_code[len] += 1;
        }
    }

    codes
}

/// Minimal MSB-first bit accumulator used to build the packed payload.
struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    filled: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            filled: 0,
        }
    }

    /// Append the low `len` bits of `code`, most-significant bit first.
    fn write_code(&mut self, code: u32, len: u8) {
        for i in (0..len).rev() {
            let bit = ((code >> i) & 1) as u8;
            self.current = (self.current << 1) | bit;
            self.filled += 1;
            if self.filled == 8 {
                self.bytes.push(self.current);
                self.current = 0;
                self.filled = 0;
            }
        }
    }

    /// Flush any partial byte (zero-padded) and return the packed bytes.
    fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            self.current <<= 8 - self.filled;
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

/// Minimal MSB-first bit reader over a borrowed byte slice.
struct BitReader<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read the next bit, or `None` once the slice is exhausted.
    fn read_bit(&mut self) -> Option<u8> {
        if self.byte_pos >= self.bytes.len() {
            return None;
        }
        let byte = self.bytes[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Some(bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic linear-congruential byte generator (NOT the `rand` crate),
    /// so the round-trip corpus is reproducible across runs and platforms.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            // Numerical Recipes LCG constants.
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.state >> 32) as u32
        }

        /// Uniform byte in `0..=255`.
        fn next_byte(&mut self) -> u8 {
            (self.next_u32() & 0xFF) as u8
        }

        /// Byte drawn from a skewed distribution: small values dominate, so the
        /// resulting stream is genuinely compressible.
        fn next_skewed_byte(&mut self) -> u8 {
            let r = self.next_u32() % 100;
            if r < 70 {
                0
            } else if r < 85 {
                1
            } else if r < 93 {
                2
            } else if r < 97 {
                3
            } else {
                (self.next_u32() & 0xFF) as u8
            }
        }
    }

    fn roundtrip(data: &[u8]) {
        let encoded = encode(data);
        let decoded = decode(&encoded).expect("decode must succeed");
        assert_eq!(decoded, data, "round-trip must be lossless");
    }

    #[test]
    fn roundtrip_empty() {
        roundtrip(&[]);
    }

    #[test]
    fn roundtrip_single_byte() {
        roundtrip(&[42]);
    }

    #[test]
    fn roundtrip_single_symbol_repeated() {
        roundtrip(&[7u8; 1000]);
        roundtrip(&[0u8; 1]);
        roundtrip(&[255u8; 4096]);
    }

    #[test]
    fn roundtrip_two_symbols() {
        let data: Vec<u8> = (0..500).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        roundtrip(&data);
    }

    #[test]
    fn roundtrip_all_byte_values() {
        let data: Vec<u8> = (0..=255u16).map(|v| v as u8).collect();
        roundtrip(&data);
    }

    #[test]
    fn roundtrip_deterministic_sizes() {
        let mut lcg = Lcg::new(0x1234_5678_9abc_def0);
        for &size in &[0usize, 1, 2, 3, 5, 16, 17, 63, 255, 256, 1000, 4097] {
            let data: Vec<u8> = (0..size).map(|_| lcg.next_byte()).collect();
            roundtrip(&data);
        }
    }

    #[test]
    fn roundtrip_skewed_distribution() {
        let mut lcg = Lcg::new(0xdead_beef_cafe_babe);
        for &size in &[64usize, 512, 4096, 20000] {
            let data: Vec<u8> = (0..size).map(|_| lcg.next_skewed_byte()).collect();
            roundtrip(&data);
        }
    }

    #[test]
    fn compresses_skewed_distribution() {
        // A heavily skewed stream must shrink (ratio strictly below 1.0).
        let mut lcg = Lcg::new(0x0badc0de_0badc0de);
        let data: Vec<u8> = (0..20000).map(|_| lcg.next_skewed_byte()).collect();
        let encoded = encode(&data);
        let ratio = encoded.len() as f64 / data.len() as f64;
        assert!(
            ratio < 1.0,
            "skewed data should compress: ratio = {ratio} ({} -> {})",
            data.len(),
            encoded.len()
        );
        // And of course it must still round-trip.
        assert_eq!(decode(&encoded).unwrap(), data);
    }

    #[test]
    fn highly_repetitive_compresses_hard() {
        // Long single-symbol run uses the explicit single-symbol form: 6 bytes.
        let data = vec![123u8; 100_000];
        let encoded = encode(&data);
        assert!(
            encoded.len() < 16,
            "single-symbol run should be tiny, got {}",
            encoded.len()
        );
        assert_eq!(decode(&encoded).unwrap(), data);
    }

    #[test]
    fn no_inflation_on_random_data() {
        // High-entropy data must never inflate beyond input + the 1-byte marker.
        let mut lcg = Lcg::new(0xfeed_face_dead_2026);
        for &size in &[16usize, 256, 4096, 65536] {
            let data: Vec<u8> = (0..size).map(|_| lcg.next_byte()).collect();
            let encoded = encode(&data);
            assert!(
                encoded.len() <= data.len() + 1,
                "random data inflated: {} -> {} (size {size})",
                data.len(),
                encoded.len()
            );
            assert_eq!(decode(&encoded).unwrap(), data);
        }
    }

    #[test]
    fn rejects_malformed_buffer() {
        // Empty buffer (no marker) is malformed.
        assert!(decode(&[]).is_err());
        // Unknown marker.
        assert!(decode(&[200, 1, 2, 3]).is_err());
        // Truncated single-symbol header.
        assert!(decode(&[FORMAT_SINGLE, 5]).is_err());
        // Truncated Huffman header.
        assert!(decode(&[FORMAT_HUFFMAN, 1, 2]).is_err());
    }
}
