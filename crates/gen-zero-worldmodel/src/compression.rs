//! Versioned, dimension-bound zstd storage for finite f32 state streams.
use crate::error::WorldModelError;
use std::io::{Read, Write};

const MAGIC: &[u8; 4] = b"GZWM";
const HEADER_LEN: usize = 4 + 1 + 1 + 8 + 8;
const VERSION: u8 = 1;

fn error(message: impl Into<String>) -> WorldModelError {
    WorldModelError::Compression(message.into())
}

pub(crate) fn encode(
    kind: u8,
    dimensions: usize,
    count: usize,
    values: impl Iterator<Item = f32>,
    level: i32,
) -> Result<Vec<u8>, WorldModelError> {
    let dimensions = u64::try_from(dimensions).map_err(|_| error("dimension count exceeds u64"))?;
    let count = u64::try_from(count).map_err(|_| error("state count exceeds u64"))?;
    let mut result = Vec::new();
    result.extend_from_slice(MAGIC);
    result.extend_from_slice(&[VERSION, kind]);
    result.extend_from_slice(&dimensions.to_le_bytes());
    result.extend_from_slice(&count.to_le_bytes());
    let mut encoder =
        zstd::stream::write::Encoder::new(result, level).map_err(|e| error(e.to_string()))?;
    encoder
        .include_checksum(true)
        .map_err(|e| error(e.to_string()))?;
    for value in values {
        if !value.is_finite() {
            return Err(error("non-finite state value"));
        }
        encoder
            .write_all(&value.to_bits().to_le_bytes())
            .map_err(|e| error(e.to_string()))?;
    }
    encoder.finish().map_err(|e| error(e.to_string()))
}

pub(crate) fn decode(
    kind: u8,
    dimensions: usize,
    fields: usize,
    compressed: &[u8],
) -> Result<Vec<f32>, WorldModelError> {
    if compressed.len() < HEADER_LEN
        || &compressed[..4] != MAGIC
        || compressed[4] != VERSION
        || compressed[5] != kind
    {
        return Err(error("invalid state stream header"));
    }
    let stored_dimensions = u64::from_le_bytes(compressed[6..14].try_into().unwrap());
    if stored_dimensions
        != u64::try_from(dimensions).map_err(|_| error("dimension count exceeds u64"))?
    {
        return Err(error("state dimension mismatch"));
    }
    let count = usize::try_from(u64::from_le_bytes(compressed[14..22].try_into().unwrap()))
        .map_err(|_| error("state count exceeds usize"))?;
    let value_count = count
        .checked_mul(fields)
        .ok_or_else(|| error("state length overflow"))?;
    let byte_count = value_count
        .checked_mul(4)
        .ok_or_else(|| error("byte length overflow"))?;
    let limit = u64::try_from(byte_count)
        .map_err(|_| error("byte length exceeds u64"))?
        .checked_add(1)
        .ok_or_else(|| error("byte length overflow"))?;
    let decoder = zstd::stream::read::Decoder::new(&compressed[HEADER_LEN..])
        .map_err(|e| error(e.to_string()))?;
    let mut reader = decoder.take(limit);
    let mut raw = Vec::new();
    reader
        .read_to_end(&mut raw)
        .map_err(|e| error(e.to_string()))?;
    if raw.len() != byte_count {
        return Err(error(format!(
            "decoded length mismatch: expected {byte_count}, got {}",
            raw.len()
        )));
    }
    let mut result = Vec::with_capacity(value_count);
    for chunk in raw.chunks_exact(4) {
        let value = f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap()));
        if !value.is_finite() {
            return Err(error("non-finite decoded state value"));
        }
        result.push(value);
    }
    Ok(result)
}
