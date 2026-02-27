/**
 * IEEE 754 half-precision (float16) decoding.
 *
 * The retrieval embeddings are stored as raw float16 binary to halve
 * download size (~60 MB vs ~120 MB for float32). We decode to Float32Array
 * on the client for all subsequent math.
 */

/**
 * Decode a single IEEE 754 binary16 value to a float32 number.
 *
 * Layout (16 bits): [sign(1)][exponent(5)][mantissa(10)]
 *   - Bias = 15, so stored_exp - 15 = true exponent
 *   - Subnormals (exp=0, mantissa!=0): value = (-1)^s * 2^-14 * (0.mantissa)
 *   - Inf/NaN (exp=31): mantissa==0 => Inf, else NaN
 */
export function float16ToFloat32(h: number): number {
  const sign = (h >>> 15) & 1;
  const exp = (h >>> 10) & 0x1f;
  const mantissa = h & 0x3ff;

  const s = sign ? -1 : 1;

  if (exp === 0) {
    // Zero or subnormal
    if (mantissa === 0) return s * 0; // signed zero
    // Subnormal: (-1)^s * 2^(-14) * (mantissa / 1024)
    return s * Math.pow(2, -14) * (mantissa / 1024);
  }

  if (exp === 31) {
    // Infinity or NaN
    return mantissa === 0 ? s * Infinity : NaN;
  }

  // Normal: (-1)^s * 2^(exp-15) * (1 + mantissa/1024)
  return s * Math.pow(2, exp - 15) * (1 + mantissa / 1024);
}

/**
 * Decode a raw ArrayBuffer of float16 values into a Float32Array.
 *
 * @param buffer - Raw bytes (2 bytes per value, little-endian as stored by numpy)
 * @param length - Number of float16 values to decode
 * @returns Float32Array of decoded values
 */
export function decodeFloat16Array(
  buffer: ArrayBuffer,
  length: number,
): Float32Array {
  const u16 = new Uint16Array(buffer, 0, length);
  const out = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    out[i] = float16ToFloat32(u16[i]);
  }

  return out;
}
