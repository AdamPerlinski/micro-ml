use crate::error::MlError;

/// Validate a flat row-major matrix, return n_samples
pub fn validate_matrix(data: &[f64], n_features: usize) -> Result<usize, MlError> {
    if n_features == 0 {
        return Err(MlError::new("n_features must be > 0"));
    }
    if data.is_empty() {
        return Err(MlError::new("data must not be empty"));
    }
    if data.len() % n_features != 0 {
        return Err(MlError::new("data length must be divisible by n_features"));
    }
    Ok(data.len() / n_features)
}

/// Get element at (row, col) from flat row-major matrix
#[inline(always)]
pub fn mat_get(data: &[f64], n_features: usize, row: usize, col: usize) -> f64 {
    data[row * n_features + col]
}

/// Squared Euclidean distance between two rows (avoids sqrt)
#[inline]
pub fn euclidean_dist_sq(data: &[f64], n_features: usize, a: usize, b: usize) -> f64 {
    let mut sum = 0.0;
    for j in 0..n_features {
        let d = mat_get(data, n_features, a, j) - mat_get(data, n_features, b, j);
        sum += d * d;
    }
    sum
}

/// Euclidean distance between a row and a point
pub fn dist_to_point(data: &[f64], n_features: usize, row: usize, point: &[f64]) -> f64 {
    let mut sum = 0.0;
    for j in 0..n_features {
        let d = mat_get(data, n_features, row, j) - point[j];
        sum += d * d;
    }
    sum.sqrt()
}

/// Simple xorshift64 PRNG (no dependencies)
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    /// Seed from data (deterministic)
    pub fn from_data(data: &[f64]) -> Self {
        let mut h: u64 = 0xcbf29ce484222325;
        for &v in data.iter().take(20) {
            h ^= v.to_bits();
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= data.len() as u64;
        Self::new(h)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Random usize in [0, max)
    pub fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() % max as u64) as usize
    }

    /// Random f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_matrix() {
        assert_eq!(validate_matrix(&[1.0, 2.0, 3.0, 4.0], 2).unwrap(), 2);
        assert!(validate_matrix(&[1.0, 2.0, 3.0], 2).is_err());
        assert!(validate_matrix(&[], 2).is_err());
        assert!(validate_matrix(&[1.0], 0).is_err());
    }

    #[test]
    fn test_euclidean_dist_sq() {
        let data = vec![0.0, 0.0, 3.0, 4.0];
        assert!((euclidean_dist_sq(&data, 2, 0, 1) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..10 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }
}
