//! Core vector data structure.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{LaurusError, Result};

/// A dense vector representation for similarity search.
///
/// The inner data is wrapped in [`Arc`] so that cloning a `Vector` is an O(1)
/// reference-count increment instead of an O(n) memory copy.  This is
/// critical on the search hot-path where vectors are read many times
/// (e.g. HNSW ef_search neighbours) but never mutated.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// The vector dimensions as floating point values.
    ///
    /// Wrapped in `Arc` for cheap cloning on read paths.
    pub data: Arc<Vec<f32>>,
}

impl Serialize for Vector {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.data.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Vector {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let data = Vec::<f32>::deserialize(deserializer)?;
        Ok(Vector {
            data: Arc::new(data),
        })
    }
}

impl Vector {
    /// Create a new vector with the given dimensions.
    ///
    /// The provided `Vec<f32>` is wrapped in an `Arc` so that subsequent
    /// clones are O(1).
    ///
    /// # Arguments
    ///
    /// * `data` - The floating point values representing the vector dimensions.
    ///
    /// # Returns
    ///
    /// A new `Vector` instance.
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data: Arc::new(data),
        }
    }

    /// Get the dimensionality of this vector.
    ///
    /// # Returns
    ///
    /// The number of dimensions (length of the underlying `Vec<f32>`).
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Calculate the L2 norm (magnitude) of this vector.
    ///
    /// # Returns
    ///
    /// The L2 norm as an `f32` value.
    pub fn norm(&self) -> f32 {
        l2_norm(&self.data)
    }

    /// Normalize this vector to unit length.
    ///
    /// Uses [`Arc::make_mut`] to obtain a mutable reference, performing a
    /// copy-on-write if other clones exist.
    pub fn normalize(&mut self) {
        let norm = self.norm();
        if norm > 0.0 {
            let data: &mut Vec<f32> = Arc::make_mut(&mut self.data);
            divide_by(data, norm);
        }
    }

    /// Get a normalized copy of this vector.
    ///
    /// # Returns
    ///
    /// A new `Vector` with unit length.
    pub fn normalized(&self) -> Self {
        let mut normalized = self.clone();
        normalized.normalize();
        normalized
    }

    /// Validate that this vector has the expected dimension.
    ///
    /// # Arguments
    ///
    /// * `expected_dim` - The expected number of dimensions.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the dimension matches, or an error describing the mismatch.
    pub fn validate_dimension(&self, expected_dim: usize) -> Result<()> {
        if self.data.len() != expected_dim {
            return Err(LaurusError::InvalidOperation(format!(
                "Vector dimension mismatch: expected {}, got {}",
                expected_dim,
                self.data.len()
            )));
        }
        Ok(())
    }

    /// Check if this vector contains any NaN or infinite values.
    ///
    /// # Returns
    ///
    /// `true` if all values are finite, `false` otherwise.
    pub fn is_valid(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }

    /// Calculate the L2 norm using parallel processing for large vectors.
    ///
    /// Falls back to the sequential [`norm`](Self::norm) for vectors with
    /// 10 000 dimensions or fewer.
    ///
    /// # Returns
    ///
    /// The L2 norm as an `f32` value.
    pub fn norm_parallel(&self) -> f32 {
        #[cfg(not(target_arch = "wasm32"))]
        if self.data.len() > 10000 {
            return self.data.par_iter().map(|x| x * x).sum::<f32>().sqrt();
        }
        self.norm()
    }

    /// Normalize this vector using parallel processing for large vectors.
    ///
    /// Falls back to sequential normalization for vectors with 10 000
    /// dimensions or fewer.  Uses [`Arc::make_mut`] for copy-on-write
    /// semantics.
    pub fn normalize_parallel(&mut self) {
        let norm = self.norm_parallel();
        if norm > 0.0 {
            let data = Arc::make_mut(&mut self.data);
            #[cfg(not(target_arch = "wasm32"))]
            if data.len() > 10000 {
                data.par_iter_mut().for_each(|value| *value /= norm);
                return;
            }
            for value in data.iter_mut() {
                *value /= norm;
            }
        }
    }

    /// Normalize multiple vectors in parallel.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Mutable slice of vectors to normalize in-place.
    pub fn normalize_batch_parallel(vectors: &mut [Vector]) {
        #[cfg(not(target_arch = "wasm32"))]
        if vectors.len() > 10 {
            vectors
                .par_iter_mut()
                .for_each(|vector| vector.normalize_parallel());
            return;
        }
        for vector in vectors {
            vector.normalize();
        }
    }
}

/// Dense vector with weight, used for internal storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredVector {
    /// The floating point values representing the vector dimensions.
    pub data: Vec<f32>,
    /// Weight/boost factor for this vector.
    pub weight: f32,
}

impl StoredVector {
    /// Create a new stored vector with default weight of 1.0.
    ///
    /// # Arguments
    ///
    /// * `data` - The floating point values representing the vector dimensions.
    ///
    /// # Returns
    ///
    /// A new `StoredVector` with `weight` set to `1.0`.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data, weight: 1.0 }
    }

    /// Set the weight for this stored vector.
    ///
    /// # Arguments
    ///
    /// * `weight` - The weight/boost factor.
    ///
    /// # Returns
    ///
    /// The modified `StoredVector` (builder pattern).
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Get the dimensionality of this stored vector.
    ///
    /// # Returns
    ///
    /// The number of dimensions.
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Convert to a [`Vector`], wrapping the data in `Arc`.
    ///
    /// # Returns
    ///
    /// A new `Vector` instance.
    pub fn to_vector(&self) -> Vector {
        Vector::new(self.data.clone())
    }
}

impl From<Vector> for StoredVector {
    fn from(vector: Vector) -> Self {
        // Unwrap the Arc if we are the sole owner; otherwise clone the inner Vec.
        let data = Arc::try_unwrap(vector.data).unwrap_or_else(|arc| (*arc).clone());
        Self { data, weight: 1.0 }
    }
}

/// L2 norm, accumulated in `f32` in element order.
fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn divide_by(values: &mut [f32], norm: f32) {
    for value in values {
        *value /= norm;
    }
}

/// L2-normalize `values` in place with exactly the arithmetic of
/// [`Vector::normalize`], so a vector normalized on ingest and one
/// re-normalized later (e.g. a pooled token vector, #1346) agree bit for
/// bit. A zero vector is left as is.
pub(crate) fn normalize_slice(values: &mut [f32]) {
    let norm = l2_norm(values);
    if norm > 0.0 {
        divide_by(values, norm);
    }
}
