//! Core data structures for vector search.
//!
//! This module contains fundamental data structures and types used throughout
//! the vector search implementation, including vector representations,
//! distance metrics, and quantization methods.

pub mod distance;
pub mod distance_pq_fastscan;
pub mod distance_quantized;
pub mod field;
pub(crate) mod late_interaction;
pub mod quantization;
pub mod rerank;
pub mod sq_int8_avx2;
pub mod sq_int8_neon;
pub mod vector;
