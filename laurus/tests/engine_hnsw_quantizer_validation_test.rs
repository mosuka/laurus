//! Engine-level validation for HNSW quantizer and distance combinations.

use std::sync::Arc;

use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::vector::core::distance::DistanceMetric;
use laurus::vector::core::field::HnswOption;
use laurus::vector::core::quantization::QuantizationMethod;
use laurus::{Engine, FieldOption, Schema};

async fn assert_unsupported_metrics_rejected(engine: &Engine, quantizer: QuantizationMethod) {
    for metric in [
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
        DistanceMetric::Angular,
    ] {
        let field_name = format!("embedding_{metric:?}");
        let error = engine
            .add_field(
                &field_name,
                FieldOption::Hnsw(HnswOption {
                    dimension: 4,
                    distance: metric,
                    quantizer,
                    ..HnswOption::default()
                }),
            )
            .await
            .expect_err("unsupported PQ metric must be rejected when defining the field");
        let message = error.to_string();
        assert!(message.contains(&field_name), "{message}");
        assert!(message.contains(&format!("{quantizer:?}")), "{message}");
        assert!(message.contains(&format!("{metric:?}")), "{message}");
    }
}

async fn assert_supported_metrics_accepted(
    engine: &Engine,
    quantizer: QuantizationMethod,
) -> laurus::Result<()> {
    for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine] {
        let field_name = format!("supported_{metric:?}");
        engine
            .add_field(
                &field_name,
                FieldOption::Hnsw(HnswOption {
                    dimension: 4,
                    distance: metric,
                    quantizer,
                    ..HnswOption::default()
                }),
            )
            .await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn engine_validates_product_quantization_metrics() -> laurus::Result<()> {
    let engine = Engine::new(
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
        Schema::new(),
    )
    .await?;
    let quantizer = QuantizationMethod::ProductQuantization { subvector_count: 2 };

    assert_unsupported_metrics_rejected(&engine, quantizer).await;
    assert_supported_metrics_accepted(&engine, quantizer).await?;
    Ok(())
}

#[cfg(feature = "pq-fastscan")]
#[tokio::test(flavor = "multi_thread")]
async fn engine_validates_product_quantization_fastscan_metrics() -> laurus::Result<()> {
    let engine = Engine::new(
        Arc::new(MemoryStorage::new(MemoryStorageConfig::default())),
        Schema::new(),
    )
    .await?;
    let quantizer = QuantizationMethod::ProductQuantizationFastScan { subvector_count: 2 };

    assert_unsupported_metrics_rejected(&engine, quantizer).await;
    assert_supported_metrics_accepted(&engine, quantizer).await?;
    Ok(())
}
