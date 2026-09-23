//! PHP bindings for the Laurus unified search library.
//!
//! Build with:
//!
//! ```bash
//! cd laurus-php
//! cargo build --release
//! ```
//!
//! Then copy the resulting `liblaurus_php.so` to the PHP extensions directory
//! and add `extension=laurus_php.so` to `php.ini`.

#![deny(clippy::all)]

pub mod analysis;
pub mod convert;
pub mod errors;
pub mod index;
pub mod query;
pub mod schema;
pub mod search;

use ext_php_rs::prelude::*;

/// Register the `laurus` PHP extension module and all classes.
#[php_module]
pub fn get_module(module: ModuleBuilder) -> ModuleBuilder {
    module
        // Core
        .class::<index::PhpIndex>()
        .class::<index::PhpWalSyncPolicy>()
        .class::<index::PhpCommitPolicy>()
        .class::<schema::PhpSchema>()
        .function(index::peek_commit_generation_function_entry())
        // Search result & request, fusion algorithms
        .class::<search::PhpRRF>()
        .class::<search::PhpWeightedSum>()
        .class::<search::PhpSearchResult>()
        .class::<search::PhpSearchRequest>()
        // Query types
        .class::<query::PhpTermQuery>()
        .class::<query::PhpPhraseQuery>()
        .class::<query::PhpFuzzyQuery>()
        .class::<query::PhpWildcardQuery>()
        .class::<query::PhpNumericRangeQuery>()
        .class::<query::PhpDateTimeRangeQuery>()
        .class::<query::PhpGeoDistanceQuery>()
        .class::<query::PhpGeoBoundingBoxQuery>()
        .class::<query::PhpGeo3dDistanceQuery>()
        .class::<query::PhpGeo3dBoundingBoxQuery>()
        .class::<query::PhpGeo3dNearestQuery>()
        .class::<query::PhpBooleanQuery>()
        .class::<query::PhpSpanQuery>()
        .class::<query::PhpVectorQuery>()
        .class::<query::PhpVectorTextQuery>()
        // Analysis pipeline
        .class::<analysis::PhpToken>()
        .class::<analysis::PhpSynonymDictionary>()
        .class::<analysis::PhpWhitespaceTokenizer>()
        .class::<analysis::PhpSynonymGraphFilter>()
}
