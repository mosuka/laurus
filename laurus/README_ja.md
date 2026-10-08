# laurus

[![Crates.io](https://img.shields.io/crates/v/laurus.svg)](https://crates.io/crates/laurus)
[![Documentation](https://docs.rs/laurus/badge.svg)](https://docs.rs/laurus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Laurus](https://github.com/mosuka/laurus) プロジェクトのコア検索エンジンライブラリです。Lexical 検索（転置インデックスによるキーワードマッチング）、Vector 検索（エンベディングによるセマンティック類似度検索）、ハイブリッド検索（両者の統合）を統一 API で提供します。

## 機能

- **Lexical 検索** -- BM25 スコアリングを備えた転置インデックスによる全文検索
- **Vector 検索** -- Flat、HNSW、IVF インデックスを使用した近似最近傍（ANN）検索
- **ハイブリッド検索** -- フュージョンアルゴリズム（RRF、WeightedSum）で Lexical と Vector の結果を統合
- **テキスト解析** -- プラガブルな解析パイプライン: トークナイザー、フィルター、ステマー、同義語（[Lindera](https://github.com/lindera/lindera) による CJK サポートを含む）
- **エンベディング** -- Candle（ローカル BERT/CLIP）、OpenAI API、カスタムエンベダーの組み込みサポート
- **プラガブルストレージ** -- インメモリ、ファイルベース、メモリマップドバックエンド
- **ファセットとハイライト** -- ファセットナビゲーションと検索結果ハイライト
- **スペル修正** -- スペルミスのあるクエリ用語の修正候補を提案
- **Write-Ahead Log** -- WAL によるデータ耐久性と再起動時の自動リカバリ

## インストール

```toml
# Lexical 検索のみ（エンベディングなし）
[dependencies]
laurus = "0.13"

# ローカル BERT エンベディング付き
[dependencies]
laurus = { version = "0.13", features = ["embeddings-candle"] }

# 全エンベディングバックエンド
[dependencies]
laurus = { version = "0.13", features = ["embeddings-all"] }
```

## フィーチャーフラグ

| フィーチャー | 説明 |
| :--- | :--- |
| `embeddings-candle` | [Candle](https://github.com/huggingface/candle) によるローカル BERT エンベディング |
| `embeddings-openai` | OpenAI API によるクラウドベースのエンベディング |
| `embeddings-multimodal` | CLIP ベースのマルチモーダル（テキスト + 画像）エンベディング |
| `embeddings-all` | すべてのエンベディングバックエンドを有効化 |

## クイックスタート

```rust
use laurus::lexical::{TermQuery, TextOption};
use laurus::storage::memory::MemoryStorageConfig;
use laurus::storage::{StorageConfig, StorageFactory};
use laurus::{Document, Engine, LexicalSearchRequest, Schema, SearchRequestBuilder};

#[tokio::main]
async fn main() -> laurus::Result<()> {
    // 1. ストレージの作成
    let storage = StorageFactory::create(StorageConfig::Memory(MemoryStorageConfig::default()))?;

    // 2. スキーマの定義
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .build();

    // 3. エンジンの作成
    let engine = Engine::new(storage, schema).await?;

    // 4. ドキュメントのインデックス
    engine
        .add_document(
            "doc1",
            Document::builder()
                .add_text("title", "Introduction to Rust")
                .add_text(
                    "body",
                    "Rust is a systems programming language focused on safety and performance.",
                )
                .build(),
        )
        .await?;
    engine.commit().await?;

    // 5. 検索
    let results = engine
        .search(
            SearchRequestBuilder::new()
                .lexical_search_request(LexicalSearchRequest::new(Box::new(TermQuery::new(
                    "body", "rust",
                ))))
                .limit(5)
                .build(),
        )
        .await?;

    for hit in &results {
        println!("[{}] score={:.4}", hit.id, hit.score);
    }

    Ok(())
}
```

## 主要な型

| 型 | モジュール | 説明 |
| :--- | :--- | :--- |
| `Engine` | `engine` | Lexical 検索と Vector 検索を統括する統一検索エンジン |
| `Schema` | `engine` | フィールド定義とルーティング設定 |
| `Document` | `data` | 名前付きフィールド値のコレクション |
| `SearchRequestBuilder` | `engine` | 統一検索リクエスト（Lexical、Vector、ハイブリッド）のビルダー |
| `FusionAlgorithm` | `engine` | 結果マージ戦略（RRF または WeightedSum） |
| `LaurusError` | `error` | 各サブシステムのバリアントを持つ包括的なエラー型 |

## サンプル

使用例は [`examples/`](examples/) ディレクトリにあります:

| サンプル | 説明 | フィーチャーフラグ |
| :--- | :--- | :--- |
| [quickstart](examples/quickstart.rs) | 基本的な全文検索 | -- |
| [lexical_search](examples/lexical_search.rs) | 全クエリタイプ（Term、Phrase、Boolean、Fuzzy、Wildcard、Range、Geo、Span） | -- |
| [vector_search](examples/vector_search.rs) | エンベディングによるセマンティック類似度検索 | -- |
| [hybrid_search](examples/hybrid_search.rs) | フュージョンによる Lexical 検索と Vector 検索の統合 | -- |
| [synonym_graph_filter](examples/synonym_graph_filter.rs) | 解析パイプラインでの同義語展開 | -- |
| [search_with_candle](examples/search_with_candle.rs) | Candle によるローカル BERT エンベディング | `embeddings-candle` |
| [search_with_openai](examples/search_with_openai.rs) | OpenAI によるクラウドベースエンベディング | `embeddings-openai` |
| [multimodal_search](examples/multimodal_search.rs) | テキストから画像、画像から画像への検索 | `embeddings-multimodal` |

## ドキュメント

- [ライブラリガイド](https://mosuka.github.io/laurus/ja/laurus.html)
- [API リファレンス (docs.rs)](https://docs.rs/laurus)
- [アーキテクチャ](https://mosuka.github.io/laurus/ja/architecture.html)
- [スキーマとフィールド](https://mosuka.github.io/laurus/ja/concepts/schema_and_fields.html)
- [テキスト解析](https://mosuka.github.io/laurus/ja/concepts/analysis.html)
- [検索](https://mosuka.github.io/laurus/ja/concepts/search.html)

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](../LICENSE) ファイルを参照してください。
