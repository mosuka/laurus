# クイックスタート

このチュートリアルでは、5 つのステップで完全な検索エンジンを構築する方法を説明します。最後には、ドキュメントをインデックスしてキーワード検索ができるようになります。

## ステップ 1 — Storage の作成

Storage は Laurus がインデックスデータを保存する場所を決定します。開発やテストには `MemoryStorage` を使用します:

```rust
use std::sync::Arc;
use laurus::storage::memory::MemoryStorage;
use laurus::Storage;

let storage: Arc<dyn Storage> = Arc::new(
    MemoryStorage::new(Default::default())
);
```

> **ヒント:** 本番環境では `FileStorage`（オプションで `use_mmap` によるメモリマップド I/O）の使用を検討してください。詳細は[ストレージ](../concepts/storage.md)を参照してください。

## ステップ 2 — Schema の定義

`Schema` はドキュメント内のフィールドと、各フィールドのインデックス方法を宣言します:

```rust
use laurus::Schema;
use laurus::lexical::TextOption;

let schema = Schema::builder()
    .add_text_field("title", TextOption::default())
    .add_text_field("body", TextOption::default())
    .add_default_field("body")  // used when no field is specified in a query
    .build();
```

各フィールドには型があります。主な型は以下の通りです:

| メソッド | フィールド型 | 値の例 |
| :--- | :--- | :--- |
| `add_text_field` | Text（全文検索可能） | `"Hello world"` |
| `add_integer_field` | 64 ビット整数 | `42` |
| `add_float_field` | 64 ビット浮動小数点数 | `3.14` |
| `add_boolean_field` | ブール値 | `true` / `false` |
| `add_datetime_field` | UTC 日時 | `2024-01-15T10:30:00Z` |
| `add_hnsw_field` | Vector（HNSW インデックス） | `[0.1, 0.2, ...]` |
| `add_flat_field` | Vector（Flat インデックス） | `[0.1, 0.2, ...]` |

> 全一覧は[スキーマとフィールド](../concepts/schema_and_fields.md)を参照してください。

## ステップ 3 — Engine の構築

`Engine` は Storage、Schema、ランタイムコンポーネントを統合します:

```rust
use laurus::Engine;

let engine = Engine::builder(storage, schema)
    .build()
    .await?;
```

テキストフィールドのみを使用する場合、デフォルトの `StandardAnalyzer` が自動的に適用されます。解析のカスタマイズや Vector エンベディングの追加については、[アーキテクチャ](../architecture.md)を参照してください。

## ステップ 4 — ドキュメントのインデックス

`DocumentBuilder` でドキュメントを作成し、Engine に追加します:

```rust
use laurus::Document;

// Each document needs a unique external ID (string)
let doc = Document::builder()
    .add_text("title", "Introduction to Rust")
    .add_text("body", "Rust is a systems programming language focused on safety and performance.")
    .build();
engine.add_document("doc-1", doc).await?;

let doc = Document::builder()
    .add_text("title", "Python for Data Science")
    .add_text("body", "Python is widely used in machine learning and data analysis.")
    .build();
engine.add_document("doc-2", doc).await?;

let doc = Document::builder()
    .add_text("title", "Web Development with JavaScript")
    .add_text("body", "JavaScript powers interactive web applications and server-side code with Node.js.")
    .build();
engine.add_document("doc-3", doc).await?;

// Commit to make documents searchable
engine.commit().await?;
```

> **重要:** ドキュメントは `commit()` が呼ばれるまで検索可能になりません。

## ステップ 5 — 検索

`SearchRequestBuilder` とクエリを使ってインデックスを検索します:

```rust
use laurus::SearchRequestBuilder;
use laurus::lexical::TermQuery;
use laurus::lexical::search::searcher::LexicalSearchQuery;

// Search for "rust" in the "body" field
let request = SearchRequestBuilder::new()
    .lexical_query(
        LexicalSearchQuery::Obj(
            Box::new(TermQuery::new("body", "rust"))
        )
    )
    .limit(10)
    .build();

let results = engine.search(request).await?.results;

for result in &results {
    println!("ID: {}, Score: {:.4}", result.id, result.score);
    if let Some(doc) = &result.document {
        if let Some(title) = doc.get("title") {
            println!("  Title: {:?}", title);
        }
    }
}
```

## 完全なサンプル

以下は、コピー・ペーストしてそのまま実行できる完全なプログラムです:

```rust
use std::sync::Arc;
use laurus::{
    Document, Engine, Result, Schema, SearchRequestBuilder,
};
use laurus::lexical::{TextOption, TermQuery};
use laurus::lexical::search::searcher::LexicalSearchQuery;
use laurus::storage::memory::MemoryStorage;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Storage
    let storage = Arc::new(MemoryStorage::new(Default::default()));

    // 2. Schema
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .add_default_field("body")
        .build();

    // 3. Engine
    let engine = Engine::builder(storage, schema).build().await?;

    // 4. Index documents
    for (id, title, body) in [
        ("doc-1", "Introduction to Rust", "Rust is a systems programming language focused on safety."),
        ("doc-2", "Python for Data Science", "Python is widely used in machine learning."),
        ("doc-3", "Web Development", "JavaScript powers interactive web applications."),
    ] {
        let doc = Document::builder()
            .add_text("title", title)
            .add_text("body", body)
            .build();
        engine.add_document(id, doc).await?;
    }
    engine.commit().await?;

    // 5. Search
    let request = SearchRequestBuilder::new()
        .lexical_query(
            LexicalSearchQuery::Obj(
                Box::new(TermQuery::new("body", "rust"))
            )
        )
        .limit(10)
        .build();

    let results = engine.search(request).await?.results;
    for r in &results {
        println!("{}: score={:.4}", r.id, r.score);
    }

    Ok(())
}
```

## 次のステップ

- Engine の内部動作を学ぶ: [アーキテクチャ](../architecture.md)
- Schema とフィールド型を理解する: [スキーマとフィールド](../concepts/schema_and_fields.md)
- Vector 検索を追加する: [Vector 検索](../concepts/search/vector_search.md)
- Lexical と Vector を統合する: [ハイブリッド検索](../concepts/search/hybrid_search.md)
