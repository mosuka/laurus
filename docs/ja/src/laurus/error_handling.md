# エラーハンドリング

Laurusはすべての操作に統一的なエラー型を使用します。エラーシステムを理解することで、障害を適切に処理する堅牢なアプリケーションを作成できます。

## LaurusError

Laurusのすべての操作は `Result<T>` を返します。これは `std::result::Result<T, LaurusError>` のエイリアスです。

`LaurusError` は、各カテゴリの障害に対応するバリアントを持つenumです。

| バリアント | 説明 | 一般的な原因 |
| :--- | :--- | :--- |
| `Io` | I/Oエラー | ファイルが見つからない、権限拒否、ディスク容量不足 |
| `Index` | インデックス操作エラー | インデックスの破損、セグメント読み取り失敗 |
| `Schema` | スキーマ関連のエラー | 不明なフィールド名、型の不一致 |
| `Analysis` | テキスト解析エラー | トークナイザーの失敗、無効なフィルタ設定 |
| `Query` | クエリの解析/実行エラー | 不正なQuery DSL、クエリ内の不明なフィールド |
| `Storage` | ストレージバックエンドエラー | ストレージのオープン失敗、書き込み失敗 |
| `Field` | フィールド定義エラー | 無効なフィールドオプション、重複するフィールド名 |
| `InvalidArgument` | 呼び出し側が渡した引数が不正 | DSL クエリ内の不明なフィールド、ドキュメントの値の型の誤り、範囲外の値 |
| `BenchmarkFailed` | ベンチマークエラー | ベンチマーク実行失敗 |
| `ThreadJoinError` | スレッド join エラー | ワーカースレッドでのパニック |
| `Json` | JSONシリアライズエラー | 不正なドキュメントJSON |
| `Anyhow` | anyhow ラップエラー | `anyhow` 経由のサードパーティクレートエラー |
| `InvalidOperation` | 無効な操作 | コミット前の検索、二重クローズ |
| `ResourceExhausted` | リソース制限超過 | メモリ不足、オープンファイル数超過 |
| `SerializationError` | バイナリシリアライズエラー | ディスク上のデータ破損 |
| `OperationCancelled` | 操作がキャンセルされた | タイムアウト、ユーザーによるキャンセル |
| `NotImplemented` | 機能が利用不可 | 未実装の操作 |
| `Other` | 汎用エラー | タイムアウト、無効な設定、見つからない |

## 基本的なエラーハンドリング

### `?` 演算子の使用

最もシンプルなアプローチ -- エラーを呼び出し元に伝播します。

```rust
use laurus::{Engine, Result};

async fn index_documents(engine: &Engine) -> Result<()> {
    let doc = laurus::Document::builder()
        .add_text("title", "Rust Programming")
        .build();

    engine.put_document("doc1", doc).await?;
    engine.commit().await?;
    Ok(())
}
```

### エラーバリアントのマッチング

エラータイプごとに異なる動作が必要な場合:

```rust
use laurus::{Engine, LaurusError};

async fn safe_search(engine: &Engine, query: &str) {
    match engine.search(/* request */).await {
        Ok(results) => {
            for result in results {
                println!("{}: {}", result.id, result.score);
            }
        }
        Err(LaurusError::Query(msg)) => {
            eprintln!("Invalid query syntax: {}", msg);
        }
        Err(LaurusError::Io(e)) => {
            eprintln!("Storage I/O error: {}", e);
        }
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
        }
    }
}
```

### `downcast` によるエラータイプの確認

`LaurusError` は `std::error::Error` を実装しているため、標準的なエラーハンドリングパターンを使用できます。

```rust
use laurus::LaurusError;

fn is_retriable(error: &LaurusError) -> bool {
    matches!(error, LaurusError::Io(_) | LaurusError::ResourceExhausted(_))
}
```

## よくあるエラーシナリオ

### スキーマの不一致

スキーマに一致しないフィールドを持つドキュメントの追加:

```rust
// スキーマには "title"（Text）と "year"（Integer）がある
let doc = Document::builder()
    .add_text("title", "Hello")
    .add_text("unknown_field", "this field is not in schema")
    .build();

// スキーマにないフィールドはインデキシング時に黙って無視されます。
// エラーは発生しません -- スキーマで定義されたフィールドのみが処理されます。
```

### クエリ解析エラー

無効なQuery DSL構文は `Query` エラーを返します。

```rust
use laurus::engine::query::UnifiedQueryParser;

let parser = UnifiedQueryParser::new();
match parser.parse("title:\"unclosed phrase") {
    Ok(request) => { /* ... */ }
    Err(LaurusError::Query(msg)) => {
        // msgには解析失敗の詳細が含まれます
        eprintln!("Bad query: {}", msg);
    }
    Err(e) => { /* その他のエラー */ }
}
```

### ストレージI/Oエラー

ファイルベースのストレージではI/Oエラーが発生する可能性があります。

```rust
use laurus::storage::{StorageConfig, StorageFactory};

match StorageFactory::open(StorageConfig::File {
    path: "/nonexistent/path".into(),
    loading_mode: Default::default(),
}) {
    Ok(storage) => { /* ... */ }
    Err(LaurusError::Io(e)) => {
        eprintln!("Cannot open storage: {}", e);
    }
    Err(e) => { /* その他のエラー */ }
}
```

## 便利なコンストラクタ

`LaurusError` はカスタム実装でエラーを作成するためのファクトリメソッドを提供しています。

| メソッド | 作成されるバリアント |
| :--- | :--- |
| `LaurusError::index(msg)` | `Index` バリアント |
| `LaurusError::schema(msg)` | `Schema` バリアント |
| `LaurusError::analysis(msg)` | `Analysis` バリアント |
| `LaurusError::query(msg)` | `Query` バリアント |
| `LaurusError::storage(msg)` | `Storage` バリアント |
| `LaurusError::field(msg)` | `Field` バリアント |
| `LaurusError::other(msg)` | `Other` バリアント |
| `LaurusError::cancelled(msg)` | `OperationCancelled` バリアント |
| `LaurusError::invalid_argument(msg)` | `InvalidArgument` バリアント |
| `LaurusError::invalid_config(msg)` | "Invalid configuration" プレフィックス付き `Other` |
| `LaurusError::not_found(msg)` | "Not found" プレフィックス付き `Other` |
| `LaurusError::timeout(msg)` | "Timeout" プレフィックス付き `Other` |

これらはカスタム [Analyzer、Embedder、またはStorage](extensibility.md) トレイトを実装する際に有用です。

```rust
use laurus::{LaurusError, Result};

fn validate_dimension(dim: usize) -> Result<()> {
    if dim == 0 {
        return Err(LaurusError::invalid_argument("dimension must be > 0"));
    }
    Ok(())
}
```

## 自動変換

`LaurusError` は一般的なエラー型に対して `From` を実装しているため、`?` で自動変換されます。

| ソース型 | ターゲットバリアント |
| :--- | :--- |
| `std::io::Error` | `LaurusError::Io` |
| `serde_json::Error` | `LaurusError::Json` |
| `anyhow::Error` | `LaurusError::Anyhow` |

## 次のステップ

- [拡張性](extensibility.md) -- 適切なエラーハンドリングでカスタムトレイトを実装
- [APIリファレンス](api_reference.md) -- 完全なメソッドシグネチャと戻り値の型
