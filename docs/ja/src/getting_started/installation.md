# インストール

## プロジェクトへの Laurus の追加

`Cargo.toml` に `laurus` と `tokio`（非同期ランタイム）を追加します:

```toml
[dependencies]
laurus = "0.13"
tokio = { version = "1", features = ["full"] }
```

## Feature Flags

Laurus はデフォルトで最小限の機能セットで提供されます。必要に応じて追加の機能を有効にしてください:

| Feature | 説明 | ユースケース |
| :--- | :--- | :--- |
| `native` *(デフォルト)* | コアライブラリ（Lexical 検索、ストレージ、アナライザ — エンベディングなし） | キーワード検索のみ |
| `embeddings-candle` | Hugging Face Candle によるローカル BERT エンベディング | 外部 API 不要の Vector 検索 |
| `embeddings-openai` | OpenAI API エンベディング（text-embedding-3-small 等） | クラウドベースの Vector 検索 |
| `embeddings-multimodal` | Candle による CLIP エンベディング（テキスト + 画像） | マルチモーダル（テキスト→画像）検索 |
| `embeddings-all` | 上記すべてのエンベディング機能 | 全エンベディング対応 |

### 例

**Lexical 検索のみ**（エンベディング不要）:

```toml
[dependencies]
laurus = "0.13"
```

**ローカルモデルによる Vector 検索**（API キー不要）:

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-candle"] }
```

**OpenAI による Vector 検索**:

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-openai"] }
```

**すべての機能**:

```toml
[dependencies]
laurus = { version = "0.13", features = ["embeddings-all"] }
```

## インストールの確認

Laurus が正しくコンパイルされることを確認するために、最小限のプログラムを作成します:

```rust
use laurus::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Laurus version: {}", laurus::VERSION);
    Ok(())
}
```

```bash
cargo run
```

バージョンが表示されれば、[クイックスタート](quickstart.md)に進む準備が整っています。
