# 開発

## 前提条件

- [Rust](https://rustup.rs/)（stable、`wasm32-unknown-unknown` ターゲット付き）
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- [Node.js](https://nodejs.org/)（テストと npm publish 用）

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## ビルド

```bash
cd laurus-wasm

# デバッグビルド（コンパイル高速）
wasm-pack build --target web --dev

# リリースビルド（最適化）
wasm-pack build --target web --release

# バンドラーターゲット（webpack、vite 等）
wasm-pack build --target bundler --release
```

## プロジェクト構成

```text
laurus-wasm/
├── Cargo.toml          # Rust 依存関係（wasm-bindgen、laurus コア）
├── package.json        # npm パッケージメタデータ
├── src/
│   ├── lib.rs          # モジュール宣言
│   ├── index.rs        # Index クラス（CRUD + 検索）
│   ├── schema.rs       # Schema ビルダー
│   ├── search.rs       # SearchRequest / SearchResult
│   ├── query.rs        # クエリ型定義
│   ├── convert.rs      # JsValue ↔ Document 変換
│   ├── analysis.rs     # トークナイザー / フィルターラッパー
│   ├── errors.rs       # LaurusError → JsValue 変換
│   └── storage.rs      # OPFS 永続化レイヤー
└── js/
    └── opfs_bridge.js  # Origin Private File System 用 JS グルーコード
```

## アーキテクチャノート

### ストレージ戦略

laurus-wasm は二層ストレージアプローチを採用しています:

1. **MemoryStorage**（ランタイム） -- すべての読み書き操作は Laurus の
   インメモリストレージを経由します。これは `Storage` トレイトの
   `Send + Sync` 要件を満たします。

2. **OPFS**（永続化） -- `commit()` 時に MemoryStorage の全状態が
   OPFS ファイルにシリアライズされます。`Index.open()` 時に OPFS
   ファイルが MemoryStorage にロードされます。

この設計により、JS ハンドルの `Send + Sync` 非互換性を回避しつつ、
コアエンジンを変更せずに永続化を実現しています。

### Feature Flags

`laurus` コアは Feature Flags で WASM をサポートしています:

```toml
# laurus-wasm はデフォルト機能なしで laurus に依存
laurus = { workspace = true, default-features = false }
```

これにより、ネイティブ専用の依存関係（tokio/full、rayon、memmap2 等）が
除外され、`#[cfg(target_arch = "wasm32")]` フォールバックで並列処理が
逐次処理に切り替わります。

### 日本語形態素解析

ブラウザ WASM にはファイルシステムが無いため、`{ "language": "japanese", "dict": "/path/to/ipadic" }` の標準アナライザープリセットは利用できません。`laurus-wasm` は [`src/analysis.rs`](https://github.com/mosuka/laurus/blob/main/laurus-wasm/src/analysis.rs) で `JapaneseAnalyzer.fromBytes(...)` を公開しており、Lindera IPADIC 辞書アーカイブを実行時に OPFS へ取得し、Lindera が必要とする 9 つの生バイト配列を読み出してアナライザーに渡せます：

```javascript
import { JapaneseAnalyzer, Schema } from "laurus-wasm";
import { downloadDictionary, loadDictionaryFiles } from "laurus-wasm/opfs";

await downloadDictionary("./dict/lindera-ipadic.zip", "ipadic");
const f = await loadDictionaryFiles("ipadic");
const ja = JapaneseAnalyzer.fromBytes(
  f.metadata, f.dictTrie, f.dictValsIdx, f.dictVals,
  f.dictWordsIdx, f.dictWords, f.matrixMtx, f.charDef, f.unk, "normal",
);

const schema = new Schema();
schema.addAnalyzer("ja-ipadic", ja);
schema.addTextField("body", undefined, undefined, undefined, undefined, "ja-ipadic");
```

OPFS ヘルパー（`downloadDictionary` / `getDictionaryVersion` / `loadDictionaryFiles` / `hasDictionary` / `listDictionaries` / `removeDictionary`）は [`js/opfs.js`](https://github.com/mosuka/laurus/blob/main/laurus-wasm/js/opfs.js) にあり、`package.json` で `laurus-wasm/opfs` サブパスとして再公開されています。引数表は [API リファレンス → JapaneseAnalyzer](api_reference.md#japaneseanalyzer) を参照してください。

### コールバック Embedder

事前計算済みベクトルを受け取る `"precomputed"` Embedder に加えて、`laurus-wasm` は JS 側から `async embed: (text) => Promise<number[]>` を渡せる `"callback"` Embedder をサポートします。エンジンはドキュメント投入時と `searchVectorText()` クエリ時にこのコールバックを呼び出すため、WASM モジュールを再ビルドすることなく任意のブラウザ向け埋め込みライブラリ（Transformers.js、ONNX Runtime Web 等）を統合できます：

```javascript
import { pipeline } from "@xenova/transformers";

const embedder = await pipeline(
  "feature-extraction",
  "Xenova/all-MiniLM-L6-v2",
);

schema.addEmbedder("minilm", {
  type: "callback",
  embed: async (text) => {
    const output = await embedder(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  },
});

schema.addHnswField(
  "embedding", 384, "cosine",
  undefined, undefined, undefined, "minilm",
);
```

wasm-bindgen のグルーコードが JS コールバックを `Closure` で保持するため、インデックスの寿命中ずっと有効です。コールバックは常にメインスレッドで実行されるため、`Send + Sync` 制約は付きません。

## テスト

```bash
# ビルド確認
cargo build -p laurus-wasm --target wasm32-unknown-unknown

# Clippy
cargo clippy -p laurus-wasm --target wasm32-unknown-unknown -- -D warnings
```

ブラウザテストは `wasm-pack test` で実行できます:

```bash
wasm-pack test --headless --chrome
```
