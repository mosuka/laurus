# laurus-wasm

[Laurus](https://github.com/mosuka/laurus) 検索ライブラリの
WebAssembly バインディング —
ブラウザ上で Lexical 検索、Vector 検索、Hybrid 検索を実行できます。

## 特徴

- **Lexical 検索** — BM25 スコアリングによる Term、Phrase、Fuzzy、
  Wildcard、Geo、Boolean、Span クエリ
- **Vector 検索** — HNSW、Flat、IVF インデックスと複数の距離メトリクス
- **Hybrid 検索** — Lexical 検索と Vector 検索を
  RRF または Weighted Sum フュージョンで組み合わせ
- **CJK 対応** — [Lindera](https://github.com/lindera/lindera) による日本語・中国語・韓国語のトークナイズ
- **OPFS 永続化** — ブラウザの Origin Private File System を使用し、
  ページリロード後もデータを保持
- **JS コールバック Embedder** — JavaScript コールバックで
  任意のエンベディング関数を提供可能（例: Transformers.js）

## クイックスタート

```javascript
import init, { Index, Schema } from "./pkg/laurus_wasm.js";

await init();

// スキーマ定義
const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");
schema.setDefaultFields(["title", "body"]);

// OPFS 永続化インデックスを作成（ページリロード後もデータ保持）
// スキーマは初回利用時にインデックスへ永続化される。後で再オープンする際は
// 名前だけで済む（`Index.open("my-index")`）— 詳細は下記を参照。
const index = await Index.open("my-index", schema);

// ドキュメントをインデックス
await index.putDocument("doc1", {
  title: "Rust プログラミング",
  body: "安全性と速度。",
});
await index.putDocument("doc2", {
  title: "Python 入門",
  body: "汎用的な言語。",
});
await index.commit();

// DSL 文字列で検索
const results = await index.search("programming", 5);
for (const r of results) {
  console.log(r.id, r.score, r.document.title);
}
```

## API 概要

### Index

```javascript
// インデックス作成（インメモリまたは OPFS 永続化）
const index = await Index.create(schema);              // インメモリ（揮発性）。schema は省略可能（既定: 空）
const index = await Index.open("my-index", schema);    // OPFS（永続化）、初回: schema を永続化
const reopened = await Index.open("my-index");         // OPFS（永続化）、以降: schema は自動的に読み込まれる

// ドキュメント CRUD
await index.putDocument("id", { field: "value" });     // upsert
await index.addDocument("id", { field: "chunk" });     // 追記 (RAG)
const docs = await index.getDocuments("id");
await index.deleteDocuments("id");
await index.commit();                                  // フラッシュ + OPFS に永続化

// 検索
const results = await index.search("query DSL", limit, offset);
const results = await index.searchTerm("field", "term", limit);
const results = await index.searchVector("field", [0.1, ...], limit);
const results = await index.searchVectorText("field", "text", limit);

// 統計情報
const stats = index.stats();
// { documentCount: 42, vectorFields: {
//     embedding: { count: 42, dimension: 384 }
// } }
```

### 永続性 / WAL

各変更は、エンジンのインメモリ先行書き込みログ（WAL）に追記されます。
デフォルトでは WAL はレコードごとにフラッシュされます。書き込みスループットを
高めるためにグループコミットを有効化するとフラッシュをまとめられます
（クラッシュ時には SQLite の `synchronous = NORMAL` と同様に最後の未同期
バッチまでを失う可能性があります）:

```javascript
import { Index, Schema, WalSyncPolicy } from "./pkg/laurus_wasm.js";

// maxRecords, maxBytes, maxIntervalMs（いずれも省略可）
const policy = WalSyncPolicy.group(4096, undefined, 1000);
const index = await Index.open("my-index", schema, policy);

await index.putDocument("doc1", { title: "Hello" });
await index.flushWal(); // エンジン WAL のみをフラッシュ
await index.commit();   // 変更を検索可能にし、かつ OPFS に永続化する
```

WASM の注意点: `maxIntervalMs` のバックグラウンドタイマーは wasm では
**no-op** です（バックグラウンドスレッドがないため）。また `flushWal()` は
インメモリエンジンの WAL のみをフラッシュし、OPFS への永続化は引き続き
`commit()` で行われます。永続的な永続化には `commit()` を呼び出してください。
`walSyncPolicy` を省略する（または `WalSyncPolicy.perRecord()` を渡す）と、
デフォルトのレコードごとの動作が維持されます。

### Schema

```javascript
const schema = new Schema();
// 第 6 引数が analyzer。ここでは組込名を指定。Japanese などのカスタム analyzer は
// 先に addAnalyzer で登録し、その名前で参照する（API リファレンス参照）。
schema.addTextField("title", true, true, false, true, "english");
schema.addIntegerField("year");
schema.addFloatField("price");
schema.addBooleanField("active");
schema.addDatetimeField("created_at");
schema.addGeoField("location");
schema.addBytesField("thumbnail");
schema.addHnswField("embedding", 384, "cosine", 16, 200, undefined, "minilm");
schema.addFlatField("embedding", 384);
schema.addIvfField("embedding", 384, "cosine", 100, 1);
schema.addEmbedder("minilm", {
  type: "callback",
  embed: async (text) => {
    // エンベディング関数を指定（例: Transformers.js）
    return [0.1, 0.2, ...];
  },
});
schema.setDefaultFields(["title", "body"]);
```

## サンプル

[examples/](examples/) ディレクトリには、複数のシングルページ
デモが配置されています。サンプル一覧トップは
[`examples/index.html`](examples/index.html) です。個別のサンプル
にも直接アクセスできます。

- [`examples/basic/`](examples/basic/) — 統合クエリ DSL を用いた
  日本語の基本的なハイブリッド検索（全文検索 + ベクトル検索、
  Transformers.js エンベディング併用）
- [`examples/geo/`](examples/geo/) — Leaflet 地図に東京の観光
  スポットをプロットし、ビューポートから生成した
  `location:geo_bbox(...)` をテキスト検索・ベクトル検索と組み
  合わせるサンプル

共通アセット（テーマスタイル、ロガー、辞書ローダー、Embedder
ヘルパ）は `examples/shared/` 配下にまとめており、各サンプルは
それぞれの本題に集中できる構成にしています。

## ソースからのビルド

```bash
cd laurus-wasm

# 開発ビルド
wasm-pack build --target web --dev

# リリースビルド
wasm-pack build --target web --release

# OPFS ヘルパ（./opfs サブパス）を pkg/ に同梱
./scripts/postbuild.sh

# デモの起動（サンプル一覧トップ）
python3 -m http.server 8080
# http://localhost:8080/examples/ を開いて、サンプルを選択
```

ポストビルドスクリプトは `js/opfs.js` と `js/opfs.d.ts` を `pkg/` に
コピーし、利用者が
`import { downloadDictionary } from "laurus-wasm/opfs"` できるよう
`pkg/package.json` を書き換えます。`wasm-pack build` 後に毎回実行
してください（冪等です）。

## ライセンス

MIT
