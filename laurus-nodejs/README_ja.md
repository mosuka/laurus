# laurus-nodejs

[Laurus](https://github.com/mosuka/laurus) 検索ライブラリの
Node.js/TypeScript バインディング —
Lexical検索、Vector検索、ハイブリッド検索を統合的に提供します。

## 特徴

- **Lexical検索** — BM25スコアリング、Term/Phrase/Fuzzy/Wildcard/Geo/Boolean/Spanクエリ
- **Vector検索** — HNSW、Flat、IVFインデックス、複数の距離指標対応
- **ハイブリッド検索** — Lexical + Vector を RRF または WeightedSum で融合
- **Late interaction による再採点** — どの検索でも上位の結果を MultiVector フィールドに対する ColBERT 型の MaxSim で並べ替え
- **CJK対応** — [Lindera](https://github.com/lindera/lindera) による日本語・中国語・韓国語トークナイズ
- **ネイティブ性能** — [napi-rs](https://napi.rs) によるRustコア直接呼び出し、C APIオーバーヘッドなし
- **TypeScript型定義** — `.d.ts` ファイルの自動生成

## インストール

```bash
npm install laurus-nodejs
```

## クイックスタート

```javascript
import { Index, Schema } from "laurus-nodejs";

// スキーマ定義
const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");
schema.setDefaultFields(["title", "body"]);

// インメモリインデックスを作成
const index = await Index.create(null, schema);

// ドキュメントをインデックス
await index.putDocument("doc1", {
  title: "Rustプログラミング",
  body: "安全性と速度。",
});
await index.putDocument("doc2", {
  title: "Python入門",
  body: "汎用的なプログラミング言語。",
});
await index.commit();

// DSL文字列で検索
const results = await index.search("programming", 5);
for (const r of results) {
  console.log(r.id, r.score, r.document.title);
}
```

## API概要

### Index

```javascript
// インデックス作成（インメモリまたはファイルベース）
const index = await Index.create();                    // インメモリ
const index = await Index.create("./myindex", schema); // 永続化（./myindex/schema.toml + ./myindex/store/ を書き込む）
const reopened = await Index.create("./myindex");       // 再オープン: schema は省略（ディスクから読み込まれる）

// ドキュメント CRUD
await index.putDocument("id", { field: "value" });     // 上書き
await index.addDocument("id", { field: "chunk" });     // 追記（RAGパターン）
const docs = await index.getDocuments("id");
await index.deleteDocuments("id");
await index.commit();

// 検索
const results = await index.search("クエリDSL", limit, offset);
const results = await index.searchTerm("field", "term", limit);
const results = await index.searchVector("field", [0.1, ...], limit);
const results = await index.searchVectorText("field", "テキスト", limit);
const results = await index.searchWithRequest(searchRequest);

// 統計情報
const stats = index.stats();
// { documentCount: 42, vectorFields: {
//     embedding: { count: 42, dimension: 384 }
// } }

// クローズ（ストレージロックを確定的に解放する。詳細は下記参照）
index.close();
```

ファイルベースのインデックスは、`Index` を開いている間ディレクトリの排他ロック
を保持するため、最初のインデックスを開いたままだと同じパスに対する2つめの
`Index.create()` は失敗します。JS のガベージコレクタのタイミングは非決定的な
ので、それに頼らず、インデックスを使い終えたら（特に同じパスを再オープンする
前に）`close()` を呼んでください。`close()` は冪等で、呼び出し後は他の全ての
メソッドが例外を投げます。

インデックスを開き直す（あるいは開く）ことすらせずに、別プロセスが変更を
コミットしたかを安く確認したい場合は、エクスポートされている
`peekCommitGeneration(path)` を使ってください:

```javascript
import { peekCommitGeneration } from "laurus-nodejs";

const before = peekCommitGeneration(path);
// ... しばらく経過 ...
if (peekCommitGeneration(path) !== before) {
  // ディスク上で何かが変わった。取り込むにはインデックスを開き直す
}
```

これはディスク上の永続化されたcommit世代を直接読むだけで、`Engine`の構築
（ストレージロック・WALリカバリ・Embedder読み込み）を一切行わないため、
このプロセスでそのパスの`Index`をまだ一度も作っていなくても使えます。
`path`がlaurusのインデックスディレクトリでない（永続化されたスキーマが
無い）場合は例外を投げます。

### 永続性 / WAL

永続インデックスはすべての変更を先行書き込みログ（WAL）に書き込みます。
デフォルトでは WAL はレコードごとに `fsync` されるため、各書き込みは完全に
永続化されます。書き込みスループットを高めるためにグループコミットを有効化
すると `fsync` をまとめられます（クラッシュ時には SQLite の
`synchronous = NORMAL` と同様に最後の未同期バッチまでを失う可能性があります）:

```javascript
import { Index, WalSyncPolicy } from "laurus-nodejs";

// maxRecords, maxBytes, maxIntervalMs（いずれも省略可）
const policy = WalSyncPolicy.group(4096, undefined, 1000);
const index = await Index.create("./myindex", schema, policy);

await index.putDocument("doc1", { title: "Hello" });
await index.flushWal(); // 必要なときに永続性バリアを強制
await index.commit();   // WAL もフラッシュされます
```

`walSyncPolicy` を省略する（または `WalSyncPolicy.perRecord()` を渡す）と、
デフォルトのレコードごとの永続性が維持されます。

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
schema.addHnswField("embedding", 384, "cosine", 16, 200, undefined, "bert");
schema.addFlatField("embedding", 384);
schema.addIvfField("embedding", 384, "cosine", 100, 1);
schema.addMultiVectorField("tokens", 128, "dot_product"); // 再採点用のトークンベクトル
schema.addEmbedder("bert", {
  type: "candle_bert",
  model: "sentence-transformers/all-MiniLM-L6-v2",
});
schema.setDefaultFields(["title", "body"]);
```

### SearchRequest（高度な検索）

```javascript
import {
  RRF,
  SearchRequest,
  TermQuery,
  VectorQuery,
  VectorTextQuery,
} from "laurus-nodejs";

// Lexical とベクトルの句を RRF で融合する。各セッターは同じ種類の句を
// 置き換える（`setLexicalPhrase`、`setWeightedSumFusion` など）。
const req = new SearchRequest({ limit: 10, offset: 0 });
req.setLexicalTerm(new TermQuery("body", "programming"));
req.setVectorQuery(new VectorQuery("embedding", [0.1, 0.2, ...]));
req.setFilterTerm(new TermQuery("category", "tech"));
req.setRrfFusion(new RRF(60.0));
const results = await index.searchWithRequest(req);

// DSL はベクトルクエリと組み合わせられ、両方で検索する。
// DSL と `setLexicalX` のセッターを組み合わせると例外になる。Lexical の句は DSL に書く。
const req2 = new SearchRequest({ queryDsl: "title:hello", limit: 10 });
req2.setVectorTextQuery(new VectorTextQuery("embedding", "クエリテキスト"));
const results2 = await index.searchWithRequest(req2);
```

### Late interaction による再採点

どの検索でも、上位の結果を MultiVector フィールドに対する MaxSim で並べ替えます。
トークンベクトルは入れ子の数値配列で渡し、保存はされません:

```javascript
const schema = new Schema();
schema.addTextField("title");
schema.addMultiVectorField("tokens", 2, "dot_product");

const index = await Index.create(null, schema);
await index.putDocument("doc1", { title: "rust", tokens: [[0.9, 0.2], [0.0, 0.5]] });
await index.commit();

const rescore = { field: "tokens", vectors: [[1, 0], [0, 1]], windowSize: 100 };
const results = await index.search("title:rust", 10, 0, undefined, rescore);
const results2 = await index.searchWithRequest(new SearchRequest({ queryDsl: "title:rust", rescore }));
```

`vectors` と `text` はちょうど一方だけを指定します。`text` クエリには、フィールドに
`candle_colbert` の Embedder が必要です（`addEmbedder` で登録し、
`addMultiVectorField` の第 4 引数に指定）。

### テキスト解析

```javascript
import { WhitespaceTokenizer, SynonymDictionary, SynonymGraphFilter } from "laurus-nodejs";

const tokenizer = new WhitespaceTokenizer();
const tokens = tokenizer.tokenize("hello world");

const synDict = new SynonymDictionary();
synDict.addSynonymGroup(["ml", "machine learning"]);

const filter = new SynonymGraphFilter(synDict, true, 0.8);
const expanded = filter.apply(tokens);
```

## データ型マッピング

| JavaScript | Laurus フィールド型 |
| --- | --- |
| `string` | Text |
| `number`（整数） | Int64 |
| `number`（浮動小数点） | Float64 |
| `boolean` | Boolean |
| `null` | Null |
| `number[]` | Vector |
| `number[][]` | MultiVector（トークンベクトル。保存されない） |
| `{ lat, lon }` | Geo |
| `Date` / ISO8601文字列 | DateTime |
| `Buffer` | Bytes |

## サンプル

[examples/](examples/) ディレクトリを参照:

- [quickstart.mjs](examples/quickstart.mjs) — 基本的なインデックス・ドキュメント・検索
- [lexical-search.mjs](examples/lexical-search.mjs) — 各種Lexicalクエリ
- [vector-search.mjs](examples/vector-search.mjs) — HNSWによるベクトル検索
- [hybrid-search.mjs](examples/hybrid-search.mjs) — RRF/WeightedSumによるハイブリッド検索

## ソースからビルド

```bash
cd laurus-nodejs
npm install
npm run build        # リリースビルド
npm run build:debug  # デバッグビルド
npm test             # テスト実行
```

## ライセンス

MIT
