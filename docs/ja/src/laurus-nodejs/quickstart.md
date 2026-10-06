# クイックスタート

## 1. インデックスの作成

```javascript
import { Index, Schema } from "laurus-nodejs";

// インメモリインデックス（揮発性、プロトタイピング向け）
const index = await Index.create();

// ファイルベースインデックス（永続化）
// `./myindex/schema.toml` と `./myindex/store/` を書き込む -- これは
// `laurus-cli create index --schema` と同じレイアウトなので、このディレクトリは
// CLI からも開ける（逆も同様）。
const schema = new Schema();
schema.addTextField("name");
schema.addTextField("description");
const persistentIndex = await Index.create("./myindex", schema);

// 後で再オープンする際はパスだけで済む -- schema を再度渡すとエラーになる
// （スキーマは既に永続化されているため）。
const reopened = await Index.create("./myindex");
```

## 2. ドキュメントのインデックス

```javascript
await index.putDocument("express", {
  name: "Express",
  description: "Fast minimalist web framework for Node.js.",
});
await index.putDocument("fastify", {
  name: "Fastify",
  description: "Fast and low overhead web framework.",
});
await index.commit();
```

## 3. Lexical 検索

```javascript
// DSL 文字列
const results = await index.search("name:express", 5);

// Term クエリ
const results2 = await index.searchTerm(
  "description", "framework", 5,
);

// 結果の表示
for (const r of results) {
  console.log(`[${r.id}] score=${r.score.toFixed(4)}  ${r.document.name}`);
}
```

## 4. Vector 検索

Vector 検索にはベクトルフィールドを持つスキーマと
事前計算済みの埋め込みベクトルが必要です。

```javascript
import { Index, Schema } from "laurus-nodejs";

const schema = new Schema();
schema.addTextField("name");
schema.addHnswField("embedding", 4);

const index = await Index.create(null, schema);
await index.putDocument("express", {
  name: "Express",
  embedding: [0.1, 0.2, 0.3, 0.4],
});
await index.putDocument("pg", {
  name: "pg",
  embedding: [0.9, 0.8, 0.7, 0.6],
});
await index.commit();

const results = await index.searchVector(
  "embedding", [0.1, 0.2, 0.3, 0.4], 3,
);
```

## 5. ハイブリッド検索

```javascript
import {
  Index,
  RRF,
  SearchRequest,
  TermQuery,
  VectorQuery,
} from "laurus-nodejs";

const req = new SearchRequest({ limit: 5 });
req.setLexicalTerm(new TermQuery("name", "express"));
req.setVectorQuery(new VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4]));
req.setRrfFusion(new RRF(60.0));

const results = await index.searchWithRequest(req);
```

## 6. Late interaction による再採点

MultiVector フィールドは文書ごとに複数のトークンベクトル（たとえば ColBERT の
トークン埋め込み）を保持します。`rescore` オブジェクトを渡すと、どの検索でも
上位の結果をそのフィールドに対する late interaction（MaxSim）で並べ替えます。

```javascript
import { Index, Schema } from "laurus-nodejs";

const schema = new Schema();
schema.addTextField("title");
schema.addMultiVectorField("tokens", 2, "dot_product");

const index = await Index.create(null, schema);
await index.putDocument("a", {
  title: "rust",
  tokens: [[0.1, 0.0]],
});
await index.putDocument("b", {
  title: "rust language",
  tokens: [[0.9, 0.2], [0.0, 0.5]],
});
await index.commit();

// Lexical 検索の上位を MaxSim で並べ替える:
// "b"（スコア 1.4）が "a"（スコア 0.1）より上位になる。
const results = await index.search("title:rust", 10, 0, undefined, {
  field: "tokens",
  vectors: [[1, 0], [0, 1]],
});
```

トークンベクトルは保存されないため、`getDocuments` や検索結果には含まれません。
`addEmbedder` で `candle_colbert` の Embedder を登録してフィールドの第 4 引数に
指定すると、文書はフィールドにテキストを渡せ、再採点も `vectors` の代わりに
`text` を受け付けます。詳細は
[Late interaction による再採点（Rescore）](api_reference.md#late-interaction-による再採点rescore)
を参照してください。

## 7. 更新と削除

```javascript
// 更新: putDocument は既存バージョンをすべて置換
await index.putDocument("express", {
  name: "Express v5",
  description: "Updated content.",
});
await index.commit();

// バージョン追記（RAG チャンキングパターン）
await index.addDocument("express", {
  name: "Express chunk 2",
  description: "Additional chunk.",
});
await index.commit();

// 全バージョンの取得
const docs = await index.getDocuments("express");

// 削除
await index.deleteDocuments("express");
await index.commit();
```

## 8. スキーマ管理

```javascript
const schema = new Schema();
schema.addTextField("name");
schema.addTextField("description");
schema.addIntegerField("stars");
schema.addFloatField("score");
schema.addBooleanField("published");
schema.addBytesField("thumbnail");
schema.addGeoField("location");
schema.addDatetimeField("createdAt");
schema.addHnswField("embedding", 384);
schema.addFlatField("smallVec", 64);
schema.addIvfField("ivfVec", 128, "cosine", 100, 1);
schema.addMultiVectorField("tokens", 128);
```

## 9. インデックス統計

```javascript
const stats = index.stats();
console.log(stats.documentCount);
console.log(stats.vectorFields);
```
