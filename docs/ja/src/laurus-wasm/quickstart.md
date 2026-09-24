# クイックスタート

## 基本的な使い方（インメモリ）

```javascript
import init, { Index, Schema } from 'laurus-wasm';

// WASM モジュールを初期化
await init();

// スキーマを定義
const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");
schema.setDefaultFields(["title", "body"]);

// インメモリインデックスを作成
const index = await Index.create(schema);

// ドキュメントを追加
await index.putDocument("doc1", {
  title: "Rust 入門",
  body: "Rust はシステムプログラミング言語です"
});
await index.putDocument("doc2", {
  title: "WebAssembly ガイド",
  body: "WASM はブラウザでネイティブに近いパフォーマンスを実現します"
});
await index.commit();

// 検索
const results = await index.search("rust");
for (const result of results) {
  console.log(`${result.id}: ${result.score}`);
  console.log(result.document);
}
```

## 永続化ストレージ（OPFS）

```javascript
import init, { Index, Schema } from 'laurus-wasm';

await init();

const schema = new Schema();
schema.addTextField("title");
schema.addTextField("body");

// 永続化インデックスを開く（ページリロード後もデータが保持される）
// 初回はスキーマもデータと一緒に永続化される。
const index = await Index.open("my-index", schema);

// ドキュメントを追加
await index.putDocument("doc1", {
  title: "Hello",
  body: "World"
});

// commit() で自動的に OPFS に永続化される
await index.commit();

// 次のページロード時は、schema 引数を省略した Index.open("my-index") で
// データと永続化済みスキーマの両方が復元される。
const reopened = await Index.open("my-index");
```

## 日本語形態素検索

ブラウザ WASM では Lindera 辞書をファイルシステムパスで指定できないため、
OPFS にロードした IPADIC のバイト列から analyzer を構築します。

```javascript
import init, { Index, Schema, JapaneseAnalyzer } from 'laurus-wasm';
import {
  downloadDictionary,
  getDictionaryVersion,
  loadDictionaryFiles,
  hasDictionary,
} from 'laurus-wasm/opfs';

await init();

// 1. 初回訪問時に IPADIC アーカイブを OPFS にキャッシュする。zip は
//    アプリと同一オリジンで配信する必要がある（GitHub Releases は CORS
//    でブロックされる）。圧縮 ~16 MB / 展開後 ~58 MB。
//    バイナリ形式は WASM にコンパイルされた Lindera バージョンに紐づく
//    ため、`version` でキャッシュにスタンプを付け、ビルドが期待する
//    バージョンと一致しなくなったら再ダウンロードする。
const LINDERA_VERSION = "5.0.2"; // Cargo.lock の lindera バージョンと揃える
if (
  !(await hasDictionary("ipadic"))
  || (await getDictionaryVersion("ipadic")) !== LINDERA_VERSION
) {
  await downloadDictionary("./dict/lindera-ipadic.zip", "ipadic", {
    version: LINDERA_VERSION,
    onProgress: ({ phase, loaded, total }) => console.log(phase, loaded, total),
  });
}

// 2. 9 つのコンポーネントファイルを読み出して analyzer を構築する。
const f = await loadDictionaryFiles("ipadic");
const ja = JapaneseAnalyzer.fromBytes(
  f.metadata, f.dictTrie, f.dictValsIdx, f.dictVals,
  f.dictWordsIdx, f.dictWords, f.matrixMtx, f.charDef, f.unk,
  "normal",
);

// 3. analyzer をスキーマに登録し、テキストフィールドから名前で参照する。
const schema = new Schema();
schema.addAnalyzer("ja-ipadic", ja);
schema.addTextField("title", undefined, undefined, undefined, undefined, "ja-ipadic");
schema.addTextField("body", undefined, undefined, undefined, undefined, "ja-ipadic");
schema.setDefaultFields(["title", "body"]);

const index = await Index.create(schema);

await index.putDocument("doc1", {
  title: "形態素解析",
  body: "Lindera は Rust 製の形態素解析ライブラリです。",
});
await index.commit();

const results = await index.search("形態素");
console.log(results[0].document.title); // "形態素解析"
```

`JapaneseAnalyzer.fromBytes` の完全なシグネチャと OPFS ヘルパ API は
[API リファレンス](api_reference.md#japaneseanalyzer) を参照してください。

## ベクトル検索

```javascript
import init, { Index, Schema } from 'laurus-wasm';

await init();

const schema = new Schema();
schema.addTextField("title");
schema.addHnswField("embedding", 3); // 3次元ベクトル

const index = await Index.create(schema);

await index.putDocument("doc1", {
  title: "Rust",
  embedding: [1.0, 0.0, 0.0]
});
await index.putDocument("doc2", {
  title: "Python",
  embedding: [0.0, 1.0, 0.0]
});
await index.commit();

// ベクトル類似度で検索
const results = await index.searchVector("embedding", [0.9, 0.1, 0.0]);
console.log(results[0].document.title); // "Rust"
```

## バンドラーでの利用

### Vite

```javascript
// vite.config.js
import wasm from 'vite-plugin-wasm';

export default {
  plugins: [wasm()]
};
```

### Webpack 5

Webpack 5 は `asyncWebAssembly` で WASM をネイティブサポートしています:

```javascript
// webpack.config.js
module.exports = {
  experiments: {
    asyncWebAssembly: true
  }
};
```
