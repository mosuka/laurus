# API リファレンス

## モジュール関数

### `version()`

laurus-wasm のビルドバージョン文字列（例: `"0.12.1"`）を返します。
laurus の状態を OPFS に永続化するアプリケーションは、この値を
スタンプとして保存しておくことで、オンディスクフォーマットが
変わった可能性のある別ビルド由来の状態を検出できます（デモの
サンプルが実際にこの方式を使っています）。

## Index

検索インデックスの作成・クエリを行うメインエントリポイントです。

### 静的メソッド

#### `Index.create(schema?, walSyncPolicy?, commitPolicy?)`

新しいインメモリ（一時）インデックスを作成します。

- **引数:**
  - `schema` (Schema, 省略可) -- スキーマ定義。省略時は空のスキーマが使用されます
  - `walSyncPolicy` (WalSyncPolicy, 省略可) -- WAL の永続性ポリシー。省略すると
    デフォルトのレコードごとの同期を使用します。
    [WAL 同期ポリシー / 永続性](#wal-同期ポリシー--永続性)を参照してください。
  - `commitPolicy` (CommitPolicy, 省略可) -- 自動コミットポリシー。省略すると
    デフォルト（manual: 呼び出し側がコミットを駆動）を使用します。
    [コミットポリシー / 自動コミット](#コミットポリシー--自動コミット)を参照してください。
- **戻り値:** `Promise<Index>`

#### `Index.open(name, schema?, walSyncPolicy?, commitPolicy?)`

OPFS で永続化されたインデックスを開くか、新規作成します。

- **引数:**
  - `name` (string) -- インデックス名（OPFS サブディレクトリ）
  - `schema` (Schema, 省略可) -- スキーマ定義、およびそのセッションで必要な
    embedder コールバック・ランタイムアナライザー。インデックスを初めて作成する
    とき、またはスキーマ永続化に対応する前に永続化されたインデックスを開くとき
    （後述）は**必須**。それ以降は**省略可能**です — フィールドスキーマ部分は
    インデックスのデータと一緒に永続化され、自動的に再読み込みされるためです。
    スキーマが既に永続化された状態で `schema` を渡した場合、そのフィールド定義
    は無視され永続化済みのものが使われます。embedder コールバック・ランタイム
    アナライザーだけは永続化できないため、必要なセッションでは毎回渡す必要が
    あります
  - `walSyncPolicy` (WalSyncPolicy, 省略可) -- WAL の永続性ポリシー。省略すると
    デフォルトのレコードごとの同期を使用します。
    [WAL 同期ポリシー / 永続性](#wal-同期ポリシー--永続性)を参照してください。
  - `commitPolicy` (CommitPolicy, 省略可) -- 自動コミットポリシー。省略すると
    デフォルト（manual: 呼び出し側がコミットを駆動）を使用します。
    [コミットポリシー / 自動コミット](#コミットポリシー--自動コミット)を参照してください。
- **戻り値:** `Promise<Index>`
- **例外:** この OPFS インデックスにスキーマ永続化対応前のデータが既に存在し、
  かつ一度きりの移行を完了するための `schema` が渡されなかった場合に発生します
  （スキーマはその場で永続化され、以降の再オープンには不要になります）。

### インスタンスメソッド

#### `putDocument(id, document)`

ドキュメントを置換（upsert）します。

- **引数:**
  - `id` (string) -- ドキュメント識別子
  - `document` (object) -- スキーマフィールドに対応するキーバリューペア
- **戻り値:** `Promise<void>`

#### `addDocument(id, document)`

ドキュメントバージョンを追加します（マルチバージョン RAG パターン）。

- **引数・戻り値:** `putDocument` と同じ

#### `putDocuments(docs)`

バッチ upsert。ペアをバッチ全体で WAL fsync 1 回で順に適用します。1 バッチ内で重複した ID はデデュープされます（最後が勝ち）。最初の不正エントリで fail-fast し、適用済みの prefix はロールバックされません（再試行は冪等）。

- **引数:**
  - `docs` (`Array<[string, object]>`) -- `[id, document]` ペアの配列
- **戻り値:** `Promise<void>`

#### `addDocuments(docs)`

バッチチャンク追記。`putDocuments` と同様ですが、繰り返した ID は別バージョンとして蓄積されます。

- **引数・戻り値:** `putDocuments` と同じ

#### `getDocuments(id)`

ドキュメントの全バージョンを取得します。

- **引数:** `id` (string)
- **戻り値:** `Promise<object[]>`

#### `deleteDocuments(id)`

ドキュメントの全バージョンを削除します。

- **引数:** `id` (string)
- **戻り値:** `Promise<void>`

#### `commit()`

書き込みをフラッシュし、変更を検索可能にします。
`Index.open()` で作成したインデックスの場合、OPFS にも自動永続化されます。

- **戻り値:** `Promise<void>`

#### `flushWal()`

インメモリエンジンの WAL に対して永続性バリアを強制します。wasm 固有の
注意点については [WAL 同期ポリシー / 永続性](#wal-同期ポリシー--永続性) を
参照してください — 特にこれは OPFS への永続化を**行いません**。永続的な
永続化には `commit()` を呼び出してください。

- **戻り値:** `Promise<void>`

#### `search(query, limit?, offset?)`

DSL 文字列クエリで検索します。

- **引数:**
  - `query` (string) -- クエリ DSL（例: `"title:hello"`）
  - `limit` (number, デフォルト 10)
  - `offset` (number, デフォルト 0)
- **戻り値:** `Promise<SearchResult[]>`

#### `searchTerm(field, term, limit?, offset?)`

完全一致タームで検索します。

- **引数:**
  - `field` (string) -- フィールド名
  - `term` (string) -- 検索ターム
  - `limit`, `offset` (number, 省略可)
- **戻り値:** `Promise<SearchResult[]>`

#### `searchVector(field, vector, limit?, offset?)`

ベクトル類似度で検索します。

- **引数:**
  - `field` (string) -- ベクトルフィールド名
  - `vector` (number[]) -- クエリ埋め込みベクトル
  - `limit`, `offset` (number, 省略可)
- **戻り値:** `Promise<SearchResult[]>`

#### `searchVectorText(field, text, limit?, offset?)`

テキストで検索します（登録された埋め込み器で変換）。

- **引数:**
  - `field` (string) -- ベクトルフィールド名
  - `text` (string) -- 埋め込み対象テキスト
  - `limit`, `offset` (number, 省略可)
- **戻り値:** `Promise<SearchResult[]>`

#### `searchGeo3dDistance(field, x, y, z, distanceM, limit?, offset?)`

3D ECEF 座標フィールドへの球距離検索。中心 `(x, y, z)` から `distanceM` メートル以内
の座標を持つドキュメントを返します。ECEF の理論については
[Geo3d の概念](../concepts/geo3d.md) を参照。

- **引数:**
  - `field` (string) -- Geo3d フィールド名
  - `x`, `y`, `z` (number) -- 中心 ECEF 座標（メートル）
  - `distanceM` (number) -- 中心からの最大距離（メートル）
  - `limit`, `offset` (number, 省略可)
- **戻り値:** `Promise<SearchResult[]>`

#### `searchGeo3dBoundingBox(field, minX, minY, minZ, maxX, maxY, maxZ, limit?, offset?)`

3D ECEF 座標フィールドへの軸並行範囲（AABB）検索。

- **引数:**
  - `field` (string) -- Geo3d フィールド名
  - `minX`, `minY`, `minZ`, `maxX`, `maxY`, `maxZ` (number) -- 範囲境界（メートル）
  - `limit`, `offset` (number, 省略可)
- **戻り値:** `Promise<SearchResult[]>`

#### `searchGeo3dNearest(field, x, y, z, k, limit?, offset?, initialRadiusM?, maxRadiusM?)`

3D ECEF 座標フィールドへの k 最近傍検索。`(x, y, z)` から最も近い `k` 件のドキュ
メントを返します。`initialRadiusM` / `maxRadiusM`（オプション）で反復拡張サーチの
探索コーンを調整できます。

- **引数:**
  - `field` (string) -- Geo3d フィールド名
  - `x`, `y`, `z` (number) -- 中心 ECEF 座標（メートル）
  - `k` (number) -- 返す近傍件数
  - `limit`, `offset` (number, 省略可)
  - `initialRadiusM`, `maxRadiusM` (number, 省略可)
- **戻り値:** `Promise<SearchResult[]>`

#### `stats()`

インデックス統計を返します。

- **戻り値:** `{ documentCount: number, vectorFields: { [name]: { count, dimension } } }`

## WAL 同期ポリシー / 永続性

各書き込みは、エンジンのインメモリ先行書き込みログ（WAL）に追記されます。
`Index.create` と `Index.open` はオプションの `walSyncPolicy` を受け付け、
WAL をどの頻度でフラッシュするかを制御します。デフォルト（引数を省略）は
レコードごとの同期です。

```typescript
class WalSyncPolicy {
  static perRecord(): WalSyncPolicy;
  static group(
    maxRecords?: number,
    maxBytes?: number,
    maxIntervalMs?: number,
  ): WalSyncPolicy;
}
```

| コンストラクタ | 説明 |
| :--- | :--- |
| `WalSyncPolicy.perRecord()` | デフォルト。WAL レコードごとにフラッシュします。 |
| `WalSyncPolicy.group(...)` | グループコミット。複数の書き込みにまたがってフラッシュをまとめます。 |

`group(...)` のパラメータ（引数を省略するとそのデフォルトを維持）:

| パラメータ | デフォルト | 説明 |
| :--- | :--- | :--- |
| `maxRecords` | `1024` | この件数のレコードが蓄積されたらフラッシュします。 |
| `maxBytes` | `1048576`（1 MiB） | この量の未同期バイトが蓄積されたらフラッシュします。 |
| `maxIntervalMs` | なし | 定期フラッシュタイマー（ミリ秒）。**wasm では no-op**（注意点を参照）。 |

グループコミットでは、`maxRecords` または `maxBytes` の**いずれか**に達した
時点でエンジン WAL がフラッシュされ、`commit()` 時にも必ずフラッシュされます。
クラッシュ時には最後の未同期バッチまでを失う可能性があります — これは
SQLite の `synchronous = NORMAL` と同じトレードオフです。

### `flushWal()`（永続性バリア）

`flushWal()` はインメモリエンジンの WAL を必要なときにフラッシュします。

- **戻り値:** `Promise<void>`

### WASM の注意点

WebAssembly にはバックグラウンドスレッドや直接のファイルシステムがないため、
ネイティブバインディングとは 2 点で動作が異なります:

- **`maxIntervalMs` は no-op です。** 定期フラッシュタイマーには
  バックグラウンドスレッドが必要ですが、wasm では利用できません。
  グループコミットは `maxRecords` / `maxBytes` のしきい値到達時と
  `commit()` 時にはフラッシュされます。
- **`flushWal()` はインメモリエンジンの WAL のみをフラッシュします。**
  OPFS への永続化は引き続き `commit()` で行われます。wasm で永続的に
  永続化するには `commit()` を呼び出してください。

```javascript
import { Index, Schema, WalSyncPolicy } from "./pkg/laurus_wasm.js";

const schema = new Schema();
schema.addTextField("title");

// グループコミットを有効化。maxIntervalMs は受け付けられますが wasm では無視されます。
const policy = WalSyncPolicy.group(4096, undefined, 1000);
const index = await Index.open("my-index", schema, policy);

for (let i = 0; i < 10000; i++) {
  await index.putDocument(`doc${i}`, { title: `Document ${i}` });
}

await index.flushWal(); // エンジン WAL をフラッシュ（OPFS ではない）
await index.commit();   // 変更を検索可能にし、かつ OPFS に永続化する
```

## コミットポリシー / 自動コミット

コミットはバッファされた書き込みを検索可能なストアに反映します。
`Index.create` と `Index.open` はオプションの `commitPolicy` を受け付け、
エンジンが代わりにコミットするかどうかを制御します。デフォルト（引数を省略）は
manual で、すべての `commit()` を自分で駆動します。

```typescript
class CommitPolicy {
  static manual(): CommitPolicy;
  static everyDocs(n: number): CommitPolicy;
  static intervalMs(ms: number): CommitPolicy;
}
```

| コンストラクタ | 説明 |
| :--- | :--- |
| `CommitPolicy.manual()` | デフォルト。自動コミットなし。呼び出し側がすべての `commit()` を駆動します。 |
| `CommitPolicy.everyDocs(n)` | `n` 件のドキュメントを適用するたびに自動コミットします。 |
| `CommitPolicy.intervalMs(ms)` | バックグラウンドタイマーで少なくとも `ms` ミリ秒ごとに自動コミットします（デフォルト: なし）。**ネイティブ専用 — wasm では no-op。** |

`everyDocs(n)` では、エンジンは `n` 件のドキュメントを適用するたびに 1 回
コミットします。カウンタは単発とバッチの両方の取り込みにまたがり、バッチの
**内部**でも発火します — `n` より大きい `putDocuments` 呼び出しはバッチの
途中で 1 回以上のコミットを引き起こします。`everyDocs(0)` は有効で、自動
コミットを無効化します。これは `CommitPolicy.manual()` と等価です。

`intervalMs(ms)` は `everyDocs` の時間ベース版です: バックグラウンドタイマーが
少なくとも `ms` ミリ秒ごとにコミットするため、取り込みがアイドル状態でも
末尾の部分的なバッチがコミットされます。これは**ネイティブ専用**です — 下記の
WASM の注意点を参照してください。

`commitPolicy` は `walSyncPolicy` と**直交**します: `walSyncPolicy` は永続性の
ために WAL をどの頻度で fsync するかを制御し、`commitPolicy` はバッファされた
書き込みをいつ検索可能な状態へ反映するかを制御します。両者は独立して設定
できます。

### WASM の注意点

`walSyncPolicy` の `maxIntervalMs` バックグラウンドタイマー（wasm では no-op）
とは異なり、`everyDocs` はバックグラウンドスレッドを**必要としません** —
ドキュメントカウンタは取り込み中にインラインでチェックされます — そのため
自動コミットは WebAssembly 上でも完全に動作します。

一方 `intervalMs` は `walSyncPolicy` の `maxIntervalMs` と同じくバックグラウンド
タイマーに依存しますが、wasm にはバックグラウンドスレッドがありません。
ファクトリはポータブルなポリシーコードがコンパイルできるよう値を構築します
が、タイマーは WebAssembly 上では決して動作せず、**`intervalMs` は wasm では
効果がありません** — タイマーによるコミットは一切発火しません。wasm での自動
コミットには `everyDocs` を使用してください。

```javascript
import { Index, Schema, CommitPolicy } from "./pkg/laurus_wasm.js";

const schema = new Schema();
schema.addTextField("title");

// 1000 件のドキュメントを適用するたびに自動コミット。
const index = await Index.open(
  "my-index",
  schema,
  undefined,
  CommitPolicy.everyDocs(1000),
);

for (let i = 0; i < 10000; i++) {
  await index.putDocument(`doc${i}`, { title: `Document ${i}` });
}
// エンジンは 10 回自動コミット済み。明示的な commit() は不要。
```

## Schema

インデックスフィールドと埋め込み器を定義するビルダーです。

### コンストラクタ

#### `new Schema()`

空のスキーマを作成します。

### メソッド

#### `addTextField(name, stored?, indexed?, termVectors?, analyzer?)`

全文検索テキストフィールドを追加します。`analyzer` にはパラメータ不要の
組込名（`"standard"` / `"english"` / `"keyword"` / `"simple"` /
`"noop"`）または `addAnalyzer()` で登録したランタイム analyzer 名を
指定します。

日本語の形態素解析を行う場合は、まず `JapaneseAnalyzer` を IPADIC の
バイト列から構築し、`addAnalyzer()` で登録してください。
[`JapaneseAnalyzer.fromBytes`](#japaneseanalyzerfrombytesmetadata-dicttrie--mode)
と [`addAnalyzer`](#addanalyzername-analyzer) を参照。

#### `addIntegerField(name, stored?, indexed?, multiValued?)`

64 ビット整数フィールドを追加します。`multiValued: true` を指定すると整数配列を受け付け、
範囲クエリは**いずれかの値**が条件を満たせばマッチ（Lucene 流の "any match"、constant スコア）します。

#### `addFloatField(name, stored?, indexed?, multiValued?)`

64 ビット浮動小数点フィールドを追加します。`multiValued: true` を指定すると浮動小数点配列を受け付け、
範囲クエリは**いずれかの値**が条件を満たせばマッチ（Lucene 流の "any match"、constant スコア）します。

#### `addBooleanField(name, stored?, indexed?)`

真偽値フィールドを追加します。

#### `addDatetimeField(name, stored?, indexed?)`

日時フィールドを追加します。

#### `addGeoField(name, stored?, indexed?)`

地理座標フィールドを追加します。

#### `addGeo3dField(name, stored?, indexed?)`

3D ECEF カルテシアン座標フィールド（x, y, z はメートル）を追加します。値は
`{ x, y, z }` オブジェクトで投入します。詳細は
[Geo3d の概念](../concepts/geo3d.md) を参照。

WASM バインディングは `Geo3dDistanceQuery` / `Geo3dBoundingBoxQuery` /
`Geo3dNearestQuery` を JS クラスとして公開していません（wasm-bindgen は
`dyn Query` トレイトオブジェクトを公開できないため）。代わりに上記の
`Index.searchGeo3dDistance` / `Index.searchGeo3dBoundingBox` /
`Index.searchGeo3dNearest` メソッドを使用してください。

#### `addBytesField(name, stored?)`

バイナリデータフィールドを追加します。

#### `addHnswField(name, dimension, distance?, m?, efConstruction?, defaultEfSearch?, embedder?, quantizer?, subvectorCount?, rerankStorage?, pqCodebookPath?, baseWeight?)`

HNSW ベクトルインデックスフィールドを追加します。

- `distance`: `"cosine"`（デフォルト）、`"euclidean"`、`"dot_product"`、`"manhattan"`、`"angular"`
- `m`: 分岐係数（デフォルト 16）
- `efConstruction`: 構築時の探索幅（デフォルト 200）
- `defaultEfSearch`: クエリ時の `ef_search`（候補リストサイズ）のスキーマレベルデフォルト。省略時は内部フォールバックの 50 を使用します
- `quantizer`: `"scalar_8bit"`（デフォルト）または `"product_quantization"`（`subvectorCount` が必須）
- `subvectorCount`: PQ サブベクトル数。`dimension` を割り切れる値を指定します
- `rerankStorage`: 省略（デフォルト）するか、`"f32"` を指定して完全精度のリランクサイドカーを保存します
- `pqCodebookPath`: 省略（デフォルト）するか、segment 間で再利用する共有 PQ codebook のストレージ相対ファイル名（Issue #631）を指定します（segment ごとの学習の代替）
- `baseWeight`: 他の vector フィールドと同時に検索されたときの、このフィールドの相対的なスコアリング優先度（デフォルト `1.0`、Issue #1084）。[ウェイト](../concepts/search/vector_search.md#ウェイト)を参照してください

#### `addFlatField(name, dimension, distance?, embedder?, baseWeight?)`

全探索ベクトルインデックスフィールドを追加します。

#### `addIvfField(name, dimension, distance?, nClusters?, nProbe?, embedder?, baseWeight?)`

IVF ベクトルインデックスフィールドを追加します。

- `nClusters`: パーティショニングクラスタ数（デフォルト 100）
- `nProbe`: 検索時にプローブするクラスタ数（デフォルト 1）

**ベクトル量子化とリランクストレージ**（HNSW フィールド）:

- `quantizer` — `"scalar_8bit"`（デフォルト、4 倍圧縮）または高圧縮率の `"product_quantization"`。Product quantization では `subvectorCount`（`dimension` を割り切れる値）が必須です。
- `rerankStorage` — `"f32"` を指定すると完全精度の `*.hnsw.f32` サイドカーを書き出し、厳密な Stage-2 リランクを有効化します。省略すると int8 のみのセグメントを維持します。
- `pqCodebookPath` — 共有 PQ codebook のストレージ相対ファイル名（Issue #631）。`laurus train pq-codebook` CLI コマンドで一度だけ学習します。`quantizer: "product_quantization"` との組み合わせでのみ意味を持ち、以後の commit は segment ごとの k-means 再学習の代わりに学習済み codebook で encode します。省略すると segment ごとの学習を維持します。

#### `addAnalyzer(name, analyzer)`

事前に構築した analyzer インスタンスを `name` で登録します。テキスト
フィールドが `Named` 形式で analyzer を参照するときに、組込名や
`schema.analyzers` 定義よりも先に解決されます。

現状は [`JapaneseAnalyzer.fromBytes`](#japaneseanalyzerfrombytesmetadata-dicttrie--mode)
で構築した `JapaneseAnalyzer` のみ受け付けます。ブラウザ WASM では
`{ "language": "japanese", "dict": ... }` プリセットがファイルシステム
パスを解決できないため、ランタイムレジストリ経由が日本語 analyzer を
利用する唯一の現実的な経路です。

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
schema.addTextField("body", undefined, undefined, undefined, "ja-ipadic");
```

#### `addEmbedder(name, config)`

名前付き埋め込み器を登録します。WASM では以下の 2 種類の `type` をサポートします:

- `"precomputed"` — 埋め込みは行いません。ベクトルは `putDocument()` /
  `searchVector()` 経由で直接渡します。
- `"callback"` — JavaScript コールバック `embed: (text) => Promise<number[]>` を
  登録します。エンジンがインジェスト時および `searchVectorText()` で呼び出します。
  Transformers.js などのブラウザ内埋め込みライブラリと組み合わせることで、
  エンジン内自動埋め込みが可能になります。

```javascript
// Precomputed embedder
schema.addEmbedder("precomputed-embedder", { type: "precomputed" });

// Callback embedder（例: Transformers.js）
schema.addEmbedder("callback-embedder", {
  type: "callback",
  embed: async (text) => {
    const output = await pipeline(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  },
});
```

#### `setDefaultFields(fields)`

デフォルト検索フィールドを設定します。

#### `setDynamicFieldPolicy(policy)`

ドキュメントに含まれるがスキーマに宣言されていないフィールドの扱いを設定します。`policy` は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"` のいずれか（大文字小文字を無視）。不正な値を渡すと例外をスローします。

- `"strict"` — ドキュメントを拒否
- `"dynamic"` — 各未宣言フィールドの型を推論してスキーマに追加。**警告**: integer フィールドに入ってきた float 値は静かに切り捨てられます（`3.14` → `3`）
- `"ignore"` — 未宣言フィールドを静かに破棄

詳細な挙動マトリクスは [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

#### `dynamicFieldPolicy()`

現在のポリシーを小文字の文字列で返します。

#### `fieldNames()`

定義済みフィールド名の配列を返します。

#### `toString()`

スキーマの文字列表現（`"Schema(fields=[...])"` 形式）を返します。

## SearchResult

```typescript
interface SearchResult {
  id: string;
  score: number;
  document: object | null;
}
```

## Analysis

### JapaneseAnalyzer

Lindera 辞書のバイト列から構築する日本語形態素解析 analyzer。
ブラウザ WASM には実ファイルシステムが無いため、標準の
`{ "language": "japanese", "dict": "/path/to/ipadic" }` プリセットは
利用できません。代わりに Lindera 辞書アーカイブ（典型的には
`lindera-ipadic-X.Y.Z.zip`）を取得して [OPFS ヘルパ](#opfs-ヘルパ) で
OPFS に保存し、9 つのコンポーネントバイト配列を
`JapaneseAnalyzer.fromBytes` に渡してください。

#### `JapaneseAnalyzer.fromBytes(metadata, dictTrie, ..., mode?)`

IPADIC のバイト列から analyzer を構築する static ファクトリ。

引数（`mode` 以外はすべて `Uint8Array`）:

| 引数 | 対応するファイル |
| ---- | ---- |
| `metadata` | `metadata.json` |
| `dictTrie` | `dict.trie`（prefix trie） |
| `dictValsIdx` | `dict.valsidx` |
| `dictVals` | `dict.vals` |
| `dictWordsIdx` | `dict.wordsidx` |
| `dictWords` | `dict.words` |
| `matrixMtx` | `matrix.mtx` |
| `charDef` | `char_def.bin` |
| `unk` | `unk.bin` |
| `mode` | `"normal"`（デフォルト）/ `"decompose"` |

いずれかのコンポーネントの deserialization に失敗した場合、または
mode 文字列が不正な場合は throw します。

```javascript
import { JapaneseAnalyzer } from "laurus-wasm";
import { loadDictionaryFiles } from "laurus-wasm/opfs";

const f = await loadDictionaryFiles("ipadic");
const ja = JapaneseAnalyzer.fromBytes(
  f.metadata, f.dictTrie, f.dictValsIdx, f.dictVals,
  f.dictWordsIdx, f.dictWords, f.matrixMtx, f.charDef, f.unk,
  "normal",
);
```

パイプラインは
`NFKC 正規化 → 日本語 iteration mark 正規化 → Lindera 形態素解析 → lowercase → 日本語 stop word フィルタ`
で、ネイティブ側の `japanese` プリセットと完全に一致します。

### OPFS ヘルパ

`laurus-wasm/opfs` サブパスは、Lindera 辞書をブラウザの Origin
Private File System にダウンロード・保存・読込するヘルパを提供します。
`JapaneseAnalyzer.fromBytes` と組み合わせて使用します。

```javascript
import {
  downloadDictionary,
  getDictionaryVersion,
  loadDictionaryFiles,
  hasDictionary,
  listDictionaries,
  removeDictionary,
} from "laurus-wasm/opfs";
```

| 関数 | 説明 |
| ---- | ---- |
| `downloadDictionary(url, name, options?)` | `.zip` を fetch し、Web の `DecompressionStream` API で展開して、Lindera 9 ファイルを OPFS の `laurus/dictionaries/<name>/` 配下に保存します。`options.onProgress({ phase, loaded?, total? })` で進捗通知を受け取れます。`options.version` を渡すとファイルと並べてバージョンスタンプを保存します（下記参照）。 |
| `getDictionaryVersion(name)` | `downloadDictionary` が保存したバージョンスタンプを返します。辞書またはスタンプが存在しない場合は `null` を返します。 |
| `loadDictionaryFiles(name)` | 9 ファイルを `{ metadata, dictTrie, dictValsIdx, dictVals, dictWordsIdx, dictWords, matrixMtx, charDef, unk }` オブジェクトとして読み出し、`JapaneseAnalyzer.fromBytes` にそのまま渡せる形にします。 |
| `hasDictionary(name)` | 辞書ディレクトリが OPFS にあれば `true`。 |
| `listDictionaries()` | 保存済み辞書名の配列を返します。 |
| `removeDictionary(name)` | 辞書ディレクトリを削除します。 |

辞書のバイナリ形式は WASM バイナリにコンパイルされた Lindera
（およびその依存 daachorse）のバージョンに紐づいており、アプリが
Lindera を更新すると OPFS にキャッシュ済みの辞書は読めなくなります
（`InvalidAutomatonError` でデシリアライズに失敗します）。ダウンロード時に
zip の対象 Lindera バージョンを `options.version` として渡し、起動時に
`getDictionaryVersion(name)` を現在のビルドが期待するバージョンと比較して、
不一致なら再ダウンロードしてください。スタンプが `null` の場合も
不一致として扱います。

ブラウザ CORS の制約により GitHub Releases から直接 fetch できないため、
zip はアプリと同一オリジンで配信してください（Laurus デモではデプロイ
時に `./dict/lindera-ipadic.zip` を WASM と同じパスに同梱します）。

### WhitespaceTokenizer

```javascript
const tokenizer = new WhitespaceTokenizer();
const tokens = tokenizer.tokenize("hello world");
// [{ text, position, startOffset, endOffset, boost, stopped, positionIncrement, positionLength }]
```

空白を境界としてテキストを分割し、`Token` オブジェクトの配列を返します。

### SynonymDictionary

```javascript
const dict = new SynonymDictionary();
dict.addSynonymGroup(["ml", "machine learning"]);
```

同義語グループの辞書。グループ内のすべての語句が互いに同義語として扱われます。

### SynonymGraphFilter

```javascript
new SynonymGraphFilter(dictionary, keepOriginal = true, boost = 1.0)
```

- `dictionary` (`SynonymDictionary`) — 同義語グループのソース。
- `keepOriginal` (boolean, デフォルト `true`) — 元のトークンを挿入された同義語と
  並べて保持します。
- `boost` (number, デフォルト `1.0`) — 挿入される同義語トークンに適用される
  スコアブースト。

```javascript
const filter = new SynonymGraphFilter(dict, true, 0.8);
const expanded = filter.apply(tokens);
```

`SynonymDictionary` の同義語でトークンを展開するトークンフィルターです。
