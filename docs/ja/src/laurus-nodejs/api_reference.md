# API リファレンス

## Index

主要なエントリポイント。Laurus 検索エンジンをラップします。

```typescript
class Index {
  static create(
    path?: string | null,
    schema?: Schema,
    walSyncPolicy?: WalSyncPolicy,
    commitPolicy?: CommitPolicy,
  ): Promise<Index>;
}
```

### ファクトリメソッド

| パラメータ | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `path` | `string \| null` | `null` | 永続化ストレージのディレクトリ。`null` でインメモリ。指定した場合、そのディレクトリは `laurus-cli create index`/`--index-dir` と同じ `<path>/schema.toml` + `<path>/store/` というレイアウトに従うため、ここで作成したインデックスは CLI からも開けます（逆も同様）。詳細は下記を参照。 |
| `schema` | `Schema` | 空 | スキーマ定義。新規にファイルベース（またはインメモリ）インデックスを*作成*する場合のみ意味を持ちます。既存のファイルベースインデックスを再オープンする場合は省略する必要があり、永続化済みのスキーマが代わりに読み込まれます。新規インデックスで省略した場合は空のスキーマが使用されます。 |
| `walSyncPolicy` | `WalSyncPolicy` | レコードごと | WAL の永続性ポリシー。省略するとデフォルトのレコードごとの `fsync` を使用します。[WAL 同期ポリシー / 永続性](#wal-同期ポリシー--永続性)を参照。 |
| `commitPolicy` | `CommitPolicy` | 手動 | 自動コミットポリシー。省略するとデフォルトの手動ポリシー（呼び出し側がすべての `commit()` を駆動）を使用します。[コミットポリシー / 自動コミット](#コミットポリシー--自動コミット)を参照。 |

**ファイルベースインデックスの作成 vs 再オープン**（`path` を指定した場合）: `<path>/schema.toml` がまだ存在しない場合、この呼び出しは新規インデックスを**作成**し、`schema`（省略時は空のスキーマ）をそこに永続化します。`<path>/schema.toml` が既に存在する場合、この呼び出しは既存インデックスを**再オープン**します -- `schema` は省略しなければならず、指定すると例外が投げられます（どちらのスキーマを優先すべきか曖昧になるため）。`path` がこの規約導入以前のレイアウト（`schema.toml` が無く、セグメントファイルが `path` 直下にある）のインデックスを含んでいる場合も例外になります。

### メソッド

| メソッド | 説明 |
| :--- | :--- |
| `putDocument(id, doc)` | ドキュメントを上書き保存。 |
| `addDocument(id, doc)` | 既存バージョンを残してチャンクを追記。 |
| `putDocuments(docs)` | バッチ upsert。`docs` は `Array<[id, doc]>` で、バッチごとに WAL fsync 1 回で順に適用（重複 ID はデデュープ、最後が勝ち）。最初の不正エントリで fail-fast し、適用済みの prefix はロールバックされません。 |
| `addDocuments(docs)` | バッチチャンク追記。`putDocuments` と同様ですが、繰り返した ID は別バージョンとして蓄積されます。 |
| `getDocuments(id)` | 指定 ID の全バージョンを取得。 |
| `deleteDocuments(id)` | 指定 ID の全バージョンを削除。 |
| `commit()` | 書き込みをフラッシュし変更を検索可能にする。 |
| `flushWal()` | WAL の永続性バリアを強制する。[WAL 同期ポリシー / 永続性](#wal-同期ポリシー--永続性)を参照。 |
| `search(query, limit?, offset?)` | DSL 文字列で検索。 |
| `searchTerm(field, term, limit?, offset?)` | 完全一致 Term 検索。 |
| `searchVector(field, vector, limit?, offset?)` | 事前計算ベクトルで検索。 |
| `searchVectorText(field, text, limit?, offset?)` | テキストを自動埋め込みして検索。 |
| `searchWithRequest(request)` | `SearchRequest` で検索。 |
| `searchBatch(queries, limit?, offset?)` | 複数の DSL 文字列クエリを並列実行します。`results[i]` は `queries[i]` に対応。戻り値は `Promise<Array<Array<JsSearchResult>>>`。空入力の場合は `[]` を返します。 |
| `stats()` | インデックス統計（`documentCount`、`vectorFields`）を返す。 |

ドキュメント操作と検索メソッドはすべて非同期で Promise を返します。
`stats()` は同期メソッドです。

`stats()` は次の形のオブジェクトを返します:

```typescript
interface IndexStats {
  documentCount: number;
  vectorFields: Record<string, { count: number; dimension: number }>;
}
```

### WAL 同期ポリシー / 永続性

永続インデックスでは、すべての書き込みが先行書き込みログ（WAL）に追記されます。
デフォルトでは WAL は**すべての**レコードごとに `fsync` されるため、Promise が
解決した時点で各書き込みは完全に永続化されます。`Index.create` はオプションの
`walSyncPolicy` を受け付け、永続性を一部犠牲にして書き込みスループットを
向上させることができます。また `flushWal()` で必要なときに永続性バリアを
強制できます。

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
| `WalSyncPolicy.perRecord()` | デフォルト。WAL レコードごとに `fsync` し、書き込みごとに完全に永続化します。 |
| `WalSyncPolicy.group(...)` | グループコミット。複数の書き込みにまたがって `fsync` をまとめます。 |

`group(...)` のパラメータ（引数を省略するとそのデフォルトを維持）:

| パラメータ | デフォルト | 説明 |
| :--- | :--- | :--- |
| `maxRecords` | `1024` | この件数のレコードが蓄積されたらフラッシュします。 |
| `maxBytes` | `1048576`（1 MiB） | この量の未同期バイトが蓄積されたらフラッシュします。 |
| `maxIntervalMs` | なし | 任意の定期フラッシュタイマー（ミリ秒）。省略するとタイマー無効。 |

グループコミットでは、`maxRecords` または `maxBytes` の**いずれか**に達した
時点で WAL がフラッシュされ、`commit()` 時にも必ずフラッシュされます。
クラッシュ時には最後の未同期バッチまでを失う可能性があります — これは
SQLite の `synchronous = NORMAL` と同じトレードオフです。完全な `commit()` を
行わずにこれまで書き込んだ内容をディスクへ強制するには `flushWal()` を
呼び出します。

| メソッド | 説明 |
| :--- | :--- |
| `flushWal()` | 今すぐ WAL の永続性バリアを強制します。`Promise<void>` を返します。 |

```javascript
import { Index, WalSyncPolicy } from "laurus-nodejs";

// 1 秒の定期フラッシュタイマー付きでグループコミットを有効化します。
const policy = WalSyncPolicy.group(4096, undefined, 1000);
const index = await Index.create("./myindex", schema, policy);

for (let i = 0; i < 10000; i++) {
  await index.putDocument(`doc${i}`, { title: `Document ${i}` });
}

// まだコミットせずに永続性バリアを強制します。
await index.flushWal();

await index.commit(); // WAL もフラッシュされます
```

`walSyncPolicy` を省略する（または `WalSyncPolicy.perRecord()` を渡す）と、
デフォルトの完全に永続的な動作が維持されます。

### コミットポリシー / 自動コミット

デフォルトでは呼び出し側がすべての `commit()` を駆動するため、保留中の変更は
明示的に呼び出したときにのみ検索可能になります。`Index.create` はオプションの
`commitPolicy` を受け付け、代わりに取り込み駆動のタイミングでエンジンに自動
コミットさせることができます。これにより明示的な `commit()` なしで書き込みが
実体化されます。

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
| `CommitPolicy.everyDocs(n)` | 適用されたドキュメント `n` 件ごとに自動コミットします。 |
| `CommitPolicy.intervalMs(ms)` | バックグラウンドタイマーにより少なくとも `ms` ミリ秒ごとに自動コミットします（デフォルト: なし）。**ネイティブ専用**。wasm では no-op（バックグラウンドスレッドがないため）。 |

`everyDocs(n)` では、エンジンは適用されたドキュメント `n` 件ごとにコミットし、
その件数は単発・バッチ両方の取り込みにまたがって数えられます — 1 つのバッチ
**内部**でもドキュメント `n` 件ごとにコミットされます。`everyDocs(0)` は有効で
自動コミットを無効化し、`CommitPolicy.manual()` と等価です。

`intervalMs(ms)` は `everyDocs(n)` の時間ベース版です。バックグラウンドタイマー
が少なくとも `ms` ミリ秒ごとに自動コミットするため、取り込みがアイドル状態でも
末尾の部分的なバッチが実体化されます。これは**ネイティブ専用**です — wasm には
バックグラウンドスレッドがないため、エンジンは `intervalMs` を no-op として扱い
ます（ファクトリは値を構築しますが、WebAssembly ではタイマーによるコミットは
発生しません）。

`commitPolicy` は `walSyncPolicy` と直交します。`walSyncPolicy` は WAL の
`fsync` 永続性を制御するのに対し、`commitPolicy` は保留中の変更を検索可能な
コミットへ**いつ**実体化するかを制御します。両者は自由に組み合わせられます。

```javascript
import { Index, CommitPolicy } from "laurus-nodejs";

// 適用されたドキュメント 1000 件ごとに自動コミットします。
const index = await Index.create(
  null,
  schema,
  undefined,
  CommitPolicy.everyDocs(1000),
);

for (let i = 0; i < 10000; i++) {
  await index.putDocument(`doc${i}`, { title: `Document ${i}` });
}
// エンジンはすでに 10 回コミット済みです。明示的な commit() は不要です。
```

`commitPolicy` を省略する（または `CommitPolicy.manual()` を渡す）と、
デフォルトの呼び出し側駆動のコミット動作が維持されます。

---

## Schema

`Index` のフィールドとインデックス型を定義します。

```typescript
class Schema {
  constructor();
}
```

### フィールドメソッド

| メソッド | 説明 |
| :--- | :--- |
| `addTextField(name, stored?, indexed?, termVectors?, analyzer?)` | 全文検索フィールド（転置インデックス、BM25）。`analyzer` にはパラメータ不要の組込名（`"standard"` / `"english"` / `"keyword"` / `"simple"` / `"noop"`、または `addAnalyzer` で登録したカスタム名）を指定します。Lindera 辞書パスが必要な Japanese プリセットを使う場合は、`lindera` tokenizer を含むカスタム analyzer を登録して、その名前を参照してください。 |
| `addIntegerField(name, stored?, indexed?, multiValued?)` | 64 ビット整数フィールド。`multiValued: true` で整数配列を受け付け（範囲クエリは "any match"）。 |
| `addFloatField(name, stored?, indexed?, multiValued?)` | 64 ビット浮動小数点フィールド。`multiValued: true` で浮動小数点配列を受け付け（範囲クエリは "any match"）。 |
| `addBooleanField(name, stored?, indexed?)` | 真偽値フィールド。 |
| `addBytesField(name, stored?)` | バイナリデータフィールド。 |
| `addGeoField(name, stored?, indexed?)` | 地理座標フィールド。 |
| `addGeo3dField(name, stored?, indexed?)` | 3D ECEF カルテシアン座標フィールド（x, y, z はメートル）。詳細は [Geo3d の概念](../concepts/geo3d.md)。 |
| `addDatetimeField(name, stored?, indexed?)` | UTC 日時フィールド。 |
| `addHnswField(name, dimension, distance?, m?, efConstruction?, defaultEfSearch?, embedder?, quantizer?, subvectorCount?, rerankStorage?, pqCodebookPath?, baseWeight?)` | HNSW ベクトルフィールド。`baseWeight` は他の vector フィールドと同時に検索されたときの相対的なスコアリング優先度（Issue #1084）。[ウェイト](../concepts/search/vector_search.md#ウェイト)を参照。 |
| `addFlatField(name, dimension, distance?, embedder?, baseWeight?)` | Flat（全探索）ベクトルフィールド。 |
| `addIvfField(name, dimension, distance?, nClusters?, nProbe?, embedder?, baseWeight?)` | IVF ベクトルフィールド。 |
| `addEmbedder(name, config)` | 名前付き Embedder を登録。 |
| `setDefaultFields(fields)` | デフォルト検索フィールドを設定。 |
| `setDynamicFieldPolicy(policy)` | 未宣言フィールドの扱いを設定。`policy` は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"`。詳細は下記を参照。 |
| `dynamicFieldPolicy()` | 現在のポリシーを小文字の文字列で返す。 |
| `fieldNames()` | 全フィールド名を返す。 |
| `toString()` | スキーマの文字列表現（`"Schema(fields=[...])"` 形式）を返す。 |

**ベクトル量子化とリランクストレージ**（HNSW フィールド）:

- `quantizer` — `"scalar_8bit"`（デフォルト、4 倍圧縮）または高圧縮率の `"product_quantization"`。Product quantization では `subvectorCount`（`dimension` を割り切れる値）が必須です。
- `rerankStorage` — `"f32"` を指定すると完全精度の `*.hnsw.f32` サイドカーを書き出し、厳密な Stage-2 リランクを有効化します。省略すると int8 のみのセグメントを維持します。
- `pqCodebookPath` — 共有 PQ codebook のストレージ相対ファイル名（Issue #631）。`laurus train pq-codebook` CLI コマンドで一度だけ学習します。`quantizer: "product_quantization"` との組み合わせでのみ意味を持ち、以後の commit は segment ごとの k-means 再学習の代わりに学習済み codebook で encode します。省略すると segment ごとの学習を維持します。

#### Dynamic field policy（動的フィールドポリシー）

ドキュメントに含まれるがスキーマに宣言されていないフィールドの扱いを制御します:

- `"strict"` — ドキュメントを拒否
- `"dynamic"`（デフォルト）— 各未宣言フィールドの型を推論してスキーマに追加。**警告**: integer フィールドに入ってきた float 値は静かに切り捨てられます（`3.14` → `3`）。厳密さが必要なら `"strict"` を使用してください
- `"ignore"` — 未宣言フィールドを静かに破棄

詳細な挙動マトリクスは [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

### 距離指標

| 値 | 説明 |
| :--- | :--- |
| `"cosine"` | コサイン類似度（デフォルト） |
| `"euclidean"` | ユークリッド距離 |
| `"dot_product"` | 内積 |
| `"manhattan"` | マンハッタン距離 |
| `"angular"` | 角度距離 |

---

## クエリクラス

### TermQuery

```typescript
new TermQuery(field: string, term: string)
```

指定フィールドで完全一致する Term を含むドキュメントにマッチ。

### PhraseQuery

```typescript
new PhraseQuery(field: string, terms: string[])
```

指定順序で Term を含むドキュメントにマッチ。

### FuzzyQuery

```typescript
new FuzzyQuery(field: string, term: string, maxEdits?: number)
```

最大 `maxEdits` 編集距離までの近似マッチ（デフォルト 2）。

### WildcardQuery

```typescript
new WildcardQuery(field: string, pattern: string)
```

パターンマッチ。`*` は任意の文字列、`?` は任意の1文字。

### NumericRangeQuery

```typescript
new NumericRangeQuery(
  field: string,
  min?: number | null,
  max?: number | null,
  numericType?: "integer" | "float",
)
```

`[min, max]` 範囲の数値にマッチします。`null`（または省略）で開放端。
`numericType` は内部の範囲型を選択します（`"integer"`（デフォルト）または
`"float"`）。それ以外の値は例外をスローします。

### GeoDistanceQuery

```typescript
GeoDistanceQuery.withinRadius(
  field: string, lat: number, lon: number, distanceM: number,
): GeoDistanceQuery
```

地理的距離検索（半径指定）。

### GeoBoundingBoxQuery

```typescript
GeoBoundingBoxQuery.withinBoundingBox(
  field: string,
  minLat: number, minLon: number,
  maxLat: number, maxLon: number,
): GeoBoundingBoxQuery
```

地理的バウンディングボックス検索。

### Geo3dDistanceQuery

```typescript
Geo3dDistanceQuery.withinSphere(
  field: string,
  x: number, y: number, z: number,
  distanceM: number,
): Geo3dDistanceQuery
```

3D ECEF 座標フィールドへの球距離検索。中心から `distanceM` メートル以内の `(x, y, z)`
座標を持つドキュメントを返します。ECEF の理論については
[Geo3d の概念](../concepts/geo3d.md) を参照。

### Geo3dBoundingBoxQuery

```typescript
Geo3dBoundingBoxQuery.withinBox(
  field: string,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): Geo3dBoundingBoxQuery
```

軸並行 3D 範囲（AABB）検索。

### Geo3dNearestQuery

```typescript
Geo3dNearestQuery.kNearest(
  field: string,
  x: number, y: number, z: number,
  k: number,
  initialRadiusM?: number,
  maxRadiusM?: number,
): Geo3dNearestQuery
```

3D ECEF 座標フィールドへの k 最近傍検索。`initialRadiusM` / `maxRadiusM` は
反復拡張サーチの探索コーンを調整します。

### BooleanQuery

```typescript
class BooleanQuery {
  constructor();
  // 各クエリタイプ X について（X は次のいずれか）:
  //   { Term, Phrase, Fuzzy, Wildcard, NumericRange,
  //     GeoDistance, GeoBoundingBox,
  //     Geo3dDistance, Geo3dBoundingBox, Geo3dNearest,
  //     Boolean, Span }
  mustX(query: X): void;
  shouldX(query: X): void;
  mustNotX(query: X): void;
}
```

MUST / SHOULD / MUST_NOT 句による複合ブーリアンクエリ。各句は対応するクエリ
クラスのインスタンスを引数に取ります。例:
`mustTerm(new TermQuery("body", "rust"))` や
`shouldGeo3dNearest(Geo3dNearestQuery.kNearest(...))`。

Node.js バインディングは多態 `must(query)` ではなく 36 個の per-type メソッド
（12 クエリタイプ × 3 極性）を公開しています。これは `js_name` を上書きした
クラスに対する `napi-derive` の `Either<&T, ...>` 引数バリデーションの制限を
回避するためです。

`must` 節はすべて一致する必要があり、`mustNot` 節は一致してはなりません。
`should` 節はスコアリングに寄与し、`must` 節が無い場合は少なくとも1つが
一致する必要があります。

```javascript
const bq = new BooleanQuery();
bq.mustTerm(new TermQuery("body", "programming"));
bq.mustNotTerm(new TermQuery("title", "python"));
bq.shouldFuzzy(new FuzzyQuery("body", "data", 1));
```

### SpanQuery

```typescript
SpanQuery.term(field: string, term: string): SpanQuery
SpanQuery.near(
  field: string, terms: string[],
  slop?: number, ordered?: boolean,
): SpanQuery
SpanQuery.nearSpans(
  field: string, clauses: SpanQuery[],
  slop?: number, ordered?: boolean,
): SpanQuery
SpanQuery.containing(
  field: string, big: SpanQuery, little: SpanQuery,
): SpanQuery
SpanQuery.within(
  field: string,
  include: SpanQuery, exclude: SpanQuery, distance: number,
): SpanQuery
```

位置・近接ベースのスパンクエリ。

### VectorQuery

```typescript
new VectorQuery(field: string, vector: number[])
```

事前計算済み埋め込みベクトルによる最近傍検索。

### VectorTextQuery

```typescript
new VectorTextQuery(field: string, text: string)
```

クエリ時にテキストを埋め込みに変換して検索。
インデックスに Embedder の設定が必要。

---

## SearchRequest

高度な制御のための全機能検索リクエスト。

```typescript
interface SearchRequestOptions {
  queryDsl?: string;
  limit?: number;   // デフォルト 10
  offset?: number;  // デフォルト 0
}

class SearchRequest {
  constructor(options?: SearchRequestOptions);
}
```

コンストラクタにはプリミティブな options を渡し、多態クエリ句は下記の
per-type セッターで設定します。`BooleanQuery` 同様、`napi-derive` の
`Either<&T, ...>` バリデーション制限を回避するため per-type 化されています。

### DSL とフュージョンセッター

| メソッド | 説明 |
| :--- | :--- |
| `setQueryDsl(dsl: string)` | DSL 文字列クエリを設定。 |
| `setRrfFusion(rrf: RRF)` | RRF フュージョンを使用。 |
| `setWeightedSumFusion(ws: WeightedSum)` | 加重和フュージョンを使用。 |

### ベクトルセッター

| メソッド | 説明 |
| :--- | :--- |
| `setVectorQuery(query: VectorQuery)` | 事前計算ベクトルクエリを設定。 |
| `setVectorTextQuery(query: VectorTextQuery)` | テキストベースのベクトルクエリを設定（登録 Embedder で自動埋め込み）。 |

### Lexical セッター（per-type）

`X` を `{ Term, Phrase, Fuzzy, Wildcard, NumericRange, GeoDistance,
GeoBoundingBox, Geo3dDistance, Geo3dBoundingBox, Geo3dNearest, Boolean,
Span }` の各クエリタイプとして、以下のメソッドが公開されています:

| メソッド | 説明 |
| :--- | :--- |
| `setLexicalX(query: X)` | 明示的なハイブリッドリクエストの Lexical コンポーネントを設定。 |
| `setFilterX(query: X)` | スコアリング後のフィルタコンポーネントを設定。 |

合計 24 個の per-type セッター（12 lexical + 12 filter）に加え、上記の DSL /
ベクトル / フュージョンセッターが利用可能です。

```javascript
const req = new SearchRequest({ limit: 5 });
req.setLexicalTerm(new TermQuery("title", "rust"));
req.setVectorQuery(new VectorQuery("embedding", [0.1, 0.2, 0.3, 0.4]));
req.setRrfFusion(new RRF(60.0));
const results = await index.searchWithRequest(req);
```

---

## SearchResult

検索メソッドが配列として返す結果。

```typescript
interface SearchResult {
  id: string;        // 外部ドキュメント識別子
  score: number;     // 関連度スコア
  document: object | null; // 取得フィールド、stored=false の場合は null
}
```

---

## 融合アルゴリズム

### RRF

```typescript
new RRF(k?: number)  // デフォルト 60.0
```

Reciprocal Rank Fusion。ランク位置で Lexical と Vector の
結果リストを統合。

### WeightedSum

```typescript
new WeightedSum(
  lexicalWeight?: number,  // デフォルト 0.5
  vectorWeight?: number,   // デフォルト 0.5
)
```

両スコアリストを個別に正規化し、加重和で結合。

---

## テキスト解析

### SynonymDictionary

```typescript
class SynonymDictionary {
  constructor();
  addSynonymGroup(terms: string[]): void;
}
```

### WhitespaceTokenizer

```typescript
class WhitespaceTokenizer {
  constructor();
  tokenize(text: string): Token[];
}
```

### SynonymGraphFilter

```typescript
class SynonymGraphFilter {
  constructor(
    dictionary: SynonymDictionary,
    keepOriginal?: boolean,  // デフォルト true
    boost?: number,          // デフォルト 1.0
  );
  apply(tokens: Token[]): Token[];
}
```

### Token

```typescript
interface Token {
  text: string;
  position: number;
  startOffset: number;
  endOffset: number;
  boost: number;
  stopped: boolean;
  positionIncrement: number;
  positionLength: number;
}
```

---

## フィールド値の型

JavaScript の値は自動的に Laurus の `DataValue` 型に変換されます:

| JavaScript 型 | Laurus 型 | 備考 |
| :--- | :--- | :--- |
| `null` | `Null` | |
| `boolean` | `Bool` | |
| `number`（整数） | `Int64` | |
| `number`（浮動小数点） | `Float64` | |
| `string` | `Text` | ISO 8601 文字列は `DateTime` になる |
| `number[]` | `Vector` | `f32` に変換 |
| `{ lat, lon }` | `Geo` | 2 つの `number` 値 |
| `{ x, y, z }` | `GeoEcef` | 3 つの `number` 値（メートル単位、3D ECEF 直交座標） |
