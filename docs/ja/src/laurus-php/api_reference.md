# API リファレンス

## Index

Laurus 検索エンジンをラップするメインクラスです。

```php
new \Laurus\Index(?string $path = null, ?Schema $schema = null, ?WalSyncPolicy $wal_sync_policy = null, ?CommitPolicy $commit_policy = null)
```

### コンストラクタ

| パラメータ | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `$path` | `string\|null` | `null` | 永続ストレージのディレクトリパス。`null` の場合はインメモリインデックスを作成します。指定した場合、そのディレクトリは `laurus-cli create index`/`--index-dir` と同じ `<path>/schema.toml` + `<path>/store/` というレイアウトに従うため、ここで作成したインデックスは CLI からも開けます（逆も同様）。詳細は下記を参照。 |
| `$schema` | `Schema\|null` | `null` | スキーマ定義。新規にファイルベース（またはインメモリ）インデックスを*作成*する場合のみ意味を持ちます。既存のファイルベースインデックスを再オープンする場合は省略（`null`）する必要があり、永続化済みのスキーマが代わりに読み込まれます。新規インデックスで省略した場合は空のスキーマが使用されます。 |
| `$wal_sync_policy` | `WalSyncPolicy\|null` | `null` | 先行書き込みログ（WAL）の耐久性ポリシー。`null` の場合はデフォルトのレコードごと fsync を維持します。[WAL 同期ポリシーと耐久性](#wal-同期ポリシーと耐久性) を参照。 |
| `$commit_policy` | `CommitPolicy\|null` | `null` | 自動コミットポリシー。`null` の場合はデフォルトの manual ポリシー（呼び出し側がすべての `commit()` を駆動）を維持します。[コミットポリシーと自動コミット](#コミットポリシーと自動コミット) を参照。 |

**ファイルベースインデックスの作成 vs 再オープン**（`$path` を指定した場合）: `<path>/schema.toml` がまだ存在しない場合、この呼び出しは新規インデックスを**作成**し、`$schema`（省略時は空のスキーマ）をそこに永続化します。`<path>/schema.toml` が既に存在する場合、この呼び出しは既存インデックスを**再オープン**します -- `$schema` は省略しなければならず、指定すると `ValueError` が投げられます（どちらのスキーマを優先すべきか曖昧になるため）。`$path` がこの規約導入以前のレイアウト（`schema.toml` が無く、セグメントファイルが `$path` 直下にある）のインデックスを含んでいる場合も `ValueError` になります。

### メソッド

| メソッド | 説明 |
| :--- | :--- |
| `putDocument(string $id, array $doc): void` | ドキュメントをアップサート（upsert）します。同じ ID の既存バージョンをすべて置換します。 |
| `addDocument(string $id, array $doc): void` | 既存バージョンを削除せずにドキュメントチャンクを追記します。 |
| `putDocuments(array $docs): void` | バッチ upsert。`$docs` は `[$id, $doc]` ペアの配列で、バッチごとに WAL fsync 1 回で順に適用します（重複 ID はデデュープ、最後が勝ち）。最初の不正エントリで fail-fast し、適用済みの prefix はロールバックされません。 |
| `addDocuments(array $docs): void` | バッチチャンク追記。`putDocuments` と同様ですが、繰り返した ID は別バージョンとして蓄積されます。 |
| `getDocuments(string $id): array` | 指定 ID の全保存バージョンを返します。 |
| `deleteDocuments(string $id): void` | 指定 ID の全バージョンを削除します。 |
| `commit(): void` | バッファリングされた書き込みをフラッシュし、すべての保留中の変更を検索可能にします。 |
| `flushWal(): void` | WAL の耐久バリアをオンデマンドで強制します。未同期の WAL レコードを同期的に fsync します。group-commit ポリシー下で実行する場合に有用です（下記参照）。 |
| `search(mixed $query, int $limit = 10, int $offset = 0): array` | 検索クエリを実行します。`SearchResult` の配列を返します。 |
| `searchBatch(array $queries, int $limit = 10, int $offset = 0): array` | 独立した複数の検索を 1 回の呼び出しで実行します。各クエリは内部の tokio ランタイム上で並列に dispatch されます。`results[i]` は `queries[i]` に対応し、`SearchResult` の配列の配列を返します。入力が空の配列の場合は `[]` を返します。 |
| `stats(): array` | インデックス統計（`"documentCount"`、`"vectorFields"`）を返します。 |

### `search` の query 引数

`$query` パラメータは以下のいずれかを受け付けます：

- **DSL 文字列**（例: `"title:hello"`、`"embedding:\"memory safety\""`)
- **Lexical クエリオブジェクト**（`TermQuery`、`PhraseQuery`、`BooleanQuery` など）
- **Vector クエリオブジェクト**（`VectorQuery`、`VectorTextQuery`）
- **`SearchRequest`**（完全な制御が必要な場合）

`searchBatch` の `$queries` 配列の各要素も同じ種類の値を受け付けます。DSL 文字列・クエリオブジェクト・`SearchRequest` を 1 つのバッチ内で混在させることもできます。

### WAL 同期ポリシーと耐久性

先行書き込みログ（WAL: Write-Ahead Log）は、コミット済みデータをクラッシュ
から保護します。デフォルトでは WAL は完全に耐久的で、すべてのレコードは
書き込みが返る前に `fsync` されます。**group commit（グループコミット）**
を有効にすると、`fsync` 呼び出しをまとめることで、耐久性をいくらか引き換えに
書き込みスループットを向上させられます。

#### WalSyncPolicy

`Laurus\WalSyncPolicy` は WAL のフラッシュ方法を記述するイミュータブルな
値オブジェクトです。`Index` コンストラクタの `$wal_sync_policy` 引数に渡します。

```php
// デフォルト: 書き込みごとに耐久（各レコードを個別に fsync）。
\Laurus\WalSyncPolicy::perRecord(): WalSyncPolicy

// Group commit: fsync をまとめてコストを償却。
\Laurus\WalSyncPolicy::group(
    ?int $max_records = null,      // このレコード数でフラッシュ（デフォルト 1024）
    ?int $max_bytes = null,        // このバイト数でフラッシュ（デフォルト 1 MiB）
    ?int $max_interval_ms = null,  // このミリ秒ごとに定期的にもフラッシュ
): WalSyncPolicy
```

| コンストラクタ | 説明 |
| :--- | :--- |
| `WalSyncPolicy::perRecord()` | デフォルト。すべてのレコードは書き込みが返る前に `fsync` されます。書き込みごとに完全に耐久的です。 |
| `WalSyncPolicy::group($max_records, $max_bytes, $max_interval_ms)` | `fsync` をまとめます。すべての引数が `null` の場合はデフォルト（`max_records = 1024`、`max_bytes = 1 MiB`、タイマーなし）を使用します。WAL は `$max_records` **または** `$max_bytes` のいずれかが蓄積したとき、および毎回の `commit()` 時にフラッシュされます。`$max_interval_ms` を指定すると、定期タイマーでもフラッシュします。 |

Group commit は SQLite の `synchronous = NORMAL` に相当します。クラッシュ時に
失われるのは最後の未同期バッチのレコードまでで、インデックスが破損する
ことはありません。レコードは常に `commit()` 時に耐久化されるため、成功した
`commit()` はポリシーに関わらず耐久バリアとなります。

#### フラッシュの強制

コミットの合間に耐久バリアを強制するには `flushWal()` を呼び出します。
例えば、バッチが安全に永続化されたことを通知する前などです。未同期の
レコードを同期的に `fsync` します。デフォルトのレコードごとポリシーでは
実質的に no-op です。

```php
// group commit を有効にし、必要に応じて耐久性を強制する。
$policy = \Laurus\WalSyncPolicy::group(4096, 4 * 1024 * 1024);
$index = new \Laurus\Index("./myindex", null, $policy);

$index->putDocument("doc1", ["title" => "Hello"]);
$index->flushWal(); // group バッチが満杯でなくてもレコードが永続化される
```

### コミットポリシーと自動コミット

コミットは、バッファリングされた書き込みを Lexical ストアと Vector ストアに
実体化（materialise）し、保留中の変更を検索可能にします。デフォルトでは
Laurus が自動でコミットすることはなく、呼び出し側がすべての `commit()` を
駆動します。代わりに、適用したドキュメント数が一定に達するたびに、あるいは
一定の時間間隔ごとに、エンジンに **自動コミット（auto-commit）**させることも
できます。

#### CommitPolicy

`Laurus\CommitPolicy` はエンジンがいつコミットするかを記述するイミュータブルな
値オブジェクトです。`Index` コンストラクタの `$commit_policy` 引数に渡します。

```php
// デフォルト: 自動コミットなし — 呼び出し側がすべての commit() を駆動。
\Laurus\CommitPolicy::manual(): CommitPolicy

// 適用したドキュメント N 件ごとに自動コミット。
\Laurus\CommitPolicy::everyDocs(
    int $n,   // このドキュメント数を適用するたびにコミット
): CommitPolicy

// 少なくとも N ミリ秒ごとに自動コミット（ネイティブ専用。wasm では no-op）。
\Laurus\CommitPolicy::intervalMs(
    int $ms,   // 少なくともこの間隔（ミリ秒）でコミット
): CommitPolicy
```

| コンストラクタ | 説明 |
| :--- | :--- |
| `CommitPolicy::manual()` | デフォルト。エンジンは自動でコミットせず、呼び出し側がすべての `commit()` を駆動します。 |
| `CommitPolicy::everyDocs($n)` | 適用したドキュメント `$n` 件ごとに自動コミットします。カウントは単一 ingest とバッチ ingest の両方にまたがり、バッチ **内** でも `$n` 件ごとにトリガーされます。 |
| `CommitPolicy::intervalMs($ms)` | バックグラウンドタイマーにより、少なくとも `$ms` ミリ秒ごとに自動コミットします。ingest がアイドル状態でも、末尾の部分バッチがコミットされます。`everyDocs` の時間ベース版です。デフォルト: なし。**ネイティブ専用** — wasm では no-op です（WebAssembly にはバックグラウンドスレッドがありません）。値は構築されますが、タイマーによるコミットは発生しません。 |

`CommitPolicy::everyDocs(0)` は有効で、自動コミットを無効化します。
`CommitPolicy::manual()` と等価です。

コミットポリシーは WAL 同期ポリシーと**直交（orthogonal）**しています。
`WalSyncPolicy` は耐久性のために WAL をいつ `fsync` するかを制御するのに対し、
`CommitPolicy` はストアをいつ実体化し、保留中の変更をいつ検索可能にするかを
制御します。両者は独立して設定します。

```php
// 適用したドキュメント 1000 件ごとに自動コミットし、WAL ポリシーはデフォルトを維持。
$index = new \Laurus\Index(null, $schema, null, \Laurus\CommitPolicy::everyDocs(1000));

foreach ($docs as $id => $doc) {
    $index->putDocument($id, $doc); // エンジンが 1000 件ごとに自動でコミットする
}
```

---

## Schema

`Index` のフィールドとインデックスタイプを定義します。

```php
new \Laurus\Schema()
```

### フィールドメソッド

| メソッド | 説明 |
| :--- | :--- |
| `addTextField(string $name, bool $stored = true, bool $indexed = true, bool $termVectors = false, ?string $analyzer = null): void` | 全文フィールド（転置インデックス、BM25）。`$analyzer` にはパラメータ不要の組込名（`"standard"` / `"english"` / `"keyword"` / `"simple"` / `"noop"`、または `addAnalyzer` で登録したカスタム名）を指定します。Lindera 辞書パスが必要な Japanese プリセットは、`lindera` tokenizer を含むカスタム analyzer として登録し、名前で参照してください。 |
| `addIntegerField(string $name, bool $stored = true, bool $indexed = true, bool $multiValued = false): void` | 64 ビット整数フィールド。`$multiValued = true` で整数配列を受け付け（範囲クエリは "any match"）。 |
| `addFloatField(string $name, bool $stored = true, bool $indexed = true, bool $multiValued = false): void` | 64 ビット浮動小数点フィールド。`$multiValued = true` で浮動小数点配列を受け付け（範囲クエリは "any match"）。 |
| `addBooleanField(string $name, bool $stored = true, bool $indexed = true): void` | ブールフィールド。 |
| `addBytesField(string $name, bool $stored = true): void` | 生バイトフィールド。 |
| `addGeoField(string $name, bool $stored = true, bool $indexed = true): void` | 地理座標フィールド（緯度/経度）。 |
| `addGeo3dField(string $name, bool $stored = true, bool $indexed = true): void` | 3D ECEF カルテシアン座標フィールド（x, y, z はメートル）。詳細は [Geo3d の概念](../concepts/geo3d.md)。 |
| `addDatetimeField(string $name, bool $stored = true, bool $indexed = true): void` | UTC 日時フィールド。 |
| `addHnswField(string $name, int $dimension, ?string $distance = "cosine", int $m = 16, int $efConstruction = 200, ?int $defaultEfSearch = null, ?string $embedder = null, ?string $quantizer = null, ?int $subvectorCount = null, ?string $rerankStorage = null, ?string $pqCodebookPath = null, float $baseWeight = 1.0): void` | HNSW 近似最近傍ベクトルフィールド。`$baseWeight` は他の vector フィールドと同時に検索されたときの相対的なスコアリング優先度（Issue #1084）。[ウェイト](../concepts/search/vector_search.md#ウェイト)を参照。 |
| `addFlatField(string $name, int $dimension, ?string $distance = "cosine", ?string $embedder = null, float $baseWeight = 1.0): void` | Flat（総当たり）ベクトルフィールド。 |
| `addIvfField(string $name, int $dimension, ?string $distance = "cosine", int $nClusters = 100, int $nProbe = 1, ?string $embedder = null, float $baseWeight = 1.0): void` | IVF 近似最近傍ベクトルフィールド。 |

**ベクトル量子化とリランクストレージ**（HNSW フィールド）:

- `quantizer` — `"scalar_8bit"`（デフォルト、4 倍圧縮）または高圧縮率の `"product_quantization"`。Product quantization では `subvectorCount`（`dimension` を割り切れる値）が必須です。
- `rerankStorage` — `"f32"` を指定すると完全精度の `*.hnsw.f32` サイドカーを書き出し、厳密な Stage-2 リランクを有効化します。省略すると int8 のみのセグメントを維持します。
- `pqCodebookPath` — 共有 PQ codebook のストレージ相対ファイル名（Issue #631）。`laurus train pq-codebook` CLI コマンドで一度だけ学習します。`$quantizer = "product_quantization"` との組み合わせでのみ意味を持ち、以後の commit は segment ごとの k-means 再学習の代わりに学習済み codebook で encode します。省略すると segment ごとの学習を維持します。

### その他のメソッド

| メソッド | 説明 |
| :--- | :--- |
| `addEmbedder(string $name, array $config): void` | 名前付きエンベダー定義を登録します。`$config` は `"type"` キーを持つ連想配列です（下記参照）。 |
| `setDefaultFields(array $fieldNames): void` | クエリでフィールドが指定されていない場合に使用するデフォルトフィールドを設定します。`$fieldNames` は文字列の配列です。 |
| `setDynamicFieldPolicy(string $policy): void` | 未宣言フィールドの扱いを設定します。`$policy` は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"`。詳細は下記を参照。 |
| `dynamicFieldPolicy(): string` | 現在のポリシーを小文字の文字列で返します。 |
| `fieldNames(): array` | このスキーマに定義されたフィールド名のリストを返します。 |

#### Dynamic field policy（動的フィールドポリシー）

ドキュメントに含まれるがスキーマに宣言されていないフィールドの扱いを制御します:

- `"strict"` — ドキュメントを拒否
- `"dynamic"`（デフォルト）— 各未宣言フィールドの型を推論してスキーマに追加。**警告**: integer フィールドに入ってきた float 値は静かに切り捨てられます（`3.14` → `3`）。厳密さが必要なら `"strict"` を使用してください
- `"ignore"` — 未宣言フィールドを静かに破棄

詳細な挙動マトリクスは [スキーマとフィールド](../concepts/schema_and_fields.md#動的スキーマ) を参照してください。

### エンベダータイプ

| `"type"` | 必須キー | Feature Flag |
| :--- | :--- | :--- |
| `"precomputed"` | -- | （常に利用可能） |
| `"candle_bert"` | `"model"` | `embeddings-candle` |
| `"candle_clip"` | `"model"` | `embeddings-multimodal` |
| `"openai"` | `"model"` | `embeddings-openai` |

### 距離メトリクス

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

```php
new \Laurus\TermQuery(string $field, string $term)
```

指定フィールドに完全一致する語句を含むドキュメントを検索します。

### PhraseQuery

```php
new \Laurus\PhraseQuery(string $field, array $terms)
```

指定した語句が順序どおりに含まれるドキュメントを検索します。`$terms` は文字列の配列です。

### FuzzyQuery

```php
new \Laurus\FuzzyQuery(string $field, string $term, int $maxEdits = 2)
```

編集距離が `$maxEdits` 以内の近似一致を検索します。

### WildcardQuery

```php
new \Laurus\WildcardQuery(string $field, string $pattern)
```

ワイルドカードパターン検索。`*` は任意の文字列、`?` は任意の1文字に一致します。

### NumericRangeQuery

```php
new \Laurus\NumericRangeQuery(string $field, mixed $min, mixed $max, ?string $numericType = "integer")
```

`[$min, $max]` の範囲内の数値を検索します。開いた境界には `null` を指定します。`$numericType` には `"integer"` または `"float"` を設定します。

### GeoDistanceQuery

```php
\Laurus\GeoDistanceQuery::withinRadius(
    string $field, float $lat, float $lon, float $distanceM,
): GeoDistanceQuery
```

地理的距離検索（半径指定）。指定した地点から `$distanceM` メートル以内の
`(lat, lon)` 座標を持つドキュメントを返します。

### GeoBoundingBoxQuery

```php
\Laurus\GeoBoundingBoxQuery::withinBoundingBox(
    string $field,
    float $minLat, float $minLon,
    float $maxLat, float $maxLon,
): GeoBoundingBoxQuery
```

地理的範囲（バウンディングボックス）検索。軸並行 `[$minLat, $maxLat] ×
[$minLon, $maxLon]` 内の `(lat, lon)` 座標を持つドキュメントを返します。

### Geo3dDistanceQuery

```php
\Laurus\Geo3dDistanceQuery::withinSphere(
    string $field,
    float $x, float $y, float $z,
    float $distanceM,
): Geo3dDistanceQuery
```

3D ECEF 座標フィールドへの球距離検索。中心 `(x, y, z)` から `$distanceM` メートル以内
の座標を持つドキュメントを返します。ECEF の理論については
[Geo3d の概念](../concepts/geo3d.md) を参照。

### Geo3dBoundingBoxQuery

```php
\Laurus\Geo3dBoundingBoxQuery::withinBox(
    string $field,
    float $minX, float $minY, float $minZ,
    float $maxX, float $maxY, float $maxZ,
): Geo3dBoundingBoxQuery
```

軸並行 3D 範囲（AABB）検索。

### Geo3dNearestQuery

```php
\Laurus\Geo3dNearestQuery::kNearest(
    string $field,
    float $x, float $y, float $z,
    int $k,
    ?float $initialRadiusM = null,
    ?float $maxRadiusM = null,
): Geo3dNearestQuery
```

3D ECEF 座標フィールドへの k 最近傍検索。`$initialRadiusM` / `$maxRadiusM`
（オプション）で反復拡張サーチの探索コーンを調整できます。

### BooleanQuery

```php
$bq = new \Laurus\BooleanQuery();
$bq->must($query);
$bq->should($query);
$bq->mustNot($query);
```

複合ブールクエリ。`must` 節はすべて一致する必要があり、`mustNot` 節は一致してはなりません。`should` 節はスコアリングに寄与し、`must` 節が無い場合は少なくとも1つが一致する必要があります。

### SpanQuery

```php
// 単一語句
\Laurus\SpanQuery::term(string $field, string $term): SpanQuery

// Near: slop 位置以内の語句
\Laurus\SpanQuery::near(string $field, array $terms, int $slop = 0, bool $ordered = true): SpanQuery

// NearSpans: slop 位置以内のネストされた SpanQuery 句
\Laurus\SpanQuery::nearSpans(string $field, array $clauses, int $slop = 0, bool $ordered = true): SpanQuery

// Containing: big スパンが little スパンを含む
\Laurus\SpanQuery::containing(string $field, SpanQuery $big, SpanQuery $little): SpanQuery

// Within: 最大距離での include スパンと exclude スパン
\Laurus\SpanQuery::within(string $field, SpanQuery $include, SpanQuery $exclude, int $distance): SpanQuery
```

位置・近接スパンクエリ。`near` は語句文字列の配列を受け取り、`nearSpans` は
ネスト式のために `SpanQuery` オブジェクトの配列を受け取ります（各句のフィールド
は外側の `$field` に再ルートされます）。

### VectorQuery

```php
new \Laurus\VectorQuery(string $field, array $vector)
```

事前計算済みエンベディングベクトルを使った近似最近傍検索を行います。`$vector` は Float の配列です。

### VectorTextQuery

```php
new \Laurus\VectorTextQuery(string $field, string $text)
```

クエリ時に `$text` をエンベディングに変換してベクトル検索を行います。インデックスにエンベダーの設定が必要です。

---

## SearchRequest

高度な制御が必要な場合の完全なリクエストクラスです。

```php
new \Laurus\SearchRequest(
    mixed $query = null,
    mixed $lexicalQuery = null,
    mixed $vectorQuery = null,
    mixed $filterQuery = null,
    mixed $fusion = null,
    int $limit = 10,
    int $offset = 0,
)
```

| パラメータ | 説明 |
| :--- | :--- |
| `$query` | DSL 文字列または単一クエリオブジェクト。`$lexicalQuery` / `$vectorQuery` と排他的。 |
| `$lexicalQuery` | 明示的なハイブリッド検索の Lexical コンポーネント。 |
| `$vectorQuery` | 明示的なハイブリッド検索の Vector コンポーネント。 |
| `$filterQuery` | スコアリング後に適用する Lexical フィルター。 |
| `$fusion` | フュージョンアルゴリズム（`RRF` または `WeightedSum`）。両コンポーネント指定時のデフォルトは `RRF(k: 60)`。 |
| `$limit` | 最大結果件数（デフォルト 10）。 |
| `$offset` | ページネーションオフセット（デフォルト 0）。 |

---

## SearchResult

`Index->search()` が返すクラスです。

```php
$result->getId()        // string   -- 外部ドキュメント識別子
$result->getScore()     // float    -- 関連性スコア
$result->getDocument()  // array|null -- 取得されたフィールド値。stored=false の場合は null
```

---

## フュージョンアルゴリズム

### RRF

```php
new \Laurus\RRF(float $k = 60.0)
```

逆順位フュージョン（Reciprocal Rank Fusion）。Lexical と Vector の結果リストを順位位置によってマージします。`$k` は平滑化定数で、値が大きいほど上位ランクの影響が小さくなります。

### WeightedSum

```php
new \Laurus\WeightedSum(float $lexicalWeight = 0.5, float $vectorWeight = 0.5)
```

両スコアリストをそれぞれ正規化した後、`$lexicalWeight * lexical_score + $vectorWeight * vector_score` として結合します。

---

## テキスト解析

### SynonymDictionary

```php
$dict = new \Laurus\SynonymDictionary();
$dict->addSynonymGroup(["fast", "quick", "rapid"]);
```

同義語グループの辞書です。グループ内のすべての語句は互いの同義語として扱われます。

### WhitespaceTokenizer

```php
$tokenizer = new \Laurus\WhitespaceTokenizer();
$tokens = $tokenizer->tokenize("hello world");
```

空白で分割してテキストをトークン化し、`Token` オブジェクトの配列を返します。

### SynonymGraphFilter

```php
new \Laurus\SynonymGraphFilter(SynonymDictionary $dictionary, bool $keepOriginal = true, float $boost = 1.0)
```

| パラメータ | 説明 |
| :--- | :--- |
| `$dictionary` | 同義語グループのソース。 |
| `$keepOriginal` | `true`（デフォルト）の場合は元のトークンも同義語と並べて保持します。 |
| `$boost` | 挿入される同義語トークンに適用されるスコアブースト（デフォルト `1.0`）。 |

```php
$filter = new \Laurus\SynonymGraphFilter($dictionary, true, 1.0);
$expanded = $filter->apply($tokens);
```

`SynonymDictionary` の同義語でトークンを展開するトークンフィルターです。

### Token

```php
$token->getText()               // string  -- トークンテキスト
$token->getPosition()           // int     -- トークンストリーム内の位置
$token->getStartOffset()        // int     -- 元テキスト内の文字開始オフセット
$token->getEndOffset()          // int     -- 元テキスト内の文字終了オフセット
$token->getBoost()              // float   -- スコアブースト係数（1.0 = 調整なし）
$token->isStopped()             // bool    -- ストップフィルターによって除去されたかどうか
$token->getPositionIncrement()  // int     -- 前のトークンの位置との差分
$token->getPositionLength()     // int     -- このトークンがカバーする位置数
```

---

## フィールド値の型マッピング

PHP の値は自動的に Laurus の `DataValue` 型に変換されます：

| PHP 型 | Laurus 型 | 備考 |
| :--- | :--- | :--- |
| `null` | `Null` | |
| `true` / `false` | `Bool` | |
| `int` | `Int64` | |
| `float` | `Float64` | |
| `string` | `Text` | |
| `array`（数値） | `Vector` | 要素は `f32` に変換 |
| `array`（`"lat"`, `"lon"`） | `Geo` | 2 つの `float` 値 |
| `array`（`"x"`, `"y"`, `"z"`） | `GeoEcef` | 3 つの `float` 値（メートル単位、3D ECEF 直交座標） |
| `string`（ISO 8601） | `DateTime` | ISO 8601 形式からパース |
