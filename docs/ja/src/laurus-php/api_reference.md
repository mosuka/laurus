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
| `search(mixed $query, int $limit = 10, int $offset = 0, ?array $highlight = null, ?LateInteractionRescore $rescore = null): array` | 検索クエリを実行します。`SearchResult` の配列を返します。`$rescore` は上位の結果を MultiVector フィールドに対する late interaction の MaxSim で並べ替えます（Issue #1351）。[LateInteractionRescore](#lateinteractionrescore) を参照。`$rescore` にそれ以外のオブジェクトを渡すと `\TypeError` になります。`$query` が `SearchRequest` の場合は、リクエスト自身の `$limit`/`$offset`/`$highlight`/`$rescore` を使い、他の引数は無視されます。 |
| `searchBatch(array $queries, int $limit = 10, int $offset = 0, ?array $highlight = null): array` | 独立した複数の検索を 1 回の呼び出しで実行します。各クエリは内部の tokio ランタイム上で並列に dispatch されます。`results[i]` は `queries[i]` に対応し、`SearchResult` の配列の配列を返します。入力が空の配列の場合は `[]` を返します。`$highlight` はバッチ内のすべてのクエリに同一に適用されます。`$rescore` パラメータはありません。`SearchRequest` の要素は、その要素自身の `$rescore` で再採点されます。 |
| `stats(): array` | インデックス統計（`"documentCount"`、`"vectorFields"`）を返します。 |

### `search` の query 引数

`$query` パラメータは以下のいずれかを受け付けます：

- **DSL 文字列**（例: `"title:hello"`、`"embedding:\"memory safety\""`)
- **Lexical クエリオブジェクト**（`TermQuery`、`PhraseQuery`、`BooleanQuery` など）
- **Vector クエリオブジェクト**（`VectorQuery`、`VectorTextQuery`）
- **`SearchRequest`**（完全な制御が必要な場合）

`searchBatch` の `$queries` 配列の各要素も同じ種類の値を受け付けます。DSL 文字列・クエリオブジェクト・`SearchRequest` を 1 つのバッチ内で混在させることもできます。

### ハイライト

`search`/`searchBatch` の `$highlight` パラメータ（Issue #1134）は、各ヒットの `SearchResult::getHighlights()` にフィールドごとのハイライト済みフラグメントを要求します。以下のいずれかを受け付けます。

- **フィールド名のリスト**: `["body"]`
- **連想配列**: 必須の `"fields"` キーに加えて、`HighlightConfig` の任意の設定（`max_fragments`、`fragment_size`、`tag`、`css_class`、`require_field_match`、`max_analyzed_chars`、`return_entire_field_if_no_highlight`）を指定 — 例: `["fields" => ["body"], "tag" => "em", "max_fragments" => 2]`

```php
$results = $index->search("body:rust", 10, 0, ["body"]);
$results[0]->getHighlights(); // ["body" => ["<mark>Rust</mark> is a systems programming language."]]
```

ハイライトは `search`/`searchBatch` に渡したクエリ（または `SearchRequest` の `$query`/`$lexicalQuery`、下記参照）に従い、`stored: true` のテキストフィールドのみハイライト可能です。存在しない、保存されていない、テキスト型でないフィールドは黙ってスキップされます。`$highlight` を省略すると、すべての結果のハイライトは空のままになります。同じ `highlight` 引数は `SearchRequest` のコンストラクタでも使用できます。

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
| `addTextField(string $name, bool $stored = true, bool $indexed = true, bool $termVectors = true, bool $docValues = true, ?string $analyzer = null, bool $multiValued = false, int $positionIncrementGap = 100): void` | 全文フィールド（転置インデックス、BM25）。`$docValues` は値を DocValues（ソート・ファセット・集計が読み取る列指向ストア）にもコピーするかどうかを制御します（Issue #1047）。`$stored` も `true` の場合のみ有効です。`$multiValued = true` で文字列のシーケンシャル配列を受け付けます（Issue #1175）: term クエリはいずれかの要素がタームを含めばマッチし、フレーズクエリは slop が `$positionIncrementGap`（デフォルト 100。`0` にすると要素を連結したものとして付番）に達しない限り 2 つの要素をまたぎません。値は文字列の配列として読み戻されます。`$analyzer` にはパラメータ不要の組込名（`"standard"` / `"english"` / `"keyword"` / `"simple"` / `"noop"`）、または `addAnalyzer` で登録済みの任意のカスタム名（Japanese/Lindera アナライザーなど）を指定できます。 |
| `addIntegerField(string $name, bool $stored = true, bool $indexed = true, bool $multiValued = false, bool $docValues = true): void` | 64 ビット整数フィールド。`$multiValued = true` で整数配列を受け付け（範囲クエリは "any match"）。`$docValues` は上記を参照。 |
| `addFloatField(string $name, bool $stored = true, bool $indexed = true, bool $multiValued = false, bool $docValues = true): void` | 64 ビット浮動小数点フィールド。`$multiValued = true` で浮動小数点配列を受け付け（範囲クエリは "any match"）。`$docValues` は上記を参照。 |
| `addBooleanField(string $name, bool $stored = true, bool $indexed = true, bool $multiValued = false, bool $docValues = true): void` | ブールフィールド。`$multiValued = true` で `bool` のシーケンシャル配列を受け付け（`flags:true` のような term クエリはいずれかの要素が値と等しければマッチ。値は `bool` の配列として読み戻されます）。`$docValues` は上記を参照。 |
| `addBytesField(string $name, bool $stored = true, bool $multiValued = false): void` | 生バイトフィールド。`$docValues` オプションはありません —— `Bytes` の値は設定にかかわらず DocValues に一切書き込まれないためです。`$multiValued = true` を渡すと base64 文字列のシーケンシャル配列を受け付け（Issue #1176）、単一の base64 文字列と同じ方法で要素ごとにデコードされます。`Bytes` はそもそもインデックスされないため、他の `$multiValued` オプションと異なりクエリ一致の意味論はなく、保存時の形と取り込み時の許容個数を変えるだけです。値はスカラーフィールドと同様、バイナリ文字列の配列として読み戻されます。 |
| `addGeoField(string $name, bool $stored = true, bool $indexed = true, bool $multiValued = false, bool $docValues = true): void` | 地理座標フィールド（緯度/経度）。`$multiValued = true` で `["lat" => .., "lon" => ..]` 配列の配列を受け付け（距離 / バウンディングボックスクエリはいずれかのポイントが条件を満たせばマッチ）。`$docValues` は上記を参照。 |
| `addGeo3dField(string $name, bool $stored = true, bool $indexed = true, bool $multiValued = false, bool $docValues = true): void` | 3D ECEF カルテシアン座標フィールド（x, y, z はメートル）。`$multiValued = true` で `["x" => .., "y" => .., "z" => ..]` 配列の配列を受け付け（距離 / バウンディングボックス / nearest クエリはいずれかのポイントが条件を満たせばマッチ）。詳細は [Geo3d の概念](../concepts/geo3d.md)。`$docValues` は上記を参照。 |
| `addDatetimeField(string $name, bool $stored = true, bool $indexed = true, bool $multiValued = false, bool $docValues = true): void` | UTC 日時フィールド。`$multiValued = true` で RFC 3339 文字列のシーケンシャル配列を受け付け（範囲クエリはいずれかの時刻が条件を満たせばマッチ。値は UTC に正規化した RFC 3339 文字列の配列として読み戻されます）。`$docValues` は上記を参照。 |
| `addHnswField(string $name, int $dimension, ?string $distance = "cosine", int $m = 16, int $efConstruction = 200, ?int $defaultEfSearch = null, ?string $embedder = null, ?string $quantizer = null, ?int $subvectorCount = null, ?string $rerankStorage = null, ?string $pqCodebookPath = null, float $baseWeight = 1.0): void` | HNSW 近似最近傍ベクトルフィールド。`$baseWeight` は他の vector フィールドと同時に検索されたときの相対的なスコアリング優先度（Issue #1084）。[ウェイト](../concepts/search/vector_search.md#ウェイト)を参照。 |
| `addFlatField(string $name, int $dimension, ?string $distance = "cosine", ?string $embedder = null, float $baseWeight = 1.0): void` | Flat（総当たり）ベクトルフィールド。 |
| `addIvfField(string $name, int $dimension, ?string $distance = "cosine", int $nClusters = 100, int $nProbe = 1, ?string $embedder = null, float $baseWeight = 1.0): void` | IVF 近似最近傍ベクトルフィールド。 |
| `addMultiVectorField(string $name, int $dimension, ?string $distance = null, ?string $embedder = null, ?string $storage = null): void` | 文書ごとに可変本数のトークンベクトルを保持する MultiVector フィールド。[late interaction による再採点](#lateinteractionrescore)が読み取ります（Issue #1351）。[MultiVector フィールド](../concepts/schema_and_fields.md#multivector-フィールド)を参照。`$dimension` は各トークンベクトルの長さで、正の値でなければなりません。`$distance` は `"cosine"`（デフォルト）または `"dot_product"` です。どちらもフィールドの追加時にチェックされます（`\ValueError`）。`$storage` は各トークンベクトルのディスク上の要素種別を指定します（Issue #1346）— `"f32"`（デフォルト、正確）、`"f16"`（2倍小さい）、`"int8"`（約4倍小さい）のいずれかで、こちらもフィールドの追加時にチェックされます（`\ValueError`）。`$embedder` にはトークン単位のエンベダー（`"candle_colbert"` のもの）の名前を指定し、フィールドのテキスト値と再採点のクエリテキストを埋め込みます。このフィールドはベクトル検索の対象にならず、トークンベクトルは保存されません。`getDocuments` や検索結果には含まれません。 |

**ベクトル量子化とリランクストレージ**（HNSW フィールド）:

- `quantizer` — `"scalar_8bit"`（デフォルト、4 倍圧縮）または高圧縮率の `"product_quantization"`。Product quantization では `subvectorCount`（`dimension` を割り切れる値）が必須です。
- `rerankStorage` — `"f32"` を指定すると完全精度の `*.hnsw.f32` サイドカーを書き出し、厳密な Stage-2 リランクを有効化します。省略すると int8 のみのセグメントを維持します。
- `pqCodebookPath` — 共有 PQ codebook のストレージ相対ファイル名（Issue #631）。`laurus train pq-codebook` CLI コマンドで一度だけ学習します。`$quantizer = "product_quantization"` との組み合わせでのみ意味を持ち、以後の commit は segment ごとの k-means 再学習の代わりに学習済み codebook で encode します。省略すると segment ごとの学習を維持します。

上記のどの `add*Field` メソッドも、`name` が `_`（`_id` を除く）で始まる場合は `\ValueError` を投げ、フィールドを追加しない。`fromToml` / `fromTomlFile` で読み込んだスキーマはそのようなフィールドを引き続き受け付けるため、永続化済みのスキーマも読み込めるが、そこから新しい `Index` を作成すると `\ValueError` になる。詳細は[フィールド命名規則](../laurus-cli/schema_format.md#フィールド命名規則)を参照。

### その他のメソッド

| メソッド | 説明 |
| :--- | :--- |
| `addEmbedder(string $name, array $config): void` | 名前付きエンベダー定義を登録します。`$config` は `"type"` キーを持つ連想配列で（下記参照）、スキーマ TOML 形式と同じ規則でデコードされます。型が無い・未知の型である・必須キーが無い場合は `\Exception`（`invalid embedder config: ...`）を投げます。 |
| `addAnalyzer(string $name, array $tokenizer, ?array $charFilters = null, ?array $tokenFilters = null): void` | カスタムアナライザー定義を登録します。`$tokenizer` は必須、`$charFilters`/`$tokenFilters` は連想配列の配列で省略可能です。各要素はスキーマ TOML/JSON 形式と同じ `{"type": "...", ...}` の形を使います（下記参照）。組み込みアナライザー用に予約された名前（`standard`、`keyword`、`english`、`simple`、`noop`）は `\ValueError` になり、その名前を定義したスキーマ（`fromToml` で読み込んだものなど）から新しい `Index` を作る場合も同じです。正規表現の構文誤りなどの意味的な妥当性は、このメソッド呼び出し時ではなく、スキーマから `Index` を構築する際にチェックされます。 |
| `analyzerNames(): array` | `addAnalyzer` で登録済み、または TOML から読み込んだカスタムアナライザー名の一覧を返します。 |
| `Schema::fromToml(string $tomlStr): Schema` | `laurus-cli create index --schema` と同じ形式の TOML 文字列からスキーマを読み込みます。TOML がスキーマとして正しくない場合は `ValueError` を投げます。 |
| `Schema::fromTomlFile(string $path): Schema` | TOML ファイルからスキーマを読み込みます。ファイルを読めない場合はパスで始まるメッセージの `Exception` を、内容がスキーマとして正しくない場合は `ValueError` を投げます。 |
| `toToml(): string` | このスキーマを `laurus-cli` と同じ形式の TOML 文字列にシリアライズします。テーブルはキーの昇順で出力されるため、往復させたスキーマはテキストではなく内容で比較してください。 |
| `toTomlFile(string $path): void` | このスキーマを TOML ファイルに書き込みます。既存のファイルは上書きします。書き込めない場合はパスで始まるメッセージの `Exception` を投げます。 |
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

各型の説明を含む正規のリファレンスは [スキーマフォーマットリファレンス → エンベダー](../laurus-cli/schema_format.md#エンベダー) を参照してください。

| `"type"` | 必須キー | 任意キー | Feature Flag |
| :--- | :--- | :--- | :--- |
| `"precomputed"` | -- | -- | （常に利用可能） |
| `"candle_bert"` | `"model"` | -- | `embeddings-candle` |
| `"candle_clip"` | `"model"` | -- | `embeddings-multimodal` |
| `"openai"` | `"model"` | -- | `embeddings-openai` |
| `"candle_colbert"` | `"model"` | `"revision"`、`"query_maxlen"`、`"doc_maxlen"` | `embeddings-candle` |

`"candle_colbert"` はトークンごとに 1 本のベクトルを出力するため、MultiVector フィールド（`addMultiVectorField`）専用です。

```php
$schema->addEmbedder("colbert", ["type" => "candle_colbert", "model" => "colbert-ir/colbertv2.0"]);
$schema->addMultiVectorField("body_colbert", 128, null, "colbert");
```

### アナライザーコンポーネント

`addAnalyzer(string $name, array $tokenizer, ?array $charFilters = null, ?array $tokenFilters = null)`
および `[analyzers.<name>]` TOML セクションで使用します。`$tokenizer` は単一の
連想配列、`$charFilters`/`$tokenFilters` は連想配列の配列で、配列の順序で
適用されます。

各コンポーネントの説明を含む正規のリファレンスは [スキーマフォーマットリファレンス → アナライザ](../laurus-cli/schema_format.md#アナライザ) を参照してください。

**トークナイザー**（`$tokenizer`、必ず1つ）:

| `"type"` | 必須キー | 任意キー |
| :--- | :--- | :--- |
| `"whitespace"` | -- | -- |
| `"unicode_word"` | -- | -- |
| `"regex"` | -- | `"pattern"`（デフォルト `\w+`）、`"gaps"`（デフォルト `false`） |
| `"ngram"` | `"min_gram"`, `"max_gram"` | -- |
| `"lindera"` | `"mode"`, `"dict"` | `"user_dict"` |
| `"whole"` | -- | -- |

**Char filter**（`$charFilters`、トークン化前の生テキストに適用）:

| `"type"` | 必須キー | 任意キー |
| :--- | :--- | :--- |
| `"unicode_normalization"` | `"form"`（`"nfc"`/`"nfd"`/`"nfkc"`/`"nfkd"`） | -- |
| `"pattern_replace"` | `"pattern"`, `"replacement"` | -- |
| `"mapping"` | `"mapping"`（文字列置換の連想配列） | -- |
| `"japanese_iteration_mark"` | -- | `"kanji"`（デフォルト `true`）、`"kana"`（デフォルト `true`） |

**Token filter**（`$tokenFilters`、トークン化後のトークン列に適用）:

| `"type"` | 必須キー | 任意キー |
| :--- | :--- | :--- |
| `"lowercase"` | -- | -- |
| `"stop"` | -- | `"words"`（デフォルト: 英語のストップワード） |
| `"stem"` | -- | `"stem_type"`（`"porter"`/`"simple"`/`"identity"`） |
| `"boost"` | `"boost"` | -- |
| `"limit"` | `"limit"` | -- |
| `"strip"` | -- | -- |
| `"remove_empty"` | -- | -- |
| `"flatten_graph"` | -- | -- |

```php
$schema = new Laurus\Schema();
$schema->addAnalyzer(
    "ja_ipadic",
    ["type" => "lindera", "mode" => "normal", "dict" => "/var/lib/lindera/ipadic"],
    [
        ["type" => "unicode_normalization", "form" => "nfkc"],
        ["type" => "japanese_iteration_mark"],
    ],
    [["type" => "lowercase"]],
);
$schema->addTextField("title", analyzer: "ja_ipadic");
```

### 距離メトリクス

| 値 | 説明 |
| :--- | :--- |
| `"cosine"` | コサイン類似度（デフォルト） |
| `"euclidean"` | ユークリッド距離 |
| `"dot_product"` | 内積 |
| `"manhattan"` | マンハッタン距離 |
| `"angular"` | 角度距離 |

`addMultiVectorField` が受け付けるのは `"cosine"` と `"dot_product"` だけです。

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

### DateTimeRangeQuery

```php
new \Laurus\DateTimeRangeQuery(string $field, ?string $min = null, ?string $max = null)
```

`[$min, $max]` の範囲内（両端を含む）の `DateTime` 値を検索します。開いた境界には `null` を指定します。境界は Query DSL が受け付ける任意の形式の文字列リテラルです: RFC 3339（`"2024-01-01T09:00:00+09:00"`、UTC に正規化）、オフセットなしの `"YYYY-MM-DDTHH:MM:SS[.fff]"`（UTC）、または `"YYYY-MM-DD"`（その日の 0 時 UTC）。`DateTimeInterface` は `$dt->format(DATE_RFC3339)` で渡します。ext-php-rs のコンストラクタは失敗できないため、不正な境界はクエリの使用時（`Index::search`、`BooleanQuery`、`SearchRequest`）に `\Throwable` として報告されます。

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
    ?array $highlight = null,
    ?LateInteractionRescore $rescore = null,
)
```

| パラメータ | 説明 |
| :--- | :--- |
| `$query` | DSL 文字列または単一クエリオブジェクト。DSL 文字列は `$vectorQuery` と組み合わせられます。両方で検索し、`$vectorQuery` は DSL のベクトル部分に加わります（融合は `$fusion`、デフォルトは `RRF(k: 60)`）。それ以外で `$lexicalQuery` / `$vectorQuery` と組み合わせると `\ValueError` になります（Issue #1372）。Lexical の句は DSL に書くか、クエリオブジェクトを `$lexicalQuery` / `$vectorQuery` として渡してください。 |
| `$lexicalQuery` | 明示的なハイブリッド検索の Lexical コンポーネント。 |
| `$vectorQuery` | 明示的なハイブリッド検索の Vector コンポーネント。 |
| `$filterQuery` | スコアリング後に適用する Lexical フィルター。 |
| `$fusion` | フュージョンアルゴリズム（`RRF` または `WeightedSum`）。両コンポーネント指定時のデフォルトは `RRF(k: 60)`。 |
| `$limit` | 最大結果件数（デフォルト 10）。 |
| `$offset` | ページネーションオフセット（デフォルト 0）。 |
| `$highlight` | `Index->search()` の `$highlight` と同じリストまたは連想配列の形式（Issue #1134）。[ハイライト](#ハイライト)を参照。`$limit`/`$offset` 以外は PHP レベルのデフォルトを持たないため、`$highlight` を名前付き引数で渡す場合もそれ以前の引数はすべて位置または名前で渡す必要があります。 |
| `$rescore` | 上位の結果を並べ替える `LateInteractionRescore`（Issue #1351）。それ以外のオブジェクトは `\TypeError` になります。[LateInteractionRescore](#lateinteractionrescore) を参照。`$highlight` と同じく、それ以前の引数もすべて渡す必要があります。 |

`SearchRequest` を `Index->search()` に渡すと、リクエスト自身の `$limit`・`$offset`・`$highlight`・`$rescore` が使われ、`search` の他の引数は無視されます。

---

## LateInteractionRescore

上位の検索結果を late interaction（ColBERT の MaxSim）で再採点します（Issue #1351）。`Index->search()` の 5 番目の引数、または `SearchRequest` の `$rescore` に渡します。仕組みは [Vector 検索 → Late Interaction による再採点](../concepts/search/vector_search.md#late-interaction-による再採点rescore) を参照してください。

```php
new \Laurus\LateInteractionRescore(string $field, string|array $query, ?int $windowSize = null)
```

| パラメータ | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `$field` | `string` | -- | MultiVector フィールド（`addMultiVectorField`）。 |
| `$query` | `string\|array` | -- | フィールドのトークン単位のエンベダー（`"candle_colbert"` のもの）が埋め込むクエリテキスト、またはクエリのトークンベクトルを数値リストのリストで渡したもの（例: `[[1.0, 0.0], [0.0, 1.0]]`。整数は拡張されます）。 |
| `$windowSize` | `int\|null` | `null`（100） | 再採点する 1 段目の上位結果の件数。最大 10,000。 |

### メソッド

| メソッド | 説明 |
| :--- | :--- |
| `getWindowSize(): int` | 再採点する上位結果の件数を返します（`$windowSize` を指定しなければ `100`）。 |
| `__toString(): string` | `LateInteractionRescore(field="tokens", window_size=100)` のような文字列表現を返します。 |

### 並び順とスコア

1 段目（lexical・vector・ハイブリッド）の上位 `$windowSize` 件を、フィールドに対する MaxSim で並べ替えます。再採点した結果の `getScore()` は MaxSim の値です。window の外の結果は 1 段目の順位とスコアのまま、再採点した結果の後に続きます（フィールドにトークンベクトルを持たない window 内の結果は、両者の間に 1 段目の順で並びます）。2 種類のスコアは比較できません。

### エラー

- コンストラクタは、`$query` が文字列でも数値リストのリストでもない場合に `\TypeError`（`query must be a string or a list of numeric lists`）を、トークンベクトルが `bool` や `string` など数値以外を含む場合に `\Exception`（`token vector N must hold only numbers`）を投げます。
- `Index->search()` と `SearchRequest` は、`$rescore` にそれ以外のオブジェクトを渡すと `\TypeError`（`rescore must be a Laurus\LateInteractionRescore`）を投げます。
- それ以外の値は検索時、検索を始める前にエンジンがチェックし、`rescore: ...` を含むメッセージの `\ValueError` を投げます。フィールドが存在しないか MultiVector フィールドでない場合、クエリがフィールドの次元を持つ有限値のベクトルを 1〜1,024 本含まない場合、テキストのクエリが空かフィールドにトークン単位のエンベダーがない場合、`$windowSize` が `1..=10,000` の範囲外の場合です。

### 例

```php
$schema = new Laurus\Schema();
$schema->addTextField("title");
$schema->addMultiVectorField("tokens", 2, "dot_product");

$index = new Laurus\Index(null, $schema);
$index->putDocument("a", ["title" => "rust", "tokens" => [[0.1, 0.0]]]);
$index->putDocument("b", ["title" => "rust language", "tokens" => [[0.9, 0.2]]]);
$index->commit();

$rescore = new Laurus\LateInteractionRescore("tokens", [[1.0, 0.0], [0.0, 1.0]], 50);
$results = $index->search("title:rust", 10, 0, null, $rescore);
$results[0]->getId();    // "b"
$results[0]->getScore(); // ≈ 1.1 = 0.9 + 0.2（MaxSim）

// SearchRequest で同じ再採点を行う（9 番目の引数）
$results = $index->search(new Laurus\SearchRequest("title:rust", null, null, null, null, 10, 0, null, $rescore));

// テキストのクエリには、フィールドにトークン単位のエンベダーが必要（「エンベダータイプ」を参照）
$rescore = new Laurus\LateInteractionRescore("body_colbert", "how do lifetimes work");
```

---

## SearchResult

`Index->search()` が返すクラスです。

```php
$result->getId()          // string   -- 外部ドキュメント識別子
$result->getScore()       // float    -- 関連性スコア
$result->getDocument()    // array|null -- 取得されたフィールド値。stored=false の場合は null
$result->getHighlights()  // array    -- 要求したフィールドごとのハイライト済みフラグメント
```

`getHighlights()` は `$highlight` で指定した各フィールドをハイライト済みフラグメント（最も良いものが先頭）にマッピングします。ハイライトされなかったフィールドは配列に現れず、`$highlight` を要求しなかった場合は `[]` を返します。詳細は[ハイライト](#ハイライト)を参照してください。

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
$token->getStartOffset()        // int     -- 元テキスト内の UTF-8 バイト開始オフセット
$token->getEndOffset()          // int     -- 元テキスト内の UTF-8 バイト終了オフセット
$token->getBoost()              // float   -- スコアブースト係数（1.0 = 調整なし）
$token->isStopped()             // bool    -- ストップフィルターによって除去されたかどうか
$token->getPositionIncrement()  // int     -- 前のトークンの位置との差分
$token->getPositionLength()     // int     -- このトークンがカバーする位置数
$token->getTokenType()          // ?string -- トークン種別（例: "alphanum"）
```

オフセットは PHP の文字列と同じくバイト数なので、`substr($text, $start, $end - $start)` でトークンのテキストを取り出せます。

`getTokenType()` は `"alphanum"`、`"num"`、`"cjk"`、`"katakana"`、`"hiragana"`、`"hangul"`、`"punctuation"`、`"whitespace"`、`"synonym"`、`"email"`、`"url"`、`"other"` のいずれか、または `null` を返します。`SynonymGraphFilter::apply()` は各トークンのオフセットと種別を保ち、挿入する同義語には種別 `"synonym"` と、置き換える語のオフセットを付けます。

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
| `array`（`int`、シーケンシャル） | `Int64Array` | 多値整数フィールド。ベクトルフィールドでは配列を `f32` にキャスト。空の `array` は空の `Int64Array` |
| `array`（数値、シーケンシャル） | `Float64Array` | 多値浮動小数点フィールド（整数は拡張）。ベクトルフィールドでは配列を `f32` にキャスト |
| `array`（`"lat"`, `"lon"`） | `Geo` | 2 つの `float` 値 |
| `array`（`"x"`, `"y"`, `"z"`） | `GeoEcef` | 3 つの `float` 値（メートル単位、3D ECEF 直交座標） |
| `array`（`["lat" => .., "lon" => ..]` 配列の配列） | `GeoArray` | シーケンシャル配列。フィールドに `$multiValued = true` が必要 |
| `array`（`["x" => .., "y" => .., "z" => ..]` 配列の配列） | `GeoEcefArray` | シーケンシャル配列。フィールドに `$multiValued = true` が必要 |
| `array`（数値リストのリスト、シーケンシャル。例: `[[0.1, 0.2], [0.3, 0.4]]`） | `VectorArray` | MultiVector フィールドのトークンベクトル（Issue #1351）。内側の配列はキー付きの地理座標ではなくリストなので、両者が衝突することはない。整数は拡張される。`bool` や `string` の要素はエラー（"token vector N must hold only numbers"）。ベクトルの本数と次元はフィールドに対してチェックされる（`\ValueError`）。保存されないため、`getDocuments` や検索結果には含まれない |
| `string`（ISO 8601） | `DateTime` | ISO 8601 形式からパース |
| `array`（ISO 8601 文字列、シーケンシャル） | `DateTimeArray` | 全要素が ISO 8601 としてパースできる場合のみ選ばれる。フィールドに `$multiValued = true` が必要 |
| `array`（`string`、シーケンシャル。すべてが ISO 8601 ではない） | `TextArray` | 多値テキストフィールド（Issue #1175）。文字列の配列として読み戻される。フィールドに `$multiValued = true` が必要。宣言済みの多値 `Bytes` フィールドでは、同じ base64 文字列の配列が要素ごとにデコードされる（Issue #1176） |
| `array`（`bool`、シーケンシャル） | `BoolArray` | 全要素が `bool` であること。`[true, 1]` のような混在配列はエラー（"numeric array elements must be numeric"）。フィールドに `$multiValued = true` が必要 |
