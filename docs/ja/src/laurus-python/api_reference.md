# API リファレンス

## Index

Laurus 検索エンジンをラップするメインクラスです。

```python
class Index:
    def __init__(
        self,
        path: str | None = None,
        schema: Schema | None = None,
        wal_sync_policy: WalSyncPolicy | None = None,
        commit_policy: CommitPolicy | None = None,
    ) -> None: ...
```

### コンストラクタ

| パラメータ | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- |
| `path` | `str \| None` | `None` | 永続ストレージのディレクトリパス。`None` の場合はインメモリインデックスを作成します。指定した場合、そのディレクトリは `laurus-cli create index`/`--index-dir` と同じ `<path>/schema.toml` + `<path>/store/` というレイアウトに従うため、ここで作成したインデックスは CLI からも開けます（逆も同様）。詳細は下記を参照してください。 |
| `schema` | `Schema \| None` | `None` | スキーマ定義。新規にファイルベース（またはインメモリ）インデックスを*作成*する場合のみ意味を持ちます。既存のファイルベースインデックスを再オープンする場合は省略（`None`）する必要があり、永続化済みのスキーマが代わりに読み込まれます。新規インデックスで省略した場合は空のスキーマが使用されます。 |
| `wal_sync_policy` | `WalSyncPolicy \| None` | `None` | WAL の永続性ポリシー。`None` の場合はデフォルトのレコードごとの `fsync` を使用します。[WAL 同期ポリシー / 永続性](#wal-同期ポリシー--永続性)を参照してください。 |
| `commit_policy` | `CommitPolicy \| None` | `None` | 自動コミットポリシー。`None` の場合はデフォルトの手動コミット（自動コミットなし）を使用します。[コミットポリシー / 自動コミット](#コミットポリシー--自動コミット)を参照してください。 |

**ファイルベースインデックスの作成 vs 再オープン**（`path` を指定した場合）: `<path>/schema.toml` がまだ存在しない場合、この呼び出しは新規インデックスを**作成**し、`schema`（省略時は空のスキーマ）をそこに永続化します。`<path>/schema.toml` が既に存在する場合、この呼び出しは既存インデックスを**再オープン**します -- `schema` は省略しなければならず、指定すると `ValueError` になります（どちらのスキーマを優先すべきか曖昧になるため）。`path` がこの規約導入以前のレイアウト（`schema.toml` が無く、セグメントファイルが `path` 直下にある）のインデックスを含んでいる場合も `ValueError` になります。

### メソッド

| メソッド | 説明 |
| :--- | :--- |
| `put_document(id, doc)` | ドキュメントをアップサート（upsert）します。同じ ID の既存バージョンをすべて置換します。 |
| `add_document(id, doc)` | 既存バージョンを削除せずにドキュメントチャンクを追記します。 |
| `put_documents(docs)` | バッチ upsert。`docs` は `(id, dict)` ペアのイテラブルで、バッチごとに WAL fsync 1 回で順に適用します（重複 ID はデデュープ、最後が勝ち）。最初の不正エントリで fail-fast し、適用済みの prefix はロールバックされません。 |
| `add_documents(docs)` | バッチチャンク追記。`put_documents` と同様ですが、繰り返した ID は別バージョンとして蓄積されます。 |
| `get_documents(id) -> list[dict]` | 指定 ID の全保存バージョンを返します。 |
| `delete_documents(id)` | 指定 ID の全バージョンを削除します。 |
| `commit()` | バッファリングされた書き込みをフラッシュし、すべての保留中の変更を検索可能にします。 |
| `flush_wal()` | WAL の永続性バリアを強制します。[WAL 同期ポリシー / 永続性](#wal-同期ポリシー--永続性)を参照してください。 |
| `search(query, *, limit=10, offset=0) -> list[SearchResult]` | 検索クエリを実行します。 |
| `search_batch(queries, *, limit=10, offset=0) -> list[list[SearchResult]]` | 独立した複数の検索を 1 回の呼び出しで実行します。各クエリは内部の tokio ランタイム上で並列に dispatch されます。`results[i]` は `queries[i]` に対応し、入力が空のリストの場合は `[]` を返します。 |
| `stats() -> dict` | インデックス統計（`document_count`、`vector_fields`）を返します。 |

### `search` の query 引数

`query` パラメータは以下のいずれかを受け付けます：

- **DSL 文字列**（例: `"title:hello"`、`"content:\"memory safety\""`)
- **Lexical クエリオブジェクト**（`TermQuery`、`PhraseQuery`、`BooleanQuery` など）
- **Vector クエリオブジェクト**（`VectorQuery`、`VectorTextQuery`）
- **`SearchRequest`**（完全な制御が必要な場合）

`search_batch` の `queries` リストの各要素も同じ種類の値を受け付けます。DSL 文字列・クエリオブジェクト・`SearchRequest` を 1 つのバッチ内で混在させることもできます。

### WAL 同期ポリシー / 永続性

永続インデックスでは、すべての書き込みが先行書き込みログ（WAL）に追記されます。
デフォルトでは WAL は**すべての**レコードごとに `fsync` されるため、呼び出しが
返った時点で各書き込みは完全に永続化されます。コンストラクタはオプションの
`wal_sync_policy` を受け付け、永続性を一部犠牲にして書き込みスループットを
向上させることができます。また `flush_wal()` で必要なときに永続性バリアを
強制できます。

```python
class WalSyncPolicy:
    @staticmethod
    def per_record() -> WalSyncPolicy: ...
    @staticmethod
    def group(
        max_records: int | None = None,
        max_bytes: int | None = None,
        max_interval_ms: int | None = None,
    ) -> WalSyncPolicy: ...
```

| コンストラクタ | 説明 |
| :--- | :--- |
| `WalSyncPolicy.per_record()` | デフォルト。WAL レコードごとに `fsync` し、書き込みごとに完全に永続化します。 |
| `WalSyncPolicy.group(...)` | グループコミット。複数の書き込みにまたがって `fsync` をまとめます。 |

`group()` のパラメータ（いずれもキーワード指定可。`None` はデフォルトを維持）:

| パラメータ | デフォルト | 説明 |
| :--- | :--- | :--- |
| `max_records` | `1024` | この件数のレコードが蓄積されたらフラッシュします。 |
| `max_bytes` | `1048576`（1 MiB） | この量の未同期バイトが蓄積されたらフラッシュします。 |
| `max_interval_ms` | `None` | 任意の定期フラッシュタイマー（ミリ秒）。`None` でタイマー無効。 |

グループコミットでは、`max_records` または `max_bytes` の**いずれか**に達した
時点で WAL がフラッシュされ、`commit()` 時にも必ずフラッシュされます。
クラッシュ時には最後の未同期バッチまでを失う可能性があります — これは
SQLite の `synchronous = NORMAL` と同じトレードオフです。完全な `commit()` を
行わずにこれまで書き込んだ内容をディスクへ強制するには `flush_wal()` を
呼び出します。

| メソッド | 説明 |
| :--- | :--- |
| `flush_wal()` | 今すぐ WAL の永続性バリアを強制します。同期メソッドで `None` を返します。 |

```python
import laurus

# 1 秒の定期フラッシュタイマー付きでグループコミットを有効化します。
policy = laurus.WalSyncPolicy.group(max_records=4096, max_interval_ms=1000)
index = laurus.Index(path="./myindex", wal_sync_policy=policy)

for i in range(10_000):
    index.put_document(f"doc{i}", {"title": f"Document {i}"})

# まだコミットせずに永続性バリアを強制します。
index.flush_wal()

index.commit()  # WAL もフラッシュされます
```

`wal_sync_policy` を省略する（または `WalSyncPolicy.per_record()` を渡す）と、
デフォルトの完全に永続的な動作が維持されます。

### コミットポリシー / 自動コミット

コミットはバッファリングされた書き込みをストアへ materialize し、すべての
保留中の変更を検索可能にします。デフォルトでは `Index` は自動コミットを
行わないため、呼び出し側がすべての `commit()` を明示的に実行します。
コンストラクタはオプションの `commit_policy` を受け付け、一定件数の
ドキュメントを適用するごと、または一定時間ごとにエンジンが自動的に
コミットするようにできます。

```python
class CommitPolicy:
    @staticmethod
    def manual() -> CommitPolicy: ...
    @staticmethod
    def every_docs(n: int) -> CommitPolicy: ...
    @staticmethod
    def interval_ms(ms: int) -> CommitPolicy: ...
```

| コンストラクタ | 説明 |
| :--- | :--- |
| `CommitPolicy.manual()` | デフォルト。自動コミットなし。呼び出し側がすべての `commit()` を実行します。 |
| `CommitPolicy.every_docs(n)` | `n` 件のドキュメントを適用するごとに自動コミットします。 |
| `CommitPolicy.interval_ms(ms)` | バックグラウンドタイマーにより、少なくとも `ms` ミリ秒ごとに自動コミットします（デフォルト: なし）。ネイティブ専用で、wasm では no-op です。 |

`every_docs(n)` は、単体（`put_document`、`add_document`）とバッチ
（`put_documents`、`add_documents`）の両方の取り込みにまたがって適用済み
ドキュメントを数え、`n` 件ごとに自動コミットします — 1 つのバッチの
**内部**でも同様です。`every_docs(0)` も有効で、自動コミットを無効化するため
`manual()` と等価になります。

`interval_ms(ms)` は `every_docs(n)` の時間ベースの対応版です。
バックグラウンドタイマーが少なくとも `ms` ミリ秒ごとにコミットするため、
取り込みがアイドル状態でも末尾の部分バッチがコミットされます。この
ファクトリは **ネイティブ専用** です — wasm にはバックグラウンドスレッドが
存在しないため、エンジンはこれを **no-op** として扱います。値の構築は
できますが、WebAssembly 上ではタイマーによるコミットは発生しません。

`commit_policy` は `wal_sync_policy` と直交します。`wal_sync_policy` は WAL の
`fsync` 永続性を制御するのに対し、`commit_policy` はストアが保留中の変更を
検索可能な状態へ materialize するタイミングを制御します。一方を設定しても
他方には影響しません。

```python
import laurus

# 100 件のドキュメントを適用するごとに自動コミットします。
index = laurus.Index(
    path="./myindex",
    commit_policy=laurus.CommitPolicy.every_docs(100),
)

for i in range(1_000):
    index.put_document(f"doc{i}", {"title": f"Document {i}"})
# エンジンはすでに 10 回コミットしています（100 件ごとに 1 回）。
```

`commit_policy` を省略する（または `CommitPolicy.manual()` を渡す）と、
デフォルトの手動コミットの動作が維持されます。

---

## Schema

`Index` のフィールドとインデックスタイプを定義します。

```python
class Schema:
    def __init__(self) -> None: ...
```

### フィールドメソッド

| メソッド | 説明 |
| :--- | :--- |
| `add_text_field(name, *, stored=True, indexed=True, term_vectors=True, analyzer=None)` | 全文フィールド（転置インデックス、BM25）。`term_vectors` はタームの位置を保存するかどうかを制御し、フレーズクエリ・スパンクエリが読み取ります。`analyzer` には組込名（`"standard"` / `"english"` / `"keyword"` / `"simple"` / `"noop"`、または `add_analyzer` で登録したカスタム名）か、`{"language": "japanese", "mode": "normal", "dict": "/var/lib/lindera/ipadic"}` のようなパラメータ付きプリセットの dict を渡せます。文字列単独の `"japanese"` は Lindera 辞書パスが必須なため拒否されます。 |
| `add_integer_field(name, *, stored=True, indexed=True, multi_valued=False)` | 64 ビット整数フィールド。`multi_valued=True` で整数配列を受け付け（範囲クエリは "any match"）。 |
| `add_float_field(name, *, stored=True, indexed=True, multi_valued=False)` | 64 ビット浮動小数点フィールド。`multi_valued=True` で浮動小数点配列を受け付け（範囲クエリは "any match"）。 |
| `add_boolean_field(name, *, stored=True, indexed=True)` | ブールフィールド。 |
| `add_bytes_field(name, *, stored=True)` | 生バイトフィールド。 |
| `add_geo_field(name, *, stored=True, indexed=True)` | 地理座標フィールド（緯度/経度）。 |
| `add_geo3d_field(name, *, stored=True, indexed=True)` | 3D ECEF カルテシアン座標フィールド（x, y, z はメートル）。詳細は [Geo3d の概念](../concepts/geo3d.md)。 |
| `add_datetime_field(name, *, stored=True, indexed=True)` | UTC 日時フィールド。 |
| `add_hnsw_field(name, dimension, *, distance="cosine", m=16, ef_construction=200, quantizer=None, subvector_count=None, rerank_storage=None, embedder=None, pq_codebook_path=None, base_weight=1.0)` | HNSW 近似最近傍ベクトルフィールド。`base_weight` は他の vector フィールドと同時に検索されたときの相対的なスコアリング優先度（Issue #1084）。[ウェイト](../concepts/search/vector_search.md#ウェイト)を参照。 |
| `add_flat_field(name, dimension, *, distance="cosine", embedder=None, base_weight=1.0)` | Flat（総当たり）ベクトルフィールド。 |
| `add_ivf_field(name, dimension, *, distance="cosine", n_clusters=100, n_probe=1, embedder=None, base_weight=1.0)` | IVF 近似最近傍ベクトルフィールド。 |

**ベクトル量子化とリランクストレージ**（HNSW フィールド）:

- `quantizer` — `"scalar_8bit"`（デフォルト、4 倍圧縮）または高圧縮率の `"product_quantization"`。Product quantization では `subvector_count`（`dimension` を割り切れる値）が必須です。
- `rerank_storage` — `"f32"` を指定すると完全精度の `*.hnsw.f32` サイドカーを書き出し、厳密な Stage-2 リランクを有効化します。省略すると int8 のみのセグメントを維持します。
- `pq_codebook_path` — 共有 PQ codebook のストレージ相対ファイル名（Issue #631）。`laurus train pq-codebook` CLI コマンドで一度だけ学習します。`quantizer="product_quantization"` との組み合わせでのみ意味を持ち、以後の commit は segment ごとの k-means 再学習の代わりに学習済み codebook で encode します。省略すると segment ごとの学習を維持します。

### その他のメソッド

| メソッド | 説明 |
| :--- | :--- |
| `add_embedder(name, config)` | 名前付きエンベダー定義を登録します。`config` は `"type"` キーを持つ辞書です（下記参照）。 |
| `set_default_fields(fields)` | デフォルト検索フィールドを設定（文字列のリスト）。 |
| `set_dynamic_field_policy(policy)` | 未宣言フィールドの扱いを設定。`policy` は `"strict"` / `"dynamic"`（デフォルト）/ `"ignore"`。詳細は下記を参照。 |
| `dynamic_field_policy()` | 現在のポリシーを小文字の文字列で返す。 |
| `field_names()` | 全フィールド名を返す。 |

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

```python
TermQuery(field: str, term: str)
```

指定フィールドに完全一致する語句を含むドキュメントを検索します。

### PhraseQuery

```python
PhraseQuery(field: str, terms: list[str])
```

指定した語句が順序どおりに含まれるドキュメントを検索します。

### FuzzyQuery

```python
FuzzyQuery(field: str, term: str, *, max_edits: int = 2)
```

編集距離が `max_edits` 以内の近似一致を検索します。`max_edits` はキーワード専用引数です。

### WildcardQuery

```python
WildcardQuery(field: str, pattern: str)
```

ワイルドカードパターン検索。`*` は任意の文字列、`?` は任意の1文字に一致します。

### NumericRangeQuery

```python
NumericRangeQuery(field: str, *, min: int | float | None = None, max: int | float | None = None)
```

`[min, max]` の範囲内の数値を検索します。開いた境界には `None` を指定する
（または省略する）と開放されます。`min` と `max` はキーワード専用引数です。
数値型（整数または浮動小数点）は `min`/`max` の Python 型から推論されます。

### GeoDistanceQuery

```python
GeoDistanceQuery.within_radius(
    field: str, lat: float, lon: float, distance_m: float,
)
```

地理的距離検索（半径指定）。指定した地点から `distance_m` メートル以内の
`(lat, lon)` 座標を持つドキュメントを返します。

### GeoBoundingBoxQuery

```python
GeoBoundingBoxQuery.within_bounding_box(
    field: str,
    min_lat: float, min_lon: float,
    max_lat: float, max_lon: float,
)
```

地理的範囲（バウンディングボックス）検索。軸並行 `[min_lat, max_lat] ×
[min_lon, max_lon]` 内の `(lat, lon)` 座標を持つドキュメントを返します。

### Geo3dDistanceQuery

```python
Geo3dDistanceQuery.within_sphere(
    field: str, x: float, y: float, z: float, distance_m: float,
)
```

3D ECEF 座標フィールドへの球距離検索。中心 `(x, y, z)` から `distance_m` メートル以内
の座標を持つドキュメントを返します。ECEF の理論については
[Geo3d の概念](../concepts/geo3d.md) を参照。

### Geo3dBoundingBoxQuery

```python
Geo3dBoundingBoxQuery.within_box(
    field: str,
    min_x: float, min_y: float, min_z: float,
    max_x: float, max_y: float, max_z: float,
)
```

軸並行 3D 範囲（AABB）検索。`[min_x, max_x] × [min_y, max_y] × [min_z, max_z]` 内
にある ECEF 座標を持つドキュメントを返します。

### Geo3dNearestQuery

```python
Geo3dNearestQuery.k_nearest(
    field: str,
    x: float, y: float, z: float,
    k: int,
    *,
    initial_radius_m: float | None = None,
    max_radius_m: float | None = None,
)
```

3D ECEF 座標フィールドへの k 最近傍検索。`(x, y, z)` から最も近い `k` 件のドキュ
メントを返します。`initial_radius_m` / `max_radius_m` は反復拡張サーチの探索コーン
を調整します。

### BooleanQuery

```python
bq = BooleanQuery()
bq.must(query)
bq.should(query)
bq.must_not(query)
```

複合ブールクエリ。引数なしでコンストラクタを呼び出し、`must` / `should` /
`must_not` メソッドで節を一つずつ追加します。各メソッドは任意のクエリ
オブジェクト（ネストされた `BooleanQuery` も含む）を受け付けます。

`must` 節はすべて一致する必要があり、`must_not` 節は一致してはなりません。
`should` 節はスコアリングに寄与し、`must` 節が無い場合は少なくとも1つが
一致する必要があります。

### SpanQuery

```python
# 単一語句
SpanQuery.term(field: str, term: str)

# Near: slop 位置以内の語句
SpanQuery.near(field: str, terms: list[str], *, slop: int = 0, ordered: bool = True)

# ネストされた SpanQuery 句を使った Near
SpanQuery.near_spans(field: str, clauses: list[SpanQuery], *, slop: int = 0, ordered: bool = True)

# Containing: big スパンが little スパンを含む
SpanQuery.containing(field: str, big: SpanQuery, little: SpanQuery)

# Within: 最大距離での include スパンと exclude スパン
SpanQuery.within(field: str, include: SpanQuery, exclude: SpanQuery, distance: int)
```

位置・近接スパンクエリ。静的ファクトリメソッドで構築します。`near` は語句
文字列のリストを受け取り、`near_spans` はネスト式のために `SpanQuery`
オブジェクトのリストを受け取ります。`slop` と `ordered` はキーワード専用
引数です。

### VectorQuery

```python
VectorQuery(field: str, vector: list[float])
```

事前計算済みエンベディングベクトルを使った近似最近傍検索を行います。

### VectorTextQuery

```python
VectorTextQuery(field: str, text: str)
```

クエリ時に `text` をエンベディングに変換してベクトル検索を行います。インデックスにエンベダーの設定が必要です。

---

## SearchRequest

高度な制御が必要な場合の完全なリクエストクラスです。

```python
class SearchRequest:
    def __init__(
        self,
        *,
        query=None,
        lexical_query=None,
        vector_query=None,
        filter_query=None,
        fusion=None,
        limit: int = 10,
        offset: int = 0,
    ) -> None: ...
```

| パラメータ | 説明 |
| :--- | :--- |
| `query` | DSL 文字列または単一クエリオブジェクト。`lexical_query` / `vector_query` と排他的。 |
| `lexical_query` | 明示的なハイブリッド検索の Lexical コンポーネント。 |
| `vector_query` | 明示的なハイブリッド検索の Vector コンポーネント。 |
| `filter_query` | スコアリング後に適用する Lexical フィルター。 |
| `fusion` | フュージョンアルゴリズム（`RRF` または `WeightedSum`）。両コンポーネント指定時のデフォルトは `RRF(k=60)`。 |
| `limit` | 最大結果件数（デフォルト 10）。 |
| `offset` | ページネーションオフセット（デフォルト 0）。 |

---

## SearchResult

`Index.search()` が返すクラスです。

```python
class SearchResult:
    id: str          # 外部ドキュメント識別子
    score: float     # 関連性スコア
    document: dict | None  # 取得されたフィールド値。stored=False の場合は None
```

---

## フュージョンアルゴリズム

### RRF

```python
RRF(k: float = 60.0)
```

逆順位フュージョン（Reciprocal Rank Fusion）。Lexical と Vector の結果リストを順位位置によってマージします。`k` は平滑化定数で、値が大きいほど上位ランクの影響が小さくなります。

### WeightedSum

```python
WeightedSum(lexical_weight: float = 0.5, vector_weight: float = 0.5)
```

両スコアリストをそれぞれ正規化した後、`lexical_weight * lexical_score + vector_weight * vector_score` として結合します。

---

## テキスト解析

### SynonymDictionary

```python
class SynonymDictionary:
    def __init__(self) -> None: ...
    def add_synonym_group(self, synonyms: list[str]) -> None: ...
```

### WhitespaceTokenizer

```python
class WhitespaceTokenizer:
    def __init__(self) -> None: ...
    def tokenize(self, text: str) -> list[Token]: ...
```

### SynonymGraphFilter

```python
class SynonymGraphFilter:
    def __init__(
        self,
        dictionary: SynonymDictionary,
        keep_original: bool = True,
        boost: float = 1.0,
    ) -> None: ...
    def apply(self, tokens: list[Token]) -> list[Token]: ...
```

### Token

```python
class Token:
    text: str
    position: int
    start_offset: int
    end_offset: int
    boost: float
    stopped: bool
    position_increment: int
    position_length: int
```

---

## フィールド値の型マッピング

Python の値は自動的に Laurus の `DataValue` 型に変換されます：

| Python 型 | Laurus 型 | 備考 |
| :--- | :--- | :--- |
| `None` | `Null` | |
| `bool` | `Bool` | `int` より先にチェック |
| `int` | `Int64` | |
| `float` | `Float64` | |
| `str` | `Text` | |
| `bytes` | `Bytes` | |
| `list[float]` | `Vector` | 要素は `f32` に変換 |
| `(lat, lon)` タプル | `Geo` | 2 つの `float` 値 |
| `(x, y, z)` タプル | `Geo3d` | 3 つの `float` 値（ECEF 直交座標系、メートル単位） |
| `datetime.datetime` | `DateTime` | `isoformat()` 経由で変換 |
