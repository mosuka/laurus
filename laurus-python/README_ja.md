# laurus-python

[![PyPI](https://img.shields.io/pypi/v/laurus.svg)](https://pypi.org/project/laurus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Laurus](https://github.com/mosuka/laurus) 検索エンジンの Python バインディングです。[PyO3](https://github.com/PyO3/pyo3) と [Maturin](https://github.com/PyO3/maturin) を使ってビルドされたネイティブ Rust 拡張を通じて、Python から Lexical 検索、Vector 検索、ハイブリッド検索を利用できます。

## 機能

- **Lexical 検索** -- BM25 スコアリングを備えた転置インデックスによる全文検索
- **Vector 検索** -- Flat、HNSW、IVF インデックスを使用した近似最近傍（ANN）検索
- **ハイブリッド検索** -- フュージョンアルゴリズム（RRF、WeightedSum）で Lexical と Vector の結果を統合
- **Late Interaction による再採点** -- MultiVector（トークンベクトル）フィールドに対する ColBERT 型の MaxSim で上位の結果を並べ替え
- **豊富なクエリ DSL** -- Term、Phrase、Fuzzy、Wildcard、NumericRange、Geo、Boolean、Span クエリ
- **テキスト解析** -- トークナイザー、フィルター、ステマー、同義語展開
- **柔軟なストレージ** -- インメモリ（一時的）またはファイルベース（永続的）インデックス
- **Python らしい API** -- 型情報を備えた直感的な Python クラス

## インストール

```bash
pip install laurus
```

ソースからビルドする場合（Rust ツールチェーンが必要）:

```bash
pip install maturin
maturin develop
```

## クイックスタート

```python
import laurus

# インメモリインデックスを作成
index = laurus.Index()

# ドキュメントをインデックス
index.put_document("doc1", {"title": "Rust 入門", "body": "システムプログラミング言語です。"})
index.put_document("doc2", {"title": "Python データサイエンス", "body": "Python によるデータ解析。"})
index.commit()

# DSL 文字列で検索
results = index.search("title:rust", limit=5)
for r in results:
    print(f"[{r.id}] score={r.score:.4f}  {r.document['title']}")

# クエリオブジェクトで検索
results = index.search(laurus.TermQuery("body", "python"), limit=5)
```

## インデックスの種類

### インメモリ（一時的）

```python
index = laurus.Index()
```

### ファイルベース（永続的）

```python
schema = laurus.Schema()
schema.add_text_field("title")
schema.add_text_field("body")
schema.add_hnsw_field("embedding", dimension=384)

index = laurus.Index(path="./myindex", schema=schema)
```

これにより `./myindex/schema.toml` と `./myindex/store/` が書き込まれます
-- `laurus-cli create index --schema` と同じレイアウトなので、どちらでも
このディレクトリを開けます。後で再オープンする際はパスだけで済みます
（`schema` は永続化済みの `schema.toml` から読み込まれるため省略します）:

```python
index = laurus.Index(path="./myindex")
```

ファイルベースのインデックスは、`Index` オブジェクトが生存している間ディレクトリ
の排他ロックを保持するため、最初のインデックスを開いたままだと同じパスに対する
2つめの `laurus.Index(path=...)` は失敗します。CPython のガベージコレクタの
破棄タイミングは全ての参照パターン（例えば参照サイクル）で保証されるわけではない
ので、それに頼らず、インデックスを使い終えたら（特に同じパスを再オープンする前に）
`index.close()` を呼んでください。`close()` は冪等で、呼び出し後は他の全ての
メソッドが例外を送出します。

```python
index.close()
```

## インデックスのリロード

`reload()` は、新規に `Index` を構築するコストを払わずに、別プロセスが
コミットした変更を取り込みます。スキーマの埋め込み設定が変わっていなければ、
既にロード済みの埋め込みモデルをゼロから再ロードせず再利用します:

```python
changed = index.reload()  # commit世代が実際に進んでいればTrue
```

`reload()` はインデックスが開いている状態でも、**既に`close()`済み**の
状態でも動作します -- どちらの場合も同じディレクトリを再オープンします。
そのため、新しい`Index`を作って参照を差し替える代わりに、同じ`Index`
オブジェクトを持ち続けたままリロードサイクルを回せます。ファイルベースの
インデックス（構築時に`path`を指定したもの）でのみ動作し、インメモリ
インデックスで呼ぶと`ValueError`を送出します。

`index.commit_generation()` は `stats()["commit_generation"]` と同じ値を、
`stats()`のようにベクトルフィールドを走査するコストを払わずにO(1)で返し
ます。ただし、これは**このIndexオブジェクトを通じたコミットのみ**を反映
するメモリ上のスナップショットで、呼び出すたびにディスクから読み直される
わけではありません。そのため`reload()`（や自分自身の`commit()`）が実際に
状態を進めたかどうかは確認できますが、**別プロセス**が新たにコミットした
かどうかをこれ単体で安く検知することはできません -- それを拾えるのは
`reload()`だけです。

`reload()`のコストを払わずに別プロセスの変更を安く確認したい場合は、
モジュールレベルの`laurus.peek_commit_generation(path)`を使ってください。
`index.commit_generation()`と異なり`Index`オブジェクトに紐付かず、
`Engine`を一切構築せず`commit_generation.json`をディスクから直接読むため、
このプロセスでまだ該当パスの`Index`を一度も開いていなくても使えます:

```python
before = laurus.peek_commit_generation(path)
# ... しばらく経過 ...
if laurus.peek_commit_generation(path) != before:
    index.reload()
```

`path`がlaurusのインデックスディレクトリでない（永続化されたスキーマが
無い）場合は`ValueError`を送出します。

## 永続性 / WAL

永続インデックスはすべての変更を先行書き込みログ（WAL）に書き込みます。
デフォルトでは WAL はレコードごとに `fsync` されるため、各書き込みは完全に
永続化されます。書き込みスループットを高めるためにグループコミットを有効化
すると `fsync` をまとめられます（クラッシュ時には SQLite の
`synchronous = NORMAL` と同様に最後の未同期バッチまでを失う可能性があります）:

```python
import laurus

policy = laurus.WalSyncPolicy.group(max_records=4096, max_interval_ms=1000)
index = laurus.Index(path="./myindex", schema=schema, wal_sync_policy=policy)

index.put_document("doc1", {"title": "Hello"})
index.flush_wal()  # 必要なときに永続性バリアを強制
index.commit()     # WAL もフラッシュされます
```

`wal_sync_policy` を省略する（または `laurus.WalSyncPolicy.per_record()` を
渡す）と、デフォルトのレコードごとの永続性が維持されます。

## クエリタイプ

| クエリクラス | 説明 |
| :--- | :--- |
| `TermQuery(field, term)` | 完全一致検索 |
| `PhraseQuery(field, [terms])` | フレーズ検索（順序一致） |
| `FuzzyQuery(field, term, max_edits)` | 近似一致検索 |
| `WildcardQuery(field, pattern)` | ワイルドカード検索（`*`、`?`） |
| `NumericRangeQuery(field, min, max)` | 数値範囲検索（int または float） |
| `GeoDistanceQuery.within_radius(field, lat, lon, distance_m)` | 地理的距離検索（半径指定） |
| `GeoBoundingBoxQuery.within_bounding_box(field, min_lat, min_lon, max_lat, max_lon)` | 地理的範囲検索（バウンディングボックス） |
| `Geo3dDistanceQuery.within_sphere(field, x, y, z, distance_m)` | 3D ECEF 球距離検索 |
| `Geo3dBoundingBoxQuery.within_box(field, min_x, min_y, min_z, max_x, max_y, max_z)` | 3D ECEF AABB 検索 |
| `Geo3dNearestQuery.k_nearest(field, x, y, z, k)` | 3D ECEF k 最近傍検索 |
| `BooleanQuery(must, should, must_not)` | 複合ブール検索 |
| `SpanNearQuery(field, [terms], slop)` | 近接検索（スパン） |
| `VectorQuery(field, vector)` | 事前計算済みベクトルによる類似度検索 |
| `VectorTextQuery(field, text)` | テキストからベクトルへの変換と類似度検索（エンベダーが必要） |

## ハイブリッド検索

```python
request = laurus.SearchRequest(
    lexical_query=laurus.TermQuery("body", "rust"),
    vector_query=laurus.VectorQuery("embedding", query_vec),
    fusion=laurus.RRF(k=60.0),
    limit=10,
)
results = index.search(request)
```

### フュージョンアルゴリズム

| クラス | 説明 |
| :--- | :--- |
| `RRF(k=60.0)` | 逆順位フュージョン（ランクベース、ハイブリッドのデフォルト） |
| `WeightedSum(lexical_weight=0.5, vector_weight=0.5)` | スコア正規化後の加重和 |

## Late Interaction による再採点

MultiVector フィールドは、文書ごとのトークンベクトル（ColBERT の埋め込みなど）を保持します。`LateInteractionRescore` は、どの検索でも上位 `window_size` 件（デフォルト 100）を、このフィールドに対する MaxSim で並べ替えます:

```python
schema.add_multi_vector_field("tokens", dimension=2, distance="dot_product")
# ...
index.put_document("doc1", {"title": "Rust", "tokens": [[0.9, 0.2], [0.0, 0.3]]})
# ...
rescore = laurus.LateInteractionRescore("tokens", [[1.0, 0.0], [0.0, 1.0]])
results = index.search("title:rust", rescore=rescore)
```

フィールドに `"candle_colbert"` のエンベダーがあれば、文書と再採点のクエリをテキストで渡せます。

## テキスト解析

```python
syn_dict = laurus.SynonymDictionary()
syn_dict.add_synonym_group(["ml", "machine learning"])

tokenizer = laurus.WhitespaceTokenizer()
filt = laurus.SynonymGraphFilter(syn_dict, keep_original=True, boost=0.8)

tokens = tokenizer.tokenize("ml tutorial")
tokens = filt.apply(tokens)
for tok in tokens:
    print(tok.text, tok.position, tok.boost)
```

## サンプル

使用例は [`examples/`](examples/) ディレクトリにあります:

| サンプル | 説明 |
| :--- | :--- |
| [quickstart.py](examples/quickstart.py) | 基本的なインデックスと全文検索 |
| [lexical_search.py](examples/lexical_search.py) | 全クエリタイプ（Term、Phrase、Boolean、Fuzzy、Wildcard、Range、Geo、Span） |
| [vector_search.py](examples/vector_search.py) | エンベディングによるセマンティック類似度検索 |
| [hybrid_search.py](examples/hybrid_search.py) | フュージョンによる Lexical 検索と Vector 検索の統合 |
| [synonym_graph_filter.py](examples/synonym_graph_filter.py) | 解析パイプラインでの同義語展開 |
| [search_with_openai.py](examples/search_with_openai.py) | OpenAI によるクラウドベースエンベディング |
| [multimodal_search.py](examples/multimodal_search.py) | テキストから画像、画像から画像への検索 |

## ドキュメント

- [Python バインディングガイド](https://mosuka.github.io/laurus/ja/laurus-python.html)

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](../LICENSE) ファイルを参照してください。
