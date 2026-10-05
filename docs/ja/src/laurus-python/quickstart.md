# クイックスタート

## 1. インデックスを作成する

```python
import laurus

# インメモリインデックス（一時的、プロトタイピングに最適）
index = laurus.Index()

# ファイルベースインデックス（永続的）
# `./myindex/schema.toml` と `./myindex/store/` を書き込む -- これは
# `laurus-cli create index --schema` と同じレイアウトなので、このディレクトリは
# CLI からも開ける（逆も同様）。
schema = laurus.Schema()
schema.add_text_field("title")
schema.add_text_field("body")
index = laurus.Index(path="./myindex", schema=schema)

# 後で再オープンする際はパスだけで済む -- schema を再度渡すとエラーになる
# （スキーマは既に永続化されているため）。
index = laurus.Index(path="./myindex")
```

## 2. ドキュメントをインデックスする

```python
index.put_document("doc1", {
    "title": "Rust 入門",
    "body": "Rust は安全性とパフォーマンスに重点を置いたシステムプログラミング言語です。",
})
index.put_document("doc2", {
    "title": "Python データサイエンス",
    "body": "Python はデータ解析と機械学習に広く使われています。",
})
index.commit()
```

## 3. Lexical 検索

```python
# DSL 文字列
results = index.search("title:rust", limit=5)

# クエリオブジェクト
results = index.search(laurus.TermQuery("body", "python"), limit=5)

# 結果を表示
for r in results:
    print(f"[{r.id}] score={r.score:.4f}  {r.document['title']}")
```

## 4. Vector 検索

Vector 検索にはベクトルフィールドを含むスキーマと事前計算済みエンベディングが必要です。

```python
import laurus

schema = laurus.Schema()
schema.add_text_field("title")
schema.add_hnsw_field("embedding", dimension=4)

index = laurus.Index(schema=schema)
index.put_document("doc1", {"title": "Rust", "embedding": [0.1, 0.2, 0.3, 0.4]})
index.put_document("doc2", {"title": "Python", "embedding": [0.9, 0.8, 0.7, 0.6]})
index.commit()

query_vec = [0.1, 0.2, 0.3, 0.4]
results = index.search(laurus.VectorQuery("embedding", query_vec), limit=3)
```

## 5. ハイブリッド検索

```python
request = laurus.SearchRequest(
    lexical_query=laurus.TermQuery("title", "rust"),
    vector_query=laurus.VectorQuery("embedding", query_vec),
    fusion=laurus.RRF(k=60.0),
    limit=5,
)
results = index.search(request)
```

## 6. Late interaction による再採点

late interaction による再採点は、どの検索の上位の結果も ColBERT 型の MaxSim で並べ替えます。MaxSim は、文書ごとのトークンベクトルを保持する MultiVector フィールドに対して計算します。

```python
import laurus

schema = laurus.Schema()
schema.add_text_field("title")
schema.add_multi_vector_field("tokens", dimension=2, distance="dot_product")

index = laurus.Index(schema=schema)
index.put_document("doc1", {"title": "Rust", "tokens": [[0.1, 0.0]]})
index.put_document("doc2", {"title": "Rust language", "tokens": [[0.9, 0.2], [0.0, 0.3]]})
index.commit()

results = index.search(
    "title:rust",
    rescore=laurus.LateInteractionRescore("tokens", [[1.0, 0.0], [0.0, 1.0]]),
)
```

トークンベクトルの代わりにテキストを渡すには、`"candle_colbert"` のエンベダーを登録し（`schema.add_embedder("colbert", {"type": "candle_colbert", "model": "colbert-ir/colbertv2.0"})`）、MultiVector フィールドに `embedder="colbert"` を指定します。すると文書はフィールドにテキストを与えられ、再採点のクエリも文字列で渡せます。詳細は [API リファレンス → LateInteractionRescore](api_reference.md#lateinteractionrescore) を参照してください。

## 7. 更新と削除

```python
# 更新: put_document は同じ ID の全バージョンを置換する
index.put_document("doc1", {"title": "更新されたタイトル", "body": "新しいコンテンツ。"})
index.commit()

# 既存バージョンを削除せずに新しいバージョンを追記（RAG チャンキングパターン）
index.add_document("doc1", {"title": "チャンク 2", "body": "追加のチャンク。"})
index.commit()

# 全バージョンを取得
docs = index.get_documents("doc1")

# 削除
index.delete_documents("doc1")
index.commit()
```

## 8. スキーマ管理

```python
schema = laurus.Schema()
schema.add_text_field("title")
schema.add_text_field("body")
schema.add_integer_field("year")
schema.add_float_field("score")
schema.add_boolean_field("published")
schema.add_bytes_field("thumbnail")
schema.add_geo_field("location")
schema.add_datetime_field("created_at")
schema.add_hnsw_field("embedding", dimension=384)
schema.add_flat_field("small_vec", dimension=64)
schema.add_ivf_field("ivf_vec", dimension=128, n_clusters=100)
schema.add_multi_vector_field("tokens", dimension=128)
```

## 9. インデックス統計

```python
stats = index.stats()
print(stats["document_count"])
print(stats["vector_fields"])
```
