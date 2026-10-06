# クイックスタート

## 1. インデックスを作成する

```ruby
require "laurus"

# インメモリインデックス（一時的、プロトタイピングに最適）
index = Laurus::Index.new

# ファイルベースインデックス（永続的）
# `./myindex/schema.toml` と `./myindex/store/` を書き込む -- これは
# `laurus-cli create index --schema` と同じレイアウトなので、このディレクトリは
# CLI からも開ける（逆も同様）。
schema = Laurus::Schema.new
schema.add_text_field("title")
schema.add_text_field("body")
index = Laurus::Index.new(path: "./myindex", schema: schema)

# 後で再オープンする際はパスだけで済む -- schema: を再度渡すとエラーになる
# （スキーマは既に永続化されているため）。
index = Laurus::Index.new(path: "./myindex")
```

## 2. ドキュメントをインデックスする

```ruby
index.put_document("doc1", {
  "title" => "Rust 入門",
  "body" => "Rust は安全性とパフォーマンスに重点を置いたシステムプログラミング言語です。",
})
index.put_document("doc2", {
  "title" => "Ruby Web 開発",
  "body" => "Ruby は Web アプリケーションと高速プロトタイピングに広く使われています。",
})
index.commit
```

## 3. Lexical 検索

```ruby
# DSL 文字列
results = index.search("title:rust", limit: 5)

# クエリオブジェクト
results = index.search(Laurus::TermQuery.new("body", "ruby"), limit: 5)

# 結果を表示
results.each do |r|
  puts "[#{r.id}] score=#{format('%.4f', r.score)}  #{r.document['title']}"
end
```

## 4. Vector 検索

Vector 検索にはベクトルフィールドを含むスキーマと事前計算済みエンベディングが必要です。

```ruby
require "laurus"

schema = Laurus::Schema.new
schema.add_text_field("title")
schema.add_hnsw_field("embedding", 4)

index = Laurus::Index.new(schema: schema)
index.put_document("doc1", { "title" => "Rust", "embedding" => [0.1, 0.2, 0.3, 0.4] })
index.put_document("doc2", { "title" => "Ruby", "embedding" => [0.9, 0.8, 0.7, 0.6] })
index.commit

query_vec = [0.1, 0.2, 0.3, 0.4]
results = index.search(Laurus::VectorQuery.new("embedding", query_vec), limit: 3)
```

## 5. ハイブリッド検索

```ruby
request = Laurus::SearchRequest.new(
  lexical_query: Laurus::TermQuery.new("title", "rust"),
  vector_query: Laurus::VectorQuery.new("embedding", query_vec),
  fusion: Laurus::RRF.new(k: 60.0),
  limit: 5,
)
results = index.search(request)
```

## 6. Late interaction による再採点

MultiVector フィールドは、文書ごとに可変個のトークンベクトル（ColBERT のトークンごとの埋め込みなど）を保持します。`LateInteractionRescore` は、どの検索でも上位の結果をそのフィールドに対する MaxSim で並べ替え、再採点された結果のスコアはその MaxSim になります。

```ruby
require "laurus"

schema = Laurus::Schema.new
schema.add_text_field("title")
schema.add_multi_vector_field("tokens", 2, distance: "dot_product")

index = Laurus::Index.new(schema: schema)
index.put_document("doc1", { "title" => "rust", "tokens" => [[0.1, 0.0]] })
index.put_document("doc2", { "title" => "rust language", "tokens" => [[0.9, 0.2], [0.1, 0.8]] })
index.commit

results = index.search("title:rust", rescore: Laurus::LateInteractionRescore.new("tokens", [[1.0, 0.0], [0.0, 1.0]]))
# doc2（MaxSim 1.7）が doc1（MaxSim 0.1）より上位になる
```

フィールドに `candle_colbert` のエンベダーを設定すると（`schema.add_embedder("colbert", { type: "candle_colbert", model: "colbert-ir/colbertv2.0" })` と `add_multi_vector_field("tokens", 128, embedder: "colbert")`）、文書はこのフィールドにテキストを与えられ、クエリも `Laurus::LateInteractionRescore.new("tokens", "how do lifetimes work")` のようにテキストで渡せます。[LateInteractionRescore](api_reference.md#lateinteractionrescore) を参照してください。

## 7. 更新と削除

```ruby
# 更新: put_document は同じ ID の全バージョンを置換する
index.put_document("doc1", { "title" => "更新されたタイトル", "body" => "新しいコンテンツ。" })
index.commit

# 既存バージョンを削除せずに新しいバージョンを追記（RAG チャンキングパターン）
index.add_document("doc1", { "title" => "チャンク 2", "body" => "追加のチャンク。" })
index.commit

# 全バージョンを取得
docs = index.get_documents("doc1")

# 削除
index.delete_documents("doc1")
index.commit
```

## 8. スキーマ管理

```ruby
schema = Laurus::Schema.new
schema.add_text_field("title")
schema.add_text_field("body")
schema.add_integer_field("year")
schema.add_float_field("score")
schema.add_boolean_field("published")
schema.add_bytes_field("thumbnail")
schema.add_geo_field("location")
schema.add_datetime_field("created_at")
schema.add_hnsw_field("embedding", 384)
schema.add_flat_field("small_vec", 64)
schema.add_ivf_field("ivf_vec", 128, n_clusters: 100)
schema.add_multi_vector_field("tokens", 128)
```

## 9. インデックス統計

```ruby
stats = index.stats
puts stats["document_count"]
puts stats["vector_fields"]
```
