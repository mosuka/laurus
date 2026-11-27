# Ruby バインディング概要

`laurus` gem は Laurus 検索エンジンの Ruby バインディングです。[Magnus](https://github.com/matsadler/magnus) と [rb_sys](https://github.com/oxidize-rb/rb-sys) を使ってネイティブ Rust 拡張としてビルドされており、Ruby プログラムからネイティブに近いパフォーマンスで Laurus の Lexical 検索、Vector 検索、ハイブリッド検索機能を利用できます。

## 機能

- **Lexical 検索** -- BM25 スコアリングを備えた転置インデックスによる全文検索
- **Vector 検索** -- Flat、HNSW、IVF インデックスを使用した近似最近傍（ANN）検索
- **ハイブリッド検索** -- フュージョンアルゴリズム（RRF、WeightedSum）で Lexical と Vector の結果を統合
- **豊富なクエリ DSL** -- Term、Phrase、Fuzzy、Wildcard、NumericRange、Geo、Boolean、Span クエリ
- **テキスト解析** -- トークナイザー、フィルター、ステマー、同義語展開
- **柔軟なストレージ** -- インメモリ（一時的）またはファイルベース（永続的）インデックス
- **Ruby らしい API** -- `Laurus::` 名前空間の直感的な Ruby クラス

## アーキテクチャ

```mermaid
graph LR
    subgraph "laurus-ruby (gem)"
        RbIndex["Index\n(Ruby クラス)"]
        RbQuery["クエリクラス"]
        RbSearch["SearchRequest\n/ SearchResult"]
    end

    Ruby["Ruby アプリケーション"] -->|"メソッド呼び出し"| RbIndex
    Ruby -->|"クエリオブジェクト"| RbQuery
    RbIndex -->|"Magnus FFI"| Engine["laurus::Engine\n(Rust)"]
    RbQuery -->|"Magnus FFI"| Engine
    Engine --> Storage["ストレージ\n(Memory / File)"]
```

Ruby クラスは Rust エンジンの薄いラッパーです。
各呼び出しは Magnus の FFI 境界を一度だけ越え、その後
Rust エンジンが操作をネイティブコードで実行します。

Rust エンジン内部は非同期 I/O を使用していますが、
Ruby 側のメソッドはすべて**同期関数**として公開されています。
各メソッドは内部で `tokio::Runtime::block_on()` を呼び出し、
非同期 Rust を同期 Ruby にブリッジしていますが、その呼び出しの間は
GVL（Global VM Lock）を解放します（Issue #1103）。これにより、
呼び出し中も他の Ruby スレッドは動き続けられます。マルチスレッド
サーバーがワーカースレッドを増やすことで、すべての呼び出しが
GVL 上で直列化される以前とは異なり、実際にスループットの恩恵を
受けられます。

Ruby スレッドが初めて真の並行ライターになれるため、
エンジン側の既存の並行性に関する制約が Ruby からも
到達可能になります: `commit` は並行する
`put`/`add`/`delete` 呼び出しと直列化されず、
`CommitPolicy` の自動コミット保証は単一ライターでの
取り込みを前提としており、同一 `Index` への並行ライター
下では best-effort（ベストエフォート）になります。
並行実行下でこれらの保証が必要な場合は、明示的な
`commit` 呼び出し、または単一の取り込みスレッドを
使用してください。

Python バインディングとは異なり、`close` は他のスレッドの
実行中の呼び出しの完了を待たずに返ります。Magnus は
`#[magnus::wrap]` メソッドに常に素の `&self` しか渡さないため、
`close` を排他的にする借用チェッカーの仕組みがここには
存在しません。`close` の実行中に別スレッドがまだ呼び出しの
途中である場合、その呼び出し自身が保持する参照によって、
呼び出しが完了するまで内部のエンジン（およびそのストレージ
ロック）は生き続けます。

## クイックスタート

```ruby
require "laurus"

# インメモリインデックスを作成
index = Laurus::Index.new

# ドキュメントをインデックス
index.put_document("doc1", { "title" => "Rust 入門", "body" => "システムプログラミング言語です。" })
index.put_document("doc2", { "title" => "Ruby Web 開発", "body" => "Ruby による Web アプリケーション。" })
index.commit

# 検索
results = index.search("title:rust", limit: 5).results
results.each do |r|
  puts "[#{r.id}] score=#{format('%.4f', r.score)}  #{r.document['title']}"
end
```

## セクション

- [インストール](laurus-ruby/installation.md) -- gem のインストール方法
- [クイックスタート](laurus-ruby/quickstart.md) -- サンプルによるハンズオン入門
- [API リファレンス](laurus-ruby/api_reference.md) -- クラスとメソッドの完全リファレンス
- [開発](laurus-ruby/development.md) -- ソースからのビルドとテスト実行
