# Python バインディング概要

`laurus-python` パッケージは Laurus 検索エンジンの Python バインディングです。[PyO3](https://github.com/PyO3/pyo3) と [Maturin](https://github.com/PyO3/maturin) を使ってネイティブ Rust 拡張としてビルドされており、Python プログラムからネイティブに近いパフォーマンスで Laurus の Lexical 検索、Vector 検索、ハイブリッド検索機能を利用できます。

## 機能

- **Lexical 検索** -- BM25 スコアリングを備えた転置インデックスによる全文検索
- **Vector 検索** -- Flat、HNSW、IVF インデックスを使用した近似最近傍（ANN）検索
- **ハイブリッド検索** -- フュージョンアルゴリズム（RRF、WeightedSum）で Lexical と Vector の結果を統合
- **豊富なクエリ DSL** -- Term、Phrase、Fuzzy、Wildcard、NumericRange、Geo、Boolean、Span クエリ
- **テキスト解析** -- トークナイザー、フィルター、ステマー、同義語展開
- **柔軟なストレージ** -- インメモリ（一時的）またはファイルベース（永続的）インデックス
- **Python らしい API** -- 型情報を備えた直感的な Python クラス

## アーキテクチャ

```mermaid
graph LR
    subgraph "laurus-python"
        PyIndex["Index\n(Python クラス)"]
        PyQuery["クエリクラス"]
        PySearch["SearchRequest\n/ SearchResult"]
    end

    Python["Python アプリケーション"] -->|"メソッド呼び出し"| PyIndex
    Python -->|"クエリオブジェクト"| PyQuery
    PyIndex -->|"PyO3 FFI"| Engine["laurus::Engine\n(Rust)"]
    PyQuery -->|"PyO3 FFI"| Engine
    Engine --> Storage["ストレージ\n(Memory / File)"]
```

Python クラスは Rust エンジンの薄いラッパーです。
各呼び出しは PyO3 の FFI 境界を一度だけ越え、その後
Rust エンジンが操作をネイティブコードで実行します。

Rust エンジン内部は非同期 I/O を使用していますが、
Python 側のメソッドはすべて**同期関数**として公開されています。
これは Python の GIL（Global Interpreter Lock）があると
非同期 API が煩雑になる（`asyncio.run()` が常に必要になる）
ためです。代わりに、各メソッドは内部で
`tokio::Runtime::block_on()` を呼び出し、非同期 Rust を
同期 Python にブリッジしていますが、その呼び出しの間は
GIL を解放します（`Python::detach`、Issue #1103）。これにより、
呼び出し中も他の Python スレッドは動き続けられます。以前は
すべての呼び出しが GIL 上で直列化されていましたが、現在は
マルチスレッドサーバーがワーカースレッドを増やすことで
実際にスループットの恩恵を受けられます。

Python スレッドが初めて真の並行ライターになれるため、
エンジン側の既存の並行性に関する制約が Python からも
到達可能になります: `commit()` は並行する
`put`/`add`/`delete` 呼び出しと直列化されず、
`CommitPolicy` の自動コミット保証は単一ライターでの
取り込みを前提としており、同一 `Index` への並行ライター
下では best-effort（ベストエフォート）になります。
並行実行下でこれらの保証が必要な場合は、明示的な
`commit()` 呼び出し、または単一の取り込みスレッドを
使用してください。

> **注意:** Node.js バインディング（`laurus-nodejs`）では、
> 同じ Rust エンジンのメソッドをネイティブな
> `async` / `Promise` API として公開しています。
> Node.js のイベントループは非同期をネイティブにサポート
> しているためです。

## クイックスタート

```python
import laurus

# インメモリインデックスを作成
index = laurus.Index()

# ドキュメントをインデックス
index.put_document("doc1", {"title": "Rust 入門", "body": "システムプログラミング言語です。"})
index.put_document("doc2", {"title": "Python データサイエンス", "body": "Python によるデータ解析。"})
index.commit()

# 検索
results = index.search("title:rust", limit=5).results
for r in results:
    print(f"[{r.id}] score={r.score:.4f}  {r.document['title']}")
```

## セクション

- [インストール](laurus-python/installation.md) -- パッケージのインストール方法
- [クイックスタート](laurus-python/quickstart.md) -- サンプルによるハンズオン入門
- [API リファレンス](laurus-python/api_reference.md) -- クラスとメソッドの完全リファレンス
- [開発](laurus-python/development.md) -- ソースからのビルド、テスト、プロジェクト構成
