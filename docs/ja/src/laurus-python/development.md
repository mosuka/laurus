# 開発環境のセットアップ

このページでは `laurus-python` バインディングのローカル開発環境のセットアップ、ビルド、テスト実行の手順を説明します。

## 前提条件

- **Rust** 1.89 以降（Cargo 付属）
- **Python** 3.8 以降
- リポジトリのローカルクローン

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
```

## Python 仮想環境

Python ツール（Maturin、pytest など）はすべて `laurus-python/.venv` に作成した専用の仮想環境で管理します。

```bash
# venv を作成して maturin と pytest をインストール
make venv
```

これは以下と同等です：

```bash
python3 -m venv laurus-python/.venv
laurus-python/.venv/bin/pip install maturin pytest
```

> **注意:** venv を手動でアクティベートする必要はありません。
> すべての `make` ターゲットは venv 内のバイナリを直接呼び出します。

## ビルド

### 開発ビルド（編集可能インストール）

Rust 拡張をコンパイルして venv にインストールします。
Rust ソースを変更するたびに再実行してください。

```bash
cd laurus-python
VIRTUAL_ENV=$(pwd)/.venv .venv/bin/maturin develop
```

または Makefile のショートカットを使って配布用ホイールもまとめてビルドします：

```bash
make build-laurus-python
```

リリースホイールが `target/wheels/` に生成されます：

```text
target/wheels/laurus-0.x.y-cp312-cp312-manylinux_2_34_x86_64.whl
```

### ビルドの確認

```python
# venv の Python を直接指定して確認する場合:
laurus-python/.venv/bin/python -c "import laurus; print(laurus.Index())"
# Index()
```

## テスト

`make test-laurus-python` は次の2つのテストスイートを順番に実行します：

1. **Rust ユニットテスト** — `cargo test -p laurus-python`
2. **Python 統合テスト** — `maturin develop` 後に `pytest` を実行

```bash
make test-laurus-python
```

Python テストだけを実行する場合（Rust ステップをスキップ）：

```bash
cd laurus-python
VIRTUAL_ENV=$(pwd)/.venv .venv/bin/maturin develop --quiet
.venv/bin/pytest tests/ -v
```

特定のテストだけを実行する場合：

```bash
.venv/bin/pytest tests/ -v -k test_vector_query
```

## Lint とフォーマット

```bash
# Rust Lint（Clippy）
make lint-laurus-python

# Rust フォーマット
make format-laurus-python
```

## クリーンアップ

```bash
# venv だけを削除
make venv-clean

# すべて削除（venv + Cargo ビルド成果物）
make clean
```

## Makefile リファレンス

| ターゲット | 説明 |
| :--- | :--- |
| `make venv` | `.venv` を作成して `maturin` と `pytest` をインストール |
| `make venv-clean` | `.venv` を削除 |
| `make build-laurus-python` | `maturin build` でリリースホイールをビルド |
| `make test-laurus-python` | Rust ユニットテスト + Python pytest |
| `make lint-laurus-python` | `-D warnings` 付きで Clippy を実行 |
| `make format-laurus-python` | `cargo fmt -p laurus-python` |
| `make clean` | venv と Cargo ビルド成果物をすべて削除 |

## プロジェクト構成

```text
laurus-python/
├── Cargo.toml          # Rust クレートマニフェスト
├── pyproject.toml      # Python パッケージメタデータ（Maturin / PEP 517）
├── README.md           # 英語 README
├── README_ja.md        # 日本語 README
├── src/                # Rust ソース（PyO3 バインディング）
│   ├── lib.rs          # モジュール登録
│   ├── index.rs        # Index クラス
│   ├── schema.rs       # Schema クラス
│   ├── query.rs        # クエリクラス
│   ├── search.rs       # SearchRequest / SearchResult / Fusion
│   ├── analysis.rs     # Tokenizer / Filter / Token
│   ├── convert.rs      # Python ↔ DataValue 変換
│   └── errors.rs       # エラーマッピング
├── tests/              # Python pytest 統合テスト
│   └── test_index.py
└── examples/           # 実行可能な Python サンプル
    ├── quickstart.py
    ├── lexical_search.py
    ├── vector_search.py
    ├── hybrid_search.py
    ├── synonym_graph_filter.py
    ├── search_with_openai.py
    └── multimodal_search.py
```
