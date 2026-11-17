# 開発環境のセットアップ

`laurus-nodejs` バインディングのローカル開発環境の構築、
ビルド、テスト実行について説明します。

## 前提条件

- **Rust** 1.89 以降（Cargo 含む）
- **Node.js** 24.15 以降（npm 含む。`package.json` の `engines.node` と一致）
- リポジトリがローカルにクローン済み

```bash
git clone https://github.com/mosuka/laurus.git
cd laurus
```

## ビルド

### 開発ビルド

デバッグモードで Rust ネイティブアドオンをコンパイルします。
Rust ソースを変更した後は再実行してください。

```bash
cd laurus-nodejs
npm install
npm run build:debug
```

### リリースビルド

```bash
npm run build
```

### ビルドの確認

```javascript
node -e "
const { Index } = require('./index.js');
Index.create().then(idx => console.log(idx.stats()));
"
// { documentCount: 0, vectorFields: {} }
```

## テスト

テストには [Vitest](https://vitest.dev/) を使用し、
`__tests__/` に配置されています。

```bash
# 全テスト実行
npm test
```

特定のテストを名前で実行:

```bash
npx vitest run -t "searches with DSL string"
```

## リントとフォーマット

```bash
# Rust リント（Clippy）
cargo clippy -p laurus-nodejs -- -D warnings

# Rust フォーマットチェック
cargo fmt -p laurus-nodejs --check

# フォーマット適用
cargo fmt -p laurus-nodejs
```

## クリーンアップ

```bash
# ビルド成果物の削除
rm -f *.node index.js index.d.ts

# node_modules の削除
rm -rf node_modules
```

## プロジェクト構成

```text
laurus-nodejs/
├── Cargo.toml          # Rust クレートマニフェスト
├── build.rs            # napi-build セットアップ
├── package.json        # npm パッケージメタデータ
├── README.md           # 英語 README
├── README_ja.md        # 日本語 README
├── src/                # Rust ソース（napi-rs バインディング）
│   ├── lib.rs          # モジュール登録
│   ├── index.rs        # Index クラス
│   ├── schema.rs       # Schema クラス
│   ├── query.rs        # Query クラス群
│   ├── search.rs       # SearchRequest / SearchResult / Fusion
│   ├── analysis.rs     # Tokenizer / Filter / Token
│   ├── convert.rs      # JS ↔ DataValue 変換
│   └── errors.rs       # エラーマッピング
├── __tests__/          # Vitest 統合テスト
│   └── index.spec.mjs
└── examples/           # 実行可能な Node.js サンプル
    ├── quickstart.mjs
    ├── lexical-search.mjs
    ├── vector-search.mjs
    └── hybrid-search.mjs
```
