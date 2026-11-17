# インストール

## npm から

```bash
npm install laurus-nodejs
```

## ソースから

ソースからビルドするには Rust ツールチェーン（1.89 以降）と
Node.js 24.15 以上が必要です。

```bash
# リポジトリをクローン
git clone https://github.com/mosuka/laurus.git
cd laurus/laurus-nodejs

# 依存パッケージのインストール
npm install

# ネイティブモジュールのビルド（リリース）
npm run build

# デバッグモード（ビルドが速い）
npm run build:debug
```

## 確認

```javascript
import { Index } from "laurus-nodejs";
const index = await Index.create();
console.log(index.stats());
// { documentCount: 0, vectorFields: {} }
```

## 要件

- Node.js 24.15 以上（`package.json` の `engines.node` と一致）
- コンパイル済みネイティブアドオン以外のランタイム依存なし
