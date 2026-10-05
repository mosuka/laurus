# 青空文庫サンプル

[青空文庫](https://www.aozora.gr.jp/) の著作権フリー作品を、形態素解析（Lindera + IPADIC）と意味的ベクトル検索（Candle BERT embedding）を使ってインデックスして検索するサンプルです。
フィールドごとに異なるアナライザーを割り当てる方法（あるフィールドは形態素解析で部分一致、別のフィールドは完全一致）、引用符あり（フレーズ）と引用符なし（OR 緩和）でのクエリ意味論の違い、Unicode に安全な検索結果表示、そしてハイブリッド（レキシカル + ベクトル）検索を示します。

## 前提条件

- [python3](https://www.python.org/) — 作品リスト CSV の解析、本文の CP932 デコード、ルビ・注記の除去
- [curl](https://curl.se/)、[unzip](https://linux.die.net/man/1/unzip) — Lindera IPADIC 辞書のダウンロードと展開
- 初回実行時のネットワーク接続（IPADIC 辞書 約 15MB、テキスト埋め込みモデル 約 470MB、青空文庫の作品リストをダウンロードします。いずれも以降はキャッシュされます）
- `embeddings-candle` build フィーチャーの有効化が必要（このサンプルの各スクリプトは既に指定済みです）
- movies サンプルと異なり、データセットを別途クローンする必要はありません

## スキーマ

[schema.toml](schema.toml) で以下のフィールドを定義しています:

| フィールド | 型 | アナライザー / Embedder | インデックス | 保存 | 説明 |
| --------- | ---- | ---------------------- | ---- | ---- | ---- |
| `title` | Text | `ja_ipadic` | あり | あり | 作品名 |
| `author` | Text | `ja_ipadic` | あり | あり | 著者名（形態素解析。例: `author:太宰` で部分一致） |
| `author_exact` | Text | `keyword` | あり | あり | 著者名（完全一致のみ。例: `author_exact:太宰治`） |
| `body` | Text | `ja_ipadic` | あり | なし | 本文（ストアしない。表示は `excerpt` を参照） |
| `excerpt` | Text | — | なし | あり | 本文冒頭の約 200 文字（表示用） |
| `ndc` | Text | Standard（デフォルト） | あり | あり | NDC 分類番号（例: `"913"`） |
| `chars` | Integer | — | あり | あり | 本文の文字数 |
| `card_url` | Text | — | なし | あり | 図書カード URL |
| `title_vec` | Hnsw（384次元） | `ja_text_embedder` | あり | なし | 作品名の意味ベクトル |
| `body_vec` | Hnsw（384次元） | `ja_text_embedder` | あり | なし | 本文（冒頭）の意味ベクトル（切り詰めの注意点は[Embedder](#embedder)を参照） |

デフォルト検索フィールド: `title`, `author`, `body`

### 日本語アナライザー

`[analyzers.ja_ipadic]` はカスタムパイプラインを定義しています: NFKC 正規化 → 日本語の踊り字展開（々/ゝ/ゞ） → Lindera 形態素解析（IPADIC、`mode = "normal"`） → 小文字化。

これは意図的に組み込みの `{language = "japanese"}` プリセットを使っていません。このプリセットは日本語ストップワードフィルタを適用しますが、フィルタはトークン位置を振り直さずにトークンを除去するため、`PhraseQuery`（連続した位置を前提とする）は助詞を含むフレーズで黙って 0 件を返してしまいます。例えば `title:"銀河鉄道の夜"` は一致しなくなります。カスタム定義の `ja_ipadic` は助詞を含むすべての形態素を保持するため、フレーズ検索が期待通りに動作します。

### フィールド別アナライザー: `author` と `author_exact` の対比

同じ入力値を、異なるアナライザーを持つ 2 つのフィールドにインデックスしています。これは「フィールド別アナライザー」が何をもたらすかを示す、最も分かりやすい例です。

- `author:太宰` — Lindera 経由で一致（姓の部分一致）
- `author_exact:太宰治` — keyword アナライザー経由で一致（完全一致）
- `author_exact:太宰` — **一致しません**（keyword アナライザーは値を分割しないため）

### Embedder

`[embedders.ja_text_embedder]` は
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
（384 次元、HuggingFace Hub から初回のみダウンロード — 約 470MB — `$HF_HOME`（未設定なら `~/.cache/huggingface`）配下にキャッシュされます）を使用しています。これは日本語専用モデルではありません。このスキーマを作成した時点で、日本語専用の代替モデル（`cl-nagoya/sup-simcse-ja-base`、`sonoisa/sentence-bert-base-ja-mean-tokens-v2`、`pkshatech/GLuCoSE-base-ja`）はいずれも、laurus の `CandleBertEmbedder` が必要とする `model.safetensors` または `tokenizer.json` のいずれかを Hub 上に欠いており、実際にロードして動作することを確認できたのはこの多言語モデルでした。

`CandleBertEmbedder` は、sentence-transformers と同じく、`sentence_bert_config.json` にあるモデルの `max_seq_length`（このモデルでは 128 トークン）で入力を切り詰めます。そのため、青空文庫の長い本文を渡してもエラーにはならず安全に切り詰められます — `body_vec` の埋め込みは本文全体ではなく作品冒頭部分の意味を反映します。

`author` は意図的にベクトル化していません。人名は固有名詞であり、名前同士の意味的類似検索には実用上の意味がないためです。代わりに既存の `author`（部分一致）/ `author_exact`（完全一致）のレキシカルフィールドを使用してください。

## ドキュメント例

`build_dataset.py` は `examples/aozora/data/aozora.jsonl` に `put docs` 用のレコードを1行1件
（`{"id": ..., "fields": {...}}`）で書き出します。実際のファイルから2件、
`body`/`excerpt`/`body_vec` を読みやすさのため `…` で省略して示します（実ファイルでは省略せず全文が1行に入ります）:

```json
{"id": "000919", "fields": {
  "title": "いなか、の、じけん",
  "author": "夢野 久作",
  "author_exact": "夢野久作",
  "body": "いなか、の、じけん　備考\n\n　みんな、私の郷里、北九州の某地方の出来事で、私が見聞致しましたことばかりです。…",
  "excerpt": "いなか、の、じけん　備考　みんな、私の郷里、北九州の某地方の出来事で…",
  "ndc": "913",
  "chars": 169,
  "card_url": "https://www.aozora.gr.jp/cards/000919/card919.html",
  "title_vec": "いなか、の、じけん",
  "body_vec": "いなか、の、じけん　備考\n\n　みんな、私の郷里、北九州の某地方の出来事で、私が見聞致しましたことばかりです。…"
}}
{"id": "001140", "fields": {
  "title": "長崎",
  "author": "芥川 竜之介",
  "author_exact": "芥川竜之介",
  "body": "菱形の凧。サント・モンタニの空に揚つた凧。うらうらと幾つも漂つた凧。\n　路ばたに商ふ夏蜜柑やバナナ。…",
  "excerpt": "菱形の凧。サント・モンタニの空に揚つた凧。うらうらと幾つも漂つた凧。　路ばたに商ふ夏蜜柑やバナナ。…",
  "ndc": "914",
  "chars": 318,
  "card_url": "https://www.aozora.gr.jp/cards/001140/card1140.html",
  "title_vec": "長崎",
  "body_vec": "菱形の凧。サント・モンタニの空に揚つた凧。うらうらと幾つも漂つた凧。\n　路ばたに商ふ夏蜜柑やバナナ。…"
}}
```

`title_vec`/`body_vec` は `title`/`body` と同じテキストをコピーしたものです — インデックス時に
フィールドへ設定された embedder（[Embedder](#embedder) 参照）によって自動的にベクトル化されます。

## 使い方

### 1. インデックスの作成

```bash
bash examples/aozora/scripts/create_index.sh
```

release バイナリをビルドし、Lindera IPADIC 辞書をダウンロード（初回のみ。以降はキャッシュされます。[scripts/fetch_dict.sh](scripts/fetch_dict.sh) 参照）、`schema.toml` の `@IPADIC_DIR@` プレースホルダーを絶対パスに置換した上で、`examples/aozora/index/` に空のインデックスを作成します。

### 2. 作品の投入

```bash
bash examples/aozora/scripts/index_aozora.sh
```

オプション（すべて `build_dataset.py` にそのまま渡されます）:

| オプション | デフォルト | 説明 |
| --------- | ---------- | ---- |
| `--limit N` | `1000` | 作品ID 昇順で先頭 N 件のみインデックス（`0` で著作権フリー全作品 約 17,000 件） |
| `--ndc CODE` | — | 分類番号に `CODE` を含む作品のみ（例: `913` で日本の小説） |
| `--author NAME` | — | 著者名に `NAME` を含む作品のみ |
| `--parallel N` | `4` | 本文ダウンロードの並列数 |
| `--sleep SECONDS` | `0.2` | ワーカーごとのダウンロード間隔 |
| `--refresh-list` | — | キャッシュ済みでも作品リスト CSV を再取得 |
| `--yes` | — | `--limit 0` 実行時の確認待機をスキップ |

```bash
# クイックテスト
bash examples/aozora/scripts/index_aozora.sh --limit 20

# 夏目漱石の作品のみ
bash examples/aozora/scripts/index_aozora.sh --author 夏目漱石

# 著作権フリー全作品（全作品ぶん aozora.gr.jp にリクエストが飛びます。下記の注意を参照）
bash examples/aozora/scripts/index_aozora.sh --limit 0 --yes
```

このスクリプトは以下を行います:

1. release バイナリをビルド
2. `build_dataset.py` を実行し、青空文庫の作品リストを取得、著作権フリーかつ本文URLありの作品を抽出、著者・翻訳者・校訂者などの行を作品単位に畳み込み、各作品の本文をダウンロードして CP932 デコード、ルビ・注記記法を除去した上で `put docs` 用の JSONL を生成
3. `laurus put docs` で JSONL を一括投入

> 青空文庫はボランティアによって運営されています。`--limit 0` は著作権フリー全作品（数万リクエスト）の本文をダウンロードします。特別な理由がない限りデフォルトの件数のままお使いください。

青空文庫の作品IDはおおむね登録順に振られており、若い ID には有名作品が多く含まれます（例: 芥川龍之介「羅生門」、宮沢賢治「銀河鉄道の夜」、夏目漱石「こころ」）。そのため、デフォルトの「先頭 1,000 件」だけでも見覚えのあるタイトルが揃いやすくなっています。

### 3. 検索例の実行

```bash
bash examples/aozora/scripts/search_aozora.sh
```

以下の検索例を実行します:

- 全文検索 — `羅生門`
- フィールド指定検索 — `title:こころ`、`body:蜘蛛の糸`
- 著者検索の対比 — `author:芥川`（部分一致） vs `author_exact:芥川竜之介`（完全一致） vs `author_exact:芥川`（意図的に 0 件）
- フレーズ検索 vs OR 緩和検索 — `title:"銀河鉄道の夜"`（厳密） vs `title:銀河鉄道の夜`（引用符なし、形態素の OR）
- 助詞を含むフレーズ検索 — `title:"吾輩は猫である"`
- 自然文検索（句読点を含むため引用符が必須） — `"ある日の暮方の事である"`
- ブール演算子（`AND`、`OR`、`-`）、NDC 分類・文字数によるレンジ絞り込み
- ベクトル検索（`title_vec:"人間の孤独と疎外感"`）とハイブリッド検索（`title:こころ body_vec:"人間の孤独感"`、`+` でベクトル節を必須化）— [ベクトル検索・ハイブリッド検索](#ベクトル検索ハイブリッド検索)を参照
- JSON 形式での出力

## ベクトル検索・ハイブリッド検索

`title_vec` と `body_vec` は意味的（Hnsw）フィールドです（[Embedder](#embedder)参照）。上記のレキシカルフィールドと異なり、形態素ではなく意味で一致するため、クエリ文字列そのものを含まない作品も検索結果に浮かび上がることがあります。

```bash
# ベクトル検索のみ
./target/release/laurus --index-dir examples/aozora/index \
  search 'title_vec:"人間の孤独と疎外感"' --limit 5

# ハイブリッド: レキシカル OR ベクトル（デフォルトは RRF 融合）
./target/release/laurus --index-dir examples/aozora/index \
  search 'title:こころ body_vec:"人間の孤独感"' --limit 5

# ハイブリッド: ベクトル節も必須にする
./target/release/laurus --index-dir examples/aozora/index \
  search 'title:こころ +body_vec:"人間の孤独感"' --limit 5
```

完全な構文（融合アルゴリズムの詳細を含む）は [Query DSL](../../docs/ja/src/concepts/query_dsl.md) と [ハイブリッド検索](../../docs/ja/src/concepts/search/hybrid_search.md) を参照してください。

## 日本語クエリの注意点

- 引用符なしの語は Unicode の文字・数字を受け付けますが、句読点（`。、「」` など）は**受け付けません**。句読点を含む文字列は引用符で囲んでください: `"ある日の暮方の事である"` は解析できますが、`ある日の暮方の事である。` はパースエラーになります。
- 引用符付きフレーズ（`"..."`）は形態素の完全な並び（`PhraseQuery`）を要求します。引用符なしの語がアナライザーで複数の形態素に分割された場合は、それらの形態素が OR 結合（`BooleanQuery`）されるため、より緩やかに一致します — Lucene の `match` クエリに近い挙動です。
- フィールド名は ASCII のみです（`title`、`author` など）。値には日本語を使えます。

## 手動検索

```bash
./target/release/laurus --index-dir examples/aozora/index search 'title:"銀河鉄道の夜"' --limit 10
```

対話モードで操作する場合:

```bash
./target/release/laurus --index-dir examples/aozora/index repl
```

## gRPCサーバーとMCPサーバー

CLIの代わりに、このインデックスをgRPC（＋任意でHTTPゲートウェイ）でサーブしたり、
MCPクライアント（Claude Codeなど）に公開したりすることもできます。

```bash
# gRPCサーバー（--http-portでHTTPゲートウェイも同時起動）を、既に構築済みの
# aozoraインデックスに対して起動する。インデックスディレクトリ内の schema.toml
# （レンダリング済みのIPADIC辞書パスを含む）が自動的に使われる
# — --schema のようなフラグは存在せず、指定も不要
./target/release/laurus --index-dir examples/aozora/index serve --port 50051 --http-port 8080
```

```bash
# HTTPゲートウェイ: gRPCクライアント不要のプレーンなREST/JSON
curl http://localhost:8080/v1/index
curl -X POST http://localhost:8080/v1/search -H "Content-Type: application/json" -d '{"query":"title:こころ","limit":3}'
```

別のターミナルで、MCPサーバーは同じgRPCエンドポイントへ標準入出力（stdio）経由でプロキシします:

```bash
./target/release/laurus mcp --endpoint http://localhost:50051
```

Claude Codeへ登録する場合:

```bash
claude mcp add laurus-aozora -- ./target/release/laurus mcp --endpoint http://localhost:50051
```

`title_vec`/`body_vec` を使うには `embeddings-candle` フィーチャー付きでビルドしたバイナリが
必要です（[使い方](#使い方)の手順どおりであれば既に満たされています）。これが無い場合、
サーバー自体は正常に起動しますが、このインデックスへのベクトル・ハイブリッド検索クエリは
リクエスト時にエラーになります。

## トラブルシューティング

- **「Failed to resolve analyzer for field 'title'」** — IPADIC 辞書が見つからないか壊れています。`bash examples/aozora/scripts/fetch_dict.sh --force` を再実行してください。
- **インデックスを作り直したい** — `rm -rf examples/aozora/index/store examples/aozora/index/schema.toml` の後、`create_index.sh` を再実行してください。
- **すべてのクエリが 0 件になる** — 実行している `laurus` バイナリがこのブランチからビルドされているか確認してください。CLI の `search`/`repl search` コマンドは、`Engine::unified_query_parser()` 経由でスキーマ自身のフィールド別アナライザーを使う必要があり、英語向けの固定アナライザーを使ってはいけません。
- **ダウンロードが遅い・失敗する** — `--parallel 2 --sleep 0.5` で再実行してください。ダウンロード済みの作品はキャッシュされ、再実行時にはスキップされます。
- **初回実行が遅い・埋め込みモデルのダウンロードに失敗する** — `create_index.sh`/`index_aozora.sh`/`search_aozora.sh` の初回実行時、HuggingFace Hub から `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` の約 470MB をダウンロードします。以降は `$HF_HOME`（未設定なら `~/.cache/huggingface`）にキャッシュされます。途中でダウンロードが失敗した場合は、`$HF_HOME/hub` 配下の該当モデルのキャッシュディレクトリを削除して再実行してください。

## ファイル構成

```text
examples/aozora/
├── README.md
├── README_ja.md
├── schema.toml               # インデックスのスキーマテンプレート（@IPADIC_DIR@ プレースホルダー）
├── scripts/
│   ├── fetch_dict.sh         # Lindera IPADIC 辞書のダウンロードと展開
│   ├── create_index.sh       # ビルド、辞書取得、スキーマレンダリング、インデックス作成
│   ├── build_dataset.py      # 作品リスト取得→抽出→本文ダウンロード→整形→JSONL 生成
│   ├── index_aozora.sh       # データセット生成とバルク投入
│   └── search_aozora.sh      # 検索例
├── dict/                      # 展開された Lindera IPADIC 辞書（git 管理外）
├── data/                       # キャッシュされた CSV/zip と生成された JSONL データセット（git 管理外）
└── index/                     # 生成されるインデックスデータ（git 管理外）
```
