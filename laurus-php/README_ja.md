# laurus-php

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Laurus](https://github.com/mosuka/laurus) 検索エンジンの PHP バインディングです。[ext-php-rs](https://github.com/davidcole1340/ext-php-rs) を使ってビルドされたネイティブ Rust 拡張を通じて、PHP から Lexical 検索、Vector 検索、ハイブリッド検索を利用できます。

## 機能

- **Lexical 検索** -- BM25 スコアリングを備えた転置インデックスによる全文検索
- **Vector 検索** -- Flat、HNSW、IVF インデックスを使用した近似最近傍（ANN）検索
- **ハイブリッド検索** -- フュージョンアルゴリズム（RRF、WeightedSum）で Lexical と Vector の結果を統合
- **豊富なクエリ DSL** -- Term、Phrase、Fuzzy、Wildcard、NumericRange、Geo、Boolean、Span クエリ
- **テキスト解析** -- トークナイザー、フィルター、同義語展開
- **柔軟なストレージ** -- インメモリ（一時的）またはファイルベース（永続的）インデックス

## 要件

- PHP 8.1 以上
- Rust ツールチェーン（stable）

## インストール

ソースからエクステンションをビルドします:

```bash
cd laurus-php
cargo build --release
```

共有ライブラリを PHP のエクステンションディレクトリにコピーします:

```bash
cp target/release/liblaurus_php.so $(php -r 'echo ini_get("extension_dir");')/laurus_php.so
```

`php.ini` に以下の行を追加してエクステンションを有効にします:

```ini
extension=laurus_php.so
```

インストールの確認:

```bash
php -m | grep laurus
```

## クイックスタート

```php
<?php
// php.ini で extension=laurus_php.so を設定

use Laurus\Index;
use Laurus\TermQuery;

// インメモリインデックスを作成
$index = new Index();

// ドキュメントをインデックス
$index->putDocument("doc1", ["title" => "Rust 入門", "body" => "システムプログラミング言語です。"]);
$index->putDocument("doc2", ["title" => "PHP Web 開発", "body" => "PHP による Web アプリケーション開発。"]);
$index->commit();

// DSL 文字列で検索
$results = $index->search("title:rust", 5);
foreach ($results as $r) {
    $doc = $r->getDocument();
    printf("[%s] score=%.4f  %s\n", $r->getId(), $r->getScore(), $doc["title"]);
}

// クエリオブジェクトで検索
$results = $index->search(new TermQuery("body", "php"), 5);
```

## インデックスの種類

### インメモリ（一時的）

```php
$index = new Index();
```

### ファイルベース（永続的）

```php
use Laurus\Index;
use Laurus\Schema;

$schema = new Schema();
$schema->addTextField("title");
$schema->addTextField("body");
$schema->addHnswField("embedding", 384);

$index = new Index("./myindex", $schema);
```

これにより `./myindex/schema.toml` と `./myindex/store/` が書き込まれます
-- `laurus-cli create index --schema` と同じレイアウトなので、どちらでも
このディレクトリを開けます。後で再オープンする際はパスだけで済みます
（`$schema` は永続化済みの `schema.toml` から読み込まれるため省略します）:

```php
$index = new Index("./myindex");
```

ファイルベースのインデックスは、`Index` オブジェクトが生存している間ディレクトリ
の排他ロックを保持するため、最初のインデックスを開いたままだと同じパスに対する
2つめの `new Index("./myindex")` は失敗します。PHP の参照カウント/循環コレクタ
による破棄タイミングは全ての参照パターン（例えば参照サイクル）で保証される
わけではないので、それに頼らず、インデックスを使い終えたら（特に同じパスを
再オープンする前に）`$index->close()` を呼んでください。`close()` は冪等で、
呼び出し後は他の全てのメソッドが例外を投げます。

```php
$index->close();
```

インデックスを開き直す（あるいは開く）ことすらせずに、別プロセスが変更を
コミットしたかを安く確認したい場合は、名前空間付きの
`Laurus\peek_commit_generation($path)` を使ってください:

```php
$before = Laurus\peek_commit_generation($path);
// ... しばらく経過 ...
if (Laurus\peek_commit_generation($path) !== $before) {
    // ディスク上で何かが変わった。取り込むにはインデックスを開き直す
}
```

これはディスク上の永続化されたcommit世代を直接読むだけで、`Engine`の構築
（ストレージロック・WALリカバリ・Embedder読み込み）を一切行わないため、
このプロセスでそのパスの`Index`をまだ一度も作っていなくても使えます。
`$path`がlaurusのインデックスディレクトリでない（永続化されたスキーマが
無い）場合は`ValueError`を投げます。

## スキーマ

`Schema` クラスはインデックスの構造を定義します。以下のメソッドでフィールドを追加できます:

| メソッド | 説明 |
| :--- | :--- |
| `addTextField(name, stored, indexed, termVectors, docValues, analyzer, multiValued, positionIncrementGap)` | 全文検索可能なテキストフィールド（`analyzer` は組込名: `standard` / `english` / `keyword` / `simple` / `noop`） |
| `addIntegerField(name, stored, indexed, multiValued, docValues)` | 整数（i64）フィールド |
| `addFloatField(name, stored, indexed, multiValued, docValues)` | 浮動小数点（f64）フィールド |
| `addBooleanField(name, stored, indexed, multiValued, docValues)` | ブーリアンフィールド |
| `addDatetimeField(name, stored, indexed, multiValued, docValues)` | 日時フィールド |
| `addGeoField(name, stored, indexed, multiValued, docValues)` | 地理座標フィールド（緯度/経度） |
| `addGeo3dField(name, stored, indexed, multiValued, docValues)` | 3D ECEF カルテシアン座標フィールド（x, y, z はメートル） |
| `addBytesField(name, stored)` | バイナリデータフィールド |
| `addHnswField(name, dimension, distance, m, efConstruction, defaultEfSearch, embedder, quantizer, subvectorCount, rerankStorage, pqCodebookPath, baseWeight)` | HNSW ベクトルインデックスフィールド |
| `addFlatField(name, dimension, distance, embedder, baseWeight)` | Flat（総当たり）ベクトルインデックスフィールド |
| `addIvfField(name, dimension, distance, nClusters, nProbe, embedder, baseWeight)` | IVF ベクトルインデックスフィールド |
| `addEmbedder(name, config)` | 名前付きエンベダーの登録 |
| `setDefaultFields(fieldNames)` | デフォルト検索フィールドの設定 |

`name`（ベクトルフィールドでは `dimension` も）以降の引数にはすべてデフォルト値があるため、末尾の引数は省略できます。引数は位置引数なので、後ろの引数を渡すときは上記の順序を守ってください。型とデフォルト値は [API リファレンス](../docs/ja/src/laurus-php/api_reference.md) を参照してください。

## クエリタイプ

| クエリクラス | 説明 |
| :--- | :--- |
| `TermQuery(field, term)` | 完全一致検索 |
| `PhraseQuery(field, terms)` | フレーズ検索（順序一致） |
| `FuzzyQuery(field, term, maxEdits)` | 近似一致検索 |
| `WildcardQuery(field, pattern)` | ワイルドカード検索（`*`、`?`） |
| `NumericRangeQuery(field, min, max, numericType)` | 数値範囲検索（integer または float） |
| `GeoDistanceQuery::withinRadius(field, lat, lon, distanceKm)` | 地理的距離検索（半径指定） |
| `GeoBoundingBoxQuery::withinBoundingBox(field, minLat, minLon, maxLat, maxLon)` | 地理的バウンディングボックス検索 |
| `BooleanQuery` | 複合ブール検索（must/should/mustNot） |
| `SpanQuery::near(field, terms, slop, ordered)` | 近接検索（スパン） |
| `VectorQuery(field, vector)` | 事前計算済みベクトルによる類似度検索 |
| `VectorTextQuery(field, text)` | テキストからベクトルへの変換と類似度検索（エンベダーが必要） |

### ブールクエリ

```php
use Laurus\BooleanQuery;
use Laurus\TermQuery;

$bq = new BooleanQuery();
$bq->must(new TermQuery("body", "rust"));
$bq->should(new TermQuery("title", "introduction"));
$bq->mustNot(new TermQuery("body", "deprecated"));

$results = $index->search($bq, 10);
```

### 地理クエリ

```php
use Laurus\GeoBoundingBoxQuery;
use Laurus\GeoDistanceQuery;

// 半径検索
$results = $index->search(GeoDistanceQuery::withinRadius("location", 35.6895, 139.6917, 10.0), 10);

// バウンディングボックス検索
$results = $index->search(GeoBoundingBoxQuery::withinBoundingBox("location", 35.0, 139.0, 36.0, 140.0), 10);
```

## ハイブリッド検索

```php
use Laurus\SearchRequest;
use Laurus\TermQuery;
use Laurus\VectorQuery;
use Laurus\RRF;

$request = new SearchRequest(
    query: null,
    lexicalQuery: new TermQuery("body", "rust"),
    vectorQuery: new VectorQuery("embedding", $queryVec),
    filterQuery: null,
    fusion: new RRF(60.0),
    limit: 10,
    offset: 0,
);
$results = $index->search($request);
```

### フュージョンアルゴリズム

| クラス | 説明 |
| :--- | :--- |
| `RRF(k)` | 逆順位フュージョン（ランクベース、ハイブリッドのデフォルト） |
| `WeightedSum(lexicalWeight, vectorWeight)` | スコア正規化後の加重和 |

## テキスト解析

```php
use Laurus\SynonymDictionary;
use Laurus\WhitespaceTokenizer;
use Laurus\SynonymGraphFilter;

$synDict = new SynonymDictionary();
$synDict->addSynonymGroup(["ml", "machine learning"]);

$tokenizer = new WhitespaceTokenizer();
$filter = new SynonymGraphFilter($synDict, true, 0.8);

$tokens = $tokenizer->tokenize("ml tutorial");
$tokens = $filter->apply($tokens);
foreach ($tokens as $tok) {
    printf("%s position=%d boost=%.2f\n", $tok->getText(), $tok->getPosition(), $tok->getBoost());
}
```

## ドキュメント操作

```php
// ドキュメントの登録（置換）
$index->putDocument("doc1", ["title" => "Hello", "body" => "World"]);

// ドキュメントバージョンの追加
$index->addDocument("doc1", ["title" => "Hello v2", "body" => "World v2"]);

// ドキュメントの全バージョンを取得
$docs = $index->getDocuments("doc1");

// ドキュメントの全バージョンを削除
$index->deleteDocuments("doc1");

// 変更をコミットして検索可能にする
$index->commit();

// インデックスの統計情報を取得
$stats = $index->stats();
echo "ドキュメント数: " . $stats["documentCount"] . "\n";
```

## 耐久性 / WAL

先行書き込みログ（WAL）はデフォルトで完全に耐久的です。すべてのレコードは
書き込みが返る前に `fsync` されます。耐久性をいくらか引き換えに書き込み
スループットを向上させたい場合は、`Index` コンストラクタに `WalSyncPolicy`
を渡して **group commit（グループコミット）** を有効にします。group commit は
`fsync` をまとめ、レコード数（デフォルト 1024）またはバイト数（デフォルト
1 MiB）のいずれかに達したとき、および毎回の `commit()` 時にフラッシュします。
クラッシュ時に失われるのは最後の未同期バッチまでで（SQLite の
`synchronous = NORMAL` と同様）、インデックスが破損することはありません。
オンデマンドで耐久バリアを強制するには `flushWal()` を呼び出します。

```php
use Laurus\Index;
use Laurus\WalSyncPolicy;

// デフォルト: レコードごとの fsync（引数を省略する）。
$index = new Index();

// group commit を有効にし、必要なときにフラッシュを強制する。
$policy = WalSyncPolicy::group(4096, 4 * 1024 * 1024);
$index = new Index("./myindex", null, $policy);

$index->putDocument("doc1", ["title" => "Hello"]);
$index->flushWal(); // バッチが満杯になるのを待たずに今すぐ永続化
```

## 機能フラグ

オプションの Cargo 機能フラグで追加のエンベディングバックエンドを有効にできます:

| 機能フラグ | 説明 |
| :--- | :--- |
| `embeddings-candle` | [Candle](https://github.com/huggingface/candle) によるローカル BERT エンベディング |
| `embeddings-multimodal` | テキストと画像のマルチモーダル（CLIP）エンベディング |
| `embeddings-openai` | OpenAI API によるクラウドベースエンベディング |
| `embeddings-all` | 全エンベディングバックエンドを有効化 |

機能フラグを指定してビルド:

```bash
cargo build --release --features embeddings-candle
```

## ドキュメント

- [PHP バインディングガイド](https://mosuka.github.io/laurus/ja/laurus-php.html)

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](../LICENSE) ファイルを参照してください。
