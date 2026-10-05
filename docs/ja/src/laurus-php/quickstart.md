# クイックスタート

## 1. インデックスを作成する

```php
<?php

use Laurus\Index;
use Laurus\Schema;

// インメモリインデックス（一時的、プロトタイピングに最適）
$index = new Index();

// ファイルベースインデックス（永続的）
// `./myindex/schema.toml` と `./myindex/store/` を書き込む -- これは
// `laurus-cli create index --schema` と同じレイアウトなので、このディレクトリは
// CLI からも開ける（逆も同様）。
$schema = new Schema();
$schema->addTextField("title");
$schema->addTextField("body");
$index = new Index("./myindex", $schema);

// 後で再オープンする際はパスだけで済む -- $schema を再度渡すとエラーになる
// （スキーマは既に永続化されているため）。
$index = new Index("./myindex");
```

## 2. ドキュメントをインデックスする

```php
$index->putDocument("doc1", [
    "title" => "Introduction to Rust",
    "body" => "Rust is a systems programming language focused on safety and performance.",
]);
$index->putDocument("doc2", [
    "title" => "PHP for Web Development",
    "body" => "PHP is widely used for web applications and rapid prototyping.",
]);
$index->commit();
```

## 3. Lexical 検索

```php
// DSL 文字列
$results = $index->search("title:rust", 5);

// クエリオブジェクト
$results = $index->search(new \Laurus\TermQuery("body", "php"), 5);

// 結果を表示
foreach ($results as $r) {
    printf("[%s] score=%.4f  %s\n", $r->getId(), $r->getScore(), $r->getDocument()["title"]);
}
```

## 4. Vector 検索

Vector 検索にはベクトルフィールドを含むスキーマと事前計算済みエンベディングが必要です。

```php
<?php

use Laurus\Index;
use Laurus\Schema;
use Laurus\VectorQuery;

$schema = new Schema();
$schema->addTextField("title");
$schema->addHnswField("embedding", 4);

$index = new Index(null, $schema);
$index->putDocument("doc1", ["title" => "Rust", "embedding" => [0.1, 0.2, 0.3, 0.4]]);
$index->putDocument("doc2", ["title" => "PHP", "embedding" => [0.9, 0.8, 0.7, 0.6]]);
$index->commit();

$queryVec = [0.1, 0.2, 0.3, 0.4];
$results = $index->search(new VectorQuery("embedding", $queryVec), 3);
```

## 5. ハイブリッド検索

```php
use Laurus\SearchRequest;
use Laurus\TermQuery;
use Laurus\VectorQuery;
use Laurus\RRF;

$request = new SearchRequest(
    query: null,
    lexicalQuery: new TermQuery("title", "rust"),
    vectorQuery: new VectorQuery("embedding", $queryVec),
    filterQuery: null,
    fusion: new RRF(60.0),
    limit: 5,
);
$results = $index->search($request);
```

## 6. Late interaction による再採点

late interaction による再採点は、どの検索でも上位の結果を、MultiVector フィールドに対する ColBERT 型の MaxSim で並べ替えます。MultiVector フィールドは、文書ごとに可変本数のトークンベクトルを保持します。

```php
<?php

use Laurus\Index;
use Laurus\Schema;

$schema = new Schema();
$schema->addTextField("title");
$schema->addMultiVectorField("tokens", 2, "dot_product");

$index = new Index(null, $schema);
$index->putDocument("doc1", ["title" => "Rust", "tokens" => [[0.1, 0.0]]]);
$index->putDocument("doc2", ["title" => "Rust language", "tokens" => [[0.9, 0.2], [0.0, 0.3]]]);
$index->commit();

// "title:rust" の上位 100 件（デフォルト）を MaxSim で並べ替える
$results = $index->search("title:rust", 10, 0, null, new Laurus\LateInteractionRescore("tokens", [[1.0, 0.0], [0.0, 1.0]]));
```

再採点した結果のスコアは MaxSim の値です。トークンベクトルの代わりにテキストでクエリを渡すには、`candle_colbert` エンベダー（Cargo の feature `embeddings-candle`）を `addEmbedder` で登録し、`addMultiVectorField` の `$embedder` に指定します。詳細は [LateInteractionRescore](api_reference.md#lateinteractionrescore) を参照してください。

## 7. 更新と削除

```php
// 更新: putDocument は同じ ID の全バージョンを置換する
$index->putDocument("doc1", ["title" => "Updated Title", "body" => "New content."]);
$index->commit();

// 既存バージョンを削除せずに新しいバージョンを追記（RAG チャンキングパターン）
$index->addDocument("doc1", ["title" => "Chunk 2", "body" => "Additional chunk."]);
$index->commit();

// 全バージョンを取得
$docs = $index->getDocuments("doc1");

// 削除
$index->deleteDocuments("doc1");
$index->commit();
```

## 8. スキーマ管理

```php
$schema = new \Laurus\Schema();
$schema->addTextField("title");
$schema->addTextField("body");
$schema->addIntegerField("year");
$schema->addFloatField("score");
$schema->addBooleanField("published");
$schema->addBytesField("thumbnail");
$schema->addGeoField("location");
$schema->addDatetimeField("created_at");
$schema->addHnswField("embedding", 384);
$schema->addFlatField("small_vec", 64);
$schema->addIvfField("ivf_vec", 128, "cosine", 100, 1);
$schema->addMultiVectorField("colbert", 128);
```

## 9. インデックス統計

```php
$stats = $index->stats();
echo $stats["documentCount"];
echo $stats["vectorFields"];
```
