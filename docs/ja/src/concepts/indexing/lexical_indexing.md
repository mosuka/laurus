# Lexical インデキシング

Lexical インデキシングは、キーワードベースの検索を支える仕組みです。ドキュメントのテキストフィールドがインデキシングされると、Laurus は**転置インデックス（Inverted Index）** を構築します。これは、タームからそのタームを含むドキュメントへのマッピングを行うデータ構造です。

## Lexical インデキシングの仕組み

```mermaid
sequenceDiagram
    participant Doc as Document
    participant Analyzer
    participant Writer as IndexWriter
    participant Seg as Segment

    Doc->>Analyzer: "The quick brown fox"
    Analyzer->>Analyzer: Tokenize + Filter
    Analyzer-->>Writer: ["quick", "brown", "fox"]
    Writer->>Writer: Buffer in memory
    Writer->>Seg: Flush to segment on commit()
```

### ステップごとの流れ

1. **解析（Analyze）**: テキストが設定されたアナライザー（トークナイザー + フィルター）を通過し、正規化されたタームのストリームが生成される
2. **バッファリング（Buffer）**: タームはフィールドごとに整理され、インメモリの書き込みバッファに格納される
3. **コミット（Commit）**: `commit()` の呼び出し時に、バッファがストレージ上の新しいセグメントにフラッシュされる

## 転置インデックス（Inverted Index）

転置インデックスは、基本的にタームからドキュメントリストへのマップです。

```mermaid
graph LR
    subgraph "Term Dictionary"
        T1["'brown'"]
        T2["'fox'"]
        T3["'quick'"]
        T4["'rust'"]
    end

    subgraph "Posting Lists"
        P1["doc_1, doc_3"]
        P2["doc_1"]
        P3["doc_1, doc_2"]
        P4["doc_2, doc_3"]
    end

    T1 --> P1
    T2 --> P2
    T3 --> P3
    T4 --> P4
```

| コンポーネント | 説明 |
| :--- | :--- |
| **Term Dictionary** | インデックス内のユニークなタームのソート済み辞書。**ディスク形式**は Lucene `BlockTreeTermsWriter` 互換のブロックツリー（FST + 128 ターム単位の front-coded ブロック + bit-packed `TermInfo`）でファイルサイズを最小化。**メモリ形式**は load 時に構築する `AHashMap` インデックス + parallel-array 形式のクエリ層で、`get` / `iter` / `find_prefix` を parallel-array 同等のレイテンシで提供。完全一致検索、順序イテレーション、プレフィックススキャンに対応 |
| **Posting Lists** | 各タームに対する、ドキュメント ID とメタデータ（ターム頻度、位置情報）のリスト |
| **Doc Values** | 数値フィールドや日付フィールドでのソート/フィルター操作のためのカラム指向ストレージ |

### Posting List の内容

Posting List の各エントリには以下の情報が含まれます。

| フィールド | 説明 |
| :--- | :--- |
| Document ID | 内部 `u64` 識別子 |
| Term Frequency | そのドキュメント内でタームが出現する回数 |
| Positions（オプション） | ドキュメント内でタームが出現する位置（フレーズクエリに必要）。多値テキストフィールドの各要素は、フィールドの `position_increment_gap` で区切られた 1 本の昇順の位置列を共有する |
| Weight | このポスティングのスコアウェイト。既定値は `1.0` で、Laurus が書き出すポスティングは常にこの値を持つ。リスト内のすべてのウェイトが `1.0` の場合、ウェイトセクションはディスクから完全に省かれる（v3） |

### On-Disk Posting レイアウト

Posting list は **structure-of-arrays** レイアウトで保存され、各フィールドが
それぞれ連続したセクションに書かれます。ドキュメント ID とターム頻度は
固定長の **128 整数ブロック**単位でビットパックされ（ドキュメント ID は
Frame-of-Reference + ソート済みデルタ）、端数の末尾ブロックは varint に
フォールバックします。これは Tantivy および Lucene 9 が採用する on-disk
形式と同じで、[`bitpacking`](https://crates.io/crates/bitpacking) クレートに
よる SIMD デコードが効きます。

```text
[term, total_frequency, doc_frequency, posting_count N, any_positions, any_weights]
[Skip levels              — v2 以降: num_levels + レベルごと (len + u32 doc_ids)]
[Section 1: doc_ids       — N/128 ビットパックブロック + varint 末尾]
[Section 2: frequencies   — N/128 ビットパックブロック + varint 末尾]
[Section 3: weights       — N 個の生 f32（any_weights == 1 のときのみ）]
[Section 4: positions     — ポスティングごとのフラグ + varint デルタ（存在する場合のみ）]
```

2 つのヘッダフラグが 2 つの省略可能セクションを制御します。`any_positions` は
従来から Section 4 を制御しており、`any_weights` は **v3** で追加され、同じ形で
Section 3 を制御します。Laurus が書き出すポスティングは常に既定値 `1.0` の
ウェイトを持つため、writer が生成するセグメントでは Section 3 が常に省かれ、
posting list は v2 と比べて `4N - 1` バイト小さくなります ── ポスティング
あたり 4 バイトの削減から、リストあたり 1 バイトのフラグを差し引いた値です。
10 億ポスティングでは、書き込み・読み出し・キャッシュのいずれもしなくて
済むバイト数が約 4 GB になります。

この削減によって形式が lossy になることはありません。`1.0` 以外のウェイトが
1 つでもあればフラグが立って Section 3 が復活し、すべての値が厳密に往復します。

セグメント内のドキュメント ID は `u32` に収まる必要があります。`u32::MAX` を
超える値のエンコードは、ビットパックされたセグメントを黙って破損させないよう、
明確なエラーで即座に失敗します。

デコーダは SoA ネイティブの高速パス（`PostingList::decode_soa`）を提供し、
中間的な `Vec<Posting>` の再構築を挟まずに `doc_id` と `frequency` の
parallel な `Vec<u32>` スライスをディスクから直接生成します。クエリ
イテレータはこのスライスを保持するため、`next()` は 40 バイトの `Posting`
構造体をストライドせず、密なスライス上で `u32` カーソルを 1 つ進めるだけで
済みます。

### Multi-Level Skip Table

`N ≥ 8` ポスティングを持つ posting list はヘッダ直後に
Lucene-90 互換のマルチレベルスキップテーブル（v2 フォーマットで導入され、
v3 でも変更なし）を持ちます。各レベルは `doc_ids` 上の固定ストライド窓の "末尾 doc id" を
保持し、レベル 0 は `SKIP_INTERVAL = 8` ポスティングごとに 1 エントリ、
レベル 1 は `8²`、トップレベルは 1 エントリに収束するまで重ねます。

`PostingIterator::skip_to(target)` はテーブルをトップダウンに走査し、
各レベルで `partition_point` により探索窓を `SKIP_INTERVAL` で 1 段ずつ
絞り込んでから、最下層の最大 8 ポスティング窓を線形スキャンします。
1 回あたりのコストは `O(log_8 N + SKIP_INTERVAL)` ── `N = 1M` で約 25
比較。従来の単一レベル `block_cache` が支払っていた線形
`O(N / block_size)` ウォークと比べて大幅に少なくなります。

スキップテーブルは Lucene 9 / Tantivy と同様に posting list の
ファイル内に同居させ、別ファイル化はしません。古いセグメントもそのまま
読めます ── スキップテーブルを on-disk に持たない v1 形式は SoA デコーダが
ロード時に `doc_ids` から再構築し、v2 形式は無条件のウェイトセクションを
そのまま読みます。どのデコーダを使うかは、マジックとバージョン番号を持つ
唯一のセグメント単位ファイルである `.dict` のバージョンで決まるため、
posting のペイロード自体を推測的に読む必要はありません。

### Term Dictionary の 2 層構造

ディクショナリは **ディスク** と **メモリ** で別の表現を持ち、
それぞれの最適化目標を分離しています。

- **ディスク層** — `.dict` ファイルは Lucene `BlockTreeTermsWriter`
  風のブロックツリーレイアウト（マジック `LTDD`、スキーマ v3）。
  100k ユニーク 5-10 バイトタームのコーパスで `.dict` は ~12.5
  バイト/ターム と、旧 parallel-array 形式比 **約 70% 削減**
- **メモリ層** — build / load 時に `AHashMap<term, ordinal>` 索引、
  ordinal indexed の `Vec<String>`、単一コピーの `Arc<[TermInfo]>`
  を構築。`get` / `iter` / `find_prefix` / `find_range` はすべて
  この in-memory 構造のみを参照するので、per-query レイテンシは
  旧 parallel-array 実装と同等（ FST traversal や block 内線形
  スキャンのコストを払わない）

ディスク形式のスクラッチ (FST + BlockSection bytes) は
[`BlockTermDictionary::write_to_storage`] でセグメントマージ時に
再エンコードせず再シリアライズするためにのみ保持しています。

```text
[Header                ]  マジック "LTDD" + バージョン
[FstSection            ]  各ブロック末尾タームを key、ブロックの開始
                          バイトオフセットを value とする fst::Map<u64>
[BlockSection          ]  128 ターム単位のブロックを連結。各ブロックは
                          front-coded タームバイト列、bit-packed の
                          固定長 TermInfo ブロック、可変長の per-term
                          Block-Max-WAND メタデータ配列を含む
[Footer                ]  全タームカウント + ブロックカウント
```

- 検索: FST を 1 回辿って (`O(|term|)`) target を含むブロックを特定
  し、そのブロック内（≤ 128 件）を front-coded で線形スキャン
- イテレーション: FST を経由せず BlockSection を順次走査。各ブロック
  の front-coding バッファを再利用するため、per-step コストは
  front-coding decode（≈ 5–10 ns）のみ
- プレフィックススキャン: FST で先頭ブロックを特定し、prefix が一致
  しなくなるまで順次ブロックを走査

flat per-term FST と比較して block-head FST は 1〜2 桁小さく、
front-coded タームバイト列と bit-packed `TermInfo` の組み合わせで
production 規模ではディスクサイズが 50〜80 % 削減されます。

## 数値フィールドと日付フィールド

整数、浮動小数点数、日時フィールドは、**[BKD ツリー](../bkd_tree.md)** を使用してインデキシングされます。BKD ツリーは範囲クエリに最適化された空間分割データ構造です。

```mermaid
graph TB
    Root["BKD Root"]
    Root --> L["values < 50"]
    Root --> R["values >= 50"]
    L --> LL["values < 25"]
    L --> LR["25 <= values < 50"]
    R --> RL["50 <= values < 75"]
    R --> RR["values >= 75"]
```

BKD ツリーにより、`price:[10 TO 100]` や `date:[2024-01-01 TO 2024-12-31]` のような範囲クエリを効率的に評価できます。

## 地理フィールド（Geo Fields）

地理フィールドには 2 種類あり、いずれも同じ多次元
[BKD-Tree](../bkd_tree.md) プリミティブにバックアップされています：

| フィールド型 | 次元数 | 座標 | サポートされるクエリ |
| :--- | :---: | :--- | :--- |
| `Geo` | 2 | WGS84 緯度・経度（度） | 半径検索、バウンディングボックス |
| `Geo3d` | 3 | ECEF 直交座標 `(x, y, z)`（メートル） | 3D 距離検索（球）、3D バウンディングボックス、k-NN |

`Geo3d` は高度が一級の次元になる用途（ドローン・衛星・屋内 3D 測位など、
2D の `Geo` フィールドでは情報が失われたり極で歪んだりするケース）で
適しています。座標系・WGS84 変換ヘルパー・DSL 構文については
[3D 地理検索 (ECEF)](../geo3d.md) を参照してください。

## セグメント（Segments）

Lexical インデックスは**セグメント**に分割されて構成されます。各セグメントはイミュータブルで自己完結型のミニインデックスです。

```mermaid
graph TB
    LI["Lexical Index"]
    LI --> S1["Segment 0"]
    LI --> S2["Segment 1"]
    LI --> S3["Segment 2"]

    S1 --- F1[".dict (terms)"]
    S1 --- F2[".post (postings)"]
    S1 --- F3[".bkd (numerics)"]
    S1 --- F4[".docs (doc store)"]
    S1 --- F5[".dv (doc values)"]
    S1 --- F6[".meta (metadata)"]
    S1 --- F7[".norms (field-length norms)"]
```

| ファイル拡張子 | 内容 |
| :--- | :--- |
| `.dict` | Term Dictionary。v3 `LTDD` ブロックツリーレイアウト（FST + 128 ターム単位の front-coded ブロック + bit-packed `TermInfo`）。セグメント open 時に `AHashMap` バックの in-memory クエリ層へ展開 |
| `.post` | Posting Lists（ドキュメント ID、ターム頻度、位置情報） |
| `.bkd` | 数値・日付・`Geo`（2D）・`Geo3d`（3D ECEF）フィールドの [BKD ツリー](../bkd_tree.md) データ |
| `.docs` | 格納されたフィールド値（元のドキュメント内容）。チャンク単位（未圧縮で約16KiBまたは128ドキュメントのいずれか早い方）でLZ4圧縮され、圧縮による効果がないチャンクは無圧縮のままフォールバックする（`SDOC` v1、Issue #548） |
| `.dv` | ソートおよびフィルタリング用の Doc Values |
| `.meta` | セグメントメタデータ（ドキュメント数、ターム数など） |
| `.norms` | 1バイトに量子化されたフィールド長の正規化値（BM25 スコアリング用、ドキュメント×フィールドごとに1バイト） |

### セグメントのライフサイクル

1. **作成（Create）**: `commit()` が呼び出されるたびに新しいセグメントが作成される
2. **検索（Search）**: すべてのセグメントが並列に検索され、結果がマージされる
3. **マージ（Merge）**: 各 `commit()` 後に自動マージが走り、セグメント数が `max_segments` を超えると最小のセグメント群がマージされて数が有界に保たれる。手動 `optimize()` は全セグメントを 1 つに強制マージする
4. **削除（Delete）**: ドキュメントが削除された場合、物理的に削除されるのではなく、削除ビットマップに ID が追加される（[Deletions & Compaction](../../laurus/deletions.md) を参照）

## BM25 スコアリング

Laurus は Lexical 検索結果のスコアリングに BM25 アルゴリズムを使用します。BM25 は以下の要素を考慮します。

- **ターム頻度（Term Frequency, TF）**: ドキュメント内でタームが出現する頻度（多いほど良いが、収穫逓減あり）
- **逆文書頻度（Inverse Document Frequency, IDF）**: 全ドキュメントにおけるタームの希少性（希少なほど重要）
- **フィールド長の正規化（Field Length Normalization）**: 短いフィールドは長いフィールドに対してブーストされる

計算式:

```text
score(q, d) = IDF(q) * (TF(q, d) * (k1 + 1)) / (TF(q, d) + k1 * (1 - b + b * |d| / avgdl))
```

`k1 = 1.2` と `b = 0.75` がデフォルトのチューニングパラメータです。

## SIMD 最適化

ベクトル距離計算では、利用可能な場合に SIMD（Single Instruction, Multiple Data）命令が活用され、ベクトル検索における類似度計算が大幅に高速化されます。

## コード例

```rust
use std::sync::Arc;
use laurus::{Document, Engine, Schema};
use laurus::lexical::TextOption;
use laurus::lexical::core::field::IntegerOption;
use laurus::storage::memory::MemoryStorage;

#[tokio::main]
async fn main() -> laurus::Result<()> {
    let storage = Arc::new(MemoryStorage::new(Default::default()));
    let schema = Schema::builder()
        .add_text_field("title", TextOption::default())
        .add_text_field("body", TextOption::default())
        .add_integer_field("year", IntegerOption::default())
        .build();

    let engine = Engine::builder(storage, schema).build().await?;

    // Index documents
    engine.add_document("doc-1", Document::builder()
        .add_text("title", "Rust Programming")
        .add_text("body", "Rust is a systems programming language.")
        .add_integer("year", 2024)
        .build()
    ).await?;

    // Commit to flush segments to storage
    engine.commit().await?;

    Ok(())
}
```

## 次のステップ

- ベクトルインデックスの仕組みを学ぶ: [Vector インデキシング](vector_indexing.md)
- Lexical インデックスに対してクエリを実行する: [Lexical 検索](../search/lexical_search.md)
