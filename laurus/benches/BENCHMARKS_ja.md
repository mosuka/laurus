<!-- markdownlint-disable MD060 -->

# ベンチマークアーキテクチャ

このドキュメントは laurus ベンチマークスイートの設計判断（**なぜそうなっているか**）を整理したものです。各 bench ファイルの先頭 docstring が「何を測るか・どう実行するか」を扱う一方、このファイルは横断的な設計の考え方を扱います。

英語版: [BENCHMARKS.md](BENCHMARKS.md)

## 3 フェーズモデル

この suite の検索系 bench は論理的に 3 つのフェーズに分かれます:

```
┌──────────────────────────┐
│ Phase 1: コーパス生成     │  純関数 (`build_body*`)
│                          │  入力: i (整数), 出力: String
│ コスト: ~1 秒 / 100k     │  決定論的・メモリ上のみ
└──────────────────────────┘
            ↓
┌──────────────────────────┐
│ Phase 2: インデックス構築 │  engine.add_document × n + commit
│                          │  入力: コーパス, 出力: 索引ファイル群
│ コスト: ~17 分 / 100k    │  ★ 重いフェーズ。ディスクに永続化
└──────────────────────────┘
            ↓
┌──────────────────────────┐
│ Phase 3: 検索計測         │  Criterion `b.iter` で engine.search を計測
│                          │  入力: 索引 + クエリ, 出力: 結果 + latency
│ コスト: ~10 秒 / シナリオ │  bench で実際に測りたいのはここだけ
└──────────────────────────┘
```

**検索系 bench**（`lexical_search_bench`）の目的は Phase 3 を孤立して計測すること。Phase 1 と 2 は setup であり、計測対象ではない。

**インデックス系 bench**（`lexical_indexing_bench`）では Phase 2 自体が計測対象。Phase 1 は setup、Phase 3 はそもそも存在しない。

**マイクロベンチ**（`bm25_simd_bench` 等）は 3 フェーズモデルとは別の存在で、合成入力で kernel を直接叩く。Engine も corpus も Criterion-async wrapping も使わない。

## なぜディスク索引キャッシュか (#510)

キャッシュ導入前は、検索系 bench が毎回 `cargo bench` で索引をゼロから再構築していました。複数の bench function が同じ形のコーパス（uniform 100k, skewed 100k 等）を必要とし、それぞれが独立にビルドコストを支払う構造。1 度のフル run で 100k uniform ケースだけでも 4-6 回ビルドが走り、latency 計測の前に純粋なリビルドだけで 1-2 時間を消費していました。

`cached_engine` はビルド済み索引を以下に書き出します:

```
target/laurus_bench_index_cache/<shape>_<n>_segs<k>_v<N>/
```

laurus の標準的な `Engine::builder(file_storage, schema).build()` パス経由で再オープン（その内部の `recover()` が既存セグメントを再アタッチ）します。初回 run の後、後続の `cargo bench` は **1 秒未満**で Phase 3 に到達します。

### 無効化（invalidation）

| 方法                                              | 用途                                                                       |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| `BENCH_INDEX_FORMAT_VERSION` を bump（ソース内定数） | schema / analyzer / corpus 合成 / segment format 変更時。stale な cache は自動再構築 |
| `LAURUS_BENCH_REBUILD=1`                           | 手動オーバーライド。bench 自体を弄っているとき便利                            |
| `cargo clean`                                      | `target/` 全体を消す。cache も道連れに消える                                  |

### 検索系のみが対象

キャッシュは**検索系 bench のみ**。インデックス系 bench は `cached_engine` を呼ばず、毎イテレーションで新規メモリ engine を構築します。**ビルドそのものが計測対象**だからです。

## なぜ合成データか — TREC / Wikipedia を使わない理由

Lucene の benchmark フレームワークは外部コーパス（TREC, Wikipedia ダンプ, 任意の `.txt` を読む `LineDocSource`）に対応しており、検索エンジン比較の事実上のデファクトです。laurus suite はこれを意図的に採用していません。

### トレードオフ表

|                         | 外部コーパス               | 合成（現状）              |
| ----------------------- | ------------------------- | ------------------------ |
| コーパスサイズ          | GB スケール（Wikipedia ~22 GB） | ~数 MB（100k × ~300 B/doc） |
| 語彙                    | 実データ（数百万のユニーク語）  | 合成（~150 ユニーク語）    |
| 再現性                  | スナップショット日付に依存 | ソースから決定論的           |
| CI / fresh checkout コスト | 数 GB DL、ライセンス審査 | ゼロ — プロセス内生成        |
| 実世界との近さ          | 高                         | 中（Zipf 風の合成）         |
| **比率**比較（PR の before/after）に有用か | 有用 | 有用 |
| **絶対値**スループット主張に有用か | 有用 | 有用ではない（語彙が限定的） |

### なぜ合成で十分か

laurus bench suite の実際のワークフローは **perf-PR の比率比較**です。「私の変更で 100k 検索が 1.2x 速くなったか？」という質問。この目的には合成データで十分。条件:

1. **Term frequency 分布が実世界寄り**: 関連するコードパス（BMW pivot, skip list, top-K 早期終了等）を発火させるのに足る現実度。3 層 Zipf 風分布（`COMMON_TERMS`, `TOPIC_PHRASES`, `LONG_TAIL`）でこれを満たしている。#403 (BMW), #466 (enum dispatch), #503 (skip list), #506 (SIMD batch) で検証済み。
2. **Posting list が長い**: テスト対象のデータ構造が「光る」だけの長さ。100k docs なら rare-term posting は ~6k entries → `log_8(6000) ≈ 4` 段の skip-list 階層が形成される。1M docs ならその +1 段（`log_8(60_000) ≈ 5`）。同オーダー。
3. **キャッシュ効果**: 100k で典型的な L3 (~32-64 MB) を溢れ DRAM パスを叩く。10k はキャッシュ内に収まりクリーンなコントロール計測になる。

### 合成データでは答えられないこと

- 「laurus は 10M docs の enwiki で Lucene とどう比較されるか？」 → 外部コーパス対応が必要。本ドキュメントの範囲外。本当に必要になれば別 issue を切る。
- 「analyzer が実世界の long-tail 語彙で本当に効くか？」 → 合成語彙が小さいため analyzer の相対コストが現実より小さく見える。

`#463` umbrella で進めている perf work では、合成データは「CI 速度・決定論性・外部依存なし・ライセンス簡素」のあらゆる観点で勝っている。

## なぜ Lucene の bench フレームワークを採用しないか

Lucene の `benchmark/` モジュールは Task DSL（`.alg` ファイル）・Content Source・DocMaker・大量の計測タスク（`AddDocTask`, `SearchTravRetTask` 等）を持ちます。プロジェクトの規模（数十年の貢献者、数百のバリエーション）には適切です。

laurus は Criterion を素の Rust 関数形式で使います。理由:

1. **Criterion で十分**: warmup / sample collection / 統計分析 / `--baseline` / `--save-baseline` のワークフローは Criterion が既に提供。perf PR で必要なものは揃っている。
2. **コードを読めば分かる**: bench function は上から下に読めば分かる。DSL を経由する利点が薄い。
3. **オンボーディングコスト**: Rust が分かる contributor が bench を読めば即座に内容を理解できる。Lucene DSL は学習層を 1 つ増やす。

将来 laurus が Lucene 風の「N docs/sec を入れながら同時 M searcher を回す」mixed workload を必要としたら、それは別種の bench（soak test）であり専用 harness を作るべき。単発の perf bench にこの仕組みは不要。

## ファイルごとのスコープ（早見表）

| Bench ファイル                | 計測するフェーズ | キャッシュ使用 | 備考 |
| ---------------------------- | -------------- | ------------- | ---- |
| `lexical_search_bench`        | Phase 3        | 使う           | Term / Boolean / Phrase / Fuzzy / DSL 検索スループット |
| `lexical_indexing_bench`      | Phase 2        | 使わない        | `add_document` + `commit` コスト |
| `bm25_simd_bench`             | Microbench     | n/a           | SIMD BM25 kernel のみ（engine なし）|
| `posting_skip_to_bench`       | Microbench     | 独自 setup     | Skip-list seek microbench |
| `posting_merge_bench`         | Microbench     | 独自 setup     | Segment-merge microbench |
| `dict_lookup_bench`           | Phase 3 / micro | 独自 setup     | 辞書 lookup バリエーション |
| `hybrid_search_bench`         | Phase 3        | 使う           | Lexical + vector hybrid（`cached_hybrid_engine` で cache 化、#513 Stage 1）|
| `bkd_bench`                   | Microbench     | 独自 setup     | Geo / numeric range |
| `distance_bench`              | Microbench     | n/a           | 距離 kernel |
| `facet_bench`                 | Phase 3        | 独自 setup     | Facet collection |
| `highlight_bench`             | Phase 3        | 独自 setup     | Highlighter |
| `mutation_bench`              | Phase 2 相当   | n/a           | Update / delete スループット |
| `search_perf`                 | Phase 3        | 独自 setup     | 旧エントリポイント |
| `spell_correction_bench`     | Phase 3        | n/a           | スペル補正 |
| `store_fetch_bench`           | Phase 2 相当   | 独自 setup     | Document store fetch |
| `synonym_bench`               | Phase 3        | n/a           | Synonym 展開 |
| `text_analysis_bench`         | Microbench     | n/a           | Analyzer パイプライン |
| `vector_indexing_bench`       | Phase 2 相当   | n/a           | Vector index build |
| `vector_search_bench`         | Phase 3        | 使う           | Vector kNN（検索系 bench は `cached_vector_reader` で cache 化、#513 Stage 2。Construction 系は変更なし）|
| `merge_stress_bench`          | Phase 2 相当   | 使わない        | `optimize()` 経由の `MergeEngine::perform_merge`。独自の `TrackingAllocator` グローバルアロケータでピークバイトを報告（#1167）|

「独自 setup」は小さなコーパス形か、汎用キャッシュを必要としない別の setup パターンを使うもの。#513 Stage 1 (`hybrid_search_bench`) と Stage 2 (`vector_search_bench`) の merge により、上表の検索系 bench はすべて on-disk cache を再利用するようになった。cache helper は各 bench ファイルに同居しており、`benches/common.rs` への抽出は別途設計議論が必要（issue #513 本文参照）。

## 新規 search-side bench の追加手順

1. コーパス形を選ぶ: `EngineShape::Uniform`（プレーンテキスト）または `EngineShape::Skewed`（TF skew fixture）
2. サイズゲートヘルパーを選ぶ:
   - [`corpus_sizes`](lexical_search_bench.rs): `(small … LARGE で 100k 追加)`
   - [`skewed_corpus_sizes`](lexical_search_bench.rs): `(1k, 10k … LARGE で 100k 追加)`
   - [`seek_skewed_sizes`](lexical_search_bench.rs): skip-list 形状
3. `cached_engine(&rt, shape, n, segment_count)` で `Arc<Engine>` を取得。キャッシュキーは `(shape, n, segment_count, BENCH_INDEX_FORMAT_VERSION)`。初回はビルド、以降は再オープン
4. `b.iter` の前に `probe` クエリを 1 回投げて「fixture とクエリがズレていないこと」を assert。コーパス／クエリ drift を早期に検出
5. `b.to_async(&rt).iter(|| { let engine = &engine; async move { … } })` のパターンを使う。`Arc<Engine>` を参照キャプチャすれば auto-deref でメソッド呼び出しが通る

索引内容を変える変更（schema・analyzer・corpus 合成・segment format）を入れたら `BENCH_INDEX_FORMAT_VERSION` を bump。stale な cache は次回 run で自動再構築される。

## ピークメモリ計測（`merge_stress_bench`）

`merge_stress_bench` は本スイート初のアロケータ計装 bench。
`TrackingAllocator`（`#[global_allocator]`）でネット割り当てバイト数の
ハイウォーターマークを追跡し、タイムド Criterion ループの開始前に
サイズごとに1回`println!`で報告する。`#[global_allocator]`はコンパイル
済みバイナリ単位で適用され、各`[[bench]]`エントリは独立したバイナリの
ため、他のbench・`cargo test`・ライブラリ本体には一切影響しない —
同じシグナルが必要な将来のbenchで、`common.rs`に触れずに独立して
安全に再利用できる。

これは（同一マシン上でのメモリ関連perf PRの前後比較のような）A/B比較
用の方向性を示すプロキシであり、出版品質のプロファイリングではない。
また計測対象がシングルスレッドであることを前提とする（並行アロケー
ション活動があると単一のグローバルハイウォーターマークが壊れる）
— このパターンを再利用する前に、対象コードに`rayon`/スレッド生成が
無いことを確認すること。

## 背景

- `#510` が `lexical_search_bench` 向けのディスクキャッシュとサイズゲート整理を導入
- `#513` がキャッシュを `hybrid_search_bench`（Stage 1, PR #514）および `vector_search_bench`（Stage 2, `cached_vector_reader` 経由で検索系 bench のみ）へ拡張
- `#403` が BMW acceptance gate を定義し、参照サイズとして 100k を確立
- `#503` が `bench_seek_skewed` を導入。1M docs ケースは #510 で削除（100k からの追加 1 階層は情報量が少ないと判断）
