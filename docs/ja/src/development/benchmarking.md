# ベンチマーク

このガイドでは、laurus のベンチマークの実行方法、ベースライン（baseline）の保存と比較方法、プルリクエストでの結果報告方法について説明します。

ベンチマークスイートは [`laurus/benches/`](https://github.com/mosuka/laurus/tree/main/laurus/benches) 配下にあり、[Criterion](https://github.com/bheisler/criterion.rs) で構築されています。衛生ルール（決定的シード、ファイル冒頭ドキュメント、sanity assert、`sample_size` ポリシー）は [`laurus/benches/common.rs`](https://github.com/mosuka/laurus/blob/main/laurus/benches/common.rs) で一元管理されています。

## スイート一覧

| ファイル | スコープ |
| --- | --- |
| `bkd_bench.rs` | BKD ツリーの範囲検索（range search）、交差判定（intersect）、構築（1D / 2D / 3D、10k / 100k / 1M ポイント） |
| `distance_bench.rs` | `DistanceMetric::distance` の cosine / Euclidean / Manhattan / dot product（現状は単一次元、次元スイープは #424 で対応予定） |
| `lexical_search_bench.rs` | `Engine::search` 経由のエンドツーエンド lexical 検索（term / boolean / phrase / fuzzy / DSL） |
| `search_perf.rs` | Posting iterator の `skip_to`、`BM25Scorer::score`、SIMD バッチスコアリング、コンパクト posting 変換 |
| `spell_correction_bench.rs` | `SpellingCorrector::correct`、各イテレーションごとに fresh corrector を渡す cold-state 計測 |
| `synonym_bench.rs` | `SynonymDictionary::get_synonyms` のルックアップ（100 / 1k / 10k グループ）と構築コスト |
| `text_analysis_bench.rs` | `StandardAnalyzer::analyze` のシングルドキュメントとバッチ（100 ドキュメント）解析 |
| `vector_search_bench.rs` | Flat / IVF / HNSW の構築と検索（1k / 5k ベクタ、dim 128、top-10）。加えてクラスタ選択パス用の大 K IVF ケース（512 / 2048 クラスタ、#668） |

各ファイル冒頭の `//!` ドキュメントコメントにスコープ・シナリオ・フィルタ方法が書かれています。実行前に確認してください。

## ベンチマークの実行

単一ベンチファイルを実行：

```bash
cargo bench -p laurus --bench distance_bench
```

criterion id でフィルタ（部分一致）：

```bash
cargo bench -p laurus --bench distance_bench -- cosine
cargo bench -p laurus --bench vector_search_bench -- "HNSW Search/top10"
```

コンパイル確認のみ（CI やリファクタリング時に有用）：

```bash
cargo bench -p laurus --bench distance_bench --no-run
```

ワークスペースの全ベンチを実行：

```bash
cargo bench -p laurus
```

## ベースラインの保存と比較

Criterion は名前付きベースラインをサポートしており、フィーチャーブランチを `main`（または任意の参照状態）と比較できます。

現在の状態を `main` という名前のベースラインとして保存：

```bash
cargo bench -p laurus --bench distance_bench -- --save-baseline main
```

その後の実行結果をベースラインと比較：

```bash
cargo bench -p laurus --bench distance_bench -- --baseline main
```

出力には `change:` 行がベンチマーク単位で表示され、変化率と判定（`No change in performance detected`、`Performance has improved`、`Performance has regressed`）が示されます。Criterion はベースラインを `target/criterion/<bench-id>/<baseline>/` 配下に保存します。

perf PR の推奨フロー：

1. `main`（または変更前の状態）で — `cargo bench --bench RELEVANT -- --save-baseline main`
2. ブランチで変更を実装
3. ブランチで — `cargo bench --bench RELEVANT -- --baseline main`
4. `change:` 行を PR 説明にコピーする

## 推奨環境

µs / ns 単位のマイクロベンチマークはシステムノイズに敏感です。意味のある数値を得るには：

- **CPU governor**: `performance` に設定（Linux）：

  ```bash
  sudo cpupower frequency-set -g performance
  ```

- **Turbo boost**: 周波数スケーリングが結果を歪めないように無効化：

  ```bash
  echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo   # Intel
  ```

  AMD システムや BIOS レベルの設定は異なるためベンダーのドキュメントを参照してください。

- **バックグラウンド負荷**: ブラウザ・IDE・ビルドウォッチャー・Docker を停止する。CPU を共有するものは短時間ベンチを歪めます。

- **コア固定（任意）**: 利用可能なら固定コアにピン留め：

  ```bash
  taskset -c 2 cargo bench -p laurus --bench distance_bench
  ```

  あるいは同梱のラッパー `scripts/bench-stable.sh` を使えば、`taskset` による
  単一コアへのピン留めと `nice` によるスケジューリング優先度引き上げを 1 行で実行できます：

  ```bash
  ./scripts/bench-stable.sh --bench distance_bench
  ./scripts/bench-stable.sh --bench distance_bench -- --baseline main
  ```

  ラッパーは Linux 限定です。CPU governor や turbo 状態には触りません（root が必要なため）。
  さらに精度が必要なら別途設定してください。

- **再実行**: 2 回実行して比較する。チューニング済みマシンで ~5 % 以下の差はノイズ。共有ワークステーションでは ~5 % を超える差もノイズの可能性がある。1 回の実行結果を過大解釈しない。

環境を安定化できない場合は、不安定な数値を権威ある数値として提示せず、PR で明示的に断ること（例:「共有ノート PC で測定、~10 % のノイズを想定」）。

## Make ターゲット

`Makefile` から共通エントリポイントを利用できます：

```bash
make bench             # cargo bench -p laurus
make bench-baseline    # cargo bench -p laurus -- --save-baseline main
make bench-compare     # cargo bench -p laurus -- --baseline main
```

単一ベンチを指定する場合は `BENCH=name` を渡します：

```bash
make bench BENCH=distance_bench
make bench-baseline BENCH=distance_bench
make bench-compare BENCH=distance_bench
```

## PR 説明テンプレート

PR で計測可能なパフォーマンス変化を主張する場合、下記のようなテーブルを説明に貼り付けてください：

```markdown
## Performance

Environment: <CPU モデル>, governor=performance, turbo disabled, dedicated machine.

Baseline: `main` at <commit-sha>. After: this branch at <commit-sha>.

| Bench | Before | After | Δ | Verdict |
| --- | --- | --- | --- | --- |
| `distance_metrics/cosine` | 4.20 µs | 3.10 µs | -26 % | improved |
| `distance_metrics/euclidean` | 2.18 µs | 2.16 µs | -1 % | no change |

Reproduce: `cargo bench -p laurus --bench distance_bench -- --baseline main`
```

比較が再現可能となるように、ベースラインと変更後の commit SHA を必ず含めてください。チューニング済みマシンで実行した場合でも環境を明示してください。

## 新しいベンチマークの追加

新規ベンチファイルを追加する際は、[`laurus/benches/common.rs`](https://github.com/mosuka/laurus/blob/main/laurus/benches/common.rs) のスイート全体衛生ルールに従ってください：

1. `common::DEFAULT_SEED`（または `lcg_*` ヘルパ）で決定的シードを使う。`rand::rng()` は使わない。
2. ファイル冒頭に `//!` ドキュメントコメントでスコープ・シナリオ・実行コマンド・フィルタ例を記載する。
3. タイミング `b.iter` の外側で 1 度だけ `assert!` を実行し、空結果を出す regression を黙ってパスさせない。
4. `SAMPLE_SIZE_FAST`（デフォルト、50 ms 以下の操作向け）または `SAMPLE_SIZE_SLOW`（構築パス向け）のいずれかを選ぶ。中間値は使わない。
5. `laurus/Cargo.toml` に `[[bench]] name = "..." harness = false` で登録する。クレートは `autobenches = false` を設定しているため、`benches/` 配下のファイルは自動検出されない。

ファイル間でヘルパを共有する必要がある場合は、`benches/common.rs` を拡張してコード重複を避けてください。

## 検索品質の測定（probe）

Criterion は速度を測ります。順位付けの品質は BEIR データセットで [`laurus/examples/beir_multi_vector_probe.rs`](https://github.com/mosuka/laurus/blob/main/laurus/examples/beir_multi_vector_probe.rs) を使って測ります。この probe は [`MultiVector` のストレージ形式](../concepts/indexing/vector_indexing.md#multivector-ストレージ)ごとに（`Binary` を含む）、ColBERT の MaxSim による rescore の nDCG@10 を出力します。今後の方式（token pooling）もこの probe で測ります。

```bash
./scripts/fetch-beir.sh scifact          # BEIR の zip を MD5 で検証して .cache/beir/ に展開
cargo run --release --features embeddings-candle \
    --example beir_multi_vector_probe -- --dataset scifact
make beir-storage-eval BEIR_DATASET=nfcorpus   # 上の 2 手順をまとめたターゲット
```

- コーパスは `answerdotai/answerai-colbert-small-v1`（revision 固定）で 1 回だけエンコードし、f32 のトークンベクトルを `.cache/beir/<dataset>/` にキャッシュします。2 回目以降はエンコードを省きます。
- 1 つの engine にストレージ形式ごとのフィールドを置き、全形式に同じベクトルを取り込みます。第 1 段は同一で、違いはディスク上の要素形式だけです。
- 条件は 2 つです。コーパス全体に対する MaxSim（ストレージ形式の影響だけを取り出す。コーパスは `RescoreOptions::MAX_WINDOW_SIZE` に収まる必要がある）と、BM25 の上位 100 件を MaxSim で rescore するものです。
- nDCG@10 は BEIR と同じく trec_eval の `ndcg_cut_10` に従います。`--trec-run <dir>` で TREC 形式の run ファイルを書き出し、trec_eval や pytrec_eval で別途検算できます。
- F32 との差には、対応のあるブートストラップ（paired bootstrap、10,000 回、seed 固定）による 95% 信頼区間を添えます。検索時間は参考値です。

probe の指標とキャッシュのコードは `cargo test` の単体テストで検証します（ダウンロード不要）。probe 自体は CI では実行しません。

## CI 連携

現状、CI ではリグレッション検出のベンチジョブは実行していません。perf 系の PR は、推奨環境下で取得した baseline-vs-after 数値を手動で投稿することが想定されています。

将来的には大きなリグレッションで失敗するスモークセットのベンチジョブを追加する案があり、アンブレラ Issue [#429](https://github.com/mosuka/laurus/issues/429) で追跡されています。
