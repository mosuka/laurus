# 検索結果のグループ化

長い文書を複数の chunk に分けて登録すると、1 つの文書の chunk が検索結果の
上位を埋め、他の文書がページから押し出されることがよくあります。検索結果の
グループ化（Issue #1297）は、同じフィールド値を持つ hit を 1 つのグループに
まとめ、グループごとに上位の数件だけを残します。Elasticsearch の `collapse`
や Qdrant の `group_by` と同じ機能で、lexical・vector・ハイブリッドのどの
検索でも同じように動きます。

## 基本的な使い方

`Engine::add_document` で登録した chunk は、元の文書の `_id` を共有します。
そのため `_id` でグループ化すると、各文書が最上位の chunk を代表として 1 回
ずつ返ります。

```rust
use laurus::{GroupBy, SearchRequestBuilder};

engine.add_document("manual", chapter_1).await?;
engine.add_document("manual", chapter_2).await?;
engine.add_document("faq", faq).await?;
engine.commit().await?;

let request = SearchRequestBuilder::new()
    .query_dsl("body:lifetimes")
    .group_by(GroupBy::new("_id").group_size(3))
    .limit(10)
    .build();
for result in engine.search(request).await? {
    // グループの最上位の chunk と、同じ文書の残り最大 2 件。
    println!("{} {}", result.id, result.score);
    for hit in &result.inner_hits {
        println!("  {}", hit.score);
    }
}
```

Rust API のほか、[gRPC](../../laurus-server/grpc_api.md#groupbyparams) と
[HTTP gateway](../../laurus-server/http_gateway.md#グループ化付き検索)、
[MCP の `search`・`search_batch` ツール](../../laurus-mcp/tools.md#グループ化の例)、
[`laurus search --group-by`](../../laurus-cli/commands.md#search)、
[Python](../../laurus-python/api_reference.md#groupby)・
[Node.js](../../laurus-nodejs/api_reference.md#検索結果のグループ化)・
[Ruby](../../laurus-ruby/api_reference.md#groupby)・
[PHP](../../laurus-php/api_reference.md#groupby)・
[WASM](../../laurus-wasm/api_reference.md#検索結果のグループ化) のバインディングでも
使えます。

## 仕組み

```text
クエリ ─→ 1 段目: lexical / vector / ハイブリッド（通常の検索）
              ▼
          再採点（指定したとき）
              ▼
          順位リストの hit をフィールド値でグループ化
              ▼
          グループに対する offset / limit → 文書を取得
```

グループ化は、融合と再採点の後、ページを切る前の最終的な順位リストに対して
行います。

1. グループは、そのグループの最上位の hit の順位の順に並びます。
2. グループの中では、hit は順位の順に最大 `group_size` 件まで残ります。
3. グループの先頭の hit が結果として返り、残りはその `inner_hits` になります。

グループ化は順位リストの位置だけに従い、スコアを比べません。そのため、
lexical の `sort_by` にも、再採点した hit としなかった hit でスコアの種類が
異なる再採点の結果にも、そのまま従います。

## オプション

| フィールド | 型 | 既定値 | 説明 |
| :--- | :--- | :--- | :--- |
| `field` | `String` | — | hit をグループに分けるフィールド（下記） |
| `group_size` | `usize` | `1` | 1 グループに残す hit の数（`1..=100`）。返る結果 1 件と、最大 `group_size - 1` 件の inner hit |

## グループ化に使えるフィールド

グループ化に使えるのは `_id` か、次の条件をすべて満たすスキーマのフィールド
です。

- 型が `Text`・`Integer`・`Float`・`Boolean`・`DateTime` のいずれか
- 単一値（`multi_valued: false`）
- `stored` で、`doc_values` が有効（どちらも既定で有効）

値は保存した形のまま比べます。`Text` の値は、解析した語ではなく元の文字列
全体で比べます。等しいとみなされる `Float` の値（`0.0` と `-0.0`）は同じ
グループになり、NaN 同士も同じグループになります。フィールドに値がない文書
は、`group_key` のない単独のグループになります。hit を落とさず、無関係な
文書をまとめないためです。

フィールドが存在しない、型が違う、複数値、保存されていない、`doc_values` が
無効、`group_size` が `1..=100` の範囲外、`offset + limit` が 10,000 を
超える、のいずれかに当たると、検索を始める前に
`LaurusError::InvalidArgument` で失敗します。`limit` が 0 のときは、検索を
せずに空の結果を返します。

## 結果

`limit` と `offset` は、hit ではなくグループの数を数えます。返る各
`SearchResult` はグループの最上位の hit で、次の 2 つのフィールドが入ります。

| フィールド | 型 | 説明 |
| :--- | :--- | :--- |
| `group_key` | `Option<DataValue>` | グループの hit が共有する値。値のない hit では `None` |
| `inner_hits` | `Vec<SearchResult>` | グループの 2 番目以降の hit（順位の順） |

各 inner hit は同じ `group_key` を持ち、自身の inner hit は持ちません。
ハイライトは inner hit にも付きます。グループ化しない検索では、どちらの
フィールドも空です。

## 候補の数

グループ化した検索の 1 段目は、`(offset + limit) × group_size × 2` 件の
候補（再採点の window の方が大きければ window の件数）を順位付けします。
その候補に含まれるグループが `offset + limit` に満たないときは、候補の数を
2 倍にして 1 段目をやり直します。グループが足りるか、1 段目にそれ以上の
hit がないか、候補が 10,000 件に達するまで繰り返します。クエリの埋め込みは
1 回だけで、再採点も各候補を 1 回だけ採点します。2 倍ずつ広げるので、全体の
仕事量は最後の 1 回のおよそ 2 倍に収まります。

この範囲で、グループの数は欠けません。inner hit の数は保証しません。
グループの 2 番目以降の hit も同じ候補の中から選ぶので、一致する文書がもっと
あっても、グループの hit が `group_size` 件に届かないことがあります。

`lexical_timeout_ms` は、全ラウンドの lexical 検索の時間をまとめて制限します。
広げているうちに超えると、グループ化しない検索と同じく失敗します。

グループの値は、フィールドの DocValues の列から読みます。あるフィールドで
最初にグループ化した検索は、たどった各セグメントのその列をメモリに読み込み、
列はその後の検索のために残ります。フィールドで並べ替えるときと同じです。
`_id` が 12 文字の文書 1,000,000 件では、全セグメントの `_id` の列が約 111 MB
（1 文書あたり約 111 バイト）を占めました。

## 次のステップ

- [Lexical 検索](lexical_search.md)
- [Vector 検索](vector_search.md)
- [ハイブリッド検索](hybrid_search.md)
