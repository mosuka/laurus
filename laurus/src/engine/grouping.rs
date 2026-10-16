//! Result grouping for [`Engine::search`](super::Engine::search)
//! (Issue #1297).
//!
//! A grouped search ranks a candidate pool as usual, then folds the ranked
//! hits that share a field value into groups, in rank order. When the pool
//! holds too few groups, the search ranks a larger pool and groups again.
//! This module holds the parts that do not touch the stores: checking the
//! request, the pool schedule, the rule that says the first stage has run
//! out, and the grouping itself.

use ahash::AHashMap;

use crate::data::DataValue;
use crate::engine::schema::{FieldOption, RESERVED_ID_FIELD, Schema};
use crate::engine::search::GroupBy;
use crate::error::{LaurusError, Result};

/// Check `group_by` against the schema and the requested page.
///
/// `_id` is always accepted: every document stores it. Any other field must
/// be a `Text`, `Integer`, `Float`, `Boolean` or `DateTime` field with one
/// value per document, `stored`, with `doc_values`. Without a DocValues
/// column every candidate's key would be read from its stored document.
///
/// # Arguments
///
/// * `schema` - The engine's schema.
/// * `group_by` - The grouping request.
/// * `offset` - The request's offset, in groups.
/// * `limit` - The request's limit, in groups.
///
/// # Errors
///
/// Returns [`LaurusError::InvalidArgument`] when `group_size` is outside
/// `1..=GroupBy::MAX_GROUP_SIZE`, `offset + limit` exceeds
/// [`GroupBy::MAX_CANDIDATES`], or the field cannot group results.
pub(super) fn validate(
    schema: &Schema,
    group_by: &GroupBy,
    offset: usize,
    limit: usize,
) -> Result<()> {
    if !(1..=GroupBy::MAX_GROUP_SIZE).contains(&group_by.group_size) {
        return Err(invalid(format!(
            "group_size must be between 1 and {}, got {}",
            GroupBy::MAX_GROUP_SIZE,
            group_by.group_size
        )));
    }
    let page_end = offset.saturating_add(limit);
    if page_end > GroupBy::MAX_CANDIDATES {
        return Err(invalid(format!(
            "offset + limit must be at most {}, got {page_end}",
            GroupBy::MAX_CANDIDATES
        )));
    }

    let field = group_by.field.as_str();
    if field == RESERVED_ID_FIELD {
        return Ok(());
    }
    let option = match schema.fields.get(field) {
        Some(
            option @ (FieldOption::Text(_)
            | FieldOption::Integer(_)
            | FieldOption::Float(_)
            | FieldOption::Boolean(_)
            | FieldOption::DateTime(_)),
        ) => option,
        Some(_) => {
            return Err(invalid(format!(
                "field '{field}' cannot group results; use _id or a Text, Integer, Float, \
                 Boolean or DateTime field"
            )));
        }
        None => return Err(invalid(format!("unknown field '{field}'"))),
    };
    let Some(option) = option.to_lexical() else {
        return Err(invalid(format!("field '{field}' cannot group results")));
    };
    if option.multi_valued() {
        return Err(invalid(format!(
            "field '{field}' is multi_valued; a grouping field holds one value per document"
        )));
    }
    if !option.stored() || option.doc_values() != Some(true) {
        return Err(invalid(format!(
            "field '{field}' must be stored with doc_values enabled to group results"
        )));
    }
    Ok(())
}

fn invalid(reason: String) -> LaurusError {
    LaurusError::invalid_argument(format!("group_by: {reason}"))
}

/// The first round's candidate count: twice as many hits as `needed`
/// groups of `group_size` hits hold, between 1 and
/// [`GroupBy::MAX_CANDIDATES`].
///
/// # Arguments
///
/// * `needed` - `offset + limit`, in groups.
/// * `group_size` - The hits kept per group.
pub(super) fn first_request(needed: usize, group_size: usize) -> usize {
    needed
        .saturating_mul(group_size)
        .saturating_mul(2)
        .clamp(1, GroupBy::MAX_CANDIDATES)
}

/// The next round's candidate count after a round that ranked
/// `fetch_count` candidates: twice as many, up to
/// [`GroupBy::MAX_CANDIDATES`]. `None` once a round reached the cap.
///
/// Doubling what the round actually ranked, rather than what it was asked
/// for, keeps a rescore window larger than the request from repeating the
/// same round.
///
/// # Arguments
///
/// * `fetch_count` - How many candidates the last round ranked.
pub(super) fn next_request(fetch_count: usize) -> Option<usize> {
    (fetch_count < GroupBy::MAX_CANDIDATES)
        .then(|| fetch_count.saturating_mul(2).min(GroupBy::MAX_CANDIDATES))
}

/// How many hits one first-stage list was asked for, and how many it
/// returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ListCount {
    /// The list's limit.
    pub(super) requested: usize,
    /// The hits it returned.
    pub(super) returned: usize,
}

/// Whether a larger pool cannot add candidates: every list of the first
/// stage has run out.
///
/// A lexical list has run out when it returned fewer hits than requested:
/// its collector applies `min_score` and the sort itself, so a short list
/// holds every match. A vector list must also have returned no more than in
/// the previous round: a filtered or deletion-heavy HNSW search stops after
/// a number of visits that grows with the requested count, so a short list
/// can still grow.
///
/// # Arguments
///
/// * `lexical` - This round's lexical list, if the search has one.
/// * `vector` - This round's vector list, if the search has one.
/// * `previous_vector` - The previous round's vector list, if any.
pub(super) fn first_stage_exhausted(
    lexical: Option<ListCount>,
    vector: Option<ListCount>,
    previous_vector: Option<ListCount>,
) -> bool {
    let lexical_done = lexical.is_none_or(|list| list.returned < list.requested);
    let vector_done = vector.is_none_or(|list| {
        list.returned < list.requested
            && previous_vector.is_some_and(|previous| list.returned <= previous.returned)
    });
    lexical_done && vector_done
}

/// One group of ranked hits.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Group {
    /// The value the group's hits share; `None` for a hit without a value,
    /// which forms a group of its own.
    pub(super) key: Option<DataValue>,
    /// The group's hits as `(doc id, score)`, best first.
    pub(super) hits: Vec<(u64, f32)>,
}

/// The hashable form of a group value. [`DataValue`] is only `PartialEq`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GroupKey {
    Text(String),
    Int(i64),
    /// The value's bits, with `-0.0` folded into `0.0` and every NaN into
    /// one, so values that compare equal share a group.
    Float(u64),
    Bool(bool),
    /// Microseconds since the epoch, the precision DateTime values keep.
    DateTime(i64),
}

impl GroupKey {
    /// The key of `value`; `None` for a value that cannot group (an array
    /// from data written before the field became single-valued, or any
    /// other non-scalar).
    fn of(value: &DataValue) -> Option<Self> {
        match value {
            DataValue::Text(text) => Some(Self::Text(text.clone())),
            DataValue::Int64(value) => Some(Self::Int(*value)),
            DataValue::Float64(value) => Some(Self::Float(if *value == 0.0 {
                0f64.to_bits()
            } else if value.is_nan() {
                f64::NAN.to_bits()
            } else {
                value.to_bits()
            })),
            DataValue::Bool(value) => Some(Self::Bool(*value)),
            DataValue::DateTime(value) => Some(Self::DateTime(value.timestamp_micros())),
            _ => None,
        }
    }
}

/// Fold `ranked` (best first) into groups, in rank order.
///
/// A group is ordered by its best hit, and keeps up to `group_size` hits in
/// rank order. A hit whose document has no value, or a value that cannot
/// group, forms a group of its own. At most `needed` groups are formed;
/// later hits still join those groups until each is full.
///
/// # Arguments
///
/// * `ranked` - The ranked hits as `(doc id, score)`, best first.
/// * `keys` - Each hit's group value, `None` for no value. A hit missing
///   from the map counts as having no value.
/// * `group_size` - The most hits kept per group.
/// * `needed` - The most groups formed.
pub(super) fn group_ranked(
    ranked: &[(u64, f32)],
    keys: &AHashMap<u64, Option<DataValue>>,
    group_size: usize,
    needed: usize,
) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    let mut index: AHashMap<GroupKey, usize> = AHashMap::new();
    for &(doc_id, score) in ranked {
        let value = keys.get(&doc_id).and_then(Option::as_ref);
        let key = value.and_then(GroupKey::of);
        if let Some(&i) = key.as_ref().and_then(|key| index.get(key)) {
            if groups[i].hits.len() < group_size {
                groups[i].hits.push((doc_id, score));
            }
            continue;
        }
        if groups.len() == needed {
            continue;
        }
        let value = match key {
            Some(key) => {
                index.insert(key, groups.len());
                value.cloned()
            }
            None => None,
        };
        groups.push(Group {
            key: value,
            hits: vec![(doc_id, score)],
        });
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::schema::{FieldOption, Schema};
    use crate::lexical::core::field::{IntegerOption, TextOption};
    use crate::vector::core::field::FlatOption;

    fn keys(entries: &[(u64, Option<DataValue>)]) -> AHashMap<u64, Option<DataValue>> {
        entries.iter().cloned().collect()
    }

    fn text(value: &str) -> Option<DataValue> {
        Some(DataValue::Text(value.to_string()))
    }

    fn doc_ids(group: &Group) -> Vec<u64> {
        group.hits.iter().map(|(doc_id, _)| *doc_id).collect()
    }

    #[test]
    fn groups_follow_their_best_hit_and_keep_rank_order_inside() {
        let ranked = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)];
        let keys = keys(&[
            (1, text("a")),
            (2, text("b")),
            (3, text("a")),
            (4, text("c")),
            (5, text("a")),
        ]);
        let groups = group_ranked(&ranked, &keys, 2, 10);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].key, text("a"));
        assert_eq!(doc_ids(&groups[0]), vec![1, 3]);
        assert_eq!(doc_ids(&groups[1]), vec![2]);
        assert_eq!(doc_ids(&groups[2]), vec![4]);
    }

    #[test]
    fn group_size_one_keeps_only_the_best_hit() {
        let ranked = [(1, 0.9), (2, 0.8), (3, 0.7)];
        let keys = keys(&[(1, text("a")), (2, text("a")), (3, text("b"))]);
        let groups = group_ranked(&ranked, &keys, 1, 10);

        assert_eq!(
            groups.iter().map(doc_ids).collect::<Vec<_>>(),
            vec![vec![1], vec![3]]
        );
    }

    #[test]
    fn a_hit_without_a_value_forms_its_own_group() {
        let ranked = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)];
        // 2 has no value, 3 is missing from the map, 4 holds an array.
        let keys = keys(&[
            (1, text("a")),
            (2, None),
            (4, Some(DataValue::Int64Array(vec![1, 2]))),
        ]);
        let groups = group_ranked(&ranked, &keys, 3, 10);

        assert_eq!(
            groups.iter().map(doc_ids).collect::<Vec<_>>(),
            vec![vec![1], vec![2], vec![3], vec![4]]
        );
        assert!(groups[1..].iter().all(|group| group.key.is_none()));
    }

    #[test]
    fn stops_forming_groups_at_needed_but_still_fills_them() {
        let ranked = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)];
        let keys = keys(&[
            (1, text("a")),
            (2, text("b")),
            (3, text("c")),
            (4, text("a")),
        ]);
        let groups = group_ranked(&ranked, &keys, 2, 2);

        assert_eq!(
            groups.iter().map(doc_ids).collect::<Vec<_>>(),
            vec![vec![1, 4], vec![2]]
        );
    }

    #[test]
    fn equal_floats_share_a_group() {
        let ranked = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)];
        let keys = keys(&[
            (1, Some(DataValue::Float64(0.0))),
            (2, Some(DataValue::Float64(-0.0))),
            (3, Some(DataValue::Float64(f64::NAN))),
            (4, Some(DataValue::Float64(-f64::NAN))),
        ]);
        let groups = group_ranked(&ranked, &keys, 2, 10);

        assert_eq!(
            groups.iter().map(doc_ids).collect::<Vec<_>>(),
            vec![vec![1, 2], vec![3, 4]]
        );
    }

    #[test]
    fn the_first_request_scales_with_the_page_and_is_capped() {
        assert_eq!(first_request(10, 1), 20);
        assert_eq!(first_request(10, 3), 60);
        assert_eq!(first_request(0, 5), 1);
        assert_eq!(first_request(9_000, 2), GroupBy::MAX_CANDIDATES);
        assert_eq!(
            first_request(usize::MAX, usize::MAX),
            GroupBy::MAX_CANDIDATES
        );
    }

    #[test]
    fn the_next_request_doubles_what_was_ranked_until_the_cap() {
        assert_eq!(next_request(20), Some(40));
        // A rescore window of 100 ranked 100, not the 20 requested: the next
        // round asks for 200 instead of repeating 100.
        assert_eq!(next_request(100), Some(200));
        assert_eq!(next_request(6_000), Some(GroupBy::MAX_CANDIDATES));
        assert_eq!(next_request(GroupBy::MAX_CANDIDATES), None);
    }

    fn count(requested: usize, returned: usize) -> Option<ListCount> {
        Some(ListCount {
            requested,
            returned,
        })
    }

    #[test]
    fn a_short_lexical_list_has_run_out() {
        assert!(first_stage_exhausted(count(20, 7), None, None));
        assert!(!first_stage_exhausted(count(20, 20), None, None));
    }

    #[test]
    fn a_short_vector_list_runs_out_only_when_it_stopped_growing() {
        // First round: short, but a wider search might visit more nodes.
        assert!(!first_stage_exhausted(None, count(20, 7), None));
        // Grew since the last round: keep widening.
        assert!(!first_stage_exhausted(None, count(40, 12), count(20, 7)));
        // Did not grow: run out.
        assert!(first_stage_exhausted(None, count(40, 12), count(20, 12)));
        // A full list never ran out.
        assert!(!first_stage_exhausted(None, count(40, 40), count(20, 20)));
    }

    #[test]
    fn a_hybrid_search_runs_out_when_both_lists_have() {
        assert!(!first_stage_exhausted(
            count(40, 3),
            count(40, 40),
            count(20, 20)
        ));
        assert!(!first_stage_exhausted(
            count(40, 40),
            count(40, 9),
            count(20, 9)
        ));
        assert!(first_stage_exhausted(
            count(40, 3),
            count(40, 9),
            count(20, 9)
        ));
    }

    #[test]
    fn no_first_stage_has_run_out() {
        assert!(first_stage_exhausted(None, None, None));
    }

    fn schema() -> Schema {
        let mut schema = Schema::new();
        schema
            .fields
            .insert("parent".into(), FieldOption::Text(TextOption::default()));
        schema.fields.insert(
            "tags".into(),
            FieldOption::Text(TextOption {
                multi_valued: true,
                ..Default::default()
            }),
        );
        schema.fields.insert(
            "unstored".into(),
            FieldOption::Integer(IntegerOption {
                stored: false,
                ..Default::default()
            }),
        );
        schema.fields.insert(
            "no_doc_values".into(),
            FieldOption::Integer(IntegerOption {
                doc_values: false,
                ..Default::default()
            }),
        );
        schema
            .fields
            .insert("embedding".into(), FieldOption::Flat(FlatOption::default()));
        schema
    }

    fn check(field: &str, group_size: usize, offset: usize, limit: usize) -> Result<()> {
        validate(
            &schema(),
            &GroupBy::new(field).group_size(group_size),
            offset,
            limit,
        )
    }

    fn rejects(result: Result<()>, fragment: &str) {
        match result {
            Err(LaurusError::InvalidArgument(message)) => {
                assert!(message.contains(fragment), "{message}");
            }
            other => panic!("expected InvalidArgument containing {fragment:?}, got {other:?}"),
        }
    }

    #[test]
    fn validate_accepts_id_and_single_valued_stored_doc_values_fields() {
        assert!(check("_id", 1, 0, 10).is_ok());
        assert!(check("parent", GroupBy::MAX_GROUP_SIZE, 0, 10).is_ok());
        assert!(check("parent", 1, GroupBy::MAX_CANDIDATES - 10, 10).is_ok());
    }

    #[test]
    fn validate_rejects_bad_requests() {
        rejects(check("parent", 0, 0, 10), "group_size");
        rejects(
            check("parent", GroupBy::MAX_GROUP_SIZE + 1, 0, 10),
            "group_size",
        );
        rejects(
            check("parent", 1, GroupBy::MAX_CANDIDATES, 1),
            "offset + limit",
        );
        rejects(check("parent", 1, usize::MAX, 1), "offset + limit");
        rejects(check("missing", 1, 0, 10), "unknown field");
        rejects(check("embedding", 1, 0, 10), "cannot group");
        rejects(check("tags", 1, 0, 10), "multi_valued");
        rejects(check("unstored", 1, 0, 10), "doc_values");
        rejects(check("no_doc_values", 1, 0, 10), "doc_values");
    }
}
