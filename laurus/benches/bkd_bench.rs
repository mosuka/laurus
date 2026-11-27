//! Criterion benchmarks for the BKD tree.
//!
//! Covers `BKDReader::range_search` (the legacy axis-aligned API) and
//! `BKDReader::intersect` (the visitor-driven primitive that 3D distance,
//! k-NN, and other custom-shape queries will be built on) across 1D, 2D,
//! and 3D datasets at 10k / 100k / 1M points, and, through a `LexicalStore`
//! of 100k points (Issue #1596): `GeoPolygonQuery` against
//! `GeoBoundingBoxQuery` over the same extent, and `GeoPolylineQuery` and
//! `Geo3dPolylineQuery` against the distance query of their field type.
//!
//! # Running
//!
//! Run every bench in this file:
//!
//! ```sh
//! cargo bench --bench bkd_bench
//! ```
//!
//! Filter by group / case (substring match against the criterion id):
//!
//! ```sh
//! cargo bench --bench bkd_bench -- range_search/3d/100000
//! cargo bench --bench bkd_bench -- intersect_counting
//! cargo bench --bench bkd_bench -- geo_polygon_count
//! cargo bench --bench bkd_bench -- polyline_count
//! ```
//!
//! Compile-only smoke check (skips the runtime, used by CI):
//!
//! ```sh
//! cargo bench --bench bkd_bench --no-run
//! ```
//!
//! All trees are built once per `(num_dims, n)` combination and reused
//! across iterations. Synthetic data is generated with an inline LCG so
//! the benches stay deterministic without pulling in `rand`.

mod common;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use laurus::lexical::core::field::{FieldOption, Geo3dOption, GeoOption};
use laurus::lexical::index::structures::aabb::AABB;
use laurus::lexical::index::structures::bkd_tree::{BKDReader, BKDTree, BKDWriter};
use laurus::lexical::index::structures::visitor::{CellRelation, IntersectVisitor};
use laurus::lexical::query::{
    Geo3dDistanceQuery, Geo3dPolyline, Geo3dPolylineQuery, GeoBoundingBoxQuery, GeoDistanceQuery,
    GeoPoint, GeoPolygonQuery, GeoPolylineQuery, Query,
};
use laurus::lexical::{LexicalIndexConfig, LexicalSearchRequest, LexicalStore};
use laurus::storage::Storage;
use laurus::storage::memory::{MemoryStorage, MemoryStorageConfig};
use laurus::util::ecef::wgs84_to_ecef;
use laurus::{DataValue, Document};
use std::hint::black_box;
use std::sync::Arc;

use common::{DEFAULT_SEED, lcg_next};

/// Build an in-memory BKD tree of `n` random points in `num_dims` dimensions
/// with coordinates uniformly distributed in `[0, 1000)`. Returns the open
/// `BKDReader` ready for query benchmarking.
fn build_tree(num_dims: usize, n: usize) -> BKDReader {
    let storage: Arc<MemoryStorage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let mut points: Vec<f64> = Vec::with_capacity(n * num_dims);
    let mut doc_ids: Vec<u64> = Vec::with_capacity(n);
    let mut rng_state: u64 = DEFAULT_SEED;
    for i in 0..n {
        for _ in 0..num_dims {
            points.push(lcg_next(&mut rng_state));
        }
        doc_ids.push(i as u64);
    }
    let path = format!("bench_{num_dims}d_{n}.bkd");
    let output = storage.create_output(&path).unwrap();
    let mut writer = BKDWriter::new(output, num_dims as u32);
    writer.write(&points, &doc_ids).unwrap();
    writer.finish().unwrap();
    BKDReader::open(storage, &path).unwrap()
}

/// Visitor that just counts hits, used to benchmark the raw `intersect`
/// path without paying the `RangeQueryVisitor` boundary-handling cost.
struct CountingVisitor {
    query: AABB,
    count: u64,
}

impl IntersectVisitor for CountingVisitor {
    fn compare(&self, cell: &AABB) -> CellRelation {
        let qmin = self.query.min();
        let qmax = self.query.max();
        let cmin = cell.min();
        let cmax = cell.max();
        for d in 0..cell.num_dims() {
            if cmax[d] < qmin[d] || cmin[d] > qmax[d] {
                return CellRelation::Outside;
            }
        }
        for d in 0..cell.num_dims() {
            if cmin[d] < qmin[d] || cmax[d] > qmax[d] {
                return CellRelation::Crosses;
            }
        }
        CellRelation::Inside
    }
    fn visit_inside(&mut self, _doc_id: u64) {
        self.count += 1;
    }
    fn visit(&mut self, _doc_id: u64, point: &[f64]) {
        if self.query.contains_point(point) {
            self.count += 1;
        }
    }
}

/// Sizes to sweep across the dimensionality axis.
const SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

/// Dimensionalities to sweep.
const DIMS: &[usize] = &[1, 2, 3];

/// Narrow (selective) query: 1% of the per-dimension data range.
/// Hit counts: ~1% × n in 1D, ~0.01% × n in 2D, ~0.0001% × n in 3D.
const QUERY_NARROW_LO: f64 = 100.0;
const QUERY_NARROW_HI: f64 = 110.0;

/// Wide query: 50% of the per-dimension data range.
/// Hit counts: ~50% × n in 1D, ~25% × n in 2D, ~12.5% × n in 3D.
const QUERY_WIDE_LO: f64 = 100.0;
const QUERY_WIDE_HI: f64 = 600.0;

fn bench_range_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("bkd_range_search");
    for &num_dims in DIMS {
        for &n in SIZES {
            let reader = build_tree(num_dims, n);
            for (label, lo, hi) in [
                ("narrow", QUERY_NARROW_LO, QUERY_NARROW_HI),
                ("wide", QUERY_WIDE_LO, QUERY_WIDE_HI),
            ] {
                let mins: Vec<Option<f64>> = (0..num_dims).map(|_| Some(lo)).collect();
                let maxs: Vec<Option<f64>> = (0..num_dims).map(|_| Some(hi)).collect();
                let id = format!("{num_dims}d/{label}/{n}");
                group.throughput(Throughput::Elements(n as u64));
                group.bench_with_input(BenchmarkId::from_parameter(id), &(mins, maxs), |b, qs| {
                    let (mins, maxs) = qs;
                    b.iter(|| {
                        let hits = reader
                            .range_search(black_box(mins), black_box(maxs), true, true)
                            .unwrap();
                        black_box(hits);
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_intersect_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("bkd_intersect_counting");
    for &num_dims in DIMS {
        for &n in SIZES {
            let reader = build_tree(num_dims, n);
            for (label, lo, hi) in [
                ("narrow", QUERY_NARROW_LO, QUERY_NARROW_HI),
                ("wide", QUERY_WIDE_LO, QUERY_WIDE_HI),
            ] {
                let qmin: Vec<f64> = (0..num_dims).map(|_| lo).collect();
                let qmax: Vec<f64> = (0..num_dims).map(|_| hi).collect();
                let id = format!("{num_dims}d/{label}/{n}");
                group.throughput(Throughput::Elements(n as u64));
                group.bench_with_input(BenchmarkId::from_parameter(id), &(qmin, qmax), |b, qs| {
                    let (qmin, qmax) = qs;
                    b.iter(|| {
                        let mut v = CountingVisitor {
                            query: AABB::new(qmin.clone(), qmax.clone()).unwrap(),
                            count: 0,
                        };
                        reader.intersect(&mut v).unwrap();
                        black_box(v.count);
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_build(c: &mut Criterion) {
    // Tree construction itself, useful as a baseline and to gauge the
    // amortized cost of widest-axis splitting + AABB computation
    // introduced in #291–#293.
    let mut group = c.benchmark_group("bkd_build");
    // Smaller sweep: building 1M points × 3 dims repeatedly is slow.
    for &num_dims in DIMS {
        for &n in &[10_000usize, 100_000] {
            let id = format!("{num_dims}d/{n}");
            group.throughput(Throughput::Elements(n as u64));
            group.bench_function(BenchmarkId::from_parameter(id), |b| {
                // Pre-generate input outside the iter loop so we measure
                // build, not data generation.
                let mut points: Vec<f64> = Vec::with_capacity(n * num_dims);
                let mut doc_ids: Vec<u64> = Vec::with_capacity(n);
                let mut rng_state: u64 = DEFAULT_SEED;
                for i in 0..n {
                    for _ in 0..num_dims {
                        points.push(lcg_next(&mut rng_state));
                    }
                    doc_ids.push(i as u64);
                }
                b.iter(|| {
                    let storage: Arc<MemoryStorage> =
                        Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
                    let path = "build.bkd";
                    let output = storage.create_output(path).unwrap();
                    let mut writer = BKDWriter::new(output, num_dims as u32);
                    writer.write(&points, &doc_ids).unwrap();
                    writer.finish().unwrap();
                    black_box(storage);
                });
            });
        }
    }
    group.finish();
}

/// A `LexicalStore` of `n` random `Geo` points in latitudes `[30, 45)` and
/// longitudes `[130, 145)`, in one segment with a BKD tree.
fn build_geo_store(n: usize) -> LexicalStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field("location", FieldOption::Geo(GeoOption::default()))
        .build();
    let store = LexicalStore::new(storage, config).unwrap();
    let mut rng_state: u64 = DEFAULT_SEED;
    for id in 0..n as u64 {
        let lat = 30.0 + lcg_next(&mut rng_state) * 0.015;
        let lon = 130.0 + lcg_next(&mut rng_state) * 0.015;
        let doc = Document::builder()
            .add_field("location", DataValue::Geo(GeoPoint::new(lat, lon)))
            .build();
        store.upsert_document(id, doc).unwrap();
    }
    store.commit().unwrap();
    store
}

/// A regular polygon of `sides` vertices around `(lat, lon)`, `radius`
/// degrees from it.
fn regular_polygon(lat: f64, lon: f64, radius: f64, sides: usize) -> Vec<(f64, f64)> {
    (0..sides)
        .map(|i| {
            let angle = std::f64::consts::TAU * i as f64 / sides as f64;
            (lat + radius * angle.sin(), lon + radius * angle.cos())
        })
        .collect()
}

/// `GeoPolygonQuery` against `GeoBoundingBoxQuery` over the same extent,
/// 35 to 40 degrees of latitude and 135 to 140 of longitude (about 11% of
/// the points): the box, the square as a polygon, the square with a hole,
/// and a 64-sided polygon inside the box. The polygon takes the cells inside
/// it whole, and the box tests every point of the cells it reaches.
fn bench_geo_polygon_count(c: &mut Criterion) {
    let n = 100_000;
    let store = build_geo_store(n);
    let square = [(35.0, 135.0), (35.0, 140.0), (40.0, 140.0), (40.0, 135.0)];
    let hole = [(36.0, 136.0), (36.0, 139.0), (39.0, 139.0), (39.0, 136.0)];
    let circle = regular_polygon(37.5, 137.5, 2.5, 64);
    let queries: Vec<(&str, Box<dyn Query>)> = vec![
        (
            "bbox",
            Box::new(
                GeoBoundingBoxQuery::within_bounding_box("location", 35.0, 135.0, 40.0, 140.0)
                    .unwrap(),
            ),
        ),
        (
            "polygon_square",
            Box::new(GeoPolygonQuery::within_polygon("location", &square, &[]).unwrap()),
        ),
        (
            "polygon_square_with_a_hole",
            Box::new(GeoPolygonQuery::within_polygon("location", &square, &[&hole]).unwrap()),
        ),
        (
            "polygon_64_sides",
            Box::new(GeoPolygonQuery::within_polygon("location", &circle, &[]).unwrap()),
        ),
    ];

    let mut group = c.benchmark_group("geo_polygon_count");
    group.throughput(Throughput::Elements(n as u64));
    for (label, query) in &queries {
        group.bench_function(BenchmarkId::from_parameter(format!("{label}/{n}")), |b| {
            b.iter(|| {
                let request = LexicalSearchRequest::new(query.clone_box());
                black_box(store.count(black_box(request)).unwrap());
            });
        });
    }
    group.finish();
}

/// A `LexicalStore` of the points of [`build_geo_store`] on the surface in
/// ECEF, in a `Geo3d` field `position`, in one segment with a BKD tree.
fn build_geo3d_store(n: usize) -> LexicalStore {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
    let config = LexicalIndexConfig::builder()
        .add_field("position", FieldOption::Geo3d(Geo3dOption::default()))
        .build();
    let store = LexicalStore::new(storage, config).unwrap();
    let mut rng_state: u64 = DEFAULT_SEED;
    for id in 0..n as u64 {
        let lat = 30.0 + lcg_next(&mut rng_state) * 0.015;
        let lon = 130.0 + lcg_next(&mut rng_state) * 0.015;
        let doc = Document::builder()
            .add_field("position", DataValue::GeoEcef(wgs84_to_ecef(lat, lon, 0.0)))
            .build();
        store.upsert_document(id, doc).unwrap();
    }
    store.commit().unwrap();
    store
}

/// A zigzag of `vertices` vertices across latitudes 33 to 42 and longitudes
/// 133 to 142.
fn zigzag(vertices: usize) -> Vec<(f64, f64)> {
    (0..vertices)
        .map(|i| {
            let t = i as f64 / (vertices - 1).max(1) as f64;
            (if i % 2 == 0 { 33.0 } else { 42.0 }, 133.0 + 9.0 * t)
        })
        .collect()
}

/// Count `queries` over `store` in the group `name`.
fn bench_counts(
    c: &mut Criterion,
    name: &str,
    n: usize,
    store: &LexicalStore,
    queries: &[(&str, Box<dyn Query>)],
) {
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(n as u64));
    for (label, query) in queries {
        group.bench_function(BenchmarkId::from_parameter(format!("{label}/{n}")), |b| {
            b.iter(|| {
                let request = LexicalSearchRequest::new(query.clone_box());
                black_box(store.count(black_box(request)).unwrap());
            });
        });
    }
    group.finish();
}

/// `GeoPolylineQuery` within 50 km against `GeoDistanceQuery` of the same
/// radius: a single vertex (the circle), 2 vertices, and zigzags of 4 and 16
/// vertices across the points. The polyline prunes by each edge's bounds,
/// so a diagonal edge visits the cells of its whole bounding box.
fn bench_geo_polyline_count(c: &mut Criterion) {
    let n = 100_000;
    let store = build_geo_store(n);
    let distance_m = 50_000.0;
    let polyline = |vertices: &[(f64, f64)]| -> Box<dyn Query> {
        Box::new(GeoPolylineQuery::within_distance("location", vertices, distance_m).unwrap())
    };
    let queries: Vec<(&str, Box<dyn Query>)> = vec![
        (
            "geo_distance",
            Box::new(GeoDistanceQuery::within_radius("location", 37.5, 137.5, distance_m).unwrap()),
        ),
        ("polyline_1_vertex", polyline(&[(37.5, 137.5)])),
        (
            "polyline_2_vertices",
            polyline(&[(35.0, 135.0), (40.0, 140.0)]),
        ),
        ("polyline_4_vertices", polyline(&zigzag(4))),
        ("polyline_16_vertices", polyline(&zigzag(16))),
    ];
    bench_counts(c, "geo_polyline_count", n, &store, &queries);
}

/// `Geo3dPolylineQuery` within 50 km against `Geo3dDistanceQuery` of the
/// same radius, over the points of [`bench_geo_polyline_count`] in ECEF.
fn bench_geo3d_polyline_count(c: &mut Criterion) {
    let n = 100_000;
    let store = build_geo3d_store(n);
    let distance_m = 50_000.0;
    let polyline = |vertices: &[(f64, f64)]| -> Box<dyn Query> {
        let path = Geo3dPolyline::new(
            vertices
                .iter()
                .map(|&(lat, lon)| wgs84_to_ecef(lat, lon, 0.0))
                .collect(),
        )
        .unwrap();
        Box::new(Geo3dPolylineQuery::new("position", path, distance_m))
    };
    let queries: Vec<(&str, Box<dyn Query>)> = vec![
        (
            "geo3d_distance",
            Box::new(Geo3dDistanceQuery::new(
                "position",
                wgs84_to_ecef(37.5, 137.5, 0.0),
                distance_m,
            )),
        ),
        ("polyline_1_vertex", polyline(&[(37.5, 137.5)])),
        (
            "polyline_2_vertices",
            polyline(&[(35.0, 135.0), (40.0, 140.0)]),
        ),
        ("polyline_4_vertices", polyline(&zigzag(4))),
        ("polyline_16_vertices", polyline(&zigzag(16))),
    ];
    bench_counts(c, "geo3d_polyline_count", n, &store, &queries);
}

criterion_group!(
    benches,
    bench_range_search,
    bench_intersect_counting,
    bench_build,
    bench_geo_polygon_count,
    bench_geo_polyline_count,
    bench_geo3d_polyline_count
);
criterion_main!(benches);
