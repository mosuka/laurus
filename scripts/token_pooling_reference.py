# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy==2.5.4",
#     "scipy==1.18.1",
# ]
# ///
"""Generate SciPy reference partitions for laurus' Ward token pooling (Issue #1346).

`laurus/src/vector/core/token_pooling.rs` clusters a document's token
vectors the way Sentence Transformers' `HierarchicalTokenPooling` and PyLate
do: `scipy.cluster.hierarchy.linkage(condensed, method="ward")` on the
cosine distances `1 - cos`. This script writes a JSON fixture of row sets
with the partition SciPy's linkage gives when cut into `k` clusters; the
Rust unit test checks that its clustering reproduces every one.

- The distances are computed in plain Python in the same order as the Rust
  code (sequential f64 sums, explicit loops: `sum()` compensates on Python
  3.12+), so SciPy receives exactly the matrix the Rust code builds.
- The cut applies the first `n - k` rows of SciPy's (height-sorted)
  linkage. `fcluster(criterion="maxclust")` is not used: on tied heights it
  returns fewer than `k` clusters.
- Every case is also clustered by a plain-Python port of the Rust
  algorithm; a case where the two disagree (an exact tie decided
  differently, e.g. by fused multiply-adds in SciPy's compiled code) is
  left out and reported.
- Some cases are kept because Ward partitions them differently from the
  average, complete, single and weighted linkages, so the test pins the
  Ward update itself.

Usage:

    uv run scripts/token_pooling_reference.py \
        laurus/tests/fixtures/token_pooling/ward_scipy.json
"""

import base64
import json
import math
import struct
import sys

import numpy as np
import scipy
from scipy.cluster import hierarchy


def seq_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


def distances(rows):
    """Condensed `1 - cos` distances, as `cosine_distances` in Rust."""
    n = len(rows)
    norms = [math.sqrt(seq_sum(v * v for v in row)) for row in rows]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = seq_sum(a * b for a, b in zip(rows[i], rows[j]))
            norm = norms[i] * norms[j]
            cos = dot / norm if norm > 0.0 else 0.0
            d = 1.0 - cos
            out.append(min(max(d, 0.0), 2.0) if math.isfinite(d) else 2.0)
    return out


def condensed_index(n, i, j):
    if i > j:
        i, j = j, i
    return n * i - i * (i + 1) // 2 + (j - i - 1)


def first_appearance(roots):
    seen = {}
    return [seen.setdefault(r, len(seen)) for r in roots]


def cut_linkage(z, n, k):
    """Partition after the first `n - k` merges of SciPy's sorted linkage."""
    members = {i: [i] for i in range(n)}
    for row in range(n - k):
        a, b = int(z[row, 0]), int(z[row, 1])
        members[n + row] = members.pop(a) + members.pop(b)
    owner = [0] * n
    for node, leaves in members.items():
        for leaf in leaves:
            owner[leaf] = node
    return first_appearance(owner)


def port_labels(condensed, n, k):
    """Plain-Python port of `ward_labels` (nearest-neighbour chain)."""
    d = list(condensed)
    size = [1] * n
    chain = []
    merges = []
    for _ in range(n - 1):
        if not chain:
            chain.append(next(i for i in range(n) if size[i] > 0))
        while True:
            x = chain[-1]
            if len(chain) > 1:
                y = chain[-2]
                nearest = d[condensed_index(n, x, y)]
            else:
                y, nearest = None, math.inf
            for i in range(n):
                if size[i] == 0 or i == x:
                    continue
                dist = d[condensed_index(n, x, i)]
                if dist < nearest:
                    nearest, y = dist, i
            if len(chain) > 1 and y == chain[-2]:
                break
            chain.append(y)
        del chain[-2:]
        if x > y:
            x, y = y, x
        nx, ny = size[x], size[y]
        merges.append((nearest, x, y))
        size[x] = 0
        size[y] = nx + ny
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue
            t = 1.0 / (nx + ny + ni)
            d_xi = d[condensed_index(n, i, x)]
            d_yi = d[condensed_index(n, i, y)]
            v = (ni + nx) * t * d_xi * d_xi + (ni + ny) * t * d_yi * d_yi - ni * t * nearest * nearest
            d[condensed_index(n, i, y)] = math.sqrt(max(v, 0.0))
    order = sorted(range(len(merges)), key=lambda m: merges[m][0])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for m in order[: n - k]:
        _, x, y = merges[m]
        parent[find(x)] = find(y)
    return first_appearance([find(i) for i in range(n)])


def method_labels(condensed, n, k, method):
    return cut_linkage(hierarchy.linkage(np.array(condensed), method=method), n, k)


def rows_for(rng, n, dimension, normalize):
    centers = rng.normal(size=(max(2, n // 4), dimension))
    picks = rng.integers(0, len(centers), size=n)
    rows = centers[picks] + 0.35 * rng.normal(size=(n, dimension))
    if normalize:
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return [[float(np.float32(v)) for v in row] for row in rows]


def main():
    out_path = sys.argv[1]
    cases, skipped, distinct = [], 0, 0
    for seed in (1346, 2409):
        rng = np.random.default_rng(seed)
        for dimension in (2, 4, 8):
            for n in (3, 5, 8, 13, 21, 34):
                for factor in (2, 3):
                    k = max(n // factor, 1)
                    if k >= n:
                        continue
                    rows = rows_for(rng, n, dimension, normalize=(n % 2 == 1))
                    condensed = distances(rows)
                    ward = method_labels(condensed, n, k, "ward")
                    if port_labels(condensed, n, k) != ward:
                        skipped += 1
                        continue
                    others = [
                        method_labels(condensed, n, k, m)
                        for m in ("average", "complete", "single", "weighted")
                    ]
                    distinct += all(o != ward for o in others)
                    packed = struct.pack(f"<{n * dimension}f", *[v for row in rows for v in row])
                    cases.append(
                        {
                            "dimension": dimension,
                            "clusters": k,
                            "rows": base64.b64encode(packed).decode(),
                            "labels": ward,
                        }
                    )
    assert distinct >= 5, f"only {distinct} cases tell Ward from the other linkages"
    fixture = {
        "generator": "scripts/token_pooling_reference.py",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cases": cases,
    }
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")
    print(f"{len(cases)} cases ({distinct} where Ward differs from every other linkage), "
          f"{skipped} skipped on SciPy/port disagreement", file=sys.stderr)


if __name__ == "__main__":
    main()
