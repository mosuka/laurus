#!/usr/bin/env bash
# Fetch BEIR datasets into .cache/beir/ for the multi-vector storage probe
# (Issue #1346, laurus/examples/beir_multi_vector_probe.rs).
#
# Downloads the original BEIR zip, verifies its MD5 against the checksum
# published in the BEIR README, and unpacks it. This is the same layout
# scripts/beir_ladr_eval.py uses, so both share one download:
#   .cache/beir/<name>/corpus.jsonl
#   .cache/beir/<name>/queries.jsonl
#   .cache/beir/<name>/qrels/test.tsv
#
# Datasets (counts from the BEIR paper, Table 1):
#   - scifact:  5 183 documents /   300 test queries (relevance 1)
#   - nfcorpus: 3 633 documents /   323 test queries (relevance 1, 2)
#
# Provenance:
#   https://github.com/beir-cellar/beir (README lists the URLs and MD5s).
#   Licenses: SciFact CC BY-NC 2.0; NFCorpus "free to use for academic
#   purposes" (https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/).
#   The datasets are downloaded at run time and never committed.
#
# Usage:
#   ./scripts/fetch-beir.sh scifact
#   ./scripts/fetch-beir.sh nfcorpus
#
# Skips the download when the dataset is already unpacked.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/.cache/beir"
BASE_URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

usage() {
    echo "usage: $0 <scifact|nfcorpus>" >&2
    exit 2
}

[ $# -eq 1 ] || usage
NAME="$1"
case "${NAME}" in
    scifact) EXPECTED_MD5="5f7d1de60b170fc8027bb7898e2efca1" ;;
    nfcorpus) EXPECTED_MD5="a89dba18a62ef92f7d323ec890a0d38d" ;;
    *) usage ;;
esac

if [ -f "${CACHE_DIR}/${NAME}/corpus.jsonl" ]; then
    echo "[fetch-beir] ${NAME}: already present, skipping"
    exit 0
fi

mkdir -p "${CACHE_DIR}"
ZIP_PATH="${CACHE_DIR}/${NAME}.zip"

if [ ! -f "${ZIP_PATH}" ]; then
    echo "[fetch-beir] downloading ${NAME}.zip"
    curl -fL --retry 3 --retry-delay 5 -o "${ZIP_PATH}.part" "${BASE_URL}/${NAME}.zip"
    mv "${ZIP_PATH}.part" "${ZIP_PATH}"
fi

if command -v md5sum >/dev/null 2>&1; then
    ACTUAL_MD5="$(md5sum "${ZIP_PATH}" | cut -d' ' -f1)"
else
    ACTUAL_MD5="$(md5 -q "${ZIP_PATH}")"
fi
if [ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]; then
    echo "[fetch-beir] MD5 mismatch for ${NAME}.zip: expected ${EXPECTED_MD5}, got ${ACTUAL_MD5}" >&2
    echo "[fetch-beir] delete ${ZIP_PATH} and retry" >&2
    exit 1
fi

echo "[fetch-beir] extracting ${NAME}.zip"
unzip -q -o "${ZIP_PATH}" -d "${CACHE_DIR}"

echo "[fetch-beir] done — files under ${CACHE_DIR}/${NAME}"
