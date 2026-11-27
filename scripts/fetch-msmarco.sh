#!/usr/bin/env bash
# Fetch the MS MARCO passage collection into .cache/msmarco/ for the
# total-hit-count probe (Issue #1472).
#
# Source: Microsoft's MS MARCO passage ranking collection,
#   https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
#   (about 1.0 GB; 8,841,823 passages as `pid<TAB>passage` lines in
#   collection.tsv).
# Terms: "The MS MARCO datasets are intended for non-commercial research
#   purposes only" (https://microsoft.github.io/msmarco/). The collection is
#   downloaded at run time and never committed.
#
# Usage:
#   ./scripts/fetch-msmarco.sh
#
# Skips the download when collection.tsv is already unpacked. Prints the
# SHA-256 of the archive, since the source publishes no checksum.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/.cache/msmarco"
URL="https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"

if [ -f "${CACHE_DIR}/collection.tsv" ]; then
    echo "[fetch-msmarco] collection.tsv: already present, skipping"
    exit 0
fi

mkdir -p "${CACHE_DIR}"
ARCHIVE="${CACHE_DIR}/collection.tar.gz"

if [ ! -f "${ARCHIVE}" ]; then
    echo "[fetch-msmarco] downloading collection.tar.gz"
    curl -fL --retry 3 --retry-delay 5 -o "${ARCHIVE}.part" "${URL}"
    mv "${ARCHIVE}.part" "${ARCHIVE}"
fi

echo "[fetch-msmarco] sha256: $(sha256sum "${ARCHIVE}" | cut -d' ' -f1)"
tar -xzf "${ARCHIVE}" -C "${CACHE_DIR}"
rm -f "${ARCHIVE}"
echo "[fetch-msmarco] passages: $(wc -l < "${CACHE_DIR}/collection.tsv")"
