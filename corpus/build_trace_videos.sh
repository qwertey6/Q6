#!/usr/bin/env bash
# corpus/build_trace_videos.sh — materialize the 306 TRACE benchmark videos.
#
# Determinism contract:
#   * opencv-python / numpy / Pillow versions are pinned in requirements.txt.
#   * The TRACE video_config.json is overridden in the work tree to
#     codec=mp4v / extension=mp4 for cross-tool compatibility; recorded in
#     PROVENANCE.md. (Upstream default I420/AVI targets the legacy PEAT
#     tool and is poorly supported by modern detectors.)
#   * Codec sensitivity is exercised by the OURS-extended codec round-trip
#     set in build_extended_corpus.py, not here.
#
# Output: corpus/generated/pse-test-media/<set>/<stem>.mp4 for each of the
# 306 tests across 15 sets. Output is git-ignored and reproducible.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO_ROOT/corpus/sources/pse-test-media"
OUT_BASE="$REPO_ROOT/corpus/generated/pse-test-media"
PYTHON="${PYTHON:-python3}"

if [[ ! -d "$SRC" ]]; then
  echo "FATAL: TRACE pse-test-media not cloned. Run corpus/fetch_sources.sh first." >&2
  exit 1
fi

mkdir -p "$OUT_BASE"

# TRACE's generator hard-codes 'python' (not 'python3') and walks
# ../.. from the JSON's directory to find pattern files, so the JSON
# must live at <root>/video_creation/<set>/<stem>.json. We stage a work
# tree of that exact shape.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# 'python' -> python3 shim so subprocess calls inside make_single_video.py
# resolve correctly on systems without an unversioned `python`.
mkdir -p "$WORK/bin"
ln -sf "$(command -v "$PYTHON")" "$WORK/bin/python"
export PATH="$WORK/bin:$PATH"

# Symlink generator files at depth-1.
for f in generate_frames.py generate_video.py clean_up.py \
         make_single_video.py make_videos.py utils \
         area_patterns lum_flash_patterns red_flash_patterns \
         combo_flash_patterns; do
  ln -sf "$SRC/$f" "$WORK/$f"
done

# video_creation/ must be a real dir (so we can write videos/ into each set
# subdir) but its contents can be symlinked.
mkdir -p "$WORK/video_creation"
for d in "$SRC/video_creation/"*/; do
  set_name="$(basename "$d")"
  if [[ -d "$d" ]]; then
    cp -R "$d" "$WORK/video_creation/$set_name"
  fi
done

# Our codec override.
cat > "$WORK/video_config.json" <<'JSON'
{
    "codec": "mp4v",
    "video_extension": "mp4",
    "padding": 10,
    "comment1": "Overridden by pse-bench: H.264-family for cross-tool compatibility.",
    "comment2": "Recorded in PROVENANCE.md. Codec sensitivity exercised by OURS-extended round-trip set."
}
JSON

SETS=(
  30fps_alternating_01
  broadcast_30fps_01 broadcast_30fps_combo01 broadcast_30fps_inf01
  broadcast_30fps_inf02 broadcast_30fps_red01 broadcast_30fps_red02
  trace24_30fps_01 trace24_30fps_combo01 trace24_30fps_inf01
  trace24_30fps_red01 trace24_30fps_red02
  wcagc_30fps_area01 wcagc_30fps_area02 wcagc_30fps_area03
)

THREADS="${TRACE_BUILD_THREADS:-4}"
ONLY_SETS="${TRACE_BUILD_ONLY:-}"     # comma-separated subset for smoke runs

cd "$WORK"
for s in "${SETS[@]}"; do
  if [[ -n "$ONLY_SETS" && ",$ONLY_SETS," != *",$s,"* ]]; then
    continue
  fi
  echo "[trace] set=$s"
  "$PYTHON" make_videos.py "video_creation/$s/" --silent --max_threads "$THREADS"
  mkdir -p "$OUT_BASE/$s"
  if [[ -d "video_creation/$s/videos" ]]; then
    mv "video_creation/$s/videos/"*.mp4 "$OUT_BASE/$s/" 2>/dev/null || true
    rmdir "video_creation/$s/videos" 2>/dev/null || true
  fi
done

COUNT="$(find "$OUT_BASE" -name '*.mp4' | wc -l | tr -d ' ')"
echo "[trace] materialized $COUNT mp4 files under $OUT_BASE"

if [[ -z "$ONLY_SETS" && "$COUNT" -ne 306 ]]; then
  echo "WARN: expected 306 TRACE videos, got $COUNT — investigate per-set CSV vs JSON parity." >&2
fi
