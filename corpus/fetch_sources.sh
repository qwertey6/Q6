#!/usr/bin/env bash
# corpus/fetch_sources.sh — fetch every upstream source at its pinned commit.
#
# Idempotent: if a clone exists, fetch + checkout to the pin instead.
# Pins come from environment.lock; if you update them, update both files together.

set -euo pipefail

CORPUS_SOURCES="$(cd "$(dirname "$0")/sources" && pwd)"
cd "$CORPUS_SOURCES"

# Format: name|url|pinned_commit
PINS=(
  "pse-test-media|https://github.com/traceRERC/pse-test-media.git|edf799a15cc1a8817a58c0120a7b25b2b28a1932"
  "pseGuidelines|https://github.com/traceRERC/pseGuidelines.git|48d0c20f22a3333f64f444159b52c8c9eb097c71"
  "IRIS|https://github.com/electronicarts/IRIS.git|d96978ac1107f3463b77f69a9c1b1ec5d45291a0"
  "VideoFlashingReduction|https://github.com/apple/VideoFlashingReduction.git|7357d2f347c8659cc5ab4804b1338cfb0e95f362"
  "IRIS-Unreal-Plugin|https://github.com/electronicarts/IRIS-Unreal-Plugin.git|85311532a588d951b833a7b942234bcc9b578bd1"
)

for entry in "${PINS[@]}"; do
  IFS='|' read -r name url pin <<< "$entry"
  if [[ -d "$name/.git" ]]; then
    echo "[exists] $name — fetching to verify pin $pin"
    git -C "$name" fetch --depth 50 origin "$pin" 2>/dev/null || git -C "$name" fetch origin
    git -C "$name" checkout -q "$pin"
  else
    echo "[clone] $name @ $pin"
    git clone --depth 50 "$url" "$name"
    git -C "$name" fetch --depth 50 origin "$pin" 2>/dev/null || git -C "$name" fetch origin
    git -C "$name" checkout -q "$pin"
  fi
done

# Apple VFR sanity: the three demo videos must be byte-identical.
APPLE_HASH_EXPECTED="896551b3857a8096d0243046ce21655f858a1e3310d5cf8b43156504b071a25b"
for v in \
    VideoFlashingReduction/VideoFlashingReduction_MATLAB/TestContent/TestVideo.mp4 \
    VideoFlashingReduction/VideoFlashingReduction_Mathematica/Resources/movie.mp4 \
    VideoFlashingReduction/VideoFlashingReduction_Xcode/VideoFlashingReduction/Resources/movie.mp4
do
  got="$(shasum -a 256 "$v" | awk '{print $1}')"
  if [[ "$got" != "$APPLE_HASH_EXPECTED" ]]; then
    echo "FATAL: Apple VFR fixture $v hash drift: $got vs expected $APPLE_HASH_EXPECTED" >&2
    exit 1
  fi
done

echo "All upstream sources pinned + verified."
