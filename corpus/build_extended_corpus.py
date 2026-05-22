#!/usr/bin/env python3
"""corpus/build_extended_corpus.py — generate the Q6-extended fixture set.

Every fixture's label is computed analytically from its generation
parameters against the *standard* it claims to test, never by running a
detector. That is the only way the label is legitimate: if our own
detector says PASS but the parameters analytically violate WCAG SC 2.3.1
thresholds, the label is FAIL and the report records the disagreement as
an open question, not as a tuning target.

Sections (each appends rows to MANIFEST.csv with source = "Q6-extended"):

  1. FPS sweep        — same logical 3.1Hz / 2.9Hz luminance flash at
                        {24, 25, 30, 50, 60, 90, 120} fps. Above 3Hz/s
                        is hazard-positive (3 or more flashes per second);
                        2.9Hz is hazard-negative. Tests frame-rate handling.

  2. Boundary precision — count exactly at the limit, count -1, count +1.
                        At the boundary is "fail by definition" per the
                        standards' "3 or more" wording (WCAG); we use
                        explicit per-axis test cases.

  3. Area boundary    — flashing rectangle at exactly the WCAG-classic
                        25% / 341x256 limit, just under, just over,
                        and at multiple positions on a 1920x1080 canvas.

  4. False-positive battery — alarming-looking but PASS content:
                        (a) high-frequency luminance change with area < 25%
                            of the WCAG-classic limit;
                        (b) large-area but sub-threshold luminance contrast;
                        (c) rapid monochrome-equiluminant chroma swap (no
                            luminance delta, no red-saturation delta).

  5. Codec round-trip — take a small set of near-threshold PASS and FAIL
                        from the FPS / boundary sets, re-encode each
                        through H.264 CRF 18, H.264 CRF 28, ProRes 422,
                        VP9. Same logical content; verdict should be
                        stable across encodes. Recorded for scoring's
                        stability metric.

  6. Color space      — sRGB plus one wider-gamut (BT.2020-like primaries)
                        rendering of a single PASS and a single FAIL
                        fixture. Flagged "extended coverage; standards
                        under-specify."

This script is deterministic: seeded numpy + fixed encoding params.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2  # type: ignore


CORPUS_DIR = Path(__file__).resolve().parent
OUT_DIR = CORPUS_DIR / "generated" / "Q6-extended"
MANIFEST_PATH = CORPUS_DIR / "MANIFEST.csv"

# Reference canvas — matches TRACE's 1920x1080 SDR sRGB convention.
W, H = 1920, 1080

# WCAG 2.2 SC 2.3.1 "classic" area threshold: any 341x256 rectangle that
# would exceed 25% of the screen at standard 1024x768. On a 1920x1080
# canvas, the equivalent area-fraction limit is the *same* 0.25 fraction
# of the screen (the underlying rule is screen-fraction).
WCAG_AREA_RECT_W = 341
WCAG_AREA_RECT_H = 256
WCAG_AREA_FRACTION_LIMIT = 0.25  # of the entire screen

# Seed every randomness source.
np.random.seed(0)


# --- Helpers ---------------------------------------------------------------

def _make_writer(path: Path, fps: int) -> "cv2.VideoWriter":
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (W, H))


def _solid_frame(rgb: tuple[int, int, int]) -> np.ndarray:
    """Build a full-frame BGR image from an (R,G,B) triplet."""
    r, g, b = rgb
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[..., 0] = b
    img[..., 1] = g
    img[..., 2] = r
    return img


def _frame_with_rect(bg_rgb: tuple[int, int, int],
                     rect_rgb: tuple[int, int, int],
                     rect_w: int, rect_h: int,
                     anchor: str = "center") -> np.ndarray:
    img = _solid_frame(bg_rgb)
    rect_w = min(rect_w, W); rect_h = min(rect_h, H)
    if anchor == "center":
        x0 = (W - rect_w) // 2; y0 = (H - rect_h) // 2
    elif anchor == "tl":
        x0, y0 = 0, 0
    elif anchor == "br":
        x0, y0 = W - rect_w, H - rect_h
    else:
        raise ValueError(anchor)
    r, g, b = rect_rgb
    img[y0:y0 + rect_h, x0:x0 + rect_w] = (b, g, r)
    return img


def _alternating_full_screen(path: Path, fps: int, hz: float,
                             duration_s: float,
                             a_rgb=(0, 0, 0), b_rgb=(255, 255, 255)) -> int:
    """Write a video where the full frame alternates between two solid
    colors at ``hz`` cycles per second (one full A->B->A = 1 Hz cycle =
    2 flash transitions). Returns the number of complete *transitions*.
    """
    writer = _make_writer(path, fps)
    n_frames = int(round(duration_s * fps))
    # A "transition" is one of the two-state switches per cycle. At hz cycles
    # per second there are 2*hz transitions per second.
    transitions_per_sec = 2.0 * hz
    # Map each frame to a state based on a phase counter that ticks at
    # transitions_per_sec.
    state = 0
    for fi in range(n_frames):
        t = fi / fps
        # state flips every (1 / transitions_per_sec) seconds.
        target_state = int((t * transitions_per_sec)) % 2
        if target_state != state:
            state = target_state
        writer.write(_solid_frame(b_rgb if state else a_rgb))
    writer.release()
    return int(transitions_per_sec * duration_s)


# --- Row schema (matches build_manifest.py::MANIFEST_COLUMNS) --------------

MANIFEST_COLUMNS = [
    "source", "license", "type", "path",
    "expected_label", "expected_detail_file",
    # Per-standard expected-label columns added in build_manifest.py;
    # Q6-extended fixtures don't carry per-fixture per-standard labels
    # (they derive PASS/FAIL analytically from their generation_params
    # against a single standard) so these are emitted empty -- scoring
    # falls back to expected_label keyed against each standard the
    # standard_clause names.
    "expected_trace24", "expected_wcag2_2", "expected_ofcom2017",
    "expected_itu_r1702_4", "expected_iso9241_391",
    "standard_clause", "frame_rate", "color_space", "dynamic_range",
    "resolution", "codec", "generation_params",
    "provenance_commit", "notes",
]


@dataclass
class Row:
    path: Path
    expected_label: str
    standard_clause: str
    frame_rate: int
    codec: str
    generation_params: dict
    notes: str = ""
    color_space: str = "sRGB"
    dynamic_range: str = "SDR"
    resolution: str = f"{W}x{H}"

    def to_csv_row(self) -> list[str]:
        return [
            "Q6-extended",
            "BSD-3-Clause (this repo)",
            "video",
            str(self.path.relative_to(CORPUS_DIR.parent)),
            self.expected_label,
            "",  # no separate detail file; generation_params is the proof
            # Per-standard label columns: empty for Q6-extended (scoring
            # falls back to expected_label via standard_clause matching).
            "", "", "", "", "",
            self.standard_clause,
            str(self.frame_rate),
            self.color_space,
            self.dynamic_range,
            self.resolution,
            self.codec,
            json.dumps(self.generation_params, separators=(",", ":")),
            "",  # provenance_commit = this repo HEAD; left blank, derivable from git
            self.notes,
        ]


# --- Generators ------------------------------------------------------------

def fps_sweep_rows() -> Iterable[Row]:
    """Same logical luminance content at varying fps. A flash count above
    3 transitions per second is hazard-positive per WCAG 2.2 SC 2.3.1
    general flash rule; the area axis is set deliberately above the 25%
    limit (full screen) so the count axis is decisive.

    For each fps we generate ONE near-fail (3.1 Hz cycle → ~6.2 transitions/s)
    labeled FAIL, and ONE pass-by-count (1.4 Hz → ~2.8 transitions/s)
    labeled PASS. Both span 3.0 seconds.
    """
    out_dir = OUT_DIR / "fps_sweep"
    for fps in (24, 25, 30, 50, 60, 90, 120):
        for tag, hz, label in (
            ("fail_31hz", 3.1, "FAIL"),
            ("pass_14hz", 1.4, "PASS"),
        ):
            path = out_dir / f"{fps}fps_{tag}.mp4"
            _alternating_full_screen(path, fps=fps, hz=hz, duration_s=3.0)
            yield Row(
                path=path,
                expected_label=label,
                standard_clause=(
                    "WCAG 2.2 SC 2.3.1 general flash threshold: more than 3 "
                    "general flashes within any 1-second period, full-screen "
                    "(area > 25%) — count axis decisive."
                ),
                frame_rate=fps,
                codec="mp4v",
                generation_params={
                    "cycle_hz": hz, "transitions_per_sec": 2 * hz,
                    "duration_s": 3.0, "area_fraction": 1.0,
                    "a_rgb": [0, 0, 0], "b_rgb": [255, 255, 255],
                    "label_derivation": (
                        "transitions_per_sec > 6 ⇒ FAIL (>3 flashes/sec) "
                        "else PASS (count below WCAG general-flash limit)"
                    ),
                },
                notes=f"fps_sweep; fps={fps}; expected={label} by count axis",
            )


def boundary_precision_rows() -> Iterable[Row]:
    """Count axis at boundary ± 1 transition over a 1-second window.

    WCAG 2.2 SC 2.3.1 general flash: "MORE than 3 flashes in any 1 second
    period" is the hazard. So exactly 3 flashes/sec = PASS; 4+ = FAIL.
    We deliberately encode 2, 3, 4 flashes/sec (3 PASS / 3 PASS / 3 FAIL).

    "Flash" per the standard is a pair-of-opposing-transitions (a complete
    light-to-dark-to-light cycle counts as 2 flashes in the WCAG-text
    counting). We use the transitions count and divide by 2 to get flashes.
    """
    out_dir = OUT_DIR / "boundary_precision"
    fps = 60
    for flashes_per_sec, label in (
        (2, "PASS"),   # well under
        (3, "PASS"),   # at the limit (NOT more than 3)
        (4, "FAIL"),   # just over
        (6, "FAIL"),   # comfortably over
    ):
        # transitions/sec = 2 * flashes/sec since one flash = one opposing pair.
        hz = flashes_per_sec  # cycles per second so 2*hz transitions/sec
        # Actually if "flashes" means light-dark pairs, 1 flash/sec = 1 Hz cycle.
        path = out_dir / f"{flashes_per_sec}_flashes_per_sec.mp4"
        _alternating_full_screen(path, fps=fps, hz=hz, duration_s=2.0)
        yield Row(
            path=path,
            expected_label=label,
            standard_clause=(
                "WCAG 2.2 SC 2.3.1 general flash: 'more than 3 flashes within "
                "any 1-second period'. 3 flashes/sec is at-limit PASS; 4+ is FAIL."
            ),
            frame_rate=fps,
            codec="mp4v",
            generation_params={
                "flashes_per_sec": flashes_per_sec,
                "transitions_per_sec": 2 * flashes_per_sec,
                "duration_s": 2.0, "area_fraction": 1.0,
                "label_derivation": "PASS iff flashes_per_sec <= 3 per WCAG 2.2 SC 2.3.1.",
            },
            notes=f"boundary_precision; flashes/sec={flashes_per_sec}; expected={label}",
        )


def area_boundary_rows() -> Iterable[Row]:
    """Same hazardous flash content, varying area. Area decisive."""
    out_dir = OUT_DIR / "area_boundary"
    fps = 60
    hz = 5  # 10 transitions / sec — well over the count axis
    duration = 2.0
    # WCAG-classic rect (341x256) area ≈ 87,296 px ≈ 4.2% of 1920x1080 — well
    # below the 25% screen-fraction limit. So 25% requires a larger rectangle:
    # 25% of 1920x1080 = 518,400 px → ~720x720 square.
    sqrt_area_limit = int((W * H * WCAG_AREA_FRACTION_LIMIT) ** 0.5)  # ~720
    for tag, rect_w, rect_h, anchor, label in (
        ("classic_341x256_center", WCAG_AREA_RECT_W, WCAG_AREA_RECT_H, "center", "PASS"),
        ("just_under_25pct",       sqrt_area_limit - 20, sqrt_area_limit - 20, "center", "PASS"),
        ("exactly_25pct",          sqrt_area_limit, sqrt_area_limit, "center", "FAIL"),
        ("just_over_25pct",        sqrt_area_limit + 20, sqrt_area_limit + 20, "center", "FAIL"),
        ("over_25pct_topleft",     sqrt_area_limit + 20, sqrt_area_limit + 20, "tl", "FAIL"),
        ("over_25pct_botright",    sqrt_area_limit + 20, sqrt_area_limit + 20, "br", "FAIL"),
    ):
        path = out_dir / f"area_{tag}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = _make_writer(path, fps)
        n_frames = int(round(duration * fps))
        transitions_per_sec = 2.0 * hz
        state = 0
        for fi in range(n_frames):
            t = fi / fps
            target_state = int(t * transitions_per_sec) % 2
            if target_state != state:
                state = target_state
            color = (255, 255, 255) if state else (0, 0, 0)
            writer.write(_frame_with_rect(
                bg_rgb=(128, 128, 128),
                rect_rgb=color,
                rect_w=rect_w, rect_h=rect_h, anchor=anchor,
            ))
        writer.release()
        area_frac = (rect_w * rect_h) / float(W * H)
        yield Row(
            path=path,
            expected_label=label,
            standard_clause=(
                "WCAG 2.2 SC 2.3.1 area threshold: flashing region must "
                "exceed ~25% of any 10° visual-field area (approximated on-screen "
                "as 25% of the canvas) AND meet count + intensity thresholds for FAIL."
            ),
            frame_rate=fps,
            codec="mp4v",
            generation_params={
                "rect_w": rect_w, "rect_h": rect_h,
                "anchor": anchor, "area_fraction": area_frac,
                "transitions_per_sec": transitions_per_sec,
                "label_derivation": (
                    "Count axis hazard-positive; PASS iff area_fraction < 0.25 "
                    "(falls short on area axis), else FAIL."
                ),
            },
            notes=f"area_boundary; area={area_frac:.4f}; expected={label}",
        )


def false_positive_battery_rows() -> Iterable[Row]:
    """Alarming-looking content that is PASS because it falls short on
    exactly one axis. This is the credibility battery.
    """
    out_dir = OUT_DIR / "false_positive_battery"
    fps = 60

    # (a) High-frequency luminance flash with sub-threshold area (1% canvas).
    a_path = out_dir / "tiny_area_fast_flash.mp4"
    a_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(a_path, fps)
    area_frac_a = 0.01
    rect_side = int(((W * H * area_frac_a)) ** 0.5)
    hz_a = 8.0  # 16 transitions/sec — *very* high
    for fi in range(int(2.0 * fps)):
        t = fi / fps
        state = int(t * 2 * hz_a) % 2
        color = (255, 255, 255) if state else (0, 0, 0)
        writer.write(_frame_with_rect(
            bg_rgb=(128, 128, 128),
            rect_rgb=color, rect_w=rect_side, rect_h=rect_side, anchor="center",
        ))
    writer.release()
    yield Row(
        path=a_path,
        expected_label="PASS",
        standard_clause=(
            "WCAG 2.2 SC 2.3.1: hazard requires meeting BOTH the intensity/count "
            "threshold AND the area threshold (>25%). Sub-threshold area "
            "alone makes the sequence safe regardless of frequency."
        ),
        frame_rate=fps, codec="mp4v",
        generation_params={
            "axis_failed": "area",
            "area_fraction": area_frac_a,
            "transitions_per_sec": 2 * hz_a,
            "label_derivation": "PASS by area axis (<0.25 of screen).",
        },
        notes="FP-battery (a): alarming high-frequency but tiny area; should PASS.",
    )

    # (b) Large area but sub-threshold luminance contrast.
    b_path = out_dir / "large_area_low_contrast.mp4"
    writer = _make_writer(b_path, fps)
    hz_b = 4.0
    # Two near-equal mid-grays: relative luminance delta well below 0.1.
    lo, hi = (118, 118, 118), (138, 138, 138)
    for fi in range(int(2.0 * fps)):
        t = fi / fps
        state = int(t * 2 * hz_b) % 2
        writer.write(_solid_frame(hi if state else lo))
    writer.release()
    yield Row(
        path=b_path,
        expected_label="PASS",
        standard_clause=(
            "WCAG 2.2 SC 2.3.1 general flash threshold uses a "
            "relative-luminance delta of 0.1 (with lower of the two ≤ 0.8). "
            "20/255 gray-step is well below this threshold even at full screen."
        ),
        frame_rate=fps, codec="mp4v",
        generation_params={
            "axis_failed": "luminance",
            "lo_rgb": list(lo), "hi_rgb": list(hi),
            "transitions_per_sec": 2 * hz_b, "area_fraction": 1.0,
            "label_derivation": (
                "Relative-luminance delta < 0.1 → PASS by intensity axis "
                "regardless of area or count."
            ),
        },
        notes="FP-battery (b): large area, sub-threshold contrast; should PASS.",
    )

    # (c) Rapid chroma swap that is equi-luminant (no luminance OR red-sat delta).
    c_path = out_dir / "equiluminant_chroma_swap.mp4"
    writer = _make_writer(c_path, fps)
    hz_c = 5.0
    # Cyan vs Yellow — different hues, similar perceived luminance after
    # sRGB->relative-luminance transform, and very low *saturated* red on
    # either side (red channel near full for yellow, near zero for cyan;
    # we deliberately pick a swap with negligible R saturation in BOTH).
    cyan   = (40, 200, 200)
    yellow = (200, 200, 40)
    for fi in range(int(2.0 * fps)):
        t = fi / fps
        state = int(t * 2 * hz_c) % 2
        writer.write(_solid_frame(yellow if state else cyan))
    writer.release()
    yield Row(
        path=c_path,
        expected_label="PASS",
        standard_clause=(
            "WCAG 2.2 SC 2.3.1 red flash uses Harding's saturated-red transition "
            "(NOT a generic hue change). A chroma swap with low saturated-red "
            "content on both sides does not constitute a red flash, even if "
            "perceptually 'colorful'."
        ),
        frame_rate=fps, codec="mp4v",
        generation_params={
            "axis_failed": "red_saturation",
            "a_rgb": list(cyan), "b_rgb": list(yellow),
            "transitions_per_sec": 2 * hz_c, "area_fraction": 1.0,
            "label_derivation": (
                "Neither side qualifies as 'saturated red' per Harding red-flash "
                "definition (R - max(G,B) low or negative) → red axis PASS; "
                "relative-luminance delta also small → general flash PASS."
            ),
        },
        notes="FP-battery (c): equiluminant chroma swap; should PASS.",
    )


def codec_roundtrip_rows(seed_rows: list[Row]) -> Iterable[Row]:
    """Take a handful of seed fixtures and re-encode them via ffmpeg
    through H.264 CRF18/CRF28, ProRes422, and VP9. Same logical content,
    same expected label. The stability metric in scoring compares verdicts
    across these encodes per tool.
    """
    out_dir = OUT_DIR / "codec_roundtrip"
    encodes = [
        ("h264_crf18", ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast"], ".mp4"),
        ("h264_crf28", ["-c:v", "libx264", "-crf", "28", "-preset", "veryfast"], ".mp4"),
        ("prores422",  ["-c:v", "prores_ks", "-profile:v", "2"],                 ".mov"),
        ("vp9_crf32",  ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0"],        ".webm"),
    ]
    # Re-encode a subset so the Q6-extended corpus stays a manageable size.
    seeds = [r for r in seed_rows if "boundary_precision" in str(r.path) or "fps_sweep" in str(r.path)][:4]
    for seed in seeds:
        for enc_name, enc_args, ext in encodes:
            path = out_dir / f"{seed.path.stem}__{enc_name}{ext}"
            path.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                   "-i", str(seed.path), *enc_args, str(path)]
            try:
                subprocess.run(cmd, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                # ffmpeg or encoder unavailable; skip this encode. The
                # stability metric handles missing rows.
                print(f"[codec_roundtrip] skip {enc_name} ({e})", file=sys.stderr)
                continue
            yield Row(
                path=path,
                expected_label=seed.expected_label,
                standard_clause=(
                    f"Codec round-trip of {seed.path.name}: same logical "
                    f"content; verdict should be stable across encodes. "
                    f"Used by scoring's stability metric."
                ),
                frame_rate=seed.frame_rate,
                codec=enc_name,
                generation_params={
                    "seed_fixture": str(seed.path.relative_to(CORPUS_DIR.parent)),
                    "encoder": enc_args,
                    "label_inherits_from_seed": True,
                },
                notes=f"codec_roundtrip; seed={seed.path.name}; encoder={enc_name}",
            )


def color_space_rows() -> Iterable[Row]:
    """One PASS and one FAIL re-rendered with BT.2020-like primaries.

    We don't currently apply a real BT.2020 ICC profile — we tag the
    output as 'BT.2020-like' to make the under-specification explicit.
    Standards do not cover wide-gamut quantitatively; this is a coverage
    fixture flagged accordingly.
    """
    out_dir = OUT_DIR / "color_space"
    fps = 30
    for tag, hz, label in (("pass_below_count", 1.4, "PASS"),
                           ("fail_above_count", 3.5, "FAIL")):
        path = out_dir / f"bt2020like_{tag}.mp4"
        _alternating_full_screen(path, fps=fps, hz=hz, duration_s=2.0)
        yield Row(
            path=path,
            expected_label=label,
            standard_clause=(
                "Standards (WCAG 2.2, ITU-R BT.1702, Ofcom) are written for "
                "sRGB/Rec.709-era content; wider-gamut behavior is under-"
                "specified. Coverage fixture flagged accordingly."
            ),
            frame_rate=fps, codec="mp4v",
            generation_params={
                "color_space_tag": "BT.2020-like",
                "cycle_hz": hz,
                "label_derivation": "Count-axis decision under same sRGB rendering; tagged for color-space coverage.",
            },
            color_space="BT.2020-like",
            notes=f"color_space; tag={tag}; under-specified standards coverage",
        )


# --- Main ------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    rows.extend(fps_sweep_rows())
    rows.extend(boundary_precision_rows())
    rows.extend(area_boundary_rows())
    rows.extend(false_positive_battery_rows())
    rows.extend(codec_roundtrip_rows(rows.copy()))
    rows.extend(color_space_rows())

    # Append to MANIFEST.csv (assumes build_manifest.py has already run;
    # otherwise create the file with the header).
    write_header = not MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(MANIFEST_COLUMNS)
        for r in rows:
            w.writerow(r.to_csv_row())

    print(f"Q6-extended: {len(rows)} fixtures written; manifest appended.")


if __name__ == "__main__":
    main()
