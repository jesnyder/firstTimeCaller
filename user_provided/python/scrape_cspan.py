#!/usr/bin/env python3
"""
C-SPAN Washington Journal caller scraper — YouTube edition.

c-span.org blocks plain HTTP requests via CloudFront WAF.
This script pulls episodes from C-SPAN's official YouTube channel using
yt-dlp, which handles YouTube properly and downloads professional closed
captions.

Caption format notes (YouTube VTT):
  - Speaker changes are marked with ">>" (encoded as "&gt;&gt;" in HTML)
  - The rolling-window format duplicates each line; we deduplicate per-line
  - No "CALLER:" labels; callers follow host intros like:
      "Rick calling from Austin Texas on our Republican line"
      "on the Democrat line, Sarah from Ohio, go ahead"

Pipeline:
  1. Search / list Washington Journal episodes on YouTube via yt-dlp
  2. Filter to videos whose title contains "Washington Journal"
  3. Download VTT closed-caption files to a temp directory
  4. Parse captions: deduplicate lines, group into speaker turns at ">>"
  5. Detect caller turns: host turn contains party-line intro → next turn is caller
  6. Extract caller name from intro, infer gender via gender_guesser / sir/ma'am
  7. Compute text metrics, save CSV, print comparison table

Install:
    pip install yt-dlp pandas gender-guesser

Optional (richer POS features):
    pip install spacy && python -m spacy download en_core_web_sm

Run:
    python user_provided/python/scrape_cspan.py
    python user_provided/python/scrape_cspan.py --episodes 40 --output results.csv
    python user_provided/python/scrape_cspan.py \\
        --source "https://www.youtube.com/playlist?list=PLAYLIST_ID"
"""

import re
import sys
import os
import glob
import html
import tempfile
import shutil
from dataclasses import dataclass
from collections import Counter
from typing import Optional

import yt_dlp
import pandas as pd

# ── optional spacy ────────────────────────────────────────────────────────────
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False

# ── optional gender_guesser ───────────────────────────────────────────────────
try:
    import gender_guesser.detector as _ggd
    _detector = _ggd.Detector()
    HAS_GENDER_GUESSER = True
except ImportError:
    HAS_GENDER_GUESSER = False
    print("WARNING: gender_guesser not installed — name inference disabled.\n"
          "  pip install gender-guesser\n")

# ── default source ────────────────────────────────────────────────────────────
# The official @cspan YouTube channel does not post full Washington Journal
# episodes.  @thecspanreview is a fan channel that posts combined
# "Open Forum" segments (~90-110 min each) with consistent caption quality
# and many callers per video.  It is the best default source.
DEFAULT_SOURCE = "https://www.youtube.com/@thecspanreview/videos"

# Only keep videos whose title contains "Washington Journal"
TITLE_FILTER = re.compile(r'washington\s+journal', re.I)

# Minimum duration in seconds — Open Forum combined videos run ~90 min.
# Keep bar low to also catch shorter single-segment uploads.
MIN_DURATION_SEC = 600

# ── VTT inline cleanup ────────────────────────────────────────────────────────
_INLINE_TS = re.compile(r'<\d{2}:\d{2}:\d{2}\.\d{3}>')
_HTML_TAGS = re.compile(r'<[^>]+>')

# ── Washington Journal caption patterns ──────────────────────────────────────

PARTY_RE = re.compile(r'\b(republican|democrat(?:ic)?|independent)\b', re.I)

# Matches the host's caller introduction:
#   "Rick calling from Austin Texas on our Republican line"
#   "on the Democrat line, Sarah from Ohio"
#   "Democrat line, John from Maryland, go ahead"
INTRO_PARTY_RE = re.compile(
    r'\b(republican|democrat(?:ic)?|independent)\s+line\b', re.I
)

# Several name patterns that appear in Washington Journal intros
NAME_PATTERNS = [
    # "Rick calling from …"  or  "Rick's calling from …"
    re.compile(r"\b([A-Z][a-z]{1,20})(?:'s)?\s+(?:calling|is calling)\b"),
    # "on the Republican line, Sarah from …"  or  "line, Sarah go ahead"
    re.compile(r'\bline[.,]?\s+([A-Z][a-z]{1,20})\b'),
    # "go ahead Mary"  /  "welcome Mary"  /  "yes Mary"
    re.compile(r'\b(?:go ahead|welcome|yes)[,.]?\s+([A-Z][a-z]{1,20})\b', re.I),
    # "Mary from Ohio"  (generic — lower priority)
    re.compile(r'\b([A-Z][a-z]{1,20})\s+from\s+[A-Z][a-z]'),
]

# Signals end of a caller turn (host taking back the floor)
TURN_END_RE = re.compile(
    r'\b(thank you|thanks for (your )?call|we go next|next (up|caller)|'
    r'let\'?s go to|moving on|on our|on the)\b', re.I
)

# Salutation
SIR_RE  = re.compile(r'\b(sir|gentleman)\b', re.I)
MAAM_RE = re.compile(r'\b(ma\'?am|madam)\b', re.I)

# ── Known Washington Journal hosts ────────────────────────────────────────────
# Keys are distinctive first-name or surname tokens (lowercase).
# Values are (display_name, gender).
# Intentionally excludes ambiguous tokens like "john", "bill", "casey".
_HOST_LOOKUP: dict[str, tuple[str, str]] = {
    "greta":      ("Greta Brawner",    "female"),
    "brawner":    ("Greta Brawner",    "female"),
    "kimberly":   ("Kimberly Adams",   "female"),
    "libby":      ("Libby Casey",      "female"),
    "swain":      ("Susan Swain",      "female"),
    "jeslyn":     ("Jeslyn Rollins",   "female"),
    "rollins":    ("Jeslyn Rollins",   "female"),
    "chloe":      ("Chloe Veltman",    "female"),
    "pedro":      ("Pedro Echevarria", "male"),
    "echevarria": ("Pedro Echevarria", "male"),
    "mcardle":    ("John McArdle",     "male"),
    "scanlan":    ("Bill Scanlan",     "male"),
    "scully":     ("Steve Scully",     "male"),
    "harleston":  ("Rob Harleston",    "male"),
    "khalil":     ("Khalil Garriott",  "male"),
    "garriott":   ("Khalil Garriott",  "male"),
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CallerTurn:
    name:        str
    gender:      str   # "male" | "female" | "unknown"
    gender_src:  str   # "name" | "salutation" | "unknown"
    party:       str
    text:        str
    episode_id:  str = ""
    upload_date: str = ""  # YYYY-MM-DD from YouTube video metadata
    host_name:   str = ""
    host_gender: str = ""  # "male" | "female" | "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  VIDEO DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def fetch_video_urls(source: str, max_videos: int) -> list[str]:
    """
    Return YouTube watch URLs for Washington Journal episodes.

    When the primary source is exhausted (fewer matches than requested),
    falls back to a set of supplemental YouTube search queries so we get
    more unique episodes.
    """
    SUPPLEMENTAL_SEARCHES = [
        # Primary fan channel (secondary playlist page)
        "https://www.youtube.com/@thecspanreview/videos",
        # Official C-SPAN channel — has some shorter WJ segments
        "https://www.youtube.com/@cspan/videos",
        # Year-specific Open Forum searches (100 results each)
        "ytsearch100:Washington Journal Open Forum C-SPAN 2026",
        "ytsearch100:Washington Journal Open Forum C-SPAN 2025",
        "ytsearch100:Washington Journal Open Forum C-SPAN 2024",
        "ytsearch100:Washington Journal Open Forum C-SPAN 2023",
        "ytsearch100:Washington Journal Open Forum C-SPAN 2022",
        "ytsearch100:Washington Journal Open Forum C-SPAN 2021",
        "ytsearch100:Washington Journal Open Forum C-SPAN 2020",
        # Caller / phone-line specific
        "ytsearch100:\"Washington Journal\" caller C-SPAN open phones",
        "ytsearch100:Washington Journal C-SPAN Republican Democrat line",
        "ytsearch100:\"Washington Journal\" \"open phones\" CSPAN callers",
        "ytsearch100:C-SPAN Washington Journal caller republican democrat independent",
        # Host-name searches to surface older episodes
        "ytsearch100:Washington Journal CSPAN Greta Brawner callers",
        "ytsearch100:Washington Journal CSPAN Pedro Echevarria callers",
        "ytsearch100:Washington Journal CSPAN John McArdle callers",
        "ytsearch100:Washington Journal CSPAN Kimberly Adams callers",
        # Broader CSPAN caller searches
        "ytsearch100:CSPAN Washington Journal callers 2024",
        "ytsearch100:CSPAN Washington Journal callers 2023",
        "ytsearch100:CSPAN Washington Journal callers 2022",
        "ytsearch100:\"Washington Journal\" open forum independent line",
        "ytsearch100:Washington Journal C-SPAN open phones republican line caller",
    ]

    seen_ids: set[str] = set()
    urls: list[str] = []

    def _drain_source(src: str, limit: int) -> None:
        is_channel = "/videos" in src or ("/@" in src and "{n}" not in src)
        # For ytsearchN: sources the N is baked into the query string;
        # use a generous playlist_items window so we don't prune early.
        m_search = re.match(r'ytsearch(\d+):', src)
        if m_search:
            fetch_n = int(m_search.group(1))
        elif is_channel:
            fetch_n = min(limit * 10, 500)
        else:
            fetch_n = limit * 3
        query = src.format(n=fetch_n) if "{n}" in src else src

        ydl_opts = {
            "quiet": True, "no_warnings": True,
            "extract_flat": True,
            "playlist_items": f"1-{fetch_n}",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(query, download=False)
        except Exception:
            return

        for e in (info.get("entries") or [])[:fetch_n]:
            if not e or len(urls) >= limit:
                break
            title    = e.get("title", "")
            vid_id   = e.get("id", "")
            duration = e.get("duration") or 0
            if not TITLE_FILTER.search(title):
                continue
            if duration and duration < MIN_DURATION_SEC:
                continue
            if vid_id in seen_ids:
                continue
            url = e.get("webpage_url") or e.get("url", "")
            if not url.startswith("http") and vid_id:
                url = f"https://www.youtube.com/watch?v={vid_id}"
            if url.startswith("http"):
                seen_ids.add(vid_id)
                urls.append(url)

    # Primary source
    _drain_source(source, max_videos)

    # Supplemental searches if still under quota
    for search in SUPPLEMENTAL_SEARCHES:
        if len(urls) >= max_videos:
            break
        _drain_source(search, max_videos)

    return urls


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CAPTION DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_captions(video_url: str, tmp_dir: str) -> tuple[list[str], str]:
    """
    Download VTT subtitle file(s) for a video.
    Returns (new_vtt_paths, upload_date) where upload_date is YYYY-MM-DD or "".
    """
    ydl_opts = {
        "quiet": True, "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": os.path.join(tmp_dir, "%(id)s.%(ext)s"),
    }
    before = set(glob.glob(os.path.join(tmp_dir, "*.vtt")))
    upload_date = ""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            if info:
                raw = info.get("upload_date", "")  # "YYYYMMDD"
                if raw and len(raw) == 8:
                    upload_date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    except Exception:
        pass
    after = set(glob.glob(os.path.join(tmp_dir, "*.vtt")))
    return list(after - before), upload_date


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VTT PARSING → deduplicated line list
# ══════════════════════════════════════════════════════════════════════════════

def parse_vtt(vtt_text: str) -> list[str]:
    """
    Parse a WebVTT file into a deduplicated list of clean text lines.

    YouTube's rolling-window VTT repeats each line across adjacent cues.
    We deduplicate at the individual-line level (not cue level) and decode
    HTML entities so ">>" speaker markers survive.
    """
    # Decode HTML entities first so &gt;&gt; becomes >>
    vtt_text = html.unescape(vtt_text)
    # Strip inline word-level timestamps and tags
    vtt_text = _INLINE_TS.sub("", vtt_text)
    vtt_text = _HTML_TAGS.sub("", vtt_text)

    seen: set[str] = set()
    lines: list[str] = []

    for raw in vtt_text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if "-->" in ln or re.fullmatch(r"\d+", ln):
            continue
        if ln.upper().startswith(("WEBVTT", "NOTE", "STYLE", "KIND:", "LANGUAGE:")):
            continue
        if ln not in seen:
            seen.add(ln)
            lines.append(ln)

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# 4.  GROUP LINES INTO SPEAKER TURNS
# ══════════════════════════════════════════════════════════════════════════════

def lines_to_turns(lines: list[str]) -> tuple[list[str], bool]:
    """
    Group caption lines into speaker turns.

    Returns (turns, has_markers).
    has_markers=True  → ">>" splits used; caller = turn AFTER intro turn
    has_markers=False → no ">>" found; caller text is WITHIN the intro turn,
                        after the "go ahead" / "what's on your mind" signal
    """
    has_markers = any(ln.startswith(">>") for ln in lines)
    turns: list[str] = []
    current: list[str] = []

    if has_markers:
        for ln in lines:
            if ln.startswith(">>"):
                if current:
                    turns.append(" ".join(current).strip())
                rest = ln[2:].strip()
                current = [rest] if rest else []
            else:
                current.append(ln)
        if current:
            turns.append(" ".join(current).strip())
    else:
        # No ">>" markers — split each time we see a new party-line intro.
        # The intro line STARTS the new segment; caller speech follows in
        # the same segment until the next intro.
        for ln in lines:
            if INTRO_PARTY_RE.search(ln) and current:
                turns.append(" ".join(current).strip())
                current = [ln]
            else:
                current.append(ln)
        if current:
            turns.append(" ".join(current).strip())

    return [t for t in turns if t], has_markers


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _gender_from_name(first_name: str) -> tuple[str, str]:
    if not HAS_GENDER_GUESSER or not first_name:
        return "unknown", "unknown"
    result = _detector.get_gender(first_name.title())
    if result in ("male", "mostly_male"):
        return "male", "name"
    if result in ("female", "mostly_female"):
        return "female", "name"
    return "unknown", "unknown"


def detect_host(turns: list[str]) -> tuple[str, str]:
    """
    Scan all speaker turns for a known WJ host token.
    Returns (host_name, host_gender).

    Strategy: vote across every turn (callers often address the host by first
    name — "Good morning, Pedro") and return the most-mentioned host.
    """
    votes: Counter = Counter()
    for turn in turns:
        for token in re.findall(r"[a-z']+", turn.lower()):
            if token in _HOST_LOOKUP:
                votes[_HOST_LOOKUP[token][0]] += 1

    if not votes:
        return ("", "unknown")

    top_name = votes.most_common(1)[0][0]
    # Retrieve gender for the winning name
    for display, gender in _HOST_LOOKUP.values():
        if display == top_name:
            return (top_name, gender)
    return (top_name, "unknown")


def _gender_from_salutation(text: str) -> str:
    m = len(SIR_RE.findall(text))
    f = len(MAAM_RE.findall(text))
    if f > m:
        return "female"
    if m > f:
        return "male"
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CALLER EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_name(text: str) -> str:
    """Try each NAME_PATTERN against text; return first match or ''."""
    for pat in NAME_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = m.group(1)
            # Reject common non-name words that slip through
            if candidate.lower() not in {"the", "our", "your", "thank", "good",
                                          "this", "that", "what", "well", "just",
                                          "line", "next", "caller", "and"}:
                return candidate
    return ""


def extract_callers(turns: list[str], episode_id: str = "",
                    has_markers: bool = True,
                    upload_date: str = "",
                    host_name: str = "",
                    host_gender: str = "") -> list[CallerTurn]:
    """
    Walk speaker turns and identify caller turns.

    has_markers=True  (">>" episodes):
        Host intro is turn[i]; caller text is turn[i+1].
    has_markers=False (no ">>" — continuous caption text):
        Intro and caller speech are in the same turn[i].
        We split at the "go ahead" / "what's on your mind" signal.

    Word-count bounds:
        min 15 words  — filters out short host greetings caught as callers
        max 350 words — filters out multi-caller lumps in the no-">>" path
    """
    # Signals that hand the floor to the caller, used to split no-">>" turns
    _HANDOFF_RE = re.compile(
        r'\b(go ahead|what\'?s on your mind|you\'?re on|your turn|'
        r'please go ahead|please proceed)\b', re.I
    )

    callers: list[CallerTurn] = []
    n = len(turns)

    for i, turn in enumerate(turns):
        if not INTRO_PARTY_RE.search(turn):
            continue

        # ── get caller text based on caption mode ─────────────────────────
        if has_markers:
            if i + 1 >= n:
                continue
            caller_text = turns[i + 1].strip()
            # Skip if next turn is itself another host intro (no caller in between)
            if INTRO_PARTY_RE.search(caller_text) and len(caller_text.split()) < 25:
                continue
        else:
            # Extract speech from WITHIN this turn, after the handoff signal
            ho = _HANDOFF_RE.search(turn)
            if ho:
                caller_text = turn[ho.end():].strip()
            else:
                # No explicit handoff — skip first sentence and use the rest
                first_end = re.search(r'[.?!]\s+', turn)
                caller_text = turn[first_end.end():].strip() if first_end else ""

        # ── word-count bounds ──────────────────────────────────────────────
        wc = len(caller_text.split())
        if wc < 15 or wc > 350:
            continue

        # ── party ─────────────────────────────────────────────────────────
        pm = PARTY_RE.search(turn)
        party = pm.group(1).lower() if pm else "unknown"
        if party == "democratic":
            party = "democrat"

        # ── name ──────────────────────────────────────────────────────────
        first_name = _extract_name(turn)

        # ── gender ────────────────────────────────────────────────────────
        gender, gender_src = _gender_from_name(first_name)
        if gender == "unknown":
            # Look for sir/ma'am in the host's closing of this caller
            # (the turn after the caller, if it exists)
            closing_ctx = turns[i + 2] if i + 2 < n else ""
            sal = _gender_from_salutation(turn + " " + caller_text + " " + closing_ctx)
            if sal != "unknown":
                gender, gender_src = sal, "salutation"

        callers.append(CallerTurn(
            name=first_name,
            gender=gender,
            gender_src=gender_src,
            party=party,
            text=caller_text,
            episode_id=episode_id,
            upload_date=upload_date,
            host_name=host_name,
            host_gender=host_gender,
        ))

    return callers


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_Q_START    = re.compile(
    r'^(do|does|did|is|are|was|were|will|would|can|could|have|has|'
    r'why|what|how|when|where|who|which|whose)\b', re.I
)


def analyze(callers: list[CallerTurn]) -> pd.DataFrame:
    rows = []
    for c in callers:
        text  = c.text.strip()
        words = text.split()
        sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()] or [text]
        qs    = [s for s in sents if s.endswith("?") or _Q_START.match(s)]

        row = {
            "gender":                 c.gender,
            "gender_src":             c.gender_src,
            "party":                  c.party,
            "episode_id":             c.episode_id,
            "upload_date":            c.upload_date,
            "name":                   c.name,
            "word_count":             len(words),
            "sentence_count":         len(sents),
            "question_count":         len(qs),
            "question_ratio":         round(len(qs) / max(len(sents), 1), 3),
            "avg_words_per_sentence": round(len(words) / max(len(sents), 1), 2),
            "unique_word_ratio":      round(len(set(w.lower() for w in words)) / max(len(words), 1), 3),
            "host_name":              c.host_name,
            "host_gender":            c.host_gender,
            "text":                   text,
        }

        if HAS_SPACY and text:
            doc   = _nlp(text[:8000])
            pos   = Counter(t.pos_ for t in doc)
            total = max(len(doc), 1)
            row["noun_ratio"] = round(pos.get("NOUN", 0) / total, 3)
            row["verb_ratio"] = round(pos.get("VERB", 0) / total, 3)
            row["adj_ratio"]  = round(pos.get("ADJ",  0) / total, 3)

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No caller turns extracted.")
        return

    labeled = df[df["gender"].isin(["male", "female"])]

    print("\n" + "=" * 60)
    print("C-SPAN Washington Journal — Caller Gender Comparison")
    print("=" * 60)
    print(f"\nTotal turns : {len(df)}")
    print(f"  female    : {(labeled['gender'] == 'female').sum()}")
    print(f"  male      : {(labeled['gender'] == 'male').sum()}")
    print(f"  unknown   : {(df['gender'] == 'unknown').sum()}")

    if labeled.empty:
        print("\nNo gender-labeled turns — try more episodes or a specific playlist.")
        return

    metrics = ["word_count", "sentence_count", "question_count",
               "question_ratio", "avg_words_per_sentence", "unique_word_ratio"]
    if HAS_SPACY:
        metrics += ["noun_ratio", "verb_ratio", "adj_ratio"]

    print("\n--- Mean metrics by gender ---")
    print(labeled.groupby("gender")[metrics].mean().round(3).to_string())

    known = labeled[labeled["party"] != "unknown"]
    if not known.empty:
        print("\n--- Mean word count — gender × party ---")
        print(known.groupby(["gender", "party"])["word_count"]
              .agg(mean="mean", n="count").round(1).to_string())

    print("\n--- Gender inference source ---")
    print(labeled["gender_src"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scrape C-SPAN Washington Journal callers via YouTube."
    )
    parser.add_argument("--source",   default=None,
                        help="YouTube URL (playlist/channel/search). "
                             "Default: @thecspanreview channel + supplemental searches.")
    parser.add_argument("--episodes", type=int, default=15,
                        help="Max NEW episodes to target (default 15).")
    parser.add_argument("--output",   default="cspan_callers.csv",
                        help="Output CSV path.")
    parser.add_argument("--append",   action="store_true",
                        help="Load existing CSV, skip already-scraped episodes, "
                             "and append new rows. Safe to run repeatedly.")
    parser.add_argument("--debug",    action="store_true",
                        help="Print first 20 speaker turns for the first episode.")
    args = parser.parse_args()

    source = args.source or DEFAULT_SOURCE

    # ── load existing data when appending ─────────────────────────────────
    existing_df: pd.DataFrame | None = None
    skip_ids: set[str] = set()
    if args.append and os.path.exists(args.output):
        existing_df = pd.read_csv(args.output)
        if "episode_id" in existing_df.columns:
            skip_ids = set(existing_df["episode_id"].dropna().astype(str).unique())
        print(f"Existing data: {len(existing_df)} rows from {len(skip_ids)} episode(s) — will skip those.")

    # Fetch more URLs than requested so we have room to skip already-seen ones
    fetch_target = args.episodes + len(skip_ids) + 20
    print(f"Fetching up to {fetch_target} candidate video(s) (targeting {args.episodes} new) ...")
    video_urls = fetch_video_urls(source, max_videos=fetch_target)
    print(f"  Found {len(video_urls)} matching video(s).")

    if not video_urls:
        print("No videos found. Try --source with a specific playlist URL.")
        sys.exit(1)

    tmp_dir = tempfile.mkdtemp(prefix="cspan_caps_")
    all_callers: list[CallerTurn] = []
    new_episodes = 0

    try:
        for idx, url in enumerate(video_urls, 1):
            if new_episodes >= args.episodes:
                break
            m = re.search(r'[?&]v=([^&]+)', url)
            vid_id = m.group(1) if m else str(idx)

            if vid_id in skip_ids:
                print(f"[{idx}/{len(video_urls)}] {vid_id} ... already scraped, skipping.")
                continue

            print(f"[{idx}/{len(video_urls)}] {vid_id} ...", end=" ", flush=True)

            vtt_files, upload_date = download_captions(url, tmp_dir)
            if not vtt_files:
                print("no captions.")
                new_episodes += 1  # count it even if empty so we don't loop forever
                continue

            # Prefer official captions over auto-generated
            vtt_files.sort(key=lambda p: ("auto" in p.lower(), p))
            with open(vtt_files[0], encoding="utf-8", errors="replace") as f:
                vtt_text = f.read()

            lines = parse_vtt(vtt_text)
            turns, has_markers = lines_to_turns(lines)
            host_name, host_gender = detect_host(turns)
            callers = extract_callers(turns, episode_id=vid_id,
                                      has_markers=has_markers, upload_date=upload_date,
                                      host_name=host_name, host_gender=host_gender)
            print(f"{len(callers)} caller(s)  [{len(turns)} turns, "
                  f"{'>>markers' if has_markers else 'no markers'}]")
            new_episodes += 1

            if args.debug and idx == 1:
                print("\n--- First 20 turns (debug) ---")
                for t in turns[:20]:
                    print(f"  {repr(t[:120])}")
                print()

            all_callers.extend(callers)

            for p in vtt_files:
                try:
                    os.remove(p)
                except OSError:
                    pass

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not all_callers:
        if args.append and existing_df is not None:
            print("\nNo new caller turns found — all candidate episodes were already scraped.")
            print_summary(existing_df)
            return
        print("\nNo caller turns found.\n"
              "Try: make scrape-debug  to see raw turns.\n"
              "Or:  make scrape SOURCE=\"<specific playlist URL>\"")
        sys.exit(1)

    new_df = analyze(all_callers)

    if args.append and existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nAppended {len(new_df)} new rows → {args.output}  "
              f"({len(combined)} total, {new_episodes} new episode(s) processed)")
        print_summary(combined)
    else:
        new_df.to_csv(args.output, index=False)
        print(f"\nSaved {len(new_df)} rows → {args.output}")
        print_summary(new_df)


if __name__ == "__main__":
    main()
