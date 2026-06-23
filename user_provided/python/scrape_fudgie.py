#!/usr/bin/env python3
"""
C-SPAN Washington Journal caller scraper — fight.fudgie.org edition.

fudgie.org hosts machine-generated transcripts for all C-SPAN broadcasts,
with per-line speaker attribution already done (data-speaker-name attribute).
Callers appear as "name in state" (e.g. "roy in north dakota") or "Unidentified".
This script extracts caller turns and produces the same CSV schema as
scrape_cspan.py so both can be combined.

Install:
    pip install requests pandas gender-guesser

Run:
    python user_provided/python/scrape_fudgie.py --episodes 200 --append --output results/scraped/cspan_callers.csv
"""

import re
import sys
import os
import html as html_lib
import time
import random
from dataclasses import dataclass, field
from collections import Counter
from datetime import datetime
from typing import Optional

import requests
import pandas as pd

try:
    import gender_guesser.detector as _ggd
    _detector = _ggd.Detector()
    HAS_GENDER_GUESSER = True
except ImportError:
    HAS_GENDER_GUESSER = False
    print("WARNING: gender_guesser not installed — name inference disabled.\n"
          "  pip install gender-guesser\n")

BASE_URL = "https://fight.fudgie.org"
LIST_URL = f"{BASE_URL}/search/show/cspan/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Only keep full morning Washington Journal episodes (the 3-hour open-phone shows)
# Pattern: 0700-1000 or 0700-1001 etc. OR contains "Open_Phones"
WJ_EPISODE_RE = re.compile(
    r'Washington_Journal.*(?:0[67]\d{2}-1[01]\d{2}|Open_Phones|Open_Forum)',
    re.I,
)
WJ_ANY_RE = re.compile(r'Washington_Journal', re.I)

# Known WJ hosts (lowercase)
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

KNOWN_HOSTS = {
    "john mcardle", "greta brawner", "kimberly adams", "libby casey",
    "susan swain", "jeslyn rollins", "chloe veltman", "pedro echevarria",
    "bill scanlan", "steve scully", "rob harleston", "khalil garriott",
}

INTRO_PARTY_RE = re.compile(
    r'\b(republican|democrat(?:ic)?|independent)\s+line\b', re.I
)
PARTY_RE = re.compile(r'\b(republican|democrat(?:ic)?|independent)\b', re.I)
SIR_RE   = re.compile(r'\b(sir|gentleman)\b', re.I)
MAAM_RE  = re.compile(r'\b(ma\'?am|madam)\b', re.I)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_Q_START    = re.compile(
    r'^(do|does|did|is|are|was|were|will|would|can|could|have|has|'
    r'why|what|how|when|where|who|which|whose)\b', re.I
)
HEDGE_RE   = re.compile(
    r"\b(i think|i feel|i believe|i guess|maybe|perhaps|possibly|"
    r"it seems|sort of|kind of|i was wondering|i'm not sure|"
    r"i don't know|might be|could be)\b", re.I
)
MODAL_RE   = re.compile(r"\b(would|could|should|might|may|ought)\b", re.I)
FIRST_P_RE = re.compile(r"\b(i|me|my|mine|myself)\b", re.I)
SECOND_P_RE = re.compile(r"\b(you|your|yours|yourself)\b", re.I)


US_STATES: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# Sort longest first so multi-word states match before single-word prefixes
_STATE_RE = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in sorted(US_STATES, key=len, reverse=True)) + r')\b',
    re.I,
)

# "city, state" pattern in intro text
_CITY_STATE_RE = re.compile(
    r'\b(?:in|from)\s+([A-Za-z][A-Za-z ]{1,22}?),\s*('
    + '|'.join(re.escape(s) for s in sorted(US_STATES, key=len, reverse=True))
    + r')\b',
    re.I,
)


def _extract_location(text: str) -> tuple[str, str]:
    """Return (city, state_abbrev) from intro or speaker text."""
    m = _CITY_STATE_RE.search(text)
    if m:
        city = m.group(1).strip().title()
        state = US_STATES[m.group(2).lower()]
        return city, state
    m = _STATE_RE.search(text)
    if m:
        return "", US_STATES[m.group(1).lower()]
    return "", ""


def _ep_start_hour(episode_id: str) -> int:
    """Extract the broadcast start hour (ET) from an episode ID like 20260620_CSPAN_0700-..."""
    m = re.search(r'_(\d{2})\d{2}-', episode_id)
    return int(m.group(1)) if m else -1


@dataclass
class CallerTurn:
    name:             str
    gender:           str
    gender_src:       str
    party:            str
    text:             str
    episode_id:       str   = ""
    upload_date:      str   = ""
    host_name:        str   = ""
    host_gender:      str   = ""
    caller_city:      str   = ""
    caller_state:     str   = ""
    call_hour:        int   = -1    # -1 = unknown; ET hour (0-23)
    call_duration_sec: float = 0.0
    day_of_week:      str   = ""    # Monday … Sunday
    host_response_text: str = ""   # host's first turn after caller finishes


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def fetch_episode_list(max_episodes: int, skip_ids: set[str]) -> list[tuple[str, str]]:
    """
    Fetch the fudgie.org cspan listing page and return (episode_id, url) pairs
    for Washington Journal episodes, skipping already-seen IDs.
    """
    print(f"Fetching episode list from {LIST_URL} ...")
    try:
        r = requests.get(LIST_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR fetching episode list: {e}")
        return []

    all_links = re.findall(r'href="(/search/show/cspan/episode/([^"]+))"', r.text)
    results = []
    seen = set()
    for path, ep_id in all_links:
        # Strip URL #fragment (e.g. #speaker-21) — same page, different anchor
        ep_id = ep_id.split('#')[0]
        path  = path.split('#')[0]
        if not WJ_ANY_RE.search(ep_id):
            continue
        if ep_id in seen or ep_id in skip_ids:
            continue
        seen.add(ep_id)
        results.append((ep_id, BASE_URL + path))
        if len(results) >= max_episodes * 3:  # fetch extra for filtering
            break

    # Prioritise full morning shows (0700-10xx) and open phones
    priority = [(eid, url) for eid, url in results if WJ_EPISODE_RE.search(eid)]
    rest     = [(eid, url) for eid, url in results if not WJ_EPISODE_RE.search(eid)]
    ordered  = priority + rest

    print(f"  {len(ordered)} WJ episodes found ({len(priority)} priority morning/open-phones).")
    return ordered


# ══════════════════════════════════════════════════════════════════════════════
# PAGE PARSING
# ══════════════════════════════════════════════════════════════════════════════

_ATTRIBUTED_ROW_RE = re.compile(
    r'<td id="line(\d+)"[^>]*'
    r'data-speaker-name="([^"]+)"[^>]*>'
    r'.*?'
    r'<td id="text-\1"[^>]*>\s*(.*?)\s*<a class="quote-favourite',
    re.DOTALL,
)

_TEXT_CELL_RE = re.compile(
    r'<td id="line(\d+)"([^>]*)>'
    r'.*?'
    r'<td id="text-\1"[^>]*>\s*(.*?)\s*<a class="quote-favourite',
    re.DOTALL,
)

# Captures play_ep offset (seconds into audio file) from timestamp cell
_PLAY_EP_RE = re.compile(r"play_ep\([^,]+,\s*([\d.]+)")

# Turn type: (speaker_name, joined_text, start_offset_sec, end_offset_sec)
Turn = tuple[str, str, float, float]


def _build_timestamps(html: str) -> dict[str, float]:
    """Return {line_id: offset_seconds} by scanning play_ep() calls."""
    ts: dict[str, float] = {}
    for m in re.finditer(r'id="line(\d+)"', html):
        lid = m.group(1)
        chunk = html[m.start(): m.start() + 400]
        pm = _PLAY_EP_RE.search(chunk)
        if pm:
            ts[lid] = float(pm.group(1))
    return ts


def _detect_host(speaker_turns: list[Turn]) -> tuple[str, str]:
    votes: Counter = Counter()
    for sp, _, _, _ in speaker_turns:
        for token in sp.split():
            if token in _HOST_LOOKUP:
                votes[_HOST_LOOKUP[token][0]] += 1
        if sp in KNOWN_HOSTS:
            canonical = next(
                (_HOST_LOOKUP[part][0] for part in sp.split() if part in _HOST_LOOKUP),
                None,
            )
            if canonical:
                votes[canonical] += 5
    if not votes:
        return "", "unknown"
    top = votes.most_common(1)[0][0]
    host_gender = next((g for d, g in _HOST_LOOKUP.values() if d == top), "unknown")
    return top, host_gender


def parse_episode(html: str) -> tuple[list[Turn], str, str]:
    """
    Parse a fudgie.org episode page.
    Returns (turns, host_name, host_gender).
    Each turn: (speaker_name, text, start_sec, end_sec).
    """
    timestamps = _build_timestamps(html)

    # ── attributed format ────────────────────────────────────────────────────
    if 'data-speaker-name' in html:
        rows = _ATTRIBUTED_ROW_RE.findall(html)
        speaker_turns: list[Turn] = []
        cur_speaker: str | None = None
        cur_lines: list[str] = []
        cur_start = cur_end = -1.0
        for lid, speaker, raw_text in rows:
            speaker = speaker.strip().lower()
            text = html_lib.unescape(re.sub(r'<[^>]+>', '', raw_text)).strip()
            if not text:
                continue
            t = timestamps.get(lid, -1.0)
            if speaker != cur_speaker:
                if cur_speaker is not None and cur_lines:
                    speaker_turns.append((cur_speaker, ' '.join(cur_lines), cur_start, cur_end))
                cur_speaker = speaker
                cur_lines   = [text]
                cur_start   = t
                cur_end     = t
            else:
                cur_lines.append(text)
                if t >= 0:
                    cur_end = t
        if cur_speaker is not None and cur_lines:
            speaker_turns.append((cur_speaker, ' '.join(cur_lines), cur_start, cur_end))
        host_name, host_gender = _detect_host(speaker_turns)
        return speaker_turns, host_name, host_gender

    # ── unattributed format ───────────────────────────────────────────────────
    rows = _TEXT_CELL_RE.findall(html)
    speaker_turns_u: list[Turn] = []
    cur_lines_u: list[str] = []
    cur_start_u = cur_end_u = -1.0
    for lid, td_attrs, raw_text in rows:
        is_new = 'new-speaker-cell' in td_attrs
        text = html_lib.unescape(re.sub(r'<[^>]+>', '', raw_text)).strip()
        if not text:
            continue
        t = timestamps.get(lid, -1.0)
        if is_new and cur_lines_u:
            speaker_turns_u.append(("?", ' '.join(cur_lines_u), cur_start_u, cur_end_u))
            cur_lines_u = [text]
            cur_start_u = cur_end_u = t
        else:
            cur_lines_u.append(text)
            if t >= 0:
                if cur_start_u < 0:
                    cur_start_u = t
                cur_end_u = t
    if cur_lines_u:
        speaker_turns_u.append(("?", ' '.join(cur_lines_u), cur_start_u, cur_end_u))

    all_text = ' '.join(t for _, t, _, _ in speaker_turns_u).lower()
    votes: Counter = Counter()
    for token, (name, _) in _HOST_LOOKUP.items():
        count = len(re.findall(r'\b' + re.escape(token) + r'\b', all_text))
        if count:
            votes[name] += count
    host_name = votes.most_common(1)[0][0] if votes else ""
    host_gender = next((g for d, g in _HOST_LOOKUP.values() if d == host_name), "unknown") if host_name else "unknown"
    return speaker_turns_u, host_name, host_gender


# ══════════════════════════════════════════════════════════════════════════════
# CALLER EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _is_caller_name(name: str) -> bool:
    """True if speaker name looks like a caller (not a recognised host/figure)."""
    if name in ("unidentified", "?"):
        return True
    if re.search(r'\b(?:in|from)\b', name):
        return True
    return False


def _gender_from_name(first_name: str) -> tuple[str, str]:
    if not HAS_GENDER_GUESSER or not first_name:
        return "unknown", "unknown"
    result = _detector.get_gender(first_name.title())
    if result in ("male", "mostly_male"):
        return "male", "name"
    if result in ("female", "mostly_female"):
        return "female", "name"
    return "unknown", "unknown"


def _gender_from_salutation(text: str) -> str:
    m = len(SIR_RE.findall(text))
    f = len(MAAM_RE.findall(text))
    if f > m: return "female"
    if m > f: return "male"
    return "unknown"


_NAME_PATS = [
    re.compile(r'\b([A-Z][a-z]{1,20})(?:\'s)?\s+(?:calling|is calling)\b'),
    re.compile(r'\bline[.,]?\s+([A-Z][a-z]{1,20})\b'),
    re.compile(r'\b(?:here is|here\'?s|talk to)\s+([A-Z][a-z]{1,20})\b', re.I),
    re.compile(r'\b([A-Z][a-z]{1,20})\s+(?:in|from)\s+[A-Z][a-z]'),
    re.compile(r',\s+([A-Z][a-z]{1,20}),\s+(?:you\'?re on|good morning)'),
]
_BAD_NAMES = {
    "The", "Our", "Your", "Thank", "Good", "This", "That", "What", "Well",
    "Just", "Line", "Next", "Caller", "And", "Here", "Talk", "Washington",
    "Independent", "Republican", "Democrat", "Open", "Forum", "Phones",
}


def extract_callers(
    speaker_turns: list[Turn],
    episode_id: str,
    upload_date: str,
    host_name: str,
    host_gender: str,
) -> list[CallerTurn]:
    callers: list[CallerTurn] = []
    n = len(speaker_turns)
    unattributed = any(sp == "?" for sp, _, _, _ in speaker_turns)

    ep_start_hr = _ep_start_hour(episode_id)

    day_of_week = ""
    if upload_date:
        try:
            day_of_week = datetime.strptime(upload_date, "%Y-%m-%d").strftime("%A")
        except ValueError:
            pass

    for i, (speaker, text, start_sec, end_sec) in enumerate(speaker_turns):
        # ── unattributed mode: use party-line intro pattern ───────────────
        if unattributed:
            if not INTRO_PARTY_RE.search(text):
                continue

            pm = INTRO_PARTY_RE.search(text)
            party = pm.group(1).lower() if pm else "unknown"
            if party == "democratic":
                party = "democrat"

            # Collect all turns from i+1 until the next party-line intro.
            # Turns > 4 words = caller speech; turns that follow the last
            # long turn are likely the host's closing remark.
            block = []  # (text, start_sec, end_sec)
            for k in range(i + 1, min(i + 60, n)):
                _, kt, kt_start, kt_end = speaker_turns[k]
                if INTRO_PARTY_RE.search(kt):
                    break
                block.append((kt, kt_start, kt_end))

            caller_parts = []
            caller_start = -1.0
            caller_end   = -1.0
            last_long_bidx = -1
            for bidx, (kt, kt_start, kt_end) in enumerate(block):
                if len(kt.split()) > 4:
                    caller_parts.append(kt)
                    if caller_start < 0 and kt_start >= 0:
                        caller_start = kt_start
                    if kt_end >= 0:
                        caller_end = kt_end
                    last_long_bidx = bidx

            # Host response: short turns that trail after the last long caller turn
            _host_resp_parts = []
            if last_long_bidx >= 0 and last_long_bidx < len(block) - 1:
                _host_resp_parts = [t for t, _, _ in block[last_long_bidx + 1:] if t.strip()]
            host_response = ' '.join(_host_resp_parts).strip()

            caller_text = ' '.join(caller_parts).strip()

            words = caller_text.split()
            if len(words) < 15 or len(words) > 600:
                continue

            # Timing
            call_hour = -1
            if ep_start_hr >= 0 and caller_start >= 0:
                call_hour = ep_start_hr + int(caller_start // 3600)
            call_duration_sec = max(0.0, caller_end - caller_start) if caller_start >= 0 and caller_end >= caller_start else 0.0

            # Location from host intro text
            caller_city, caller_state = _extract_location(text)

            # Caller first name from intro line
            first_name = ""
            for pat in _NAME_PATS:
                mm = pat.search(text)
                if mm:
                    cand = mm.group(1)
                    if cand not in _BAD_NAMES:
                        first_name = cand
                        break

            gender, gender_src = _gender_from_name(first_name)
            if gender == "unknown":
                sal = _gender_from_salutation(caller_text)
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
                caller_city=caller_city,
                caller_state=caller_state,
                call_hour=call_hour,
                call_duration_sec=call_duration_sec,
                day_of_week=day_of_week,
                host_response_text=host_response,
            ))
            continue

        # ── attributed mode: use speaker name ─────────────────────────────
        if not _is_caller_name(speaker):
            continue

        words = text.split()
        if len(words) < 15 or len(words) > 400:
            continue

        # Party: look at preceding turns for host intro
        party = "unknown"
        for j in range(i - 1, max(i - 4, -1), -1):
            prev_sp, prev_text, _, _ = speaker_turns[j]
            if prev_sp in KNOWN_HOSTS or prev_sp == host_name.lower():
                pm = INTRO_PARTY_RE.search(prev_text)
                if pm:
                    party = pm.group(1).lower()
                    if party == "democratic":
                        party = "democrat"
                break

        # Location: try speaker name first ("roy in north dakota"), then host intro
        caller_city, caller_state = _extract_location(speaker)
        if not caller_state:
            for j in range(i - 1, max(i - 4, -1), -1):
                prev_sp, prev_text, _, _ = speaker_turns[j]
                if prev_sp in KNOWN_HOSTS or prev_sp == host_name.lower():
                    c, s = _extract_location(prev_text)
                    if s:
                        caller_city, caller_state = c, s
                    break

        # Timing
        call_hour = -1
        if ep_start_hr >= 0 and start_sec >= 0:
            call_hour = ep_start_hr + int(start_sec // 3600)
        call_duration_sec = max(0.0, end_sec - start_sec) if start_sec >= 0 and end_sec > start_sec else 0.0

        # Caller name from speaker tag
        first_name = ""
        if speaker not in ("unidentified", "?"):
            parts = speaker.split()
            if parts[0] not in {"penned", "tax", "the", "caller", "viewer"}:
                first_name = parts[0].capitalize()

        gender, gender_src = _gender_from_name(first_name)
        if gender == "unknown":
            closing = speaker_turns[i + 1][1] if i + 1 < n else ""
            sal = _gender_from_salutation(text + " " + closing)
            if sal != "unknown":
                gender, gender_src = sal, "salutation"

        # Host response: first host turn after caller finishes
        host_response = ""
        for j in range(i + 1, min(i + 6, n)):
            sp_j, txt_j, _, _ = speaker_turns[j]
            if sp_j in KNOWN_HOSTS or sp_j == host_name.lower():
                host_response = txt_j.strip()
                break
            if _is_caller_name(sp_j):
                break

        callers.append(CallerTurn(
            name=first_name,
            gender=gender,
            gender_src=gender_src,
            party=party,
            text=text,
            episode_id=episode_id,
            upload_date=upload_date,
            host_name=host_name,
            host_gender=host_gender,
            caller_city=caller_city,
            caller_state=caller_state,
            call_hour=call_hour,
            call_duration_sec=call_duration_sec,
            day_of_week=day_of_week,
            host_response_text=host_response,
        ))

    return callers


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze(callers: list[CallerTurn]) -> pd.DataFrame:
    rows = []
    for c in callers:
        text  = c.text.strip()
        words = text.split()
        sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()] or [text]
        qs    = [s for s in sents if s.endswith("?") or _Q_START.match(s)]
        wc    = len(words)
        rows.append({
            "gender":                 c.gender,
            "gender_src":             c.gender_src,
            "party":                  c.party,
            "episode_id":             c.episode_id,
            "upload_date":            c.upload_date,
            "name":                   c.name,
            "word_count":             wc,
            "sentence_count":         len(sents),
            "question_count":         len(qs),
            "question_ratio":         round(len(qs) / max(len(sents), 1), 3),
            "avg_words_per_sentence": round(wc / max(len(sents), 1), 2),
            "unique_word_ratio":      round(len(set(w.lower() for w in words)) / max(wc, 1), 3),
            "hedge_rate":             round(len(HEDGE_RE.findall(text)) / max(wc, 1), 4),
            "modal_rate":             round(len(MODAL_RE.findall(text)) / max(wc, 1), 4),
            "first_p_rate":           round(len(FIRST_P_RE.findall(text)) / max(wc, 1), 4),
            "second_p_rate":          round(len(SECOND_P_RE.findall(text)) / max(wc, 1), 4),
            "host_name":              c.host_name,
            "host_gender":            c.host_gender,
            "caller_city":            c.caller_city,
            "caller_state":           c.caller_state,
            "call_hour":              c.call_hour if c.call_hour >= 0 else None,
            "call_duration_sec":      round(c.call_duration_sec, 1) if c.call_duration_sec > 0 else None,
            "day_of_week":            c.day_of_week,
            "text":                   text,
            "host_response_text":     c.host_response_text,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scrape C-SPAN Washington Journal callers via fight.fudgie.org transcripts."
    )
    parser.add_argument("--episodes", type=int, default=50,
                        help="Max new episodes to scrape (default 50).")
    parser.add_argument("--output",   default="results/scraped/cspan_callers.csv",
                        help="Output CSV path.")
    parser.add_argument("--append",   action="store_true",
                        help="Load existing CSV, skip already-scraped episodes.")
    parser.add_argument("--delay",    type=float, default=1.5,
                        help="Seconds to wait between requests (default 1.5).")
    args = parser.parse_args()

    # ── load existing data ────────────────────────────────────────────────
    existing_df: pd.DataFrame | None = None
    skip_ids: set[str] = set()
    if args.append and os.path.exists(args.output):
        existing_df = pd.read_csv(args.output)
        if "episode_id" in existing_df.columns:
            # Strip #fragment from stored IDs so they match normalized incoming IDs
            skip_ids = set(
                existing_df["episode_id"].dropna().astype(str).str.split('#').str[0].unique()
            )
        print(f"Existing data: {len(existing_df)} rows from {len(skip_ids)} episode(s) — will skip those.")

    episodes = fetch_episode_list(args.episodes, skip_ids)
    if not episodes:
        print("No episodes found.")
        sys.exit(1)

    all_callers: list[CallerTurn] = []
    new_episodes = 0

    for idx, (ep_id, url) in enumerate(episodes, 1):
        if new_episodes >= args.episodes:
            break

        print(f"[{idx}/{len(episodes)}] {ep_id[:60]} ...", end=" ", flush=True)

        # Extract upload_date from episode ID (YYYYMMDD prefix)
        date_m = re.match(r'(\d{4})(\d{2})(\d{2})', ep_id)
        upload_date = f"{date_m.group(1)}-{date_m.group(2)}-{date_m.group(3)}" if date_m else ""

        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"fetch error: {e}")
            new_episodes += 1
            continue

        speaker_turns, host_name, host_gender = parse_episode(r.text)
        callers = extract_callers(speaker_turns, ep_id, upload_date, host_name, host_gender)
        print(f"{len(callers)} caller(s)  [{len(speaker_turns)} turns, host={host_name or '?'}]")
        new_episodes += 1
        all_callers.extend(callers)

        time.sleep(args.delay + random.uniform(0, 0.5))

    if not all_callers:
        if args.append and existing_df is not None:
            print("\nNo new callers found.")
            return
        print("\nNo caller turns found.")
        sys.exit(1)

    new_df = analyze(all_callers)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.append and existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nAppended {len(new_df)} rows → {args.output}  ({len(combined)} total)")
    else:
        new_df.to_csv(args.output, index=False)
        print(f"\nSaved {len(new_df)} rows → {args.output}")

    # Quick summary
    labeled = new_df[new_df["gender"].isin(["male","female"])]
    print(f"New rows: {len(new_df)}  (female={len(labeled[labeled.gender=='female'])}, "
          f"male={len(labeled[labeled.gender=='male'])}, "
          f"unknown={(new_df.gender=='unknown').sum()})")


if __name__ == "__main__":
    main()
