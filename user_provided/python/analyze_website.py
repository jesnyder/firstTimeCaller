#!/usr/bin/env python3
"""
C-SPAN Caller Gender Analysis — Static Website Generator

Reads cspan_callers.csv, computes stats in Python, and writes:
  docs/index.html  — interactive page (Plotly.js charts + Tabulator table)
  docs/style.css   — external stylesheet

No running server needed — open docs/index.html directly in any browser.

Install:
    pip install pandas scipy

Run:
    python user_provided/python/analyze_website.py
    python user_provided/python/analyze_website.py \\
        --csv results/scraped/cspan_callers.csv --output docs/index.html
"""

import re
import json
import argparse
import sys
import os
from collections import Counter

import pandas as pd

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderAnalyzer
    _vader = _VaderAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("WARNING: vaderSentiment not installed — sentiment analysis disabled.\n"
          "  pip install vaderSentiment\n")

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

# ── syntactic patterns ────────────────────────────────────────────────────────
FILLER = {"um", "uh", "well", "yes", "yeah", "no", "so", "and", "but",
          "i", "you", "know", "like", "okay", "ok", "hi", "hello", "good"}

HEDGE_RE    = re.compile(
    r"\b(i think|i feel|i believe|i guess|maybe|perhaps|possibly|"
    r"it seems|sort of|kind of|i was wondering|i'm not sure|"
    r"i don't know|might be|could be)\b", re.I
)
MODAL_RE    = re.compile(r"\b(would|could|should|might|may|ought)\b", re.I)
FIRST_P_RE  = re.compile(r"\b(i|me|my|mine|myself)\b", re.I)
SECOND_P_RE = re.compile(r"\b(you|your|yours|yourself)\b", re.I)
Q_OPENER_RE = re.compile(
    r"^(why|what|how|when|where|who|which|whose|wouldn't|don't|"
    r"isn't|aren't|can't|didn't)\b", re.I
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def _sentiment_turn(text: str) -> dict:
    """Run VADER on each sentence; return mean compound + fraction pos/neg/neu."""
    if not HAS_VADER or not text.strip():
        return {"sent_compound": 0.0, "sent_pos": 0.0, "sent_neg": 0.0, "sent_neu": 1.0}
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        sentences = [text]
    scores = [_vader.polarity_scores(s) for s in sentences]
    compounds = [s["compound"] for s in scores]
    n = len(compounds)
    return {
        "sent_compound": round(sum(compounds) / n, 4),
        "sent_pos":      round(sum(1 for c in compounds if c >  0.05) / n, 3),
        "sent_neg":      round(sum(1 for c in compounds if c < -0.05) / n, 3),
        "sent_neu":      round(sum(1 for c in compounds if -0.05 <= c <= 0.05) / n, 3),
    }


def load_and_enrich(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df.sort_values(["upload_date", "episode_id"], na_position="last")
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    if len(df) < before:
        print(f"  Removed {before - len(df)} duplicate rows (same text, different episode replay).")
    text = df["text"].fillna("")
    df["hedge_rate"]    = text.apply(lambda t: round(len(HEDGE_RE.findall(t))    / max(len(t.split()), 1), 4))
    df["modal_rate"]    = text.apply(lambda t: round(len(MODAL_RE.findall(t))    / max(len(t.split()), 1), 4))
    df["first_p_rate"]  = text.apply(lambda t: round(len(FIRST_P_RE.findall(t))  / max(len(t.split()), 1), 4))
    df["second_p_rate"] = text.apply(lambda t: round(len(SECOND_P_RE.findall(t)) / max(len(t.split()), 1), 4))
    df["q_opener"]      = text.apply(lambda t: bool(Q_OPENER_RE.match(t.strip())))

    def _opener_word(t):
        for w in t.lower().split():
            w = re.sub(r"[^a-z']", "", w)
            if w and w not in FILLER:
                return w
        return ""
    df["opener_word"] = text.apply(_opener_word)

    # Normalise host name casing variants ("John Mcardle" → "John McArdle")
    _HOST_NORM = {"John Mcardle": "John McArdle"}
    _GENDER_NORM = {"John McArdle": "male"}
    if "host_name" in df.columns:
        df["host_name"] = df["host_name"].replace(_HOST_NORM)
    if "host_gender" in df.columns and "host_name" in df.columns:
        for name, gender in _GENDER_NORM.items():
            mask = (df["host_name"] == name) & (df["host_gender"] == "unknown")
            df.loc[mask, "host_gender"] = gender

    if HAS_VADER:
        print("  Computing sentence-level sentiment (VADER) ...")
        sent_df = text.apply(_sentiment_turn).apply(pd.Series)
        df = pd.concat([df, sent_df], axis=1)

    return df


def _mwu_p(a, b) -> str | None:
    if not HAS_SCIPY or len(a) < 3 or len(b) < 3:
        return None
    _, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    if p < 0.001:
        return "p < 0.001 ***"
    if p < 0.01:
        return f"p = {p:.3f} **"
    if p < 0.05:
        return f"p = {p:.3f} *"
    return f"p = {p:.3f} (ns)"


def _compute_interaction_data(df: pd.DataFrame) -> dict:
    """
    All permutations of caller gender × host gender.
    Returns empty dict if host_gender column is missing or has no data.
    """
    if "host_gender" not in df.columns:
        return {}

    metrics = ["word_count", "sentence_count", "question_ratio",
               "avg_words_per_sentence", "hedge_rate", "unique_word_ratio"]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for caller_g in ["female", "male", "unknown"]:
        for host_g in ["female", "male", "unknown"]:
            sub = df[(df["gender"] == caller_g) & (df["host_gender"] == host_g)]
            if sub.empty:
                continue
            entry = {
                "caller_gender": caller_g,
                "host_gender":   host_g,
                "label":         f"{caller_g.capitalize()} caller\n→ {host_g.capitalize()} host",
                "n":             len(sub),
            }
            for m in metrics:
                entry[m] = round(float(sub[m].mean()), 3)
            rows.append(entry)

    # Host breakdown summary (how many episodes per host)
    host_counts = []
    if "host_name" in df.columns:
        hc = df[df["host_name"] != ""].groupby(["host_name", "host_gender"]).size().reset_index(name="n_turns")
        for _, r in hc.iterrows():
            host_counts.append({"name": r["host_name"], "gender": r["host_gender"], "n_turns": int(r["n_turns"])})

    return {"combos": rows, "hostCounts": host_counts}


def _compute_time_series(df: pd.DataFrame) -> dict:
    """
    Monthly caller counts and total words, split by gender, plus cumulative totals.
    Returns empty dict if upload_date column is missing or all blank.
    """
    if "upload_date" not in df.columns:
        return {}
    dated = df[df["upload_date"].notna() & (df["upload_date"] != "")].copy()
    if dated.empty:
        return {}

    dated["month"] = pd.to_datetime(dated["upload_date"], errors="coerce").dt.to_period("M").astype(str)
    dated = dated[dated["month"].notna() & (dated["month"] != "NaT")]
    if dated.empty:
        return {}

    all_months = sorted(dated["month"].unique())
    result: dict = {"months": all_months}

    for g in ("all", "female", "male", "unknown"):
        sub = dated if g == "all" else dated[dated["gender"] == g]
        counts = sub.groupby("month").size().reindex(all_months, fill_value=0)
        words  = sub.groupby("month")["word_count"].sum().reindex(all_months, fill_value=0)
        result[f"{g}_counts"]     = counts.tolist()
        result[f"{g}_words"]      = words.tolist()
        result[f"cum_{g}_counts"] = counts.cumsum().tolist()
        result[f"cum_{g}_words"]  = words.cumsum().tolist()

    # Female ratio among labeled callers (kept for backwards compat)
    labeled = dated[dated["gender"].isin(["female", "male"])]
    if not labeled.empty:
        lgrp = labeled.groupby("month")["gender"].value_counts().unstack(fill_value=0)
        lgrp = lgrp.reindex(columns=["female", "male"], fill_value=0).reindex(all_months, fill_value=0)
        lgrp["total"] = lgrp["female"] + lgrp["male"]
        result["female_ratio"] = (
            (lgrp["female"] / lgrp["total"].replace(0, float("nan"))).fillna(0).round(3).tolist()
        )
    else:
        result["female_ratio"] = [0] * len(all_months)

    return result


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "you", "he", "she",
    "we", "they", "it", "this", "that", "these", "those", "my", "your",
    "his", "her", "our", "their", "its", "me", "him", "us", "them",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "not", "no", "so", "if", "as", "by", "from", "up", "about", "into",
    "through", "during", "just", "more", "also", "than", "then", "there",
    "all", "any", "both", "each", "few", "most", "other", "some", "such",
    "own", "same", "too", "very", "s", "t", "don", "m", "re", "ve", "ll",
    "um", "uh", "yeah", "okay", "ok",
}


def _pos_tag_words(words: list[str]) -> dict[str, str]:
    """Return {word: simplified_pos} using NLTK averaged_perceptron_tagger if available."""
    _TAG_MAP = {
        'NN': 'noun',   'NNS': 'noun',   'NNP': 'noun',  'NNPS': 'noun',
        'VB': 'verb',   'VBD': 'verb',   'VBG': 'verb',  'VBN': 'verb',
        'VBP': 'verb',  'VBZ': 'verb',   'MD': 'verb',
        'JJ': 'adjective', 'JJR': 'adjective', 'JJS': 'adjective',
        'RB': 'adverb', 'RBR': 'adverb', 'RBS': 'adverb',
        'IN': 'preposition', 'TO': 'preposition',
        'PRP': 'pronoun', 'PRP$': 'pronoun', 'WP': 'pronoun', 'WP$': 'pronoun',
        'CC': 'conjunction',
        'DT': 'determiner', 'PDT': 'determiner', 'WDT': 'determiner',
        'UH': 'interjection',
    }
    try:
        import nltk
        try:
            tagged = nltk.pos_tag(words)
        except LookupError:
            for res in ('averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'):
                try:
                    nltk.download(res, quiet=True)
                except Exception:
                    pass
            tagged = nltk.pos_tag(words)
        return {w: _TAG_MAP.get(t, 'other') for w, t in tagged}
    except Exception:
        # Heuristic fallback
        result = {}
        for w in words:
            if w.endswith('ly'):
                result[w] = 'adverb'
            elif w.endswith(('tion', 'ment', 'ness', 'ity', 'ism', 'ance', 'ence')):
                result[w] = 'noun'
            elif w.endswith(('ous', 'ful', 'less', 'ive', 'ic', 'able', 'ible')):
                result[w] = 'adjective'
            elif w.endswith(('ing', 'ize', 'ise')):
                result[w] = 'verb'
            else:
                result[w] = '—'
        return result


def _compute_word_freq(df: pd.DataFrame) -> list[dict]:
    """
    Count word occurrences and unique-caller reach, split by gender.
    Returns % of callers (per gender) who used each word at least once.
    """
    from collections import defaultdict
    _word_re = re.compile(r"[a-z']+")
    counts: dict[str, Counter] = {"female": Counter(), "male": Counter(), "all": Counter()}
    # sets of row indices per gender per word
    caller_sets: dict[str, dict[str, set]] = defaultdict(lambda: {"all": set(), "female": set(), "male": set()})

    n_female = n_male = n_all = 0
    for idx, row in df.iterrows():
        text   = str(row.get("text", "") or "")
        gender = row.get("gender", "unknown")
        n_all += 1
        if gender == "female": n_female += 1
        if gender == "male":   n_male   += 1
        word_list = [w for w in _word_re.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]
        counts["all"].update(word_list)
        if gender in counts:
            counts[gender].update(word_list)
        for w in set(word_list):
            caller_sets[w]["all"].add(idx)
            if gender == "female":
                caller_sets[w]["female"].add(idx)
            elif gender == "male":
                caller_sets[w]["male"].add(idx)

    top_words = sorted(counts["all"].keys(), key=lambda w: -counts["all"][w])[:2000]
    pos_map = _pos_tag_words(top_words)

    result = []
    for w in top_words:
        nc_all = len(caller_sets[w]["all"])
        nc_f   = len(caller_sets[w]["female"])
        nc_m   = len(caller_sets[w]["male"])
        result.append({
            "word":            w,
            "pos":             pos_map.get(w, "—"),
            "count":           counts["all"][w],
            "female_count":    counts["female"].get(w, 0),
            "male_count":      counts["male"].get(w, 0),
            "n_callers":       nc_all,
            "n_female_callers": nc_f,
            "n_male_callers":  nc_m,
            "pct_all":   round(100 * nc_all / n_all,    1) if n_all    else 0,
            "pct_female": round(100 * nc_f  / n_female, 1) if n_female else 0,
            "pct_male":   round(100 * nc_m  / n_male,   1) if n_male   else 0,
            "pct_diff":  round((100 * nc_f / n_female if n_female else 0) -
                               (100 * nc_m / n_male   if n_male   else 0), 1),
        })
    return result


def _compute_geo_data(df: pd.DataFrame) -> dict:
    """Aggregate caller counts by state, split by gender and party."""
    if "caller_state" not in df.columns:
        return {}
    geo = df[df["caller_state"].notna() & (df["caller_state"] != "")].copy()
    if geo.empty:
        return {}

    # State centroids (lat, lon) for bubble map
    CENTROIDS = {
        "AL":(32.8,-86.8),"AK":(64.2,-153.4),"AZ":(34.3,-111.1),"AR":(34.8,-92.2),
        "CA":(36.8,-119.4),"CO":(39.0,-105.5),"CT":(41.6,-72.7),"DE":(39.0,-75.5),
        "FL":(28.7,-82.5),"GA":(32.7,-83.4),"HI":(20.9,-157.0),"ID":(44.4,-114.6),
        "IL":(40.0,-89.2),"IN":(40.3,-86.1),"IA":(42.0,-93.2),"KS":(38.5,-98.4),
        "KY":(37.5,-85.3),"LA":(31.1,-91.9),"ME":(45.4,-69.2),"MD":(39.0,-76.8),
        "MA":(42.3,-71.8),"MI":(44.3,-85.4),"MN":(46.4,-93.1),"MS":(32.7,-89.7),
        "MO":(38.4,-92.5),"MT":(47.0,-110.0),"NE":(41.5,-99.9),"NV":(38.5,-117.1),
        "NH":(43.7,-71.6),"NJ":(40.1,-74.5),"NM":(34.5,-106.2),"NY":(42.9,-75.5),
        "NC":(35.5,-79.4),"ND":(47.5,-100.5),"OH":(40.4,-82.8),"OK":(35.6,-96.9),
        "OR":(44.6,-122.1),"PA":(40.6,-77.3),"RI":(41.7,-71.6),"SC":(33.9,-80.9),
        "SD":(44.4,-100.2),"TN":(35.9,-86.4),"TX":(31.5,-99.3),"UT":(39.3,-111.1),
        "VT":(44.0,-72.7),"VA":(37.5,-79.5),"WA":(47.4,-120.6),"WV":(38.9,-80.5),
        "WI":(44.3,-89.8),"WY":(43.0,-107.6),"DC":(38.9,-77.0),
    }

    by_state = []
    grp = geo.groupby("caller_state")
    for state, rows in grp:
        if state not in CENTROIDS:
            continue
        lat, lon = CENTROIDS[state]
        parties = {}
        for p in ["republican", "democrat", "independent"]:
            parties[p] = int((rows["party"] == p).sum())
        by_state.append({
            "state":       state,
            "lat":         lat,
            "lon":         lon,
            "total":       int(len(rows)),
            "female":      int((rows["gender"] == "female").sum()),
            "male":        int((rows["gender"] == "male").sum()),
            "unknown":     int((rows["gender"] == "unknown").sum()),
            "republican":  parties["republican"],
            "democrat":    parties["democrat"],
            "independent": parties["independent"],
        })
    by_state.sort(key=lambda r: -r["total"])
    return {"byState": by_state}


def _compute_sentiment_data(df: pd.DataFrame) -> dict:
    """Aggregate VADER sentiment by gender and by party × gender."""
    if "sent_compound" not in df.columns:
        return {}

    parties  = ["republican", "democrat", "independent"]
    genders  = ["female", "male", "unknown"]

    def _agg(sub):
        if sub.empty:
            return None
        return {
            "n":        int(len(sub)),
            "compound": round(float(sub["sent_compound"].mean()), 4),
            "pos":      round(float(sub["sent_pos"].mean()), 3),
            "neg":      round(float(sub["sent_neg"].mean()), 3),
            "neu":      round(float(sub["sent_neu"].mean()), 3),
        }

    # By gender
    by_gender = {}
    for g in genders:
        sub = df[df["gender"] == g]
        by_gender[g] = _agg(sub)

    # By party × gender
    by_party = {}
    for p in parties:
        by_party[p] = {}
        for g in genders:
            sub = df[(df["party"] == p) & (df["gender"] == g)]
            by_party[p][g] = _agg(sub)

    # Violin data for compound score (female / male)
    labeled = df[df["gender"].isin(["female", "male"])]
    violin = {
        "female": labeled[labeled["gender"] == "female"]["sent_compound"].dropna().round(4).tolist(),
        "male":   labeled[labeled["gender"] == "male"]["sent_compound"].dropna().round(4).tolist(),
    }

    # Scatter: pct positive sentences vs call duration
    dur_col = "call_duration_sec" if "call_duration_sec" in df.columns else None
    scatter = []
    if dur_col:
        sub = df[df[dur_col].notna() & (df[dur_col] > 0) & df["sent_pos"].notna()].copy()
        for _, row in sub.iterrows():
            scatter.append({
                "x":       round(float(row["sent_pos"]), 3),
                "y":       round(float(row[dur_col]), 1),
                "gender":  str(row.get("gender", "unknown")),
                "party":   str(row.get("party", "")),
                "name":    str(row.get("name", "")),
                "compound": round(float(row["sent_compound"]), 3),
                "text":    str(row.get("text", ""))[:400],
            })

    return {
        "byGender":  by_gender,
        "byParty":   by_party,
        "violin":    violin,
        "scatter":   scatter,
    }


_COMPLIMENT_RE = re.compile(
    r"\b(good question|great question|excellent question|"
    r"good point|great point|excellent point|very interesting|"
    r"interesting point|good comment|great comment|good observation|"
    r"excellent comment|that'?s a good|that'?s a great|well said)\b", re.I
)
_WORD_RE = re.compile(r"[a-z']+")


def _jaccard(text1: str, text2: str) -> float:
    w1 = set(_WORD_RE.findall(text1.lower())) - _STOPWORDS
    w2 = set(_WORD_RE.findall(text2.lower())) - _STOPWORDS
    if not w1 or not w2:
        return 0.0
    return round(len(w1 & w2) / len(w1 | w2), 4)


def _compute_responsiveness(df: pd.DataFrame) -> dict:
    """Compute host-response metrics for calls that have host_response_text."""
    if "host_response_text" not in df.columns:
        return {}
    resp = df[df["host_response_text"].notna() & (df["host_response_text"].str.strip() != "")].copy()
    if resp.empty:
        return {}

    resp["host_resp_words"]  = resp["host_response_text"].apply(lambda t: len(t.split()))
    resp["word_overlap"]     = resp.apply(
        lambda row: _jaccard(str(row.get("text", "")), str(row["host_response_text"])), axis=1
    )
    resp["host_compliment"]  = resp["host_response_text"].apply(lambda t: bool(_COMPLIMENT_RE.search(t)))
    resp["host_followup_q"]  = resp["host_response_text"].apply(lambda t: "?" in t)
    if HAS_VADER:
        resp["host_resp_sentiment"] = resp["host_response_text"].apply(
            lambda t: round(_vader.polarity_scores(t)["compound"], 4) if t.strip() else 0.0
        )

    genders = ["female", "male", "unknown"]
    by_gender = {}
    for g in genders:
        sub = resp[resp["gender"] == g]
        if sub.empty:
            continue
        by_gender[g] = {
            "n":               int(len(sub)),
            "avg_host_words":  round(float(sub["host_resp_words"].mean()), 1),
            "avg_overlap":     round(float(sub["word_overlap"].mean()), 4),
            "compliment_rate": round(float(sub["host_compliment"].mean()), 4),
            "followup_rate":   round(float(sub["host_followup_q"].mean()), 4),
        }

    # Table rows
    keep_cols = ["gender", "party", "name", "upload_date", "word_count",
                 "host_resp_words", "word_overlap", "host_compliment", "host_followup_q"]
    if HAS_VADER and "host_resp_sentiment" in resp.columns:
        keep_cols.append("host_resp_sentiment")
    keep_cols = [c for c in keep_cols if c in resp.columns]

    table_data = []
    for _, row in resp.iterrows():
        entry = {c: (bool(row[c]) if c in ("host_compliment", "host_followup_q") else row[c])
                 for c in keep_cols}
        entry["text"]               = str(row.get("text", ""))[:300]
        entry["host_response_text"] = str(row.get("host_response_text", ""))[:400]
        table_data.append(entry)

    return {
        "byGender":        by_gender,
        "tableData":       table_data,
        "n_with_response": int(len(resp)),
    }


_DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def _compute_day_of_week(df: pd.DataFrame) -> dict:
    """Caller counts and female/male ratios by day of week."""
    if "day_of_week" not in df.columns:
        return {}
    dated = df[df["day_of_week"].notna()].copy()
    if dated.empty:
        return {}

    days = [d for d in _DOW_ORDER if d in dated["day_of_week"].unique()]

    result = {"days": days}
    for g in ("all", "female", "male", "unknown"):
        sub = dated if g == "all" else dated[dated["gender"] == g]
        counts = sub.groupby("day_of_week").size().reindex(days, fill_value=0)
        result[f"{g}_counts"] = counts.tolist()

    # Female fraction of gender-labeled callers per day
    labeled = dated[dated["gender"].isin(["female", "male"])]
    f_counts = labeled[labeled["gender"] == "female"].groupby("day_of_week").size().reindex(days, fill_value=0)
    total_labeled = labeled.groupby("day_of_week").size().reindex(days, fill_value=0)
    result["female_fraction"] = [
        round(float(f) / float(t), 4) if t > 0 else None
        for f, t in zip(f_counts, total_labeled)
    ]
    result["total_labeled"] = total_labeled.tolist()
    return result


def _compute_sentiment_over_time(df: pd.DataFrame) -> dict:
    """Monthly mean sentiment compound score, % negative sentences, by gender."""
    if "sent_compound" not in df.columns or "upload_date" not in df.columns:
        return {}
    dated = df[df["upload_date"].notna()].copy()
    dated["month"] = pd.to_datetime(dated["upload_date"], errors="coerce").dt.to_period("M").astype(str)
    dated = dated[dated["month"].notna() & (dated["month"] != "NaT")]
    if dated.empty:
        return {}

    all_months = sorted(dated["month"].unique())
    result = {"months": all_months}
    for g in ("all", "female", "male", "unknown"):
        sub = dated if g == "all" else dated[dated["gender"] == g]
        compound = sub.groupby("month")["sent_compound"].mean().reindex(all_months)
        neg_pct  = sub.groupby("month")["sent_neg"].mean().reindex(all_months)
        result[f"{g}_compound"] = [round(float(v), 4) if pd.notna(v) else None for v in compound]
        result[f"{g}_neg_pct"]  = [round(float(v), 4) if pd.notna(v) else None for v in neg_pct]
    return result


def _compute_sankey(df: pd.DataFrame) -> dict:
    """Build node/link data for all Sankey diagrams."""

    OUT_VALS   = ["Substantive answer", "Acknowledged", "Brief / dismissed"]
    OUT_COLORS = ["#2E7D32", "#FFC107", "#EF5350"]
    CTYPE_VALS   = ["Statement (no question)", "Has questions"]
    CTYPE_COLORS = ["#607D8B", "#1565C0"]

    def _tab(df_sub, src_col, tgt_col, src_vals, tgt_vals, src_off, tgt_off):
        links = []
        for si, sv in enumerate(src_vals):
            for ti, tv in enumerate(tgt_vals):
                v = int(((df_sub[src_col] == sv) & (df_sub[tgt_col] == tv)).sum())
                if v > 0:
                    links.append({"source": si + src_off, "target": ti + tgt_off, "value": v})
        return links

    def _sk(nodes, colors, *link_batches):
        return {"nodes": nodes, "colors": colors,
                "links": [lk for batch in link_batches for lk in batch]}

    # ── shared derived columns ────────────────────────────────────────────────
    df = df.copy()
    df["call_type"] = df["question_ratio"].apply(
        lambda q: "Has questions" if q > 0 else "Statement (no question)")
    df["caller_len"] = df["word_count"].apply(
        lambda w: "Short (≤50w)" if w <= 50 else ("Medium (51–150w)" if w <= 150 else "Long (>150w)"))

    # ── host-response subset ──────────────────────────────────────────────────
    has_resp = df["host_response_text"].notna() & (df["host_response_text"].str.strip() != "")
    resp = df[has_resp].copy()

    if not resp.empty:
        resp["host_resp_words"] = resp["host_response_text"].apply(lambda t: len(str(t).split()))
        resp["host_followup_q"] = resp["host_response_text"].apply(lambda t: "?" in str(t))
        resp["resp_len_bin"] = resp["host_resp_words"].apply(
            lambda w: "Brief (<20w)" if w < 20 else ("Medium (20–60w)" if w <= 60 else "Long (>60w)"))

        def _outcome(row):
            rw, fu = row["host_resp_words"], row["host_followup_q"]
            if rw > 60 or (rw >= 20 and fu): return "Substantive answer"
            if rw >= 20:                      return "Acknowledged"
            return "Brief / dismissed"
        resp["outcome"] = resp.apply(_outcome, axis=1)

        # sentiment bucket
        if "sent_compound" in resp.columns:
            resp["tone"] = resp["sent_compound"].apply(
                lambda c: "Negative tone" if c < -0.05 else ("Positive tone" if c > 0.05 else "Neutral tone"))

        # question count bucket
        resp["q_bucket"] = resp["question_count"].apply(
            lambda x: "0 questions" if x == 0 else
                      ("1 question"  if x == 1 else
                       ("2 questions" if x == 2 else "3+ questions")))

        # vocabulary diversity
        resp["vocab_bin"] = resp["unique_word_ratio"].apply(
            lambda r: "High variety\n(TTR >0.80)" if r > 0.80 else
                      ("Medium variety\n(TTR 0.60–0.80)" if r >= 0.60 else "Low variety\n(TTR <0.60)"))

        # hedging
        resp["hedge_bin"] = resp["hedge_rate"].apply(
            lambda h: "Uses hedging" if h > 0 else "No hedging language")

        # day of week (keep only days with ≥20 rows in resp subset)
        if "day_of_week" in resp.columns:
            day_counts = resp["day_of_week"].value_counts()
            resp["dow_bin"] = resp["day_of_week"].apply(
                lambda d: d if day_counts.get(d, 0) >= 20 else None)

    # ══════════════════════════════════════════════════════════════════════════
    # Sankey 1  Party → Call Type → Caller Length  (ALL rows)
    # ══════════════════════════════════════════════════════════════════════════
    party_vals   = ["republican", "democrat", "independent", "unknown"]
    party_labels = ["Republican", "Democrat", "Independent", "Unknown party"]
    clen_vals    = ["Short (≤50w)", "Medium (51–150w)", "Long (>150w)"]
    s1 = _sk(
        party_labels + CTYPE_VALS + clen_vals,
        ["#c0392b","#2980b9","#27ae60","#95a5a6"] + CTYPE_COLORS + ["#FFA726","#42A5F5","#AB47BC"],
        _tab(df, "party",     "call_type",  party_vals,  CTYPE_VALS, 0,                   4),
        _tab(df, "call_type", "caller_len", CTYPE_VALS,  clen_vals,  4,                   6),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Sankey 2  Call Type → Response Length → Follow-up  (resp rows)
    # ══════════════════════════════════════════════════════════════════════════
    rlen_vals = ["Brief (<20w)", "Medium (20–60w)", "Long (>60w)"]
    fu_vals   = ["Host asks follow-up", "Host responds only"]
    s2 = _sk(
        CTYPE_VALS + rlen_vals + fu_vals,
        CTYPE_COLORS + ["#EF9A9A","#FFB74D","#66BB6A"] + ["#2E7D32","#9E9E9E"],
        _tab(resp, "call_type",    "resp_len_bin",  CTYPE_VALS, rlen_vals, 0, 2),
        _tab(resp, "resp_len_bin", "followup_label" if "followup_label" in resp.columns else "host_followup_q",
             rlen_vals, fu_vals, 2, 5) if not resp.empty else [],
    ) if not resp.empty else {"nodes": [], "colors": [], "links": []}

    if not resp.empty:
        resp["followup_label"] = resp["host_followup_q"].apply(
            lambda b: "Host asks follow-up" if b else "Host responds only")
        s2 = _sk(
            CTYPE_VALS + rlen_vals + fu_vals,
            CTYPE_COLORS + ["#EF9A9A","#FFB74D","#66BB6A"] + ["#2E7D32","#9E9E9E"],
            _tab(resp, "call_type",    "resp_len_bin",   CTYPE_VALS, rlen_vals, 0, 2),
            _tab(resp, "resp_len_bin", "followup_label", rlen_vals,  fu_vals,   2, 5),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Sankey 3  Caller Length → Call Type → Outcome  (resp rows)
    # ══════════════════════════════════════════════════════════════════════════
    clen3 = ["Short (≤50w)", "Medium (51–150w)", "Long (>150w)"]
    s3 = _sk(
        clen3 + CTYPE_VALS + OUT_VALS,
        ["#FFA726","#42A5F5","#AB47BC"] + CTYPE_COLORS + OUT_COLORS,
        _tab(resp, "caller_len", "call_type", clen3,      CTYPE_VALS, 0, 3),
        _tab(resp, "call_type",  "outcome",   CTYPE_VALS, OUT_VALS,   3, 5),
    ) if not resp.empty else {"nodes": [], "colors": [], "links": []}

    # ══════════════════════════════════════════════════════════════════════════
    # Sankey 4  Individual Host → Call Type → Outcome
    #           Which host gives the most substantive answers?
    # ══════════════════════════════════════════════════════════════════════════
    host_vals = ["Greta Brawner", "John McArdle", "Kimberly Adams", "Pedro Echevarria"]
    host_colors = ["#AD1457", "#1565C0", "#6A1B9A", "#00695C"]
    s4 = _sk(
        host_vals + CTYPE_VALS + OUT_VALS,
        host_colors + CTYPE_COLORS + OUT_COLORS,
        _tab(resp, "host_name", "call_type", host_vals,  CTYPE_VALS, 0, 4),
        _tab(resp, "call_type", "outcome",   CTYPE_VALS, OUT_VALS,   4, 6),
    ) if not resp.empty else {"nodes": [], "colors": [], "links": []}

    # ══════════════════════════════════════════════════════════════════════════
    # Sankey 5  Day of Week → Call Type → Outcome
    #           Are some days better for getting a substantive answer?
    # ══════════════════════════════════════════════════════════════════════════
    if not resp.empty and "day_of_week" in resp.columns:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_counts = resp["day_of_week"].value_counts()
        dow_vals = [d for d in dow_order if day_counts.get(d, 0) >= 20]
        dow_colors = ["#5C6BC0","#26A69A","#EF5350","#FFA726","#66BB6A","#AB47BC","#78909C"][:len(dow_vals)]
        s5 = _sk(
            dow_vals + CTYPE_VALS + OUT_VALS,
            dow_colors + CTYPE_COLORS + OUT_COLORS,
            _tab(resp, "day_of_week", "call_type", dow_vals,   CTYPE_VALS, 0,             len(dow_vals)),
            _tab(resp, "call_type",   "outcome",   CTYPE_VALS, OUT_VALS,   len(dow_vals), len(dow_vals) + 2),
        )
    else:
        s5 = {"nodes": [], "colors": [], "links": []}

    # ══════════════════════════════════════════════════════════════════════════
    # Sankey 6  Question Count → Caller Length → Outcome
    #           Is there a sweet spot for how many questions to ask?
    # ══════════════════════════════════════════════════════════════════════════
    q_bucket_vals   = ["0 questions", "1 question", "2 questions", "3+ questions"]
    q_bucket_colors = ["#607D8B", "#FFA726", "#EF6C00", "#B71C1C"]
    s6 = _sk(
        q_bucket_vals + clen3 + OUT_VALS,
        q_bucket_colors + ["#FFA726","#42A5F5","#AB47BC"] + OUT_COLORS,
        _tab(resp, "q_bucket",   "caller_len", q_bucket_vals, clen3,    0,                    4),
        _tab(resp, "caller_len", "outcome",    clen3,         OUT_VALS, 4,                    7),
    ) if not resp.empty else {"nodes": [], "colors": [], "links": []}

    # ══════════════════════════════════════════════════════════════════════════
    # Sankey 7  Caller Tone → Call Type → Outcome
    #           Does the emotional tone of the call affect the response?
    # ══════════════════════════════════════════════════════════════════════════
    tone_vals   = ["Negative tone", "Neutral tone", "Positive tone"]
    tone_colors = ["#EF5350", "#90A4AE", "#66BB6A"]
    s7 = _sk(
        tone_vals + CTYPE_VALS + OUT_VALS,
        tone_colors + CTYPE_COLORS + OUT_COLORS,
        _tab(resp, "tone",      "call_type", tone_vals,  CTYPE_VALS, 0, 3),
        _tab(resp, "call_type", "outcome",   CTYPE_VALS, OUT_VALS,   3, 5),
    ) if not resp.empty and "tone" in resp.columns else {"nodes": [], "colors": [], "links": []}

    # ══════════════════════════════════════════════════════════════════════════
    # Sankey 8  Hedging Language → Call Type → Outcome
    #           Does hedging / tentative language help land a substantive answer?
    # ══════════════════════════════════════════════════════════════════════════
    hedge_vals   = ["No hedging language", "Uses hedging"]
    hedge_colors = ["#78909C", "#7B1FA2"]
    s8 = _sk(
        hedge_vals + CTYPE_VALS + OUT_VALS,
        hedge_colors + CTYPE_COLORS + OUT_COLORS,
        _tab(resp, "hedge_bin", "call_type", hedge_vals,  CTYPE_VALS, 0, 2),
        _tab(resp, "call_type", "outcome",   CTYPE_VALS,  OUT_VALS,   2, 4),
    ) if not resp.empty else {"nodes": [], "colors": [], "links": []}

    # ══════════════════════════════════════════════════════════════════════════
    # Sankey 9  Vocabulary Diversity → Call Type → Outcome
    #           Does using richer or more varied vocabulary help?
    # ══════════════════════════════════════════════════════════════════════════
    vocab_vals   = ["Low variety\n(TTR <0.60)", "Medium variety\n(TTR 0.60–0.80)", "High variety\n(TTR >0.80)"]
    vocab_colors = ["#FF7043", "#FFA726", "#42A5F5"]
    s9 = _sk(
        vocab_vals + CTYPE_VALS + OUT_VALS,
        vocab_colors + CTYPE_COLORS + OUT_COLORS,
        _tab(resp, "vocab_bin", "call_type", vocab_vals,  CTYPE_VALS, 0, 3),
        _tab(resp, "call_type", "outcome",   CTYPE_VALS,  OUT_VALS,   3, 5),
    ) if not resp.empty else {"nodes": [], "colors": [], "links": []}

    return {
        "n_all":  int(len(df)),
        "n_resp": int(len(resp)),
        "sankey1": s1, "sankey2": s2, "sankey3": s3,
        "sankey4": s4, "sankey5": s5, "sankey6": s6,
        "sankey7": s7, "sankey8": s8, "sankey9": s9,
    }


def _compute_effective_calls(df: pd.DataFrame) -> dict:
    """
    Compare linguistic / temporal features between calls that received a
    Substantive answer vs those that received a Brief response.
    Returns data for the 'What makes an effective call?' section.
    """
    if "host_response_text" not in df.columns:
        return {}
    has_resp = df["host_response_text"].notna() & (df["host_response_text"].str.strip() != "")
    resp = df[has_resp].copy()
    if resp.empty:
        return {}

    resp["host_resp_words"] = resp["host_response_text"].apply(lambda t: len(str(t).split()))
    resp["host_followup_q"] = resp["host_response_text"].apply(lambda t: "?" in str(t))

    def _outcome(row):
        rw, fu = row["host_resp_words"], row["host_followup_q"]
        if rw > 60 or (rw >= 20 and fu): return "Substantive"
        if rw >= 20:                      return "Acknowledged"
        return "Brief"
    resp["outcome"] = resp.apply(_outcome, axis=1)

    sub = resp[resp["outcome"] == "Substantive"]
    ack = resp[resp["outcome"] == "Acknowledged"]
    bri = resp[resp["outcome"] == "Brief"]

    # ── Feature comparison ────────────────────────────────────────────────────
    METRIC_LABELS = {
        "hedge_rate":            "Hedging language rate",
        "avg_words_per_sentence":"Avg words per sentence",
        "word_count":            "Caller word count",
        "modal_rate":            "Modal verb rate",
        "question_count":        "# questions asked",
        "question_ratio":        "Question ratio",
        "first_p_rate":          "1st-person pronoun rate",
        "second_p_rate":         "2nd-person pronoun rate",
        "unique_word_ratio":     "Vocab diversity (TTR)",
        "sent_compound":         "Caller sentiment (compound)",
        "sent_neg":              "% negative sentences",
    }
    feature_diff = []
    for col, label in METRIC_LABELS.items():
        if col not in resp.columns:
            continue
        s_mean = float(sub[col].dropna().mean())
        b_mean = float(bri[col].dropna().mean())
        a_mean = float(ack[col].dropna().mean())
        pct = round(100 * (s_mean - b_mean) / abs(b_mean), 1) if b_mean != 0 else 0.0
        feature_diff.append({
            "metric":      label,
            "subst_mean":  round(s_mean, 5),
            "ack_mean":    round(a_mean, 5),
            "brief_mean":  round(b_mean, 5),
            "pct_diff":    pct,
        })
    feature_diff.sort(key=lambda x: x["pct_diff"], reverse=True)

    # ── Outcome rates by host ─────────────────────────────────────────────────
    by_host = []
    for host, grp in resp.groupby("host_name"):
        n = len(grp)
        by_host.append({
            "host":       str(host),
            "n":          n,
            "subst_pct":  round(100 * (grp["outcome"] == "Substantive").mean(), 1),
            "ack_pct":    round(100 * (grp["outcome"] == "Acknowledged").mean(), 1),
            "brief_pct":  round(100 * (grp["outcome"] == "Brief").mean(), 1),
        })
    by_host.sort(key=lambda x: x["subst_pct"], reverse=True)

    # ── Outcome rates by day of week ──────────────────────────────────────────
    by_day = []
    if "day_of_week" in resp.columns:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        for day in dow_order:
            grp = resp[resp["day_of_week"] == day]
            if len(grp) < 10:
                continue
            by_day.append({
                "day":       day,
                "n":         int(len(grp)),
                "subst_pct": round(100 * (grp["outcome"] == "Substantive").mean(), 1),
                "ack_pct":   round(100 * (grp["outcome"] == "Acknowledged").mean(), 1),
                "brief_pct": round(100 * (grp["outcome"] == "Brief").mean(), 1),
            })

    # ── Outcome rates by question count ──────────────────────────────────────
    resp["q_bucket"] = resp["question_count"].apply(
        lambda x: "0" if x == 0 else ("1" if x == 1 else ("2" if x == 2 else "3+")))
    by_qcount = []
    for qb in ["0", "1", "2", "3+"]:
        grp = resp[resp["q_bucket"] == qb]
        if grp.empty:
            continue
        by_qcount.append({
            "q_count":   qb,
            "n":         int(len(grp)),
            "subst_pct": round(100 * (grp["outcome"] == "Substantive").mean(), 1),
            "ack_pct":   round(100 * (grp["outcome"] == "Acknowledged").mean(), 1),
            "brief_pct": round(100 * (grp["outcome"] == "Brief").mean(), 1),
        })

    # ── Opener words by outcome ───────────────────────────────────────────────
    from collections import Counter
    sub_openers = Counter(w for w in sub["opener_word"] if w)
    bri_openers = Counter(w for w in bri["opener_word"] if w)
    all_openers = Counter(w for w in resp["opener_word"] if w)
    top_words   = [w for w, _ in all_openers.most_common(15)]
    n_sub = max(len(sub), 1); n_bri = max(len(bri), 1)
    by_opener = []
    for w in top_words:
        by_opener.append({
            "word":        w,
            "subst_rate":  round(100 * sub_openers.get(w, 0) / n_sub, 2),
            "brief_rate":  round(100 * bri_openers.get(w, 0) / n_bri, 2),
            "total":       all_openers[w],
        })

    return {
        "n_subst":     int(len(sub)),
        "n_ack":       int(len(ack)),
        "n_brief":     int(len(bri)),
        "n_total":     int(len(resp)),
        "featureDiff": feature_diff,
        "byHost":      by_host,
        "byDay":       by_day,
        "byQcount":    by_qcount,
        "byOpener":    by_opener,
    }


def compute_payload(df: pd.DataFrame) -> dict:
    """Compute all chart/table data as a JSON-serializable dict."""
    labeled = df[df["gender"].isin(["male", "female"])]
    f = labeled[labeled["gender"] == "female"]
    m = labeled[labeled["gender"] == "male"]

    def vals(sub, col):
        return sub[col].dropna().round(4).tolist()

    def mean_sem(sub, col):
        s = sub[col].dropna()
        return {
            "mean": round(float(s.mean()), 4) if len(s) else 0,
            "sem":  round(float(s.sem()),  4) if len(s) > 1 else 0,
            "n":    len(s),
        }

    def top_openers(sub, n=14):
        counts = Counter(w for w in sub["opener_word"] if w)
        top = counts.most_common(n)
        return {"words": [x[0] for x in top], "counts": [x[1] for x in top]}

    # Mann-Whitney p-values for violin titles
    violin_cols = ["word_count", "avg_words_per_sentence", "question_ratio",
                   "unique_word_ratio", "hedge_rate", "modal_rate",
                   "first_p_rate", "second_p_rate"]
    pvals = {}
    for col in violin_cols:
        pvals[col] = _mwu_p(m[col].dropna().tolist(), f[col].dropna().tolist())

    # Party × gender mean word counts
    parties = ["republican", "democrat", "independent"]
    party_data = {}
    for party in parties:
        fs = f[f["party"] == party]["word_count"]
        ms = m[m["party"] == party]["word_count"]
        party_data[party] = {
            "female": round(float(fs.mean()), 1) if len(fs) else None,
            "male":   round(float(ms.mean()), 1) if len(ms) else None,
            "n_f":    len(fs),
            "n_m":    len(ms),
        }

    summary = {
        "total":        len(df),
        "n_female":     len(f),
        "n_male":       len(m),
        "n_unknown":    int((df["gender"] == "unknown").sum()),
        "n_episodes":   int(df["episode_id"].nunique()) if "episode_id" in df.columns else None,
        "avg_words_f":  round(float(f["word_count"].mean()), 1) if len(f) else 0,
        "avg_words_m":  round(float(m["word_count"].mean()), 1) if len(m) else 0,
        "q_ratio_f":    round(float(f["question_ratio"].mean()), 3) if len(f) else 0,
        "q_ratio_m":    round(float(m["question_ratio"].mean()), 3) if len(m) else 0,
        "hedge_f":      round(float(f["hedge_rate"].mean()), 3) if len(f) else 0,
        "hedge_m":      round(float(m["hedge_rate"].mean()), 3) if len(m) else 0,
    }

    # Tabulator table — all rows; text truncated to 400 chars to keep data.json small
    table_cols = ["gender", "party", "name", "host_name", "host_gender",
                  "word_count", "sentence_count",
                  "question_count", "question_ratio", "avg_words_per_sentence",
                  "unique_word_ratio", "hedge_rate", "modal_rate",
                  "first_p_rate", "second_p_rate", "upload_date", "episode_id", "text"]
    table_cols = [c for c in table_cols if c in df.columns]
    tbl = df[table_cols].copy()
    for col in ["question_ratio", "avg_words_per_sentence", "unique_word_ratio",
                "hedge_rate", "modal_rate", "first_p_rate", "second_p_rate"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].round(3)
    if "text" in tbl.columns:
        tbl["text"] = tbl["text"].fillna("").str[:200]
    table_data = tbl.fillna("").to_dict(orient="records")

    # Compact per-row data for scatter charts (stays in data.json — text truncated to 80 chars)
    scatter_cols = ["gender", "word_count", "unique_word_ratio", "question_count",
                    "name", "party", "upload_date"]
    scatter_cols = [c for c in scatter_cols if c in df.columns]
    scat_tbl = df[scatter_cols].copy()
    if "unique_word_ratio" in scat_tbl.columns:
        scat_tbl["unique_word_ratio"] = scat_tbl["unique_word_ratio"].round(3)
    if "text" in df.columns:
        scat_tbl["text"] = df["text"].fillna("").str[:80]
    points_data = scat_tbl.fillna("").to_dict(orient="records")

    # Sentiment table — numeric sentiment cols + short text preview (no full text duplicate)
    sent_cols = ["gender", "party", "name", "upload_date", "episode_id",
                 "sent_compound", "sent_pos", "sent_neg", "sent_neu", "word_count", "text"]
    sent_cols = [c for c in sent_cols if c in df.columns]
    sent_tbl = df[sent_cols].copy()
    for col in ["sent_compound", "sent_pos", "sent_neg", "sent_neu"]:
        if col in sent_tbl.columns:
            sent_tbl[col] = sent_tbl[col].round(3)
    if "text" in sent_tbl.columns:
        sent_tbl["text"] = sent_tbl["text"].fillna("").str[:120]
    sent_table_data = sent_tbl.fillna("").to_dict(orient="records")

    # Responsiveness — keep chart data (small) in main payload; table rows go to tables.json
    resp_full = _compute_responsiveness(df)
    resp_chart = {k: v for k, v in resp_full.items() if k != "tableData"}

    main_payload = {
        "summary": summary,
        "pvals": pvals,
        "female": {col: vals(f, col) for col in violin_cols},
        "male":   {col: vals(m, col) for col in violin_cols},
        "barMetrics": {
            "cols": ["Question ratio", "Vocab diversity", "Sentence count",
                     "1st-person rate", "2nd-person rate"],
            "female": [mean_sem(f, c) for c in ["question_ratio", "unique_word_ratio",
                                                  "sentence_count", "first_p_rate", "second_p_rate"]],
            "male":   [mean_sem(m, c) for c in ["question_ratio", "unique_word_ratio",
                                                  "sentence_count", "first_p_rate", "second_p_rate"]],
        },
        "styleMetrics": {
            "cols": ["Hedging rate", "Modal verb rate", "1st-person rate", "2nd-person rate"],
            "female": [mean_sem(f, c) for c in ["hedge_rate", "modal_rate", "first_p_rate", "second_p_rate"]],
            "male":   [mean_sem(m, c) for c in ["hedge_rate", "modal_rate", "first_p_rate", "second_p_rate"]],
        },
        "openers": {
            "female": top_openers(f),
            "male":   top_openers(m),
        },
        "partyData":     party_data,
        "pointsData":    points_data,
        "interactions":  _compute_interaction_data(df),
        "timeSeries":    _compute_time_series(df),
        "wordFreq":      _compute_word_freq(df),
        "sentiment":        _compute_sentiment_data(df),
        "geo":              _compute_geo_data(df),
        "responsiveness":   resp_chart,
        "dayOfWeek":        _compute_day_of_week(df),
        "sentOverTime":     _compute_sentiment_over_time(df),
        "sankey":           _compute_sankey(df),
        "effectiveCalls":   _compute_effective_calls(df),
    }

    # Heavy table rows live in tables.json — fetched lazily so charts appear first
    tables_payload = {
        "tableData":    table_data,
        "sentTableData": sent_table_data,
        "respTableData": resp_full.get("tableData", []),
    }

    return main_payload, tables_payload


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

CSS = """\
*, *::before, *::after { box-sizing: border-box; }

body {
  font-family: Inter, Segoe UI, Arial, sans-serif;
  background: #F4F6F8;
  color: #1a1a2e;
  margin: 0;
  padding: 0;
}

.page {
  max-width: 1400px;
  margin: 0 auto;
  padding: 28px 32px 60px;
}

header {
  margin-bottom: 32px;
}
header h1 {
  font-size: 1.9rem;
  font-weight: 700;
  margin: 0 0 6px;
  color: #0d1b2a;
}
header p.subtitle {
  color: #555;
  margin: 0;
  font-size: 0.97rem;
  max-width: 820px;
  line-height: 1.55;
}

.female-label { color: #D81B60; font-weight: 600; }
.male-label   { color: #1565C0; font-weight: 600; }

/* ── summary cards ─────────────────────────────────────────────────────── */
.cards {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  margin-bottom: 36px;
}
.card {
  background: #fff;
  border-radius: 10px;
  padding: 16px 22px;
  flex: 1 1 130px;
  box-shadow: 0 1px 5px rgba(0,0,0,.09);
  text-align: center;
}
.card .label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .05em;
  color: #777;
  margin: 0 0 4px;
}
.card .value {
  font-size: 1.45rem;
  font-weight: 700;
  color: #0d1b2a;
  margin: 0;
}
.card .subvalue {
  font-size: 0.8rem;
  color: #999;
  margin: 2px 0 0;
}

/* ── sections ──────────────────────────────────────────────────────────── */
section {
  margin-bottom: 48px;
}
section h2 {
  font-size: 1.15rem;
  font-weight: 700;
  color: #0d1b2a;
  margin: 0 0 4px;
  padding-bottom: 8px;
  border-bottom: 3px solid #e0e4ea;
}
section .note {
  font-size: 0.83rem;
  color: #666;
  margin: 4px 0 14px;
}

/* ── chart grid ────────────────────────────────────────────────────────── */
.chart-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 18px;
}
.chart-card {
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 1px 5px rgba(0,0,0,.08);
  padding: 16px 14px 10px;
}
.fig-label {
  font-size: 0.72rem;
  font-weight: 600;
  color: #888;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.sk-head {
  font-weight: 700;
  font-size: 0.95rem;
  margin: 24px 0 4px;
  color: #0d1b2a;
}
.sk-head .sk-sub {
  font-weight: 400;
  color: #666;
  font-size: 0.85rem;
}
.sk-card {
  margin-bottom: 8px;
}

/* ── term glossary below each section ──────────────────────────────────── */
.term-list {
  background: #f7f9fc;
  border-left: 3px solid #c8d0db;
  border-radius: 0 8px 8px 0;
  padding: 14px 20px;
  margin-top: 14px;
  font-size: 0.84rem;
  line-height: 1.7;
  color: #444;
}
.term-list p {
  margin: 0 0 5px;
}
.term-list p:last-child { margin-bottom: 0; }
.term-list strong { color: #1a1a2e; }

/* ── Tabulator overrides ───────────────────────────────────────────────── */
.tabulator {
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 1px 5px rgba(0,0,0,.08);
  font-size: 13px;
}
.tabulator .tabulator-header {
  background: #e8ecf1;
  font-weight: 600;
  font-size: 12px;
}
.tabulator .tabulator-row.tabulator-row-even {
  background: #f9fafb;
}
.tabulator .tabulator-row:hover {
  background: #EEF2FF !important;
}

.table-actions {
  display: flex;
  gap: 10px;
  margin-bottom: 12px;
  align-items: center;
  flex-wrap: wrap;
}
.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #1565C0;
  color: #fff;
  border: none;
  border-radius: 6px;
  padding: 7px 14px;
  font-size: 13px;
  cursor: pointer;
  font-family: inherit;
  font-weight: 500;
  transition: background .15s;
}
.btn:hover { background: #0d47a1; }
.btn.secondary {
  background: #fff;
  color: #1565C0;
  border: 1.5px solid #1565C0;
}
.btn.secondary:hover { background: #EEF2FF; }
.table-count {
  font-size: 12px;
  color: #666;
  margin-left: auto;
}

/* ── about block ───────────────────────────────────────────────────────── */
.about-text {
  margin-top: 22px;
  margin-bottom: 8px;
  max-width: 900px;
  font-size: 0.88rem;
  line-height: 1.72;
  color: #333;
}
.about-text h3 {
  font-size: 0.82rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .07em;
  color: #555;
  margin: 22px 0 5px;
  padding-bottom: 3px;
  border-bottom: 1px solid #e0e4ea;
}
.about-text h3:first-child { margin-top: 0; }
.about-text p { margin: 0 0 6px; }
.about-text ul { margin: 4px 0 8px 20px; padding: 0; }
.about-text li { margin-bottom: 4px; }
.about-text a { color: #1565C0; text-decoration: none; }
.about-text a:hover { text-decoration: underline; }
.about-text code {
  background: #f0f2f5;
  border-radius: 3px;
  padding: 1px 5px;
  font-size: 0.84em;
}

footer {
  color: #aaa;
  font-size: 11px;
  margin-top: 48px;
  padding-top: 14px;
  border-top: 1px solid #dde1e8;
}

/* ── Leaflet map ────────────────────────────────────────────────────────── */
#caller-map {
  height: 520px;
  border-radius: 10px;
  box-shadow: 0 1px 5px rgba(0,0,0,.08);
}

/* ── section groups ─────────────────────────────────────────────────────── */
.section-group {
  margin-bottom: 64px;
}
.group-header {
  margin-bottom: 28px;
  padding-bottom: 14px;
  border-bottom: 3px solid #0d1b2a;
}
.group-header h2 {
  font-size: 1.5rem;
  font-weight: 700;
  color: #0d1b2a;
  margin: 0 0 5px;
}
.group-header p {
  font-size: 0.9rem;
  color: #555;
  margin: 0;
  max-width: 860px;
  line-height: 1.6;
}
.group-section {
  margin-bottom: 42px;
}
.group-section h3 {
  font-size: 1.05rem;
  font-weight: 700;
  color: #1a1a2e;
  margin: 0 0 4px;
  padding-bottom: 6px;
  border-bottom: 2px solid #e8ecf1;
}
.group-section .note {
  font-size: 0.83rem;
  color: #666;
  margin: 3px 0 12px;
}

/* ── App shell (sidebar + main) ─────────────────────────────── */
.app-shell {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  width: 200px;
  min-width: 200px;
  background: #0d1b2a;
  color: #c8d6e5;
  display: flex;
  flex-direction: column;
  position: fixed;
  top: 0;
  left: 0;
  height: 100vh;
  overflow-y: auto;
  z-index: 200;
  padding-bottom: 24px;
}

.sidebar-brand {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #fff;
  padding: 22px 20px 18px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  line-height: 1.4;
}

.nav-item {
  display: block;
  padding: 11px 20px;
  color: #c8d6e5;
  text-decoration: none;
  font-size: 0.88rem;
  font-weight: 500;
  border-left: 3px solid transparent;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}

.nav-item:hover {
  background: rgba(255,255,255,0.07);
  color: #fff;
}

.nav-item.active {
  background: rgba(255,255,255,0.10);
  color: #fff;
  border-left-color: #4fc3f7;
  font-weight: 600;
}

.main-area {
  margin-left: 200px;
  flex: 1;
  min-width: 0;
  padding: 0 24px 40px;
  max-width: 1400px;
}

/* Each nav-page hidden by default except .active */
.nav-page { display: none; }
.nav-page.active { display: block; }

/* Remove the old .page top margin/padding that assumed full-width layout */
.page { margin: 0; padding: 0; max-width: none; }
"""


# ══════════════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>C-SPAN Washington Journal — Caller Gender Analysis</title>
  <link rel="stylesheet" href="style.css">

  <!-- Plotly.js -->
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>

  <!-- Tabulator -->
  <link href="https://unpkg.com/tabulator-tables@6.2.1/dist/css/tabulator.min.css" rel="stylesheet">
  <script src="https://unpkg.com/tabulator-tables@6.2.1/dist/js/tabulator.min.js"></script>

  <!-- Leaflet -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>
<div class="app-shell">

  <!-- Fixed left navigation sidebar -->
  <nav class="sidebar">
    <div class="sidebar-brand">C-SPAN<br>Analysis</div>
    <a class="nav-item active" data-page="overview" href="#">Overview</a>
    <a class="nav-item" data-page="who" href="#">Who&rsquo;s Calling?</a>
    <a class="nav-item" data-page="questions" href="#">Questions &amp; Sentiment</a>
    <a class="nav-item" data-page="style" href="#">Speaking Style</a>
    <a class="nav-item" data-page="tables" href="#">Data Tables</a>
  </nav>

  <!-- Main scrollable area -->
  <main class="main-area">

    <!-- Page: Overview -->
    <div class="nav-page active" id="page-overview">

<header>
  <h1>C-SPAN Washington Journal &mdash; Caller Gender Analysis</h1>
  <p class="subtitle">
    Do <span class="female-label">female</span> and
    <span class="male-label">male</span> callers speak differently?
    This project scrapes machine-generated transcripts of every Washington Journal episode,
    extracts individual caller turns, and measures differences in word length, sentence structure,
    hedging language, question-asking behavior, vocabulary, and sentence-level sentiment
    across gender and party affiliation.
  </p>

  <div class="about-text">

    <h3>The Show</h3>
    <p>
      <a href="https://en.wikipedia.org/wiki/Washington_Journal" target="_blank">Washington Journal</a>
      is a live, three-hour call-in program broadcast on C-SPAN that has aired daily since
      January&nbsp;4,&nbsp;1995. Each morning, callers dial in on three separate phone lines
      designated by self-reported party affiliation &mdash; Republican, Democrat, and Independent
      &mdash; and speak directly and unscripted with the on-air host about the day&rsquo;s news
      and politics. The host introduces the caller (often including their name and location), listens
      to their comment or question without interruption, and responds briefly before taking the next
      call. The show averages roughly 15&ndash;20 calls per hour. Because each caller has an
      uninterrupted speaking turn of variable length, and callers self-select into party lines,
      Washington Journal is one of the most accessible long-running corpora of spontaneous
      American political speech available for research.
    </p>

    <h3>Data Sources</h3>
    <p>
      Transcripts are collected from two sources. The primary source is
      <a href="https://fight.fudgie.org/search/show/cspan/" target="_blank">fight.fudgie.org</a>,
      a third-party public archive of machine-generated transcripts for C-SPAN broadcasts,
      produced by automated speech-to-text and covering over 10,000 episodes. The fudgie.org
      HTML stores per-line speaker attribution via <code>data-speaker-name</code> attributes
      in the attributed format (episodes prior to 2026) and uses a CSS class (<code>new-speaker-cell</code>)
      to mark speaker changes in the unattributed format (2026+ episodes).
      A supplementary scraper extracts closed captions from C-SPAN&rsquo;s official YouTube
      channel using <a href="https://github.com/yt-dlp/yt-dlp" target="_blank">yt-dlp</a>.
      The current dataset covers
      <strong id="about-n-episodes">&mdash;</strong> episodes and
      <strong id="about-n-turns">&mdash;</strong> caller turns
      (<strong id="about-n-labeled">&mdash;</strong> gender-labeled).
    </p>

    <h3>Corpus Construction</h3>
    <p>
      Caller turns are identified by one of two methods depending on the fudgie.org episode format.
      In the <strong>attributed format</strong> (pre-2026 episodes), each transcript line carries a
      speaker name. Lines attributed to speakers matching the pattern
      <em>[firstname] in [state]</em> or labeled &ldquo;Unidentified&rdquo; are treated as caller
      turns; host turns are identified by a lookup table of known Washington Journal hosts.
      In the <strong>unattributed format</strong> (2026+ episodes), no speaker names are present.
      Caller turns are identified by the host&rsquo;s party-line introduction phrase &mdash;
      e.g., &ldquo;Republican line, go ahead&rdquo; or &ldquo;Democrat line, good morning&rdquo; &mdash;
      and all turns between two such introductions are concatenated to form the caller&rsquo;s text.
      Turns shorter than 15 words are excluded to remove off-air fragments; turns longer than 600
      words are excluded to remove transcript artifacts and advertisements. Each caller turn is treated
      as one independent observation. Duplicate rows arising from the same episode appearing at
      multiple URL fragments are removed by normalizing the episode identifier and deduplicating on
      (episode&nbsp;ID,&nbsp;caller&nbsp;text).
    </p>

    <h3>Metadata per Caller Turn</h3>
    <p>For each extracted turn the following fields are recorded:</p>
    <ul>
      <li><strong>Caller name</strong> &mdash; First name extracted from the host&rsquo;s spoken
        introduction using regular expressions (e.g., &ldquo;Republican line, John from
        Texas&rdquo; &rarr; &ldquo;John&rdquo;). Absent when the host does not name the caller.</li>
      <li><strong>Party line</strong> &mdash; Republican, Democrat, or Independent, parsed directly
        from the host&rsquo;s introduction phrase.</li>
      <li><strong>Host name &amp; gender</strong> &mdash; Identified by matching known Washington
        Journal host names against the full episode transcript; host gender is drawn from a
        manually curated lookup table.</li>
      <li><strong>Caller city &amp; state</strong> &mdash; Parsed from introduction phrases such as
        &ldquo;from Portland, Oregon&rdquo; or &ldquo;Roy in North Dakota&rdquo; using a regex
        against a list of all 50 US state names. Only available when the host mentions a location.</li>
      <li><strong>Call hour (Eastern Time)</strong> &mdash; Derived from the episode broadcast
        start hour (encoded in the episode identifier string) plus the caller&rsquo;s audio offset
        in seconds, extracted from <code>play_ep()</code> JavaScript calls embedded in the
        fudgie.org HTML, divided by 3600.</li>
      <li><strong>Call duration (seconds)</strong> &mdash; The caller&rsquo;s last transcript
        timestamp minus their first, from the same <code>play_ep()</code> timestamps.</li>
      <li><strong>Day of week</strong> &mdash; Derived from the broadcast date in the episode
        identifier (YYYYMMDD prefix).</li>
    </ul>

    <h3>Gender Inference</h3>
    <p>
      Caller gender is inferred automatically in order of precedence. First, if the host names the
      caller, the first name is queried against the
      <a href="https://pypi.org/project/gender-guesser/" target="_blank">gender-guesser</a>
      Python library, which classifies names as <em>male</em>, <em>mostly_male</em>,
      <em>female</em>, <em>mostly_female</em>, or <em>andy</em> (androgynous / ambiguous)
      based on a large international name database. Names classified as male or mostly_male
      are labeled <em>male</em>; female or mostly_female are labeled <em>female</em>;
      andy, absent, or unrecognized names yield <em>unknown</em>.
      Second, when name inference fails, the caller&rsquo;s transcript text is scanned for
      gendered salutations: <em>sir</em> or <em>gentleman</em> indicate male;
      <em>ma&rsquo;am</em> or <em>madam</em> indicate female.
      Turns where neither signal resolves a gender are labeled <em>unknown</em> and excluded
      from gender-comparative analyses. No manual labeling was performed. The automated approach
      systematically underpredicts labeling rates: gender-neutral names, non-English names,
      and calls where the host does not introduce the caller by name all produce unknown labels.
    </p>

    <h3>Linguistic Features</h3>
    <p>
      All features are computed from the raw transcript text of the caller&rsquo;s turn using
      regular expressions and word-list lookups; no external NLP models are used for feature
      extraction (sentiment is a separate step, below).
    </p>
    <ul>
      <li><strong>Word count</strong> &mdash; Total whitespace-delimited tokens in the
        caller&rsquo;s turn.</li>
      <li><strong>Sentence count</strong> &mdash; Number of segments produced by splitting on
        sentence-ending punctuation (<code>.&nbsp;!&nbsp;?</code>). Empty segments are
        discarded.</li>
      <li><strong>Question count / Question ratio</strong> &mdash; A sentence is classified as a
        question if it ends with &ldquo;?&rdquo; or begins with one of the following interrogative
        or auxiliary-inversion words: <em>do, does, did, is, are, was, were, will, would, can,
        could, have, has, why, what, how, when, where, who, which, whose, wouldn&rsquo;t,
        don&rsquo;t, isn&rsquo;t, aren&rsquo;t, can&rsquo;t, didn&rsquo;t</em>. Question ratio
        is question count &divide; sentence count.</li>
      <li><strong>Avg words per sentence</strong> &mdash; Word count &divide; sentence count.
        A proxy for syntactic complexity.</li>
      <li><strong>Vocabulary diversity (type&ndash;token ratio)</strong> &mdash; Number of distinct
        lowercased word types &divide; total word tokens. Ranges from near 0 (maximal repetition)
        to 1.0 (no word used more than once). Equivalent to the type&ndash;token ratio (TTR).</li>
      <li><strong>Hedging rate</strong> &mdash; Count of hedging phrases &divide; word count.
        Hedging signals epistemic uncertainty or tentativeness. Phrases counted:
        <em>I think, I feel, I believe, I guess, maybe, perhaps, possibly, it seems, sort of,
        kind of, I was wondering, I&rsquo;m not sure, I don&rsquo;t know, might be, could be</em>.
        Matches are case-insensitive; multi-word phrases are matched as contiguous substrings.</li>
      <li><strong>Modal verb rate</strong> &mdash; Count of modal verbs
        (<em>would, could, should, might, may, ought</em>) &divide; word count. Modal verbs
        express conditionality, possibility, or obligation and are commonly used to soften
        assertions.</li>
      <li><strong>First-person pronoun rate</strong> &mdash; Count of
        <em>I, me, my, mine, myself</em> &divide; word count. A measure of self-referential
        speech; higher values indicate callers foregrounding their own experience or
        perspective.</li>
      <li><strong>Second-person pronoun rate</strong> &mdash; Count of
        <em>you, your, yours, yourself</em> &divide; word count. A measure of direct address
        to the host or a general audience.</li>
      <li><strong>First meaningful word</strong> &mdash; The first word of the caller&rsquo;s
        turn after stripping filler tokens
        (<em>um, uh, well, yes, yeah, no, so, and, but, i, you, know, like, okay, ok, hi,
        hello, good</em>). Reveals whether callers open with a declarative concept, a
        question word, or a named entity.</li>
    </ul>

    <h3>Sentiment Analysis</h3>
    <p>
      Sentence-level sentiment is scored with
      <a href="https://github.com/cjhutto/vaderSentiment" target="_blank">VADER</a>
      (Valence Aware Dictionary and sEntiment Reasoner; Hutto &amp; Gilbert, 2014), a
      rule-based lexicon model designed for informal and spoken language. VADER assigns each
      sentence a <strong>compound score</strong> in [&minus;1,&nbsp;+1], constructed by summing
      valence scores from a sentiment lexicon and applying rules for negation
      (&ldquo;not good&rdquo;), intensifiers (&ldquo;very&rdquo;), capitalization emphasis,
      and punctuation. Each caller turn is split into sentences; per-sentence compound scores are
      averaged to produce a per-turn mean compound score. Three additional fractions are recorded:
      the share of sentences with compound&nbsp;&gt;&nbsp;0.05 (<strong>% positive</strong>),
      compound&nbsp;&lt;&nbsp;&minus;0.05 (<strong>% negative</strong>), and within
      [&minus;0.05,&nbsp;0.05] (<strong>% neutral</strong>).
    </p>

    <h3>Host Responsiveness</h3>
    <p>
      For each caller turn, the host&rsquo;s first spoken turn immediately following the caller
      is captured as <code>host_response_text</code>. Four metrics are derived:
    </p>
    <ul>
      <li><strong>Host response length</strong> &mdash; Word count of the host&rsquo;s
        response.</li>
      <li><strong>Topic overlap (Jaccard similarity)</strong> &mdash; The Jaccard index between
        the set of non-stop-words in the caller&rsquo;s text and those in the host&rsquo;s
        response: |caller&nbsp;&cap;&nbsp;host|&nbsp;&divide;&nbsp;|caller&nbsp;&cup;&nbsp;host|.
        Ranges from 0 (no shared vocabulary) to 1 (identical vocabulary). Higher values
        indicate the host directly addressed the caller&rsquo;s specific topics.</li>
      <li><strong>Compliment detection</strong> &mdash; A Boolean flag: True when the
        host&rsquo;s response contains a phrase from a predefined list including
        <em>good question, great point, excellent point, very interesting, well said</em>.</li>
      <li><strong>Follow-up question</strong> &mdash; A Boolean flag: True when the host&rsquo;s
        response contains a question mark, indicating active engagement rather than a simple
        acknowledgment.</li>
    </ul>
    <p>
      Host response data is only available for episodes scraped after this feature was added
      to the scraper; earlier rows in the dataset have an empty <code>host_response_text</code>
      field. In the unattributed format (2026+), the host response is approximated as turns
      following the caller&rsquo;s last long speech segment, which is a heuristic.
    </p>

    <h3>Statistical Methods</h3>
    <p>
      Group differences between female and male callers on all continuous linguistic features are
      tested with the <strong>Mann&ndash;Whitney U test</strong> (implemented in
      <a href="https://scipy.org/" target="_blank">SciPy</a>), a non-parametric rank-sum test
      that makes no assumption of normality and is appropriate for the skewed, bounded
      distributions typical of linguistic rate measures. Tests are two-tailed. Significance
      thresholds: p&nbsp;&lt;&nbsp;0.001&nbsp;(***), p&nbsp;&lt;&nbsp;0.01&nbsp;(**),
      p&nbsp;&lt;&nbsp;0.05&nbsp;(*); p&nbsp;&ge;&nbsp;0.05 is labeled not significant (ns).
      Error bars on bar charts show &plusmn;1 standard error of the mean (SE).
    </p>

    <h3>Visualization &amp; Tools</h3>
    <p>
      Interactive charts are rendered with
      <a href="https://plotly.com/javascript/" target="_blank">Plotly.js</a>
      (all violin, bar, scatter, histogram, and time-series plots; supports pan, zoom, and
      PNG/SVG export). Sortable, filterable data tables are built with
      <a href="https://tabulator.info/" target="_blank">Tabulator</a>.
      The geographic bubble map uses
      <a href="https://leafletjs.com/" target="_blank">Leaflet.js</a>
      with CartoDB light tile layers. All analysis code is in Python using
      <a href="https://pandas.pydata.org/" target="_blank">pandas</a> and
      <a href="https://scipy.org/" target="_blank">SciPy</a>.
    </p>

    <h3>Limitations</h3>
    <p>
      Machine-generated transcripts contain systematic errors, particularly for proper nouns,
      names, overlapping speech, and accented speakers. Gender inference is approximate and
      produces high unknown rates (&gt;90% in 2026+ unattributed episodes) because many hosts
      do not introduce callers by name. Call duration, call hour, and host response text are
      only available for episodes scraped with the current scraper version; earlier rows have
      these fields empty. Geographic state data is only available when the host verbally mentions
      the caller&rsquo;s location. The dataset covers a sample of episodes and grows
      incrementally &mdash; run <code>make fudgie-big</code> to append additional episodes.
    </p>

  </div>
</header>

      <div class="cards" id="summary-cards"></div>
    </div>

    <!-- Page: Who's Calling? -->
    <div class="nav-page" id="page-who">

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- WHO'S CALLING?                                                         -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div class="section-group" id="group-who">
  <div class="group-header">
    <h2>Who&rsquo;s Calling?</h2>
    <p>Scope of the dataset: total callers captured, their self-reported party affiliation, inferred gender, call volume over time, how long callers speak, and where they&rsquo;re calling from.</p>
  </div>

  <!-- Call volume over time -->
  <div class="group-section" id="sec-callsovertime">
    <h3>Call Volume &amp; Words Over Time</h3>
    <p class="note">Monthly count of caller turns (solid lines, left axis) and total words spoken (dashed lines, right axis), split by gender. Click legend entries to toggle series. Use the Plotly camera icon to download as PNG.</p>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-ts-monthly"></div></div>
      <div class="chart-card"><div id="c-ts-cumulative"></div></div>
    </div>
    <div class="term-list">
      <p><strong>Monthly call volume</strong> &mdash; The count of individual caller turns extracted from episodes broadcast in that calendar month. Each turn is one caller&rsquo;s uninterrupted speaking segment, bounded by the host&rsquo;s party-line introduction and the next introduction.</p>
      <p><strong>Total words per month</strong> &mdash; The sum of word counts across all caller turns in that month. Word count for each turn is the number of whitespace-delimited tokens in the transcript text.</p>
      <p><strong>Gender series: All / Female / Male / Unknown</strong> &mdash; <em>All callers</em> counts every extracted turn. <em>Female</em> and <em>Male</em> include only turns where gender was inferred via caller name (gender-guesser library) or host salutation (<em>sir / ma&rsquo;am</em>). <em>Unknown</em> includes turns where neither signal resolved a gender label.</p>
      <p><strong>Cumulative plot</strong> &mdash; Running total from the earliest episode in the dataset to the most recent. The slope of the curve reflects the scraping rate. Flat periods indicate coverage gaps (episodes not yet scraped). Run <code>make fudgie-big</code> to append additional episodes.</p>
      <p><strong>Dual y-axes</strong> &mdash; The left axis (solid lines) counts callers; the right axis (dashed lines) counts total words spoken. The two axes are independent &mdash; they share the same x-axis (month) but have different scales.</p>
    </div>
  </div>

  <!-- How long do callers speak? -->
  <div class="group-section">
    <h3>How Long Do Callers Speak?</h3>
    <p class="note">Distribution of word counts per caller turn, by gender. Word count is the primary proxy for call duration (call timestamps are only available after the scraper was updated).</p>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-wordcount"></div></div>
      <div class="chart-card"><div id="c-words-per-sent"></div></div>
      <div class="chart-card"><div id="c-wc-hist"></div></div>
      <div class="chart-card"><div id="c-wc-dist"></div></div>
    </div>
    <div class="term-list">
      <p><strong>Word count</strong> &mdash; Total whitespace-delimited tokens in a caller&rsquo;s turn. Turns shorter than 15 words or longer than 600 words are excluded during scraping to remove fragments and artifacts. Word count is the best available proxy for speaking time in the current dataset.</p>
      <p><strong>Avg words per sentence</strong> &mdash; Word count &divide; sentence count. Sentence count is the number of segments after splitting on terminal punctuation (<code>.&nbsp;!&nbsp;?</code>). Higher values indicate longer, more syntactically complex sentences.</p>
      <p><strong>Violin plot</strong> &mdash; Kernel density estimate of the distribution. The width at each height is proportional to the density of observations at that value &mdash; wider means more callers. The box inside marks the median (center line) and interquartile range (IQR, 25th&ndash;75th percentile). The thin horizontal line shows the arithmetic mean. Adapted from Hintze &amp; Nelson (1998). <em>The American Statistician</em>, 52(2), 181&ndash;187. <a href="https://doi.org/10.2307/2685873" target="_blank">doi:10.2307/2685873</a></p>
      <p><strong>Per-integer bar chart</strong> &mdash; Each x-position is an exact integer word count; bar height is the number of female or male callers who spoke exactly that many words. Zoom in by clicking and dragging within the chart.</p>
      <p><strong>Word count range bar chart</strong> &mdash; Callers are grouped into bins: 15&ndash;30, 31&ndash;50, 51&ndash;75, 76&ndash;100, 101&ndash;150, 151&ndash;200, 201&ndash;300, 301+. Bars show female and male counts side by side.</p>
      <p><strong>Mann&ndash;Whitney U test (p-value)</strong> &mdash; A non-parametric two-sample rank test of the null hypothesis that the female and male distributions are identical (Mann &amp; Whitney, 1947). <em>Annals of Mathematical Statistics</em>, 18(1), 50&ndash;60. <a href="https://doi.org/10.1214/aoms/1177730491" target="_blank">doi:10.1214/aoms/1177730491</a>. Thresholds: *** p&nbsp;&lt;&nbsp;0.001 &nbsp; ** p&nbsp;&lt;&nbsp;0.01 &nbsp; * p&nbsp;&lt;&nbsp;0.05 &nbsp; (ns) not significant.</p>
    </div>
  </div>

  <!-- Day of week -->
  <div class="group-section" id="sec-dow">
    <h3>Calls by Day of the Week</h3>
    <p class="note">Do women and men call on different days? Left chart: raw caller counts per day of the week, by gender series. Right chart: female share of gender-labeled callers (female + male) for each day.</p>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-dow-counts"></div></div>
      <div class="chart-card"><div id="c-dow-fraction"></div></div>
    </div>
    <div class="term-list">
      <p><strong>Day of week</strong> &mdash; Derived from the episode&rsquo;s broadcast date (<code>upload_date</code> field) using Python&rsquo;s <code>datetime.strftime('%A')</code>. Washington Journal airs Monday through Saturday; Sunday entries reflect occasional special programming or classification artifacts.</p>
      <p><strong>Caller counts per day</strong> &mdash; Total extracted caller turns for each day across all episodes in the dataset. Series: <em>All callers</em> (every turn), <em>Female</em>, <em>Male</em>, <em>Unknown gender</em>.</p>
      <p><strong>Female fraction</strong> &mdash; The fraction of gender-labeled callers (female&nbsp;+&nbsp;male only; unknown excluded) who are female for each day: F<sub>day</sub> / (F<sub>day</sub>&nbsp;+&nbsp;M<sub>day</sub>). A value of 0.5 indicates equal female and male representation among labeled callers on that day. The dashed line marks 0.5 (parity). Sample sizes (<em>n</em>) are shown in the hover tooltip.</p>
      <p><strong>Time-of-day note</strong> &mdash; Within-episode call timing (e.g., 7:15&nbsp;am vs 9:45&nbsp;am) is not yet available in this dataset because the <code>call_hour</code> column requires per-caller audio timestamps that are populated by re-running the scraper (<code>make fudgie-big</code>) on episodes with the updated scraper version.</p>
    </div>
  </div>

  <!-- Geographic distribution -->
  <div class="group-section" id="sec-geo">
    <h3>Where Are Callers Calling From?</h3>
    <p class="note">Each circle is one US state. Circle size &prop; total callers. Hover for gender and party breakdown. Only calls where the host&rsquo;s introduction mentioned a state are shown (<strong id="geo-n">&mdash;</strong> of <strong id="geo-total">&mdash;</strong> turns).</p>
    <div id="caller-map"></div>
    <div class="table-actions" style="margin-top:14px;">
      <button class="btn" onclick="geoTable.download('csv','callers_by_state.csv')">&#8659; Download CSV</button>
      <button class="btn secondary" onclick="geoTable.clearHeaderFilter()">Clear filters</button>
      <span class="table-count" id="geo-table-count"></span>
    </div>
    <div id="geo-table"></div>
    <div class="term-list">
      <p><strong>State extraction</strong> &mdash; The caller&rsquo;s state is parsed from the host&rsquo;s spoken introduction using a regular expression matched against all 50 US state names plus the District of Columbia (e.g., &ldquo;Roy in North Dakota&rdquo; or &ldquo;from Portland, Oregon&rdquo;). Only the attributed fudgie.org format reliably provides this; most unattributed (2026+) episodes yield no state data.</p>
      <p><strong>Bubble map</strong> &mdash; Rendered with <a href="https://leafletjs.com/" target="_blank">Leaflet.js</a> using CartoDB light tiles. Circle radius scales as &radic;(n / n<sub>max</sub>), making area proportional to caller count. State centroids are fixed geographic coordinates.</p>
      <p><strong>Party breakdown in popup</strong> &mdash; Republican, Democrat, and Independent counts are derived from the host&rsquo;s party-line introduction phrase for each call.</p>
    </div>
  </div>
</div><!-- /group-who -->

    </div>

    <!-- Page: Questions & Sentiment -->
    <div class="nav-page" id="page-questions">

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- DO YOU HAVE A QUESTION?                                                -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div class="section-group" id="group-questions">
  <div class="group-header">
    <h2>Do You Have a Question?</h2>
    <p>How often do callers ask questions versus making statements? Does gender, party affiliation, or the host&rsquo;s gender shape that behaviour? How does tone — positive, negative, neutral — vary across groups?</p>
  </div>

  <!-- Questions vs Statements by gender -->
  <div class="group-section">
    <h3>Questions vs Statements &mdash; by Gender</h3>
    <p class="note">Question ratio and key speech metrics, female vs male callers (gender-labeled turns only).</p>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-q-ratio-violin"></div></div>
      <div class="chart-card"><div id="c-key-metrics-bar"></div></div>
    </div>
    <div class="term-list">
      <p><strong>Question ratio</strong> &mdash; The fraction of sentences classified as questions: Q<sub>ratio</sub> = Q<sub>count</sub> / S<sub>count</sub>, where Q<sub>count</sub> is the number of question sentences and S<sub>count</sub> is total sentence count. Ranges from 0 (no questions) to 1 (every sentence is a question).</p>
      <p><strong>How a question is identified</strong> &mdash; A sentence is classified as a question if (a) it ends with &ldquo;?&rdquo;, or (b) its first word is one of: <em>do, does, did, is, are, was, were, will, would, can, could, have, has, why, what, how, when, where, who, which, whose, wouldn&rsquo;t, don&rsquo;t, isn&rsquo;t, aren&rsquo;t, can&rsquo;t, didn&rsquo;t</em>. This heuristic captures both direct questions and auxiliary-inversion questions.</p>
      <p><strong>Vocab diversity (type&ndash;token ratio, TTR)</strong> &mdash; TTR = |V| / N, where |V| is the number of distinct word types (lowercased) and N is the total number of tokens. TTR near 1.0 indicates near-zero repetition; lower values indicate more repeated vocabulary. TTR is sensitive to text length and should be compared within similar word-count ranges (Templin, 1957).</p>
      <p><strong>1st-person rate</strong> &mdash; Count of {<em>I, me, my, mine, myself</em>} &divide; N. Measures self-reference; higher values indicate the caller foregrounds their own experience or identity.</p>
      <p><strong>2nd-person rate</strong> &mdash; Count of {<em>you, your, yours, yourself</em>} &divide; N. Measures direct address to the host or a generalised &ldquo;you&rdquo; audience.</p>
      <p><strong>Error bars</strong> &mdash; Each bar shows the group mean &plusmn;&nbsp;1 standard error of the mean (SE = &sigma; / &radic;n). SE quantifies the precision of the mean estimate, not the spread of individual observations.</p>
    </div>
  </div>

  <!-- By party line -->
  <div class="group-section">
    <h3>Average Words by Gender &amp; Party Line</h3>
    <p class="note">Mean word count for gender-labeled callers, broken out by party line and gender.</p>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-party"></div></div>
    </div>
    <div class="term-list">
      <p><strong>Party line</strong> &mdash; Washington Journal operates three call-in lines by self-reported political affiliation: Republican, Democrat, and Independent. Callers select a line before connecting, and the host announces the line when introducing each caller. Party label is derived directly from the host&rsquo;s spoken introduction (e.g., &ldquo;Republican line, good morning&rdquo;).</p>
      <p><strong>Avg word count</strong> &mdash; Arithmetic mean of word counts across all gender-labeled turns for each gender&nbsp;&times;&nbsp;party cell. Only female and male labeled callers are shown; unknown-gender turns are excluded.</p>
    </div>
  </div>

  <!-- Host gender effect -->
  <div class="group-section" id="sec-interactions">
    <h3>Does the Host&rsquo;s Gender Matter?</h3>
    <p class="note">Mean word count, question ratio, and hedging rate for callers of each gender when speaking to a female, male, or unknown-gender host. A two-way comparison: the x-axis is host gender; the bar color/series is caller gender.</p>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-interaction-words"></div></div>
      <div class="chart-card"><div id="c-interaction-qratio"></div></div>
      <div class="chart-card"><div id="c-interaction-hedge"></div></div>
    </div>
    <div id="host-breakdown" style="margin-top:18px;"></div>
    <div class="term-list">
      <p><strong>Host gender identification</strong> &mdash; The episode host is detected by matching known Washington Journal host surnames against all text in the episode transcript. Each token matched contributes one vote to that host&rsquo;s name; the most-voted name is selected as the episode host. Host gender is drawn from a curated lookup table of known hosts. Episodes where no host name is detected are labeled <em>unknown host</em>.</p>
      <p><strong>Known hosts in this dataset</strong> &mdash; Pedro Echevarria (male), John McArdle (male), Greta Brawner (female), Kimberly Adams (female), Jeslyn Rollins (female), Libby Casey (female), Susan Swain (female), Bill Scanlan (male), Steve Scully (male), Rob Harleston (male), Khalil Garriott (male), Chloe Veltman (female).</p>
      <p><strong>Word count, Question ratio, Hedging rate</strong> &mdash; See definitions above. Each bar is the mean for callers of that gender when speaking to a host of that gender. Differences across host genders suggest accommodation effects &mdash; callers adjusting their speech style to their interlocutor.</p>
    </div>
  </div>

  <!-- Sentiment / tone -->
  <div class="group-section" id="sec-sentiment">
    <h3>Tone of the Comment &mdash; Sentiment Analysis</h3>
    <p class="note">Sentence-level sentiment scored with VADER (compound: &minus;1&nbsp;=&nbsp;most negative, +1&nbsp;=&nbsp;most positive). Bar charts show six groups: Dem&middot;Female, Dem&middot;Male, Rep&middot;Female, Rep&middot;Male, Ind&middot;Female, Ind&middot;Male. <span class="female-label">Pink = female</span>, <span class="male-label">blue = male</span>.</p>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-sent-violin"></div></div>
      <div class="chart-card"><div id="c-sent-gender-bar"></div></div>
      <div class="chart-card"><div id="c-sent-party-bar"></div></div>
      <div class="chart-card"><div id="c-sent-neg-bar"></div></div>
      <div class="chart-card"><div id="c-sent-scatter"></div></div>
    </div>
    <div class="table-actions" style="margin-top:18px;">
      <button class="btn" onclick="sentTable.download('csv','caller_sentiment.csv')">&#8659; Download CSV</button>
      <button class="btn secondary" onclick="sentTable.clearHeaderFilter()">Clear filters</button>
      <span class="table-count" id="sent-table-count"></span>
    </div>
    <div id="sent-table"></div>
    <div class="term-list">
      <p><strong>VADER sentiment model</strong> &mdash; Valence Aware Dictionary and sEntiment Reasoner (Hutto &amp; Gilbert, 2014). A rule-based lexicon model designed for informal and social-media language. For each sentence VADER computes: positive valence score <em>p</em>, negative valence score <em>n</em>, neutral valence score <em>neu</em>, and a compound score C = normalize(&Sigma; valence), where the normalization uses &alpha;&nbsp;=&nbsp;15 to bound C to [&minus;1,&nbsp;+1]. VADER handles negation (&ldquo;not good&rdquo;), degree modifiers (&ldquo;very&rdquo;, &ldquo;barely&rdquo;), ALL-CAPS emphasis, and punctuation. Reference: Hutto, C.J. &amp; Gilbert, E.E. (2014). <em>ICWSM</em>. <a href="https://doi.org/10.1609/icwsm.v8i1.14550" target="_blank">doi:10.1609/icwsm.v8i1.14550</a></p>
      <p><strong>Per-turn compound score</strong> &mdash; Each caller turn is split into sentences on terminal punctuation; VADER scores each sentence; the mean of sentence-level compound scores is the per-turn score. Sentences with |C|&nbsp;&le;&nbsp;0.05 are neutral; C&nbsp;&gt;&nbsp;0.05 is positive; C&nbsp;&lt;&nbsp;&minus;0.05 is negative.</p>
      <p><strong>% positive / negative / neutral sentences</strong> &mdash; The fraction of sentences in a turn falling into each sentiment category. These three fractions sum to 1 for each turn.</p>
      <p><strong>Scatter: % positive sentences vs call duration</strong> &mdash; Only visible when call-duration data is available (requires re-scraping with the updated scraper). X-axis is the fraction of positive sentences; Y-axis is call duration in seconds from <code>play_ep()</code> timestamps.</p>
    </div>
  </div>

  <!-- Sentiment over time -->
  <!-- Sankey: question effectiveness -->
  <div class="group-section" id="sec-sankey">
    <h3>Does a Question Land? &mdash; Sankey Diagrams</h3>
    <p class="note">Nine flow diagrams tracing every path from caller intent to host outcome. Flow width &prop; caller count. Toggle nodes by clicking. Download any chart with the Plotly camera icon. Diagrams 4&ndash;9 use the <strong id="sk-n-resp">&mdash;</strong> calls for which host response text was captured.</p>

    <p class="sk-head">Diagram 1 &mdash; Who calls, what do they say, and how long? <span class="sk-sub">(all <strong id="sk-n-all">&mdash;</strong> calls)</span></p>
    <p class="note">Party line &rarr; statement vs. question &rarr; caller word-count. Shows whether political groups differ in questioning behaviour and call length.</p>
    <div class="chart-card sk-card"><div id="c-sankey-1" style="min-height:400px;"></div></div>

    <p class="sk-head">Diagram 2 &mdash; Does asking get a longer response? <span class="sk-sub">(calls with response data)</span></p>
    <p class="note">Call type &rarr; host response length &rarr; host follow-up. Traces whether question-callers receive more substantive responses than statement-callers.</p>
    <div class="chart-card sk-card"><div id="c-sankey-2" style="min-height:400px;"></div></div>

    <p class="sk-head">Diagram 3 &mdash; Caller investment &rarr; outcome <span class="sk-sub">(calls with response data)</span></p>
    <p class="note">Caller word count &rarr; call type &rarr; outcome tier (substantive / acknowledged / brief). Does investing more words, or framing a question, produce a better answer?</p>
    <div class="chart-card sk-card"><div id="c-sankey-3" style="min-height:400px;"></div></div>

    <p class="sk-head">Diagram 4 &mdash; Does it matter which host you get? <span class="sk-sub">(calls with response data)</span></p>
    <p class="note">Individual host &rarr; call type &rarr; outcome. Reveals each host&rsquo;s disposition toward questions vs. statements — who is most likely to engage substantively, and does it change based on whether you ask a question?</p>
    <div class="chart-card sk-card"><div id="c-sankey-4" style="min-height:400px;"></div></div>

    <p class="sk-head">Diagram 5 &mdash; Does the day of the week matter? <span class="sk-sub">(calls with response data)</span></p>
    <p class="note">Day of broadcast &rarr; call type &rarr; outcome. Explores whether certain broadcast days produce more substantive host engagement. Tuesday and Sunday stand out in the data.</p>
    <div class="chart-card sk-card"><div id="c-sankey-5" style="min-height:400px;"></div></div>

    <p class="sk-head">Diagram 6 &mdash; How many questions is optimal? <span class="sk-sub">(calls with response data)</span></p>
    <p class="note">Question count bucket &rarr; caller length &rarr; outcome. Tests the hypothesis that 1&ndash;2 well-focused questions outperform a barrage of 3+ questions. The data suggest asking 3 or more questions reduces your chance of a substantive answer.</p>
    <div class="chart-card sk-card"><div id="c-sankey-6" style="min-height:400px;"></div></div>

    <p class="sk-head">Diagram 7 &mdash; Does caller tone affect the response? <span class="sk-sub">(calls with response data)</span></p>
    <p class="note">VADER sentiment bucket (negative / neutral / positive) &rarr; call type &rarr; outcome. Examines whether the emotional valence of the call — how positive or negative the caller sounds — influences the quality of the host&rsquo;s response.</p>
    <div class="chart-card sk-card"><div id="c-sankey-7" style="min-height:400px;"></div></div>

    <p class="sk-head">Diagram 8 &mdash; Does hedging language help? <span class="sk-sub">(calls with response data)</span></p>
    <p class="note">Hedging presence &rarr; call type &rarr; outcome. Callers who soften their statements with phrases like &ldquo;I think&rdquo; or &ldquo;maybe&rdquo; show a notably higher substantive-answer rate. Does this hold across question and statement calls alike?</p>
    <div class="chart-card sk-card"><div id="c-sankey-8" style="min-height:400px;"></div></div>

    <p class="sk-head">Diagram 9 &mdash; Does vocabulary richness change the outcome? <span class="sk-sub">(calls with response data)</span></p>
    <p class="note">Vocabulary diversity (type&ndash;token ratio) &rarr; call type &rarr; outcome. Tests whether callers with varied (high-TTR) or repetitive (low-TTR) vocabulary get more substantive responses. Medium diversity shows the best outcome rate in this dataset.</p>
    <div class="chart-card sk-card"><div id="c-sankey-9" style="min-height:400px;"></div></div>

    <div class="term-list">
      <p><strong>Sankey diagram</strong> &mdash; A flow diagram where link width &prop; the number of calls following that path. Each column represents one decision stage in the caller&ndash;host exchange. Reference: Schmidt, M. (2008). <em>Journal of Industrial Ecology</em>, 12(1), 82&ndash;94. <a href="https://doi.org/10.1111/j.1530-9290.2008.00004.x" target="_blank">doi:10.1111/j.1530-9290.2008.00004.x</a></p>
      <p><strong>Outcome classification</strong> &mdash; <em>Substantive answer</em>: host response &gt;&nbsp;60&nbsp;words, OR &ge;&nbsp;20&nbsp;words AND host asked a follow-up question (active engagement). <em>Acknowledged</em>: host response 20&ndash;60&nbsp;words, no follow-up (addressed but not probed). <em>Brief / dismissed</em>: host response &lt;&nbsp;20&nbsp;words, no follow-up (minimal engagement).</p>
      <p><strong>Question count bucket</strong> &mdash; Total sentences in the caller&rsquo;s turn identified as questions (see question heuristic above). Grouped as 0, 1, 2, or 3+. The data show a sweet spot at 1&ndash;2 questions; callers who ask 3 or more questions receive a substantive answer only ~27% of the time vs ~37% for callers who ask exactly one question.</p>
      <p><strong>Hedging language</strong> &mdash; Boolean: the caller used at least one phrase from the hedging lexicon (see Hedging section). Callers who hedge show a ~7 percentage-point higher substantive-answer rate than those who do not (38% vs 31%).</p>
      <p><strong>Vocabulary diversity (TTR)</strong> &mdash; Type&ndash;token ratio = distinct word types &divide; total tokens. High TTR (&gt;&nbsp;0.80) can indicate very short calls or unusually varied diction; medium TTR (0.60&ndash;0.80) appears optimal for engagement in this dataset. See Templin (1957) for TTR discussion.</p>
      <p><strong>Host response coverage</strong> &mdash; Diagrams 2&ndash;9 draw on <strong id="sk-n-resp-2">&mdash;</strong> calls with host response text. Run <code>make fudgie-big</code> to expand this subset.</p>
    </div>
  </div>

  <!-- What makes an effective call? -->
  <div class="group-section" id="sec-effective">
    <h3>What All Effective Calls Have in Common &mdash; and What Ineffective Ones Share</h3>
    <p class="note">Comparing <strong id="ec-n-subst">&mdash;</strong> calls that received a <em>Substantive</em> response against <strong id="ec-n-brief">&mdash;</strong> calls that received a <em>Brief</em> response across every available dimension: linguistic style, syntax, timing, and host identity. Bars above zero (green) are features more common in calls that landed a substantive answer; bars below zero (red) are features that predict a brief dismissal.</p>

    <div class="chart-grid">
      <div class="chart-card"><div id="c-ec-feature-diff"></div></div>
      <div class="chart-card"><div id="c-ec-feature-raw"></div></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-ec-by-host"></div></div>
      <div class="chart-card"><div id="c-ec-by-day"></div></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-ec-qcount"></div></div>
      <div class="chart-card"><div id="c-ec-openers"></div></div>
    </div>

    <div class="term-list">
      <p><strong>Feature difference chart (top left)</strong> &mdash; Each bar is the percentage difference in the mean of that feature between <em>Substantive</em> and <em>Brief</em> calls: &Delta;% = 100 &times; (mean<sub>subst</sub> &minus; mean<sub>brief</sub>) / |mean<sub>brief</sub>|. Positive bars (green) indicate the feature is higher for calls that received a substantive response; negative bars (red) indicate it is higher for calls that were briefly dismissed. The largest positive signal is hedging language (+27%); the largest negative signal is question count (&minus;19%).</p>
      <p><strong>Raw feature comparison (top right)</strong> &mdash; Mean values for the same features shown for all three outcome tiers (Substantive / Acknowledged / Brief) to contextualise the direction and magnitude of each difference.</p>
      <p><strong>% Substantive by host (middle left)</strong> &mdash; For each host the proportion of their calls that received a Substantive response. Pedro Echevarria and Greta Brawner show the highest engagement rates; John McArdle processes the most calls and has a lower substantive rate, partly reflecting his higher call volume.</p>
      <p><strong>% Substantive by day (middle right)</strong> &mdash; Tuesday is the standout day, with nearly 47% of calls receiving a Substantive response — 22 percentage points above Wednesday (25%). Weekend data are sparse and should be interpreted with caution.</p>
      <p><strong>Question count sweet spot (bottom left)</strong> &mdash; The substantive-answer rate peaks at 1&ndash;2 questions (~37%) and drops sharply to ~27% for callers who ask 3 or more questions. Callers with 0 questions (pure statements) also do well at ~35%, suggesting that the quality and framing of a single focused question matters more than asking many questions.</p>
      <p><strong>Opener word rates (bottom right)</strong> &mdash; For the 15 most frequent first meaningful words, the bars show what fraction of <em>Substantive</em> vs <em>Brief</em> calls began with each word. Calls opening with &ldquo;think&rdquo; (as in &ldquo;I think&hellip;&rdquo;) have a notably elevated substantive rate; calls opening with &ldquo;thank&rdquo; skew toward Brief responses, suggesting that leading with gratitude or small talk is less productive than leading with a substantive opinion or framing.</p>
    </div>
  </div>

  <div class="group-section" id="sec-sent-ot">
    <h3>Negativity Over Time</h3>
    <p class="note">Monthly mean VADER compound score (left) and mean % of sentences scored negative (right), split by gender series. Higher negativity % means a greater share of sentences in that month were negative-valenced. Toggle series in the interactive legend.</p>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-sent-ot-compound"></div></div>
      <div class="chart-card"><div id="c-sent-ot-neg"></div></div>
    </div>
    <div class="term-list">
      <p><strong>Monthly mean compound score</strong> &mdash; For each calendar month, the arithmetic mean of per-turn compound scores across all caller turns in that month (optionally filtered by gender series). A declining trend indicates the corpus is becoming more negative on average. Compound score C &isin; [&minus;1,&nbsp;+1]; see VADER definition above.</p>
      <p><strong>Monthly mean % negative sentences</strong> &mdash; For each month, the mean across turns of each turn&rsquo;s proportion of negative sentences (sent_neg). Negative sentences are those with sentence-level compound C&nbsp;&lt;&nbsp;&minus;0.05. Plotted as a fraction (0&ndash;1), where 1 means every sentence in every turn that month was negative.</p>
      <p><strong>Gender series: All / Female / Male / Unknown</strong> &mdash; See gender inference description in &ldquo;Call Volume Over Time&rdquo; above. Months with fewer than 5 turns for a given series are plotted but should be interpreted cautiously given small sample sizes.</p>
    </div>
  </div>
</div><!-- /group-questions -->

    </div>

    <!-- Page: Speaking Style -->
    <div class="nav-page" id="page-style">

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- HOW DO THEY SAY IT?                                                    -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div class="section-group" id="group-how">
  <div class="group-header">
    <h2>How Do They Say It?</h2>
    <p>Linguistic style beyond simple question-asking: hedging language, modal verb use, pronoun patterns, first-word choices, vocabulary scatter, and — once enough re-scraped episodes accumulate — how the host responds.</p>
  </div>

  <!-- Hedging & pronouns -->
  <div class="group-section">
    <h3>Hedging, Modal Verbs &amp; Pronoun Use</h3>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-style-bar"></div></div>
      <div class="chart-card"><div id="c-hedge-violin"></div></div>
    </div>
    <div class="term-list">
      <p><strong>Hedging rate</strong> &mdash; Count of hedging phrases &divide; N. Hedging language signals epistemic uncertainty or face-saving tentativeness (Hyland, 1996. <em>Written Communication</em>, 13(2), 251&ndash;281. <a href="https://doi.org/10.1177/0741088396013002004" target="_blank">doi:10.1177/0741088396013002004</a>). Phrases counted: <em>I think, I feel, I believe, I guess, maybe, perhaps, possibly, it seems, sort of, kind of, I was wondering, I&rsquo;m not sure, I don&rsquo;t know, might be, could be</em>. All matches are case-insensitive continuous substrings.</p>
      <p><strong>Modal verb rate</strong> &mdash; Count of {<em>would, could, should, might, may, ought</em>} &divide; N. Modal verbs express epistemic possibility, deontic obligation, or conditionality and are frequently used to soften assertions or frame hypothetical situations.</p>
      <p><strong>1st- and 2nd-person pronoun rates</strong> &mdash; See definitions in the &ldquo;Questions vs Statements&rdquo; section above. All rates are divided by N (total word count) so that longer turns do not inflate absolute counts.</p>
    </div>
  </div>

  <!-- First meaningful word -->
  <div class="group-section">
    <h3>First Meaningful Word</h3>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-openers"></div></div>
    </div>
    <div class="term-list">
      <p><strong>First meaningful word</strong> &mdash; The first word of each caller&rsquo;s turn after removing the following filler tokens: <em>um, uh, well, yes, yeah, no, so, and, but, i, you, know, like, okay, ok, hi, hello, good</em>. The opener reveals whether a caller leads with a declarative concept, a question word, or a named entity &mdash; a proxy for call intent (statement vs. question) that does not require sentence boundary detection.</p>
    </div>
  </div>

  <!-- Vocabulary scatter -->
  <div class="group-section">
    <h3>Vocabulary &amp; Questions Scatter</h3>
    <p class="note">Every dot is one caller turn. Hover for full details. Toggle gender series in the legend.</p>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-wc-scatter"></div></div>
      <div class="chart-card"><div id="c-q-scatter"></div></div>
    </div>
    <div class="term-list">
      <p><strong>Unique words</strong> &mdash; Distinct word types used in a turn: |V| = TTR &times; N. A turn with 100 words and TTR&nbsp;=&nbsp;0.65 contains 65 distinct word types. Plotted against total word count N, the slope of the point cloud indicates how vocabulary grows with call length.</p>
      <p><strong>Question count</strong> &mdash; The total number of sentences identified as questions (see question identification above). Plotted against N, positive slope indicates that longer turns tend to contain more questions in absolute terms (not necessarily in ratio).</p>
      <p><strong>Hover tooltip</strong> &mdash; Shows caller name, gender, party, date, word count, and the full transcript text of that turn.</p>
    </div>
  </div>

  <!-- Host responsiveness -->
  <div class="group-section" id="sec-responsiveness">
    <h3>Host Responsiveness</h3>
    <p class="note">How does the host respond after each caller? Metrics are computed from the host&rsquo;s first turn immediately following the caller. Only available for episodes scraped after this feature was added. <strong id="resp-n">&mdash;</strong> calls have host response data.</p>
    <div class="chart-grid">
      <div class="chart-card"><div id="c-resp-words"></div></div>
      <div class="chart-card"><div id="c-resp-overlap"></div></div>
      <div class="chart-card"><div id="c-resp-compliment"></div></div>
    </div>
    <div class="table-actions" style="margin-top:14px;">
      <button class="btn" onclick="respTable.download('csv','host_responsiveness.csv')">&#8659; Download CSV</button>
      <button class="btn secondary" onclick="respTable.clearHeaderFilter()">Clear filters</button>
      <span class="table-count" id="resp-table-count"></span>
    </div>
    <div id="resp-table"></div>
    <div class="term-list">
      <p><strong>Host response length</strong> &mdash; Word count of the host&rsquo;s first spoken turn immediately after the caller finishes. Longer responses may indicate the host found the call more substantive.</p>
      <p><strong>Topic overlap (Jaccard similarity)</strong> &mdash; J(A,B) = |A&nbsp;&cap;&nbsp;B| / |A&nbsp;&cup;&nbsp;B|, where A is the set of non-stop-words in the caller&rsquo;s text and B is the set of non-stop-words in the host&rsquo;s response. J = 0 means no shared vocabulary; J = 1 means identical vocabulary. Higher values indicate the host directly addressed the caller&rsquo;s specific topics. Stop-words (function words, pronouns, common auxiliaries) are excluded to focus on content words.</p>
      <p><strong>Compliment detection</strong> &mdash; Boolean flag: True when the host&rsquo;s response matches any phrase in the set {<em>good question, great question, excellent question, good point, great point, very interesting, interesting point, well said</em>}, case-insensitive. Indicates explicit praise from the host.</p>
      <p><strong>Coverage</strong> &mdash; Host response text is captured by the scraper only for episodes fetched after the <code>host_response_text</code> column was added. In unattributed format (2026+ episodes), the host response is approximated as turns that follow the caller&rsquo;s last long speech segment, which is a heuristic. Run <code>make fudgie-big</code> to populate new episodes.</p>
    </div>
  </div>
</div><!-- /group-how -->

    </div>

    <!-- Page: Data Tables -->
    <div class="nav-page" id="page-tables">

<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- DATA TABLES                                                            -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->
<div class="section-group" id="group-data">
  <div class="group-header">
    <h2>Data Tables</h2>
    <p>Sortable, filterable tables for the full dataset. Use column header inputs to filter; click column headers to sort. All tables are downloadable as CSV.</p>
  </div>

  <div class="group-section">
    <h3>Word Frequency Across All Comments</h3>
    <p class="note">All words by total occurrence (stop-words removed). % columns show the share of callers of that gender who used the word at least once. POS = part of speech (from NLTK Penn Treebank tagger).</p>
    <div class="table-actions">
      <button class="btn" onclick="wordTable.download('csv','word_frequency.csv')">&#8659; Download CSV</button>
      <button class="btn secondary" onclick="wordTable.clearHeaderFilter()">Clear filters</button>
      <span class="table-count" id="word-table-count"></span>
    </div>
    <div id="word-freq-table"></div>
  </div>

  <div class="group-section">
    <h3>All Caller Turns</h3>
    <p class="note">Every extracted caller turn with all computed metrics. Sortable and filterable.</p>
    <div class="table-actions">
      <button class="btn" onclick="table.download('csv','cspan_callers.csv')">&#8659; Download CSV</button>
      <button class="btn secondary" onclick="table.clearHeaderFilter()">Clear filters</button>
      <span class="table-count" id="table-count"></span>
    </div>
    <div id="caller-table"></div>
  </div>
</div><!-- /group-data -->

    </div>

    <footer>Generated by analyze_website.py &middot; C-SPAN Washington Journal caller analysis</footer>
  </main>
</div>

<!-- ══ Charts & table ════════════════════════════════════════════════════ -->
<script>
(function () {
  var DATA = null;
  var TABLES = null;
  const F_COLOR = '#D81B60';
  const M_COLOR = '#1565C0';
  const LAYOUT_BASE = {
    paper_bgcolor: '#fff',
    plot_bgcolor: '#F8F9FA',
    margin: { t: 50, b: 40, l: 55, r: 20 },
    legend: { orientation: 'h', y: -0.18 },
    font: { family: 'Inter, Segoe UI, Arial, sans-serif', size: 12 },
  };

  function pLabel(col) {
    const p = DATA.pvals[col];
    return p ? '  [' + p + ']' : '';
  }

  // ── violin ──────────────────────────────────────────────────────────────
  function violin(divId, col, title, yLabel) {
    const traces = [
      { type: 'violin', y: DATA.female[col], name: 'Female',
        box: { visible: true }, meanline: { visible: true },
        fillcolor: F_COLOR, line: { color: F_COLOR }, opacity: 0.72 },
      { type: 'violin', y: DATA.male[col],   name: 'Male',
        box: { visible: true }, meanline: { visible: true },
        fillcolor: M_COLOR, line: { color: M_COLOR }, opacity: 0.72 },
    ];
    Plotly.newPlot(divId, traces, Object.assign({}, LAYOUT_BASE, {
      title: { text: title + pLabel(col), font: { size: 13 } },
      yaxis: { title: yLabel },
    }), { responsive: true, displayModeBar: false });
  }

  // ── word count histogram ─────────────────────────────────────────────────
  function initWcHistogram() {
    Plotly.newPlot('c-wc-hist', [
      { type: 'histogram', name: 'Female',
        x: DATA.female.word_count,
        autobinx: false, xbins: { size: 1 },
        marker: { color: F_COLOR }, opacity: 0.85 },
      { type: 'histogram', name: 'Male',
        x: DATA.male.word_count,
        autobinx: false, xbins: { size: 1 },
        marker: { color: M_COLOR }, opacity: 0.85 },
    ], Object.assign({}, LAYOUT_BASE, {
      barmode: 'group',
      title: { text: 'Callers per word count — female vs male', font: { size: 13 } },
      xaxis: { title: 'Number of words in turn' },
      yaxis: { title: 'Number of callers' },
      height: 420,
    }), { responsive: true, displayModeBar: true });
  }

  // ── grouped bar ─────────────────────────────────────────────────────────
  function groupedBar(divId, metricsKey, title, yLabel) {
    const d = DATA[metricsKey];
    const traces = [
      { type: 'bar', name: 'Female',
        x: d.cols,
        y: d.female.map(v => v.mean),
        error_y: { type: 'data', array: d.female.map(v => v.sem), visible: true },
        marker: { color: F_COLOR } },
      { type: 'bar', name: 'Male',
        x: d.cols,
        y: d.male.map(v => v.mean),
        error_y: { type: 'data', array: d.male.map(v => v.sem), visible: true },
        marker: { color: M_COLOR } },
    ];
    Plotly.newPlot(divId, traces, Object.assign({}, LAYOUT_BASE, {
      barmode: 'group',
      title: { text: title, font: { size: 13 } },
      yaxis: { title: yLabel || 'Mean value' },
    }), { responsive: true, displayModeBar: false });
  }

  // ── opener words horizontal bar ─────────────────────────────────────────
  function openerBar(divId) {
    const fo = DATA.openers.female;
    const mo = DATA.openers.male;
    const traces = [
      { type: 'bar', orientation: 'h',
        name: 'Female', x: fo.counts, y: fo.words,
        marker: { color: F_COLOR }, opacity: 0.82 },
      { type: 'bar', orientation: 'h',
        name: 'Male',   x: mo.counts, y: mo.words,
        marker: { color: M_COLOR }, opacity: 0.82 },
    ];
    Plotly.newPlot(divId, traces, Object.assign({}, LAYOUT_BASE, {
      barmode: 'overlay',
      title: { text: 'Top opener words by gender', font: { size: 13 } },
      xaxis: { title: 'Count' },
      yaxis: { autorange: 'reversed' },
      height: 420,
      margin: { t: 50, b: 40, l: 90, r: 20 },
    }), { responsive: true, displayModeBar: false });
  }

  // ── party × gender ──────────────────────────────────────────────────────
  function partyBar(divId) {
    const pd = DATA.partyData;
    const parties = ['republican', 'democrat', 'independent'];
    const fVals = parties.map(p => pd[p].female);
    const mVals = parties.map(p => pd[p].male);
    const labels = parties.map(p => p.charAt(0).toUpperCase() + p.slice(1));
    Plotly.newPlot(divId, [
      { type: 'bar', name: 'Female', x: labels, y: fVals, marker: { color: F_COLOR } },
      { type: 'bar', name: 'Male',   x: labels, y: mVals, marker: { color: M_COLOR } },
    ], Object.assign({}, LAYOUT_BASE, {
      barmode: 'group',
      title: { text: 'Avg word count by gender × party line', font: { size: 13 } },
      yaxis: { title: 'Avg word count' },
    }), { responsive: true, displayModeBar: false });
  }

  // ── summary cards ────────────────────────────────────────────────────────
  function summaryCards() {
    const s = DATA.summary;
    // Fill about-block live counts
    const ep = document.getElementById('about-n-episodes');
    const tu = document.getElementById('about-n-turns');
    const lb = document.getElementById('about-n-labeled');
    if (ep) ep.textContent = s.n_episodes != null ? s.n_episodes : '—';
    if (tu) tu.textContent = s.total;
    if (lb) lb.textContent = (s.n_female + s.n_male) + ' of ' + s.total;
    const cards = [
      { label: 'Total turns',       value: s.total,     sub: s.n_episodes != null ? s.n_episodes + ' episodes' : null },
      { label: 'Female callers',    value: s.n_female,  sub: null },
      { label: 'Male callers',      value: s.n_male,    sub: null },
      { label: 'Unknown gender',    value: s.n_unknown, sub: null },
      { label: 'Avg words — female', value: s.avg_words_f, sub: null },
      { label: 'Avg words — male',   value: s.avg_words_m, sub: null },
      { label: 'Q ratio — female',  value: s.q_ratio_f, sub: null },
      { label: 'Q ratio — male',    value: s.q_ratio_m, sub: null },
    ];
    const el = document.getElementById('summary-cards');
    el.innerHTML = cards.map(c =>
      '<div class="card">' +
      '<p class="label">' + c.label + '</p>' +
      '<p class="value">' + c.value + '</p>' +
      (c.sub ? '<p class="subvalue">' + c.sub + '</p>' : '') +
      '</div>'
    ).join('');
  }

  // ── caller × host gender interactions ───────────────────────────────────
  function initInteractions() {
    const sec = document.getElementById('sec-interactions');
    const ix = DATA.interactions;
    if (!ix || !ix.combos || ix.combos.length === 0) {
      if (sec) sec.style.display = 'none';
      return;
    }

    const combos = ix.combos;
    const CALLER_COLORS = { female: F_COLOR, male: M_COLOR, unknown: '#9E9E9E' };
    const HOST_PATTERNS = { female: '', male: '/', unknown: 'x' };

    // Build one grouped-bar chart per metric
    function interactionBar(divId, metric, title, yLabel) {
      // Group by caller gender, one bar series per caller gender
      const callerGenders = [...new Set(combos.map(c => c.caller_gender))];
      const hostLabels    = [...new Set(combos.map(c => c.host_gender + ' host'))];

      const traces = callerGenders.map(function(cg) {
        const subset = combos.filter(c => c.caller_gender === cg);
        return {
          type: 'bar',
          name: cg.charAt(0).toUpperCase() + cg.slice(1) + ' caller',
          x: subset.map(c => c.host_gender.charAt(0).toUpperCase() + c.host_gender.slice(1) + ' host'),
          y: subset.map(c => c[metric] || 0),
          text: subset.map(c => 'n=' + c.n),
          textposition: 'outside',
          marker: { color: CALLER_COLORS[cg] || '#aaa', opacity: 0.85 },
        };
      });

      Plotly.newPlot(divId, traces, Object.assign({}, LAYOUT_BASE, {
        barmode: 'group',
        title: { text: title, font: { size: 13 } },
        xaxis: { title: 'Host gender' },
        yaxis: { title: yLabel },
        legend: { orientation: 'h', y: -0.22 },
      }), { responsive: true, displayModeBar: false });
    }

    interactionBar('c-interaction-words',  'word_count',      'Avg word count — caller × host gender',     'Words per turn');
    interactionBar('c-interaction-qratio', 'question_ratio',  'Question ratio — caller × host gender',     'Ratio');
    interactionBar('c-interaction-hedge',  'hedge_rate',      'Hedging rate — caller × host gender',       'Rate per word');

    // Host breakdown table
    if (ix.hostCounts && ix.hostCounts.length > 0) {
      const bd = document.getElementById('host-breakdown');
      if (bd) {
        const sorted = ix.hostCounts.slice().sort((a, b) => b.n_turns - a.n_turns);
        bd.innerHTML =
          '<p style="font-size:12px;color:#666;margin:0 0 8px;font-weight:600;">DETECTED HOSTS IN DATASET</p>' +
          '<div style="display:flex;flex-wrap:wrap;gap:10px;">' +
          sorted.map(function(h) {
            const col = h.gender === 'female' ? F_COLOR : h.gender === 'male' ? M_COLOR : '#999';
            return '<div style="background:#fff;border:1.5px solid ' + col + ';border-radius:8px;' +
                   'padding:8px 14px;font-size:13px;">' +
                   '<span style="color:' + col + ';font-weight:600;">' + h.name + '</span>' +
                   ' <span style="color:#999;font-size:11px;">(' + h.gender + ' · ' + h.n_turns + ' turns)</span>' +
                   '</div>';
          }).join('') +
          '</div>';
      }
    }
  }

  // ── word count distribution grouped bar ─────────────────────────────────
  function initWcDist() {
    const BINS = [
      [15,  30,  '15–30'],
      [31,  50,  '31–50'],
      [51,  75,  '51–75'],
      [76,  100, '76–100'],
      [101, 150, '101–150'],
      [151, 200, '151–200'],
      [201, 300, '201–300'],
      [301, 1e9, '301+'],
    ];
    const fCounts = BINS.map(() => 0);
    const mCounts = BINS.map(() => 0);
    DATA.pointsData.forEach(function(r) {
      const wc = r.word_count || 0;
      const bi = BINS.findIndex(function(b) { return wc >= b[0] && wc <= b[1]; });
      if (bi < 0) return;
      if (r.gender === 'female') fCounts[bi]++;
      else if (r.gender === 'male') mCounts[bi]++;
    });
    const labels = BINS.map(function(b) { return b[2]; });
    Plotly.newPlot('c-wc-dist', [
      { type: 'bar', name: 'Female', x: labels, y: fCounts, marker: { color: F_COLOR }, opacity: 0.85 },
      { type: 'bar', name: 'Male',   x: labels, y: mCounts, marker: { color: M_COLOR }, opacity: 0.85 },
    ], Object.assign({}, LAYOUT_BASE, {
      barmode: 'group',
      title: { text: 'Word count distribution by gender', font: { size: 13 } },
      xaxis: { title: 'Word count range' },
      yaxis: { title: 'Number of callers' },
      height: 400,
    }), { responsive: true, displayModeBar: false });
  }

  // ── words vs unique words scatter ────────────────────────────────────────
  function initWcScatter() {
    const groups = { female: [], male: [], unknown: [] };
    DATA.pointsData.forEach(function(r) {
      const g = (r.gender === 'female' || r.gender === 'male') ? r.gender : 'unknown';
      groups[g].push(r);
    });

    function makeTrace(rows, label, color) {
      return {
        type: 'scatter',
        mode: 'markers',
        name: label,
        x: rows.map(function(r) { return r.word_count; }),
        y: rows.map(function(r) { return Math.round((r.unique_word_ratio || 0) * (r.word_count || 0)); }),
        customdata: rows,
        hoverinfo: 'none',
        marker: { color: color, size: 8, opacity: 0.72,
                  line: { width: 0.5, color: 'rgba(255,255,255,0.6)' } },
      };
    }

    Plotly.newPlot('c-wc-scatter', [
      makeTrace(groups.unknown, 'Unknown', '#9E9E9E'),
      makeTrace(groups.male,    'Male',    M_COLOR),
      makeTrace(groups.female,  'Female',  F_COLOR),
    ], Object.assign({}, LAYOUT_BASE, {
      title: { text: 'Total words vs unique words — every caller turn', font: { size: 13 } },
      xaxis: { title: 'Total words in turn' },
      yaxis: { title: 'Unique words in turn' },
      hovermode: 'closest',
      legend: { orientation: 'h', y: -0.15 },
      height: 480,
    }), { responsive: true, displayModeBar: false });

    // ── custom hover tooltip ──────────────────────────────────────────────
    var tip = document.createElement('div');
    tip.style.cssText = [
      'position:fixed', 'display:none', 'background:#fff',
      'border:1px solid #d0d5dd', 'border-radius:10px', 'padding:14px 18px',
      'max-width:460px', 'max-height:55vh', 'overflow-y:auto',
      'box-shadow:0 6px 24px rgba(0,0,0,.15)', 'font-size:13px',
      'line-height:1.65', 'z-index:9999', 'pointer-events:none',
    ].join(';');
    document.body.appendChild(tip);

    var el = document.getElementById('c-wc-scatter');

    el.on('plotly_hover', function(data) {
      var r = data.points[0].customdata;
      var uniqueWc = Math.round((r.unique_word_ratio || 0) * (r.word_count || 0));
      var gColor = r.gender === 'female' ? F_COLOR : r.gender === 'male' ? M_COLOR : '#9E9E9E';
      tip.innerHTML =
        '<div style="font-weight:700;font-size:14px;margin-bottom:3px;">' +
          (r.name || '<em style="color:#999">no name</em>') +
        '</div>' +
        '<div style="font-size:12px;color:#666;margin-bottom:10px;">' +
          '<span style="color:' + gColor + ';font-weight:600;">' + (r.gender || '—') + '</span>' +
          ' &middot; ' + (r.party || '—') +
          (r.upload_date ? ' &middot; ' + r.upload_date : '') +
          ' &middot; ' + r.word_count + ' words' +
          ' &middot; ' + uniqueWc + ' unique' +
        '</div>' +
        '<div style="border-top:1px solid #eee;padding-top:10px;color:#222;">' +
          String(r.text || '') +
        '</div>';
      tip.style.display = 'block';
    });

    el.on('plotly_unhover', function() { tip.style.display = 'none'; });

    el.addEventListener('mousemove', function(e) {
      if (tip.style.display === 'none') return;
      var pad = 18;
      var x = e.clientX + pad;
      var y = e.clientY + pad;
      if (x + tip.offsetWidth  > window.innerWidth)  x = e.clientX - tip.offsetWidth  - pad;
      if (y + tip.offsetHeight > window.innerHeight) y = e.clientY - tip.offsetHeight - pad;
      tip.style.left = Math.max(0, x) + 'px';
      tip.style.top  = Math.max(0, y) + 'px';
    });
  }

  // ── words vs questions scatter ───────────────────────────────────────────
  function initQScatter() {
    const groups = { female: [], male: [], unknown: [] };
    DATA.pointsData.forEach(function(r) {
      const g = (r.gender === 'female' || r.gender === 'male') ? r.gender : 'unknown';
      groups[g].push(r);
    });

    function makeTrace(rows, label, color) {
      return {
        type: 'scatter',
        mode: 'markers',
        name: label,
        x: rows.map(function(r) { return r.word_count; }),
        y: rows.map(function(r) { return r.question_count || 0; }),
        customdata: rows,
        hoverinfo: 'none',
        marker: { color: color, size: 8, opacity: 0.72,
                  line: { width: 0.5, color: 'rgba(255,255,255,0.6)' } },
      };
    }

    Plotly.newPlot('c-q-scatter', [
      makeTrace(groups.unknown, 'Unknown', '#9E9E9E'),
      makeTrace(groups.male,    'Male',    M_COLOR),
      makeTrace(groups.female,  'Female',  F_COLOR),
    ], Object.assign({}, LAYOUT_BASE, {
      title: { text: 'Total words vs questions asked — every caller turn', font: { size: 13 } },
      xaxis: { title: 'Total words in turn' },
      yaxis: { title: 'Number of questions' },
      hovermode: 'closest',
      legend: { orientation: 'h', y: -0.15 },
      height: 480,
    }), { responsive: true, displayModeBar: false });

    var tip = document.createElement('div');
    tip.style.cssText = [
      'position:fixed', 'display:none', 'background:#fff',
      'border:1px solid #d0d5dd', 'border-radius:10px', 'padding:14px 18px',
      'max-width:460px', 'max-height:55vh', 'overflow-y:auto',
      'box-shadow:0 6px 24px rgba(0,0,0,.15)', 'font-size:13px',
      'line-height:1.65', 'z-index:9999', 'pointer-events:none',
    ].join(';');
    document.body.appendChild(tip);

    var el = document.getElementById('c-q-scatter');

    el.on('plotly_hover', function(data) {
      var r = data.points[0].customdata;
      var gColor = r.gender === 'female' ? F_COLOR : r.gender === 'male' ? M_COLOR : '#9E9E9E';
      tip.innerHTML =
        '<div style="font-weight:700;font-size:14px;margin-bottom:3px;">' +
          (r.name || '<em style="color:#999">no name</em>') +
        '</div>' +
        '<div style="font-size:12px;color:#666;margin-bottom:10px;">' +
          '<span style="color:' + gColor + ';font-weight:600;">' + (r.gender || '—') + '</span>' +
          ' &middot; ' + (r.party || '—') +
          (r.upload_date ? ' &middot; ' + r.upload_date : '') +
          ' &middot; ' + r.word_count + ' words' +
          ' &middot; ' + (r.question_count || 0) + ' questions' +
        '</div>' +
        '<div style="border-top:1px solid #eee;padding-top:10px;color:#222;">' +
          String(r.text || '') +
        '</div>';
      tip.style.display = 'block';
    });

    el.on('plotly_unhover', function() { tip.style.display = 'none'; });

    el.addEventListener('mousemove', function(e) {
      if (tip.style.display === 'none') return;
      var pad = 18;
      var x = e.clientX + pad;
      var y = e.clientY + pad;
      if (x + tip.offsetWidth  > window.innerWidth)  x = e.clientX - tip.offsetWidth  - pad;
      if (y + tip.offsetHeight > window.innerHeight) y = e.clientY - tip.offsetHeight - pad;
      tip.style.left = Math.max(0, x) + 'px';
      tip.style.top  = Math.max(0, y) + 'px';
    });
  }

  // ── calls & words over time (dual y-axis) ───────────────────────────────
  function initCallsOverTime() {
    const ts = DATA.timeSeries;
    const sec = document.getElementById('sec-callsovertime');
    if (!ts || !ts.months || ts.months.length === 0) {
      if (sec) sec.style.display = 'none';
      return;
    }

    const GENDER_SERIES = [
      { key: 'all',     label: 'All callers',     color: '#555',    dash: 'solid' },
      { key: 'female',  label: 'Female callers',   color: F_COLOR,   dash: 'solid' },
      { key: 'male',    label: 'Male callers',     color: M_COLOR,   dash: 'solid' },
      { key: 'unknown', label: 'Unknown gender',   color: '#aaa',    dash: 'dot'   },
    ];

    const sharedLayout = {
      legend: { x: 1.03, xanchor: 'left', y: 1, bgcolor: 'rgba(255,255,255,0.85)',
                bordercolor: '#ddd', borderwidth: 1 },
      yaxis:  { title: 'Callers (count)', titlefont: { size: 11 } },
      yaxis2: { title: 'Words spoken', overlaying: 'y', side: 'right',
                titlefont: { size: 11 }, showgrid: false },
      xaxis:  { title: 'Month', tickangle: -35 },
      margin: { r: 160 },
      height: 360,
    };

    function makeMonthlyTraces() {
      const traces = [];
      GENDER_SERIES.forEach(g => {
        traces.push({
          type: 'scatter', mode: 'lines+markers',
          x: ts.months, y: ts[g.key + '_counts'],
          name: g.label + ' (callers)',
          line: { color: g.color, width: 2, dash: g.dash },
          marker: { size: 5, color: g.color },
        });
        traces.push({
          type: 'scatter', mode: 'lines',
          x: ts.months, y: ts[g.key + '_words'],
          name: g.label + ' (words)',
          yaxis: 'y2',
          line: { color: g.color, width: 1.5, dash: 'dash' },
          opacity: 0.7,
        });
      });
      return traces;
    }

    function makeCumulativeTraces() {
      const traces = [];
      GENDER_SERIES.forEach(g => {
        traces.push({
          type: 'scatter', mode: 'lines',
          x: ts.months, y: ts['cum_' + g.key + '_counts'],
          name: g.label + ' (callers)',
          line: { color: g.color, width: 2, dash: g.dash },
        });
        traces.push({
          type: 'scatter', mode: 'lines',
          x: ts.months, y: ts['cum_' + g.key + '_words'],
          name: g.label + ' (words)',
          yaxis: 'y2',
          line: { color: g.color, width: 1.5, dash: 'dash' },
          opacity: 0.7,
        });
      });
      return traces;
    }

    Plotly.newPlot('c-ts-monthly', makeMonthlyTraces(),
      Object.assign({}, LAYOUT_BASE, sharedLayout, {
        title: { text: 'Monthly callers &amp; words by gender', font: { size: 13 } },
      }),
      { responsive: true, displayModeBar: true,
        modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d'],
        toImageButtonOptions: { format: 'png', filename: 'cspan_monthly', scale: 2 } });

    Plotly.newPlot('c-ts-cumulative', makeCumulativeTraces(),
      Object.assign({}, LAYOUT_BASE, sharedLayout, {
        title: { text: 'Cumulative callers &amp; words by gender', font: { size: 13 } },
      }),
      { responsive: true, displayModeBar: true,
        modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d'],
        toImageButtonOptions: { format: 'png', filename: 'cspan_cumulative', scale: 2 } });
  }

  // ── geographic map & table ───────────────────────────────────────────────
  var geoTable;
  function initGeo() {
    const sec = document.getElementById('sec-geo');
    const G = DATA.geo;
    if (!G || !G.byState || G.byState.length === 0) {
      if (sec) sec.style.display = 'none';
      return;
    }
    const states = G.byState;
    const totalGeo = states.reduce(function(s, r) { return s + r.total; }, 0);
    const el_n = document.getElementById('geo-n');
    const el_t = document.getElementById('geo-total');
    if (el_n) el_n.textContent = totalGeo.toLocaleString();
    if (el_t) el_t.textContent = DATA.summary.total.toLocaleString();

    // ── Leaflet map ─────────────────────────────────────────────────────
    var map = L.map('caller-map').setView([38.5, -96], 4);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(map);

    const maxTotal = Math.max.apply(null, states.map(function(s) { return s.total; }));

    states.forEach(function(s) {
      var radius = 6 + 38 * Math.sqrt(s.total / maxTotal);
      var fPct   = s.female + s.male > 0
        ? Math.round(100 * s.female / (s.female + s.male)) : 0;
      var circle = L.circleMarker([s.lat, s.lon], {
        radius:      radius,
        fillColor:   '#1565C0',
        color:       '#fff',
        weight:      1.5,
        opacity:     1,
        fillOpacity: 0.65,
      }).addTo(map);

      circle.bindPopup(
        '<div style="font-size:13px;line-height:1.7;">' +
        '<strong style="font-size:15px;">' + s.state + '</strong><br>' +
        '<b>Total callers:</b> ' + s.total + '<br>' +
        '<span style="color:#D81B60;"><b>Female:</b> ' + s.female + '</span>  ' +
        '<span style="color:#1565C0;"><b>Male:</b> ' + s.male + '</span>  ' +
        '<span style="color:#999;"><b>Unknown:</b> ' + s.unknown + '</span><br>' +
        (s.female + s.male > 0 ? '<b>% female (labeled):</b> ' + fPct + '%<br>' : '') +
        '<hr style="margin:4px 0;">' +
        '<span style="color:#B71C1C;"><b>Republican:</b> ' + s.republican + '</span><br>' +
        '<span style="color:#1565C0;"><b>Democrat:</b> ' + s.democrat + '</span><br>' +
        '<span style="color:#2E7D32;"><b>Independent:</b> ' + s.independent + '</span>' +
        '</div>'
      );
    });

    // ── Tabulator ──────────────────────────────────────────────────────
    geoTable = new Tabulator('#geo-table', {
      data: states,
      layout: 'fitColumns',
      pagination: true,
      paginationSize: 25,
      paginationSizeSelector: [10, 25, 50],
      initialSort: [{ column: 'total', dir: 'desc' }],
      columns: [
        { title: 'State',       field: 'state',       width: 80,  headerFilter: 'input' },
        { title: 'Total',       field: 'total',       width: 80,  sorter: 'number', hozAlign: 'right' },
        { title: 'Female',      field: 'female',      width: 80,  sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v > 0 ? '<span style="color:#D81B60;font-weight:600;">' + v + '</span>' : v;
          }
        },
        { title: 'Male',        field: 'male',        width: 80,  sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v > 0 ? '<span style="color:#1565C0;font-weight:600;">' + v + '</span>' : v;
          }
        },
        { title: 'Unknown',     field: 'unknown',     width: 90,  sorter: 'number', hozAlign: 'right' },
        { title: 'Republican',  field: 'republican',  width: 105, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v > 0 ? '<span style="color:#B71C1C;">' + v + '</span>' : v;
          }
        },
        { title: 'Democrat',    field: 'democrat',    width: 100, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v > 0 ? '<span style="color:#1565C0;">' + v + '</span>' : v;
          }
        },
        { title: 'Independent', field: 'independent', width: 110, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v > 0 ? '<span style="color:#2E7D32;">' + v + '</span>' : v;
          }
        },
      ],
    });
    geoTable.on('dataFiltered', function(filters, rows) {
      document.getElementById('geo-table-count').textContent = rows.length + ' states';
    });
    document.getElementById('geo-table-count').textContent = states.length + ' states';
  }

  // ── sentiment charts & table ─────────────────────────────────────────────
  var sentTable;
  function initSentiment() {
    const sec = document.getElementById('sec-sentiment');
    const S = DATA.sentiment;
    if (!S || !S.byGender) { if (sec) sec.style.display = 'none'; return; }

    const POS_COLOR = '#2E7D32';
    const NEG_COLOR = '#C62828';
    const NEU_COLOR = '#9E9E9E';

    // ── violin: compound score by gender ──────────────────────────────────
    if (S.violin && (S.violin.female.length || S.violin.male.length)) {
      Plotly.newPlot('c-sent-violin', [
        { type: 'violin', name: 'Female', y: S.violin.female,
          box: { visible: true }, meanline: { visible: true },
          fillcolor: F_COLOR, line: { color: F_COLOR }, opacity: 0.72 },
        { type: 'violin', name: 'Male', y: S.violin.male,
          box: { visible: true }, meanline: { visible: true },
          fillcolor: M_COLOR, line: { color: M_COLOR }, opacity: 0.72 },
      ], Object.assign({}, LAYOUT_BASE, {
        title: { text: 'Compound sentiment score by gender', font: { size: 13 } },
        yaxis: { title: 'Compound score (−1 to +1)', zeroline: true, zerolinecolor: '#ccc' },
      }), { responsive: true, displayModeBar: false });
    }

    // ── party × gender sentiment charts ───────────────────────────────────
    // Six groups: Democrat·Female, Democrat·Male, Republican·Female, …
    const COMBOS = [
      { p: 'democrat',    g: 'female', label: 'Dem · Female',  fc: F_COLOR, pat: '' },
      { p: 'democrat',    g: 'male',   label: 'Dem · Male',    fc: M_COLOR, pat: '/' },
      { p: 'republican',  g: 'female', label: 'Rep · Female',  fc: F_COLOR, pat: '' },
      { p: 'republican',  g: 'male',   label: 'Rep · Male',    fc: M_COLOR, pat: '/' },
      { p: 'independent', g: 'female', label: 'Ind · Female',  fc: F_COLOR, pat: '' },
      { p: 'independent', g: 'male',   label: 'Ind · Male',    fc: M_COLOR, pat: '/' },
    ];
    const comboLabels = COMBOS.map(function(c) { return c.label; });

    function comboVal(field) {
      return COMBOS.map(function(c) {
        const d = S.byParty[c.p] && S.byParty[c.p][c.g];
        return d ? d[field] : null;
      });
    }

    // Colour each bar by gender, with a text annotation showing n=
    function comboTrace(field, label, color, borderColor) {
      return {
        type: 'bar', name: label,
        x: comboLabels,
        y: comboVal(field),
        text: COMBOS.map(function(c) {
          const d = S.byParty[c.p] && S.byParty[c.p][c.g];
          return d ? 'n=' + d.n : '';
        }),
        textposition: 'outside',
        textfont: { size: 10 },
        marker: {
          color: COMBOS.map(function(c) { return c.fc; }),
          opacity: 0.82,
          line: { color: '#fff', width: 1 },
        },
      };
    }

    // Chart 1: mean compound score
    Plotly.newPlot('c-sent-gender-bar', [comboTrace('compound', 'Compound score', null, null)],
      Object.assign({}, LAYOUT_BASE, {
        title: { text: 'Mean compound sentiment score by party &amp; gender', font: { size: 13 } },
        yaxis: { title: 'Mean compound score (−1 to +1)', zeroline: true, zerolinecolor: '#bbb' },
        showlegend: false,
        height: 400,
      }), { responsive: true, displayModeBar: false });

    // Chart 2: % positive sentences
    Plotly.newPlot('c-sent-party-bar', [comboTrace('pos', '% positive', null, null)],
      Object.assign({}, LAYOUT_BASE, {
        title: { text: '% positive sentences by party &amp; gender', font: { size: 13 } },
        yaxis: { title: 'Fraction of sentences', tickformat: '.0%' },
        showlegend: false,
        height: 400,
      }), { responsive: true, displayModeBar: false });

    // Chart 3: % negative sentences (new div)
    Plotly.newPlot('c-sent-neg-bar', [comboTrace('neg', '% negative', null, null)],
      Object.assign({}, LAYOUT_BASE, {
        title: { text: '% negative sentences by party &amp; gender', font: { size: 13 } },
        yaxis: { title: 'Fraction of sentences', tickformat: '.0%' },
        showlegend: false,
        height: 400,
      }), { responsive: true, displayModeBar: false });

    // ── scatter: % positive sentences vs call duration ─────────────────────
    if (S.scatter && S.scatter.length > 0) {
      const sg = { female: [], male: [], unknown: [] };
      S.scatter.forEach(function(r) {
        const g = (r.gender === 'female' || r.gender === 'male') ? r.gender : 'unknown';
        sg[g].push(r);
      });
      function sTrace(rows, label, color) {
        return {
          type: 'scatter', mode: 'markers', name: label,
          x: rows.map(function(r) { return r.x; }),
          y: rows.map(function(r) { return r.y; }),
          customdata: rows,
          hoverinfo: 'none',
          marker: { color: color, size: 8, opacity: 0.72,
                    line: { width: 0.5, color: 'rgba(255,255,255,0.6)' } },
        };
      }
      Plotly.newPlot('c-sent-scatter', [
        sTrace(sg.unknown, 'Unknown', '#9E9E9E'),
        sTrace(sg.male,    'Male',    M_COLOR),
        sTrace(sg.female,  'Female',  F_COLOR),
      ], Object.assign({}, LAYOUT_BASE, {
        title: { text: '% positive sentences vs call duration', font: { size: 13 } },
        xaxis: { title: '% positive sentences', tickformat: '.0%' },
        yaxis: { title: 'Call duration (seconds)' },
        hovermode: 'closest',
        legend: { orientation: 'h', y: -0.15 },
        height: 480,
      }), { responsive: true, displayModeBar: false });

      var stip = document.createElement('div');
      stip.style.cssText = [
        'position:fixed', 'display:none', 'background:#fff',
        'border:1px solid #d0d5dd', 'border-radius:10px', 'padding:14px 18px',
        'max-width:460px', 'max-height:55vh', 'overflow-y:auto',
        'box-shadow:0 6px 24px rgba(0,0,0,.15)', 'font-size:13px',
        'line-height:1.65', 'z-index:9999', 'pointer-events:none',
      ].join(';');
      document.body.appendChild(stip);

      var sel = document.getElementById('c-sent-scatter');
      sel.on('plotly_hover', function(data) {
        var r = data.points[0].customdata;
        var gColor = r.gender === 'female' ? F_COLOR : r.gender === 'male' ? M_COLOR : '#9E9E9E';
        stip.innerHTML =
          '<div style="font-weight:700;font-size:14px;margin-bottom:3px;">' +
            (r.name || '<em style="color:#999">no name</em>') +
          '</div>' +
          '<div style="font-size:12px;color:#666;margin-bottom:10px;">' +
            '<span style="color:' + gColor + ';font-weight:600;">' + (r.gender || '—') + '</span>' +
            ' &middot; ' + (r.party || '—') +
            ' &middot; ' + Math.round(r.x * 100) + '% positive' +
            ' &middot; compound ' + r.compound +
            ' &middot; ' + r.y + 's' +
          '</div>' +
          '<div style="border-top:1px solid #eee;padding-top:10px;color:#222;">' +
            String(r.text || '') +
          '</div>';
        stip.style.display = 'block';
      });
      sel.on('plotly_unhover', function() { stip.style.display = 'none'; });
      sel.addEventListener('mousemove', function(e) {
        if (stip.style.display === 'none') return;
        var pad = 18, x = e.clientX + pad, y = e.clientY + pad;
        if (x + stip.offsetWidth  > window.innerWidth)  x = e.clientX - stip.offsetWidth  - pad;
        if (y + stip.offsetHeight > window.innerHeight) y = e.clientY - stip.offsetHeight - pad;
        stip.style.left = Math.max(0, x) + 'px';
        stip.style.top  = Math.max(0, y) + 'px';
      });
    } else {
      var sdiv = document.getElementById('c-sent-scatter');
      if (sdiv) sdiv.closest('.chart-card').style.display = 'none';
    }

  }

  function initSentTable() {
    if (!TABLES || !TABLES.sentTableData || TABLES.sentTableData.length === 0) return;
    sentTable = new Tabulator('#sent-table', {
      data: TABLES.sentTableData,
      layout: 'fitColumns',
      pagination: true,
      paginationSize: 25,
      paginationSizeSelector: [10, 25, 50, 100],
      movableColumns: true,
      initialSort: [{ column: 'sent_compound', dir: 'desc' }],
      columns: [
        { title: 'Gender', field: 'gender', width: 85, headerFilter: 'select',
          headerFilterParams: { values: { '': 'All', female: 'Female', male: 'Male', unknown: 'Unknown' } },
          formatter: function(cell) {
            const v = cell.getValue();
            if (v === 'female') return '<span style="color:#D81B60;font-weight:600;">Female</span>';
            if (v === 'male')   return '<span style="color:#1565C0;font-weight:600;">Male</span>';
            return v || '';
          }
        },
        { title: 'Party', field: 'party', width: 110, headerFilter: 'select',
          headerFilterParams: { values: { '': 'All', republican: 'Republican', democrat: 'Democrat', independent: 'Independent', unknown: 'Unknown' } },
        },
        { title: 'Name',    field: 'name',          width: 100, headerFilter: 'input' },
        { title: 'Date',    field: 'upload_date',   width: 100 },
        { title: 'Compound', field: 'sent_compound', width: 100, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = parseFloat(cell.getValue());
            if (isNaN(v)) return '—';
            const col = v > 0.05 ? POS_COLOR : v < -0.05 ? NEG_COLOR : NEU_COLOR;
            const sign = v > 0 ? '+' : '';
            return '<span style="color:' + col + ';font-weight:600;">' + sign + v.toFixed(3) + '</span>';
          }
        },
        { title: '% Pos', field: 'sent_pos', width: 80, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v !== '' ? '<span style="color:' + POS_COLOR + ';">' + Math.round(v * 100) + '%</span>' : '—';
          }
        },
        { title: '% Neg', field: 'sent_neg', width: 80, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v !== '' ? '<span style="color:' + NEG_COLOR + ';">' + Math.round(v * 100) + '%</span>' : '—';
          }
        },
        { title: '% Neu', field: 'sent_neu', width: 80, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v !== '' ? Math.round(v * 100) + '%' : '—';
          }
        },
        { title: 'Words', field: 'word_count', width: 70, sorter: 'number', hozAlign: 'right' },
        { title: 'Text (excerpt)', field: 'text', minWidth: 260,
          formatter: function(cell) {
            const v = cell.getValue() || '';
            const short = v.length > 180 ? v.slice(0, 180) + '…' : v;
            return '<span title="' + v.replace(/"/g, '&quot;') + '">' + short + '</span>';
          },
          headerFilter: 'input',
        },
      ],
      rowFormatter: function(row) {
        const g = row.getData().gender;
        if (g === 'female') row.getElement().style.background = '#FFF0F5';
        if (g === 'male')   row.getElement().style.background = '#EFF4FF';
      },
    });
    sentTable.on('dataFiltered', function(filters, rows) {
      document.getElementById('sent-table-count').textContent = rows.length + ' of ' + TABLES.sentTableData.length + ' rows';
    });
    document.getElementById('sent-table-count').textContent = TABLES.sentTableData.length + ' rows';
  }

  // ── host responsiveness ──────────────────────────────────────────────────
  var respTable;
  function initResponsivenessCharts() {
    const sec = document.getElementById('sec-responsiveness');
    const R = DATA.responsiveness;
    if (!R || !R.byGender || Object.keys(R.byGender).length === 0) {
      if (sec) sec.style.display = 'none';
      return;
    }

    const rn = document.getElementById('resp-n');
    if (rn) rn.textContent = (R.n_with_response || 0).toLocaleString();

    const genders = Object.keys(R.byGender);
    const gColors = genders.map(function(g) {
      return g === 'female' ? F_COLOR : g === 'male' ? M_COLOR : '#9E9E9E';
    });
    const gLabels = genders.map(function(g) { return g.charAt(0).toUpperCase() + g.slice(1); });
    const nAnnot  = genders.map(function(g) { return 'n=' + R.byGender[g].n; });

    function respBar(divId, field, title, yLabel, fmt) {
      var yVals = genders.map(function(g) { return R.byGender[g][field]; });
      Plotly.newPlot(divId, [{
        type: 'bar', x: gLabels, y: yVals,
        text: nAnnot, textposition: 'outside', textfont: { size: 10 },
        marker: { color: gColors, opacity: 0.85 },
      }], Object.assign({}, LAYOUT_BASE, {
        title: { text: title, font: { size: 13 } },
        yaxis: { title: yLabel, tickformat: fmt || '' },
        showlegend: false,
        height: 380,
      }), { responsive: true, displayModeBar: false });
    }

    respBar('c-resp-words',     'avg_host_words',  'Avg host response length by caller gender', 'Words in host response');
    respBar('c-resp-overlap',   'avg_overlap',     'Avg topic overlap (Jaccard) by caller gender', 'Jaccard similarity (0–1)');
    respBar('c-resp-compliment','compliment_rate', 'Host compliment rate by caller gender', 'Fraction of calls', '.0%');
  }

  function initRespTable() {
    if (!TABLES || !TABLES.respTableData || TABLES.respTableData.length === 0) return;
    var hasHostSent = 'host_resp_sentiment' in TABLES.respTableData[0];

    respTable = new Tabulator('#resp-table', {
      data: TABLES.respTableData,
      layout: 'fitColumns',
      pagination: true,
      paginationSize: 25,
      paginationSizeSelector: [10, 25, 50, 100],
      initialSort: [{ column: 'host_resp_words', dir: 'desc' }],
      movableColumns: true,
      columns: [
        { title: 'Gender', field: 'gender', width: 85, headerFilter: 'select',
          headerFilterParams: { values: { '': 'All', female: 'Female', male: 'Male', unknown: 'Unknown' } },
          formatter: function(cell) {
            var v = cell.getValue();
            if (v === 'female') return '<span style="color:#D81B60;font-weight:600;">Female</span>';
            if (v === 'male')   return '<span style="color:#1565C0;font-weight:600;">Male</span>';
            return v || '';
          }
        },
        { title: 'Party', field: 'party', width: 110, headerFilter: 'select',
          headerFilterParams: { values: { '': 'All', republican: 'Republican', democrat: 'Democrat', independent: 'Independent', unknown: 'Unknown' } },
        },
        { title: 'Name',        field: 'name',          width: 100, headerFilter: 'input' },
        { title: 'Date',        field: 'upload_date',   width: 100 },
        { title: 'Caller wds',  field: 'word_count',    width: 90,  sorter: 'number', hozAlign: 'right' },
        { title: 'Host wds',    field: 'host_resp_words', width: 85, sorter: 'number', hozAlign: 'right' },
        { title: 'Overlap',     field: 'word_overlap',  width: 85,  sorter: 'number', hozAlign: 'right',
          formatter: function(cell) { return (parseFloat(cell.getValue()) || 0).toFixed(3); }
        },
        { title: 'Compliment',  field: 'host_compliment', width: 100, hozAlign: 'center',
          formatter: function(cell) {
            return cell.getValue() ? '<span style="color:#2E7D32;font-weight:700;">&#10003;</span>' : '';
          }
        },
        { title: 'Follow-up?', field: 'host_followup_q', width: 95, hozAlign: 'center',
          formatter: function(cell) {
            return cell.getValue() ? '<span style="color:#1565C0;font-weight:700;">?</span>' : '';
          }
        },
        (hasHostSent ? { title: 'Host sent.', field: 'host_resp_sentiment', width: 90,
          sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            var v = parseFloat(cell.getValue());
            if (isNaN(v)) return '—';
            var col = v > 0.05 ? '#2E7D32' : v < -0.05 ? '#C62828' : '#9E9E9E';
            return '<span style="color:' + col + ';font-weight:600;">' + (v > 0 ? '+' : '') + v.toFixed(3) + '</span>';
          }
        } : null),
        { title: 'Caller text', field: 'text', minWidth: 180,
          formatter: function(cell) {
            var v = cell.getValue() || '';
            var s = v.length > 120 ? v.slice(0, 120) + '…' : v;
            return '<span style="color:#666;" title="' + v.replace(/"/g, '&quot;') + '">' + s + '</span>';
          },
          headerFilter: 'input',
        },
        { title: 'Host response', field: 'host_response_text', minWidth: 200,
          formatter: function(cell) {
            var v = cell.getValue() || '';
            var s = v.length > 140 ? v.slice(0, 140) + '…' : v;
            return '<span title="' + v.replace(/"/g, '&quot;') + '">' + s + '</span>';
          },
          headerFilter: 'input',
        },
      ].filter(Boolean),
      rowFormatter: function(row) {
        var g = row.getData().gender;
        if (g === 'female') row.getElement().style.background = '#FFF0F5';
        if (g === 'male')   row.getElement().style.background = '#EFF4FF';
      },
    });
    respTable.on('dataFiltered', function(filters, rows) {
      document.getElementById('resp-table-count').textContent = rows.length + ' of ' + TABLES.respTableData.length + ' rows';
    });
    document.getElementById('resp-table-count').textContent = TABLES.respTableData.length + ' rows';
  }

  // ── Tabulator ────────────────────────────────────────────────────────────
  var table;
  function initTable() {
    const hasText = TABLES.tableData.length && 'text' in TABLES.tableData[0];
    const cols = [
      { title: 'Gender',  field: 'gender',  width: 80,  headerFilter: 'select',
        headerFilterParams: { values: { '': 'All', female: 'Female', male: 'Male', unknown: 'Unknown' } },
        formatter: function(cell) {
          const v = cell.getValue();
          if (v === 'female') return '<span style="color:#D81B60;font-weight:600;">Female</span>';
          if (v === 'male')   return '<span style="color:#1565C0;font-weight:600;">Male</span>';
          return v || '';
        }
      },
      { title: 'Party',   field: 'party',   width: 110, headerFilter: 'select',
        headerFilterParams: { values: { '': 'All', republican: 'Republican', democrat: 'Democrat', independent: 'Independent', unknown: 'Unknown' } },
      },
      { title: 'Name',    field: 'name',    width: 100, headerFilter: 'input' },
      { title: 'Words',   field: 'word_count',             width: 75,  sorter: 'number', hozAlign: 'right' },
      { title: 'Sentences', field: 'sentence_count',        width: 90,  sorter: 'number', hozAlign: 'right' },
      { title: 'Questions', field: 'question_count',        width: 90,  sorter: 'number', hozAlign: 'right' },
      { title: 'Q ratio', field: 'question_ratio',          width: 80,  sorter: 'number', hozAlign: 'right' },
      { title: 'Wds/sent', field: 'avg_words_per_sentence', width: 90,  sorter: 'number', hozAlign: 'right' },
      { title: 'Vocab div', field: 'unique_word_ratio',     width: 90,  sorter: 'number', hozAlign: 'right' },
      { title: 'Hedge',   field: 'hedge_rate',              width: 75,  sorter: 'number', hozAlign: 'right' },
      { title: 'Modal',   field: 'modal_rate',              width: 75,  sorter: 'number', hozAlign: 'right' },
      { title: '1st-P',   field: 'first_p_rate',            width: 70,  sorter: 'number', hozAlign: 'right' },
      { title: '2nd-P',   field: 'second_p_rate',           width: 70,  sorter: 'number', hozAlign: 'right' },
      { title: 'Episode', field: 'episode_id',              width: 110, headerFilter: 'input' },
    ];
    if (hasText) {
      cols.push({
        title: 'Text (excerpt)',
        field: 'text',
        minWidth: 260,
        formatter: function(cell) {
          const v = cell.getValue() || '';
          const short = v.length > 180 ? v.slice(0, 180) + '…' : v;
          return '<span title="' + v.replace(/"/g, '&quot;') + '">' + short + '</span>';
        },
        headerFilter: 'input',
      });
    }

    table = new Tabulator('#caller-table', {
      data: TABLES.tableData,
      columns: cols,
      layout: 'fitColumns',
      pagination: true,
      paginationSize: 25,
      paginationSizeSelector: [10, 25, 50, 100],
      movableColumns: true,
      initialSort: [{ column: 'word_count', dir: 'desc' }],
      rowFormatter: function(row) {
        const g = row.getData().gender;
        if (g === 'female') row.getElement().style.background = '#FFF0F5';
        if (g === 'male')   row.getElement().style.background = '#EFF4FF';
      },
    });

    // Live row count
    table.on('dataFiltered', function(filters, rows) {
      document.getElementById('table-count').textContent = rows.length + ' of ' + TABLES.tableData.length + ' rows';
    });
    document.getElementById('table-count').textContent = TABLES.tableData.length + ' rows';
  }

  // ── word frequency table ──────────────────────────────────────────────────
  var wordTable;
  function initWordFreqTable() {
    if (!DATA.wordFreq || DATA.wordFreq.length === 0) return;
    const POS_COLORS = {
      noun: '#1565C0', verb: '#2E7D32', adjective: '#E65100', adverb: '#6A1B9A',
      preposition: '#795548', pronoun: '#00838F', conjunction: '#546E7A',
      determiner: '#9E9E9E', interjection: '#F57F17', other: '#9E9E9E',
    };
    wordTable = new Tabulator('#word-freq-table', {
      data: DATA.wordFreq,
      layout: 'fitData',
      pagination: true,
      paginationSize: 25,
      paginationSizeSelector: [10, 25, 50, 100],
      initialSort: [{ column: 'count', dir: 'desc' }],
      columns: [
        { title: 'Word', field: 'word', width: 130, headerFilter: 'input',
          formatter: function(cell) { return '<strong>' + cell.getValue() + '</strong>'; }
        },
        { title: 'POS', field: 'pos', width: 100, headerFilter: 'select',
          headerFilterParams: { values: { '': 'All', noun: 'noun', verb: 'verb',
            adjective: 'adjective', adverb: 'adverb', preposition: 'preposition',
            pronoun: 'pronoun', conjunction: 'conjunction', other: 'other' } },
          formatter: function(cell) {
            const v = cell.getValue() || '—';
            const col = POS_COLORS[v] || '#999';
            return '<span style="color:' + col + ';font-weight:600;font-size:11px;' +
                   'background:' + col + '18;padding:1px 6px;border-radius:3px;">' + v + '</span>';
          }
        },
        { title: 'Total uses', field: 'count',        width: 100, sorter: 'number', hozAlign: 'right' },
        { title: 'F uses',     field: 'female_count', width: 80,  sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v > 0 ? '<span style="color:#D81B60;">' + v + '</span>' : v;
          }
        },
        { title: 'M uses',     field: 'male_count',   width: 80,  sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v > 0 ? '<span style="color:#1565C0;">' + v + '</span>' : v;
          }
        },
        { title: '# callers',   field: 'n_callers',        width: 90,  sorter: 'number', hozAlign: 'right' },
        { title: '# F callers', field: 'n_female_callers',  width: 95,  sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v > 0 ? '<span style="color:#D81B60;">' + v + '</span>' : v;
          }
        },
        { title: '# M callers', field: 'n_male_callers',    width: 95,  sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            return v > 0 ? '<span style="color:#1565C0;">' + v + '</span>' : v;
          }
        },
        { title: '% all callers',    field: 'pct_all',    width: 115, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) { return cell.getValue() + '%'; }
        },
        { title: '% F callers',  field: 'pct_female', width: 105, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            return '<span style="color:#D81B60;">' + cell.getValue() + '%</span>';
          }
        },
        { title: '% M callers',  field: 'pct_male',   width: 105, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            return '<span style="color:#1565C0;">' + cell.getValue() + '%</span>';
          }
        },
        { title: '%F − %M', field: 'pct_diff', width: 100, sorter: 'number', hozAlign: 'right',
          formatter: function(cell) {
            const v = cell.getValue();
            if (v === null || v === '') return '—';
            const col = v > 0 ? '#D81B60' : v < 0 ? '#1565C0' : '#999';
            const sign = v > 0 ? '+' : '';
            return '<span style="color:' + col + ';font-weight:600;">' + sign + v + '%</span>';
          }
        },
      ],
    });
    wordTable.on('dataFiltered', function(filters, rows) {
      document.getElementById('word-table-count').textContent = rows.length + ' of ' + DATA.wordFreq.length + ' words';
    });
    document.getElementById('word-table-count').textContent = DATA.wordFreq.length + ' words';
  }

  // ── init ─────────────────────────────────────────────────────────────────
  // ── day of week ─────────────────────────────────────────────────────────
  function initDayOfWeek() {
    const D = DATA.dayOfWeek;
    const sec = document.getElementById('sec-dow');
    if (!D || !D.days || D.days.length === 0) {
      if (sec) sec.style.display = 'none';
      return;
    }
    const days = D.days;
    const GENDER_CFG = [
      { key: 'all',     label: 'All callers',   color: '#555' },
      { key: 'female',  label: 'Female',         color: F_COLOR },
      { key: 'male',    label: 'Male',           color: M_COLOR },
      { key: 'unknown', label: 'Unknown gender', color: '#aaa' },
    ];
    const legendCfg = {
      x: 1.03, xanchor: 'left', y: 1,
      bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ddd', borderwidth: 1,
    };

    // Counts chart
    Plotly.newPlot('c-dow-counts',
      GENDER_CFG.map(g => ({
        type: 'bar', name: g.label,
        x: days, y: D[g.key + '_counts'],
        marker: { color: g.color }, opacity: 0.85,
      })),
      Object.assign({}, LAYOUT_BASE, {
        title: { text: 'Caller count by day of week', font: { size: 13 } },
        barmode: 'group',
        xaxis: { title: 'Day of week', categoryorder: 'array', categoryarray: days },
        yaxis: { title: 'Number of callers' },
        legend: legendCfg,
        margin: { r: 160 },
        height: 340,
      }),
      { responsive: true, displayModeBar: true,
        modeBarButtonsToRemove: ['select2d','lasso2d'],
        toImageButtonOptions: { format: 'png', filename: 'cspan_day_of_week_counts', scale: 2 } });

    // Female fraction chart
    const n_labeled = D.total_labeled || days.map(() => 0);
    Plotly.newPlot('c-dow-fraction',
      [{
        type: 'bar', name: 'Female fraction',
        x: days, y: D.female_fraction,
        marker: { color: F_COLOR }, opacity: 0.85,
        text: days.map((d, i) => 'n=' + (n_labeled[i] || 0)),
        hovertemplate: '%{x}<br>Female fraction: %{y:.1%}<br>%{text}<extra></extra>',
      }, {
        type: 'scatter', mode: 'lines', name: '50% parity',
        x: days, y: days.map(() => 0.5),
        line: { color: '#ccc', dash: 'dot', width: 1.5 },
        hoverinfo: 'skip',
      }],
      Object.assign({}, LAYOUT_BASE, {
        title: { text: 'Female share of labeled callers by day', font: { size: 13 } },
        xaxis: { title: 'Day of week', categoryorder: 'array', categoryarray: days },
        yaxis: { title: 'Female fraction', tickformat: '.0%', range: [0, 1] },
        legend: legendCfg,
        margin: { r: 160 },
        height: 340,
      }),
      { responsive: true, displayModeBar: true,
        modeBarButtonsToRemove: ['select2d','lasso2d'],
        toImageButtonOptions: { format: 'png', filename: 'cspan_day_of_week_fraction', scale: 2 } });
  }

  // ── sentiment over time ──────────────────────────────────────────────────
  function initSentOverTime() {
    const S = DATA.sentOverTime;
    const sec = document.getElementById('sec-sent-ot');
    if (!S || !S.months || S.months.length === 0) {
      if (sec) sec.style.display = 'none';
      return;
    }
    const months = S.months;
    const GENDER_CFG = [
      { key: 'all',     label: 'All callers',   color: '#555',  dash: 'solid' },
      { key: 'female',  label: 'Female',         color: F_COLOR, dash: 'solid' },
      { key: 'male',    label: 'Male',           color: M_COLOR, dash: 'solid' },
      { key: 'unknown', label: 'Unknown gender', color: '#aaa',  dash: 'dot'   },
    ];
    const legendCfg = {
      x: 1.03, xanchor: 'left', y: 1,
      bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ddd', borderwidth: 1,
    };

    function makeTrace(g, field, yaxis) {
      return {
        type: 'scatter', mode: 'lines+markers',
        x: months, y: S[g.key + field],
        name: g.label,
        line: { color: g.color, width: 2, dash: g.dash },
        marker: { size: 5, color: g.color },
        connectgaps: false,
        yaxis: yaxis || 'y',
      };
    }

    Plotly.newPlot('c-sent-ot-compound',
      GENDER_CFG.map(g => makeTrace(g, '_compound')),
      Object.assign({}, LAYOUT_BASE, {
        title: { text: 'Mean sentiment (compound) per month', font: { size: 13 } },
        xaxis: { title: 'Month', tickangle: -35 },
        yaxis: { title: 'Mean compound score', zeroline: true,
                 zerolinecolor: '#ccc', zerolinewidth: 1.5 },
        legend: legendCfg,
        margin: { r: 160 },
        height: 360,
      }),
      { responsive: true, displayModeBar: true,
        modeBarButtonsToRemove: ['select2d','lasso2d'],
        toImageButtonOptions: { format: 'png', filename: 'cspan_sentiment_over_time', scale: 2 } });

    Plotly.newPlot('c-sent-ot-neg',
      GENDER_CFG.map(g => makeTrace(g, '_neg_pct')),
      Object.assign({}, LAYOUT_BASE, {
        title: { text: '% negative sentences per month', font: { size: 13 } },
        xaxis: { title: 'Month', tickangle: -35 },
        yaxis: { title: '% negative sentences', tickformat: '.0%' },
        legend: legendCfg,
        margin: { r: 160 },
        height: 360,
      }),
      { responsive: true, displayModeBar: true,
        modeBarButtonsToRemove: ['select2d','lasso2d'],
        toImageButtonOptions: { format: 'png', filename: 'cspan_negativity_over_time', scale: 2 } });
  }

  // ── sankey diagrams ──────────────────────────────────────────────────────
  function initSankey() {
    const SK = DATA.sankey;
    if (!SK) return;

    // Populate coverage counts
    ['sk-n-all','sk-n-all2'].forEach(function(id) {
      var el = document.getElementById(id);
      if (el) el.textContent = (SK.n_all || 0).toLocaleString();
    });
    ['sk-n-resp','sk-n-resp-2'].forEach(function(id) {
      var el = document.getElementById(id);
      if (el) el.textContent = (SK.n_resp || 0).toLocaleString();
    });

    function renderSankey(divId, data, title) {
      if (!data || !data.nodes || data.links.length === 0) {
        var el = document.getElementById(divId);
        if (el) el.innerHTML = '<p style="color:#888;padding:20px;">No data available for this diagram.</p>';
        return;
      }
      // Build link colors from source node color with opacity
      var linkColors = data.links.map(function(lk) {
        var hex = data.colors[lk.source] || '#aaa';
        var r = parseInt(hex.slice(1,3),16);
        var g = parseInt(hex.slice(3,5),16);
        var b = parseInt(hex.slice(5,7),16);
        return 'rgba(' + r + ',' + g + ',' + b + ',0.35)';
      });

      Plotly.newPlot(divId, [{
        type: 'sankey',
        orientation: 'h',
        arrangement: 'snap',
        node: {
          label:     data.nodes,
          color:     data.colors,
          pad:       18,
          thickness: 24,
          line:      { color: '#ffffff', width: 0.5 },
        },
        link: {
          source: data.links.map(function(l) { return l.source; }),
          target: data.links.map(function(l) { return l.target; }),
          value:  data.links.map(function(l) { return l.value;  }),
          color:  linkColors,
        },
      }], Object.assign({}, LAYOUT_BASE, {
        title:  { text: title, font: { size: 13 } },
        height: 420,
        margin: { l: 20, r: 20, t: 50, b: 20 },
        font:   { size: 11 },
      }), {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d','lasso2d'],
        toImageButtonOptions: { format: 'png', filename: divId, scale: 2 },
      });
    }

    var diagrams = [
      ['c-sankey-1', SK.sankey1, 'Party → Call Type → Caller Length  (all calls)'],
      ['c-sankey-2', SK.sankey2, 'Call Type → Response Length → Follow-up'],
      ['c-sankey-3', SK.sankey3, 'Caller Length → Call Type → Outcome'],
      ['c-sankey-4', SK.sankey4, 'Host Identity → Call Type → Outcome'],
      ['c-sankey-5', SK.sankey5, 'Day of Week → Call Type → Outcome'],
      ['c-sankey-6', SK.sankey6, 'Question Count → Caller Length → Outcome'],
      ['c-sankey-7', SK.sankey7, 'Caller Tone → Call Type → Outcome'],
      ['c-sankey-8', SK.sankey8, 'Hedging Language → Call Type → Outcome'],
      ['c-sankey-9', SK.sankey9, 'Vocabulary Diversity → Call Type → Outcome'],
    ];
    diagrams.forEach(function(d) { renderSankey(d[0], d[1], d[2]); });
  }

  // ── effective calls analysis ─────────────────────────────────────────────
  function initEffectiveCalls() {
    const EC = DATA.effectiveCalls;
    if (!EC || !EC.featureDiff || EC.featureDiff.length === 0) return;

    var el;
    el = document.getElementById('ec-n-subst'); if (el) el.textContent = EC.n_subst.toLocaleString();
    el = document.getElementById('ec-n-brief'); if (el) el.textContent = EC.n_brief.toLocaleString();

    const OUT_COLORS = { 'Substantive': '#2E7D32', 'Acknowledged': '#FFC107', 'Brief': '#EF5350' };
    const legendCfg = {
      x: 1.03, xanchor: 'left', y: 1,
      bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ddd', borderwidth: 1,
    };
    const imgOpts = { format: 'png', scale: 2 };

    // ── Feature difference chart (% diff: substantive vs brief) ──────────────
    const fd = EC.featureDiff.slice().sort(function(a,b){ return a.pct_diff - b.pct_diff; });
    const barColors = fd.map(function(d) { return d.pct_diff >= 0 ? '#2E7D32' : '#EF5350'; });
    Plotly.newPlot('c-ec-feature-diff', [{
      type: 'bar', orientation: 'h',
      x: fd.map(function(d){ return d.pct_diff; }),
      y: fd.map(function(d){ return d.metric; }),
      marker: { color: barColors },
      text: fd.map(function(d){ return (d.pct_diff>0?'+':'')+d.pct_diff+'%'; }),
      textposition: 'outside',
      hovertemplate: '%{y}<br>Diff: %{x:.1f}%<extra></extra>',
    }], Object.assign({}, LAYOUT_BASE, {
      title: { text: '% difference: Substantive vs Brief calls', font: { size: 13 } },
      xaxis: { title: '% difference (positive = more in substantive calls)',
               zeroline: true, zerolinecolor: '#333', zerolinewidth: 1.5 },
      yaxis: { automargin: true },
      margin: { l: 200, r: 80, t: 50, b: 60 },
      height: 420,
      shapes: [{ type:'line', x0:0, x1:0, y0:-0.5, y1:fd.length-0.5,
                 line:{ color:'#333', width:1.5 } }],
    }), { responsive: true, displayModeBar: true,
          modeBarButtonsToRemove:['select2d','lasso2d'],
          toImageButtonOptions: Object.assign({filename:'ec_feature_diff'}, imgOpts) });

    // ── Raw feature comparison (3 grouped bars) ───────────────────────────────
    const rawMetrics = EC.featureDiff.map(function(d){ return d.metric; });
    Plotly.newPlot('c-ec-feature-raw', [
      { type:'bar', name:'Substantive', x: rawMetrics,
        y: EC.featureDiff.map(function(d){ return d.subst_mean; }),
        marker:{ color: OUT_COLORS['Substantive'] }, opacity: 0.85 },
      { type:'bar', name:'Acknowledged', x: rawMetrics,
        y: EC.featureDiff.map(function(d){ return d.ack_mean; }),
        marker:{ color: OUT_COLORS['Acknowledged'] }, opacity: 0.85 },
      { type:'bar', name:'Brief', x: rawMetrics,
        y: EC.featureDiff.map(function(d){ return d.brief_mean; }),
        marker:{ color: OUT_COLORS['Brief'] }, opacity: 0.85 },
    ], Object.assign({}, LAYOUT_BASE, {
      title: { text: 'Mean feature value by outcome tier', font: { size: 13 } },
      barmode: 'group',
      xaxis: { tickangle: -35, automargin: true },
      yaxis: { title: 'Mean value' },
      legend: legendCfg,
      margin: { r: 160 },
      height: 400,
    }), { responsive: true, displayModeBar: true,
          modeBarButtonsToRemove:['select2d','lasso2d'],
          toImageButtonOptions: Object.assign({filename:'ec_feature_raw'}, imgOpts) });

    // ── % Substantive by host ─────────────────────────────────────────────────
    if (EC.byHost && EC.byHost.length > 0) {
      const hosts = EC.byHost.map(function(h){ return h.host + ' (n='+h.n+')'; });
      Plotly.newPlot('c-ec-by-host', [
        { type:'bar', name:'Substantive', x:hosts,
          y: EC.byHost.map(function(h){ return h.subst_pct; }),
          marker:{ color: OUT_COLORS['Substantive'] }, opacity: 0.85 },
        { type:'bar', name:'Acknowledged', x:hosts,
          y: EC.byHost.map(function(h){ return h.ack_pct; }),
          marker:{ color: OUT_COLORS['Acknowledged'] }, opacity: 0.85 },
        { type:'bar', name:'Brief', x:hosts,
          y: EC.byHost.map(function(h){ return h.brief_pct; }),
          marker:{ color: OUT_COLORS['Brief'] }, opacity: 0.85 },
      ], Object.assign({}, LAYOUT_BASE, {
        title: { text: '% outcome by host', font: { size: 13 } },
        barmode: 'stack',
        xaxis: { tickangle: -20, automargin: true },
        yaxis: { title: '% of calls', tickformat: 'd' },
        legend: legendCfg,
        margin: { r: 160 },
        height: 360,
      }), { responsive: true, displayModeBar: true,
            modeBarButtonsToRemove:['select2d','lasso2d'],
            toImageButtonOptions: Object.assign({filename:'ec_by_host'}, imgOpts) });
    }

    // ── % Substantive by day ──────────────────────────────────────────────────
    if (EC.byDay && EC.byDay.length > 0) {
      const days = EC.byDay.map(function(d){ return d.day + '\n(n='+d.n+')'; });
      Plotly.newPlot('c-ec-by-day', [
        { type:'bar', name:'Substantive', x:days,
          y: EC.byDay.map(function(d){ return d.subst_pct; }),
          marker:{ color: OUT_COLORS['Substantive'] }, opacity: 0.85 },
        { type:'bar', name:'Acknowledged', x:days,
          y: EC.byDay.map(function(d){ return d.ack_pct; }),
          marker:{ color: OUT_COLORS['Acknowledged'] }, opacity: 0.85 },
        { type:'bar', name:'Brief', x:days,
          y: EC.byDay.map(function(d){ return d.brief_pct; }),
          marker:{ color: OUT_COLORS['Brief'] }, opacity: 0.85 },
      ], Object.assign({}, LAYOUT_BASE, {
        title: { text: '% outcome by broadcast day', font: { size: 13 } },
        barmode: 'stack',
        xaxis: { automargin: true },
        yaxis: { title: '% of calls', tickformat: 'd' },
        legend: legendCfg,
        margin: { r: 160 },
        height: 360,
      }), { responsive: true, displayModeBar: true,
            modeBarButtonsToRemove:['select2d','lasso2d'],
            toImageButtonOptions: Object.assign({filename:'ec_by_day'}, imgOpts) });
    }

    // ── Question count sweet spot ─────────────────────────────────────────────
    if (EC.byQcount && EC.byQcount.length > 0) {
      const qlabels = EC.byQcount.map(function(q){ return q.q_count+' question'+(q.q_count==='1'?'':q.q_count==='0'?'s':q.q_count==='2'?'s':'s+')+'\n(n='+q.n+')'; });
      Plotly.newPlot('c-ec-qcount', [
        { type:'bar', name:'Substantive', x:qlabels,
          y: EC.byQcount.map(function(q){ return q.subst_pct; }),
          marker:{ color: OUT_COLORS['Substantive'] }, opacity: 0.85 },
        { type:'bar', name:'Acknowledged', x:qlabels,
          y: EC.byQcount.map(function(q){ return q.ack_pct; }),
          marker:{ color: OUT_COLORS['Acknowledged'] }, opacity: 0.85 },
        { type:'bar', name:'Brief', x:qlabels,
          y: EC.byQcount.map(function(q){ return q.brief_pct; }),
          marker:{ color: OUT_COLORS['Brief'] }, opacity: 0.85 },
      ], Object.assign({}, LAYOUT_BASE, {
        title: { text: '% outcome by question count — the sweet spot', font: { size: 13 } },
        barmode: 'stack',
        xaxis: { automargin: true },
        yaxis: { title: '% of calls', tickformat: 'd' },
        legend: legendCfg,
        margin: { r: 160 },
        height: 360,
      }), { responsive: true, displayModeBar: true,
            modeBarButtonsToRemove:['select2d','lasso2d'],
            toImageButtonOptions: Object.assign({filename:'ec_qcount'}, imgOpts) });
    }

    // ── Opener word rates ─────────────────────────────────────────────────────
    if (EC.byOpener && EC.byOpener.length > 0) {
      const openers = EC.byOpener.map(function(o){ return '"'+o.word+'" (n='+o.total+')'; });
      Plotly.newPlot('c-ec-openers', [
        { type:'bar', name:'Rate in Substantive calls (%)',
          x: openers,
          y: EC.byOpener.map(function(o){ return o.subst_rate; }),
          marker:{ color: OUT_COLORS['Substantive'] }, opacity: 0.85 },
        { type:'bar', name:'Rate in Brief calls (%)',
          x: openers,
          y: EC.byOpener.map(function(o){ return o.brief_rate; }),
          marker:{ color: OUT_COLORS['Brief'] }, opacity: 0.85 },
      ], Object.assign({}, LAYOUT_BASE, {
        title: { text: 'First meaningful word: rate in substantive vs brief calls', font: { size: 13 } },
        barmode: 'group',
        xaxis: { tickangle: -35, automargin: true },
        yaxis: { title: '% of calls in that outcome group' },
        legend: legendCfg,
        margin: { r: 160 },
        height: 380,
      }), { responsive: true, displayModeBar: true,
            modeBarButtonsToRemove:['select2d','lasso2d'],
            toImageButtonOptions: Object.assign({filename:'ec_openers'}, imgOpts) });
    }
  }

  // ── auto figure numbers ──────────────────────────────────────────────────
  function autoFigureNumbers() {
    var n = 0;
    document.querySelectorAll('.chart-card > div[id]').forEach(function(el) {
      n++;
      var label = document.createElement('div');
      label.className = 'fig-label';
      label.textContent = 'Figure ' + n;
      el.parentNode.insertBefore(label, el);
    });
  }

  var _initDone = {};   // tracks which pages have had their charts initialized

  function initPageCharts(pageId) {
    if (_initDone[pageId]) return;
    _initDone[pageId] = true;

    if (pageId === 'overview') {
      summaryCards();
      autoFigureNumbers();

    } else if (pageId === 'who') {
      violin('c-wordcount',      'word_count',             'Word count per turn',    'Words');
      violin('c-words-per-sent', 'avg_words_per_sentence', 'Avg words per sentence', 'Words');
      initWcHistogram();
      initWcDist();
      initCallsOverTime();
      initDayOfWeek();
      initGeo();
      autoFigureNumbers();

    } else if (pageId === 'questions') {
      violin('c-q-ratio-violin', 'question_ratio', 'Fraction of sentences that are questions', 'Ratio');
      partyBar('c-party');
      initInteractions();
      initSentiment();
      initSentOverTime();
      initSankey();
      initEffectiveCalls();
      autoFigureNumbers();

    } else if (pageId === 'style') {
      violin('c-hedge-violin', 'hedge_rate', 'Hedging language rate', 'Rate (per word)');
      groupedBar('c-key-metrics-bar', 'barMetrics',  'Key metrics by gender  (error bars = SE)',  'Mean value');
      groupedBar('c-style-bar',       'styleMetrics', 'Style metrics by gender  (error bars = SE)', 'Rate per word');
      openerBar('c-openers');
      initWcScatter();
      initQScatter();
      initResponsivenessCharts();
      autoFigureNumbers();

    } else if (pageId === 'tables') {
      initWordFreqTable();
      if (TABLES) {
        initTable();
        initSentTable();
        initRespTable();
      }
      // else: tables.json hasn't loaded yet — the "Loading…" placeholder remains
      // and _initDone['tables'] stays true; when tables.json arrives we re-init below
      autoFigureNumbers();
    }
  }

  function switchPage(pageId) {
    document.querySelectorAll('.nav-page').forEach(function(p) { p.classList.remove('active'); });
    var pg = document.getElementById('page-' + pageId);
    if (pg) pg.classList.add('active');
    document.querySelectorAll('.nav-item').forEach(function(n) { n.classList.remove('active'); });
    var ni = document.querySelector('[data-page="' + pageId + '"]');
    if (ni) ni.classList.add('active');
    _currentPage = pageId;
    if (DATA) initPageCharts(pageId);
  }

  var _currentPage = 'overview';

  // Wire nav clicks
  document.querySelectorAll('.nav-item').forEach(function(item) {
    item.addEventListener('click', function(e) {
      e.preventDefault();
      switchPage(this.dataset.page);
      window.location.hash = this.dataset.page;
    });
  });

  document.addEventListener('DOMContentLoaded', function () {
    // Determine start page from URL hash
    var startPage = (window.location.hash || '').replace('#', '') || 'overview';
    var validPages = ['overview','who','questions','style','tables'];
    if (validPages.indexOf(startPage) < 0) startPage = 'overview';

    // Show correct nav-page and nav-item for start page (CSS default hides all)
    document.querySelectorAll('.nav-page').forEach(function(p) { p.classList.remove('active'); });
    var startEl = document.getElementById('page-' + startPage);
    if (startEl) startEl.classList.add('active');
    document.querySelectorAll('.nav-item').forEach(function(n) { n.classList.remove('active'); });
    var startNav = document.querySelector('[data-page="' + startPage + '"]');
    if (startNav) startNav.classList.add('active');
    _currentPage = startPage;

    // Fetch data.json (charts) and tables.json (tables) in parallel
    fetch('data.json')
      .then(function(r) {
        if (!r.ok) throw new Error('data.json fetch failed: ' + r.status);
        return r.json();
      })
      .then(function(json) {
        DATA = json;
        initPageCharts(_currentPage);
      })
      .catch(function(err) {
        document.body.insertAdjacentHTML('afterbegin',
          '<div style="background:#ffebee;color:#b71c1c;padding:16px 24px;font-family:monospace;font-size:14px;margin-left:200px;">' +
          '<strong>Could not load data.json:</strong> ' + err.message +
          '<br>If running locally, use <code>make serve</code> instead of opening the file directly.' +
          '</div>');
      });

    // Load table rows in background — big file, not blocking
    ['caller-table','sent-table','resp-table'].forEach(function(id) {
      var el = document.getElementById(id);
      if (el) el.innerHTML = '<p style="color:#888;font-style:italic;padding:12px 0;">Loading table data…</p>';
    });
    fetch('tables.json')
      .then(function(r) {
        if (!r.ok) throw new Error('tables.json fetch failed: ' + r.status);
        return r.json();
      })
      .then(function(tbl) {
        TABLES = tbl;
        // If user is on the tables page and initPageCharts already ran, re-init tables
        if (_currentPage === 'tables') {
          initTable();
          initSentTable();
          initRespTable();
        }
      })
      .catch(function(err) {
        ['caller-table','sent-table','resp-table'].forEach(function(id) {
          var el = document.getElementById(id);
          if (el) el.innerHTML = '<p style="color:#b71c1c;">Could not load table data: ' + err.message + '</p>';
        });
      });
  });
})();
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate docs/index.html + docs/style.css analysing C-SPAN caller gender differences."
    )
    parser.add_argument("--csv",    default="results/scraped/cspan_callers.csv",
                        help="Input CSV from scrape_cspan.py")
    parser.add_argument("--output", default="docs/index.html",
                        help="Output HTML file (default: docs/index.html)")
    args = parser.parse_args()

    css_path = os.path.join(os.path.dirname(args.output), "style.css")

    print(f"Loading {args.csv} ...")
    df = load_and_enrich(args.csv)
    n_f = (df["gender"] == "female").sum()
    n_m = (df["gender"] == "male").sum()
    n_u = (df["gender"] == "unknown").sum()
    print(f"  {len(df)} total turns  ({n_f} female, {n_m} male, {n_u} unknown)")

    if n_f + n_m == 0:
        sys.exit(
            f"No male/female rows found in {args.csv}.\n"
            "Run scrape_cspan.py first to generate caller data."
        )

    payload, tables_payload = compute_payload(df)

    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)

    data_path   = os.path.join(out_dir, "data.json")
    tables_path = os.path.join(out_dir, "tables.json")

    print(f"Writing {data_path} ...")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Writing {tables_path} ...")
    with open(tables_path, "w", encoding="utf-8") as f:
        json.dump(tables_payload, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Writing {css_path} ...")
    with open(css_path, "w", encoding="utf-8") as f:
        f.write(CSS)

    print(f"Writing {args.output} ...")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)

    data_mb   = os.path.getsize(data_path)   / 1e6
    tables_mb = os.path.getsize(tables_path) / 1e6
    html_mb   = os.path.getsize(args.output) / 1e6
    print(f"Done.  HTML: {html_mb:.1f} MB  |  data.json: {data_mb:.1f} MB  |  tables.json: {tables_mb:.1f} MB")
    print(f"Open {args.output} in your browser (requires a local server — use 'make serve').")


if __name__ == "__main__":
    main()
