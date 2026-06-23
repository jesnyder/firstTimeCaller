C-SPAN Washington Journal Caller Analysis
==========================================

Project overview
----------------
This project scrapes caller transcripts from Washington Journal (C-SPAN) episodes
hosted at fudgie.org, computes linguistic and demographic metrics, and produces
a single-page interactive website (docs/index.html).

Quick start
-----------
1. Install dependencies:
       make install

2. Scrape transcripts (append mode — skips already-scraped episodes):
       make fudgie          # 50 episodes
       make fudgie-big      # 200 episodes (recommended for first run)

3. Build the website:
       make website

4. Preview in browser:
       make serve
   Then open http://localhost:8000 in your browser.

To do all of the above in one step:
       make all

Running the scraper
-------------------
The primary scraper for fudgie.org transcripts is:
    user_provided/python/scrape_fudgie.py

It runs in append mode by default — it reads the existing CSV, finds which
episode IDs are already present, and only fetches new ones.  Safe to re-run.

    python3 user_provided/python/scrape_fudgie.py \
        --episodes 200 \
        --append \
        --output results/scraped/cspan_callers.csv

Output CSV: results/scraped/cspan_callers.csv
Output HTML: docs/index.html
Output CSS:  docs/style.css

Re-building the website without re-scraping
-------------------------------------------
    make website

This reads the existing CSV and regenerates docs/index.html + docs/style.css.
Useful after editing analyze_website.py without wanting to re-scrape.

Known hosts (manually verified)
--------------------------------
The following Washington Journal hosts are recognised by the scraper and
assigned gender labels used in the host-interaction analyses:

  Pedro Echevarria  — male
  John McArdle      — male       [NOTE: Python .title() gives "John Mcardle" not
  Bill Scanlan      — male        "John McArdle"; scraper uses token-based lookup
  Steve Scully      — male        to avoid this bug — do not revert to .title()]
  Rob Harleston     — male
  Khalil Garriott   — male
  Greta Brawner     — female
  Kimberly Adams    — female
  Libby Casey       — female
  Susan Swain       — female
  Jeslyn Rollins    — female
  Chloe Veltman     — female

If a new host appears and is labeled "unknown gender", add their first and last
name tokens to _HOST_LOOKUP in scrape_fudgie.py (see lines ~59-76) and their
full lowercased name to KNOWN_HOSTS (lines ~78-82).

Website display standards (mandatory — enforce in analyze_website.py)
----------------------------------------------------------------------
These rules apply to every plot and every table on the website.

PLOTS (all built with Plotly.js):
  - Every plot must have its interactive Plotly toolbar visible so the user
    can pan, zoom, select, and download the chart.  Pass displayModeBar: true
    and set toImageButtonOptions to provide a labelled PNG download.
  - Every plot must have a legend positioned OFF the plot, on the far right,
    so it never obscures data.  Use:
        legend: { x: 1.03, xanchor: 'left', y: 1,
                  bgcolor: 'rgba(255,255,255,0.85)',
                  bordercolor: '#ddd', borderwidth: 1 }
    and set margin: { r: 160 } (or wider) to leave room for the legend.
  - Every series in a multi-series plot must be independently toggleable by
    clicking or double-clicking its legend entry (this is Plotly's default
    behaviour — do not disable it).
  - Every plot is assigned an auto-generated figure number ("Figure N")
    displayed in small caps above the chart, assigned by DOM order via
    autoFigureNumbers() in the page JS.

TABLES (all built with Tabulator):
  - Every table must be sortable: clicking any column header sorts ascending /
    descending.  Use Tabulator's default sort behaviour.
  - Every table must be filterable: use headerFilter: 'input' on text columns
    and headerFilter: 'number' / 'tickCross' on numeric/boolean columns.
  - Every table must be paginated: set pagination: 'local', paginationSize: 50
    (or 25 for wide tables).
  - Every table must have a "Download CSV" button that calls
    table.download('csv', '<filename>.csv').
  - Every table must show a row-count label that updates with
    table.getDataCount('active') when data is filtered.

METHODS BLOCK (peer-reviewed quality):
Every plot on the website includes a "methods" block below it.  These blocks
must be written as if preparing a methods section for a peer-reviewed journal:
  - Define every term used in the plot
  - State mathematical equations where applicable
  - Cite peer-reviewed references with DOI links
  - Describe how each metric is computed from the raw transcript text

This policy is enforced in analyze_website.py.  Do not abbreviate or simplify
these descriptions — they are the primary documentation of the analysis.

File layout
-----------
  README.txt                        This file
  Makefile                          Build targets
  requirements.txt                  Python dependencies
  user_provided/python/
    scrape_fudgie.py                Primary scraper (fudgie.org)
    scrape_cspan.py                 Legacy YouTube scraper
    analyze_website.py              Generates docs/index.html + docs/style.css
  results/scraped/
    cspan_callers.csv               Scraped caller turns (output of scraper)
  docs/
    index.html                      Generated website (open this in browser)
    style.css                       Generated stylesheet

Dependencies (see requirements.txt)
------------------------------------
  pandas              — data wrangling
  gender-guesser      — first-name gender inference
  vaderSentiment      — sentence-level sentiment scoring (Hutto & Gilbert 2014)
  scipy               — Mann-Whitney U tests
  requests            — HTTP scraping
  beautifulsoup4      — HTML parsing
  (browser-side)
  Plotly.js           — interactive charts with download
  Tabulator           — sortable/filterable data tables
  Leaflet.js          — geographic bubble map
