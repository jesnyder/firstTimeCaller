PYTHON      := python3
SCRAPER     := user_provided/python/scrape_cspan.py
FUDGIE      := user_provided/python/scrape_fudgie.py
SITE_SCRIPT := user_provided/python/analyze_website.py
OUTPUT_CSV  := results/scraped/cspan_callers.csv
OUTPUT_HTML := docs/index.html
OUTPUT_CSS  := docs/style.css
EPISODES    := 15
# Override SOURCE on the CLI if needed:
#   make scrape SOURCE="https://www.youtube.com/playlist?list=XXXXX"
SOURCE      :=

DEPS_STAMP  := .deps-installed

.PHONY: all update scrape scrape-big scrape-more scrape-debug \
        fudgie fudgie-big website serve install clean clean-all help

# ── default ───────────────────────────────────────────────────────────────────
## Append new episodes from both YouTube and fudgie.org, then rebuild the site
all: scrape-more fudgie website

# ── dependency installation ───────────────────────────────────────────────────
install: $(DEPS_STAMP)

$(DEPS_STAMP): requirements.txt
	$(PYTHON) -m pip install -r requirements.txt
	@touch $(DEPS_STAMP)
	@echo "Dependencies installed."

# ── scrape targets ────────────────────────────────────────────────────────────

## Fresh scrape — REPLACES the CSV (default 15 episodes); use EPISODES=N to override
scrape: $(DEPS_STAMP)
	@mkdir -p results/scraped
	$(PYTHON) $(SCRAPER) \
	  --episodes $(EPISODES) \
	  --output   $(OUTPUT_CSV) \
	  $(if $(SOURCE),--source "$(SOURCE)",)

## Fresh larger scrape — REPLACES the CSV (50 episodes by default)
scrape-big: $(DEPS_STAMP)
	@mkdir -p results/scraped
	$(PYTHON) $(SCRAPER) \
	  --episodes 50 \
	  --output   $(OUTPUT_CSV) \
	  $(if $(SOURCE),--source "$(SOURCE)",)

## Append mode — skips episodes already in the CSV, adds new ones
scrape-more: $(DEPS_STAMP)
	@mkdir -p results/scraped
	$(PYTHON) $(SCRAPER) \
	  --episodes 100 \
	  --append \
	  --output  $(OUTPUT_CSV) \
	  $(if $(SOURCE),--source "$(SOURCE)",)

## Debug — show the first 20 speaker turns for one episode without saving
scrape-debug: $(DEPS_STAMP)
	$(PYTHON) $(SCRAPER) \
	  --episodes 1 \
	  --output   /tmp/cspan_debug_$$(date +%s).csv \
	  --debug \
	  $(if $(SOURCE),--source "$(SOURCE)",)

## Scrape fudgie.org transcripts (50 episodes, append mode) — much richer than YouTube
fudgie: $(DEPS_STAMP)
	@mkdir -p results/scraped
	$(PYTHON) $(FUDGIE) --episodes 50 --append --output $(OUTPUT_CSV)

## Scrape fudgie.org transcripts (200 episodes, append mode)
fudgie-big: $(DEPS_STAMP)
	@mkdir -p results/scraped
	$(PYTHON) $(FUDGIE) --episodes 200 --append --output $(OUTPUT_CSV)

# ── website ───────────────────────────────────────────────────────────────────

## Build docs/index.html + docs/style.css from the scraped CSV
website: $(DEPS_STAMP)
	@if [ ! -f $(OUTPUT_CSV) ]; then \
	  echo "ERROR: $(OUTPUT_CSV) not found. Run 'make scrape' first."; exit 1; \
	fi
	@mkdir -p docs
	$(PYTHON) $(SITE_SCRIPT) \
	  --csv    $(OUTPUT_CSV) \
	  --output $(OUTPUT_HTML)

## Serve docs/ at http://localhost:8000
serve:
	@if [ ! -f $(OUTPUT_HTML) ]; then \
	  echo "ERROR: $(OUTPUT_HTML) not found. Run 'make website' first."; exit 1; \
	fi
	$(PYTHON) -m http.server 8000 --bind 127.0.0.1 --directory docs

# ── convenience combos ────────────────────────────────────────────────────────

## Alias for 'all': append new episodes from both sources then rebuild site
update: scrape-more fudgie website

## Fresh full run: replace CSV then rebuild site
fresh: scrape website

## Fresh big run: 50-episode replace then rebuild site
fresh-big: scrape-big website

# ── housekeeping ──────────────────────────────────────────────────────────────

## Delete generated CSV and HTML (keeps installed deps)
clean:
	rm -f $(OUTPUT_CSV) $(OUTPUT_HTML) $(OUTPUT_CSS)

## Delete generated files AND force dep reinstall next run
clean-all: clean
	rm -f $(DEPS_STAMP)

help:
	@echo ""
	@echo "Usage:  make [target] [VAR=value ...]"
	@echo ""
	@echo "Main targets:"
	@echo "  all / update      Append from YouTube + fudgie.org, rebuild site  (default)"
	@echo "  fudgie-big        Append 200 fudgie.org episodes then rebuild site"
	@echo "  serve             Serve docs/ at http://localhost:8000"
	@echo ""
	@echo "Individual steps:"
	@echo "  scrape-more       Append new YouTube episodes (skips already-scraped)"
	@echo "  fudgie            Append 50 fudgie.org episodes"
	@echo "  fudgie-big        Append 200 fudgie.org episodes"
	@echo "  scrape            Fresh YouTube scrape → $(OUTPUT_CSV)  (EPISODES=N)"
	@echo "  scrape-debug      Show raw speaker turns for 1 YouTube episode"
	@echo "  website           Rebuild $(OUTPUT_HTML) from existing CSV"
	@echo "  install           pip install all dependencies (auto-runs on first use)"
	@echo "  clean             Delete CSV + HTML"
	@echo "  clean-all         Delete CSV + HTML + deps stamp"
	@echo ""
	@echo "Overrides:"
	@echo "  make scrape       EPISODES=30"
	@echo "  make scrape       SOURCE=\"https://www.youtube.com/playlist?list=XXXXX\""
	@echo "  make fresh-big    OUTPUT_CSV=my.csv OUTPUT_HTML=docs/report.html"
