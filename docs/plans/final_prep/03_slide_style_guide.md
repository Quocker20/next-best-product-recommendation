# Final deck: slide style guide (anti "AI look")

Prepared 2026-10-11 for the final presentation (report due 2026-10-21). Scope: how a coding agent
should generate the final HTML/PDF deck so it reads as a research talk, not a generated template.
Staged skills (not installed) are in `docs/plans/final_prep/skills_staging/`.

---

## (a) Sources, ranked

Stars and dates are from the GitHub API on 2026-10-11.

| # | Source | Type, license | Verdict |
|---|---|---|---|
| 1 | **Michael Alley, assertion-evidence approach** (Penn State). https://writing.engr.psu.edu/research.html ; book *The Craft of Scientific Presentations* (Springer, 2nd ed. 2013) | Human guide, research-backed | The method to build the deck around. Every slide gets a sentence headline and visual evidence, not bullets. Controlled studies by Alley et al. found better understanding and retention. |
| 2 | **Naegle, "Ten simple rules for effective presentation slides"**, PLOS Comput Biol 17(12) e1009554, 2021. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8638955/ | Human guide, CC-BY | Short, practical rules for one slide at a time: one idea per slide, then rules for design and for rehearsal. Good to hand to every team member. |
| 3 | **antislop-decks** (jhuse25/anti-slop-kit). https://github.com/jhuse25/anti-slop-kit | Agent skill, MIT, **0 stars**, created 2026-08 | The only skill found that targets *decks* and lists the AI tells. It covers assertion-evidence, a titles-only test, a word ceiling, chart data-ink and speaker notes. Provenance is weak, so use it as a checklist. Some of its defaults (a "signature element", archetype rotation, VARIANCE 6) suit pitch decks more than a 12-minute research talk. **Staged.** |
| 4 | **Anthropic `frontend-design`** (anthropics/skills, ~180k stars on the repo; skill updated 2026-09-03). https://github.com/anthropics/skills/tree/main/skills/frontend-design | Agent skill, Apache-2.0 | Written for web UI, but it names the exact traits of AI-generated design: the SaaS card kit, all-caps eyebrows, middle-dot meta strings, single-word accents, numbered markers. Its "plan tokens, review against the brief, then build" process transfers directly. **Staged.** |
| 5 | **impeccable** (pbakaus/impeccable, 79,313 stars, pushed 2026-10-09). https://github.com/pbakaus/impeccable | Agent skill, Apache-2.0 | `reference/craft-floor.md` is the best ban list found: eyebrow labels, the hero-metric template, colored `border-left` callouts, identical cards, monospace as costume, emoji icons. The full workflow is web/app oriented and runs a downloaded binary (see warnings). **Staged partially, text only.** |
| 6 | **Rougier, Droettboom, Bourne, "Ten simple rules for better figures"**, PLOS Comput Biol 2014. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4161295/ | Human guide, CC-BY, made with matplotlib | For the charts: don't trust the defaults, use color with intent, no chartjunk. Directly applicable to `results/figures/`. |
| 7 | **Edward Tufte**: data-ink ratio and chartjunk (*The Visual Display of Quantitative Information*), and *The Cognitive Style of PowerPoint* essay. https://www.edwardtufte.com | Human guide (books) | The basis for "remove non-data ink" and his critique of bullet outlines. Use it for the charts and tables, not as a reason to drop slides altogether. |
| 8 | **Better Posters** (Zen Faulkes). https://betterposters.blogspot.com | Human guide (blog) | Advice on hierarchy, reading order and white space. Useful for the dense appendix slides. |
| 9 | academic-pptx-skill (Gabberflast, 1,145 stars). https://github.com/Gabberflast/academic-pptx-skill | Agent skill, **license unclear** | Strong academic rules: action titles, ghost-deck test, one exhibit per slide, end on conclusions, at most ~40 body words, three colors. **Not staged:** the repo LICENSE says MIT, but `SKILL.md` says "license: Proprietary. LICENSE.txt has complete terms", and the skill depends on Anthropic's source-available pptx skill. Its ideas are paraphrased in (b). |
| 10 | frontend-slides (zarazhangrui, 30,410 stars, MIT). https://github.com/zarazhangrui/frontend-slides | Agent skill, MIT | Popular single-file HTML deck generator, and its fixed 16:9 stage rules are sound. Its aesthetic works against a research talk: atmospheric gradient backgrounds, "animation-rich", presets with 01/02 section numbers. It also offers to deploy to a public URL with Vercel. **Not staged.** |
| 11 | Anthropic `pptx` | Agent skill, **source-available, restrictive** | Its LICENSE forbids keeping copies outside Anthropic services or making derivative works. **Not staged.** It already ships as the built-in `anthropic-skills:pptx` if a .pptx is ever needed. |
| 12 | Anthropic `theme-factory`, `canvas-design`, `brand-guidelines` | Agent skills, Apache-2.0 | Not useful here. theme-factory offers 10 generic preset themes ("Ocean Depths", "Tech Innovation"), which is template look by design. canvas-design makes art posters as PNG. brand-guidelines applies *Anthropic's* brand, which would be wrong on this deck. |

Toolchain references: reveal.js (72,395 stars, MIT), Slidev (49,001 stars, MIT), Marp CLI
(3,861 stars, MIT), python-pptx (MIT).

---

## (b) Do / don't checklist for the agent building the final deck

### Narrative and titles
- **DO** write every content slide title as one complete-sentence claim that the slide's evidence proves (assertion-evidence). Use no more than ~15 words; if it needs more, it is two slides.
- **DO** run the titles-only test before building. List the titles in order in the build script's docstring or a plan file; read on their own, they must tell the argument: data problem → protocol → main model vs ItemKNN and logistic regression → slices (L, season, trip context, known/new destination) → accuracy vs diversity → limitations → conclusion.
- **DO** end on a conclusions slide that stays up during Q&A, with the limitations on it: `hotel_cluster` is a proxy, one dataset, offline only, two simple baselines. **DON'T** end on "Cảm ơn / Q&A", and **DON'T** use an agenda slide that only lists sections.
- **DO** put methods detail (formulas, feature tables, cleaning rules) in an appendix after the conclusion slide.
- **DON'T** write topic titles ("Kết quả", "Kết luận, giới hạn và bước tiếp") or titles that pack two claims with a semicolon.
- **DON'T** use filler words in either language: leverage, seamless, robust, delve, unlock, elevate, "comprehensive", "toàn diện", "vượt trội", "mạnh mẽ". State the number instead.

### Words and density
- **DO** cap body text at **~40 words per content slide**, excluding the title, axis labels and the source line. The week-4 deck has 82–178 words on main slides (see (c)).
- **DO** move explanation to speaker notes (see below). A slide line should be the label for something visible, not a paragraph.
- **DO** round numbers to what the comparison needs: NDCG@5 to 3 decimals, shares to whole percents, and mean ± std only where seeds matter. **DON'T** put 4-decimal signed deltas in titles.
- **DON'T** use more than 3–4 bullets on a slide. Use bullets only for parallel items, never as a chopped paragraph.

### Typography
- **DO** use one family with full Vietnamese diacritics. Keep Be Vietnam Pro (already loaded) or switch to Source Sans 3 / IBM Plex Sans; either works with Vietnamese. Show hierarchy with size and weight only.
- **DO** use this scale on the 1280×720 stage: title 32–36 px / 600–700; body 20–22 px; chart labels and table cells ≥ 16 px; source line 13–14 px, the only text allowed below 16 px.
- **DO** use sentence case everywhere and tabular numerals in tables (`font-variant-numeric: tabular-nums`).
- **DON'T** use ALL-CAPS tracked labels: no eyebrow/kicker above titles, no uppercase table headers.
- **DON'T** join meta strings with middle dots ("Tuần 4 · Làm sạch dữ liệu"), and **DON'T** color a single word in a headline.
- **DON'T** set formulas in a monospace face. Use the body face with italic variables, or KaTeX if the formulas get complex.

### Palette (one, shared by the deck and matplotlib)
Use a white background (projectors wash out cream and tints) and the colors below. Most of them already appear in the matplotlib figures (`scripts/expedia_eda_clean.py`, `scripts/week3_report_figures.py`), so the deck and the figures finally match. Contrast ratios on white were computed with the WCAG formula.

| Token | Hex | Contrast on white | Use |
|---|---|---|---|
| `--bg` | `#ffffff` | – | slide background |
| `--ink` | `#20242a` | 15.6 | titles, body |
| `--muted` | `#5c6166` | 6.3 | secondary text, source line |
| `--rule` | `#d9dadb` | – | hairlines, gridlines, table rules |
| `--main` | `#3b6ea5` | 5.3 | **main model (SMLP4Rec + query token + sameDest) only** |
| `--main-dark` | `#2f5a88` | 7.2 | text that must carry the main color |
| `--contrast` | `#c27c3f` | 3.4 | second series (e.g. logistic regression, "cold"); large marks only, never small text |
| `--context` | `#a8a79f` | 2.4 | baselines, earlier variants, background data (marks only) |
| `--alert` | `#a63d32` | 6.3 | leak, defect, a drop; at most one use per slide |

Rules:
- **DO** give each model the same color on every slide and figure: main model blue, baselines gray, a second highlighted series ochre.
- **DO** use color to point at the finding. Everything else is gray.
- **DON'T** use the Tailwind defaults now in the deck (`#1e3a8a`, `#2563eb`, `#0f766e`, `#dc2626`, slate `#94a3b8`/`#e2e8f0`), indigo (`#6366f1`/`#eef2ff`), purple, or gradients of any kind.

### Layout grid
- **DO** use a 12-column grid on the 1280×720 stage: 64 px outer margins, 24 px gutters, and one baseline unit of 8 px for all spacing (8/16/24/32/48).
- **DO** keep the title at a fixed position on every slide, top-left aligned. Left-align body text and tables, and right-align numbers.
- **DO** use 2–3 slide layouts for real reasons: (1) full-width evidence (chart or diagram) under the title; (2) evidence on 8 columns with a ≤ 3-line annotation on 4 columns; (3) a table. Pick a layout by content type, not to look varied. A research talk does not need the antislop-decks "VARIANCE 6".
- **DON'T** put content in rounded cards or panels by default. Separate with white space and a hairline at most. Use no box-shadow, no `border-radius` > 4 px, and no colored left borders on callouts.
- **DON'T** use the "big number + small label" stat stack as the default right column, three-column feature grids, or icons in circles. Use numbered circles only for a real sequence (the pipeline steps).
- **DON'T** repeat the same footer sentence on every slide. Show a short source line only where the slide shows data (e.g. "Expedia, warm test rows, n = …"); the protocol already requires it.

### Charts
- **DO** render charts as inline SVG from the results JSON (current approach) or embed the matplotlib PNG/SVG from `results/figures/{en,vi}/`. Either way, use the palette above and the same font. Set `font.family` in matplotlib to the deck face and export SVG (or PNG at ≥ 200 dpi) at the size it will appear.
- **DO** label series directly at the end of the line or bar instead of using a legend. Use light gridlines (`--rule`) or none, no top/right spines, and a bar baseline at zero.
- **DO** annotate the one finding on the chart (an arrow or label at the bar or gap). The title says what to see; the annotation shows where.
- **DO** show seed spread as dots and whiskers (the dot plot already does this) and the paired-bootstrap CI in the annotation, not in a separate box.
- **DON'T** round bar corners (`rx="3"`), use 3D, use more than 4 colors, or show value labels on every bar when an axis would do.
- **DO** state the dataset and row set (warm / cold / all) under every result chart or table. The protocol in `docs/eval_protocol.md` requires it.

### Tables
- **DO** use horizontal rules only: top and bottom of the table plus one under the header. No cell borders, no zebra fill. Right-align numbers, use tabular numerals, and use the same decimals within a column.
- **DO** highlight at most one row (the main model) with weight or the `--main-dark` color, not a tinted background.
- **DO** keep a table under ~6 rows × 6 columns on a content slide; anything bigger goes in the appendix.

### Model architecture diagrams
- **DO** draw the architecture as one hand-authored SVG in the same palette. Gray boxes for inputs, the main color only for the new parts (query token, sameDest), thin `--muted` arrows, labels in the body face at ≥ 16 px.
- **DO** build it up over 2–3 slides when it is explained step by step: history only → + query token → + sameDest. Use the same diagram each time with the new part colored; the slide title states what the step adds.
- **DO** stay faithful to the SMLP4Rec paper's block structure (the three mixers: sequence, channel/feature, embedding) and label it "research adaptation".
- **DON'T** use colored pastel boxes for every block, rounded pill shapes, icon glyphs inside boxes, or gradient arrows.

### Speaker notes
- **DO** give every slide notes: 60–120 words in Vietnamese, covering what to say, the exact number with its source file under `results/`, and the likely question.
- **DO** keep notes in the build script next to the slide, rendered as `<aside class="notes">` (hidden on screen, printable as a separate notes PDF), or use reveal.js's built-in notes if the toolchain moves there (see (d)).

### Process the agent must follow
1. Write the token plan (palette, type scale, grid, the 2–3 layouts) and the titles-only outline. Show them to the user before building (frontend-design's "plan, review against the brief, build" step).
2. Build. Read every number from `results/` (no typed statistics, per CLAUDE.md §5).
3. Render a PDF with headless Chrome and look at every page image. Check: overflow, text < 16 px, words per slide, and any banned pattern from this list. Fix everything in one batch and re-check once.
4. Cross-check each cited number against its JSON (CLAUDE.md §9).

---

## (c) Critique of the current week-4 deck

Files: `scripts/week4_clean_slides.py` (builder) and the CSS / diagram it pulls from
`scripts/week2_methodology_slides.py`. Output: `reports/summary/week4_rebuild/week4_clean_slides.html`.

**What already works (keep):** most content titles are claims with numbers (assertion style), not
topics. Every number is read from JSON. Baselines are drawn gray and the model of interest in
color (e.g. `hbar(..., [LGRAY, LGRAY, GRAY, BLUE, TEAL])`, week4 l. 300). The dot plot shows
seeds rather than only means. The background is white, and the font supports Vietnamese.

**AI-looking or weak patterns found:**

| # | Pattern | Where |
|---|---|---|
| 1 | **Tailwind default palette**, the most recognisable generated-UI look: navy-900 / blue-600 / teal-700 / amber-700 / red-600 plus slate grays. It also differs from the matplotlib figures (`#3b6ea5`, `#c27c3f`, `#7a9e7e`, `#898781`), so embedded figures and SVG charts don't match. | week4 l. 30 (`NAVY, BLUE, TEAL, …`); week2 CSS `:root` l. 542; SVG text and gridline colors `#334155`, `#94a3b8`, `#e2e8f0` throughout l. 94–170 |
| 2 | **Indigo boxes** (`#6366f1` / `#eef2ff`) in the architecture diagram and flow boxes: the "purple/indigo" tell. | week2 `architecture_svg` l. 127–240; week2 CSS `.fbox.model` l. 601, `.m2` l. 621 |
| 3 | **Same layout on almost every slide.** `lay()` (chart left + 2–3 big-number stats right) followed by `take()` (callout with a navy left border) is used on 9 of the 12 main slides. That is the "hero-metric template", repeated. | `lay` / `take` defined l. 207–214; called at l. 299, 321, 354, 370, 395, 421, 446, 474, 500 |
| 4 | **Big-number stat stacks** (36 px colored numbers over 14.5 px gray sentences) carry most of the text, so each slide has 3 competing focal points beside the chart. | `.stat b` l. 41–43, `.stat b.s` l. 57 |
| 5 | **Colored left-border callouts and cover bar**: `.take` (4 px navy), `.cover` (6 px navy), `.card.good`, `.sum-main`, `.wk`. Explicitly flagged by impeccable's craft floor. | week4 l. 44; week2 CSS l. 578, 632, 640 |
| 6 | **ALL-CAPS tracked eyebrows** above every title (`.meta`, "1 · Nhắc lại tuần trước") and on the cover (`.kicker`, "TUẦN 4 · LÀM SẠCH DỮ LIỆU"). Uppercase table headers too. Both are on frontend-design's list of tells, and impeccable bans eyebrows outright. | week2 CSS `.meta` l. 549, `.kicker` l. 641, `.tbl th` l. 585; emitted at week4 l. 269, 640 |
| 7 | **Middle-dot meta strings** everywhere: eyebrows, cover lead ("Next-Best-Product Recommendation · Vinpearl × GSM · …"), footer. | week4 l. 269–271, 640–642 |
| 8 | **Rounded panels and boxes** (`border-radius` 12–14 px on `.panel`, `.card`, `.kpi`, `.eq`, `.ag`, `.th-card`; pill `.tag` and `.cover-tags` at 20 px), plus rounded bar ends `rx="3"`. This is the "SaaS card kit" look. | week2 CSS l. 556–580, 610, 618, 623, 646; week4 `.eq` l. 47; bars l. 97, 115, 138 |
| 9 | **Equations in a monospace box** (Consolas on a tinted rounded panel): monospace used as a "technical" costume. | week4 `.eq` l. 47–48; used at l. 562, 582, 584 |
| 10 | **Too many words.** Visible words per slide (excluding SVG), counted with a script over the HTML: main slides 2–12 have **82–134**, the conclusion has **178**, the appendix **207–324**. The target is ≤ ~40. | slides 2–16 of the HTML |
| 11 | **Text too small to project**: chart labels 11–12.5 px, `.sm` 13 px, `.tbl.tight` 13 px, footer 11.5 px. | week4 l. 52, 58, 96, 130, 159; week2 CSS `.ft` l. 553 |
| 12 | **Titles overloaded with precision** (e.g. "-0,0165 NDCG@5", "+0,0156 NDCG@5", "-0,0002") and some with two claims joined by ";" or ":". The conclusion slide has a topic title ("Kết luận, giới hạn và bước tiếp") and two columns of bullets. | titles at l. 298, 394, 420, 529; conclusion l. 544–556 |
| 13 | **Repeated chrome**: the same source footer and "Tuần 4" tag on every slide. It is noise on slides without data, and it isn't specific (no row set) on slides with data. | week4 l. 640–642 |
| 14 | **No speaker notes**, so the explanation has to live on the slide, which drives #10. | entire builder |
| 15 | **Fragile style reuse** (not visual, but it blocks fixing the above): CSS, JS and the diagram are pulled out of the week-2 script by regex and `exec()`. A style change for one deck silently changes or breaks the others. | week4 l. 20, 28–35 |

---

## (d) Toolchain recommendation

| Option | New dependency | Pros | Cons |
|---|---|---|---|
| **Keep the Python → single-HTML builder (recommended)** | none | Numbers flow straight from `results/*.json`, satisfying CLAUDE.md §5. Inline SVG charts already work, PDF prints via headless Chrome, and the team knows it. | Speaker view and notes have to be added (small: `<aside class="notes">` plus a `?notes` print mode). The style needs a proper shared module. |
| reveal.js (one CDN script and CSS, MIT) | light, no build step | Built-in speaker view with notes and timer (`S` key), `?print-pdf` export, fragments. The Python builder can keep generating the `<section>`s. | Its default themes look like reveal.js; you must replace the theme CSS with the tokens above. Fragments invite gratuitous builds. Needs a pinned CDN version and internet at talk time, or a vendored file. |
| Marp (Markdown → HTML/PDF/PPTX via `marp-cli`, Node) | Node + npm package | Markdown is pleasant to edit by hand, and it exports PPTX. | The numbers would have to be templated into Markdown by Python anyway. Custom SVG charts and layouts are awkward, and it's a new runtime for the team. |
| Slidev (Vue + Vite) | heavy (Node, Vite, Vue) | Powerful, good presenter mode. | Heavy dependency for one talk; not justified. |
| python-pptx | pip package (MIT) | Gives an editable .pptx if the mentor or university requires one. | Weaker typographic control and no inline SVG (charts become images). Anthropic's pptx skill is source-available and restrictive, so don't vendor it. |

**Recommendation.** Keep the HTML builder. Make three changes:
1. Create `scripts/deck_style.py` (or `src/nbp/report/`) holding the palette tokens, type scale, base CSS and the diagram function. Every deck script imports it, which removes the regex/`exec` reuse (critique #15). Make `results/figures` scripts import the same palette so the charts match.
2. Add speaker notes per slide and a `--notes` print mode that writes a notes PDF.
3. Rewrite `lay()` / `take()` into the 2–3 layouts in (b) and delete the card, kpi, pill and eyebrow CSS.

Switch to reveal.js only if the presenter view during the talk matters; the generated sections stay the same. Use python-pptx only if a .pptx is required. (New folders or modules outside the CLAUDE.md §7 table need that section updated in the same change.)

---

## (e) Installing a staged skill (only after you approve)

Staged copies are not active: Claude Code loads skills only from `.claude/skills/<name>/SKILL.md`
(project) or `~/.claude/skills/<name>/SKILL.md` (personal). Review `SKILL.md` and `SOURCE.txt`
first. Prefer the **project** location so the skill applies only to this repo; it can be committed
or gitignored as you like.

Git Bash, from the repo root:

```bash
# antislop-decks (MIT)
mkdir -p .claude/skills/antislop-decks
cp docs/plans/final_prep/skills_staging/antislop-decks/{SKILL.md,LICENSE} .claude/skills/antislop-decks/

# frontend-design (Apache-2.0)
mkdir -p .claude/skills/frontend-design
cp docs/plans/final_prep/skills_staging/frontend-design/{SKILL.md,LICENSE.txt} .claude/skills/frontend-design/

# impeccable (Apache-2.0), reference-only copy: first remove the pre-approved Bash tools
mkdir -p .claude/skills/impeccable/reference
cp docs/plans/final_prep/skills_staging/impeccable/{SKILL.md,LICENSE} .claude/skills/impeccable/
cp docs/plans/final_prep/skills_staging/impeccable/reference/*.md .claude/skills/impeccable/reference/
# then edit .claude/skills/impeccable/SKILL.md and delete the `allowed-tools:` block (3 lines)
```

PowerShell equivalent (one skill shown):

```powershell
New-Item -ItemType Directory -Force .claude\skills\antislop-decks | Out-Null
Copy-Item docs\plans\final_prep\skills_staging\antislop-decks\SKILL.md, docs\plans\final_prep\skills_staging\antislop-decks\LICENSE .claude\skills\antislop-decks\
```

For a personal install, replace `.claude/skills` with `~/.claude/skills` (PowerShell:
`$HOME\.claude\skills`). Restart the Claude Code session or start a new one, then check that the
skill appears in the skill list. Remove it by deleting its folder.

Notes per skill:
- **impeccable:** the staged copy has no `scripts/`, so its Setup step (which runs a downloaded binary) falls back to "launcher unavailable". That is intended. Don't install the full skill with `npx skills add pbakaus/impeccable` without reviewing what it downloads. Its "go all out, dream big" stance and web modes don't fit a research talk; tell the agent to use only `reference/craft-floor.md` as a ban list.
- **antislop-decks:** it refers to companion skills (`antislop-web`, `antislop-docs`) that are not staged. Ignore those references, and override its "signature element" and VARIANCE defaults with section (b) of this guide.
- **frontend-design:** web-oriented. Tell the agent it is building a projected 16:9 research deck and that this guide's palette and grid take precedence.
- A simpler option than installing anything: tell the agent "follow `docs/plans/final_prep/03_slide_style_guide.md`" when it builds the deck. Section (b) already holds the parts of these skills that matter here.
