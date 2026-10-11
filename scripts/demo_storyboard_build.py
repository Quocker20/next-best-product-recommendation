"""Build the self-contained demo storyboard page from the precomputed JSON.

Input:  results/week5_final/demo_storyboard.json (scripts/demo_storyboard_export.py)
Output: results/week5_final/demo_storyboard.html, one file with inline CSS and JS, no network
        access; open it by double-click. It is a demo asset (not a report), hence results/.

The page shows, for each curated test booking, what the system knows (earlier bookings, current
search) and the top 5 clusters, built up in three steps the presenter clicks through:
original SMLP4Rec (history only) -> + query token -> + sameDest, with the rank of the booked
cluster at each step next to the two baselines. A what-if toggle (key W) swaps in one
precomputed variant of the search. A last frame repeats the test Recall@5 of the five models.
Keys: right / space = next, left = back, W = what-if, 1-4 = jump to a booking, S = summary.

Language: `--lang en` (default) or `--lang vi` (writes demo_storyboard_vi.html). All wording
lives in `texts(d, lang)` (page UI strings are passed to the page as JSON); numbers come only
from the JSON.

Usage (any Python 3.11, stdlib only):
    python scripts/demo_storyboard_build.py [--lang en|vi]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "week5_final" / "demo_storyboard.json"

FOOTER = {
    "en": (
        "Simulation on public Expedia data; hotel_cluster = anonymous proxy for room/package "
        "category; offline evaluation"
    ),
    "vi": (
        "Mô phỏng trên dữ liệu Expedia công khai; hotel_cluster là proxy ẩn danh cho hạng "
        "phòng/gói; đánh giá offline"
    ),
}
SIDE = {
    "en": (
        "Real test bookings picked for illustration; D is a miss. Baselines: notebook 07b, "
        "trained on the same rows as our model."
    ),
    "vi": (
        "Lượt đặt thật của tập test, chọn để minh họa; D là chỗ sai. Baseline: notebook 07b, "
        "train trên cùng các dòng với mô hình chính."
    ),
}

CSS = """
:root{--bg:#ffffff;--ink:#20242a;--muted:#5c6166;--rule:#d9dadb;--main:#3b6ea5;
--main-dark:#2f5a88;--contrast:#c27c3f;--context:#a8a79f;--alert:#a63d32}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#ffffff;color:var(--ink);overflow:hidden}
body{font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif;font-size:20px;line-height:1.35;
font-variant-numeric:tabular-nums}
#wrap{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:#ffffff}
#stage{width:1280px;height:720px;position:relative;padding:24px 64px 0 64px;transform-origin:center}
h1{font-size:32px;font-weight:600;line-height:1.2}
.sub{color:var(--muted);font-size:18px;margin-top:4px}
.bar{display:flex;align-items:center;margin-top:12px;padding-bottom:8px;border-bottom:1px solid var(--rule)}
.steps{display:flex;gap:28px;font-size:20px;color:var(--context);white-space:nowrap}
.steps span.done{color:var(--muted)}
.steps span.on{color:var(--main-dark);font-weight:600}
.cols{display:grid;grid-template-columns:500px 1fr;gap:56px;margin-top:12px}
h2{font-size:20px;font-weight:600;margin-bottom:6px}
.k{color:var(--muted)}
table{border-collapse:collapse;width:100%}
td,th{text-align:left;padding:2px 10px 2px 0;font-size:18px;font-weight:400;vertical-align:baseline}
th{color:var(--muted);border-bottom:1px solid var(--rule)}
td.n,th.n{text-align:right}
tr.same td{color:var(--main-dark);font-weight:600}
.search{margin-bottom:14px}
.search td{font-size:19px;padding:0 10px 0 0}
.hist td{padding:1px 10px 1px 0}
.hist{table-layout:fixed}
.search.off td{color:var(--context)}
.search.off h2{color:var(--context)}
.offnote{color:var(--muted);font-size:18px;font-weight:400;margin-top:2px}
td.chg{color:var(--alert);font-weight:600}
.expl{font-size:20px;margin-bottom:6px;min-height:54px}
ol.top{list-style:none}
ol.top li{display:grid;grid-template-columns:36px 150px 1fr;align-items:baseline;
border-bottom:1px solid var(--rule);padding:1px 0;font-size:24px}
ol.top li .r{color:var(--muted);font-size:20px}
ol.top li .c{font-weight:400}
ol.top li .m{font-size:18px;color:var(--muted);word-spacing:2px}
ol.top li .sd{color:var(--main-dark);font-weight:600}
ol.top li.hit .c{color:var(--main-dark);font-weight:700}
ol.top li.hit .b{color:var(--main-dark);font-weight:700}
.ranks{margin-top:12px}
.ranks td{font-size:20px;padding:1px 10px 1px 0}
.ranks tr.cur td{color:var(--main-dark);font-weight:600}
.ranks tr.base td{color:var(--muted)}
.ranks tr.future td.n{color:var(--context)}
.note{font-size:18px;color:var(--ink);margin-top:10px}
.foot{position:absolute;left:64px;right:64px;bottom:12px;font-size:18px;line-height:1.3;color:var(--muted);
border-top:1px solid var(--rule);padding-top:6px}
.nav{display:flex;gap:8px;flex:none;margin-left:auto;align-items:center;font-size:18px;color:var(--muted);white-space:nowrap}
.sub.note3{color:var(--ink)}
button{font:inherit;font-size:18px;color:var(--ink);background:#ffffff;border:1px solid var(--rule);
border-radius:3px;padding:4px 12px;cursor:pointer}
button:hover{border-color:var(--muted)}
button.on{border-color:var(--main-dark);color:var(--main-dark);font-weight:600}
button:disabled{color:var(--context);cursor:default}
.sum td,.sum th{font-size:20px;padding:6px 24px 6px 0}
.sum tr.main td{color:var(--main-dark);font-weight:600}
.sum tr.base td{color:var(--muted)}
.sum{margin-top:24px;width:auto}
.src{font-size:18px;color:var(--muted);margin-top:16px;max-width:1100px}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const T = JSON.parse(document.getElementById('text').textContent);
const U = T.ui;
const fmt = (s, o) => s.replace(/\{(\w+)\}/g, (_, k) => o[k]);
const B = D.bookings;
const frames = [];
B.forEach((b, i) => { for (let s = 1; s <= 3; s++) frames.push({b: i, s: s}); });
frames.push({summary: true});
let cur = 0, whatif = false;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const STEP_KEYS = ['plain', 'query', 'samedest'];
const STEP_NAMES = T.step_names;

function fit() {
  const s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
  $('stage').style.transform = 'scale(' + s + ')';
}

function party(sr) {
  const a = sr.adults, c = sr.children, r = sr.rooms;
  return `${a} ${a === 1 ? U.adult1 : U.adultN}, ${c} ${c === 1 ? U.child1 : U.childN}, ${r} ${r === 1 ? U.room1 : U.roomN}`;
}

function searchBlock(b, step, wi) {
  const sr = Object.assign({}, b.search);
  const chg = wi && wi.changed ? wi.changed : {};
  for (const k in chg) sr[k] = chg[k];
  const pc = ('adults' in chg || 'children' in chg || 'rooms' in chg) ? ' class=chg' : '';
  const known = b.steps.samedest.known_destination;
  const off = step === 1;
  let h = `<div class="search${off ? ' off' : ''}"><h2>${esc(U.current_search)}${off ? `&emsp;<span class=offnote>${esc(T.search_unused)}</span>` : ''}</h2>`;
  h += '<table>';
  const hasHist = b.n_prior && !(wi && wi.no_history);
  h += `<tr><td class=k>${esc(U.destination)}</td><td>${sr.dest}${hasHist ? (known ? U.known_dest : U.new_dest) : ''}</td></tr>`;
  h += `<tr><td class=k>${esc(U.checkin)}</td><td>${esc(sr.ci)}, ${sr.nights} ${sr.nights === 1 ? U.night1 : U.nightN}</td></tr>`;
  h += `<tr><td class=k>${esc(U.party)}</td><td${pc}>${party(sr)}</td></tr>`;
  h += `<tr><td class=k>${esc(U.package)}</td><td>${sr.package ? U.yes : U.no} (${U.channel} ${sr.channel}${sr.mobile ? U.mobile : ''})</td></tr>`;
  return h + '</table></div>';
}

function historyBlock(b, step, wi) {
  const MAXR = 10;
  let h = `<h2>${esc(U.earlier)}</h2>`;
  if (wi && wi.no_history) return h + `<div class=offnote>${esc(T.history_removed)}</div>`;
  if (!b.history.length) return h + `<div class=offnote>${esc(T.no_history)}</div>`;
  const dest = b.search.dest;
  const mark = step === 3 && b.steps.samedest.w_samedest > 0;
  const rows = b.history.slice(-MAXR);
  h += '<table class=hist><colgroup><col style="width:130px"><col style="width:120px"><col style="width:90px"><col></colgroup>';
  h += `<tr><th>${esc(U.booked_on)}</th><th class=n>${esc(U.destination)}</th><th class=n>${esc(U.cluster)}</th><th></th></tr>`;
  if (b.history.length > MAXR) h += `<tr><td class=k colspan=4>${fmt(U.not_shown, {n: b.history.length - MAXR})}</td></tr>`;
  for (const r of rows) {
    const same = mark && r.dest === dest;
    h += `<tr${same ? ' class=same' : ''}><td>${esc(r.booked_on)}</td><td class=n>${r.dest}</td><td class=n>${r.cluster}</td><td>${same ? esc(U.same_dest) : ''}</td></tr>`;
  }
  return h + '</table>';
}

function topBlock(b, st, step) {
  const key = STEP_KEYS[step - 1];
  const s = st[key];
  const boosted = new Set(st.samedest.boosted_clusters);
  let h = '<ol class=top>';
  for (let i = 0; i < s.top.length; i++) {
    const t = s.top[i];
    const hit = t.cluster === b.truth;
    let m = '';
    if (step > 1 && t.prev_rank !== i + 1) m = fmt(t.prev_rank > i + 1 ? U.up_from : U.down_from, {n: t.prev_rank});
    let tag = '';
    if (step === 3 && boosted.has(t.cluster)) tag = `<span class=sd>${esc(U.booked_here)}</span>`;
    const parts = [hit ? `<span class=b>${esc(U.booked)}</span>` : '', tag, m ? `<span>${m}</span>` : ''].filter(x => x);
    h += `<li class="${hit ? 'hit' : ''}"><span class=r>${i + 1}</span><span class=c>${esc(U.cluster_lc)} ${t.cluster}</span><span class=m>${parts.join('&emsp;')}</span></li>`;
  }
  return h + '</ol>';
}

function rankBlock(st, step) {
  let h = `<table class=ranks><tr><th>${esc(T.rank_head)}</th><th class=n>${esc(U.rank)}</th></tr>`;
  STEP_KEYS.forEach((k, i) => {
    const cls = i + 1 === step ? 'cur' : (i + 1 > step ? 'future' : '');
    const v = i + 1 <= step ? st[k].truth_rank : '';
    h += `<tr class="${cls}"><td>${esc(STEP_NAMES[i])}</td><td class=n>${v}</td></tr>`;
  });
  h += `<tr class=base><td>${esc(T.lr_name)}</td><td class=n>${st.baselines.lr.truth_rank}</td></tr>`;
  h += `<tr class=base><td>${esc(T.knn_name)}</td><td class=n>${st.baselines.knn.truth_rank}</td></tr>`;
  return h + '</table>';
}

function explanation(b, st, step, wi) {
  if (step === 1) return b.history.length && !(wi && wi.no_history) ? T.expl[0] : T.expl_cold;
  if (step === 2) return T.expl[1];
  return st.samedest.w_samedest > 0 ? T.expl[2] : T.expl_newdest;
}

function renderBooking(f) {
  const b = B[f.b], step = f.s;
  const wi = whatif && b.whatif ? b.whatif : null;
  const st = wi ? wi.steps : b.steps;
  $('title').textContent = `${b.key}. ${T.roles[b.key] || b.role}`;
  let sub = fmt(U.real_booking, {user: b.user_id, truth: b.truth, on: b.booked_on});
  if (wi) sub = fmt(U.whatif_sub, {label: T.whatif_labels[b.key] || wi.label, truth: b.truth});
  const n3 = step === 3 && !wi && T.notes[b.key];
  if (n3) sub = T.notes[b.key];
  $('sub').textContent = sub;
  $('sub').classList.toggle('note3', !!n3);
  $('steps').innerHTML = T.bar_names.map((n, i) =>
    `<span class="${i + 1 === step ? 'on' : (i + 1 < step ? 'done' : '')}">${i + 1}&ensp;${esc(n)}</span>`).join('');
  $('body').innerHTML = `<div class=cols><div>${searchBlock(b, step, wi)}${historyBlock(b, step, wi)}</div>
    <div><h2>${esc(U.top5)}</h2><div class=expl>${esc(explanation(b, st, step, wi))}</div>
    ${topBlock(b, st, step)}${rankBlock(st, step)}</div></div>`;
  $('wi').disabled = !b.whatif;
  $('wi').classList.toggle('on', !!wi);
  $('wi').textContent = b.whatif ? (wi ? U.back_real : U.whatif) : U.no_whatif;
}

function renderSummary() {
  const S = D.summary;
  $('title').textContent = T.summary_title;
  $('sub').classList.remove('note3');
  $('sub').textContent = fmt(U.summary_sub, {split: S.split, seed: S.seed});
  $('steps').innerHTML = '';
  const rows = [['plain', STEP_NAMES[0], ''], ['query', STEP_NAMES[1], ''], ['samedest', STEP_NAMES[2], 'main'],
                ['lr', T.lr_name_short, 'base'], ['knn', T.knn_name_short, 'base']];
  const m = S.models;
  let h = `<table class=sum><tr><th>${esc(U.model)}</th><th class=n>${fmt(U.recall_warm, {n: m.samedest.warm.n.toLocaleString('en-US')})}</th>
    <th class=n>${fmt(U.recall_cold, {n: m.samedest.cold.n.toLocaleString('en-US')})}</th></tr>`;
  for (const [k, name, cls] of rows)
    h += `<tr class="${cls}"><td>${esc(name)}</td><td class=n>${m[k].warm['recall@5'].toFixed(3)}</td><td class=n>${m[k].cold['recall@5'].toFixed(3)}</td></tr>`;
  h += '</table>';
  h += `<div class=src>${esc(T.summary_note)} ${fmt(U.source_line, {source: esc(S.source), base: esc(S.baselines_from)})}</div>`;
  $('body').innerHTML = h;
  $('wi').disabled = true; $('wi').classList.remove('on'); $('wi').textContent = U.no_whatif;
}

function render() {
  const f = frames[cur];
  if (f.summary) renderSummary(); else renderBooking(f);
  $('pos').textContent = f.summary ? U.summary : fmt(U.pos, {b: f.b + 1, n: B.length, s: f.s});
  $('prev').disabled = cur === 0;
  $('next').disabled = cur === frames.length - 1;
}

function go(i) {
  i = Math.max(0, Math.min(frames.length - 1, i));
  if (frames[i].b !== frames[cur].b) whatif = false;
  cur = i; render();
}

document.addEventListener('keydown', e => {
  if (['ArrowRight', 'PageDown', ' ', 'Enter'].includes(e.key)) { go(cur + 1); e.preventDefault(); }
  else if (['ArrowLeft', 'PageUp', 'Backspace'].includes(e.key)) { go(cur - 1); e.preventDefault(); }
  else if (e.key === 'w' || e.key === 'W') toggleWhatif();
  else if (e.key === 'Home') go(0);
  else if (e.key === 's' || e.key === 'S' || e.key === 'End') go(frames.length - 1);
  else if (/^[1-9]$/.test(e.key) && +e.key <= B.length) go((+e.key - 1) * 3);
});

function toggleWhatif() {
  const f = frames[cur];
  if (f.summary || !B[f.b].whatif) return;
  whatif = !whatif; render();
}

$('prev').onclick = () => go(cur - 1);
$('next').onclick = () => go(cur + 1);
$('wi').onclick = toggleWhatif;
window.addEventListener('resize', fit);
fit(); render();
"""


def texts(d: dict, lang: str = "en") -> dict:
    """Page wording; numbers are filled from the JSON, never typed here."""
    vi = lang == "vi"
    by = {b["key"]: b for b in d["bookings"]}
    notes = {}
    a = by.get("A")
    if a:
        pr, qr = a["steps"]["plain"]["truth_rank"], a["steps"]["query"]["truth_rank"]
        notes["A"] = (
            f"Không có lịch sử, mô hình gốc hiển thị cùng một danh sách cho mọi khách mới "
            f"(cluster được đặt ở hạng {pr}); chỉ riêng tìm kiếm đã đưa nó lên hạng {qr}."
            if vi
            else f"Without history the original model shows every new customer the same list "
            f"(booked cluster at rank {pr}); the search alone moves it to rank {qr}."
        )
    b = by.get("B")
    if b and b["history"]:
        first = next((h for h in b["history"] if h["dest"] == b["search"]["dest"]), None)
        if first:
            sr = b["steps"]["samedest"]["truth_rank"]
            notes["B"] = (
                f"Khách đã đặt cluster {first['cluster']} tại điểm đến {first['dest']} "
                f"vào {first['booked_on']}; sameDest đưa nó trở lại hạng {sr}."
                if vi
                else f"The customer booked cluster {first['cluster']} at destination "
                f"{first['dest']} on {first['booked_on']}; sameDest brings it back to rank {sr}."
            )
    c = by.get("C")
    if c:
        cr = c["steps"]["query"]["truth_rank"]
        notes["C"] = (
            f"Chưa có chuyến đi nào đến điểm đến này: chỉ riêng tìm kiếm của gia đình đã xếp "
            f"cluster {c['truth']} ở hạng {cr}."
            if vi
            else f"No earlier trip to this destination: the family search alone ranks cluster "
            f"{c['truth']} at {cr}."
        )
    dd = by.get("D")
    if dd:
        top = dd["steps"]["samedest"]["top"][0]["cluster"]
        pr, kr = dd["steps"]["plain"]["truth_rank"], dd["steps"]["baselines"]["knn"]["truth_rank"]
        notes["D"] = (
            f"Sai: cluster {dd['truth']} lặp lại một chuyến đi khác; các mô hình chỉ dùng lịch sử "
            f"xếp nó ở hạng {pr} và {kr}, mô hình của chúng tôi đặt {top} lên đầu."
            if vi
            else f"Miss: cluster {dd['truth']} repeats another trip; history-only models rank it "
            f"{pr} and {kr}, ours puts {top} first."
        )
    ui_en = {
        "adult1": "adult", "adultN": "adults", "child1": "child", "childN": "children",
        "room1": "room", "roomN": "rooms", "night1": "night", "nightN": "nights",
        "current_search": "Current search", "destination": "Destination", "checkin": "Check-in",
        "party": "Party", "package": "Package", "yes": "yes", "no": "no",
        "known_dest": " (booked here before)", "new_dest": " (new for this customer)",
        "channel": "sales channel", "mobile": ", mobile",
        "earlier": "Earlier bookings", "booked_on": "Booked on", "cluster": "Cluster",
        "not_shown": "{n} earlier not shown", "same_dest": "same destination",
        "up_from": "up from {n}", "down_from": "down from {n}",
        "booked_here": "booked here before", "booked": "booked", "cluster_lc": "cluster",
        "rank": "Rank", "top5": "Top 5 clusters",
        "real_booking": "Real test booking: user {user}, booked cluster {truth} on {on}",
        "whatif_sub": "What-if: {label}. Booked cluster stays {truth}.",
        "back_real": "Back to the real search", "whatif": "What-if", "no_whatif": "No what-if",
        "summary_sub": "Expedia, {split} split, seed {seed}, full ranking over 100 clusters",
        "model": "Model",
        "recall_warm": "Recall@5, returning (n = {n})",
        "recall_cold": "Recall@5, first booking (n = {n})",
        "source_line": "Source: {source} (baselines there: {base}).",
        "summary": "Summary", "pos": "Booking {b}/{n}, step {s}/3",
        "back": "Back", "next": "Next",
    }  # fmt: skip
    ui_vi = {
        "adult1": "người lớn", "adultN": "người lớn", "child1": "trẻ em", "childN": "trẻ em",
        "room1": "phòng", "roomN": "phòng", "night1": "đêm", "nightN": "đêm",
        "current_search": "Tìm kiếm hiện tại", "destination": "Điểm đến", "checkin": "Check-in",
        "party": "Nhóm khách", "package": "Gói", "yes": "có", "no": "không",
        "known_dest": " (đã đặt ở đây trước đó)", "new_dest": " (mới với khách này)",
        "channel": "kênh bán", "mobile": ", mobile",
        "earlier": "Các lượt đặt trước", "booked_on": "Ngày đặt", "cluster": "Cluster",
        "not_shown": "{n} lượt cũ hơn không hiển thị", "same_dest": "cùng điểm đến",
        "up_from": "tăng từ {n}", "down_from": "giảm từ {n}",
        "booked_here": "đã đặt ở đây trước đó", "booked": "được đặt", "cluster_lc": "cluster",
        "rank": "Hạng", "top5": "Top 5 cluster",
        "real_booking": "Lượt đặt thật trong tập test: người dùng {user}, đặt cluster {truth} ngày {on}",
        "whatif_sub": "Giả định: {label}. Cluster được đặt vẫn là {truth}.",
        "back_real": "Về tìm kiếm thật", "whatif": "Giả định", "no_whatif": "Không giả định",
        "summary_sub": "Expedia, tập {split}, seed {seed}, xếp hạng đầy đủ trên 100 cluster",
        "model": "Mô hình",
        "recall_warm": "Recall@5, khách quay lại (n = {n})",
        "recall_cold": "Recall@5, lượt đặt đầu tiên (n = {n})",
        "source_line": "Nguồn: {source} (baseline ở đó: {base}).",
        "summary": "Tóm tắt", "pos": "Lượt {b}/{n}, bước {s}/3",
        "back": "Quay lại", "next": "Tiếp",
    }  # fmt: skip
    if not vi:
        return {
            "ui": ui_en,
            "roles": {},
            "whatif_labels": {},
            "bar_names": ["Original SMLP4Rec", "+ query token", "+ sameDest"],
            "step_names": [
                "Original SMLP4Rec (history only)",
                "+ query token",
                "+ sameDest (our model)",
            ],
            "expl": [
                "The original model reads only the earlier clusters; it cannot see the search.",
                "The current search becomes one extra token that the model mixes with the history.",
                "sameDest adds clusters this customer booked before at the searched destination.",
            ],
            "expl_cold": "No earlier bookings: the original model gives every new customer the same list.",
            "expl_newdest": "Destination is new for this customer, so sameDest changes nothing here.",
            "search_unused": "not used by the original model",
            "no_history": "none: this is the customer's first booking",
            "history_removed": "removed for this what-if (treated as a new customer)",
            "rank_head": "Rank of the booked cluster (of 100)",
            "lr_name": "Logistic regression (search, no history)",
            "knn_name": "ItemKNN (history only)",
            "lr_name_short": "Logistic regression",
            "knn_name_short": "ItemKNN",
            "notes": notes,
            "summary_title": "On all test bookings, the query token adds most; sameDest helps returning customers",
            "summary_note": (
                "The four bookings were picked for illustration; these are the measured "
                "numbers on every test booking."
            ),
        }
    return {
        "ui": ui_vi,
        "roles": {
            "A": "Khách mới: lượt đặt đầu tiên, chưa có lịch sử",
            "B": "Khách quay lại, trở về điểm đến của một chuyến đi trước",
            "C": "Gia đình có ba trẻ em, check-in vào tháng Mười Hai",
            "D": "Chỗ mô hình sai: khách lặp lại một lựa chọn cũ",
        },
        "whatif_labels": {
            "A": "cùng tìm kiếm nhưng của một gia đình (2 người lớn, 2 trẻ em)",
            "B": "cùng tìm kiếm, bỏ các lượt đặt trước",
            "C": "cùng tìm kiếm nhưng của một người đi một mình",
        },
        "bar_names": ["SMLP4Rec gốc", "+ query token", "+ sameDest"],
        "step_names": [
            "SMLP4Rec gốc (chỉ lịch sử)",
            "+ query token",
            "+ sameDest (mô hình chính)",
        ],
        "expl": [
            "Mô hình gốc chỉ đọc các cluster trước đó; nó không thấy tìm kiếm.",
            "Tìm kiếm hiện tại trở thành một token bổ sung, mô hình trộn với lịch sử.",
            "sameDest cộng các cluster khách đã đặt trước đây tại điểm đến đang tìm.",
        ],
        "expl_cold": "Chưa có lượt đặt trước: mô hình gốc đưa cùng một danh sách cho mọi khách mới.",
        "expl_newdest": "Điểm đến mới với khách này, nên sameDest không thay đổi gì ở đây.",
        "search_unused": "mô hình gốc không dùng",
        "no_history": "không có: đây là lượt đặt đầu tiên của khách",
        "history_removed": "đã bỏ trong kịch bản này (coi như khách mới)",
        "rank_head": "Hạng của cluster được đặt (trong 100)",
        "lr_name": "Logistic regression (có tìm kiếm)",
        "knn_name": "ItemKNN (chỉ lịch sử)",
        "lr_name_short": "Logistic regression",
        "knn_name_short": "ItemKNN",
        "notes": notes,
        "summary_title": "Trên mọi lượt đặt của tập test, query token đóng góp nhiều nhất; sameDest giúp khách quay lại",
        "summary_note": (
            "Bốn lượt đặt trên được chọn để minh họa; đây là số liệu đo trên mọi lượt đặt của tập test."
        ),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese notes on a cp1258 console
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=("en", "vi"), default="en")
    lang = ap.parse_args().lang
    out = SRC.with_name("demo_storyboard_vi.html" if lang == "vi" else "demo_storyboard.html")
    d = json.loads(SRC.read_text(encoding="utf-8"))
    t = texts(d, lang)
    u = t["ui"]
    for k, v in t["notes"].items():
        print(f"note {k}: {v}")
    title = "Storyboard gợi ý" if lang == "vi" else "Recommendation storyboard"
    data_json = json.dumps(d, separators=(",", ":")).replace("</", "<\\/")
    text_json = json.dumps(t, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style></head>
<body><div id="wrap"><div id="stage">
<h1 id="title"></h1><div class="sub" id="sub"></div>
<div class="bar"><div class="steps" id="steps"></div>
<div class="nav"><span id="pos" style="margin-right:8px"></span>
<button id="prev">{u["back"]}</button><button id="wi">{u["whatif"]}</button><button id="next">{u["next"]}</button></div></div>
<div id="body"></div>
<div class="foot">{FOOTER[lang]}.<br>{SIDE[lang]}</div>
</div></div>
<script type="application/json" id="data">{data_json}</script>
<script type="application/json" id="text">{text_json}</script>
<script>{JS}</script>
</body></html>
"""
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html.encode('utf-8')) / 1024:.0f} KB, {len(d['bookings'])} bookings)")


if __name__ == "__main__":
    main()
