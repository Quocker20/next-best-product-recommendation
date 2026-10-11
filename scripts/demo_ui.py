"""Quick demo UI: simulate a user and a search, get the top-5 clusters from the main model.

Main model = SMLP4Rec + query token trained with the first bookings (notebook 01d), plus the
sameDest term with the weights tuned on valid (`samedest_hybrid` in the run json). Offline demo,
nothing is written to the data. Cluster ids are anonymous (proxy of room category / package).

Inputs: a user id (blank = new user, empty history) and the 14 query fields of the search.
"Random test booking" fills the form from a booking of the test split (seed taken from the clock, shown on the page);
the actual booking is then shown next to the recommendation. The history of a user is the
bookings before the chosen event (a typed user id without an event uses the full history).

Usage (smlp4rec venv, Python 3.11):
    C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/demo_ui.py [--port 8765]
then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch  # before pandas: on Windows, pandas first makes c10.dll fail to load

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pandas as pd  # noqa: E402

from nbp.data.query_features import LEAD_BINS, QUERY_FIELDS, encode  # noqa: E402
from nbp.hybrid.fusion import same_dest_distribution, smoothed_log  # noqa: E402
from nbp.hybrid.same_dest import same_dest_scores  # noqa: E402
from nbp.models.smlprec_query import SMLPRECQuery  # noqa: E402

TOP_K = 5  # hard-coded on purpose
RECENCY = 0.7
N_CLUSTERS = 100
INTER = ROOT / "data" / "interim" / "recbole" / "expedia_clean_rowid" / "expedia_clean_rowid.inter"
CODES = ROOT / "data" / "interim" / "expedia_query_codes_clean.npz"
RUNS = {  # model seed -> (experiment folder, run json)
    2022: ("2026-10-09_expedia_smlp4rec_query-1d-clean-cold", "smlprec_query_cold_train_run.json"),
    2023: ("2026-10-09_expedia_smlp4rec_query-1d-clean-cold-s2023", "smlprec_query_cold_train_run_seed2023.json"),
    2024: ("2026-10-09_expedia_smlp4rec_query-1d-clean-cold-s2024", "smlprec_query_cold_train_run_seed2024.json"),
}
SMALL = {"dest_type", "ci_month", "lead", "stay", "adults", "children", "rooms", "package",
         "mobile", "channel", "site", "posa"}  # dropdowns; dest and country are free numbers


BASELINES = ROOT / "data" / "interim" / "demo_baselines.npz"  # scripts/export_baselines_for_demo.py
LR_ORDER = ["dest", "dest_type", "ci_month", "stay", "lead", "adults", "children", "rooms",
            "package", "channel", "mobile", "site", "posa", "country"]  # notebook 07 LR_FEATURES


class Baselines:
    """ItemKNN (history only) and logistic regression (search context only) from notebook 07."""

    def __init__(self, path: Path = BASELINES) -> None:
        z = np.load(path)
        self.sim, self.pop, self.coef, self.icpt = z["knn_similarity"], z["pop"], z["coef"], z["intercept"]
        # empty-history fallback of ItemKNN: first-booking popularity if exported (07b), else `pop`
        self.pop_cold = z["pop_first"] if "pop_first" in z.files else self.pop
        self.cats = [list(z[f"cat{i}"]) for i in range(int(z["n_cat"]))]
        self.offset = np.concatenate([[0], np.cumsum([len(c) for c in self.cats])[:-1]])

    def _rank(self, score: np.ndarray) -> list[int]:
        return [int(c) for c in np.lexsort((np.arange(N_CLUSTERS), -self.pop, -score))]  # ties -> popular

    def _lr_value(self, f: str, c: int) -> str:
        if f in ("ci_month", "adults", "rooms"):
            return str(c if c > 0 else -1)
        if f in ("stay", "lead"):
            return str(c - 1 if c > 0 else -1)  # code 0 unknown; stay nights / lead bucket = code - 1
        return str(c)

    def recommend(self, hist_clusters: list[int], fields: dict[str, int], truth: int | None) -> dict:
        h = np.zeros(N_CLUSTERS, np.float32)
        for c in hist_clusters[-20:]:
            h[c] += 1
        knn_score = h @ self.sim if len(hist_clusters) else self.pop_cold.astype(np.float64)
        cols = []
        for i, f in enumerate(LR_ORDER):
            v = self._lr_value(f, fields[f])
            if v in self.cats[i]:
                cols.append(self.offset[i] + self.cats[i].index(v))  # unseen value -> all-zero block
        z = self.coef[:, cols].sum(1) + self.icpt
        lr_p = np.exp(z - z.max())
        lr_p /= lr_p.sum()
        out = {}
        for name, score, note in (("ItemKNN", knn_score, "history only" + ("" if hist_clusters else " (empty: popularity)")),
                                  ("Logistic regression", lr_p, "destination + context, no history")):
            order = self._rank(np.asarray(score, np.float64))
            e = {"note": note, "top": order[:TOP_K]}
            if truth is not None:
                e["truth_rank"] = order.index(truth) + 1
            out[name] = e
        return out


class Cfg(dict):
    """Minimal stand-in for the RecBole Config (only the keys the model reads)."""


class Data:
    """Bookings table, query codes and the test split, loaded once."""

    def __init__(self) -> None:
        df = pd.read_csv(INTER, sep="\t")
        self.user = df["user_id:token"].to_numpy()
        row_id = df["row_id:float"].to_numpy().astype(np.int64)
        assert (row_id == np.arange(len(df))).all(), "row_id must equal the line index"
        tok = df["item_id:token"].to_numpy()
        # RecBole item ids: [PAD] = 0, then first-appearance order of the tokens
        codes_first, uniques = pd.factorize(tok)
        self.item = tok.astype(np.int64)  # raw cluster id of every booking
        self.cluster2id = {int(c): i + 1 for i, c in enumerate(uniques)}
        self.id2cluster = {v: k for k, v in self.cluster2id.items()}
        self.cluster_ids = np.array([self.cluster2id[k] for k in range(N_CLUSTERS)])
        z = np.load(CODES)
        self.codes = z["codes"]
        assert list(z["fields"]) == list(QUERY_FIELDS) and len(self.codes) == len(df)
        order = np.lexsort((np.arange(len(df)), self.user))  # by user, then event
        self.by_user = order
        su = self.user[order]
        self.u_start = {}
        uniq, first = np.unique(su, return_index=True)
        last = np.append(first[1:], len(su))
        for u, a, b in zip(uniq, first, last):
            self.u_start[int(u)] = (int(a), int(b))
        self.n_prior = np.zeros(len(df), np.int64)  # bookings of the same user before this one
        pos = np.arange(len(su)) - np.repeat(first, last - first)
        self.n_prior[order] = pos
        self.test_first = None  # set per model run (same split for all seeds)

    def history(self, user: int | None, before_event: int | None) -> np.ndarray:
        """Event ids (oldest first) of the user's bookings before `before_event` (None = all)."""
        if user is None or user not in self.u_start:
            return np.zeros(0, np.int64)
        a, b = self.u_start[user]
        ev = self.by_user[a:b]
        if before_event is not None:
            ev = ev[ev < before_event]
        return ev


class Runner:
    """One trained checkpoint + its sameDest weights."""

    def __init__(self, seed: int, data: Data) -> None:
        folder, js = RUNS[seed]
        ck = torch.load(ROOT / "experiments" / folder / "best.pth", weights_only=False)
        meta = json.loads((ROOT / "results" / "week4_rebuild" / js).read_text(encoding="utf-8"))
        vz = np.load(ROOT / "experiments" / folder / "vocab.npz")
        self.vocab = [vz[f"vocab_{f}"] for f in QUERY_FIELDS]
        cfg = Cfg(ck["config"])
        cfg.update(USER_ID_FIELD="user_id", ITEM_ID_FIELD="item_id", LIST_SUFFIX="_list",
                   ITEM_LIST_LENGTH_FIELD="item_length", NEG_PREFIX="neg_", n_heads=1,
                   selected_features=[], pooling_mode="mean", device="cpu")
        n_items = ck["state_dict"]["item_embedding.weight"].shape[0]
        ds = type("DS", (), {"num": staticmethod(lambda f: n_items)})()
        self.model = SMLPRECQuery(cfg, ds, torch.zeros((1, len(QUERY_FIELDS))), ck["field_sizes"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.s_len = cfg["MAX_ITEM_LIST_LENGTH"]
        self.n_items = n_items
        sd = meta["samedest_hybrid"]["tuned_on_valid"]["A"]
        self.alpha = sd["alpha"]
        self.ws = {int(k): v[1] for k, v in sd["weights"].items()}  # 0 = new dest, 1 = known dest
        self.test_first = meta["splits"]["test"]["first_target_event"]
        self.seed = seed
        self.data = data

    @torch.no_grad()
    def score_all(self, hist_events: np.ndarray, fields: dict[str, int]) -> dict:
        """Scores of all 100 clusters (raw cluster order) for one history + search.

        Returns `p` (model softmax, query token, no sameDest), `sd_p` (sameDest distribution),
        `score` (final log-score = model + sameDest), `known` (destination in the history),
        `ws` (sameDest weight used), `length` (history length used), `codes` (1, F) query codes.
        """
        d = self.data
        ev = hist_events[-self.s_len:]
        length = len(ev)
        seq = np.zeros((1, self.s_len), np.int64)
        seq[0, :length] = [d.cluster2id[int(d.item[e])] for e in ev]
        h_dest = np.zeros((1, self.s_len), np.int64)
        h_dest[0, :length] = d.codes[ev, 0]
        codes = np.array([[fields[f] for f in QUERY_FIELDS]], np.int32)
        q_idx = torch.from_numpy(encode(codes, self.vocab)).long()
        s = self.model.scores(torch.from_numpy(seq), q_idx)
        s[:, 0] = float("-inf")
        p = torch.softmax(s, dim=1).numpy()[:, d.cluster_ids]  # (1, 100) in raw cluster order
        q_dest = np.array([fields["dest"]], np.int64)
        sd = same_dest_scores(seq, np.array([length]), h_dest, q_dest, self.n_items,
                              base=RECENCY, normalize=True)[:, d.cluster_ids]
        known = bool(length and (h_dest[0, :length] == q_dest[0]).any())
        ws = self.ws[1 if known else 0]
        sd_p = same_dest_distribution(sd)
        score = smoothed_log(p, self.alpha) + ws * smoothed_log(sd_p, self.alpha)
        return {"p": p, "sd_p": sd_p, "score": score, "known": known, "ws": ws,
                "length": length, "codes": codes}

    def recommend(self, hist_events: np.ndarray, fields: dict[str, int], truth: int | None) -> dict:
        r = self.score_all(hist_events, fields)
        p, sd_p, score, known, ws = r["p"], r["sd_p"], r["score"], r["known"], r["ws"]
        length, codes = r["length"], r["codes"]
        final = np.exp(score - score.max())
        final = (final / final.sum())[0]
        order = np.argsort(-score[0], kind="stable")
        top = [{"rank": i + 1, "cluster": int(c), "score": round(float(final[c]), 4),
                "model_p": round(float(p[0, c]), 4), "samedest_share": round(float(sd_p[0, c]), 4)}
               for i, c in enumerate(order[:TOP_K])]
        out = {"top": top, "known_destination": known, "w_samedest": ws, "alpha": self.alpha,
               "history_len": int(length), "dest_in_vocab": bool(encode(codes, self.vocab)[0, 0] > 0)}
        if truth is not None:
            out["truth"] = {"cluster": truth, "rank": int(np.flatnonzero(order == truth)[0]) + 1,
                            "in_top": any(t["cluster"] == truth for t in top)}
        return out


def field_meta(vocab: list[np.ndarray]) -> list[dict]:
    """Form description: one entry per query field."""
    out = []
    for f, v in zip(QUERY_FIELDS, vocab):
        e = {"name": f}
        if f in SMALL:
            opts = []
            for c in [0] + [int(x) for x in v if int(x) != 0]:
                if f == "lead":
                    lab = ("unknown" if c == 0 else f"{LEAD_BINS[c - 1] + 1}-{LEAD_BINS[c]} days"
                           if LEAD_BINS[c] < 10_000 else f">{LEAD_BINS[c - 1]} days")
                elif f == "stay":
                    lab = "unknown" if c == 0 else f"{c - 1} nights"
                elif f == "ci_month":
                    lab = "unknown" if c == 0 else f"month {c}"
                else:
                    lab = "unknown/0" if c == 0 else str(c)
                opts.append([c, f"{c}: {lab}"])
            e["options"] = opts
        else:
            e["hint"] = f"{len(v)} ids in the model vocabulary; others map to 'unknown'"
            e["datalist"] = [int(x) for x in v[:400]]
        out.append(e)
    return out


STATE: dict = {}
LOCK = threading.Lock()


def get_runner(seed: int) -> Runner:
    if seed not in STATE["runners"]:
        STATE["runners"][seed] = Runner(seed, STATE["data"])
    return STATE["runners"][seed]


def api_seed(req: dict) -> dict:
    d: Data = STATE["data"]
    first = get_runner(min(RUNS)).test_first
    rows = np.arange(first, len(d.user))
    mode = req.get("mode", "any")
    if mode == "warm":
        rows = rows[d.n_prior[rows] >= 1]
    elif mode == "cold":
        rows = rows[d.n_prior[rows] == 0]
    raw_seed = req.get("seed")
    if raw_seed not in (None, ""):
        seed = int(raw_seed)
        seed_type = "manual"
    else:
        seed = int(time.time_ns() % (2**32))  # tự động lấy seed theo thời gian thực
        seed_type = "time"
    e = int(rows[np.random.default_rng(seed).integers(len(rows))])
    return {"seed": seed, "seed_type": seed_type, "user_id": int(d.user[e]), "event": e, "truth": int(d.item[e]),
            "n_prior": int(d.n_prior[e]),
            "fields": {f: int(c) for f, c in zip(QUERY_FIELDS, d.codes[e])}}


def api_recommend(req: dict) -> dict:
    d: Data = STATE["data"]
    uid = req.get("user_id")
    uid = int(uid) if uid not in (None, "") else None
    if uid is not None and uid not in d.u_start:
        raise ValueError(f"user {uid} not found")
    event = req.get("event")
    event = int(event) if event not in (None, "") else None
    fields = {f: int(req["fields"][f]) for f in QUERY_FIELDS}
    truth = req.get("truth")
    r = get_runner(int(req.get("model_seed", 2022)))
    hist = d.history(uid, event)
    out = r.recommend(hist, fields, int(truth) if truth not in (None, "") else None)
    out["history"] = [{"event": int(e), "cluster": int(d.item[e]), "dest": int(d.codes[e, 0])}
                      for e in hist[-r.s_len:]]
    out["model_seed"] = r.seed
    out["baselines"] = STATE["baselines"].recommend(
        [int(d.item[e]) for e in hist], fields, int(truth) if truth not in (None, "") else None)
    return out


PAGE = """<!doctype html><meta charset=utf-8><title>NBP demo</title>
<style>body{font-family:sans-serif;max-width:980px;margin:16px auto;padding:0 12px}
fieldset{margin:8px 0}label{display:inline-block;margin:3px 10px 3px 0}
input,select{width:150px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 8px}
.hit{background:#cfc}.err{color:#b00}.note{color:#555;font-size:90%}</style>
<h2>Next-best hotel cluster: user simulator</h2>
<p class=note>Offline demo. Main model: SMLP4Rec + query token (trained with first bookings) + sameDest; plus the two baselines. Clusters are
anonymous ids (proxy of room category / package). Nothing is saved. Top-5 fixed.</p>
<fieldset><legend>User</legend>
<label>user id (blank = new user) <input id=user_id></label>
<label>history before event (blank = all) <input id=event></label>
<label>model seed <select id=model_seed></select></label>
<label>actual cluster (test) <input id=truth readonly></label></fieldset>
<fieldset><legend>Test-split seed</legend>
<label>rows <select id=mode><option>any<option>warm<option>cold</select></label>
<label>seed (auto: time) <input id=split_seed placeholder="auto (clock)" title="Tự động lấy seed theo thời gian thực (để trống để tự đổi)"></label>
<button onclick="pick(true)">Random test booking</button>
<label style="width:auto;margin-left:8px"><input id=auto_cycle type=checkbox style="width:auto" onchange="toggleAuto()"> auto-change every 5s</label>
<span class=note id=seedinfo></span></fieldset>
<fieldset><legend>Search (query fields)</legend><div id=form></div></fieldset>
<button onclick=rec()>Recommend top 5</button> <span id=msg class=err></span>
<div id=out></div>
<script>
let META=null, autoTimer=null;
const $=id=>document.getElementById(id);
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',
 body:body?JSON.stringify(body):undefined});const j=await r.json();if(j.error)throw new Error(j.error);return j}
function fieldVals(){const o={};for(const f of META.fields)o[f.name]=$('f_'+f.name).value;return o}
function toggleAuto(){
 if($('auto_cycle').checked){if(!autoTimer)autoTimer=setInterval(()=>{if($('auto_cycle').checked)pick(true)},5000)}
 else{if(autoTimer){clearInterval(autoTimer);autoTimer=null}}
}
async function init(){META=await api('/api/meta');
 $('model_seed').innerHTML=META.seeds.map(s=>`<option>${s}`).join('');
 let h='';for(const f of META.fields){h+=`<label>${f.name} `;
  if(f.options)h+=`<select id=f_${f.name}>${f.options.map(o=>`<option value=${o[0]}>${o[1]}`).join('')}</select>`;
  else h+=`<input id=f_${f.name} type=number value=0 list=dl_${f.name} title="${f.hint}"><datalist id=dl_${f.name}>${f.datalist.map(x=>`<option value=${x}>`).join('')}</datalist>`;
  h+='</label>'}
 $('form').innerHTML=h;
 for(const f of META.fields)$('f_'+f.name).addEventListener('input',()=>{$('truth').value='';$('seedinfo').textContent='(form edited: no comparison)'});
 await pick(true)}
async function pick(forceNewTime=false){try{$('msg').textContent='';
 if(forceNewTime)$('split_seed').value='';
 const s=$('split_seed').value.trim();
 const body={mode:$('mode').value};
 if(s)body.seed=parseInt(s,10);
 const j=await api('/api/seed',body);
 $('split_seed').value=j.seed;
 $('user_id').value=j.user_id;$('event').value=j.event;$('truth').value=j.truth;
 for(const k in j.fields)$('f_'+k).value=j.fields[k];
 const note=j.seed_type==='time'?`auto time (${new Date().toLocaleTimeString()})`:'manual';
 $('seedinfo').textContent=`seed ${j.seed} [${note}], event ${j.event}, user ${j.user_id}, ${j.n_prior} earlier bookings`;
 await rec()}catch(e){$('msg').textContent=e.message}}
async function rec(){try{$('msg').textContent='';
 const j=await api('/api/recommend',{user_id:$('user_id').value,event:$('event').value,
  truth:$('truth').value,model_seed:$('model_seed').value,fields:fieldVals()});
 const t=j.truth?j.truth.cluster:null;
 let h=`<h3>Top 5 (history ${j.history_len} bookings, model seed ${j.model_seed})</h3>
 <p class=note>destination ${j.known_destination?'is':'is not'} in the user's history, sameDest weight ${j.w_samedest}${j.dest_in_vocab?'':'; destination outside model vocabulary'}</p>
 <table><tr><th>rank<th>cluster<th>final score<th>model p<th>sameDest share</tr>`;
 for(const r of j.top)h+=`<tr class="${r.cluster===t?'hit':''}"><td>${r.rank}<td>${r.cluster}<td>${r.score}<td>${r.model_p}<td>${r.samedest_share}</tr>`;
 h+='</table>';
 if(j.truth)h+=`<p><b>Actual booking: cluster ${t}</b> - ${j.truth.in_top?'in the top 5':'not in the top 5'}, rank ${j.truth.rank} of 100.</p>`;
 h+='<h3>History used (oldest first, last 20)</h3>';
 h+=j.history.length?'<table><tr><th>event<th>cluster<th>destination</tr>'+j.history.map(x=>`<tr><td>${x.event}<td>${x.cluster}<td>${x.dest}</tr>`).join('')+'</table>':'<p>empty (new user)</p>';
 h+='<h3>Baselines (top 5)</h3><table><tr><th>baseline<th>input<th>top 5 clusters<th>actual rank</tr>';
 for(const k in j.baselines){const b=j.baselines[k];
  h+=`<tr><td>${k}<td>${b.note}<td>${b.top.map(c=>c===t?`<b class=hit>${c}</b>`:c).join(', ')}<td>${b.truth_rank===undefined?'':b.truth_rank}</tr>`}
 h+='</table>';
 $('out').innerHTML=h}catch(e){$('msg').textContent=e.message}}
init();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/meta":
            r = get_runner(min(RUNS))
            body = json.dumps({"fields": field_meta(r.vocab), "seeds": list(RUNS)})
            self._send(200, body.encode(), "application/json")
        else:
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with LOCK:
                out = {"/api/seed": api_seed, "/api/recommend": api_recommend}[self.path](req)
        except Exception as e:  # shown in the page
            out = {"error": f"{type(e).__name__}: {e}"}
        self._send(200, json.dumps(out).encode(), "application/json")

    def log_message(self, *a) -> None:  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    STATE["data"] = Data()
    STATE["runners"] = {}
    STATE["baselines"] = Baselines()
    for s in RUNS:
        get_runner(s)
    print(f"ready: http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
