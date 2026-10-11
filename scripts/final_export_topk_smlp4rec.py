"""Top-20 lists and target ranks on every test event for the SMLP4Rec models of the final report.

No training. Loads the saved checkpoints and scores the 263,331 test-window events
(warm L >= 1 and cold L = 0), so that `scripts/final_results_summary.py` can compute the
beyond-accuracy metrics (intra-list diversity, popularity bias, coverage), which need the
recommended lists and not only the rank of the booked cluster.

Models (seed 2022):
- final model, 01d with early stopping (`2026-10-10_..._query-1d-clean-cold-es/best.pth`):
  query token alone, and query token + sameDest with the weights tuned on valid in
  `smlprec_query_cold_train_es_run.json` (`samedest_hybrid.tuned_on_valid.A`);
- original code, plain SMLP4Rec (history only) trained on all events, 01e
  (`2026-10-10_..._plain-1e-clean-cold-es/best.pth`); cold rows = all-padding history, length 1,
  exactly as in notebook 01e;
- original code, plain SMLP4Rec trained on warm rows only, nb01 (`2026-10-08_..._clean-plain/best.pth`);
  warm rows only (it never saw an empty history; cold rows are filled with -1).

The scoring code is the batched version of `scripts/demo_ui.py` / notebook 08 (same Data loader,
query encoding, sameDest and fusion). Each target rank is checked against the per-row ranks saved
by notebooks 01d / 01e (`ranks_test_*.npz`) and the agreement is printed and stored.

Output: `data/interim/final_topk_test.npz` (gitignored). Cluster ids are anonymous (proxy for
room category / package); offline only.

Usage (smlp4rec venv): C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/final_export_topk_smlp4rec.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch  # before pandas: on Windows, pandas first makes c10.dll fail to load

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import demo_ui as demo
import yaml

from nbp.data.query_features import encode
from nbp.hybrid.fusion import same_dest_distribution, smoothed_log
from nbp.hybrid.same_dest import same_dest_scores
from src.models.smlprec import SMLPREC

RES = ROOT / "results" / "week4_rebuild"
EXP = ROOT / "experiments"
FINAL_DIR = "2026-10-10_expedia_smlp4rec_query-1d-clean-cold-es"
FINAL_JSON = "smlprec_query_cold_train_es_run.json"
PLAIN_ALL_DIR = EXP / "2026-10-10_expedia_smlp4rec_plain-1e-clean-cold-es"
PLAIN_WARM_DIR = EXP / "2026-10-08_expedia_smlp4rec_clean-plain"
OUT = ROOT / "data" / "interim" / "final_topk_test.npz"
TOP = 20
BATCH = 4096
SEED = 2022


def plain_model(path: Path, n_items: int) -> SMLPREC:
    """History-only SMLPREC from a checkpoint (architecture keys from the project yaml)."""
    cfg_y = yaml.safe_load((ROOT / "configs" / "smlprec_expedia.yaml").read_text(encoding="utf-8"))
    cfg = demo.Cfg(
        {
            k: cfg_y[k]
            for k in (
                "MAX_ITEM_LIST_LENGTH",
                "n_layers",
                "n_heads",
                "hidden_size",
                "hidden_dropout_prob",
                "hidden_act",
                "layer_norm_eps",
                "initializer_range",
                "selected_features",
                "pooling_mode",
                "loss_type",
            )
        }
    )
    cfg.update(
        USER_ID_FIELD="user_id",
        ITEM_ID_FIELD="item_id",
        LIST_SUFFIX="_list",
        ITEM_LIST_LENGTH_FIELD="item_length",
        NEG_PREFIX="neg_",
        device="cpu",
    )
    ds = type("DS", (), {"num": staticmethod(lambda f: n_items)})()
    m = SMLPREC(cfg, ds)
    m.load_state_dict(torch.load(path, weights_only=False, map_location="cpu")["state_dict"])
    m.eval()
    return m


def rank_and_top(score: np.ndarray, truth_col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """score (n, 100) in raw cluster order -> (rank of truth (n,), top-TOP cluster ids (n, TOP)).

    Ties: lower raw cluster id first (stable sort), the same order as notebook 08.
    """
    order = np.argsort(-score, axis=1, kind="stable")
    rank = (order == truth_col[:, None]).argmax(1) + 1
    return rank.astype(np.int16), order[:, :TOP].astype(np.int8)


def main() -> None:
    t0 = time.time()
    demo.RUNS[SEED] = (FINAL_DIR, FINAL_JSON)  # the early-stopping checkpoint, not the 3-epoch one
    data = demo.Data()
    runner = demo.Runner(SEED, data)
    meta = json.loads((RES / FINAL_JSON).read_text(encoding="utf-8"))
    test_first = meta["splits"]["test"]["first_target_event"]
    events = np.arange(test_first, len(data.user))
    print(
        f"loaded in {time.time() - t0:.0f}s | test-window events {len(events):,} "
        f"(warm {(data.n_prior[events] > 0).sum():,}, cold {(data.n_prior[events] == 0).sum():,})"
    )
    print("sameDest: alpha", runner.alpha, "w_s (new, known destination)", runner.ws)

    p_all = plain_model(PLAIN_ALL_DIR / "best.pth", runner.n_items)
    p_warm = plain_model(PLAIN_WARM_DIR / "best.pth", runner.n_items)
    names = ("query_token", "main", "plain_all", "plain_warm")
    ranks = {k: np.empty(len(events), np.int16) for k in names}
    tops = {k: np.empty((len(events), TOP), np.int8) for k in names}
    known = np.zeros(len(events), bool)
    hist_len = np.zeros(len(events), np.int64)
    d, r = data, runner
    with torch.no_grad():
        for s in range(0, len(events), BATCH):
            ev_b = events[s : s + BATCH]
            n = len(ev_b)
            seq = np.zeros((n, r.s_len), np.int64)
            h_dest = np.zeros((n, r.s_len), np.int64)
            length = np.zeros(n, np.int64)
            for i, e in enumerate(ev_b):
                a, _ = d.u_start[int(d.user[e])]
                h = d.by_user[a : a + d.n_prior[e]][-r.s_len :]
                length[i] = len(h)
                seq[i, : len(h)] = [d.cluster2id[int(d.item[x])] for x in h]
                h_dest[i, : len(h)] = d.codes[h, 0]
            codes = d.codes[ev_b].astype(np.int32)
            q_idx = torch.from_numpy(encode(codes, r.vocab)).long()
            sc = r.model.scores(torch.from_numpy(seq), q_idx)
            sc[:, 0] = float("-inf")
            p = torch.softmax(sc, dim=1).numpy()[:, d.cluster_ids]
            q_dest = codes[:, 0].astype(np.int64)
            sd = same_dest_scores(
                seq, length, h_dest, q_dest, r.n_items, base=demo.RECENCY, normalize=True
            )[:, d.cluster_ids]
            kn = ((h_dest == q_dest[:, None]) & (np.arange(r.s_len)[None] < length[:, None])).any(1)
            ws = np.where(kn, r.ws[1], r.ws[0])
            fused = smoothed_log(p, r.alpha) + ws[:, None] * smoothed_log(
                same_dest_distribution(sd), r.alpha
            )
            truth = d.item[ev_b].astype(np.int64)
            sl = slice(s, s + n)
            ranks["query_token"][sl], tops["query_token"][sl] = rank_and_top(np.log(p), truth)
            ranks["main"][sl], tops["main"][sl] = rank_and_top(fused, truth)
            # plain models: empty history -> one all-padding position (length 1), as in 01e
            ln_t = torch.from_numpy(np.maximum(length, 1))
            for key, m in (("plain_all", p_all), ("plain_warm", p_warm)):
                ps = m.full_sort_predict(
                    {"item_id_list": torch.from_numpy(seq), "item_length": ln_t}
                )
                ps[:, 0] = float("-inf")
                ranks[key][sl], tops[key][sl] = rank_and_top(ps.numpy()[:, d.cluster_ids], truth)
            known[sl], hist_len[sl] = kn, length
            if (s // BATCH) % 10 == 0:
                print(f"  {s + n:,}/{len(events):,} ({time.time() - t0:.0f}s)")
    cold = d.n_prior[events] == 0
    ranks["plain_warm"][cold] = -1  # trained on warm rows only: not scored on cold rows
    tops["plain_warm"][cold] = -1

    # agreement with the per-row ranks saved by the training notebooks
    zw = np.load(EXP / FINAL_DIR / "ranks_test_warm.npz")
    zc = np.load(EXP / FINAL_DIR / "ranks_test_cold.npz")
    ze = np.load(PLAIN_ALL_DIR / "ranks_test.npz")
    pos = {int(e): i for i, e in enumerate(events)}
    iw = np.array([pos[int(e)] for e in zw["row"]])
    ic = np.array([pos[int(e)] for e in zc["row"]])
    agree = {
        "main vs 01d ranks_test_warm[query_token_sameDest]": float(
            (ranks["main"][iw] == zw["query_token_sameDest"]).mean()
        ),
        "query_token vs 01d ranks_test_warm[query_token]": float(
            (ranks["query_token"][iw] == zw["query_token"]).mean()
        ),
        "plain_warm vs 01d ranks_test_warm[plain] (nb01 rescored)": float(
            (ranks["plain_warm"][iw] == zw["plain"]).mean()
        ),
        "query_token (cold) vs 01d ranks_test_cold[query_token_alone]": float(
            (ranks["query_token"][ic] == zc["query_token_alone"]).mean()
        ),
        "plain_all (warm) vs 01e ranks_test.npz[warm]": float(
            (ranks["plain_all"][iw] == ze["warm"]).mean()
        ),
        "plain_all (cold) vs 01e ranks_test.npz[cold], cold order of 01d": float(
            (ranks["plain_all"][ic] == ze["cold"]).mean()
        ),
    }
    for k, v in agree.items():
        print(f"rank agreement {k}: {v:.6f}")
    np.savez_compressed(
        OUT,
        event=events,
        n_prior=d.n_prior[events],
        known_dest_last20=known,
        hist_len=hist_len,
        truth=d.item[events].astype(np.int16),
        q_dest=d.codes[events, 0],
        **{f"rank_{k}": v for k, v in ranks.items()},
        **{f"top_{k}": v for k, v in tops.items()},
        agreement=json.dumps(agree),
        alpha=r.alpha,
        ws_new=r.ws[0],
        ws_known=r.ws[1],
    )
    print(f"wrote {OUT.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
