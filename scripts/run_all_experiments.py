"""Generate all 16 canonical MPNN-DEGBSAT-CERT-1.0 certificates and 4 exact-Q studies
for Paper 43 (Marco Sälzer, arXiv:2611.32595, Nov 24, 2026).

Author: Lluis Eriksson
License: MIT
"""
from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mpnn_certifier import SCHEMA_VERSION, verify_mpnn_certificate_exact
from src.mpnn_exact import (
    build_ntm_delta_and_F_polyhedra,
    enc_config_q,
    enc_word_q,
    eval_union_relu_fnn_q,
    format_rat,
    in_interval_I,
    interval_I_bounds,
    pop_q,
    push_q,
    run_mpnn_Mw_exact_q,
)

CERT_DIR = ROOT / "certificates"
DATA_DIR = ROOT / "data"


def _ntm_spec_dict(
    k: int,
    blank: int,
    ns: int,
    q_acc: int,
    transitions: Sequence[Tuple[int, int, int, int, str]],
) -> Dict[str, Any]:
    return {
        "alphabet_k": k,
        "blank_sym": blank,
        "num_states": ns,
        "q_accept": q_acc,
        "transitions": [
            {"q1": q1, "a1": a1, "q2": q2, "a_prime": ap, "dir": dr}
            for (q1, a1, q2, ap, dr) in transitions
        ],
    }


def _make_mode1(cid: str, k: int, blank: int, word: List[int], push_sym: int) -> Dict[str, Any]:
    ew = enc_word_q(word, k, blank)
    ep = push_q(push_sym, ew, k)
    hd = word[0] if word else blank
    et = pop_q(hd, ew, k)
    flags = [in_interval_I(pop_q(a, ew, k), k) for a in range(1, k + 1)]
    return {
        "schema": SCHEMA_VERSION,
        "mode": "tape_encoding_lemma31_cert",
        "cert_id": cid,
        "alphabet_k": k,
        "blank_sym": blank,
        "word": word,
        "push_sym": push_sym,
        "claimed_enc_word": format_rat(ew),
        "claimed_enc_pushed": format_rat(ep),
        "claimed_enc_tail": format_rat(et),
        "claimed_head_sym": hd,
        "claimed_pop_in_I_flags": flags,
    }


def _sim_step(
    cfg: Tuple[List[int], int, int, List[int]],
    tr: Tuple[int, int, int, int, str] | None,
    blank: int,
) -> Tuple[List[int], int, int, List[int]]:
    l_t, st, hd, r_t = cfg
    if tr is None:
        return (list(l_t), st, hd, list(r_t))
    q1, a1, q2, ap, dr = tr
    assert st == q1 and hd == a1
    if dr == "R":
        new_l = [ap] + list(l_t)
        new_hd = r_t[0] if r_t else blank
        new_r = list(r_t[1:]) if r_t else []
        return (new_l, q2, new_hd, new_r)
    new_r = [ap] + list(r_t)
    new_hd = l_t[0] if l_t else blank
    new_l = list(l_t[1:]) if l_t else []
    return (new_l, q2, new_hd, new_r)


def _make_mode2(
    cid: str,
    k: int,
    blank: int,
    ns: int,
    q_acc: int,
    transitions: List[Tuple[int, int, int, int, str]],
    input_word: List[int],
    height_h: int,
    chosen_trs: List[Tuple[int, int, int, int, str] | None],
) -> Tuple[Dict[str, Any], List[List[Fraction]]]:
    assert len(chosen_trs) == (1 << height_h)
    init_hd = input_word[0] if input_word else blank
    init_r = list(input_word[1:]) if input_word else []
    cfgs = [([], 0, init_hd, init_r)]
    for tr in chosen_trs:
        cfgs.append(_sim_step(cfgs[-1], tr, blank))
    encoded = [enc_config_q(l_t, st, hd, r_t, k, blank, ns) for (l_t, st, hd, r_t) in cfgs]
    cert = {
        "schema": SCHEMA_VERSION,
        "mode": "polyhedral_ntm_delta_lemma41_cert",
        "cert_id": cid,
        "ntm": _ntm_spec_dict(k, blank, ns, q_acc, transitions),
        "input_word": input_word,
        "height_h": height_h,
        "run_configs": [
            {"left": l_t, "state": st, "head": hd, "right": r_t}
            for (l_t, st, hd, r_t) in cfgs
        ],
        "claimed_delta_poly_count": k * (len(transitions) + 1),
        "claimed_num_steps": 1 << height_h,
        "claimed_enc_init": [format_rat(q) for q in encoded[0]],
        "claimed_enc_final": [format_rat(q) for q in encoded[-1]],
        "claimed_all_steps_in_delta": True,
        "claimed_final_in_F": True,
    }
    return cert, encoded


def _make_mode3(
    cid: str,
    m: int,
    polys: List[Tuple[List[List[Fraction]], List[Fraction]]],
    queries: List[List[Fraction]],
) -> Dict[str, Any]:
    outs = [eval_union_relu_fnn_q(polys, q) for q in queries]
    mems = [o == Fraction(0, 1) for o in outs]
    return {
        "schema": SCHEMA_VERSION,
        "mode": "relu_polyhedra_union_lemma51_cert",
        "cert_id": cid,
        "ambient_dim_m": m,
        "polyhedra": [
            {
                "A": [[format_rat(v) for v in row] for row in A_mat],
                "b": [format_rat(v) for v in b_vec],
            }
            for A_mat, b_vec in polys
        ],
        "query_points": [[format_rat(v) for v in q] for q in queries],
        "claimed_num_polyhedra": len(polys),
        "claimed_fnn_outputs": [format_rat(o) for o in outs],
        "claimed_memberships": mems,
    }


def _build_node_label(height_h: int, level: int, side: int, A: List[Fraction], B: List[Fraction]) -> List[Fraction]:
    num_types = 2 * (height_h + 1)
    dim = 7 * num_types
    lbl = [Fraction(0, 1)] * dim
    tid = 2 * level + side
    off = 7 * tid
    lbl[off] = Fraction(1, 1)
    for j in range(3):
        lbl[off + 1 + j] = A[j]
        lbl[off + 4 + j] = B[j]
    return lbl


def _build_binary_tree_graph(
    height_h: int,
    encoded_chain: List[List[Fraction]],
) -> Tuple[int, List[Tuple[int, int]], List[List[Fraction]]]:
    """Build exact Theorem 6.1 full binary tree of height h (2^{h+1}-1 nodes, degree <= 3)."""
    assert len(encoded_chain) == (1 << height_h) + 1
    labels: List[List[Fraction]] = []
    edges: List[Tuple[int, int]] = []

    def rec(level: int, side: int, lo: int, hi: int) -> int:
        nid = len(labels)
        labels.append(_build_node_label(height_h, level, side, encoded_chain[lo], encoded_chain[hi]))
        if level < height_h:
            mid = (lo + hi) // 2
            l_id = rec(level + 1, 0, lo, mid)
            r_id = rec(level + 1, 1, mid, hi)
            edges.append((nid, l_id))
            edges.append((nid, r_id))
        return nid

    rec(0, 0, 0, 1 << height_h)
    return len(labels), edges, labels


def _make_mode4(
    cid: str,
    k: int,
    blank: int,
    ns: int,
    q_acc: int,
    transitions: List[Tuple[int, int, int, int, str]],
    input_word: List[int],
    height_h: int,
    num_nodes: int,
    edges: List[Tuple[int, int]],
    root_v: int,
    labels: List[List[Fraction]],
) -> Dict[str, Any]:
    init_hd = input_word[0] if input_word else blank
    init_r = list(input_word[1:]) if input_word else []
    enc_cw = enc_config_q([], 0, init_hd, init_r, k, blank, ns)
    delta_polys, F_polys = build_ntm_delta_and_F_polyhedra(k, ns, q_acc, transitions)
    res = run_mpnn_Mw_exact_q(height_h, num_nodes, edges, root_v, labels, enc_cw, delta_polys, F_polys)
    deg = [0] * num_nodes
    for u, w in edges:
        deg[u] += 1
        deg[w] += 1
    return {
        "schema": SCHEMA_VERSION,
        "mode": "local_sum_mpnn_deg3_theorem61_cert",
        "cert_id": cid,
        "ntm": _ntm_spec_dict(k, blank, ns, q_acc, transitions),
        "input_word": input_word,
        "height_h": height_h,
        "num_nodes": num_nodes,
        "edges": [[u, w] for (u, w) in edges],
        "root_v": root_v,
        "node_labels": [[format_rat(v) for v in lbl] for lbl in labels],
        "claimed_max_degree": max(deg),
        "claimed_label_dim": 14 * (height_h + 1),
        "claimed_final_e1_v": format_rat(res["final_e1_v"]),
        "claimed_final_e2_v": format_rat(res["final_e2_v"]),
        "claimed_mpnn_output": format_rat(res["output_v"]),
        "claimed_accepted": bool(res["accepted"]),
    }


def main() -> None:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Shared compact NTM T1: k=3 (1='a', 2='b', 3=blank), states 0(q0), 1(q1), 2(qF)
    # Nondeterministic palindrome/flip machine with both R and L transitions!
    t1_trs = [
        (0, 1, 0, 2, "R"),  # flip 1->2 moving right
        (0, 2, 1, 1, "R"),  # flip 2->1 moving right into q1
        (0, 3, 2, 3, "L"),  # hit blank -> accept moving left
        (1, 3, 2, 3, "L"),  # from q1 hit blank -> accept moving left
        (1, 1, 1, 1, "R"),  # scan 1 in q1
    ]
    # NTM T2: k=4 (1,2,3 non-blank, 4=blank), states 0,1,2,3(qF)
    t2_trs = [
        (0, 1, 1, 2, "R"),
        (1, 2, 1, 3, "R"),
        (1, 3, 2, 1, "L"),
        (2, 3, 3, 3, "R"),
        (1, 4, 3, 4, "L"),
    ]

    certs: List[Dict[str, Any]] = []

    # --- Mode 1: 4 Lemma 3.1 exact tape encoding certificates ---
    certs.append(_make_mode1("m1_tape_k3_empty_fixedpt", 3, 3, [], 2))
    certs.append(_make_mode1("m1_tape_k3_word_121", 3, 3, [1, 2, 1], 3))
    certs.append(_make_mode1("m1_tape_k4_word_3142", 4, 4, [3, 1, 4, 2], 1))
    certs.append(_make_mode1("m1_tape_k6_word_52613", 6, 6, [5, 2, 6, 1, 3], 4))

    # --- Mode 2: 4 Lemma 4.1 polyhedral NTM Delta^{2^h} runs ---
    c2_1, ch2_1 = _make_mode2(
        "m2_ntm_h1_2steps_exact", 3, 3, 3, 2, t1_trs, [1], 1,
        [(0, 1, 0, 2, "R"), (0, 3, 2, 3, "L")],
    )
    c2_2, ch2_2 = _make_mode2(
        "m2_ntm_h2_4steps_padded", 3, 3, 3, 2, t1_trs, [1, 2], 2,
        [(0, 1, 0, 2, "R"), (0, 2, 1, 1, "R"), (1, 3, 2, 3, "L"), None],
    )
    c2_3, ch2_3 = _make_mode2(
        "m2_ntm_h2_4steps_k4_lr", 4, 4, 4, 3, t2_trs, [1, 2, 3], 2,
        [(0, 1, 1, 2, "R"), (1, 2, 1, 3, "R"), (1, 3, 2, 1, "L"), (2, 3, 3, 3, "R")],
    )
    c2_4, ch2_4 = _make_mode2(
        "m2_ntm_h3_8steps_padded", 3, 3, 3, 2, t1_trs, [1, 2, 1, 1], 3,
        [
            (0, 1, 0, 2, "R"),
            (0, 2, 1, 1, "R"),
            (1, 1, 1, 1, "R"),
            (1, 1, 1, 1, "R"),
            (1, 3, 2, 3, "L"),
            None,
            None,
            None,
        ],
    )
    certs.extend([c2_1, c2_2, c2_3, c2_4])

    # --- Mode 3: 4 Lemma 5.1 ReLU-FNN polyhedra union recognition certificates ---
    z0, one, neg1 = Fraction(0, 1), Fraction(1, 1), Fraction(-1, 1)
    # 3.1: 2D disjoint squares [0,1]^2 U [3,4]^2
    p_sq1 = ([[one, z0], [neg1, z0], [z0, one], [z0, neg1]], [one, z0, one, z0])
    p_sq2 = ([[one, z0], [neg1, z0], [z0, one], [z0, neg1]], [Fraction(4, 1), Fraction(-3, 1), Fraction(4, 1), Fraction(-3, 1)])
    certs.append(_make_mode3(
        "m3_relu_2d_disjoint_boxes", 2, [p_sq1, p_sq2],
        [[Fraction(1, 2), Fraction(1, 2)], [Fraction(7, 2), Fraction(13, 4)], [Fraction(2, 1), Fraction(1, 2)], [Fraction(-1, 4), Fraction(1, 2)]],
    ))
    # 3.2: Exact accepting set F for T1 (3 polyhedra in Q^3!)
    delta_t1, F_t1 = build_ntm_delta_and_F_polyhedra(3, 3, 2, t1_trs)
    certs.append(_make_mode3(
        "m3_relu_ntm_accept_set_F", 3, F_t1,
        [ch2_1[-1], ch2_2[-1], ch2_1[0], [Fraction(1, 10), ch2_1[-1][1], ch2_1[-1][2]]],
    ))
    # 3.3: Exact transition relation Delta subset for T1 (first 6 polyhedra in Q^6: P_1..P_3 + R_{t0,1..3})
    certs.append(_make_mode3(
        "m3_relu_ntm_delta_subset_6d", 6, delta_t1[:6],
        [
            ch2_1[0] + ch2_1[1],        # valid step in R_{t0, 3}
            ch2_2[3] + ch2_2[4],        # valid padding step in P_1
            ch2_1[1] + ch2_1[0],        # reversed invalid step
            ch2_1[0] + ch2_1[0],        # non-accepting self-loop invalid
        ],
    ))
    # 3.4: 3D L-shaped non-convex union of 3 rational simplices/boxes
    p_a = ([[one, one, z0], [neg1, z0, z0], [z0, neg1, z0], [z0, z0, one], [z0, z0, neg1]], [one, z0, z0, one, z0])
    p_b = ([[one, z0, z0], [neg1, z0, z0], [z0, one, z0], [z0, neg1, z0], [z0, z0, one], [z0, z0, neg1]], [Fraction(2, 1), Fraction(-1, 1), Fraction(1, 2), z0, Fraction(1, 2), z0])
    p_c = ([[one, z0, z0], [neg1, z0, z0], [z0, one, z0], [z0, neg1, z0], [z0, z0, one], [z0, z0, neg1]], [Fraction(1, 2), z0, Fraction(2, 1), Fraction(-1, 1), Fraction(1, 2), z0])
    certs.append(_make_mode3(
        "m3_relu_3d_nonconvex_3union", 3, [p_a, p_b, p_c],
        [
            [Fraction(1, 3), Fraction(1, 3), Fraction(1, 2)],
            [Fraction(3, 2), Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 4), Fraction(3, 2), Fraction(1, 4)],
            [Fraction(3, 2), Fraction(3, 2), Fraction(1, 4)],
            [Fraction(1, 2), Fraction(1, 2), Fraction(3, 2)],
        ],
    ))

    # --- Mode 4: 4 Theorem 6.1 full local-sum MPNN M_w executions on deg <= 3 graphs ---
    # 4.1: h=1 (3 nodes, 2 steps, deg <= 2) ACCEPTING
    n1, e1, l1 = _build_binary_tree_graph(1, ch2_1)
    certs.append(_make_mode4("m4_mpnn_h1_tree_accept", 3, 3, 3, 2, t1_trs, [1], 1, n1, e1, 0, l1))

    # 4.2: h=2 (7 nodes, 4 steps, deg <= 3) ACCEPTING
    n2, e2, l2 = _build_binary_tree_graph(2, ch2_2)
    certs.append(_make_mode4("m4_mpnn_h2_tree_deg3_accept", 3, 3, 3, 2, t1_trs, [1, 2], 2, n2, e2, 0, l2))

    # 4.3: h=3 (15 nodes, 8 steps, deg <= 3) ACCEPTING
    n3, e3, l3 = _build_binary_tree_graph(3, ch2_4)
    certs.append(_make_mode4("m4_mpnn_h3_tree_8steps_accept", 3, 3, 3, 2, t1_trs, [1, 2, 1, 1], 3, n3, e3, 0, l3))

    # 4.4: h=2 (7 nodes, deg <= 3) CORRUPTED LEAF TRANSITION -> REJECTED (soundness demo!)
    l2_bad = [list(row) for row in l2]
    # Perturb leaf node 6's B_0 coordinate by +1/5 so (A_6, B_6) leaves Delta!
    off_leaf_r = 7 * (2 * 2 + 1)
    l2_bad[6][off_leaf_r + 4] += Fraction(1, 5)
    certs.append(_make_mode4("m4_mpnn_h2_corrupted_leaf_reject", 3, 3, 3, 2, t1_trs, [1, 2], 2, n2, e2, 0, l2_bad))

    # Write & benchmark all 16 canonical certificates
    replay_rows: List[Dict[str, Any]] = []
    for cert in certs:
        out_path = CERT_DIR / f"{cert['cert_id']}.json"
        out_path.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
        t0 = time.perf_counter()
        summary = verify_mpnn_certificate_exact(cert)
        ms = round((time.perf_counter() - t0) * 1000.0, 3)
        replay_rows.append({"cert_id": cert["cert_id"], "mode": cert["mode"], "verify_ms": ms, "summary": summary})

    (DATA_DIR / "certificate_replay_summary.json").write_text(
        json.dumps({"count": len(replay_rows), "certificates": replay_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Experiment 1: Lemma 3.1 pop_a separation margin across alphabet sizes k = 2..8
    exp1_rows = []
    for k in range(2, 9):
        b = k + 2
        lo, hi = interval_I_bounds(k)
        width = hi - lo
        ew = enc_word_q([1, k], k, k)
        # Minimum distance of any wrong pop_a(enc(w)) to interval I
        wrong_gaps = []
        for a in range(2, k + 1):
            pv = pop_q(a, ew, k)
            gap = lo - pv if pv < lo else pv - hi
            wrong_gaps.append(gap)
        min_gap = min(wrong_gaps)
        exp1_rows.append({
            "k": k,
            "base_b": b,
            "I_lo": format_rat(lo),
            "I_hi": format_rat(hi),
            "interval_width_float": round(width.numerator / width.denominator, 6),
            "min_wrong_pop_gap_float": round(min_gap.numerator / min_gap.denominator, 6),
        })
    (DATA_DIR / "exp1_lemma31_pop_separation.json").write_text(
        json.dumps({"rows": exp1_rows}, indent=2) + "\n", encoding="utf-8"
    )

    # Experiment 2: Theorem 6.1 polynomial MPNN size vs exponential verified NTM run length 2^h
    exp2_rows = []
    for h in range(1, 13):
        exp2_rows.append({
            "height_h": h,
            "mpnn_layers": h + 2,
            "label_dim": 14 * (h + 1),
            "run_length_2h": 1 << h,
            "tree_nodes": (1 << (h + 1)) - 1,
        })
    (DATA_DIR / "exp2_mpnn_vs_run_scaling.json").write_text(
        json.dumps({"rows": exp2_rows}, indent=2) + "\n", encoding="utf-8"
    )

    # Experiment 3: Lemma 5.1 ReLU-FNN violation profile along diagonal (x, x) across disjoint boxes [0,1]^2 U [3,4]^2
    exp3_rows = []
    for idx in range(0, 41):
        xv = Fraction(idx, 10)
        val = eval_union_relu_fnn_q([p_sq1, p_sq2], [xv, xv])
        exp3_rows.append({
            "x": round(xv.numerator / xv.denominator, 3),
            "fnn_violation": round(val.numerator / val.denominator, 6),
            "in_union": val == Fraction(0, 1),
        })
    (DATA_DIR / "exp3_relu_fnn_profile.json").write_text(
        json.dumps({"rows": exp3_rows}, indent=2) + "\n", encoding="utf-8"
    )

    # Experiment 4: Mode 4 MPNN exact error propagation comparison (accept vs corrupted)
    exp4_rows = [
        {
            "cert_id": r["cert_id"],
            "nodes": r["summary"]["nodes"],
            "max_deg": r["summary"]["max_deg"],
            "accepted": r["summary"]["accepted"],
            "output": r["summary"]["output"],
            "verify_ms": r["verify_ms"],
        }
        for r in replay_rows
        if r["mode"] == "local_sum_mpnn_deg3_theorem61_cert"
    ]
    (DATA_DIR / "exp4_mpnn_execution_audit.json").write_text(
        json.dumps({"rows": exp4_rows}, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Generated {len(replay_rows)} canonical {SCHEMA_VERSION} certificates and 4 experiment JSONs.")
    for r in replay_rows:
        print(f"  {r['cert_id']:40s} {r['verify_ms']:6.3f} ms  ({r['mode']})")


if __name__ == "__main__":
    main()
