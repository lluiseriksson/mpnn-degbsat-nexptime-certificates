"""Zero-floating-point exact-Q certificate verifier: MPNN-DEGBSAT-CERT-1.0.
Audits all four rational-encoding, polyhedral-NTM, ReLU-FNN, and local-sum MPNN pillars of
Marco Sälzer (arXiv:2611.32595, Nov 24, 2026)
"Degree-bounded Satisfiability of Local MPNNs with Sum Aggregation is NEXPTIME-hard":
  - Mode 1: tape_encoding_lemma31_cert (Lemma 3.1(1)-(3) exact base-b tape encoding & push/pop)
  - Mode 2: polyhedral_ntm_delta_lemma41_cert (Lemma 4.1 bounded NTM run <-> Delta^{2^h} & F)
  - Mode 3: relu_polyhedra_union_lemma51_cert (Lemma 5.1 exact ReLU-FNN recognition of S = U P_i)
  - Mode 4: local_sum_mpnn_deg3_theorem61_cert (Theorem 6.1 & Cor. 6.2 full M_w execution on deg <= 3)

Author: Lluis Eriksson
License: MIT
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Dict, List, Set, Tuple

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

SCHEMA_VERSION = "MPNN-DEGBSAT-CERT-1.0"
_CANON_INT_RE = re.compile(r"\A(0|-?[1-9][0-9]*)\Z")
_CANON_FRAC_RE = re.compile(r"\A-?[1-9][0-9]*/[1-9][0-9]*\Z")
MAX_RAT_STR_LEN = 1024
MAX_INT_BITS = 2048
MAX_ID_LEN = 128
MAX_JSON_DEPTH = 32


class MPNNCertificateError(ValueError):
    """Raised whenever an MPNN-DEGBSAT-CERT-1.0 certificate fails any invariant."""


def _assert_strict_json_types(obj: Any, path: str = "$", depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise MPNNCertificateError(f"JSON nesting depth > {MAX_JSON_DEPTH} at {path}")
    t = type(obj)
    if t is int:
        if obj.bit_length() > MAX_INT_BITS:
            raise MPNNCertificateError(f"Integer bit-length > {MAX_INT_BITS} at {path}")
        return
    if t is str or t is bool:
        return
    if t is list:
        for i, v in enumerate(obj):
            _assert_strict_json_types(v, f"{path}[{i}]", depth + 1)
        return
    if t is dict:
        for k, v in obj.items():
            if type(k) is not str:
                raise MPNNCertificateError(f"Non-string dict key at {path}")
            _assert_strict_json_types(v, f"{path}.{k}", depth + 1)
        return
    raise MPNNCertificateError(f"Forbidden JSON type {t.__name__} at {path}")


def _parse_int(val: Any, name: str) -> int:
    if type(val) is not int:
        raise MPNNCertificateError(f"{name} must be strict int, got {type(val).__name__}")
    if val.bit_length() > MAX_INT_BITS:
        raise MPNNCertificateError(f"{name} integer too large")
    return val


def _parse_rat(val: Any, name: str) -> Fraction:
    if type(val) is int:
        if val.bit_length() > MAX_INT_BITS:
            raise MPNNCertificateError(f"{name} integer too large")
        return Fraction(val, 1)
    if type(val) is str:
        if len(val) > MAX_RAT_STR_LEN:
            raise MPNNCertificateError(f"{name} rational string too long")
        if _CANON_INT_RE.match(val):
            iv = int(val)
            if iv.bit_length() > MAX_INT_BITS:
                raise MPNNCertificateError(f"{name} integer bit-length > {MAX_INT_BITS}")
            return Fraction(iv, 1)
        if _CANON_FRAC_RE.match(val):
            num_s, den_s = val.split("/")
            num, den = int(num_s), int(den_s)
            if num.bit_length() > MAX_INT_BITS or den.bit_length() > MAX_INT_BITS:
                raise MPNNCertificateError(f"{name} rational bit-length > {MAX_INT_BITS}")
            if den < 2:
                raise MPNNCertificateError(f"{name} denominator must be >= 2: {val!r}")
            q = Fraction(num, den)
            if q.numerator != num or q.denominator != den:
                raise MPNNCertificateError(f"{name} unreduced fraction: {val!r}")
            return q
        raise MPNNCertificateError(f"{name} non-canonical rational string: {val!r}")
    raise MPNNCertificateError(f"{name} invalid type {type(val).__name__}")


def _parse_exact_keys(cert: Dict[str, Any], req: Set[str], mode: str) -> None:
    keys = set(cert.keys())
    if keys != req:
        raise MPNNCertificateError(f"{mode} key mismatch: {sorted(keys ^ req)}")


def _parse_ntm_spec(raw: Any) -> Tuple[int, int, int, int, List[Tuple[int, int, int, int, str]]]:
    if type(raw) is not dict:
        raise MPNNCertificateError("ntm must be object")
    _parse_exact_keys(raw, {"alphabet_k", "blank_sym", "num_states", "q_accept", "transitions"}, "ntm")
    k = _parse_int(raw["alphabet_k"], "ntm.alphabet_k")
    blank = _parse_int(raw["blank_sym"], "ntm.blank_sym")
    ns = _parse_int(raw["num_states"], "ntm.num_states")
    q_acc = _parse_int(raw["q_accept"], "ntm.q_accept")
    if not (2 <= k <= 8) or not (1 <= blank <= k) or not (2 <= ns <= 8) or not (0 <= q_acc < ns):
        raise MPNNCertificateError("Invalid NTM dimensions or indices")
    raw_tr = raw["transitions"]
    if type(raw_tr) is not list or not (1 <= len(raw_tr) <= 32):
        raise MPNNCertificateError("ntm.transitions must have length in [1, 32]")
    transitions: List[Tuple[int, int, int, int, str]] = []
    seen_tr: Set[Tuple[int, int, int, int, str]] = set()
    for i, tr in enumerate(raw_tr):
        if type(tr) is not dict:
            raise MPNNCertificateError(f"transitions[{i}] must be dict")
        _parse_exact_keys(tr, {"q1", "a1", "q2", "a_prime", "dir"}, f"transitions[{i}]")
        q1 = _parse_int(tr["q1"], "q1")
        a1 = _parse_int(tr["a1"], "a1")
        q2 = _parse_int(tr["q2"], "q2")
        ap = _parse_int(tr["a_prime"], "a_prime")
        dr = tr["dir"]
        if not (0 <= q1 < ns and 0 <= q2 < ns and 1 <= a1 <= k and 1 <= ap <= k):
            raise MPNNCertificateError(f"transitions[{i}] out of range")
        if dr not in ("L", "R"):
            raise MPNNCertificateError(f"transitions[{i}].dir must be 'L' or 'R'")
        tup = (q1, a1, q2, ap, dr)
        if tup in seen_tr:
            raise MPNNCertificateError(f"Duplicate transition {tup}")
        seen_tr.add(tup)
        transitions.append(tup)
    return k, blank, ns, q_acc, transitions


# =============================================================================
# Mode 1: Lemma 3.1 Exact Rational Tape Encoding, Push & Pop Disambiguation
# =============================================================================


def _verify_mode1_tape_encoding(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema",
        "mode",
        "cert_id",
        "alphabet_k",
        "blank_sym",
        "word",
        "push_sym",
        "claimed_enc_word",
        "claimed_enc_pushed",
        "claimed_enc_tail",
        "claimed_head_sym",
        "claimed_pop_in_I_flags",
    }
    _parse_exact_keys(cert, req, "Mode 1")
    k = _parse_int(cert["alphabet_k"], "alphabet_k")
    blank = _parse_int(cert["blank_sym"], "blank_sym")
    push_a = _parse_int(cert["push_sym"], "push_sym")
    if not (2 <= k <= 12) or not (1 <= blank <= k) or not (1 <= push_a <= k):
        raise MPNNCertificateError("Invalid alphabet_k, blank_sym, or push_sym")

    raw_w = cert["word"]
    if type(raw_w) is not list or len(raw_w) > 16:
        raise MPNNCertificateError("word must be list of length <= 16")
    word = [_parse_int(c, "word[i]") for c in raw_w]
    if any(not (1 <= c <= k) for c in word):
        raise MPNNCertificateError("word symbols must lie in 1..k")

    ew = enc_word_q(word, k, blank)
    if not in_interval_I(ew, k):
        raise MPNNCertificateError("Lemma 3.1(1) failed: enc(w) not in I")

    ep_direct = enc_word_q([push_a] + word, k, blank)
    ep_push = push_q(push_a, ew, k)
    if ep_direct != ep_push or not in_interval_I(ep_push, k):
        raise MPNNCertificateError("Lemma 3.1(2) push identity failed")

    head_sym = word[0] if word else blank
    tail_word = word[1:] if word else []
    et_direct = enc_word_q(tail_word, k, blank)
    et_pop = pop_q(head_sym, ew, k)
    if et_direct != et_pop or not in_interval_I(et_pop, k):
        raise MPNNCertificateError("Lemma 3.1(2) pop(head) identity failed")

    # Lemma 3.1(3): exhaustively check pop_a(enc(w)) in I <=> a == head(w) for all a in 1..k
    pop_flags = [in_interval_I(pop_q(a, ew, k), k) for a in range(1, k + 1)]
    expected_flags = [a == head_sym for a in range(1, k + 1)]
    if pop_flags != expected_flags:
        raise MPNNCertificateError("Lemma 3.1(3) pop disambiguation failed")

    if _parse_rat(cert["claimed_enc_word"], "claimed_enc_word") != ew:
        raise MPNNCertificateError("claimed_enc_word mismatch")
    if _parse_rat(cert["claimed_enc_pushed"], "claimed_enc_pushed") != ep_push:
        raise MPNNCertificateError("claimed_enc_pushed mismatch")
    if _parse_rat(cert["claimed_enc_tail"], "claimed_enc_tail") != et_pop:
        raise MPNNCertificateError("claimed_enc_tail mismatch")
    if _parse_int(cert["claimed_head_sym"], "claimed_head_sym") != head_sym:
        raise MPNNCertificateError("claimed_head_sym mismatch")
    raw_pf = cert["claimed_pop_in_I_flags"]
    if type(raw_pf) is not list or not all(type(b) is bool for b in raw_pf) or raw_pf != pop_flags:
        raise MPNNCertificateError("claimed_pop_in_I_flags mismatch")

    return {
        "verified": True,
        "mode": cert["mode"],
        "k": k,
        "base_b": k + 2,
        "enc_word": format_rat(ew),
        "head_sym": head_sym,
    }


# =============================================================================
# Mode 2: Lemma 4.1 Polyhedral NTM Transition Relation Delta^{2^h} & F Audit
# =============================================================================


def _verify_mode2_polyhedral_ntm(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema",
        "mode",
        "cert_id",
        "ntm",
        "input_word",
        "height_h",
        "run_configs",
        "claimed_delta_poly_count",
        "claimed_num_steps",
        "claimed_enc_init",
        "claimed_enc_final",
        "claimed_all_steps_in_delta",
        "claimed_final_in_F",
    }
    _parse_exact_keys(cert, req, "Mode 2")
    k, blank, ns, q_acc, transitions = _parse_ntm_spec(cert["ntm"])
    h = _parse_int(cert["height_h"], "height_h")
    if not (1 <= h <= 5):
        raise MPNNCertificateError("Require 1 <= height_h <= 5")
    num_steps = 1 << h

    raw_iw = cert["input_word"]
    if type(raw_iw) is not list or len(raw_iw) > 16:
        raise MPNNCertificateError("input_word must be list of length <= 16")
    iw = [_parse_int(c, "input_word") for c in raw_iw]
    if any(not (1 <= c <= k) or c == blank for c in iw):
        raise MPNNCertificateError("input_word symbols must be non-blank in 1..k")

    delta_polys, F_polys = build_ntm_delta_and_F_polyhedra(k, ns, q_acc, transitions)
    expected_delta_cnt = k * (len(transitions) + 1)
    if len(delta_polys) != expected_delta_cnt:
        raise MPNNCertificateError("Internal Delta polyhedron count mismatch")

    raw_cfgs = cert["run_configs"]
    if type(raw_cfgs) is not list or len(raw_cfgs) != num_steps + 1:
        raise MPNNCertificateError(f"run_configs must have length 2^h + 1 = {num_steps + 1}")

    encoded_chain: List[List[Fraction]] = []
    for idx, cf in enumerate(raw_cfgs):
        if type(cf) is not dict:
            raise MPNNCertificateError(f"run_configs[{idx}] must be dict")
        _parse_exact_keys(cf, {"left", "state", "head", "right"}, f"run_configs[{idx}]")
        if type(cf["left"]) is not list or type(cf["right"]) is not list:
            raise MPNNCertificateError(f"run_configs[{idx}] tapes must be lists")
        if len(cf["left"]) > 32 or len(cf["right"]) > 32:
            raise MPNNCertificateError(f"run_configs[{idx}] tape too long")
        l_tape = [_parse_int(v, "left") for v in cf["left"]]
        r_tape = [_parse_int(v, "right") for v in cf["right"]]
        st = _parse_int(cf["state"], "state")
        hd = _parse_int(cf["head"], "head")
        if not (0 <= st < ns and 1 <= hd <= k):
            raise MPNNCertificateError(f"run_configs[{idx}] invalid state/head")
        if any(not (1 <= c <= k) for c in l_tape) or any(not (1 <= c <= k) for c in r_tape):
            raise MPNNCertificateError(f"run_configs[{idx}] invalid tape symbol")
        if idx == 0:
            exp_head = iw[0] if iw else blank
            exp_right = iw[1:] if iw else []
            if l_tape != [] or st != 0 or hd != exp_head or r_tape != exp_right:
                raise MPNNCertificateError("run_configs[0] does not match initial configuration c_w")
        encoded_chain.append(enc_config_q(l_tape, st, hd, r_tape, k, blank, ns))

    # Verify every step (x_{j-1}, x_j) in Delta via exact polyhedral violation == 0
    for j in range(num_steps):
        pair = encoded_chain[j] + encoded_chain[j + 1]
        if eval_union_relu_fnn_q(delta_polys, pair) != Fraction(0, 1):
            raise MPNNCertificateError(f"Step {j} -> {j+1} is NOT in polyhedral relation Delta")

    # Verify final endpoint in F
    if eval_union_relu_fnn_q(F_polys, encoded_chain[-1]) != Fraction(0, 1):
        raise MPNNCertificateError("Final configuration encoding is NOT in accepting set F")

    if _parse_int(cert["claimed_delta_poly_count"], "claimed_delta_poly_count") != expected_delta_cnt:
        raise MPNNCertificateError("claimed_delta_poly_count mismatch")
    if _parse_int(cert["claimed_num_steps"], "claimed_num_steps") != num_steps:
        raise MPNNCertificateError("claimed_num_steps mismatch")
    raw_init = cert["claimed_enc_init"]
    raw_fin = cert["claimed_enc_final"]
    if type(raw_init) is not list or len(raw_init) != 3 or [_parse_rat(v, "enc_init") for v in raw_init] != encoded_chain[0]:
        raise MPNNCertificateError("claimed_enc_init mismatch")
    if type(raw_fin) is not list or len(raw_fin) != 3 or [_parse_rat(v, "enc_final") for v in raw_fin] != encoded_chain[-1]:
        raise MPNNCertificateError("claimed_enc_final mismatch")
    if cert["claimed_all_steps_in_delta"] is not True or cert["claimed_final_in_F"] is not True:
        raise MPNNCertificateError("claimed_all_steps_in_delta and claimed_final_in_F must be true")

    return {
        "verified": True,
        "mode": cert["mode"],
        "delta_polys": expected_delta_cnt,
        "steps": num_steps,
    }


# =============================================================================
# Mode 3: Lemma 5.1 Exact ReLU-FNN Recognition of Finite Union of Polyhedra
# =============================================================================


def _verify_mode3_relu_polyhedra(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema",
        "mode",
        "cert_id",
        "ambient_dim_m",
        "polyhedra",
        "query_points",
        "claimed_num_polyhedra",
        "claimed_fnn_outputs",
        "claimed_memberships",
    }
    _parse_exact_keys(cert, req, "Mode 3")
    m = _parse_int(cert["ambient_dim_m"], "ambient_dim_m")
    if not (1 <= m <= 8):
        raise MPNNCertificateError("Require 1 <= ambient_dim_m <= 8")

    raw_polys = cert["polyhedra"]
    if type(raw_polys) is not list or not (1 <= len(raw_polys) <= 16):
        raise MPNNCertificateError("polyhedra must have length in [1, 16]")
    polys: List[Tuple[List[List[Fraction]], List[Fraction]]] = []
    for pi, p in enumerate(raw_polys):
        if type(p) is not dict:
            raise MPNNCertificateError(f"polyhedra[{pi}] must be dict")
        _parse_exact_keys(p, {"A", "b"}, f"polyhedra[{pi}]")
        raw_A, raw_b = p["A"], p["b"]
        if type(raw_A) is not list or type(raw_b) is not list or not raw_A or len(raw_A) != len(raw_b) or len(raw_A) > 32:
            raise MPNNCertificateError(f"polyhedra[{pi}] invalid A/b dimensions")
        A_mat: List[List[Fraction]] = []
        b_vec: List[Fraction] = []
        for ri, row in enumerate(raw_A):
            if type(row) is not list or len(row) != m:
                raise MPNNCertificateError(f"polyhedra[{pi}].A[{ri}] width != {m}")
            A_mat.append([_parse_rat(v, f"A[{pi}][{ri}]") for v in row])
            b_vec.append(_parse_rat(raw_b[ri], f"b[{pi}][{ri}]"))
        polys.append((A_mat, b_vec))

    raw_pts = cert["query_points"]
    if type(raw_pts) is not list or not (2 <= len(raw_pts) <= 32):
        raise MPNNCertificateError("query_points must have length in [2, 32]")

    outputs: List[Fraction] = []
    memberships: List[bool] = []
    for qi, pt in enumerate(raw_pts):
        if type(pt) is not list or len(pt) != m:
            raise MPNNCertificateError(f"query_points[{qi}] length != {m}")
        xv = [_parse_rat(v, f"query_points[{qi}]") for v in pt]
        fnn_val = eval_union_relu_fnn_q(polys, xv)
        if fnn_val < Fraction(0, 1):
            raise MPNNCertificateError("ReLU-FNN output negative")
        # Direct algebraic check: exists i with A_i x <= b_i
        in_union = any(
            all(
                sum((a * x for a, x in zip(row, xv)), Fraction(0, 1)) <= rhs
                for row, rhs in zip(A_mat, b_vec)
            )
            for A_mat, b_vec in polys
        )
        if (fnn_val == Fraction(0, 1)) != in_union:
            raise MPNNCertificateError(f"Lemma 5.1 soundness mismatch at query {qi}")
        outputs.append(fnn_val)
        memberships.append(in_union)

    # Require non-trivial audit covering at least one inside point and one outside point
    if not (any(memberships) and any(not b for b in memberships)):
        raise MPNNCertificateError("Mode 3 query_points must include both inside and outside points")

    if _parse_int(cert["claimed_num_polyhedra"], "claimed_num_polyhedra") != len(polys):
        raise MPNNCertificateError("claimed_num_polyhedra mismatch")
    raw_out = cert["claimed_fnn_outputs"]
    if type(raw_out) is not list or len(raw_out) != len(outputs) or [_parse_rat(v, "fnn_out") for v in raw_out] != outputs:
        raise MPNNCertificateError("claimed_fnn_outputs mismatch")
    raw_mem = cert["claimed_memberships"]
    if type(raw_mem) is not list or not all(type(b) is bool for b in raw_mem) or raw_mem != memberships:
        raise MPNNCertificateError("claimed_memberships mismatch")

    return {
        "verified": True,
        "mode": cert["mode"],
        "num_polyhedra": len(polys),
        "num_queries": len(outputs),
        "inside_count": sum(1 for b in memberships if b),
    }


# =============================================================================
# Mode 4: Theorem 6.1 & Corollary 6.2 Exact Local-Sum ReLU MPNN Execution (deg <= 3)
# =============================================================================


def _verify_mode4_local_sum_mpnn(cert: Dict[str, Any]) -> Dict[str, Any]:
    req = {
        "schema",
        "mode",
        "cert_id",
        "ntm",
        "input_word",
        "height_h",
        "num_nodes",
        "edges",
        "root_v",
        "node_labels",
        "claimed_max_degree",
        "claimed_label_dim",
        "claimed_final_e1_v",
        "claimed_final_e2_v",
        "claimed_mpnn_output",
        "claimed_accepted",
    }
    _parse_exact_keys(cert, req, "Mode 4")
    k, blank, ns, q_acc, transitions = _parse_ntm_spec(cert["ntm"])
    h = _parse_int(cert["height_h"], "height_h")
    if not (1 <= h <= 4):
        raise MPNNCertificateError("Require 1 <= height_h <= 4")
    label_dim = 14 * (h + 1)

    raw_iw = cert["input_word"]
    if type(raw_iw) is not list or len(raw_iw) > 16:
        raise MPNNCertificateError("input_word invalid")
    iw = [_parse_int(c, "input_word") for c in raw_iw]
    if any(not (1 <= c <= k) or c == blank for c in iw):
        raise MPNNCertificateError("input_word symbols must be non-blank in 1..k")
    exp_head = iw[0] if iw else blank
    exp_right = iw[1:] if iw else []
    enc_cw = enc_config_q([], 0, exp_head, exp_right, k, blank, ns)

    n_nodes = _parse_int(cert["num_nodes"], "num_nodes")
    if not (3 <= n_nodes <= 64):
        raise MPNNCertificateError("Require 3 <= num_nodes <= 64")
    root_v = _parse_int(cert["root_v"], "root_v")
    if not (0 <= root_v < n_nodes):
        raise MPNNCertificateError("root_v out of range")

    raw_edges = cert["edges"]
    if type(raw_edges) is not list or len(raw_edges) > 128:
        raise MPNNCertificateError("edges invalid")
    edges: List[Tuple[int, int]] = []
    seen_e: Set[Tuple[int, int]] = set()
    deg = [0] * n_nodes
    for ei, e in enumerate(raw_edges):
        if type(e) is not list or len(e) != 2:
            raise MPNNCertificateError(f"edges[{ei}] must be pair [u, w]")
        u, w = _parse_int(e[0], "u"), _parse_int(e[1], "w")
        if not (0 <= u < n_nodes and 0 <= w < n_nodes) or u == w:
            raise MPNNCertificateError(f"edges[{ei}] invalid endpoints or self-loop")
        canon = (u, w) if u < w else (w, u)
        if canon in seen_e:
            raise MPNNCertificateError(f"Duplicate undirected edge {canon}")
        seen_e.add(canon)
        edges.append(canon)
        deg[u] += 1
        deg[w] += 1

    max_deg = max(deg) if deg else 0
    if max_deg > 3:
        raise MPNNCertificateError(f"Graph degree {max_deg} exceeds Corollary 6.2 fixed bound d = 3")

    raw_labels = cert["node_labels"]
    if type(raw_labels) is not list or len(raw_labels) != n_nodes:
        raise MPNNCertificateError("node_labels length != num_nodes")
    labels: List[List[Fraction]] = []
    for u, lbl in enumerate(raw_labels):
        if type(lbl) is not list or len(lbl) != label_dim:
            raise MPNNCertificateError(f"node_labels[{u}] dimension != 14*(h+1)={label_dim}")
        labels.append([_parse_rat(v, f"node_labels[{u}]") for v in lbl])

    delta_polys, F_polys = build_ntm_delta_and_F_polyhedra(k, ns, q_acc, transitions)
    res = run_mpnn_Mw_exact_q(h, n_nodes, edges, root_v, labels, enc_cw, delta_polys, F_polys)

    if _parse_int(cert["claimed_max_degree"], "claimed_max_degree") != max_deg:
        raise MPNNCertificateError("claimed_max_degree mismatch")
    if _parse_int(cert["claimed_label_dim"], "claimed_label_dim") != label_dim:
        raise MPNNCertificateError("claimed_label_dim mismatch")
    if _parse_rat(cert["claimed_final_e1_v"], "claimed_final_e1_v") != res["final_e1_v"]:
        raise MPNNCertificateError("claimed_final_e1_v mismatch")
    if _parse_rat(cert["claimed_final_e2_v"], "claimed_final_e2_v") != res["final_e2_v"]:
        raise MPNNCertificateError("claimed_final_e2_v mismatch")
    if _parse_rat(cert["claimed_mpnn_output"], "claimed_mpnn_output") != res["output_v"]:
        raise MPNNCertificateError("claimed_mpnn_output mismatch")
    if cert["claimed_accepted"] is not res["accepted"]:
        raise MPNNCertificateError("claimed_accepted mismatch")

    return {
        "verified": True,
        "mode": cert["mode"],
        "nodes": n_nodes,
        "edges": len(edges),
        "max_deg": max_deg,
        "accepted": res["accepted"],
        "output": format_rat(res["output_v"]),
    }


def verify_mpnn_certificate_exact(cert: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level zero-float entry point for MPNN-DEGBSAT-CERT-1.0."""
    if type(cert) is not dict:
        raise MPNNCertificateError("Certificate root must be a JSON object")
    _assert_strict_json_types(cert)
    if cert.get("schema") != SCHEMA_VERSION:
        raise MPNNCertificateError(f"Unsupported schema: {cert.get('schema')!r}")
    cid = cert.get("cert_id")
    if type(cid) is not str or not (1 <= len(cid) <= MAX_ID_LEN):
        raise MPNNCertificateError(f"cert_id must be string of length in [1, {MAX_ID_LEN}]")
    mode = cert.get("mode")
    if mode == "tape_encoding_lemma31_cert":
        return _verify_mode1_tape_encoding(cert)
    if mode == "polyhedral_ntm_delta_lemma41_cert":
        return _verify_mode2_polyhedral_ntm(cert)
    if mode == "relu_polyhedra_union_lemma51_cert":
        return _verify_mode3_relu_polyhedra(cert)
    if mode == "local_sum_mpnn_deg3_theorem61_cert":
        return _verify_mode4_local_sum_mpnn(cert)
    raise MPNNCertificateError(f"Unknown verification mode: {mode!r}")
