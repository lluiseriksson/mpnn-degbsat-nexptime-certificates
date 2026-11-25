"""Comprehensive pytest & 60-attack live adversarial red-team suite for Paper 43:
MPNN-DEGBSAT-CERT-1.0 (Marco Sälzer, arXiv:2611.32595).

Author: Lluis Eriksson
License: MIT
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mpnn_certifier import MPNNCertificateError, verify_mpnn_certificate_exact

CERT_DIR = ROOT / "certificates"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((CERT_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_all_16_canonical_certificates_verify() -> None:
    paths = sorted(CERT_DIR.glob("*.json"))
    assert len(paths) == 16, f"Expected 16 certificates, found {len(paths)}"
    for p in paths:
        res = verify_mpnn_certificate_exact(json.loads(p.read_text(encoding="utf-8")))
        assert res["verified"] is True


def _mut(base: Dict[str, Any], fn: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    c = copy.deepcopy(base)
    fn(c)
    return c


def build_60_adversarial_attacks() -> List[Tuple[str, Any]]:
    m1 = _load("m1_tape_k3_word_121")
    m2 = _load("m2_ntm_h2_4steps_padded")
    m3 = _load("m3_relu_2d_disjoint_boxes")
    m4 = _load("m4_mpnn_h2_tree_deg3_accept")

    attacks: List[Tuple[str, Any]] = [
        # --- Global Schema / Type / Rational Hardening (12 attacks) ---
        ("g01_non_dict_root", [1, 2, 3]),
        ("g02_wrong_schema", _mut(m1, lambda c: c.__setitem__("schema", "MPNN-DEGBSAT-CERT-2.0"))),
        ("g03_empty_cert_id", _mut(m1, lambda c: c.__setitem__("cert_id", ""))),
        ("g04_unknown_mode", _mut(m1, lambda c: c.__setitem__("mode", "bogus"))),
        ("g05_float_leak", _mut(m3, lambda c: c["query_points"][0].__setitem__(0, 0.5))),
        ("g06_none_injection", _mut(m1, lambda c: c.__setitem__("alphabet_k", None))),
        ("g07_bool_as_int", _mut(m1, lambda c: c.__setitem__("blank_sym", True))),
        ("g08_unreduced_fraction", _mut(m3, lambda c: c["query_points"][0].__setitem__(0, "2/4"))),
        ("g09_den_one_fraction", _mut(m3, lambda c: c["query_points"][2].__setitem__(0, "2/1"))),
        ("g10_trailing_newline_rat", _mut(m3, lambda c: c["query_points"][0].__setitem__(0, "1/2\n"))),
        ("g11_oversized_rat_str", _mut(m3, lambda c: c["query_points"][0].__setitem__(0, "1/" + "9" * 1200))),
        ("g12_huge_int_bitlen", _mut(m1, lambda c: c.__setitem__("alphabet_k", 1 << 3000))),
        # --- Mode 1: Lemma 3.1 Tape Encoding & Pop Disambiguation (12 attacks) ---
        ("m1_01_extra_key", _mut(m1, lambda c: c.__setitem__("extra", 1))),
        ("m1_02_k_too_small", _mut(m1, lambda c: c.__setitem__("alphabet_k", 1))),
        ("m1_03_blank_out_of_range", _mut(m1, lambda c: c.__setitem__("blank_sym", 4))),
        ("m1_04_push_sym_zero", _mut(m1, lambda c: c.__setitem__("push_sym", 0))),
        ("m1_05_word_sym_zero", _mut(m1, lambda c: c["word"].__setitem__(0, 0))),
        ("m1_06_word_too_long", _mut(m1, lambda c: c.__setitem__("word", [1] * 20))),
        ("m1_07_wrong_enc_word", _mut(m1, lambda c: c.__setitem__("claimed_enc_word", "1/2"))),
        ("m1_08_wrong_enc_pushed", _mut(m1, lambda c: c.__setitem__("claimed_enc_pushed", "1/2"))),
        ("m1_09_wrong_enc_tail", _mut(m1, lambda c: c.__setitem__("claimed_enc_tail", "1/2"))),
        ("m1_10_wrong_head_sym", _mut(m1, lambda c: c.__setitem__("claimed_head_sym", 2))),
        ("m1_11_wrong_pop_flags", _mut(m1, lambda c: c.__setitem__("claimed_pop_in_I_flags", [False, True, False]))),
        ("m1_12_pop_flags_wrong_len", _mut(m1, lambda c: c["claimed_pop_in_I_flags"].append(False))),
        ("m1_13_int_as_bool_pop_flags", _mut(m1, lambda c: c.__setitem__("claimed_pop_in_I_flags", [1, 0, 0]))),
        # --- Mode 2: Lemma 4.1 Polyhedral NTM Delta^{2^h} & F (12 attacks) ---
        ("m2_01_extra_key", _mut(m2, lambda c: c.__setitem__("extra", 1))),
        ("m2_02_invalid_dir", _mut(m2, lambda c: c["ntm"]["transitions"][0].__setitem__("dir", "S"))),
        ("m2_03_duplicate_transition", _mut(m2, lambda c: c["ntm"]["transitions"].append(c["ntm"]["transitions"][0]))),
        ("m2_04_blank_in_input", _mut(m2, lambda c: c["input_word"].append(3))),
        ("m2_05_wrong_init_config", _mut(m2, lambda c: c["run_configs"][0].__setitem__("head", 2))),
        ("m2_06_invalid_step_not_in_delta", _mut(m2, lambda c: c["run_configs"][1].__setitem__("state", 2))),
        ("m2_07_non_accepting_final", _mut(m2, lambda c: c["run_configs"][-1].__setitem__("state", 1))),
        ("m2_08_wrong_chain_len", _mut(m2, lambda c: c["run_configs"].pop())),
        ("m2_09_wrong_delta_count", _mut(m2, lambda c: c.__setitem__("claimed_delta_poly_count", 1))),
        ("m2_10_wrong_enc_init", _mut(m2, lambda c: c["claimed_enc_init"].__setitem__(0, "1/2"))),
        ("m2_11_wrong_enc_final", _mut(m2, lambda c: c["claimed_enc_final"].__setitem__(0, "1/2"))),
        ("m2_12_false_final_in_F", _mut(m2, lambda c: c.__setitem__("claimed_final_in_F", False))),
        # --- Mode 3: Lemma 5.1 ReLU-FNN Union of Polyhedra (12 attacks) ---
        ("m3_01_extra_key", _mut(m3, lambda c: c.__setitem__("extra", 1))),
        ("m3_02_dim_zero", _mut(m3, lambda c: c.__setitem__("ambient_dim_m", 0))),
        ("m3_03_empty_polyhedra", _mut(m3, lambda c: c.__setitem__("polyhedra", []))),
        ("m3_04_A_b_len_mismatch", _mut(m3, lambda c: c["polyhedra"][0]["b"].pop())),
        ("m3_05_A_row_width_mismatch", _mut(m3, lambda c: c["polyhedra"][0]["A"][0].append("0"))),
        ("m3_06_query_dim_mismatch", _mut(m3, lambda c: c["query_points"][0].append("0"))),
        ("m3_07_all_inside_trivial", _mut(m3, lambda c: (c.__setitem__("query_points", c["query_points"][:2]), c.__setitem__("claimed_fnn_outputs", ["0", "0"]), c.__setitem__("claimed_memberships", [True, True])))),
        ("m3_08_all_outside_trivial", _mut(m3, lambda c: (c.__setitem__("query_points", c["query_points"][2:]), c.__setitem__("claimed_fnn_outputs", c["claimed_fnn_outputs"][2:]), c.__setitem__("claimed_memberships", [False, False])))),
        ("m3_09_wrong_num_polys", _mut(m3, lambda c: c.__setitem__("claimed_num_polyhedra", 1))),
        ("m3_10_wrong_fnn_output", _mut(m3, lambda c: c["claimed_fnn_outputs"].__setitem__(2, "0"))),
        ("m3_11_wrong_membership", _mut(m3, lambda c: c["claimed_memberships"].__setitem__(0, False))),
        ("m3_12_single_query_only", _mut(m3, lambda c: c.__setitem__("query_points", c["query_points"][:1]))),
        ("m3_13_int_as_bool_memberships", _mut(m3, lambda c: c.__setitem__("claimed_memberships", [1, 1, 0, 0]))),
        # --- Mode 4: Theorem 6.1 Local-Sum ReLU MPNN Execution (12 attacks) ---
        ("m4_01_extra_key", _mut(m4, lambda c: c.__setitem__("extra", 1))),
        ("m4_02_degree_4_exceeds_3", _mut(m4, lambda c: (c["edges"].append([1, 4]), c["edges"].append([1, 5])))),
        ("m4_03_self_loop_edge", _mut(m4, lambda c: c["edges"].append([0, 0]))),
        ("m4_04_duplicate_edge", _mut(m4, lambda c: c["edges"].append([1, 0]))),
        ("m4_05_root_out_of_range", _mut(m4, lambda c: c.__setitem__("root_v", 99))),
        ("m4_06_label_dim_wrong", _mut(m4, lambda c: c["node_labels"][0].pop())),
        ("m4_07_wrong_max_deg", _mut(m4, lambda c: c.__setitem__("claimed_max_degree", 2))),
        ("m4_08_wrong_label_dim", _mut(m4, lambda c: c.__setitem__("claimed_label_dim", 14))),
        ("m4_09_wrong_final_e1", _mut(m4, lambda c: c.__setitem__("claimed_final_e1_v", "1"))),
        ("m4_10_wrong_final_e2", _mut(m4, lambda c: c.__setitem__("claimed_final_e2_v", "1"))),
        ("m4_11_wrong_output_accept", _mut(m4, lambda c: c.__setitem__("claimed_mpnn_output", "0"))),
        ("m4_12_forged_accept_on_corrupted", _mut(m4, lambda c: c["node_labels"][0].__setitem__(1, "1/2"))),
    ]
    assert len(attacks) == 62
    return attacks


def test_62_live_adversarial_attacks_rejected() -> None:
    attacks = build_60_adversarial_attacks()
    rejected = 0
    for name, payload in attacks:
        try:
            verify_mpnn_certificate_exact(payload)
        except MPNNCertificateError:
            rejected += 1
        else:
            raise AssertionError(f"Adversarial attack {name!r} was NOT rejected!")
    assert rejected == 62


def test_exact_no_float_source_audit() -> None:
    for rel in ["src/mpnn_exact.py", "src/mpnn_certifier.py"]:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "float(" not in src, f"float() found in {rel}"
        assert "math." not in src, f"math. found in {rel}"
        assert "numpy" not in src, f"numpy found in {rel}"
