"""Zero-floating-point exact-Q primitives for Paper 43:
Marco Sälzer (arXiv:2611.32595, Nov 24, 2026)
"Degree-bounded Satisfiability of Local MPNNs with Sum Aggregation is NEXPTIME-hard".

Strict rule: zero floating-point arithmetic — Python fractions.Fraction only.

Author: Lluis Eriksson
License: MIT
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence, Tuple


def format_rat(q: Fraction) -> str:
    """Canonical reduced rational string ('n' if den == 1 else 'n/d')."""
    if q.denominator == 1:
        return str(q.numerator)
    return f"{q.numerator}/{q.denominator}"


def relu_q(x: Fraction) -> Fraction:
    """Exact ReLU(x) = max(0, x) over Q."""
    return x if x > Fraction(0, 1) else Fraction(0, 1)


def min_relu_q(x: Fraction, y: Fraction) -> Fraction:
    """Exact Lemma 5.1 ReLU-FNN minimum gadget: relu(x - relu(x - y))."""
    return relu_q(x - relu_q(x - y))


def interval_I_bounds(k: int) -> Tuple[Fraction, Fraction]:
    """Exact rational encoding interval I = [1/(b-1), k/(b-1)] with b = k + 2."""
    b = k + 2
    return Fraction(1, b - 1), Fraction(k, b - 1)


def in_interval_I(x: Fraction, k: int) -> bool:
    lo, hi = interval_I_bounds(k)
    return lo <= x <= hi


def enc_word_q(word: Sequence[int], k: int, blank_sym: int) -> Fraction:
    """Exact Section 3 rational tape encoding:
    enc(a_1 ... a_m) = sum_{i=1}^m a_i b^{-i} + blank_sym / ((b-1) b^m),  b = k + 2.
    """
    b = k + 2
    val = Fraction(0, 1)
    pow_b = 1
    for a in word:
        pow_b *= b
        val += Fraction(a, pow_b)
    val += Fraction(blank_sym, (b - 1) * pow_b)
    return val


def push_q(a: int, x: Fraction, k: int) -> Fraction:
    """Exact push_a(x) = (x + a) / b where b = k + 2."""
    return (x + Fraction(a, 1)) / Fraction(k + 2, 1)


def pop_q(a: int, x: Fraction, k: int) -> Fraction:
    """Exact pop_a(x) = b * x - a where b = k + 2."""
    return Fraction(k + 2, 1) * x - Fraction(a, 1)


def eval_polyhedron_violation_q(
    A_rows: Sequence[Sequence[Fraction]],
    b_vec: Sequence[Fraction],
    x_vec: Sequence[Fraction],
) -> Fraction:
    """Exact Lemma 5.1 single-polyhedron ReLU violation rho_P(x) = sum_j relu((A x - b)_j)."""
    total = Fraction(0, 1)
    for row, rhs in zip(A_rows, b_vec):
        dot = sum((a * xv for a, xv in zip(row, x_vec)), Fraction(0, 1))
        total += relu_q(dot - rhs)
    return total


def eval_union_relu_fnn_q(
    polyhedra: Sequence[Tuple[Sequence[Sequence[Fraction]], Sequence[Fraction]]],
    x_vec: Sequence[Fraction],
) -> Fraction:
    """Exact Lemma 5.1 ReLU-FNN N_S(x) = min_{1 <= i <= n} rho_{P_i}(x)
    computed via iterated ReLU min gadget relu(acc - relu(acc - v)).
    """
    if not polyhedra:
        return Fraction(1, 1)
    acc = eval_polyhedron_violation_q(polyhedra[0][0], polyhedra[0][1], x_vec)
    for A_rows, b_vec in polyhedra[1:]:
        v = eval_polyhedron_violation_q(A_rows, b_vec, x_vec)
        acc = min_relu_q(acc, v)
    return acc


def state_sym_code_q(q: int, a: int, num_states: int, k: int) -> Fraction:
    """Deterministic injective rational encoding enc(q, a) in [0, 1] for Q x Gamma."""
    idx = q * k + (a - 1)
    denom = max(1, num_states * k)
    return Fraction(idx, denom)


def build_ntm_delta_and_F_polyhedra(
    k: int,
    num_states: int,
    q_accept: int,
    transitions: Sequence[Tuple[int, int, int, int, str]],
) -> Tuple[
    List[Tuple[List[List[Fraction]], List[Fraction]]],
    List[Tuple[List[List[Fraction]], List[Fraction]]],
]:
    """Construct the exact rational polyhedron families Delta (in Q^6) and F (in Q^3)
    from Section 4 Eq. (1):
      - R_{t, a2} for right transitions t = (q1, a1, q2, a_prime, 'R'), a2 in 1..k
      - L_{t, a2} for left transitions t = (q1, a1, q2, a_prime, 'L'), a2 in 1..k
      - P_a for accepting padding a in 1..k
    Vector layout for Delta: [x1, h1, y1, x2, h2, y2].
    Vector layout for F: [x, h, y].
    """
    lo, hi = interval_I_bounds(k)
    b = Fraction(k + 2, 1)
    inv_b = Fraction(1, k + 2)
    z0 = Fraction(0, 1)
    one = Fraction(1, 1)
    neg1 = Fraction(-1, 1)

    def add_eq(A: List[List[Fraction]], bv: List[Fraction], row: List[Fraction], rhs: Fraction) -> None:
        A.append(row)
        bv.append(rhs)
        A.append([-c for c in row])
        bv.append(-rhs)

    def add_box_I_6(A: List[List[Fraction]], bv: List[Fraction]) -> None:
        for idx in (0, 2, 3, 5):
            r_hi = [z0] * 6
            r_hi[idx] = one
            A.append(r_hi)
            bv.append(hi)
            r_lo = [z0] * 6
            r_lo[idx] = neg1
            A.append(r_lo)
            bv.append(-lo)

    delta_polys: List[Tuple[List[List[Fraction]], List[Fraction]]] = []

    # 1. Accepting padding polyhedra P_a (|Gamma| polyhedra)
    for a in range(1, k + 1):
        A: List[List[Fraction]] = []
        bv: List[Fraction] = []
        add_box_I_6(A, bv)
        h_acc = state_sym_code_q(q_accept, a, num_states, k)
        add_eq(A, bv, [z0, one, z0, z0, z0, z0], h_acc)
        add_eq(A, bv, [z0, z0, z0, z0, one, z0], h_acc)
        add_eq(A, bv, [one, z0, z0, neg1, z0, z0], z0)  # x1 = x2
        add_eq(A, bv, [z0, z0, one, z0, z0, neg1], z0)  # y1 = y2
        delta_polys.append((A, bv))

    # 2. Transition polyhedra R_{t, a2} and L_{t, a2} (|delta| * |Gamma| polyhedra)
    for q1, a1, q2, a_prime, direction in transitions:
        h1_code = state_sym_code_q(q1, a1, num_states, k)
        for a2 in range(1, k + 1):
            A = []
            bv = []
            add_box_I_6(A, bv)
            h2_code = state_sym_code_q(q2, a2, num_states, k)
            add_eq(A, bv, [z0, one, z0, z0, z0, z0], h1_code)
            add_eq(A, bv, [z0, z0, z0, z0, one, z0], h2_code)
            if direction == "R":
                # x2 = (x1 + a_prime)/b  <=>  x2 - (1/b) x1 = a_prime / b
                add_eq(A, bv, [-inv_b, z0, z0, one, z0, z0], Fraction(a_prime, k + 2))
                # y2 = b * y1 - a2       <=>  y2 - b * y1 = -a2
                add_eq(A, bv, [z0, z0, -b, z0, z0, one], Fraction(-a2, 1))
            elif direction == "L":
                # y2 = (y1 + a_prime)/b  <=>  y2 - (1/b) y1 = a_prime / b
                add_eq(A, bv, [z0, z0, -inv_b, z0, z0, one], Fraction(a_prime, k + 2))
                # x2 = b * x1 - a2       <=>  x2 - b * x1 = -a2
                add_eq(A, bv, [-b, z0, z0, one, z0, z0], Fraction(-a2, 1))
            else:
                raise ValueError(f"Invalid NTM move direction: {direction!r}")
            delta_polys.append((A, bv))

    # 3. Accepting set F (|Gamma| polyhedra in Q^3)
    F_polys: List[Tuple[List[List[Fraction]], List[Fraction]]] = []
    for a in range(1, k + 1):
        A = []
        bv = []
        for idx in (0, 2):
            r_hi = [z0] * 3
            r_hi[idx] = one
            A.append(r_hi)
            bv.append(hi)
            r_lo = [z0] * 3
            r_lo[idx] = neg1
            A.append(r_lo)
            bv.append(-lo)
        h_acc = state_sym_code_q(q_accept, a, num_states, k)
        add_eq(A, bv, [z0, one, z0], h_acc)
        F_polys.append((A, bv))

    return delta_polys, F_polys


def enc_config_q(
    left_tape: Sequence[int],
    state: int,
    head_sym: int,
    right_tape: Sequence[int],
    k: int,
    blank_sym: int,
    num_states: int,
) -> List[Fraction]:
    """Encode NTM configuration c = (l, (q, a), r) as enc(c) in Q^3."""
    return [
        enc_word_q(left_tape, k, blank_sym),
        state_sym_code_q(state, head_sym, num_states, k),
        enc_word_q(right_tape, k, blank_sym),
    ]


def abs_diff_q(a: Fraction, b: Fraction) -> Fraction:
    """Exact |a - b| = relu(a - b) + relu(b - a) realizable by a single ReLU pair."""
    return relu_q(a - b) + relu_q(b - a)


def run_mpnn_Mw_exact_q(
    height_h: int,
    num_nodes: int,
    edges: Sequence[Tuple[int, int]],
    root_v: int,
    node_labels: Sequence[Sequence[Fraction]],
    enc_cw: Sequence[Fraction],
    delta_polys: Sequence[Tuple[Sequence[Sequence[Fraction]], Sequence[Fraction]]],
    F_polys: Sequence[Tuple[Sequence[Sequence[Fraction]], Sequence[Fraction]]],
) -> Dict[str, object]:
    """Exact-Q execution of Theorem 6.1 local-sum ReLU MPNN M_w = (N_1, ..., N_{h+2})
    on pointed graph (G, v).
    Label vector dimension per node: d = 2 * (h + 1) * 7 = 14 * (h + 1).
    Type index for (l, side) with l in 0..h, side in {0 (L), 1 (R)}:
      tid = 2 * l + side,  block slice [7*tid : 7*(tid+1)] = [z, A0, A1, A2, B0, B1, B2].
    """
    num_types = 2 * (height_h + 1)
    dim = 7 * num_types
    z0 = Fraction(0, 1)
    one = Fraction(1, 1)

    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, w in edges:
        adj[u].append(w)
        adj[w].append(u)

    # Layer 1: compute non-negative error flags (e1_u, e2_u) via exact Lemma 5.1 ReLU-FNNs
    e1: List[Fraction] = []
    e2: List[Fraction] = []

    for u in range(num_nodes):
        lbl = node_labels[u]
        agg = [z0] * dim
        for nb in adj[u]:
            nb_lbl = node_labels[nb]
            for j in range(dim):
                agg[j] += nb_lbl[j]

        # Flag 1: min over all 2(h+1) valid types t = (l, side) of local condition violation
        best_e1: Fraction | None = None
        for l in range(height_h + 1):
            for side in (0, 1):
                tid = 2 * l + side
                off = 7 * tid
                # Type-t conditions: z_t = 1, A_t, B_t in [0, 1]^3, all other blocks = 0
                viol = abs_diff_q(lbl[off], one)
                for j in range(1, 7):
                    viol += relu_q(-lbl[off + j]) + relu_q(lbl[off + j] - one)
                for other_tid in range(num_types):
                    if other_tid != tid:
                        o2 = 7 * other_tid
                        for j in range(7):
                            viol += abs_diff_q(lbl[o2 + j], z0)

                Au = lbl[off + 1 : off + 4]
                Bu = lbl[off + 4 : off + 7]

                if l < height_h:
                    off_L = 7 * (2 * (l + 1) + 0)
                    off_R = 7 * (2 * (l + 1) + 1)
                    viol += abs_diff_q(agg[off_L], one)
                    viol += abs_diff_q(agg[off_R], one)
                    for j in range(3):
                        viol += abs_diff_q(Au[j], agg[off_L + 1 + j])  # A_u = A'_{(l+1, L)}
                        viol += abs_diff_q(Bu[j], agg[off_R + 4 + j])  # B_u = B'_{(l+1, R)}
                        viol += abs_diff_q(agg[off_L + 4 + j], agg[off_R + 1 + j])  # B'_L = A'_R
                else:
                    # Leaf check: (A_u, B_u) in Delta via exact Lemma 5.1 ReLU-FNN
                    viol += eval_union_relu_fnn_q(delta_polys, list(Au) + list(Bu))

                best_e1 = viol if best_e1 is None else min_relu_q(best_e1, viol)

        assert best_e1 is not None
        e1.append(best_e1)

        # Flag 2 (root condition): min over side in {L, R} of [type=(0,side), A_u=enc(c_w), B_u in F]
        best_e2: Fraction | None = None
        for side in (0, 1):
            tid = side
            off = 7 * tid
            viol2 = abs_diff_q(lbl[off], one)
            for other_tid in range(num_types):
                if other_tid != tid:
                    o2 = 7 * other_tid
                    for j in range(7):
                        viol2 += abs_diff_q(lbl[o2 + j], z0)
            Au = lbl[off + 1 : off + 4]
            Bu = lbl[off + 4 : off + 7]
            for j in range(3):
                viol2 += abs_diff_q(Au[j], enc_cw[j])
            viol2 += eval_union_relu_fnn_q(F_polys, list(Bu))
            best_e2 = viol2 if best_e2 is None else min_relu_q(best_e2, viol2)

        assert best_e2 is not None
        e2.append(best_e2)

    # Layers 2 .. h+1: propagate non-negative flag 1 via local sum aggregation; preserve flag 2
    cur_e1 = list(e1)
    for _ in range(height_h):
        nxt_e1: List[Fraction] = []
        for u in range(num_nodes):
            s = cur_e1[u]
            for nb in adj[u]:
                s += cur_e1[nb]
            nxt_e1.append(s)
        cur_e1 = nxt_e1

    # Final layer N_{h+2}: M_w(G, v) = relu(1 - e1_v^{(h+1)} - e2_v^{(h+1)})
    final_e1_v = cur_e1[root_v]
    final_e2_v = e2[root_v]
    out_v = relu_q(one - final_e1_v - final_e2_v)
    return {
        "layer1_e1": e1,
        "layer1_e2": e2,
        "final_e1_v": final_e1_v,
        "final_e2_v": final_e2_v,
        "output_v": out_v,
        "accepted": out_v == one,
    }
