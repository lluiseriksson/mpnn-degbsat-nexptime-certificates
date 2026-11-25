# Certified Verification of NEXPTIME-Hardness for Degree-Bounded Local-Sum MPNN Satisfiability

[![License: MIT](https://img.shields.io/badge/Code%20%26%20Lean-MIT-blue.svg)](LICENSE)
[![Paper License: CC-BY-4.0](https://img.shields.io/badge/Paper-CC--BY--4.0-green.svg)](paper/main.tex)

**Author:** Lluis Eriksson

Machine-checked **Lean 4.35.0** formalization and zero-floating-point exact-$\mathbb{Q}$ certificate audit (`MPNN-DEGBSAT-CERT-1.0`) for **Marco Sälzer**'s breakthrough `NEXPTIME`-hardness proof for degree-bounded satisfiability (`DEGBSAT`) of local-sum ReLU Message Passing Neural Networks at fixed degree bound $d = 3$ ([arXiv:2611.32595](https://arxiv.org/abs/2611.32595), Nov 24, 2026).

---

## Key Verified Artifacts

| Artifact | Verification Metric | Result |
| :--- | :--- | :--- |
| **Lean 4.35.0 Kernel** (`lean/MPNNDegBSat.lean`) | 18 theorems, zero `sorry`, zero warnings, zero `Classical.choice` | **18 / 18 PASS** (`2` `none`, `16` `[propext, Quot.sound]`) |
| **Exact-$\mathbb{Q}$ Verifier** (`src/mpnn_certifier.py`) | Zero `float()`, `math`, or `numpy`; 4-mode `MPNN-DEGBSAT-CERT-1.0` | **16 / 16 PASS** (`0.080–114.497 ms`) |
| **Adversarial Red-Team Suite** (`tests/test_suite.py`) | 62 live forged payloads (12 global + 12–13/mode) | **62 / 62 REJECTED** |
| **SHA-256 Integrity Manifest** (`SHA256_MANIFEST.json`) | Full deterministic replay check (`scripts/verify_all.py`) | **41 / 41 MATCH** |
| **Manuscript** (`paper/main.pdf`) | 3 pages, 3 publication figures (`300 DPI`) | **0 warnings, 0 overfulls** |

---

## Four Certificate Modes (`MPNN-DEGBSAT-CERT-1.0`)

1. **`tape_encoding_lemma31_cert` (Mode 1 — Lemma 3.1):** Verifies exact base-$b$ ($b = k + 2$) rational tape encoding $\operatorname{enc}(w) \in I = [1/(b-1), k/(b-1)]$, exact $\operatorname{push}_a$ / $\operatorname{pop}_{\operatorname{head}(w)}$ identities, and exhaustive symbol disambiguation $\operatorname{pop}_a(\operatorname{enc}(w)) \in I \iff a = \operatorname{head}(w)$ over all $a \in \{1,\dots,k\}$.
2. **`polyhedral_ntm_delta_lemma41_cert` (Mode 2 — Lemma 4.1):** Constructs the exact rational polyhedron families $\Delta$ ($|\Gamma|(|\delta|+1)$ polyhedra $R_{t,a}, L_{t,a}, P_a$ in $\mathbb{Q}^6$) and $\mathcal{F}$ ($|\Gamma|$ polyhedra in $\mathbb{Q}^3$) and certifies complete $2^h$-step (possibly padded) NTM accepting runs from $\operatorname{enc}(c_w)$ to $\mathcal{F}$.
3. **`relu_polyhedra_union_lemma51_cert` (Mode 3 — Lemma 5.1):** Evaluates exact rational ReLU-FNNs $N_S(x) = \min_i \sum_j \operatorname{ReLU}((A_i x - b_i)_j)$ via the iterated ReLU minimum gadget $\operatorname{ReLU}(u - \operatorname{ReLU}(u - v))$ on non-trivial inside and outside query sets, verifying $N_S(x) \ge 0$ and $N_S(x) = 0 \iff x \in S$.
4. **`local_sum_mpnn_deg3_theorem61_cert` (Mode 4 — Theorem 6.1 & Corollary 6.2):** Executes Sälzer's full $(h+2)$-layer local-sum ReLU MPNN $M_w = (N_1, \dots, N_{h+2})$ over exact $\mathbb{Q}$ on explicit pointed graphs $(G, v)$ of degree $\le 3$ with $14(h+1)$-dimensional block labels, certifying both accepting runs ($M_w(G,v) = 1$) and corrupted-leaf rejection ($M_w(G,v) = 2/5 \ne 1$).

---

## Quick Reproduction

```bash
cd lean && lake build && lake env lean AuditAxioms.lean && cd ..
python scripts/run_all_experiments.py
python scripts/make_figures.py
python scripts/verify_all.py
```

---

## Licenses & Transparency Disclosure

- **Code, Lean proofs, certificates, and scripts:** [MIT License](LICENSE).
- **Manuscript (`paper/main.tex`, `paper/main.pdf`) and figures:** CC-BY-4.0.
- **AI Assistance Disclosure:** Prepared by sole author Lluis Eriksson with AI coding, formalization, and drafting assistance from Google Antigravity; all Lean proofs, exact-$\mathbb{Q}$ certificates, and adversarial tests were independently compiled and machine-verified locally.
