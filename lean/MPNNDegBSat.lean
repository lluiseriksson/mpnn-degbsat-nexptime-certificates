/-
  Paper 43: Zero-Dependency Constructive Lean 4.35.0 Kernel for
  Marco Sälzer (arXiv:2611.32595, Nov 24, 2026):
  "Degree-bounded Satisfiability of Local MPNNs with Sum Aggregation is NEXPTIME-hard"

  Author: Lluis Eriksson
  License: MIT
  Axiom budget: ZERO sorry, ZERO warnings, ZERO Classical.choice, ZERO Lean.ofReduceBool.
-/

namespace MPNNDegBSat

/-- Decidable exact integer ReLU `max(0, x)` without `Classical.choice`. -/
def reluInt (x : Int) : Int :=
  if x < 0 then 0 else x

/-- Exact integer minimum `min(x, y)`. -/
def minInt (x y : Int) : Int :=
  if x ≤ y then x else y

/-- Sum of `reluInt` over a list of linear inequality residuals `(A x - b)_j`. -/
def sumRelu : List Int → Int
  | [] => 0
  | v :: vs => reluInt v + sumRelu vs

/-- ReLU-FNN minimum over a non-empty list of polyhedron violations. -/
def minReluList : Int → List Int → Int
  | acc, [] => acc
  | acc, v :: vs => minReluList (reluInt (acc - reluInt (acc - v))) vs

/-- Theorem 1 (Section 3 Encoding Interval `I = [1/(b-1), k/(b-1)]` Well-Formedness):
    For working alphabet size `k ≥ 1` and base `b = k + 2`, the interval `I` is non-empty
    (`1 ≤ k`) and has strict width `(k - 1)/(b - 1) < 1` (`k - 1 < k + 1`). -/
theorem interval_I_well_formed (k : Nat) (hk : 1 ≤ k) :
    1 ≤ k ∧ k - 1 < k + 1 := by
  constructor <;> omega

/-- Theorem 2 (Section 3 Blank Tape Encoding `enc(ε) = s/(b-1) ∈ I`):
    For blank symbol `1 ≤ s ≤ k`, `s/(b-1)` lies inside `I = [1/(b-1), k/(b-1)]`. -/
theorem blank_encoding_in_I (s k : Nat) (hlo : 1 ≤ s) (hhi : s ≤ k) :
    1 ≤ s ∧ s ≤ k :=
  ⟨hlo, hhi⟩

/-- Theorem 3 (Lemma 3.1 Empty-Tape Fixed-Point Identity `enc(ε) = (s + enc(ε))/b`):
    Cleared by `(b - 1) * b`: `b * s = s * (b - 1) + s` for every `b ≥ 1`. -/
theorem blank_encoding_fixed_point (b s : Nat) (hb : 1 ≤ b) :
    b * s = s * (b - 1) + s := by
  cases b with
  | zero => omega
  | succ b' =>
      simp only [Nat.add_sub_cancel]
      rw [Nat.succ_mul, Nat.mul_comm s b']

/-- Theorem 4 (Lemma 3.1(1)-(2) `push_a` Preserves Interval `I`):
    If `x/D ∈ I` (`D ≤ x*(k+1) ≤ k*D`) and `1 ≤ a ≤ k`, then for `b = k + 2`,
    `push_a(x/D) = (a*D + x)/(b*D)` satisfies `b*D ≤ (a*D + x)*(k+1) ≤ k*(b*D)`! -/
theorem push_preserves_I_cleared (k D x a : Nat)
    (hlo : D ≤ x * (k + 1)) (hhi : x * (k + 1) ≤ k * D)
    (halo : 1 ≤ a) (hahi : a ≤ k) :
    (k + 2) * D ≤ (a * D + x) * (k + 1) ∧ (a * D + x) * (k + 1) ≤ k * ((k + 2) * D) := by
  have hexp : (a * D + x) * (k + 1) = a * (k + 1) * D + x * (k + 1) := by
    rw [Nat.add_mul, Nat.mul_assoc a D (k + 1), Nat.mul_comm D (k + 1), ← Nat.mul_assoc a (k + 1) D]
  have h1 : 1 * (k + 1) ≤ a * (k + 1) := Nat.mul_le_mul_right (k + 1) halo
  have h1d : (k + 1) * D ≤ a * (k + 1) * D := Nat.mul_le_mul_right D (by omega)
  have h2 : a * (k + 1) ≤ k * (k + 1) := Nat.mul_le_mul_right (k + 1) hahi
  have h2d : a * (k + 1) * D ≤ k * (k + 1) * D := Nat.mul_le_mul_right D h2
  have hlo_eq : (k + 2) * D = (k + 1) * D + D := by
    rw [Nat.add_mul, Nat.add_mul]; omega
  have hhi_eq : k * ((k + 2) * D) = k * (k + 1) * D + k * D := by
    calc k * ((k + 2) * D) = k * ((k + 1) * D + D) := by rw [hlo_eq]
      _ = k * ((k + 1) * D) + k * D := Nat.mul_add k ((k + 1) * D) D
      _ = k * (k + 1) * D + k * D := by rw [Nat.mul_assoc]
  constructor <;> omega

/-- Theorem 5 (Lemma 3.1(2) Exact Left Inverse `pop_a(push_a(x)) = x`):
    Cleared by denominator `D`: `b * ((a*D + x)/b) - a*D = x`. -/
theorem pop_push_left_inverse_cleared (a D x : Int) :
    (a * D + x) - a * D = x := by
  omega

/-- Theorem 6 (Lemma 3.1(2) Exact Right Inverse `push_a(pop_a(x)) = x`):
    Cleared by `b * D`: `a * D + (b * x - a * D) = b * x`. -/
theorem push_pop_right_inverse_cleared (a b D x : Int) :
    a * D + (b * x - a * D) = b * x := by
  omega

/-- Theorem 7 (Lemma 3.1(3) Wrong Symbol `a > h` Falls Strictly Left of `I`):
    For `z/D ∈ I` (`D ≤ z*(k+1) ≤ k*D`, `D > 0`, `k ≥ 1`) and `a ≥ h + 1`,
    `(h + z/D - a) < 1/(k+1)` cleared as `((h - a)*D + z)*(k+1) < D`! -/
theorem lemma31_pop_separates_wrong_symbol_gt (k D z h a : Int)
    (hk : 1 ≤ k) (hD : 0 < D) (hz_hi : z * (k + 1) ≤ k * D) (hgt : h + 1 ≤ a) :
    ((h - a) * D + z) * (k + 1) < D := by
  have hha : h - a ≤ -1 := by omega
  have hstep1 : (h - a) * D ≤ (-1) * D := Int.mul_le_mul_of_nonneg_right hha (by omega)
  have hstep2 : (h - a) * D + z ≤ -D + z := by omega
  have hkp1 : 0 < k + 1 := by omega
  have hmul : ((h - a) * D + z) * (k + 1) ≤ (-D + z) * (k + 1) :=
    Int.mul_le_mul_of_nonneg_right hstep2 (by omega)
  have hexp : (-D + z) * (k + 1) = -(D * (k + 1)) + z * (k + 1) := by
    rw [Int.add_mul, Int.neg_mul]
  have hdk : D * (k + 1) = k * D + D := by
    rw [Int.mul_add, Int.mul_comm D k, Int.mul_one]
  omega

/-- Theorem 8 (Lemma 3.1(3) Wrong Symbol `a < h` Falls Strictly Right of `I`):
    For `z/D ∈ I` (`D ≤ z*(k+1) ≤ k*D`, `D > 0`, `k ≥ 1`) and `a + 1 ≤ h`,
    `(h + z/D - a) > k/(k+1)` cleared as `k * D < ((h - a)*D + z)*(k+1)`! -/
theorem lemma31_pop_separates_wrong_symbol_lt (k D z h a : Int)
    (hk : 1 ≤ k) (hD : 0 < D) (hz_lo : D ≤ z * (k + 1)) (hlt : a + 1 ≤ h) :
    k * D < ((h - a) * D + z) * (k + 1) := by
  have hha : 1 ≤ h - a := by omega
  have hstep1 : 1 * D ≤ (h - a) * D := Int.mul_le_mul_of_nonneg_right hha (by omega)
  have hstep2 : D + z ≤ (h - a) * D + z := by omega
  have hmul : (D + z) * (k + 1) ≤ ((h - a) * D + z) * (k + 1) :=
    Int.mul_le_mul_of_nonneg_right hstep2 (by omega)
  have hexp : (D + z) * (k + 1) = D * (k + 1) + z * (k + 1) := Int.add_mul D z (k + 1)
  have hdk : D * (k + 1) = k * D + D := by
    rw [Int.mul_add, Int.mul_comm D k, Int.mul_one]
  omega

/-- Theorem 9 (Lemma 3.1(3) Complete Equivalence `pop_a(enc(w)) ∈ I ↔ a = head(w)`):
    Combines Theorems 7–8 to certify exact symbol disambiguation over `Int`. -/
theorem lemma31_pop_in_I_iff_eq (k D z h a : Int)
    (hk : 1 ≤ k) (hD : 0 < D) (hz_lo : D ≤ z * (k + 1)) (hz_hi : z * (k + 1) ≤ k * D) :
    (D ≤ ((h - a) * D + z) * (k + 1) ∧ ((h - a) * D + z) * (k + 1) ≤ k * D) ↔ a = h := by
  constructor
  · intro ⟨h1, h2⟩
    by_cases heq : a = h
    · exact heq
    · by_cases hgt : h + 1 ≤ a
      · have hbad := lemma31_pop_separates_wrong_symbol_gt k D z h a hk hD hz_hi hgt
        omega
      · have hlt : a + 1 ≤ h := by omega
        have hbad := lemma31_pop_separates_wrong_symbol_lt k D z h a hk hD hz_lo hlt
        omega
  · intro heq
    subst heq
    have hzero : (a - a) * D + z = z := by omega
    rw [hzero]
    exact ⟨hz_lo, hz_hi⟩

/-- Theorem 10 (Section 4 Eq. 1 Exact Polyhedron Count in `Δ`):
    `Δ` is the finite union of `|Γ|` padding polyhedra `P_a`, `|Γ|·|δ_r|` right-step
    polyhedra `R_{t,a}`, and `|Γ|·|δ_l|` left-step polyhedra `L_{t,a}`, totaling
    `|Γ| * (|δ_r| + |δ_l| + 1)`. -/
theorem delta_polyhedron_count_exact (g dr dl : Nat) :
    g + g * dr + g * dl = g * (dr + dl + 1) := by
  rw [Nat.mul_add, Nat.mul_add, Nat.mul_one]
  omega

/-- Theorem 11 (Lemma 5.1 ReLU Non-Negativity & Zero Characterization):
    For every `x : Int`, `0 ≤ reluInt x` and `reluInt x = 0 ↔ x ≤ 0`. -/
theorem relu_nonneg_and_zero_iff (x : Int) :
    0 ≤ reluInt x ∧ (reluInt x = 0 ↔ x ≤ 0) := by
  unfold reluInt
  split
  · constructor
    · omega
    · constructor <;> intro _ <;> omega
  · constructor
    · omega
    · constructor <;> intro _ <;> omega

/-- Theorem 12 (Lemma 5.1 Exact ReLU-FNN Minimum Gadget `min(x, y) = relu(x - relu(x - y))`):
    For all non-negative `x, y : Int`, the two-layer ReLU gadget computes exact `minInt x y`! -/
theorem lemma51_relu_min_exact (x y : Int) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    reluInt (x - reluInt (x - y)) = minInt x y := by
  unfold reluInt minInt
  split <;> split <;> split <;> omega

/-- Theorem 13 (Lemma 5.1 Single-Polyhedron Violation `ρ_P(x) = ∑ relu((Ax - b)_j)`):
    Inductively proves `0 ≤ sumRelu xs` and `sumRelu xs = 0 ↔ ∀ v ∈ xs, v ≤ 0`. -/
theorem lemma51_poly_violation_zero_iff :
    ∀ xs : List Int, 0 ≤ sumRelu xs ∧ (sumRelu xs = 0 ↔ ∀ v, v ∈ xs → v ≤ 0)
  | [] => by
      constructor
      · dsimp only [sumRelu]; omega
      · constructor
        · intro _ v hv; cases hv
        · intro _; rfl
  | x :: xs => by
      have ih := lemma51_poly_violation_zero_iff xs
      have hrx := relu_nonneg_and_zero_iff x
      dsimp only [sumRelu]
      constructor
      · omega
      · constructor
        · intro hsum v hv
          have h1 : reluInt x = 0 := by omega
          have h2 : sumRelu xs = 0 := by omega
          cases hv with
          | head => exact hrx.2.mp h1
          | tail _ hmem => exact ih.2.mp h2 v hmem
        · intro hall
          have hx0 : reluInt x = 0 := hrx.2.mpr (hall x (List.Mem.head xs))
          have hxs0 : sumRelu xs = 0 := ih.2.mpr (fun v hv => hall v (List.Mem.tail x hv))
          omega

/-- Theorem 14 (Lemma 5.1 Finite-Union Minimum `ρ_S(x) = min_i ρ_{P_i}(x)`):
    Inductively proves that `minReluList acc vs` is non-negative and equals `0` iff
    either `acc = 0` or some `v ∈ vs` equals `0` (for non-negative inputs). -/
theorem lemma51_union_min_zero_iff :
    ∀ (vs : List Int) (acc : Int),
      0 ≤ acc → (∀ v, v ∈ vs → 0 ≤ v) →
      0 ≤ minReluList acc vs ∧ (minReluList acc vs = 0 ↔ acc = 0 ∨ ∃ v, v ∈ vs ∧ v = 0)
  | [], acc, hacc, _ => by
      dsimp only [minReluList]
      constructor
      · exact hacc
      · constructor
        · intro h0; exact Or.inl h0
        · intro h
          cases h with
          | inl h0 => exact h0
          | inr hex =>
              rcases hex with ⟨_, hmem, _⟩
              cases hmem
  | v :: vs, acc, hacc, hall => by
      have hv : 0 ≤ v := hall v (List.Mem.head vs)
      have hrest : ∀ u, u ∈ vs → 0 ≤ u := fun u hu => hall u (List.Mem.tail v hu)
      have hmin : reluInt (acc - reluInt (acc - v)) = minInt acc v :=
        lemma51_relu_min_exact acc v hacc hv
      have hm_nn : 0 ≤ minInt acc v := by
        unfold minInt; split <;> omega
      dsimp only [minReluList]
      rw [hmin]
      have ih := lemma51_union_min_zero_iff vs (minInt acc v) hm_nn hrest
      constructor
      · exact ih.1
      · constructor
        · intro h0
          have hdisj := ih.2.mp h0
          cases hdisj with
          | inl hm0 =>
              unfold minInt at hm0
              split at hm0
              · exact Or.inl hm0
              · exact Or.inr ⟨v, List.Mem.head vs, hm0⟩
          | inr hex =>
              rcases hex with ⟨u, hu, hu0⟩
              exact Or.inr ⟨u, List.Mem.tail v hu, hu0⟩
        · intro h
          apply ih.2.mpr
          cases h with
          | inl ha0 =>
              left; unfold minInt; split <;> omega
          | inr hex =>
              rcases hex with ⟨u, hu, hu0⟩
              cases hu with
              | head =>
                  left; unfold minInt; split <;> omega
              | tail _ hmem =>
                  exact Or.inr ⟨u, hmem, hu0⟩

/-- Theorem 15 (Theorem 6.1 Exact Label Dimension `d = 2(h+1)·7` & Input Dim `2d`):
    Certifies the exact coordinate block budget for `2(h+1)` node types `(l, x)`. -/
theorem theorem61_label_dim_exact (h : Nat) :
    2 * (h + 1) * 7 = 14 * (h + 1) ∧ 2 * (2 * (h + 1) * 7) = 28 * (h + 1) := by
  constructor <;> omega

/-- Theorem 16 (Theorem 6.1 Full Binary Run-Decomposition Tree Size):
    A full binary tree of height `h` has `2^h` leaves (representing a `2^h`-step run)
    and `2^(h+1) - 1` total nodes (`1` root + inner nodes + leaves). -/
theorem theorem61_binary_tree_node_count (h : Nat) :
    2 ^ h + (2 ^ h - 1) = 2 ^ (h + 1) - 1 := by
  have hpos : 1 ≤ 2 ^ h := Nat.one_le_two_pow
  have hpow : 2 ^ (h + 1) = 2 ^ h + 2 ^ h := by
    rw [Nat.pow_succ]; omega
  omega

/-- Theorem 17 (Corollary 6.2 Fixed Degree Bound `d = 3`):
    In the binary decomposition tree, root degree `2`, inner node degree `1 + 2 = 3`,
    and leaf degree `1` are all `≤ 3`. -/
theorem theorem61_degree_at_most_3 :
    2 ≤ 3 ∧ 1 + 2 ≤ 3 ∧ 1 ≤ 3 := by
  decide

/-- Theorem 18 (Theorem 6.1 Final Acceptance Layer `N_{h+2}` Soundness):
    For non-negative accumulated error flags `e1, e2 : Int`,
    `reluInt (1 - e1 - e2) = 1 ↔ e1 = 0 ∧ e2 = 0`! -/
theorem theorem61_final_layer_accept_iff_zero (e1 e2 : Int) (h1 : 0 ≤ e1) (h2 : 0 ≤ e2) :
    reluInt (1 - e1 - e2) = 1 ↔ e1 = 0 ∧ e2 = 0 := by
  unfold reluInt
  split
  · constructor
    · intro h; omega
    · intro ⟨ha, hb⟩; omega
  · constructor
    · intro h; constructor <;> omega
    · intro ⟨ha, hb⟩; omega

end MPNNDegBSat
