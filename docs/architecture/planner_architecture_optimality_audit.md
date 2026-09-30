# Rust Planning Architecture Optimality and Authenticity Audit

Audit date: 2026-09-27. Worktree: `/workspace/pj/gen-zero-worktree/b0927-opt-planner-audit`. Code baseline: `025360f5c1699e13312b885874d66282a0ad89d2`. `git status --short` was empty before the audit. This pass only adds the report and audit evidence; it does not patch the production implementation, and no commit or push was made.

## 1. Final Verdict

**The current implementation should be classified as [severely suboptimal / structurally deficient].** This is not a claim that another architecture has been proven optimal across all tasks; rather, the current system does not even satisfy the necessary conditions for "six genuine planning paradigms + end-to-end formal gating." Keeping an independent safety-adjudication layer and consolidating duplicate implementations is a reasonable engineering direction; but fixing the count at six named engines has neither a mathematical optimality proof nor can it substitute for a capability acceptance test.

The most damning problem is not that the six engines are missing a seventh, but rather:

1. **Algorithmic identity is misrepresented.** MCTS is a depth-one PUCT bandit; A* is a single-step sort; GFlowNet has no flow/TB training or sampling; CFR has no game tree, information sets, accumulated counterfactual regret, or average strategy; CP-SAT has no SAT, constraint propagation, or branch-and-bound. Only CEM genuinely performs finite-horizon candidate-trajectory evaluation, but it is a discrete categorical variant, not a continuous Gaussian MPC.
2. **The formal boundary is not threaded through end to end.** The planner does not pass concurrency context, agent id, or heat certificate; PolicyGate itself does not receive the predicted successor, nor does it call any Nanocore invariant. Passing a local rule does not imply dynamic trajectory safety.
3. **There are reproduced fail-closed breaches.** A release-mode integer overflow can let a violated linear constraint pass; MCTS swallows a NaN reward the model returns as `Ok`; both MCTS and CFR accept a NaN successor.
4. **"High concurrency" and "2ms" cannot be inferred from the structure's name.** The committee is internally serial, and MCTS simulations are serial; TypedArena allocation is guarded by a mutex; planning calls have no deadline parameter or actual timeout check. Measured, the default CEM median already exceeds 2ms.
5. **The validation environment itself is weak.** The default "real Rust world model" is a linear contraction system with fixed sinusoidal perturbation, not a learned dynamics model validated on task data. Passing tests proves the software path is executable, not that real decision-making capability exists.

### 1.1 What is implemented

- Six `PlanningEngine` implementations plus a unified `ProductionPipeline`, seven selectable decide modes; basic numeric validation, model `Err` propagation, and returning `NoFeasibleAction` when every action is rejected by the gate are all backed by real code and tests.
- PUCT visit counting, finite-horizon categorical CEM, immediate-reward/distance scoring, a single round of regret matching, and linear inequality checking genuinely exist.
- Graph revocation checks, confirmation/escalation tiers, rejection for missing evidence on registered heat requirements, and independent recomputation/validation of the terminal certificate.
- Nodes with 64-byte/64-byte alignment, a pre-allocated arena, pre-release initialization, atomic reads/writes with serial allocation; this is not full parallel MCTS.
- Actual compile/test run for this audit: **112 tests passed, 0 failed**; an independent release probe compiled and ran successfully. See Section 8 for details.

### 1.2 Unverified

- Planning success rate on real tasks, long-horizon performance, calibration under a stochastic model, adversarial exploitability, and capability retention from Python to Rust.
- End-to-end P99/P999 under production load, WCET, sustained high concurrency, NUMA/allocator/cache contention, executor confirmation, and landing of safe actions.
- Any claim of "theoretically optimal," "SOTA," "all critical state machines covered," or "99.5% capability retention." This document runs no industry-wide leaderboard experiment, and does not treat supplementary literature as an up-to-date SOTA ranking.

### 1.3 Incomplete / fatal defects

- P0: gate integer overflow; model numeric contamination can pass through some engines; safety approval has no closed-loop semantics for successor/concurrency context.
- P1: multi-step MCTS/A*, genuine CFR/GFlowNet/CP-SAT, continuous action optimization, deadlines, and a verifiable safe incumbent are all incomplete.
- P1: heat-certificate support remains confined to a standalone gate API; the pipeline has no certificate entry point; an invariant not wired to Nanocore cannot be advertised externally as "formalized."
- P2: uncalibrated entropy is reused across engines; the committee has fixed vote weights and order bias; an arena is constructed and a success-path string is allocated on every call.

## 2. Evidence Boundary and Method

The primary thread inspected, section by section, all six engine implementations in `engine.rs`, the decision and rollout logic in `pipeline.rs`, the decision chain in `policy.rs`, and `constraint.rs`, and traced `WorldModelDynamics`, the default dynamics, and the service-pipeline wiring. Two read-only Luna side threads separately checked the Python capability mapping and the router/tree/historical benchmarks; the primary thread integrated the final verdict. Search hits were never treated as proof of implementation, and no legacy Python benchmark was substituted for actual Rust measurement.

Evidence directory: `planner_audit_evidence/` (historical evidence, not included in the current commit). It contains raw test logs, the release probe log, probe source code, the dependency lock file, environment information, and reproduction scripts. The probes use unmodified repository crates; fault models are explicitly labeled as synthetic input. **A probe exiting cleanly only means the observation succeeded, not that the audited behavior is safe.**

The `path:line` references in this report correspond to the HEAD noted above; line ranges describe locatable entry points. Judgments of "not implemented" are strictly scoped to the call chains actually read: the existence of a same-named math utility in another crate does not imply the six engines call it.

## 3. Raw Implementation and Real Capability of the Six Engines

| Name | What is actually computed | Missing algorithmic core | Conclusion and code location |
|---|---|---|---|
| MctsEngine | Root plus one child per legal action; 128 default simulations, each calling `step` from the same root and backing up only the immediate reward; action chosen by visit count | Successor expansion, rollout/value bootstrap, terminal safety semantics, tree reuse, parallel workers | An effective single-step bandit prototype, not full MCTS. `crates/gen-zero-planner/src/engine.rs:85`, `:144`, `:168`, `:206`, `:221` |
| AStarEngine | One `step` per action; minimizes `-r + 0.05 λ ||s'-s|| + 0.1`; the BinaryHeap is popped only once | Goal predicate, accumulated g, admissible h, closed/reopen, path reconstruction, multi-level frontier | A heap-sorted one-step cost ranker, not A* graph search. `crates/gen-zero-planner/src/engine.rs:286`, `:338`, `:354`, `:366` |
| MpcCemEngine | Default 32 samples × 3 iterations × horizon 4; categorical-distribution sampling, elite update, returns the first action of the best trajectory | Continuous control variables, Gaussian mean/covariance, an independent distribution per timestep, warm start, recursive feasibility | Genuinely performs multi-step computation, but it is only a constrained discrete CEM. `crates/gen-zero-planner/src/engine.rs:403`, `:460`, `:471`, `:497`, `:544` |
| ManifoldGFlowNetEngine | Argmax of `reward - 0.1 * L2 distance`; softmax is used only to produce output entropy | P_F/P_B, Z, trajectory-balance loss, learning, reward-proportional stochastic sampling, mixed-curvature operations | Name does not match algorithm. `crates/gen-zero-planner/src/engine.rs:598`, `:625`, `:644` |
| CfrNashEngine | Single-step reward per action; subtracts the mean reward, keeps the positive part, normalizes, and returns the action with the highest positive regret | Multi-player payoffs, information sets, reach probabilities, counterfactual values, iterative accumulated regret, average strategy, exploitability | Single-round regret-shaped greedy, not a CFR/Nash solver. `crates/gen-zero-planner/src/engine.rs:661`, `:684`, `:700` |
| CpSatFormalEngine | The gate filters each candidate, the model is stepped once per candidate, and the action with the highest immediate reward is returned; entropy is always 0 | Constraint model variables and search space, SAT/CP propagation, branch-and-bound, optimality bounds, solver statuses | A gate wrapper plus greedy selection; not a CP-SAT/ILP solver. `crates/gen-zero-planner/src/engine.rs:745`, `:801`, `:819` |

**None of the six can be signed off as the full production-grade algorithm its name implies.** This does not mean all six are mocks that return constants: they perform real arithmetic, call a replaceable model, and do change the chosen action. The problem is a mismatch between capability tier, naming, and burden of proof. The code already carries honest corrective documentation in places, e.g. `engine.rs:85`, `:286`, and `pipeline.rs:164`; but the file header and several sections still retain names/promises beyond what is implemented.

### 3.1 Directly Derivable Capability Overlap

Let F denote the set of legal candidates, r(a) the one-step reward from the fixed root state, and d(a)=||s'(a)-s||.

- A* is in fact `argmax_F [r(a) - 0.05 λ d(a)]`; GFlowNet is in fact `argmax_F [r(a) - 0.1 d(a)]`. **Setting A*'s λ to 2 makes its action objective identical to GFlowNet's** (ignoring floating-point implementation/tie-break order differences; the entropy temperature still differs). The default λ=0.5 is just a different weighting within the same objective family.
- CFR's `R(a)=max(r(a)-mean(r),0)` does not change which action has the highest reward. Aside from tie-breaking, it selects the same optimal-reward action as the CP-SAT wrapper's immediate greedy choice. This follows directly from the code's formula and needs no appeal to CFR convergence theorems.
- Under the current deterministic single-step model, MCTS repeatedly re-estimates the same immediate reward for the same action; more simulations do not produce multi-step capability. With a stochastic model, repeated sampling can estimate the immediate mean, but there is no risk-sensitive or belief-state semantics.

Hence "strict orthogonality" is not merely unproven — **the existing implementation admits an explicit equivalence relation that refutes it.**

### 3.2 The Real Limitations of CEM

CEM stores only a single length-K `probs` array shared across all horizon steps; each elite retains only `(first_act_idx, total_reward)`, not the full action sequence. The update statistic uses only the first action, yet the updated distribution is then reused for every subsequent timestep (`engine.rs:526`, `:544`). For tasks that require "step one is A, step two is B," this parameterization cannot independently express time-conditioned behavior.

In addition, `total_reward += sub_r * 0.9` (`engine.rs:519`) applies the same 0.9 factor to every subsequent step rather than the usual `γ^t` discount. This could be defined as a deliberate objective, but it must be stated explicitly and cannot be accepted as a standard discounted MPC. The retained best first action keeps the highest single-trajectory score across iterations, which for a stochastic model is a best-of-samples estimate, not a reliable expected-optimal estimate.

`done` does stop the rollout, which is a correctly implemented boundary; **stopping does not mean the action is judged unsafe and rejected.** In the default model, `done` signals divergence, yet CEM can still select a high-reward `done` action.

## 4. Mathematical Boundaries of Orthogonality, Coverage, and Consolidation

### 4.1 The "Six Methods" Are Not an Orthogonal Basis for the Decision-Problem Space

Strict orthogonality requires, at minimum, a defined object space, an inner-product/independence definition, and a coverage mapping. The methods here mix search strategy (MCTS/A*), receding-horizon control structure (MPC), a distributional learning objective (GFlowNet), game solving (CFR), and feasibility modeling/solving (CP-SAT). These layers are naturally composable: MPC can call CP-SAT; MCTS can use a learned proposal; GFlowNet can generate candidates for tree search. They cannot be claimed to be pairwise orthogonal and jointly complete simply because there are six of them, as if they were basis vectors in linear algebra.

The number of algorithms likewise cannot prove optimality. One would first need a task distribution D, a loss L, and a compute/memory/safety budget B, then compare the Pareto frontier of `E_D[L]` against the constraint-satisfaction rate. No such objective or experimental matrix currently exists. "Fewer modules to maintain" is a genuine engineering benefit; "no loss of capability" is a separate, as-yet-unmet acceptance requirement.

### 4.2 Coverage Matrix: Representational Space Is Not Solving Capability

| Dimension | Currently visible capability | Uncovered or insufficient |
|---|---|---|
| Discrete actions | A local frame of at most 16 `ActionId`s; finite-horizon CEM | Large action spaces, combinatorial actions, dynamic successor legal-action generation |
| Continuous state | 1024-dimensional `FullLatent` | Does not automatically yield continuous control-optimization capability; a continuous action vector is not part of the contract |
| Continuous control | None of the six engines above can accept continuous action parameters | Gaussian CEM, iLQR/DDP, MPPI, mixed control variables |
| Deterministic problems | Single-step sorting and fixed-model rollout | Goal-conditioned shortest path, admissible bounds, proof of optimal termination |
| Stochastic problems | `step` can implement stochastic sampling at the caller's discretion; MCTS/CEM can call it multiple times | Explicit transition probabilities, belief updates, chance constraints, CVaR, sample confidence intervals |
| Multi-agent/game | A single action id, a single scalar reward | Players/joint actions, opponent models, information sets, equilibrium definitions and exploitability |
| Hard constraints | Registered linear rules, graph revocation, an independent heat-certificate checker | Trajectory constraints, live resource context, combinatorial optimization, temporal logic/reachability |
| Partial observability / non-stationarity | A single external entropy scalar | Belief state, change detection, online model adaptation, dynamic regret |
| Causal/language tasks | Can be manually encoded into numeric input, but with no proven capability | Causal intervention, natural-language transitions, a formal step-verifier interface |

Locations: the model contract at `crates/gen-zero-core/src/traits.rs:40` has only `step(state, ActionId)->(state,reward,done)`; the plan output at `crates/gen-zero-planner/src/engine.rs:72` has only `(ActionId, NormalizedEntropy)`; `crates/gen-zero-planner/src/pipeline.rs:207` has no goal, player, control vector, certificate, or deadline.

### 4.3 Are Diffusion, Goal Reachability, and Non-Stationary Games Blind Spots?

- **Diffusion Planner: a proposal/trajectory-prior capability that is not yet present, but this is not a mathematical reason to build a mandatory "seventh engine."** Trajectory diffusion generates candidates by learning a trajectory distribution with conditioning/guidance, which is not equivalent to the current categorical CEM. It could enter the optimizer as a candidate-generation plugin, to be validated afterward by the independent gate. Its value must be demonstrated with task data, feasibility rate, and latency; the generated output itself is not a safety proof. [Planning with Diffusion for Flexible Behavior Synthesis](https://proceedings.mlr.press/v162/janner22a.html)
- **Goal-conditioned reachability: a more fundamental semantic gap.** There is currently no goal predicate and no backward-reachable/viability set, so the system cannot answer "does a given safe first action necessarily lead into a future dead end?" Backward search on a graph, HJ reachability for control systems, and a learned goal-conditioned value function are different tiers of guarantee and are not interchangeable. A learned reachable tube still requires verified error/probability guarantees; it cannot be treated as a certificate merely because it borrows PDE terminology. [Verification of neural reachable tubes](https://proceedings.mlr.press/v242/lin24a.html)
- **Non-stationary adaptive games: not implemented.** There is no persistent regret, opponent state, time window, or change detection locally. The choice among discounted/sliding-window regret, an opponent model, or online planning should be driven by the real task, with metrics stated explicitly relative to a moving comparator; standard static Nash convergence is not a promise that is automatically inherited.

## 5. PolicyGate: Actual Contract, Mathematical Risk, and Safety Fault Lines

### 5.1 Current Call Chain

```text
ProductionPipeline.decide
  validate request
  prune(candidates) -> gate.evaluate(action, entropy=0, graph, agent=None)
  selected engine / router
    -> gate.evaluate_basic(action, entropy=0) repeatedly
    -> world_model.step(...)
  check returned action belongs to feasible frame
  gate.evaluate(action, REQUEST entropy, graph, agent=None)
  optionally run an independent greedy rollout for display
  return Decision { action, gate_tier, requires_confirmation, trajectory }
```

Evidence: `crates/gen-zero-planner/src/pipeline.rs:521`, `:534`, `:549`, `:565`, `:572`, `:580`, `:605`. `PolicyGate::evaluate` always forwards `active_context=[]`, `certificate=None` (`crates/gen-zero-gate/src/policy.rs:258`).

**This is not an implementation of "hard decoupling" between front/back gating and the algorithm.** Every engine depends on a concrete `PolicyGate` and calls it repeatedly; yet the full safety context is still not threaded through to the internals. This currently pays both costs at once: redundant overhead and an incomplete contract.

### 5.2 How Far 0-1 ILP, Sheaf, and Nanocore Are Actually Implemented

**0-1 ILP:** `policy.rs:145` iterates over the rules, and `constraint.rs:100` computes the left-hand-side value for a given candidate plus active context and compares it against the RHS. Checking a given assignment is a feasibility check, not solving an integer program. There is no search over optimization variables, no bounding, and no infeasibility proof. The CP-SAT engine likewise does not call any external solver. A genuine CP-SAT solver distinguishes at least OPTIMAL, FEASIBLE, INFEASIBLE, and UNKNOWN; failing to find a solution within a timeout cannot be passed off as a proof of infeasibility. [OR-Tools CP-SAT status contract](https://developers.google.com/optimization/cp/cp_solver)

**Sheaf:** There is a substantive numerical implementation here, and it should not be misreported as a pure placeholder. `crates/gen-zero-gate/src/sheaf_gate.rs:398` recomputes the residual, energy, and gradient, and checks the diagnostic value/step size; it explicitly states that it only proves terminal compliance for a given problem, not the relaxation history. `policy.rs:179` performs read-only certificate validation for a registered problem and hard-stops if it is missing. This is a valuable check, but there is no evidence binding this terminal state to the planner's `world_model.step` prediction, the action actually executed, current physical observations, or the full trajectory.

**Nanocore:** No Nanocore invariant call or evidence parameter was found anywhere in the `engine/pipeline/policy` chain examined in this audit. The service's pipeline branch also only assembles the world model, PolicyGate, and LodGraph (`crates/gen-zero-service/src/pipeline_verb.rs:35`, `:51`). The existence of Nanocore on other service paths does not fill this contract gap. The service's audit metadata also explicitly states `formal_certificate: "unavailable"` (`crates/gen-zero-service/src/zero.rs:1560`).

### 5.3 Reproduced Boundary Issues

| ID / Severity | Evidence and observation | Judgment |
|---|---|---|
| F01 / P0 | `constraint.rs:105`-`:114` uses i32 multiply-add. The release probe constructs two `i32::MAX` terms for the same action, rhs=0: the true mathematical LHS is 4294967294 > 0, yet the actual gate returns Tier0Proceed | **A reproduced rejection failure.** Whether the config source can be remotely controlled is a separate question; a legitimate public structure can express this input. A debug-mode overflow panic is also not an acceptable typed fail-closed behavior |
| F02 / P0 | `engine.rs:208` calls `add_value` directly on the reward with no validation; the probe's NaN reward returns `Ok(ActionId(1), H≈0.0659)` | This does not just contaminate the decision, it also reports low entropy — "the model returned Ok" cannot be read as "the model's output is trustworthy" |
| F03 / P0 | MCTS ignores the successor; CFR takes only the `.1` reward (`engine.rs:692`); both return Ok for a NaN successor | Pipeline input validation does not cover model output; because the existing real model actively rejects NaN, it masks this defect in any replaceable model |
| F04 / P0 contract gap | Under `evaluate_basic`, mutual-exclusion rules pass; under `evaluate_with_context` with another active action present, it HardStops. The planner always uses an empty context | Concurrent action mutual exclusion / running quota cannot be guaranteed. It is not that the checker fails to check — the facts are simply never passed to the entry point |
| F05 / P1 availability | After a heat requirement is registered, the pipeline has no certificate field, so all related actions are pruned (there is already a pipeline test for this) | This is a safety rejection, not a bypass; but "support for certificate-bearing planning" is incomplete |
| F06 / P0 dynamic safety semantics | A synthetic model gives action 1 reward=10 and done=true, action 2 reward=0 and done=false; across the pipeline, all six engines choose action 1 | `decide`'s Tier0 reflects rule permission, not model trajectory safety. `pipeline.rs:7` claims that `done` is treated as a hazard, but this condition is not enforced when the action is decided |
| F07 / P1 diagnostic semantics | `simulate` on a forbidden action returns gate=HardStop, steps=1, `is_safe=true`. `pipeline.rs:710` calls the model before the gate check at `:743`; `is_safe` looks only at hazard (`:90`) | This disproves the general claim that "all methods prune first." `simulate` is hypothetical and does not prove an external action was ever executed; if policy-violating counterfactual simulation is to be allowed, `policy_allowed` and `hazard_free` must be explicitly distinguished |
| F08 / P1 bypass API | `GenZeroPlanner::evaluate_reflex` returns `action_slice[0]` directly, with no gate (`crates/gen-zero-planner/src/lib.rs:51`, `:61`) | A legacy public interface exists with an ungated suggestion path; it must not be wrongly claimed that ProductionPipeline's Reflex mode uses this implementation — it in fact uses the gated CpSat wrapper |
| F09 / P0 direct routing entry point | `router.rs:120` gets a HardStop for illegal entropy, yet still selects K2 at `:77`; subsequent engines re-check with entropy=0. The probe returns Ok for both NaN and 1.5 entropy | **A reproduced rejection failure caused by context loss.** ProductionPipeline's checked scope is unaffected by this same path; the direct router and the legacy lookahead still expose this contract problem |

Independent probe source: `planner_audit_evidence/probe.rs` (historical evidence, not included in the current commit); raw observations: `planner_audit_evidence/probe-release.txt` (historical evidence, not included in the current commit). F01/F02/F03 are not "possibly present" guesses. Supplementary raw observations for F09: `planner_audit_evidence/faults-extra.txt` (historical evidence, not included in the current commit).

Additional boundary: `audit_action` does reject when a safety estimate is missing, which is a substantive protection; but an uncalibrated estimate only adds to `reasons` (`pipeline.rs:493`) and does not automatically block Approved. The default margin estimate has `calibrated=false` (`crates/gen-zero-worldmodel/src/dynamics.rs:121`). "The margin passed inside the model" and "measured risk approved after calibration" should be separate types.

### 5.4 Three Convergence Risks Raised by the User: Fact Must Be Separated from Hypothesis

**a) MCTS's PUCT value distortion / dead-end penalty.** The current gate filters at the root, with the uniform prior renormalized over feasible actions (`engine.rs:130`, `:148`); there is no deep tree, and therefore no implemented mechanism for "gate pruning after expansion causing a deep dead-end exploration penalty." The actual problem today is that only the immediate reward is used, so future dead ends are invisible. For a genuine future constrained MCTS, if the legal set is a fixed and correct `A_safe(s)`, searching over the constrained MDP does not by itself constitute a value bias; bias would instead come from treating a gate error as an ordinary low reward, ignoring the denominator for rejected samples, reusing a stale mask after a state change, or conflating infeasible with unknown. Illegal terminal/cost, backup semantics, remaining budget, and viability should all be explicitly defined, and comparisons should use the same constrained problem as the baseline — it is not valid to fault safety pruning for "losing optimality" relative to the unconstrained optimum.

**b) CFR's non-convex constraints / Nash convergence.** There is currently no CFR iteration at all, so it is not meaningful to talk about "pruning breaking an existing Nash convergence." In general, after fixed illegal pure actions are removed, the mixed-strategy simplex over the remaining pure actions is still a convex set; "the original action domain is non-convex" does not by itself imply CFR fails to converge. The genuinely dangerous cases are constraints that depend on hidden state, different legal sets offered within the same information set causing information leakage/abstraction distortion, or coupled feasible sets arising from shared resources between the two sides — in which case the problem may become a constrained/generalized Nash problem rather than a standard two-player zero-sum game. Standard CFR's average-strategy guarantee carries game-theoretic assumptions; it is not a guarantee for a single argmax. [Original CFR paper](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html); [Last-iterate Convergence in Extensive-Form Games](https://arxiv.org/abs/2106.14326)

**c) CEM's particle rejection / covariance degeneracy.** This implementation has no Gaussian particles or covariance, so "covariance degeneracy has already occurred" cannot be diagnosed here. It rejects an entire trajectory that contains a disallowed action (`engine.rs:504`), and zero survivors yields NoFeasibleAction (`:529`). Once the pipeline has already removed statically forbidden actions upfront, the same static gate internally usually does not reject further on those actions; calling the engine directly makes wasted sampling more likely. If future state-dependent feasibility has a per-step acceptance rate approximately p, then under an independence approximation an entire H-step trajectory survives with probability p^H; this is a modeling assumption illustrating sample starvation, **not a probability measured in this audit.** In a genuine Gaussian CEM, when the elite count m is smaller than dimension d+1, the sample covariance has rank ≤ m−1 — this is a linear-algebra fact, not a feature of this repository. Feasibility-aware proposals, staged constraints, covariance smoothing/lower bounds, and a minimum number of feasible elites should be adopted; after any repair/projection, re-certification is required — a soft penalty cannot substitute for the final hard check.

### 5.5 What a Reasonable Boundary Design Would Look Like

The reasonable compromise is to keep an independent, safety-first **final adjudication** that the optimizer cannot override, while providing search with a pure-functional/snapshot-bound feasibility oracle. Checking only the action id at entry and exit cannot guarantee trajectory safety; mixing all safety fixes into an unauditable optimizer is equally unacceptable.

Proposed contract: `PlanningProblem{state, goal, action_space, model_version, policy_snapshot, active_context, agent_id, budget}`; `PlanCandidate{trajectory, predicted_states, objective, uncertainty, status, evidence}`; `GateResult=Allowed(certificate)|Rejected(reason)|Unknown(reason)`. Unknown must never authorize execution, and must never silently become the first action or a zero reward. A repair component is responsible for proposing a repaired candidate; the validator only validates and never quietly mutates state. Before execution, bind the observation/model/policy epoch, action digest, and resource reservation, then perform a final validation, resolving concurrent state and TOCTOU issues. This design is a proposal — it is not yet implemented, and it is not proven globally optimal.

## 6. Router: A Rule-Based Dispatcher, Not a Learned Dynamics MoE

`crates/gen-zero-planner/src/router.rs:75` directly does if-else dispatch based on external entropy, the worst gate tier, and fixed thresholds of 0.2/0.7; K1 calls A* (`:141`), K2 runs MCTS after a gate filter (`:145`), and K3 runs MCTS, CEM, and A* in sequence (`:175`). There is no model Jacobian, controllability analysis, reward-landscape estimate, branching-factor prediction, online performance feedback, or learned gating network.

It does change route based on input entropy, so it cannot be called "entirely non-dynamic"; the accurate description is **input-conditioned dispatch with fixed thresholds.** Auto has only four internal components and does not route to CFR or GFlowNet (`:27`). It is therefore also not "six experts competing adaptively."

K3's weights are 2:1:1, with the three members called sequentially, not in parallel; when MCTS supports A and the other two support B, the 2:2 tie is broken by request order (`:185`). The three algorithms share the same model and similar immediate objectives, so their errors are strongly correlated; a "committee" does not automatically add evidentiary independence. A member's `Err` propagates via `?` and fails the whole call, which is indeed fail-closed (`:173`).

Entropy is also not a unified risk scale: MCTS uses visit-frequency entropy, A* uses an arbitrary-temperature Boltzmann entropy, GFlowNet uses a softmax at yet another temperature, CFR uses positive-regret entropy, the CP wrapper is always 0, and K3 returns the input entropy unchanged (`:195`). The claim in `pipeline.rs:221` about an engine's "own entropy" does not hold for K3; the final gate, moreover, uses the request entropy rather than the engine's entropy (`:574`). The same threshold cannot be assumed to give these quantities the same "confidence" meaning. It is recommended to report observation uncertainty, model uncertainty, search uncertainty, and disagreement as separate fields, calibrated against task data.

## 7. The 2.0ms and High-Concurrency Memory Claims: Verified

### 7.1 There Is No Actual 2ms Deadline

Searching for deadline/timeout/Instant in `crates/gen-zero-planner/src` finds only the `TimeoutExceeded` error definition in `error.rs:13`; no timing path that actually produces this error was found. `PlanningEngine::plan` has no budget/deadline, and neither does the pipeline. `PlannerConfig` even allows 50,000 simulations, 10,000 CEM samples, and a horizon of 100 (`config.rs:69`, `:81`, `:87`). This is an upper bound on work count, not on wall-clock time.

In this audit, two `step` calls were each made to sleep 3ms, and A* still returned Ok only **after 6.295ms.** This sleep-injection fault is a deadline-contract probe, not a normal performance sample. It is sufficient by itself to refute the claim of a universal 2ms hard timeout at the entry point.

The default K3 mode needs roughly `128 + 32×3×4 + K = 528` model steps when there is no early `done` (K=16), not counting the gate or allocations. If the total budget were 2ms, the average per-step cost including amortized overhead would need to be about 3.79µs. This is a call-count derivation, not a WCET. The CEM path measured on this machine already runs about 5.8ms, over budget.

2ms is not a mathematically impossible bound for all tree search: small problems, precomputation, warm start, batched model calls, and a verifiable incumbent can all deliver a bounded-quality result within a deadline. **"Any problem + multi-step global optimum + a uniform 2ms" does not hold.** For hard real-time requirements, a fixed-budget safety-response layer should be separated from an anytime planning layer, with the deadline/cancellation propagated step by step, a certified incumbent retained, and a clear status returned on timeout. If a synchronous `step` call can block, checking the clock only outside the loop cannot provide a hard time limit; model execution, resource scheduling, and preemption must themselves be bounded. An unverified first action must never be returned as a fallback on timeout.

### 7.2 The 64-Byte Claim Is True; "Zero Allocation / Lock-Free Throughout" Is False

`crates/gen-zero-planner/src/tree.rs:15`'s `repr(C, align(64))` and the static assertion at `:52` are consistent with the observed `size=64 align=64`. The 64 bytes are node metadata; they do not include the 1024×f32 latent state. A single cache-line layout does not, by itself, imply the cache-hit rate or throughput of the full search.

`TypedArena::new` uses `alloc_zeroed` (`:173`), and every MCTS plan call creates a new default 1024-capacity arena (`engine.rs:144`), i.e., roughly 64KiB of node storage; with 16 candidates, only 17 nodes are currently used. `arena.alloc` does not heap-allocate per node, but it does hold a mutex on every call (`tree.rs:198`); reads after publication use `Acquire` (`:220`). This should be described as "pre-allocated, serially allocated, lock-free once published for reads," not a lock-free allocator. Using CAS for node values does not mean the whole search is parallel.

MCTS does not wire up first_child/sibling: the branch at `engine.rs:160` is empty, and subsequent addressing relies on index+1; the state_hash/hot/cold state fields do not form an actual state arena/transposition table. `act.0 as u16` (`:153`) can truncate a u32 `ActionId`; the currently returned action is taken from an independent `valid_actions` list, so this cannot be falsely reported as an observed mis-selection, but it would cause information loss if node-id backtracking is relied on in the future.

## 8. Execution Evidence and Authenticity Limits for This Audit

### 8.1 Actual Tests

Command: `cargo test -p gen-zero-planner -p gen-zero-gate`, raw exit code **0**. gate unit: 43; planner unit: 23; numeric integration: 7; latent model integration: 5; config integration: 6; pipeline integration: 28; total **112**. Both doc-test suites: 0. Raw log: `planner_audit_evidence/cargo-test.txt` (historical evidence, not included in the current commit).

This is not an acceptance test of algorithmic performance for 112 cases; for example, MCTS's "real model rejects NaN" can be achieved by the model itself rejecting it, rather than the engine performing unified output validation. The probe with `Ok(NaN)` explicitly exposes this gap. The full workspace was not run, and service live acceptance is not treated as completed.

### 8.2 Release Single-Engine Latency / Allocation

Environment: x86_64 Linux VM, 24 visible vCPUs, Microsoft hypervisor; rustc 1.96.0; release-optimized build; 16 candidates, a 1024-dimensional zero-initialized state, default engine parameters, the default `LatentDynamicsWorldModel`, an empty PolicyGate. Each engine was warmed up 20 times and measured 200 times; nearest-rank P50/P95/P99. Results were kept alive via `black_box`, and every plan call had to succeed or the program would panic. A custom global allocator counted alloc/alloc_zeroed/realloc calls; **this is not a count of live objects or bytes.**

| Engine | P50 µs | P95 µs | P99 µs | max µs | alloc/realloc calls / plan |
|---|---:|---:|---:|---:|---:|
| MCTS | 1774.930 | 2167.115 | 2679.594 | 2892.086 | 17 |
| A* | 246.390 | 329.587 | 409.684 | 473.682 | 51 |
| CEM | 5822.570 | 7283.112 | 8337.970 | 9431.527 | 1440 |
| GFlowNet named ranker | 257.390 | 318.888 | 368.086 | 548.578 | 48 |
| CFR named ranker | 216.692 | 264.389 | 275.989 | 291.889 | 16 |
| CP-SAT named ranker | 233.090 | 301.588 | 349.587 | 423.283 | 48 |

MCTS's 17 calls can be explained by the 16 gate-verdict reason strings plus the arena; A* additionally has a `format!` call on the success path plus heap growth; CEM repeatedly constructs label strings, successor labels, and so on across 384 transitions. This does not contradict the "SmallVec/stack array" claims: a local container not allocating does not mean the call chain it participates in does not allocate (`engine.rs:45`, `:491`; `policy.rs:247`).

### 8.3 Request-Concurrency and Shared-Arena Stress Probes

Using the same `Arc<ProductionPipeline>`, Auto K3, 50 requests per worker, started simultaneously via a barrier; wall time was measured, including join but not thread creation. No HTTP/serialization/real model-serving cost is included.

| Request workers | Total requests | elapsed s | requests/s |
|---|---:|---:|---:|
| 1 | 50 | 0.408776 | 122.32 |
| 4 | 200 | 0.406330 | 492.21 |
| 8 | 400 | 0.413471 | 967.42 |

**Outer-level concurrency does scale throughput in practice,** which is worth keeping, but it is not the same as parallel rollout within a single tree, and it does not license any claim of a 2ms response.

A shared TypedArena was pre-allocated with 100,000 nodes, and 1/4/8 workers contended for allocation; only the fill phase and join were timed, excluding arena creation and release:

| workers | nodes | elapsed ms | nodes/s |
|---|---:|---:|---:|
| 1 | 100000 | 2.663 | 37551718 |
| 4 | 100000 | 12.129 | 8244689 |
| 8 | 100000 | 18.251 | 5478858 |

Lock contention is significant under this load, and adding threads actually reduces throughput; this supports the claim that "high-concurrency lock-free allocation cannot be inferred from an atomic cursor alone." It does not measure shared-node backup or NUMA effects, and it is not the real tree contention of the currently serial MCTS.

**Limitations:** this was a single short run, cores were not pinned, no exclusive machine was used, and there was no long-term soak test; the global allocator's atomic counters also add overhead and may themselves become a source of shared-counter contention under concurrency. The P99 from 200 samples is only this run's empirical quantile and cannot serve as an SLA/WCET. The default dynamics, while a genuine repository implementation, still encode a synthetic formula for its business content: `next_i=.95*s_i+.05*sin(.05*i+.17*a)`, with reward as `sum(next_i)*.001` clamped (`crates/gen-zero-worldmodel/src/dynamics.rs:70`). These numbers cannot be generalized to a real language world model, GPU inference, or actual environment execution.

Historical reproduction command (the script and lock file were not published with the current commit and cannot be re-run directly on this checkout): `python3 docs/architecture/planner_audit_evidence/reproduce.py`. The script builds in a fresh temporary directory, retains the full log and exit code, and pins the dependency lock for that run; production sources are unchanged. `reproduce.py --faults-only` has already been compiled and run in a fresh temporary directory with `--locked --release`, exit code 0. Performance will vary with machine/scheduling and should not be expected to match the table bit-for-bit.

## 9. Python's 11/12 Items to Rust's Six Engines: What Capability Was Actually Lost

"12 planners" was already a mix of engines, models, validators, and opponent analyzers; the counts cannot be equated. The current Python client already aliases `bidirectional_planner` to AStarEngine and `continuous_mpc_planner` to MpcCemEngine, while still keeping the text world model, PRM, D-SCM, and the bluff detector (`python/gen_zero/client.py:465`, `:574`). This shows that "consolidating modules" can preserve sub-capabilities; whether Rust actually preserves them must be verified item by item.

| Python module | Real capability and limits of the original code | Judgment on Rust migration |
|---|---|---|
| Bidirectional Search | Two heaps/frontiers, meet-in-the-middle detection, reverse-action mapping, and path stitching — genuinely performs graph search; correctness still depends on reverse-transition/cost semantics. `python/gen_zero/planner/engines/astar_engine.py:327`; forwarded by the client at `client.py:1604` | **Real multi-step path-search capability was not preserved.** Rust's A* has no goal predicate or frontier expansion and cannot be called functionally equivalent. It could be folded into a future graph-search backend option, but it does not need to be deployed as an independent engine |
| Text World Model | Explicitly an untrained, uncalibrated rule-based model; the string branch just concatenates `" -> action"`, returns a fixed reward, and ends after one step (`python/gen_zero/world_model/text_world_model.py:1`, `:84`) | The string/dict environment-adapter semantics were lost, but this should not be inflated into "a trained language world model was lost." A typed adapter and tested semantics should be restored — not a false claim of model quality |
| Continuous Latent MPC | Genuinely has an N×H×D Gaussian trajectory, bounds clipping/simplex projection, and per-timestep mean/std updates; but the simulation `curr_z += .1*act` does not actually use the passed-in latent model (`python/gen_zero/planner/engines/mpc_cem_engine.py:107`, `:125`, `:154`, `:166`) | **Continuous action parameterization, bounds/simplex, and time-conditioned distributions are indeed missing.** Rust, conversely, does genuinely call `WorldModelDynamics` step by step — not every dimension regressed. The next step should be wiring continuous actions into the real model, not copying Python's fake dynamics |
| PRM | In this repository this means **ProcessRewardModel**, not the robotics Probabilistic Roadmap. It is fatal-reward, flood-fill pocket, financial stop-loss, and text keyword rules (`python/gen_zero/model/prm.py:65`); it can be injected by the caller as an MCTS invariant lambda (`client.py:1002`) | This domain-specific transition-verifier capability has no equivalent migration; it belongs to the safety/model-scoring plugin category and should not have its semantics discarded on the grounds of "it's not a planner," nor should it be called a learned PRM |
| D-SCM / Bluff detector | Hand-written structural equations, noise abduction, and counterfactual simulation with noise locked (`python/gen_zero/multiagent/decentralized_scm.py:68`, `:168`); intent inference is a logistic/threshold rule (`:207`) | Rust CFR's single-shot payoff comparison cannot substitute for causal intervention, opponent state, or intent features. **The interface capability was lost, though its authenticity/calibration was already limited to begin with.** It is suited to being an independent opponent/causal-model service, not an in-place substitute inside the equilibrium solver |

"Irreplaceable" here is relative to Rust's current six public contracts: these capabilities cannot be reconstructed without adding information; it does not mean these Python implementations are the only possible algorithm or are worth porting line by line.

### 9.1 Why the Historical Benchmarks Do Not Prove 12→6 Was Lossless

- The Text World Model in `run_12_planners_benchmark.py:156` uses a grid dictionary and does not test arbitrary natural language; the MPC-CEM Trajectory at `:168` uses BUY/HOLD/SELL and takes the discrete branch.
- The continuous MPC success condition is only that the output action has length 4 (`:248`), not goal reward, constraint maintenance, or dynamics-model correctness.
- PRM is tested against a generic pos dictionary (`:258`) that does not trigger the main domain rules; D-SCM supplies a next state that is exactly model-consistent along with revealed_strength=0.20 (`:272`, `:282`), and success is judged by a preset bluff type/threshold (`:286`).
- `benchmarks/results/r4_evidence/planners_after_report.json:2` explicitly states `algorithm fixtures; does not establish learned-model quality`. Its 50-run fixture figures — Bidirectional 100%/mean 0.059ms, Text 100%/0.028ms, discrete MPC 100%/1.852ms, continuous MPC 100%/3.808ms, PRM 100%/0.001ms, D-SCM 100%/0.036ms — are **historical records, not re-run for this audit** (corresponding to lines 129, 251, 312, 556, 617, 678). These figures use a completely different methodology from the Rust engine timing in Section 8.
- `benchmarks/results/planners_and_constraints_fixed_results.json:59` explicitly records that the strict run achieved only **10/12, exit=1**; CP-SAT's 50-run success rate was 96% (`:418`), not 12/12; `:60` also records a historical 2.0ms test with **127/1000 fallbacks and wall-deadline misses.** This data cannot be transposed onto the current Rust failure rate, but it must be preserved rather than only citing the passing items.
- The six-engine Python benchmark script writes its output to `python/results/gen_zero/issue_93_6_planners_benchmark_report.json` (`run_6_orthogonal_planners_benchmark.py:232`); this result file is not tracked at the current HEAD. A script existing does not mean the experiment was completed.
- There is no comparable same-methodology benchmark for the Rust arena's historical throughput; `README.md:121` has already retracted the old node-allocation/planning SLA numbers, and `benchmarks/suites/latency_suite.py:108` marks in-engine MCTS as `not_measured`. The new probes added in this audit fill in local observations, but they still do not fill the gap for production acceptance.

## 10. Minimal Counterexamples and Burden of Proof for the Mathematical Claims

### 10.1 A Single-Step Interface Cannot Imply Global Path Optimality

Construct a root state with two actions: A has immediate reward 1, but its total reward along the rest of the path is −100; B has immediate reward 0, but 100 is available on the next step. If the two first-step state displacements are equal, the current A*/GFlowNet/CFR/CP wrapper all favor A; MCTS only compares the root's immediate reward and cannot learn the second-step payoff from more simulations either. This is an **analytical counterexample** derived from the objective functions actually read in the code, not a claim of a new executed experiment; the independent immediate terminal-hazard counterexample executed in Section 8 is a separate case. CEM can see some future reward within a finite horizon, but horizon truncation, sampling, and the shared categorical distribution still provide no global optimality guarantee.

### 10.2 A Single Pure Action Cannot Represent a General Nash Solution

In zero-sum rock-paper-scissors, the equilibrium is the uniform mixture; always picking the pure action with maximum regret can be exploited by the opponent. The current CFR implementation outputs a single ActionId and entropy value; it does not output and execute a mixed strategy, and there is no opponent payoff matrix or extensive-form game. Even when all three payoffs are equal, it simply selects the first legal action and returns high entropy — **high-entropy metadata does not mean the behavior is actually randomized.**

Standard counterfactual regret accumulates action advantage per information set, weighted by the corresponding reach probabilities; the finite-regret bounds that are typically derived apply to average-strategy exploitability, not to a single vector-minus-mean computation. When constraints change, it should be proven which constrained game is actually being solved; it is not valid to demand recovery of the unconstrained Nash equilibrium that the rules have excluded.

### 10.3 A Softmax Score Is Not Trajectory Balance

Trajectory balance requires a positive terminal reward, forward/backward path probabilities, and a normalizing quantity, with a typical objective:

`L_TB(τ) = [log Z + Σ log P_F(s_t|s_{t−1}) − log R(x) − Σ log P_B(s_{t−1}|s_t)]²`.

The conclusions of reward-proportional sampling depend on conditions such as the objective and support set, not on "computing a softmax" by itself. None of these quantities exist in the current Rust code, and no action is learned or sampled: it directly does an argmax. Mixed-curvature geometry utilities exist in other modules, but this likewise does not prove that a manifold GFlowNet is implemented here. [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)

### 10.4 A Certificate Must Match the Proposition It Claims

`||Ds-b|| <= ε` can prove a small residual under a given matrix/bound; without additional modeling and error bounds, it cannot imply that the action is harmless in the real environment, that a long-term constraint can be maintained, or that the goal is reachable. An integer-inequality checker likewise can only prove that a given assignment satisfies the registered constraints, and even that requires guaranteeing the arithmetic does not overflow in the first place. This audit does not accept the inferential leap of "a mathematical formula exists, therefore the whole system is formalized."

## 11. Next Architecture: Evolve by Capability and Evidence, Not by Padding Out an Engine Count

What follows is a proposed roadmap, and **all of it is unfinished work**; this audit delivers only the assessment and does not present any of it as already implemented.

### P0: Close the Real Fail-Closed Breaches First

1. Validate coefficients, duplicate terms, and context capacity at constraint compile/registration time; use checked arithmetic, or integers wide enough with a proven bound, for multiply-add; overflow must be a typed rejection. Verify the same rejection outcome on both debug and release profiles.
2. Have every engine uniformly check input, successor, reward, accumulated values, and internal scores. `Ok` must not exempt anything from validation. Add NaN/Inf/overflow model-output test cases for MCTS/CFR; a model `Err` or illegal data must never turn into a low reward, a skipped failing member, or the first action.
3. Unify entry-point validation and close off the public paths where the legacy reflex and the direct router bypass the full gate; the type-level distinction between "suggestion" and "authorization" must be made explicit.
4. Distinguish `Terminal::GoalReached`, `Hazard`, `Truncated`, and `Unknown`. Deciding an action must invoke the state/transition safety contract; if only policy eligibility is provided, the field and API documentation must not call it certified safe.
5. Thread active resource context, agent identity, model/policy epoch, and candidate certificates through end to end; when required facts are missing, return Unknown/Reject. Use atomic reservation/commit for high-concurrency quotas — it must not be possible for multiple requests to each check an empty context and pass simultaneously.
6. If `simulate` retains the ability to run a policy-violating counterfactual, it must explicitly return `policy_allowed=false` to avoid confusion with `is_safe=true`; it must never be presented as an executable authorization. Audit whether `audit_action`'s Approved outcome allows an uncalibrated safety reading, and use the type system to distinguish levels of guarantee.

**Acceptance bar:** the independent counterexamples F01-F09 in this document must be stably rejected at any public entry point where rejection is expected; illegal input must not produce an unflagged Ok, and engine defects must not be masked by relying on the default model to error out on its own.

### P1: Establish a Core Contract That Can Express the Real Problem

Separate the five responsibilities of world model, proposal, search, verification, and execution. The goal is not to split them into five network services, but to make the replaceable boundaries and evidence flow explicit.

| Responsibility | Contract to add | Verification metric |
|---|---|---|
| Dynamics | Discrete/continuous/hybrid actions, batched step, terminal reason, model version and calibration error | Held-out transition error, calibration, rejection rate under model mismatch |
| Problem | Goal, legal-action generation, cost, horizon, players/information sets or belief | Comparison against a small exact oracle, semantic consistency |
| Search | Anytime budget, persistent workspace, candidate trajectory, objective bounds/status | Quality-vs-budget, optimality gap, deadline misses |
| Safety | Stateful feasibility oracle, certificate request/validation, Unknown | Counterexample coverage, resource mutual exclusion, certificate binding/expiry rejection |
| Execution | Final recheck, reservation, confirmation, safe fallback | TOCTOU/revocation race conditions, real execution trace |

### P1: Add the Missing Algorithms or Honestly Drop the Algorithmic Identity

- Rename the current shallow MCTS/A*/GFlowNet/CFR/CP wrapper to baseline names that describe their actual behavior, or explicitly flag them as approximations externally. They can share a single `OneStepScorer` to avoid maintaining multiple mathematically equivalent implementations.
- Real MCTS: successor tree, terminal/value backup, legal mask, transposition/state storage, reusable search, batched rollout; concurrency correctness before throughput. Where there is no value model, state the rollout policy and its error explicitly.
- Real graph search: goal predicate, g+h, an admissibility statement, duplicate-state handling/reopen, path reconstruction; bidirectional search is just a backend variant and needs a correctly implemented reverse transition.
- Continuous MPC: a continuous action space, per-timestep mean/covariance (or diagonal std) with genuine model dynamics, a bounds/manifold adapter, warm start, recursive feasibility; keep the categorical backend, but do not pass off a continuous latent as continuous control.
- Build real CFR only where there genuinely is an imperfect-information game; add information sets/players, counterfactual reach, accumulated strategy, average strategy, and an exploitability benchmark. Otherwise, drop the Nash claim and keep the greedy baseline.
- Build a real GFlowNet only where diversity/reward-proportional proposals are actually needed and training data exists; provide TB/flow loss, sampling consistency, and diversity metrics. It need not serve as an independent safety adjudicator.
- If a real CP-SAT is needed for combinatorial constraint optimality, connect a stateful, bound-tracking solver backend; distinguish feasible, optimal, infeasible, unknown, and timeout. Move simple action eligibility back into the shared verifier instead of listing it as "the sixth independent planning paradigm."

### P2: Optimize by Budget-Aware Scheduling and Authenticity Benchmarks

- First turn the router into an auditable budget/capability scheduler: route based on problem type, action cardinality, model latency, horizon, and safety requirements; only evaluate whether a learned gating network beats the rules once there is data. Do not upgrade the "MoE" naming first.
- Separate the small-step action response from background anytime search. Under any deadline, only return a certified candidate; if there is no certifiable action, explicitly reject/request takeover. A so-called safe fallback must carry its own feasibility evidence.
- Pool workspaces/arenas, precompile static masks, make success-path formatting lazy, and wire in `step_batch`; then run a stable load comparison across 1/4/8/16 workers. The lock that currently protects release-mode correctness must not simply be removed for the sake of throughput.
- Performance acceptance should report quality and rejection rate together, and it must be forbidden to satisfy a latency table by shrinking the horizon, disabling the gate, swallowing errors, or quietly substituting a default action.

### P3: Add Missing Capabilities Once There Is Demand and Evidence

Diffusion proposals, a goal-conditioned value/reachability oracle, the PRM transition verifier, and causal/opponent models should all be versioned plugins with explicit input semantics; SOTA is not a checklist of algorithms. Any of them should enter the default route only once a clear task comparison shows it improves the success-rate/safety/budget Pareto frontier.

### 11.1 Proposed Acceptance Matrix

| Workload | Required ground truth / control | Acceptance condition that cannot be gamed |
|---|---|---|
| Delayed reward and trap maps | A small graph with exhaustive DP/Dijkstra, a fixed successor model | Report the optimal-path gap; checking only that the action belongs to the candidate set is forbidden |
| Stochastic / POMDP | A toy model with known probabilities, a belief oracle, multiple seeds | Calibration/confidence intervals, risk budget, number of model calls |
| Continuous control | An LQR closed-form solution, a small constrained system, non-convex obstacles | Trajectory cost, constraint violation, feasible-elite ratio |
| Two-player zero-sum imperfect information | Fixed rules with computable exploitability, e.g. Kuhn/Leduc | Average-strategy exploitability; "returned a legal action" is forbidden as Nash verification |
| Hard constraints and resource concurrency | An exhaustively solved small ILP, two requests competing for the same resource, fault injection | Every authorization must be independently re-checkable; overflow/timeout/missing certificate must never authorize |
| Generative proposal | A fixed reward target and a sampling baseline | Diversity, distributional error, constraint success rate, training/inference cost reported separately |
| Real-task migration | The same model, same gate, same budget and seeds for Python and Rust | Paired success rate/quality; direct cross-machine historical latency ratios are forbidden |
| Performance | Long mixed load, slow model requests, cancellation, OOM/capacity limits | P50/P99/P999, wall-deadline misses, throughput, allocated bytes, reported together with quality |

## 12. Delivery Status and Remaining Scope

**Delivered in this audit:** the report, source-code locations, the six-engine formula comparison, the Python capability-loss mapping, an actual run of the 112 existing tests, release-mode numeric/gate fault probes, six-engine latency/allocation observations, and request/arena concurrency stress probes. The read-only side threads are complete and report a consistent workspace/HEAD; the primary thread verified the key evidence and is responsible for the verdict.

**Unverified:** the real production model and environment, long-term load, the safety executor, full-workspace integration, and current industry-best rankings. Papers cited in this document are used to define algorithms and guarantee conditions; they are not used to claim this repository already has that capability.

**Incomplete / fatal defects:** the production-code problems listed in this document were not fixed as part of this audit task; they will not disappear just because the report is written or the tests are all green. Launch/safety sign-off must not rely on the unfulfilled claims of "six orthogonal engines," "formally complete," "2ms hard real-time," or "zero allocation on every path."
