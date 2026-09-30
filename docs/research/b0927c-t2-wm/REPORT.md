# b0927c-t2-wm: Dense-Teacher-Guided Zero-Token Continuous World Model

Research and engineering design; 2026-09-27; working tree `/ebs/pj/gen-zero`; verification baseline HEAD `acb2c0ccf3f30a708cd9a4f638248973c4709188`.

This work only wrote the research report and evidence records; it did not modify code, train a model, run a heavy compile, deploy, or dispatch a subagent. Below, "proposal," "goal," and "acceptance threshold" are all not-yet-implemented capabilities; code-reading evidence and execution evidence are each labeled separately. The existence of an existing checkpoint or historical JSON does not constitute a reproduction experiment in this work.

## 1. Decision Conclusion: What Can Be Done, What Cannot Be Promised

The executable main route is: **treat the Dense large model as an expensive representation teacher, use real action transitions to train a small, action-conditioned latent-space model with risk and termination semantics, then have the Rust planner run language-decoding-free lookahead on that model.** The sub-millisecond candidate is this small model's local transition kernel, not a full 405B-model forward pass, not a full search, and not cold-start end-to-end decision-making.

The project must not be founded on "Dense naturally has a smooth semantic potential surface, so symplectic integration alone gives reasoning." That chains together four things that have not yet been established: that hidden states have Markov sufficiency, that hidden states have a known symplectic structure, that task transitions can be characterized by conservative dynamics, and that energy-preserving motion improves task correctness. If any one of these fails, the result can be a system that is extremely fast, stable, and has a nice-looking energy plot, but predicts wrongly.

The recommendation is to first build an equal-data, equal-budget residual MLP baseline, then compare Neural ODE, conservative Hamiltonian, controlled dissipative Hamiltonian, and hybrid event models. **The Hamiltonian structure is an inductive bias to be tested, not a preset winner.** The goal is reliable prediction and decision benefit, not a low-energy-drift plot.

"Zero token" must be broken down by scope:

| Scope | Computation allowed | Must be reported |
|---|---|---|
| Offline teacher extraction | Input tokenization, large-model forward pass, optional training trajectories with clear provenance | Extraction FLOPs/time, model and quantization version, training data provenance |
| Online root-state encoding | One teacher or trained-student encoding pass; reading a real observation | Whether the large model was invoked, prefill time, history/KV cost |
| Imagination inner loop | Small latent-space transition, risk/value head, legal action selection | `generated_tokens=0`, `teacher_forward_calls=0`, real transition call count |
| Environment execution / re-observation | Real tool actions and state updates | Tool time, failures, necessary re-encoding cost |

A speed measured using cached features only should be called "compute latency after freezing features," not "end-to-end zero-token inference latency for a new task." Zero language decoding also does not mean zero candidate actions, zero discrete events, or zero computation.

## 2. Repository Reality: Capabilities That Already Exist and Dangerous Gaps

The table below cites source-code locations in the current working tree; a static confirmation is not the same as production verification. Full commands, raw exit codes, and outputs are stored in this directory's `01`–`06` logs and `evidence.json`.

| Fact | Evidence location (relative to repo root) | Engineering impact |
|---|---|---|
| The Python neural world model is a residual MLP with sigmoid outcome/safety heads | `python/gen_zero/world_model/neural_dynamics.py:119` | There is no need to build a training model from scratch, but its semantics must be upgraded |
| `step` requires a loaded checkpoint; `forward` itself does not check the load state | Same file `:153`, `:119`, `:197` | Production must go through the checked interface; a correctly structured checkpoint still does not prove real training or generalization |
| The reward is a safety/outcome probability; `done = r_hat < threshold`, and the data has no independent done label | Same file `:17`, `:169` | Goal, collision, trap, and timeout cannot be distinguished; a safety probability cannot substitute for a task reward |
| The Python client can load a neural checkpoint and mount MCTS/MPC | `python/gen_zero/client.py:432`, `:467` | There is already a Python path; it should not be misreported as a complete island |
| When both a Hamiltonian and a neural model are configured, the client passes `None` as the neural model to MCTS | Same file `:471` | An ambiguous configuration must be rejected; "loaded" must not be understood as "in effect" |
| The Python Hamiltonian potential/action matrix is randomly initialized; string actions become hash-driven random vectors | `python/gen_zero/world_model/hamiltonian_dynamics.py:113`, `:186`, `:247` | This is not learned action semantics; the string hash also does not guarantee cross-process stability |
| This model silently zero-pads/truncates when the action dimension is wrong | Same file `:220` | This is a clear violation of this task's fail-closed requirement; a follow-up implementation must remove this behavior |
| Rust already has symplectic integration, contact dynamics, and a corresponding world model | `crates/gen-zero-worldmodel/src/lib.rs:7` | A duplicate numerical-integration island should not be started separately |
| The Rust symplectic world model uses a hand-set harmonic potential well and sinusoidal ActionId encoding, explicitly labeled untrained and uncalibrated | `crates/gen-zero-worldmodel/src/symplectic_dynamics.rs:20`, `:34`, `:132`, `:233` | It can only prove numerical structure, not language or environment prediction capability |
| The symplectic model does have a service constructor and planner-trait entry point | `crates/gen-zero-service/src/worldsim.rs:240`, `:275` | It cannot be said its production reference count is zero; but it is not proven that a Dense-trained model is wired in |
| The general service's default world model is still constructed as `LatentDynamicsWorldModel::default()` | `crates/gen-zero-service/src/zero.rs:818` | The optional symplectic model in `worldsim` and the default wiring of the general `pipeline` are different chains |
| `pipeline` is the shared main-chain entry point for CLI/MCP/HTTP | `crates/gen-zero-service/src/pipeline_verb.rs:1`, `:35`; `zero.rs:3439` | A new model must be injected into this chain; adding only a new simulate endpoint is not enough |
| The Rust state is fixed at 1024 dimensions, and a `step_batch` trait already exists | `crates/gen-zero-core/src/types.rs:152`; `traits.rs:40` | A 64-dimensional Python checkpoint cannot be plugged in directly; an explicit representation protocol and export are needed |
| The current Rust MCTS expands sequentially, caches deterministic successors, and reuses the same action set at each depth | `crates/gen-zero-planner/src/engine.rs:213`, `:247`, `:355`, `:397` | Having a batch interface does not mean search is already batched; having an arena does not mean GPU-scale search exists |
| The pipeline treats every done as danger, and the search wrapper errors out directly on encountering terminal | `crates/gen-zero-planner/src/pipeline.rs:4`, `:49` | If a new model returns done for a goal, it will wrongly reject a successful path; this is a blocker to fix before integration |
| The Dense feature schema is a static train/test/candidate matrix with IDs | `benchmarks/suites/gpu_extract_llama405b_13tasks.py:38` | It does not contain sufficient `(s,a,s')` causal trajectories; having an extractor does not mean features for all five models have been successfully extracted |

The `python/gen_zero/model/set_choice_head.py` mentioned in the user's background was not found in this work's file listing; the existing directory contains `choice_head.py`, `dual_head.py`, and similar files. This report does not treat a stale path as a verified interface. System 1's role in the new design is defined as a candidate-action prior and ranking; the concrete export contract must follow the actual deployed implementation.

### 2.1 Three Categories of Existing Experiments That Can No Longer Be Used to "Prove a Breakthrough"

1. `python/gen_zero/scripts/benchmark_issue_86_hamiltonian_world_model.py:40` uses a random MLP; `:101` uses a random Hamiltonian model; `:112` has no action rollout. The two have different dynamics and different energy definitions, and there is no paired real successor. It compares the numerical behavior of constructed systems; it does not prove that a learned Hamiltonian outperforms the already-trained `NeuralDynamicsWorldModel`.
2. `benchmarks/suites/evaluate_cpu_zero_token_dynamics.py:68` tests a random-matrix recurrence; the convergence process at `:99` directly holds the target vector. It is a small-kernel arithmetic experiment; it does not prove unknown-answer reasoning or Dense world-modeling.
3. `benchmarks/suites/benchmark_world_model_mcts_ablation.py:21`'s model directly accesses the environment map and traps; `:133` already honestly labels it privileged exact dynamics. It can serve as a planning-diagnostic reference; it cannot serve as performance evidence for a neural world model or the Rust production planner.

In addition, the state in `scripts/extract_trajectories.py:16` includes safe-successor density and forward-blocked/goal-reachable-type features. Whether these are visible at deployment time must be decided case by case; if they can only be obtained from a full map or future simulation, they must be placed on the privileged track and must not be compared against a limited-observation model and then reported as generalization. This issue is more urgent than swapping the integrator.

A light behavioral reproduction this time found: a Python Hamiltonian model can be constructed with no checkpoint; a two-dimensional action and a manually zero-padded four-dimensional action produce identical forces; an untrained model accepts a string action and returns a finite state. The raw exit code was 0. This only verifies the risky behaviors above; it does not verify predictive capability or latency; see `13-behavior.log`.

## 3. Support Boundaries Given by the Literature

The following literature is used as method grounding; its benchmark results are not transferred to Gen-Zero. Firecrawl Research performed multi-direction search, reference expansion, and key body-text reading; key body text/metadata are kept in logs `07`–`12`. The search is not an exhaustive audit of the latest leaderboards.

| Method family | What can be borrowed | Inferences it does not support |
|---|---|---|
| [Neural ODE, Chen et al.](https://arxiv.org/abs/1806.07366) | Define continuous depth with a learned vector field, advanced by a solver | The continuous form does not guarantee low NFE, low latency, or safety |
| [Hamiltonian Neural Networks, Greydanus et al.](https://arxiv.org/abs/1906.01563) | Learn a scalar Hamiltonian, then take partial derivatives to construct dynamics | That LLM hidden states are naturally a canonical phase space; that energy conservation equals correct reasoning |
| [Hamiltonian Generative Networks](https://arxiv.org/abs/1909.13789), [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630) | Learn a dynamics representation from observations; compare different mechanical structures and coordinate assumptions | That arbitrarily naming the two halves of a vector q/p identifies physical structure |
| [Deep Hamiltonian Networks based on symplectic integrators](https://arxiv.org/abs/2004.13830) | Incorporate a discrete integration scheme during training; study the discrete model and modified equations | That arbitrary control force, quantization, and projection remain strictly symplectic |
| [Port-Hamiltonian Neural Networks](https://arxiv.org/abs/2107.08024), [Dissipative HNN](https://arxiv.org/abs/2201.10085) | Explicitly model input work and dissipation, avoiding forcing energy conservation on an open system | That dissipation alone identifies logically irreversible events |
| [MuZero](https://arxiv.org/abs/1911.08265), [TD-MPC2](https://arxiv.org/abs/2310.16828), [UniZero](https://arxiv.org/abs/2406.10667) | Learn a latent dynamics useful for predicting reward, value, and policy, and plan on it | That a frozen LLM representation becomes a correct world model without training on a real environment |
| [Coconut](https://arxiv.org/abs/2412.06769) | Train continuous thought; a hidden vector feeds back in as the next input | That bypassing discrete tokens bypasses the large-model forward pass, or naturally reaches sub-millisecond |
| [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538) | Top-k routing produces a piecewise computation structure | That all MoE is necessarily discontinuous, that all Dense is necessarily smoother |

Coconut's cost boundary is especially different from this design: continuous thought is still fed into the language model for processing. This design wants the search's inner loop to stop traversing the teacher network, which requires a genuinely trained small proxy dynamics; the former's experimental results cannot be treated as proof of the latter.

## 4. Dense, Smoothness, and "Potential Surface": A Correct Mathematical Statement

### 4.1 Dense's Real Advantage Is a Local Computation-Graph Regularity, Not Inherent Conservativeness

Fix the sequence length, position, and attention mask, and treat the input embedding as a continuous variable. If the network is composed of smooth operators and normalization denominators have a positive epsilon, the Dense map is differentiable on this continuous input domain; with ReLU and similar, it is generally only piecewise smooth. The map from text to token is not continuous, and quantization, truncation, caching, and discrete events do not inherit the above property either.

A hard top-k MoE can be written as

\[
F(x)=\sum_{i\in S_k(x)}g_i(x)E_i(x).
\]

It can be smooth on the region where the expert set is unchanged; a Jacobian jump can appear at a set-switch point, and if expert outputs do not match at the switch, a function jump can also appear. Soft routing, expert matching, or other constraints can soften these effects. Dense simply lacks this one routing-switch source; this cannot be used to rank the two model types' actual Lipschitz constant, curvature, or task predictability.

For a generic vector field \(f(q)\), the existence of a local scalar potential \(f=-\nabla V\) needs a corresponding integrability condition; on a simply connected domain, for a sufficiently smooth field this requires the Jacobian to be symmetric. An LLM's residual map generally has no such constraint. **"The representation is differentiable" does not imply "it has a potential"; "it has a potential" does not imply "that potential represents ground truth or safety."** Dense hidden states also are typically not sampled from some already-identified energy function.

### 4.2 Geometric Properties That Must Be Measured, Not Asserted

At the same input, the same semantic perturbation, and a comparable compute budget, measure for both Dense and MoE teachers: local Jacobian norm, finite-difference slope change, routing-flip condition statistics, successor-prediction error, and task-information loss after low-dimensional projection. Use JVP/VJP or random-direction estimation to avoid constructing a huge full Jacobian.

In a differentiable implementation, compare \(\|J_F(x+\epsilon v)-J_F(x)\|/\epsilon\); if only a black-box extraction API is available, only a finite-difference proxy can be reported, and it cannot be called complete curvature. An embedding perturbation may leave the natural-language distribution, so it must also be evaluated with real text rewrites, real state perturbations, and action interventions. Match quantization precision, layer/pooling method, sample ID, context truncation, and scale; do not attribute different quantization errors to Dense vs. MoE.

The smoothness of an HNN's own potential is controlled by the parameterization of the small model; it does not require the teacher to have first proven it has a physical potential. The teacher may provide a more predictable representation, and that is the causal path that needs ablation to verify.

## 5. State Construction: Solve Sufficiency First, Then Talk About q/p

Let the real environment state be \(s_t\), the visible observation \(o_t\), the structured action \(a_t\), and the task/goal condition \(c\). The Dense teacher \(E_m\) gives \(h_t^m=E_m(o_{\le t},a_{<t})\). What needs to be learned is

\[
x_t=\Phi_\phi(h_t^m,\text{observable history}),\qquad
p_\theta(x_{t+1},r_t,e_t\mid x_t,a_t,c).
\]

A single final-token pooled hidden state is not necessarily a sufficient statistic. If two histories have similar \(x_t\) but the same action produces conflicting successors, there is state aliasing; history encoding/memory or a probabilistic belief should be added, rather than using energy regularization to average conflicts into a "stable prediction."

The recommendation is to use \(x=(q,p)\in\mathbb R^{2d}\), \(d\in\{64,128,256\}\), for the first round, saving context, event, and uncertainty separately. This dimension is an experimental grid, not a performance promise. A minimal end-to-end pass could first use the existing 1024-dimensional, 512+512 Rust contract, and only later decide whether to migrate to a smaller, version-constrained latent type. Different dimensions must have an explicit schema; zero-padding, truncation, or relying on a shape happening to fit is prohibited.

\(q_t\) is encoded from the observation; \(p_t\) is inferred from observation history and the previous action. \(p=M\Delta q/\Delta t\) should only be considered when the observation genuinely has an identifiable velocity. Pseudo-time for text tasks and physical time must not be mixed; \(q_{t+1}\) must not be used to construct an online \(p_t\), or it leaks the future. Splitting a raw h in half is only an array layout, not learned canonical coordinates.

Only if a known original phase space and symplectic form exist can the encoder \(\Phi\) be required to satisfy the corresponding symplectic-map condition. A generic LLM hidden state has no known original symplectic form; in that case only "impose a canonical inductive bias in the learned coordinates" can be said. A local canonical chart does not guarantee a single global coordinate; a topology like the torus in particular needs periodic representations or multiple charts with explicit switching.

Multiple teachers each use their own \(\Phi_m\), aligned on the same observation/action/episode. The existing ID/label consistency check in `cross_model_manifold_alignment.py:128` can be reused, but static CKA or Procrustes similarity does not prove dynamics conjugacy. The latter requires verifying that

\[
A_{m\to n}T_m(x,a)\approx T_n(A_{m\to n}x,a)
\]

holds on unseen trajectories and actions. Alignment, normalization, and dimensionality reduction are fit only on the training set; the validation set is used for selection; the test set stays sealed.

## 6. Continuous Dynamics, Symplectic Integration, and Irreversibility

### 6.1 Establish a General Controlled Neural ODE First

\[
\dot x=f_\theta(x,u,c),\quad u=\psi(a,\text{parameters},c).
\]

Use fixed-step RK2/RK4 and an equal-parameter-count discrete residual model as the baseline; record NFE. Deployment should prioritize a fixed small NFE for predictable latency. An adaptive solver has value for error control, but stiffness can spike NFE and p99; when the budget is reached, return an explicit rejection rather than silently integrating fewer steps.

The ODE's layer depth, imagined time, and real action duration are three different quantities. When only one-step discrete training data is available, the intermediate continuous trajectory is generally unidentifiable; multiple different vector fields may give the same one-step map. So a sub-step of the integrator must not be called a real intermediate environment state without intermediate-observation verification.

### 6.2 Separable Hamiltonian and Störmer–Verlet

In the conservative branch, choose a constant positive-definite mass matrix M (diagonal at first) and an action-conditioned potential:

\[
\mathcal H_\theta(q,p;u,c)=\tfrac12p^TM^{-1}p+V_\theta(q;u,c),
\quad
\dot q=M^{-1}p,\quad\dot p=-\nabla_qV_\theta.
\]

Freeze u,c within each sub-step, with step size \(\delta\):

\[
p_{k+1/2}=p_k-\tfrac\delta2\nabla V(q_k;u,c),
\]
\[
q_{k+1}=q_k+\delta M^{-1}p_{k+1/2},
\]
\[
p_{k+1}=p_{k+1/2}-\tfrac\delta2\nabla V(q_{k+1};u,c).
\]

Under the corresponding smoothness and numerical conditions, this scheme is second order, symplectic, and reversible for a separable system under frozen control; it is not exact energy conservation. The usual bounded-long-term-energy-error result also has preconditions such as step size, bounded trajectory, and smoothness. For a harmonic oscillator, the stability interval needs \(\delta\omega<2\); a high-curvature trained potential may force a smaller step size. If \(M=M(q)\), H is no longer separable in this form, and these three lines cannot be reused while continuing to claim symplectic.

A single-step action change alters the Hamiltonian, and legitimately can do work. Integration residual, control work, dissipation, and event jumps should be recorded separately; a real control-induced energy change must not be uniformly recorded as integrator failure.

A variational route can choose a discrete Lagrangian \(L_d(q_k,q_{k+1};u_k)\), solved via the forced discrete Euler–Lagrange equation

\[
D_2L_d(q_{k-1},q_k)+D_1L_d(q_k,q_{k+1})+F_d^++F_d^-=0
\]

It applies to more general mechanical structures, but implicit-solve overhead, convergence failure, and export complexity are real; it is not adopted as the default sub-millisecond MVP route for now. Non-converging iteration must error out; the last iterate must not be treated as a valid solution.

### 6.3 Low-Cost Potential Parameterization, Without Repeatedly Traversing the Dense Teacher

One exportable, learnable candidate parameterization is

\[
V(q;u,c)=\tfrac12q^TKq+b(u,c)^Tq+
\sum_{j=1}^{w}\alpha_j\,\operatorname{softplus}(w_j^Tq+\beta_j(u,c)),
\]

\[
\nabla V=Kq+b+W^T[\alpha\odot\sigma(Wq+\beta)].
\]

K is diagonal at first; keep an appropriate spectral bound. This analytic gradient corresponds exactly to the trained potential, avoiding embedding Python autograd inside Rust. Allowing signed \(\alpha\) is needed to express non-convex structure; if it is all forced convex, the model cannot then be claimed to have learned multiple potential wells. A positive-definite quadratic term can bound the far field, but it does not prove correctness of prediction inside the training region.

Independently training an arbitrary "force network" and then claiming it necessarily equals the gradient of this potential is prohibited. If an independent force network is used, it should be explicitly classified as a general Neural ODE, and its curl/integrability residual should be measured.

### 6.4 Prefer a Controlled Dissipative Model That Can Express an Open System

For real tool and logic tasks, a more reasonable candidate family is

\[
\dot x=(J-R_\theta(x))\nabla\mathcal H_\theta(x)+G_\theta(x)u,
\quad J^T=-J,\quad R_\theta\succeq0.
\]

Under an autonomous H:

\[
\dot{\mathcal H}=-\nabla\mathcal H^TR\nabla\mathcal H+
\nabla\mathcal H^TGu.
\]

Explicit time dependence adds \(\partial_t\mathcal H\). This is an energy-budget relation, not a safety proof. The first version can restrict to \(\dot p=-\nabla V-\Gamma p+B(q)u\), using dissipative/conservative/dissipative Strang splitting; a constant diagonal damping allows an exact exponential sub-step.

"Conformally symplectic" can only be claimed under specific conditions such as uniform damping; a generic state-dependent R/G or an event system cannot inherit this label. When reusing the existing Rust `contact` numerical foundation, check its actual equations against the target model, rather than attaching it just because the name is similar.

### 6.5 Irreversible Events Must Be Modeled Explicitly

A smooth ODE with a unique solution generally defines a reversible flow over a finite time interval; a pure Hamiltonian also preserves phase volume, and is not suited to compressing multiple states directly into the same absorbing failure state. Even a damped ODE does not automatically become a discrete-sense many-to-one reset over a finite time interval.

Add mode e and event guards/resets:

\[
\dot x=f_{\theta,e}(x,u),\qquad
g_j(x,u)=0\Rightarrow(x^+,e^+)=\mathcal R_j(x^-,u,e^-).
\]

Goal, collision, tool failure, resource exhaustion, and irreversible commitment are different event types. Only a model-predicted event probability should be reported, and it must not be disguised as a fact that has already occurred; only real environment execution produces an observed event label. Guard localization and reset both need supervision or a clear mechanistic source.

"Phase transition" in this report refers to a sudden change in behavior pattern/reachability; it does not automatically carry the thermodynamic phase-transition meaning from statistical physics. A change in a finite-dimensional neural network's output does not constitute a thermodynamic proof.

## 7. Action Geometry: Pullback, Impulse, and Real Semantics

Actions use a stable `ActionId + typed parameters + schema/version + duration`; a name may be displayed, but semantics must not be replaced by a hashed random vector. A finite fixed action set can use a trained embedding or an explicit coordinate encoding; this is only a representation, not "hardcoded reasoning." Unknown actions, illegal parameters, and a missing embedding must all be rejected. Teacher encoding of a tool's text description can be done at the root node/cache stage, not by invoking the 405B model at every search sub-step.

Let the learnable immersion/local decoding from latent space to teacher representation be \(h=D_\psi(q)\), with Jacobian \(J_D\). A force in teacher space is a covector \(\alpha_h\), whose natural pullback is

\[
\alpha_q=D^*\alpha_h=J_D(q)^T\alpha_h.
\]

A vector field has no analogous pullback that holds unconditionally under an arbitrary map. To project a target velocity v in teacher space onto the latent tangent space, one can solve, under a specified metric,

\[
v_q=(J_D^TJ_D+\lambda I)^{-1}J_D^Tv_h,
\]

This is a regularized least-squares projection; it is not a lossless geometric isomorphism. Rank deficiency and a large off-manifold projection residual must be exposed. When the full Jacobian is too expensive, distill it offline into a low-rank B(q), and measure the action-response error.

If an action produces a scalar potential change \(U_a(q)\), then \(-dU_a\) is a one-form of force; the symplectic form establishes its relation to the Hamiltonian vector field. The exterior derivative d is a mathematical operator, not a module that automatically digs a causal effect out of an action string.

Impulse is modeled as \(q^+=q^-\), \(p^+=p^-+I_\theta(q,u)\). This map is only symplectic under conditions such as the impulse corresponding to a suitable closed one-form (locally a scalar gradient); an arbitrary neural B(q)u does not automatically satisfy this. A non-conservative tool action should use a controlled-force/event branch, explicitly giving up an inapplicable conservation claim.

A potential well can be changed by an action, e.g. \(V(q;u)=V_0(q)+\sum_i u_iU_i(q)\). It helps represent attraction/repulsion and decision boundaries, but the well depth must be learned from successors and task supervision; it must not be hand-set so that "the correct answer has low energy" and then have the answer fed in at test time.

MCTS always searches from a legal action set; a continuous vector is only an action encoding. A discrete tool must not run CEM on an embedding and then arbitrarily nearest-neighbor-map it to an action while claiming continuous control. If the existing Rust MPC uses discrete candidates, keep a categorical distribution; only genuinely continuous parameters may be optimized within a legal range, and parameter-coupling constraints need explicit handling.

## 8. Safety, Goals, and Irreversible Traps: Separating Supervision Semantics from Planning Semantics

The model should return at least six independent pieces of information: successor distribution, task reward, immediate hazard, goal, terminal reason, and epistemic uncertainty. The `safe_or_goal` merged label loses a critical distinction, and requires new data to fix; it cannot be recovered out of a single current binary r.

The safe set \(\mathcal S\) and the goal set \(\mathcal G\) are defined separately. A learnable barrier margin \(b_\theta(x)\) can be defined, but it is only a safety certificate after corresponding verification/formal proof. Collision can happen between two safe endpoints; when intermediate observations exist, train path risk on them, and when they don't, only a one-step event probability and its blind spot can be reported: latent-space interpolation must not be called real swept-volume collision detection.

A trap is a reachability question: for finite horizon H, the optimal probability of reaching the goal without failing along the way can be defined as

\[
V_0(s)=1_{\mathcal G}(s),\qquad
V_{h+1}(s)=1_{\mathcal G}(s)+1_{\mathcal S\setminus\mathcal G}(s)
\max_{a\in\mathcal A(s)}\mathbb E[V_h(s')].
\]

The real definition of an irreversible failure depends on the action space, visible information, and time horizon. Not reaching the goal by H=10 does not mean it is never reachable. `hazard_now`, `unreachable_within_horizon`, `irreversible_failure`, and `unknown` must be distinguished; the last category must not be automatically counted as safe.

Risk-constrained planning can optimize

\[
\max_\pi\;\mathbb E[\sum_{t=0}^{H-1}\gamma^tr_t+\gamma^HV(x_H)]
\quad\text{s.t.}\quad P(\exists t\le H:\text{hazard}_t)\le\epsilon.
\]

Only a correctly conditioned survival probability can be multiplied along the sequence; a generic calibrated score cannot be assumed independent. If each step has a valid probability upper bound, a union bound can serve as a conservative budget, but it is not guaranteed to keep holding out of distribution. Record Brier score, NLL, reliability diagrams, risk-coverage, and confidence intervals; normalized action entropy must not be used as the model's epistemic uncertainty.

A model error (NaN, unknown schema, checkpoint mismatch, insufficient budget, missing risk estimate) is `Err`, which by default rejects the whole plan; a predicted legal hazard is a model output, which can be pruned by an explicit policy with the reason recorded. Goal is a successful terminal and should not trigger a danger rejection. The semantics of the current `HazardCheckedDynamics` must migrate together with this; it must not be worked around by turning goal's done into false.

## 9. Performance: Sub-Millisecond Comes From Reducing Compute Scale, Not From Mathematical Terminology

### 9.1 The Compute Bill for Repeated Large-Model Forward Passes

Use an idealized roofline estimate rather than a measured promise: the parameter-dominated compute for one batch=1 Dense token/latent pass is about \(2P\) FLOPs, not counting attention-context and similar costs; if the weights must be read from high-bandwidth storage, at least the corresponding weight traffic is also needed. A rough lower bound:

\[
t\gtrsim\max(2P/F_{\rm eff},\;Pb/B_{\rm eff})+t_{\rm communication}.
\]

As an example, **assuming** an aggregate effective bandwidth of 10 TB/s, BF16 at 2 bytes/parameter, and the weights streamed through that layer of storage each time: 405B weights are about 810 GB, giving a bandwidth term of 81 ms; an idealized 4-bit raw weight set is about 202.5 GB, giving 20.25 ms, not yet counting quantization metadata, dequantization, and communication. This example is not a measurement on an actual machine, nor an absolute lower bound across all hardware; it shows that "skip sampling/softmax" is not enough to conclude <1 ms.

The Dense teacher does not enter every \(\nabla V\) computation. If every gradient had to backpropagate through the 405B model, Verlet's two gradient evaluations could be more expensive than an ordinary single forward pass.

### 9.2 A Measurable Target for the Small Kernel

For the shallow potential above, each gradient is mainly two d×w matrix-vector operations, about 4dw FLOPs; Verlet's two gradient evaluations are about 8dw, plus action, risk, value, dissipation, and memory cost. At d=128, w=256, the gradient part alone is about 262,144 FLOPs. Multiple potential layers, an ensemble, and sub-step counts all expand this cost; this is only a design estimate, **not a measured microsecond figure**.

Candidate implementation: resident Rust weights, pre-allocated scratch, SIMD/GEMV; use GEMM/GPU at larger batch sizes, after measuring the crossover point. Do not call Python/HTTP per node. Do not sacrifice B=1 scheduling latency for the sake of a GPU label. Quantization must be validated against paired trajectories and event boundaries, not by substituting single-output similarity.

The total ledger:

\[
T_{decision}=T_{encode}+T_{legal/actions}+T_{search}+T_{gate/audit}+T_{serialization}.
\]

A single H=10 trajectory still has time dependence; the throughput-amortized time across B trajectories must not be treated as the single-trajectory latency. The number of model calls in sequential MCTS is determined by caching and rollout, and can reach the order of simulations×H; it must actually be counted. Even at 0.2 ms each, 1000 calls already cost about 200 ms of pure serial kernel time.

Acceptance should separately report: B=1 full-transition p50/p95/p99; B=8/32/128 batch latency and throughput; H=1/5/10/20 rollout; the full planner at a fixed decision budget; end-to-end results with and without teacher encoding. GPU timing must be synchronized; cold start, warm-up, concurrent load, CPU affinity, hardware/frequency, precision, NFE, and rejection rate must all be published alongside the report. An initial threshold can be set as `B=1 transition p99<1ms`, but until measured this is only called a target.

## 10. Production Integration Plan: The Full Chain From Export to a Rust Arena

Target chain: real observation → versioned encoder → learned latent → legal action/PolicyGate → `ProductionPipeline` → MCTS/MPC → Rust world model with trained weights loaded → transition/risk/event → backed-up value → real action execution → correction from new observation. System 1 provides a prior over legal candidates; it does not replace the transition model.

### 10.1 Model Package and State Protocol

A model package must bind: format version, model family, checkpoint hash, training-run ID, training/validation data hash, encoder/model/tokenizer/quantization hash, layer and pooling, normalization, latent schema, action schema, event-label definition, integrator/dt/substeps, calibration-set hash, and export precision and numerical tolerance. A self-reported `trained=true` field cannot serve as training evidence; traceable training logs and acceptance artifacts are needed.

At load time, verify shape, finiteness, mass-matrix positivity, integration parameters, the action registry, and the calibration identity. A safety-decision path with a missing file, incompatibility, or lack of calibration must fail explicitly. Automatically falling back to a harmonic oscillator, an old residual default model, a zero vector, or a string hash is prohibited.

A suggested versioned logical interface (design draft, not yet implemented):

```text
ModelIdentity = weights_hash + latent_schema + action_schema + calibration_hash
LatentState = values + schema_id + encoder_id + observation_version
Action = stable_id + typed_parameters + duration + schema_id
Transition = next_state + task_reward + event_distribution
             + terminal_reason + hazard_probability + uncertainty
             + numerical_diagnostics + model_identity
TerminalReason = None | Goal | Hazard | EnvironmentFailure | TimeLimit
WorldModel.transition_batch(requests, output_buffer) -> Result<(), ModelError>
ActionProvider.legal_actions(state, context) -> Result<ActionSet, ActionError>
```

A source/action-dependent hazard must not be smuggled into a `safety_estimate` that currently only receives `(next_state,reward,done)`, then retrieved via a global "last prediction cache": under concurrency this cross-wires. The safety estimate should be returned atomically as part of the same Transition. On a failed batch call, the caller is forbidden from consuming partially written output; either a whole-batch atomic publish or explicit per-item status must be defined: a missing item must not default to safe.

### 10.2 Migration Order and File Ownership

| Stage | Main existing location | Behavior that must actually be completed |
|---|---|---|
| Python reference model and training | `python/gen_zero/world_model/neural_dynamics.py`; `scripts/train_world_model_dynamics.py` | Independent reward/safety/goal/event, trained on real multi-step data; keep the trained residual baseline |
| Unified trait/event contract | `crates/gen-zero-core/src/traits.rs`; `types.rs` | Transition v2, schema and action parameters, error semantics; update all implementations and callers |
| Rust model core and export | `crates/gen-zero-worldmodel/src/` existing integration modules | Import trained weights, align Python/Rust per sample; analytic gradient consistent with the scalar potential |
| Planner semantics | `crates/gen-zero-planner/src/engine.rs`, `pipeline.rs` | Goal is a normal termination, hazard is auditably prunable, model errors are rejected, dynamic legal actions and risk budget |
| Service assembly | `crates/gen-zero-service/src/zero.rs:818`; `pipeline_verb.rs`; `worldsim.rs` | Inject the same model instance into the pipeline and related entry points; remove the implicit choice of the hand-set default production model |
| External acceptance | CLI, HTTP `/v1/pipeline/{op}`, MCP | Requests, responses, call counts, and model hash consistent on the same real fixture; not testing only a simulate function |

A single PR does not need to complete every research variant at once, but a production release must complete the end-to-end contract migration for the corresponding model. A half-finished state of "the trait is updated but the entry point still uses the old model" must not be released. What is listed here is the future scope of change; this work did not modify these files.

### 10.3 MCTS Arena and MPC Batching

The first stage should first verify that the real model works on the existing sequential MCTS, avoiding conflating algorithm change with model change. The second stage then changes the arena: separate node metadata from the continuous-state buffer; record model/schema/version, parent node/action, terminal, risk, and uncertainty; bind the lifetime to one immutable model snapshot.

The current 1024 f32 state needs 4096 bytes of data alone per node; one million nodes is about 4.096 GB, not counting children, values, indices, and allocator. A so-called 64-byte node descriptor cannot represent the full state memory.

Batch MCTS uses explicit leaf selection, in-flight marking/virtual loss, batch transition, and a backup protocol; it must guarantee the same simulation is not enqueued twice, that a failure releases its reservation, and that a timeout does not consume incomplete results. Multiple independent root requests are easier to batch. MPC should batch by horizon order and by different candidates within the same layer. The existing `step_batch` loop implementation must be replaced with a real batched kernel before claiming SIMD/GPU acceleration.

If the dynamics is a stochastic distribution, caching a single deterministic successor per edge is no longer correct. Fixed particle/ensemble semantics, chance nodes, or an explicit risk estimate is needed; a single random sample must not be treated as the permanent true successor forever. If the experiment first uses a deterministic mean model, the risk of losing multimodality must be acknowledged.

The cache key includes state, model version, action parameters, and time, not just ActionId. Near-neighbor merging/quantization collision must be ablated separately; different semantic states must not be silently merged for speed.

### 10.4 Fail-Closed and Auditable Degradation

An unknown action, wrong dimension, non-finite value, encoder mismatch, missing checkpoint, missing risk head, unsupported integration mode, an overly dense event set, implicit-solve failure, and NFE/time-budget exhaustion are all typed errors. Explicitly distinguish "numerical failure," "epistemic insufficiency," "illegal action," and "predicted danger."

If the business allows returning the previously verified incumbent, the response must include `status=partial_budget_exhausted`, the actual horizon, the completed simulation count, the model identity, risk-check coverage, and the reason; it must not share an indistinguishable result with a fully successful plan. A high-risk path can still specify that a timeout is always rejected. The current timeout-returns-incumbent behavior at `engine.rs:188` must be folded into this status contract; giving only entropy=1 does not count as an explanation of completion.

The reuse/replacement boundary needs a symbol checklist listed before implementation begins: if the deprecated random Hamiltonian model, hashed actions, silent padding, and the old production default choice are replaced, they should be removed together from code, exports, configuration, callers, tests, and documentation; a whole-repo `rg` should be run against that checklist, with no matches expected (exit code 1). A correct underlying integrator can be reused directly; it should not be rewritten just for the sake of "deleting everything." The retained residual baseline is an explicit experimental control; it must not also serve as a failure fallback; this work replaced no code, so it does not claim the old symbols are already cleared.

## 11. Training Data and Objective Function: Most of the Work Is Here

Static classification data only gives `(observation, candidates, label)`; a hidden state across Transformer layers is network computation depth, not environment time. Treating a layer difference as `(s_t,s_{t+1})` can be studied as a computation flow, but it must be named independently; it cannot be disguised as an action-conditioned world model.

Each real transition record should log episode/environment/layout/family ID, observed history, the legal action set, the actual action and its parameters, execution time, next observation, reward, terminal reason, hazard/goal/viability labels and their provenance, teacher identity, and feature hash. The teacher encodes an observation that has actually happened next; it must not use its own generated prediction as ground truth. Exploration/intervention data must include failures, rare events, and different actions from the same state; with only single-policy logs and no action support, identification of the full action-causal effect cannot be claimed.

Split data by episode, layout, document/question family, and source, not random rows; set up an independent calibration set. Pre-freeze the test manifest. Changes to action parameters, phrasing, and model quantization distribution also go into the OOD test. Teacher tokenization and budget must be bound to the schema.

Suggested multi-step training objective:

\[
\mathcal L=\sum_{k=1}^{K}w_k\{\lambda_z\ell_z(\hat x_{t+k},\operatorname{sg}(\bar\Phi(o_{\le t+k})))
+\lambda_r\ell_r(\hat r,r)+\lambda_e\operatorname{CE}(\hat e,e)
+\lambda_s\operatorname{BCE}(\hat p_{hazard},y_{hazard})
+\lambda_v\ell_v(\hat V,V^{target})\}+\lambda_{reg}\mathcal R.
\]

The latent-variable target uses a frozen or EMA target encoder, plus a retained task-prediction/variance constraint to prevent the encoding from collapsing to all zeros. When the state is multimodal, use a distributional loss rather than only MSE; the mean may fall at a physically meaningless location. The target value must be explicitly labeled as coming from a real return, a verifiable simulator, or bootstrapping, with each error reported separately.

\(\mathcal R\) can include Jacobian/curvature control, a legal energy-budget residual, and representation stability; it must not force \(\Delta H=0\) on data that has control work/dissipation/events. Use the same integrator during training as at deployment, starting with one step and gradually increasing to real multi-step unroll; report both teacher forcing and free rollout, not only the former.

First freeze the teacher and train projection + residual + heads; then swap the dynamics under the same data/parameter/tuning budget; only afterward evaluate multi-teacher fusion and student encoding. Data scale, teacher, model capacity, and planner budget must not all be changed at once and then have the improvement attributed to symplectic integration.

## 12. A Reproducible Experiment Matrix Based on Existing Suites

### 12.1 Unify the Benchmark Identity First

There are currently two different "13 tasks" sets: `grand_challenge_data.py:49` is MASSIVE en/de, MultiNLI, PubMedQA, VitaminC, BoolQ, SQuAD2, PAWS, Civil Comments, Aegis, HelpSteer2, SummEval relevance/consistency; `evaluate_full_suite_cpu_dynamics.py:78` uses an older set that includes ARC/GSM8K and similar, documented as 30 frozen 9B samples per task.

They must not be mixed into one "13-task score." A run identity needs the suite file hash, task list, split, row count, and ID/family hash. The 3880-row and similar figures noted in comments need to be recounted from the data at the time of a formal run. This work did not audit whether all five teacher artifacts actually exist, are complete, and correspond to the same manifest.

### 12.2 Four Experiment Tracks

| Track | Reused entry point/data | What it can prove and what it needs |
|---|---|---|
| A: Static representation and zero-decode readout | `grand_challenge_data.py`, Dense extractor, `cross_model_manifold_alignment.py`, `equivariance_suite.py` | Measures candidate accuracy/ranking, teacher contribution, permutation equivariance; cannot prove environment rollout |
| B: Real controlled transitions | `deadlock_torus_env.py`, `scripts/extract_trajectories.py`, `scripts/evaluate_world_model_dataset.py` | Distinguish between the `TorusWorld` and `DeadlockTorusEnv` APIs found in the files; state observation visibility explicitly, and add goal/hazard/trap/duration labels |
| C: Production planning benefit | Rust MCTS/MPC + CLI/HTTP/MCP, unseen Torus layouts | Replace the privileged exact model with an actually exported trained model; pair the same planner/budget, and record real environment results |
| D: Numerical and system cost | Existing latency/profile suites + real model artifacts | Numerical fidelity, batching benefit, end-to-end latency/memory; the small random kernel is only an independent microbenchmark |

If the static NLP track wants to study multi-step "logic actions," a new environment with real execution semantics must be built, for example a verified evidence-retrieval action, a constraint update, or a tool query; the action should not be "guess the next piece of text." The raw static samples themselves do not provide this transition supervision; missing data should be recorded as not completed.

### 12.3 Required Ablations

| Question | Held constant | Control |
|---|---|---|
| Is planning even necessary? | encoder, training data, test episodes | System 1, H=1, H=5, H=10; match wall-clock budget |
| Is the continuous form useful? | parameter count/training steps and data, event head | residual MLP, fixed-NFE Neural ODE |
| Is the symplectic structure useful? | same coordinates, potential capacity, data | non-symplectic vs. symplectic integration of the same learned system; also compare a trained unrestricted MLP |
| Is the conservation assumption harmful? | sample/action and capacity budget | conservative, damped/port, hybrid events |
| Does the action actually affect the prediction? | state encoder | real action, action removed, action shuffled during training as a negative control; a negative control must never be used as a production implementation |
| Is a bigger teacher better? | paired samples, downstream training budget, precision/extraction metadata | each Dense teacher, no teacher/small encoder; also control for budget against MoE |
| Does q/p provide information? | training data | learned history momentum, single state, known physical velocity (observable track only) |
| Does the improvement come from spending more time? | wall clock and call count each matched | more integration sub-steps, more simulations, an MLP with the same total compute |
| Is it just terminal classification? | prediction head/data | direct readout, one-step model, multi-step free rollout |
| Does quantization/batching break semantics? | same weights and per-sample input | Python FP32, Rust FP32, later quantization; B=1 vs B>1 |

Also test candidate permutation. Policy scores should be equivariant under candidate permutation; when there are tied optima, compare the optimal set or break ties with a stable action ID; the difference caused by slot order must not be hidden.

### 12.4 Paired Statistics and the "Can Be Claimed" Threshold

For each sample/episode, save the baseline's and the variant's prediction/action, real outcome, hazard/goal, risk, latency, model call count, seed, model/data hash, and error status. Use McNemar for paired classification; use episode/family cluster bootstrap for return/success-rate differences; multiple seeds reflect training randomness, and multiple inferences on the same sample must not be treated as independent samples expanding n. Multiple ablations must pre-specify the primary endpoint and correct for multiple comparisons.

The accuracy difference \(\Delta=\frac1N\sum_i(I_i^{new}-I_i^{base})\) must be accompanied by a paired confidence interval and the raw b/c discordant counts; with only a mean change and no per-sample pairing, no breakthrough can be claimed. A safety miss rate needs a one-sided upper confidence bound; for example, with zero misses among n independent risk samples, the 95% upper bound is approximately 3/n, not zero risk. Correlated sampling cannot use an independent-sample conclusion.

Suggested pre-registered acceptance: the lower bound of the paired interval for accuracy/success-rate improvement is >0; the safety miss-rate upper bound is no worse than a preset tolerance; B=1 transition p99<1ms; the planning benefit still holds at a fixed wall clock; the new entry point's call count is non-zero; error injection is entirely fail-closed. The specific risk tolerance is set by the deployment task; "safety passed" must not be published before it is set.

Numerical stability of the model, risk calibration, task effectiveness, and production wiring are four independent gates; they cannot substitute for each other. If the Hamiltonian shows no benefit while the residual meets the bar, the report should clearly state that the structural hypothesis was not supported, rather than switching metrics to rescue the conclusion.

## 13. Technical Roadmap and Stop Conditions

| Stage | Deliverable | Exit condition |
|---|---|---|
| P0 Facts and contract | Data/model manifest; termination, safety, action, and schema definitions; legacy-symbol replacement checklist | No mixing of static/trajectory data; goal is not treated as hazard |
| P1 Trustworthy baseline | Residual trained on deployable observations; independent calibration and test kept separate; model package exported | Paired prediction statistics on unseen episodes; training and artifacts traceable |
| P2 Minimal production wiring | Python/Rust parity; CLI/HTTP/MCP use the same model; the existing sequential MCTS calls a real checkpoint | Model-kernel call count >0 in a real request trace; a request fails when the model is missing; goal/hazard correctly separated |
| P3 Dynamics-structure experiments | Ablations among ODE, Hamiltonian, controlled dissipative, and hybrid, all controlled | Structure is chosen only when the data supports it; if not supported, keep the trained baseline and report a negative result |
| P4 Performance | Analytic gradient, pre-allocation, MPC batching, then batched MCTS; quantization as needed | Full-transition p99 and fixed-budget decision both meet the bar; failures/rejections counted in the results |
| P5 Dense scale-out | Per-sample comparison across five teachers, cross-model dynamics consistency, student root encoding | Benefit exceeds cost and OOD does not worsen; parameter count is not substituted for evidence |
| P6 Production migration and cleanup | Unified assembly, error injection, zero legacy-symbol residue, rollback-capable version artifacts | Both numerical/semantic checks and real acceptance at every entry point pass; independent reviewer review |

The stages are not date commitments. The most likely blockers are missing real transition data, representation state aliasing, and insufficient calibration, not necessarily Rust compute. With only static question data available, complete track A first and explicitly state that B/C are not done; a "virtual trajectory" must not be fabricated to fake closure.

When heavy training/compilation is genuinely needed later, execute per the user-specified remote standard: first verify each target machine's 1m/15m load, CPU count, available RAM ≥8GB, disk ≥10GB, with headroom added based on the task's actual peak; only send a content-verified source snapshot to a `.git`-free sandbox; isolate reusable caches to avoid multiple full-core tasks competing; use `CARGO_BUILD_JOBS=$(nproc)` only when that node's load and memory allow it. Save the command, environment, full log, raw exit code, and artifact hash; verify provenance when pulling results back. Keep the final report/acceptance log locally, and clean up the temporary remote log and sandbox. Committing and pushing require explicit future task authorization; this work performed no remote heavy task, commit, or push.

## 14. Evidence, Recheck Commands, and Three Categories of Outcome

### Implemented / Completed in This Work

Completed a source-code check, literature check, mathematical design, interface/migration route, ablation and statistics plan, and this report. The number of new algorithms implemented is zero, consistent with "do not modify code."

The source-code evidence commands are the argv values saved in `evidence.json`, which can be run as-is from the repo root. For example:

```bash
rg -n 'pub type FullLatent|fn step_batch|fn safety_estimate|world_model.step|struct MctsNode' crates/gen-zero-core/src crates/gen-zero-planner/src/engine.rs
rg -n 'world_model: Arc::new|execute_pipeline|SymplecticWorldModelDynamics::default|as_dyn|neural_dynamics_model|dynamics_model=None' crates/gen-zero-service/src/zero.rs crates/gen-zero-service/src/worldsim.rs crates/gen-zero-service/src/pipeline_verb.rs python/gen_zero/client.py
rg -n 'rng.uniform|rng.randn|Pad or truncate|hash\(action\)|r_hat <|weights_loaded|load_checkpoint' python/gen_zero/world_model/hamiltonian_dynamics.py python/gen_zero/world_model/neural_dynamics.py
```

The raw exit codes for the above formal check logs `01`–`06` are all 0. The tail of `05`'s output includes `hamiltonian_dynamics.py:221: # Pad or truncate to match action_dim` and `:249: h = abs(hash(action)) % (2**31)`. The behavioral reproduction's exit code was 0; output is in `13-behavior.log`. These are risk evidence, not evidence that the model passed acceptance.

During exploration, `rg` was run against non-existent paths `gen-zero-api/gen-zero-mcp/gen-zero-server/src`, and the tool returned exit code 2; the check was then completed using the real `gen-zero-service` path. Those failed lookups were not treated as evidence of "zero references across the whole repo." The initial listing search also used truncated output at times, used only for locating things, not as proof of completeness or test success. The formal evidence commands have no build-pipe truncation.

### Unverified

The completeness of the five teachers' features and the actual usability of the model weights; the training quality of the checkpoints already in this working tree; the Python/Rust numerical consistency of the new model; Dense's predictability advantage over MoE; the Hamiltonian's benefit over a trained residual; sub-millisecond full transition, end-to-end latency, and safety calibration. All of these need future real execution.

### Not Completed

Filling in real transition data, training/exporting the new model, the trait and terminal migration, Rust main-chain assembly, batched MCTS, deletion of the old module, production deployment, and independent reviewer acceptance. The reason is that this task was explicitly scoped to research and design, with code changes prohibited; design text must not be written up as an already-shipped outcome.

Conclusions of the four-part review: some existing models were found to have real entry points, so it cannot be generally called an island; silent Python action-shape correction and non-semantic random actions were found; the existing numerical benchmarks are not enough to support a claim of a world-modeling breakthrough; the proposed new model has not yet been implemented, deployed, or replaced the old logic. What this report provides is an implementation plan that can be falsified stage by stage.
