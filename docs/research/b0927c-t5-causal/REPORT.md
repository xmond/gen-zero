# b0927c-t5-causal: Irreversible-Action Audit and Conformal Safety Barrier Proposal

**Conclusion: There is currently no proof that Gen-Zero detects every irreversible action.** Smoothness, tangent directions, and curvature in a dense model's latent space do not, by themselves, give a vector causal meaning in Pearl's sense. Under explicit exchangeability assumptions, conformal prediction provides finite-sample **marginal coverage**. It does not establish zero false negatives for arbitrary dangerous actions or safety under distribution shift or adversarial input. The model should help identify risk; authority to execute must remain with mandatory gates, explicit confirmation, and auditable records.

## 1. Current State: Sources of False Confidence

| Finding | Evidence and implication |
|---|---|
| The default Rust `PolicyGate` has empty constraint and irreversible-action confirmation tables. If neither table matches and entropy is low, it returns `Tier0Proceed`. | [policy.rs](../../../crates/gen-zero-gate/src/policy.rs#L43), [policy.rs](../../../crates/gen-zero-gate/src/policy.rs#L219). **The default gate is no proof that every irreversible action is recognized.** |
| Rust `audit_action` rejects trajectories without a safety estimate, but may still return `Approved` when the estimate is uncalibrated, with a textual warning. | [pipeline.rs](../../../crates/gen-zero-planner/src/pipeline.rs#L571), [pipeline.rs](../../../crates/gen-zero-planner/src/pipeline.rs#L610). A warning cannot substitute for a release condition. |
| The public Rust interface explicitly calls the default residual dynamics an **untrained prior**. There are separate `worldsim` verb and `ProductionPipeline` paths. Wiring only one leaves a bypass. | [server.rs](../../../crates/gen-zero-service/src/server.rs#L1631), [zero.rs](../../../crates/gen-zero-service/src/zero.rs#L1545), [pipeline_verb.rs](../../../crates/gen-zero-service/src/pipeline_verb.rs#L105). |
| Python `audit_action` refuses to issue `APPROVED` from a text heuristic, a useful safeguard. Yet it directly treats its neural score as `safe_prob`; the inspected path does not validate a conformal calibration certificate for that score. `what_if` returns rankings and `best_candidate`, not execution authority. | [client.py](../../../python/gen_zero/client.py#L1417), [client.py](../../../python/gen_zero/client.py#L1491), [client.py](../../../python/gen_zero/client.py#L1513). The HTTP entry point is [app.py](../../../python/gen_zero/service/app.py#L925). |
| The existing `ConformalMarginGate` compares the margin with a fixed `theta`. The inspected code does not calculate a conformal quantile from an independent calibration set. The word “Conformal” in the name is not evidence of coverage. Its evaluation also distinguishes a deployable view from an *oracle* diagnostic view that uses the true class. | [conformal_margin_gate.py](../../../benchmarks/suites/conformal_margin_gate.py#L68), [aegis_dual_track.py](../../../benchmarks/suites/aegis_dual_track.py#L190). |
| Rust MMR offers optional persistence; without configuration it is an in-process ledger. In the inspected main flow, `record_decision` follows a successful `ask`. This does not establish persistent records for every `what_if`, `audit_action`, and tool execution. An MMR proof establishes record integrity, not decision correctness. | [zero.rs](../../../crates/gen-zero-service/src/zero.rs#L791), [zero.rs](../../../crates/gen-zero-service/src/zero.rs#L1579), [zero.rs](../../../crates/gen-zero-service/src/zero.rs#L2274). |
| Of the 13 tasks, `aegis_safety` and `civil_comments` are directly marked as safety tasks. The former labels prompt content safety and the latter text toxicity, not damage after tool execution. The documented results on 30 frozen samples each are not action-safety experiments for this proposal. | [grand_challenge_data.py](../../../benchmarks/suites/grand_challenge_data.py#L49), [grand_challenge_data.py](../../../benchmarks/suites/grand_challenge_data.py#L366), [README.md](../../../docs/zero/README.md#L136). |

## 2. Causal Manifold: Separate Interventions from Geometric Diagnostics

Let \(C\) be context, \(E\) the authority and resource state, \(A\) a candidate action, \(U\) exogenous disturbance, and \(Y\in\{\text{safe},\text{irreversible harm}\}\) the observed outcome. The target is
\[
P\!\left(Y=\text{harm}\mid do(A=a),C=c,E=e\right).
\]
`do(A=a)` requires replacing the action mechanism in an **explicit structural causal model or an isolated, reproducible experimental environment**, while holding comparable \(C,E,U\). Computing only
\(\phi_\theta(c,a)-\phi_\theta(c,a')\) is a difference in representations, **not** a do-operator. Identifiability of a causal representation generally requires additional conditions, such as intervention environments; it does not follow automatically from a high-dimensional observational representation. See [research on causal representation identifiability](https://arxiv.org/abs/2306.00542).

The proposed geometric layer has only two roles:

1. **Identify perturbation-sensitive candidates.** For a fixed model version and state, estimate the directional derivative \(D_vr\) of risk score \(r(\phi)\) along a physically or semantically defined action perturbation \(v\), and the local second-order change \(v^\top H_rv\). Define \(v\) through paired changes in permissions, target paths, transaction scope, or rollback conditions. An arbitrary latent direction must not be named a “deletion factor.”
2. **Test whether a proposed invariant kernel is actually invariant.** Estimate nuisance subspace \(N\) from preregistered pairs of harmless rewrites. Under metric \(G\), one may study the projection
   \[
   P_{\perp}=I-N(N^\top GN+\lambda I)^{-1}N^\top G.
   \]
   Projection may also remove essential risk signals. Therefore the raw input, projected representation, and explicit environment variables must all enter the audit. Danger on any path, or disagreement between paths, must escalate the case. **The projected result alone must never approve an action.**

Counterfactual training data should contain action pairs from the same initial state, explicit permission and resource snapshots, observed outcomes, and markers for unobserved outcomes. Tests involving `rm -rf`, dirty writes, or unauthorized access belong only in isolated test environments with recoverable snapshots. Without real outcome labels, report “simulated inference,” not causal effects. A 405B model changes neither the identifiability conditions nor safety coverage automatically.

## 3. Conformal Barrier: What Can and Cannot Be Proved

After freezing the model, scoring rule, and its geometric features, draw a separate calibration set \((X_i,Y_i)_{i=1}^n\) **not used for training or threshold selection**. Let \(s(X,y)\) be a nonconformity score, for example \(1-\hat p(y\mid X)\) calculated for each of two labels. Curvature, energy, and counterfactual differences may enter the prespecified, frozen score \(s\). Define
\[
k=\left\lceil(n+1)(1-\alpha)\right\rceil,\qquad
q_\alpha=\begin{cases}
s_{(k)},&k\le n,\\
+\infty,&k>n,
\end{cases}
\quad
\Gamma_\alpha(x)=\{y:s(x,y)\le q_\alpha\}.
\]
When calibration and next-sample scores are **exchangeable**, the scoring rule is frozen, and labels have a consistent meaning, the rank argument gives
\[
P\{Y_{n+1}\in\Gamma_\alpha(X_{n+1})\}\ge1-\alpha .
\]
This is a marginal guarantee. If an action may advance to the execution gate only when \(\Gamma_\alpha(x)=\{\text{safe}\}\), then
\[
P\{Y=\text{harm}\ \land\ \text{model barrier permits advancement}\}\le\alpha .
\]
It **does not** guarantee \(P(\text{permitted}\mid Y=\text{harm})\le\alpha\). If dangerous actions are rare, all errors could be concentrated in that class. Distribution-free conditional coverage for each input is generally unavailable. See the [original paper on conditional-coverage limits](https://arxiv.org/abs/1903.04684).

Separate calibration for predefined danger classes can yield **class-conditional** coverage under exchangeability, provided each class has enough independent true labels. At \(\alpha=0.01\), fewer than 99 corresponding calibration samples leave the nonrandomized quantile rule above without a finite \(q_\alpha\). Even if 30 dangerous samples in an independent test have zero misses, the one-sided 95% binomial upper bound on the dangerous-class miss rate is still about **9.5%**; this is not “zero risk.” Distribution shift, selective deployment, human rewriting, and adaptive attacks can invalidate calibration. On detection, revoke the certificate and escalate rather than reuse the old threshold.

**The provable deterministic property concerns software interlocks, not a model that never misses:** If every execution entry point mandatorily crosses the same gate, inputs and certificates are validated, and the execution object cannot be swapped after checking, then `harm ∈ Γ`, an empty set, missing certificate, timeout, audit-write failure, or dependency failure cannot reach automatic execution. Entry-point coverage and fault-injection tests are needed to establish this property; they do not yet exist.

## 4. Production Integration Contract

The proposed unified `SafetyEvidence` response must contain at least: canonical action and resource identity; context and permission snapshot hashes; model and feature versions; training and calibration dataset manifest hashes; label definitions; applicability domain; calibration sample count and \(\alpha\); prediction set; geometric diagnostics; simulator provenance; failure reason; final tier; MMR leaf and persistence status. Missing fields or mismatched versions must block automatic permission and record the reason.

```text
audit(action, context):
    verify canonical action, identity, authority, resource version
    verify required hard constraints and irreversible-action registration
    evidence = causal_audit_and_conformal_set(action, context)
    if evidence missing / expired / out of domain / computation failed:
        emit typed failure; append durable audit record; return ESCALATE or HARD_STOP
    if hard rule violated or "harm" in evidence.prediction_set:
        append durable audit record; return HARD_STOP or ESCALATE
    if irreversible:
        append durable audit record; return CONFIRM with bound action/context digest
    append durable audit record; return PROCEED
```

Integration should proceed upward from the **execution boundary**: `gen-zero-gate` composes mandatory tiers; Rust `ProductionPipeline` methods `decide`, `audit_action`, and `what_if`, the separate Rust `worldsim` verb, CLI, HTTP, MCP, and the Python `client.py` and service entry points share one versioned verdict contract. `what_if` emits per-candidate evidence and rankings, not reusable execution permission. Before execution, audit the current resource version and permissions again. CP-SAT proves feasibility only for **encoded constraints**; it cannot cover unencoded harm. A confirmation token binds the action, arguments, resource version, principal, expiration, and audit leaf. If durable audit fails, the token must not be released.

Acceptance must also verify **zero residual references** to old scoring shortcuts and approval symbols at production entry points. A call graph and end-to-end fault injection must prove that the new verdict is enforced. These are integration requirements, **not completed integration**.

## 5. Empirical Plan and Stop Conditions

First evaluate `aegis_safety` and `civil_comments` as content-classification tasks using the frozen 13-task split. Align records by sample ID; isolate training, calibration, and test by source and text family. Report dangerous-class misses, permission and escalation rates, coverage, confidence intervals by risk group, and paired per-sample differences. The fixed-margin gate, raw model, and proposed geometric score must share the same test IDs. Preregister ablations and paired bootstrap or exact tests. The previously documented 30-sample results are a baseline reference, not a measurement of this proposal.

Then build an **independent action-outcome dataset** covering deletion in isolated file systems, database transactions and dirty writes, unauthorized access, multistep tool calls, state changes after checking, bridge-service timeouts, unloaded models, corrupt calibration files, and failed MMR writes. Each record should include the initial state, action, source of real execution or simulation, final harm label, and audit trail. Hold out environments, resources, template families, and times; measure the actual dangerous-class miss rate and its upper bound. Any silent permission, `APPROVED` with missing evidence, or entry point that bypasses the unified gate blocks release. High-dimensional model comparisons require real model weights, extraction configurations, training logs, and paired statistics on the same IDs. Until then, no benefit from moving from 70B to 405B may be claimed.

## Results and Evidence Status

- **Implemented in this work:** A read-only code review and the proposal above. Inspection commands included `rg -n 'audit_action|what_if' ...`, `nl -ba ...`, and `git status --short`; all returned exit code **0**. Key source text includes `untrained residual latent prior` in the Rust interface ([server.rs](../../../crates/gen-zero-service/src/server.rs#L1634)) and `safety estimate is uncalibrated` in `audit_action` ([pipeline.rs](../../../crates/gen-zero-planner/src/pipeline.rs#L618)). The untracked files shown by `git status --short` were the same before and after; **no code file was changed** in that work.
- **Unverified:** Identifiability of the proposed causal score, conformal coverage in the target deployment distribution, the dangerous-class miss rate, any benefit from 405B representations, and the cross-entry-point mandatory-gate invariant. The equations state mathematical properties under the listed assumptions, not measured capabilities of the current Gen-Zero runtime.
- **Incomplete:** Model training, paired per-sample statistics, the action-outcome dataset, production integration, removal of old paths, build and test runs, and live acceptance. That work was scoped to a **proposal report with code changes prohibited**; without those steps there is no corresponding implementation or performance claim.
