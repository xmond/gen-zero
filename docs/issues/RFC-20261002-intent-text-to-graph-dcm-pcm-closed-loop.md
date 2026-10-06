# [RFC/Issue] Neural-Causal Autonomous Closed-Loop Architecture: Text-to-Graph Intent Identification, Graph-Theoretic Scheduling, Dual-Track DCM/PCM Execution, and Dynamic Tool Crystallization

- **Issue Type**: `Architecture RFC` / `Core Design Specification`
- **Related Modules**: `crates/gen-zero-lod`, `crates/gen-zero-service`, `crates/gen-zero-model`, `crates/gen-zero-gate`
- **Proposed Date**: 2026-10-02
- **Status**: `PROPOSED` / `IN_REVIEW`

---

## 1. Background and Problem Statement

Existing Agent architectures centered on autoregressive large language models (Generative LLMs) face three fundamental engineering bottlenecks when deployed in industrial control and high-throughput production environments:

1. **Token waste and tail latency**: The entire workflow of planning, intent parsing, parameter extraction, tool invocation, and safety review is delegated to a 100-billion-parameter-scale autoregressive model for one-step generation. Each decision produces hundreds to thousands of generated tokens, driving end-to-end latency to several or even tens of seconds and failing to meet microsecond-level requirements at the edge and under high concurrency.
2. **Weak determinism and unauthorized-action risk (Hallucination & Jailbreak)**: An LLM is fundamentally a probability-based token sampler. It is susceptible to prompt-injection attacks and hallucination-induced instructions and, without physics- and mathematics-level gates, can directly invoke physical tools with destructive side effects (such as database deletion, high-risk transfers, or privilege escalation). It therefore lacks a formal, immutable security boundary.
3. **Rigid tools and broken extensibility**: Traditional systems restrict tool definitions to hard-coded code (APIs/scripts). When they encounter obscure or long-tail ambiguous scenarios for which no ready-made tool exists, they fail immediately and cannot generalize dynamically. Conversely, relying entirely on an LLM to hand-write a temporary script for every request introduces inefficiency and the risk of unauditable code injection.

---

## 2. Core Vision and Principles

Build an end-to-end autonomous closed-loop agent architecture based on **Neural Perception (No-LLM high-speed bidirectional core) + Symbolic Causal Manifold (LOD-Graph) + Deterministic Tools (DCM) + Cognitive Reporting and Soft Operators (PCM)**:

- **0-Token core control loop**: Routine intent parsing, causal-topology validation, and orchestration of existing tools are completed inside the graph using microsecond-scale pure Rust computation, consuming no LLM-generated tokens;
- **Dual-operator unification of hard and soft operators (DCM & PCM Dual Operators)**:
  - **DCM (Dynamic Causal Mechanisms)**: Physical, deterministic atomic hard tools (APIs, SQL, Shell, system calls);
  - **PCM (Predictive Causal Models)**: Flexible, generalizing cognitive soft operators (a local 0.5B small model or lightweight bidirectional core) that fill long-tail gaps when no ready-made hard tool exists;
- **Dynamic tool crystallization flywheel (Tool Crystallization)**: When PCM soft execution recurs in a given class of scenarios, the system triggers the LLM to automatically crystallize the reasoning logic and compile it into a purely functional WASM/Python deterministic DCM operator. The graph becomes faster and more complete with use.

---

## 3. End-to-End Topology and Execution Flow

```text
                  [User Natural-Language Input: Intent / Task]
                                │
                                ▼
    ┌───────────────────────────────────────────────────────────┐
    │ 1. Intent Identification and Causal Topology Extraction   │
    │    (Text to Graph)                                        │
    │    • Lightweight bidirectional core (MiniLM / DeBERTa)   │
    │      extracts entities and action dependencies           │
    │    • SQuAD 2.0-style Answerability gate (AUROC > 0.93)  │
    │    • Parses preconditions and resource consumption,      │
    │      directly generating Causal DAG/LodNode              │
    └─────────────────────────────┬─────────────────────────────┘
                                  │ Explicit Topological Structure
                                  │ (Causal DAG / LodNode)
                                  ▼
    ┌───────────────────────────────────────────────────────────┐
    │ 2. LOD-Graph Topology Hub and Hard Gate                   │
    │    (The Arbiter of Truth)                                 │
    │    • Projects state onto a mixed-curvature manifold       │
    │      (H^4 x S^3 x R^8)                                    │
    │    • PPR diffusion-based context association and Falsifies │
    │      conflict resolution                                  │
    │    • CausalGate / ILP topological hard-gate checks        │
    │      (0 unauthorized actions, 0 violations)               │
    │    • Orchestrates composite tasks into a topologically    │
    │      ordered tool-call pipeline                           │
    └──────────────┬─────────────────────────────┬──────────────┘
                   │ Existing Tool Composition    │ Missing Capability
                   │ (DCM)                        │ (Missing Tool)
                   ▼                             ▼
    ┌───────────────────────────┐ ┌─────────────────────────────┐
    │ 3. Dynamic Causal          │ │ ★ Flywheel A: LLM Dynamically│
    │    Mechanisms (DCM)        │ │   Synthesizes New Tools      │
    │    • Single-shot/parallel │ │   • LLM writes standalone    │
    │      deterministic calls  │ │     scripts/WASM             │
    │    • Executes external    │ │   • Registers them in graph  │
    │      APIs/system actions  │ │     after sandbox tests pass  │
    │    • Carries a one-time   │ └──────────────┬──────────────┘
    │      authorization (Nonce)│                │
    └──────────────┬────────────┘                │
                   │                             │
                   └──────────────┬──────────────┘
                                  │ Real Execution Evidence and Trace
                                  │ (Exit Code, Log)
                                  ▼
    ┌───────────────────────────────────────────────────────────┐
    │ 4. Cognitive Prediction and Result Reporting               │
    │    (PCM / LLM Reporting)                                   │
    │    • Predictive causal model (PCM) evaluates state         │
    │      transitions and residual evolution                    │
    │    • Small model (0.5B) or LLM aggregates a natural-       │
    │      language report from the real-evidence Trace           │
    │    • Zero hallucinations: every conclusion is supported    │
    │      by real DCM execution evidence                         │
    └───────────────────────────────────────────────────────────┘
```

---

## 4. Core Technical Design Specification

### 1. Unified Operator Algebra Contract (Causal Operator Interface)

Abstract the state-transition operator interface in `crates/gen-zero-lod`:

```rust
pub trait CausalOperator: Send + Sync {
    /// Operator metadata and model-space identity
    fn signature(&self) -> OperatorSignature;
    
    /// Hard-gate check of precondition invariants
    fn check_preconditions(&self, state: &GraphState) -> Result<(), GateRejection>;

    /// State-transition execution: DCM uses physical code; PCM uses model-based soft computation
    fn transit(&self, state: &GraphState, input: &OperatorInput) -> Result<OperatorOutput, ExecutionError>;

    /// Post-condition compliance verification
    fn verify_postconditions(&self, output: &OperatorOutput, input: &OperatorInput, state_after: &LodNode) -> Result<PostconditionReport, LodError>;

    /// Whether the operator has irreversible physical side effects
    fn is_pure(&self) -> bool;
}
```

### 2. PCM Flexible Soft Fallback When a Tool Is Missing (Soft Fallback)

- When an action node parsed by LOD-Graph does not match any locally registered DCM operator, the graph **does not throw an exception and halt**. Instead, it automatically routes the action to a built-in PCM (such as a Native Qwen or MiniLM semantic head);
- **Constraint guarantees**: The inputs and outputs of the PCM soft operator remain constrained by the graph Schema and post-condition assertion gates, ensuring a valid output format and blocking illegal-state contamination.

### 3. Automatic Tool Crystallization Pipeline (Tool Crystallization Compiler)

1. **Trace Harvesting**: Record in persistent logs the input/output pairs $(x_i, y_i)$ for successful PCM soft executions that pass gate validation;
2. **Operator Synthesis**: Trigger the LLM to write a purely functional Python module or Rust/WASM source;
3. **Property Checking and Sandbox Audit (Verification)**: Run property-based testing and AST static analysis in an independently isolated WASM sandbox, strictly prohibiting unauthorized system calls;
4. **Seal into Graph**: Compute the BLAKE3 signature and register the result as a persistent DCM hard node. Subsequent identical intents use microsecond-scale offline code 100% of the time.

### 4. Multi-Scale LOD Layered Mapping (Multi-Scale Execution)

- **Lod3 (System Strategy Layer)**: Ingest unstructured user intent (Natural Language Intent) and parse global goals;
- **Lod2 (Tactical Planning Layer)**: Solve the coarse-grained causal topology graph (Macro-DAG) and its dependency relationships;
- **Lod1 (Operator Scheduling Layer)**: Bind atomic action operators (dynamically selecting a DCM hard tool or PCM soft operator);
- **Lod0 (Physical Evidence Layer)**: Ground Truth Bytes generated by physical calls (exit codes, diffs, hashes, and system logs), serving as the sole factual basis for reporting.

---

## 5. Implementation Roadmap and Acceptance Criteria

| Phase | Deliverable Module | Key Outputs and Acceptance Metrics |
| :--- | :--- | :--- |
| **Phase 0** | Interface Contract Definition | Define `CausalOperator`, `OperatorSignature`, and `OperatorKind::{HardDcm, SoftPcm}` in `gen-zero-lod`. |
| **Phase 1** | Text-to-Graph Extractor | Implement intent and precondition-dependency extraction based on the lightweight bidirectional core; output a valid `CausalDagSpec`; per-step latency $\le 5\text{ms}$. |
| **Phase 2** | PCM Soft-Fallback Channel | In `graph_verb.rs`, connect automatic PCM proxying and post-condition verification when a tool is missing, eliminating error-and-halt behavior for undefined tools. |
| **Phase 3** | WASM Sandbox and Automatic Crystallization | Implement a `wasmtime`-based secure sandbox and offline tool-compilation pipeline; validate the end-to-end closed loop that automatically solidifies high-frequency PCM patterns into DCM. |

---

## 6. Security and Anti-Corruption Principles

1. **No Silent Bypass**: Soft execution (PCM) and hard execution (DCM) must explicitly annotate `operator_kind` in the Trace and must not be conflated or impersonated;
2. **No Privilege Escalation**: Every DCM tool that produces external physical side effects must hold a one-shot Nonce token for the current clock cycle;
3. **Strict Fail-Closed Behavior**: If a post-condition assertion is not satisfied or an irreversible exception occurs, the graph immediately triggers a reverse `Falsifies` fixed-point rollback along causal dependencies, rejecting downstream propagation of dirty data.
