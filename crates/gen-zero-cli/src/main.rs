mod reflex_cmd;
use anyhow::Context;
use clap::{Parser, Subcommand};
use gen_zero_service::zero::{DEFAULT_TENANT, DEFAULT_WORKSPACE};
use gen_zero_service::{McpServer, MountKey, MountRegistry};
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "gen-zero", author, version, about = "Gen-Zero SOTA Rust Cognitive Decision Engine", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Commands {
    /// Execute a pipeline decision and print its in-process audit commitment.
    AuditLedger {
        /// Pipeline decision JSON object, or @file containing one.
        #[arg(long)]
        request: String,
        /// MctsEngine::max_simulations override (modes mcts and auto's K2/K3 tiers).
        #[arg(long)]
        max_simulations: Option<usize>,
        /// MctsEngine::c_puct override: PUCT exploration weight (modes mcts and
        /// auto's K2/K3 tiers).
        #[arg(long)]
        c_puct: Option<f32>,
        /// MpcCemEngine::num_samples override (modes mpc_cem and auto's K3 tier).
        #[arg(long)]
        num_samples: Option<usize>,
        /// MpcCemEngine::horizon override: steps per sampled CEM trajectory, not
        /// the rollout horizon inside --request (modes mpc_cem and auto's K3 tier).
        #[arg(long)]
        cem_horizon: Option<usize>,
        /// AStarEngine::uncertainty_penalty_weight override (modes astar and
        /// auto's K1/K3 tiers).
        #[arg(long)]
        uncertainty_weight: Option<f32>,
        /// DynamicKMoERouter::entropy_threshold_low override (mode auto only).
        #[arg(long)]
        entropy_threshold_low: Option<f32>,
        /// DynamicKMoERouter::entropy_threshold_high override (mode auto only).
        #[arg(long)]
        entropy_threshold_high: Option<f32>,
    },
    /// Launch polymorphic MCP service in stdio or SSE mode
    Serve {
        #[arg(long, default_value = "stdio", value_parser = ["stdio", "sse"])]
        mode: String,
        /// IP address to bind in SSE / HTTP mode.
        #[arg(long, default_value = "127.0.0.1")]
        host: IpAddr,
        #[arg(long, default_value_t = gen_zero_service::server::DEFAULT_SERVE_PORT)]
        port: u16,
        #[arg(long, env = "GENZERO_API_KEY")]
        token: Option<String>,
        /// Cognitive assets JSON (gen-zero/cognitive-assets/v1) to mount
        /// on the default key before serving.
        #[arg(long, env = "GENZERO_MOUNT_ASSETS")]
        mount_assets: Option<String>,
    },
    /// Alias for MCP server command
    Mcp {
        /// Run MCP over the process stdio stream.
        #[arg(long, conflicts_with = "sse")]
        stdio: bool,
        /// Run MCP over the SSE/HTTP server (the default when neither flag is set).
        #[arg(long, conflicts_with = "stdio")]
        sse: bool,
        #[arg(long, default_value_t = gen_zero_service::server::DEFAULT_MCP_SSE_PORT)]
        port: u16,
        #[arg(long, env = "GENZERO_API_KEY")]
        token: Option<String>,
        #[arg(long, env = "GENZERO_MOUNT_ASSETS")]
        mount_assets: Option<String>,
    },
    /// Generate a cryptographically secure Gen-Zero connection token
    Keygen {
        #[arg(long, default_value = "gz_live_")]
        prefix: String,
    },
    /// Evaluate single-step reflex decision
    Reflex {
        #[arg(short, long)]
        context: String,
        // No short flag: `-c` is `--context` (clap refuses the duplicate).
        #[arg(long, value_delimiter = ',')]
        candidates: Option<Vec<String>>,
        /// Numeric manifold request (JSON object) for the cognitive runtime.
        #[arg(long)]
        cognitive: Option<String>,
        #[arg(long, env = "GENZERO_MOUNT_ASSETS")]
        mount_assets: Option<String>,
    },
    /// Decide among candidates (verb `ask`/`decide`): semantic ask, semantic
    /// PUCT lookahead, or a latent planner, chosen by `--mode`
    Decide {
        #[arg(long)]
        context: String,
        /// Comma-separated candidate actions (default `proceed,wait`).
        #[arg(long, value_delimiter = ',')]
        candidates: Option<Vec<String>>,
        /// auto and reflex: semantic ask. mcts: semantic PUCT lookahead, or the
        /// planner crate's MCTS with --latent. mpc_cem and astar need --latent.
        #[arg(long, default_value = "auto", value_parser = ["auto", "mcts", "mpc_cem", "astar", "reflex"])]
        mode: String,
        #[arg(long, default_value = "generic", value_parser = ["generic", "nanocore"])]
        engine: String,
        #[arg(long, default_value = "linear", value_parser = ["linear", "etf"])]
        head: String,
        /// JSON core parameters or @file; required by nanocore.
        #[arg(long)]
        nanocore_core: Option<String>,
        /// JSON array of 128 floats or @file; required by nanocore.
        #[arg(long)]
        decision_state: Option<String>,
        /// JSON float array or @file; the decision state for ETF on the generic engine.
        #[arg(long)]
        etf_rep: Option<String>,
        /// JSON object candidate -> float array, or @file; required by ETF.
        #[arg(long)]
        candidate_reps: Option<String>,
        /// Softmax temperature over ETF cosine scores (default: uncalibrated 0.25).
        #[arg(long)]
        etf_temperature: Option<f32>,
        /// ETF cosine geometry: JSON object `{"kind": ...}` or @file (default: isotropic).
        #[arg(long)]
        etf_metric: Option<String>,
        /// Numeric manifold request (JSON object) for the cognitive runtime.
        #[arg(long)]
        cognitive: Option<String>,
        /// Latent state: JSON array of exactly 1024 numbers, or `@file`.
        /// Read only by the planner modes mcts, mpc_cem and astar.
        #[arg(long)]
        latent: Option<String>,
        /// Explicit A* goal JSON {state: [1024 finite numbers], tolerance: number}, or @file.
        #[arg(long)]
        astar_goal: Option<String>,
        /// Also roll the chosen action forward (needs --latent; a text state
        /// reports `trajectory_status: unsupported_without_latent_state`).
        #[arg(long)]
        return_trajectory: bool,
        /// Lookahead depth (mcts on text) or trajectory length (with --latent).
        #[arg(long)]
        horizon: Option<u64>,
        /// World model of the latent modes (needs --latent): `residual`
        /// (default), `symplectic` or `contact`.
        #[arg(long, value_parser = ["residual", "symplectic", "contact"])]
        dynamics: Option<String>,
        /// Conformal damping rate gamma >= 0 of `--dynamics contact`
        /// (default 0.5). Refused with any other dynamics.
        #[arg(long)]
        damping: Option<f64>,
        #[arg(long, env = "GENZERO_MOUNT_ASSETS")]
        mount_assets: Option<String>,
    },
    /// Roll a fixed action plan forward on the latent world model
    Simulate {
        /// Latent state: JSON array of exactly 1024 numbers, or `@file`.
        #[arg(long)]
        state: String,
        /// Action names: comma-separated, or a JSON array of strings.
        #[arg(long)]
        actions: String,
        /// Steps to simulate (default: number of actions; at most 256).
        #[arg(long)]
        horizon: Option<u64>,
        /// World model: `residual` (default), `symplectic` or `contact`.
        #[arg(long, value_parser = ["residual", "symplectic", "contact"])]
        dynamics: Option<String>,
        /// Conformal damping rate gamma >= 0 of `--dynamics contact`
        /// (default 0.5). Refused with any other dynamics.
        #[arg(long)]
        damping: Option<f64>,
    },
    /// Compare candidate first actions on the latent world model
    #[command(name = "what-if")]
    WhatIf {
        /// Latent state: JSON array of exactly 1024 numbers, or `@file`.
        #[arg(long)]
        state: String,
        /// Candidate names: comma-separated, or a JSON array of strings (1 to 16).
        #[arg(long)]
        candidates: String,
        /// Steps per rollout (default 5; at most 256).
        #[arg(long)]
        horizon: Option<u64>,
        /// World model: `residual` (default), `symplectic` or `contact`.
        #[arg(long, value_parser = ["residual", "symplectic", "contact"])]
        dynamics: Option<String>,
        /// Conformal damping rate gamma >= 0 of `--dynamics contact`
        /// (default 0.5). Refused with any other dynamics.
        #[arg(long)]
        damping: Option<f64>,
    },
    /// Shadow risk review of one planned action (never returns an approval)
    Audit {
        /// Latent state: JSON array of exactly 1024 numbers, or `@file`.
        #[arg(long)]
        state: String,
        /// The planned action to review.
        #[arg(long)]
        action: String,
        /// Steps to roll forward (default 5; at most 256).
        #[arg(long)]
        horizon: Option<u64>,
        /// Action names for the greedy continuation after the audited action:
        /// comma-separated, or a JSON array. Default: repeat the audited action.
        #[arg(long)]
        continuation: Option<String>,
        /// World model: `residual` (default), `symplectic` or `contact`.
        #[arg(long, value_parser = ["residual", "symplectic", "contact"])]
        dynamics: Option<String>,
        /// Conformal damping rate gamma >= 0 of `--dynamics contact`
        /// (default 0.5). Refused with any other dynamics.
        #[arg(long)]
        damping: Option<f64>,
    },
    /// Busemann entailment `passage ⊃ question` on the mounted preset geometry
    Entail {
        /// JSON array of `[H | R | S]` coordinates, as wide as the mounted
        /// preset: 64 (compact_64d), 128 (balanced_128d, boolq_128d) or 256
        /// (extended_256d). Another width exits 1 with `FiberMismatch`.
        #[arg(long)]
        passage: String,
        /// JSON array, same width and layout as `--passage`.
        #[arg(long)]
        question: String,
        /// Scheme 2: JSON array of `{time_ns, input}` tangent events at the
        /// passage point. Needs `--question-events` and `--window-start-ns`.
        #[arg(long)]
        passage_events: Option<String>,
        /// Scheme 2: JSON array of `{time_ns, input}` tangent events at the
        /// question point, paired index by index with `--passage-events`.
        #[arg(long)]
        question_events: Option<String>,
        /// Scheme 2: start of the event window in nanoseconds.
        #[arg(long)]
        window_start_ns: Option<u64>,
        /// Assets JSON whose `entailment` block seals the geometry.
        #[arg(long, env = "GENZERO_MOUNT_ASSETS")]
        mount_assets: Option<String>,
    },
    /// Causal relation trace fold on the discrete relation semiring (S1 Left, S3 Chart)
    Fold {
        /// JSON array of relation IDs, e.g. `[1, 2, 3]`
        #[arg(long, conflicts_with = "sets", required_unless_present = "sets")]
        edges: Option<String>,
        /// JSON array of node genders, e.g. `["Male", "Female", "Unknown"]`
        #[arg(long, required_unless_present = "gender")]
        genders: Option<String>,
        /// JSON array of nonempty relation sets, e.g. [[1], [2, 3]]
        #[arg(long, conflicts_with = "edges", required_unless_present = "edges")]
        sets: Option<String>,
        /// Uniform composition gender for sets: Male, Female or Unknown
        #[arg(long, required_unless_present = "genders")]
        gender: Option<String>,
        /// Path to custom axioms JSON mapping (optional, defaults to an empty table)
        #[arg(long)]
        axioms: Option<String>,
        /// Path to axiom weights JSON file (for weighted strategies)
        #[arg(long)]
        weights: Option<String>,
        /// Margin threshold for weighted strategies (e.g. 0.1)
        #[arg(long)]
        margin_threshold: Option<f64>,
        /// Strategy: weighted_tropical, weighted_logprob, "chart" (S3), "tiered" (default with weights), "left" (S1)
        #[arg(long)]
        strategy: Option<String>,
    },
    /// SQLite reflex feedback trace counts: unlabeled, unconsumed, trained
    #[command(name = "reflex-feedback-status")]
    ReflexFeedbackStatus {
        #[arg(long)]
        db: PathBuf,
        /// Must match the reflex operator's input width the traces were recorded with.
        #[arg(long)]
        input_dim: usize,
        /// Restrict counts to one task (default: every task in the database).
        #[arg(long)]
        task: Option<String>,
    },
    /// Attach a ground-truth label to a previously recorded reflex feedback trace
    #[command(name = "reflex-feedback-record")]
    ReflexFeedbackRecord {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        input_dim: usize,
        #[arg(long)]
        trace_id: String,
        /// Must be one of the head's candidate labels for a later adaptation
        /// cycle to accept it; recording does not itself check membership.
        #[arg(long)]
        label: String,
        #[arg(long, default_value = "human_review")]
        feedback_type: String,
        /// Unix epoch milliseconds (default: now).
        #[arg(long)]
        joined_at_unix_ms: Option<i64>,
    },
    /// Prune old reflex feedback traces
    #[command(name = "reflex-feedback-prune")]
    ReflexFeedbackPrune {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        input_dim: usize,
        /// Delete traces created before this Unix epoch millisecond cutoff.
        #[arg(long)]
        before_unix_ms: i64,
        /// Also delete labeled traces not yet consumed by training (default:
        /// keep them, since deleting would silently destroy untrained feedback).
        #[arg(long)]
        force: bool,
    },
    /// Compute a differential patch between two reflex plugin checkpoints
    #[command(name = "reflex-patch-create")]
    ReflexPatchCreate {
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        target: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        metadata: Option<String>,
    },
    /// Apply a differential patch to a base checkpoint, producing a target checkpoint
    #[command(name = "reflex-patch-apply")]
    ReflexPatchApply {
        #[arg(long)]
        base: PathBuf,
        #[arg(long)]
        patch: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Inspect a reflex patch's metadata, hashes, and delta statistics
    #[command(name = "reflex-patch-inspect")]
    ReflexPatchInspect {
        #[arg(long)]
        patch: PathBuf,
    },
    /// Run one online-adaptation cycle: fetch unconsumed SQLite feedback for
    /// a task/head, take a gradient step, verify the safety gate, hot-swap
    /// into a fresh in-process registry, and write the patched checkpoint
    #[command(name = "reflex-adapt")]
    ReflexAdapt {
        /// Base plugin checkpoint (written by a training run or a prior
        /// reflex-adapt / reflex-patch-apply).
        #[arg(long)]
        plugin: PathBuf,
        /// Where to write the patched checkpoint (only written if a cycle ran).
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        task: String,
        #[arg(long)]
        head: String,
        #[arg(long, default_value_t = 256)]
        limit: usize,
        #[arg(long, default_value_t = gen_zero_service::reflex_adapter::DEFAULT_LEARNING_RATE)]
        learning_rate: f32,
    },
    /// Micro-benchmark reflex registry predict latency, and live patch
    /// hot-swap latency under concurrent reader load, if --patch is given
    #[command(name = "reflex-bench")]
    ReflexBench {
        #[arg(long)]
        plugin: PathBuf,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        #[arg(long, default_value_t = 10_000)]
        iterations: usize,
        /// A patch to hot-swap in mid-benchmark, to measure swap latency
        /// under concurrent reader load (omit to measure predict latency only).
        #[arg(long)]
        patch: Option<PathBuf>,
    },
}

/// Publish an assets file as the next generation of the default mount. Any
/// failure stops the process: serving without the requested geometry would
/// be a silent downgrade.
fn mount_assets_file(server: &McpServer, path: &str) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let assets: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {path}"))?;
    let key = MountKey::new(DEFAULT_TENANT, DEFAULT_WORKSPACE);
    let base = server
        .engine
        .mounts()
        .load(&key)
        .map_err(|r| anyhow::anyhow!("load default mount: {r}"))?;
    let published = server
        .engine
        .publish_assets(&serde_json::json!({
            "base_version": base.version().0,
            "assets": assets,
            "reason": format!("startup --mount-assets {path}"),
        }))
        .map_err(|r| anyhow::anyhow!("mount {path}: {} ({})", r.code, r.detail))?;
    tracing::info!(
        "mounted cognitive assets from {path}: {}",
        published["published"]
    );
    Ok(())
}

/// A JSON argument, inline or `@path`.
fn json_arg(flag: &str, raw: &str) -> anyhow::Result<serde_json::Value> {
    let text = match raw.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("read {path}"))?,
        None => raw.to_string(),
    };
    serde_json::from_str(&text).with_context(|| format!("--{flag} must be valid JSON"))
}

/// Action names: a JSON array of strings, or a comma-separated list. An empty
/// piece is an error, never dropped.
fn name_list_arg(flag: &str, raw: &str) -> anyhow::Result<serde_json::Value> {
    if raw.trim_start().starts_with('[') {
        return json_arg(flag, raw);
    }
    let names: Vec<&str> = raw.split(',').map(str::trim).collect();
    anyhow::ensure!(
        names.iter().all(|n| !n.is_empty()),
        "--{flag} has an empty name"
    );
    Ok(serde_json::json!(names))
}

/// Print an outcome; a refused or gated one is a failed command (exit 1).
fn finish(label: &str, outcome: &gen_zero_service::ZeroToolOutcome) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(outcome)?);
    if outcome.is_error {
        let code = outcome
            .rejection
            .as_ref()
            .map(|r| r.code.as_str())
            .or_else(|| outcome.meta["tier"].as_str())
            .unwrap_or("error");
        let detail = outcome
            .content
            .first()
            .map_or("", |block| block.text.as_str());
        eprintln!("gen-zero {label}: refused ({code}): {detail}");
        std::process::exit(1);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();

    match cli.command {
        Commands::AuditLedger {
            request,
            max_simulations,
            c_puct,
            num_samples,
            cem_horizon,
            uncertainty_weight,
            entropy_threshold_low,
            entropy_threshold_high,
        } => {
            let mut block = json_arg("request", &request)?;
            if block.get("op").and_then(|v| v.as_str()) != Some("decide") {
                anyhow::bail!("audit-ledger requires a pipeline decide request");
            }
            let mut planner_config = serde_json::Map::new();
            if let Some(v) = max_simulations {
                planner_config.insert("mcts_max_simulations".into(), v.into());
            }
            if let Some(v) = c_puct {
                planner_config.insert("mcts_c_puct".into(), v.into());
            }
            if let Some(v) = num_samples {
                planner_config.insert("cem_num_samples".into(), v.into());
            }
            if let Some(v) = cem_horizon {
                planner_config.insert("cem_horizon".into(), v.into());
            }
            if let Some(v) = uncertainty_weight {
                planner_config.insert("astar_uncertainty_penalty_weight".into(), v.into());
            }
            if let Some(v) = entropy_threshold_low {
                planner_config.insert("router_entropy_threshold_low".into(), v.into());
            }
            if let Some(v) = entropy_threshold_high {
                planner_config.insert("router_entropy_threshold_high".into(), v.into());
            }
            if !planner_config.is_empty() {
                let obj = block
                    .as_object_mut()
                    .context("--request must be a JSON object")?;
                anyhow::ensure!(
                    !obj.contains_key("planner_config"),
                    "--request already sets planner_config; use --request or the \
                     --c-puct-style flags, not both"
                );
                obj.insert(
                    "planner_config".into(),
                    serde_json::Value::Object(planner_config),
                );
            }
            let server = McpServer::new();
            let outcome = server
                .engine
                .execute(&serde_json::json!({"action": "pipeline", "pipeline": block}))
                .await?;
            finish("audit-ledger", &outcome)?;
            let (count, root) = server.engine.audit_ledger();
            println!(
                "ledger leaves: {count}, root: {}",
                root.iter().map(|b| format!("{b:02x}")).collect::<String>()
            );
        }
        Commands::Serve {
            mode,
            host,
            port,
            token,
            mount_assets,
        } => {
            let server = McpServer::new().with_auth_token(token.clone());
            if let Some(path) = &mount_assets {
                mount_assets_file(&server, path)?;
            }
            if mode == "stdio" {
                server.run_stdio().await?;
            } else {
                let addr = SocketAddr::new(host, port);
                if token.is_some() {
                    println!(
                        "🔒 Gen-Zero SSE service running with Token Authorization on {}",
                        addr
                    );
                } else {
                    println!(
                        "⚠️ Gen-Zero SSE service running in OPEN mode (no token set) on {}",
                        addr
                    );
                }
                server.run_sse(addr).await?;
            }
        }
        Commands::Mcp {
            stdio,
            sse: _,
            port,
            token,
            mount_assets,
        } => {
            let server = McpServer::new().with_auth_token(token.clone());
            if let Some(path) = &mount_assets {
                mount_assets_file(&server, path)?;
            }
            if stdio {
                server.run_stdio().await?;
            } else {
                let addr: SocketAddr = format!("0.0.0.0:{}", port).parse()?;
                if token.is_some() {
                    println!(
                        "🔒 Gen-Zero MCP SSE service running with Token Authorization on {}",
                        addr
                    );
                } else {
                    println!(
                        "⚠️ Gen-Zero MCP SSE service running in OPEN mode (no token set) on {}",
                        addr
                    );
                }
                server.run_sse(addr).await?;
            }
        }
        Commands::Keygen { prefix } => {
            let mut bytes = [0u8; 32];
            rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut bytes);
            let hex_str: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
            let token = format!("{}{}", prefix, hex_str);
            println!("============================================================");
            println!("🔑 Gen-Zero Cryptographic API Token Generated");
            println!("============================================================");
            println!("Token: {}", token);
            println!();
            println!("Quick-Start Configuration:");
            println!("  1. Local SSE / HTTP Server:");
            println!(
                "     GENZERO_API_KEY={} gen-zero serve --mode sse --port 8999",
                token
            );
            println!();
            println!("  2. Production Environment Variable:");
            println!("     export GENZERO_API_KEY={}", token);
            println!();
            println!("  3. Client MCP (Header Authorization):");
            println!(
                "     \"headers\": {{ \"Authorization\": \"Bearer {}\" }}",
                token
            );
            println!();
            println!("  4. Client SSE (Query Parameter Authorization):");
            println!(
                "     \"url\": \"http://127.0.0.1:8999/sse?token={}\"",
                token
            );
            println!("============================================================");
        }
        Commands::Reflex {
            context,
            candidates,
            cognitive,
            mount_assets,
        } => {
            let server = McpServer::new();
            if let Some(path) = &mount_assets {
                mount_assets_file(&server, path)?;
            }
            let cands =
                candidates.unwrap_or_else(|| vec!["proceed".to_string(), "wait".to_string()]);
            let mut request = serde_json::json!({
                "action": "ask",
                "context": context,
                "candidates": cands
            });
            if let Some(block) = cognitive {
                request["cognitive"] =
                    serde_json::from_str(&block).context("--cognitive must be a JSON object")?;
            }
            let outcome = server.engine.execute(&request).await?;
            println!("{}", serde_json::to_string_pretty(&outcome)?);
            if outcome.is_error {
                // A refused or gated decision is a failed command, not a
                // success with an error inside the JSON.
                let code = outcome
                    .rejection
                    .as_ref()
                    .map(|r| r.code.as_str())
                    .or_else(|| outcome.meta["tier"].as_str())
                    .unwrap_or("error");
                eprintln!("gen-zero reflex: refused ({code})");
                std::process::exit(1);
            }
        }
        Commands::Decide {
            context,
            candidates,
            mode,
            engine,
            head,
            nanocore_core,
            decision_state,
            etf_rep,
            candidate_reps,
            etf_temperature,
            etf_metric,
            cognitive,
            latent,
            astar_goal,
            return_trajectory,
            horizon,
            dynamics,
            damping,
            mount_assets,
        } => {
            let server = McpServer::new();
            if let Some(path) = &mount_assets {
                mount_assets_file(&server, path)?;
            }
            let cands =
                candidates.unwrap_or_else(|| vec!["proceed".to_string(), "wait".to_string()]);
            let mut request = serde_json::json!({
                "action": "decide",
                "context": context,
                "candidates": cands,
                "mode": mode,
                "engine": engine,
                "head": head,
                "return_trajectory": return_trajectory,
            });
            if let Some(raw) = &nanocore_core {
                request["nanocore_core"] = json_arg("nanocore_core", raw)?;
            }
            if let Some(raw) = &decision_state {
                request["decision_state"] = json_arg("decision_state", raw)?;
            }
            if let Some(raw) = &etf_rep {
                request["etf_rep"] = json_arg("etf_rep", raw)?;
            }
            if let Some(raw) = &candidate_reps {
                request["candidate_reps"] = json_arg("candidate_reps", raw)?;
            }
            if let Some(t) = etf_temperature {
                request["etf_temperature"] = t.into();
            }
            if let Some(raw) = &etf_metric {
                request["etf_metric"] = json_arg("etf_metric", raw)?;
            }
            if let Some(block) = cognitive {
                request["cognitive"] =
                    serde_json::from_str(&block).context("--cognitive must be a JSON object")?;
            }
            if let Some(raw) = &latent {
                request["latent"] = json_arg("latent", raw)?;
            }
            if let Some(raw) = &astar_goal {
                request["astar_goal"] = json_arg("astar_goal", raw)?;
            }
            if let Some(h) = horizon {
                request["horizon"] = h.into();
            }
            if let Some(d) = dynamics {
                request["dynamics"] = d.into();
            }
            if let Some(g) = damping {
                request["damping"] = g.into();
            }
            let outcome = server.engine.execute(&request).await?;
            finish("decide", &outcome)?;
        }
        Commands::Simulate {
            state,
            actions,
            horizon,
            dynamics,
            damping,
        } => {
            let server = McpServer::new();
            let mut request = serde_json::json!({
                "action": "simulate",
                "state": json_arg("state", &state)?,
                "actions": name_list_arg("actions", &actions)?,
            });
            if let Some(h) = horizon {
                request["horizon"] = h.into();
            }
            if let Some(d) = dynamics {
                request["dynamics"] = d.into();
            }
            if let Some(g) = damping {
                request["damping"] = g.into();
            }
            let outcome = server.engine.execute(&request).await?;
            finish("simulate", &outcome)?;
        }
        Commands::WhatIf {
            state,
            candidates,
            horizon,
            dynamics,
            damping,
        } => {
            let server = McpServer::new();
            let mut request = serde_json::json!({
                "action": "what_if",
                "state": json_arg("state", &state)?,
                "candidates": name_list_arg("candidates", &candidates)?,
            });
            if let Some(h) = horizon {
                request["horizon"] = h.into();
            }
            if let Some(d) = dynamics {
                request["dynamics"] = d.into();
            }
            if let Some(g) = damping {
                request["damping"] = g.into();
            }
            let outcome = server.engine.execute(&request).await?;
            finish("what-if", &outcome)?;
        }
        Commands::Audit {
            state,
            action,
            horizon,
            continuation,
            dynamics,
            damping,
        } => {
            let server = McpServer::new();
            // The JSON key `action` selects the verb, so the audited action
            // travels as `target_action`.
            let mut request = serde_json::json!({
                "action": "audit",
                "state": json_arg("state", &state)?,
                "target_action": action,
            });
            if let Some(h) = horizon {
                request["horizon"] = h.into();
            }
            if let Some(d) = dynamics {
                request["dynamics"] = d.into();
            }
            if let Some(g) = damping {
                request["damping"] = g.into();
            }
            if let Some(raw) = &continuation {
                request["continuation_actions"] = name_list_arg("continuation", raw)?;
            }
            let outcome = server.engine.execute(&request).await?;
            finish("audit", &outcome)?;
        }
        Commands::Entail {
            passage,
            question,
            passage_events,
            question_events,
            window_start_ns,
            mount_assets,
        } => {
            let server = McpServer::new();
            if let Some(path) = &mount_assets {
                mount_assets_file(&server, path)?;
            }
            let mut request = serde_json::json!({
                "action": "entail",
                "entailment": {
                    "passage": serde_json::from_str::<serde_json::Value>(&passage)
                        .context("--passage must be a JSON array")?,
                    "question": serde_json::from_str::<serde_json::Value>(&question)
                        .context("--question must be a JSON array")?,
                },
            });
            // Forwarded as given: the runtime refuses a partial set, so the
            // CLI never silently drops one of them.
            if let Some(ev) = &passage_events {
                request["entailment"]["passage_events"] =
                    serde_json::from_str(ev).context("--passage-events must be a JSON array")?;
            }
            if let Some(ev) = &question_events {
                request["entailment"]["question_events"] =
                    serde_json::from_str(ev).context("--question-events must be a JSON array")?;
            }
            if let Some(t) = window_start_ns {
                request["entailment"]["window_start_ns"] = t.into();
            }
            let outcome = server.engine.execute(&request).await?;
            println!("{}", serde_json::to_string_pretty(&outcome)?);
            if outcome.is_error {
                let code = outcome
                    .rejection
                    .as_ref()
                    .map_or("error", |r| r.code.as_str());
                eprintln!("gen-zero entail: refused ({code})");
                std::process::exit(1);
            }
        }
        Commands::Fold {
            sets,
            gender,
            edges,
            genders,
            axioms,
            weights,
            margin_threshold,
            strategy,
        } => {
            let server = McpServer::new();
            let mut causal_fold = serde_json::json!({});
            if let Some(strategy) = strategy {
                causal_fold["strategy"] = serde_json::json!(strategy);
            }
            for (name, value) in [("edges", edges), ("genders", genders), ("sets", sets)] {
                if let Some(value) = value {
                    causal_fold[name] = serde_json::from_str::<serde_json::Value>(&value)
                        .with_context(|| format!("--{name} must be a JSON array"))?;
                }
            }
            if let Some(gender) = gender {
                causal_fold["gender"] = serde_json::json!(gender);
            }
            if let Some(path) = &axioms {
                let text = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
                causal_fold["axioms"] =
                    serde_json::from_str(&text).with_context(|| format!("parse {path}"))?;
            }
            if let Some(path) = weights {
                let text =
                    std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
                causal_fold["weights"] =
                    serde_json::from_str(&text).with_context(|| format!("parse {path}"))?;
            }
            if let Some(threshold) = margin_threshold {
                anyhow::ensure!(
                    threshold.is_finite() && threshold >= 0.0,
                    "--margin-threshold must be finite and >= 0"
                );
                causal_fold["margin_threshold"] = serde_json::json!(threshold);
            }
            let request = serde_json::json!({
                "action": "causal_fold",
                "causal_fold": causal_fold,
            });
            let outcome = server.engine.execute(&request).await?;
            println!("{}", serde_json::to_string_pretty(&outcome)?);
            if outcome.is_error {
                let code = outcome
                    .rejection
                    .as_ref()
                    .map_or("error", |r| r.code.as_str());
                let detail = outcome.rejection.as_ref().map_or("", |r| r.detail.as_str());
                eprintln!("gen-zero fold: refused ({code}): {detail}");
                std::process::exit(1);
            }
        }
        Commands::ReflexFeedbackStatus {
            db,
            input_dim,
            task,
        } => {
            reflex_cmd::feedback_status(&db, input_dim, task.as_deref())?;
        }
        Commands::ReflexFeedbackRecord {
            db,
            input_dim,
            trace_id,
            label,
            feedback_type,
            joined_at_unix_ms,
        } => {
            reflex_cmd::feedback_record(
                &db,
                input_dim,
                &trace_id,
                &label,
                &feedback_type,
                joined_at_unix_ms,
            )?;
        }
        Commands::ReflexFeedbackPrune {
            db,
            input_dim,
            before_unix_ms,
            force,
        } => {
            reflex_cmd::feedback_prune(&db, input_dim, before_unix_ms, force)?;
        }
        Commands::ReflexPatchCreate {
            base,
            target,
            out,
            metadata,
        } => {
            reflex_cmd::patch_create(&base, &target, &out, metadata)?;
        }
        Commands::ReflexPatchApply { base, patch, out } => {
            reflex_cmd::patch_apply(&base, &patch, &out)?;
        }
        Commands::ReflexPatchInspect { patch } => {
            reflex_cmd::patch_inspect(&patch)?;
        }
        Commands::ReflexAdapt {
            plugin,
            out,
            db,
            task,
            head,
            limit,
            learning_rate,
        } => {
            reflex_cmd::adapt(&plugin, &out, &db, &task, &head, limit, learning_rate)?;
        }
        Commands::ReflexBench {
            plugin,
            threads,
            iterations,
            patch,
        } => {
            reflex_cmd::bench(&plugin, threads, iterations, patch.as_deref())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod cli_parse_tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn mcp_rejects_conflicting_transport_flags() {
        let error = Cli::try_parse_from(["gen-zero", "mcp", "--stdio", "--sse"])
            .expect_err("stdio and sse must not be accepted together");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }
}
