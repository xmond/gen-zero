//! Text-to-Graph (RFC-20261002 Phase 1): a short imperative task text becomes
//! a causal action DAG, behind an answerability gate. Served as the
//! `graph_induce` verb ([`crate::graph_verb`]).
//!
//! What this is, plainly: a deterministic lexical rule parser
//! ([`INDUCE_ENGINE`]). It is not the RFC's bidirectional encoder
//! (MiniLM / DeBERTa); no such model or trained answerability head exists in
//! this repository, and nothing here was trained or calibrated. The
//! answerability `confidence` is a product of three heuristic factors (below),
//! not a probability, and carries no AUROC claim.
//!
//! # Grammar
//!
//! The text is split into clauses. Every clause must begin (after fillers such
//! as `first`, `please`, `then`) with a verb from a fixed lexicon
//! ([`ACTION_VERBS`]); the action's name is the clause,
//! lowercased, with the verb lemmatized.
//!
//! - Sequence is the default: `then`, `;`, `.`, `->`, `=>`, and `,` or `and`
//!   followed by a verb start a new step that depends on every action of the
//!   step before. So `fetch data then clean it and save to db` is the chain
//!   fetch -> clean -> save.
//! - Parallel needs an explicit marker ([`PARALLEL_MARKERS`]: `in parallel`,
//!   `at the same time`, `meanwhile`, ...), before the clause or at
//!   its end. A parallel clause joins the current step: same parents.
//! - `A after B` and `before B, A` run B first. A chain of two such
//!   inversions is ambiguous and refused, never guessed.
//!
//! There is no coreference: `it` stays `it`. Clauses are never merged or
//! deduplicated; a repeated action name is refused.
//!
//! # Answerability gate (fail-closed)
//!
//! Before any topology is built the text, and the optional `context`, pass
//! the gate. A refusal is `answerable: false` with a `refusal_reason`, no DAG
//! and no actions; nothing is deposited.
//!
//! - Hard refusals (confidence 0): blank text; a phrase from the fixed
//!   prompt-injection blocklist ([`INJECTION_MARKERS`], a blocklist, not a
//!   classifier) in the text or the context; no clause; no clause that
//!   opens with a lexicon verb (whatever the threshold); more than
//!   [`MAX_INDUCED_ACTIONS`] clauses; a clause name over
//!   [`MAX_ACTION_NAME_BYTES`]; a repeated action; an ambiguous inversion
//!   chain; a last step with more than one action (the DAG has one target).
//! - Otherwise `confidence = char_validity * word_shape * verb_coverage`:
//!   the share of non-space characters that are letters, digits or plain
//!   punctuation; the share of Latin words that are word-shaped (a vowel, no
//!   consonant run over four); the share of clauses that begin with a lexicon
//!   verb. Below the threshold (default
//!   [`DEFAULT_ANSWERABILITY_THRESHOLD`], an uncalibrated preset) the text is
//!   refused. Destructive verbs (`delete`, `drop`) are not refused here: the
//!   PolicyGate decides those.
//!
//! # Output
//!
//! Each action's id is `zero::action_id` of its name, the key the
//! PolicyGate checks, so a pruned induced action is hard-stopped like any
//! other. The DAG is a [`CausalDagSpec`]: all AND rules, a nominal cost of 1
//! per action (not estimated from the text), value 1 on the target and 0
//! elsewhere, budget = the number of actions. It is validated by
//! [`CausalDag::from_spec`] before it is returned, and `pipeline decide`
//! (`causal_triad` mode) takes it as `causal_dag` unchanged.

use crate::cognitive::Rejection;
use crate::zero::action_id;
use gen_zero_core::ActionId;
use gen_zero_planner::triad::{CausalDag, CausalDagSpec};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

/// Engine name echoed in every outcome. Rule-based, not a model.
pub const INDUCE_ENGINE: &str = "lexical_rule_parser_v1";
/// Uncalibrated preset; echoed in every outcome.
pub const DEFAULT_ANSWERABILITY_THRESHOLD: f32 = 0.8;
/// Most bytes of `text`, and of `context`. Larger is a malformed request (413).
pub const MAX_INDUCE_TEXT_BYTES: usize = 4096;
/// Most actions one text may yield: the `pipeline decide` candidate cap
/// ([`crate::worldsim::MAX_CANDIDATES`]), so every induced DAG can be decided.
pub const MAX_INDUCED_ACTIONS: usize = crate::worldsim::MAX_CANDIDATES;
/// Most bytes of one action name; a graph label holds 256.
pub const MAX_ACTION_NAME_BYTES: usize = 200;

const STAGE: &str = "text_to_graph";

/// Imperative verbs that open an English action clause (lemma form).
pub const ACTION_VERBS: &[&str] = &[
    "add",
    "aggregate",
    "analyze",
    "apply",
    "approve",
    "archive",
    "assign",
    "audit",
    "back",
    "backup",
    "build",
    "calculate",
    "call",
    "cancel",
    "check",
    "clean",
    "cleanse",
    "clear",
    "clone",
    "close",
    "collect",
    "commit",
    "compare",
    "compile",
    "compress",
    "compute",
    "configure",
    "connect",
    "convert",
    "copy",
    "create",
    "crawl",
    "debug",
    "decode",
    "decrypt",
    "dedupe",
    "deduplicate",
    "delete",
    "deploy",
    "detect",
    "disable",
    "download",
    "drop",
    "email",
    "embed",
    "enable",
    "encode",
    "encrypt",
    "enrich",
    "evaluate",
    "execute",
    "export",
    "extract",
    "fetch",
    "filter",
    "find",
    "fix",
    "format",
    "generate",
    "get",
    "grant",
    "group",
    "hash",
    "import",
    "index",
    "ingest",
    "initialize",
    "insert",
    "install",
    "join",
    "label",
    "launch",
    "lint",
    "load",
    "log",
    "login",
    "measure",
    "merge",
    "migrate",
    "monitor",
    "mount",
    "move",
    "normalize",
    "notify",
    "open",
    "optimize",
    "pack",
    "package",
    "parse",
    "patch",
    "ping",
    "plot",
    "poll",
    "post",
    "predict",
    "prepare",
    "preprocess",
    "print",
    "process",
    "provision",
    "publish",
    "pull",
    "push",
    "query",
    "read",
    "rebuild",
    "receive",
    "record",
    "reboot",
    "refresh",
    "register",
    "reindex",
    "reload",
    "remove",
    "rename",
    "render",
    "replace",
    "replicate",
    "report",
    "request",
    "reset",
    "resize",
    "restart",
    "restore",
    "retrain",
    "retrieve",
    "review",
    "revoke",
    "rollback",
    "rotate",
    "run",
    "sanitize",
    "save",
    "scale",
    "scan",
    "schedule",
    "score",
    "scrape",
    "search",
    "send",
    "serialize",
    "set",
    "setup",
    "sign",
    "snapshot",
    "sort",
    "split",
    "start",
    "stop",
    "store",
    "submit",
    "summarize",
    "sync",
    "tag",
    "test",
    "tokenize",
    "train",
    "transcode",
    "transfer",
    "transform",
    "translate",
    "trigger",
    "truncate",
    "tune",
    "unpack",
    "unzip",
    "update",
    "upgrade",
    "upload",
    "validate",
    "verify",
    "visualize",
    "wait",
    "write",
    "zip",
];

/// Phrases that mark a step as parallel to the one before (tokens, lowercase).
pub const PARALLEL_MARKERS: &[&[&str]] = &[
    &["in", "parallel"],
    &["at", "the", "same", "time"],
    &["simultaneously"],
    &["concurrently"],
    &["meanwhile"],
];

/// Fixed prompt-injection blocklist, matched as substrings of the lowercased,
/// whitespace-collapsed text. A blocklist, not a classifier: a paraphrase it
/// does not list passes this check.
pub const INJECTION_MARKERS: &[&str] = &[
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore prior instructions",
    "ignore all prior instructions",
    "ignore the above",
    "ignore your instructions",
    "disregard previous instructions",
    "disregard all previous instructions",
    "disregard the above",
    "forget your instructions",
    "forget all previous instructions",
    "reveal your system prompt",
    "print your system prompt",
    "system prompt:",
    "you are now dan",
    "developer mode enabled",
    "jailbreak",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "[inst]",
    "<<sys>>",
    "\u{5ffd}\u{7565}\u{4e4b}\u{524d}\u{7684}\u{6307}\u{4ee4}", // ignore previous instructions
    "\u{5ffd}\u{7565}\u{4ee5}\u{4e0a}",                         // ignore the above
    "\u{5ffd}\u{7565}\u{6240}\u{6709}\u{6307}\u{4ee4}",         // ignore all instructions
    "\u{5ffd}\u{7565}\u{4e0a}\u{9762}\u{7684}\u{6307}\u{4ee4}", // ignore the instructions above
    "\u{6cc4}\u{9732}\u{7cfb}\u{7edf}\u{63d0}\u{793a}",         // leak the system prompt
    "\u{8d8a}\u{72f1}",                                         // jailbreak
];

/// Words skipped before a clause's verb and stripped from its name.
const FILLERS: &[&str] = &[
    "first",
    "firstly",
    "then",
    "next",
    "finally",
    "lastly",
    "afterwards",
    "please",
    "also",
    "now",
    "just",
    "and",
    "to",
    "so",
];

/// One `graph_induce` request. Unknown fields are refused.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InduceRequest {
    /// The task text.
    pub text: String,
    /// Untrusted side text: passes the injection check and is stored with every
    /// deposited action's payload. It does not change the parse.
    #[serde(default)]
    pub context: Option<String>,
    /// In `[0, 1]`; default [`DEFAULT_ANSWERABILITY_THRESHOLD`].
    #[serde(default)]
    pub answerability_threshold: Option<f32>,
    /// Deposit the actions and their `depends_on` edges into the live graph.
    #[serde(default)]
    pub auto_deposit: Option<bool>,
}

/// One induced action.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InductedAction {
    /// Explicit audit-only action binding; no external command is inferred.
    pub operator: gen_zero_lod::OperatorSignature,
    /// `zero::action_id` of `name`.
    pub action_id: u32,
    /// Normalized clause: lowercase, verb lemmatized, fillers stripped.
    pub name: String,
    /// The lexicon verb that opened the clause, or `None` when none did.
    pub verb: Option<String>,
    /// Action ids this action needs done first (all of them: AND).
    pub parents: Vec<u32>,
    /// Nominal cost, always 1: not estimated from the text.
    pub cost: f32,
    /// The single goal action of the DAG.
    pub target: bool,
}

/// The gate's factors, echoed whether or not the text passed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnswerabilityReport {
    pub threshold: f32,
    pub char_validity: f32,
    pub word_shape: f32,
    pub verb_coverage: f32,
    pub clauses: usize,
    pub recognized_clauses: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InduceOutcome {
    pub answerable: bool,
    /// Heuristic score in `[0, 1]`, not a calibrated probability.
    pub confidence: f32,
    pub dag_spec: Option<CausalDagSpec>,
    pub actions: Vec<InductedAction>,
    /// Graph node ids of the deposited actions; `None` unless deposited.
    pub deposited_node_ids: Option<Vec<u64>>,
    pub engine: String,
    pub answerability: AnswerabilityReport,
    /// Why the gate refused; `None` when answerable.
    pub refusal_reason: Option<String>,
    /// Wall time of the induction alone (no deposit), microseconds.
    pub elapsed_us: u64,
}

/// How a clause attaches to the clause before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Link {
    /// A new step after the previous one.
    Seq,
    /// Same step as the previous clause.
    Par,
    /// The previous clause runs after this one (`A after B`).
    After,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lead {
    After,
    Before,
}

#[derive(Debug)]
struct RawClause {
    tokens: Vec<String>,
    link: Link,
    lead: Option<Lead>,
}

/// A gate refusal: `(confidence, reason)`.
type Refusal = (f32, String);

#[derive(Clone, Debug)]
struct Clause {
    name: String,
    verb: Option<String>,
}

/// Stateless; one instance serves every request.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextToGraphInducer;

impl TextToGraphInducer {
    pub fn new() -> Self {
        Self
    }

    /// Run the gate and, when it passes, induce the DAG. `Err` only for a
    /// malformed request (threshold out of range, text or context too long);
    /// a refusal by the gate is an `Ok` outcome with `answerable: false`.
    pub fn induce(&self, req: &InduceRequest) -> Result<InduceOutcome, Rejection> {
        let started = Instant::now();
        let threshold = req
            .answerability_threshold
            .unwrap_or(DEFAULT_ANSWERABILITY_THRESHOLD);
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(Rejection::invalid(
                STAGE,
                format!("answerability_threshold must be a number in [0, 1], got {threshold}"),
            ));
        }
        for (field, len) in [
            ("text", req.text.len()),
            ("context", req.context.as_ref().map_or(0, String::len)),
        ] {
            if len > MAX_INDUCE_TEXT_BYTES {
                return Err(Rejection {
                    code: "PayloadTooLarge".into(),
                    stage: STAGE.into(),
                    detail: format!("{field} has {len} bytes; at most {MAX_INDUCE_TEXT_BYTES}"),
                    http_status: 413,
                });
            }
        }
        let mut report = AnswerabilityReport {
            threshold,
            char_validity: 0.0,
            word_shape: 0.0,
            verb_coverage: 0.0,
            clauses: 0,
            recognized_clauses: 0,
        };
        let outcome = match self.gate_and_parse(req, &mut report) {
            Ok(actions) => {
                let spec = dag_spec(&actions)?;
                let confidence = report.char_validity * report.word_shape * report.verb_coverage;
                InduceOutcome {
                    answerable: true,
                    confidence,
                    dag_spec: Some(spec),
                    actions,
                    deposited_node_ids: None,
                    engine: INDUCE_ENGINE.into(),
                    answerability: report,
                    refusal_reason: None,
                    elapsed_us: 0,
                }
            }
            Err((confidence, reason)) => {
                tracing::warn!(engine = INDUCE_ENGINE, %reason, "graph_induce: text refused");
                InduceOutcome {
                    answerable: false,
                    confidence,
                    dag_spec: None,
                    actions: Vec::new(),
                    deposited_node_ids: None,
                    engine: INDUCE_ENGINE.into(),
                    answerability: report,
                    refusal_reason: Some(reason),
                    elapsed_us: 0,
                }
            }
        };
        Ok(InduceOutcome {
            elapsed_us: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
            ..outcome
        })
    }

    /// The actions in step order, or the refusal `(confidence, reason)`.
    fn gate_and_parse(
        &self,
        req: &InduceRequest,
        report: &mut AnswerabilityReport,
    ) -> Result<Vec<InductedAction>, Refusal> {
        let text = req.text.trim();
        if text.is_empty() {
            return Err((0.0, "empty_text: the text is blank".into()));
        }
        for (field, value) in [("text", Some(text)), ("context", req.context.as_deref())] {
            if let Some(marker) = value.and_then(injection_marker) {
                return Err((
                    0.0,
                    format!("injection_marker: {field} contains the blocklisted phrase {marker:?}"),
                ));
            }
        }
        report.char_validity = char_validity(text);
        let tokens = tokenize(text);
        report.word_shape = word_shape(&tokens);

        let raw = split_clauses(&tokens);
        report.clauses = raw.len();
        if raw.is_empty() {
            return Err((0.0, "no_clause: the text holds no clause".into()));
        }
        if raw.len() > MAX_INDUCED_ACTIONS {
            return Err((
                0.0,
                format!(
                    "too_many_actions: {} clauses, at most {MAX_INDUCED_ACTIONS}",
                    raw.len()
                ),
            ));
        }
        let ordered = order_clauses(raw)?;
        let clauses: Vec<(Clause, Link)> = ordered
            .into_iter()
            .map(|(tokens, link)| (clause(&tokens), link))
            .collect();
        report.recognized_clauses = clauses.iter().filter(|(c, _)| c.verb.is_some()).count();
        report.verb_coverage = report.recognized_clauses as f32 / clauses.len() as f32;
        // No action identified at all: refused even at threshold 0. The
        // threshold only decides texts where some clauses were recognized.
        if report.recognized_clauses == 0 {
            return Err((
                0.0,
                "no_action_verb: no clause opens with a known action verb".into(),
            ));
        }
        let confidence = report.char_validity * report.word_shape * report.verb_coverage;
        if confidence < report.threshold {
            let unrecognized: Vec<&str> = clauses
                .iter()
                .filter(|(c, _)| c.verb.is_none())
                .map(|(c, _)| c.name.as_str())
                .collect();
            return Err((
                confidence,
                format!(
                    "below_threshold: confidence {confidence:.3} < {:.3} (char_validity {:.3}, \
                     word_shape {:.3}, verb_coverage {:.3}; clauses without a known action \
                     verb: {unrecognized:?})",
                    report.threshold, report.char_validity, report.word_shape, report.verb_coverage
                ),
            ));
        }
        build_actions(clauses).map_err(|reason| (0.0, reason))
    }
}

fn injection_marker(text: &str) -> Option<&'static str> {
    let folded = text
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    INJECTION_MARKERS
        .iter()
        .copied()
        .find(|m| folded.contains(m))
}

fn is_plain_punct(c: char) -> bool {
    matches!(
        c,
        ',' | '.' | ';' | ':' | '!' | '?' | '\'' | '"' | '-' | '_' | '/' | '(' | ')' | '>' | '='
    ) || "，。；：！？、（）“”‘’".contains(c)
}

/// Share of non-space characters that are letters, digits or plain punctuation.
fn char_validity(text: &str) -> f32 {
    let (mut valid, mut total) = (0_usize, 0_usize);
    for c in text.chars().filter(|c| !c.is_whitespace()) {
        total += 1;
        if c.is_alphanumeric() || is_plain_punct(c) {
            valid += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        valid as f32 / total as f32
    }
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

/// A purely Latin-letter word is word-shaped when it is short (an
/// abbreviation such as `db`), or has a vowel and no consonant run over four.
/// Tokens with digits, non-Latin text and punctuation are not judged.
fn word_shaped(word: &str) -> Option<bool> {
    if word.is_empty() || !word.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    if word.len() <= 3 {
        return Some(true);
    }
    if word.len() > 24 || !word.chars().any(is_vowel) {
        return Some(false);
    }
    let mut run = 0;
    for c in word.chars() {
        run = if is_vowel(c) { 0 } else { run + 1 };
        if run > 4 {
            return Some(false);
        }
    }
    Some(true)
}

fn word_shape(tokens: &[String]) -> f32 {
    let judged: Vec<bool> = tokens.iter().filter_map(|t| word_shaped(t)).collect();
    if judged.is_empty() {
        // Nothing Latin to judge (e.g. Chinese only): neutral.
        return 1.0;
    }
    judged.iter().filter(|&&ok| ok).count() as f32 / judged.len() as f32
}

const SEPARATOR_PUNCT: &[char] = &[',', ';', '.', '!', '?', ':'];

/// Lowercased tokens. `->` and `=>` always stand alone; `, ; . ! ? :` stand
/// alone when they end a word (so `data.csv` stays one token).
fn tokenize(text: &str) -> Vec<String> {
    let text = text
        .to_lowercase()
        .replace("->", " -> ")
        .replace("=>", " => ");
    let mut out = Vec::new();
    for word in text.split_whitespace() {
        let mut rest = word;
        let mut tail = Vec::new();
        while let Some(c) = rest.chars().last().filter(|c| SEPARATOR_PUNCT.contains(c)) {
            tail.push(c.to_string());
            rest = &rest[..rest.len() - c.len_utf8()];
        }
        if !rest.is_empty() {
            out.push(rest.to_string());
        }
        out.extend(tail.into_iter().rev());
    }
    out
}

fn is_hard_separator(t: &str) -> bool {
    matches!(t, ";" | "." | "!" | "?" | "->" | "=>" | "then")
}

fn parallel_marker_at(tokens: &[String], i: usize) -> Option<usize> {
    PARALLEL_MARKERS
        .iter()
        .find(|m| tokens.len() >= i + m.len() && m.iter().zip(&tokens[i..]).all(|(a, b)| *a == b))
        .map(|m| m.len())
}

fn is_filler(t: &str) -> bool {
    FILLERS.contains(&t)
}

/// Lemma candidates of an English word: itself, and the `-ing` stems.
fn lemmas(word: &str) -> Vec<String> {
    let mut out = vec![word.to_string()];
    if let Some(stem) = word.strip_suffix("ing").filter(|s| s.len() >= 2) {
        out.push(stem.to_string());
        out.push(format!("{stem}e"));
        let b = stem.as_bytes();
        if b.len() >= 2 && b[b.len() - 1] == b[b.len() - 2] {
            out.push(stem[..stem.len() - 1].to_string());
        }
    }
    out
}

/// The lexicon verb `token` opens with, in its lemma form.
fn verb_of(token: &str) -> Option<String> {
    if let Some(lemma) = lemmas(token)
        .into_iter()
        .find(|l| ACTION_VERBS.contains(&l.as_str()))
    {
        return Some(lemma);
    }
    None
}

/// Whether a verb (or a parallel marker or `then`) follows position `i`,
/// skipping fillers.
fn step_follows(tokens: &[String], i: usize) -> bool {
    let mut j = i;
    while j < tokens.len() {
        let t = tokens[j].as_str();
        if t == "then" || parallel_marker_at(tokens, j).is_some() || verb_of(t).is_some() {
            return true;
        }
        if !is_filler(t) {
            return false;
        }
        j += 1;
    }
    false
}

fn flush(cur: &mut Vec<String>, lead: &mut Option<Lead>, link: Link, out: &mut Vec<RawClause>) {
    if !cur.is_empty() {
        out.push(RawClause {
            tokens: std::mem::take(cur),
            link,
            lead: lead.take(),
        });
    }
}

/// Clauses in text order, each with its link to the clause before.
fn split_clauses(tokens: &[String]) -> Vec<RawClause> {
    let mut out = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut lead: Option<Lead> = None;
    let mut link = Link::Seq;
    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i].as_str();
        if is_hard_separator(t) {
            flush(&mut cur, &mut lead, link, &mut out);
            link = Link::Seq;
            i += 1;
            continue;
        }
        if let Some(len) = parallel_marker_at(tokens, i) {
            let ends_clause = i + len == tokens.len()
                || is_hard_separator(&tokens[i + len])
                || tokens[i + len] == ","
                || (tokens[i + len] == "and" && step_follows(tokens, i + len + 1));
            if !cur.is_empty() && ends_clause {
                // Suffix form, `X and Y in parallel`: this clause joins the step before.
                flush(&mut cur, &mut lead, Link::Par, &mut out);
                link = Link::Seq;
            } else {
                // Prefix form, `meanwhile Y`: the next clause joins the step before.
                flush(&mut cur, &mut lead, link, &mut out);
                link = Link::Par;
            }
            i += len;
            continue;
        }
        if (t == "," || t == "and") && !cur.is_empty() && step_follows(tokens, i + 1) {
            flush(&mut cur, &mut lead, link, &mut out);
            link = Link::Seq;
            i += 1;
            continue;
        }
        if (t == "after" || t == "before") && cur.is_empty() {
            lead = Some(if t == "after" {
                Lead::After
            } else {
                Lead::Before
            });
            i += 1;
            continue;
        }
        if (t == "after" || t == "before") && step_follows(tokens, i + 1) {
            flush(&mut cur, &mut lead, link, &mut out);
            link = if t == "after" { Link::After } else { Link::Seq };
            i += 1;
            continue;
        }
        if matches!(t, "afterwards" | "next" | "finally" | "lastly")
            && !cur.is_empty()
            && step_follows(tokens, i + 1)
        {
            flush(&mut cur, &mut lead, link, &mut out);
            link = Link::Seq;
            i += 1;
            continue;
        }
        if t != "," {
            cur.push(t.to_string());
        }
        i += 1;
    }
    flush(&mut cur, &mut lead, link, &mut out);
    out
}

/// Put the clauses in execution order. `A after B` and `before B, A` swap
/// the pair; two inversions in a row are refused.
fn order_clauses(raw: Vec<RawClause>) -> Result<Vec<(Vec<String>, Link)>, Refusal> {
    let mut out: Vec<(Vec<String>, Link)> = Vec::with_capacity(raw.len());
    let mut prev_lead: Option<Lead> = None;
    let mut prev_inverted = false;
    for (k, c) in raw.into_iter().enumerate() {
        let inverted = k > 0
            && (c.link == Link::After || (c.link == Link::Seq && prev_lead == Some(Lead::Before)));
        if inverted {
            if prev_inverted {
                return Err((
                    0.0,
                    "ambiguous_order: two `after`/`before` inversions in a row".into(),
                ));
            }
            let (prev_tokens, prev_link) = out.pop().expect("k > 0");
            out.push((c.tokens, prev_link));
            out.push((prev_tokens, Link::Seq));
        } else {
            out.push((c.tokens, c.link));
        }
        prev_inverted = inverted;
        prev_lead = c.lead;
    }
    Ok(out)
}

/// Name and verb of one clause: fillers stripped, the verb lemmatized.
fn clause(tokens: &[String]) -> Clause {
    let start = tokens
        .iter()
        .position(|t| !is_filler(t))
        .unwrap_or(tokens.len());
    let words = &tokens[start..];
    let verb = words.first().and_then(|w| verb_of(w));
    let mut parts: Vec<String> = words.to_vec();
    if let (Some(v), Some(first)) = (&verb, parts.first_mut()) {
        *first = v.clone();
    }
    Clause {
        name: parts.join(" "),
        verb,
    }
}

/// Steps from the ordered clauses, then each action with the previous
/// step as its parents. Refuses a repeated action, an over-long name and a
/// last step with more than one action.
fn build_actions(clauses: Vec<(Clause, Link)>) -> Result<Vec<InductedAction>, String> {
    let mut steps: Vec<Vec<Clause>> = Vec::new();
    for (c, link) in clauses {
        if c.name.len() > MAX_ACTION_NAME_BYTES {
            return Err(format!(
                "action_name_too_long: {:?} has {} bytes, at most {MAX_ACTION_NAME_BYTES}",
                c.name,
                c.name.len()
            ));
        }
        match (link, steps.last_mut()) {
            (Link::Par, Some(step)) => step.push(c),
            _ => steps.push(vec![c]),
        }
    }
    if let Some(last) = steps.last().filter(|s| s.len() > 1) {
        let names: Vec<&str> = last.iter().map(|c| c.name.as_str()).collect();
        return Err(format!(
            "no_single_target: the last step runs {names:?} in parallel; the DAG needs one goal \
             action"
        ));
    }
    let mut names = HashSet::new();
    let mut ids = HashSet::new();
    let mut actions = Vec::new();
    let mut prev_ids: Vec<u32> = Vec::new();
    let last = steps.len() - 1;
    for (s, step) in steps.into_iter().enumerate() {
        let mut step_ids = Vec::with_capacity(step.len());
        for c in step {
            let id = action_id(&c.name).0;
            if !names.insert(c.name.clone()) {
                return Err(format!("duplicate_action: {:?} appears twice", c.name));
            }
            if !ids.insert(id) {
                return Err(format!(
                    "action_id_collision: {:?} hashes to the id {id} of another action",
                    c.name
                ));
            }
            actions.push(InductedAction {
                operator: gen_zero_lod::builtin_operator::dcm_signature(),
                action_id: id,
                name: c.name,
                verb: c.verb,
                parents: prev_ids.clone(),
                cost: 1.0,
                target: s == last,
            });
            step_ids.push(id);
        }
        prev_ids = step_ids;
    }
    Ok(actions)
}

/// The planner's wire DAG for `actions`, validated by [`CausalDag::from_spec`].
fn dag_spec(actions: &[InductedAction]) -> Result<CausalDagSpec, Rejection> {
    let target = actions
        .iter()
        .find(|a| a.target)
        .map(|a| a.action_id)
        .ok_or_else(|| Rejection::invalid(STAGE, "internal: induced DAG has no target"))?;
    let spec = CausalDagSpec {
        parents: actions
            .iter()
            .filter(|a| !a.parents.is_empty())
            .map(|a| (a.action_id, a.parents.clone()))
            .collect(),
        is_or: BTreeMap::new(),
        cost: actions.iter().map(|a| (a.action_id, 1)).collect(),
        value: BTreeMap::from([(target, 1.0)]),
        target,
        budget: u32::try_from(actions.len()).expect("at most MAX_INDUCED_ACTIONS"),
    };
    let candidates: Vec<ActionId> = actions.iter().map(|a| ActionId(a.action_id)).collect();
    CausalDag::from_spec(&candidates, &spec).map_err(|e| {
        Rejection::invalid(
            STAGE,
            format!("internal: induced DAG failed validation: {e}"),
        )
    })?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_lexicon_verb_is_recognized_once() {
        let mut seen = HashSet::new();
        for v in ACTION_VERBS.iter() {
            assert!(seen.insert(*v), "{v} listed twice");
            assert_eq!(verb_of(v).as_deref(), Some(*v));
        }
    }

    #[test]
    fn tokenize_keeps_file_names_and_splits_trailing_punctuation() {
        assert_eq!(
            tokenize("Load data.csv, then save->db."),
            ["load", "data.csv", ",", "then", "save", "->", "db", "."]
        );
    }

    #[test]
    fn lemmas_cover_ing_forms() {
        assert_eq!(verb_of("cleaning").as_deref(), Some("clean"));
        assert_eq!(verb_of("saving").as_deref(), Some("save"));
        assert_eq!(verb_of("running").as_deref(), Some("run"));
        assert_eq!(verb_of("banana"), None);
    }

    #[test]
    fn word_shape_flags_keyboard_mash() {
        assert_eq!(word_shaped("asdfghjkl"), Some(false));
        assert_eq!(word_shaped("db"), Some(true));
        assert_eq!(word_shaped("fetch"), Some(true));
        assert_eq!(word_shaped("v2"), None);
    }
}
