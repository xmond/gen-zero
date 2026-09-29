"""Gen-Zero Hard Safety: Declarative DSL / AST Constraint Compiler.

RFC-069 & Issue #76 Implementation:
1. Natural Language & AST Constraint Parsing: Translates declarative safety rules
   and Python AST expressions into formal 0-1 linear inequality constraints.
2. Versioned Typed Schema (CONSTRAINT_SCHEMA_VERSION): each single-variable predicate is
   (schema version, dtype, variable, operator, threshold), bound to a coordinate of the
   continuous latent state z in R^D via a name-sorted (order-of-rules-independent) map.
   Truth is decided by applying the EXACT comparator to the extracted coordinate -- never a
   fixed epsilon shifting the decision boundary, which either opens a real gap between
   complementary predicates or gets silently swallowed by float rounding at large magnitudes.
   W_sat / b_sat retain a full-rank orthonormal structure for diagnostic/shape purposes only;
   they are never the source of a proposition's truth value.
3. Strict Fail-Closed Verification: Unresolved variables, NaNs, missing metrics, unsupported
   syntax (internal OR/NOT, and any AST term that is not a comparison or an AND of
   comparisons) and latent coordinates missing from a shorter-than-schema vector
   all resolve to the explicit UNKNOWN/ABSTAIN tri-state (see ``Tristate``) and
   fail closed, to eliminate unauthorized action pass-throughs. Every term of an
   AND-of-comparisons is schema-bound and evaluated -- not just its first term --
   so ``project_latent_propositions`` and ``evaluate_condition`` never disagree
   on a compound proposition's truth value.
4. Safe Case-Insensitive Action Matching: Action IDs are strictly normalized with .strip().upper().
5. Unsafe State Attribution: If all candidates are barred or fallback action itself is forbidden,
   solver strictly sets is_safe=False and marks solver_status="UNSAFE_NO_FEASIBLE_ACTIONS".
6. Neuro-Symbolic Hard Safety Enforcement: Neural proposals are formally vetted by
   a 0-1 CP-SAT integer linear programming solver, ensuring 100% hard blocking of safety violations.
7. Dynamic Runtime Updating: Safety rules can be compiled, modified, or appended in sub-2ms
   without retraining or fine-tuning neural weights.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union, Any, Set, Sequence
import ast
import hashlib
import logging
import numbers
import operator
import os
import sys
import time
import math
import re
import numpy as np

# The largest finite magnitude representable in float64 (and thus assignable into
# the diagnostic W_sat/b_sat matrices without OverflowError). Threshold literals
# beyond this -- whether a huge int like 10**400 or a float overflowed to inf by
# Python's own literal folding (`1e309` -> `inf`) -- are rejected at parse time
# (X-C03/X-C04): silently keeping them would either fold a real threshold to a
# no-op inf-or-negative-inf comparison or crash the matrix assignment.
_MAX_FINITE_THRESHOLD = sys.float_info.max

from gen_zero.gate.action_constraints import (
    FV_NO_ACTION,
    FV_NOT_REQUESTED,
    FV_UNAVAILABLE,
    ActionConstraintSpec,
    ProjectionResult,
    cpsat_available,
    cpsat_verify_selection,
    parse_action_constraints,
    project_distribution,
    resolve_disabled_actions,
)
from gen_zero.gate.cpsat_formal_solver import CPSATVerdict

logger = logging.getLogger(__name__)

# R13-01: the versioned typed IR that replaces the eps-biased linear threshold.
# Every predicate is (schema_version, dtype, variable, operator, threshold);
# truth is decided by applying the EXACT operator below to the extracted
# value, never by shifting the decision boundary with a fixed epsilon. A
# fixed epsilon either opens a real gap in the reals (both `x > 0` and
# `x <= 0` false at np.nextafter(0, +inf)) or gets silently swallowed by
# float rounding at larger magnitudes (`x = 1024` in float32), so both
# complementary predicates read True. Exact comparison on the same value has
# neither failure mode: strict/non-strict is a property of the operator, not
# of the threshold.
CONSTRAINT_SCHEMA_VERSION = "gate.constraint_compiler.schema.v2"
CONSTRAINT_SCHEMA_DTYPE = "float64"

_EXACT_OPERATORS: Dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


# T3-C01: the largest integer float64 can represent exactly. float64 has 52
# explicit mantissa bits plus an implicit leading 1 (53 bits of precision), so
# every |n| <= 2**53 round-trips through float64 unchanged (2**53 itself is
# just 1.0 * 2**53); every |n| beyond it (starting at the first odd integer,
# 2**53+1) rounds to some other representable value the moment it is cast,
# which is exactly the silent narrowing this module must reject rather than
# perform.
_MAX_EXACT_FLOAT64_INT = 1 << 53  # 9007199254740992 == 2**53


class LossyNumericConversionError(ValueError):
    """Raised when a latent value cannot be cast to float64 without narrowing.

    T3-C01: ``_validate_latent`` unconditionally cast every input to float64
    before this class existed. An input integer beyond
    ``_MAX_EXACT_FLOAT64_INT`` (Python int, ``np.integer``, or an
    integer-dtype array/list element) is not representable exactly in
    float64 -- casting it anyway silently rounds it to a different value,
    which can flip a threshold comparison at the 2**53 boundary. Subclasses
    ``ValueError`` so every existing fail-closed ``except ValueError`` around
    this compiler's latent-consuming entry points still catches it.
    """


class SchemaMismatchError(ValueError):
    """Raised when a latent vector is evaluated without the schema it was bound to.

    X-C01: the compiler's variable-to-coordinate map (``_variable_dimensions``) is
    rebuilt every time ``compile_rules`` runs, sorted by variable name. Adding a
    rule for a new, alphabetically-earlier variable silently shifts every later
    variable's coordinate. A caller holding a ``z_latent`` produced under the old
    map has no way to detect the shift unless the map itself is checked -- so any
    entry point that consumes ``z_latent`` requires the caller to also pass the
    ``schema_fingerprint`` (``ConstraintLinearProjectionCompiler.schema_fingerprint``)
    that was current when the vector was produced. A missing or stale fingerprint
    fails closed here rather than silently reinterpreting the vector under today's
    (possibly different) coordinate map.
    """


class Tristate(Enum):
    """Fail-closed truth value for a single predicate (R13-01 tri-state margin).

    TRUE / FALSE are ordinary boolean outcomes reached by exact comparison.
    UNKNOWN is the explicit ABSTAIN margin: the predicate could not be
    resolved (missing metric, non-finite value, unsupported operator, or a
    variable with no schema-bound latent coordinate). Callers must never
    treat UNKNOWN as FALSE; ``evaluate_condition`` collapses it to whichever
    branch is SAFE for the rule kind (forbid/require the action rather than
    silently authorizing it).
    """
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    def __bool__(self) -> bool:
        # Every Enum member is truthy by default, so `if prop_map[k]:` would read
        # Tristate.FALSE as true -- silently reintroducing the exact "treat
        # UNKNOWN/FALSE as authorized" bug this type exists to prevent. Force
        # callers to compare against a specific member (`is Tristate.TRUE`)
        # instead of ever branching on truthiness.
        raise TypeError(
            "Tristate has no truth value -- compare explicitly against "
            "Tristate.TRUE / Tristate.FALSE / Tristate.UNKNOWN"
        )


@dataclass
class CompiledConstraintRule:
    """Represents a compiled linear safety rule."""
    rule_id: str
    raw_expression: str
    rule_type: str  # "FORBID_IF", "ALLOW_ONLY_IF", "REQUIRE_WHEN", "MUTUAL_EXCLUSIVE", "THRESHOLD"
    target_action: Optional[str]
    condition_expr: str
    proposition_index: int
    variable_name: Optional[str] = None
    comparator: Optional[str] = None
    # X-C04: kept as the parsed Python type (int stays int) so comparisons against
    # a same-typed or float metric are exact -- Python's mixed int/float rich
    # comparison never narrows either side, unlike an explicit float() cast.
    threshold: Union[int, float] = 0.0
    sub_conditions: List[Dict[str, Any]] = field(default_factory=list)
    is_hard: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "raw_expression": self.raw_expression,
            "rule_type": self.rule_type,
            "target_action": self.target_action,
            "condition_expr": self.condition_expr,
            "proposition_index": self.proposition_index,
            "variable_name": self.variable_name,
            "comparator": self.comparator,
            "threshold": self.threshold,
            "sub_conditions": self.sub_conditions,
            "is_hard": self.is_hard,
        }


@dataclass
class CompilationReport:
    """Telemetry report produced by compiling a constraint specification."""
    total_rules_compiled: int
    num_propositions: int
    latent_dim: int
    compilation_time_ms: float
    is_orthonormal: bool
    rules: List[CompiledConstraintRule]
    schema_version: str = CONSTRAINT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rules_compiled": self.total_rules_compiled,
            "num_propositions": self.num_propositions,
            "latent_dim": self.latent_dim,
            "compilation_time_ms": round(self.compilation_time_ms, 3),
            "is_orthonormal": self.is_orthonormal,
            "rules": [r.to_dict() for r in self.rules],
            "schema_version": self.schema_version,
        }


class ASTConstraintParser:
    """Parses safe Python AST expressions into linear comparative propositions."""

    SAFE_COMPARATORS = {
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Eq: "==",
        ast.NotEq: "!=",
    }

    @classmethod
    def _safe_threshold(cls, value: Any) -> Optional[Union[int, float]]:
        """Validates a parsed numeric constant (X-C03/X-C04).

        ``type(value) in (int, float)`` (not ``isinstance``) so a stray ``bool``
        constant -- a subclass of ``int`` in Python -- is never silently read as
        0/1. An int is returned unchanged (never narrowed to float, so exact
        comparison against a same-magnitude metric never loses precision). Either
        type is rejected outside +/-float64-max: a float there is already `inf`/
        `-inf` from Python folding an overflowing literal (`1e309`), and an int
        there would raise OverflowError the moment it is written into the
        diagnostic float32 W_sat/b_sat matrices. Both cases must become
        UNSUPPORTED_SYNTAX rather than a threshold that silently never fires.
        """
        if type(value) not in (int, float):
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if abs(value) > _MAX_FINITE_THRESHOLD:
            return None
        return value

    @classmethod
    def parse_single_comparison(cls, compare_node: ast.Compare) -> Optional[Tuple[str, str, Union[int, float]]]:
        """Parses a single ast.Compare node into (var_name, comparator_symbol, threshold)."""
        if (
            isinstance(compare_node.left, ast.Name)
            and len(compare_node.ops) == 1
            and len(compare_node.comparators) == 1
        ):
            var_name = compare_node.left.id
            op_type = type(compare_node.ops[0])
            comp_symbol = cls.SAFE_COMPARATORS.get(op_type, None)
            if comp_symbol is None:
                return None

            right_node = compare_node.comparators[0]
            if isinstance(right_node, ast.Constant):
                thresh = cls._safe_threshold(right_node.value)
                if thresh is not None:
                    return var_name, comp_symbol, thresh
            elif isinstance(right_node, ast.UnaryOp) and isinstance(right_node.op, ast.USub):
                if isinstance(right_node.operand, ast.Constant):
                    thresh = cls._safe_threshold(right_node.operand.value)
                    if thresh is not None:
                        return var_name, comp_symbol, -thresh
        return None

    @classmethod
    def parse_expression(
        cls, expr_str: str
    ) -> Tuple[Optional[str], Optional[str], float, List[Dict[str, Any]]]:
        """Parses simple and compound (and) comparative expressions.

        Returns:
            Tuple of (primary_var, primary_comp, primary_thresh, list_of_all_sub_conditions).
        """
        expr_clean = expr_str.strip()
        sub_conditions: List[Dict[str, Any]] = []

        try:
            tree = ast.parse(expr_clean, mode="eval")
            if isinstance(tree.body, ast.Compare):
                single = cls.parse_single_comparison(tree.body)
                if single is not None:
                    var_name, comp_symbol, thresh = single
                    cond_dict = {"variable_name": var_name, "comparator": comp_symbol, "threshold": thresh}
                    return var_name, comp_symbol, thresh, [cond_dict]
            elif isinstance(tree.body, ast.BoolOp) and isinstance(tree.body.op, ast.And):
                all_parsed = True
                for val_node in tree.body.values:
                    if isinstance(val_node, ast.Compare):
                        sub = cls.parse_single_comparison(val_node)
                        if sub is not None:
                            sub_conditions.append({
                                "variable_name": sub[0],
                                "comparator": sub[1],
                                "threshold": sub[2]
                            })
                        else:
                            all_parsed = False
                            break
                    else:
                        all_parsed = False
                        break
                if all_parsed and sub_conditions:
                    first = sub_conditions[0]
                    return first["variable_name"], first["comparator"], first["threshold"], sub_conditions
        except Exception:
            pass

        # Regex fallback for single natural expressions
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(<=|>=|==|!=|<|>)\s*([-+]?[0-9]*\.?[0-9]+)$", expr_clean)
        if m:
            var_name, comp = m.group(1), m.group(2)
            # X-C03/X-C04: route through the same overflow/precision guard as the
            # AST path -- a bare digit run this long (e.g. 10**400 as text) would
            # otherwise reach float() unfiltered and either overflow to inf or
            # silently narrow, bypassing the AST branch's rejection entirely.
            digits = m.group(3)
            raw_value: Union[int, float] = int(digits) if re.fullmatch(r"[-+]?[0-9]+", digits) else float(digits)
            thresh = cls._safe_threshold(raw_value)
            if thresh is not None:
                cond_dict = {"variable_name": var_name, "comparator": comp, "threshold": thresh}
                return var_name, comp, thresh, [cond_dict]

        return None, None, 0.0, []


class ConstraintLinearProjectionCompiler:
    """Automated Natural Language and AST Constraint Linear Projection Compiler for CP-SAT."""

    def __init__(
        self,
        latent_dim: int = 1024,
        seed: int = 42,
        hard_timeout_ms: Optional[float] = None,
    ) -> None:
        if hard_timeout_ms is None:
            hard_timeout_ms = float(os.environ.get("GENZERO_CPSAT_TIMEOUT_MS", "50.0"))
        if not math.isfinite(hard_timeout_ms) or hard_timeout_ms <= 0:
            raise ValueError("hard_timeout_ms must be finite and positive")
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            logger.error("OR-Tools unavailable: constraint solving will fail closed")
        self.latent_dim = latent_dim
        self.seed = seed
        self.hard_timeout_ms = float(hard_timeout_ms)

        self._variable_dimensions: Dict[str, int] = {}
        self._rules: List[CompiledConstraintRule] = []
        self._w_sat: Optional[np.ndarray] = None  # Shape: [num_propositions, latent_dim]
        self._b_sat: Optional[np.ndarray] = None  # Shape: [num_propositions]
        self._proposition_names: List[str] = []
        self._prop_meta: List[Dict[str, Any]] = []
        self._schema_version: str = CONSTRAINT_SCHEMA_VERSION
        self._action_to_rules: Dict[str, List[CompiledConstraintRule]] = {}
        self._action_constraints: Tuple[ActionConstraintSpec, ...] = ()

    @property
    def num_action_constraints(self) -> int:
        return len(self._action_constraints)

    @property
    def num_rules(self) -> int:
        return len(self._rules)

    @property
    def num_propositions(self) -> int:
        return len(self._proposition_names)

    @property
    def projection_matrix(self) -> Optional[np.ndarray]:
        return self._w_sat

    @property
    def projection_bias(self) -> Optional[np.ndarray]:
        return self._b_sat

    @property
    def schema_fingerprint(self) -> str:
        """The versioned variable->coordinate schema active after the last ``compile_rules``.

        X-C01: any caller that wants to evaluate a ``z_latent`` against this
        compiler must capture this value at the time the vector's producer last
        observed the schema, and pass it back in as ``schema_fingerprint`` to
        ``evaluate_condition`` / ``project_latent_propositions`` /
        ``solve_safest_action``. A caller that re-reads this property at call
        time instead of caching it defeats the check -- it would always match.
        """
        return self._schema_version

    def compile_rules(self, raw_rules: Sequence[str]) -> CompilationReport:
        """Compiles declarative natural language and AST rules into 0-1 linear CP-SAT constraints."""
        t0 = time.perf_counter()
        compiled: List[CompiledConstraintRule] = []
        unique_propositions: List[str] = []
        action_map: Dict[str, List[CompiledConstraintRule]] = {}

        for idx, rule_str in enumerate(raw_rules):
            r = rule_str.strip()
            if not r or r.startswith("#"):
                continue

            rule_id = f"RULE_{idx + 1:03d}"
            rule_type = "FORBID_IF"
            target_action = None
            condition_expr = ""
            var_name = None
            comparator = None
            threshold = 0.0
            sub_conditions: List[Dict[str, Any]] = []
            metadata: Dict[str, Any] = {}

            # Pattern 1: FORBID / BAN / DO NOT / DENY <ACTION> IF <EXPR>
            m_forbid = re.match(r"^(?:FORBID|BAN|DO NOT|DENY)\s+([a-zA-Z0-9_\-:]+)\s+(?:IF|WHEN)\s+(.+)$", r, re.IGNORECASE)
            # Pattern 2: ALLOW <ACTION> ONLY IF <EXPR>
            m_allow = re.match(r"^ALLOW\s+([a-zA-Z0-9_\-:]+)\s+ONLY\s+IF\s+(.+)$", r, re.IGNORECASE)
            # Pattern 3: REQUIRE <ACTION> WHEN <EXPR>
            m_require = re.match(r"^REQUIRE\s+([a-zA-Z0-9_\-:]+)\s+(?:WHEN|IF)\s+(.+)$", r, re.IGNORECASE)
            # Pattern 4: MUTUAL_EXCLUSIVE <ACT1> <ACT2>
            m_mutex = re.match(r"^MUTUAL_EXCLUSIVE\s+([a-zA-Z0-9_\-:]+)\s+([a-zA-Z0-9_\-:]+)$", r, re.IGNORECASE)
            # Pattern 5: Arrow syntax "<EXPR> -> BAN <ACTION>"
            m_arrow = re.match(r"^(.+)\s*->\s*(?:BAN|FORBID)\s+([a-zA-Z0-9_\-:]+)$", r, re.IGNORECASE)

            if m_forbid:
                rule_type = "FORBID_IF"
                target_action = m_forbid.group(1).strip().upper()
                condition_expr = m_forbid.group(2).strip()
            elif m_allow:
                rule_type = "ALLOW_ONLY_IF"
                target_action = m_allow.group(1).strip().upper()
                condition_expr = m_allow.group(2).strip()
            elif m_require:
                rule_type = "REQUIRE_WHEN"
                target_action = m_require.group(1).strip().upper()
                condition_expr = m_require.group(2).strip()
            elif m_mutex:
                rule_type = "MUTUAL_EXCLUSIVE"
                target_action = m_mutex.group(1).strip().upper()
                condition_expr = m_mutex.group(2).strip().upper()
            elif m_arrow:
                rule_type = "FORBID_IF"
                target_action = m_arrow.group(2).strip().upper()
                condition_expr = m_arrow.group(1).strip()
            else:
                # Unsupported syntax is strictly recorded and not mapped to dummy actions (S15)
                rule_type = "UNSUPPORTED_SYNTAX"
                target_action = None
                condition_expr = r
                metadata = {"unsupported_reason": "rule does not match any recognized grammar "
                                                    "(FORBID/ALLOW/REQUIRE/MUTUAL_EXCLUSIVE/arrow)"}

            # Parse condition AST
            if rule_type not in ("MUTUAL_EXCLUSIVE", "UNSUPPORTED_SYNTAX"):
                v, c, t, subs = ASTConstraintParser.parse_expression(condition_expr)
                if v is not None:
                    var_name, comparator, threshold = v, c, t
                    sub_conditions = subs
                else:
                    # A recognized action verb ("FORBID X IF ...") whose condition does
                    # not reduce to a pure AND of comparisons (an internal OR, a NOT, a
                    # nested BoolOp, or an unevaluated call) must never be left looking
                    # like an ordinary rule with a merely-empty condition: that is exactly
                    # the "silently project down and return a plain bool" failure mode.
                    # Relabel it so both the CompilationReport and solve_safest_action's
                    # fail-closed gate see the real reason.
                    unsupported_reason = "condition does not reduce to a supported AND-of-comparisons"
                    try:
                        parsed_tree = ast.parse(condition_expr.strip(), mode="eval")
                        if isinstance(parsed_tree.body, ast.BoolOp) and isinstance(parsed_tree.body.op, ast.Or):
                            unsupported_reason = "internal OR is not supported"
                        elif isinstance(parsed_tree.body, ast.UnaryOp) and isinstance(parsed_tree.body.op, ast.Not):
                            unsupported_reason = "internal NOT is not supported"
                    except Exception:
                        pass
                    logger.error(
                        "Rule %s: %s -- %r. Marking UNSUPPORTED_SYNTAX (fail-closed).",
                        rule_id, unsupported_reason, r,
                    )
                    metadata = {
                        "attempted_rule_type": rule_type,
                        "attempted_target_action": target_action,
                        "unsupported_reason": unsupported_reason,
                    }
                    rule_type = "UNSUPPORTED_SYNTAX"
                    target_action = None

            # Register proposition
            prop_key = condition_expr
            if prop_key not in unique_propositions:
                unique_propositions.append(prop_key)
            prop_idx = unique_propositions.index(prop_key)

            rule_obj = CompiledConstraintRule(
                rule_id=rule_id,
                raw_expression=r,
                rule_type=rule_type,
                target_action=target_action,
                condition_expr=condition_expr,
                proposition_index=prop_idx,
                variable_name=var_name,
                comparator=comparator,
                threshold=threshold,
                sub_conditions=sub_conditions,
                is_hard=True,
                metadata=metadata,
            )
            compiled.append(rule_obj)

            if target_action:
                action_map.setdefault(target_action, []).append(rule_obj)

        self._rules = compiled
        self._proposition_names = unique_propositions
        self._action_to_rules = action_map

        # Generate Coordinate-Aligned Orthonormal / Complementary Projection Basis W_sat & b_sat (S08, S10, S17)
        num_props = max(1, len(unique_propositions))
        w_sat = np.zeros((num_props, self.latent_dim), dtype=np.float32)
        b_sat = np.zeros(num_props, dtype=np.float32)

        # 1. Extract each proposition's primary (variable, comparator, threshold) --
        #    used only for the W_sat/b_sat diagnostic row below -- AND its full
        #    sub_conditions list, which is what truth evaluation actually consumes
        #    (C07: a proposition's truth can never be decided from the primary term
        #    alone once it has siblings from an "and").
        prop_meta: List[Dict[str, Any]] = []

        for p_idx, prop in enumerate(unique_propositions):
            matched_rule = next((r for r in compiled if r.condition_expr == prop), None)
            var_name = matched_rule.variable_name if matched_rule else None
            comp = matched_rule.comparator if matched_rule else None
            thresh = matched_rule.threshold if matched_rule else 0.0
            sub_conditions = matched_rule.sub_conditions if matched_rule else []

            if not var_name:
                v, c, t, subs = ASTConstraintParser.parse_expression(prop)
                if v is not None:
                    var_name, comp, thresh, sub_conditions = v, c, t, subs

            prop_meta.append({
                "prop": prop,
                "var_name": var_name,
                "comparator": comp,
                "threshold": thresh,
                "sub_conditions": sub_conditions,
            })

        # 2. Bind every variable referenced by ANY sub_condition of ANY rule --
        #    not just each proposition's primary term -- to a fixed coordinate
        #    from the CANONICAL (sorted) variable name set, never from encounter
        #    order in `raw_rules`. C07: leaving a compound's secondary variables
        #    (the "y" in "x > 0 and y > 0") unbound is exactly what forced
        #    project_latent_propositions to silently drop them and answer from
        #    "x > 0" alone; binding every referenced variable gives the
        #    projection path the same information the metrics-dict path already
        #    had, so the two truth entries can agree instead of split.
        #    R13-01: assigning dims in rule-list order means the same z_latent can
        #    flip a predicate's truth value merely because an unrelated rule moved
        #    earlier in the list -- a FORBID guarded by "x > 0" could silently stop
        #    firing on reorder alone. Sorting by name makes the same rule *set*
        #    always produce the same coordinate map, independent of order, and the
        #    fingerprint below makes that binding auditable/versioned.
        all_referenced_vars: Set[str] = set()
        for r in compiled:
            for sub in r.sub_conditions:
                sub_var = sub.get("variable_name")
                if sub_var:
                    all_referenced_vars.add(sub_var)
        distinct_vars = sorted(all_referenced_vars)
        var_to_dim: Dict[str, int] = {
            name: i for i, name in enumerate(distinct_vars) if i < self.latent_dim
        }
        # X-C01: latent_dim is folded into the fingerprint so two compilers with
        # the identical variable->dim map but different capacities (which changes
        # how many further variables could still fit) are never mistaken for the
        # same schema.
        schema_fingerprint = "latent_dim={}|{}".format(
            self.latent_dim,
            "|".join(f"{name}:{dim}" for name, dim in sorted(
                var_to_dim.items(), key=lambda kv: kv[1]
            )),
        )
        self._schema_version = "{}:{}".format(
            CONSTRAINT_SCHEMA_VERSION,
            hashlib.sha256(schema_fingerprint.encode("utf-8")).hexdigest()[:12],
        )

        # 3. Assign coordinate axes and complementary opposing vectors. The bias is
        #    the EXACT threshold -- no epsilon shift. Strict-vs-non-strict is a
        #    property of the comparator applied at decision time (see
        #    project_latent_propositions / _EXACT_OPERATORS), never baked into the
        #    boundary itself. This is what actually guarantees the excluded middle:
        #    the same finite value fed through complementary operators always
        #    yields one True and one False, at every magnitude, with no gap.
        unassigned_indices: List[int] = []

        for p_idx, meta in enumerate(prop_meta):
            var_name = meta["var_name"]
            comp = meta["comparator"]
            thresh = meta["threshold"]

            if var_name and var_name in var_to_dim and comp in _EXACT_OPERATORS:
                dim_idx = var_to_dim[var_name]
                if comp in (">", ">="):
                    w_sat[p_idx, dim_idx] = 1.0
                    b_sat[p_idx] = -thresh
                elif comp in ("<", "<="):
                    w_sat[p_idx, dim_idx] = -1.0
                    b_sat[p_idx] = thresh
                elif comp == "==":
                    w_sat[p_idx, dim_idx] = 1.0
                    b_sat[p_idx] = -thresh
                else:  # "!="
                    w_sat[p_idx, dim_idx] = -1.0
                    b_sat[p_idx] = thresh
            else:
                unassigned_indices.append(p_idx)

        # 4. Propositions whose primary term still has no schema-bound coordinate
        #    (an unsupported/unparseable condition with no variable at all, or
        #    latent_dim too small to hold every distinct variable) still get a
        #    structural orthonormal filler row so W_sat stays full-rank for the
        #    diagnostic orthonormality check below. This filler is NEVER a source
        #    of truth: project_latent_propositions / evaluate_condition resolve
        #    truth only from the typed sub_conditions schema above (every
        #    referenced variable, not just the primary one -- see step 2), and
        #    report an unresolvable proposition as Tristate.UNKNOWN rather than
        #    fabricating a boolean from an arbitrary random projection.
        if unassigned_indices:
            used_dims = len(var_to_dim)
            rem_dims = max(len(unassigned_indices), self.latent_dim - used_dims)
            sub_basis = self._generate_orthonormal_basis(len(unassigned_indices), rem_dims, self.seed)
            for i, p_idx in enumerate(unassigned_indices):
                if used_dims + rem_dims <= self.latent_dim:
                    w_sat[p_idx, used_dims:used_dims + rem_dims] = sub_basis[i]
                else:
                    w_sat[p_idx, :] = sub_basis[i, :self.latent_dim]
                b_sat[p_idx] = -prop_meta[p_idx]["threshold"]

        self._variable_dimensions = var_to_dim
        self._prop_meta = prop_meta
        self._w_sat = w_sat
        self._b_sat = b_sat

        dur_ms = (time.perf_counter() - t0) * 1000.0
        is_orthonormal = self._verify_orthonormality(self._w_sat)

        return CompilationReport(
            total_rules_compiled=len(compiled),
            num_propositions=len(unique_propositions),
            latent_dim=self.latent_dim,
            compilation_time_ms=dur_ms,
            is_orthonormal=is_orthonormal,
            schema_version=self._schema_version,
            rules=compiled,
        )

    def _generate_orthonormal_basis(self, num_vectors: int, dim: int, seed: int) -> np.ndarray:
        """Generates strictly orthonormal projection basis via QR decomposition."""
        rng = np.random.RandomState(seed + 77)
        raw_mat = rng.randn(dim, num_vectors).astype(np.float32)
        q, _ = np.linalg.qr(raw_mat, mode="reduced")
        w_sat = q.T.astype(np.float32)
        norms = np.linalg.norm(w_sat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        w_sat = w_sat / norms
        return w_sat

    def _verify_orthonormality(self, matrix: np.ndarray, tol: float = 1e-4) -> bool:
        """Verifies that rows of W_sat are unit-normalized and pairwise orthogonal or anti-aligned."""
        if matrix is None or len(matrix) == 0:
            return False
        gram = np.dot(matrix, matrix.T)
        n = len(matrix)
        # Check diagonal: all norms must be 1.0
        if not np.allclose(np.diag(gram), 1.0, atol=tol):
            return False
        # Check off-diagonal: must be 0.0 (orthogonal) or -1.0 (complementary/opposing)
        for i in range(n):
            for j in range(i + 1, n):
                val = gram[i, j]
                if abs(val) > tol and abs(val - (-1.0)) > tol:
                    return False
        return True

    @staticmethod
    def _reject_lossy_integers(z_latent: Any) -> None:
        """T3-C01: reject any exact integer beyond +/-2**53 before the float64
        cast below has a chance to silently round it to a different value.

        A plain Python ``list``/``tuple`` is walked element-by-element at the
        PYTHON level, before any NumPy involvement: ``np.asarray`` on a
        *mixed* int/float list (e.g. ``[2**53+1, 1.5]``, both magnitudes
        within int64 range) itself upcasts the whole list to a float64 array
        immediately -- NumPy's own dtype-unification is the lossy step there,
        and it would erase the very int this check exists to catch before a
        dtype-based inspection ever got a chance to run. Walking the raw
        Python objects first sidesteps that: each element is still whatever
        the caller actually put in the list.

        Anything else array-like (an ``np.ndarray``, a CPU ``torch.Tensor``,
        or any other object implementing the array protocol) is routed
        through ``np.asarray`` with no dtype forced, so an integer-dtype
        array's exact values are inspected via ``.tolist()`` -- native Python
        ints, compared with ordinary (exact, arbitrary-precision) int
        arithmetic, never narrowed through a float or promoted through a
        version-dependent NumPy comparison rule.

        The one input this cannot catch is a caller who already built a
        float-dtype ndarray/tensor themselves (e.g. ``np.asarray([2**53+1],
        dtype=np.float64)`` or ``torch.tensor([2**53+1], dtype=torch.float32)``
        before ever calling into this compiler) -- the original integer is
        already gone by the time ``z_latent`` reaches us, and no inspection
        performed here can recover it.

        T3-C01 (WebGPT round 4): an ``np.ndarray`` -- 0-d or multi-dim -- can
        also appear as a LEAF inside a list/tuple (``[np.array(2**53+1,
        dtype=np.int64)]``), not just as the top-level ``z_latent``. The
        original ``_walk`` recursed into list/tuple but routed every other
        leaf straight to ``_check_scalar``, which only recognizes
        ``(int, np.integer)`` -- an ndarray leaf matched neither branch and
        silently passed through, reaching ``_validate_latent``'s float64 cast
        unchecked. Every leaf that is an ``np.ndarray`` is now routed through
        the same dtype-based ``_check_ndarray`` used for the top-level array
        case, at whatever nesting depth it is found.

        R5-N01 (WebGPT round 5): the round-4 fix above still only special-cased
        ``np.ndarray`` leaves. A ``torch.Tensor`` leaf (``[torch.tensor(2**53+1,
        dtype=torch.int64)]``) is neither a list/tuple nor an ``np.ndarray``, so
        it fell into the exact same ``_check_scalar`` gap the round-4 fix
        closed for ndarrays -- silently passing through to the float64 cast,
        which rounds ``2**53+1`` down to ``2**53``. The top-level (non-list)
        branch below already handles this correctly by converting any
        non-ndarray argument via ``np.asarray`` (the array protocol), so a
        bare top-level ``torch.Tensor`` was never vulnerable; only a
        Tensor *nested* inside a list/tuple was. ``_walk`` now applies the
        same array-protocol conversion to any leaf that exposes ``__array__``
        (a CPU ``torch.Tensor``, or any other array-like object) before
        falling back to ``_check_scalar``. Plain Python ``int``/``float``/
        ``bool`` do not implement ``__array__`` (only ``np.integer``/
        ``np.floating`` scalars and true array-like objects do), so ordinary
        scalar leaves are unaffected and still go through ``_check_scalar``
        exactly as before.

        R6-N01 (ChatGPT 6 Pro round 6): every check above only ever inspected
        values that were ALREADY known to be ``(int, np.integer)`` -- anything
        else (a ``str``, ``Decimal``, ``Fraction``, or Python ``complex``) fell
        through every branch with no check at all, reaching
        ``_validate_latent``'s unconditional ``np.asarray(..., dtype=float64)``
        cast unguarded. That cast then performs the exact silent damage this
        module exists to prevent: NumPy parses a numeric string and rounds it
        (``"9007199254740993"`` -> ``9007199254740992.0``), calls ``__float__``
        on a ``Decimal``/``Fraction`` with the same silent rounding, and casts a
        complex array/tensor to float64 by dropping the imaginary part outright
        (``1+2j`` -> ``1.0``). ``_check_scalar`` and ``_check_ndarray`` now
        enforce a CLOSED real-scalar whitelist instead of only checking
        magnitude: any scalar that is not a Python ``bool``/``int``/``float``
        or a NumPy real integer/float scalar, and any array/tensor whose dtype
        kind is not in ``('i', 'u', 'f')``, is rejected with ``TypeError``
        before it can reach the lossy cast -- str, Decimal, Fraction, complex
        scalars, and complex/string/object-incompatible arrays all fail closed
        here rather than silently converting or truncating.
        """
        def _check_scalar(value: Any) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, (int, np.integer)):
                if abs(int(value)) > _MAX_EXACT_FLOAT64_INT:
                    raise LossyNumericConversionError(
                        f"z_latent contains integer {int(value)} outside the float64 "
                        f"exact-integer range (+/-{_MAX_EXACT_FLOAT64_INT}); casting it "
                        "to float64 would silently narrow its value"
                    )
                return
            if isinstance(value, float):
                return
            if isinstance(value, np.floating):
                # R7-N01: np.longdouble/float128/float96 report dtype.kind == "f"
                # on this platform (NOT a distinct "g" kind) -- itemsize is the
                # only signal that separates them from float16/32/64. Their 80-
                # or 128-bit mantissa (63 bits on this box, vs float64's 52) has
                # no lossless path to float64: it silently rounds a distinguishable
                # value to an indistinguishable one (1.0000000000000000009 -> 1.0),
                # or silently underflows a strictly-positive subnormal to 0.0
                # (ldexp(longdouble(1), -1075)). Reject before the cast, not after.
                if value.dtype.itemsize > 8:
                    raise TypeError(
                        f"z_latent contains an extended-precision float scalar of "
                        f"dtype {value.dtype!r} (itemsize={value.dtype.itemsize} "
                        "bytes); only IEEE-754 float16/float32/float64 (itemsize "
                        "<= 8 bytes) have a lossless path to float64 -- "
                        "np.longdouble/float128/float96 are platform-dependent "
                        "extended-precision types and would be silently narrowed "
                        "or underflowed by the float64 cast"
                    )
                return
            # R6-N01: closed whitelist -- anything else (str, Decimal, Fraction,
            # complex, np.complexfloating, None, ...) has no verified lossless
            # path to float64 and must fail closed rather than pass through to
            # np.asarray's implicit (and silently lossy) conversion.
            raise TypeError(
                f"z_latent contains a non-real-scalar leaf of type "
                f"{type(value).__name__!r} ({value!r}); only Python "
                "bool/int/float and NumPy real integer/float scalars are "
                "permitted -- str, Decimal, Fraction, and complex values are "
                "rejected rather than silently narrowed or truncated"
            )

        def _check_ndarray(arr: np.ndarray) -> None:
            # .flatten() first: also covers a bare 0-d array (arr.ndim == 0),
            # whose .tolist() would otherwise return a plain scalar, not an
            # iterable.
            kind = arr.dtype.kind
            if kind in ("i", "u"):
                for value in arr.flatten().tolist():
                    _check_scalar(value)
            elif kind == "f":
                # R7-N01: itemsize, not kind, is what separates float16/32/64
                # from np.longdouble/float128/float96 -- all report kind "f" on
                # this platform. See the matching check in _check_scalar above
                # for why extended precision cannot be losslessly cast.
                if arr.dtype.itemsize > 8:
                    raise TypeError(
                        f"z_latent contains an extended-precision float array/"
                        f"tensor of dtype {arr.dtype!r} (itemsize="
                        f"{arr.dtype.itemsize} bytes); only IEEE-754 float16/"
                        "float32/float64 (itemsize <= 8 bytes) have a lossless "
                        "path to float64 -- np.longdouble/float128/float96 are "
                        "platform-dependent extended-precision types and would "
                        "be silently narrowed or underflowed by the float64 cast"
                    )
                return
            elif kind == "O":
                # An object-dtype array can itself hold further nested
                # ints/arrays/lists -- walk each element rather than assuming
                # it is already a plain scalar.
                for value in arr.flatten().tolist():
                    _walk(value)
            else:
                # R6-N01: closed whitelist -- complex ('c'), string ('U'/'S'),
                # bool ('b'), datetime, and every other dtype kind has no
                # verified lossless path to float64 and must fail closed
                # instead of silently truncating (complex -> real part) or
                # being implicitly parsed (numeric string -> float).
                raise TypeError(
                    f"z_latent contains an array/tensor of disallowed dtype "
                    f"{arr.dtype!r} (kind={kind!r}); only integer, unsigned "
                    "integer, and float dtypes (kind in 'i', 'u', 'f') are "
                    "permitted -- complex and string dtypes are rejected "
                    "rather than silently truncated or parsed"
                )

        def _walk(obj: Any) -> None:
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    _walk(item)
            elif isinstance(obj, np.ndarray):
                _check_ndarray(obj)
            elif hasattr(obj, "__array__"):
                # R5-N01: any other array-like leaf (a CPU torch.Tensor, or
                # anything else implementing the array protocol). Plain
                # Python int/float/bool do NOT implement __array__ -- only
                # np.integer/np.floating scalars and true array-like objects
                # do -- so this cannot swallow an ordinary scalar leaf into
                # the wrong branch.
                _check_ndarray(np.asarray(obj))
            else:
                _check_scalar(obj)

        if isinstance(z_latent, (list, tuple)):
            _walk(z_latent)
            return

        # Not a list/tuple: any array-like object (np.ndarray, torch.Tensor,
        # ...) is converted via the array protocol with no dtype forced, so
        # the dtype below reflects the caller's own array, not a cast we
        # impose. This deliberately has no other branch -- an object that is
        # neither list/tuple nor array-like raises here (TypeError from
        # np.asarray), which is fail-closed, not a silent skip.
        raw = z_latent if isinstance(z_latent, np.ndarray) else np.asarray(z_latent)
        _check_ndarray(raw)

    @classmethod
    def _validate_latent(cls, z_latent: np.ndarray) -> np.ndarray:
        # T3-C01: reject a lossy integer BEFORE the float64 cast, not after --
        # once cast, the original exact value is gone and there is nothing left
        # to check.
        cls._reject_lossy_integers(z_latent)
        # Validate before dimensional truncation or any early return. float64 (the
        # schema dtype, CONSTRAINT_SCHEMA_DTYPE) so a probe like
        # np.nextafter(np.float32(0), np.float32(inf)) is not silently underflowed
        # to 0.0 by a float32 cast before the finite check even runs.
        with np.errstate(over="ignore", invalid="ignore"):
            z = np.asarray(z_latent, dtype=np.float64).flatten()
        if not np.all(np.isfinite(z)):
            raise ValueError("z_latent contains NaN or Inf")
        return z

    def _bind_latent(
        self, z_latent: np.ndarray, schema_fingerprint: Optional[str]
    ) -> Dict[str, float]:
        """Validates ``z_latent`` and extracts its schema-bound variables as a metrics dict.

        Order of checks matters: malformed data (NaN/Inf) is rejected before the
        schema is even consulted, so a caller debugging a bad vector always sees
        the same "z_latent contains NaN or Inf" message regardless of whether it
        also got the fingerprint right.

        X-C01: a missing or stale ``schema_fingerprint`` fails closed
        (``SchemaMismatchError``) rather than silently reinterpreting the vector
        under whatever coordinate map ``compile_rules`` currently holds.

        X-C02: dimensions beyond ``len(z_latent)`` are left OUT of the returned
        dict -- never zero-padded. A variable with no entry reads as
        Tristate.UNKNOWN downstream (``_evaluate_sub_condition``), never as 0.0.
        """
        z = self._validate_latent(z_latent)
        if schema_fingerprint is None:
            raise SchemaMismatchError(
                "z_latent was passed without schema_fingerprint; pass "
                "compiler.schema_fingerprint captured when this vector's "
                "coordinate map was produced -- an unlabeled vector cannot be "
                "safely reinterpreted against whatever schema compile_rules "
                "currently holds."
            )
        if schema_fingerprint != self._schema_version:
            raise SchemaMismatchError(
                f"schema_fingerprint {schema_fingerprint!r} does not match this "
                f"compiler's current schema {self._schema_version!r}; the "
                "variable->coordinate map has changed (e.g. compile_rules ran "
                "again with a different variable set) since this vector's "
                "schema was captured."
            )
        return {
            var: z[dim] for var, dim in self._variable_dimensions.items() if dim < len(z)
        }

    def project_latent_propositions(
        self, z_latent: np.ndarray, schema_fingerprint: Optional[str] = None
    ) -> Dict[str, Tristate]:
        """Projects continuous latent state z into discrete proposition truth values.

        C07: this is no longer a second, weaker truth path. Each proposition's
        full ``sub_conditions`` list (not just its primary term) is evaluated
        through the exact same ``_evaluate_proposition`` combinator that
        ``evaluate_condition`` uses, against the exact same schema-bound metrics
        dict (``_bind_latent``) that a ``current_metrics`` caller would supply by
        hand. A compound "x > 0 and y > 0" is fully resolved when both x and y
        have schema-bound coordinates within ``len(z_latent)``, and abstains
        (Tristate.UNKNOWN) -- never a partial answer from "x > 0" alone -- the
        moment either operand is unresolvable.
        """
        metrics = self._bind_latent(z_latent, schema_fingerprint)
        if len(self._proposition_names) == 0:
            return {}

        results: Dict[str, Tristate] = {}
        for idx, prop_name in enumerate(self._proposition_names):
            meta = self._prop_meta[idx] if idx < len(self._prop_meta) else {}
            sub_conditions = meta.get("sub_conditions") or []
            results[prop_name] = self._evaluate_proposition(sub_conditions, metrics)

        return results

    def _evaluate_sub_condition(self, cond: Dict[str, Any], metrics: Dict[str, Any]) -> Tristate:
        """Evaluates a single comparative term against metrics with EXACT comparison.

        Returns Tristate.TRUE / Tristate.FALSE if resolvable, else Tristate.UNKNOWN
        (missing variable, non-finite value, or unsupported operator). No tolerance
        window is applied to "==" or "!=": a fixed epsilon there either misses a
        real mismatch (x=5e-7 read as x==0) or misses a real match, and both are
        safety escapes for a discrete/indicator predicate.

        X-C04: neither ``val`` nor the rule's ``threshold`` is narrowed through
        ``float()``. Python's mixed int/float rich comparison is exact (no
        `x=2**53 == x+1` false positive), so an int metric or an int threshold
        parsed by ``ASTConstraintParser`` is compared as-is. ``numbers.Real`` (not
        a bare ``(int, float)`` isinstance check) so a numpy scalar in a
        caller-supplied metrics dict (``np.int64``, ``np.float64``) is still
        accepted rather than silently reading UNKNOWN; ``bool`` is excluded even
        though it is technically ``numbers.Integral``.

        T3-C01: Python's exact mixed int/float comparison only applies once
        BOTH operands are native Python objects. NumPy's own comparison
        operator does not honor it: comparing an ``np.floating`` against a
        Python/NumPy integer (or vice versa) upcasts both sides to float64
        first, which rounds any magnitude beyond 2**53 and can flip the
        verdict (``np.float64(2**53) < 2**53+1`` reads False; correct is
        True). Every NumPy scalar is therefore unwrapped via ``.item()`` to
        its native Python int/float BEFORE comparing, so the exact-comparison
        guarantee above actually holds instead of being silently bypassed by
        NumPy's ufunc casting rules the moment a ``z_latent``-derived metric
        (always an ``np.float64``, via ``_bind_latent``) is involved.
        """
        var_name = cond.get("variable_name")
        if not var_name or var_name not in metrics:
            return Tristate.UNKNOWN

        val = metrics[var_name]
        if val is None or isinstance(val, bool) or not isinstance(val, numbers.Real):
            return Tristate.UNKNOWN
        # Integral types (Python int, np.int64, ...) skip the isfinite probe: they
        # are always finite by definition, and math.isfinite raises OverflowError
        # on an int too large to convert to a C double (e.g. 10**400). Anything
        # else real-valued (float, np.float32/64, ...) is checked regardless of
        # its concrete type -- math.isfinite handles np.float32 correctly even
        # though it is not a Python `float` subclass.
        if not isinstance(val, numbers.Integral) and not math.isfinite(val):
            return Tristate.UNKNOWN

        op_fn = _EXACT_OPERATORS.get(cond.get("comparator"))
        if op_fn is None:
            return Tristate.UNKNOWN

        thresh = cond.get("threshold", 0.0)

        # T3-C01: unwrap NumPy scalars to native Python int/float -- a bit-exact
        # widening, never a precision change -- so the comparison below runs
        # entirely in Python's own (exact) int/float domain rather than
        # NumPy's (lossy-at-2**53) one.
        if isinstance(val, np.generic):
            val = val.item()
        if isinstance(thresh, np.generic):
            thresh = thresh.item()
        if type(val) not in (int, float) or type(thresh) not in (int, float):
            # Neither Python int nor float after unwrapping (e.g. a Decimal or
            # some other Real subclass) -- there is no verified-exact
            # comparison path for it here, so abstain rather than risk an
            # unaudited implicit conversion.
            return Tristate.UNKNOWN

        return Tristate.TRUE if op_fn(val, thresh) else Tristate.FALSE

    def _evaluate_proposition(
        self, sub_conditions: Sequence[Dict[str, Any]], metrics: Dict[str, Any]
    ) -> Tristate:
        """Evaluates a full AND-of-comparisons proposition, the single truth combinator
        shared by ``project_latent_propositions`` and ``evaluate_condition`` (C07).

        No sub_conditions (unsupported/unparseable syntax, e.g. an internal OR or
        NOT) -> UNKNOWN: never fabricated from the diagnostic W_sat basis. Any
        unresolved term -> UNKNOWN, matching evaluate_condition's existing
        any-unknown-wins collapse (deliberately more conservative than Kleene
        false-domination: an unrelated unresolvable term still abstains the whole
        conjunction rather than being short-circuited away by an already-False
        sibling).
        """
        if not sub_conditions:
            return Tristate.UNKNOWN
        outcomes = [self._evaluate_sub_condition(cond, metrics) for cond in sub_conditions]
        if any(outcome is Tristate.UNKNOWN for outcome in outcomes):
            return Tristate.UNKNOWN
        return Tristate.TRUE if all(outcome is Tristate.TRUE for outcome in outcomes) else Tristate.FALSE

    def evaluate_condition(
        self,
        rule: CompiledConstraintRule,
        z_latent: Optional[np.ndarray] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        schema_fingerprint: Optional[str] = None,
    ) -> bool:
        """Evaluates whether a rule's condition is currently ACTIVE (True).

        Applies strict Fail-Closed principle:
        - If condition has named variables and metrics is provided but variable is missing or NaN:
          FORBID_IF / REQUIRE_WHEN -> condition evaluates to True (Fail-Closed: forbid the dangerous action).
          ALLOW_ONLY_IF -> condition evaluates to False (Fail-Closed: disallowed because authorization unproven).
        - If sub_conditions (e.g. 'cond1 and cond2') are present, all must hold.
        - Tristate.UNKNOWN on any sub-condition (see ``_evaluate_sub_condition``) is
          never treated as FALSE: it collapses to the same fail-closed branch as a
          missing metric, per rule kind.

        X-C01: when ``current_metrics`` is explicitly supplied, it -- not
        ``z_latent`` -- is authoritative (a metrics dict is keyed by variable
        name and needs no coordinate map), so ``z_latent`` is only validated for
        NaN/Inf in that case and ``schema_fingerprint`` is not required. When
        ``current_metrics`` is None and ``z_latent`` is the sole source of truth,
        a matching ``schema_fingerprint`` is mandatory (``_bind_latent``).
        """
        metrics = current_metrics
        if z_latent is not None:
            if metrics is None:
                metrics = self._bind_latent(z_latent, schema_fingerprint)
            else:
                self._validate_latent(z_latent)
        conditions = rule.sub_conditions
        if not conditions:
            logger.error("Unverifiable constraint condition: %s", rule.condition_expr)
            return rule.rule_type in ("FORBID_IF", "REQUIRE_WHEN")
        outcome = self._evaluate_proposition(conditions, metrics or {})
        if outcome is Tristate.UNKNOWN:
            logger.warning("Unresolved constraint metrics; applying fail-closed rule %s", rule.rule_id)
            return rule.rule_type in ("FORBID_IF", "REQUIRE_WHEN")
        return outcome is Tristate.TRUE

    def solve_safest_action(
        self,
        candidate_utilities: Dict[str, float],
        z_latent: Optional[np.ndarray] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        fallback_safe_action: str = "HOLD",
        schema_fingerprint: Optional[str] = None,
    ) -> CPSATVerdict:
        """Run CP-SAT within a wall-time acceptance budget; reject late results.

        CP-SAT and the OS are not hard real-time: completion itself can be late.
        Such calls return an explicit unsafe timeout verdict, never a proved action.

        X-C01: when ``z_latent`` is the sole source of truth (``current_metrics``
        is None), ``schema_fingerprint`` must match ``compiler.schema_fingerprint``
        captured when the vector was produced, or this raises
        ``SchemaMismatchError`` (see ``_bind_latent``). When ``current_metrics``
        is supplied, it is authoritative and ``z_latent`` is only checked for
        NaN/Inf.
        """
        t0 = time.perf_counter()
        resolved_metrics = current_metrics
        if z_latent is not None:
            if resolved_metrics is None:
                resolved_metrics = self._bind_latent(z_latent, schema_fingerprint)
            else:
                self._validate_latent(z_latent)

        # Case-insensitive normalization of candidates
        norm_utilities: Dict[str, float] = {}
        orig_candidate_map: Dict[str, str] = {}
        for orig_act, util in candidate_utilities.items():
            norm_key = str(orig_act).strip().upper()
            if not norm_key or norm_key in norm_utilities:
                raise ValueError("Candidate action IDs must be nonempty and unique after normalization")
            if not math.isfinite(float(util)):
                raise ValueError("Candidate utilities must be finite")
            norm_utilities[norm_key] = float(util)
            if norm_key not in orig_candidate_map:
                orig_candidate_map[norm_key] = orig_act

        candidates = list(norm_utilities.keys())
        if not candidates:
            return CPSATVerdict(
                selected_action=fallback_safe_action,
                is_safe=False,
                solve_time_ms=0.0,
                timed_out=False,
                fallback_used=True,
                solver_status="NO_CANDIDATES",
                applied_constraints=[],
            )

        # Fail-Closed: Unparseable/unsupported syntax cannot be verified safe (S15)
        unsupported_rules = [r for r in self._rules
                             if r.rule_type == "UNSUPPORTED_SYNTAX"
                             or (r.rule_type != "MUTUAL_EXCLUSIVE" and not r.sub_conditions)]
        if unsupported_rules:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return CPSATVerdict(
                selected_action=fallback_safe_action,
                is_safe=False,
                solve_time_ms=elapsed_ms,
                timed_out=False,
                fallback_used=True,
                solver_status="UNSUPPORTED_SYNTAX_FAIL_CLOSED",
                applied_constraints=[f"{r.rule_id}: {r.raw_expression}" for r in unsupported_rules],
            )

        applied_constraints: List[str] = []
        forbidden_actions: Set[str] = set()
        required_actions: Set[str] = set()

        # Evaluate compiled rules
        for rule in self._rules:
            if rule.rule_type == "MUTUAL_EXCLUSIVE":
                continue

            # resolved_metrics was already bound once above (schema-checked if it
            # came from z_latent); no need to re-validate or re-check per rule.
            is_active = self.evaluate_condition(rule, current_metrics=resolved_metrics)
            norm_target = rule.target_action.strip().upper() if rule.target_action else None

            if is_active:
                applied_constraints.append(f"{rule.rule_id}: {rule.raw_expression}")
                if rule.rule_type == "FORBID_IF" and norm_target:
                    forbidden_actions.add(norm_target)
                elif rule.rule_type == "REQUIRE_WHEN" and norm_target:
                    required_actions.add(norm_target)
                elif rule.rule_type == "ALLOW_ONLY_IF" and norm_target:
                    pass  # Active condition means authorized
            else:
                if rule.rule_type == "ALLOW_ONLY_IF" and norm_target:
                    forbidden_actions.add(norm_target)
                    applied_constraints.append(f"{rule.rule_id} [ALLOW_VIOLATION]: {rule.raw_expression}")

        # Fail-Closed: Detect conflicting requirements in single-choice action space (S07)
        # 1. More than one distinct REQUIRED action cannot be satisfied simultaneously.
        # 2. An action is simultaneously REQUIRED and FORBIDDEN.
        conflicting_requirements = (len(required_actions) > 1) or bool(required_actions.intersection(forbidden_actions))
        if conflicting_requirements:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return CPSATVerdict(
                selected_action=fallback_safe_action,
                is_safe=False,
                solve_time_ms=elapsed_ms,
                timed_out=False,
                fallback_used=True,
                solver_status="CONFLICTING_REQUIREMENTS",
                applied_constraints=applied_constraints,
            )

        # 0-1 ILP Feasibility filtering
        feasible_candidates: Dict[str, float] = {}
        for act, util in norm_utilities.items():
            if act in forbidden_actions:
                continue
            if required_actions and act not in required_actions:
                continue
            feasible_candidates[act] = util

        def rejected(status: str, timed_out: bool = False) -> CPSATVerdict:
            logger.error("Constraint solve failed closed: %s", status)
            return CPSATVerdict(
                selected_action=fallback_safe_action,
                is_safe=False,
                solve_time_ms=(time.perf_counter() - t0) * 1000.0,
                timed_out=timed_out,
                fallback_used=True,
                solver_status=status,
                applied_constraints=applied_constraints,
            )

        if not feasible_candidates:
            if candidates and all(a in forbidden_actions for a in candidates):
                return rejected("ALL_FORBIDDEN_FAILSAFE")
            return rejected("UNSAFE_NO_FEASIBLE_ACTIONS")
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            return rejected("ORTOOLS_UNAVAILABLE_FALLBACK")
        try:
            model = cp_model.CpModel()
            # Eliminate fixed-zero variables and redundant bounds before CP-SAT.
            # REQUIRE constraints have already restricted this single-choice domain.
            actions = list(feasible_candidates)
            # Write the compact 0-1 model directly to its public protobuf. This avoids
            # allocating Python LinearExpr/IntVar wrappers on every decision.
            proto = model.Proto()
            for _ in actions:
                proto.variables.add().domain.extend((0, 1))
            proto.constraints.add().exactly_one.literals.extend(range(len(actions)))
            # Ordinal utility is equivalent for exactly-one choice, and preserves
            # distinctions that int(utility * 10000) used to truncate away.
            ranks = {u: i for i, u in enumerate(sorted(set(feasible_candidates.values())))}
            proto.objective.vars.extend(range(len(actions)))
            proto.objective.coeffs.extend(-ranks[feasible_candidates[a]] for a in actions)
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.cp_model_presolve = True
            solver.parameters.cp_model_probing_level = 0
            solver.parameters.symmetry_level = 0
            solver.parameters.linearization_level = 0
            remaining = self.hard_timeout_ms - (time.perf_counter() - t0) * 1000.0
            if remaining <= 0:
                return rejected("TIMEOUT_EARLY_FALLBACK", True)
            solver.parameters.max_time_in_seconds = remaining / 1000.0
            status = solver.Solve(model)
            total_ms = (time.perf_counter() - t0) * 1000.0
            if total_ms > self.hard_timeout_ms:
                return rejected("CP_SAT_DEADLINE_EXCEEDED", True)
            if status != cp_model.OPTIMAL:
                return rejected(f"CPSAT_NO_OPTIMUM:{solver.StatusName(status)}", status == cp_model.UNKNOWN)
            solution = solver.ResponseProto().solution
            chosen = next(a for i, a in enumerate(actions) if solution[i])
            total_ms = (time.perf_counter() - t0) * 1000.0
            if total_ms > self.hard_timeout_ms:
                return rejected("CP_SAT_DEADLINE_EXCEEDED", True)
            return CPSATVerdict(
                selected_action=orig_candidate_map[chosen],
                is_safe=True,
                solve_time_ms=total_ms,
                timed_out=False,
                fallback_used=False,
                solver_status="CP_SAT_OPTIMAL",
                applied_constraints=applied_constraints,
            )
        except Exception as exc:
            return rejected(f"CPSAT_EXCEPTION_FALLBACK:{type(exc).__name__}:{exc}")

    # ------------------------------------------------------------------
    # Structured action constraints (caller-supplied dicts; see gate/action_constraints.py)
    # ------------------------------------------------------------------
    def compile_action_constraints(self, specs: Any) -> int:
        """Validate and install structured action constraints. Replaces any earlier set.

        Raises ValueError on a malformed spec; nothing is installed in that case.
        """
        self._action_constraints = parse_action_constraints(specs)
        return len(self._action_constraints)

    def project_probabilities(self, probs: Dict[str, float]) -> ProjectionResult:
        """Zero the disabled actions and project ``probs`` onto the installed bounds (sum == 1)."""
        return project_distribution(self._action_constraints, probs)

    def resolve_disabled_actions(
        self, priorities: Dict[str, float], pre_disabled: Optional[Set[str]] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """Hard-mask half of the projection for callers that hold scores, not probabilities."""
        return resolve_disabled_actions(self._action_constraints, priorities, pre_disabled)

    def formal_verification_status(
        self,
        selected: Optional[str],
        candidates: Sequence[str],
        disabled: Sequence[str],
    ) -> str:
        """Truthful `formal_verification` label for a decision made under these constraints.

        no constraints -> not_requested; OR-Tools missing -> unavailable_missing_dependency
        (checked first, so the caller always learns the proof did not run); nothing selected
        -> not_applicable_no_action_selected; otherwise the real CP-SAT result.
        """
        if not self._action_constraints:
            return FV_NOT_REQUESTED
        if not cpsat_available():
            logger.warning("Constraints were supplied but OR-Tools is missing: %s", FV_UNAVAILABLE)
            return FV_UNAVAILABLE
        if selected is None or selected not in candidates:
            return FV_NO_ACTION
        return cpsat_verify_selection(self._action_constraints, candidates, disabled, selected)
