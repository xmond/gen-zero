"""Probabilistic Boolean Algebra Engine over Noul Model Predictions.

Evaluates complex propositional logic expressions over scalar probabilities in [0.0, 1.0].
Supports:
1. Fuzzy/Probabilistic Logic closures:
   - Zadeh / Gödel t-norm: NOT(A) = 1 - P(A), AND(A,B) = min(P(A), P(B)), OR(A,B) = max(P(A), P(B))
   - Product / Independent t-norm: AND(A,B) = P(A)*P(B), OR(A,B) = 1 - (1 - P(A))*(1 - P(B))
   - Lukasiewicz t-norm: AND(A,B) = max(0, A+B-1), OR(A,B) = min(1, A+B)
2. Infix Propositional Expression Parser:
   - Supports compounds: `(A AND B) OR NOT C`
   - Supports operators: `AND`, `&&`, `&`, `OR`, `||`, `|`, `NOT`, `!`, `~`
   - Supports quoted phrases: `"database deadlock" AND NOT 'connection timeout'`
"""

from enum import Enum
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class BooleanSemantics(str, Enum):
    ZADEH = "zadeh"            # min / max
    PRODUCT = "product"        # algebraic product / algebraic sum
    LUKASIEWICZ = "lukasiewicz"# bounded sum / bounded difference


def _sanitize_prob(p: float) -> float:
    """Clamps probability strictly to finite float in [0.0, 1.0]."""
    if not isinstance(p, (int, float)) or not math.isfinite(p):
        return 0.0
    return max(0.0, min(1.0, float(p)))


def p_not(p: float) -> float:
    """Probabilistic NOT: NOT(A) = 1 - P(A)."""
    return round(1.0 - _sanitize_prob(p), 6)


def p_and(p1: float, p2: float, semantics: Union[str, BooleanSemantics] = BooleanSemantics.ZADEH) -> float:
    """Probabilistic AND closure across semantics."""
    s1 = _sanitize_prob(p1)
    s2 = _sanitize_prob(p2)
    sem = BooleanSemantics(semantics) if isinstance(semantics, str) else semantics

    if sem == BooleanSemantics.ZADEH:
        res = min(s1, s2)
    elif sem == BooleanSemantics.PRODUCT:
        res = s1 * s2
    elif sem == BooleanSemantics.LUKASIEWICZ:
        res = max(0.0, s1 + s2 - 1.0)
    else:
        res = min(s1, s2)
    return round(res, 6)


def p_or(p1: float, p2: float, semantics: Union[str, BooleanSemantics] = BooleanSemantics.ZADEH) -> float:
    """Probabilistic OR closure across semantics."""
    s1 = _sanitize_prob(p1)
    s2 = _sanitize_prob(p2)
    sem = BooleanSemantics(semantics) if isinstance(semantics, str) else semantics

    if sem == BooleanSemantics.ZADEH:
        res = max(s1, s2)
    elif sem == BooleanSemantics.PRODUCT:
        res = 1.0 - (1.0 - s1) * (1.0 - s2)
    elif sem == BooleanSemantics.LUKASIEWICZ:
        res = min(1.0, s1 + s2)
    else:
        res = max(s1, s2)
    return round(res, 6)


# ---------------------------------------------------------
# AST Definitions
# ---------------------------------------------------------

class BooleanNode:
    """Base class for AST expression nodes."""
    def evaluate(self, probs: Dict[str, float], semantics: BooleanSemantics = BooleanSemantics.ZADEH) -> float:
        raise NotImplementedError

    def get_variables(self) -> List[str]:
        raise NotImplementedError


class LiteralNode(BooleanNode):
    """Leaf node representing a proposition / variable name."""
    def __init__(self, name: str):
        self.name = name.strip()

    def evaluate(self, probs: Dict[str, float], semantics: BooleanSemantics = BooleanSemantics.ZADEH) -> float:
        # 1. Exact match
        if self.name in probs:
            return _sanitize_prob(probs[self.name])
        # 2. Case-insensitive match
        name_lower = self.name.lower()
        for k, v in probs.items():
            if k.lower() == name_lower:
                return _sanitize_prob(v)
        # 3. Strip quotes match
        stripped = self.name.strip("'\"")
        if stripped in probs:
            return _sanitize_prob(probs[stripped])
        for k, v in probs.items():
            if k.strip("'\"").lower() == stripped.lower():
                return _sanitize_prob(v)
        return 0.0

    def get_variables(self) -> List[str]:
        return [self.name]

    def __repr__(self) -> str:
        return f"Literal({self.name!r})"


class NotNode(BooleanNode):
    """Unary NOT node."""
    def __init__(self, child: BooleanNode):
        self.child = child

    def evaluate(self, probs: Dict[str, float], semantics: BooleanSemantics = BooleanSemantics.ZADEH) -> float:
        return p_not(self.child.evaluate(probs, semantics))

    def get_variables(self) -> List[str]:
        return self.child.get_variables()

    def __repr__(self) -> str:
        return f"NOT({self.child})"


class AndNode(BooleanNode):
    """Binary/N-ary AND node."""
    def __init__(self, children: List[BooleanNode]):
        self.children = children

    def evaluate(self, probs: Dict[str, float], semantics: BooleanSemantics = BooleanSemantics.ZADEH) -> float:
        if not self.children:
            return 1.0
        acc = self.children[0].evaluate(probs, semantics)
        for child in self.children[1:]:
            acc = p_and(acc, child.evaluate(probs, semantics), semantics)
        return acc

    def get_variables(self) -> List[str]:
        vars_: List[str] = []
        for c in self.children:
            vars_.extend(c.get_variables())
        return list(dict.fromkeys(vars_))

    def __repr__(self) -> str:
        return f"AND({', '.join(str(c) for c in self.children)})"


class OrNode(BooleanNode):
    """Binary/N-ary OR node."""
    def __init__(self, children: List[BooleanNode]):
        self.children = children

    def evaluate(self, probs: Dict[str, float], semantics: BooleanSemantics = BooleanSemantics.ZADEH) -> float:
        if not self.children:
            return 0.0
        acc = self.children[0].evaluate(probs, semantics)
        for child in self.children[1:]:
            acc = p_or(acc, child.evaluate(probs, semantics), semantics)
        return acc

    def get_variables(self) -> List[str]:
        vars_: List[str] = []
        for c in self.children:
            vars_.extend(c.get_variables())
        return list(dict.fromkeys(vars_))

    def __repr__(self) -> str:
        return f"OR({', '.join(str(c) for c in self.children)})"


# ---------------------------------------------------------
# Tokenizer & Recursive Descent Parser
# ---------------------------------------------------------

TOKEN_REGEX = re.compile(
    r'\s*(?:'
    r'(\()|'                           # 1: LPAREN
    r'(\))|'                           # 2: RPAREN
    r'(\bAND\b|\&\&|\&)|'              # 3: AND
    r'(\bOR\b|\|\||\|)|'               # 4: OR
    r'(\bNOT\b|\!|\~)|'                # 5: NOT
    r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|' # 6: Quoted string
    r'([^\s\(\)\&\|\!\~\'\"]+)'         # 7: Identifier/word
    r')',
    re.IGNORECASE
)


class BooleanEngine:
    """Engine to parse and evaluate probabilistic Boolean expressions."""

    def __init__(self, default_semantics: BooleanSemantics = BooleanSemantics.ZADEH):
        self.default_semantics = default_semantics

    def tokenize(self, expression: str) -> List[Tuple[str, str]]:
        """Tokenizes expression into list of (type, value) tuples."""
        tokens: List[Tuple[str, str]] = []
        pos = 0
        while pos < len(expression):
            m = TOKEN_REGEX.match(expression, pos)
            if not m:
                # Skip unknown character
                pos += 1
                continue
            lparen, rparen, op_and, op_or, op_not, quoted, ident = m.groups()
            if lparen:
                tokens.append(("LPAREN", "("))
            elif rparen:
                tokens.append(("RPAREN", ")"))
            elif op_and:
                tokens.append(("AND", "AND"))
            elif op_or:
                tokens.append(("OR", "OR"))
            elif op_not:
                tokens.append(("NOT", "NOT"))
            elif quoted:
                # Remove surrounding quotes and unescape internal quotes
                unescaped = quoted[1:-1].replace('\\"', '"').replace("\\'", "'")
                tokens.append(("LITERAL", unescaped))
            elif ident:
                tokens.append(("LITERAL", ident))
            pos = m.end()
        return tokens

    def parse(self, expression: str) -> BooleanNode:
        """Parses infix Boolean expression string into AST."""
        tokens = self.tokenize(expression.strip())
        if not tokens:
            raise ValueError(f"Empty Boolean expression: {expression!r}")

        parser = _Parser(tokens)
        node = parser.parse_or()
        if parser.current_token() is not None:
            raise ValueError(f"Unexpected token '{parser.current_token()}' in expression: {expression!r}")
        return node

    def evaluate(
        self,
        expression: Union[str, BooleanNode],
        probabilities: Dict[str, float],
        semantics: Optional[Union[str, BooleanSemantics]] = None
    ) -> float:
        """Evaluates expression against probability mapping."""
        sem = (
            BooleanSemantics(semantics) if isinstance(semantics, str)
            else semantics or self.default_semantics
        )
        if isinstance(expression, str):
            ast = self.parse(expression)
        else:
            ast = expression
        return ast.evaluate(probabilities, semantics=sem)

    def evaluate_batch(
        self,
        expression: Union[str, BooleanNode],
        batch_probabilities: List[Dict[str, float]],
        semantics: Optional[Union[str, BooleanSemantics]] = None
    ) -> List[float]:
        """Evaluates expression across a list of probability dictionaries."""
        if isinstance(expression, str):
            ast = self.parse(expression)
        else:
            ast = expression
        sem = (
            BooleanSemantics(semantics) if isinstance(semantics, str)
            else semantics or self.default_semantics
        )
        return [ast.evaluate(p, semantics=sem) for p in batch_probabilities]

    @staticmethod
    def build_expression_from_patterns(
        or_patterns: Optional[List[str]] = None,
        and_patterns: Optional[List[str]] = None,
        not_patterns: Optional[List[str]] = None
    ) -> str:
        """Builds normalized compound logic expression from CLI pattern lists.

        - or_patterns (-e): at least one must match (OR group)
        - and_patterns (-a): all must match (AND group)
        - not_patterns (-v): none must match (AND NOT group)
        """
        parts: List[str] = []

        def _escape_pattern(pat: str) -> str:
            return pat.replace('"', '\\"')

        if or_patterns:
            clean_or = [f'"{_escape_pattern(p)}"' for p in or_patterns if p]
            if len(clean_or) == 1:
                parts.append(clean_or[0])
            elif len(clean_or) > 1:
                parts.append(f"({' OR '.join(clean_or)})")

        if and_patterns:
            clean_and = [f'"{_escape_pattern(p)}"' for p in and_patterns if p]
            for p in clean_and:
                parts.append(p)

        if not_patterns:
            clean_not = [f'NOT "{_escape_pattern(p)}"' for p in not_patterns if p]
            for p in clean_not:
                parts.append(p)

        if not parts:
            return '""'
        return " AND ".join(parts)


class _Parser:
    """Recursive descent parser with precedence: OR < AND < NOT < Literal/Parentheses."""

    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Optional[Tuple[str, str]]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: Optional[str] = None) -> Tuple[str, str]:
        tok = self.current_token()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if expected_type is not None and tok[0] != expected_type:
            raise ValueError(f"Expected token type {expected_type}, got {tok[0]} ('{tok[1]}')")
        self.pos += 1
        return tok

    def parse_or(self) -> BooleanNode:
        """Parse OR terms: Term (OR Term)*"""
        nodes = [self.parse_and()]
        while True:
            tok = self.current_token()
            if tok and tok[0] == "OR":
                self.consume("OR")
                nodes.append(self.parse_and())
            else:
                break
        if len(nodes) == 1:
            return nodes[0]
        return OrNode(nodes)

    def parse_and(self) -> BooleanNode:
        """Parse AND terms: Factor (AND Factor)*"""
        nodes = [self.parse_not()]
        while True:
            tok = self.current_token()
            if tok and tok[0] == "AND":
                self.consume("AND")
                nodes.append(self.parse_not())
            else:
                break
        if len(nodes) == 1:
            return nodes[0]
        return AndNode(nodes)

    def parse_not(self) -> BooleanNode:
        """Parse NOT factor: NOT Factor | Primary"""
        tok = self.current_token()
        if tok and tok[0] == "NOT":
            self.consume("NOT")
            child = self.parse_not()
            return NotNode(child)
        return self.parse_primary()

    def parse_primary(self) -> BooleanNode:
        """Parse Primary: ( Expr ) | Literal"""
        tok = self.current_token()
        if tok is None:
            raise ValueError("Unexpected end of expression while expecting primary token")

        if tok[0] == "LPAREN":
            self.consume("LPAREN")
            expr = self.parse_or()
            self.consume("RPAREN")
            return expr
        elif tok[0] == "LITERAL":
            self.consume("LITERAL")
            return LiteralNode(tok[1])
        else:
            raise ValueError(f"Unexpected token in expression: {tok[0]} ('{tok[1]}')")
