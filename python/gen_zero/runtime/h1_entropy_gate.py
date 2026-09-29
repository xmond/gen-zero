"""H1 Entropy-Gated Adaptive Sampling Engine (Issue #32 & RFC-032).

Implements:
1. First-order Logits Shannon Entropy computation over Top-20 probabilities:
   H_1 = - sum_{i=1}^{K} p_i * ln(p_i)
2. Fast-Path (< 6.5ms): Single-pass early exit when max_j(H_1) <= tau_entropy (0.10).
3. Multi-Read Path (~18ms): Triggered when ambiguity arises (H_1 > 0.10) AND
   `perturbation_enabled=True`, firing K=4 genuine perturbation passes to produce
   empirical mean (mu) and standard deviation (sigma) error bars. When
   perturbation is not enabled, an ambiguous read is reported honestly as a
   single, variance-unmeasured sample instead of firing fabricated idle
   re-reads.
4. Downstream conservative safety integration with CP-SAT and Fail-Closed scheduler.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


def _coerce_probability_values(probabilities: Any) -> Tuple[Optional[List[float]], int]:
    """Best-effort extraction of a flat list of numeric probability values.

    Returns ``(values, sized_len)``:
    - ``values`` is ``None`` when ``probabilities`` is not a well-formed
      numeric probability container (wrong type, non-numeric entries, a bare
      string/bytes, a scalar, ...). This function never raises.
    - ``sized_len`` is ``len(probabilities)`` when the input supports
      ``len()``, else ``0``. Used for ``EntropyEvaluation.num_candidates`` on
      invalid input so the caller still knows roughly how large the rejected
      input was, without ever mistaking it for a valid read.
    """
    try:
        sized_len = len(probabilities)
    except TypeError:
        sized_len = 0

    # Strings/bytes are sized and iterable but must never be treated as a
    # sequence of probabilities: iterating a string yields its characters.
    if isinstance(probabilities, (str, bytes, bytearray)):
        return None, sized_len

    if isinstance(probabilities, dict):
        raw_iter: Any = probabilities.values()
    elif isinstance(probabilities, (list, tuple)):
        raw_iter = probabilities
    else:
        # Anything else not already covered (None, int, float, bool, an
        # arbitrary non-iterable object, ...) is not an accepted probability
        # container. Some other iterables (sets, generators, numpy arrays)
        # still work via this fallback.
        try:
            raw_iter = list(probabilities)
        except TypeError:
            return None, sized_len

    try:
        # OverflowError: a Python int too large for a C double (e.g. 10**1000)
        # raises this distinctly from TypeError/ValueError; it must be treated
        # as illegal input too, not left to crash the caller.
        values = [float(p) for p in raw_iter]
    except (TypeError, ValueError, OverflowError):
        return None, sized_len

    return values, len(values)


@dataclass
class EntropyEvaluation:
    """Quantitative entropy metrics for an evaluated probability distribution."""
    entropy: float
    top_prob: float
    is_ambiguous: bool
    threshold: float
    num_candidates: int
    # False for empty/all-zero/non-finite input: the gate must never mistake a
    # missing or degenerate distribution for a confident, low-entropy one.
    is_valid: bool = True


@dataclass
class MultiReadStatistics:
    """Empirical statistics computed across K perturbation passes.

    ``variance_measured`` distinguishes a genuine multi-sample std_dev
    (``n >= 2`` real perturbation reads) from a single-read stat where no
    variance was ever observed. For the latter, ``std_dev`` is ``math.nan``:
    a fabricated ``0.0`` would misrepresent "we never measured spread" as
    "we measured zero spread", which downstream conservative-bound math would
    silently treat as perfect confidence.
    """
    mean: float
    std_dev: float
    samples: List[float]
    reads_count: int
    variance_measured: bool = True

    @property
    def error_bar(self) -> str:
        if not self.variance_measured:
            return f"{round(self.mean, 4)} ± unmeasured"
        return f"{round(self.mean, 4)} ± {round(self.std_dev, 4)}"

    @property
    def two_sigma_margin(self) -> float:
        """Heuristic conservative margin: mu - 2 * sigma, clamped at 0.

        This is NOT a statistically rigorous confidence bound: with K as low as
        1-4 empirical reads there is no valid normal-approximation guarantee of
        95% coverage (that requires much larger n, or a small-sample correction
        such as a t-distribution). It is a downside-biased point estimate useful
        for conservative gating decisions, nothing more precise than that.

        Returns math.nan when variance was never measured (single-read stats):
        there is no sigma to subtract, and clamping a fabricated one to 0.0
        would report a margin as if spread had actually been observed.
        """
        if not self.variance_measured:
            return math.nan
        return max(0.0, self.mean - 2.0 * self.std_dev)


class H1EntropyGate:
    """Adaptive sampling scheduler utilizing first-order Logits entropy gating."""

    DEFAULT_THRESHOLD = 0.10  # RFC-032 recommended tau_entropy
    DEFAULT_MULTI_READ_K = 4  # 4 perturbation reads on ambiguous samples
    TOP_K_TRUNCATION = 20     # RFC-032 standard Top-20 Logprobs evaluation

    def __init__(
        self,
        tau_entropy: float = DEFAULT_THRESHOLD,
        multi_read_k: int = DEFAULT_MULTI_READ_K,
        perturbation_scale: float = 0.04,
        perturbation_enabled: bool = False,
    ):
        try:
            tau = float(tau_entropy)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"tau_entropy must be a finite float >= 0.0, got {tau_entropy!r}"
            ) from exc
        # Fail-closed: a negative or non-finite threshold would make every
        # (or no) read pass evaluate_distribution's `h1 > tau_entropy` gate,
        # silently disabling the ambiguity check instead of raising at setup.
        if not (math.isfinite(tau) and tau >= 0.0):
            raise ValueError(
                f"tau_entropy must be finite and >= 0.0, got {tau_entropy!r}"
            )
        self.tau_entropy = tau
        self.multi_read_k = int(multi_read_k)
        try:
            self.perturbation_scale = float(perturbation_scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"perturbation_scale must be a finite float, got {perturbation_scale!r}"
            ) from exc
        if not (math.isfinite(self.perturbation_scale) and self.perturbation_scale >= 0.0):
            raise ValueError(
                f"perturbation_scale must be finite and >= 0.0, got {perturbation_scale!r}"
            )
        # Fail-closed default: without an explicit opt-in, an ambiguous read
        # is reported honestly as unmeasured variance rather than firing K-1
        # extra forward_fn passes whose caller may not expect (or budget for)
        # the extra latency, and whose old fallback fabricated a std_dev.
        self.perturbation_enabled = bool(perturbation_enabled)

    @classmethod
    def compute_h1_entropy(
        cls,
        probabilities: Sequence[float],
        top_k: int = TOP_K_TRUNCATION
    ) -> float:
        """Computes standardized first-order Shannon entropy H_1 over top-K probabilities.

        Formula: H_1 = - sum_{i=1}^{K} p_i * ln(p_i)

        Fail-closed contract: an empty, all-zero, negative, or non-finite (NaN/Inf)
        distribution carries NO real information and must NEVER be reported as
        zero entropy (which downstream callers read as "maximally certain" and use
        to fast-path past the ambiguity gate). Such inputs return +inf instead, so
        they are unconditionally treated as ambiguous by evaluate_distribution
        regardless of tau_entropy.
        """
        values = list(probabilities)
        if not values:
            return math.inf

        try:
            floats = [float(p) for p in values]
        except (TypeError, ValueError, OverflowError):
            return math.inf

        # Silently dropping NaN/Inf/negative entries would hide corrupt upstream
        # data behind a plausible-looking entropy number; reject the whole
        # distribution instead.
        if any((not math.isfinite(p)) or p < 0.0 for p in floats):
            return math.inf

        # Sort and take top-K
        sorted_probs = sorted(floats, reverse=True)[:top_k]
        total_p = sum(sorted_probs)
        # A sum that overflows to inf (e.g. two 1e308 entries) is just as
        # uninformative as a non-positive sum: dividing by it below would
        # silently collapse every p_i to 0.0, producing a fake entropy of
        # exactly 0.0 (maximal fabricated certainty) instead of failing closed.
        if not math.isfinite(total_p) or total_p <= 0.0:
            return math.inf

        # Renormalize top-K to sum to 1.0 for true distribution entropy
        norm_probs = [p / total_p for p in sorted_probs]

        entropy = 0.0
        for p in norm_probs:
            if p > 1e-12:
                entropy -= p * math.log(p)

        if not math.isfinite(entropy):
            return math.inf

        return max(0.0, entropy)

    def evaluate_distribution(self, probabilities: Any) -> EntropyEvaluation:
        """Evaluates whether a probability distribution is ambiguous according to H_1 entropy.

        Fail-closed contract: illegal input (``None``, a bare string/bytes, a
        scalar, a non-iterable object, or a dict/sequence whose values are not
        coercible to ``float``) must never raise. It is reported as an
        invalid, maximally ambiguous distribution (``entropy=+inf``,
        ``is_valid=False``, ``top_prob=0.0``) instead of crashing the caller
        or being silently treated as a confident read.
        """
        raw_p, sized_len = _coerce_probability_values(probabilities)

        if raw_p is None:
            return EntropyEvaluation(
                entropy=math.inf,
                top_prob=0.0,
                is_ambiguous=True,
                threshold=self.tau_entropy,
                num_candidates=sized_len,
                is_valid=False,
            )

        h1 = self.compute_h1_entropy(raw_p, self.TOP_K_TRUNCATION)
        # h1 is +inf exactly when compute_h1_entropy rejected the distribution
        # (empty / all-zero / non-finite); that is never a valid, complete read.
        is_valid = math.isfinite(h1)
        # Only compute top_prob for a valid read: max() of a rejected
        # distribution is not a meaningful "top probability".
        top_prob = max(raw_p) if (is_valid and raw_p) else 0.0
        # inf > tau_entropy is always True, so invalid distributions are always
        # ambiguous regardless of the configured threshold (fail closed).
        is_ambiguous = (h1 > self.tau_entropy)

        return EntropyEvaluation(
            entropy=round(h1, 4) if is_valid else h1,
            top_prob=round(top_prob, 4) if is_valid else 0.0,
            is_ambiguous=is_ambiguous,
            threshold=self.tau_entropy,
            num_candidates=len(raw_p),
            is_valid=is_valid,
        )

    def compute_multi_read_statistics(self, sample_probabilities: Sequence[float]) -> MultiReadStatistics:
        """Computes empirical mean and Bessel-corrected sample standard deviation.

        Raises:
            ValueError: given zero samples, or a sample that is not a finite
                real number (including an int too large for a float, e.g.
                10**1000, which raises OverflowError). There is no statistic
                to report, and fabricating a mean=0.0/std_dev=0.0 result would
                misrepresent an absent/corrupt read as a measured one.
        """
        try:
            samples = [float(s) for s in sample_probabilities]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"compute_multi_read_statistics received a non-numeric or "
                f"unconvertible sample: {exc}"
            ) from exc
        for s in samples:
            if not math.isfinite(s):
                raise ValueError(
                    f"compute_multi_read_statistics received non-finite sample: {s}"
                )
        n = len(samples)
        if n == 0:
            raise ValueError("compute_multi_read_statistics requires at least one sample")
        if n == 1:
            # A single read has no measured spread: std_dev=nan and
            # variance_measured=False, never a fabricated std_dev=0.0.
            return MultiReadStatistics(
                mean=samples[0],
                std_dev=math.nan,
                samples=samples,
                reads_count=1,
                variance_measured=False,
            )

        mu = sum(samples) / n
        variance = sum((s - mu) ** 2 for s in samples) / (n - 1)
        sigma = math.sqrt(max(0.0, variance))

        return MultiReadStatistics(
            mean=round(mu, 4),
            std_dev=round(sigma, 4),
            samples=samples,
            reads_count=n,
            variance_measured=True,
        )

    def schedule_execution(
        self,
        forward_fn: Callable[[float], Dict[str, Any]],
        extract_slot_probs_fn: Callable[[Dict[str, Any]], Dict[str, Dict[str, float]]],
    ) -> Tuple[Dict[str, Any], bool, float, Dict[str, MultiReadStatistics]]:
        """Executes adaptive two-stage sampling schedule.

        Stage 2 (K perturbation passes) only fires when ``self.perturbation_enabled``
        is True. If it is False, an ambiguous read is reported as a single sample
        with ``variance_measured=False`` instead of firing extra forward_fn calls.

        Args:
            forward_fn: Callable taking perturbation scale (0.0 for initial pass) and returning raw output.
            extract_slot_probs_fn: Callable extracting per-slot probability dicts from raw output.

        Returns:
            Tuple of:
            - final_raw_output: dict
            - is_multi_read: bool
            - max_h1_entropy: float
            - slot_statistics: Dict[slot_name, MultiReadStatistics]
        """
        # ----------------------------------------------------
        # STAGE 1: Fast-Path Single Forward Pass (< 6.5ms)
        # ----------------------------------------------------
        first_output = forward_fn(0.0)
        slot_probs = extract_slot_probs_fn(first_output)

        max_h1 = 0.0
        is_any_ambiguous = False
        slot_evals: Dict[str, EntropyEvaluation] = {}

        for slot_name, probs in slot_probs.items():
            ev = self.evaluate_distribution(probs)
            slot_evals[slot_name] = ev
            if ev.entropy > max_h1:
                max_h1 = ev.entropy
            if ev.is_ambiguous:
                is_any_ambiguous = True

        # FAST-PATH EARLY EXIT (85%~90% of samples): a single confident read.
        # std_dev=0.0 would be a fabricated zero-variance claim for a
        # quantity that was never actually measured across repeated reads.
        if not is_any_ambiguous:
            single_stats: Dict[str, MultiReadStatistics] = {}
            for s_name, ev in slot_evals.items():
                single_stats[s_name] = MultiReadStatistics(
                    mean=ev.top_prob,
                    std_dev=math.nan,
                    samples=[ev.top_prob],
                    reads_count=1,
                    variance_measured=False,
                )
            return first_output, False, max_h1, single_stats

        # An ambiguous slot was found, but without perturbation_enabled there
        # is no genuine second read to measure variance from. Re-invoking
        # forward_fn here would be an idle re-read producing a fabricated
        # std_dev, exactly what this gate must not do. Report the single read
        # honestly instead.
        if not self.perturbation_enabled:
            single_stats = {}
            for s_name, ev in slot_evals.items():
                single_stats[s_name] = MultiReadStatistics(
                    mean=ev.top_prob,
                    std_dev=math.nan,
                    samples=[ev.top_prob],
                    reads_count=1,
                    variance_measured=False,
                )
            return first_output, False, max_h1, single_stats

        # ----------------------------------------------------
        # STAGE 2: Multi-Read Path (K=4 Perturbations, ~18ms)
        # Only entered when perturbation_enabled=True: these are genuine
        # perturbed forward_fn re-invocations, not idle re-reads.
        # ----------------------------------------------------
        slot_samples: Dict[str, List[float]] = {
            s_name: [slot_evals[s_name].top_prob] for s_name in slot_probs.keys()
        }
        # A perturbed pass that returns an illegal/non-finite distribution
        # taints that slot: its samples must never be handed to
        # compute_multi_read_statistics, which would silently compute a
        # NaN mean/std_dev while still claiming variance_measured=True.
        # A slot whose stage-1 (unperturbed) read was itself illegal/non-finite
        # starts tainted: its top_prob is a fabricated 0.0 placeholder (see
        # evaluate_distribution), not a real sample, so it must never seed
        # slot_samples as if it were legitimate.
        slot_tainted: Dict[str, bool] = {
            s_name: not slot_evals[s_name].is_valid for s_name in slot_probs.keys()
        }

        # Fire remaining K-1 perturbation passes
        for k_idx in range(1, self.multi_read_k):
            # Scale perturbation smoothly: e.g. +scale, -scale, +2*scale
            pert = self.perturbation_scale * (1.0 if k_idx % 2 == 1 else -1.0) * ((k_idx + 1) // 2)
            k_out = forward_fn(pert)
            k_slot_probs = extract_slot_probs_fn(k_out)
            for s_name in list(slot_samples.keys()):
                if s_name not in k_slot_probs:
                    slot_tainted[s_name] = True
                    continue
                p_dict = k_slot_probs[s_name]
                k_ev = self.evaluate_distribution(p_dict)
                if not k_ev.is_valid:
                    slot_tainted[s_name] = True
                    continue
                slot_samples[s_name].append(k_ev.top_prob)

        # Aggregate empirical statistics
        multi_stats: Dict[str, MultiReadStatistics] = {}
        for s_name, samples in slot_samples.items():
            if slot_tainted[s_name] or len(samples) < 2:
                # Honest report: either a perturbed read was illegal, or too
                # few legal samples survived to measure real spread. Never
                # wrap a tainted/short sample set as variance_measured=True.
                multi_stats[s_name] = MultiReadStatistics(
                    mean=samples[0],
                    std_dev=math.nan,
                    samples=samples,
                    reads_count=len(samples),
                    variance_measured=False,
                )
            else:
                multi_stats[s_name] = self.compute_multi_read_statistics(samples)

        return first_output, True, max_h1, multi_stats
