"""Service for evaluating retrieval relevance and evidence sufficiency (Phase 9.15).

Provides:
- RelevanceChecker: Deterministic evaluator assessing whether retrieved document
  evidence meets the configured relevance and sufficiency policies.
- check_retrieval_relevance: Convenience function for one-off relevance checking.

Guarantees:
- Distance Semantics: Operates strictly on DISTANCE metrics (lower = closer match).
  Relevant condition is chunk.distance <= max_distance.
- Safe Boundary: Detects empty or weak contexts and prevents downstream LLM fabrication.
- Partial Relevance: Filters weak chunks (distance > max_distance) and passes forward
  only relevant chunks in a derived unmutated RetrievalContext if min_relevant_chunks
  is satisfied.
- Immutability: The original RetrievalContext is never modified.
- Deterministic: 100% offline rule-based evaluation with zero external network or LLM.
"""

from typing import Optional

from app.core.logging import get_logger
from app.models.context import RetrievalContext
from app.models.relevance import (
    REASON_INSUFFICIENT_RELEVANT_CHUNKS,
    REASON_NO_EVIDENCE_WITHIN_THRESHOLD,
    REASON_NO_RETRIEVED_EVIDENCE,
    REASON_NOT_ALL_CHUNKS_RELEVANT,
    REASON_SUFFICIENT_EVIDENCE,
    RelevanceCheckResult,
    RelevanceConfig,
)

logger = get_logger("app.services.relevance_checker")


class RelevanceChecker:
    """Deterministic relevance and evidence sufficiency evaluator (Phase 9.15)."""

    def __init__(self, default_config: Optional[RelevanceConfig] = None) -> None:
        """Initialize RelevanceChecker with optional default policy configuration.

        Args:
            default_config: Default RelevanceConfig. If omitted, uses standard defaults.
        """
        self._default_config = default_config or RelevanceConfig()

    @property
    def default_config(self) -> RelevanceConfig:
        """Return the default RelevanceConfig instance."""
        return self._default_config

    def check_relevance(
        self,
        context: RetrievalContext,
        config: Optional[RelevanceConfig] = None,
    ) -> RelevanceCheckResult:
        """Evaluate whether RetrievalContext has sufficiently relevant evidence.

        Args:
            context: RetrievalContext produced by Phase 9.11 Context Construction.
            config: Optional RelevanceConfig overriding the default policy.

        Returns:
            RelevanceCheckResult: Machine-readable evaluation result containing
                                  decision, reasons, counts, best distance, and
                                  derived context.

        Raises:
            TypeError: If context is not an instance of RetrievalContext.
        """
        if not isinstance(context, RetrievalContext):
            raise TypeError(
                f"context must be an instance of RetrievalContext, "
                f"got {type(context).__name__}."
            )

        active_config = config or self._default_config
        threshold = active_config.max_distance
        min_chunks = active_config.min_relevant_chunks

        # 1. Empty context handling (no documents or zero retrieved chunks)
        if not context.has_evidence or context.is_empty or len(context.chunks) == 0:
            logger.info(
                "RetrievalContext is empty; returning insufficient evidence "
                "(reason=%s)",
                REASON_NO_RETRIEVED_EVIDENCE,
            )
            return RelevanceCheckResult(
                is_relevant=False,
                reason=REASON_NO_RETRIEVED_EVIDENCE,
                total_chunks=0,
                relevant_chunks_count=0,
                rejected_chunks_count=0,
                best_distance=None,
                threshold_used=threshold,
                min_chunks_required=min_chunks,
                relevant_chunks=[],
                rejected_chunks=[],
                derived_context=None,
            )

        # 2. Evaluate distance for all candidate chunks (lower distance = closer match)
        total_chunks = len(context.chunks)
        best_distance = min(c.distance for c in context.chunks)

        relevant_chunks = [c for c in context.chunks if c.distance <= threshold]
        rejected_chunks = [c for c in context.chunks if c.distance > threshold]

        relevant_count = len(relevant_chunks)
        rejected_count = len(rejected_chunks)

        # 3. Apply policy logic
        is_relevant = False
        reason = REASON_NO_EVIDENCE_WITHIN_THRESHOLD

        if active_config.require_all_relevant:
            if rejected_count > 0:
                is_relevant = False
                reason = REASON_NOT_ALL_CHUNKS_RELEVANT
            elif relevant_count < min_chunks:
                is_relevant = False
                reason = REASON_INSUFFICIENT_RELEVANT_CHUNKS
            else:
                is_relevant = True
                reason = REASON_SUFFICIENT_EVIDENCE
        else:
            if relevant_count == 0:
                is_relevant = False
                reason = REASON_NO_EVIDENCE_WITHIN_THRESHOLD
            elif relevant_count < min_chunks:
                is_relevant = False
                reason = REASON_INSUFFICIENT_RELEVANT_CHUNKS
            else:
                is_relevant = True
                reason = REASON_SUFFICIENT_EVIDENCE

        # 4. Build derived context if evidence is sufficient
        derived_context: Optional[RetrievalContext] = None
        if is_relevant:
            # Preserve original ranking order, text, and complete provenance
            derived_metadata = dict(context.retrieval_metadata)
            derived_metadata["relevance_filtered"] = rejected_count > 0
            derived_metadata["original_chunks_count"] = total_chunks
            derived_metadata["relevance_threshold_applied"] = threshold

            derived_context = RetrievalContext(
                query=context.query,
                collection_name=context.collection_name,
                distance_metric=context.distance_metric,
                total_retrieved=relevant_count,
                total_characters=sum(c.character_count for c in relevant_chunks),
                total_words=sum(c.word_count for c in relevant_chunks),
                chunks=relevant_chunks,
                has_evidence=True,
                max_distance_threshold=threshold,
                exceeds_budget=context.exceeds_budget,
                built_at=context.built_at,
                retrieval_metadata=derived_metadata,
            )

        logger.info(
            "Relevance check completed: is_relevant=%s, reason=%s, total=%d, "
            "relevant=%d, rejected=%d, best_distance=%.4f, threshold=%.4f",
            is_relevant,
            reason,
            total_chunks,
            relevant_count,
            rejected_count,
            best_distance,
            threshold,
        )

        return RelevanceCheckResult(
            is_relevant=is_relevant,
            reason=reason,
            total_chunks=total_chunks,
            relevant_chunks_count=relevant_count,
            rejected_chunks_count=rejected_count,
            best_distance=best_distance,
            threshold_used=threshold,
            min_chunks_required=min_chunks,
            relevant_chunks=relevant_chunks,
            rejected_chunks=rejected_chunks,
            derived_context=derived_context,
        )


def check_retrieval_relevance(
    context: RetrievalContext,
    config: Optional[RelevanceConfig] = None,
) -> RelevanceCheckResult:
    """Functional convenience entry point for checking retrieval relevance.

    Args:
        context: RetrievalContext instance from Phase 9.11.
        config: Optional RelevanceConfig policy override.

    Returns:
        RelevanceCheckResult: Comprehensive evaluation result.
    """
    checker = RelevanceChecker()
    return checker.check_relevance(context=context, config=config)
