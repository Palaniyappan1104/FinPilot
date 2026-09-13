"""Source/Evidence Tracking Service (Phase 9.13).

Provides:
- EvidenceTracker: Provider-agnostic evidence tracking service managing
  verified provenance, citation resolution, and multi-document registries.
- track_evidence: Functional helper to instantiate and register an EvidenceTracker.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

from app.core.logging import get_logger
from app.models.context import ContextChunk, RetrievalContext
from app.models.evidence import (
    ConflictingProvenanceError,
    EvidenceNotFoundError,
    EvidenceValidationError,
    FabricatedProvenanceError,
    ResearchEvidence,
    compute_evidence_id,
)

if TYPE_CHECKING:
    from app.agents.research_schema import ResearchAnalysisOutput, ResearchEvidenceRef

logger = get_logger("app.services.evidence_tracker")


class EvidenceTracker:
    """Provider-agnostic registry and tracker for RAG source evidence.

    Phase 9.13: Guarantees that every finding and citation maps deterministically
    to its originating document, chunk, page numbers, and vector metadata.

    Responsibilities:
    - Register evidence from a RetrievalContext (Phase 9.11).
    - Expose deterministic evidence IDs ('ev_{doc_id}_{chunk_id}').
    - Resolve citations by evidence_id or chunk_id.
    - Deterministically verify citations against registered provenance.
    - Preserve ranking/order of retrieved evidence.
    - Support multi-document contexts with document-scoped indexing.
    - Deduplicate identical chunks while rejecting conflicting metadata.
    - Ensure zero fabricated provenance and safe operational logging.
    """

    def __init__(self, context: Optional[RetrievalContext] = None) -> None:
        """Initialize EvidenceTracker with an optional RetrievalContext.

        Args:
            context: Optional RetrievalContext to register immediately.
        """
        self._by_evidence_id: Dict[str, ResearchEvidence] = {}
        self._by_chunk_id: Dict[str, ResearchEvidence] = {}
        self._ordered_evidence: List[ResearchEvidence] = []
        self._by_document_id: Dict[str, List[ResearchEvidence]] = defaultdict(list)
        self._context: Optional[RetrievalContext] = None

        if context is not None:
            self.register_context(context)

    # =======================================================================
    # REGISTRATION
    # =======================================================================

    def register_context(self, context: RetrievalContext) -> List[ResearchEvidence]:
        """Register all evidence chunks from a structured RetrievalContext.

        Preserves retrieval ordering while deduplicating identical chunks and
        validating provenance integrity.

        Args:
            context: Structured RetrievalContext from Phase 9.11.

        Returns:
            List of registered ResearchEvidence objects in retrieval order.

        Raises:
            EvidenceValidationError: If context is invalid or chunk lacks provenance.
            ConflictingProvenanceError: If identical chunk has conflicting metadata.
        """
        if not isinstance(context, RetrievalContext):
            raise EvidenceValidationError(
                f"Expected RetrievalContext instance, got {type(context).__name__}."
            )

        self._context = context

        if context.is_empty or not context.chunks:
            logger.info("Registered empty RetrievalContext with 0 evidence chunks.")
            return []

        registered_this_batch: List[ResearchEvidence] = []

        for chunk in context.chunks:
            evidence = self._register_chunk(chunk)
            if evidence is not None:
                registered_this_batch.append(evidence)

        logger.info(
            "Evidence registration complete: %d total evidence items across "
            "%d distinct documents (collection='%s').",
            len(self._ordered_evidence),
            len(self._by_document_id),
            context.collection_name,
        )

        return self._ordered_evidence

    def _register_chunk(self, chunk: ContextChunk) -> Optional[ResearchEvidence]:
        """Validate and index a single ContextChunk.

        Returns the registered ResearchEvidence if newly added, or None if deduplicated.
        """
        if not isinstance(chunk, ContextChunk):
            raise EvidenceValidationError(
                f"Expected ContextChunk, got {type(chunk).__name__}."
            )

        # Enforce mandatory document_id provenance
        document_id = chunk.document_id
        if not document_id or not document_id.strip():
            raise EvidenceValidationError(
                f"Chunk '{chunk.chunk_id}' lacks mandatory document_id provenance."
            )
        document_id = document_id.strip()

        # Enforce mandatory source_document provenance
        source_doc = chunk.source or (
            chunk.metadata.get("source_document") if chunk.metadata else None
        )
        if not source_doc or not str(source_doc).strip():
            raise EvidenceValidationError(
                f"Chunk '{chunk.chunk_id}' lacks mandatory source_document provenance."
            )
        source_document = str(source_doc).strip()

        chunk_id = chunk.chunk_id.strip()
        evidence_id = compute_evidence_id(document_id, chunk_id)

        # Check for existing chunk registration
        if chunk_id in self._by_chunk_id:
            existing = self._by_chunk_id[chunk_id]
            # Verify conflicting provenance
            if (
                existing.document_id != document_id
                or existing.source_document != source_document
            ):
                raise ConflictingProvenanceError(
                    f"Conflicting provenance detected for chunk '{chunk_id}': "
                    f"existing doc '{existing.document_id}' "
                    f"(source '{existing.source_document}') vs new doc "
                    f"'{document_id}' (source '{source_document}')."
                )
            # Identical chunk repeated in retrieval pipeline -> deduplicate gracefully
            logger.debug(
                "Deduplicated repeated chunk '%s' (evidence_id='%s').",
                chunk_id,
                evidence_id,
            )
            return None

        # Build clean vector metadata excluding raw document text
        clean_meta = {
            k: v
            for k, v in chunk.metadata.items()
            if k not in ("text", "document_text", "raw_text")
        }

        evidence = ResearchEvidence(
            evidence_id=evidence_id,
            chunk_id=chunk_id,
            document_id=document_id,
            source_document=source_document,
            ticker=chunk.ticker,
            document_type=chunk.document_type,
            page_numbers=list(chunk.pages) if chunk.pages else [],
            section=chunk.section,
            chunk_index=chunk.chunk_index,
            distance=chunk.distance,
            metadata=clean_meta,
        )

        self._by_evidence_id[evidence_id] = evidence
        self._by_chunk_id[chunk_id] = evidence
        self._ordered_evidence.append(evidence)
        self._by_document_id[document_id].append(evidence)

        return evidence

    # =======================================================================
    # RESOLUTION
    # =======================================================================

    def resolve_by_evidence_id(self, evidence_id: str) -> ResearchEvidence:
        """Resolve a ResearchEvidence record by its deterministic evidence_id.

        Args:
            evidence_id: Evidence identifier (e.g. 'ev_doc123_chunk45').

        Returns:
            Matching ResearchEvidence record.

        Raises:
            EvidenceNotFoundError: If evidence_id is not registered.
        """
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise EvidenceValidationError("evidence_id must be a non-empty string.")

        eid = evidence_id.strip()
        if eid not in self._by_evidence_id:
            raise EvidenceNotFoundError(
                f"Evidence ID '{eid}' is not present in the evidence registry."
            )
        return self._by_evidence_id[eid]

    def resolve_by_chunk_id(self, chunk_id: str) -> ResearchEvidence:
        """Resolve a ResearchEvidence record by its chunk_id.

        Args:
            chunk_id: Chunk identifier.

        Returns:
            Matching ResearchEvidence record.

        Raises:
            EvidenceNotFoundError: If chunk_id is not registered.
        """
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise EvidenceValidationError("chunk_id must be a non-empty string.")

        cid = chunk_id.strip()
        if cid not in self._by_chunk_id:
            raise EvidenceNotFoundError(
                f"Chunk ID '{cid}' is not present in the evidence registry."
            )
        return self._by_chunk_id[cid]

    def resolve(self, ref: Union[str, ResearchEvidenceRef]) -> ResearchEvidence:
        """Flexible resolver supporting evidence_id, chunk_id, or ResearchEvidenceRef.

        Args:
            ref: Identifier string (evidence_id or chunk_id) or ResearchEvidenceRef.

        Returns:
            Resolved ResearchEvidence record.

        Raises:
            EvidenceNotFoundError: If reference cannot be found.
            EvidenceValidationError: If reference is invalid.
        """
        if isinstance(ref, str):
            clean_ref = ref.strip()
            if clean_ref in self._by_evidence_id:
                return self._by_evidence_id[clean_ref]
            if clean_ref in self._by_chunk_id:
                return self._by_chunk_id[clean_ref]
            raise EvidenceNotFoundError(
                f"Reference '{clean_ref}' could not be resolved as an "
                "evidence_id or chunk_id."
            )

        if hasattr(ref, "chunk_id") and hasattr(ref, "document_id"):
            return self.verify_citation(ref)

        raise EvidenceValidationError(
            f"Unsupported citation reference type: {type(ref).__name__}."
        )

    # =======================================================================
    # CITATION VERIFICATION
    # =======================================================================

    def verify_citation(
        self, ref: Union[Any, "ResearchEvidenceRef"]
    ) -> ResearchEvidence:
        """Verify that a cited ResearchEvidenceRef matches registered provenance.

        Args:
            ref: Citation reference from ResearchFinding or ResearchAnalysisOutput.

        Returns:
            Matching verified ResearchEvidence record.

        Raises:
            FabricatedProvenanceError: If chunk_id does not exist in registry.
            ConflictingProvenanceError: If document_id or source_document mismatches.
            EvidenceValidationError: If cited page numbers are invalid.
        """
        if not hasattr(ref, "chunk_id") or not hasattr(ref, "document_id"):
            raise EvidenceValidationError(
                f"Expected citation reference with chunk_id and document_id, "
                f"got {type(ref).__name__}."
            )

        chunk_id = ref.chunk_id.strip()
        if chunk_id not in self._by_chunk_id:
            raise FabricatedProvenanceError(
                f"Cited chunk_id '{chunk_id}' does not exist in evidence registry."
            )

        target = self._by_chunk_id[chunk_id]

        # Verify document_id correspondence
        if ref.document_id.strip() != target.document_id:
            raise ConflictingProvenanceError(
                f"Cited document_id '{ref.document_id}' does not match registered "
                f"document_id '{target.document_id}' for chunk '{chunk_id}'."
            )

        # Verify source_document correspondence
        if ref.source_document.strip() != target.source_document:
            raise ConflictingProvenanceError(
                f"Cited source_document '{ref.source_document}' does not match "
                f"registered source_document '{target.source_document}' for "
                f"chunk '{chunk_id}'."
            )

        # Verify page numbers consistency
        if target.page_numbers and ref.page_numbers:
            for page in ref.page_numbers:
                if page not in target.page_numbers:
                    raise EvidenceValidationError(
                        f"Cited page {page} for chunk '{chunk_id}' is not present "
                        f"in registered pages {target.page_numbers}."
                    )

        return target

    def verify_analysis_output(
        self, output: ResearchAnalysisOutput
    ) -> List[ResearchEvidence]:
        """Verify all citations in a ResearchAnalysisOutput against this registry.

        Args:
            output: ResearchAnalysisOutput from ResearchAnalystAgent.

        Returns:
            List of verified ResearchEvidence records matching cited references.
        """
        if output.insufficient_evidence:
            return []

        verified: List[ResearchEvidence] = []
        seen_chunks: Set[str] = set()

        # Verify evidence references
        for ref in output.evidence:
            evidence = self.verify_citation(ref)
            if evidence.chunk_id not in seen_chunks:
                seen_chunks.add(evidence.chunk_id)
                verified.append(evidence)

        # Verify finding references
        for finding in output.key_findings:
            if not finding.evidence:
                raise EvidenceValidationError(
                    f"Finding '{finding.claim}' has no supporting evidence citations."
                )
            for ref in finding.evidence:
                evidence = self.verify_citation(ref)
                if evidence.chunk_id not in seen_chunks:
                    seen_chunks.add(evidence.chunk_id)
                    verified.append(evidence)

        return verified

    # =======================================================================
    # MULTI-DOCUMENT AND REGISTRY INSPECTION
    # =======================================================================

    def has_chunk(self, chunk_id: str) -> bool:
        """Return True if chunk_id is registered."""
        return isinstance(chunk_id, str) and chunk_id.strip() in self._by_chunk_id

    def has_evidence_id(self, evidence_id: str) -> bool:
        """Return True if evidence_id is registered."""
        return (
            isinstance(evidence_id, str) and evidence_id.strip() in self._by_evidence_id
        )

    def get_document_evidence(self, document_id: str) -> List[ResearchEvidence]:
        """Return all registered evidence items associated with a document_id."""
        if not isinstance(document_id, str):
            return []
        return list(self._by_document_id.get(document_id.strip(), []))

    def get_all_evidence(self) -> List[ResearchEvidence]:
        """Return all registered evidence items in preserved retrieval order."""
        return list(self._ordered_evidence)

    @property
    def total_evidence(self) -> int:
        """Return the count of distinct registered evidence items."""
        return len(self._ordered_evidence)

    @property
    def total_documents(self) -> int:
        """Return the count of distinct documents in the registry."""
        return len(self._by_document_id)

    @property
    def document_ids(self) -> List[str]:
        """Return the list of distinct document IDs present in the registry."""
        return list(self._by_document_id.keys())

    @property
    def is_empty(self) -> bool:
        """Return True if no evidence items are registered."""
        return len(self._ordered_evidence) == 0


def track_evidence(context: RetrievalContext) -> EvidenceTracker:
    """Construct and register an EvidenceTracker from a RetrievalContext.

    Args:
        context: Structured RetrievalContext from Phase 9.11.

    Returns:
        Fully initialized EvidenceTracker instance.
    """
    return EvidenceTracker(context=context)
