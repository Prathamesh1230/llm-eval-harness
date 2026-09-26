"""Retrieval metrics. Deliberately hand-implemented — no RAGAS.

These need no LLM calls, so they run in milliseconds and are fully
deterministic, unlike anything the judge produces.

Relevance can be declared at two levels:

  file level   "hr_policy.pdf"        any chunk from that file counts
  chunk level  "hr_policy.pdf::c3"    only that exact chunk counts

File level is the default because it's the only thing a user uploading a
PDF can reasonably know, and because it's the only way a comparison across
chunk sizes is valid: at chunk_size 400 a fact may sit in ::c1 while at 800
the same fact sits in ::c0. Chunk-level matching would score the smaller
chunks as a miss for finding the *more precise* passage.
"""

from statistics import median

from evalharness.schemas import CaseResult, RetrievalMetrics, RetrievedChunk

CHUNK_SEP = "::"


def source_of(doc_id: str) -> str:
    """'hr_policy.pdf::c3' -> 'hr_policy.pdf'. Plain filenames pass through."""
    return doc_id.split(CHUNK_SEP, 1)[0]


def is_relevant(doc_id: str, relevant_ids: set[str]) -> bool:
    """True if this retrieved chunk satisfies any declared relevance.

    Exact match wins first, then file-level: a declared 'hr_policy.pdf'
    is satisfied by 'hr_policy.pdf::c0', '::c1', or any other chunk of it.
    """
    if doc_id in relevant_ids:
        return True
    return source_of(doc_id) in relevant_ids


def score_retrieval(
    retrieved: list[RetrievedChunk],
    relevant_doc_ids: list[str],
    k: int = 5,
) -> RetrievalMetrics:
    """Compute hit@k, reciprocal rank, and precision@k for ONE question.

    hit@k             — did any relevant chunk make the top k? (binary)
    reciprocal_rank   — 1/position of the first relevant chunk, else 0.
                        Rank 1 -> 1.0, rank 2 -> 0.5, rank 4 -> 0.25.
                        Averaged across questions this becomes MRR.
    precision@k       — what fraction of the returned k were relevant?
    """
    if not relevant_doc_ids:
        # No ground truth for this case — don't punish the pipeline.
        return RetrievalMetrics(hit=True, reciprocal_rank=1.0, precision_at_k=1.0)

    top_k = sorted(retrieved, key=lambda c: c.rank)[:k]
    relevant = set(relevant_doc_ids)

    hit = any(is_relevant(c.doc_id, relevant) for c in top_k)

    rr = 0.0
    for c in top_k:
        if is_relevant(c.doc_id, relevant):
            rr = 1.0 / c.rank      # rank is 1-indexed — this is why that mattered
            break                  # only the FIRST relevant hit counts

    n_relevant_retrieved = sum(1 for c in top_k if is_relevant(c.doc_id, relevant))
    precision = n_relevant_retrieved / len(top_k) if top_k else 0.0

    return RetrievalMetrics(
        hit=hit,
        reciprocal_rank=rr,
        precision_at_k=precision,
    )


def aggregate_retrieval(results: list[CaseResult]) -> dict[str, float]:
    """Roll per-case retrieval metrics up to run level."""
    if not results:
        return {"hit_rate": 0.0, "mrr": 0.0, "precision_at_k": 0.0}

    n = len(results)
    return {
        "hit_rate": sum(r.retrieval.hit for r in results) / n,
        "mrr": sum(r.retrieval.reciprocal_rank for r in results) / n,
        "precision_at_k": sum(r.retrieval.precision_at_k for r in results) / n,
    }


def latency_percentiles(results: list[CaseResult]) -> tuple[float, float]:
    """Return (p50, p95) latency in ms.

    p95 matters more than the mean — one very slow query is a real user
    problem that an average would hide.
    """
    if not results:
        return 0.0, 0.0
    lat = sorted(r.latency_ms for r in results)
    p50 = median(lat)
    idx = min(int(0.95 * len(lat)), len(lat) - 1)
    return p50, lat[idx]


if __name__ == "__main__":
    passed = failed = 0

    def check(label: str, condition: bool) -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  OK    {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}")

    def chunk(doc_id: str, rank: int) -> RetrievedChunk:
        return RetrievedChunk(doc_id=doc_id, text="", score=0.0, rank=rank)

    print("Chunk-level matching:")
    m = score_retrieval([chunk("a.txt::c0", 1), chunk("b.txt::c0", 2)], ["a.txt::c0"])
    check("rank 1 -> rr 1.00", m.hit and m.reciprocal_rank == 1.0)

    m = score_retrieval(
        [chunk("x.txt::c0", 1), chunk("y.txt::c0", 2), chunk("a.txt::c0", 3)], ["a.txt::c0"]
    )
    check("rank 3 -> rr 0.33", m.hit and round(m.reciprocal_rank, 2) == 0.33)

    m = score_retrieval([chunk("x.txt::c0", 1)], ["a.txt::c0"])
    check("missing -> rr 0.00", not m.hit and m.reciprocal_rank == 0.0)

    m = score_retrieval([chunk("a.txt::c1", 1)], ["a.txt::c0"])
    check("wrong chunk of right file -> miss", not m.hit)

    print("\nFile-level matching:")
    m = score_retrieval([chunk("a.txt::c0", 1)], ["a.txt"])
    check("c0 satisfies file", m.hit and m.reciprocal_rank == 1.0)

    m = score_retrieval([chunk("a.txt::c7", 1)], ["a.txt"])
    check("any chunk satisfies file", m.hit and m.reciprocal_rank == 1.0)

    m = score_retrieval([chunk("b.txt::c0", 1), chunk("a.txt::c3", 2)], ["a.txt"])
    check("rank 2 -> rr 0.50", m.hit and m.reciprocal_rank == 0.5)

    m = score_retrieval([chunk("b.txt::c0", 1)], ["a.txt"])
    check("other file -> miss", not m.hit)

    print("\nThe chunk-size problem this fixes:")
    # Same fact, different chunking. File-level scores both as a hit.
    at_800 = score_retrieval([chunk("it.txt::c0", 1)], ["it.txt"])
    at_400 = score_retrieval([chunk("it.txt::c1", 1)], ["it.txt"])
    check("chunk_800 hit", at_800.hit)
    check("chunk_400 hit (was a false miss before)", at_400.hit)
    check("comparable rr", at_800.reciprocal_rank == at_400.reciprocal_rank)

    print("\nMixed and edge cases:")
    m = score_retrieval([chunk("a.txt::c0", 1), chunk("b.txt::c2", 2)], ["a.txt", "b.txt::c2"])
    check("file and chunk ids together", m.precision_at_k == 1.0)

    m = score_retrieval([chunk("a.txt::c0", 1)], [])
    check("no ground truth -> not penalised", m.hit and m.reciprocal_rank == 1.0)

    m = score_retrieval([], ["a.txt"])
    check("nothing retrieved -> miss", not m.hit and m.precision_at_k == 0.0)

    m = score_retrieval([chunk("plain.txt", 1)], ["plain.txt"])
    check("id without chunk suffix", m.hit)

    print(f"\n{passed} passed, {failed} failed")