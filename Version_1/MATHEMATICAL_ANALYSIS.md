# Mathematical Analysis & Technical Implementation

This document explains the implementation and the underlying mathematical approach used by the project (Open_Hack — CCPA Violation Detector). It maps architecture components to code, formalizes the algorithms with equations, and discusses complexity, failure modes, and suggested improvements.

**Scope**: covers embedding model usage, vector store retrieval, Mixture-of-Experts (MoE) gating, expert retrieval and confidence scoring, RAG prompt construction and LLM classification, and the heuristic fallback classifier.

---

**Notation**

- $d$ — embedding dimensionality (e.g., `EMBEDDING_DIM`).
- $N$ — number of experts.
- $n$ — total number of chunks in the global store.
- $n_e$ — number of chunks in expert $e$'s sub-index.
- $\mathbf{q} \in \mathbb{R}^d$ — query embedding (vector).
- $\mathbf{e}_i \in \mathbb{R}^d$ — embedding for expert description $i$.
- $\mathbf{C} \in \mathbb{R}^{n\times d}$ — matrix of chunk embeddings in the global store.
- $\mathbf{P} \in \mathbb{R}^{m\times d}$ — pattern embeddings for heuristic fallback (m patterns).
- $k$ — retrieval top-k (per expert) (`TOP_K_RETRIEVAL`).
- $K$ — number of experts selected by the MoE (`TOP_K_EXPERTS`).
- $\tau$ — gating similarity threshold (`SIMILARITY_THRESHOLD`).

All vector inner-products in the code are used as cosine similarity because embeddings are L2-normalised before adding to the FAISS inner-product index.

---

**1. Embedding Engine** (`app/embeddings.py`)

- The embedding model maps text $t$ to an embedding vector $\phi(t) \in \mathbb{R}^d$.
- The code normalises embeddings (optionally via `normalize=True`) so that L2-normalised vectors satisfy $\|\phi(t)\|_2 = 1$.

Mathematical effect:

- When both query and stored vectors are normalized, the inner product equals cosine similarity:

$$\text{cosine}(\mathbf{u},\mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|} = \mathbf{u}^\top \mathbf{v} \quad\text{if }\|\mathbf{u}\|=\|\mathbf{v}\|=1.$$ 

Implication: Index similarity scores lie in $[-1,1]$ and larger is more similar.

---

**2. Vector Store** (`app/vector_store.py`)

- Uses FAISS `IndexFlatIP` (inner product) to store vectors. With normalized vectors this implements cosine similarity search.
- Ingest cost: storing $n$ vectors costs $O(n d)$ memory.
- Search cost for `IndexFlatIP`: $O(n d)$ compute per query (exhaustive inner products), but FAISS offers faster indexes if needed.

Search operation returns top-$k$ results

- For a normalized query $\mathbf{q}$, scores are computed as

$$ s_j = \mathbf{c}_j^\top \mathbf{q}, \quad j=1\dots n. $$

- Returned results are sorted by $s_j$ descending.

---

**3. Mixture-of-Experts (Gating Network)** (`app/moe_router.py`)

Registration:

- For N experts, the router computes description embeddings and stores them in matrix $\mathbf{E} \in \mathbb{R}^{N\times d}$ (each row normalized).

Routing (for query text):

1. Compute query embedding $\mathbf{q} = \phi(\text{query})$ and normalize: $\mathbf{q} := \mathbf{q}/\|\mathbf{q}\|_2$.
2. Compute gate scores (cosine similarity) via matrix product:

$$ \mathbf{s} = \mathbf{E}\,\mathbf{q} \in \mathbb{R}^N, \quad s_i = \mathbf{e}_i^\top \mathbf{q}. $$

3. Rank experts by $s_i$ and select the top-$K$ where $s_i \ge \tau$. If none exceed $\tau$, the highest-scoring expert is chosen.

Design note: The router uses raw cosine scores as gate scores (no softmax). These scores are in $[-1,1]$ and later used to scale expert confidences multiplicatively.

Probabilistic extension (optional): convert to probabilities via softmax:

$$ p_i = \frac{\exp(\alpha s_i)}{\sum_j \exp(\alpha s_j)} $$

with temperature $1/\alpha$ for calibration. The current code uses raw similarity to preserve interpretability and avoid over-smoothing.

---

**4. Expert Retrieval & Opinion** (`app/experts.py`)

Per-Expert pipeline (expert $e$):

1. Given query embedding $\mathbf{q}$, the expert normalizes it then retrieves its top-$k$ chunks from its private FAISS index: scores $r_{e,1},\dots,r_{e,k}$.

2. Keyword relevance is computed as a normalized hit-count score

$$ \text{kw}(q,e) = \frac{\#\{\text{keywords}_e\ \text{found in }q\}}{\max(1,|\text{keywords}_e|)} \in [0,1]. $$

3. Retrieval-based score is computed as a weighted combination:

- Let $\text{max}_r = \max_j r_{e,j}$ and $\text{avg3}_r = \frac{1}{\min(3, k)}\sum_{j=1}^{\min(3,k)} r_{e,j}$. Then

$$ r_{score} = 0.4\cdot\text{max}_r + 0.6\cdot\text{avg3}_r. $$

4. Final expert confidence (before gating) is

$$ c_e = \text{clip}\big(0.7\cdot r_{score} + 0.3\cdot \text{kw}(q,e),\ 0,\ 1\big). $$

5. After the MoE activates expert $e$ with gate score $g_e$ (cosine similarity), the pipeline scales confidence:

$$ c'_e = c_e \cdot g_e. $$

Interpretation: $g_e$ downweights expert opinions when the query is weakly matched to the expert's description.

Aggregation: Expert opinions are sorted by $c'_e$ and their retrieved chunks are merged (deduplicated) into context for the LLM.

---

**5. RAG Prompt Construction & LLM Classification** (`app/reasoning_rag.py`, `app/llm_engine.py`)

- The pipeline builds a context by taking the most confident experts and concatenating their top chunks (prefixing with section and page labels). The context size is capped (e.g. up to 10 chunks).

- The LLM receives the context and the original prompt and must output a JSON-like classification with keys `harmful` (bool) and `articles` (list).

Mathematical role of LLM: The LLM approximates a classifier / reasoning function

$$ f_{LLM} : (\text{context}, \text{query}) \mapsto \{\text{harmful: bool},\ \text{articles: list}\}. $$

Validation ensures the LLM response matches the expected schema. If the LLM is unavailable or its response fails validation, the system falls back to the heuristic classifier.

---

**6. Heuristic Fallback Classifier**

- Predefined patterns are embedded into $\mathbf{P} \in \mathbb{R}^{m\times d}$.
- For query embedding $\mathbf{q}$ the similarity vector is

$$ \mathbf{s}_p = \mathbf{P}\,\mathbf{q} \in \mathbb{R}^m. $$

- The heuristic picks the pattern index $i^* = \arg\max_i s_{p,i}$ and score $s^* = s_{p,i^*}$. If $s^*$ exceeds a threshold (e.g. 0.45) and the pattern is marked as harmful, return harmful with the pattern's sections.

- Additionally, expert signals are aggregated: if any expert with $c'_e > 0.3$ reports sections, compute

$$ c_{expert} = \max_e c'_e, $$

and if $c_{expert} > 0.5$ return harmful with those expert-indicated sections. This logic blends pattern-based evidence with expert retrieval evidence.

Decision boundary summary:

- If best pattern is harmful and $s^*>0.45$ → harmful.
- Else if best pattern is non-harmful and $s^*>0.5$ → not harmful.
- Else if $c_{expert} > 0.5$ → harmful.
- Else → not harmful.

This deterministic rule-set is simple and auditable; it provides a guaranteed output when the LLM fails.

---

**7. Mapping of Key Files to Responsibilities**

- `app/embeddings.py` — wraps sentence-transformers; produces normalized embeddings.
- `app/vector_store.py` — FAISS index, `IndexFlatIP` for cosine similarity when vectors are normalized.
- `app/experts.py` — `Expert` class, keyword scoring, per-expert vector store, confidence computation (see Section 4).
- `app/moe_router.py` — registers experts, computes gating scores, selects experts.
- `app/reasoning_rag.py` — end-to-end orchestration: chunking, building indices, routing, LLM invocation, fallback classifier, response validation.
- `app/llm_engine.py` — wrapper around the local quantized LLM, responsible for producing the final JSON output from context.
- `app/chunker.py` / `app/pdf_processor.py` — chunking and PDF extraction (create chunk embeddings, metadata used by vector stores).

Refer to these files when tracing execution for a single query.

---

**8. Per-query Computational Complexity (approximate)**

Let d be embedding dim, n total chunks, N experts. Assume each expert's sub-index sizes sum to n.

- Embedding the query: cost dominated by model inference; treat as $O(T_{embed})$.
- Routing: matrix product $\mathbf{E}\mathbf{q}$ costs $O(N d)$.
- Retrieval across activated experts: for each expert $e$ with index size $n_e$ and using `IndexFlatIP` the search is $O(n_e d)$; if K experts are activated total cost $O(d \sum_{e\in A} n_e)$ (worst-case $O(n d)$).
- Context assembly: concatenating top chunks $O(K k L)$ where $L$ is average chunk length in characters/tokens.
- LLM reasoning: $O(T_{LLM})$ dominated by LLM inference, dependent on model and context length.

Overall wall-clock is dominated by embedding & LLM steps; nearest-neighbour search can become the bottleneck if `IndexFlatIP` is used with large $n$. Consider IVF / HNSW indexes for sublinear search.

---

**9. Score Calibration & Interpretation**

- Cosine similarity values $s\in[-1,1]$. In practice sentence-transformer embeddings for short texts give scores in a narrower range (often [0,1] for related texts).

- Gate score $g_e$ multiplicatively scales expert confidence. If $g_e$ is small or negative, it can suppress an expert; the code ensures at least one expert is returned even if all $g_e<\tau$.

- Expert `confidence` is a heuristic, interpretable blending of retrieval evidence and keyword matches. For production, consider calibrating these scores against labeled data (e.g., isotonic regression or Platt scaling) to convert them to well-calibrated probabilities.

---

**10. Failure Modes & Mitigations**

- LLM Unavailability: covered by deterministic embedding-based fallback.
- Unnormalized embeddings: would break cosine semantics. The code normalizes query and description embeddings; ensure any third-party embedding call also normalizes.
- Exhaustive FAISS search (`IndexFlatIP`) scales poorly with large corpora. Replace with `IndexIVFFlat` or `IndexHNSW` for large `n`.
- Mis-calibrated thresholds ($\tau$, fallback thresholds): tune on validation set.
- Keyword matching brittle to phrasing: augment keywords with fuzzy matching or use learned classifiers.

---

**11. Potential Improvements**

- Use a soft gating distribution (softmax over scores with temperature) and compute a weighted ensemble of expert opinions rather than multiplicative gate scaling.

- Replace `IndexFlatIP` with HNSW or IVF for scalability:

  - HNSW: $O(\log n)$ search in practice, minimal recall loss with appropriate parameters.
  - IVF: faster at scale but requires training/quantization.

- Calibrate expert confidences and gating scores on labeled data.

- Use vector quantization or compressed indexes to reduce memory.

- Consider using learned rerankers / cross-encoders for final chunk ranking before context building.

---

**12. Example Math Walkthrough (one query)**

1. Query text → embedding $\mathbf{q}$, normalize ($\|\mathbf{q}\|_2=1$).
2. Gating: compute $\mathbf{s}=\mathbf{E}\mathbf{q}$; pick experts with $s_i\ge\tau$.
3. For each selected expert $e$:
   - Retrieve top-$k$ chunks with scores $r_{e,1}\ge r_{e,2}\ge\dots$.
   - Compute $r_{score}=0.4\max_j r_{e,j} + 0.6\text{avg3}(r_{e,1:3})$.
   - Compute $c_e = \text{clip}(0.7 r_{score} + 0.3\,\text{kw}(q,e), 0, 1)$.
   - Scale $c'_e = c_e \cdot s_e$.
4. Build context from retrieved chunks ordered by $c'_e$.
5. LLM returns `harmful` / `articles` or fail.
6. If LLM fails, compute $\mathbf{s}_p = \mathbf{P}\mathbf{q}$, take best pattern and evaluate threshold rules; also examine $c'_e$ to decide final result.

---

**13. Practical Notes**

- When adding chunks, ensure the same normalization policy is used for chunk embeddings and query embeddings.
- Keep metadata (section_id, page, domain) with each chunk to produce traceable explanations to the user.
- For explainability, surface the top contributing chunks and experts (the code provides `reasoning_hint` strings per expert).

---

**14. Quick Checklist for Reproducibility**

- Ensure `EMBEDDING_DIM` matches the model final layer.
- Verify `normalize=True` when encoding descriptions and patterns.
- Choose appropriate thresholds `SIMILARITY_THRESHOLD`, fallback thresholds, and `TOP_K_RETRIEVAL` after validation.
- If deploying at scale, change vector store to HNSW/IVF and persist the index to disk.

---

If you'd like, I can:
- add a labelled example showing numeric values for a toy query, or
- produce a small validation script that computes gating and retrieval scores for sample prompts.

File created: `Version_1/MATHEMATICAL_ANALYSIS.md`.
