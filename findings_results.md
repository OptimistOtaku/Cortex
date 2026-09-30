# Findings & Results

## 2026-09-29 — Prior-art + Qdrant Edge capability research (Code Cubicle PS03)

Mode: INVESTIGATE. No code written. Project dir was empty, so "existing repo / Kaggle work"
was read as public prior art: Qdrant docs, demos and course, plus competitor repos for this PS.

### Confirmed (evidence in parentheses)

- `qdrant-edge-py` latest **0.8.0** (2026-08-05), Apache-2.0, wheels include **win_amd64**,
  manylinux x86_64/aarch64, musllinux, macOS. No Android/iOS Python wheels. (PyPI JSON API)
- 0.8.0 API from the wheel's stub `qdrant_edge/__init__.pyi` (3396 lines; wheel downloaded to
  scratchpad and unzipped, NOT installed). Signatures confirmed; **runtime behaviour not tested**:
  - `QueryRequest(limit, offset, query, prefetches, with_vector, with_payload, filter, score_threshold, params)`
  - `Prefetch(limit, query, prefetches, params, filter, score_threshold)`
  - `Fusion.Rrf(k, weights=None)`, `Fusion.Dbsf()`, `Formula`, `Mmr(vector, lambda_, candidates_limit, using)`, `OrderBy`
  - `Expression.{Constant, Variable, Condition, GeoDistance, Datetime, DatetimeKey, Mult, Sum, Neg, Div, Sqrt, Pow, Exp, Log10, Ln, Abs, Decay}`, `DecayKind.{Lin, Gauss, Exp}`
  - `Query.{Nearest, RecommendBestScore, RecommendSumScores, Discover, Context, FeedbackNaive}`
  - `UpdateOperation.upsert_points(points, condition: Filter | None, update_mode: UpdateMode | None)`;
    `UpdateMode.{Upsert, InsertOnly, UpdateOnly}`; delete / set / overwrite payload by ids or filter;
    vector ops; field-index ops
  - Filters: `HasIdCondition`, `RangeDateTime`, `IsNullCondition`, `IsEmptyCondition`, `GeoRadius`,
    `MatchText/MatchPhrase/MatchTextAny`, `NestedCondition`, `MinShould`
  - `EdgeShard.{load, create, flush, close, optimize, update, query, search, scroll, count, facet, retrieve, info, unpack_snapshot, snapshot_manifest, update_from_snapshot(snapshot_path, tmp_dir)}`
    — **no snapshot creation on Edge**, so edge→server sync must be point-level.
  - Quantization: Scalar, Product, Binary, TurboQuant. `AcornSearchParams` present.
- Update durability: "written to the write-ahead log before it is applied … survives a crash". (Edge docs, updating-data)
- Qdrant server latest **v1.19.1** (2026-09-04) ships `qdrant-x86_64-pc-windows-msvc.zip`, so Docker
  is not needed. (GitHub releases API)
- Server conditional writes: `update_filter` (v1.16+), `update_mode` (v1.17+). On upsert the condition
  applies only to existing points; new points insert regardless. (Qdrant points docs, via search)
- `fastembed` 0.8.1. Multilingual text models: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
  `paraphrase-multilingual-mpnet-base-v2`, `intfloat/multilingual-e5-large`, `jinaai/jina-embeddings-v3`,
  `minishlab/potion-multilingual-128M`. Cross-modal: `jinaai/jina-clip-v1`, CLIP ViT-B-32. (fastembed source, main branch)
- Edge BM25: `Bm25(Bm25Config(language=...))`, tokenizers `prefix | whitespace | word | multilingual`;
  needs `EdgeSparseVectorParams(modifier=Modifier.Idf)`. (Edge BM25 docs)
- Vendor-reported numbers (NOT measured by us): ~11 MB footprint; 10k-vector query ~0.1 ms on-device
  vs ~52 ms cloud RTT; hybrid 0.51 ms. (Qdrant blog "Memory at the Edge")
- Qdrant's Vector Space Hackathon 2026 excluded RAG and simple chatbots; criteria Innovation,
  Creativity, Technical Depth. (Qdrant winners blog)
- Local hardware: Ryzen 5 4600H, 16 GB RAM, GTX 1650 4 GB, Win 11; Python 3.12.7, Node 20.20.2;
  no Docker, no Rust; Ollama model `qwen3-vl:2b-instruct`. (nvidia-smi, wmic, ollama list)

### Gaps in Qdrant's reference sync pattern (edge-synchronization-guide)

Derived by reading the documented code. The logic is certain, but **none of these has been
reproduced at runtime yet.**

1. Merge sorts both shards' hits by score and keeps the first ID seen. If a point exists in both
   shards (local edit of a synced point), the higher-scoring copy wins, even when it is the stale
   server copy.
2. The queue carries upserts only. There is no delete path, so a server-origin point cannot be
   deleted locally and keeps returning from the mirror shard.
3. The purge uses wall-clock `time.time()`: it deletes mutable points with `timestamp <= sync_timestamp`.
   Any point not yet uploaded (partial batch, dropped link, NTP clock jump) vanishes from local
   search, and the in-memory queue is lost on restart. The guide says to flush the queue first but
   has no guard for it.
4. The example uses integer IDs, which collide across devices if copied.
5. Every device mirrors shard 0 of the whole collection.

The docs say themselves that deletes, conflict resolution, dedup and ordering are not addressed.

### Competitor teardown (README/spec only, not their code)

- **FieldEdge** `choksi2212/code-cubicle-qdrant` ("Code Cubicle × Paytm × Qdrant 2025"): RN Android +
  Rust FFI + CLIP ONNX + FastAPI → Qdrant Cloud. One-way push, manual "Sync now", conflicts decided
  by "timestamp + vector checksum", no delete path documented, EXIF GPS synced, 270 ms offline search.
- **Aegis Edge** `tanishka4481/edge-node` (fork `KA-1205/edge-node`; 7 commits, spec-heavy):
  decision engine (novelty/urgency/sensitivity/completeness), trust-weighted N-way consensus with a
  DISPUTED state, event-sourced hub, outbox as a filtered scroll, priority lanes, network simulator,
  tactical-console UI, disaster vertical. Weaknesses:
  - Vote weight = trust × corroboration × e^(−λ·age) with one global λ, so it cannot separate
    "world changed" from "devices disagree". A stale majority beats a fresh first-hand report.
  - Trust = agreement rate. This penalises the first device to report a change, and relayed
    rumours count as independent corroboration because provenance is not tracked.
  - HYPOTHESIS (spec-level): priority-ordered push combined with "push only rows above the cloud's
    max client_sequence" can skip lower-seq, low-priority rows forever.
  - Ordering uses device wall clocks (`client_timestamp_ns`); there is no HLC.
  - Entities are matched by a string `corroboration_key`, and semantic conflicts "never auto-merge".
  - CONTRADICTION: they state that Edge has no Prefetch/fusion. The 0.8.0 stub has Prefetch,
    Fusion.Rrf(k, weights), Dbsf and Formula (runtime unverified).
  - Self-admitted: full hub copy per device, hub grows forever, no working-memory tier, embedding
    drift, answer layer needs ≥4 GB.
  - Their "anonymized embedding" privacy path is weak because embeddings are invertible (vec2text,
    Morris et al., EMNLP 2023; from own knowledge, not fetched).
  - PII regex is US-centric (SSN).
- Official prior art:
  - `qdrant/qdrant-edge-demo`: smart glasses, CLIP, SQLite queue, single device.
  - Video-anomaly edge→cloud tutorial: escalation threshold with ~6× bandwidth cut; no split-brain,
    privacy or UI handling.
  - DeepLearning.AI on-device memory course: 0.7.2, teaching-grade sync appendix.
  - `RGGH/qd-edge-ice`: full snapshot → edge only.
- Agent memory:
  - Mem0: LLM call per write (ADD/UPDATE/DELETE/NOOP); deletes contradicted memories.
  - Zep/Graphiti: bi-temporal invalidation, which is the right model, but it needs a graph DB and
    an LLM per write, and supports a single writer.
- Kaggle: Gemma 3n Impact Challenge winners (Klypt, Vite Vere) are single-device offline apps with
  no multi-device memory. "End-to-End RAG with Groq and Qdrant" depends on a cloud API.
- Motivation data: Maha Kumbh 2025 ran 10 digital Khoya-Paya centres with face recognition;
  reported reunions range from 20k (mid-Feb) to 35k–50k+ (end). There were network problems on
  Mauni Amavasya (29 Jan 2025).

### What did not work

- Kaggle pages (writeups, Gemma 4 Good project list) are JS-rendered; WebFetch returned titles only.
- The Code Cubicle site found is the 2024 edition. The 6.0 format, round deliverables and
  pre-existing-code rule are unconfirmed.
- The PyPI HTML page failed to load; the JSON API worked.

### Unfinished

- Runtime spike on 0.8.0: native Prefetch+Fusion, Formula decay on `DatetimeKey`, conditional upsert
  on Edge, partial snapshot against the Windows server binary, and reproductions of gaps 1–3.
  Needs approval to pip install into a scratch venv.
- Concept direction decision pending with user (A Khoya-Paya mesh / B truth kernel + disaster /
  C yatra data-mule), proposed in chat 2026-09-29.
  → SUPERSEDED same day: see "Decision" below.

## 2026-09-29 (later) — Qdrant docs deep read + solution decision

Mode: INVESTIGATE → DECIDE. Sources: all 12 Edge doc pages, server concept pages (hybrid queries,
explore, multitenancy), release notes, relevance-feedback article, competitor READMEs. Doc reads,
not runtime.

### Confirmed from docs
- Edge = full query engine in one embedded shard, **beta**. Dense, sparse, multivector, named vectors,
  quantization, HNSW, payload index, filter, facet, count, scroll, retrieve. Query: Prefetch nesting,
  RRF (k, weights), DBSF, Formula, MMR, OrderBy, Sample, Recommend, Discover, Context, relevance
  feedback. (edge-api/reading-data, edge-vs-qdrant-cluster)
- Edge vs server: no collections; `optimize()` manual and **synchronous**; "Only one `EdgeShard` may be
  open on a given directory at a time"; snapshots **restore-only**; groups and search-matrix **Rust-only**. (edge-vs-qdrant-cluster, shard-lifecycle)
- `info()` counts are approximate (a point can sit in >1 segment before optimize). (shard-lifecycle)
- Optimizer `prevent_unoptimized` defers point visibility until indexed. Default `vacuum_min_vector_number`=1000. (configuration)
- Conditional upsert on Edge: "the filter applies only to existing points … new points are inserted
  regardless". Named vectors can be added later but existing points are not populated. (updating-data)
- Sync primitives: full shard snapshot by server shard ID; partial = POST `snapshot_manifest()` to
  `/collections/{c}/shards/{id}/snapshot/partial/create`, which returns **changed segments** only; apply
  "rewrites files … pause or buffer writes … make sure queued updates have already reached the server".
  Upload = app dual-write from an in-memory queue ("for production consider persisting"). No deletes,
  conflicts or ordering in any pattern. (edge-data-synchronization-patterns, edge-api/snapshots)
- BM25 on Edge: IDF computed **within the shard**; stemming languages documented: english, german;
  tokenizer `multilingual` exists. `embed_query` vs `embed_document` must not be mixed. (edge-bm25)
- Server: weighted RRF and `update_mode` (v1.17); per-query IDF corpus, TurboQuant storage (v1.19);
  sparse IDF not tenant-scoped by default. (release notes, multitenancy)
- Relevance feedback: F = a·score + Σ confidence^b · c · delta; the feedback source can be a model or
  humans; the gain depends on the retriever's expressiveness. (relevance-feedback article)
- Qdrant's "Memory at the Edge" blog frames: local first, "uncertain answers … go to Qdrant Cloud",
  "fleet learning up". It does not address multi-device conflicts, deletes or privacy boundaries.

### Derived (logic, not run)
- The reference cross-shard merge sorts by raw score, but BM25 IDF differs per shard, so sparse and hybrid
  scores are not comparable across `local` and `mirror` shards. Merge by rank plus version instead.
- Partial-snapshot delta size depends on the server's segment layout. Unmeasured.

### New competitor
- **EDGE.MEM** `hemv-857/edge-mem` (27 commits): industrial notes, residency rule engine
  (synced/queued/local_only), bge-small + BM25 RRF, federation, Next.js console, 124-check Playwright
  audit. Reports that the `qdrant_edge` Python binding **segfaults off the main thread** (their claim,
  unverified). No conflict resolution, no tombstones. ⇒ static residency rules plus a hybrid dashboard are table stakes.

### Other
- Code Cubicle 6.0 details page is JS-rendered/empty; Devfolio page is the 2024 edition. Round
  format is still unconfirmed from the web. User states 3 Oct deliverable = **working prototype**.
- Datasets checked (not used): PlantVillage (54k lab images), PlantDoc (~2.6k field images).

### Decision (user-approved plan, 2026-09-29)
Build **Cortex**, a platform, not a vertical: Edge Node (Qdrant Edge `local` + `cloud`-mirror shards,
hybrid search, local LLM, durable sync agent) + Cloud (Qdrant Server + sync/escalation service) +
Console. Differentiators: correct sync semantics (outbox, HLC, tombstones, ack-based purge,
rank+version merge), dynamic per-point residency with reasons, supersede vs dispute, an
"edge answers, cloud refines" escalation + cache-back loop. Demo data: field-service fleet (swappable).
No face recognition. Plan: `~/.claude/plans/we-are-working-on-splendid-corbato.md`.

## 2026-09-29 — Day-1 runtime spike (qdrant-edge-py 0.8.0 + Qdrant server 1.19.1, Windows 11)

Mode: VALIDATE. Scripts are in `spikes/` (01 engine, 02 crash, 03 flush cost, 04 server sync, 05 wire size, 06 AI).
Venv: Python **3.13.1** (the default `python` on this machine, not 3.12.7).

### Confirmed at runtime
- Engine (spike_01, all PASS): UUID ids + reload; BM25 sparse; hybrid `Prefetch` dense+BM25 → `Fusion.Rrf(k=2, weights=[1,3])`;
  `Formula` = `$score` + 0.5·exp-decay(`DatetimeKey`) reorders by recency; `FeedbackNaive` runs; delete by id/filter.
- **Conditional upsert works as a version guard on Edge**: `condition=ver < new` rejects a stale write (v3 after v5) and
  applies a newer one (v7). `InsertOnly` is a no-op on an existing id. A new id inserts regardless of the condition.
- **Cross-shard BM25 scores are incomparable**: the same doc and query scored 4.50 in shard A and 8.92 in shard B (IDF differs).
  The reference raw-score merge is invalid for sparse/hybrid.
- **Threading: no crash** with 200 ops on a worker thread or 4 concurrent threads × 200 mixed ops. EDGE.MEM's segfault
  claim does NOT reproduce on 0.8.0/Windows. (A single-writer actor is still sensible for flush/snapshot ordering.)
- Second `EdgeShard.load` on an open dir is rejected ("failed to open WAL").
- Stub/runtime mismatch: `retrieve()` requires `with_vector` although the stub says optional.
- **CRASH DURABILITY CONTRADICTS DOCS** (spike_02/03): after upsert + `os._exit`, recovered 0/1, 0/50, 0/2000 points,
  even after a 2 s sleep. After `flush()`: 100% recovered. Writes after an earlier flush are also lost (10/10 flushed,
  0/10 unflushed). The WAL files exist (64 MB preallocated) but are not replayed. Docs claim "an update that has
  returned survives a crash". Only win_amd64 tested.
- Cost: upsert 0.13 ms (1 pt) to 7.4 ms (1000 pts); **flush ≈ 16–23 ms, roughly constant** → group commit.
- Server: custom sharding **needs cluster mode** (`QDRANT__CLUSTER__ENABLED=true`, `--uri`), even on one node.
- **Windows MAX_PATH**: server storage under a long path fails ("Gridstore IO error … os error 3"; segment dirs
  reach 248 chars). Run the server from a short path (used `C:\Users\ADITYA SINGH\.cortex\qdrant`).
- Per-zone mirror: shard-key `zone_a` → its shard snapshot → Edge holds only zone_a points. PASS.
- **Partial snapshot propagates inserts, payload updates AND deletes.** PASS. Server-side deletes reach the edge.
- **Edge-local writes into a mirror shard are wiped by `update_from_snapshot`**, so the separate local shard is mandatory.
- `update_from_snapshot(tmp_dir=…)` requires tmp_dir to exist. A no-change partial = 0 bytes (cheap poll).
- **Snapshot size is dominated by preallocated 33.6 MB pages**: full = 596 MB for 100 pts (8-d); partial after
  +10/5 upd/5 del = 453 MB, apply 8 s; after a 1-payload change = 139 MB raw.
- **On the wire the server compresses with Brotli** when sent `Accept-Encoding`: 663 MB full → 0.4 MB.
  Client gzip of the 1-change partial = 203 KB. So the wire is fine, but device disk writes and apply time are heavy.
- fastembed `paraphrase-multilingual-MiniLM-L12-v2`: 384-d, 8.4 ms single, 50 ms batch-4, 32 s first load.
  EN↔HI same fact cos 0.70 vs unrelated 0.30.
- Ollama `llama3.2:1b`: warm 4.4 s per classification, 41 tok/s, and it **misclassified** "gate code 4471" as internal, not PII.
  `qwen3-vl:2b-instruct` with `format: json` returned an empty response.

### Design consequences
1. Durability: our SQLite outbox is the source of truth for un-acked writes, replayed into the shard on start.
   Group-commit `flush()` every ~250 ms / N writes. Do not trust the Edge WAL alone.
2. Down-sync is hybrid. Brotli snapshot for bootstrap and big catch-up; **point-level delta** (payload `hlc`
   watermark + server tombstone points) for small changes (~KB vs ~200 KB compressed / 139 MB on disk).
3. Two shards per device (local + mirror) are required. Cross-shard merge by rank + version, never raw score.
4. Version guard for writes = conditional upsert (`ver < new`) on Edge; mirror it on the server with `update_filter`.
5. Write-path residency must be deterministic (PII detectors + embedding prototypes). It runs in ms. The local LLM is too slow
   and inaccurate here, so use it only for answer synthesis off the hot path.
6. Pitch material: "Qdrant Edge on Windows loses un-flushed writes on crash; we measured it and designed around it".
   Consider filing an upstream issue (outward-facing, so ask the team first).

### TypeSafe Jev (cloud API) — spike_07/08, 2026-09-29
- Cloud-only: `https://api.typesafe.ai/v1/systemone`, bearer key, model `jev-latest` (answered as jev-1.13.0).
  Docs mention no offline or self-host option, so it **cannot run on the edge**. Key is kept in env `TYPESAFE_API_KEY` only, never in the repo.
- Latency from this laptop: first call 856–1140 ms, then **median ~436 ms**; 4 questions batched in one call ≈ 365 ms.
  (Local llama3.2:1b: 4.4 s.)
- Accuracy on small synthetic sets (n is tiny, indicative only): escalation routing 3/3, answer-quality score 2/2,
  sensitivity 4/4 (gate code and phone → pii, where llama was wrong).
- **Conflict "supersede vs dispute" asked directly: 2/4.** It called 1-minute-apart contradictory reports from
  different devices "supersede". Reframed as a pure semantic question ("incompatible claims about the same property?"):
  **8/8**, but two negatives were near the threshold (0.46, 0.41).
- ⇒ Split: HLC/causality decides *when* (deterministic, on the edge); Jev decides *whether* two notes contradict
  (cloud, ~0.4 s). Treat 0.35–0.65 as "unsure" and show both. Jev fits Cortex Cloud: escalation routing, answer-quality
  gate before caching and fan-out, contradiction check. On-device decisions stay local (embeddings + patterns).

### Ollaya 0.7.5 local decision models vs Jev — spike_09, 2026-09-29
- Ollaya installed by the user (`%LOCALAPPDATA%\Programs\Ollaya\bin\ollaya.exe`); the installer auto-starts a server on
  127.0.0.1:11435. `pull laya` fetched `laya:en` (853 MB) and `laya:multilingual` (683 MB): 322M params, ONNX,
  run on **cuda:0 F16** on the GTX 1650. The wire API is identical to Jev, so the same code works against both.
- Same 17 synthetic cases as spike_07/08 (tiny n, indicative only):

  | | sensitivity | routing | quality | contradiction (noul) | latency median | 3-q batch |
  |---|---|---|---|---|---|---|
  | laya (→ laya:en) | 3/4 | 1/3 | 1/2 | 6/8 | 314 ms | 651 ms |
  | laya:multilingual | 3/4 | 2/3 | 2/2 | 5/8 | **122 ms** | 229 ms |
  | Jev cloud | 4/4 | 3/3 | 2/2 | 7/8 | 406 ms | 359 ms |
- laya:en probabilities on our custom sensitivity labels are nearly flat (e.g. 0.42/0.35/0.23), and it **missed the gate code**.
  laya:multilingual caught the gate code (pii 0.97) but **missed the phone number** (pii 0.41 vs public 0.42).
  multilingual over-calls contradiction on related-but-compatible pairs (0.95 for "E42 active" vs "fan filter dusty").
- ⇒ Zero-shot laya alone is not reliable enough for residency or conflict decisions. Usable as a fast *signal*:
  combine it with deterministic PII patterns (which catch the phone case) and escalate to Jev when laya's top-2 margin is low.
  Next to test: an embedding-head classifier distilled from Jev labels (step 2).

### Step 2: Jev-distilled edge head — spike_10, 2026-09-29
- Method: 300 template notes + 150 template questions (EN + Hinglish), labelled by Jev (cached in `spikes/data/`,
  ~14 s with 8 threads). Logistic regression on normalised multilingual-MiniLM embeddings. Evaluated on a
  **hand-written held-out set** with different phrasing (24 notes, 12 questions; gold labels by us). Small n: ±~10 pts.
- **v1 labels failed because of label wording, not the models**: "public/internal/pii" made Jev call 96/100 technical notes
  "internal" (it read "public" as "publishable"). The head then scored 62.5%. Lesson: name classes by *destination*.
- **v2 labels** `personal_or_secret` / `business_confidential` / `technical_knowledge` (Jev agrees with template
  intent 99%):

  | | sensitivity | routing | latency/item | offline |
  |---|---|---|---|---|
  | distilled head (embed + LR) | **95.8%** | **91.7%** | **4–5 ms** | yes |
  | laya:multilingual (Ollaya) | 62.5% | 50.0% | 120–220 ms | yes |
  | Jev cloud | 100% | 100% | ~360 ms | no |
- Remaining misses:
  - Sensitivity: health info without identifiers ("owner is recovering from surgery") → business_confidential at conf
    0.72, so it would NOT escalate. PII regex doesn't cover health. Add health/family terms to the training data.
  - **Routing: "colleague fell from the tower ladder and is not responding" → fleet_knowledge (conf 0.64).** This is a
    safety-critical false negative. Needs a deterministic safety override (injury / unresponsive / fire / smoke / shock /
    gas / sparks → human_expert) ahead of the head, plus injury examples in training.
- ⇒ On-device decision stack: safety keywords → PII patterns → distilled head (4 ms) → if confidence < 0.6, mark
  "uncertain", act conservatively (keep local / show both), and ask Jev when online. Jev's answers become new
  training labels, and the retrained head syncs down to the fleet ("cloud teaches edge"). laya is not used.

## 2026-09-30 — Product decision: SIETCH + image-retrieval go/no-go (spike_11)

Decision: build **SIETCH** (fleet of camera nodes, value-per-byte sync gate under scarce comms windows,
one-sentence retasking, ground control on Qdrant Server). Milaap is the fallback. Plan in `~/.claude/plans/`.

Gallery: 45 Wikimedia Commons images in 9 folders (`spikes/photos/commons`, manifest has sources). **Folder labels
were wrong in places, checked by eye on the contact sheet**: "logo_sign" holds building facades and Ferris wheels,
not logos; 2 of 5 "glasses" images show no glasses. So scoring uses hand-checked relevance per intent (`QRELS_COMMONS`).
The logo intent is untested and needs our own photos.

| model (FastEmbed, offline) | dim | hit@1 | precision@3 | image embed | text embed | load |
|---|---|---|---|---|---|---|
| Qdrant/clip-ViT-B-32 (vision+text) | 512 | 89% | 78% | 61 ms/img | 10 ms | 79 s first (download) |
| **google/siglip2-base-patch16-224** | 768 | **100%** | **85%** | 222 ms/img | 55 ms | 6 s cached |
- 9 intents (red, bottle, glasses, layered rock, water, plant, laptop, woman portrait, Ferris wheel). n is small.
- **GO** for image retrieval. Primary: SigLIP2. CLIP is the fallback for slow laptops.
- Gotchas: (1) HF Xet CDN download failed once, so use `HF_HUB_DISABLE_XET=1`. (2) **fastembed reads SigLIP2's
  tokenizer config as cp1252 on Windows and crashes (UnicodeDecodeError), so run Python with `PYTHONUTF8=1`.**
- FastEmbed image models available offline: clip-ViT-B-32, resnet50, Unicom-ViT-B-16/32, jina-clip-v1,
  nomic-embed-vision-v1.5(-Q), siglip2-base-patch16-224. Text↔image pairs: CLIP, jina-clip-v1, SigLIP2.

Privacy (faces), OpenCV YuNet 2023mar (`spikes/models/`), score threshold 0.7:
- 36 ms/img. Faces found in 9/10 people images, plus 1 in a street scene (a small motorcyclist, a correct detection).
- **Missed a frame-filling close-up face**. A padded second pass did NOT recover it.
- SigLIP zero-shot max-sim to {"a photo of a person's face", "a photo of a person", "a portrait of a human"}: all 10 people
  images ≥ 0.092, all 35 non-people ≤ 0.071, so the classes separate. The missed close-up scored highest (0.140). SigLIP misses
  small people (the motorcyclist scored below 0.071), and YuNet catches those.
- ⇒ Privacy gate = YuNet face (blur the regions) **OR** SigLIP person score > ~0.085 (the whole image is sensitive: the full-res
  image never leaves; only a fully blurred thumbnail or the embedding). The threshold is from 45 images; recheck on venue photos.

### Sync gate as a Qdrant query (2026-09-30, node core in `sietch/node/`, tests in `tests/`)
- Edge `Formula` reads payload variables and divides: value/KB ordering matched hand computation. `Expression.Div`
  requires `by_zero_default` positionally (stub says optional).
- `Mmr` as an outer query over a `Formula` prefetch **discards the formula ordering** (it re-scores by vector sim),
  so near-duplicate suppression is done in Python during the greedy budget fill instead.
- **Raw SigLIP cosine is too small and bunched (~0.0–0.15) to weight directly**. A first version let novelty and cheap
  blurred thumbnails dominate (retask test failed: the top 3 for "layered rock" were a bottle and 2 people). Fix: relevance =
  min-max of intent sim over the current queue (bounds passed as Formula constants), **squared**, plus a 6 KB cost
  floor (`cost_kb` payload). Still a single Formula query.
- Result on the 45-image gallery (shuffled capture order), 40 KB window: "evidence of past water" → gate sent 5/6
  water/rock, FIFO 1/6; "find anything red" → gate 3 red of 6 (the gallery has ~5 red), FIFO 0; "a person wearing
  glasses" → gate 5 glasses (face-blurred) of 7, FIFO 1. FIFO's window is the same whatever the intent.
- All 10 people images were marked sensitive with no full-res payload. Hard-exit replay of our SQLite log: 20/20 writes restored.

### End-to-end build (2026-09-30): `sietch/`, `run_fleet.py`, `tests/e2e_scenario.py`
- Stack: 2 rover processes (FastAPI + Qdrant Edge + SigLIP2 + YuNet) and ground (FastAPI + Qdrant Server 1.19.1),
  one laptop, `run_fleet.py --fast` (8–12 s gaps, 30 KB budget per channel per window). 45-image gallery split
  across the rovers, with the water folder on both.
- **e2e: 11/11 checks pass** (two fresh runs). Numbers from the last run:
  - offline search in airplane mode: median 56–65 ms, of which the **Qdrant Edge query is ~0.5 ms** (the rest is
    SigLIP text embedding). The first query after start: ~200 ms.
  - two gate windows per rover after retasking to "evidence of past water": **gate 8/15 useful in 107 KB vs FIFO
    3/17 in 115 KB** (run 1), **9/15 vs 0/18** (run 2). The "judge" is simulated with gallery relevance; a person rates live.
  - 0 downlinks over budget (the budget is enforced on the real serialized request body). The gate never paid twice for a site
    another rover had delivered (FIFO did: 0–5 duplicate sites depending on timing).
  - full-res of a 62 KB image requested by ground: streamed in 3–11 chunks across windows, reassembled. Full-res of a
    person image is refused.
  - relabel → rover label superseded with history; discard → rover memory and files deleted.
- **Qdrant Edge 0.8.0 (Python) bug: boolean `MatchValue(True)` never matches**, with or without a Bool index, whether
  set via upsert or set_payload (the payload shows `True`). Keyword and integer matches work. Workaround: 0/1 integer flags.
- **Full-res vs budget**: a 305 KB full-res could never fit a 30 KB window. Now demand-first chunked streaming.
- **sqlite3 connection shared across threads** (web handlers + comms loop + flusher) threw "bad parameter or other API
  misuse" under load. Fix: `sietch/common/db.py` LockedDB (every statement + fetch under one lock).
- Gate v1 kept re-ranking fleet-redundant items at #1 with novelty 0 (held back every window, but they looked pending).
  They now move to a `held` state with the reason ("rover-2 already delivered this to ground"), shown on the rover UI.
- `run_fleet.py` stdout was block-buffered when redirected, so progress was invisible. Fix: line buffering.
- UIs verified via headless Edge screenshots (ground big screen + rover console render with live data).

## 2026-09-30 — Reframe + local LLM brain spike (spike_12)

Reframe (user): SIETCH = **Qdrant-powered edge memory for the LLM brains of robots and drones** (planetary rovers,
exploration, disaster drones, ops in denied comms). Pull stays as is (no formula-driven pull).

**CORRECTION: this laptop has 7.3 GB RAM** (Ollama log + Win32_OperatingSystem), not 16 GB as recorded on 2026-09-29.
With VS Code, Firefox, WebView2 and 2 Claude processes open, free RAM was 0.1–0.9 GB and commit was 21/28 GB.
- Orphaned SIETCH fleet processes (ground, 2 rovers, Qdrant, launchers) survived the previous session and held
  memory. They were stopped. `run_fleet.py` children must be cleaned up when the parent dies.
- Ollama then failed every load with "cudaMalloc failed: out of memory" (even for a 1.3 GB model; nvidia-smi showed 0 MiB
  used), and the idle `ollama serve` held 1.7 GB private. Restarting it with `OLLAMA_CONTEXT_LENGTH=4096` fixed it.
  All layers are on the GPU.

Brain benchmark (Ollama 0.24.0, GTX 1650 4 GB, num_ctx 4096, temperature 0):

| | qwen3-vl:2b-instruct | llama3.2:1b |
|---|---|---|
| capabilities | completion, vision, tools | completion, tools |
| stream (warm) | first token 2.3 s, 76 tok/s | first token 2.3 s, 99 tok/s |
| tool loop recall→remember→act | **valid args**, 11.5 s / 3 steps | **broken args** (echoes JSON schema strings) |
| grounded answer, 5 memories | first token 2.7 s, 6.4 s total; relevant reasoning but also mentions distractors | 6.0 s |
| caption of a camera frame | **accurate** ("muddy road, large puddle reflecting auto-rickshaw…"), 10.8 s | n/a |
| placement | 3.30 GB, all in VRAM | 1.74 GB |

- ⇒ Brain = **qwen3-vl:2b-instruct** (sees, uses tools, answers). ~10 s per deliberation or caption, so the brain must be
  **event-driven** (the Formula gate is the ms reflex; the LLM deliberates only on triggers).
- Its `remember` note restated the observation instead of an inference. The prompt needs "write what it MEANS for the
  mission, cite evidence ids". Measure insight quality in the e2e.
- The earlier `format: json` empty reply (2026-09-29) is avoided by using native tool calling instead of JSON mode.

### Brain e2e, first runs (2026-09-30, `tests/e2e_brain.py`, CLIP, brain on rover-1 only, 3 MB storage)
- Checks 0 and 1 PASS: rover-1 has the brain and rover-2 is reflex-only. The Formula reflex woke the brain for **4 of 20** frames.
- **Insight quality is good.** One reflex-woken frame produced this insight: "The reflective surface of the puddle [obs-550efee5] and its mirror-like quality strongly suggest it formed from standing water, which is a key indicator for past water presence [obs-c7131786]." It is an inference, and it cites a second recalled frame. The brain then logged `investigate`, and the UI raised the closer-look prompt.
- **The Ollama runner then died with host OOM** (`GGML_ASSERT(ctx->mem_buffer != NULL)`) while free RAM was 0.4–1.0 GB. Every reload after that failed with "memory layout cannot be allocated". Details:
  - Ollama loads with `UseMmap:false`, so it copies the model into committed host RAM before moving it to the GPU. Passing `use_mmap: true` in the options did not change that.
  - Commit charge was 25/30 GB with ground + 2 robots (~1.2 GB private each), VS Code, Firefox and Claude running.
  - Restarting `ollama serve` recovers its leaked ~1.7 GB only while there is headroom.
- Fixes:
  - `Brain._chat` retries 5xx and connection errors with backoff (5/15/30 s), and the UI shows "brain unavailable, retrying". A failed examine leaves the frame unexamined, so it can be retried.
  - The e2e no longer crashes when check 2 produces no insight.
- ⇒ On a 7.3 GB laptop, ground + 2 robots + the brain fits only with browsers closed. For the demo, put the brain robot on its own laptop.

### Brain e2e: 12/12 (2026-09-30, after the user closed Firefox; same setup)

| Check | Result |
|---|---|
| Reflex wake | 4 of 20 frames |
| Operator-requested examine: insight + decision | 34 s; `investigate` |
| Insight-first | the 1.7 KB insight reached ground; its picture is still not down |
| Offline Q&A (airplane mode) | 6 s warm, 4 citations |
| Ask over the link | rover-1 (brain) 15.4 s round trip; rover-2 (reflex-only) 3.2 s |
| Dispute | Jev **p = 0.92**; resolved; the loser was superseded on rover-2 and dropped out of its recall |
| Forged discard | refused |
| Storage tiers | ~2.9 MB against the 3 MB budget |

- **Bug found and fixed: disputes never opened when the evidence was a fleet duplicate.**
  - The gate correctly held back rover-1's picture because rover-2 had already delivered the same scene. So that picture never reached ground, and rover-1's insight kept the provisional site `a4e92b00`, while rover-2's note sat on site `4097a461`.
  - Fix: a text memory's package carries `evidence_alias` {evidence id → the fleet item it duplicates}. Ground's `_evidence_site` uses the alias when the picture is absent.
- Test race fixed: `act` follows `remember` a few seconds later, so the e2e now waits for the decision.
- Quality of the 2B model (for the pitch, and something to improve):
  - Insights are real inferences with citations.
  - Answers sometimes cite distractors: a plastic bottle offered as "suggests a past water source".
  - One answer contradicted itself ("No standing water … a puddle of water").
  - Captions of the low-resolution gallery frames can be vague ("blurry, hazy landscape").
  - Mitigations to try: stricter answer prompt, drop low-relevance memories from the context, bigger model on a stronger rover.

### Brain prompt fixes (2026-09-30), final e2e 12/12 on a fresh fleet; `e2e_scenario.py` still 11/11
Latest run: gate 9/19 useful vs FIFO 0/21 at ~140 KB.
- **The few-shot example was parroted, and its ids were cited as if real.** Insights read "Polygonal mud cracks here … suggest standing water once pooled in this basin" and cited `[obs-aa11bb22]`, the prompt's example id. Fixes:
  - The examine prompt gives a *shape* ("<what you see> [<this frame's real id>] together with <what you recalled> [<its id>] suggests …") instead of a copyable sentence.
  - `NodeMemory.strip_unknown_refs` removes citations of nonexistent memories and template placeholders from insights, decision reasons and answers.
- **Removing the concrete format hint made answers stop citing** (0 citations). The answer prompt now uses the first *real* recalled ref as its example; the next run cited 3–6 real memories.
- **Brain recall now covers only knowledge** (insights, notes). The brain's own decisions and earlier answers were crowding out observations in its context.
- e2e check 3 is n/a when the picture went down before the brain concluded, which happens when the brain is busy with reflex frames. `test_conclusions_travel_before_pictures` covers the ordering deterministically.
- Still weak (2B model): "Did you see any standing water?" sometimes gets "No … only a muddy puddle", a semantic quibble. Captions of blurry gallery frames can start "Based on the description provided…".

## 2026-09-30 (evening): Mars simulation demo, all on this laptop

User direction:
- **Drop the FIFO comparison.**
- **Demo = a 3D simulated Mars terrain with our robots in it.**
- **Everything on this laptop.**
- Cast: rover + scout drone, both with an LLM brain.
- Real NASA imagery at the science targets.
- A 3-minute video.

### What changed
- **FIFO removed** from the node, comms, ground, both UIs and the tests. The gate e2e check now asserts "after a retask, what comes down is at least 2× as mission-relevant as what the rovers hold".
- **One process hosts ground + rover-1 + scout-1** (`sietch/sim/host.py`, launched by `run_sim.py`) with one shared model (`Perception.shared`):
  - host private memory **2.28 GB with SigLIP2**, versus ~1.2 GB for each separate process before;
  - Ollama's runner shows 5.6 GB private, mostly paged out;
  - free RAM with VS Code and Claude open: 0.3 GB, so close them to record.
- **The AMD Radeon iGPU drives the display; the GTX 1650 is headless.** Edge renders the sim on the iGPU by default, and the brain keeps the 4 GB card.
- **Geotagged memory.**
  - A capture's `X-Pos` becomes payload `pos` plus `geo`.
  - qdrant-edge-py 0.8.0 supports `PayloadSchemaType.Geo`, `GeoRadius(GeoPoint(lon, lat), m)` and `FieldCondition(geo_radius=)`. These power `/api/near` and `recall(near=)`.
  - Qdrant's geo distance uses Earth's radius, so sim metres map with 111,195 m/deg (`sietch/common/geo.py`).
  - The brain's recall lines carry "38 m NE". A live answer said "…observed at 141 meters SE [ins-f937574c]".
- **Ground merges sightings from different robots within 12 m into one site** (Qdrant Server `GeoRadius`). A conclusion whose picture hasn't come down yet joins what the fleet knows at that place (`_site_near`). Verified live: the scout's aerial view and the rover's close-up of the slab merged into one site.
- **Comms windows are driven by the simulated orbiter** (`SIETCH_GAP=manual`, `/api/window_now {budget}`).
- **Navigation-camera frames never wake the brain.** Only science cameras (mastcam, aerial) do.

### spike_13: SigLIP2 on real Mars imagery (16 NASA images, `sietch/sim/fetch_assets.py`)
- **Retrieval hit@1 7/13, hit@3 8/13.**
  - Good: mud cracks, the meteorite, the spotted rock, dunes, sunset, aerial ripples, spacecraft debris.
  - Misses: layered sediment, stream pebbles, veins, spherules and "evidence of past water"; the leopard-spots close-up acts as a hub.
- **Zero-shot tags suffered from hubness**: "spacecraft debris", "crater" and "light-toned rock" topped nearly every image.
  - **Fix:** subtract each tag's mean similarity over the Mars image set (`Perception.calibrate_tags`, run at host start).
  - Tags then read, for example: mud_cracks_close "cracked rock, mud cracks, white mineral veins"; dust_devil "dust devil, hills, sunset"; sunset "sunset, horizon, sky"; leopard_spots "spotted rock…"; aerial_backshell "spacecraft debris, drill hole, parachute".
  - The calibration set is the same 16 images, so this is indicative, not a held-out measurement.

### qwen3-vl:2b agrees with whatever mission it is given
- With the mission "evidence of past water", the brain concluded water from Ingenuity's aerial **sand ripples** ("ancient river channels") and from the dark Namib **dune**. A cross-robot dispute therefore never arose: Jev correctly scored the scout's and rover's conclusions compatible (p = 0.08).
- **Direct visual yes/no** ("Does this image directly show evidence of past water (e.g. mud cracks…)?"): YES for 7 of 8 images, including the meteorite and the backshell ("shows mud cracks"). NO only for the dune.
- **Fixes that helped description quality, but not the bias:**
  - The caption prompt no longer contains the mission. Told what to look for, the model saw it everywhere; neutral captions now describe fractured rock, veins and cracks correctly.
  - The examine prompt asks for sceptical judgement.
  - Examine recalls frames only, because the model copied its own earlier insights verbatim.
  - `strip_unknown_refs` also drops placeholders like "[its id]".
- **Consequence for the demo:** the scout's "only sand ripples, no sign of water" is a scripted survey note (`/api/note`), and the captions say so. The dispute detection (Jev), the resolution and the supersede are real. The scout's own brain examines the landing debris.
- A bigger or better-calibrated local VLM is the upgrade path.

### First full rehearsal (headless Edge + CDP watcher, `?demo&autoplay`)
- All chapters ran with no script errors in ~10.5 min of real time.
- **Brain:** the rover's brain decided `investigate` at the slab, and the sim drove it closer for a close-up.
- **Pass 1:** sent 2 insights + 2 decisions + 4 pictures in 26.8/30 KB, with 12 memories waiting.
- **Leopard-spot rock after the retask:** `investigate`.
- **Blackout:** answered offline with a citation and a distance.
- **Trust:** the forged discard was refused.
- **Conflict (chapter 4) was skipped** (see above).

### Making the conflict beat real, and rehearsal 3 (12/12 beats)
- **Rehearsal 2:** the sceptical prompt made the rover say "No sign of past water" at the slab, with no insight, and it moved on. The 2B model is unreliable in both directions on the dark, wide panorama.
- **Fixes:**
  - Every examination now ends with a conclusion; negative conclusions sync too. If the model acts without remembering, it is asked once more.
  - The rover's mastcam at the slab shows the clear polygonal-crack close-up (PIA21261).
- **Jev also scored "No sign of water here" against "…suggests possible evidence of past water" as compatible (p = 0.19–0.21).** "Both cannot be true" lets a hedge coexist with a negative.
- **New wording: "reach opposite assessments of the same place: one sees evidence (even tentative) for something that the other says is absent?"**

  | Pair | old p | new p |
  |---|---|---|
  | demo pair (hedged) | 0.19 | **0.76** |
  | clear disagreement | 0.71 | **0.93** |
  | complementary | 0.04 | 0.03 |
  | both negative | 0.05 | 0.02 |
  | different things | 0.08 | 0.05 |

  5/5 correct, with wide margins.
- **Focused test:** the dispute opens at p = 0.79.
- **Rehearsal 3, ~11 min real time, all chapters, no script errors:**
  - the brain decided `investigate` at the slab ("fractured rock with white mineral veins and mud cracks");
  - dispute **p = 0.88**, resolved; the scout's note was superseded and struck through;
  - retask; leopard spots → `investigate`;
  - recall "mud cracks": 318 ms, of which Qdrant Edge 1.7 ms;
  - blackout answer citing the slab frames and the insight;
  - forged command refused;
  - end card.
- **Pacing:** the director no longer waits for the scout's brain in chapter 1. Both brains share one GPU, and that wait cost ~5 min in rehearsal 2.

### Unfinished
- Confirmation pass on our own venue-style photos (`spikes/photos/own/<category>/`, run with `--own`), incl. logos.
- Reproductions of reference-pattern gaps 1 and 3 as a side-by-side demo (logic already derived).
- Linux/macOS durability check (only Windows tested).
- Server-side conditional update (`update_filter`) not yet exercised.
- qwen3-vl JSON-mode empty response: not investigated (try without `format`, or a `/no_think` prompt).
