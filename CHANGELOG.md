# Changelog

All notable changes to the identity + activity pipeline are recorded here.

## 2026-08-13 to 2026-08-15 — `marvin/end-to-end-pipeline-mevid`

### Identity-matching accuracy
- Face-detection confidence threshold lowered `0.5 → 0.15` (fixes faces the
  detector was missing outright, notably darker-skinned faces).
- Identity match threshold lowered `0.4 → 0.2`.
- Face-sampling per track widened `5 → 20`, with a confidence-based early
  exit: stop once several samples *agree* above `0.6`, instead of stopping
  on the first lucky/unconfirmed frame.
- Fragmented-track merging: stitches a person's tracker-ID switches (after
  an occlusion) back into one track using spatial + appearance continuity.

### Speed
- Shared `ArcFaceEmbedder` across a batch instead of reloading insightface's
  models once per item.
- Frame-read caching in `score_tracks` (was re-reading the same frame from
  disk once per track that referenced it).
- `onnxruntime-gpu` on Linux, so face embedding runs on GPU instead of CPU.
- Auto-detected static-camera skip for BoT-SORT's global motion
  compensation — real cost, zero benefit on fixed/mounted cameras; detects
  per video, so genuinely moving footage is untouched.

### Batch reporting infrastructure
- Per-item and batch-level timing, an exact config snapshot, and a summary
  (matched/non-matched/error counts, mean scores) written into the output.
- Per-item error resilience in `run_batch` — one bad item no longer kills
  the whole batch.
- Phase 2 resilience — a single appearance's malformed VLM output no longer
  discards the whole item's Phase 1 results.

### VLM tuning
- `max_new_tokens` `256 → 768` (was truncating mid-JSON on longer
  descriptions, causing parse failures).

### Bug fixes
- CLI argument defaults now derive from the config classes instead of a
  separately hardcoded (and silently stale) copy — this is what caused an
  early full-batch run to silently use threshold `0.4` instead of the
  intended `0.2`.

### Not included on this branch
- The appearance-grouping continuity check and face-based evidence-gating
  fix for the `258`/`269` cross-contamination bug (two different people
  independently clearing the identity threshold and getting merged into one
  person's evidence) were reverted here. That work is complete and verified
  but lives separately on `marvin/mevid-identity-fixes`, pending
  confirmation that an earlier slow/failed batch run was a Colab compute
  budget issue rather than a bug in the fix.
