# v2 Build Log

A running log of the v2 lakehouse rebuild: what I did, what broke, and why.

The "what broke" entries are the point. They become the **Engineering lessons** section of the
README, which is the most credible thing in the repo — it's the part that can't be generated.

Format: one entry per working session. Date, what I was trying to do, what happened.

---

## 2026-08-21 — Phase 0: planning

Tagged v1 as `v1.0` and branched `v2-lakehouse` so the old pipeline stays runnable and
reviewable while the rewrite happens.

Decided the rewrite is worth doing for one reason: **v1's fact grain was wrong.** One row per
card meant ~79k rows after filtering, which fits in pandas — so using Spark was decorative.
Re-graining to `card × game_format × provider × list_type × finish × date` off MTGJSON's
90-day price history takes that into the tens of millions and makes the distributed tooling
actually load-bearing.

Wrote `docs/ARCHITECTURE.md` before installing anything, on the theory that if I can't explain
why each component is in the stack, it shouldn't be in the stack.

Decisions made:

- Full lakehouse stack, but added **strictly one component at a time**, each proved working
  and committed before the next goes in.
- If any single component costs more than half a day of configuration, drop it and document
  the fallback rather than sinking the schedule.
- No synthetic data, ever. v1 had "graceful fallbacks to synthetic data" — that is a
  liability, not a feature. Pipelines fail loudly instead.

Nothing broken yet.
