# Working preferences

## Scientific integrity (highest priority)
This is research. Report results exactly as they are, especially negative ones.

- If something doesn't work, say so plainly. A null or negative result is a valid
  result — report it, don't bury it.
- Never change what is being measured, swap in a different metric, relax a threshold,
  or reword an outcome to make a failure look like a success or "less bad". No
  sugarcoating, no hedging, no ambiguous phrasing that lets a reader assume it worked.
- State the actual number/outcome, not a euphemism for it. E.g. "the shock finder
  recovers M≈1.3–1.5, not the true M=2" — not "lands in the high-error band".
- When a test of a tool/method fails, the finding is that it failed. Do not "fix" the
  appearance by altering inputs, cherry-picking, or reframing. Surface the failure and
  its cause; let Yujie decide what to do.

## Conserve tokens
Long sessions get expensive because the whole conversation is replayed each turn, so
large tool outputs are the main cost. Keep context lean:

- **Images:** the cost is in *reading* an image (it gets tokenized into context and
  stays). Viewing on the user's side is free. So **don't Read frames/plots** — instead
  produce them to a known path and give the user the path to open themselves. Verify
  programmatically (frame counts, file sizes, `sacct` state, value ranges). Only Read an
  image when *you* must judge it for a decision and the user can't do it for you.
- **Logs / listings:** prefer `grep`/`tail`/`head -n`/`wc -l` over dumping whole files
  or big directory listings (e.g. SLURM `.err` are mostly yt INFO spam — grep for
  Traceback/Error/MaxRSS instead of `cat`/`tail -large`).
- **Sub-agents:** use sparingly; their full reports land in context. Ask for the
  conclusion, not the dump.
- Avoid re-reading files already read; trust Edit/Write success instead of re-Reading
  to verify.
- Suggest a fresh session / `/clear` when starting an unrelated task.

## Units — use unyt natively
richio attaches `unyt` units to every field on purpose. Lean on them; don't strip to
cgs everywhere.

- **Don't `.in_cgs()` (or `.in_units(...)`) on every intermediate.** Carry `unyt_array`
  through the arithmetic and let unyt track units. For ratios especially, the units
  cancel on their own and you get a genuinely dimensionless result — no manual
  conversion needed.
- That auto-cancellation is a *feature*: it validates dimensions for free. If a "fraction"
  comes out with leftover units, that's a real dimensional bug — surface it, don't paper
  over it with a conversion. Sprinkling `.in_cgs()` around destroys this check and the
  whole point of richio's unyt integration.
- Only convert at the very end for display, or when a specific unit is genuinely required.

## Naming
- Use meaningful variable names. Avoid terse physics abbreviations like `KE`, `IE`, `P`
  — write `kinetic_energy`, `internal_energy`, `dissipation_power`, etc.

## Notebook style
- Don't number section headings (no `## 1.`, `## 2.`) — just `## Load snapshot`.
- Keep titles super concise, a couple words.
