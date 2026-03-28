# Research

Research automation is enabled in a bounded form and now follows the strongest transferable ideas
from Andrej Karpathy's public `autoresearch` repo:

- a clear `program.md` contract
- one promoted incumbent per invocation
- append-only experiment ledgers
- explicit keep/discard semantics
- a small comparable budget

Active contract:

- dataset scope: `nuScenes v1.0-mini` only
- split scope: `mini_train` / `mini_val`
- max recipes per invocation: `5`
- fixed comparable train budget per recipe: `max_train_steps = 960`
- loop shape: baseline or carry-over incumbent, bounded exploration, bounded exploitation
- optional teacher path: cache-backed external LiDAR teacher providers only

Required artifacts:

- `results.jsonl`
- `results.tsv`
- `summary.json`
- per-run `manifest.json`
- per-run official `mini_val` eval output
- per-run source-mix diagnostics

This directory is still not a place for an unbounded self-modifying loop. The implementation is
intentionally narrow, evidence-first, and gated before larger compute spend.
