# Research

Research automation is now enabled in a bounded form.

Current contract:

- dataset scope: `nuScenes v1.0-mini` only
- split scope: `mini_train` / `mini_val`
- experiment count: a small fixed recipe list per invocation
- optional teacher path: cache-backed external LiDAR teacher providers only
- artifact output: append-only result ledger and summary JSON
- per-recipe evidence: forward latency, official `mini_val` export/eval, and source-mix diagnostics
- decision policy: keep/discard based on `NDS`, then `mAP`, then validation loss

This directory is still not a place for an unbounded self-modifying loop. The implementation is intentionally narrow and evidence-first.
