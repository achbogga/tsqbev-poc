# Research

Research automation is now enabled in a bounded form.

Current contract:

- dataset scope: `nuScenes v1.0-mini` only
- split scope: `mini_train` / `mini_val`
- experiment count: a small fixed recipe list per invocation
- artifact output: append-only result ledger and summary JSON
- decision policy: keep/discard based on measured validation loss and runtime success

This directory is still not a place for an unbounded self-modifying loop. The implementation is intentionally narrow and evidence-first.
