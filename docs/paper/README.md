# Papers

Current sources:

- `tsqbev_short_paper.tex`
- `tsqbev_frontier_proposal.tex`
- `tsqbev_frontier_program.md`

Recommended build command:

```bash
research/scripts/build_frontier_paper.sh docs/paper/tsqbev_frontier_proposal.tex
```

Expected output:

- `docs/paper/tsqbev_frontier_proposal.pdf`

Supervisor / planner context:

- `docs/paper/tsqbev_frontier_program.md`

The build script prefers `tectonic` on `PATH`, then falls back to `$HOME/.local/bin/tectonic`.
One user-space install path is:

```bash
mkdir -p "$HOME/.local/bin" "$HOME/.cache/tectonic-download"
cd "$HOME/.cache/tectonic-download"
curl -L -o tectonic.tar.gz \
  https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-gnu.tar.gz
tar -xzf tectonic.tar.gz
install -m 0755 tectonic "$HOME/.local/bin/tectonic"
```

Legacy short paper evidence came from:

- `artifacts/benchmarks/rtx5000_full_model_default.json`
- `artifacts/trt/benchmark.json`
- `artifacts/baselines/research_loop/summary.json`
- `artifacts/research_v3/research_loop/results.tsv`
- `artifacts/research_v3/research_loop/summary.json`
- `artifacts/baselines/mini_selected/nuscenes/history.json`
- `artifacts/baselines/mini_selected/eval/nuscenes/metrics_summary.json`

The frontier proposal paper should be read alongside:

- `artifacts/research_v29_continuation_v1/research_loop/summary.json`
- `artifacts/bevfusion_repro/bevfusion_bbox_summary.json`
- `artifacts/foundation_v3_dinov3_teacher_vits16_36ep_v1/epoch022_probe_r4/metrics/nuscenes/metrics_summary.json`
- `artifacts/joint_public_v2_manual_eval/official_eval/epoch_031/nuscenes/metrics/metrics_summary.json`
- `artifacts/knowledge_assets/coverage_summary.json`
- `artifacts/memory/sync_manifest.json`
