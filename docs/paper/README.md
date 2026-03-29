# Short Paper

Source:

- `tsqbev_short_paper.tex`

Build command used locally:

```bash
./tectonic --outdir docs/paper/build docs/paper/tsqbev_short_paper.tex
cp docs/paper/build/tsqbev_short_paper.pdf docs/paper/tsqbev_short_paper.pdf
```

The PDF in this directory was built locally after collecting the RTX 5000 measurements and the bounded `nuScenes v1.0-mini` baseline artifacts recorded in:

- `artifacts/benchmarks/rtx5000_full_model_default.json`
- `artifacts/trt/benchmark.json`
- `artifacts/baselines/research_loop/summary.json`
- `artifacts/research_v3/research_loop/results.tsv`
- `artifacts/research_v3/research_loop/summary.json`
- `artifacts/baselines/mini_selected/nuscenes/history.json`
- `artifacts/baselines/mini_selected/eval/nuscenes/metrics_summary.json`
