# Short Paper

Source:

- `tsqbev_short_paper.tex`

Build command used locally:

```bash
./tectonic --outdir docs/paper/build docs/paper/tsqbev_short_paper.tex
cp docs/paper/build/tsqbev_short_paper.pdf docs/paper/tsqbev_short_paper.pdf
```

The PDF in this directory was built locally after collecting the RTX 5000 measurements recorded in:

- `artifacts/benchmarks/rtx5000_full_model_default.json`
- `artifacts/trt/benchmark.json`
