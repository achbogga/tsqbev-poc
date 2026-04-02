# Autoresearch Supervisor
_Generated: `2026-04-02T22:03:28.113858+00:00`_

## Status
- status: `waiting_external_run`
- branch: `main`
- repo sha: `fda66e5`
- dataset root: `/home/achbogga/projects/research/nuscenes`
- artifact root: `/home/achbogga/projects/tsqbev-poc/artifacts/autoresearch`
- attempted invocations: `0`
- completed invocations: `0`
- memory mode: `server`
- memory embedder: `hash`
- last invocation dir: `-`
- last selected recipe: `-`
- last NDS: `-`
- last mAP: `-`
- last publish status: `-`
- last publish message: `-`

## Notes
- External `tsqbev research-loop` process detected; waiting instead of contending for the same GPU.
- pid `1708596`: `SCREEN -dmS 155042.tsqbev_research_v21_teacher_region_aug bash -ic cd /home/achbogga/projects/tsqbev-poc && mkdir -p artifacts/research_v21_teacher_region_aug_v1 && uv run tsqbev research-loop --dataset-root /home/achbogga/projects/research/nuscenes --artifact-dir /home/achbogga/projects/tsqbev-poc/artifacts/research_v21_teacher_region_aug_v1 --device cuda --max-experiments 5 --teacher-cache-dir /home/achbogga/projects/tsqbev-poc/artifacts/teacher_cache/centerpoint_pointpillar_mini 2>&1 | tee /home/achbogga/projects/tsqbev-poc/artifacts/research_v21_teacher_region_aug_v1/research_loop_v21.log`
- pid `1708598`: `bash -ic cd /home/achbogga/projects/tsqbev-poc && mkdir -p artifacts/research_v21_teacher_region_aug_v1 && uv run tsqbev research-loop --dataset-root /home/achbogga/projects/research/nuscenes --artifact-dir /home/achbogga/projects/tsqbev-poc/artifacts/research_v21_teacher_region_aug_v1 --device cuda --max-experiments 5 --teacher-cache-dir /home/achbogga/projects/tsqbev-poc/artifacts/teacher_cache/centerpoint_pointpillar_mini 2>&1 | tee /home/achbogga/projects/tsqbev-poc/artifacts/research_v21_teacher_region_aug_v1/research_loop_v21.log`
- pid `1709019`: `uv run tsqbev research-loop --dataset-root /home/achbogga/projects/research/nuscenes --artifact-dir /home/achbogga/projects/tsqbev-poc/artifacts/research_v21_teacher_region_aug_v1 --device cuda --max-experiments 5 --teacher-cache-dir /home/achbogga/projects/tsqbev-poc/artifacts/teacher_cache/centerpoint_pointpillar_mini`

## Pointers
- current PI brief: [docs/reports/current.md](docs/reports/current.md)
- supervisor ledger: [artifacts/autoresearch/ledger.jsonl](artifacts/autoresearch/ledger.jsonl)
- supervisor stop file: `artifacts/autoresearch/STOP`
