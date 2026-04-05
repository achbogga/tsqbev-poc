# Frontier Knowledge Base

The repo now treats `research/knowledge/` as a structured research database, not just note
storage.

Current packs:

- `mit_han_efficient_ml_kb.json`
- `mit_han_frontier_systems_kb.json`
- `frontier_vision_foundations_kb.json`
- `frontier_fusion_temporal_mechanics_kb.json`
- `frontier_distillation_playbook_kb.json`
- `frontier_efficiency_mechanics_kb.json`
- `nvidia_frontier_autonomy_efficiency_kb.json`

Rules:

- only primary sources are authoritative
- every technique card should explain intuition, ablations, `apply_when`, `avoid_when`, failure
  modes, and concrete `tsqbev_actions`
- informal repo phrases should be stored as aliases and mapped back to official source terms
- the knowledge base should answer both:
  - `what does this technique actually do?`
  - `where does it belong in our stack: control plane, teacher, student, or deployment layer?`

Local asset mirroring:

- run `uv run tsqbev knowledge-assets-sync`
- mirrored assets live outside the repo under `/home/achbogga/projects/research_assets`
- the mirror is URL-deduped so repeated references across cards share one cached asset
- generated status lives in `artifacts/knowledge_assets/manifest.json`

The memory stack indexes both the structured cards and the generated asset manifests so the
planner/critic can retrieve original papers, repos, and checkpoints quickly.

Current research policy:

- build the knowledge base and source mirror first
- derive breakthrough hypotheses from the curated packs
- only then resume the local research loop with architecture changes tied back to the cards
