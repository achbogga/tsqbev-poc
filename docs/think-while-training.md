# Think While Training

This document defines what the lab should do while GPU experiments are running. The goal is to
increase confidence in the research program without contaminating the active run or wasting GPU.

## Principle

When a run is in flight, the lab should still be working. But the work must be CPU-side,
evidence-producing, and decision-relevant.

Allowed during active training:

- reading and summarizing primary sources
- improving the local knowledge base and memory layer
- auditing retrieval quality and proposal quality
- checking launch contracts, kill switches, and pivot rules
- writing next-step research agendas and control documents
- cleaning repo health and instrumentation gaps

Not allowed during active training unless the run is already invalid:

- disruptive changes to the active training branch
- ad hoc GPU contention
- speculative code churn that is not tied to a tangible hypothesis

## Required Outputs Per Active Run

Every active run should have at least one of these produced in parallel:

1. `confidence_review.md`
   - why the current branch should continue
   - what evidence would kill it
   - what would count as a breakthrough

2. `retrieval_audit.md`
   - whether memory returned the right cards, evidence, and failure signatures
   - which missing sources or aliases should be added

3. `next_step_shortlist.md`
   - top 3 bounded next moves
   - expected ROI
   - integration cost
   - kill condition

4. `infra_gap_note.md`
   - concrete repo/control-plane problems discovered during the run
   - exact fix priority

If none of these outputs are tangible, do not force activity for its own sake.

## Confidence Axes

The lab should keep assessing confidence on these axes:

### 1. Research Program
- are we targeting the real bottleneck?
- is the architecture family still justified?
- are we learning from failures or just iterating near the same point?

### 2. Control Plane
- does the planner propose the right family?
- does the critic reject rabbit holes?
- does the executor launch what the proposal actually asked for?

### 3. Kill / Pivot Logic
- do catastrophic failures stop quickly?
- do repeated stalls actually force a pivot?
- are we preserving failure signatures in memory?

### 4. Memory / Retrieval
- are exact records current?
- is retrieval surfacing the right papers, code, and repo-local evidence?
- are aliases and source terms mapped cleanly?

### 5. Model / Training
- is the branch actually testing the thesis, or a proxy?
- are official metrics available soon enough to matter?
- are controls and ablations honest?

### 6. Repo Infra
- are docs aligned with reality?
- can another contributor understand the active thesis from repo files alone?
- are release/bootstrap paths still valid?

## Common Practice

Treat the following as routine:

1. While a run is active, do one CPU-only confidence-building task.
2. Store the result in repo docs or memory, not in chat only.
3. If the task finds a structural flaw, queue the fix for immediately after the run or stop the
   run if the flaw already invalidates it.
4. Before the next launch, convert the best parallel-thinking artifact into a concrete checklist
   item for the supervisor.

## Tangibility Rule

Planning is only worth doing if it changes one of these:

- what we run next
- what we stop running
- what we retrieve from memory
- what we measure
- what we publish as the thesis

If the planning output does not change any of the above, it is not a good use of time.
