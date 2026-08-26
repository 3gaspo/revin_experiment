# Pending updates

Last successful maintenance: 2026-08-11 10:45 +02:00.

## Pending

- 2026-08-17: Simplify `publish_job.sh`: a numeric job ID now selects only its
  exact stdout/stderr pair, while an omitted ID stages the `logs/` and
  lightweight `outputs/` parent trees directly. Publisher, focused contract
  test, README, and shared guidance changed. The project publisher contract
  test and Git Bash syntax passed, and all nine copies have matching SHA-256
  hashes. No scientific rerun or artifact migration is required. Deferred
  maintenance: reconcile and render the experiment guideline; retain the
  existing real-cluster publisher integration check.

- 2026-08-16: Adopt the thesis-standard `publish_job.sh`: source the proxy and
  fast-forward pull `origin/main` before artifact selection, staging, or commit,
  then publish only the lightweight selected paths. Affected contracts:
  publisher, focused contract test, README, and shared experiment guidance.
  Checks passed: Bash syntax for all nine standard copies, matching SHA-256
  hashes, and the RevIN publisher contract test. No scientific rerun or artifact
  migration is required. Deferred maintenance: reconcile
  `latex/experiment_guideline.tex` and exercise one real cluster publish with a
  remote update present.

- 2026-08-12: Synchronize Adaptation's terminal lifecycle: remove automatic
  publisher submission, add the manual root `publish_job.sh`, restrict overall
  manifests to `not_run|running|interrupted|completed`, and allow result
  aggregation to consume seed-ready artifacts only from its own active launch.
  Affected files/contracts: manifest helper, RevIN runner, result reader,
  publisher files, focused tests, README, and parent experiment guidance.
  Checks passed: 14 focused lifecycle/publisher/Slurm/result tests and Bash
  syntax for the runner and manual publisher. No scientific rerun or artifact
  migration is required. Deferred maintenance: reconcile and render
  `latex/experiment_guideline.tex`; cluster-check one successful and one
  failed/cancelled launch, then run the manual publisher once.
  Maintenance 2026-08-13: direct inspection confirmed the four-state overall
  manifest, seed-only `ready`, same-launch result selection, and final exit
  promotion. `src/tests/results_test.py` passed in the shared thesis runtime;
  this focused aggregation boundary was repeated because the prepared `uv`
  environment is unavailable and the broader CPU smoke requires OmegaConf,
  which the shared runtime does not provide. The README and guideline now
  describe the exact lifecycle and manual publisher. Two pdfLaTeX passes
  completed without warnings, and all three rendered pages passed visual
  inspection. Remaining blocker: observe one successful and one
  failed/cancelled cluster launch, then run `publish_job.sh` once.

- 2026-08-13: Complete every successful RevIN configuration immediately,
  preserve it across later workflow failure, interrupt only unfinished runs,
  and retain per-seed artifact lists. Affected contracts: shared manifest
  helper, runner, tests, README, and experiment guideline. Checks passed: 11
  lifecycle tests, publisher and four workflow contracts, Python AST parsing,
  Bash syntax, clean LaTeX compilation, and visual inspection of all three PDF
  pages. No artifact migration, scientific rerun, or schema bump is required.
  Remaining cluster work: exercise successful and failed/cancelled launches and
  run the manual publisher once.

Maintenance 2026-08-16: no source, configuration, artifact, documentation, or
cluster-handoff file changed after the previous pass. Direct inspection again
found completed-only reuse, same-launch ready aggregation, and the manual
publisher contract. The already successful result aggregation and PDF checks
were not repeated because there is no changed integration boundary. Live
successful/failed launch observations and one manual publisher run remain the
sole blocker; no scientific rerun is required.

Maintenance 2026-08-17: direct inspection found no new source, artifact, or
cluster-status change and reconfirmed completed-only reuse and same-launch ready
aggregation. The README was current; the experiment guideline was reconciled
with the canonical proxy-first, fast-forward-pull publisher. Bash syntax passed
for all nine byte-identical copies. Three pdfLaTeX passes completed with a clean
log, and all three rendered guideline pages passed visual inspection. The prior
aggregation/lifecycle checks were not repeated because those boundaries did not
change. Live successful/failed launch observations and one real publisher run
remain the blockers; no scientific rerun is required.

Maintenance 2026-08-18: direct inspection confirmed that the shared manifest
helper is byte-identical to the canonical schema-1 copies and that the 94
current migrated runs declare no upstream dependency; the already successful
13 focused manifest tests therefore close that standalone entry. The README
was current, and the experiment guideline was corrected to describe exact-log
job publication and unscoped lightweight-tree publication. Git Bash syntax
passed for all nine byte-identical publishers. Aggregation/training tests were
not repeated because those boundaries did not change. Three pdfLaTeX passes
completed with a clean log, and all three rendered guideline pages passed
visual inspection. Live successful/failed launch observations and one real
publisher run remain the blockers; no
scientific rerun is required.

Maintenance 2026-08-19: direct inspection found no source, configuration,
new artifact, or cluster-status change; the 94 migrated configurations and
their documented scope are unchanged. The README and guideline remain current,
and all nine publisher copies remain byte-identical at SHA-256
`0A9E87E51517B9F5816BB92CDE726B9E383AB6B8A70DC251FEF429BF7B53B45C`.
Aggregation/training, lifecycle, Bash-syntax, and PDF checks were not repeated
because no corresponding boundary changed. Live successful and failed or
cancelled launch observations plus one real publisher run remain the blockers;
no scientific rerun is required.

Maintenance 2026-08-20: direct timestamp, source, artifact, and cluster-handoff
inspection found no change after the previous pass; the 94 migrated
configurations and their analyzed scope are unchanged. The README and guideline
remain current, and the publisher remains byte-identical across all nine
projects at SHA-256
`0A9E87E51517B9F5816BB92CDE726B9E383AB6B8A70DC251FEF429BF7B53B45C`.
Aggregation/training, lifecycle, Bash-syntax, and PDF checks were deliberately
skipped because no corresponding boundary changed. Live successful and failed
or cancelled launch observations plus one real publisher run remain the
blockers; no scientific rerun is required.

Maintenance 2026-08-23: direct inspection confirmed the shared nested
selection and deterministic latest-run behavior, and the helper plus focused
test file are byte-identical to the other four maintained copies. The
complementary dependency-free `src/tests/results_test.py` aggregation consumer
passed in the shared thesis runtime. README selection documentation and the 94
migrated-run scope remain current; no LaTeX, result claim, migration, or rerun
changed. The selector entry is resolved, while live lifecycle and publisher
checks remain pending.

Maintenance 2026-08-24: direct package, import, Slurm, test, migrated-artifact,
and cluster-handoff inspection confirmed clean proposal/external-model
boundaries, cohesive owners, and no compatibility paths. As complementary
scientific coverage, `RevIN` and identity-initialized `MIN` both completed a
forward/inverse tensor round trip in the shared thesis runtime. README, both
LaTeX documents, the 94 migrated-run scope, and rerun requirements are
unchanged, so the reorganization and guidance entries are resolved. The direct
training smoke remains deliberately skipped because the documented runtime
lacks OmegaConf; live lifecycle and publisher checks remain pending.

## 2026-08-24 — Pinned PatchTST and DLinear source packages

- Behavior and affected contracts: replaced the divergent flat external-model
  files with byte-identical pinned PatchTST and DLinear source-adapted packages
  and one common lags, dim, horizon tensor boundary.
- Focused checks and outcomes: Python compilation, external-package layout and
  revision guards, and direct multivariate PatchTST/DLinear forwards passed.
  The full smoke was not runnable because OmegaConf is absent from the shared
  runtime; no dependency was installed.
- Deferred integration: execute the focused experiment smoke in the prepared
  project environment and compare fresh cluster outputs before reuse.
- README/LaTeX and reruns: README and local guidance document exact provenance.
  The 94 migrated configurations were produced by the superseded model code and
  must not be treated as current-contract comparisons; rerun every affected
  DLinear/PatchTST configuration and reconcile the guideline and executive
  summary during maintenance.

Maintenance 2026-08-25: direct source-package, selector, workflow, manifest,
artifact, README, guideline, summary, and handoff inspection confirmed that all
94 schema-1 manifests are complete but scientifically superseded by the pinned
PatchTST/DLinear packages. The complementary
`src/tests/test_slurm_workflow.py` check passed (4 tests), covering current
launcher selection and recovery without repeating the recorded model forwards.
The direct experiment smoke remains inapplicable because the documented runtime
lacks OmegaConf. README and cluster instructions now require
`RUN_CONFLICT_POLICY=new`; the guideline and executive summary explicitly scope
the old numbers as historical evidence. Two pdfLaTeX passes per document
produced clean three- and two-page PDFs, and every page passed visual
inspection. Current-package test and full reruns, followed by new artifact
analysis, remain required; live lifecycle and publisher checks also remain
pending.

Maintenance 2026-08-26: direct assertion, workflow-stage, ignore-rule, README,
LaTeX, archived-evidence, and handoff inspection confirmed that the change is
test/housekeeping-only. Dependency-light compilation of
`src/tests/test_slurm_workflow.py` passed and `.venv/` is ignored; the focused
workflow and workspace-wide Bash checks were not repeated. The entry is
resolved. Current-package cluster reruns, new artifact analysis, lifecycle
observations, and publisher validation remain pending.
