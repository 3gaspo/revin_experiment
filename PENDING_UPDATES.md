# Pending updates

- 2026-08-29: Standardized CSV missing-value handling: NaNs default to zero
  after aggregation, `error` rejects them, and infinity is always rejected.
  README, architecture, and guideline now state the contract. The focused
  loader regression and documentation validator passed; the broader smoke was
  skipped because the shared runtime lacks `omegaconf`. The three-page
  guideline was rebuilt and visually inspected. Deferred integration: publish
  the code and rerun only missing-affected data identities; finite inputs are
  unchanged, while existing superseded model runs retain their prior rerun
  requirement.

- 2026-08-28: Reconciled the five-view documentation contract by reducing the
  public README to a 103-line goal/setup/execution quickstart and leaving
  formulation, architecture, the four experiment families, protocol, and
  current-versus-historical evidence in their designated views. The validator
  now enforces README ownership, owner links, complete DGX-front coverage, and
  absence of stale LaTeX artifacts. The shared six-project documentation check
  passed; the current three-page guideline was rebuilt and visually inspected,
  and the two-page method note remains current. No scientific rerun or
  executive-summary change is required. Deferred integration: preview code
  sync on DGX, inspect every `*deleting` line, then perform the real sync.

Last successful maintenance: 2026-08-29 00:26 +02:00.

## Pending

- 2026-08-28: Made result transfer tiered: sync now defaults to aggregate
  lightweight analysis artifacts, `detailed` adds row-level/per-run
  diagnostics, and `full` explicitly retrieves binary recovery payloads;
  publication defaults to the same lightweight scope and offers non-binary
  `detailed` output. Affected contracts: both result-transfer scripts, README,
  and focused transfer checks. Git Bash syntax passed for both scripts, the
  two transfer-tier checks passed, all nine active publisher copies were
  byte-identical, and the existing Slurm/publisher checks passed. No
  scientific rerun or LaTeX update is required. Deferred integration: exercise
  each sync tier against Selena and inspect one detailed publication on DGX.

- 2026-08-27: Added a stable oversized-sample header recording the first UTC
  time and file-size reason that the associated artifact became stale on Git.
  Affected contracts: `publish_job.sh`, README publication guidance, and the
  shared focused publisher regressions. All five publisher checks passed; Git
  Bash syntax and byte parity passed for all nine active publisher copies. No
  scientific rerun or LaTeX change is required. Deferred integration: exercise
  one real oversized publication and inspect the generated header on DGX.

Maintenance 2026-08-27: direct loader, configuration, workflows, README,
guideline, handoff, archived-evidence, and placeholder inspection confirmed the
replacement contract and no current-package result. The complementary results
script exited successfully. The three-page guideline compiled in two clean
passes and every rendered page passed visual inspection. Publisher Bash
syntax, byte parity, and the representative regression passed. The prepared-
environment smoke, current-package cluster reruns, lifecycle observation, and
real oversized publication remain pending.

- 2026-08-27: Hardened the thesis-standard publisher against GitHub's
  100 MB file limit. Before staging, each selected non-excluded file above
  100,000,000 bytes is excluded literally and represented by
  `<original>.sample.txt`; text samples contain source metadata and the first
  10% capped at 10,000,000 bytes, while binary samples retain metadata only.
  Affected contracts: publisher, README, shared publication guidance, and the
  five maintained publisher regressions where present. Git Bash syntax passed
  for all nine active copies, all five focused publisher checks passed, and
  both publisher and test copies are byte-identical. No scientific rerun,
  artifact migration, or LaTeX change is required. Deferred integration:
  exercise one real oversized log publication on DGX.

- 2026-08-26: Added QoS `an_preemptable` to all four Selena fronts and aligned
  the shared scheduler contract, README, and focused workflow regression. All
  five direct workflow tests passed. No scientific configuration, artifact
  contract, result, or rerun requirement changed. Deferred integration: mirror
  the updated DGX tree to Selena before the next overflow submission.

- 2026-08-26: Standardized the validated Selena transfer/publication flow.
  `sync_results_to_dgx.sh` now runs on DGX and pulls the isolated Selena trees;
  unscoped publication includes paired `logs_selena/` and lightweight
  `outputs_selena/` under the existing heavy-payload exclusions, while numeric
  job-ID mode remains standard-log-only. Affected contracts: result helper,
  publisher, focused workflow/publisher regressions, README, shared guidance,
  and cluster handoff. Bash syntax passed for all 15 maintained scripts, all
  five publisher checks and all five RevIN workflow checks passed, and the nine
  publisher copies plus five suffix-result helpers are each byte-identical. The
  README changed; the guideline's all-log/lightweight-output wording remains
  accurate, so LaTeX/PDF files are unchanged. No scientific rerun or migration
  is required. Deferred integration: exercise
  one real pull and unscoped publication after a Selena test job.

- 2026-08-26: Added matching Selena fronts for all four RevIN families and
  made runtime roots explicit. `LOGS_ROOT` and `OUTPUTS_ROOT` default to
  `logs/` and `outputs/` but remain overridable; Selena selects
  `logs_selena/` and `outputs_selena/`, and report roots derive from the
  selected output root. Code sync protects both artifact namespaces plus
  cluster-local dependency state, and result sync returns only the
  Selena-named trees without deletion. Affected contracts: shared runner,
  eight fronts, sync pair, ignored placeholders, workflow regression, README,
  local/shared guidance, cluster handoff, and experiment-guideline source/PDF.
  Git Bash syntax and all five focused workflow tests passed; two LaTeX
  passes produced three pages, all visually inspected without clipping or
  overlap. No scientific rerun or artifact migration is required. Deferred
  integration: submit one Selena test front and exercise both sync directions.

- 2026-08-26: Removed all ten historical tracked Slurm log files from active
  `logs/` and preserved `logs/.gitkeep`, matching the workspace policy that
  retired projects alone retain old payloads. Direct inventory confirmed that
  active `logs/` and `outputs/` now contain only their placeholders. The
  cluster handoff was updated; no scientific result, analysis, README/LaTeX
  contract, rerun requirement, or deferred integration work changed.

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

## 2026-08-27 — Replacement CSV exclusions and requeue-capable fronts

- Behavior and affected contracts: `revin.drop_users` now replaces the shared
  dataset default and a non-null run value replaces both; `[]` retains every
  CSV user. All four Selena fronts no longer disable scheduler requeue.
- Focused checks completed: a direct temporary-CSV precedence check and all
  five Slurm workflow tests passed; changed Python compiled and all active
  experiment Bash/Slurm syntax passed. The full smoke remained unavailable
  because the documented runtime lacks OmegaConf, and no environment changed.
- Deferred integration: execute the smoke in the user-prepared environment and
  observe one live Selena requeue lifecycle.
- README/LaTeX and reruns: README, guidance, and guideline source describe the
  replacement contract; re-render during maintenance. The already-required
  current-package reruns must use the new effective exclusions.

## 2026-08-28 — Report-only default artifact transfer

- Behavior and affected contracts: lightweight result sync and publication now
  select logs plus only `outputs*/reports/`, without traversing run or
  diagnostics trees. Detailed/full tiers retain explicit deeper transfer.
- Focused check completed: the shared transfer-tier contract check passed in
  all six active experiment repositories (13 tests total).
- Deferred integration: exercise one real DGX pull and manual publisher run;
  no synchronization, commit, or push was performed locally.
- README/LaTeX and reruns: README and guideline source document the compact
  default; re-render the guideline during maintenance. This transfer-only
  change adds no scientific rerun beyond the already-required current-package
  reruns.

## 2026-08-29 — Terminal Slurm completion records

- Behavior and affected contracts: every RevIN configuration and report
  subtask, both workflow stages, and the final exit trap now emit explicit
  terminal states on success, skip, or failure after manifest finalization.
- Focused check completed: `src/tests/test_slurm_workflow.py` passed (6 tests)
  in the shared thesis runtime.
- Deferred integration: observe the new markers in one successful and one
  failed cluster job; no experiment was launched locally.
- README/LaTeX and reruns: public and scientific behavior are unchanged, so no
  documentation update or additional rerun is required.
