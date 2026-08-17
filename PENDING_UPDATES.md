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
