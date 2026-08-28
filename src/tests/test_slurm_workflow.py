"""Static contract checks for the resumable Slurm workflows."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SlurmWorkflowTest(unittest.TestCase):
    def test_cluster_sync_scripts(self):
        code = (ROOT / "sync_code_to_selena.sh").read_text(encoding="utf-8")
        results = (ROOT / "sync_results_to_dgx.sh").read_text(encoding="utf-8")
        for script in (code, results):
            self.assertIn('PROJECT_NAME="$(basename "$PROJECT_ROOT")"', script)
            self.assertIn("sed -n '1p'", script)
        for excluded in (
            ".git/",
            ".venv/",
            ".secrets/",
            "pyproject.toml",
            "uv.lock",
            "datasets/",
            "weights/",
            "outputs/",
            "logs/",
        ):
            self.assertIn(f"--exclude='{excluded}'", code)
        self.assertIn("selena.hpc.edf.fr", code)
        self.assertIn("--delete", code)
        self.assertNotIn("dgx-front.retd.edf.fr", results)
        self.assertIn(
            'SOURCE_ROOT="$nni@selena.hpc.edf.fr:~/codes/$PROJECT_NAME"',
            results,
        )
        self.assertIn('DESTINATION_ROOT="$PROJECT_ROOT"', results)
        self.assertIn('mkdir -p "$DESTINATION_ROOT/outputs_selena"', results)
        self.assertIn("--include='outputs_selena/.gitkeep'", code)
        self.assertIn("--exclude='outputs_selena/***'", code)
        self.assertIn("--include='logs_selena/.gitkeep'", code)
        self.assertIn("--exclude='logs_selena/***'", code)
        self.assertIn('"$SOURCE_ROOT/outputs_selena/"', results)
        self.assertIn('"$SOURCE_ROOT/logs_selena/"', results)
        self.assertIn("pulled from Selena to DGX", results)
        self.assertNotIn("--delete", results)

    def setUp(self):
        self.runner = (ROOT / "src/slurm/run_revin_experiment.sh").read_text(
            encoding="utf-8"
        )

    def test_family_fronts_and_internal_stages(self):
        fronts = {
            "revin.slurm": ("core", "test"),
            "nmse.slurm": ("nmse", "small"),
            "exotic.slurm": ("exotic", "small"),
            "min.slurm": ("min", "small"),
        }
        self.assertEqual(
            {path.name for path in ROOT.glob("*.slurm")},
            set(fronts) | {name.replace(".slurm", "_selena.slurm") for name in fronts},
        )
        for filename, (family, default_mode) in fronts.items():
            front = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn('STAGES="${STAGES:-train,tables}"', front)
            self.assertIn(f"EXPERIMENT_FAMILY={family}", front)
            self.assertIn(f'EXPERIMENT_MODE="${{EXPERIMENT_MODE:-{default_mode}}}"', front)
            self.assertIn('source "$PROJECT_ROOT/src/slurm/run_revin_experiment.sh"', front)
            self.assertIn("#SBATCH --ntasks=1", front)
            selena_path = ROOT / filename.replace(".slurm", "_selena.slurm")
            selena = selena_path.read_text(encoding="utf-8")
            self.assertIn(f"EXPERIMENT_FAMILY={family}", selena)
            self.assertIn(f'EXPERIMENT_MODE="${{EXPERIMENT_MODE:-{default_mode}}}"', selena)
            self.assertIn("#SBATCH --partition=an", selena)
            self.assertIn("#SBATCH --qos=an_preemptable", selena)
            self.assertIn("#SBATCH --output=logs_selena/%x_%j.out", selena)
            self.assertIn("#SBATCH --exclusive", selena)
            self.assertNotIn("#SBATCH --no-requeue", selena)
            self.assertIn("#SBATCH --wckey=P12CU:DATASCIENCE", selena)
            self.assertIn('OUTPUTS_ROOT="$PROJECT_ROOT/outputs_selena"', selena)
            self.assertIn('LOGS_ROOT="$PROJECT_ROOT/logs_selena"', selena)
            self.assertIn('EXPERIMENT_LAUNCH_ID="selena_${SLURM_JOB_ID', selena)

        self.assertNotIn("RUN_MODE", self.runner)
        self.assertNotIn("full|large", self.runner)
        self.assertIn('LOGS_ROOT="${LOGS_ROOT:-$ROOT/logs}"', self.runner)
        self.assertIn('OUTPUTS_ROOT="${OUTPUTS_ROOT:-$ROOT/outputs}"', self.runner)
        self.assertIn('DEFAULT_OUT_ROOT="$OUTPUTS_ROOT/$EXPERIMENT_FAMILY"', self.runner)
        for mode in ("test)", "small)", "full)", "ultra)"):
            self.assertIn(mode, self.runner)
        self.assertIn(
            "srun --ntasks=1 python -m scripts.experiment", self.runner
        )
        self.assertIn("srun --ntasks=1 python -m scripts.report", self.runner)
        self.assertTrue((ROOT / "src/slurm/stage_train.sh").is_file())
        self.assertNotIn(
            "tables.complete", (ROOT / "src/slurm/stage_tables.sh").read_text()
        )

    def test_modes_differ_only_by_datasets_then_models(self):
        self.assertIn(
            'DEFAULT_SETTINGS="168:24 336:48 504:168 336:96 336:720"',
            self.runner,
        )
        self.assertIn('DEFAULT_DATASETS="traffic electricity solar"', self.runner)
        publication_datasets = (
            'DEFAULT_DATASETS="traffic electricity solar weather exchange_rate '
            'ETTh1 ETTh2 ETTm1 ETTm2"'
        )
        self.assertEqual(self.runner.count(publication_datasets), 2)
        self.assertEqual(self.runner.count('DEFAULT_MODELS="patchtst"'), 3)
        self.assertIn('DEFAULT_MODELS="dlinear patchtst"', self.runner)

    def test_method_families(self):
        expected = {
            "CORE_METHODS": (
                "none_mse standard_mse instance_mse instance_nmse "
                "revin_mse revin_nmse"
            ),
            "NMSE_METHODS": "none_nmse standard_nmse",
            "EXOTIC_METHODS": (
                "mean_only_mse mean_only_nmse scale_only_mse scale_only_nmse "
                "median_mad_mse median_mad_nmse revin_last_mse revin_last_nmse "
                "revin_arcsinh_mse revin_arcsinh_nmse"
            ),
            "MIN_METHODS": "min_mse min_nmse",
            "TEST_METHODS": "standard_mse instance_mse instance_nmse min_nmse",
        }
        for variable, methods in expected.items():
            self.assertIn(f'{variable}="{methods}"', self.runner)

    def test_manifest_contract_reuses_matching_paths_across_modes(self):
        self.assertNotIn("run.complete", self.runner)
        self.assertIn("python -m pipeline.runs allocate", self.runner)
        self.assertIn("python -m pipeline.runs pending-seeds", self.runner)
        self.assertIn("python -m pipeline.runs status", self.runner)
        self.assertIn("--status ready", self.runner)
        self.assertIn("python -m pipeline.runs ready", self.runner)
        self.assertIn("python -m pipeline.runs complete-launch", self.runner)
        self.assertIn("python -m pipeline.runs complete --run-dir", self.runner)
        self.assertNotIn("--status completed", self.runner)
        self.assertIn('--input "dataset_config=$dataset_config"', self.runner)
        self.assertIn('--mode "$EXPERIMENT_MODE"', self.runner)
        self.assertIn(
            'identity_root="$OUT_ROOT/$dataset/${L}_${H}/${model,,}/${normalization,,}/${loss,,}"',
            self.runner,
        )
        self.assertIn(
            'TABLE_OUTPUT_ROOT="${TABLE_OUTPUT_ROOT:-$OUTPUTS_ROOT/reports/$EXPERIMENT_FAMILY/$EXPERIMENT_MODE}"',
            self.runner,
        )
        self.assertIn('summary_${model}_${split}_mse.tex', self.runner)


if __name__ == "__main__":
    unittest.main()
