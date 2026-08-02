"""Static contract checks for the resumable Slurm workflows."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SlurmWorkflowTest(unittest.TestCase):
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
        self.assertEqual({path.name for path in ROOT.glob("*.slurm")}, set(fronts))
        for filename, (family, default_mode) in fronts.items():
            front = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn('STAGES="${STAGES:-train,tables}"', front)
            self.assertIn(f"EXPERIMENT_FAMILY={family}", front)
            self.assertIn(f'EXPERIMENT_MODE="${{EXPERIMENT_MODE:-{default_mode}}}"', front)
            self.assertIn('source "$PROJECT_ROOT/src/slurm/run_revin_experiment.sh"', front)
            self.assertIn("#SBATCH --ntasks=1", front)

        self.assertNotIn("RUN_MODE", self.runner)
        self.assertNotIn("full|large", self.runner)
        for mode in ("test)", "small)", "full)", "ultra)"):
            self.assertIn(mode, self.runner)
        self.assertTrue((ROOT / "src/slurm/stage_train.sh").is_file())
        self.assertIn(
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

    def test_completion_contract_reuses_small_in_full(self):
        self.assertIn("run.complete", self.runner)
        self.assertIn("configuration_signature()", self.runner)
        self.assertEqual(self.runner.count('configuration_signature "$dataset"'), 2)
        signature_line = next(
            line for line in self.runner.splitlines() if 'RUN_SIGNATURE="v2|' in line
        )
        self.assertNotIn("mode=", signature_line)
        self.assertNotIn("family=", signature_line)
        self.assertIn("dataset_config_sha256=", signature_line)
        self.assertNotIn('-nt "$seed_root/results.json"', self.runner)
        self.assertIn(
            'WORKFLOW_STATE_DIR="$OUT_ROOT/.workflow/$EXPERIMENT_FAMILY"',
            self.runner,
        )
        self.assertIn(
            'TABLE_OUTPUT_ROOT="$OUT_ROOT/tables/$EXPERIMENT_FAMILY"', self.runner
        )
        self.assertIn('summary_${model}_${split}_mse.tex', self.runner)


if __name__ == "__main__":
    unittest.main()
