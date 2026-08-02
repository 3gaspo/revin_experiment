"""Static contract checks for the resumable Slurm workflow."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SlurmWorkflowTest(unittest.TestCase):
    def test_complete_front_and_internal_stages(self):
        front = (ROOT / "revin.slurm").read_text(encoding="utf-8")
        runner = (ROOT / "src/slurm/run_revin_experiment.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('STAGES="${STAGES:-train,tables}"', front)
        self.assertNotIn("RUN_MODE", front + runner)
        self.assertNotIn("full|large", runner)
        for mode in ("test)", "small)", "full)", "ultra)"):
            self.assertIn(mode, runner)
        self.assertIn('DEFAULT_SMALL_SETTINGS="168:24 336:48 504:168"', runner)
        self.assertIn(
            'DEFAULT_FULL_SETTINGS="$DEFAULT_SMALL_SETTINGS 336:96 336:720"',
            runner,
        )
        self.assertIn('DEFAULT_DATASETS="electricity"', runner)
        self.assertIn('DEFAULT_SETTINGS="504:168"', runner)
        self.assertIn('DEFAULT_SEEDS="1"', runner)
        self.assertIn(
            'DEFAULT_DATASETS="traffic electricity solar weather exchange_rate"',
            runner,
        )
        self.assertEqual(
            runner.count(
                'DEFAULT_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 traffic electricity solar weather exchange_rate"'
            ),
            2,
        )
        self.assertIn("run.complete", runner)
        self.assertIn("configuration_signature()", runner)
        self.assertEqual(runner.count("configuration_signature \"$dataset\""), 2)
        self.assertIn('summary_${model}_${split}_mse.tex', runner)
        self.assertIn("tables.complete", (ROOT / "src/slurm/stage_tables.sh").read_text())
        self.assertTrue((ROOT / "src/slurm/stage_train.sh").is_file())


if __name__ == "__main__":
    unittest.main()
