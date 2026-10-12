"""Contract tests for utility extraction; no model weights or GPU are used."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import utils


class UtilityContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp.name)
        self.output = io.StringIO()
        self.capture = contextlib.redirect_stdout(self.output)
        self.capture.__enter__()
        utils._SESSION = None
        self.workshop = utils.configure(utils.WorkshopConfig(work_dir=self.work_dir))

    def tearDown(self):
        utils._SESSION = None
        self.capture.__exit__(None, None, None)
        self.temp.cleanup()

    def test_import_has_no_runtime_side_effects(self):
        code = (
            "import os, pathlib; before = dict(os.environ); import utils; "
            "assert utils._SESSION is None; assert dict(os.environ) == before; "
            "assert not list(pathlib.Path('.').iterdir())"
        )
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT)
        with tempfile.TemporaryDirectory() as cwd:
            subprocess.run([sys.executable, '-c', code], cwd=cwd, env=env, check=True)

    def test_reconfigure_preserves_results(self):
        self.assertIsNone(utils.compute_pas(None))
        previous = self.workshop
        current = utils.configure(utils.WorkshopConfig(work_dir=self.work_dir, mss_mode='fp16'))
        self.assertIs(current, previous)
        self.assertEqual(len(current.results), 1)
        self.assertEqual(current.config.mss_mode, 'fp16')

    def test_missing_file_is_failed_not_zero(self):
        self.assertIsNone(utils.compute_ois(self.work_dir / 'missing.mp4'))
        self.assertEqual(self.workshop.results[-1]['status'], 'failed')
        self.assertIsNone(self.workshop.results[-1]['score'])
        self.assertTrue(Path(self.workshop.results[-1]['log']).is_file())

    def test_no_gpu_is_skipped_before_install(self):
        video = self.work_dir / 'input.mp4'
        video.write_bytes(b'file fixture; no decoding should occur')
        self.workshop.ensure_environment = lambda name: self.fail('No-GPU path must not install')
        self.assertIsNone(utils.compute_pas(video))
        self.assertEqual(self.workshop.results[-1]['status'], 'skipped')

    def test_nonhuman_ois_is_not_applicable(self):
        video = self.work_dir / 'dog.mp4'
        video.write_bytes(b'file fixture; no decoding should occur')
        self.workshop.config.video_meta[video.name] = {
            'subject_noun': 'dog', 'subject_type': 'animal', 'prompt': 'A dog moves.'}
        self.assertIsNone(utils.compute_ois(video))
        self.assertEqual(self.workshop.results[-1]['status'], 'not_applicable')

    def test_cache_environment_is_scoped_to_child(self):
        before = dict(os.environ)
        env = self.workshop.evaluator_env()
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(env['HF_HOME'], str(self.workshop.cache / 'huggingface'))
        self.assertEqual(env['UV_CACHE_DIR'], str(self.work_dir / '.cache' / 'uv'))
        self.assertIn(str(self.workshop.repo / 'VideoMAEv2'), env['PYTHONPATH'])

    def test_runner_is_materialized_without_dependencies(self):
        self.assertEqual(self.workshop.runner.read_text(), utils.RUNNER_SOURCE)
        compile(self.workshop.runner.read_text(), 'runner.py', 'exec')

    def test_invalid_configuration_fails_early(self):
        with self.assertRaises(ValueError):
            utils.WorkshopConfig(input_mode='invalid')
        with self.assertRaises(ValueError):
            utils.WorkshopConfig(metrics_to_run=['UNKNOWN'])

    def test_child_command_uses_workshop_cache(self):
        log = self.workshop.logged_command(
            [sys.executable, '-c', 'import os; print(os.environ["HF_HOME"])'], 'child.log')
        self.assertIn(str(self.workshop.cache / 'huggingface'), log.read_text())


if __name__ == '__main__':
    unittest.main()
