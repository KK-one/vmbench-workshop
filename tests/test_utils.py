"""Contract tests for utility extraction; no model weights or GPU are used."""
import contextlib
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def test_explicit_submodule_path_is_used(self):
        path = self.work_dir / 'vendor' / 'vmbench'
        session = utils.Workshop(utils.WorkshopConfig(work_dir=self.work_dir, repo_dir=path))
        self.assertEqual(session.repo, path.resolve())
        self.assertEqual(session.cache, path.resolve() / '.cache')

    def test_missing_submodule_does_not_clone_another_repository(self):
        path = self.work_dir / 'missing-submodule'
        session = utils.Workshop(utils.WorkshopConfig(work_dir=self.work_dir, repo_dir=path))
        session.logged_command = lambda *a, **k: self.fail('Explicit missing submodule must not clone elsewhere')
        session.install_repository()
        self.assertIn('Submodule', session.setup_errors['repository'])
        self.assertFalse(path.exists())

    def test_changing_repo_path_creates_new_session(self):
        other = utils.configure(utils.WorkshopConfig(work_dir=self.work_dir, repo_dir=self.work_dir / 'vendor'))
        self.assertIsNot(other, self.workshop)

    def initialization_fixture(self, metrics, mss_mode='precomputed', gpu=True):
        session = utils.Workshop(utils.WorkshopConfig(
            work_dir=self.work_dir, repo_dir=self.work_dir / 'vendor' / 'vmbench',
            metrics_to_run=metrics, mss_mode=mss_mode))
        session.install_repository = Mock()
        session.install_system_dependencies = Mock()
        session.install_host_packages = Mock()
        session.ensure_environment = Mock()
        session.ensure_checkpoints = Mock()
        session.load_precomputed = Mock()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: gpu,
            get_device_properties=lambda i: SimpleNamespace(total_memory=15 * 2**30),
            get_device_name=lambda i: 'test-GPU'))
        return session, fake_torch

    def test_initialization_rejects_no_gpu_before_downloads(self):
        session, torch = self.initialization_fixture(['PAS'], gpu=False)
        with patch.dict(sys.modules, {'torch': torch}), patch('utils.subprocess.run') as run:
            run.return_value = subprocess.CompletedProcess([], 0, '', '')
            with self.assertRaisesRegex(RuntimeError, 'No CUDA GPU'):
                session.initialize()
        session.install_system_dependencies.assert_not_called()
        session.ensure_checkpoints.assert_not_called()

    def test_initialization_downloads_selected_weights_and_skips_teacher_mss(self):
        session, torch = self.initialization_fixture(['OIS', 'MSS', 'CAS'])
        with patch.dict(sys.modules, {'torch': torch}), patch('utils.subprocess.run') as run:
            run.return_value = subprocess.CompletedProcess([], 0, '', '')
            self.assertIs(session.initialize(), session)
        self.assertEqual([call.args[0] for call in session.ensure_checkpoints.call_args_list], ['OIS', 'CAS'])
        self.assertEqual([call.args[0] for call in session.ensure_environment.call_args_list], ['core', 'ois'])
        self.assertTrue(session.gpu_available)

    def test_live_mss_initialization_downloads_pas_dependency(self):
        session, torch = self.initialization_fixture(['MSS'], mss_mode='8bit_experimental')
        with patch.dict(sys.modules, {'torch': torch}), patch('utils.subprocess.run') as run:
            run.return_value = subprocess.CompletedProcess([], 0, '', '')
            session.initialize()
        self.assertEqual([call.args[0] for call in session.ensure_checkpoints.call_args_list], ['PAS', 'MSS'])

    def test_initialization_rejects_modified_submodule(self):
        session, torch = self.initialization_fixture(['PAS'])
        with patch.dict(sys.modules, {'torch': torch}), patch('utils.subprocess.run') as run:
            run.return_value = subprocess.CompletedProcess([], 1, '', '')
            with self.assertRaisesRegex(RuntimeError, 'tracked files'):
                session.initialize()
        session.ensure_checkpoints.assert_not_called()

    def test_fp16_mss_on_t4_is_rejected_before_downloads(self):
        session, torch = self.initialization_fixture(['MSS'], mss_mode='fp16')
        with patch.dict(sys.modules, {'torch': torch}), patch('utils.subprocess.run') as run:
            run.return_value = subprocess.CompletedProcess([], 0, '', '')
            with self.assertRaisesRegex(RuntimeError, 'larger GPU'):
                session.initialize()
        session.ensure_checkpoints.assert_not_called()

    def test_prepared_folder_is_recursive_read_only_and_limited(self):
        folder = self.work_dir / 'vids'
        files = [folder / 'a' / 'one.mp4', folder / 'b' / 'two.MOV', folder / 'b' / 'three.webm']
        for video in files:
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b'prepared source fixture')
        (folder / 'labels.txt').write_text('not a video')
        original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
        self.workshop.normalize_uploaded = lambda path: self.fail('Prepared videos must not be converted')
        selected = self.workshop.load_videos(folder=folder, limit=2)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(p in files for p in selected))
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}, original)
        self.assertEqual(self.workshop.config.video_meta, {})

    def test_empty_prepared_folder_reuses_uploaded_video(self):
        folder = self.work_dir / 'vids'; folder.mkdir()
        video = self.workshop.video_dir / 'existing.mp4'; video.write_bytes(b'fixture')
        self.assertEqual(self.workshop.load_videos(folder=folder), [video])

    def test_child_command_uses_workshop_cache(self):
        log = self.workshop.logged_command(
            [sys.executable, '-c', 'import os; print(os.environ["HF_HOME"])'], 'child.log')
        self.assertIn(str(self.workshop.cache / 'huggingface'), log.read_text())

    def test_nested_kaggle_zip_preserves_videos_and_ignores_answer_files(self):
        nested = io.BytesIO()
        with zipfile.ZipFile(nested, 'w') as z:
            z.writestr('contest_vids/0abcde.mp4', b'first video bytes')
            z.writestr('../../a0b1c2.mp4', b'second video bytes')
        archive = self.work_dir / 'kaggle.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('contest_vids.zip', nested.getvalue())
            z.writestr('contest_solution.csv', 'hidden answers')
        dest = self.work_dir / 'extracted'
        paths = utils.unpack_contest_archive(archive, dest)
        self.assertEqual([p.name for p in paths], ['0abcde.mp4', 'a0b1c2.mp4'])
        self.assertEqual(paths[0].read_bytes(), b'first video bytes')
        self.assertEqual(paths[1].read_bytes(), b'second video bytes')
        self.assertEqual(len(list(dest.iterdir())), 2)
        self.assertFalse((self.work_dir / 'a0b1c2.mp4').exists())

    def test_duplicate_contest_ids_in_archive_are_rejected(self):
        archive = self.work_dir / 'duplicate.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('first/abcdef.mp4', b'one')
            z.writestr('second/abcdef.mp4', b'two')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            utils.unpack_contest_archive(archive, self.work_dir / 'dest')

    def test_cached_kaggle_data_needs_no_credentials_or_network(self):
        cache = self.work_dir / 'sample-contest'
        cache.mkdir()
        with zipfile.ZipFile(cache / 'sample-contest.zip', 'w') as z:
            z.writestr('contest_vids/00abcd.mp4', b'video fixture')
        with patch('utils.subprocess.run', side_effect=AssertionError('No network needed')):
            first = utils.download_contest('https://www.kaggle.com/competitions/sample-contest/data', self.work_dir)
            second = utils.download_contest('sample-contest', self.work_dir)
        self.assertEqual(first, second)
        self.assertEqual(utils.contest_video_paths(second)[0].read_bytes(), b'video fixture')

    def test_invitation_url_is_not_used_as_api_competition_id(self):
        with self.assertRaisesRegex(ValueError, 'invitation'):
            utils.download_contest('https://www.kaggle.com/t/test-invitation', self.work_dir)

    def test_binary_submission_preserves_hex_ids_and_covers_all_videos(self):
        paths = [self.work_dir / '00abcd.mp4', self.work_dir / '123456.mp4']
        path = utils.write_binary_submission(paths, self.work_dir / 'baseline.csv')
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows, [{'video_id': '00abcd', 'label': '0'},
                                {'video_id': '123456', 'label': '0'}])
        with self.assertRaisesRegex(ValueError, 'missing'):
            utils.write_binary_submission(paths, path, {'00abcd': 1})
        with self.assertRaisesRegex(ValueError, 'Labels'):
            utils.write_binary_submission(paths, path, {'00abcd': 1, '123456': 0.5})
        with path.open(newline='') as f:
            self.assertEqual(list(csv.DictReader(f)), rows)

    def test_default_t4_metrics_exclude_mss(self):
        self.assertEqual(self.workshop.config.metrics_to_run, ['PAS', 'OIS', 'TCS', 'CAS'])

    def test_kaggle_token_is_only_in_environment_and_redacted_from_log(self):
        token = 'fixture-secret-not-a-real-key'

        def download(command, **kwargs):
            self.assertNotIn(token, command)
            self.assertEqual(kwargs['env']['KAGGLE_API_TOKEN'], token)
            archive = Path(command[command.index('-p') + 1]) / 'test-contest.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('abcdef.mp4', b'video bytes')
            return subprocess.CompletedProcess(command, 0, token, '')

        with patch.dict(os.environ, {'KAGGLE_API_TOKEN': token}), \
             patch('importlib.util.find_spec', return_value=object()), \
             patch('utils.subprocess.run', side_effect=download):
            result = utils.download_contest('test-contest', self.work_dir)
        self.assertTrue((result / 'abcdef.mp4').is_file())
        log = (result.parent / 'download.log').read_text()
        self.assertNotIn(token, log)
        self.assertIn('[REDACTED]', log)


if __name__ == '__main__':
    unittest.main()
