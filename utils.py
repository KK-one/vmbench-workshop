"""Utilities for the VMBench student workshop.

Import is side-effect free: no installation, downloads, GPU initialization or directory creation.
Call configure(WorkshopConfig(...)) before using notebook convenience functions.
The original metric adapter is preserved in RUNNER_SOURCE and runs in a separate process.
Optional notebook dependencies are imported only when their features are used.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path


CORE_PACKAGES = ['numpy==1.26.4', 'transformers==4.36.1', 'tokenizers==0.15.0', 'huggingface-hub==0.25.2', 'timm==0.6.13', 'accelerate==0.21.0', 'sentencepiece==0.1.99', 'einops==0.6.1', 'einops-exts==0.0.4', 'opencv-python-headless==4.10.0.84', 'pandas==2.2.3', 'scipy==1.14.1', 'matplotlib==3.9.2', 'pillow==10.4.0', 'decord==0.6.0', 'supervision==0.22.0', 'hydra-core==1.3.2', 'iopath==0.1.10', 'imageio==2.36.1', 'imageio-ffmpeg==0.6.0', 'addict==2.4.0', 'yapf==0.40.1', 'pycocotools==2.0.8', 'tensorboardX==2.6.2.2', 'icecream==2.1.3', 'setuptools==75.8.2', 'wheel==0.45.1', 'tqdm==4.67.1']

OIS_PACKAGES = ['numpy==1.26.4', 'mmengine==0.10.7', 'mmdet==3.2.0', 'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl', 'opencv-python==4.10.0.84', 'scipy==1.14.1', 'matplotlib==3.9.2', 'pandas==2.2.3', 'json-tricks==3.17.3', 'munkres==1.1.4', 'xtcocotools==1.14.3', 'einops==0.6.1', 'timm==0.6.13', 'transformers==4.36.1', 'huggingface-hub==0.25.2', 'setuptools==75.8.2', 'wheel==0.45.1']

SOURCE_PATHS = ['.', 'Grounded-Segment-Anything', 'Grounded-Segment-Anything/GroundingDINO', 'Grounded-Segment-Anything/segment_anything', 'Grounded-SAM-2', 'co-tracker', 'Q-Align', 'VideoMAEv2', 'mmpose']

HF_REVISIONS = {'GD-ML/VMBench': '437cf3b7c667cd23e3f1e24e19e0af7868088907', 'q-future/one-align': 'dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb', 'google-bert/bert-base-uncased': '86b5e0934494bd15c9632b12f734a8a67f723594'}

HF_SIZES = {'groundingdino_swinb_cogcoor.pth': 938057991, 'sam_vit_h_4b8939.pth': 2564550879, 'sam2.1_hiera_large.pt': 898083611, 'scaled_offline.pth': 101890938, 'vit_g_vmbench.pt': 14162848906, 'model.safetensors': 440449768, 'pytorch_model-00001-of-00002.bin': 9991591698, 'pytorch_model-00002-of-00002.bin': 6417830970}

OIS_URLS = {'rtmdet_person.pth': 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth', 'rtmpose_body7.pth': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'}

OIS_SIZES = {'rtmdet_person.pth': 99021988, 'rtmpose_body7.pth': 54551099}

SCORE_KEYS = {'PAS': 'perceptible_amplitude_score', 'OIS': 'object_integrity_score', 'TCS': 'temporal_coherence_score', 'MSS': 'motion_smoothness_score', 'CAS': 'commonsense_adherence_score'}

METRIC_NAMES = ['PAS', 'OIS', 'TCS', 'MSS', 'CAS']

LABELS = {'REAL_NORMAL', 'GENERATED', 'FAST', 'SLOW'}

RUNNER_SOURCE = r'''
import argparse, os, sys, json, gc, traceback
from pathlib import Path
import torch

class NotApplicable(RuntimeError):
    pass

def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Версия upstream отличается: patch anchor ' + repr(old))
    return source.replace(old, new, 1)

def run_official(name, argv, patches=()):
    filename = Path.cwd() / name
    source = filename.read_text()
    for old, new in patches:
        source = replace_once(source, old, new)
    sys.argv = [str(filename), *argv]
    namespace = {'__name__': '__main__', '__file__': str(filename),
                 'NotApplicable': NotApplicable}
    exec(compile(source, str(filename), 'exec'), namespace)

def prepare_dino_fallback():
    # Namespace matches the official scripts' import.
    from GroundingDINO.groundingdino.models.GroundingDINO import ms_deform_attn
    if not hasattr(ms_deform_attn, '_C'):
        original_class = ms_deform_attn.MultiScaleDeformableAttention
        source = Path(ms_deform_attn.__file__).read_text()
        source = replace_once(source,
            'if torch.cuda.is_available() and value.is_cuda:',
            "if torch.cuda.is_available() and value.is_cuda and '_C' in globals():")
        exec(compile(source, ms_deform_attn.__file__, 'exec'), ms_deform_attn.__dict__)
        # Transformer already holds this class by reference: update its method in place.
        original_class.forward = ms_deform_attn.MultiScaleDeformableAttention.forward
        ms_deform_attn.MultiScaleDeformableAttention = original_class
        print('GroundingDINO: official PyTorch deformable-attention fallback on GPU.')

def pas(meta):
    prepare_dino_fallback()
    run_official('perceptible_amplitude_score.py', [
        '--meta_info_path', meta, '--box_threshold', '0.25', '--text_threshold', '0.20',
        '--grid_size', '30', '--device', 'cuda'], [
        ('sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])',
         'sam_predictor.transform.apply_boxes_torch(boxes_filt, image_array.shape[:2])'),
        ("# save meta info per video", "meta_info['_pas_detected'] = int(boxes_filt.shape[0])\n        # save meta info per video")])

def ois(meta):
    run_official('object_integrity_score.py', [
        '--meta-info-path', meta, '--save-predictions', '--device', 'cuda:0',
        '--det_checkpoint', str(Path('.cache/rtmdet_person.pth').resolve()),
        '--pose_checkpoint', str(Path('.cache/rtmpose_body7.pth').resolve())], [
        ('# predict keypoints',
         "if len(bboxes) == 0:\n        raise NotApplicable('No person bbox in at least one frame; human OIS is not applicable.')\n\n    # predict keypoints"),
        ('# show the results',
         "if not pose_results or not any(len(r.pred_instances.keypoints) for r in pose_results):\n        raise NotApplicable('No pose instances detected.')\n\n    # show the results")])

def tcs(meta):
    prepare_dino_fallback()
    from sam2 import build_sam
    try:
        from sam2 import _C
        print('SAM2 CUDA extension: available')
    except ImportError:
        print('WARNING: SAM2 CUDA extension unavailable; upstream skips hole filling.')
    # Video and state offload are official SAM2 init_state arguments.
    patches = [
        ('offload_video_to_cpu=True, async_loading_frames=True',
         'offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True'),
        ("meta_info['temporal_coherence_score'] = (vanish_score + emerge_score) / 2",
         "meta_info['_tcs_objects'] = objects_count\n        meta_info['temporal_coherence_score'] = (vanish_score + emerge_score) / 2")]
    patches.append(('torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()',
                    'torch.autocast(device_type="cuda", enabled=False).__enter__()'))
    print('T4-compatible TCS: BF16 autocast disabled; FP32 on every GPU.')
    run_official('temporal_coherence_score.py', [
        '--meta_info_path', meta, '--box_threshold', '0.25', '--text_threshold', '0.20',
        '--grid_size', '50', '--device', 'cuda'], patches)

def mss(meta, mode):
    import motion_smoothness_score as official
    if mode == '8bit_experimental':
        # Official builder exposes load_8bit; no alternate scoring model.
        original_loader = official.load_pretrained_model
        def low_memory_loader(*args, **kwargs):
            kwargs.update(load_8bit=True)
            return original_loader(*args, **kwargs)
        official.load_pretrained_model = low_memory_loader
        print('WARNING: experimental 8-bit Q-Align; numerical results may differ.')
    scorer = official.QAlignVideoScorer(pretrained='.cache/q-future/one-align', device='cuda:0').eval()
    with open(meta) as stream:
        records = json.load(stream)
    for item in records:
        windows = official.load_video_sliding_window(item['filepath'], window_size=3)
        scores = []
        for window in windows:
            _, _, score = scorer([window])
            scores.append(float(score.item()))
        threshold = official.set_threshold(item['perceptible_amplitude_score'])
        artifacts = official.get_artifacts_frames(scores, threshold)
        item['motion_smoothness_score'] = 1 - len(artifacts) / len(scores)
    Path(meta).write_text(json.dumps(records, indent=2))
    del scorer

def cas(meta):
    # Import from vendored VideoMAEv2 before importing its utility namespace.
    import models
    from dataset import build_dataset
    from accelerate import init_empty_weights
    import bench_utils.cas_utils as official
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    import csv
    directory = Path(meta).parent
    records = json.loads(Path(meta).read_text())
    with (directory / 'test.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter=' ')
        for item in records:
            writer.writerow([item['filepath'], 0])
    args = SimpleNamespace(data_path=str(directory), data_root='',
        data_set='Commonsense-Adherence', sparse_sample=False, nb_classes=5,
        num_frames=16, sampling_rate=4, test_num_segment=10, test_num_crop=3,
        input_size=224, short_side_size=224, reprob=0.25,
        aa='rand-m7-n4-mstd0.5-inc1', train_interpolation='bicubic')
    dataset, _ = build_dataset(is_train=False, test_mode=True, args=args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                        pin_memory=True, persistent_workers=True, drop_last=False)
    with init_empty_weights():
        model = models.vit_giant_patch14_224(pretrained=False, img_size=224,
            num_classes=5, all_frames=16, tubelet_size=2, drop_rate=0.,
            drop_path_rate=0.3, attn_drop_rate=0., head_drop_rate=0.,
            use_mean_pooling=True, init_scale=0.001, with_cp=False)
    # Only official, pinned, trusted checkpoint; mmap avoids loading optimizer arrays.
    checkpoint = torch.load('.cache/vit_g_vmbench.pt', map_location='cpu',
                            mmap=True, weights_only=False)
    state = checkpoint['module']
    model.to_empty(device='cuda')  # FP32 parameters, as in the upstream eval model.
    model.load_state_dict(state, strict=True)  # Abort rather than accept a random/partial model.
    del state, checkpoint
    gc.collect()
    model.eval()
    torch.manual_seed(0)
    official.final_test(loader, model, torch.device('cuda'), str(directory / '0.txt'))
    class SerialPool:
        def __init__(self, *args): pass
        def map(self, fn, values): return list(map(fn, values))
    official.Pool = SerialPool  # Same compute_video, no 64 forked workers.
    records = official.final_merge(str(directory), 1, meta)
    Path(meta).write_text(json.dumps(records, indent=2))
    del model, loader, dataset

def main():
    parser = argparse.ArgumentParser(description='Workshop adapter; calls official VMBench evaluators.')
    parser.add_argument('--metric', choices=['PAS', 'OIS', 'TCS', 'MSS', 'CAS'], required=True)
    parser.add_argument('--meta', required=True)
    parser.add_argument('--mss-mode', default='fp16', choices=['fp16', '8bit_experimental'])
    args = parser.parse_args()
    report = {'status': 'ok'}
    try:
        if args.metric == 'MSS': mss(args.meta, args.mss_mode)
        else: {'PAS': pas, 'OIS': ois, 'TCS': tcs, 'CAS': cas}[args.metric](args.meta)
        torch.cuda.synchronize()
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        report['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
    except NotApplicable as error:
        report.update(status='not_applicable', message=str(error))
        traceback.print_exc()
    except Exception as error:
        report.update(status='failed', message=str(error))
        traceback.print_exc()
    finally:
        Path(args.meta).with_name('process_report.json').write_text(json.dumps(report, indent=2))
        gc.collect()
        torch.cuda.empty_cache()
    if report['status'] == 'failed':
        sys.exit(1)

if __name__ == '__main__': main()
'''

@dataclass
class WorkshopConfig:
    work_dir: Path = field(default_factory=lambda: Path('/content'))
    videos: dict[str, str] = field(default_factory=dict)
    input_mode: str = 'auto'
    video_meta: dict[str, dict] = field(default_factory=dict)
    default_subject: dict = field(default_factory=lambda: {
        'subject_noun': 'person', 'subject_type': 'human', 'prompt': 'A person moves in the scene.'})
    metrics_to_run: list[str] = field(default_factory=lambda: ['PAS','OIS','TCS','MSS','CAS'])
    mss_mode: str = 'precomputed'
    precomputed_url: str = ''
    create_variants: bool = False
    run_variant_comparisons: bool = False
    run_dataset: bool = True
    download_csv: bool = False
    timeout_seconds: int = 3600
    max_input_frames: int = 240
    max_input_pixels: int = 1280 * 720
    reuse_results: bool = True
    vmbench_commit: str = '9772be365784bc3f641d64fca018b1debb749116'
    adapter_profile: str = 'colab-v1-fp32-tcs-pas-boxfix'
    gpu_available: bool = False
    gpu_gib: float = 0.0

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        if self.input_mode not in {'auto','upload','demo'}:
            raise ValueError('input_mode должен быть auto, upload или demo.')
        if self.mss_mode not in {'precomputed','fp16','8bit_experimental'}:
            raise ValueError('Неизвестный mss_mode.')
        if set(self.metrics_to_run) - set(METRIC_NAMES):
            raise ValueError('Неизвестные metrics_to_run.')


class Workshop:
    """Owns configuration and runtime state; no notebook globals are read."""

    def __init__(self, config: WorkshopConfig):
        self.config = config
        self.work_dir = config.work_dir
        self.repo = self.work_dir / 'VMBench'
        self.cache = self.repo / '.cache'
        self.video_dir = self.work_dir / 'videos'
        self.run_dir = self.work_dir / 'vmbench_runs'
        self.env_root = self.work_dir / 'vmbench_envs'
        self.pythons = {name: self.env_root / name / 'bin' / 'python' for name in ('core','ois')}
        self.precomputed_path = self.work_dir / 'vmbench_precomputed.csv'
        self.runner = self.run_dir / 'workshop_runner.py'
        self.setup_errors = {}
        self.results, self.result_cache = [], {}
        self.input_paths, self.variants = [], {}
        self.video_paths, self.video_path = [], None
        self.one_video_result = {'video': '(no video)'}
        for directory in (self.video_dir, self.run_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.prepare_runner()

    @property
    def gpu_available(self):
        return self.config.gpu_available

    @property
    def gpu_gib(self):
        return self.config.gpu_gib

    @property
    def pd(self):
        import pandas
        return pandas

    @property
    def plt(self):
        import matplotlib.pyplot
        return matplotlib.pyplot

    @property
    def requests(self):
        import requests
        return requests

    @property
    def display(self):
        from IPython.display import display
        return display

    @property
    def Video(self):
        from IPython.display import Video
        return Video

    def prepare_runner(self):
        if not self.runner.exists() or self.runner.read_text() != RUNNER_SOURCE:
            self.runner.write_text(RUNNER_SOURCE)
        return self.runner


    def logged_command(self, command, log_name, cwd=None, env=None, timeout=None):
        if timeout is None:
            timeout = self.config.timeout_seconds
        if env is None:
            env = self.cache_environment()
        log = self.run_dir / log_name
        with log.open('w', encoding='utf-8') as stream:
            stream.write('COMMAND: ' + repr([str(x) for x in command]) + '\n')
            stream.flush()
            try:
                subprocess.run([str(x) for x in command], cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)
            except Exception as error:
                raise RuntimeError(f'{type(error).__name__}: подробнее в {log}') from error
        return log

    def evaluator_env(self):
        env = self.cache_environment()
        env['PYTHONPATH'] = os.pathsep.join((str(self.repo / p) for p in SOURCE_PATHS))
        env['OMP_NUM_THREADS'] = '1'
        env['MAX_JOBS'] = '2'
        env['MPLBACKEND'] = 'Agg'
        env['TOKENIZERS_PARALLELISM'] = 'false'
        return env

    def cache_environment(self):
        """Set cache paths for children without changing the importing kernel's environment."""
        env = os.environ.copy()
        env['HF_HOME'] = str(self.cache / 'huggingface')
        env['TORCH_HOME'] = str(self.cache / 'torch')
        env['UV_CACHE_DIR'] = str(self.work_dir / '.cache' / 'uv')
        return env

    def ensure_environment(self, name):
        if 'repository' in self.setup_errors or 'host' in self.setup_errors:
            raise RuntimeError('Сначала исправьте clone/pip: ' + repr(self.setup_errors))
        target = self.pythons[name].parent.parent
        marker = target / '.workshop_ready_v1'
        if marker.exists():
            return self.pythons[name]
        print(f'Установка {name}: отдельная Python 3.11 среда; лог в {self.run_dir}.')
        uv = shutil.which('uv')
        if not uv:
            raise RuntimeError('uv не установлен: повторите pip ячейку.')
        if not self.pythons[name].exists():
            self.logged_command([uv, 'venv', '--python', '3.11', target], f'{name}_venv.log')
        versions, index = (['torch==2.5.1', 'torchvision==0.20.1'], 'cu124') if name == 'core' else (['torch==2.1.0', 'torchvision==0.16.0'], 'cu121')
        self.logged_command([uv, 'pip', 'install', '--python', self.pythons[name], *versions, '--index-url', f'https://download.pytorch.org/whl/{index}'], f'{name}_torch.log')
        self.logged_command([uv, 'pip', 'install', '--python', self.pythons[name], *(CORE_PACKAGES if name == 'core' else OIS_PACKAGES)], f'{name}_pip.log')
        env = self.evaluator_env()
        if name == 'core':
            env['SAM2_BUILD_ALLOW_ERRORS'] = '1'
            self.logged_command([uv, 'pip', 'install', '--python', self.pythons[name], '--no-deps', '--no-build-isolation', '-e', self.repo / 'Grounded-SAM-2'], 'sam2_install.log', env=env)
            smoke = 'import perceptible_amplitude_score, temporal_coherence_score, motion_smoothness_score; import models, dataset, bench_utils.cas_utils; import torch; print(torch.__version__, torch.cuda.is_available())'
        else:
            self.logged_command([uv, 'pip', 'install', '--python', self.pythons[name], '--no-deps', '--no-build-isolation', '-e', self.repo / 'mmpose'], 'mmpose_install.log', env=env)
            mim = self.repo / 'mmpose/mmpose/.mim'
            mim.mkdir(exist_ok=True)
            config_link = mim / 'configs'
            if not config_link.exists():
                config_link.symlink_to((self.repo / 'mmpose/configs').resolve(), target_is_directory=True)
            smoke = 'import mmcv, mmengine, mmdet; from mmcv.ops import nms; import object_integrity_score; import torch; print(torch.__version__, mmcv.__version__, torch.cuda.is_available())'
        self.logged_command([self.pythons[name], '-c', smoke], f'{name}_imports.log', cwd=self.repo, env=env)
        freeze = subprocess.check_output([uv, 'pip', 'freeze', '--python', self.pythons[name]], text=True)
        (self.run_dir / f'{name}_freeze.txt').write_text(freeze)
        marker.write_text(self.config.vmbench_commit)
        print(f"{name}: imports проверены. {self.run_dir / (name + '_imports.log')}")
        return self.pythons[name]

    def download_if_missing(self, url, path, expected_size=None):
        from tqdm.auto import tqdm
        path = Path(path)
        if path.is_file() and path.stat().st_size > 0:
            if expected_size is None or path.stat().st_size == expected_size:
                return path
            print('Неполный файл будет загружен заново:', path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if expected_size and shutil.disk_usage(path.parent).free < expected_size + 2 ** 30:
            raise RuntimeError(f'Недостаточно диска для {path.name}: нужен {expected_size / 1000000000.0:.2f} GB + запас.')
        partial = path.with_name(path.name + '.part')
        try:
            with self.requests.get(url, stream=True, timeout=(30, 120), headers={'Accept-Encoding': 'identity'}) as response:
                response.raise_for_status()
                header_size = int(response.headers.get('Content-Length', 0)) or None
                total = expected_size or header_size
                print(f'Скачиваем {path.name}: ' + (f'{total / 1000000000.0:.3f} GB' if total else 'размер не сообщён') + f' → {path}')
                with partial.open('wb') as stream, tqdm(total=total, unit='B', unit_scale=True) as progress:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            progress.update(len(chunk))
            size = partial.stat().st_size
            if not size or (total and size != total):
                raise RuntimeError(f'Неполная загрузка {path.name}: {size} bytes вместо {total}.')
            partial.replace(path)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return path

    def hf_file(self, repo, filename, destination):
        revision = HF_REVISIONS[repo]
        return self.download_if_missing(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}', destination, HF_SIZES.get(filename))

    def ensure_checkpoints(self, metric):
        names = {'PAS': ['groundingdino_swinb_cogcoor.pth', 'sam_vit_h_4b8939.pth', 'scaled_offline.pth'], 'TCS': ['groundingdino_swinb_cogcoor.pth', 'sam2.1_hiera_large.pt', 'scaled_offline.pth'], 'CAS': ['vit_g_vmbench.pt']}.get(metric, [])
        for name in names:
            self.hf_file('GD-ML/VMBench', name, self.cache / name)
        if metric in ('PAS', 'TCS'):
            for name in ['config.json', 'model.safetensors', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt']:
                self.hf_file('google-bert/bert-base-uncased', name, self.cache / 'google-bert/bert-base-uncased' / name)
        if metric == 'OIS':
            for name, url in OIS_URLS.items():
                self.download_if_missing(url, self.cache / name, OIS_SIZES[name])
        if metric == 'MSS':
            for name in ['config.json', 'generation_config.json', 'preprocessor_config.json', 'special_tokens_map.json', 'tokenizer.model', 'tokenizer_config.json', 'pytorch_model.bin.index.json', 'pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin']:
                self.hf_file('q-future/one-align', name, self.cache / 'q-future/one-align' / name)

    def normalize_uploaded(self, path):
        path = Path(path)
        if path.suffix.lower() == '.mp4':
            return path
        destination = self.video_dir / (path.stem + '_converted.mp4')
        self.logged_command(['ffmpeg', '-y', '-i', path, '-map', '0:v:0', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', destination], 'convert_' + path.stem + '.log')
        if path.name in self.config.video_meta:
            self.config.video_meta[destination.name] = self.config.video_meta[path.name].copy()
        return destination

    def upload_more(self):
        from google.colab import files
        for name, content in files.upload().items():
            destination = self.video_dir / Path(name).name
            if destination.suffix.lower() in {'.mp4', '.mov', '.webm'}:
                destination.write_bytes(content)
                new_path = self.normalize_uploaded(destination)
                if new_path not in self.input_paths:
                    self.input_paths.append(new_path)
        print('Повторите preview/config ячейки, чтобы обновить список.')

    def get_video_info(self, path):
        path = Path(path)
        raw = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,duration:format=duration', '-of', 'json', str(path)], text=True)
        data = json.loads(raw)
        if not data.get('streams'):
            raise ValueError(f'{path.name}: видеопоток не найден.')
        stream = data['streams'][0]
        from fractions import Fraction
        rate = stream.get('avg_frame_rate', '0/1')
        fps = float(Fraction(rate)) if rate not in ('0/0', 'N/A') else 0.0
        duration = float(stream.get('duration') or data.get('format', {}).get('duration') or 0)
        count = stream.get('nb_read_frames')
        return {'filename': path.name, 'fps': fps, 'frame_count': int(count) if count and count != 'N/A' else None, 'width': int(stream['width']), 'height': int(stream['height']), 'duration_sec': duration}

    def show_video(self, path):
        self.display(self.Video(str(path), embed=True, width=520))

    def create_variants(self, path):
        path = Path(path)
        info = self.get_video_info(path)
        fps = info['fps']
        if fps < 2:
            raise ValueError('Для эксперимента нужен FPS ≥2.')
        prefix = path.stem + '_experiment'
        variants = {}
        filters = {'original': f'setpts=PTS-STARTPTS,fps={fps:.8f}', 'fast_x2': f'setpts=0.5*(PTS-STARTPTS),fps={fps:.8f}', 'slow_x05': f'setpts=2*(PTS-STARTPTS),fps={fps:.8f}', 'frame_drop': f"select='not(mod(n,2))',setpts=N/({fps / 2:.8f}*TB)", 'frame_duplication': f"select='not(mod(n,2))',setpts=N/({fps / 2:.8f}*TB),fps={fps:.8f}"}
        for name, vf in filters.items():
            output = self.video_dir / f'{prefix}_{name}.mp4'
            rate = fps / 2 if name == 'frame_drop' else fps
            self.logged_command(['ffmpeg', '-y', '-i', path, '-an', '-vf', vf, '-r', f'{rate:.8f}', '-vsync', 'cfr', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', output], 'variant_' + name + '.log')
            variants[name] = output
            self.config.video_meta[output.name] = self.config.video_meta.get(path.name, self.config.default_subject).copy()
        return variants

    def clear_gpu(self, verbose=False):
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if verbose:
                print(f'Notebook CUDA allocated={torch.cuda.memory_allocated()/2**20:.1f} MiB; '
                      f'reserved={torch.cuda.memory_reserved()/2**20:.1f} MiB')

    def video_sha256(self, path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def metric_profile(self, metric):
        return self.config.adapter_profile + ('-mss8bit' if metric == 'MSS' and self.config.mss_mode == '8bit_experimental' else '')

    def precomputed_score(self, digest, metric):
        if not self.precomputed_path.is_file():
            return None
        table = self.pd.read_csv(self.precomputed_path)
        required = {'video_sha256', 'metric', 'score', 'runtime_sec', 'vmbench_commit', 'adapter_profile'}
        if not required.issubset(table.columns):
            raise ValueError('Precomputed CSV: отсутствуют колонки ' + str(required - set(table.columns)))
        rows = table[(table.video_sha256 == digest) & (table.metric == metric) & (table.vmbench_commit == self.config.vmbench_commit) & (table.adapter_profile == self.metric_profile(metric))]
        if rows.empty:
            return None
        if len(rows) != 1:
            raise ValueError('Precomputed CSV: неоднозначные дубли для данного hash/metric/profile.')
        row = rows.iloc[0]
        import math
        value, seconds = (float(row.score), float(row.runtime_sec))
        if not math.isfinite(value) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Precomputed CSV: score/runtime должны быть конечными числами.')
        if metric != 'PAS' and (not 0 <= value <= 1) or (metric == 'PAS' and value < 0):
            raise ValueError('Precomputed CSV: score вне ожидаемого диапазона.')
        return (value, seconds)

    def failure_hint(self, error):
        message = str(error)
        lower = message.lower()
        if 'out of memory' in lower or 'cuda error' in lower:
            return 'Недостаточно VRAM или CUDA error. Возьмите короткий вход / runtime с большей GPU; смотрите лог.'
        if 'no person' in lower or 'no pose' in lower:
            return 'OIS not applicable: человек/поза не найдены хотя бы в одном кадре.'
        if 'strict' in lower or 'state_dict' in lower:
            return 'Checkpoint не соответствует официальной модели. Смотрите подробности в логе.'
        return message[:350]

    def compute_metric(self, path, metric):
        import math
        start_total = time.perf_counter()
        row = {'video': Path(path).name if path else '(no video)', 'metric': metric, 'score': None, 'runtime_sec': 0.0, 'original_runtime_sec': None, 'setup_sec': 0.0, 'status': 'skipped', 'source': 'none', 'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'vmbench_commit': self.config.vmbench_commit, 'adapter_profile': self.metric_profile(metric), 'video_sha256': None, 'message': '', 'log': None, 'peak_allocated_gib': None}
        cache_key, process_log, inference_start = (None, None, None)
        try:
            if not path:
                row['message'] = 'Нет входного видео.'
                return None
            path = Path(path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f'Видео не найдено: {path}')
            subject = self.config.video_meta.get(path.name, self.config.default_subject).copy()
            if not str(subject.get('subject_noun', '')).strip():
                raise ValueError('Укажите английское subject_noun в VIDEO_META.')
            digest = self.video_sha256(path)
            row['video_sha256'] = digest
            cache_key = (digest, metric, self.metric_profile(metric), json.dumps(subject, sort_keys=True))
            if self.config.reuse_results and cache_key in self.result_cache:
                previous = self.result_cache[cache_key]
                row.update(score=previous['score'], status=previous['status'], source='cache:' + previous['source'], message=previous['message'], log=previous['log'], original_runtime_sec=previous.get('original_runtime_sec') or previous['runtime_sec'])
                return row['score']
            if metric == 'OIS' and subject.get('subject_type') != 'human':
                row.update(status='not_applicable', message='Официальный OIS CLI оценивает human pose, не этот тип субъекта.')
                return None
            teacher = self.precomputed_score(digest, metric)
            if teacher is not None:
                row.update(score=teacher[0], original_runtime_sec=teacher[1], status='ok', source='precomputed', message='Score преподавателя, совпали hash/revision/profile.')
                return row['score']
            if metric == 'MSS' and self.config.mss_mode == 'precomputed':
                row['message'] = 'MSS FP16 не помещается в T4. Нужен CSV преподавателя; либо явно выберите 8bit_experimental.'
                return None
            if not self.gpu_available:
                row['message'] = 'GPU отсутствует: включите T4 runtime.'
                return None
            if metric == 'MSS' and self.config.mss_mode == 'fp16' and (self.gpu_gib < 22):
                row['message'] = 'MSS FP16: нужен GPU ≥22 GiB или precomputed CSV; только веса ≈15.28 GiB.'
                return None
            if metric == 'MSS' and self.config.mss_mode not in {'fp16', '8bit_experimental'}:
                raise ValueError('Неизвестный MSS_MODE.')
            info = self.get_video_info(path)
            if not info['frame_count'] or info['frame_count'] < 3 or info['fps'] < 2:
                raise ValueError('Для evaluator’ов нужны ≥3 кадров и FPS≥2 (TCS использует int(FPS)-1).')
            if info['frame_count'] > self.config.max_input_frames or info['width'] * info['height'] > self.config.max_input_pixels:
                row['message'] = 'Вход превышает workshop guard. Подготовьте короткий clip или явно увеличьте лимиты в config.'
                return None
            pas_value = None
            if metric == 'MSS':
                print('MSS требует PAS для adaptive threshold; PAS выполняется отдельным процессом.')
                pas_value = self.compute_pas(str(path))
                if pas_value is None:
                    row['message'] = 'MSS пропущен: не получен обязательный PAS.'
                    return None
            setup_start = time.perf_counter()
            python = self.ensure_environment('ois' if metric == 'OIS' else 'core')
            if metric == 'MSS' and self.config.mss_mode == '8bit_experimental':
                marker = self.pythons['core'].parent.parent / '.bitsandbytes_ready'
                if not marker.exists():
                    self.logged_command([shutil.which('uv'), 'pip', 'install', '--python', python, 'bitsandbytes==0.44.1'], 'mss_bitsandbytes_install.log')
                    marker.write_text('0.44.1')
            self.ensure_checkpoints(metric)
            row['setup_sec'] = time.perf_counter() - setup_start
            job = self.run_dir / f'{digest[:16]}_{metric}_{time.time_ns()}'
            job.mkdir(parents=True)
            local_video = job / 'input.mp4'
            local_video.symlink_to(path)
            record = {'filepath': str(local_video), 'subject_noun': subject['subject_noun'], 'prompt': subject.get('prompt', subject['subject_noun'])}
            if pas_value is not None:
                record[SCORE_KEYS['PAS']] = pas_value
            meta = job / 'results.json'
            meta.write_text(json.dumps([record], indent=2))
            process_log = job / 'metric.log'
            row['log'] = str(process_log)
            command = [str(python), str(self.runner), '--metric', metric, '--meta', str(meta)]
            if metric == 'MSS':
                command += ['--mss-mode', self.config.mss_mode]
            print(f"{metric}: начало {row['timestamp_utc']} | {path.name} | log: {process_log}")
            inference_start = time.perf_counter()
            with process_log.open('w') as stream:
                result = subprocess.run(command, cwd=self.repo, env=self.evaluator_env(), stdout=stream, stderr=subprocess.STDOUT, timeout=self.config.timeout_seconds, check=False)
            row['runtime_sec'] = time.perf_counter() - inference_start
            report_path = job / 'process_report.json'
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            if report.get('status') == 'not_applicable':
                row.update(status='not_applicable', message=report['message'])
                return None
            if result.returncode or report.get('status') != 'ok':
                tail = '\n'.join(process_log.read_text(errors='replace').splitlines()[-6:])
                raise RuntimeError(report.get('message') or tail or f'Process exit={result.returncode}')
            record = json.loads(meta.read_text())[0]
            value = float(record[SCORE_KEYS[metric]])
            if not math.isfinite(value):
                raise ValueError('Evaluator вернул NaN/Inf.')
            notes = []
            if metric == 'PAS' and record.get('_pas_detected') == 0:
                notes.append('Subject не обнаружен: upstream PAS использовал background motion.')
            if metric == 'TCS' and record.get('_tcs_objects') == 0:
                notes.append('Объекты не обнаружены: высокий upstream TCS не подтверждает coherence.')
            if metric == 'TCS' and 'WARNING: SAM2 CUDA extension unavailable' in process_log.read_text(errors='replace'):
                notes.append('SAM2 hole filling недоступен; см. ограничения адаптера.')
            if metric == 'MSS' and self.config.mss_mode == '8bit_experimental':
                notes.append('Экспериментальная 8-bit точность; не точная FP16 реплика.')
            row.update(score=value, status='ok', source='live:' + ('8bit' if metric == 'MSS' and self.config.mss_mode == '8bit_experimental' else 'official-adapted'), message=' '.join(notes), peak_allocated_gib=report.get('peak_allocated_gib'))
            return value
        except Exception as error:
            if inference_start is not None:
                row['runtime_sec'] = time.perf_counter() - inference_start
            error_log = self.run_dir / f'{metric}_wrapper_error_{time.time_ns()}.log'
            error_log.write_text(traceback.format_exc())
            row.update(status='failed', message=self.failure_hint(error), log=str(process_log or error_log))
            print(f"{metric} failed: {row['message']}; подробности: {row['log']}")
            return None
        finally:
            row['total_wall_sec'] = time.perf_counter() - start_total
            self.results.append(row)
            if cache_key is not None and row['status'] == 'ok':
                self.result_cache[cache_key] = row.copy()
            self.clear_gpu()
            score_text = f"{row['score']:.4f}" if row['score'] is not None else 'N/A'
            print(f"{metric}: {score_text} | {row['runtime_sec']:.1f} sec | {row['status']} | {row['source']}")
            if row['message']:
                print(row['message'])

    def compute_pas(self, path):
        return self.compute_metric(path, 'PAS')

    def compute_ois(self, path):
        return self.compute_metric(path, 'OIS')

    def compute_tcs(self, path):
        return self.compute_metric(path, 'TCS')

    def compute_mss(self, path):
        return self.compute_metric(path, 'MSS')

    def compute_cas(self, path):
        return self.compute_metric(path, 'CAS')

    def compare_variants(self, metric):
        if not self.config.run_variant_comparisons or not self.variants:
            print('Для сравнения включите CREATE_VARIANTS и RUN_VARIANT_COMPARISONS.')
            return
        compute = {'PAS': self.compute_pas, 'OIS': self.compute_ois, 'TCS': self.compute_tcs, 'MSS': self.compute_mss, 'CAS': self.compute_cas}[metric]
        self.display(self.pd.DataFrame([{'variant': name, metric: compute(str(path))} for name, path in self.variants.items()]))

    def evaluate_video(self, path, metrics=('PAS', 'OIS', 'TCS', 'MSS', 'CAS')):
        functions = {'PAS': self.compute_pas, 'OIS': self.compute_ois, 'TCS': self.compute_tcs, 'MSS': self.compute_mss, 'CAS': self.compute_cas}
        unknown = set(metrics) - set(functions)
        if unknown:
            raise ValueError(f'Неизвестные метрики: {unknown}')
        result = {'video': Path(path).name if path else '(no video)'}
        for metric in metrics:
            result[metric] = functions[metric](path)
            self.clear_gpu()
        return result

    def validate_submission(self, df, expected_ids=None):
        errors = []
        if list(df.columns) != ['video_id', 'prediction', 'confidence']:
            return (False, ['Нужны ровно колонки video_id,prediction,confidence в таком порядке.'])
        if df.empty:
            errors.append('Submission пуст.')
        ids = df.video_id.fillna('').astype(str).str.strip()
        if (ids == '').any() or ids.duplicated().any():
            errors.append('video_id должны быть непустыми и уникальными.')
        if expected_ids is not None and set(ids) != set(expected_ids):
            errors.append('video_id не совпадают с набором задания.')
        if not df.prediction.isin(LABELS).all():
            errors.append('Заполните prediction одним из: ' + ', '.join(sorted(LABELS)))
        confidence = self.pd.to_numeric(df.confidence, errors='coerce')
        if confidence.isna().any() or not confidence.between(0, 1).all():
            errors.append('confidence должны быть конечными числами от 0 до 1.')
        return (not errors, errors)

    def install_repository(self):
        try:
            if not self.repo.exists():
                self.logged_command(['git', 'clone', 'https://github.com/AMAP-ML/VMBench.git', self.repo], 'clone.log')
                self.logged_command(['git', 'checkout', '--detach', self.config.vmbench_commit], 'checkout.log', cwd=self.repo)
            actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
            if actual != self.config.vmbench_commit:
                raise RuntimeError(f'Checkout {actual} отличается от {self.config.vmbench_commit}; используйте свежий runtime.')
            required = ['perceptible_amplitude_score.py', 'object_integrity_score.py', 'temporal_coherence_score.py', 'motion_smoothness_score.py', 'commonsense_adherence_score.py', 'requirements.txt', 'Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py', 'Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml', 'mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py', 'mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py']
            for name in required:
                if not (self.repo / name).is_file():
                    raise RuntimeError(f'В репозитории отсутствует {name}')
            self.cache.mkdir(exist_ok=True)
            print('Официальный код:', actual)
            print('Upstream requirements:\n' + (self.repo / 'requirements.txt').read_text())
        except Exception as error:
            self.setup_errors['repository'] = str(error)
            print('Clone failed:', error)

    def install_system_dependencies(self):
        try:
            self.logged_command(['apt-get', '-qq', 'update'], 'apt_update.log')
            self.logged_command(['apt-get', '-qq', 'install', '-y', 'ffmpeg', 'git', 'build-essential', 'libgl1', 'libglib2.0-0'], 'apt_install.log')
            print('apt: готово; ffmpeg:', shutil.which('ffmpeg'))
        except Exception as error:
            self.setup_errors['apt'] = str(error)
            print('apt failed:', error)

    def install_host_packages(self):
        try:
            self.logged_command([sys.executable, '-m', 'pip', '-q', 'install', 'uv==0.8.22'], 'host_pip.log')
            import requests
            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import display, Video
            print('Host packages: готовы. PyTorch Colab сохранён.')
        except Exception as error:
            self.setup_errors['host'] = str(error)
            print('pip failed:', error)

    def install_evaluators(self):
        if self.gpu_available and set(self.config.metrics_to_run) & {'PAS', 'TCS', 'CAS', 'MSS'}:
            try:
                self.ensure_environment('core')
            except Exception as error:
                self.setup_errors['core'] = str(error)
                print('Evaluator setup failed:', error)
        else:
            print('Установка GPU evaluator’ов отложена: GPU отсутствует или они не выбраны.')

    def load_videos(self):
        self.input_paths = []
        urls = {name: url.strip() for name, url in self.config.videos.items() if url.strip()}
        try:
            if self.config.input_mode == 'demo' or (self.config.input_mode == 'auto' and urls):
                if not urls:
                    print('Demo URLs не заданы. Выберите INPUT_MODE="upload".')
                for name, url in urls.items():
                    if Path(name).name != name or not name:
                        raise ValueError('Имя demo должно быть простым именем без папок.')
                    from urllib.parse import urlparse
                    suffix = Path(urlparse(url).path).suffix.lower()
                    if suffix not in {'.mp4', '.mov', '.webm'}:
                        suffix = '.mp4'
                    try:
                        self.input_paths.append(self.normalize_uploaded(self.download_if_missing(url, self.video_dir / (name + suffix))))
                    except Exception as error:
                        print(f'Demo {name} failed:', error)
            elif self.config.input_mode in ('upload', 'auto'):
                existing = sorted(self.video_dir.glob('*.mp4'))
                if existing:
                    self.input_paths = existing
                    print('Используем уже загруженные файлы. Для новых вызовите upload_more().')
                else:
                    from google.colab import files
                    uploaded = files.upload()
                    for name, content in uploaded.items():
                        safe_name = Path(name).name
                        if Path(safe_name).suffix.lower() not in {'.mp4', '.mov', '.webm'}:
                            print('Пропускаем неподдерживаемый файл:', safe_name)
                            continue
                        destination = self.video_dir / safe_name
                        destination.write_bytes(content)
                        self.input_paths.append(self.normalize_uploaded(destination))
                    del uploaded
            else:
                raise ValueError('INPUT_MODE должен быть auto, upload или demo.')
        except Exception as error:
            (self.run_dir / 'video_input_error.log').write_text(traceback.format_exc())
            print('Video input failed:', error, '| лог:', self.run_dir / 'video_input_error.log')
        print('Видео:', [p.name for p in self.input_paths])
        return self.input_paths

    def preview_videos(self):
        valid_paths, metadata_rows = ([], [])
        for path in self.input_paths:
            try:
                metadata_rows.append(self.get_video_info(path))
                valid_paths.append(path)
                self.show_video(path)
            except Exception as error:
                print(f'{Path(path).name}: нельзя прочитать видео: {error}')
        self.input_paths = valid_paths
        video_info_df = self.pd.DataFrame(metadata_rows, columns=['filename', 'fps', 'frame_count', 'width', 'height', 'duration_sec'])
        self.display(video_info_df)
        return video_info_df

    def prepare_variants(self):
        self.variants = {}
        if self.config.create_variants and self.input_paths:
            try:
                self.variants = self.create_variants(self.input_paths[0])
                for path in self.variants.values():
                    if path not in self.input_paths:
                        self.input_paths.append(path)
                self.display(self.pd.DataFrame([self.get_video_info(path) for path in self.variants.values()]))
            except Exception as error:
                print('Variant creation failed:', error)
        else:
            print('Optional section пропущен. Чтобы создать варианты, включите CREATE_VARIANTS.')
        return self.variants

    def select_videos(self):
        self.video_paths = [str(p) for p in self.input_paths]
        self.video_path = self.video_paths[0] if self.video_paths else None
        for path in self.video_paths:
            meta = self.config.video_meta.get(Path(path).name, self.config.default_subject)
            print(Path(path).name, '→', meta['subject_noun'], '/', meta['subject_type'])
        if self.video_path is None:
            print('Нет видео. Загрузите хотя бы один файл и повторите preview/config и разделы метрик.')
        return (self.video_path, self.video_paths)

    def load_precomputed(self):
        if self.config.precomputed_url:
            try:
                self.download_if_missing(self.config.precomputed_url, self.precomputed_path)
            except Exception as error:
                print('Precomputed download failed:', error)

    def show_metric_summary(self, path=None):
        self.clear_gpu(verbose=True)
        if self.results:
            first_video = Path(path).name if path else '(no video)'
            summary = self.pd.DataFrame(self.results)
            self.display(summary[summary.video == first_video].drop_duplicates('metric', keep='last')[['metric', 'score', 'runtime_sec', 'status', 'source', 'peak_allocated_gib']].rename(columns={'metric': 'Metric', 'score': 'Score', 'runtime_sec': 'Runtime (sec)'}))
        else:
            print('Пока нет запущенных метрик.')

    def evaluate_dataset(self):
        rows = []
        if self.config.run_dataset:
            rows = [self.evaluate_video(path, self.config.metrics_to_run) for path in self.video_paths]
        elif self.video_path:
            rows = [self.one_video_result]
        results_df = self.pd.DataFrame(rows).reindex(columns=['video', *METRIC_NAMES])
        results_path = self.work_dir / 'vmbench_results.csv'
        results_df.to_csv(results_path, index=False)
        self.display(results_df)
        diagnostics_df = self.pd.DataFrame(self.results)
        diagnostics_path = self.work_dir / 'vmbench_diagnostics.csv'
        diagnostics_df.to_csv(diagnostics_path, index=False)
        export_columns = ['video_sha256', 'metric', 'score', 'runtime_sec', 'vmbench_commit', 'adapter_profile']
        if not diagnostics_df.empty:
            successful_live = diagnostics_df[(diagnostics_df.status == 'ok') & diagnostics_df.source.str.startswith('live:')]
            successful_live[export_columns].drop_duplicates(['video_sha256', 'metric', 'adapter_profile'], keep='last').to_csv(self.work_dir / 'vmbench_precomputed_export.csv', index=False)
            self.display(diagnostics_df[['video', 'metric', 'status', 'source', 'message', 'runtime_sec', 'setup_sec']])
        else:
            self.pd.DataFrame(columns=export_columns).to_csv(self.work_dir / 'vmbench_precomputed_export.csv', index=False)
        print('Сохранено:', results_path, diagnostics_path)
        if self.config.download_csv:
            from google.colab import files
            files.download(str(results_path))
            files.download(str(diagnostics_path))
        return results_df

    def plot_results(self, results_df):
        for metric in METRIC_NAMES:
            numeric = self.pd.to_numeric(results_df[metric], errors='coerce')
            valid = results_df.loc[numeric.notna(), ['video']].copy()
            valid[metric] = numeric[numeric.notna()]
            if valid.empty:
                print(f'{metric}: нет valid scores для графика.')
                continue
            fig, ax = self.plt.subplots(figsize=(max(5, len(valid) * 1.1), 3.2))
            ax.bar(valid.video, valid[metric], color='#3277a8')
            ax.set_title(metric + (' (raw displacement)' if metric == 'PAS' else ''))
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=35)
            if metric != 'PAS':
                ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.25)
            fig.tight_layout()
            self.plt.show()

    def student_table(self):
        student_df = self.pd.DataFrame({'Video': ['A', 'B', 'C', 'D'], **{column: [''] * 4 for column in ['Human guess', 'PAS', 'OIS', 'TCS', 'MSS', 'CAS', 'Final guess']}})
        self.display(student_df)
        student_df.to_csv(self.work_dir / 'student_experiment.csv', index=False)
        return student_df

    def submission_template(self):
        submission_ids = [Path(path).stem for path in self.video_paths]
        submission_df = self.pd.DataFrame({'video_id': submission_ids, 'prediction': [''] * len(submission_ids), 'confidence': [''] * len(submission_ids)})
        submission_path = self.work_dir / 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        self.display(submission_df)
        ok, errors = self.validate_submission(submission_df, submission_ids)
        print('Format valid:', ok, '|', '; '.join(errors) if errors else 'OK')
        return submission_df


_SESSION: Workshop | None = None


def configure(config: WorkshopConfig) -> Workshop:
    """Configure explicitly; repeating the config cell preserves results in the same runtime."""
    global _SESSION
    if _SESSION is None or _SESSION.work_dir != config.work_dir:
        _SESSION = Workshop(config)
    else:
        _SESSION.config = config
        _SESSION.prepare_runner()
    print('Cache:', _SESSION.cache)
    return _SESSION


def current_workshop() -> Workshop:
    if _SESSION is None:
        raise RuntimeError('Сначала выполните configure(WorkshopConfig(...)).')
    return _SESSION


def compute_pas(path): return current_workshop().compute_pas(path)
def compute_ois(path): return current_workshop().compute_ois(path)
def compute_tcs(path): return current_workshop().compute_tcs(path)
def compute_mss(path): return current_workshop().compute_mss(path)
def compute_cas(path): return current_workshop().compute_cas(path)
def get_video_info(path): return current_workshop().get_video_info(path)
def show_video(path): return current_workshop().show_video(path)
def clear_gpu(verbose=False): return current_workshop().clear_gpu(verbose=verbose)
def compare_variants(metric): return current_workshop().compare_variants(metric)
def evaluate_video(path, metrics=('PAS','OIS','TCS','MSS','CAS')):
    return current_workshop().evaluate_video(path, metrics)
def validate_submission(df, expected_ids=None):
    return current_workshop().validate_submission(df, expected_ids)
