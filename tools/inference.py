import argparse
import os
import warnings
from collections import deque

import cv2
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import init_dist, load_checkpoint, wrap_fp16_model

from mmdet3d.apis import single_gpu_inference
from mmdet3d.core.bbox.structures.utils import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor


def get_P(calib_file):
    """
    Get matrix P_rect_02 (camera 2 RGB)
    and transform to 3 x 4 matrix
    """
    for line in open(calib_file):
        if 'P2' in line:
            cam_P = line.strip().split(' ')
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            matrix = np.zeros((3, 4))
            matrix = cam_P.reshape((3, 4))
            return matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--room', type=str, default='4sapujyt')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDataParallel(model, device_ids=[0])

    frame_deque = deque(maxlen=5)
    
    import glob

    import tqdm
    dataset_type = 'nuscenes'
    all_image_path = glob.glob('/mnt/ebs3/dataset/NUSC_KITTI/training/image_2/*.png')
    for image_path in tqdm.tqdm(all_image_path):
        calib_path = image_path.replace('/image_2/','/calib/').replace('.png','.txt')
        img = cv2.imread(image_path)
        cam_matrix = get_P(calib_path)
        img_info = {
                    'filename': image_path,
                    'ori_shape': (900, 1600, 3),
                    'img_shape': (900, 1600, 3), 
                    'cam_intrinsic': cam_matrix.tolist(), 
                    'pad_shape': (928, 1600, 3), 
                    'scale_factor': 1.0, 
                    'flip': False, 
                    'pcd_horizontal_flip': False, 
                    'pcd_vertical_flip': False, 
                    'img_norm_cfg': {'mean': np.array([103.53 , 116.28 , 123.675], dtype=np.float32), 'std': np.array([1., 1., 1.], dtype=np.float32), 'to_rgb': False}, 
                    'transformation_3d_flow': []
                }
        pipeline_input = dict(img_info=img_info)
        pipeline_input['img_prefix'] = None
        pipeline_input['bbox_fields'] = []
        pipeline_input['mask_fields'] = []
        pipeline_input['seg_fields'] = []
        pipeline_input['img'] = img
        pipeline_input['box_type_3d'], pipeline_input['box_mode_3d'] = get_box_type('Camera')
        pipeline_input['img_fields'] = []
        pipeline_input['bbox3d_fields'] = []
        pipeline_input['pts_mask_fields'] = []
        pipeline_input['pts_seg_fields'] = []
        pipeline = Compose(cfg.data.test['pipeline'])

        data = [pipeline(pipeline_input)]
        data = collate(data)

        img = single_gpu_inference(model, data, img, frame_deque)
        image_name = os.path.basename(image_path)
        path = f"/home/miles/mmdetection3d/DEBUG/{dataset_type}/{image_name}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)

if __name__ == '__main__':
    main()
