# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmcv.image import tensor2imgs

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)
from . import kitti_util as utils

class PlainObject3d:
    def __init__(self, x, y, z, l, w, h, ry):
        self.l = l
        self.w = w
        self.h = h
        self.ry = ry
        self.t = (x, y, z)

def draw_3d_box(img, plain_obj_3d, calib):
    box3d_pts_2d, _ = utils.compute_box_3d(plain_obj_3d, calib)
    img = utils.draw_projected_box3d(img, box3d_pts_2d, color=(0, 255, 0))
    return img

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool, optional): Whether to save viualization results.
            Default: True.
        out_dir (str, optional): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(
                    data,
                    result,
                    out_dir=out_dir,
                    show=show,
                    score_thr=show_score_thr)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def single_gpu_inference(model, data, img, frame_deque=None):
    model.eval()
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    boxes3d = []
    for index, value in enumerate(result[0]['img_bbox']['boxes_3d'].tensor):
        if float(result[0]['img_bbox']['scores_3d'][index]) > 0.3:
            # import ipdb; ipdb.set_trace()
            boxes3d.append(value.detach().cpu().numpy().tolist())
    # raw_boxes3d = result[0]['img_bbox']['boxes_3d'].tensor.cpu().numpy()
    if len(boxes3d) == 0:
        resized_img = cv2.resize(img, (1920, 1080), interpolation = cv2.INTER_AREA)
        frame_deque.appendleft(resized_img)
    boxes3d = np.array(boxes3d)
    for idx, box in enumerate(boxes3d):
        # img = cv2.imread(data['img_metas'][0].data[0][0]['filename'])

        # This might failed.
        # Nah, It's not failed.
        calib = np.array(data['img_metas'][0].data[0][0]['cam2img'])[:3]
        # print(calib)
        parsed_box = PlainObject3d(box[0], box[1], box[2], box[3], box[4], box[5], box[6])
        img = draw_3d_box(img, parsed_box, calib)

        if frame_deque is not None:
            resized_img = cv2.resize(img, (1920, 1080), interpolation = cv2.INTER_AREA)
            frame_deque.appendleft(resized_img)
    
    return img