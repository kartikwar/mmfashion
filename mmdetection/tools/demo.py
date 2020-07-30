import argparse

import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


from mmdet.apis import inference_detector, init_detector
    # show_result


def show_result(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

    ###########################################
    # remove duplicate
    new_bbox_result = []
    for ti, temp in enumerate(bbox_result):
        if len(temp) <= 1:
            new_bbox_result.append(temp)
            continue
        new_temp = sorted(temp, key=lambda x: x[-1])[-1]
        new_bbox_result.append(np.asarray([new_temp]))

    bbox_result = new_bbox_result
    #########################################

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    # draw segmentation masks
    # if segm_result is not None:
    #     segms = mmcv.concat_list(segm_result)
    #     inds = np.where(bboxes[:, -1] > score_thr)[0]
    #     np.random.seed(42)
    #     color_masks = [
    #         np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    #         for _ in range(max(labels) + 1)
    #     ]
    #     for i in inds:
    #         i = int(i)
    #         color_mask = color_masks[labels[i]]
    #         mask = maskUtils.decode(segms[i]).astype(np.bool)
    #         img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img



def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Fashion Detection and Segmentation')
    parser.add_argument(
        '--config',
        help='mmfashion config file path',
        default='configs/mmfashion/mask_rcnn_r50_fpn_1x.py')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--input',
        type=str,
        help='input image path',
        default='/home/kartik/Documents/personal/git-repos/mmfashion/demo/imgs/01_4_full.jpg')
    
    return  parser.parse_args()


def main():
    args = parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    # test a single image and show the results
    img = args.input
    result = inference_detector(model, img)

    # visualize the results in a new window
    # or save the visualization results to image files
    
    out_file = img.split('.')[0] + '_result.jpg'
    
    out_file = 'lol.jpg'
    
    print(out_file)
    
    show_result(
        img, result, model.CLASSES, out_file=out_file)


if __name__ == '__main__':
    main()