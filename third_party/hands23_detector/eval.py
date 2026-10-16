# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2 import _C
from detectron2.utils.logger import setup_logger

from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import (
    DatasetEvaluators,
    verify_results,
)
from detectron2.engine import DefaultTrainer, launch
from detectron2.data import  MetadataCatalog
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from hodetector.modeling import roi_heads
from hodetector.data.ho import load_ho_voc_instances
from hodetector.data import register_ho_pascal_voc, hoMapper
from hodetector.evaluation.evaluator import Hands23_Evaluator
import os
import argparse
import logging

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list = [
                Hands23_Evaluator('Hands23_VAL', save_dir, save_result)
            ]

            return DatasetEvaluators(evaluator_list)


def set_cfg(args):

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.MODEL.WEIGHTS = args.model_weights

    # assign values to the thresholds for bounding box prediction
    cfg.HAND = args.hand_thresh
    cfg.FIRSTOBJ = args.first_obj_thresh
    cfg.SECONDOBJ = args.second_obj_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(args.hand_thresh, args.first_obj_thresh, args.second_obj_thresh)

    # assign values to the thresholds for interaction classification
    cfg.HAND_RELA = args.hand_rela
    cfg.OBJ_RELA = args.obj_rela

    cfg.MODEL.DEVICE = "cuda"

    cfg.freeze()

    return cfg


def main(args):

    cfg = set_cfg(args)

    global save_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    global save_result
    save_result = args.save_result

    model = Trainer.build_model(cfg)

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )

    res = Trainer.test(cfg, model)
    

    verify_results(cfg, res)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    The inference of our model depends on a set of thresholds:
        1) hand_thresh, first_obj_thresh, second_obj_thresh: thresholds of bbox score
        for each class of object to be considered in later layers
        2) hand_rela, obj_rela: thresholds for a (hand, obj)  or (tool, second obj) pair
        to be considered in interaction.
    The combination of threhsolds we used are set as default. Please feel free to adjust them
    based on your need.
    '''
    parser.add_argument("--hand_thresh", type=float, default=0.7)
    parser.add_argument("--first_obj_thresh", type=float, default=0.5)
    parser.add_argument("--second_obj_thresh", type=float, default=0.3)
    parser.add_argument("--hand_rela", type=float, default=0.3)
    parser.add_argument("--obj_rela", type=float, default=0.7)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:49629")
    parser.add_argument("--config_file", default= "./faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml")

    parser.add_argument(
        "--model_weights", default=f"./model_weights/model_hands23.pth")
    parser.add_argument(
        "--data_dir", default = 'data/hands23_data_coco')
    parser.add_argument("--save_dir", default=f"./results/evaluation/")
    parser.add_argument("--save_result", type=bool, default=True)
    parser.add_argument("--image_root", type = str)
    args = parser.parse_args()

    global _datasets_root
    _datasets_root = args.data_dir

    global save_dir
    save_dir = args.save_dir

    global save_result
    save_result = args.save_result
  
    register_coco_instances(name='Hands23_VAL', metadata={}, json_file=os.path.join(
    _datasets_root, "annotations", "val.json"), image_root= args.image_root)
    MetadataCatalog.get(f'Hands23_VAL').set(evaluator_type='coco')  

    print("Command Line Args:", args)

    setup_logger()
    logging.getLogger("detectron2").setLevel(logging.DEBUG)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )
