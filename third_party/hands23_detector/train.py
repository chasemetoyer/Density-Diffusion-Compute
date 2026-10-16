# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
from datetime import timedelta
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader
)
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch
from detectron2.evaluation import DatasetEvaluators
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.projects.point_rend import add_pointrend_config
from hodetector.modeling import roi_heads

from hodetector.data import register_ho_pascal_voc, hoMapper
from hodetector.evaluation.evaluator import Hands23_Evaluator
import argparse


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list = [
                Hands23_Evaluator('Hands23_VAL', save_dir, True)
            ]

            return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=hoMapper(cfg, False)
        )

    @classmethod
    def build_train_loader(cls, cfg):
        print(cfg)
        return build_detection_train_loader(
            cfg, mapper=hoMapper(cfg, True),
        )


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)

    cfg.INPUT.RANDOM_FLIP = "none"

    # Current implementation only supports cases where batch size == num GPUs
    # For other cases, please modify hodetector.modeling.roi_heads._forward_z function
    cfg.SOLVER.IMS_PER_BATCH = args.num_gpus

    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    global save_dir
    save_dir = args.output_dir

    add_pointrend_config(cfg)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main(args):
    
    cfg = setup(args)
    print(cfg)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )

    trainer.train()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:49629")
    parser.add_argument("--config_file", default= "faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml")
    parser.add_argument("--data_root", default="./data/hands23_data_coco")
    parser.add_argument("--output_dir", default= "./model_outputs")
    args = parser.parse_args()

    print("Command Line Args:", args)

    # directory for images
    _datasets_root = args.data_root

    # directory for annotation file in .json format
    annotation_root = '/data/hands23_data_coco/'

    for d in ["TRAIN", "VAL"]:
        register_ho_pascal_voc(name=f'Hands23_{d}', dirname=_datasets_root, year=2007, split=d, json_file=os.path.join(annotation_root,"annotations", d.lower()+".json"), class_names=["hand", "firstobject", "secondobject"])
        MetadataCatalog.get(f'Hands23_{d}').set(evaluator_type='coco')
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        timeout=timedelta(minutes=60)
        )

