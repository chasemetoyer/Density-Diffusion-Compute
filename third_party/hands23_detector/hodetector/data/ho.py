# -*- coding: utf-8 -*-
# adapted from pascal_voc.py

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

import numpy as np
from parso import split_lines
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer
import pycocotools.mask as mask_util

import torch
import glob
import pdb
import random
import time
import contextlib
import logging
import json

import cv2
import io
import pdb


__all__ = ["load_ho_voc_instances", "register_ho_pascal_voc"]

# fmt: off
CLASS_NAMES = (
    "hand", "firstobject", "secondobject"
)
logger = logging.getLogger(__name__)

# fmt: on

def get_mask(class_id, row_num, im_id, mask_dir, dirname):
    mask_d = mask_dir+class_id+"_"+ str(row_num) +'_'+im_id.replace('.jpg', '')+'.png'

    try:
        mask = cv2.imread(mask_d)
        L = len(mask)
    except:
        try:
            im = cv2.imread(dirname+im_id)
            size = im.shape
        except:
            #pdb.set_trace()
            print("error")

        return np.ones(shape = (size[0], size[1], 1)).astype(bool), mask_d

    mask = mask[:,:,0] > 128
    mask = mask.astype(bool)

    return mask, ''

def seg_to_poly(x,y,w,h,box_segments):

    if box_segments.any() == None:
        return False

    contours, hierarchy = cv2.findContours(box_segments.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []

    for contour in contours:
        if contour.size >=6:
            segmentation.append(contour.flatten().tolist())
    
    float_segmentation = []

    for segseg in segmentation:
        float_segmentation.append([float(x) for x in segseg])
    
    
    return float_segmentation

def load_ho_voc_instances(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
   
    from pycocotools.coco import COCO


    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    

    assert os.path.exists(json_file)


    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    
    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
  
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])
    # print(f'extra_annotation_keys = {extra_annotation_keys}')
    # extra_annotation_keys = ['handside', 'incontact', 'offset', 'object_id']
    # print(f'extra_annotation_keys = {extra_annotation_keys}')

    num_instances_without_valid_segmentation = 0
    images_without_valid_segmentation = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []

        count = 0
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            ann_keys = anno.keys()
            obj = {key: anno[key] for key in ann_keys if key in anno}
            
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        # continue  # ignore this instance
                        height = img_dict["height"]
                        width = img_dict["width"]
                        x,y,w,h = obj["bbox"]
                        segm = seg_to_poly(x,y,w,h, np.ones((width,height,1)))
                        images_without_valid_segmentation.append(anno)

                        assert len(segm) != 0

                obj["segmentation"] = segm


            obj["bbox_mode"] = BoxMode.XYWH_ABS
            # obj["bbox_mode"] = BoxMode.XYXY_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e



            if obj["touch"] == -1:
                obj["touch"] = 100
            
            if obj["isincontact"] is None or obj['isincontact'] == -1:
                obj["isincontact"] = 100
            
            if obj["handside"] == -1:
                obj["handside"] = 2
            
            if obj["grasptype"] ==-1:
                obj["grasptype"] = 100
            
            
            objs.append(obj)


        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )

        print(images_without_valid_segmentation)
        f = open("./invalid_seg.json", "w+")
        json.dump(images_without_valid_segmentation, f, indent=4)
    
    return dataset_dicts


def register_ho_pascal_voc(name, dirname, year, split, json_file, class_names=CLASS_NAMES):
    #json_file, image_root, dataset_name=None, extra_annotation_keys=None
   
    DatasetCatalog.register(name, lambda: load_ho_voc_instances(json_file=json_file, image_root=dirname, dataset_name=name))
  
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

    

