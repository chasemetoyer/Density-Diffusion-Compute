# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from cgi import test
import os
import torch
import cv2
import random
import numpy as np
import pdb
import copy
import argparse
import json
import glob
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from hodetector.data import register_ho_pascal_voc, hoMapper
from hodetector.modeling import roi_heads
from utils.vis_utils import vis_per_image


def parse_grasp(grasp_type):
    # get names for grasp
    if grasp_type == 0:
        return "NP-Palm"
    elif grasp_type == 1:
        return  "NP-Fin"
    elif grasp_type == 2:
        return "Pow-Pris"
    elif grasp_type == 3:
        return "Pre-Pris"
    elif grasp_type == 4:
        return "Pow-Circ"
    elif grasp_type == 5:
        return "Pre-Circ"
    elif grasp_type == 6:
        return  "Later"
    elif grasp_type == 7:
        return "Other"
    else:
        raise ValueError(f"Can not parse grasp: {grasp_type}")

def parse_contact(contact):
    # get names for contact
    if contact == 0:
        return "no_contact"
    elif contact == 1:
        return "other_person_contact"
    elif contact == 2:
        return "self_contact"
    elif contact == 3:
        return "object_contact"
    elif contact == 4:
        return "obj_to_obj_contact"  
    else:
        raise ValueError(f"Can not parse contact: {contact}")
                
def parse_touch(touch):
    # get names for touch
    if touch == 0:
        return "tool_,_touched"
    elif touch == 1:
        return "tool_,_held"
    elif touch == 2:
        return "tool_,_used"
    elif touch == 3:
        return "container_,_touched"
    elif touch == 4:
        return "container_,_held"
    elif touch == 5:
        return "neither_,_touched"
    elif touch == 6:
        return "neither_,_held"
    else:
        raise ValueError(f"Can not parse touch: {touch}")



class Hands:
    def __init__(self, hand_id, hand_bbox, hand_mask, contactState, hand_side, grasp, pred_score, grasp_scores = None):
        self.id = hand_id
        self.hand_bbox = hand_bbox
        self.contactState = parse_contact(contactState)
        self.hand_side = "right_hand" if hand_side==1 else "left_hand"
        self.obj_bbox = None
        self.obj_touch = None
        self.obj_touch_score = None
        self.second_obj_bbox = None
        self.grasp = parse_grasp(grasp)
        self.grasp_scores = grasp_scores
        self.hand_mask = hand_mask
        self.pred_score = round(pred_score,2)
        self.obj_bbox = None
        self.obj_touch = None
        self.obj_touch_clean = None
        self.obj_masks = None
        self.second_obj_bbox = None
        self.second_obj_masks = None
        self.has_first = False
        self.has_second = False
        self.obj_pred_score = None
        self.sec_obj_pred_score = None
    

    def set_first_obj(self, obj_bbox , obj_touch , obj_masks, pred_score, touch_scores = None):
        self.obj_bbox = obj_bbox
        self.obj_touch = parse_touch(obj_touch)
        self.obj_masks = obj_masks
        self.obj_pred_score = round(pred_score,2)
        self.has_first  = True
        self.obj_touch_score = touch_scores
         

    def set_second_obj(self, obj_bbox, obj_masks, pred_score):
        self.second_obj_bbox = obj_bbox
        self.second_obj_masks = obj_masks
        self.sec_obj_pred_score = round(pred_score,2)
        self.has_second = True
    

    def save_masks(self, save_dir, im, img_id, mess = ''):
        ims = copy.deepcopy(im)
        ims[:,:,:] = 0
        ims[self.hand_mask, :] = 255

      
        save_dir = os.path.join(save_dir, "masks"+mess)
        img_id = img_id.strip('\n')
        os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(save_dir+'/2_'+str(self.id)+'_'+img_id.split('.')[0]+'.png', ims)

        if self.has_first:
            ims[:,:,:] = 0
            ims[self.obj_masks, :] = 255
            cv2.imwrite(save_dir+'/3_'+str(self.id)+'_'+img_id.split('.')[0]+'.png', ims)

            if self.has_second:
                ims[:,:,:] = 0
                ims[self.second_obj_masks, :] = 255
                cv2.imwrite(save_dir+'/5_'+str(self.id)+'_'+img_id.split('.')[0]+'.png', ims)

    def parse_grasp_scores(self):
        grasp_dict = {}
        for type, score in zip(["NP-Palm","NP-Fin", "Pow-Pris", "Pre-Pris", "Pow-Circ", "Pre-Circ", "Later","Other"], self.grasp_scores):
            grasp_dict[type] = str(round(score.item(),4))
        return grasp_dict


    def parse_touch_scores(self):
        touch_dict = {}
        for touch,score in zip(["tool_,_touched", "tool_,_held", "tool_,_used", "container_,_touched", "container_,_held","neither_,_touched", "neither_,_held"], self.obj_touch_score):
            touch_dict[touch] = str(round(score.item(),4))
        return touch_dict

    def get_json(self):
        info = {}
        info['hand_id'] = self.id 
        info['hand_bbox'] = [str(x) for x in self.hand_bbox]
        info['contact_state'] = self.contactState
        info['hand_side'] = self.hand_side
        info['obj_bbox'] = [str(x) for x in self.obj_bbox] if self.obj_bbox is not None else None
        info['obj_touch'] = str(self.obj_touch)
        info['obj_touch_scores'] = self.parse_touch_scores() if self.has_first else None
        info['second_obj_bbox'] = [str(x) for x in self.second_obj_bbox]  if self.second_obj_bbox is not None else None
        info['grasp'] = self.grasp
        info['grasp_scores'] = self.parse_grasp_scores() 
        info['hand_pred_score'] = str(self.pred_score)
        info['obj_pred_score'] = str(self.obj_pred_score)
        info['sec_obj_pred_score'] = str(self.sec_obj_pred_score)
        return info
    
    def message(self):
        return str(self.hand_side) + ' | ' + str(self.contactState) + ' | ' + str(self.hand_bbox).replace('[','').replace(']','') + ' | ' +  str(self.obj_bbox).replace('[','').replace(']','') + ' | ' +str(self.obj_touch) + ' | ' + str(self.second_obj_bbox).replace('[','').replace(']','') + " | " + str(self.grasp) +'\n'


def deal_output(im, predictor):
    outputs = predictor(im)

    pred_boxes = outputs["instances"].get("pred_boxes").tensor.to("cpu").detach().numpy()
    pred_dz = outputs["instances"].get("pred_dz").to("cpu").detach().numpy()
    pred_classes =  outputs["instances"].get("pred_classes").to("cpu").detach().numpy()
    pred_scores = outputs["instances"].get("scores").to("cpu").detach().numpy()
    pred_masks = outputs["instances"].get("pred_masks").to("cpu").detach().numpy()


    interaction = torch.tensor(pred_dz[:, 4])
    hand_side = torch.tensor(pred_dz[:, 5])
    grasp =  torch.tensor(pred_dz[:, 6])
    touch_type = torch.tensor(pred_dz[:, 7])
    contact_state = torch.tensor(pred_dz[:,8])

    grasp_scores = torch.tensor(pred_dz[:,10:18])
    touch_scores = torch.tensor(pred_dz[:,18:25])

    hand_list = []
    count = 0

   
    for i in range(len(pred_classes)):
        if pred_classes[i] == 0:
            curr_hand = Hands(hand_id= count, hand_bbox=pred_boxes[i], hand_mask=pred_masks[i], contactState=int(contact_state[i].item()),hand_side=hand_side[i].item(), grasp = grasp[i].item(), pred_score= pred_scores[i], grasp_scores= grasp_scores[i])
            count = count+1

            if interaction[i] >=0:
                obj_id = int(interaction[i])

                curr_hand.set_first_obj(obj_bbox=pred_boxes[obj_id], obj_touch= touch_type[obj_id].item(), obj_masks=pred_masks[obj_id], pred_score= pred_scores[obj_id], touch_scores=touch_scores[obj_id])

                if interaction[obj_id] >=0:
                    second_obj_id = int(interaction[obj_id])
              

                    curr_hand.set_second_obj(obj_bbox=pred_boxes[second_obj_id], obj_masks=pred_masks[second_obj_id], pred_score= pred_scores[second_obj_id])

            hand_list.append(curr_hand)

    return hand_list


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

    cfg.freeze()

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_thresh", type=float, default=0.7)
    parser.add_argument("--first_obj_thresh", type=float, default=0.5)
    parser.add_argument("--second_obj_thresh", type=float, default=0.3)
    parser.add_argument("--hand_rela", type=float, default=0.3)
    parser.add_argument("--obj_rela", type=float, default=0.7)
  
    parser.add_argument("--model_weights", default=f"./model_weights/model_hands23.pth")
    parser.add_argument("--data_dir", default = 'demo_example/example_images/')
    parser.add_argument("--save_dir", default="results/demo")
    parser.add_argument("--save_img", type=bool,  default = True)
    parser.add_argument("--image_list_txt", default=None)
    parser.add_argument("--config_file", default="./faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml")
    args = parser.parse_args()
    
    # set configuration
    cfg = set_cfg(args)
    
    predictor = DefaultPredictor(cfg)
    
    # inputs
    print(f' a folder of images...')
    if args.image_list_txt is not None:
        f = open(args.image_list_txt)
        images = f.readlines()
    else:
        images = [x.replace(args.data_dir, '') for x in glob.glob(f'{args.data_dir}/*')]

    # outputs
    save_dir = args.save_dir
    save_mask_dir = f"{save_dir}/masks" 
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    save_img = args.save_img 

    # save results
    res = {}
    res["save_dir"] = save_dir
    res["images"] = []
    json_path = os.path.join(save_dir, "result.json")

    # loop
    for test_img in tqdm(images):
        test_img = test_img.strip("\n")
        print(f'Processing: {test_img}')
        if os.path.exists(os.path.join(args.data_dir, test_img)) is False:
            raise FileNotFoundError(os.path.join(args.data_dir, test_img))
        im = cv2.imread(os.path.join(args.data_dir, test_img))
        im_name = os.path.split(test_img)[-1][:-4]
        
        # record img res
        img = {}
        img["file_name"] = test_img
        img["predictions"] = []

        #save masks and vis
        hand_lists = deal_output(im = im, predictor= predictor)

        for hands in hand_lists:

            if save_img:
                hands.save_masks(save_dir, im, test_img.split('/')[-1])

            img['predictions'].append(hands.get_json())
        
        # vis and save
        if save_img:
            im = vis_per_image(im, img['predictions'], im_name+'.png', save_mask_dir, use_simple=False)
            save_path = os.path.join(save_dir, im_name+'.png')
            im.save(save_path)

        res["images"].append(img)

    f = open(json_path, 'w')
    json.dump(res, f, indent=4)


if __name__ == '__main__':
    main()

