# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from os import O_NONBLOCK
from collections import defaultdict
from typing import Dict
import torch
import numpy as np
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou

from hodetector.modeling.roi_heads.mlp_layer import build_z_head
from hodetector.modeling.roi_heads.hand_side import build_h_head
from hodetector.modeling.roi_heads.tool import build_t_head
from hodetector.modeling.roi_heads.grasp import build_g_head

from hodetector.utils.positional_encoding import *
from typing import Dict, List, Optional, Tuple
from torch.nn import functional as F

import cv2
import pdb
import math
from math import nan
import time
import pdb

from datetime import datetime


def write_error_mess(mess = ''):
    f = open("../../error_mess.txt", "a")
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")

    f.write("Error with " + mess +" at " + str(current_time) +'\n')
    f.close()


@ROI_HEADS_REGISTRY.register()
class hoRCNNROIHeads(StandardROIHeads):
    """
    The ROI specific heads for ho R-CNN
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self.z_head = build_z_head(cfg, 2089)  # Interaction
        self.h_head = build_h_head(cfg, 2089)  # Hand Side
        self.t_head = build_t_head(cfg, 2089)  # Tool
        self.g_head = build_g_head(cfg, 2089)  # Grasp

        self.cfg = cfg
        
        self._misc = {}

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
       
       
        if self.training:
            assert targets, "'targets' argument is required during training"

            proposals = self.label_and_sample_proposals(proposals, targets)
           

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.

            del images
            del targets
            
            return proposals, losses
        else:

            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            
            return pred_instances, {}


    def _forward_box(self, features_org: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        
        features = [features_org[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)


        if self.training:
            # select hand features
            fore_proposals, _ = select_foreground_proposals(proposals, 3)
    
        del box_features

      

        if self.training:


            losses_z, losses_h, losses_t, losses_g= self._forward_z(features, fore_proposals)

            losses = self.box_predictor.losses(predictions, proposals)

            losses.update(self._forward_mask(features_org, proposals))

            losses.update(losses_z)
            losses.update(losses_h)
            losses.update(losses_t)
            losses.update(losses_g)

           
            for val in losses.values():
                
                assert torch.isnan(val)== False and torch.isfinite(val)
               
            return losses
        else:
            
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)

            z_instances, pred_instances = self._inference_z(features, pred_instances)
           
            try:
                pred_instances = self.forward_with_given_boxes(features_org, pred_instances)
            except:
                
               pred_instances[0].set("pred_masks", torch.zeros((0,1,0,0), dtype=z_instances[0].dtype, device=z_instances[0].device))
             
            
            for z, instances in zip(z_instances, pred_instances):
                instances.pred_dz = z

            
            return pred_instances


    # Get Pairwise features 
    def get_PF(self, z_feature, bbox, classes, touch):

        N,F = z_feature.shape

        PF = torch.zeros((N,N,2*F), device= z_feature.device, dtype=z_feature.dtype)

        v = torch.zeros((N,N,2), device= z_feature.device, dtype=z_feature.dtype )
        min_dist = torch.zeros((N,N, 2), device= z_feature.device, dtype=z_feature.dtype )
        max_dist = torch.zeros((N,N, 2), device= z_feature.device, dtype=z_feature.dtype )
        v_norm = torch.zeros((N,N,1), device= z_feature.device, dtype=z_feature.dtype)
        v_over_v_norm = torch.zeros((N,N,2), device= z_feature.device, dtype=z_feature.dtype)

        def sdf(hmin, hmax, val):
            if val < hmin:
                return hmin - val
            elif val > hmax:
                return val -hmax
            else:
                return min(hmin-val, val-hmax)
        
        diag = 0
        for i in range(N):
            if classes[i]==0 or classes[i]==1:
                x1,y1,x2,y2 = bbox[i]
                x1 = x1.item()
                x2 = x2.item()
                y1 = y1.item()
                y2 = y2.item()
                diag = math.sqrt((x2-x1)**2+(y2-y1)**2)

            for j in range(N):
                PF[i,j,:F] = z_feature[i]
                PF[i,j, F:] = z_feature[j]

                is_tool = True if touch is None else touch[i] == 2 
                
                #hand-versus-objects and objects-versus-second objects
                if (classes[i]==0 and classes[j]==1) or (classes[i]==1 and is_tool  and classes[j]==2):
                    X1,Y1,X2,Y2 = bbox[j]
                    X1 = X1.item()
                    X2 = X2.item()
                    Y1 = Y1.item()
                    Y2 = Y2.item()
                    v[i][j][0] = (X1+X2 - x1 - x2)/2
                    v[i][j][1] = (Y1+Y2 - y1 - y2)/2
                    v_norm[i][j][0] = math.sqrt( v[i][j][0]**2 + v[i][j][1]**2)
                    min_dist[i][j][0] =  min( sdf(x1,x2,X1), sdf(x1,x2,X2))/diag
                    min_dist[i][j][1] =  min( sdf(y1,y2,Y1), sdf(y1,y2,Y2))/diag
                    max_dist[i][j][0] =  max( sdf(x1,x2,X1), sdf(x1,x2,X2))/diag
                    max_dist[i][j][1] =  max( sdf(y1,y2,Y1), sdf(y1,y2,Y2))/diag
                    v_over_v_norm[i][j] = v[i][j]/v_norm[i][j]

        return torch.cat((PF,v,v_norm,min_dist, max_dist, v_over_v_norm), 2).to(z_feature.device)


    def _inference_z(self, features, pred_instances):

        image_size = pred_instances[0].image_size
    
        pred_boxes_raw =  [x.pred_boxes for x in pred_instances][0].tensor
        pred_classes_raw =  [x.pred_classes for x in pred_instances][0]
        pred_scores = [x.scores for x in pred_instances][0].reshape(-1,1)
        device = pred_boxes_raw.device

        pred_boxes = []
        pred_classes = []
        pred_score = []

        for i in range(pred_scores.shape[0]):
            if (pred_classes_raw[i] == 1 and pred_scores[i] >= float(self.cfg.FIRSTOBJ)) or (pred_classes_raw[i] == 2 and  pred_scores[i] >= float(self.cfg.SECONDOBJ) ) or (pred_classes_raw[i] == 0 and  pred_scores[i] >= float(self.cfg.HAND)):

                pred_boxes.append([x for x in pred_boxes_raw[i]])
                pred_classes.append(pred_classes_raw[i].item())
                pred_score.append(pred_scores[i])
        

        
        pred_boxes = torch.tensor(pred_boxes).to(device)
        pred_boxes = [Boxes(pred_boxes)]
        pred_boxes_ = pred_boxes[0]
        

        pred_scores_ = torch.tensor(pred_score).to(device)
        

        pred_classes = torch.tensor(pred_classes).to(device)
        pred_classes_ = pred_classes.clone().detach().to(device)
       

        z_feature = self.box_pooler(features, pred_boxes)
        z_feature = self.box_head(z_feature)

        boxes = pred_boxes[0].tensor[range(len(z_feature))]
       

        try:
            Ids = F.one_hot(pred_classes, num_classes = 3)
        except:
            temp = torch.zeros(0,1).to(device)
            temp_g = torch.zeros(0,8).to(device)
            temp_t = torch.zeros(0,7).to(device)
            return [torch.cat((boxes,temp, temp,temp, temp, temp, temp_g, temp_t), dim=1)], [Instances(image_size = image_size, pred_boxes = pred_boxes_, pred_classes = pred_classes_, scores = pred_scores_)]
        
        
       
        PF = self.get_PF(z_feature, boxes, pred_classes, None)

        
        score_z = self.z_head(PF)
        score_h = self.h_head(z_feature)
        score_t = self.t_head(z_feature)
        score_g = self.g_head(z_feature)
    

        cls_prob = F.softmax(score_z, 2)
        hand_side = F.softmax(score_h, 1)
        touch_type = F.softmax(score_t, 1)
        grasp = F.softmax(score_g, 1)

        inter_prob = torch.zeros_like(cls_prob)
        inter_prob[:,:,0] = 1

        length = len(inter_prob)

       
        try:

            for i in range(length):
                if pred_classes[i] == 0:
                    for j in range(length):
                        if pred_classes[j] == 1:
                            inter_prob[i][j] = cls_prob[i][j]
                elif pred_classes[i] == 1:
                    for j in range(length):
                        if pred_classes[j] == 2:
                            inter_prob[i][j] = cls_prob[i][j]
        except:
            raise
            print("error in assigning interaction probability")

        
    
        score = torch.zeros(score_z.shape[0], 1).to(device)
        interaction = -1 * torch.ones(score_z.shape[0], 1).to(device)
        contact_state = torch.zeros(score_z.shape[0], 1).to(device)

        h_side = torch.zeros(hand_side.shape[0],1).to(device)
        t_type = 100*torch.ones(hand_side.shape[0],1).to(device)
        g_type = 100*torch.ones(hand_side.shape[0],1).to(device)

        # first objects are only included for evaluation if they are associated to a hand
        # second objects are only included for evaluation if they are associated to a first object
        idx_to_include = torch.arange(start = 0, end = len(boxes), dtype = int, device=device)[pred_classes==0].tolist()


        for i in range(len(z_feature)):
            try:
                if pred_classes[i] == 0:
                    h_side[i] = torch.argmax(hand_side[i]).item()
                    g_type[i] =  torch.argmax(grasp[i]).item()
                
                    cls_prob_i = 1 - inter_prob[i][:,0].reshape(-1)
                    max_val  = torch.max(cls_prob_i)
                    idx = torch.argmax(cls_prob_i)


                    if max_val.item() >= self.cfg.HAND_RELA:

                        if idx.item() not in idx_to_include:
                            interaction[i] = len(idx_to_include)
                            idx_to_include.append(idx.item())
                        else:
                           
                            interaction[i] = idx_to_include.index(idx.item())
                    else:
                        interaction[i] = -1

                    contact_state[i] = 0 if interaction[i] == -1 else torch.argmax(inter_prob[i][int(idx.item())][1:]).item()+1
                    score[i] = max_val
                    
                elif pred_classes[i] == 1:
                    t_type[i] = torch.argmax(touch_type[i]).item()

                    try:
                        assert t_type[i] <= 8
                    except:
                        raise

                    cls_prob_i = 1 - inter_prob[i][:,0].reshape(-1)
                    max_val  = torch.max(cls_prob_i)
                    idx = torch.argmax(cls_prob_i)

                    if max_val.item() >= self.cfg.OBJ_RELA and t_type[i] == 2:
                        if idx.item() not in idx_to_include:
                            interaction[i] = len(idx_to_include)
                            idx_to_include.append(idx.item())
                        else:
                            
                            interaction[i] = idx_to_include.index(idx.item())

                    else:
                        interaction[i] = -1

                    score[i] = max_val
            except Exception as e:
                raise
                print(e)
                print("error in assign values")

        return [torch.cat((boxes, interaction, h_side, g_type, t_type, contact_state, score, grasp, touch_type,  1-inter_prob[:,:,0]), dim=1)[idx_to_include]], [Instances(image_size = image_size, pred_boxes = pred_boxes_[idx_to_include], pred_classes = pred_classes_[idx_to_include], scores = pred_scores_[idx_to_include])]
           

    def _forward_z(self, features, proposals):

        z_feature = self.box_pooler(features,  [x.proposal_boxes for x in proposals])
        z_feature = self.box_head(z_feature)
        
        # Get the ground truth information for each head
        interaction = proposals[0].get('gt_interaction').clone().detach()
        gt_id = proposals[0].get('gt_id').clone().detach() # unique instance id for interaction
        bboxes = proposals[0].get('proposal_boxes').tensor.clone().detach()
        gt_boxes = proposals[0].get('gt_boxes').tensor.clone().detach()
        gt_handSide = proposals[0].get('gt_handSide').clone().detach()
        gt_contactState = proposals[0].get('gt_contactState').clone().detach()
        gt_touch = proposals[0].get('gt_touch').clone().detach()
        gt_grasp = proposals[0].get('gt_grasp').clone().detach()
        gt_classes = proposals[0].get('gt_classes').clone().detach()

        length = z_feature.shape[0]
   
        iou = []

        # only keep instances with bbox over 0.5 IoU with group truth
        for i in range(length):
            x1,y1,x2,y2 = bboxes[i]
            X1,Y1,X2,Y2 = gt_boxes[i]
            x1 = x1.item()
            x2 = x2.item()
            y1 = y1.item()
            y2 = y2.item()
            X1 = X1.item()
            X2 = X2.item()
            Y1 = Y1.item()
            Y2 = Y2.item()

            intersect = (min(x2,X2)-max(x1,X1))*(min(y2,Y2)-max(y1,Y1))
            union = (x2-x1)*(y2-y1)+(X2-X1)*(Y2-Y1)
            if (intersect/union)>=0.5:
                iou.append(i)
        
        length = len(iou)

        # feature for pairwise interaction prediction
        PF = self.get_PF(z_feature[iou], bboxes[iou],gt_classes[iou], gt_touch[iou])
       
        # ground truth for pairwise interaction prediction
        PL = 100 * torch.ones((length, length), device= interaction.device, dtype=interaction.dtype)
        
        # go through each pair of instances to determine the pairwise ground truth interaction
        for i in range(length):
            if gt_classes[iou[i]] == 0 or gt_classes[iou[i]] == 1:
                for j in range(length):
                    if (gt_classes[iou[i]] == 0 and gt_classes[iou[j]] == 1) :
                        PL[i][j] = gt_contactState[iou[i]] if interaction[iou[i]] == gt_id[iou[j]] else 0
                    elif (gt_classes[iou[i]] == 1 and gt_touch[iou[i]] == 2 and gt_classes[iou[j]] == 2):
                        PL[i][j] = gt_contactState[iou[i]] if interaction[iou[i]] == gt_id[iou[j]] else 0
                       
        # forward to each head
        pred_z = self.z_head(torch.flatten(PF, start_dim=0, end_dim=1))
        pred_h = self.h_head(z_feature[iou])
        pred_t = self.t_head(z_feature[iou])
        pred_g = self.g_head(z_feature[iou])
       
        
        assert pred_z.shape[0] == torch.flatten(PL, start_dim=0, end_dim=1).shape[0]
        losses_inter = self.z_head.losses(pred_z, torch.flatten(PL, start_dim=0, end_dim=1))  if torch.all(PL.eq(100)).item() is False else {"loss_relation": torch.tensor(0, device = PL.device)}

        assert pred_h.shape[0] == gt_handSide[iou].shape[0]
        losses_side = self.h_head.losses(pred_h, gt_handSide[iou]) if torch.all(gt_grasp[iou].eq(100)).item() is False else {"loss_hand_side": torch.tensor(0, device = pred_h.device)}
        assert pred_t.shape[0] == gt_touch[iou].shape[0]
        losses_touch = self.t_head.losses(pred_t, gt_touch[iou]) if torch.all(gt_touch[iou].eq(100)).item() is False else {"loss_touch": torch.tensor(0, device = pred_t.device)}

        assert pred_g.shape[0] == gt_touch[iou].shape[0] 
        losses_grasp = self.g_head.losses(pred_g, gt_grasp[iou]) if torch.all(gt_grasp[iou].eq(100)).item() is False else {"loss_grasp": torch.tensor(0, device = pred_t.device)}

        return losses_inter, losses_side, losses_touch, losses_grasp