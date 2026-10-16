import json, glob, os,shutil, pdb, random, cv2, argparse, shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from data_util import *
from detectron2.structures import BoxMode

random.seed(0)

coco = {}

handv2_info = {
    "description": "Hands23 Labels.",
    "url": "",
    "version": "1.0",
    "year": 2023,
    "contributor": "Tianyi Cheng*, Dandan Shan*, Ayda Hassen, Richard Higgins, David Fouhey"
}

licenses = []

images = []



'''
object categories: 
    1:hand
    2:firstobject
    3:secondobject
    
hand features
    side:
        0:left
        1:right
        2:ignore
    contact:
        0:not in contact
        1:self_contact
        2:other_person_contact
        3:object_contact
        100:ignore
    Box:
        [x1, y1, w, h]
    segment:
    grasp:
        0:NP-Palm
        1:NP-Fin
        2:Pow-Pris
        3:Pre-Pris
        4:Pow-Circ
        5:Pre-Circ
        6:Exten
        7:Later
        8:Other
        100:ignore
    interaction:
        the unique instance id of the instance in contact
        -1: ignore

firstobject: 
    Box: 
        [x1, y1, w, h]
    segment:
    touch:
        0:tool,touched
        1:tool,held
        2:tool,used
        3:container,touched
        4:container,held
        5:neither,touched
        6:neither,held
        100:ignore
    contact(if touch is tool,used):
        0:not in contact
        4:tool-second object contact
        100:ignore
    interaction:
        the unique instance id of the instance in contact
        -1: ignore
    
secondobject:
    Box:
        [x1, y1, w, h]
    segment:
'''


def add_item(image_id=None, 
              category_id=None,
              id=None,
              bbox=None,
              area=None,
              segmentation=None,
              iscrowd=0,
              handside=2,
              incontact=100,
              grasptype=100,
              touch=100,
              interaction = -1,
              bbox_mode = BoxMode.XYWH_ABS
              ):
    item = {}
    item['id'] = id
    item['image_id'] = image_id
    item['category_id'] = category_id
    #
    item['bbox'] = bbox
    item['area'] = area
    item['segmentation'] = segmentation
    item['iscrowd'] = iscrowd
    # additional hand
    item['handside'] = handside
    item['isincontact'] = incontact
   
    item['grasptype'] = grasptype
    # additional obj
    item['touch'] = touch
    item['interaction'] = interaction
    item['bbox_mode'] = bbox_mode

    return item




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=None, help='Number of jsons to process.')
    parser.add_argument('--copy_img', action='store_true', help='Whether to copy image.')
    parser.add_argument('--split', nargs='+', required=True, help='Which split to generate COCO annotations.')
    parser.add_argument('--data_dir', required=True, help='Path to downloaded dataset', default = "../data/hands23_data")
    parser.add_argument('--save_dir', required=True, help='Directory to save the generated json file', default = "../data/hands23_data_coco")
    args = parser.parse_args()
    
    data_dir = args.data_dir

    txtBase = os.path.join(data_dir, 'allMergedTxt')
    src      = os.path.join(data_dir, 'allMergedBlur')
    maskBase = os.path.join(data_dir, 'masks_sam')
    splitBase = os.path.join(data_dir, 'allMergedSplit')

    root_dir = args.save_dir
   
    os.makedirs(root_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test', 'hard','fail', 'annotations']:
        folder = f'{root_dir}/{split}'
        os.makedirs(folder, exist_ok=True)
        
    # loop for each image
    allImages = sorted([fn for fn in os.listdir(src) if fn.endswith(".jpg")])
    random.shuffle(allImages)
    print(f'#(total img) = {len(allImages)}')
    
    graspType_ls = []
    touch_ls  = []
    side_ls      = []
    state_ls     = []

    empty_list = []

    for split in args.split:

        splitPath = os.path.join(splitBase, split.upper()+'.txt')
        splitContent = open(splitPath).read().strip()
        images = [] if len(splitContent) == 0 else splitContent.split("\n")
        print(f'{split} - {len(images)}')
        
        img_ls, annot_ls = [], []
        img_id, annot_id = 0, 0

        for fn in tqdm(images):
            imagePath = os.path.join(src, fn)
            textPath = os.path.join(txtBase, fn+".txt")
            
            data = open(textPath).read().strip()
            lines = [] if len(data) == 0 else data.split("\n")
            
            I = cv2.imread(imagePath)
            h, w = I.shape[0], I.shape[1]
            
            img_item = {
                'id': img_id,
                'file_name': fn,
                'height': h,
                'width': w
            }
            
            # copy imag
            if args.copy_img:
                newPath = os.path.join(root_dir, split, fn)
                shutil.copy(imagePath, newPath)
                
                
            # loop for each object in current image
            for lineI, line in enumerate(lines):
                side, state, handBox, objectBox, touch, secObjectBox, graspType = map(lambda x: x.strip(), line.split("|"))
                print(lineI, line)

                first_objs = {}
                sec_objs = {}
                
                if graspType not in graspType_ls:  
                    graspType_ls.append(graspType)
                if touch not in touch_ls:
                    touch_ls.append(touch)
                if side not in side_ls:
                    side_ls.append(side)
                if state not in state_ls:
                    state_ls.append(state)
            
                handMask = cv2.imread(os.path.join(maskBase, ("2_%d_" % (lineI))+fn.replace(".jpg",".png")))

                handMask = (handMask[:, :, 0] > 128).astype(np.uint8)
                handArea, handPolygon = parseMask(handMask)     
                

                
                if objectBox == "None":
                    # hand only without first object
                    item = add_item(
                        image_id     = img_id, 
                        category_id  = 1,
                        id           = annot_id,
                        bbox         = boxStr2xywh(handBox, h, w),
                        bbox_mode    = BoxMode.XYWH_ABS,
                        area         = handArea,
                        segmentation = handPolygon,
                        handside     = parseSide(side),
                        incontact    = parseState(state),
                        grasptype    = parseGraspType(graspType)
                    )
                    annot_ls.append(item)
                    annot_id += 1

    
                else:
                    objectBbox = boxStr2xywh(objectBox, h, w)
                    objectMask = cv2.imread(os.path.join(maskBase, ("3_%d_" % (lineI))+fn.replace(".jpg",".png")))
                    objectMask = (objectMask[:,:,0] > 128).astype(np.uint8)
                    objectArea, objectPolygon = parseMask(objectMask)

                    object_id = annot_id+1

                    first_object_unique = tuple(objectBbox) not in first_objs

                    if first_object_unique:
                        first_objs[tuple(objectBbox)] = object_id
                    else:
                        object_id = first_objs[tuple(objectBbox)]

                    
                    # hand with first-object 
                    item =  add_item(
                        image_id     = img_id, 
                        category_id  = 1,
                        id           = annot_id,
                        bbox         = boxStr2xywh(handBox, h, w),
                        bbox_mode    = BoxMode.XYWH_ABS,
                        area         = handArea,
                        segmentation = handPolygon,
                        handside     = parseSide(side),
                        incontact    = parseState(state),
                        grasptype    = parseGraspType(graspType),
                        interaction  = object_id
                    )
                    annot_ls.append(item)
                    annot_id += 1
                    
                    
                    if secObjectBox == "None":
                        # firs-tobject only, without second-object

                        if first_object_unique:
                            item = add_item(
                                image_id     = img_id, 
                                category_id  = 2,
                                id           = annot_id,
                                bbox         = objectBbox,
                                bbox_mode    = BoxMode.XYWH_ABS,
                                area         = objectArea,
                                segmentation = objectPolygon,
                                touch     = parseTouch(touch)
                            )
                            annot_ls.append(item)
                            annot_id += 1

                    else:
                        secObjectMask = cv2.imread(os.path.join(maskBase, ("5_%d_" % (lineI))+fn.replace(".jpg",".png")))
                        secObjectMask = (secObjectMask[:,:,0] > 128).astype(np.uint8)
                        secObjectArea, secObjectPolygon = parseMask(secObjectMask)
                        secObjectBbox = boxStr2xywh(secObjectBox, h, w)

                        second_object_id = annot_id+1

                        second_object_unqiue = tuple(secObjectBbox) not in sec_objs

                        if second_object_id:
                            sec_objs[tuple(secObjectBbox)] = second_object_id
                        else:
                            second_object_id = sec_objs[tuple(secObjectBbox)]

                        
                        # first-object with second-object
                        if first_object_unique:
                            item = add_item(
                                image_id     = img_id, 
                                category_id  = 2,
                                id           = annot_id,
                                bbox         = objectBbox,
                                bbox_mode    = BoxMode.XYWH_ABS,
                                area         = objectArea,
                                segmentation = objectPolygon,
                                incontact    = 4,
                                touch     = parseTouch(touch),
                                interaction  = second_object_id
                            )
                            annot_ls.append(item)
                            annot_id += 1
                        
                        
                        # second-object
                        if second_object_unqiue:
                            item = add_item(
                                image_id     = img_id, 
                                category_id  = 3,
                                id           = annot_id,
                                bbox         = secObjectBbox,
                                bbox_mode    = BoxMode.XYWH_ABS,
                                area         = secObjectArea,
                                segmentation = secObjectPolygon
                            )
                            annot_ls.append(item)
                            annot_id += 1
                
            img_ls.append(img_item)
            img_id += 1     
                
        print(f'hand side:{side_ls}')
        print(f'hand state:{state_ls}') 
        print(f'tool type:{touch_ls}')
        print(f'grasp type:{graspType_ls}')   
        
        
        # assembly
        categories = [
            {"id": 1, "name": "hand"},
            {"id": 2, "name": "firstobject"}, # tool / object
            {"id": 3, "name": "secondobject"} # tool interacted object
        ]
        
        coco['info']         = handv2_info
        coco['licenses']     = licenses
        coco['categories']   = categories  # 0: hand, 1: first object, 2: second object
        coco['images']       = img_ls
        coco['annotations']  = annot_ls

        # save
        f = open(f'{root_dir}/annotations/{split}.json', 'w')
        json.dump(coco, f, indent=4, cls=NpEncoder)
        
        # print
        print(f'#image = {len(img_ls)}')
        print(f'#annot = {len(annot_ls)}\n\n')

        