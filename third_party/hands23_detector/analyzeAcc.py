#!/usr/bin/python

import os
import pdb
import json
import argparse


def getIou(box1, box2):
    """
    From stackoverflow....

    Implement the intersection over union (IoU) between box1 and box2

    Arguments:
        box1 -- first box, numpy array with coordinates (ymin, xmin, ymax, xmax)
        box2 -- second box, numpy array with coordinates (ymin, xmin, ymax, xmax)
    """
    # ymin, xmin, ymax, xmax = box

    y11, x11, y21, x21 = box1
    y12, x12, y22, x22 = box2

    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)

    if (yi2 - yi1 < 0) or (xi2 - xi1 < 0):
        return 0

    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)
    union_area = box1_area + box2_area - inter_area
    # compute the IoU
    iou = inter_area / union_area
    return iou

def boxStr2float(s):
    # print(s)
    data = [float(c) for c in s.split(",")]
    return (data[0], data[1], data[2], data[3])

def isRight(o):
    return 1 if o[1]['hand_side'].startswith("right") else 0

def tupleCorrect(t):
    if t[0] == None:
        return 0 
    return 1 if t[0] == t[1] else 0

def tupleCounts(t):
    return 1 if t[0] != None else 0

def evalTuples(ts):
    return float(sum(map(tupleCorrect,ts))) / sum(map(tupleCounts,ts))


def cmat(ts,abbr,order):
    #(gt,pred)
    cats = list(set(t[0] for t in ts).union(set(t[1] for t in ts)))
    cats.sort(key=lambda k:(order.index(k) if k in order else len(order)+1))
    cats = [c for c in cats if c != None]

    K = len(cats)

    CM = [ [0]*K for i in range(K) ]

    for t0, t1 in ts:
        if t0 == None or t1 == None:
            continue 
        CM[cats.index(t0)][cats.index(t1)] += 1

    print("%-7s " % "",end="")
    for i in range(K):
        print("%7s " % abbr[cats[i]], end="")
    print("") 
    for i in range(K):
        print("%-7s " % abbr[cats[i]], end="")
        for j in range(K):
            print("%7d " % CM[i][j], end="")
        print("") 
    


contacts = {}


toolTypeOrder = [ 'tool_,_used', 'tool_,_held', 'tool_,_touched', 'container_,_held', 'container_,_touched', 'neither_,_held', 'neither_,_touched']
toolTypes2Abbr = {None:"?", 'tool_,_held':"Tool H", 'tool_,_used':"Tool U", 'container_,_touched':"Con T", 'neither_,_held':"Nei H", 'neither_,_touched':"Nei T", 'tool_,_touched':"Tool T", 'container_,_held':"Con H"}
contactsOrder = ["no_contact","object_contact","self_contact","other_person_contact"]
contacts2Abbr = {None:"?", 'inconclusive':"?", "obj_to_obj_contact":"?", 'no_contact':"No", 'object_contact':"Obj", 'other_person_contact':"Other", 'self_contact':"Self"}
sidesOrder = ["right_hand","left_hand"]
sides2Abbr = {'right_hand':"Right", "left_hand":"Left"}
graspsOrder = ["NP-Fin","NP-Palm","Pow-Pris","Pow-Circ","Pre-Pris","Pre-Circ","Lat","None"]
grasps2Abbr = {'NP-Fin':"NP-F", 'Pre-Pris':"Pre P", 'Pow-Pris':"Pow P", 'Pre-Circ':"Pre C", None:"?", 'Lat':"Lat", 'NP-Palm':"NP-P", 'Pow-Circ':"Pow C"}


grasps23Way = {'NP-Fin':"NP", 'Pre-Pris':"Pre", 'Pow-Pris':"Pow", 'Pre-Circ':"Pre", None:None, 'Lat':"Pre", 'NP-Palm':"NP", 'Pow-Circ':"Pow"}
grasps3Order = ["NP","Pow","Pre"]
grasps3Abbr = {s:s for s in grasps3Order}

contacts2UHT = {None:None, 'tool_,_held':"Held", 'tool_,_used':"Used", 'container_,_touched':"Touched", 'neither_,_held':"Held", 'neither_,_touched':"Touched", 'tool_,_touched':"Touched", 'container_,_held':"Held"}
usedHeldTouchedOrder = ["Used","Held","Touched"]
usedHeldTouchedAbbr = {s:s for s in usedHeldTouchedOrder}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate EPIC-Kitchens Hand Object Segmentation challenge results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--label_src", type=str, default = './data/hands23_data/allMergedTxt'
    )
    parser.add_argument(
        "--output_src", type=str, default = './results/result.json'
    ) 
    args = parser.parse_args()


    labelSrc = args.label_src
    

    for handThreshold in [0]:

            outputSrc = args.output_src

            print(f'{handThreshold}')
            print(outputSrc)

            ObjectBoxIOUThresh = 0.1
            output = json.load(open(outputSrc))
            images = output['images']
            print("%d loaded" % len(images))

            contacts = []
            leftRights = []
            grasps = []
            toolTypes = []
            foundMatches = []

            #0 = trueNegative (gt=none, pred=none)
            #1 = truePositive (gt=box1,pred=box2, boxes match)
            #2 = falsePositive (gt=box1,pred=box2, boxes don't match) || (gt=none, pred=box)
            #3 = falseNegative (gt=box1,pred=None)
            boxStatuses = []

            for imageResultI, imageResult in enumerate(images):

                if imageResultI % 1000 == 0:
                    print("%d/%d" % (imageResultI, len(images)))

                gtData = []
        
                for line in open(os.path.join(labelSrc, imageResult['file_name']+".txt")):
                    #load the ground truth into something sensible
                    if not line.strip():
                        continue
                    label, state, handBox, objectBox, toolClass, toolBox, graspType = map(lambda x: x.strip(), line.split("|"))
                    handBox = boxStr2float(handBox)
                    objectBox = boxStr2float(objectBox) if objectBox != "None" else None
                    toolBox = boxStr2float(toolBox) if toolBox != "None" else None
                    graspType = graspType if graspType != "None" else None
                    state = state if state not in ["None","inconclusive"] else None
                    toolClass = toolClass if toolClass not in ["None"] else None
                    gtData.append( (label, state, handBox, objectBox, toolClass, toolBox, graspType) )
                
                if len(gtData) < handThreshold:
                    continue
                
                for prediction in imageResult['predictions']:
                    predHandBox = boxStr2float(", ".join([ str(x) for x in prediction['hand_bbox']]))
        

                    #whether we found a match
                    foundMatch = 0
                    
                    ious = [getIou(predHandBox, gtBox[2]) for gtBox in gtData]
                    maxIou = max(ious) if len(ious) else -1

                    if maxIou >= 0.5:
                        gtBox = gtData[ious.index(maxIou)]
                        contacts.append( (gtBox[1], prediction['contact_state']) )
                        leftRights.append( (gtBox[0], prediction['hand_side']) )

                       
                        grasps.append( (gtBox[6], prediction['grasp']) )
                        objTouch =  prediction['obj_touch'] if prediction['obj_touch'] != "None" else None
                        toolTypes.append((gtBox[4], objTouch))

                        try:
                            predObjBox = boxStr2float(",".join(prediction['obj_bbox'])) if prediction['obj_bbox'] is not None else None
                        except:
                            pdb.set_trace()

                        if gtBox[3] == None:
                            if predObjBox == None:
                                boxStatuses.append("tn") #true negative
                            else:
                                boxStatuses.append("fp") #false positive: no gt box, but predicted one
                        else:
                            if predObjBox == None:
                                boxStatuses.append("fn") #false negative: gt box, but no predicted one
                            else:
                                iou = getIou(predObjBox, gtBox[3])
                                boxStatuses.append("tp" if iou > ObjectBoxIOUThresh else "fpiou") #gt box, pred box, then 1 if match else 0

                        foundMatch = 1

                    foundMatches.append(foundMatch)

            print("Side\nAccuracy: %.3f" % evalTuples(leftRights))
            cmat(leftRights, sides2Abbr, sidesOrder)
            print("")

            print("Tool Types\nAccuracy: %.2f" % evalTuples(toolTypes))
            cmat(toolTypes, toolTypes2Abbr, toolTypeOrder)
            print("")

            usedHeldTouched = [(contacts2UHT[t0], contacts2UHT[t1]) for t0, t1 in toolTypes]
            print("Used/Held/Touched\nAccuracy: %.2f" % evalTuples(usedHeldTouched))
            cmat(usedHeldTouched, usedHeldTouchedAbbr, usedHeldTouchedOrder)
            print("")

            print("Grasps\nAccuracy: %.2f" % evalTuples(grasps))
            cmat(grasps, grasps2Abbr, graspsOrder)
            print("")

            grasps3Way = [(grasps23Way[t0], grasps23Way[t1]) for t0, t1 in grasps]
            print("Grasps 3\nAccuracy: %.2f" % evalTuples(grasps3Way))
            cmat(grasps3Way, grasps3Abbr, grasps3Order)
            print("")

            print("Contacts\nAccuracy: %.3f" % evalTuples(contacts))
            cmat(contacts, contacts2Abbr, contactsOrder)

            print("Matches found %d/%d = %.2f%%" % (sum(foundMatches) , len(foundMatches), float(sum(foundMatches))/len(foundMatches)))

            print("")
            print("First Box")
            toCheck = ["tp","tn","fp","fpiou","fn"]
            toCheckLabel = ["True Positive","True Negative","False Positive","False Positive due to IoU (thresh:%.2f)" % ObjectBoxIOUThresh, "False Negative"]
            counts = []
            for s in toCheck:
                counts.append(len([v for v in boxStatuses if v == s]))
            
            print("(TP+TN) / all: %.2f" % ((counts[0]+counts[1]+0.0) / sum(counts)))
            for i in range(len(toCheck)):
                print("%s: %d (%.2f)" % (toCheckLabel[i], counts[i], counts[i]*1.0/sum(counts)))