# Hands23 Detector 
This is the official Detectron2 implementation for our paper, **Towards a richer 2d understanding of hands at scale** (NeurIPS 2023).

Tianyi Cheng*, Dandan Shan*, Ayda Sultan Hassen, Richard Ely Locke Higgins, David Fouhey

## Environment Installation

The environment depends on [pytorch](https://pytorch.org/get-started/locally/) and [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). We provide the versions tested on our machine below. You may also use versions match with your own machine. 

Feel free to use [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) which is a faster env management tool. After you install `mamba` correctly, simply switch `conda` to `mamba`. 
```
conda create -n hands23  python=3.10
conda activate hands23
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python parso tqdm pillow==9.5.0
```

## Pre-trained model weights:
Please download the model weights and save under `model_weights` 
```
mkdir model_weights && cd model_weights
wget https://fouheylab.eecs.umich.edu/~dandans/projects/hands23/model_weights/model_hands23.pth
```


## Demo

```
python demo.py [choose to use default arg values or customize on your own]
```
We support adjustmentss on inference thresholds for different purposes. Details about the threhsolds are defined below plus additional file path flags. By default, the thresholds are 0.7-0.5-0.3-3-0.7. We found thresholds that qualitatively had good trade offs between false positives and negatives. There are other threshold settings that have lower false positives at the cost of higher false negatives and vice-versa. For example, for a low false negative rate, you can try threshold setting of 0.5-0.1-0.3-0.3-0.7; and for a low false positive rate, you can try 0.9-0.8-0.3-0.8-0.7.

#### Available Flags

| Flag | Description | Default Value |
|------|-------------|---------------|
| `--hand_thresh` | Bbox score threshold for a hand bbox to be considered(float) | 0.7 |
| `--first_obj_thresh` | Bbox score threshold for a first object bbox to be considered(float) | 0.5 |
| `--second_obj_thresh` | Bbox score threshold for a second object bbox to be considered(float) | 0.3 |
| `--hand_rela` | Interaction score threshold for a (hand,first-object) pair to be considered in interaction(float) | 0.3 |
| `--obj_rela` | Interaction score threshold for a (tool, second-object) pair to be considered in interaction(float) | 0.7 |
| `--model_weights` | Path to model weight(str) | `model_weights/model_hands23.pth` |
| `--data_dir` | Path to the data(images) used for demo(str) | `demo_example/example_images` |
| `--image_list_txt` | Path to the .txt file with the list of images to run demo on. If not specified, the program will run on all images in data_dir(str) | `None` |
| `--save_dir` | Path to directory where demo results will be saved(str) | `result/demo` |
| `--save_img` | Whether visualized images will be saved(bool)  | `True` |

#### Results
The results of demo includes visualized images(if `--save_img` is True), per-instance masks in a folder named `masks`, and one `result.json` file which records the prediction results for all images at `save_dir`.

Segmentation predictions per instance are saved under folder `masks` in `classID_handID_imageName` (for `classID` `2` is `hand`, `3` is `first object` and `5` is `second object`, `handID` is the hand that the instance is associated with, and `imageName` is the image the instance is in).

The `result.json` takes the structure:
```json
{
    "save_dir": "example_result",
    "images": [
        {
            "file_name": "str",
            "predictions": [
                {
                    "hand_id": "int",
                    "hand_bbox": ["float"],
                    "contact_state": "str",
                    "hand_side": "str",
                    "obj_bbox": "[float in format of str]/'None'",
                    "obj_touch": "tool_,_used",
                    "obj_touch_scores": {"per category prediction score"},
                    "second_obj_bbox": "[float in format of str]/'None'",
                    "grasp": "str",
                    "grasp_scores": {"per category prediction score"},
                    "hand_pred_score": "float in format of str",
                    "obj_pred_score": "float in format of str/'None'",
                    "sec_obj_pred_score": "float in format of str/'None'"
                }
            ]
        }
    ]
}
```
Where each dict in "predictions" represents the prediction result for a hand object(and first object it interacts with and second object that interacts with the first object if they exists). For an example of demo result, please refer to [demo example](./demo_example) which contains [example raw images](./demo_example/example_images) and [example output](./demo_example/example_result).


## Hands23 Dataset: download and preparation
To get the Hands23 dataset, please follow the instructions in the [hands23_data repo](https://github.com/ddshan/hands23_data.pre_release) to download it and prepare the SAM masks. Put/Softlink Hands23 dataset here `data/hands23_data`

To generate the COCO JSON format of the data for evaluation and training, run:
```
python data_prep/gen_coco_format.py
```

## Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python trainval.py --num-gpus 8 --config-file faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml --data_root ./data/hands23_data_coco --output_dir ./model_outputs
```
Under the `data_root` directory, there should be an `annotations` directory which contains `train.json` and `val.json` in COCO format for training and evaluation.

## Evaluation
### Detection and Segmentation APs
```
CUDA_VISIBLE_DEVICES=0 python eval.py --num-gpus 1 --config_file faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml --model_weights path/to/model.pth
```
Similar to demo script `demo.py`, the evaluation script `eval.py` also supports adjustments on thresholds plus other file path flags.

#### Available Flags

| Flag | Description | Default Value |
|------|-------------|---------------|
| `--data_dir` | Path to the evaluation dataset in COCO format(str) | `./data/hands23_data_coco` |
| `--save_dir` | Path to directory where evaluation results will be saved(str) | `result` |
| `--save_result` | Whether to save numeric evaluation results(bool)  | `True` |
| `--image_root` | Path to the image for evaluation  | `./data/hands23_data/allMergedBlur` |

The per-category (totally 3 categories: hand, firstobject, secondobject) bounding box and segmentation evaluation results can be saved as a JSON file at `save_dir` (default: results/evaluation/result.json). The JSON file is in the format as:


```json
{
  "bbox": {
    "AP-50": {
      "hand": "float",
      "firstobject": "float",
      "secondobject": "float"
    },
    "mAP": {
      "hand": "float",
      "firstobject": "float",
      "secondobject": "float"
    }
  },
  "segm": {
    "AP-50": {
      "hand": "float",
      "firstobject":"float",
      "secondobject": "float"
    },
    "mAP": {
      "hand": "float",
      "firstobject": "float",
      "secondobject": "float"
    }
  }
}
```

### Classification Accuracies
To get the classification accuracies for states (hand side, contact, touch, grasp), the evaluation involves 2 steps: generate `result.json` using `demo.py` and calculate accuracies.

#### Step1 - Generate result.json

```
python demo.py --save_img=False
```

#### Step2 - Calculate state accuracy

```
python analyzeAcc.py --label_src [source of .txt files] --output_src [directory of saved result.json]
```

## Citation

If this work is helpful in your research, please cite:
```
@inproceedings{cheng2023towards,
  title={Towards a richer 2d understanding of hands at scale},
  author={Cheng, Tianyi and Shan, Dandan and Hassen, Ayda Sultan and Higgins, Richard Ely Locke and Fouhey, David},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
