from pycocotools.cocoeval import COCOeval, Params
from detectron2 import _C
from detectron2.utils.logger import create_small_table
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
import logging
import os
import numpy as np
import copy
from collections import defaultdict
from tabulate import tabulate
import itertools
import logging
import json
logger = logging.getLogger(__name__)

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


results_to_save = {}
save_dir = ''


class Hands23_Evaluator(COCOEvaluator):
    '''
    customed evaluator that calaculates per-class AP-50 and mAP
    store results at output directory if needed

    extended from COCOEvaluator from Detectron2 
    more information can be found at: https://detectron2.readthedocs.io/en/latest/modules/evaluation.html
    '''
    def __init__(
        self,
        dataset_name,
        save_dir,
        save_result,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,
    ):
        super().__init__(dataset_name)
        self.save_result = save_result
        self.save_dir = save_dir

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(
            *[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(
                all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k,
                                  v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(
                self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )

        for task in sorted(tasks):
            assert task in {"bbox", "segm",
                            "keypoints"}, f"Got unknown task: {task}!"

            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    cocoeval_fn=COCOeval_opt,  # if self._use_fast_impl else COCOeval,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get(
                    "thing_classes")
            )
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }

        self._logger.info(
            "Evaluation results for {}: \n".format(
                iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info(
                "Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[0, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it - added information about per-category AP-50
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP-50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        curr_result = {}
        curr_result["AP-50"] = {key: value for (key, value) in zip(
            results_flatten[0::2], results_flatten[1::2])}

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it - added information about per-category AP
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        # save the mAP information
        curr_result["mAP"] = {key: value for (key, value) in zip(
            results_flatten[0::2], results_flatten[1::2])}
        
        results_to_save = {}
        result_json_file_dir = os.path.join(self.save_dir, "result.json")

        if os.path.isfile(result_json_file_dir):
            results_to_save = json.load(open(result_json_file_dir, 'r'))
        
        results_to_save[coco_eval.params.iouType] = curr_result

        # if save_result is True, save the evaluation results in json format at save_dir
        if self.save_result:
            f = open(result_json_file_dir, "w")
            json.dump(results_to_save, f, indent=4)
            f.close()

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results