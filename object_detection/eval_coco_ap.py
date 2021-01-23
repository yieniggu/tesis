from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

def get_ap(detections_path, gt_path):
    # get annotations from detections and ground truth
    COCO_gts = COCO(gt_path)
    COCO_dets = COCO_gts.loadRes(detections_path)

    #results_output = detections_path + "ap_results.json"

    cocoEval = COCOeval(COCO_gts, COCO_dets, 'bbox')
    cocoEval.params.imgIds = sorted(COCO_gts.getImgIds())
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Perform tracking offline on detections results")

    parser.add_argument('--detections_path', type=str, default=None,
                        help='Json containing detections bboxes')

    parser.add_argument('--gt_path', type=str,
                        help='Json containing ground truth bboxes')

    args = parser.parse_args()

    get_ap(args.detections_path, args.gt_path)
