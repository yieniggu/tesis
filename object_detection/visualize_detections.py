import argparse
import os, glob
import pandas as pd
import re
import image_utils as imgutils
import numpy as np

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def valid_detection(detection):
    if "N/D" in detection:
        return False
    
    return True

def warmup(warmup_iters=10):
    print("[WARMUP] Starting visualization warmup on {} iters...".format(warmup_iters))
    for i in range(warmup_iters):
        random_frame = np.random.normal(size=(1920, 1080, 3)).astype(np.uint8)
        imgutils.display_frame(random_frame, scaled=True)
    
    print("[WARMUP] Visualization warmup finished!")

def visualize_detections(output_path, detections_path, frames_path, save_results=False):
    labels = ['pig', 'person']
    warmup(20)
    
    # load detections data
    detections = pd.read_csv(detections_path)

    # get frame paths
    frame_paths = sorted(glob.glob(frames_path + "*.png"), key=numericalSort)

    frame_count = 1
    for frame_path in frame_paths:
        # load the frame
        frame = imgutils.read_image_from_cv2(frame_path)
        # get frame dimensions
        height, width, _ = frame.shape

        # get detections
        frame_detections = detections.loc[detections["frame"] == frame_count]

        # copy the image to draw on it
        drawed_frame = frame.copy()

        for _, row in frame_detections.iterrows():
            # prepare bbox detections 
            bbox = [row.ymin, row.xmin, row.ymax, row.xmax]

            # check if there is a valid detection
            if valid_detection(bbox):
                print("[VIS-DETECTIONS] Detection is valid!")

                # get label params
                label = row.detection_class
                score = float(row.detection_score)

                print("[VIS-DETECTIONS] Label: {}".format(label))
                print("[VIS-DETECTIONS] Score: {}".format(score))

                det_width = frame_detections["width"].unique()[0]
                det_height = frame_detections["height"].unique()[0]

                # if frame imensions are diferent from frame detections dims
                # inference was performed with network input size image
                # we will resize bounding boxes to original frame size
                if (det_width != width) | (det_height != height):
                    bbox = imgutils.resize_bounding_box(frame.shape, (det_height, det_width, 3), bbox)

                drawed_frame = imgutils.draw_bounding_box(drawed_frame, bbox, label, score)
            else:
                break

        imgutils.display_frame(drawed_frame, scaled=True)
        if save_results:
            imgutils.save_frame(drawed_frame, output_path, detections_path, frame_count)

        frame_count+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform tracking offline on detections results")

    parser.add_argument('--detections_path', type=str, default=None,
                        help='File containing detections per frame summary')

    parser.add_argument('--frames_path', type=str, default='dogs.jpeg',
                        help='Path to frames to perform tracking')

    parser.add_argument('--save', action='store_true',
                        help='Select to save results')

    args = parser.parse_args()

    # returns /home/basti/Desktop/tesis/test/raw/video1/results/efficientdet_d0/
    output_path = os.path.dirname(os.path.dirname(os.path.dirname(args.detections_path)))
    
    visualize_detections(output_path, args.detections_path, args.frames_path, args.save)