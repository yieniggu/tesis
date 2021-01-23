import os, glob
import time
import numpy as np
import argparse
import pandas as pd
import re
import image_utils as imgutils
from bsort import Sort
from results_utils import SortResults, SortPerformanceResults
import sys

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
        imgutils.display_frame(random_frame, scaled=True, message="Tracker")
    
    print("[WARMUP] Visualization warmup finished!")

def track_objects(detections_path, frames_path, output_path, display_results=False, save_results=False):    
    # perform warmup to visualize results in "real time"
    warmup(20)
    
    # instantiate a tracker results object
    sort_results = SortResults()
    sort_performance = SortPerformanceResults()

    # get the detections file
    detections = pd.read_csv(args.detections_path).iloc[:, 1:]

    # get image dims to set boundaries
    width = detections["width"].unique()[0]
    height = detections["height"].unique()[0]

    # instantiate the tracker with given boundaries
    tracker = Sort(dims=(width, height))

    # read frames sorted by numerical order
    frame_paths = sorted(glob.glob(frames_path + "*.png"), key=numericalSort)

    frame_count = 1

    for frame_path in frame_paths:
        frame_filename = frame_path.split('/')[-1]
        # init sort results metadata (imaage and frame count)
        sort_results.init_frame_metadata(frame_path, frame_count)
        sort_performance.init_frame_metadata(frame_path, frame_count)

        # init a bbox array to store bboxes
        current_bboxes = np.array([[]]).astype(int)
        
        # get detections of current frame
        pig_detections_data = detections.loc[(detections["frame"] == frame_count) & (detections["detection_class"] == 'pig')]
        if pig_detections_data.shape[0] == 0:
            print("[SORT] No detections for pigs here, skipping frame {}"
                    .format(frame_count))
            frame_count += 1
            continue            
        print(pig_detections_data.head())
        #sys.exit()

        # get frame and its dimensions
        print("[SORT] Loading image with opencv...")
        start_loading = time.time()
        frame = imgutils.read_image_from_cv2(frame_path)
        total_loading = (time.time()-start_loading)*1000
        print("[SORT] Done!, it took {}ms".format(total_loading))
        frame_height, frame_width, _ = frame.shape

        # get detections results image dims
        width = pig_detections_data["width"].unique()[0]
        height = pig_detections_data["height"].unique()[0]

        print("[SORT] Starting image preprocessing...")
        start_preprocessing = time.time()
        # resize image to match detections results
        if (frame_height != height) or (frame_width != width):
            print("[SORT] Frame dims are not the same from the detections")
            print("[SORT] Resizing...")
            resized_frame = imgutils.resize_image(frame, (width, height))
            print("[SORT] Done!")

        else:
            print("[SORT] Frame dims are the same from the detections")
            resized_frame = frame
        total_preprocessing = (time.time()-start_preprocessing)*1000
        print("[SORT] Done!, it took {}ms".format(total_preprocessing))
        
        print("[SORT] Preparing detections for tracker...")
        start_preparing = time.time()
        for _, row in pig_detections_data.iterrows():
            # prepare bbox detections to format required by sort tracker
            x1 = row.xmin
            x2 = row.xmax
            y1 = row.ymin
            y2 = row.ymax

            # check if there is a valid detection
            if valid_detection([x1, x2, y1, y2]):
                print("[SORT] Detection is valid!")
                # fomat detection for sort input
                bbox = np.array([[x1, y1, x2, y2]]).astype(int)

                print("[SORT] Bbox: ", bbox)
                if current_bboxes.size == 0:
                    current_bboxes = bbox
                else:
                    np.append(current_bboxes, bbox, axis=0).astype(int)
            else:
                print("[SORT] Detection is invalid!, skipping...")

            
        total_preparing = (time.time()-start_preparing)*1000
        print("[SORT] Done!, it took {}ms".format(total_preparing))

        # send bbox to tracker
        # if detection is unmatched, it will initialize a new tracker
        # if its matched it should should, predict and update
        print("[SORT] Updating trackers...")
        start_update = time.time()
        objects_tracked = tracker.update(current_bboxes)
        total_update = (time.time()-start_update)*1000
        print("[SORT] Done!, it took {}ms".format(total_update))

        print("[SORT] Actual trackers: ", tracker.total_trackers)
        print("[SORT] Objects tracked: ", objects_tracked)
        #sys.exit()
        # get tracking results
        print("[SORT] Getting tracker results...")
        start_results = time.time()

        # set a copy of the current frame to display results
        drawed_frame = np.copy(resized_frame)
        for object_tracked in objects_tracked:
            # get trackers info like state and time since update
            tracker_id = object_tracked.id
            tracker_state = "active" if object_tracked.active else "inactive"
            time_since_update = object_tracked.time_since_update
            initialized_in_roi = object_tracked.initialized_in_ROI

            # get the bbox returned as [x1, y1, x2, y2]
            bbox = object_tracked.get_state().astype(int)
            first_centroid = (object_tracked.first_centroid[0], object_tracked.first_centroid[1])
            last_centroid = (object_tracked.last_centroid[0], object_tracked.last_centroid[1])

            # get x-y coords
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]

            print("[SORT] Bbox from tracker: ", bbox)
            print("[SORT] Active?: ", tracker_state)
            print("[SORT] Time since update: ", time_since_update)
            print("[SORT] Initialized in roi?: " , initialized_in_roi)

            # draw bboxes and label on frame
            drawed_frame = imgutils.draw_trackers_bounding_box(drawed_frame, "PIG", object_tracked)
         
            # add a new result to trackers results
            sort_results.add_new_result(width, height,
                                    tracker_id, tracker_state, 
                                    time_since_update,
                                    initialized_in_roi,
                                    first_centroid, last_centroid,
                                    [xmin, xmax, ymin, ymax])

        total_results = (time.time()-start_results)*1000
        print("[SORT] Tracker results done!, it took {}ms".format(total_results))

        # update trackers state to active/inactive depending on position
        print("[SORT] Updating state to  trackers")
        start_update_state = time.time()
        tracker.update_trackers_state()
        total_update_state = (time.time()-start_update_state)*1000
        print("[SORT] States update!, it took {}ms".format(total_update_state))

        # draw trackers info on the image
        drawed_frame = imgutils.draw_tracker_info(drawed_frame, "PIG", tracker)
        # display results on screen, if scaled will adapt frame to screen dimensions
        if display_results:
            imgutils.display_frame(drawed_frame, scaled=True, message="Tracker")
        # save results to a given path
        if save_results:
            imgutils.save_frame(drawed_frame, output_path, detections_path, frame_count, "tracker_frames")
            
        total_trackers = tracker.total_trackers
        sort_performance.add_new_result(width, height, total_loading, total_preprocessing,
                                        total_preparing, total_update, total_results,
                                        total_update_state, total_trackers)

        frame_count+=1

    # save tracker results on given path
    sort_results.save_results(output_path, detections_path)
    # save performance results on given path
    sort_performance.save_results(output_path, detections_path)

    return tracker.total_trackers
    #print(detections.head(10))
    #print(frame_paths)

def save_total_objects(total_objects, output_path, detections_path):
    last_path_of_detections_path = os.path.basename(os.path.normpath(detections_path))
    splitted_detections_path = last_path_of_detections_path.split("_")

    model_name = "_".join(splitted_detections_path[0:1])
    precision = splitted_detections_path[2]
    threshold = splitted_detections_path[3]
    frame_dims = splitted_detections_path[6]
    loading_backend = splitted_detections_path[7].split(".")[0]

    if output_path[-1] != '/':
        output_path+= '/'

    output_path += "tracker_results/"

    imgutils.create_dir(output_path)

    output_path += "{}_{}_{}_total_objects_{}_{}.txt".format(model_name, precision, threshold, frame_dims, loading_backend)

    print("[SORT] Writing total objects: {} to {}".format(total_objects, output_path))
    total_objects_file = open(output_path,"w") 

    total_objects_file.write(str(total_objects))
    total_objects_file.close()
    print("[SORT] Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform tracking offline on detections results")

    parser.add_argument('--detections_path', type=str, default=None,
                        help='File containing detections per frame summary')

    parser.add_argument('--frames_path', type=str, default='dogs.jpeg',
                        help='Path to frames to perform tracking')

    parser.add_argument('--display', action='store_true',
                        help='Select to show results')

    parser.add_argument('--save', action='store_true',
                        help='Select to save results')

    #parser.add_argument('--label', type=str, default='../models/detection/coco_labels.txt',
    #                   help='Path to labels of the model')

    args = parser.parse_args()

    output_path = os.path.dirname(os.path.dirname(args.detections_path))

    total_objects = track_objects(detections_path=args.detections_path, frames_path=args.frames_path,
                            output_path=output_path, display_results=args.display, 
                            save_results=args.save)

    save_total_objects(total_objects, output_path, args.detections_path)




