import argparse
import pandas as pd
import json
import os

def create_dir(output_path):
	    # creating the output directory that will contain the model
    print("[DIR-CREATION] Attempting to create output directory...")
    print("[DIR-CREATION] Files will be saved under {}"
        .format(output_path))
    if not os.path.exists(output_path):
        print("[DIR-CREATION] Directory doesnt exists, creating...") 
        os.makedirs(output_path)
        print("[DIR-CREATION] Directory created succesfully...") 
    else:
        print("[DIR-CREATION] Output directory already exists...")

def get_output_path(detections_path):
    output = os.path.dirname(detections_path)

    # get filename
    last_path_of_detections_path = os.path.basename(os.path.normpath(detections_path))
    splitted_detections_path = last_path_of_detections_path.split("_")

    model_name = "_".join(splitted_detections_path[0:2])
    precision = splitted_detections_path[2]
    threshold = splitted_detections_path[3]
    frame_dims = splitted_detections_path[6]
    loading_backend = splitted_detections_path[7].split(".")[0]

    if output[-1] != '/':
        output += '/'

    create_dir(output)   

    output += "{}_{}_{}_{}_{}.json".format(model_name, precision, 
                                        threshold, frame_dims, loading_backend)

                   
    return output

def convert_to_json(detections_path):
    detections = pd.read_csv(detections_path)

    output_path = get_output_path(detections_path)

    detections['detection_class'].replace({"pig": 1, "person": 2}, inplace=True)

    coco_detections = []

    frames = detections["frame"].unique()

    for frame in frames:
        frame_detections = detections.loc[detections["frame"] == frame]

        if "N/D" in frame_detections.values:
            continue

        for _, row in frame_detections.iterrows():
            image_id = row.frame
            class_id = row.detection_class
            xmin = int(row.xmin)
            ymin = int(row.ymin)
            xmax = int(row.xmax)
            ymax = int(row.ymax)

            bbox_coco_fmt = [
                xmin,  # xmin
                ymin,  # xmax
                (xmax - xmin),  # width
                (ymax - ymin),  # height
            ]

            score = row.detection_score

            coco_detection = {
                'image_id': image_id,
                'category_id': class_id,
                'bbox': [int(coord) for coord in bbox_coco_fmt],
                'score': float(score)
            }

            coco_detections.append(coco_detection)

    print("[CSV-TO-JSON] Saving json results to {}".format(output_path))
    with open(output_path, 'w') as f:
        json.dump(coco_detections, f)
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Perform tracking offline on detections results")

    parser.add_argument('--detections_path', type=str, default=None,
                        help='csv containing detections')

    args = parser.parse_args()

    convert_to_json(args.detections_path)
