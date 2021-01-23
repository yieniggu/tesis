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
    output = os.path.dirname(os.path.dirname(os.path.dirname((detections_path))))
    
    if output[-1] != '/':
        output += '/'

    # get filename
    last_path_of_detections_path = os.path.basename(os.path.normpath(detections_path))
    splitted_detections_path = last_path_of_detections_path.split("_")

    model_name = "_".join(splitted_detections_path[0:2])
    precision = splitted_detections_path[2]
    threshold = splitted_detections_path[3]
    frame_dims = splitted_detections_path[6]
    loading_backend = splitted_detections_path[7].split(".")[0]

    output += "predictions-txt/{}-{}-{}-{}-{}/".format(model_name, precision, threshold,
                                                        frame_dims, loading_backend)

    create_dir(output)   
                   
    return output

def detections_to_txt(detections_path):
    detections = pd.read_csv(detections_path)

    output_path = get_output_path(detections_path)

    filenames = detections["filename"].unique()

    for filename in filenames:
        file_detections = detections.loc[detections["filename"] == filename]

        if "N/D" in file_detections.values:
            continue

        filename_only = filename.split('.')[0]
        filepath = "{}{}.txt".format(output_path, filename_only)
        
        with open(filepath, "a") as output_file:
            for index, row in file_detections.iterrows():
                detection_class = row.detection_class
                score = float(row.detection_score)
                xmin = int(row.xmin)
                xmax = int(row.xmax)
                ymin = int(row.ymin)
                ymax = int(row.ymax)

                content = "{} {} {} {} {} {}".format(detection_class, score,
                                                    xmin, ymin, xmax, ymax)

                if index != file_detections.size-1:
                    content += '\n'
                        
                output_file.write(content)

def gt_to_txt(gt_anns_path):
    annotations = pd.read_csv(gt_anns_path)

    annotations.rename(columns={"class": "detection_class"}, inplace=True)

    output_path = os.path.dirname(gt_anns_path)
    if output_path[-1] != '/':
        output_path += '/'

    output_path+= 'annotations-txt/'

    create_dir(output_path)

    filenames = annotations["filename"].unique()

    for filename in filenames:
        file_annotations = annotations.loc[annotations["filename"] == filename]

        filename_only = filename.split('.')[0]
        filepath = "{}{}.txt".format(output_path, filename_only)
        
        with open(filepath, "a") as output_file:
            for index, row in file_annotations.iterrows():
                detection_class = row.detection_class
                xmin = int(row.xmin)
                xmax = int(row.xmax)
                ymin = int(row.ymin)
                ymax = int(row.ymax)

                content = "{} {} {} {} {}".format(detection_class, xmin, ymin,
                                                    xmax, ymax)

                if index != file_annotations.size-1:
                    content += '\n'
                        
                output_file.write(content)


def convert_to_txt(detections_path, gt_anns_path, convert_gt=True, convert_det=True):
    # convert detections to txt
    if convert_det:
        detections_to_txt(detections_path)

    #convert gt annotations to txt
    if convert_gt:
        gt_to_txt(gt_anns_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Convert annotations to txt special format")

    parser.add_argument('--detections_path', type=str, 
                        help='csv containing detections')

    parser.add_argument('--gt_anns_path', type=str, 
                        help='csv containing ground truth annotations')

    parser.add_argument('--convert_gt', action='store_true',
                         help='Select to convert gt ann paths')

    parser.add_argument('--convert_det', action='store_true',
                         help='Select to convert det paths')

    args = parser.parse_args()

    convert_to_txt(args.detections_path, args.gt_anns_path, args.convert_gt, args.convert_det)