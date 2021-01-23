import pandas as pd
import argparse
import re
import glob, os
import image_utils as imgutils
import xml.etree.ElementTree as ET
import numpy as np

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

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

def get_output_path(detections_path, image_paths):
    # get filename
    last_path_of_detections_path = os.path.basename(os.path.normpath(detections_path))
    splitted_detections_path = last_path_of_detections_path.split("_")

    model_name = "_".join(splitted_detections_path[0:2])
    precision = splitted_detections_path[2]
    threshold = splitted_detections_path[3]
    frame_dims = splitted_detections_path[6]
    loading_backend = splitted_detections_path[7].split(".")[0]
    
    if image_paths[-1] != '/':
        image_paths+= '/'
    
    output = image_paths + "results/detections-annos/" 
    output += "{}-{}-{}-{}-{}/".format(model_name, precision, 
                                    threshold, frame_dims, loading_backend)

    create_dir(output)                  
    return output

def get_filename_only(filename):
    return filename.split('.')[0]

def get_detections_csv(detections_path):
    return pd.read_csv(detections_path)

def convert_to_xml(detections_path, images_path):  
    # get output path to save files into
    output_path = get_output_path(detections_path, images_path)

    # get detections dataframe
    detections = get_detections_csv(detections_path)

    # get image paths
    image_paths = sorted(glob.glob(images_path + "*.png"), key=numericalSort)

    # iterate over all images
    for image_path in image_paths:
        splitted_path = image_path.split('/')
        filename = splitted_path[-1]
        folder = splitted_path[-2]

        image = imgutils.read_image_from_cv2(image_path)

        # get the shape of the image to compare to
        # image dimensions from detections
        height, width, depth = image.shape

        print("[CONVERT-TO-XML] Processing file {} from folder {}".format(filename, folder))
        image_detections = detections.loc[detections["filename"] == filename]

        if "N/D" in image_detections.values:
            continue

        # set file annotations
        root = ET.Element('annotation')
        folder_anno = ET.SubElement(root, "folder")
        folder_anno.text = folder
        filename_anno = ET.SubElement(root, "filename")
        filename_anno.text = filename
        path_anno = ET.SubElement(root, "path")
        path_anno.text = image_path

        # set source annotations
        source_anno = ET.SubElement(root, "source")
        db_anno = ET.SubElement(source_anno, "database")
        db_anno.text = "Unknown"

        # set size annotations
        size_anno = ET.SubElement(root, "size")
        width_anno = ET.SubElement(size_anno, "width")
        width_anno.text = str(width)
        height_anno = ET.SubElement(size_anno, 'height')
        height_anno.text = str(height)
        depth_anno = ET.SubElement(size_anno, "depth")
        depth_anno.text = str(depth)
         
        # set segmented
        segmented_anno = ET.SubElement(root, "segmented")
        segmented_anno.text = '0'

        for _, row in image_detections.iterrows():
            detection_class = row.detection_class
            det_width, det_height = row.width, row.height

            # format bbox to resize
            bbox = np.array([row.ymin, row.xmin, 
                            row.ymax, row.xmax]).astype(int)

            # check for resize bounding box
            if (width != det_width) | (height != det_height):
                bbox = imgutils.resize_bounding_box((height, width, 3), 
                                                    (det_height, det_width, 3),
                                                    bbox)

            ymin = bbox[0]
            xmin = bbox[1]
            ymax = bbox[2]
            xmax = bbox[3]

            object_anno = ET.SubElement(root, "object")

            name_anno = ET.SubElement(object_anno, "name")
            name_anno.text = detection_class
            pose_anno = ET.SubElement(object_anno, "pose")
            pose_anno.text = "Unspecified"
            truncated_anno = ET.SubElement(object_anno, "truncated")
            truncated_anno.text = '0'
            difficult_anno = ET.SubElement(object_anno, "difficult")
            difficult_anno.text = '0'

            bbox_anno = ET.SubElement(object_anno, "bndbox")
            xmin_anno = ET.SubElement(bbox_anno, "xmin")
            xmin_anno.text = str(xmin)
            ymin_anno = ET.SubElement(bbox_anno, "ymin")
            ymin_anno.text = str(ymin)
            xmax_anno = ET.SubElement(bbox_anno, "xmax")
            xmax_anno.text = str(xmax)
            ymax_anno = ET.SubElement(bbox_anno, "ymax")
            ymax_anno.text = str(ymax)

        tree = ET.ElementTree(root)
            
        filename_only = get_filename_only(filename) + ".xml"
        print("[CSV-TO-XML] Saving file to {}{}".format(output_path, filename_only))
        tree.write(output_path + filename_only)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform tracking offline on detections results")

    parser.add_argument('--detections_path', type=str, default=None,
                        help='File containing detections per frame summary')

    parser.add_argument('--images_path', type=str, default='dogs.jpeg',
                        help='Path to frames to perform tracking')

    args = parser.parse_args()

    convert_to_xml(args.detections_path, args.images_path)