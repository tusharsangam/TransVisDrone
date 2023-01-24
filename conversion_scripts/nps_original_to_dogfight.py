import os 
import numpy as np
from glob import glob
from tqdm import tqdm

def permute_annotations(annotations):
    for i in range(len(annotations)):
        y1, x1, y2, x2 = annotations[i]
        annotations[i] = [x1, y1, x2, y2]
    return annotations

def read_annotations(file_path):
    lines = open(file_path, "r").readlines()
    lines = [line.split("\n")[0] for line in lines]
    annotations_dict = {}
    
    for line in lines:
        _, frameid, annotations = line.split(":")
        frameid = int(frameid.replace("detections", "").strip())
        annotations = annotations.split(",")[:-1]
        if len(annotations) > 0:
            assert len(annotations) % 4 == 0, print(f"len of annotations expected to be multiple of 4, got {len(annotations)}, annotations {annotations}")
            annotations = np.array([int(annot.replace("(", "").replace(")", "").strip()) for annot in annotations]).reshape(-1, 4)
            annotations = permute_annotations(annotations) #convert from yxyx to xyxy
            annotations_dict[frameid-1] = annotations
    assert len(annotations_dict) > 0
    clip_id = int(os.path.basename(file_path).split("_")[1].strip())
    return clip_id, annotations_dict


def convert_original_to_dogfight_format(original_annotations_root_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    list_of_original_annotation_file_paths = glob(os.path.join(original_annotations_root_dir, "*"))
    for original_annotation_file_path in tqdm(list_of_original_annotation_file_paths, total=len(list_of_original_annotation_file_paths), desc="Converting to dogfight format.."):
        clip_id, annotations_dict = read_annotations(original_annotation_file_path)
        out_file = os.path.join(dest_dir, f"Clip_{str(clip_id).zfill(3)}.txt")
        file_content = ""
        for frameid, annotations in annotations_dict.items():
            file_content += f"{frameid},{len(annotations)},"
            for i, ai in enumerate(annotations.flatten()):
                file_content += f"{ai}," if i != annotations.flatten().shape[0] - 1 else f"{ai}"
            file_content += "\n"
        with open(out_file, "w") as f:
            f.write(file_content)

def main():
    original_annotations_root_dir = "/home/tu666280/tph-yolov5/nps_annotations"
    dest_dir = "/home/tu666280/NPS/annotations/NPS-Drones-Dataset-Original"
    convert_original_to_dogfight_format(original_annotations_root_dir, dest_dir)

if __name__ == "__main__":
    main()

        
            

            

