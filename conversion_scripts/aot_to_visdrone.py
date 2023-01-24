from aotcore.dataset import Dataset as AOTDataset
import os.path as osp
import random, pickle, math, os, json, yaml
from glob import glob
from tqdm.auto import tqdm
from p_tqdm import p_umap, p_map

dataset_root = "/home/c3-0/datasets/Amazon-AOT/data/part1"
dataset = AOTDataset(local_path=dataset_root, 
                    s3_path='s3://airborne-obj-detection-challenge-training/part1/', 
                    download_if_required=False, 
                    partial=False
                    )
def check_whether_object_passes_criterion(object_location, distance_threshold:int=700):
    if object_location.planned:
        if object_location.range_distance_m is not None:
            if not math.isnan(object_location.range_distance_m):
                if int(object_location.range_distance_m) <= distance_threshold :
                    return True
    return False

classes_that_matter = ["Airplane", "Helicopter", "Drone"]

def get_class_str(class_name:str):
    new_class_name = [c for c in class_name if not c.isdigit()]
    return "".join(new_class_name)

def decide_class_index(class_name:str):
    class_name_without_digits = get_class_str(class_name)
    return 0 if class_name_without_digits in classes_that_matter else 1

def convert_to_cxcywhn(object_location, height=2048., width=2448.):
    class_index = decide_class_index(object_location.object.id)
    x1, y1, x2, y2 = object_location.bb.get_bbox_traditional()
    cx, cy, w, h = (x1+x2)/2., (y1+y2)/2., (x2-x1), (y2-y1)
    cxn, cyn, wn, hn = cx/width, cy/height, w/width, h/height
    return (cxn, cyn, wn, hn), class_index

def convert_annotation_to_yolo(frame, clip_id, frameid, labels_dir, distance_range=700):
    object_locations = list(frame.detected_object_locations.values())
    object_locations = [object_location for object_location in object_locations if check_whether_object_passes_criterion(object_location, distance_range)] 
    object_locations = [convert_to_cxcywhn(object_location) for object_location in object_locations]
    if len(object_locations) > 0:
        yolo_label_path = osp.join(labels_dir, f"Clip_{str(clip_id)}_{str(frameid).zfill(5)}.txt")
        with open(yolo_label_path, "w") as file:
            for object_location in object_locations:
                bbox, class_index = object_location
                file.write(str(class_index) + " " + " ".join(str(f'{x:.6f}') for x in bbox) + '\n')
            file.close()
        
def extract_flight_labels(arg):
    try:
        flight_id, clip_id, frames_dir, labels_dir = arg
        flight = dataset.get_flight(flight_id)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        frame_ids = list(flight.frames.keys())
        for fid, frame_id in enumerate(frame_ids):
            frame = flight.frames[frame_id]
            frame_path = osp.join(dataset_root, frame.image_path())
            yolo_frame_path = osp.join(frames_dir, f"Clip_{str(clip_id)}_{str(fid).zfill(5)}.png")
            assert osp.exists(frame_path), print(f"Fatal error {frame_path}, doesn't exists")
            os.symlink(frame_path, yolo_frame_path)
            if len(frame.detected_object_locations) > 0:
                convert_annotation_to_yolo(frame, clip_id, fid, labels_dir)
    except Exception as e:
         return True, f"exception {e}, args {arg}"
    return False, ""
def create_yolo_dataset_from_aot():
    aot_flight_id_paths = "./aot_flight_ids"
    aot_yolo_root_dir = "/home/tu666280/aot_part_1_yolo_data"
    os.makedirs(aot_yolo_root_dir, exist_ok=True)
   
    train_ids = json.load(open(osp.join(aot_flight_id_paths, "trainflightidsfull1.json"), "r" ))
    val_ids = json.load(open(osp.join(aot_flight_id_paths, "valflightidsfull1.json"), "r" ))
    test_ids = json.load(open(osp.join(aot_flight_id_paths, "testflightidsfull1.json"), "r" ))
    all_ids = train_ids + val_ids + test_ids
   
    flight_id_to_clip_id_dict = {flightid:clip_id for clip_id, flightid in enumerate(all_ids)}
    pickle.dump(flight_id_to_clip_id_dict, open(osp.join(aot_flight_id_paths, "aot_flight_id_to_clip_id.pkl"), "wb"))
    print("Flight_id to clip_id dictonary saved")
    
    clip_id_to_flight_id_lookup = {clip_id:flight_id for flight_id, clip_id in flight_id_to_clip_id_dict.items()}
    pickle.dump(clip_id_to_flight_id_lookup, open(osp.join(aot_flight_id_paths, "aot_clip_id_to_flight_id.pkl"), "wb"))
    print("Clip_id to flight_id dictonary saved")

    #split into train, val, test in a different directory with os symlinking
    train_root = osp.join(aot_yolo_root_dir, "train")
    train_frames_dir = osp.join(train_root, "frames")
    train_labels_dir = osp.join(train_root, "labels")

    val_root = osp.join(aot_yolo_root_dir, "val")
    
    val_full_root = osp.join(val_root, "full")
    val_full_frames_dir = osp.join(val_full_root, "frames")
    val_full_labels_dir = osp.join(val_full_root, "labels")
    
    val_partial_root = osp.join(val_root, "partial")
    val_partial_frames_dir = osp.join(val_partial_root, "frames")
    val_partial_labels_dir = osp.join(val_partial_root, "labels")
    
    test_root = osp.join(aot_yolo_root_dir, "test")
    
    test_full_root = osp.join(test_root, "full")
    test_full_frames_dir = osp.join(test_full_root, "frames")
    test_full_labels_dir = osp.join(test_full_root, "labels")
    
    
    #symlink as per train, val, test split
    random.shuffle(val_ids)
    val_partial_ids = val_ids[:10]
    random.shuffle(test_ids)
    

    print("saving partial ids")
    json.dump(val_partial_ids, open(osp.join(aot_flight_id_paths, "valflightidspartial1.json"), "w"))
    #json.dump(test_partial_ids, open(osp.join(aot_flight_id_paths, "testflightidspartial1.json"), "w"))

    print("Creating train split...")
    args = [(flight_id, flight_id_to_clip_id_dict[flight_id], train_frames_dir, train_labels_dir) for flight_id in train_ids]
    results = p_umap(extract_flight_labels, args, **{"num_cpus": 10})
    for result in results:
        if result[0]:   
            print(f"{result[1]}")
    
    
    print("Creating val full split...")
    args = [(flight_id, flight_id_to_clip_id_dict[flight_id], val_full_frames_dir, val_full_labels_dir) for flight_id in val_ids]
    results = p_umap(extract_flight_labels, args, **{"num_cpus": 10})
    for result in results:
        if result[0]:   
            print(f"{result[1]}")
    
   
    print("Creating val partial split...")
    args = [(flight_id, flight_id_to_clip_id_dict[flight_id], val_partial_frames_dir, val_partial_labels_dir) for flight_id in val_partial_ids]
    results = p_umap(extract_flight_labels, args, **{"num_cpus": 10})
    for result in results:
        if result[0]:   
            print(f"{result[1]}")
   
    print("Creating test full split...")
    args = [(flight_id, flight_id_to_clip_id_dict[flight_id], test_full_frames_dir, test_full_labels_dir) for flight_id in test_ids]
    results = p_umap(extract_flight_labels, args, **{"num_cpus": 10})
    for result in results:
        if result[0]:   
            print(f"{result[1]}")
    
    


def extract_video_length(arg):
    clip_id, flight_id, frames_dir = arg
    clip_id = int(clip_id)
    flight = dataset.get_flight(flight_id)
    frames_len = len(list(flight.frames.keys()))
    # all_files_exists = [osp.exists(osp.join(dataset_root, frame.image_path())) for frame in flight.values()]
    # assert all(all_files_exists)
    return clip_id, frames_len

def create_video_length_dict():
    aot_yolo_root_dir = "/home/tu666280/aot_part_1_yolo_data"
    aot_flight_id_paths = "./aot_flight_ids"
    train_ids = json.load(open(osp.join(aot_flight_id_paths, "trainflightidsfull1.json"), "r" ))
    val_ids = json.load(open(osp.join(aot_flight_id_paths, "valflightidsfull1.json"), "r" ))
    test_ids = json.load(open(osp.join(aot_flight_id_paths, "testflightidsfull1.json"), "r" ))
    val_partial_ids = json.load(open(osp.join(aot_flight_id_paths, "valflightidspartial1.json"), "r" ))
    
    flight_id_to_clip_id_dict = pickle.load(open(osp.join(aot_flight_id_paths, "aot_flight_id_to_clip_id.pkl"), "rb"))

    #video_frame_dir_tuple_list = [("train/videos", "train/frames"), ("val/full/videos", "val/full/frames"), ("val/partial/videos", "val/partial/frames"), ("test/full/videos", "test/full/frames"), ("test/partial/videos", "test/partial/frames")]

    args = [ (flight_id_to_clip_id_dict[flight_id], flight_id, osp.join(aot_yolo_root_dir, "train/frames") ) for flight_id in train_ids]
    print("Extracting train video lengths...")
    results = p_map(extract_video_length, args, **{"num_cpus": 10})
    video_len_dict = {res[0]:res[1] for res in results}
    video_dir = osp.join(aot_yolo_root_dir, "train/videos")
    os.makedirs(video_dir, exist_ok=True)
    pickle.dump(video_len_dict, open(osp.join(video_dir, "video_length_dict.pkl"), "wb"))

    print(list(video_len_dict.items())[:10])

    args = [ (flight_id_to_clip_id_dict[flight_id], flight_id, osp.join(aot_yolo_root_dir, "val/full/frames") ) for flight_id in val_ids]
    print("Extracting val full video lengths...")
    results = p_map(extract_video_length, args, **{"num_cpus": 10})
    video_len_dict = {res[0]:res[1] for res in results}
    video_dir = osp.join(aot_yolo_root_dir, "val/full/videos")
    os.makedirs(video_dir, exist_ok=True)
    pickle.dump(video_len_dict, open(osp.join(video_dir, "video_length_dict.pkl"), "wb"))

    print(list(video_len_dict.items())[:10])

    args = [ (flight_id_to_clip_id_dict[flight_id], flight_id, osp.join(aot_yolo_root_dir, "val/partial/frames") ) for flight_id in val_partial_ids]
    print("Extracting val partial video lengths...")
    results = p_map(extract_video_length, args, **{"num_cpus": 10})
    video_len_dict = {res[0]:res[1] for res in results}
    video_dir = osp.join(aot_yolo_root_dir, "val/partial/videos")
    os.makedirs(video_dir, exist_ok=True)
    pickle.dump(video_len_dict, open(osp.join(video_dir, "video_length_dict.pkl"), "wb"))

    print(list(video_len_dict.items())[:10])

    args = [ (flight_id_to_clip_id_dict[flight_id], flight_id,  osp.join(aot_yolo_root_dir, "test/full/frames") ) for flight_id in test_ids]
    print("Extracting test full video lengths...")
    results = p_map(extract_video_length, args, **{"num_cpus": 10})
    video_len_dict = {res[0]:res[1] for res in results}
    video_dir = osp.join(aot_yolo_root_dir, "test/full/videos")
    os.makedirs(video_dir, exist_ok=True)
    pickle.dump(video_len_dict, open(osp.join(video_dir, "video_length_dict.pkl"), "wb"))

    print(list(video_len_dict.items())[:10])

    

   
import math

def generate_split(arg):
    clip_id_to_length_dict = {}
    try:
        flight_ids, clip_ids, labels_root, dest_frame_root, dest_labels_root = arg
        for flight_id, clip_id in zip(flight_ids, clip_ids):
            flight = dataset.get_flight(flight_id)
            frame_ids = list(flight.frames.keys())
            clip_id_to_length_dict[clip_id] = len(frame_ids)
            for fid, frame_id in enumerate(frame_ids):
                frame = flight.frames[frame_id]
                frame_path = osp.join(dataset_root, frame.image_path())
                yolo_frame_path = osp.join(dest_frame_root, f"Clip_{str(clip_id)}_{str(fid).zfill(5)}.png")
                assert osp.exists(frame_path), print(f"Fatal error {frame_path}, doesn't exists")
                os.symlink(frame_path, yolo_frame_path)
                yolo_label_path = osp.join(labels_root, f"Clip_{str(clip_id)}_{str(fid).zfill(5)}.txt")
                if osp.exists(yolo_label_path):
                    new_yolo_label_path = osp.join(dest_labels_root, f"Clip_{str(clip_id)}_{str(fid).zfill(5)}.txt")
                    os.symlink(yolo_label_path, new_yolo_label_path)
    except Exception as e:
        print(f"exception {e}, args {arg}")
    return clip_id_to_length_dict
def create_test_splits(size_of_each_split=10):
    aot_yolo_root_dir = "/home/tu666280/aot_part_1_yolo_data"
    #frames_root = osp.join(aot_yolo_root_dir, "test/full/frames")
    labels_root = osp.join(aot_yolo_root_dir, "test/full/labels")
    aot_flight_id_paths = "./aot_flight_ids"
    test_ids = json.load(open(osp.join(aot_flight_id_paths, "testflightidsfull1.json"), "r" ))
    nth_split = math.ceil(len(test_ids)/float(size_of_each_split))
    data_yaml_root = "./data/AOTTestSplits"
    flight_id_to_clip_id_path = "./aot_flight_ids/aot_flight_id_to_clip_id.pkl"
    flight_id_to_clip_id_dict = pickle.load(open(flight_id_to_clip_id_path, "rb"))
    os.makedirs(data_yaml_root, exist_ok=True)
    for split_id in tqdm(range(nth_split), total=nth_split, desc="Generating test split ..."):
        start = size_of_each_split*split_id
        end = min((1+split_id)*size_of_each_split, len(test_ids))
        flight_ids = test_ids[start:end]
        dest_frame_root = osp.join(aot_yolo_root_dir, f"test/part{split_id}/frames")
        dest_labels_root = osp.join(aot_yolo_root_dir, f"test/part{split_id}/labels")
        dest_videos_root = osp.join(aot_yolo_root_dir, f"test/part{split_id}/videos")
        os.makedirs(dest_frame_root, exist_ok=True)
        os.makedirs(dest_labels_root, exist_ok=True)
        os.makedirs(dest_videos_root, exist_ok=True)
        clip_ids = [flight_id_to_clip_id_dict[flight_id] for flight_id in flight_ids]
        clip_id_to_length_dict = generate_split((flight_ids, clip_ids, labels_root, dest_frame_root, dest_labels_root))
        pickle.dump(clip_id_to_length_dict, open(osp.join(dest_videos_root, "video_length_dict.pkl"), "wb"))
        yaml_dict = {
                    "path":"/home/tu666280/aot_part_1_yolo_data", 
                    "train": "train/frames", 
                    "val": "val/full/frames",
                    "test": f"test/part{split_id}/frames",
                    "annotation_path": "/home/tu666280/aot_part_1_yolo_data",
                    "annotation_train": "train/labels",
                    "annotation_val": "val/full/labels",
                    "annotation_test": f"test/part{split_id}/labels",
                    "video_root_path": "/home/tu666280/aot_part_1_yolo_data",
                    "video_root_path_train": "train/videos",
                    "video_root_path_val": "val/full/videos",
                    "video_root_path_test": f"test/part{split_id}/videos",
                    "nc": 2,
                    "names": ["drone", "airborne"]
                    }
        data_yaml_file_path = osp.join(data_yaml_root, f"AOTTest_{split_id}.yaml")
        yaml.dump(yaml_dict, open(data_yaml_file_path, "w"))


if __name__ == "__main__":
    #create_yolo_dataset_from_aot()
    #create_video_length_dict()
    create_yolo_dataset_from_aot()
    create_video_length_dict()
    create_test_splits()