from os.path import join
import json, pickle, os, argparse, time, re
from glob import glob
from pathlib import Path
from tqdm import tqdm 
from aotcore.dataset import Dataset as AOTDataset      

dataset_path = "/home/c3-0/datasets/Amazon-AOT/data/part1"

clip_id_to_flight_id_path = "./aot_flight_ids/aot_clip_id_to_flight_id.pkl"
assert os.path.exists(clip_id_to_flight_id_path)
clip_id_to_flight_id_dict = pickle.load(open(clip_id_to_flight_id_path, "rb"))

dataset = AOTDataset(local_path=dataset_path, 
                    s3_path='s3://airborne-obj-detection-challenge-training/part1/', 
                    download_if_required=False, 
                    partial=False
                    )

def convert_clip_id_path_to_flight_id_path(clip_id_path):
    clip_id, frame_id = os.path.basename(clip_id_path).split(".")[0].split("_")[1:]
    clip_id, frame_id = int(clip_id), int(frame_id)
    flight_id = clip_id_to_flight_id_dict[clip_id]
    flight = dataset.get_flight(flight_id)
    framekey_flight = list(flight.frames.keys())[frame_id]
    flight_image_path = os.path.basename(flight.frames[framekey_flight].image_path())
    return flight_image_path, flight_id



def generate_partial_gt(flight_ids, dataset_path, destination_folder):
    gt = json.loads(open(join(dataset_path, "ImageSets/groundtruth.json")).read())
    for sample in list(gt['samples'].keys()):
        if sample not in flight_ids:
            del gt['samples'][sample]
    os.makedirs(destination_folder, exist_ok=True)
    with open((f"{destination_folder}/groundtruth.json"), 'w') as fp:
        json.dump(gt, fp)
    print("Ground truths written")

def result_parts_joiner(parent_folder_name, destination_folder_name):
    file_paths_list = glob(parent_folder_name+"/*")
    if len(file_paths_list) == 0:
        print(f"fatal error {parent_folder_name} contains no files")
        exit()
    combined_results = []
    flight_ids = set()
    for file_path in file_paths_list:
        combined_results += pickle.load(open(f"{file_path}", "rb"))
    for result in tqdm(combined_results, desc="Converting Clip_id paths to flight_id paths..", total=len(combined_results)):
        result["img_name"], flight_id = convert_clip_id_path_to_flight_id_path(result["img_name"])
        flight_ids.add(flight_id)
    os.makedirs(destination_folder_name, exist_ok=True)
    with open( f"{destination_folder_name}/result.json", 'w') as fp:
        json.dump(combined_results, fp)
    print("Results written")
    #del dataset, clip_id_to_flight_id_dict
    return list(flight_ids)

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def evaluate(result_parts_root_folder="./runs/val/exp", min_score=0.5, evaluation_folder_name = "evaluationforaot/exp"):
    from aotcore.metrics.run_airborne_metrics import rerun, add_flags
    evaluation_folder_name = increment_path(evaluation_folder_name)
    os.makedirs(evaluation_folder_name, exist_ok=True)
    evaluation_parser = argparse.ArgumentParser(description='Evaluate airborne detection results with airborne metrics')
    add_flags(evaluation_parser)
    args_eval = evaluation_parser.parse_args()
    args_eval.dataset_folder = f"{evaluation_folder_name}/gt"
    args_eval.results_folder = f"{evaluation_folder_name}/result"
    args_eval.summaries_folder = f"{evaluation_folder_name}/summaries"
    args_eval.min_score = min_score
    print("Compiling result parts into one")
    flight_ids = result_parts_joiner(result_parts_root_folder, args_eval.results_folder)
    
    print("Generating GT required for evaluation")
    #flight_ids = json.load(open("./aot_flight_ids/testflightidsfull1.json", "r"))
    generate_partial_gt(flight_ids, dataset_path, args_eval.dataset_folder)
    
    print("Running Evaluation")
    print(f"Arguments recieved for AOT eval script : {args_eval}")
    rerun(args_eval)


if __name__ == "__main__":
    evaluation_parser = argparse.ArgumentParser('AOT evaluation script')
    evaluation_parser.add_argument('--results_folder', '-r', required=False, type=str, default="./runs/val/AOTtest/epoch_17/aotpredictions",
                        help='Name of the folder where you stored parts of the predictions')
    evaluation_parser.add_argument('--evaluation_folder', '-e', required=False, type=str, default="./evaluationaot/epoch_17",
                        help='Destination folder to run AOT evaluations')
    evaluation_parser.add_argument('--detection_threshold', '-t', required=False, type=float, default=0.75,
                        help='Detection threshold for evlaution')
    args_eval = evaluation_parser.parse_args()
    print(f"Arguments recieved for my eval script : {args_eval}")
    evaluate(args_eval.results_folder, args_eval.detection_threshold, args_eval.evaluation_folder)