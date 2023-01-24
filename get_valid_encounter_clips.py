import pickle, os
from aotcore.dataset import Dataset as AOTDataset
dataset_path = "/home/c3-0/datasets/Amazon-AOT/data/part1"

Dataset = AOTDataset(local_path=dataset_path, 
                    s3_path='s3://airborne-obj-detection-challenge-training/part1/', 
                    download_if_required=False, 
                    partial=True
                    )
valid_encounter_flight_ids = set(Dataset.valid_encounter.keys())
clip_id_to_flight_id_path = "/home/tu666280/tph-yolov5/aot_flight_ids/aot_clip_id_to_flight_id.pkl"
assert os.path.exists(clip_id_to_flight_id_path)
aot_clip_id_to_flight_id_dict = pickle.load(open(clip_id_to_flight_id_path, "rb"))

clip_ids, flight_ids = list(aot_clip_id_to_flight_id_dict.keys()), list(aot_clip_id_to_flight_id_dict.values())
valid_clip_id = {clip_ids[i] for i in range(len(clip_ids)) if flight_ids[i] in valid_encounter_flight_ids}

pickle.dump(valid_clip_id, open("valid_clip_ids.pkl", "wb"))