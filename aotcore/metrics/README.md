__Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.__
This package provides the evaluation code for PrimeAir airborne detection challenge 
The driving script is ```run_airborne_metrics.py``` 
```
usage: run_airborne_metrics.py [-h] --dataset-folder DATASET_FOLDER
                               --results-folder RESULTS_FOLDER
                               [--summaries-folder SUMMARIES_FOLDER]
                               [--min-score MIN_SCORE]
                               [--min-track-len MIN_TRACK_LEN]
                               [--log-level LOG_LEVEL]
run_airborne_metrics.py: error: the following arguments are required: --dataset-folder/-d, --results-folder/-r
```
The metrics gets a folder with the dataset ground truth file or files (```--dataset-folder```) 
and a folder with results (```--results-folder```)
It evaluates all the results present in the result folder and saves the json files with evaluation 
details into the summary folder (```--summaries-folder ``` if provided, otherwise ```summaries``` folder
is created in ```--results-folder```)  
Additional options is to perform filtering based on minimum detection score or/ and minimum track length
by providing ```--min-score``` or/ and ```--min-track-len```accordingly 

Before running the examples below you will need to place results in resulst folder (named results_example below)
and groundtruth.csv (preferred over .json) in ground truth folder (named validation_gt) 

For example:
```
cd challenge_metrics 
python3.6 -m pip install .  # need to install once unless you make changes to the code
python3.6 run_airborne_metrics.py -d ./validation_gt  -r ./results_example -s ./summaries 
```
will evaluate all the detections
OR
```
python3.6 run_airborne_metrics.py -d ./validation_gt  -r ./results_example -s ./summaries --min-track-len 10
```
will use evaluate only detections that correspond to tracks with track_len of 10 and above (in on-line fashion)

Detection results json file should contain detection records per image (img_name)  with the following fields:
'img_name' - img_name as appears in the ground truth file
'detections' - List[Dict]: with the following fields:
    'n' - name of the class (typically airborne)
    'x' - x coordinate  of the center of the bounding box
    'y' - y of the center of the bounding box
    'w' - width 
    'h' - height
    's' - score
    'track_id' / 'object_id' - optional track or object id associated with the detection 
Please see sample detection results file in ```results_example```

The results will be found in:
1) the results folder in a sub-folder named as results with 'metrics' suffix appended 
2) the summaries folder you provided as input 
