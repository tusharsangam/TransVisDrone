#!/bin/bash

# How to Run: 
 #Step 1 : Code your streaming interface (Important)
 # Please modify utils/datasets.py ->  LoadClipsStream Class -> function sample_temporal_frames_from_stream & __len__
 # sample_temporal_frames_from_stream should return a clip of num of frames you passed as an argument, I'm currently loading frames from Clip_50.mov which is present 
 # in NPS dataset videos, Videos.zip can be downloaded from https://engineering.purdue.edu/~bouman/UAV_Dataset/
 # I'm setting __len__ as length of the video however for real usecase you can set it to sys.maxsize which is equivalent of int(np.inf)

# Step 2: Customize your visualization (Optional)
# Please look at realtimepredict.py -> function visualize_detections. 
# this function is currently taking imgs loaded from the loader & predictions from the model & plots them & saves at given save_dir 
# please note that images loaded from dataloader are resized thus the predictions from the model are also on resized image, thus to view predictions on
# original image size use scaled predictions -> line 382 - 383. 

# Step 3: Choose appropriate model checkpoint & parameters
# --data data/NPS.yaml possible options  ->  data/FLDrone.yaml , data/AOT.yaml however this is irrelevant argument for streaming usecase
# --num-rames & --img should match the model checkpoint you are choosing
# --batch-size should be 1 & --task should be test 
# --project & --name can be choosen anything you want

#activate pytorch-ampere
#To Run : sh run_realtime_inference.sh 

python realtimepredict.py --data ./data/NPS.yaml \
--weights ./runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt \
--batch-size 1 --img 1280 --num-frames 5 \
--project ./runs/realtimeNPS --name realtimetest1 \
--task test --exist-ok