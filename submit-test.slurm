#!/bin/bash
#SBATCH -o ./slurmlogs/visual_%A_%a_out.txt ## %A %a
#SBATCH --job-name=Visualization
##SBATCH --time=05:00:00
##SBATCH -p gpu
##SBATCH -C gmem24
##SBATCH --qos preempt
##SBATCH --gres=gpu:1
#parallel details
#SBATCH --ntasks=1
#SBATCH -c 6
#SBATCH --mem-per-cpu=6144M
##SBATCH --gres-flags=enforce-binding
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user tusharsangam5@gmail.com

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

nvidia-smi

# Load the TensorFlow module
module load anaconda3/2020.07



source activate pytorch-ampere

which python

#export PYTHONPATH=$PYTHONPATH:/home/tu666280/VisTR

echo "Environment Activated"

cd /home/tu666280/tph-yolov5/visualization_scripts

python video_plotter.py 1
python video_plotter.py 2
python video_plotter.py 3
python video_plotter.py 4
python video_plotter.py 5 
python video_plotter.py 6
python video_plotter.py 7
python video_plotter.py 8
python video_plotter.py 9
python video_plotter.py 10
python video_plotter.py 11
python video_plotter.py 12
python video_plotter.py 13
python video_plotter.py 14
python video_plotter.py 15
python video_plotter.py 16
python video_plotter.py 17

python inference.py --data ./data/NPS.yaml \
--weights ./runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt \
--batch-size 2 --img 1280 --num-frames 5 \
--project ./runs/inference/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0 --name best_augment_full_save \
--task inference --exist-ok --save-aot-predictions

# python val.py --data ./data/NPS.yaml \
# --weights ./runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt \
# --batch-size 1 --img 1280 --num-frames 5 \
# --project ./runs/val/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0 --name best_augment_full_save \
# --task test --exist-ok --save-json  --augment --save-json-gt

# python val.py --data ./data/FLDrone.yaml \
# --weights ./runs/train/FL/image_size_1280_temporal_YOLO5L_5_frames_FL_end_to_end/weights/best.pt \
# --batch-size 1 --img 1280 --num-frames 5 \
# --project ./runs/val/FL/image_size_1280_temporal_YOLO5L_5_frames_FL_end_to_end --name best_augment_full_save \
# --task val --exist-ok --save-json  --augment --save-json-gt


# python speedtest.py --data ./data/NPS.yaml \
# --weights ./runs/train/NPS/image_size_640_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_3_fixed/weights/best.pt \
# --batch-size 2 --img 640 --num-frames 5 \
# --project ./runs/val/NPS/image_size_640_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_3_fixed --name speedtest \
# --task test --exist-ok --save-json-gt

# python val.py --data ./data/AOTTestSplits/AOTTest_$SLURM_ARRAY_TASK_ID.yaml \
# --weights ./runs/train/AOT/image_size_1280_temporal_SwinTR_1_frames_AOT_with_yolo_weights/weights/best.pt \
# --batch-size 5 --img 1280 --num-frames 1 \
# --project ./runs/val/AOT/image_size_1280_temporal_SwinTR_1_frames_AOT_with_yolo_weights --name best \
# --task test --exist-ok --save-aot-predictions --save-json-gt


# python val.py --data ./data/AOTTestSplits/AOTTest_$SLURM_ARRAY_TASK_ID.yaml \
# --weights ./runs/train/AOT/image_size_1280_YOLOXL_1_frames_SwinTR_AOT_with_yolo_weights/weights/best.pt \
# --batch-size 5 --img 1280 --num-frames 1 \
# --project ./runs/val/AOT/image_size_1280_YOLOXL_1_frames_SwinTR_AOT_with_yolo_weights --name best \
# --task test --exist-ok --save-aot-predictions --save-json-gt


# python3 speedtest.py --data ./data/NPS.yaml \
# --weights ./weights/nps_r_640_t_3_s_0_s.pt \
# --batch-size 1 --img 640 --num-frames 3 \
# --project ./runs/val/NPS/nps_r_640_t_3_s_0_s.pt --name speedtest \
# --task test --exist-ok --save-json 

python val.py --data ./data/BRIAR.yaml \
--weights ./runs/train/BRIAR/image_size_640_temporal_YOLO5l_3_frames_BRIAR_end_to_end_skip_0/weights/last.pt \
--batch-size 6 --img 640 --num-frames 3 \
--project ./runs/val/BRIAR/image_size_640_temporal_YOLO5l_3_frames_BRIAR_end_to_end_skip_0 --name last \
--task test --exist-ok --save-aot-predictions --save-json-gt