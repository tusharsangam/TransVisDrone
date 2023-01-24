#!/bin/bash
#SBATCH -o ./slurmlogs/train_aot_%A_out.txt ## %A %a
#SBATCH --job-name=TrainYolo3D
#SBATCH -p gpu --gres=gpu:ampere:2
##SBATCH --qos preempt
#SBATCH --ntasks=1
#SBATCH -c 10 --mem-per-cpu=10240M
##SBATCH --gres-flags=enforce-binding
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user tusharsangam5@gmail.com

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

nvidia-smi

# Load the TensorFlow module
module load anaconda3/2020.07



source activate /home/tu666280/.conda/envs/pytorch-ampere

which python

#export PYTHONPATH=$PYTHONPATH:/home/tu666280/VisTR

echo "Environment Activated"


python train.py --img 1280 --adam --batch 1 --epochs 80 --data ./data/NPS.yaml \
--weights ./pretrained/yolo5l.pt --hy ./data/hyps/hyp.VisDrone.yaml --cfg ./models/yolov5l.yaml \
--project ./runs/train/NPS --name visualization_5_frames --exist-ok


# python train.py --img 1280 --adam --batch 4 --epochs 80 --data ./data/AOT.yaml --weights ./pretrained/yolo5l.pt \
# --hy ./data/hyps/hyp.VisDrone_1.yaml --cfg ./models/yolov5l-xl.yaml --project ./runs/train/AOT \
# --name image_size_1280_YOLOXL_1_frames_SwinTR_AOT_with_yolo_weights --exist-ok


# python -m torch.distributed.launch --nproc_per_node=2 --master_port 47769 \
# --use_env train.py --img 1280 --adam --batch 4 --epochs 80 --data ./data/NPS.yaml \
# --hy ./data/hyps/hyp.VisDrone_3.yaml --cfg ./models/yolov5l.yaml \
# --name image_size_1280_temporal_YOLO5l_3_frames_NPS_end_to_end --project ./runs/train/NPS \
# --weights ./pretrained/yolov5l.pt --exist-ok


# python -m torch.distributed.launch --nproc_per_node=2 --master_port 47769 \
# --use_env train.py --img 1280 --adam --batch 4 --epochs 80 --data ./data/NPS.yaml \
# --hy ./data/hyps/hyp.VisDrone.yaml --cfg ./models/yolov5l.yaml \
# --name image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0 --project ./runs/train/NPS \
# --weights ./pretrained/yolo5l.pt --exist-ok

python -m torch.distributed.launch --nproc_per_node=1 --master_port 47769 \
--use_env train.py --img 640 --adam --batch 4 --epochs 80 --data ./data/NPS.yaml \
--hy ./data/hyps/hyp.VisDrone_3.yaml --cfg ./models/yolov5s.yaml \
--name image_size_640_temporal_YOLO5s_3_frames_NPS_end_to_end_skip_0 --project ./runs/train/NPS \
--weights ./pretrained/yolov5s.pt --exist-ok

# python -m torch.distributed.launch --nproc_per_node=1 --master_port 47770 \
# --use_env train.py --img 640 --adam --batch 4 --epochs 80 --data ./data/NPS.yaml \
# --hy ./data/hyps/hyp.VisDrone.yaml --cfg ./models/yolov5l.yaml \
# --resume ./runs/train/NPS/image_size_640_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0_window_3/weights/last.pt --exist-ok

#  python -m torch.distributed.launch --nproc_per_node=2 --master_port 47770 \
# --use_env train.py --img 1280 --adam --batch 4 --epochs 80 --data ./data/NPS.yaml \
# --hy ./data/hyps/hyp.VisDrone.yaml --cfg ./models/yolov5l.yaml \
# --name image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0_window_3 --project ./runs/train/NPS \
# --weights ./pretrained/yolov5l.pt --exist-ok

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 47769 \
--use_env train.py --img 640 --adam --batch 4 --epochs 80 --data ./data/BRIAR.yaml \
--hy ./data/hyps/hyp.VisDrone_3.yaml --cfg ./models/yolov5l.yaml \
--name image_size_640_temporal_YOLO5l_3_frames_BRIAR_end_to_end_skip_0 --project ./runs/train/BRIAR \
--weights ./pretrained/yolov5l.pt --exist-ok --workers 2

CUDA_VISIBLE_DEVICES=1 nohup python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 47769 \
--use_env train.py --img 640 --adam --batch 4 --epochs 80 --data ./data/BRIAR.yaml \
--hy ./data/hyps/hyp.VisDrone_3.yaml --cfg ./models/yolov5l.yaml \
--name image_size_640_temporal_YOLO5l_3_frames_BRIAR_end_to_end_skip_0 --project ./runs/train/BRIAR \
--weights ./pretrained/yolov5l.pt --exist-ok --workers 2 > output.log &