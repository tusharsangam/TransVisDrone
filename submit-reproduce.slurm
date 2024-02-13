#!/bin/bash
#SBATCH -o ./slurmlogs/reproduce_%A_%a_out.txt ## %A %a
#SBATCH --job-name=Visualization
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 10

echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

nvidia-smi

source activate pytorch-ampere
conda activate pytorch-ampere
which python
python val.py --data ./data/NPS.yaml \
--weights ./runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt \
--batch-size 1 --img 1280 --num-frames 5 \
--project ./runs/val/NPS/reproduce --name best_augment_full_save \
--task test --exist-ok --save-json  --augment --save-json-gt

