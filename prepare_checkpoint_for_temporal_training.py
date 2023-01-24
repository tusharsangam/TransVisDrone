import argparse
from utils.general import strip_optimizer
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, default='yolov5s.pt', help='path of weights to convert')
    parser.add_argument('--converted_path', type=str, required=True, default='yolov5s.pt', help='path to save converted weights')
    opt = parser.parse_args()
    strip_optimizer(opt.input_path, opt.converted_path)

# python prepare_checkpoint_for_temporal_training.py \
# --input_path ./runs/train/FL/image_size_1280_temporal_SwinTR_1_frames_FL_with_yolo_weights_half_split/weights/last.pt \
# --converted_path ./pretrained/sodlast_FL.pt