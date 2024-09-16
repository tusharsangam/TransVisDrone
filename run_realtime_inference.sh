python realtimepredict.py --data ./data/NPS.yaml \
--weights ./runs/train/NPS/image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0/weights/best.pt \
--batch-size 1 --img 1280 --num-frames 5 \
--project ./runs/realtimeNPS --name realtimetest1 \
--task test --exist-ok