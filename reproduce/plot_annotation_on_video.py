import os
from glob import glob
from os import path

import cv2 as cv
from tqdm import tqdm
import numpy as np

# folder containing videos
videos_folder = '/data/home/sangamtushar/DroneDataset/Videos'
# file extension of the videos
video_file_extension = '.mov'
# folder containing
annotations_folder = '/data/home/sangamtushar/DroneDataset/NPS-Drones-Dataset'

# if save_frame = 4, this means every 4th frame is saved
# if save_frame = 1, every frame will be saved
save_frame = 1
# value which will be placed in mask where an object exists
# use 1 for training data and 255 for testing data


def parse_yolo_annotations(file_path, vwidth, vheight):
    bboxes = []
    with open(file_path, "r") as file:
        data_list = [ln.strip().replace('\n', '') for ln in file.readlines()]
        for data in data_list:
            _, ncx, ncy, nw, nh = map(float, data.split(" "))
            cx, w = ncx*vwidth, nw*vwidth
            cy, h = ncy*vheight, nh*vheight
            x1, x2 = cx - (w/2), cx + (w/2)
            y1, y2 = cy - (h/2), cy + (h/2)
            bboxes.append([x1, y1, x2, y2])
    return bboxes

def parse_annotations_data(data_list):
    annotations = {}
    for d in data_list:
        d = d.replace(' ', '').split(',')
        frame_number = int(d[0])
        bb_coords = []
        bbs = [int(i) for i in d[2:]]
        for i in range(0, len(bbs), 4):
            bb_coords.append(bbs[i:i + 4])

        annotations[frame_number] = bb_coords
    return annotations


def get_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        data_list = [ln.strip().replace('\n', '') for ln in f.readlines()]
    annotations = parse_annotations_data(data_list)
    return annotations

def main():
    
    #os.makedirs(masks_output_folder, exist_ok=True)

    video_paths = glob(path.join(videos_folder, f'*{video_file_extension}'))

    for video_path in tqdm(video_paths, 'Processing videos'):
        filename = path.basename(video_path)
        filename_wo_ext, file_ext = path.splitext(filename)
        clip_id = int(filename_wo_ext.split("_")[-1])
        annotations_path = path.join(annotations_folder, f'Clip_{clip_id:03}.txt')
        annotations = get_annotations(annotations_path)

        video_cap = cv.VideoCapture(video_path)
        total_frames = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))

        frame_number = 0
        pbar = tqdm(desc='Processing frames', total=total_frames)
        
        width  = video_cap.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = video_cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`
        fps = video_cap.get(cv.CAP_PROP_FPS)
        
        writecap = cv.VideoWriter("output.mp4", cv.VideoWriter_fourcc(*'MPEG'), int(fps), (int(width), int(height))) 
        
        while video_cap.isOpened():
            ret, frame = video_cap.read()

            if not ret:
                break

            if True:#frame_number % save_frame == 0:
                
                
                
                if frame_number in annotations.keys():
                    bounding_boxes = annotations[frame_number]
                    for bb in bounding_boxes:
                        x1, y1, x2, y2 = bb
                        frame =  cv.rectangle(frame, (int(x1),int(y1)), (int(x2), int(y2)), (255, 0, 0))
                
                writecap.write(frame)
            frame_number += 1
            pbar.update()
        pbar.close()
        video_cap.release()
        writecap.release()
        break


def plot_yolo_labels_with_video():
    import random
    video_paths = glob(path.join(videos_folder, f'*{video_file_extension}'))
    random.shuffle(video_paths)
    for video_path in tqdm(video_paths, 'Processing videos'):
        filename = path.basename(video_path)
        filename_wo_ext, file_ext = path.splitext(filename)
        clip_id = int(filename_wo_ext.split("_")[-1])
        split = "train"
        if clip_id >= 37 and clip_id < 41:
            split = "val"
        elif clip_id >= 41:
            split = "test"
        annotation_folder = f"/data/home/sangamtushar/DroneDataset/{split}/annotations"
        video_frames_folder = f"/data/home/sangamtushar/DroneDataset/{split}/images"
        video_cap = cv.VideoCapture(video_path)
        num_frames = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
        width  = video_cap.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = video_cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float `height`
        fps = video_cap.get(cv.CAP_PROP_FPS)
        writecap = cv.VideoWriter("output_yolo.avi", cv.VideoWriter_fourcc(*'MPEG'), int(fps), (int(width), int(height))) 
        video_cap.release()
        for frame_id in tqdm(range(num_frames), total=num_frames):
            frame_path = os.path.join(video_frames_folder, f"Clip_{clip_id}_{frame_id:05}.png")
            annotation_path = os.path.join(annotation_folder, f"Clip_{clip_id}_{frame_id:05}.txt")
            frame = cv.imread(frame_path)
            if os.path.exists(annotation_path):
                bboxes = parse_yolo_annotations(annotation_path, float(width), float(height))
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    frame =  cv.rectangle(frame, (int(x1),int(y1)), (int(x2), int(y2)), (255, 0, 0))
            writecap.write(frame)
        writecap.release()
        break
if __name__ == '__main__':
    #main()
    plot_yolo_labels_with_video()