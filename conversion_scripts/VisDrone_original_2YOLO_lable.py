import os
import pandas as pd
from PIL import Image

split= "val"
YOLO_LABELS_PATH = f"/home/tu666280/NPSvisdroneStyle/{split}/labels_original"
VISANN_PATH = f"/home/tu666280/NPSvisdroneStyle/{split}/annotations_original"
VISIMG_PATH = f"/home/tu666280/NPS-Data-Uncompressed/AllFrames/{split}"

def convert(bbox, img_size):
    #将标注visDrone数据集标注转为yolov5
    #bbox top_left_x top_left_y width height
    dw = 1/(img_size[0])
    dh = 1/(img_size[1])
    x = bbox[0] + bbox[2]/2
    y = bbox[1] + bbox[3]/2
    x = x * dw
    y = y * dh
    w = bbox[2] * dw
    h = bbox[3] * dh
    return (x,y,w,h) 

def get_image_name(annotation_path, image_dir):
    clip_id, frameid = os.path.basename(annotation_path).split("_")[1:3]
    clip_id, frameid = int(clip_id.strip()), int(frameid.split(".")[0].strip())
    return os.path.join(image_dir, f"Clip_{str(clip_id)}_{str(frameid+1).zfill(5)}.png")

def ChangeToYolo5():
    if not os.path.exists(YOLO_LABELS_PATH):
        os.makedirs(YOLO_LABELS_PATH)
    print(len(os.listdir(VISANN_PATH)))
    for file in os.listdir(VISANN_PATH):
        image_path =  get_image_name(file, VISIMG_PATH) #VISIMG_PATH + '/' + file.replace('txt', 'png')
        ann_file = os.path.join(VISANN_PATH, file)
        out_file = open( os.path.join(YOLO_LABELS_PATH, file), 'w')
        bbox = pd.read_csv(ann_file,header=None).values
        img = Image.open(image_path)
        img_size = img.size
        for row in bbox:
            if(row[4]==1 and 0<row[5]<11):   
                label = convert(row[:4], img_size)
                out_file.write(str(row[5]-1) + " " + " ".join(str(f'{x:.6f}') for x in label) + '\n')
        out_file.close()

if __name__ == '__main__':
    ChangeToYolo5()