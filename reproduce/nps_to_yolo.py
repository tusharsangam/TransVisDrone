# This script is to convert the NPS annotations into the VisDrone2021 style
# NPS style 0,1,1778,499,1792,511: frame no, num_obj, bbox: X1Y1X2Y2, Multiple predictions in a line
# VisDrone style 684,8,273,116,0,0,0,0: Each prediction in different lines, XYHW, 1,1,0,0 https://github.com/VisDrone/VisDrone2018-DET-toolkit
import glob, os, cv2, pickle
split = 'test'
splits = ['test', 'val', 'train']
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

for split in splits:
    if split == 'train':
        valid_clips = range(37)
    elif split == 'val':
        valid_clips = range(37,41)
    elif split == 'test':
        valid_clips = range(41,51)

    video_folder = "/data/home/sangamtushar/DroneDataset/Videos"
    anno_files = glob.glob('/data/home/sangamtushar/DroneDataset/NPS-Drones-Dataset/*') #Clip_001.txt
    frame_directory = '/data/home/sangamtushar/DroneDataset/AllFrames' #Clip_1_000000.png
    output_image_folder = '/data/home/sangamtushar/DroneDataset/' + split + '/images'
    output_annos_folder = '/data/home/sangamtushar/DroneDataset/' + split + '/annotations'
    output_videos_folder = '/data/home/sangamtushar/DroneDataset/' + split + '/videos'
    
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_annos_folder):
        os.makedirs(output_annos_folder)
    if not os.path.exists(output_videos_folder):
        os.makedirs(output_videos_folder)    
    #Annotations only (no 1:1 for annotations & frames)
    for anno_file in anno_files:
        clip_id = int(os.path.basename(anno_file).split('.')[0].split('_')[1])
        if clip_id not in valid_clips: continue
        video_path = os.path.join(video_folder, f"Clip_{clip_id}.mov")
        videocap = cv2.VideoCapture(video_path)
        width  = int(videocap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height = int(videocap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
        num_frames = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
        videocap.release()
        anno_content = open(anno_file, 'r').read().splitlines()
        for row in anno_content:
            frame_id, num_obj = int(row.split(',')[0]), int(row.split(',')[1])
            output_file_name = '/Clip_' + str(clip_id) + '_' + f"{frame_id:05}"
            if not os.path.exists(output_annos_folder + output_file_name +'.txt'):
                temp = open(output_annos_folder + output_file_name +'.txt', 'w')
                for obj in range(num_obj):
                    x1, y1, w, h = int(row.split(',')[obj*4 + 2]), int(row.split(',')[obj*4 + 3]), int(row.split(',')[obj*4 + 4]) - int(row.split(',')[obj*4 + 2]), int(row.split(',')[obj*4 + 5]) - int(row.split(',')[obj*4 + 3])
                    label = convert([x1, y1, w, h], (width, height))
                    content = str(0) + " " + " ".join(str(f'{x:.6f}') for x in label) + '\n'
                    #content = str(row.split(',')[obj*4 + 2]) + ',' +str(row.split(',')[obj*4 + 3]) + ','+ str(int(row.split(',')[obj*4 + 4]) - int(row.split(',')[obj*4 + 2])) + ','+ str(int(row.split(',')[obj*4 + 5]) - int(row.split(',')[obj*4 + 3])) + ',' + '1,1,0,0 \n'
                    temp.writelines(content)
                temp.close()
        
        print(f'{os.path.basename(anno_file)} done')
    
    #Video & frames
    video_length = {}
    video_pkl_path = os.path.join(output_videos_folder, "video_length_dict.pkl")
    for clip_id in valid_clips:
        if clip_id == 0: continue
        video_path = os.path.join(video_folder, f"Clip_{clip_id}.mov")
        videocap = cv2.VideoCapture(video_path)
        num_frames = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
        videocap.release()
        for frame_id in range(num_frames):
            corresponding_frame =  os.path.join(frame_directory, 'Clip_' + str(clip_id) + '_' + f"{frame_id:05}.png")
            output_file_name = '/Clip_' + str(clip_id) + '_' + f"{frame_id:05}"
            os.symlink(corresponding_frame, output_image_folder + output_file_name +'.png')

        video_length[clip_id] = num_frames
    with open(video_pkl_path, "wb") as f:
        pickle.dump(video_length, f)
    
    print(f'{split} folders done')


    
        # exit()
# /home/c3-0/tu666280/NPS-Data-Uncompressed/AllFrames/train/Clip_30_00000.png
# /home/c3-0/tu666280/NPS-Data-Uncompressed/AllFrames/val/Clip_39_00000.png
# /home/c3-0/tu666280/NPS/annotations/NPS-Drones-Dataset/Clip_039.txt
