# This script is to convert the NPS annotations into the VisDrone2021 style
# NPS style 0,1,1778,499,1792,511: frame no, num_obj, bbox: X1Y1X2Y2, Multiple predictions in a line
# VisDrone style 684,8,273,116,0,0,0,0: Each prediction in different lines, XYHW, 1,1,0,0 https://github.com/VisDrone/VisDrone2018-DET-toolkit
from curses.ascii import isdigit
import glob, os, random, cv2, shutil
from tqdm import tqdm

def parse_annotation_txt_to_dict(text_file_path):
    annotations = open(text_file_path, "r").readlines()
    annot_dict = {}
    for line in annotations:
        lineparts = line.split(",")
        frame_number, n_annot  = lineparts[:2]
        lineparts = lineparts[2:]
        annot_dict[int(frame_number)] = []
        for i in range(0, len(lineparts), 4):
            annotation = []
            for j in range(i, i+4):
                annotation.append(int(lineparts[j]))
            annot_dict[int(frame_number)].append(annotation)
        assert len(annot_dict[int(frame_number)]) == int(n_annot)
    return annot_dict

def get_dimension(video_src):
    video_name = os.path.basename(video_src).split(".")[0]
    in_video_cap = cv2.VideoCapture(video_src)
    num_frames = int(in_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(in_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(in_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (video_name, (num_frames, height, width))


def rename_frames_(frames_root):
    frame_files = glob.glob(os.path.join(frames_root, "*"))
    for frame_file in tqdm(frame_files, total=len(frame_files), desc="Renaming please wait..."):
        extention = os.path.basename(frame_file).split(".")[1].strip()
        clip_id, frame_id = os.path.basename(frame_file).split(".")[0].split("_")[1:]
        clip_id = int(clip_id.strip())
        frame_id = int(frame_id.strip())
        new_path =  os.path.join(frames_root, f"Clip_{str(clip_id)}_{str(frame_id).zfill(5)}.{extention}")
        os.rename(frame_file, new_path)

import pickle
def generate_half_video_split():
    splits = ['train', 'val']
    videos_list = glob.glob('/home/c3-0/tu666280/FL-Drones-Dataset/dronevideos/*')
    dest_dir = '/home/c3-0/tu666280/FL-Drones-Dataset/yolo-tph-data-half-split'
    annotation_dir = '/home/c3-0/tu666280/FL-Drones-Dataset/yololabelsall'
    frames_dir = '/home/c3-0/tu666280/FL-Drones-Dataset/frames' #zero index frames
    train_video_lens, val_video_lens = {}, {}
    for video_path in tqdm(videos_list):

        for split in splits:
            dest_dir_images = os.path.join(dest_dir, f"{split}", "frames")
            dest_dir_annotations = os.path.join(dest_dir, f"{split}", "labels")
            dest_dir_videos = os.path.join(dest_dir, f"{split}", "videos")
            os.makedirs(dest_dir_annotations, exist_ok=True)
            os.makedirs(dest_dir_images, exist_ok=True)
            #os.makedirs(os.path.join(dest_dir, f"{split}", "labels"), exist_ok=True)
            os.makedirs(dest_dir_videos, exist_ok=True)
            
            video_name = os.path.basename(video_path).split(".")[0]
            video_id = int("".join([x for x in video_name if isdigit(x)]))
            in_video_cap = cv2.VideoCapture(video_path)
            num_frames = int(in_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            in_video_cap.release()
            
            every_fourth = list(range(0, num_frames, 4))
            mid = len(every_fourth)//2
            train_start_frame, val_start_frame = 0, num_frames//2 
            
            if split == "train":
                iterator = range(train_start_frame, val_start_frame)
                train_video_lens[video_id] = len(iterator)
            elif split == "val":
                iterator = range(val_start_frame, num_frames) #every_fourth[mid:]
                val_video_lens[video_id] = len(iterator)
            
            for fiddest, fid in enumerate(iterator):
                dest_image_path = os.path.join(dest_dir_images, f"{video_name}_{str(fiddest).zfill(5)}.png")
                src_image_path = os.path.join(frames_dir, f"{video_name}_{str(fid).zfill(5)}.png")
                assert os.path.exists(src_image_path), f"Fatal error {src_image_path} doesn't exsists"
                os.symlink(src_image_path, dest_image_path)
                src_label_path = os.path.join(annotation_dir, f"{video_name}_{str(fid).zfill(5)}.txt")
                dest_label_path = os.path.join(dest_dir_annotations,  f"{video_name}_{str(fiddest).zfill(5)}.txt")
                if os.path.exists(src_label_path):
                    os.symlink(src_label_path, dest_label_path)
    train_video_len_path = os.path.join(dest_dir, "train", "videos", "video_length_dict.pkl")
    val_video_len_path = os.path.join(dest_dir, "val", "videos", "video_length_dict.pkl")
    pickle.dump(train_video_lens, open(train_video_len_path, "wb"))
    pickle.dump(val_video_lens, open(val_video_len_path, "wb"))
            
            


#split = 'val'
def generate_visdrone_annotations():
    splits = ['train', 'val']

    videos_list = glob.glob('/home/c3-0/tu666280/FL-Drones-Dataset/dronevideos/*') 
    #random.shuffle(videos_list)

    #train_ids = [ 11, 29, 37, 49, 53, 55, 56 ]
    #val_ids = [1, 12, 18, 19, 46, 47, 48]
    train_ids = [56, 53, 47, 46, 19, 12, 11, 1]
    val_ids = [55, 49, 48, 37, 29, 18]
    
    train_videos_list = [video_path for video_path in videos_list if int(os.path.basename(video_path).split(".")[0].split("_")[-1].strip()) in train_ids]
    val_videos_list = [video_path for video_path in videos_list if int(os.path.basename(video_path).split(".")[0].split("_")[-1].strip()) in val_ids]

    dest_dir = '/home/c3-0/tu666280/FL-Drones-Dataset/yolo-tph-data-rohit-split'
    annotation_dir = '/home/c3-0/tu666280/FL-Drones-Dataset/yololabelsall'
    frames_dir = '/home/c3-0/tu666280/FL-Drones-Dataset/frames' #zero index frames
    for split in splits:
        dest_dir_images = os.path.join(dest_dir, f"{split}", "frames")
        dest_dir_annotations = os.path.join(dest_dir, f"{split}", "labels")
        dest_dir_videos = os.path.join(dest_dir, f"{split}", "videos")
        os.makedirs(dest_dir_annotations, exist_ok=True)
        os.makedirs(dest_dir_images, exist_ok=True)
        #os.makedirs(os.path.join(dest_dir, f"{split}", "labels"), exist_ok=True)
        os.makedirs(dest_dir_videos, exist_ok=True)
        videos_list = train_videos_list if split == "train" else val_videos_list
        for video_path in videos_list:
            video_name = os.path.basename(video_path).split(".")[0]
            #annotation_file_path = os.path.join(annotation_dir, f"{video_name}.txt")
            #annotation_dict = parse_annotation_txt_to_dict(annotation_file_path)
            in_video_cap = cv2.VideoCapture(video_path)
            num_frames = int(in_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            in_video_cap.release()
            dest_video_path = os.path.join(dest_dir_videos, os.path.basename(video_path))
            os.symlink(video_path, dest_video_path)
            for fid in tqdm(range(num_frames), desc=f"Copying Video frames for.. {video_name}", total=num_frames):
                #_, frame = in_video_cap.read()
                #out_image_file_name = f"{dest_dir_images}/{video_name}_{str(fid).zfill(5)}.png"
                #cv2.imwrite(out_image_file_name, frame)
                dest_image_path = os.path.join(dest_dir_images, f"{video_name}_{str(fid).zfill(5)}.png")
                src_image_path = os.path.join(frames_dir, f"{video_name}_{str(fid).zfill(5)}.png")
                assert os.path.exists(src_image_path), print(f"Fatal error {src_image_path} doesn't exsists")
                os.symlink(src_image_path, dest_image_path)
                label_path = os.path.join(annotation_dir, f"{video_name}_{str(fid).zfill(5)}.txt")
                if os.path.exists(label_path):
                    os.symlink(label_path, os.path.join(dest_dir_annotations, os.path.basename(label_path) ) )
                # if fid in annotation_dict:
                #     out_annotation_file_name = f"{dest_dir_annotations}/{video_name}_{str(fid).zfill(5)}.txt"
                #     annotations = annotation_dict[fid]
                #     annotation_str = ""
                #     for annotation in annotations:
                #         x1, y1, x2, y2 = annotation
                #         annotation_str += f"{x1},{y1},{x2-x1},{y2-y1},1,1,0,0\n"
                #     file = open(out_annotation_file_name, "w")
                #     file.write(annotation_str)
                #     file.close()

def renaming_operation(fl_drones_yolo_data_root):#change fl-drones naming to NPS naming
    splits = ["train", "val"]
    for split in splits:
        frames_dir = os.path.join(fl_drones_yolo_data_root, f"{split}", "frames")
        annotations_dir = os.path.join(fl_drones_yolo_data_root, f"{split}", "labels")
        frames_paths = glob.glob(os.path.join(frames_dir, "*"))
        #labels_paths = glob.glob(annotation_dir+"/*")
        for frame_path in tqdm(frames_paths, desc=f"Renaming for {split}..", total=len(frames_paths)):
            file_name_without_extention = os.path.basename(frame_path).split(".")[0]
            if file_name_without_extention.find("Clip") > -1:
                continue
            
            video_id, frame_id = file_name_without_extention.split("_")[1:]
            video_id = int(video_id.strip())
            new_name_wo_extention = f"Clip_{str(video_id)}_{frame_id.strip()}"
            new_frame_path = os.path.join(frames_dir, f"{new_name_wo_extention}.png")
            os.rename(frame_path, new_frame_path)
            
            label_path = os.path.join(annotations_dir, f"{file_name_without_extention}.txt")
            if os.path.exists(label_path):
                new_label_path = os.path.join(annotations_dir, f"{new_name_wo_extention}.txt")
                os.rename(label_path, new_label_path)
def assert_file_name_problems(root_dir):
    for split in ["train", "val"]:
        frames_dir = os.path.join(root_dir, f"{split}", "frames")
        label_dir = os.path.join(root_dir, f"{split}", "label")
        frames_paths = glob.glob(os.path.join(frames_dir, "*"))
        labels_paths = glob.glob(os.path.join(label_dir, "*"))
        for frame_path in tqdm(frames_paths, desc=f"Checking frames for {split}", total=len(frames_paths)):
            assert frame_path.find("Video") == -1, print(f"Problem with {frame_path}")
        for label_path in tqdm(labels_paths, desc=f"Checking labels for {split}", total=len(labels_paths)):
            assert label_path.find("Video") == -1, print(f"Problem with {label_path}")


def copy_current_annotations_to_annotations_mains():
    '''
    copy yolo labels split in different folders train & val into one folder then symlink from there
    '''
    splits = ["train", "val"]
    annotations_root_dir = "/home/c3-0/tu666280/FL-Drones-Dataset/yololabelsall"
    video_root_dir = "/home/c3-0/tu666280/FL-Drones-Dataset/dronevideos"
    for split in splits:
        current_annotations_dir = f"/home/c3-0/tu666280/FL-Drones-Dataset/yolo-tph-data/{split}/labels"
        current_video_dir = f"/home/c3-0/tu666280/FL-Drones-Dataset/dronevideos/{split}"
        list_of_labels = glob.glob(os.path.join(current_annotations_dir, "*"))
        list_of_videos = glob.glob(os.path.join(current_video_dir, "*"))
        for label_path in tqdm(list_of_labels, total=len(list_of_labels), desc=f"Copying labels to root...for {split}"):
            dest_path = os.path.join(annotations_root_dir, os.path.basename(label_path))
            shutil.copy(label_path, dest_path)
        for video_path in tqdm(list_of_videos, total=len(list_of_videos), desc=f"Copying videos to root...for {split}"):
            dest_path = os.path.join(video_root_dir, os.path.basename(video_path))
            shutil.copy(video_path, dest_path)
        
if __name__ == "__main__":
    #renaming_operation("/home/tu666280/FL-Drones-Dataset/yolo-tph-data")
    #assert_file_name_problems("/home/tu666280/FL-Drones-Dataset/yolo-tph-data")
    #rename_frames_("/home/tu666280/FL-Drones-Dataset/frames")
    #generate_visdrone_annotations()
    generate_half_video_split()
            
    