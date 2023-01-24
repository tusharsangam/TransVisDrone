# This script is to convert the NPS annotations into the VisDrone2021 style
# NPS style 0,1,1778,499,1792,511: frame no, num_obj, bbox: X1Y1X2Y2, Multiple predictions in a line
# VisDrone style 684,8,273,116,0,0,0,0: Each prediction in different lines, XYHW, 1,1,0,0 https://github.com/VisDrone/VisDrone2018-DET-toolkit
import glob, os
split = 'val'
splits = ['train', 'val', 'test']
for split in splits:
    if split == 'train':
        valid_clips = range(37)
    elif split == 'val':
        valid_clips = range(37,41)
    elif split == 'test':
        valid_clips = range(41,51)


    anno_files = glob.glob('/home/c3-0/tu666280/NPS/annotations/NPS-Drones-Dataset/*') #Clip_001.txt
    frame_directory = '/home/c3-0/tu666280/NPS-Data-Uncompressed/AllFrames/' + split #Clip_1_000000.png
    output_image_folder = 'NPSvisdroneStyle/' + split + '/images'
    output_annos_folder = 'NPSvisdroneStyle/' + split + '/annotations'

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_annos_folder):
        os.makedirs(output_annos_folder)  

    for anno_file in anno_files:
        clip_id = int(os.path.basename(anno_file).split('.')[0].split('_')[1])
        if clip_id not in valid_clips:
            print('frames are not extracted yet for ', os.path.basename(anno_file))
            continue
        anno_content = open(anno_file, 'r').read().splitlines()
        for row in anno_content:
            frame_id, num_obj = int(row.split(',')[0]), int(row.split(',')[1])
            frame_id +=1 #To get the corresponding frame
            corresponding_frame =  os.path.join(frame_directory, 'Clip_' + str(clip_id) + '_' + f"{frame_id:05}.png")
            # print(anno_file)
            frame_id -=1 #To rename the corresponding frame

            # print(corresponding_frame)
            # exit()
            # print(f"anno file, {anno_file}")
            # print(f"frame_id, {frame_id}")
            # print(f"corresponding_frame, {corresponding_frame}")
            # exit()
            output_file_name = '/Clip_' + str(clip_id) + '_' + f"{frame_id:05}"
            # if not os.path.exists(output_image_folder + output_file_name +'.png'):
            os.symlink(corresponding_frame, output_image_folder + output_file_name +'.png')
            if not os.path.exists(output_annos_folder + output_file_name +'.txt'):
                temp = open(output_annos_folder + output_file_name +'.txt', 'w')
                for obj in range(num_obj):
                    content = str(row.split(',')[obj*4 + 2]) + ',' +str(row.split(',')[obj*4 + 3]) + ','+ str(int(row.split(',')[obj*4 + 4]) - int(row.split(',')[obj*4 + 2])) + ','+ str(int(row.split(',')[obj*4 + 5]) - int(row.split(',')[obj*4 + 3])) + ',' + '1,1,0,0 \n'
                    temp.writelines(content)
                temp.close()
        print(f'{os.path.basename(anno_file)} done')

    print(f'{split} done')
        # exit()
# /home/c3-0/tu666280/NPS-Data-Uncompressed/AllFrames/train/Clip_30_00000.png
# /home/c3-0/tu666280/NPS-Data-Uncompressed/AllFrames/val/Clip_39_00000.png
# /home/c3-0/tu666280/NPS/annotations/NPS-Drones-Dataset/Clip_039.txt
