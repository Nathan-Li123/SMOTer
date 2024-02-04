import os
import json
import tqdm


DATA_PATH = '/data3/InsCap/imgs'
OUT_PATH = 'datasets/cvid/annotations/'
SPLITS = ['train', 'val']
ADD_TEXT = True

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    if ADD_TEXT:
        with open('datasets/cvid/video_summary.json', 'r') as f:
            summary_dict = json.load(f)
        with open('datasets/cvid/instance_caption.json', 'r') as f:
            caption_dict = json.load(f)
        with open('datasets/cvid/relation.json', 'r') as f:
            relation_dict = json.load(f)
    seqmap = open('datasets/cvid/seqmaps/trainval.txt', 'w')
    seqmap.write('name\n')
    for split in SPLITS:
        out = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'person'}], 'videos': []}
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        cate_dirs = os.listdir(DATA_PATH)
        for cate_dir in tqdm.tqdm(cate_dirs, desc=split + ' set'):
            cate_dir_path = os.path.join(DATA_PATH, cate_dir)
            seq_dirs = os.listdir(cate_dir_path)
            for seq_dir in seq_dirs:
                seq_dir_path = os.path.join(cate_dir_path, seq_dir)
                seq_name = cate_dir + '/' + seq_dir
                if ADD_TEXT:
                    if seq_name not in summary_dict or seq_name not in caption_dict or seq_name not in relation_dict:
                        continue
                    video_summary = summary_dict[seq_name]
                    inst_caption = caption_dict[seq_name]
                    inter_relation = relation_dict[seq_name]
                video_cnt += 1
                if split == 'train' and video_cnt % 6 == 0:
                    continue
                if split == 'val' and video_cnt % 6 != 0:
                    continue
                if ADD_TEXT:
                    out['videos'].append({'id': video_cnt, 'file_name': seq_name,\
                                          'summary': video_summary, 'caption': inst_caption, 'relation': inter_relation})
                else:
                    out['videos'].append({'id': video_cnt, 'file_name': seq_name})
                if split == 'val':
                    seqmap.write(seq_name + '\n')
                
                frame_files = os.listdir(seq_dir_path)
                jpg_files, json_files = [], []
                for frame_file in frame_files:
                    if frame_file[0] == '.':
                        continue
                    if frame_file.split('.')[-1] == 'json':
                        json_files.append(frame_file)
                    elif frame_file.split('.')[-1] == 'jpg':
                        jpg_files.append(frame_file)
                    else:
                        raise ValueError('Unkonw file type:', frame_file)
                jpg_files.sort()
                fisrt_json_path = os.path.join(seq_dir_path, json_files[0])
                with open(fisrt_json_path, 'r') as f:
                    data = json.load(f)
                img_h, img_w = data['imageHeight'], data['imageWidth']
                for i in range(len(jpg_files)):
                    jpg_file = jpg_files[i]
                    img_name = int(jpg_file.split('.')[0])
                    image_cnt += 1
                    file_path = os.path.join(seq_dir_path, jpg_file.replace('jpg', 'json'))
                    img_path = seq_name + '/' + jpg_file
                    image_info = {
                        'file_name': img_path,
                        'id': image_cnt,
                        'frame_id': i + 1,
                        'prev_image_id': image_cnt - 1 if i > 0 else -1,
                        'next_image_id': image_cnt + 1 if i != len(jpg_files) - 1 else -1,
                        'video_id': video_cnt,
                        'height': img_h,
                        'width': img_w
                    }
                    out['images'].append(image_info)
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            tgts = json.load(f)['shapes']
                        for tgt in tgts:
                            track_id = int(tgt['label'])
                            bbox = [tgt['points'][0][0], tgt['points'][0][1],\
                                    tgt['points'][1][0], tgt['points'][1][1]]
                            bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                            if bbox[2] <= 0 or bbox[3] <= 0:
                                continue
                            ann_cnt += 1
                            ann_info = {
                                'id': ann_cnt,
                                'category_id': 1,
                                'image_id': image_cnt,
                                'instance_id': track_id,
                                'bbox': bbox,
                                'conf': 1.0,
                                'iscrowd': 0,
                                'area': bbox[2] * bbox[3]
                            }
                            out['annotations'].append(ann_info)

        if ADD_TEXT:
            out_path = os.path.join(OUT_PATH, split + '_vu.json')
        else:
            out_path = os.path.join(OUT_PATH, split + '.json')
        with open(out_path, 'w') as f:
            json.dump(out, f)
    seqmap.close()
