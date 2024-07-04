import json


def cap():
    preds_path = 'output/records/smoter/caption_results.json'
    lbl_path = 'datasets/bensmot/instance_caption.json'
    with open(preds_path, 'r') as f:
        preds_data = json.load(f)
    with open(lbl_path, 'r') as f:
        lbl_data = json.load(f)
    preds_list, gts_list = preds_data['preds'], preds_data['gts']
    assert len(preds_list) == len(gts_list)

    # text = 'a woman in a white shirt holding a folding chair, looking at a woman in a white floral dress and listens to the woman, then puts the woman\'s hands on her lap.'
    # idx = preds_list.index(text)
    # gt_text = gts_list[idx]
    gt_text = 'A man in the dark blue shirt with pimples on his face raises his arms to his sides as instructed by the man with the ID in front of him and patiently submits to being examined by the man with the ID.'
    flag = False
    for seq_name, seq_data in lbl_data.items():
        texts = []
        for obj_id, obj_text in seq_data.items():
            texts.append(obj_text)
        if gt_text in texts:
            flag = True
            break
    print(texts)
    if flag:
        print(seq_name)
        for t in texts:
            assert t in gts_list
            print(preds_list[gts_list.index(t)])

def video_cap(seq_name):
    preds_path = 'output/records/smoter/summary_results.json'
    lbl_path = 'datasets/bensmot/video_summary.json'
    with open(preds_path, 'r') as f:
        preds_data = json.load(f)
    preds_list, gts_list = preds_data['preds'], preds_data['gts']
    with open(lbl_path, 'r') as f:
        lbl_data = json.load(f)
    gt_text = lbl_data[seq_name]
    preds_text = preds_list[gts_list.index(gt_text)]
    print(gt_text)
    print(preds_text)

def get_relation(seq_name):
    lbl_path = 'datasets/bensmot/relation.json'
    with open(lbl_path, 'r') as f:
        lbl_data = json.load(f)
    print(lbl_data[seq_name])


if __name__ == '__main__':
    # cap()
    video_cap('security_checking/5YzxbjqhiFU_0')
    get_relation('security_checking/5YzxbjqhiFU_0')
