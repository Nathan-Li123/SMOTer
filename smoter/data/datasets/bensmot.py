import os
import io
import logging
import contextlib

from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


logger = logging.getLogger(__name__)

def load_bensmot_json(json_file, image_root, dataset_name=None, map_inst_id=True, extra_annotation_keys=None):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))
    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    if map_inst_id:
        assert 'instance_id' in extra_annotation_keys
        instance_ids = set(
            x['instance_id'] for x in coco_api.dataset['annotations'] \
                if x['instance_id'] > 0)
        inst_id_map = {x: i + 1 for i, x in enumerate(sorted(instance_ids))}
        if len(instance_ids) > 0: 
            print('Maping instances len/ min/ max', \
              len(inst_id_map), min(inst_id_map.values()), max(inst_id_map.values()))
        inst_id_map[0] = 0
        inst_id_map[-1] = 0
        
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        # modified 
        video_id = img_dict.get('video_id', -1)
        record['video_id'] = video_id
        # finish modified
        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            if map_inst_id:
                obj['instance_id'] = inst_id_map[obj['instance_id']]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


def register_bensmot_instances(name, metadata, json_file, image_root):
    """
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    DatasetCatalog.register(name, lambda: load_bensmot_json(
        json_file, image_root, name, extra_annotation_keys=['instance_id'], map_inst_id=True
    ))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="bensmot", **metadata
    )

categories = [
    {'id': 1, 'name': 'person'},
]

def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


_PREDEFINED_SPLITS = {
    "bensmot_train": ("/data3/InsCap/imgs", 
        "bensmot/annotations/train_vu.json"),
    "bensmot_val": ("/data3/InsCap/imgs", 
        "bensmot/annotations/val_vu.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
    register_bensmot_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )