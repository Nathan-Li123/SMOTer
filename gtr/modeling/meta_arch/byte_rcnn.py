import cv2
from detectron2.layers.shape_spec import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
import torch
import random
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.structures import Boxes, pairwise_iou, Instances
from detectron2.structures import ImageList
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from gtr.modeling.roi_heads.association_head import FCHead
from gtr.modeling.roi_heads.grit_roi_heads import get_relation_labels
from .custom_rcnn import CustomRCNN
from ..roi_heads.custom_fast_rcnn import custom_fast_rcnn_inference
from ..tracker.byte_tracker import BYTETracker


@META_ARCH_REGISTRY.register()
class BYTERCNN(CustomRCNN):
    @configurable
    def __init__(self, **kwargs):
        """
        """
        self.cfg = kwargs.pop('cfg')
        self.test_len = kwargs.pop('test_len')
        self.overlap_thresh = kwargs.pop('overlap_thresh')
        self.min_track_len = kwargs.pop('min_track_len')
        self.max_center_dist = kwargs.pop('max_center_dist')
        self.decay_time = kwargs.pop('decay_time')
        self.asso_thresh = kwargs.pop('asso_thresh')
        self.with_iou = kwargs.pop('with_iou')
        self.local_track = kwargs.pop('local_track')
        self.local_no_iou = kwargs.pop('local_no_iou')
        self.local_iou_only = kwargs.pop('local_iou_only')
        self.not_mult_thresh = kwargs.pop('not_mult_thresh')
        super().__init__(**kwargs)

        self._init_asso_head(self.cfg, self.backbone.output_shape())


    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['cfg'] = cfg
        ret['test_len'] = cfg.INPUT.VIDEO.TEST_LEN
        ret['overlap_thresh'] = cfg.VIDEO_TEST.OVERLAP_THRESH     
        ret['asso_thresh'] = cfg.MODEL.ASSO_HEAD.ASSO_THRESH
        ret['min_track_len'] = cfg.VIDEO_TEST.MIN_TRACK_LEN
        ret['max_center_dist'] = cfg.VIDEO_TEST.MAX_CENTER_DIST
        ret['decay_time'] = cfg.VIDEO_TEST.DECAY_TIME
        ret['with_iou'] = cfg.VIDEO_TEST.WITH_IOU
        ret['local_track'] = cfg.VIDEO_TEST.LOCAL_TRACK
        ret['local_no_iou'] = cfg.VIDEO_TEST.LOCAL_NO_IOU
        ret['local_iou_only'] = cfg.VIDEO_TEST.LOCAL_IOU_ONLY
        ret['not_mult_thresh'] = cfg.VIDEO_TEST.NOT_MULT_THRESH
        return ret


    def _init_asso_head(self, cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES        
        asso_in_channels = input_shape[in_features[0]].channels
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.asso_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.asso_head = FCHead(
            input_shape=ShapeSpec(
                channels=asso_in_channels, 
                height=pooler_resolution, width=pooler_resolution),
            fc_dim=cfg.MODEL.ASSO_HEAD.FC_DIM,
            num_fc=cfg.MODEL.ASSO_HEAD.NUM_FC,
        )


    def forward(self, batched_inputs, iteration=None):
        """
        All batched images are from the same video
        During testing, the current implementation requires all frames 
            in a video are loaded.
        TODO (Xingyi): one-the-fly testing
        """
        if not self.training:
            return self.local_tracker_inference(batched_inputs)
        assert iteration is not None

        video_info = batched_inputs[-1]
        batched_inputs = batched_inputs[:-1]
        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
        num_images = len(images)
        if iteration >= 20000:
            min_num = 6
        else:
            min_num = 10
        if num_images > min_num:
            idx = random.randint(0, num_images - min_num)
            tmp_images = torch.stack(images.tensor.unbind(0)[idx:idx+min_num])
            tmp_images_sizes = images.image_sizes[idx:idx+min_num]
            tmp_images = ImageList(tmp_images, tmp_images_sizes)
            tmp_batched_inputs = batched_inputs[idx:idx+min_num]
            tmp_gt_instances = [x["instances"].to(self.device) for x in tmp_batched_inputs]

            features = self.backbone(tmp_images.tensor)
            _, proposal_losses = self.proposal_generator(tmp_images, features, tmp_gt_instances)
        else:
            features = self.backbone(images.tensor)
            # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            _, proposal_losses = self.proposal_generator(images, features, gt_instances)
        
        losses = {}
        losses.update(proposal_losses)
        
        # strat tracking
        if iteration >= 20000:
            pred_instances, video_features = self.inferene_tracks(batched_inputs)
            # end tracking
            pred_tracks = self._post_process_tracks(pred_instances, gt=False)
            gt_tracks = self._post_process_tracks(gt_instances, gt=True)
            gt_tracks = self._check_gt_tracks(gt_tracks, video_info['caption'])
            transfered_pred_tracks, gt_ids = self._transfer_track_ids(gt_tracks=gt_tracks, pred_tracks=pred_tracks)

            summary_losses = self.roi_heads({
                'feats': video_features, 'pred_tracks': transfered_pred_tracks, 'gt_ids': gt_ids, \
                'text': [video_info['summary']], 'mode': 'summary'
            })
            caption_losses = self.roi_heads({
                'pred_tracks': transfered_pred_tracks, 'gt_ids': gt_ids, \
                'texts': video_info['caption'], 'mode': 'caption'
            })
            relation_losses = self.roi_heads({
                'pred_tracks': transfered_pred_tracks, 'gt_ids': gt_ids, \
                'texts': video_info['relation'], 'mode': 'relation'
            })
            if summary_losses is not None:
                losses.update(summary_losses)
            if caption_losses is not None:
                losses.update(caption_losses)
            if relation_losses is not None:
                losses.update(relation_losses)
        return losses

    
    def _check_gt_tracks(self, gt_tracks, caption_dict):
        remove_list =[]
        for k in gt_tracks:
            if k not in caption_dict:
                remove_list.append(k)
        for k in remove_list:
            del gt_tracks[k]
        return gt_tracks


    def _post_process_tracks(self, instances, gt):
        tracks_dict = {}
        for i in range(len(instances)):
            frame_id = i + 1
            if not gt:
                feats = instances[i]['instances'].get_fields()['reid_features']
                boxes = instances[i]['instances'].get_fields()['pred_boxes'].tensor
                track_ids = instances[i]['instances'].get_fields()['track_ids']
            else:
                boxes = instances[i].get_fields()['gt_boxes'].tensor
                track_ids = instances[i].get_fields()['gt_instance_ids']
            assert boxes.shape[0] == track_ids.shape[0]
            for j in range(boxes.shape[0]):
                track_id = str(track_ids[j].item())
                if track_id not in tracks_dict:
                    tracks_dict[track_id] = {}
                tracks_dict[track_id][str(frame_id)] = {'bbox': boxes[j]}
                if not gt:
                    tracks_dict[track_id][str(frame_id)]['feat'] = feats[j]
        return tracks_dict


    def _remove_short_track(self, instances):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        num_insts_track = id_inds.sum(dim=1) # M
        remove_track_id = num_insts_track < self.min_track_len # M
        unique_ids[remove_track_id] = -1
        ids = unique_ids[torch.where(id_inds.permute(1, 0))[1]]
        ids = ids.split([len(x) for x in instances])
        for k in range(len(instances)):
            instances[k] = instances[k][ids[k] >= 0]
        return instances


    def _delay_cls(self, instances, video_id):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        M = len(unique_ids) # #existing tracks
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        # update scores
        cls_scores = torch.cat(
            [x.cls_scores for x in instances], dim=0) # N x (C + 1)
        traj_scores = torch.mm(id_inds, cls_scores) / \
            (id_inds.sum(dim=1)[:, None] + 1e-8) # M x (C + 1)
        _, traj_inds = torch.where(id_inds.permute(1, 0)) # N
        cls_scores = traj_scores[traj_inds] # N x (C + 1)

        n_t = [len(x) for x in instances]
        boxes = [x.pred_boxes.tensor for x in instances]
        track_ids = ids.split(n_t, dim=0)
        cls_scores = cls_scores.split(n_t, dim=0)
        instances, _ = custom_fast_rcnn_inference(
            boxes, cls_scores, track_ids, [None for _ in n_t],
            [x.image_size for x in instances],
            self.roi_heads.box_predictor[-1].test_score_thresh,
            self.roi_heads.box_predictor[-1].test_nms_thresh,
            self.roi_heads.box_predictor[-1].test_topk_per_image,
            self.not_clamp_box,
        )
        for inst in instances:
            inst.track_ids = inst.track_ids + inst.pred_classes * 10000 + \
                video_id * 100000000
        return instances


    def _get_insts_with_feats(self, proposals, features):
        if self.training:
            asso_thresh = self.cfg.MODEL.ASSO_HEAD.ASSO_THRESH
        else:
            asso_thresh = self.cfg.MODEL.ASSO_HEAD.ASSO_THRESH_TEST
        fg_inds = [x.objectness_logits > asso_thresh for x in proposals]
        filtered_proposals = [x[inds] for (x, inds) in zip(proposals, fg_inds)]
        features = [features[f] for f in self.cfg.MODEL.ROI_HEADS.IN_FEATURES]
        proposal_boxes = [x.proposal_boxes for x in filtered_proposals]
        pool_features = self.asso_pooler(features, proposal_boxes)
        reid_features = self.asso_head(pool_features)
        reid_features = reid_features.view(1, -1, self.cfg.MODEL.ASSO_HEAD.FC_DIM)
        n_t = [len(x) for x in filtered_proposals]
        instances_wo_id = [inst[inds] for inst, inds in zip(proposals, fg_inds)]
        features = reid_features.view(-1, self.cfg.MODEL.ASSO_HEAD.FC_DIM).split(n_t, dim=0)
        for inst, feat in zip(instances_wo_id, features):
            inst.reid_features = feat
        pred_instances = instances_wo_id
        for p in pred_instances:
            p.pred_boxes = p.proposal_boxes
            p.scores = p.objectness_logits
            p.pred_classes = torch.zeros(
                (len(p),), dtype=torch.long, device=p.pred_boxes.device)
            p.remove('proposal_boxes')
            p.remove('objectness_logits')
        return pred_instances
    

    def _transfer_track_ids(self, gt_tracks, pred_tracks):
        id_dict = {}
        for pred_id in pred_tracks.keys():
            pred_track = pred_tracks[pred_id]
            max_iou, max_iou_id = 0, -1
            for gt_id in gt_tracks.keys():
                gt_track = gt_tracks[gt_id]
                ious, num = 0, 0
                for frame_id in pred_track.keys():
                    if frame_id not in gt_track:
                        continue
                    pred_box = pred_track[frame_id]['bbox'].cpu().numpy()
                    gt_box = gt_track[frame_id]['bbox'].cpu().numpy()
                    ious += calculate_iou(pred_box, gt_box)
                    num += 1
                if num == 0:
                    continue
                iou = ious / num
                if iou >= max_iou:
                    max_iou = iou
                    max_iou_id = gt_id
            if max_iou > 0.2:
                id_dict[pred_id] = max_iou_id
        out_pred_tracks, gt_ids = [], []
        for pred_id, pred_track in pred_tracks.items():
            if pred_id in id_dict:
                out_pred_tracks.append(pred_track)
                gt_ids.append(id_dict[pred_id])
        return out_pred_tracks, gt_ids

    
    @torch.no_grad()
    def inferene_tracks(self, batched_inputs):
        if self.training:
            self.backbone.eval()
            self.proposal_generator.eval()
        local_tracker = BYTETracker()
        video_len = len(batched_inputs)
        instances = []
        ret_instances = []
        video_features = []
        for frame_id in range(video_len):
            # instances_wo_id = self.inference(
            #     batched_inputs[frame_id: frame_id + 1], 
            #     do_postprocess=False)
            images = self.preprocess_image(batched_inputs[frame_id: frame_id + 1])
            features = self.backbone(images.tensor)
            video_features.append(features)
            proposals, _ = self.proposal_generator(images, features, None)
            instances_wo_id = self._get_insts_with_feats(proposals=proposals, features=features)            
            instances.extend([x for x in instances_wo_id])
            inst = instances[frame_id]
            dets = torch.cat([
                inst.pred_boxes.tensor, 
                inst.scores[:, None]], dim=1).cpu()
            # id_feature = inst.reid_features.cpu()
            tracks = local_tracker.update(dets)
            track_inds = [x.ind for x in tracks]
            ret_inst = inst[track_inds]
            track_ids = [x.track_id for x in tracks]
            ret_inst.track_ids = ret_inst.pred_classes.new_tensor(track_ids)
            ret_instances.append(ret_inst)
        instances = ret_instances

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        pred_instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        if self.training:
            self.backbone.train()
            self.proposal_generator.train()
        return pred_instances, video_features


    def local_tracker_inference(self, batched_inputs):
        video_info = batched_inputs[-1]
        batched_inputs = batched_inputs[:-1]
        # get gt for evalutaions only
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        pred_instances, video_features = self.inferene_tracks(batched_inputs)
        # track finished
        pred_tracks = self._post_process_tracks(pred_instances, gt=False)
        gt_tracks = self._post_process_tracks(gt_instances, gt=True)
        gt_tracks = self._check_gt_tracks(gt_tracks, video_info['caption'])
        transfered_pred_tracks, gt_ids = self._transfer_track_ids(gt_tracks=gt_tracks, pred_tracks=pred_tracks)
        
        pred_summary = self.roi_heads({'feats': video_features, 'pred_tracks': transfered_pred_tracks, 'mode': 'summary'})
        gt_summary = video_info['summary']
        
        pred_captions = self.roi_heads({'pred_tracks': transfered_pred_tracks, 'mode': 'caption'})
        gt_captions = []
        for gt_id in gt_ids:
            gt_captions.append(video_info['caption'][gt_id])

        pred_relations = self.roi_heads({'pred_tracks': transfered_pred_tracks, 'gt_ids': gt_ids, \
                                        'texts': video_info['relation'],'mode': 'relation'})
        if pred_relations is None:
            pred_relations = []
        else:
            pred_relations = pred_relations.cpu().tolist()
        gt_relations = []
        for i in range(len(gt_ids)):
            for j in range(len(gt_ids)):
                gt_id_i = gt_ids[i]
                gt_id_j = gt_ids[j]
                k = str(gt_id_i) + '-' + str(gt_id_j)
                if k in video_info['relation']:
                    gt_relations.append(get_relation_labels(video_info['relation'][k]))
        gt_relations = [gt.cpu().tolist() for gt in gt_relations]
        assert len(pred_relations) == len(gt_relations)
        
        return pred_instances, {'pred': pred_summary, 'gt': [gt_summary]}, {'pred': pred_captions, 'gt': gt_captions}, {'pred': pred_relations, 'gt': gt_relations}


def calculate_iou(box1, box2):
    # 计算交集的坐标范围
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 计算交集的面积
    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # 计算并集的面积
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area

    # 计算交并比
    iou = intersection_area / union_area

    return iou
