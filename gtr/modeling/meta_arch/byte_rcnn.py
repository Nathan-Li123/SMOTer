import cv2
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.structures import Boxes, pairwise_iou, Instances

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .custom_rcnn import CustomRCNN
from ..roi_heads.custom_fast_rcnn import custom_fast_rcnn_inference
from ..tracker.byte_tracker import BYTETracker


@META_ARCH_REGISTRY.register()
class BYTERCNN(CustomRCNN):
    @configurable
    def __init__(self, **kwargs):
        """
        """
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

        self.tracker = BYTETracker()


    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
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


    def forward(self, batched_inputs):
        """
        All batched images are from the same video
        During testing, the current implementation requires all frames 
            in a video are loaded.
        TODO (Xingyi): one-the-fly testing
        """
        if not self.training:
            return self.local_tracker_inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        # _, detector_losses = self.roi_heads(
        #     images, features, proposals, gt_instances)
        losses = {}
        # losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


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

    def local_tracker_inference(self, batched_inputs):
        from ...tracking.local_tracker.fairmot import FairMOT
        local_tracker = BYTETracker()

        video_len = len(batched_inputs)
        instances = []
        ret_instances = []
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], 
                do_postprocess=False)
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
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances