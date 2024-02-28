import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Optional

class FusionModule(torch.nn.Module):
    def __init__(self, mode='summary', avg_num=5, min_track_len=20, max_track_len=256) -> None:
        super().__init__()
        self.mode = mode
        if mode == 'summary':
            self.avg_num = avg_num
            # self.resizer = nn.Linear(299, 192)
            self.decoder = Decoder(d_model=256, dim_kv=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu')
            self.vt_decoder = Decoder(d_model=256, dim_kv=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu')
            self.norm = nn.LayerNorm(normalized_shape=256)
        elif mode == 'caption':
            # self.encoder = Encoder(d_model=256, nhead=8,  dropout=0.1)
            pass
        elif mode == 'relation':
            self.decoder = Decoder(d_model=256, dim_kv=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu')
            self.avg_pool = torch.nn.AdaptiveAvgPool1d(128)
        else:
            raise ValueError('Unkonwn mode %s.' % self.mode)

    def forward(self, inputs, extra=None):
        if self.mode == 'summary':
            for i in range(inputs.shape[0]):
                frame_id = i + 1
                # frame_feature -> (1, 160, 256)
                frame_feature = inputs[i].unsqueeze(0)
                if int(frame_id) == 1:
                    video_outputs = frame_feature
                elif int(frame_id) <= self.avg_num:
                    video_outputs = video_outputs + frame_feature
                else:
                    if int(frame_id) == 6:
                        video_outputs /= self.avg_num
                        video_outputs = video_outputs.permute(1, 0, 2)
                    frame_feature = frame_feature.permute(1, 0, 2)
                    video_outputs = self.decoder(video_outputs, frame_feature)
                    video_outputs = self.norm(video_outputs)
            video_outputs = video_outputs.permute(1, 0, 2)

            track_outputs = []
            for track in extra:
                track_feats = []
                for frame_id in track.keys():
                    feat = track[frame_id]['feat'].cuda().view(4, 256)
                    track_feats.append(feat)
                track_feats = torch.cat(track_feats).unsqueeze(0)
                track_outputs.append(track_feats)
            
            for track_feat in track_outputs:
                video_outputs = video_outputs.permute(1, 0, 2)
                track_feat = track_feat.permute(1, 0, 2)
                video_outputs = self.vt_decoder(video_outputs, track_feat)
                video_outputs = video_outputs.permute(1, 0, 2)
            outputs = [video_outputs]
        elif self.mode == 'caption':
            outputs = []
            for track in inputs:
                track_feats = []
                for frame_id in track.keys():
                    feat = track[frame_id]['feat'].cuda().view(4, 256)
                    track_feats.append(feat)
                track_feats = torch.cat(track_feats).unsqueeze(0)
                outputs.append(track_feats)
        elif self.mode == 'relation':
            outputs = []
            for feats in inputs:
                source_track = feats['source']
                source_feat = []
                for frame_id in source_track.keys():
                    feat = source_track[frame_id]['feat'].cuda().view(4, 256)
                    source_feat.append(feat)
                source_feat = torch.cat(source_feat).unsqueeze(0)
                target_track = feats['target']
                target_feat =[]
                for frame_id in target_track.keys():
                    feat = target_track[frame_id]['feat'].cuda().view(4, 256)
                    target_feat.append(feat)
                target_feat = torch.cat(target_feat).unsqueeze(0)
                
                source_feat = source_feat.permute(1, 0, 2)
                target_feat = target_feat.permute(1, 0, 2)
                fused_feat = self.decoder(source_feat, target_feat)
                fused_feat = fused_feat.permute(1, 0, 2)
                fused_feat = self.avg_pool(fused_feat.transpose(1, 2))
                
                outputs.append(fused_feat)
            outputs = torch.stack(outputs)
        else:
            raise ValueError('Unkonwn mode %s.' % self.mode)
        return outputs


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos_src: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos_src)
        tmp = self.self_attn(q, k, value=src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(tmp)
        src = self.norm(src)
        return src


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, dim_kv, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=dim_kv, vdim=dim_kv)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None):

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos_dec),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")