import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = Adapter(num_in=128*256, num_out=360, num_hidden=2048)
        self.loss_func = FocalLoss()
    
    def forward(self, batch):
        feat = batch['feat']
        feat = feat.view(feat.shape[0], -1)
        out = self.adapter(feat)
        if not self.training:
            return out
        else:
            label = torch.stack(batch['labels']).cuda()
            # print('out', out.shape)
            # print('label', label.shape)
            assert out.shape == label.shape 
            loss = self.loss_func(out.float(), label.float())
            return loss
        

class Adapter(nn.Module):
    def __init__(self, num_in, num_out, num_hidden):
        super().__init__()
        self.linear_1 = torch.nn.Linear(num_in, num_hidden)
        self.linear_2 = torch.nn.Linear(num_hidden, num_hidden)
        self.linear_3 = torch.nn.Linear(num_hidden, num_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.relu(self.linear_2(x))
        x = self.linear_3(x)
        return x


# def multi_label_classification_hamming_loss(preds, targets):
#     """
#     计算多标签分类Hamming Loss的函数。
#     :param preds: 预测的概率值，大小为 [batch_size, num_classes]
#     :param targets: 目标标签值，大小为 [batch_size, num_classes]
#     :return: 多标签分类Hamming Loss的值, 大小为 [1]
#     """
#     # 将概率值转换为二进制标签（0或1）
#     binary_preds = torch.round(torch.sigmoid(preds))
#     # 计算Hamming Loss
#     hamming_loss = 1 - (binary_preds == targets).float().mean()
#     return hamming_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss

        return focal_loss
    