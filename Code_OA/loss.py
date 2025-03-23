import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import cal_dice

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='macro'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        # 可学习的权重，初始化为均匀分布
        if reduction == 'macro':
            self.weights = nn.Parameter(torch.tensor([0.5, 0.5, 0.1]))
        else:
            self.weights = None

    def forward(self, logits, targets):
        if self.reduction == 'macro':
            dice_et, dice_tc, dice_wt = cal_dice(logits, targets)
            
            # 使用log函数计算损失
            dice_loss_et = 1 - torch.mean(dice_et)
            dice_loss_tc = 1 - torch.mean(dice_tc)
            dice_loss_wt = 1 - torch.mean(dice_wt)

            # 归一化权重
            weights = torch.softmax(self.weights, dim=0)
            
            # 加权求和
            dice_loss = dice_loss_et * weights[0] + dice_loss_tc * weights[1] + dice_loss_wt * weights[2]

            return dice_loss, dice_et, dice_tc, dice_wt
        
        elif self.reduction == 'target':
            dice_et, dice_tc, dice_wt = cal_dice(logits, targets)
            
            # 使用log函数计算损失
            dice_loss_et = -torch.log(torch.mean(dice_et) + 1e-8)  # 加上一个小的epsilon防止log(0)
            dice_loss_tc = -torch.log(torch.mean(dice_tc) + 1e-8)
            dice_loss_wt = -torch.log(torch.mean(dice_wt) + 1e-8)

            # 加权求和
            dice_loss = dice_loss_et * 0.44 + dice_loss_tc * 0.55 + dice_loss_wt * 0.01

            return dice_loss
            
        elif self.reduction == 'micro':
            # 自动获取输入数据所在的设备
            device = logits.device
            
            num_classes = logits.shape[1]
            probs = torch.softmax(logits, dim=1)
            
            # 确保one_hot编码在正确设备上
            targets_onehot = F.one_hot(targets, num_classes).to(device).permute(0, 4, 1, 2, 3)

            intersection = torch.sum(probs * targets_onehot, dim=(2, 3, 4))
            union = torch.sum(probs + targets_onehot, dim=(2, 3, 4))

            dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1.0 - torch.sum(dice_scores) / (num_classes * targets.shape[0])
            dice = torch.sum(dice_scores) / (num_classes * targets.shape[0])
            valid_classes = torch.sum(targets_onehot, dim=(2, 3, 4)) > 0
            dice_loss = 1.0 - torch.sum(dice_scores * valid_classes) / (torch.sum(valid_classes) + 1e-8)
            return dice_loss,dice,dice,dice
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=2.0, dice_reduction='micro', class_weights=None, device=None):
        super().__init__()
        
        # 自动检测设备（优先使用传入的device参数）
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dice_reduction = dice_reduction
        
        # 转换class_weights到指定设备
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(reduction=dice_reduction)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        if self.dice_reduction == 'target':
            ce_loss = self.ce(logits, targets)
            targets = targets.to(self.device).squeeze(dim=1).long()
            logits = logits.to(self.device)

            dice_loss = self.dice(logits, targets)
            return  self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        targets = targets.to(self.device).squeeze(dim=1).long()
        logits = logits.to(self.device)
        ce_loss = self.ce(logits, targets)
        dice_loss, dice_et, dice_tc, dice_wt = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss, dice_et, dice_tc, dice_wt

    def to(self, device):
        # 重写to方法以保持设备同步
        self.device = device
        return super().to(device)


if __name__ == "__main__":
    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 模拟数据（自动传输到设备）
    batch_size = 2
    num_classes = 4
    spatial_size = (64, 128, 128)
    
    logits = torch.randn(batch_size, num_classes, *spatial_size).to(device)
    targets = torch.randint(0, num_classes, (batch_size, *spatial_size)).to(device)
    
    # 初始化损失函数（显式指定设备）
    class_weights = torch.tensor([0.1, 1.0, 2.0, 3.0], device=device)
    
    criterion = CombinedLoss(
        ce_weight=1.0,
        dice_weight=0.5,
        dice_reduction='macro',
        class_weights=class_weights,
        device=device
    )
    
    # 计算损失
    loss = criterion(logits, targets)
    print(f"Total Loss: {loss.item():.4f}")