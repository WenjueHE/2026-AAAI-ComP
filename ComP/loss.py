import torch
import torch.nn as nn
import torch.nn.functional as F


## iemocap loss function: same with CE loss
class MaskedCELoss(nn.Module):

    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target, umask, mask_m=None, first_stage=True):
        """
        pred -> [batch*seq_lentrain_transformer_expert_missing_softmoe.py, n_classes]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        if first_stage:
            umask = umask.view(-1,1) # [batch*seq_len, 1]
            mask = umask.clone()

            if mask_m == None:
                mask_m = mask
            mask_m = mask_m.reshape(-1, 1)  # [batch*seq_len, 1]

            target = target.view(-1,1) # [batch*seq_len, 1]
            pred = F.log_softmax(pred, 1) # [batch*seqlen, n_classes]
            loss = self.loss(pred*mask*mask_m, (target*mask*mask_m).squeeze().long()) / torch.sum(mask*mask_m)
            return loss
        else:
            assert first_stage == False
            umask = umask.view(-1, 1)  # [batch*seq_len, 1]
            mask = umask.clone()

            # l = mask.size(0)//7
            # mask[:4*l] = 0
            # mask[1*l:] = 0

            if mask_m == None:
                mask_m = mask
            mask_m = mask_m.reshape(-1, 1)  # [batch*seq_len, 1]

            target = target.view(-1, 1)  # [batch*seq_len, 1]
            pred = F.log_softmax(pred, 1)  # [batch*seqlen, n_classes]
            loss = self.loss(pred * mask * mask_m, (target * mask * mask_m).squeeze().long()) / torch.sum(mask * mask_m)
            if torch.isnan(loss) == True:
                loss = 0
            return loss


## for cmumosi and cmumosei loss calculation
class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        umask = umask.view(-1, 1)  # [batch*seq_len, 1]
        mask = umask.clone()

        pred = pred.view(-1, 1) # [batch*seq_len, 1]
        target = target.view(-1, 1) # [batch*seq_len, 1]

        loss = self.loss(pred*mask, target*mask) / torch.sum(mask)

        return loss

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor, masks):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())).sum() / max(1e-6, masks.sum())

class DisentangledLoss(nn.Module):
    # https://github.com/dorothy-yao/drfuse
    def __init__(self):
        super(DisentangledLoss, self).__init__()
        self.alignment_cos_sim = nn.CosineSimilarity(dim=-1)
        self.jsd = JSD()

    def _masked_abs_cos_sim(self, x, y, mask):
        return (self.alignment_cos_sim(x, y).abs().view(-1,1) * mask).sum() / max(mask.sum(), 1e-6)

    def forward(self, private_0, private_1, umask):
        umask = umask.view(-1, 1)  # [batch*seq_len, 1]
        mask = umask.clone()
        loss = self._masked_abs_cos_sim(private_0, private_1, mask)
        return loss
    
    def foward_1(self, private_0, private_1, shared_0, shared_1, umask):
        umask = umask.view(-1, 1)  # [batch*seq_len, 1]
        mask = umask.clone()
        loss = self._masked_abs_cos_sim(private_0, private_1, mask) + self._masked_abs_cos_sim(private_0, shared_0, mask) + self._masked_abs_cos_sim(private_1, shared_1, mask) 
        return loss
    
    def foward_2(self, private_0, private_1, shared_0, shared_1, umask):
        umask = umask.view(-1, 1)  # [batch*seq_len, 1]
        mask = umask.clone()
        jsd = self.jsd(shared_0.sigmoid(), shared_1.sigmoid(), torch.ones_like(mask))
        loss = self._masked_abs_cos_sim(private_0, private_1, mask) + self._masked_abs_cos_sim(private_0, shared_0, mask) + self._masked_abs_cos_sim(private_1, shared_1, mask) 
        return loss + jsd


class SampleContrastiveLoss(nn.Module):
    def __init__(self):
        super(SampleContrastiveLoss, self).__init__()

    def forward(self, fea0, fea1,mask, temperature=0.07):
        batch_size, seq_len, dim = fea0.shape

        # 展平特征和标签 [batch*seq_len, dim] & [batch*seq_len]
        mask_flat = mask.reshape(-1).bool()
        fea0_flat = fea0.reshape(-1, dim)[mask_flat]
        fea1_flat = fea1.reshape(-1, dim)[mask_flat]

        # 特征和原型归一化 (余弦相似度)
        fea0_norm = F.normalize(fea0_flat, p=2, dim=1)  # [N, dim]
        fea1_norm = F.normalize(fea1_flat, p=2, dim=1)  # [N, dim]

        # 计算相似度矩阵 [N, N]
        similarity = torch.matmul(fea0_norm, fea1_norm.T) / temperature

        # 创建正样本掩码 [N, C]
        pos_mask = torch.eye(fea0_norm.shape[0], dtype=torch.bool, device=fea0.device)

        # 计算对比损失
        exp_sim = torch.exp(similarity)
        numerator = torch.sum(exp_sim * pos_mask, dim=1)  # 正样本项
        denominator = torch.sum(exp_sim, dim=1)  # 所有样本项

        # 避免除零错误
        loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))
        return loss.mean()

    
class LabelContrastiveLoss(nn.Module):
    def __init__(self):
        super(LabelContrastiveLoss, self).__init__()

    def forward(self, fea, prototype, label, mask, temperature=0.07):
        batch_size, seq_len, dim = fea.shape
        num_classes = prototype.size(0)

        # 展平特征和标签 [batch*seq_len, dim] & [batch*seq_len]
        mask_flat = mask.reshape(-1).bool()
        fea_flat = fea.reshape(-1, dim)[mask_flat]
        label_flat = label.reshape(-1).long()[mask_flat]

        # 特征和原型归一化 (余弦相似度)
        fea_norm = F.normalize(fea_flat, p=2, dim=1)  # [N, dim]
        proto_norm = F.normalize(prototype, p=2, dim=1)  # [C, dim]

        # 计算相似度矩阵 [N, C]
        similarity = torch.matmul(fea_norm, proto_norm.T) / temperature

        # 创建正样本掩码 [N, C]
        pos_mask = torch.zeros_like(similarity)
        pos_mask[torch.arange(fea_flat.size(0)), label_flat] = 1  # 对角线位置为正样本

        # 计算对比损失
        exp_sim = torch.exp(similarity)
        numerator = torch.sum(exp_sim * pos_mask, dim=1)  # 正样本项
        denominator = torch.sum(exp_sim, dim=1)  # 所有样本项

        # 避免除零错误
        loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))
        return loss.mean()

class PrototypeLoss(nn.Module):
    def __init__(self, D_e, shared_all_proj):
        super(PrototypeLoss, self).__init__()
        self.D_e = D_e
        self.shared_a_proj = shared_all_proj[:,:,:D_e]
        self.shared_t_proj = shared_all_proj[:,:,D_e:2*D_e]
        self.shared_v_proj = shared_all_proj[:,:,2*D_e:3*D_e]
        self.shared_atv_proj = shared_all_proj[:,:,3*D_e:]
