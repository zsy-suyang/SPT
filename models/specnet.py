import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.nn.init as init
from models.Conv_Blocks import Inception_Block_V1

class TGA(nn.Module):
    def __init__(self, feat_dim, num_neighbors, device):
        super(TGA, self).__init__()
        self.feat_dim = feat_dim
        self.num_neighbors = num_neighbors
        self.device = device
        self.similarity_proj = nn.Linear(2 * feat_dim, 1)
        init.xavier_uniform_(self.similarity_proj.weight)
        
        self.register_buffer('local_mask_template', None)

    def _create_mask(self, T, device):
        q_indices = torch.arange(T, device=device)
        k_indices = torch.arange(T, device=device)

        dist = torch.abs(q_indices.unsqueeze(1) - k_indices.unsqueeze(0))

        mask = (dist <= self.num_neighbors).bool()
        self.local_mask_template = mask

    def forward(self, x):
        B, T, N = x.size()

        x_i = x.unsqueeze(2).expand(-1, -1, T, -1)  # [B, T, T, N]
        x_j = x.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, T, N]

        node_pairs = torch.cat([x_i, x_j], dim=-1)  # [B, T, T, 2N]
        e = self.similarity_proj(node_pairs).squeeze(-1)  # [B, T, T]

        if self.local_mask_template is None or self.local_mask_template.size(0) != T:
            self._create_mask(T, self.device)
        mask = self.local_mask_template.unsqueeze(0).expand(B, -1, -1)

        e_masked = e.masked_fill(~mask, float('-inf'))
        attention = torch.softmax(e_masked, dim=-1)  # [B, T, T]

        out = torch.bmm(attention, x)  # [B, T, N]

        return out


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class SpecBlock(nn.Module):
    def __init__(self, leng, seq_len, pred_len, top_k, d_model, d_ff, num_kernels, device):
        super(SpecBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(d_model, self.k)
        self.graph = TGA(feat_dim=d_model, num_neighbors=3, device=device)

    def forward(self, x):
        B, T, N = x.size() 
        period_list, period_weight_fft = FFT_for_Period(x, self.k)
        
        xx = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        period_weight = self.linear1(xx)

        res = []
        
        for i in range(self.k):
            period = period_list[i]
            if T % period != 0:
                padding_len = period - (T % period)
                out = F.pad(x, (0, 0, 0, padding_len)) 
                length = T + padding_len
            else:
                out = x
                length = T

            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            
            res.append(out[:, :T, :])
            
        res = torch.stack(res, dim=-1) 
        
        period_weight = F.softmax(period_weight, dim=1) 
        period_weight = period_weight.unsqueeze(1).unsqueeze(1) 
        res = torch.sum(res * period_weight, -1)
        
        res = res + x
        graph = self.graph(x)
        res = res + graph 

        return res