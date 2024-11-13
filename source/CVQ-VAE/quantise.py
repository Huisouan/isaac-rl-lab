import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange


class VectorQuantiser(nn.Module):
    """
    改进版向量量化器，具有动态初始化未优化的“死”点。
    
    参数:
    - num_embed: 代码本条目数量
    - embed_dim: 代码本条目维度
    - beta: 承诺损失权重
    - distance: 查找最近代码的距离度量方法 ('cos' 或 'l2')
    - anchor: 锚定采样方法 ('probrandom', 'random', 'closest')
    - first_batch: 如果为真，则使用模型的离线版本
    - contras_loss: 如果为真，则使用对比损失进一步提高性能
    """
    def __init__(self, num_embed, embed_dim, beta, distance='cos', 
                 anchor='probrandom', first_batch=False, contras_loss=False):
        super().__init__()
        
        self.num_embed = num_embed  # 代码本大小
        self.embed_dim = embed_dim  # 嵌入维度
        self.beta = beta  # 承诺损失权重
        self.distance = distance  # 距离度量方法
        self.anchor = anchor  # 锚定采样方法
        self.first_batch = first_batch  # 是否为第一个批次
        self.contras_loss = contras_loss  # 是否使用对比损失
        self.decay = 0.99  # 衰减率
        self.init = False  # 初始化标志
        
        self.pool = FeaturePool(self.num_embed, self.embed_dim)  # 特征池
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)  # 嵌入层
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)  # 初始化嵌入权重
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))  # 注册缓冲区用于存储嵌入概率
    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "仅用于与Gumbel接口兼容"
        assert rescale_logits == False, "仅用于与Gumbel接口兼容"
        assert return_logits == False, "仅用于与Gumbel接口兼容"
        
        # 重塑输入张量并展平
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)
        
        # 计算距离
        if self.distance == 'l2':
            # L2距离计算
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # 余弦距离计算
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))
        
        # 编码
        sort_distance, indices = d.sort(dim=1)
        encoding_indices = indices[:, -1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # 量化并恢复形状
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # 计算嵌入损失
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        # 保留梯度
        z_q = z + (z_q - z).detach()
        # 恢复原始输入形状
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        
        # 统计
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings
        
        # 在线聚类重新初始化未优化的点
        if self.training:
            # 计算代码条目的平均使用情况
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            # 运行平均更新
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                if self.anchor == 'closest':
                    # 最近采样
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1, :]]
                elif self.anchor == 'random':
                    # 基于特征池的随机采样
                    random_feat = self.pool.query(z_flattened.detach())
                elif self.anchor == 'probrandom':
                    # 基于概率的随机采样
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # 基于平均使用情况的衰减参数
                decay = torch.exp(-(self.embed_prob * self.num_embed * 10) / (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            
            # 对比损失
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0) / self.num_embed)):, :].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0) * 1 / 2), :]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss += contra_loss
        
        return z_q, loss, (perplexity, min_encodings, encoding_indices)

class FeaturePool():
    """
    实现一个特征缓冲区，用于存储先前编码的特征。
    
    该缓冲区使我们能够使用生成特征的历史记录来初始化代码本，而不是最新编码器生成的特征。
    """
    def __init__(self, pool_size, dim=64):
        """
        初始化FeaturePool类。
        
        参数:
        - pool_size(int): 特征缓冲区大小
        - dim(int): 特征维度，默认为64
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1) / pool_size
    
    def query(self, features):
        """
        从池中返回特征。
        """
        self.features = self.features.to(features.device)
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size:  # 如果批量足够大，直接更新整个代码本
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # 如果批量不够大，暂时存储以备下次更新
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features
        
        return self.features