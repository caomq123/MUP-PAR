import numpy as np
import torch
#import cv2
from gcn_model import *
from roi_align.roi_align import RoIAlign  # RoIAlign module
from spectral_cluster.spectralcluster import SpectralClusterer
from utils import *
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch import einsum, nn
import torch.nn.functional as F
from backbone import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from collections import OrderedDict
import torch.utils.checkpoint as checkpoint


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        device = torch.device('cuda')
        self.unc = torch.zeros((4,)).to(device=device)
        self.unc.requires_grad = True

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        device = torch.device('cuda')
        self.unc = torch.zeros((4,)).to(device=device)
        self.unc.requires_grad = True

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        device = torch.device('cuda')
        self.unc = torch.zeros((4,)).to(device=device)
        self.unc.requires_grad = True

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        device = torch.device('cuda')
        self.unc = torch.zeros((4,)).to(device=device)
        self.unc.requires_grad = True

    def forward(self, x):
        device = torch.device('cuda')
        for attn, ff in self.layers.to(device):
            x = (attn(x)) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, pool='cls',
                 dim_head=768, dropout=0., emb_dropout=0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100 + 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        device = torch.device('cuda')
        self.unc = torch.zeros((4,)).to(device=device)
        self.unc.requires_grad = True

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        b, n, _ = x.shape

        device = x.device
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b).to(device)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)].to(device)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return x

class Block28(nn.Module):
    def __init__(self, cfg):
        super(Block28, self).__init__()
        self.cfg = cfg
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        self.backbone = MyResnet_18()

        if not self.cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.roi_align = RoIAlign(*self.cfg.crop_size)

        self.fc_emb_1 = nn.Linear(K * K * D, NFB)
        self.nl_emb_1 = nn.LayerNorm([NFB])

        self.global_fc_emb_1 = nn.Linear(D, NFB)

        self.gcn_ind = GCN_w_position_embedding(self.cfg)

        self.vit = ViT(
            dim=768,
            depth=1,
            heads=12,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.dropout_global = nn.Dropout(p=self.cfg.train_dropout_prob)

        self.FFN_actions = nn.Sequential(
            nn.Linear(NFG, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=self.cfg.train_dropout_prob),
            nn.Linear(256, self.cfg.num_actions)
        )
        self.FFN_social_activities = nn.Sequential(
            nn.Linear(NFG, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=self.cfg.train_dropout_prob),
            nn.Linear(256, self.cfg.num_social_activities)
        )
        self.FFN_activities = nn.Sequential(
            nn.Linear(NFG, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=self.cfg.train_dropout_prob),
            nn.Linear(512, self.cfg.num_activities)
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4),
            nn.Linear(8192, 768)
        )

        self.PTL_individual = GroupViT(img_size=32, patch_size=4, in_chans=512, num_classes=0, embed_dim=768,
                                       embed_factors=[1, 1], depths=[3, 1], num_heads=[8, 8],
                                       num_group_tokens=[8, 0], num_output_groups=[8], w=1,
                                       mlp_ratio=4., drop_rate=0.1,  attn_drop_rate=0.1, drop_path_rate=0.1)

        self.PTL_social = GroupViT(img_size=32, patch_size=4, in_chans=512, num_classes=0,
                                   embed_dim=768, embed_factors=[1, 1], depths=[3, 1], num_heads=[12, 12],
                                   num_group_tokens=[8, 0], num_output_groups=[8], w=1,
                                   mlp_ratio=4., drop_rate=0.1,
                                   attn_drop_rate=0.1, drop_path_rate=0.1)

        self.PTL_global = GroupViT(img_size=32, patch_size=4, in_chans=512, num_classes=0, embed_dim=768,
                                   embed_factors=[1, 1], depths=[3, 1], num_heads=[12, 12], num_group_tokens=[8, 0],
                                   num_output_groups=[8], w=1,
                                   mlp_ratio=4., drop_rate=0.1,
                                   attn_drop_rate=0.1, drop_path_rate=0.1)  # 本来都是0.1的

        self.Li = nn.Linear(768 + 768, 768)
        self.Lg = nn.Linear(768 + 768, 768)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def savemodel(self, filepath):
        state = {
            'backbone_state_dict': self.backbone.state_dict(),
        }
        torch.save(state, filepath)
        print('model saved to:', filepath)

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        print('Load model states from: ', filepath)

    def predict_social_activity_score_train(self, social_group_id, boxes_states):
        # calculate the pooled feature for each bbox according to the gt group they belong to
        social_group_id = social_group_id.reshape(-1)
        social_group_gt = create_P_matrix(np.array(social_group_id.cpu()))
        social_group_gt = generate_list(social_group_gt)
        social_group_gt.sort()
        group_states_all = []
        for this_group in social_group_gt:
            group_states_list_this = []
            for person_id in this_group:
                group_states_list_this.append(boxes_states[person_id])
            group_states_this = torch.stack(group_states_list_this)
            group_states_this = torch.reshape(group_states_this, (1, len(this_group), -1))
            group_states_this_vit = self.vit(group_states_this)
            group_states_this_group = self.PTL_social(group_states_this, False)
            group_states_this = group_states_this_group + 0.2 * group_states_this_vit
            group_states_all.append(group_states_this)

        # in case there is no group in a image:
        if not group_states_all:
            group_states_all.append(boxes_states[0])
        group_states_all = torch.stack(group_states_all)

        return group_states_all

    def predict_social_activity_score_test(self, relation_graph, boxes_states):
        this_N = relation_graph.shape[-1]
        this_cluster = SpectralClusterer(
            min_clusters=max(int(this_N * 0.6), 1),
            max_clusters=int(this_N),
            autotune=None,
            laplacian_type=None,
            refinement_options=None)
        # use clustering to get the groups
        social_group_predict = this_cluster.predict(np.array(relation_graph.cpu()))
        social_group_predict = create_P_matrix(np.array(social_group_predict))
        social_group_predict = generate_list(social_group_predict)
        social_group_predict.sort()

        group_states_all = []
        for this_group in social_group_predict:
            group_states_list_this = []
            for person_id in this_group:
                group_states_list_this.append(boxes_states[person_id])
            group_states_this = torch.stack(group_states_list_this)
            group_states_this = torch.reshape(group_states_this, (1, len(this_group), -1))
            group_states_this_vit = self.vit(group_states_this)  # (1, 3, 1024) -> (1, 1024)
            group_states_this_group = self.PTL_social(group_states_this, False)
            group_states_this = group_states_this_group + 0.2 * group_states_this_vit
            group_states_all.append(group_states_this)

        # in case there is no group in a image:
        if not group_states_all:
            group_states_all.append(boxes_states[0])
        group_states_all = torch.stack(group_states_all)
        return group_states_all

    def forward(self, batch_data):

        global graph_boxes_features
        images_in, boxes_in, bboxes_num_in, bboxes_social_group_id_in = batch_data

        # read config parameters
        B = images_in.shape[0]  # image_in: B * T * 3 * H *W
        T = images_in.shape[1]  # equals to 1 here
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        MAX_N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG = self.cfg.num_features_relation, self.cfg.num_features_gcn

        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (B * T, 3, H, W))  # B*T, 3, H, W-[8, 3, 360, 2640]
        images_pri = images_in_flat
        boxes_in = boxes_in.reshape(B * T, MAX_N, 4)  # [8, 60, 4]

        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)

        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # B*T, D, OH, OW--[8, 512, 15, 118]

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MAX_N, 4))  # B*T*MAX_N, 4

        boxes_idx = [i * torch.ones(MAX_N, dtype=torch.int) for i in range(B * T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, MAX_N
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MAX_N,))  # B*T*MAX_N,

        # RoI Align--每个image中提取出60个K*K的特征
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False

        boxes_features_all = self.roi_align(features_multiscale.float(),
                                            boxes_in_flat,
                                            boxes_idx_flat)  # B*T*MAX_N, D, K, K,

        boxes_features_all_i = self.to_patch_embedding(boxes_features_all)
        boxes_features_all_i = self.vit(boxes_features_all_i)
        boxes_features_all_i = F.relu(boxes_features_all_i)  # Norm
        boxes_features_all_i = boxes_features_all_i.reshape(B, T, MAX_N, NFB)

        boxes_features_all = F.relu(boxes_features_all)         # Norm

        boxes_in = boxes_in.reshape(B, T, MAX_N, 4)
        bboxes_social_group_id_in = bboxes_social_group_id_in.reshape(B, T, MAX_N, 1)      # 读取每个person的小组编号

        actions_scores = []
        social_acty_scores = []
        activities_scores = []
        relation_graphs = []
        bboxes_num_in = bboxes_num_in.reshape(B, T)  # B,T

        po, c, oh, ow = boxes_features_all.shape  # K个learnable tokens
        boxes_features_all = boxes_features_all.reshape(B, T, MAX_N, c, ow, oh)

        for b in range(B):
            N = bboxes_num_in[b][0]
            boxes_features1 = boxes_features_all_i[b, :, :N, :].reshape(1, T * N, NFB)  # 1,T,N,NFB--[1, 9, 1024]

            boxes_features_all4 = boxes_features_all[b, :, :N, :, :, :].reshape(N, 1, 512, ow,
                                                                                oh)
            boxes_features_all2 = []

            for boxes_features_all1 in boxes_features_all4:
                result = self.PTL_individual(boxes_features_all1, True, return_attn=True)
                boxes_features_all1 = result[0]

                boxes_features_all2.append(boxes_features_all1)



            boxes_features_all5 = torch.cat(boxes_features_all2, dim=0)  # ----[N,1024]

            # # self attention
            boxes_features_all5 = boxes_features_all5.reshape(1, T, N, NFB)

            boxes_features = boxes_features_all5[:, :, :N, :].reshape(1, T * N, NFB)

            boxes_features = 0.3 * boxes_features + boxes_features1
            # boxes_features = boxes_features

            boxes_positions = boxes_in[b, :, :N, :].reshape(T * N, 4)
            bboxes_social_group_id = bboxes_social_group_id_in[b, :, :N, :].reshape(1, T * N)

            # Node graph, for individual
            graph_boxes_features, relation_graph = self.gcn_ind(boxes_features, boxes_positions)

            # graph attention module
            graph_boxes_features = graph_boxes_features.reshape(1, T * N, NFB)

            # handle the relation graph
            relation_graph = relation_graph.reshape(relation_graph.shape[1], relation_graph.shape[1])
            relation_graphs.append(relation_graph)

            # cat graph_boxes_features with boxes_features
            boxes_features = boxes_features.reshape(1, T * N, NFB)
            boxes_states = graph_boxes_features + boxes_features
            boxes_states = self.dropout_global(boxes_states)
            NFS = NFG
            boxes_states = boxes_states.reshape(T * N, NFS)

            # Predict social activities
            if self.training:
                boxes_states_for_social_activity = self.predict_social_activity_score_train(
                    bboxes_social_group_id, boxes_states.reshape(N * T, NFS))
            else:
                boxes_states_for_social_activity = self.predict_social_activity_score_test(
                    relation_graph, boxes_states.reshape(N * T, NFS))
            boxes_states_for_social_activity = boxes_states_for_social_activity.reshape(-1, NFS)  # [4, 1024]

            # Predict activities
            boxes_states_pooled_1 = self.vit(boxes_states.reshape(1, T * N, NFS))
            group_states_pooled = self.PTL_global(boxes_states_for_social_activity.reshape(1, -1, NFS), False)

            # [1, 1024]
            global_f = group_states_pooled + boxes_states_pooled_1
            acty_score = self.FFN_activities(global_f)

            # Predict actions
            actn_score = self.FFN_actions(boxes_states.reshape(-1, NFS) + global_f)  # T,N, actn_num
            social_acty_score = self.FFN_social_activities(boxes_states_for_social_activity + global_f)

            actions_scores.append(actn_score)
            social_acty_scores.append(social_acty_score)
            activities_scores.append(acty_score)

        # exit()

        actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        social_acty_scores = torch.cat(social_acty_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.cat(activities_scores, dim=0)  # B,acty_num

        return actions_scores, activities_scores, relation_graphs, social_acty_scores

#-----------------------------------------------GroupViT------------------------------------------------#
# ------------------------------------------------------------------------------------------------------#
class Result:

    def __init__(self, as_dict=False):
        if as_dict:
            self.outs = {}
        else:
            self.outs = []

    @property
    def as_dict(self):
        return isinstance(self.outs, dict)

    def append(self, element, name=None):
        if self.as_dict:
            assert name is not None
            self.outs[name] = element
        else:
            self.outs.append(element)

    def update(self, **kwargs):
        if self.as_dict:
            self.outs.update(**kwargs)
        else:
            for v in kwargs.values():
                self.outs.append(v)

    def as_output(self):
        if self.as_dict:
            return self.outs
        else:
            return tuple(self.outs)

    def as_return(self):
        outs = self.as_output()
        if self.as_dict:
            return outs
        if len(outs) == 1:
            return outs[0]
        return outs


def interpolate_pos_encoding(pos_embed, H, W):
    num_patches = H * W

    N = pos_embed.shape[1]
    if num_patches == N and W == H:
        return pos_embed
    patch_pos_embed = pos_embed
    dim = pos_embed.shape[-1]
    patch_pos_embed = F.interpolate(
        # patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        patch_pos_embed.reshape(1, H, W, dim).permute(0, 3, 1, 2),
        size=(H, W),
        mode='bicubic',
        align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'


class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 w,
                 dim,
                 out_dim,
                 num_heads,
                 num_group_token,
                 num_output_group,
                 norm_layer,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.):
        super(GroupingBlock, self).__init__()
        self.w = w
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self):
        return f'hard={self.hard}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'num_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        # [B, S_2, C]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, x)
        new_x, attn_dict = self.assign(projected_group_tokens, x, return_attn=return_attn)
        new_x += self.w * projected_group_tokens     # 残差在这里，权重应该也加在这里

        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))

        return new_x, attn_dict


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            # print(query.shape)
            # print(self.num_heads)
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


class AttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_fuse=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GroupingLayer(nn.Module):
    """A Transformer layer with Grouping Block for one stage.

    Args:
        dim (int): Number of input channels.
        num_input_token (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            In GroupViT setting, Grouping Block serves as the downsampling layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        group_projector (nn.Module | None, optional): Projector for the grouping layer. Default: None.
        zero_init_group_token (bool): Whether to initialize the grouping token to 0. Default: False.
    """

    def __init__(self,
                 dim,
                 num_input_token,
                 depth,
                 num_heads,
                 num_group_token,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 group_projector=None,
                 zero_init_group_token=False):

        super().__init__()
        self.dim = dim
        self.input_length = num_input_token
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            if not zero_init_group_token:
                trunc_normal_(self.group_token, std=.02)
        else:
            self.group_token = None

        # build blocks
        self.depth = depth
        blocks = []
        for i in range(depth):
            blocks.append(
                AttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)

        self.downsample = downsample
        self.input_resolution = num_input_token
        self.use_checkpoint = use_checkpoint

        self.group_projector = group_projector

    @property
    def with_group_token(self):
        return self.group_token is not None

    def extra_repr(self):
        return f'dim={self.dim}, \n' \
               f'input_resolution={self.input_resolution}, \n' \
               f'depth={self.depth}, \n' \
               f'num_group_token={self.num_group_token}, \n'

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, x, prev_group_token=None, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            prev_group_token (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention maps
        """
        if self.with_group_token:
            group_token = self.group_token.expand(x.size(0), -1, -1)
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None

        B, L, C = x.shape
        cat_x = self.concat_x(x, group_token)      # learnable tokens + input tokens
        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                cat_x = checkpoint.checkpoint(blk, cat_x)
            else:
                cat_x = blk(cat_x)

        x, group_token = self.split_x(cat_x)     # 摘出来 learnable tokens/input tokens

        attn_dict = None
        if self.downsample is not None:
            x, attn_dict = self.downsample(x, group_token, return_attn=return_attn)

        return x, group_token, attn_dict


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, kernel_size=7, stride=4, padding=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.img_size = img_size
        self.patches_resolution = (
            int((img_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1] + 1),
            int((img_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0] + 1),
        )

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            # FIXME look at relaxing size constraints
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape


class GroupViT(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=0,
                 embed_dim=384,
                 embed_factors=[1, 1, 1],
                 depths=[6, 3, 3],
                 num_heads=[6, 6, 6],
                 num_group_tokens=[64, 8, 0],
                 num_output_groups=[64, 8],
                 w=0.5,
                 hard_assignment=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 patch_norm=True,
                 use_checkpoint=False,
                 pos_embed_type='simple',
                 freeze_patch_embed=False):
        super().__init__()
        assert patch_size in [4, 8, 16]
        self.num_classes = num_classes
        assert len(embed_factors) == len(depths) == len(num_group_tokens)
        assert all(_ == 0 for _ in num_heads) or len(depths) == len(num_heads)
        assert len(depths) - 1 == len(num_output_groups)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * embed_factors[len(depths) - 1])
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        self.pos_embed_type = pos_embed_type
        assert pos_embed_type in ['simple', 'fourier']

        norm_layer = nn.LayerNorm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if pos_embed_type == 'simple':
            self.pos_embed = self.build_simple_position_embedding()
        elif pos_embed_type == 'fourier':
            self.pos_embed = self.build_2d_sincos_position_embedding()
        else:
            raise ValueError

        if freeze_patch_embed:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.pos_embed.requires_grad = False

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        num_input_token = num_patches
        num_output_token = num_input_token
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            dim = int(embed_dim * embed_factors[i_layer])
            downsample = None
            if i_layer < self.num_layers - 1:
                out_dim = embed_dim * embed_factors[i_layer + 1]
                downsample = GroupingBlock(
                    w=w,
                    dim=dim,
                    out_dim=out_dim,
                    num_heads=num_heads[i_layer],
                    num_group_token=num_group_tokens[i_layer],
                    num_output_group=num_output_groups[i_layer],
                    norm_layer=norm_layer,
                    hard=hard_assignment,
                    gumbel=hard_assignment)
                num_output_token = num_output_groups[i_layer]

            if i_layer > 0 and num_group_tokens[i_layer] > 0:
                prev_dim = int(embed_dim * embed_factors[i_layer - 1])
                group_projector = nn.Sequential(
                    norm_layer(prev_dim),
                    MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))

                if dim != prev_dim:
                    group_projector = nn.Sequential(group_projector, norm_layer(prev_dim),
                                                    nn.Linear(prev_dim, dim, bias=False))
            else:
                group_projector = None
            layer = GroupingLayer(
                dim=dim,
                num_input_token=num_input_token,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_group_token=num_group_tokens[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                group_projector=group_projector,
                # only zero init group token if we have a projection
                zero_init_group_token=group_projector is not None)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                num_input_token = num_output_token

        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
        if self.pos_embed_type == 'simple' and 'pos_embed' in state_dict:
            load_pos_embed = state_dict['pos_embed']
            pos_embed = self.pos_embed
            if load_pos_embed.shape != pos_embed.shape:
                H_new = int(self.patch_embed.num_patches**0.5)
                W_new = H_new
                H_ori = int(load_pos_embed.shape[1]**0.5)
                W_ori = H_ori
                load_pos_embed = F.interpolate(
                    rearrange(load_pos_embed, 'b (h w) c -> b c h w', h=H_ori, w=W_ori, b=1),
                    size=(H_new, W_new),
                    mode='bicubic',
                    align_corners=False)
                load_pos_embed = rearrange(load_pos_embed, 'b c h w -> b (h w) c', h=H_new, w=W_new)
                state_dict['pos_embed'] = load_pos_embed
        return super().load_state_dict(state_dict, strict)

    def build_simple_position_embedding(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.patches_resolution
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pos_embed = nn.Parameter(pos_emb)
        pos_embed.requires_grad = False
        return pos_embed

    @property
    def width(self):
        return self.num_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pos_embed(self, B, H, W):
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed
        pos_embed = interpolate_pos_encoding(pos_embed, H, W)
        return pos_embed

    def forward_features(self, x, *, return_attn=False, flag=False):
        if flag:                                      # flag = true 是global 和 individual
            B = x.shape[0]
            x, hw_shape = self.patch_embed(x)

            x = x + self.get_pos_embed(B, *hw_shape)
            x = self.pos_drop(x)
        else:
            num_tokens = x.shape[1]                  # flag = false 是social
            dim = x.shape[2]
            pose = nn.Parameter(torch.randn(1, num_tokens, dim))
            device = torch.device('cuda')
            pose = pose.to(device)
            x = x + pose
            x = self.pos_drop(x)

        group_token = None
        attn_dict_list = []
        for layer in self.layers:
            x, group_token, attn_dict = layer(x, group_token, return_attn=return_attn)
            attn_dict_list.append(attn_dict)

        x = self.norm(x)

        return x, group_token, attn_dict_list

    def forward_image_head(self, x):
        """

        Args:
            x: shape [B, L, C]

        Returns:

        """
        # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward(self, x, flag, *, return_feat=False, return_attn=False, as_dict=False):
        x, group_token, attn_dicts = self.forward_features(x, return_attn=return_attn, flag=flag)
        x_feat = x if return_feat else None

        outs = Result(as_dict=as_dict)

        outs.append(self.forward_image_head(x), name='x')

        if return_feat:
            outs.append(x_feat, name='feat')

        if return_attn:
            outs.append(attn_dicts, name='attn_dicts')

        return outs.as_return()