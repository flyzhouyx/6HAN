import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import dgl.function as fn
from dgl.nn.pytorch.hetero import HeteroGraphConv
from dgl.nn.pytorch.conv.gatconv import GATConv
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import random
import warnings

warnings.filterwarnings("ignore")

# ===================== 配置项（对齐论文参数）=====================
GRAPH_PATH = "Data/ipv6_hetero_graph.bin"
ENTITY_MAP_PATH = "Data/entity_mapping.csv"
MODEL_SAVE_PATH = "Data/6han_model.pth"
EMBEDDING_SAVE_PATH = "Data/prefix_embeddings_6han.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 64  # 论文指定的隐藏层维度
NUM_HEADS = 1  # 论文无多头注意力，单头即可
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
TAU = 0.05  # InfoNCE温度系数
CONTRAST_WEIGHT = 0.6  # 论文指定对比损失权重
LINK_WEIGHT = 0.4  # 论文指定链路损失权重


# ===================== 1. 数据加载（少量修正）=====================
def load_data() -> Tuple[dgl.DGLHeteroGraph, pd.DataFrame, Dict[str, Dict[str, int]], Dict[str, Dict[int, str]]]:
    graphs, _ = dgl.load_graphs(GRAPH_PATH)
    g = graphs[0].to(DEVICE)
    print(f"成功加载异构图，节点类型：{g.ntypes}，边类型：{g.canonical_etypes}")

    mapping_df = pd.read_csv(ENTITY_MAP_PATH)
    ent2id = {}
    id2ent = {}  # 新增：id到实体值的映射（用于链路预测负例生成）
    for ent_type in mapping_df["entity_type"].unique():
        sub_df = mapping_df[mapping_df["entity_type"] == ent_type]
        ent2id[ent_type] = dict(zip(sub_df["entity_value"], sub_df["entity_id"]))
        id2ent[ent_type] = dict(zip(sub_df["entity_id"], sub_df["entity_value"]))
    print(f"成功加载实体映射，含{len(ent2id)}类实体")

    return g, mapping_df, ent2id, id2ent  # 新增id2ent返回


g, mapping_df, ent2id, id2ent = load_data()


# ===================== 2. 正/负例采样（核心修复：分离索引和Prefix信息）=====================
def build_prefix_info(g: dgl.DGLHeteroGraph, mapping_df: pd.DataFrame, ent2id: Dict[str, Dict[str, int]],
                      id2ent: Dict[str, Dict[int, str]]) -> Tuple[Dict[int, Dict], Dict[str, defaultdict]]:
    """
    构建Prefix的基础信息 + 独立的索引字典（修复KeyError核心）
    Returns:
        prefix_info: {p_id: {"netname": "", "mnt_by": "", "country": "", "id": p_id}}
        index_dict: {"mnt_netname": defaultdict, "mnt": defaultdict, "country": defaultdict}
    """
    prefix_info = {}
    # 提取各类边的映射关系
    p2n_edges = g.edges(etype=("Prefix", "prefix_to_netname", "Netname"), form="uv")
    p2n_dict = {p.item(): id2ent["Netname"][n.item()] for p, n in zip(p2n_edges[0], p2n_edges[1])}

    p2m_edges = g.edges(etype=("Prefix", "prefix_to_mnt", "Mnt"), form="uv")
    p2m_dict = {p.item(): id2ent["Mnt"][m.item()] for p, m in zip(p2m_edges[0], p2m_edges[1])}

    p2c_edges = g.edges(etype=("Prefix", "prefix_to_country", "Country"), form="uv")
    p2c_dict = {p.item(): id2ent["Country"][c.item()] for p, c in zip(p2c_edges[0], p2c_edges[1])}

    # 初始化每个Prefix的信息（仅包含整数ID键）
    prefix_ids = list(range(g.num_nodes("Prefix")))
    for p_id in prefix_ids:
        prefix_info[p_id] = {
            "netname": p2n_dict.get(p_id, ""),
            "mnt_by": p2m_dict.get(p_id, ""),
            "country": p2c_dict.get(p_id, ""),
            "id": p_id
        }

    # 构建独立的索引字典（不再混入prefix_info）
    index_dict = {
        "mnt_netname": defaultdict(list),  # (mnt, netname) → [p_ids]
        "mnt": defaultdict(list),  # mnt → [p_ids]
        "country": defaultdict(list)  # country → [p_ids]
    }

    for p_id, info in prefix_info.items():
        index_dict["mnt_netname"][(info["mnt_by"], info["netname"])].append(p_id)
        index_dict["mnt"][info["mnt_by"]].append(p_id)
        index_dict["country"][info["country"]].append(p_id)

    return prefix_info, index_dict


def sample_pos_neg_for_prefix(p_id: int, prefix_info: Dict[int, Dict], index_dict: Dict[str, defaultdict]) -> Tuple[
    List[int], int]:
    """
    对齐论文表4：为单个Prefix采样3个正例（1L1+1L2+1L3）+1个负例
    核心修复：使用独立的index_dict，避免访问prefix_info的非ID键
    Returns: [L1_p_id, L2_p_id, L3_p_id], neg_p_id
    """
    info = prefix_info[p_id]

    # Step 1: 采样L1正例（同Mnt+同Netname）
    l1_candidates = index_dict["mnt_netname"].get((info["mnt_by"], info["netname"]), [])
    l1_candidates = [x for x in l1_candidates if x != p_id]
    l1_p = random.choice(l1_candidates) if l1_candidates else p_id  # 无候选则用自身

    # Step 2: 采样L2正例（同Mnt+不同Netname）
    l2_candidates = index_dict["mnt"].get(info["mnt_by"], [])
    l2_candidates = [x for x in l2_candidates if x != p_id and prefix_info[x]["netname"] != info["netname"]]
    l2_p = random.choice(l2_candidates) if l2_candidates else p_id

    # Step 3: 采样L3正例（同Country）
    l3_candidates = index_dict["country"].get(info["country"], [])
    l3_candidates = [x for x in l3_candidates if x != p_id]
    l3_p = random.choice(l3_candidates) if l3_candidates else p_id

    # Step 4: 采样负例（不同Mnt+不同Country）
    all_prefixes = list(prefix_info.keys())  # 现在仅包含整数ID
    all_prefixes.remove(p_id)
    neg_candidates = [
        x for x in all_prefixes
        if prefix_info[x]["mnt_by"] != info["mnt_by"]
           and prefix_info[x]["country"] != info["country"]
    ]
    neg_p = random.choice(neg_candidates) if neg_candidates else random.choice(all_prefixes)

    return [l1_p, l2_p, l3_p], neg_p


# 初始化Prefix信息和索引字典（核心修复）
prefix_info, index_dict = build_prefix_info(g, mapping_df, ent2id, id2ent)


# ===================== 3. 6HAN模型（对齐Algorithm 2）=====================
class FeatureMappingLayer(nn.Module):
    """特征映射层：每类节点独立线性层到64维+ReLU（论文3.3.1）"""

    def __init__(self, in_feats_dict: Dict[str, int], embed_dim: int = 64):
        super().__init__()
        self.linear_layers = nn.ModuleDict()
        for ntype, in_dim in in_feats_dict.items():
            self.linear_layers[ntype] = nn.Linear(in_dim, embed_dim).to(DEVICE)
        self.relu = nn.ReLU().to(DEVICE)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = {}
        for ntype, feat in x.items():
            h[ntype] = self.relu(self.linear_layers[ntype](feat))
        return h


class NodeLevelAttention(nn.Module):
    """节点级注意力层（论文公式4-6）"""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(embed_dim, embed_dim).to(DEVICE)  # 64×64共享权重矩阵
        self.a = nn.Linear(2 * embed_dim, 1).to(DEVICE)  # 128×1注意力向量
        self.softmax = nn.Softmax(dim=1)

    def calc_attention_score(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        """公式4：计算注意力分数 e_ij"""
        h_src_proj = self.W(h_src)
        h_dst_proj = self.W(h_dst)
        concat_feat = torch.cat([h_src_proj, h_dst_proj], dim=-1)  # 拼接操作
        e_ij = self.a(concat_feat).squeeze(-1)  # 注意力分数
        return e_ij

    def forward(self, g: dgl.DGLHeteroGraph, h: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        三步计算：
        1. 注意力分数计算（公式4）
        2. Softmax归一化（公式5）
        3. 加权融合（公式6）
        """
        embed_dict = {}
        # 仅处理Prefix节点的嵌入（核心节点）
        prefix_embed = h["Prefix"]
        prefix_embed_new = torch.zeros_like(prefix_embed).to(DEVICE)

        # 遍历所有Prefix相关的边类型
        for etype in g.canonical_etypes:
            src_ntype, rel, dst_ntype = etype
            if src_ntype != "Prefix":
                continue  # 仅处理Prefix作为源节点的边

            # 获取该边类型的所有边
            u, v = g.edges(etype=etype)
            if len(u) == 0:
                continue

            # Step 1: 计算注意力分数 e_ij
            h_src = h[src_ntype][u]
            h_dst = h[dst_ntype][v]
            e_ij = self.calc_attention_score(h_src, h_dst)

            # Step 2: Softmax归一化（公式5）
            # 按源节点u分组归一化
            unique_u = torch.unique(u)
            for u_id in unique_u:
                mask = (u == u_id)
                if not mask.any():
                    continue
                e_ij_u = e_ij[mask]
                alpha_ij = self.softmax(e_ij_u.unsqueeze(0)).squeeze(0)

                # Step 3: 加权融合（公式6）
                h_dst_u = h_dst[mask]
                prefix_embed_new[u_id] += (alpha_ij.unsqueeze(-1) * h_dst_u).sum(dim=0)

        # 嵌入输出层：L2归一化（公式7）
        embed_dict["Prefix"] = F.normalize(prefix_embed_new, p=2, dim=-1)

        # 辅助节点嵌入直接返回（用于链路预测）
        for ntype in h:
            if ntype != "Prefix":
                embed_dict[ntype] = F.normalize(h[ntype], p=2, dim=-1)

        return embed_dict


class HAN6Model(nn.Module):
    """6HAN模型：特征映射层 + 节点级注意力层（移除语义级注意力）"""

    def __init__(self, g: dgl.DGLHeteroGraph, embed_dim: int = 64):
        super().__init__()
        # 1. 输入特征维度检测
        self.in_feats_dict = {}
        for ntype in g.ntypes:
            self.in_feats_dict[ntype] = g.nodes[ntype].data["feat"].shape[1]

        # 2. 特征映射层（论文3.3.1）
        self.feature_mapping = FeatureMappingLayer(self.in_feats_dict, embed_dim)

        # 3. 节点级注意力层（论文公式4-6）
        self.node_attention = NodeLevelAttention(embed_dim)

        self.g = g.to(DEVICE)
        self.embed_dim = embed_dim

    def forward(self) -> Dict[str, torch.Tensor]:
        # Step 1: 特征映射（统一到64维+ReLU）
        x = {ntype: self.g.nodes[ntype].data["feat"].to(DEVICE) for ntype in self.g.ntypes}
        h_mapped = self.feature_mapping(x)

        # Step 2: 节点级注意力聚合
        h_agg = self.node_attention(self.g, h_mapped)

        return h_agg


# 初始化6HAN模型（替换原HANModel）
def init_6han_model(g: dgl.DGLHeteroGraph, embed_dim: int = 64) -> HAN6Model:
    if not isinstance(g, dgl.DGLHeteroGraph):
        raise ValueError(f"需输入DGL异构图，当前类型：{type(g)}")

    model = HAN6Model(g, embed_dim)
    print(f"\n✅ 6HAN模型初始化完成（对齐论文Algorithm 2）")
    print(f" - 特征映射层：{list(model.feature_mapping.linear_layers.keys())}")
    print(f" - 嵌入维度：{embed_dim}")
    print(f" - 移除语义级注意力层（符合论文优化）")
    return model


model = init_6han_model(g, embed_dim=EMBED_DIM)


# ===================== 4. 损失函数（对齐Algorithm 3，修正采样参数）=====================
def info_nce_loss(anchor_embed: torch.Tensor, pos_embeds: torch.Tensor, neg_embed: torch.Tensor,
                  tau: float) -> torch.Tensor:
    """
    论文公式8：InfoNCE损失
    anchor_embed: [embed_dim] → 单个Prefix的嵌入
    pos_embeds: [3, embed_dim] → 3个正例嵌入（L1+L2+L3）
    neg_embed: [embed_dim] → 1个负例嵌入
    """
    # L2归一化（确保内积=余弦相似度）
    anchor = F.normalize(anchor_embed, p=2, dim=-1)
    pos = F.normalize(pos_embeds, p=2, dim=-1)
    neg = F.normalize(neg_embed, p=2, dim=-1)

    # 计算相似度（内积）
    pos_sim = torch.matmul(pos, anchor.unsqueeze(-1)).squeeze(-1) / tau  # [3]
    neg_sim = torch.matmul(neg.unsqueeze(0), anchor.unsqueeze(-1)).squeeze(-1) / tau  # [1]

    # InfoNCE损失计算
    numerator = torch.sum(torch.exp(pos_sim))
    denominator = numerator + torch.exp(neg_sim)
    loss = -torch.log(numerator / denominator + 1e-12)  # 避免log(0)
    return loss


def contrastive_loss_total(prefix_embed: torch.Tensor, prefix_info: Dict[int, Dict], index_dict: Dict[str, defaultdict],
                           tau: float) -> torch.Tensor:
    """
    对比损失总计算（对齐论文表4+公式8）
    核心修复：传入独立的index_dict，避免访问prefix_info的非ID键
    """
    total_contr_loss = 0.0
    prefix_ids = list(prefix_info.keys())  # 仅包含整数ID
    random.shuffle(prefix_ids)  # 随机遍历

    for p_id in prefix_ids:
        # 采样正/负例（传入index_dict）
        pos_p_ids, neg_p_id = sample_pos_neg_for_prefix(p_id, prefix_info, index_dict)

        # 获取嵌入
        anchor = prefix_embed[p_id]
        pos_embeds = prefix_embed[torch.tensor(pos_p_ids).to(DEVICE)]
        neg_embed = prefix_embed[neg_p_id]

        # 计算单个Prefix的InfoNCE损失
        total_contr_loss += info_nce_loss(anchor, pos_embeds, neg_embed, tau)

    # 平均损失
    return total_contr_loss / len(prefix_ids)


def generate_link_neg_edges(g: dgl.DGLHeteroGraph, etype: Tuple[str, str, str],
                            pos_edges: Tuple[torch.Tensor, torch.Tensor], id2ent: Dict[str, Dict[int, str]],
                            ent2id: Dict[str, Dict[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    论文链路预测负例：同类型尾实体替换（非随机生成）
    """
    src_ntype, _, dst_ntype = etype
    src_ids, dst_ids = pos_edges

    # 构建源节点→已连接尾节点的映射
    src2connected_dst = defaultdict(set)
    for s, d in zip(src_ids.cpu().numpy(), dst_ids.cpu().numpy()):
        src2connected_dst[s].add(d)

    # 所有尾节点ID
    all_dst_ids = list(range(g.num_nodes(dst_ntype)))

    # 为每个源节点生成未连接的尾节点
    neg_dst_ids = []
    for s in src_ids.cpu().numpy():
        connected = src2connected_dst[s]
        # 选择未连接的尾节点
        candidate_dst = [d for d in all_dst_ids if d not in connected]
        if not candidate_dst:
            candidate_dst = all_dst_ids  # 无候选则随机选
        neg_d = random.choice(candidate_dst)
        neg_dst_ids.append(neg_d)

    neg_dst_ids = torch.tensor(neg_dst_ids).to(DEVICE)
    return src_ids, neg_dst_ids


def link_pred_loss_total(h: Dict[str, torch.Tensor], g: dgl.DGLHeteroGraph, id2ent: Dict[str, Dict[int, str]],
                         ent2id: Dict[str, Dict[str, int]]) -> torch.Tensor:
    """
    论文链路预测损失（公式9+10）：
    1. 随机抽50%真实边作为正例
    2. 负例为同类型尾实体替换
    3. BCE损失计算
    """
    total_link_loss = 0.0
    valid_etypes = 0
    target_etypes = [("Prefix", "prefix_to_mnt", "Mnt"), ("Prefix", "prefix_to_netname", "Netname")]

    for etype in target_etypes:
        # Step 1: 抽取50%真实边作为正例
        all_pos_edges = g.edges(etype=etype, form="uv")
        if len(all_pos_edges[0]) == 0:
            continue
        num_pos = len(all_pos_edges[0])
        sample_idx = torch.randperm(num_pos)[:num_pos // 2]  # 随机抽50%
        pos_edges = (all_pos_edges[0][sample_idx], all_pos_edges[1][sample_idx])

        # Step 2: 生成负例边（同类型尾实体替换）
        neg_edges = generate_link_neg_edges(g, etype, pos_edges, id2ent, ent2id)

        # Step 3: 概率预测（公式9）
        src_ntype, _, dst_ntype = etype
        src_embed = h[src_ntype]
        dst_embed = h[dst_ntype]

        # 正例预测
        pos_src = src_embed[pos_edges[0]]
        pos_dst = dst_embed[pos_edges[1]]
        pos_pred = torch.sigmoid(torch.sum(pos_src * pos_dst, dim=-1))

        # 负例预测
        neg_src = src_embed[neg_edges[0]]
        neg_dst = dst_embed[neg_edges[1]]
        neg_pred = torch.sigmoid(torch.sum(neg_src * neg_dst, dim=-1))

        # Step 4: BCE损失（公式10）
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))

        total_link_loss += (pos_loss + neg_loss) / 2
        valid_etypes += 1

    return total_link_loss / valid_etypes if valid_etypes > 0 else torch.tensor(0.0).to(DEVICE)


def joint_loss_fn(h: Dict[str, torch.Tensor], g: dgl.DGLHeteroGraph, prefix_info: Dict[int, Dict],
                  index_dict: Dict[str, defaultdict], tau: float, id2ent: Dict[str, Dict[int, str]],
                  ent2id: Dict[str, Dict[str, int]]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    论文联合损失（Algorithm 3）：0.6×对比损失 + 0.4×链路预测损失
    核心修复：传入index_dict参数
    """
    # 1. 对比损失（传入index_dict）
    contr_loss = contrastive_loss_total(h["Prefix"], prefix_info, index_dict, tau)

    # 2. 链路预测损失
    link_loss = link_pred_loss_total(h, g, id2ent, ent2id)

    # 3. 联合损失（论文权重）
    total_loss = CONTRAST_WEIGHT * contr_loss + LINK_WEIGHT * link_loss

    # 损失记录
    loss_dict = {
        "total": total_loss.item(),
        "contrastive": contr_loss.item(),
        "link_pred": link_loss.item()
    }
    return total_loss, loss_dict


# ===================== 5. 模型训练（对齐Algorithm 3，修正参数传递）=====================
def train_6han_model(model: HAN6Model, g: dgl.DGLHeteroGraph, prefix_info: Dict[int, Dict],
                     index_dict: Dict[str, defaultdict], epochs: int, lr: float, weight_decay: float,
                     id2ent: Dict[str, Dict[int, str]], ent2id: Dict[str, Dict[str, int]]) -> HAN6Model:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm(range(epochs), desc="Training 6HAN Model (Paper Aligned)")

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # 前向传播
        h = model()

        # 计算联合损失（传入index_dict）
        total_loss, loss_dict = joint_loss_fn(h, g, prefix_info, index_dict, TAU, id2ent, ent2id)

        # 反向传播（NaN检查）
        if torch.isnan(total_loss):
            print(f"⚠️ Epoch {epoch + 1} 损失为NaN，跳过")
            continue
        total_loss.backward()
        optimizer.step()

        # 日志输出（每5轮）
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # 聚类评估（可选）
            prefix_embed = h["Prefix"].detach().cpu().numpy()
            p2m_edges = g.edges(etype=("Prefix", "prefix_to_mnt", "Mnt"), form="uv")
            p2m_dict = {p.item(): m.item() for p, m in zip(p2m_edges[0], p2m_edges[1])}
            prefix_ids = list(range(g.num_nodes("Prefix")))
            mnt_labels = [p2m_dict.get(p, -1) for p in prefix_ids]
            valid_idx = [i for i, lbl in enumerate(mnt_labels) if lbl != -1]

            if len(valid_idx) >= 2 and len(set([mnt_labels[i] for i in valid_idx])) >= 2:
                valid_embed = prefix_embed[valid_idx]
                valid_labels = [mnt_labels[i] for i in valid_idx]
                kmeans = KMeans(n_clusters=len(set(valid_labels)), random_state=42)
                cluster_pred = kmeans.fit_predict(valid_embed)
                silhouette = silhouette_score(valid_embed, cluster_pred)

                pbar.set_postfix({
                    "epoch": epoch + 1,
                    "total_loss": f"{loss_dict['total']:.4f}",
                    "contr_loss": f"{loss_dict['contrastive']:.4f}",
                    "link_loss": f"{loss_dict['link_pred']:.4f}",
                    "silhouette": f"{silhouette:.4f}"
                })
            else:
                pbar.set_postfix({
                    "epoch": epoch + 1,
                    "total_loss": f"{loss_dict['total']:.4f}",
                    "contr_loss": f"{loss_dict['contrastive']:.4f}",
                    "link_loss": f"{loss_dict['link_pred']:.4f}"
                })

    # 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ 6HAN模型保存至：{MODEL_SAVE_PATH}")
    return model


# ===================== 6. 执行训练 & 保存嵌入 =====================
if __name__ == "__main__":
    # 重新加载数据
    g, mapping_df, ent2id, id2ent = load_data()

    # 构建Prefix信息和独立索引字典（核心修复）
    prefix_info, index_dict = build_prefix_info(g, mapping_df, ent2id, id2ent)

    # 初始化模型
    model = init_6han_model(g, embed_dim=EMBED_DIM)

    # 训练模型（传入index_dict）
    trained_model = train_6han_model(
        model=model,
        g=g,
        prefix_info=prefix_info,
        index_dict=index_dict,  # 新增参数
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        id2ent=id2ent,
        ent2id=ent2id
    )

    # 保存嵌入
    trained_model.eval()
    with torch.no_grad():
        h = trained_model()
        prefix_embed = h["Prefix"].detach().cpu().numpy()

        # 处理NaN（可选）
        if np.isnan(prefix_embed).any():
            prefix_embed = np.nan_to_num(prefix_embed)

        # 保存numpy和CSV
        np.save(EMBEDDING_SAVE_PATH, prefix_embed)

        # 生成Prefix值到嵌入的映射
        prefix_id2val = id2ent["Prefix"]
        embed_data = []
        for p_id in range(len(prefix_embed)):
            embed_data.append({
                "prefix": prefix_id2val.get(p_id, f"unknown_{p_id}"),
                **{f"embed_{i}": prefix_embed[p_id][i] for i in range(EMBED_DIM)}
            })
        embed_df = pd.DataFrame(embed_data)
        embed_df.to_csv(EMBEDDING_SAVE_PATH.replace(".npy", ".csv"), index=False, encoding="utf-8")

        print(f"\n✅ 训练完成！")
        print(f"- 嵌入保存路径：{EMBEDDING_SAVE_PATH}（形状：{prefix_embed.shape}）")
        print(f"- CSV映射表：{EMBEDDING_SAVE_PATH.replace('.npy', '.csv')}")