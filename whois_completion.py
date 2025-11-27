import pandas as pd
import json
import re
import dgl
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Optional
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# 全局配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 64  # 论文指定64维嵌入
CONFIDENCE_THRESHOLD = 0.7  # 论文要求阈值0.7
MIN_CANDIDATE_FREQ = 3  # 候选实体最小频次
STOP_WORDS = set(stopwords.words('english'))

# ===================== 核心参数（必须和训练时完全一致） =====================
TRAIN_HIDDEN_DIM = 16
TRAIN_NUM_HEADS = 4


# ===================== 模型结构 =====================
class FeatureMappingLayer(torch.nn.Module):
    """特征映射层（核心层：匹配训练权重）"""

    def __init__(self, in_feats_dict: Dict[str, int], out_dim: int = 64):
        super().__init__()
        self.linear_layers = torch.nn.ModuleDict()
        for ntype, in_dim in in_feats_dict.items():
            self.linear_layers[ntype] = torch.nn.Linear(in_dim, out_dim)

    def forward(self, g: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
        h = {}
        for ntype in g.ntypes:
            if ntype in self.linear_layers and 'feat' in g.nodes[ntype].data:
                h[ntype] = F.relu(self.linear_layers[ntype](g.nodes[ntype].data['feat']))
            else:
                h[ntype] = torch.zeros((g.num_nodes(ntype), 64), device=DEVICE)
        return h


class NodeAttentionLayer(torch.nn.Module):
    """节点注意力层（适配异构图GATConv）"""

    def __init__(self, edge_types: List[str], in_dim: int = 64, hidden_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # 核心层：匹配训练权重
        self.W = torch.nn.Linear(in_dim, in_dim)
        self.a = torch.nn.Linear(2 * in_dim, 1)

        # GAT层：初始化
        self.gat_layers = torch.nn.ModuleDict()
        for etype in edge_types:
            self.gat_layers[etype] = dgl.nn.GATConv(
                in_feats=in_dim,
                out_feats=hidden_dim,
                num_heads=num_heads,
                allow_zero_in_degree=True
            )

    def forward(self, g: dgl.DGLHeteroGraph, h: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """异构图GATConv前向"""
        gat_out = {}

        # 遍历所有边类型处理GAT
        for stype, etype, dtype in g.canonical_etypes:
            if etype not in self.gat_layers:
                continue
            if stype not in h or dtype not in h:
                continue

            try:
                # 异构图GATConv输入：字典格式
                feat_src = {stype: h[stype]}
                feat_dst = {dtype: h[dtype]}
                out = self.gat_layers[etype](g[stype, etype, dtype], (feat_src, feat_dst))
                out_tensor = out[dtype].flatten(1)  # [N, num_heads*hidden_dim]
            except:
                # 兼容单节点类型输入
                out = self.gat_layers[etype](g[stype, etype, dtype], (h[stype], h[dtype]))
                out_tensor = out.flatten(1)

            # 聚合到目标节点类型
            if dtype not in gat_out:
                gat_out[dtype] = []
            gat_out[dtype].append(out_tensor)

        # 聚合所有边类型的输出
        h_out = {}
        for ntype in g.ntypes:
            if ntype in gat_out and len(gat_out[ntype]) > 0:
                h_out[ntype] = torch.stack(gat_out[ntype], dim=0).sum(dim=0)
            else:
                h_out[ntype] = torch.zeros((g.num_nodes(ntype), self.num_heads * self.hidden_dim), device=DEVICE)

        return h_out


class HANLinkPredModel(torch.nn.Module):
    """HAN模型（适配异构图）"""

    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 in_feats_dict: Dict[str, int]):
        super().__init__()
        self.g = g
        self.in_feats_dict = in_feats_dict

        # 1. 核心特征映射层
        self.feature_mapping = FeatureMappingLayer(in_feats_dict, out_dim=64)

        # 2. 核心节点注意力层
        edge_types = [e[1] for e in g.canonical_etypes]
        self.node_attention = NodeAttentionLayer(
            edge_types=edge_types,
            in_dim=64,
            hidden_dim=TRAIN_HIDDEN_DIM,
            num_heads=TRAIN_NUM_HEADS
        )

        # 3. 第二层GAT（适配异构图）
        self.gat2 = torch.nn.ModuleDict()
        for stype, etype, dtype in g.canonical_etypes:
            self.gat2[etype] = dgl.nn.GATConv(
                in_feats=64,
                out_feats=64,
                num_heads=1,
                allow_zero_in_degree=True
            )

    def forward(self, input_g: Optional[dgl.DGLHeteroGraph] = None) -> Dict[str, torch.Tensor]:
        """前向传播：仅生成嵌入"""
        g = input_g if input_g is not None else self.g

        # 1. 特征映射（输出：{节点类型: 64维特征}）
        h = self.feature_mapping(g)

        # 2. 节点注意力（输出：{节点类型: 64维特征}）
        h1 = self.node_attention(g, h)

        # 3. 第二层GAT
        h2 = {}
        for stype, etype, dtype in g.canonical_etypes:
            if etype not in self.gat2:
                continue
            if stype not in h1 or dtype not in h1:
                continue

            try:
                # 异构图输入格式
                feat_src = {stype: h1[stype][:, :64]}  # 确保64维输入
                feat_dst = {dtype: h1[dtype][:, :64]}
                out = self.gat2[etype](g[stype, etype, dtype], (feat_src, feat_dst))
                out_tensor = out[dtype].squeeze(1)
            except:
                out = self.gat2[etype](g[stype, etype, dtype], (h1[stype][:, :64], h1[dtype][:, :64]))
                out_tensor = out.squeeze(1)

            # 聚合到目标节点类型
            if dtype not in h2:
                h2[dtype] = []
            h2[dtype].append(out_tensor)

        # 最终特征聚合
        final_embeds = {}
        for ntype in g.ntypes:
            if ntype in h2 and len(h2[ntype]) > 0:
                final_embeds[ntype] = torch.stack(h2[ntype], dim=0).sum(dim=0)
            else:
                final_embeds[ntype] = h1[ntype][:, :64]  # 确保64维

            # L2归一化
            final_embeds[ntype] = F.normalize(final_embeds[ntype], p=2, dim=1)

        return final_embeds


# ===================== Whois补全核心类（修复节点-特征匹配） =====================
class WhoisLinkPredCompleter:
    def __init__(self,
                 source_csv: str,
                 merged_json: str,
                 entity_map: str,
                 model_path: str,
                 seed_graph_path: str,
                 output_path: str = "completed_whois.csv",
                 updated_graph_path: str = "updated_ipv6_graph.bin"):
        # 路径配置
        self.source_csv = source_csv
        self.merged_json = merged_json
        self.entity_map = entity_map
        self.model_path = model_path
        self.seed_graph_path = seed_graph_path
        self.output_path = output_path
        self.updated_graph_path = updated_graph_path

        # 数据存储
        self.source_df = None
        self.merged_df = None
        self.combined_df = None
        self.seed_graph = None
        self.temp_graph = None
        self.updated_graph = None

        # 实体映射
        self.ent_val_to_id = defaultdict(dict)
        self.ent_id_to_val = defaultdict(dict)
        self.next_ent_id = defaultdict(int)

        # 模型相关
        self.han_model = None
        self.seed_embeds = {}
        self.non_seed_embeds = {}
        self.candidate_pool = defaultdict(list)

        # 非种子图的节点ID映射（关键：记录所有节点ID）
        self.prefix_id_map = {}  # prefix_str -> node_id
        self.other_id_maps = {
            'Country': {}, 'Keyword': {}, 'Mnt': {},
            'Netname': {}, 'Status': {}
        }

    def load_seed_resources(self):
        """加载种子资源"""
        print("🔍 加载种子资源...")

        # 1. 加载实体映射表
        mapping_df = pd.read_csv(self.entity_map)
        for _, row in mapping_df.iterrows():
            etype = row['entity_type']
            eval = row['entity_value']
            eid = row['entity_id']
            self.ent_val_to_id[etype][eval] = eid
            self.ent_id_to_val[etype][eid] = eval
        # 初始化非种子实体ID
        for etype in self.ent_val_to_id:
            self.next_ent_id[etype] = max(self.ent_val_to_id[etype].values()) + 1 if self.ent_val_to_id[etype] else 0
        print(f"✅ 实体映射加载完成：{len(mapping_df)}个种子实体")

        # 2. 加载种子异构图
        graphs, _ = dgl.load_graphs(self.seed_graph_path)
        self.seed_graph = graphs[0].to(DEVICE)
        print(
            f"✅ 种子异构图加载完成：节点类型={self.seed_graph.ntypes}，边类型={[e[1] for e in self.seed_graph.canonical_etypes]}")

        # 3. 加载训练好的模型
        in_feats_dict = {ntype: self.seed_graph.nodes[ntype].data['feat'].shape[1] for ntype in self.seed_graph.ntypes}
        self.han_model = HANLinkPredModel(
            g=self.seed_graph,
            in_feats_dict=in_feats_dict
        ).to(DEVICE)

        # 非严格加载权重
        state_dict = torch.load(self.model_path, map_location=DEVICE)
        self.han_model.load_state_dict(state_dict, strict=False)

        # 固定所有参数
        for param in self.han_model.parameters():
            param.requires_grad = False
        self.han_model.eval()
        print(f"✅ 训练模型加载完成（核心层参数匹配）")

        # 4. 生成种子实体嵌入
        with torch.no_grad():
            self.seed_embeds = self.han_model()
        print(f"✅ 种子嵌入生成完成：{[f'{k}:{v.shape}' for k, v in self.seed_embeds.items()]}")

        # 5. 构建候选实体池
        self._build_candidate_pool()

    def _build_candidate_pool(self):
        """构建候选实体池（修复边类型遍历）"""
        print("🔍 构建候选实体池...")
        candidate_types = ['Mnt', 'Netname', 'Country', 'Status']

        for etype in candidate_types:
            # 找到所有指向该实体类型的边
            edge_types = [e for e in self.seed_graph.canonical_etypes if e[2] == etype]
            total_counts = Counter()

            for stype, edge_type, dtype in edge_types:
                if dtype != etype:
                    continue
                try:
                    # 获取边的目标节点ID
                    _, dst_ids = self.seed_graph.edges(etype=edge_type)
                    total_counts.update(dst_ids.cpu().numpy())
                except:
                    continue

            # 筛选高频实体
            for eid, freq in total_counts.most_common():
                if freq >= MIN_CANDIDATE_FREQ:
                    eval = self.ent_id_to_val[etype].get(eid)
                    if eval:
                        self.candidate_pool[etype].append((eval, eid))

            print(f"   - {etype}：{len(self.candidate_pool[etype])}个候选实体（频次≥{MIN_CANDIDATE_FREQ}）")

    def load_and_preprocess_data(self):
        """加载并预处理非种子数据"""
        print("\n📊 加载并预处理非种子数据...")

        # 1. 加载源数据
        self.source_df = pd.read_csv(self.source_csv)
        with open(self.merged_json, 'r', encoding='utf-8') as f:
            merged_dict = json.load(f)
        self.merged_df = pd.DataFrame.from_dict(merged_dict, orient='index').reset_index()
        self.merged_df.rename(columns={'index': 'inet6num'}, inplace=True)

        # 2. 数据合并
        self.combined_df = pd.merge(
            self.source_df,
            self.merged_df,
            on='inet6num',
            how='outer',
            suffixes=('', '_merged')
        )

        # 3. 标记种子/非种子前缀
        seed_prefixes = set(self.ent_val_to_id.get('Prefix', {}).keys())
        self.combined_df['is_seed'] = self.combined_df['inet6num'].isin(seed_prefixes)
        self.non_seed_df = self.combined_df[~self.combined_df['is_seed']].copy().reset_index(drop=True)

        # 过滤空的prefix
        self.non_seed_df = self.non_seed_df[~self.non_seed_df['inet6num'].isna()].reset_index(drop=True)

        print(f"✅ 数据预处理完成：总记录={len(self.combined_df)}，非种子={len(self.non_seed_df)}")

        # 4. 基础预处理
        self._basic_preprocess()

    def _basic_preprocess(self):
        """基础预处理"""
        # 标准化字段格式
        str_fields = ['netname', 'country', 'mnt-by', 'status', 'descr']
        for field in str_fields:
            if field in self.combined_df.columns:
                self.combined_df[field] = self.combined_df[field].astype(str).str.strip().str.upper()
                self.combined_df.loc[self.combined_df[field].isin(['NAN', 'NaN', 'nan']), field] = np.nan

        # 派生descr_cleaned
        if 'descr_cleaned' not in self.combined_df.columns:
            self.combined_df['descr_cleaned'] = np.nan

        mask = self.combined_df['descr_cleaned'].isna() & ~self.combined_df['descr'].isna()
        if mask.sum() > 0:
            def clean_descr(s):
                tokens = re.split(r'[\s-]+', s.lower())
                return ' '.join([t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) >= 3]).upper()

            self.combined_df.loc[mask, 'descr_cleaned'] = self.combined_df.loc[mask, 'descr'].apply(clean_descr)

        # 标记待补全字段
        self.combined_df['to_complete'] = ''
        total_missing = 0
        for field in ['netname', 'country', 'mnt-by', 'status']:
            if field in self.combined_df.columns:
                mask = self.combined_df[field].isna() & ~self.combined_df['is_seed']
                self.combined_df.loc[mask, 'to_complete'] += f'{field},'
                total_missing += mask.sum()

        print(f"✅ 待补全字段标记完成：{total_missing}条记录需补全")

    def build_non_seed_graph(self):
        """构建非种子临时异构图（核心修复：节点-特征匹配）"""
        print("\n🌐 构建非种子临时异构图...")

        # 重置ID映射
        self.prefix_id_map.clear()
        for ntype in self.other_id_maps:
            self.other_id_maps[ntype].clear()

        # 1. 第一步：分配所有节点ID（确保ID连续）
        print("   - 分配非种子节点ID...")
        for idx, row in self.non_seed_df.iterrows():
            prefix = row['inet6num']
            if pd.isna(prefix):
                continue

            # 分配Prefix ID
            if prefix not in self.prefix_id_map:
                self.prefix_id_map[prefix] = len(self.prefix_id_map)

            # 分配关联实体ID（仅分配ID，暂不添加边）
            self._assign_other_entity_ids(row)

        # 2. 第二步：收集所有边
        print("   - 收集边信息...")
        graph_data = defaultdict(list)
        for idx, row in self.non_seed_df.iterrows():
            prefix = row['inet6num']
            if pd.isna(prefix) or prefix not in self.prefix_id_map:
                continue

            prefix_id = self.prefix_id_map[prefix]
            self._add_edges_for_prefix(prefix_id, row, graph_data)

        # 3. 第三步：构建异构图（先指定节点数量，再添加边）
        print("   - 创建异构图...")
        # 定义节点数量
        num_nodes_dict = {
            'Prefix': len(self.prefix_id_map),
            'Country': len(self.other_id_maps['Country']),
            'Keyword': len(self.other_id_maps['Keyword']),
            'Mnt': len(self.other_id_maps['Mnt']),
            'Netname': len(self.other_id_maps['Netname']),
            'Status': len(self.other_id_maps['Status'])
        }

        # 过滤空边
        filtered_graph_data = {}
        for (stype, etype, dtype), edges in graph_data.items():
            if edges:
                src_ids = torch.tensor([e[0] for e in edges], dtype=torch.long, device=DEVICE)
                dst_ids = torch.tensor([e[1] for e in edges], dtype=torch.long, device=DEVICE)
                filtered_graph_data[(stype, etype, dtype)] = (src_ids, dst_ids)

        # 创建异构图（指定节点数量）
        self.temp_graph = dgl.heterograph(filtered_graph_data, num_nodes_dict=num_nodes_dict, device=DEVICE)

        # 4. 第四步：添加节点特征（关键：维度严格匹配节点数）
        print("   - 添加节点特征...")
        self._add_node_features_to_temp_graph()

        print(f"✅ 非种子临时异构图构建完成：")
        print(f"   - 节点类型：{self.temp_graph.ntypes}")
        print(f"   - 边类型：{[e[1] for e in self.temp_graph.canonical_etypes]}")
        print(f"   - Prefix节点数：{self.temp_graph.num_nodes('Prefix')}")

    def _assign_other_entity_ids(self, row):
        """只为实体分配ID（确保ID连续）"""
        ent_mapping = {
            'mnt-by': 'Mnt',
            'netname': 'Netname',
            'country': 'Country',
            'status': 'Status',
            'descr_cleaned': 'Keyword'
        }

        for field, ntype in ent_mapping.items():
            if field not in row:
                continue
            val = row[field]
            if pd.isna(val) or val in ['NAN', 'NaN', 'nan', '']:
                continue

            if val not in self.other_id_maps[ntype]:
                self.other_id_maps[ntype][val] = len(self.other_id_maps[ntype])

    def _add_edges_for_prefix(self, prefix_id, row, graph_data):
        """为单个Prefix添加边"""
        ent_mapping = {
            'mnt-by': ('Mnt', 'prefix_to_mnt'),
            'netname': ('Netname', 'prefix_to_netname'),
            'country': ('Country', 'prefix_to_country'),
            'status': ('Status', 'prefix_to_status'),
            'descr_cleaned': ('Keyword', 'prefix_to_keyword')
        }

        for field, (ntype, edge_type) in ent_mapping.items():
            if field not in row:
                continue
            val = row[field]
            if pd.isna(val) or val in ['NAN', 'NaN', 'nan', '']:
                continue

            if val in self.other_id_maps[ntype]:
                entity_id = self.other_id_maps[ntype][val]
                graph_data[('Prefix', edge_type, ntype)].append((prefix_id, entity_id))

    def _add_node_features_to_temp_graph(self):
        """添加节点特征（确保维度匹配）"""
        # 1. 添加Prefix特征（核心修复：维度严格匹配）
        if self.temp_graph.num_nodes('Prefix') > 0:
            prefix_feats = torch.zeros((self.temp_graph.num_nodes('Prefix'), 33), device=DEVICE)

            # 遍历所有Prefix节点，填充特征
            for prefix, pid in self.prefix_id_map.items():
                if pid >= self.temp_graph.num_nodes('Prefix'):
                    continue  # 跳过超出范围的ID
                # 找到对应的行
                row_mask = self.non_seed_df['inet6num'] == prefix
                if row_mask.sum() > 0:
                    row = self.non_seed_df[row_mask].iloc[0]
                    prefix_feats[pid] = self._generate_prefix_feat(prefix, row)

            self.temp_graph.nodes['Prefix'].data['feat'] = prefix_feats

        # 2. 添加其他节点特征
        for ntype in ['Country', 'Mnt', 'Netname', 'Status', 'Keyword']:
            num_nodes = self.temp_graph.num_nodes(ntype)
            if num_nodes == 0:
                continue

            feats = torch.zeros((num_nodes, 2), device=DEVICE)
            for val, eid in self.other_id_maps[ntype].items():
                if eid >= num_nodes:
                    continue

                # 复用种子特征
                if val in self.ent_val_to_id.get(ntype, {}):
                    seed_eid = self.ent_val_to_id[ntype][val]
                    if seed_eid < self.seed_graph.num_nodes(ntype):
                        feats[eid] = self.seed_graph.nodes[ntype].data['feat'][seed_eid]
                else:
                    feats[eid] = torch.randn(2, device=DEVICE)

            self.temp_graph.nodes[ntype].data['feat'] = feats

    def _generate_prefix_feat(self, prefix: str, row: pd.Series) -> torch.Tensor:
        """生成33维Prefix特征"""
        feat = []

        # 1. 前缀长度（1维）
        try:
            plen = int(prefix.split('/')[-1]) if '/' in str(prefix) else 0
            feat.append(torch.tensor([plen / 128.0], dtype=torch.float32, device=DEVICE))
        except:
            feat.append(torch.tensor([0.0], dtype=torch.float32, device=DEVICE))

        # 2. IPv6版本标识（1维）
        feat.append(torch.tensor([1.0], dtype=torch.float32, device=DEVICE))

        # 3. 网络类型独热编码（8维）
        net_type = torch.zeros(8, dtype=torch.float32, device=DEVICE)
        try:
            plen = int(prefix.split('/')[-1]) if '/' in str(prefix) else 0
            if plen <= 12:
                net_type[0] = 1.0
            elif plen <= 24:
                net_type[1] = 1.0
            elif plen <= 32:
                net_type[2] = 1.0
            elif plen <= 48:
                net_type[3] = 1.0
            elif plen <= 64:
                net_type[4] = 1.0
            elif plen <= 80:
                net_type[5] = 1.0
            elif plen <= 96:
                net_type[6] = 1.0
            else:
                net_type[7] = 1.0
        except:
            net_type[7] = 1.0
        feat.append(net_type)

        # 4. 国家独热编码（10维）
        top_countries = ['US', 'CN', 'JP', 'DE', 'UK', 'FR', 'KR', 'CA', 'AU', 'IN']
        country_onehot = torch.zeros(10, dtype=torch.float32, device=DEVICE)
        if not pd.isna(row.get('country')) and row['country'] in top_countries:
            country_onehot[top_countries.index(row['country'])] = 1.0
        feat.append(country_onehot)

        # 5. 关键字词袋（10维）
        top_keywords = ['INTERNET', 'SERVICE', 'PROVIDER', 'NETWORK', 'COMMUNICATION',
                        'TECHNOLOGY', 'CORPORATION', 'ORGANIZATION', 'GOVERNMENT', 'EDUCATION']
        keyword_bow = torch.zeros(10, dtype=torch.float32, device=DEVICE)
        if not pd.isna(row.get('descr_cleaned')):
            desc = row['descr_cleaned']
            for i, kw in enumerate(top_keywords):
                if kw in desc:
                    keyword_bow[i] = 1.0
        feat.append(keyword_bow)

        # 6. 状态独热编码（3维）
        status_onehot = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        if not pd.isna(row.get('status')):
            status = row['status']
            if 'ALLOCATED' in status:
                status_onehot[0] = 1.0
            elif 'ASSIGNED' in status:
                status_onehot[1] = 1.0
            elif 'RESERVED' in status:
                status_onehot[2] = 1.0
        feat.append(status_onehot)

        return torch.cat(feat, dim=0)

    def predict_missing_edges(self):
        """链路预测补全缺失实体"""
        print("\n🎯 链路预测补全缺失实体（阈值=0.7）...")

        # 检查临时图是否为空
        if self.temp_graph is None or self.temp_graph.num_nodes('Prefix') == 0:
            print("⚠️ 非种子图为空，跳过补全")
            return

        # 生成非种子Prefix嵌入
        with torch.no_grad():
            self.non_seed_embeds = self.han_model(input_g=self.temp_graph)

        # 检查Prefix嵌入
        if 'Prefix' not in self.non_seed_embeds or len(self.non_seed_embeds['Prefix']) == 0:
            print("⚠️ 无Prefix嵌入，跳过补全")
            return

        prefix_embeds = self.non_seed_embeds['Prefix']
        print(f"✅ 非种子Prefix嵌入生成完成：{prefix_embeds.shape}（64维）")

        # 待补全字段映射
        complete_mapping = {
            'mnt-by': ('Mnt', 'prefix_to_mnt'),
            'netname': ('Netname', 'prefix_to_netname'),
            'country': ('Country', 'prefix_to_country'),
            'status': ('Status', 'prefix_to_status')
        }

        # 逐个字段补全
        total_completed = 0
        for field, (etype, edge_type) in complete_mapping.items():
            if field not in self.combined_df.columns:
                continue
            # 跳过无候选实体的字段
            if etype not in self.candidate_pool or len(self.candidate_pool[etype]) == 0:
                print(f"   - {field}：0个候选实体，跳过")
                continue
            completed = self._complete_single_field(field, etype, edge_type, prefix_embeds)
            total_completed += completed
            print(f"   - {field}：补全{completed}条（阈值=0.7）")

        print(f"✅ 链路预测补全完成：总计补全{total_completed}条")

    def _complete_single_field(self, field: str, etype: str, edge_type: str, prefix_embeds: torch.Tensor) -> int:
        """补全单个字段"""
        # 筛选待补全记录
        mask = self.combined_df[field].isna() & ~self.combined_df['is_seed']
        if not mask.any():
            return 0

        # 候选实体
        candidate_vals = [c[0] for c in self.candidate_pool[etype]]
        candidate_ids = [c[1] for c in self.candidate_pool[etype]]

        # 检查种子嵌入
        if etype not in self.seed_embeds or len(candidate_ids) == 0:
            return 0

        candidate_embeds = self.seed_embeds[etype][candidate_ids].to(DEVICE)
        completed = 0

        # 遍历待补全记录
        for idx in self.combined_df[mask].index:
            row = self.combined_df.iloc[idx]
            prefix = row['inet6num']

            # 获取Prefix ID
            if prefix not in self.prefix_id_map:
                continue
            prefix_id = self.prefix_id_map[prefix]
            if prefix_id >= len(prefix_embeds):
                continue

            # 计算关联概率（公式9：σ(h_u·h_v)）
            p_embed = prefix_embeds[prefix_id:prefix_id + 1]  # [1, 64]
            dot_products = torch.matmul(p_embed, candidate_embeds.T).squeeze(0)
            probs = torch.sigmoid(dot_products).cpu().numpy()

            # 选择最优候选
            max_idx = np.argmax(probs)
            max_prob = probs[max_idx]

            if max_prob > CONFIDENCE_THRESHOLD:
                best_candidate = candidate_vals[max_idx]
                # 一致性校验
                if self._consistency_check(row, field, best_candidate):
                    self.combined_df.at[idx, field] = best_candidate
                    self.combined_df.at[idx, f'{field}_confidence'] = float(max_prob)
                    completed += 1

        return completed

    def _consistency_check(self, row: pd.Series, field: str, candidate: str) -> bool:
        """一致性校验"""
        # Mnt与Country冲突校验
        if field == 'mnt-by':
            country = row.get('country')
            if not pd.isna(country) and country != '':
                country_code = re.findall(r'^[A-Z]{2}', candidate)
                if country_code and country_code[0] != country:
                    return False

        # Status合理性校验
        if field == 'status':
            valid_status = ['ALLOCATED', 'ASSIGNED', 'RESERVED']
            if not any(vs in candidate for vs in valid_status):
                return False

        return True

    def update_hetero_graph(self):
        """更新异构图"""
        print("\n🔄 更新非种子异构图...")
        if self.temp_graph is None:
            print("⚠️ 无临时图，跳过更新")
            return

        try:
            # 合并种子图和非种子图
            self.updated_graph = dgl.merge([self.seed_graph, self.temp_graph])
            dgl.save_graphs(self.updated_graph_path, [self.updated_graph])
            print(f"✅ 异构图更新完成，保存至：{self.updated_graph_path}")
        except Exception as e:
            print(f"⚠️ 合并图失败：{e}")

    def save_results(self):
        """保存补全结果"""
        # 统计补全效果
        stats = []
        for field in ['netname', 'country', 'mnt-by', 'status']:
            if field not in self.combined_df.columns:
                continue
            total = len(self.non_seed_df)
            missing_before = self.non_seed_df[field].isna().sum()
            missing_after = self.combined_df[~self.combined_df['is_seed']][field].isna().sum()
            completed = missing_before - missing_after
            completion_rate = (completed / total) * 100 if total > 0 else 0
            stats.append({
                '字段': field,
                '非种子总数': total,
                '补全数': completed,
                '补全率': f'{completion_rate:.1f}%'
            })

        # 打印统计
        print("\n📈 补全效果统计：")
        print(pd.DataFrame(stats).to_string(index=False))

        # 保存结果
        try:
            self.combined_df.to_csv(self.output_path, index=False, encoding='utf-8')
            print(f"✅ 补全结果已保存至：{self.output_path}")
        except Exception as e:
            print(f"⚠️ 保存结果失败：{e}")

    def run(self):
        """执行完整流程"""
        print("=" * 80)
        print("6HAN Whois信息补全流程（基于链路预测）")
        print("=" * 80)

        try:
            self.load_seed_resources()
            self.load_and_preprocess_data()
            self.build_non_seed_graph()
            self.predict_missing_edges()
            self.update_hetero_graph()
            self.save_results()

            print("\n🎉 补全流程全部完成！")
        except Exception as e:
            import traceback
            traceback.print_exc()
            # 保存部分结果
            if self.combined_df is not None:
                self.combined_df.to_csv(f"partial_{self.output_path}", index=False, encoding='utf-8')
                print(f"⚠️ 流程中断，已保存部分结果至：partial_{self.output_path}")
            raise RuntimeError(f"补全流程失败：{str(e)}")


# ===================== 执行入口 =====================
if __name__ == "__main__":
    # 配置路径（根据实际修改）
    config = {
        "source_csv": "Data/parsed_whois.csv",
        "merged_json": "Data/merged_whois.json",
        "entity_map": "Data/entity_mapping.csv",
        "model_path": "Data/6han_model.pth",
        "seed_graph_path": "Data/ipv6_hetero_graph.bin",
        "output_path": "completed_whois.csv",
        "updated_graph_path": "updated_ipv6_graph.bin"
    }

    # 执行补全
    completer = WhoisLinkPredCompleter(**config)
    completer.run()