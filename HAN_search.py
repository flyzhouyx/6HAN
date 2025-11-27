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


# 导入训练阶段定义的HAN模型（必须与训练代码中的模型定义完全一致）
class HANModel(torch.nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 in_feats_dict: Dict[str, int],
                 embed_dim: int,
                 num_heads: int,
                 hidden_dim: int = 32):
        super().__init__()
        self.g = g
        self.node_types = g.ntypes
        self.canonical_etypes = g.canonical_etypes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # 第一层GAT + HeteroGraphConv
        self.gat1_conv_dict = {}
        for src_ntype, etype, dst_ntype in self.canonical_etypes:
            self.gat1_conv_dict[etype] = dgl.nn.GATConv(
                in_feats=(in_feats_dict[src_ntype], in_feats_dict[dst_ntype]),
                out_feats=self.hidden_dim,
                num_heads=self.num_heads,
                allow_zero_in_degree=True
            )
        self.hetero_conv1 = dgl.nn.HeteroGraphConv(
            mods=self.gat1_conv_dict,
            aggregate="sum"
        )

        # 第二层GAT + HeteroGraphConv
        self.gat2_conv_dict = {}
        for src_ntype, etype, dst_ntype in self.canonical_etypes:
            in_feat_src = self.hidden_dim * self.num_heads
            in_feat_dst = self.hidden_dim * self.num_heads
            self.gat2_conv_dict[etype] = dgl.nn.GATConv(
                in_feats=(in_feat_src, in_feat_dst),
                out_feats=self.embed_dim,
                num_heads=1,
                allow_zero_in_degree=True
            )
        self.hetero_conv2 = dgl.nn.HeteroGraphConv(
            mods=self.gat2_conv_dict,
            aggregate="sum"
        )

        self.semantic_attn1 = torch.nn.Linear(self.hidden_dim * self.num_heads, 1)
        self.semantic_attn2 = torch.nn.Linear(self.embed_dim, 1)
        self.norm1 = torch.nn.LayerNorm(self.hidden_dim * self.num_heads)
        self.norm2 = torch.nn.LayerNorm(self.embed_dim)
        self.relu = torch.nn.ReLU()

    def _semantic_attention(self, feat_dict: Dict[str, torch.Tensor], attn_layer: torch.nn.Linear) -> Dict[
        str, torch.Tensor]:
        etype_weights = {}
        for src_ntype, etype, dst_ntype in self.canonical_etypes:
            if src_ntype in feat_dict:
                avg_feat = torch.mean(feat_dict[src_ntype], dim=0, keepdim=True)
                weight = attn_layer(avg_feat).squeeze()
                etype_weights[etype] = weight

        weighted_feat = {}
        for ntype in feat_dict:
            related_weights = []
            for src_ntype, etype, dst_ntype in self.canonical_etypes:
                if src_ntype == ntype or dst_ntype == ntype:
                    related_weights.append(etype_weights[etype])
            if related_weights:
                weights = F.softmax(torch.stack(related_weights), dim=0)
                weighted_feat[ntype] = self.relu(feat_dict[ntype] * weights.mean())
            else:
                weighted_feat[ntype] = self.relu(feat_dict[ntype])
        return weighted_feat

    def forward(self) -> Dict[str, torch.Tensor]:
        x = {
            ntype: self.g.nodes[ntype].data["feat"]
            for ntype in self.node_types
        }

        # 第一层计算
        h1 = self.hetero_conv1(self.g, x)
        h1 = {ntype: h.flatten(1) for ntype, h in h1.items()}
        h1 = self._semantic_attention(h1, self.semantic_attn1)
        h1 = {ntype: self.norm1(feat) for ntype, feat in h1.items()}

        # 第二层计算
        h2 = self.hetero_conv2(self.g, h1)
        h2 = {ntype: h.squeeze(1) for ntype, h in h2.items()}
        h2 = self._semantic_attention(h2, self.semantic_attn2)
        h2 = {ntype: self.norm2(feat) for ntype, feat in h2.items()}

        return h2


# 下载NLTK停用词资源（首次运行需取消注释执行）
# nltk.download('stopwords')

# ===================== 核心配置参数 =====================
# 数据路径
SOURCE_CSV_PATH = "Data/parsed_whois.csv"  # 源Whois数据（含inet6num、netname等字段）
MERGED_JSON_PATH = "Data/merged_whois.json"  # 补充Whois数据（JSON格式）
GRAPH_PATH = "Data/ipv6_hetero_graph.bin"  # 种子前缀异构图（BIN格式）
ENTITY_MAP_PATH = "Data/entity_mapping.csv"  # 实体ID映射表（种子数据）
MODEL_WEIGHT_PATH = "Data/han_model.pth"  # 训练好的HAN模型权重
EMBEDDING_PATH = "Data/prefix_embeddings.npy"  # 种子前缀嵌入（可选，用于候选实体筛选）
OUTPUT_PATH = "completed_whois.csv"  # 补全结果输出路径
UPDATED_GRAPH_PATH = "Data/ipv6_hetero_graph_updated.bin"  # 更新后的异构图（含非种子前缀）

# 模型与补全参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 64  # 必须与训练时一致
NUM_HEADS = 4  # 必须与训练时一致
CONFIDENCE_THRESHOLD = 0.7  # 论文要求的补全置信度阈值（≥0.7才填充）
MIN_CANDIDATE_FREQ = 3  # 候选实体最小出现频次（过滤稀有实体，提高准确性）
STOP_WORDS = set(stopwords.words('english'))  # 英文停用词（用于descr_cleaned派生）

# 特征工程配置（与种子前缀特征完全对齐，确保33维）
PREFIX_FEAT_CONFIG = {
    "prefix_length": 1,  # 前缀长度（如/48→48）
    "ipv6_version": 1,  # IPv6版本标识（固定为1）
    "net_type_onehot": 8,  # 网络类型独热编码（8类）
    "country_onehot": 10,  # 国家独热编码（10类，种子数据中高频国家）
    "keyword_bow": 10,  # 关键字词袋特征（10维）
    "status_onehot": 3  # 状态独热编码（3类）
}
assert sum(PREFIX_FEAT_CONFIG.values()) == 33, "Prefix特征维度必须为33维（与种子前缀一致）"


# ===================== 补全核心类 =====================
class WhoisCompleter:
    def __init__(self):
        # 数据存储结构
        self.source_df = None  # 源CSV数据（含种子+非种子前缀）
        self.merged_df = None  # merged JSON数据
        self.combined_df = None  # 合并后的数据
        self.seed_graph = None  # 种子前缀异构图
        self.temp_graph = None  # 非种子前缀临时异构图
        self.updated_graph = None  # 更新后的完整异构图
        self.parent_mapping = {}  # 前缀-父前缀映射

        # 实体映射表（双向映射：种子+非种子）
        self.ent_val_to_id = defaultdict(dict)  # {实体类型: {实体值: 原始ID}}
        self.ent_id_to_val = defaultdict(dict)  # {实体类型: {原始ID: 实体值}}
        self.next_ent_ids = defaultdict(int)  # 非种子实体的下一个可用原始ID

        # 新增：连续ID映射（解决DGL节点ID不连续问题）
        self.raw_to_continuous_id = defaultdict(dict)  # {实体类型: {原始ID: 连续ID}}
        self.continuous_to_raw_id = defaultdict(dict)  # {实体类型: {连续ID: 原始ID}}

        # 模型相关：修复初始化问题，改为嵌套defaultdict
        self.han_model = None  # 加载的训练好的HAN模型
        self.seed_embeddings = defaultdict(dict)  # {实体类型: {实体值: 嵌入向量}} 嵌套defaultdict
        self.non_seed_embeddings = {}  # 非种子实体嵌入

        # 候选实体池（按实体类型分组）
        self.candidate_pool = defaultdict(list)  # {实体类型: [(实体值, 出现频次)]}

    def load_entity_mapping(self) -> None:
        """加载种子实体映射表，并初始化非种子实体ID计数器"""
        print("🔍 加载实体映射表...")
        try:
            mapping_df = pd.read_csv(ENTITY_MAP_PATH)
            # 构建双向映射（种子实体）
            for _, row in mapping_df.iterrows():
                ent_type = row["entity_type"]
                ent_val = row["entity_value"]
                ent_id = row["entity_id"]
                self.ent_val_to_id[ent_type][ent_val] = ent_id
                self.ent_id_to_val[ent_type][ent_id] = ent_val
            # 初始化非种子实体ID（从种子最大ID+1开始）
            for ent_type in self.ent_val_to_id:
                if self.ent_val_to_id[ent_type]:
                    self.next_ent_ids[ent_type] = max(self.ent_val_to_id[ent_type].values()) + 1
                else:
                    self.next_ent_ids[ent_type] = 0
            print(f"✅ 实体映射表加载成功：共{len(mapping_df)}个种子实体，涵盖{list(self.ent_val_to_id.keys())}类型")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 实体映射表不存在：{ENTITY_MAP_PATH}，请先运行buildGraph_6.py生成")
        except Exception as e:
            raise RuntimeError(f"❌ 实体映射表加载失败：{e}")

    def load_seed_embeddings(self) -> None:
        """加载种子实体嵌入（用于候选实体筛选和相似度计算）"""
        print("🔍 加载种子实体嵌入...")
        try:
            import os
            # 加载Prefix嵌入（修复KeyError：先检查嵌入文件是否存在且有数据）
            if os.path.exists(EMBEDDING_PATH):
                prefix_embed = np.load(EMBEDDING_PATH)
                # 检查种子Prefix数量与嵌入维度是否匹配
                seed_prefix_count = len(self.ent_val_to_id.get("Prefix", {}))
                if len(prefix_embed) >= seed_prefix_count and seed_prefix_count > 0:
                    for p_val, p_id in self.ent_val_to_id["Prefix"].items():
                        if p_id < len(prefix_embed):
                            self.seed_embeddings["Prefix"][p_val] = torch.tensor(prefix_embed[p_id], device=DEVICE)
                    print(f"   - Prefix嵌入加载成功：{len(self.seed_embeddings['Prefix'])}个种子Prefix")
                else:
                    print(
                        f"⚠️ 种子Prefix嵌入不完整：嵌入文件长度{len(prefix_embed)} < 种子Prefix数量{seed_prefix_count}")
            else:
                print(f"⚠️ 种子Prefix嵌入文件不存在：{EMBEDDING_PATH}")

            # 加载其他实体嵌入（从模型中提取）
            if self.han_model is not None:
                self.han_model.eval()
                with torch.no_grad():
                    seed_embeds = self.han_model()
                    for ent_type in ["Mnt", "Netname", "Country", "Status", "Keyword"]:
                        if ent_type in seed_embeds and ent_type in self.ent_val_to_id:
                            embed = seed_embeds[ent_type].detach().cpu().numpy()
                            ent_count = 0
                            for e_val, e_id in self.ent_val_to_id[ent_type].items():
                                if e_id < len(embed):
                                    self.seed_embeddings[ent_type][e_val] = torch.tensor(embed[e_id], device=DEVICE)
                                    ent_count += 1
                            print(f"   - {ent_type}嵌入加载成功：{ent_count}个种子实体")
            print(f"✅ 种子实体嵌入加载完成：涵盖{[k for k, v in self.seed_embeddings.items() if v]}类型")
        except Exception as e:
            print(f"⚠️ 种子实体嵌入加载警告：{e}（将使用模型实时生成候选实体嵌入）")

    def load_han_model(self) -> None:
        """加载训练好的HAN模型，固定所有参数（仅前向传播）"""
        print("🔍 加载训练好的HAN模型...")
        try:
            # 先加载种子图获取输入特征维度
            graphs, _ = dgl.load_graphs(GRAPH_PATH)
            self.seed_graph = graphs[0].to(DEVICE)
            # 构建输入特征维度字典
            in_feats_dict = {}
            for ntype in self.seed_graph.ntypes:
                in_feats_dict[ntype] = self.seed_graph.nodes[ntype].data["feat"].shape[1]
            # 初始化模型并加载权重
            self.han_model = HANModel(
                g=self.seed_graph,
                in_feats_dict=in_feats_dict,
                embed_dim=EMBED_DIM,
                num_heads=NUM_HEADS
            ).to(DEVICE)
            self.han_model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE))
            # 固定所有参数（禁止微调）
            for param in self.han_model.parameters():
                param.requires_grad = False
            self.han_model.eval()
            print(f"✅ HAN模型加载成功：权重路径={MODEL_WEIGHT_PATH}，参数已固定")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 模型权重不存在：{MODEL_WEIGHT_PATH}，请先运行HAN_train.py训练")
        except Exception as e:
            raise RuntimeError(f"❌ 模型加载失败：{e}")

    def load_data(self) -> None:
        """加载所有数据源（源CSV、merged JSON、种子图、模型）"""
        print("\n===== 加载数据源 =====")

        # 1. 加载源CSV数据（区分种子/非种子前缀：种子前缀在种子图中）
        print("1. 加载源Whois数据...")
        try:
            self.source_df = pd.read_csv(SOURCE_CSV_PATH)
            required_fields = ["inet6num", "original_inet6num"]
            missing_fields = [f for f in required_fields if f not in self.source_df.columns]
            if missing_fields:
                raise ValueError(f"源数据缺失关键字段：{missing_fields}")
            # 标记种子/非种子前缀（修复：处理种子Prefix为空的情况）
            seed_prefixes = set(self.ent_val_to_id.get("Prefix", {}).keys())
            self.source_df["is_seed"] = self.source_df["inet6num"].isin(seed_prefixes)
            seed_count = self.source_df["is_seed"].sum()
            non_seed_count = (~self.source_df["is_seed"]).sum()
            print(f"✅ 源数据加载成功：共{len(self.source_df)}条记录（种子{seed_count}条，非种子{non_seed_count}条）")
            # 警告：种子Prefix为空可能影响补全效果
            if seed_count == 0:
                print("⚠️ 警告：源数据中无种子前缀（inet6num未匹配到实体映射表中的Prefix）")
                print("   请检查：1. entity_mapping.csv是否包含种子Prefix；2. parsed_whois.csv的inet6num格式是否正确")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 源数据不存在：{SOURCE_CSV_PATH}")
        except Exception as e:
            raise RuntimeError(f"❌ 源数据加载失败：{e}")

        # 2. 加载补充Whois数据
        print("2. 加载补充Whois数据...")
        try:
            with open(MERGED_JSON_PATH, "r", encoding="utf-8") as f:
                merged_dict = json.load(f)
            self.merged_df = pd.DataFrame.from_dict(merged_dict, orient="index").reset_index()
            self.merged_df.rename(columns={"index": "inet6num"}, inplace=True)
            print(f"✅ 补充数据加载成功：{len(self.merged_df)}条记录")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 补充数据不存在：{MERGED_JSON_PATH}")
        except Exception as e:
            raise RuntimeError(f"❌ 补充数据加载失败：{e}")

        # 3. 加载实体映射表
        self.load_entity_mapping()

        # 4. 加载HAN模型
        self.load_han_model()

        # 5. 加载种子实体嵌入
        self.load_seed_embeddings()

        # 6. 构建前缀-父前缀映射
        self.parent_mapping = {
            row["inet6num"]: row["original_inet6num"]
            for _, row in self.source_df.iterrows()
            if pd.notna(row["original_inet6num"])
        }
        print(f"✅ 前缀-父前缀映射构建成功：{len(self.parent_mapping)}条映射关系")

    def merge_data(self) -> None:
        """合并源数据与补充数据（以inet6num为关联键）"""
        print("\n===== 合并数据源 =====")
        self.combined_df = pd.merge(
            self.source_df,
            self.merged_df,
            on="inet6num",
            how="outer",
            suffixes=("", "_merged")
        )

        # 初始化字段来源标记
        for col in ["netname", "descr", "country", "mnt-by", "status", "org", "descr_cleaned"]:
            if col in self.combined_df.columns:
                self.combined_df[f"{col}_source"] = "original"

        # 标记非种子前缀（后续仅对非种子执行模型补全）
        seed_prefixes = set(self.ent_val_to_id.get("Prefix", {}).keys())
        self.combined_df["is_seed"] = self.combined_df["inet6num"].isin(seed_prefixes)
        non_seed_count = (~self.combined_df["is_seed"]).sum()
        print(f"✅ 数据合并完成：共{len(self.combined_df)}条记录（非种子{non_seed_count}条）")

    def basic_preprocessing(self) -> None:
        """基础预处理：派生descr_cleaned + 统一字段格式（与种子前缀一致）"""
        print("\n===== 基础预处理 =====")

        # 1. 派生descr_cleaned（33维特征的关键字来源）
        if "descr_cleaned" not in self.combined_df.columns:
            self.combined_df["descr_cleaned"] = np.nan
        mask = self.combined_df["descr_cleaned"].isna() & self.combined_df["descr"].notna()
        if mask.sum() > 0:
            def standardize_descr(descr: str) -> str:
                if not descr or str(descr).lower() == "nan":
                    return ""
                tokens = re.split(r"[\s-]+", str(descr).lower())
                filtered = [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) >= 3]
                return " ".join(filtered)

            self.combined_df.loc[mask, "descr_cleaned"] = self.combined_df.loc[mask, "descr"].apply(standardize_descr)
            self.combined_df.loc[mask, "descr_cleaned_source"] = "derived_from_descr"
        print(f"1. descr_cleaned派生：{mask.sum()}条")

        # 2. 统一字段格式（去除空格、标准化大小写）
        str_fields = ["netname", "country", "mnt-by", "status", "org", "descr_cleaned"]
        for field in str_fields:
            if field in self.combined_df.columns:
                self.combined_df[field] = self.combined_df[field].astype(str).str.strip().str.upper()
                self.combined_df.loc[self.combined_df[field] == "NAN", field] = np.nan
        print(f"2. 字段格式标准化完成：{str_fields}")

        # 3. 基础补全（复用补充数据中的非冲突值）
        mergeable_fields = [("descr", "descr_merged"), ("country", "country_merged"), ("org", "org_merged")]
        for target_col, merge_col in mergeable_fields:
            if target_col in self.combined_df.columns and merge_col in self.combined_df.columns:
                mask = self.combined_df[target_col].isna() & self.combined_df[merge_col].notna()
                if mask.sum() > 0:
                    self.combined_df.loc[mask, target_col] = self.combined_df.loc[mask, merge_col]
                    self.combined_df.loc[mask, f"{target_col}_source"] = "merged_data"
                print(f"3. {target_col}从补充数据补全：{mask.sum()}条")

    def generate_non_seed_features(self, prefix_val: str, row: pd.Series) -> torch.Tensor:
        """生成非种子Prefix的33维特征（与种子前缀完全对齐）"""
        features = []

        # 1. 前缀长度特征（1维）：提取/后的数字（如2001::/48→48）
        try:
            prefix_len = int(prefix_val.split("/")[-1])
            features.append(torch.tensor([prefix_len / 128.0], dtype=torch.float32))  # 归一化到[0,1]
        except:
            features.append(torch.tensor([0.0], dtype=torch.float32))

        # 2. IPv6版本标识（1维）：固定为1
        features.append(torch.tensor([1.0], dtype=torch.float32))

        # 3. 网络类型独热编码（8维）：基于前缀长度分类
        net_type = torch.zeros(8, dtype=torch.float32)
        try:
            prefix_len = int(prefix_val.split("/")[-1])
            if prefix_len <= 12:
                net_type[0] = 1.0  # 骨干网
            elif prefix_len <= 24:
                net_type[1] = 1.0  # 区域网
            elif prefix_len <= 32:
                net_type[2] = 1.0  # 骨干接入网
            elif prefix_len <= 48:
                net_type[3] = 1.0  # 校园网/企业网
            elif prefix_len <= 64:
                net_type[4] = 1.0  # 子网
            elif prefix_len <= 80:
                net_type[5] = 1.0  # 终端网段
            elif prefix_len <= 96:
                net_type[6] = 1.0  # 物联网终端
            else:
                net_type[7] = 1.0  # 其他
        except:
            net_type[7] = 1.0  # 未知类型
        features.append(net_type)

        # 4. 国家独热编码（10维）：种子数据中高频国家（按出现频次排序）
        top_countries = ["US", "CN", "JP", "DE", "UK", "FR", "KR", "CA", "AU", "IN"]
        country_onehot = torch.zeros(10, dtype=torch.float32)
        if pd.notna(row["country"]) and row["country"] in top_countries:
            country_idx = top_countries.index(row["country"])
            country_onehot[country_idx] = 1.0
        features.append(country_onehot)

        # 5. 关键字词袋特征（10维）：基于descr_cleaned的Top10高频词
        top_keywords = ["INTERNET", "SERVICE", "PROVIDER", "NETWORK", "COMMUNICATION",
                        "TECHNOLOGY", "CORPORATION", "ORGANIZATION", "GOVERNMENT", "EDUCATION"]
        keyword_bow = torch.zeros(10, dtype=torch.float32)
        if pd.notna(row["descr_cleaned"]) and row["descr_cleaned"] != "":
            descr_tokens = row["descr_cleaned"].split()
            for i, kw in enumerate(top_keywords):
                if kw in descr_tokens:
                    keyword_bow[i] = 1.0
        features.append(keyword_bow)

        # 6. 状态独热编码（3维）：ALLOCATED、ASSIGNED、RESERVED
        status_onehot = torch.zeros(3, dtype=torch.float32)
        if pd.notna(row["status"]):
            status = row["status"].upper()
            if "ALLOCATED" in status:
                status_onehot[0] = 1.0
            elif "ASSIGNED" in status:
                status_onehot[1] = 1.0
            elif "RESERVED" in status:
                status_onehot[2] = 1.0
        features.append(status_onehot)

        # 拼接为33维特征
        return torch.cat(features, dim=0)

    def build_non_seed_graph(self) -> None:
        """构建非种子前缀的临时异构图（核心修复：连续ID映射+特征全量填充）"""
        print("\n===== 构建非种子前缀临时异构图 =====")
        non_seed_df = self.combined_df[~self.combined_df["is_seed"]].copy()
        if len(non_seed_df) == 0:
            print("⚠️ 无是非种子前缀，跳过临时图构建")
            return

        # ===================== 步骤1：收集所有节点和边（原始ID） =====================
        node_id_to_feat = defaultdict(dict)  # {实体类型: {原始ID: 特征向量}}
        edges = defaultdict(list)  # {边类型三元组: [(原始源ID, 原始目标ID)]}

        # 处理非种子Prefix节点
        non_seed_prefixes = []
        for idx, row in non_seed_df.iterrows():
            prefix_val = row["inet6num"]
            if pd.isna(prefix_val):
                continue
            # 分配原始ID
            if prefix_val not in self.ent_val_to_id["Prefix"]:
                self.ent_val_to_id["Prefix"][prefix_val] = self.next_ent_ids["Prefix"]
                self.ent_id_to_val["Prefix"][self.next_ent_ids["Prefix"]] = prefix_val
                self.next_ent_ids["Prefix"] += 1
            prefix_id = self.ent_val_to_id["Prefix"][prefix_val]
            # 生成特征（仅当ID未关联特征时添加）
            if prefix_id not in node_id_to_feat["Prefix"]:
                feat = self.generate_non_seed_features(prefix_val, row)
                node_id_to_feat["Prefix"][prefix_id] = feat
            non_seed_prefixes.append((prefix_val, prefix_id, row))

        # 处理关联实体节点（Mnt/Netname/Country/Status/Keyword）
        relation_mapping = {
            "mnt-by": ("Prefix", "prefix_to_mnt", "Mnt"),
            "netname": ("Prefix", "prefix_to_netname", "Netname"),
            "country": ("Prefix", "prefix_to_country", "Country"),
            "status": ("Prefix", "prefix_to_status", "Status"),
            "descr_cleaned": ("Prefix", "prefix_to_keyword", "Keyword")
        }
        for field, edge_type_triple in relation_mapping.items():
            src_type, etype_name, dst_type = edge_type_triple
            for prefix_val, prefix_id, row in non_seed_prefixes:
                ent_val = row[field]
                if pd.isna(ent_val) or ent_val == "" or str(ent_val).lower() == "nan":
                    continue
                # 分配原始ID
                if ent_val not in self.ent_val_to_id[dst_type]:
                    # 生成特征
                    if dst_type in self.seed_graph.ntypes:
                        feat_dim = self.seed_graph.nodes[dst_type].data["feat"].shape[1]
                    else:
                        feat_dim = 16
                    new_feat = torch.randn(feat_dim, dtype=torch.float32)
                    # 分配ID
                    ent_id = self.next_ent_ids[dst_type]
                    self.ent_val_to_id[dst_type][ent_val] = ent_id
                    self.ent_id_to_val[dst_type][ent_id] = ent_val
                    self.next_ent_ids[dst_type] += 1
                    node_id_to_feat[dst_type][ent_id] = new_feat
                else:
                    # 种子实体：从种子图获取特征
                    ent_id = self.ent_val_to_id[dst_type][ent_val]
                    if ent_id not in node_id_to_feat[dst_type]:
                        if dst_type in self.seed_graph.ntypes and ent_id < self.seed_graph.num_nodes(dst_type):
                            seed_feat = self.seed_graph.nodes[dst_type].data["feat"][ent_id]
                            node_id_to_feat[dst_type][ent_id] = seed_feat
                        else:
                            feat_dim = self.seed_graph.nodes[dst_type].data["feat"].shape[
                                1] if dst_type in self.seed_graph.ntypes else 16
                            node_id_to_feat[dst_type][ent_id] = torch.randn(feat_dim, dtype=torch.float32)
                # 添加边（去重）
                if (prefix_id, ent_id) not in edges[edge_type_triple]:
                    edges[edge_type_triple].append((prefix_id, ent_id))

        # ===================== 步骤2：生成连续ID映射（核心修复） =====================
        self.raw_to_continuous_id.clear()
        self.continuous_to_raw_id.clear()
        max_raw_id = defaultdict(int)  # 各实体类型的最大原始ID

        # 第一步：计算各实体类型的最大原始ID（确定DGL的节点数）
        for ntype in node_id_to_feat:
            if node_id_to_feat[ntype]:
                max_raw_id[ntype] = max(node_id_to_feat[ntype].keys())
            else:
                max_raw_id[ntype] = 0

        # 第二步：为每个实体类型生成连续ID映射（0 → max_raw_id[ntype]）
        for ntype in node_id_to_feat:
            # 初始化连续ID映射：覆盖所有0~max_raw_id的ID（包括空缺）
            for raw_id in range(max_raw_id[ntype] + 1):
                self.raw_to_continuous_id[ntype][raw_id] = raw_id  # 连续ID = 原始ID（直接对齐）
                self.continuous_to_raw_id[ntype][raw_id] = raw_id
            print(f"   - {ntype}：最大原始ID={max_raw_id[ntype]} → 连续ID范围=0~{max_raw_id[ntype]}")

        # ===================== 步骤3：构建全量特征矩阵（覆盖所有连续ID） =====================
        full_feat_matrix = {}
        for ntype in node_id_to_feat:
            max_cid = max_raw_id[ntype]
            # 确定特征维度
            if ntype in self.seed_graph.ntypes:
                feat_dim = self.seed_graph.nodes[ntype].data["feat"].shape[1]
            else:
                feat_dim = 16 if ntype != "Prefix" else 33

            # 初始化全量特征矩阵（默认值：0向量）
            full_feat = torch.zeros((max_cid + 1, feat_dim), dtype=torch.float32)

            # 填充已有节点的特征
            for raw_id, feat in node_id_to_feat[ntype].items():
                cid = self.raw_to_continuous_id[ntype][raw_id]
                if cid <= max_cid:
                    full_feat[cid] = feat

            full_feat_matrix[ntype] = full_feat.to(DEVICE)
            print(
                f"   - {ntype}：全量特征矩阵维度={full_feat_matrix[ntype].shape}（节点数={max_cid + 1}，特征维度={feat_dim}）")

        # ===================== 步骤4：转换边为连续ID =====================
        graph_data = {}
        for edge_type_triple, edge_list in edges.items():
            if not edge_list:
                continue
            src_ntype, etype, dst_ntype = edge_type_triple
            # 转换源/目标ID为连续ID
            src_cids = []
            dst_cids = []
            for src_raw, dst_raw in edge_list:
                if src_raw in self.raw_to_continuous_id[src_ntype] and dst_raw in self.raw_to_continuous_id[dst_ntype]:
                    src_cids.append(self.raw_to_continuous_id[src_ntype][src_raw])
                    dst_cids.append(self.raw_to_continuous_id[dst_ntype][dst_raw])
            # 添加到graph_data
            graph_data[edge_type_triple] = (
                torch.tensor(src_cids, dtype=torch.long),
                torch.tensor(dst_cids, dtype=torch.long)
            )

        # ===================== 步骤5：构建DGL异构图并添加特征 =====================
        self.temp_graph = dgl.heterograph(graph_data).to(DEVICE)

        # 添加全量特征矩阵（确保特征数=节点数）
        for ntype in full_feat_matrix:
            # 验证特征数与节点数一致
            assert full_feat_matrix[ntype].shape[0] == self.temp_graph.num_nodes(ntype), \
                f"❌ {ntype}特征数({full_feat_matrix[ntype].shape[0]})与节点数({self.temp_graph.num_nodes(ntype)})不匹配"
            self.temp_graph.nodes[ntype].data["feat"] = full_feat_matrix[ntype]

        # 最终校验
        print(f"✅ 非种子临时异构图构建完成：")
        print(f"   - 节点类型：{list(self.temp_graph.ntypes)}")
        print(f"   - 边类型：{[rel[1] for rel in self.temp_graph.canonical_etypes]}")
        print(
            f"   - Prefix节点数：{self.temp_graph.num_nodes('Prefix')}，特征数：{self.temp_graph.nodes['Prefix'].data['feat'].shape[0]}")

    def build_candidate_pool(self) -> None:
        """构建候选实体池（种子数据中高频实体，过滤稀有实体）"""
        print("\n===== 构建候选实体池 =====")
        # 候选实体类型与字段映射
        candidate_types = {
            "Mnt": "mnt-by",
            "Netname": "netname",
            "Country": "country",
            "Status": "status"
        }
        # 统计种子数据中各实体的出现频次
        for ent_type, field in candidate_types.items():
            edge_type_triple = ("Prefix", f"prefix_to_{ent_type.lower()}", ent_type)
            # 检查边类型是否存在
            if edge_type_triple in self.seed_graph.canonical_etypes:
                src_ids, dst_ids = self.seed_graph.edges(etype=edge_type_triple)
                ent_counts = Counter(dst_ids.cpu().numpy())
                # 筛选高频实体（≥MIN_CANDIDATE_FREQ）
                for ent_id, count in ent_counts.items():
                    if count >= MIN_CANDIDATE_FREQ:
                        ent_val = self.ent_id_to_val[ent_type].get(ent_id)
                        if ent_val:
                            self.candidate_pool[ent_type].append((ent_val, count))
            # 按频次排序
            self.candidate_pool[ent_type].sort(key=lambda x: x[1], reverse=True)
            print(f"   - {ent_type}：{len(self.candidate_pool[ent_type])}个候选实体（频次≥{MIN_CANDIDATE_FREQ}）")

    def calculate_association_prob(self, prefix_embed: torch.Tensor, candidate_embeds: List[torch.Tensor]) -> List[
        float]:
        """按论文公式计算非种子Prefix与候选实体的关联概率（点积+sigmoid）"""
        # L2归一化（与训练时一致）
        prefix_embed = F.normalize(prefix_embed, p=2, dim=0)
        candidate_embeds = [F.normalize(embed, p=2, dim=0) for embed in candidate_embeds]
        # 计算点积并映射到[0,1]
        probs = []
        for cand_embed in candidate_embeds:
            dot_product = torch.sum(prefix_embed * cand_embed)
            prob = torch.sigmoid(dot_product).item()
            probs.append(prob)
        return probs

    def check_consistency(self, prefix_row: pd.Series, completed_ent_type: str, completed_val: str) -> bool:
        """一致性校验：补全结果与已有信息无冲突（论文要求）"""
        # 1. Mnt与Country冲突校验（Mnt名称含国家码，需与Prefix的Country一致）
        if completed_ent_type == "Mnt" and pd.notna(prefix_row["country"]):
            # 提取Mnt中的国家码（如MAINT-JP-WIDE→JP）
            country_code = prefix_row["country"].upper()
            if country_code in completed_val:
                return True
            else:
                # 常见国家码映射（处理缩写变体）
                country_map = {"US": ["USA", "AMERICA"], "CN": ["CHINA", "PRC"], "JP": ["JAPAN"],
                               "DE": ["GERMANY"], "UK": ["BRITAIN", "UNITEDKINGDOM"]}
                for code, variants in country_map.items():
                    if country_code == code and any(var in completed_val for var in variants):
                        return True
                print(f"⚠️ 冲突：Mnt={completed_val} 与 Country={country_code} 不匹配，丢弃补全结果")
                return False
        # 2. 其他实体类型暂无需校验（可扩展）
        return True

    def model_based_completion(self) -> None:
        """模型基补全：利用训练好的HAN模型预测关联概率（论文核心逻辑）"""
        print("\n===== 模型基补全（论文3.4.1节核心） =====")
        non_seed_df = self.combined_df[~self.combined_df["is_seed"]].copy()
        if len(non_seed_df) == 0 or self.temp_graph is None:
            print("⚠️ 无是非种子前缀或临时图未构建，跳过模型基补全")
            return

        # 1. 生成非种子实体嵌入（固定模型参数，仅前向传播）
        print("1. 生成非种子实体嵌入（L2归一化）...")
        self.han_model.g = self.temp_graph  # 替换为非种子临时图
        with torch.no_grad():
            non_seed_embeds = self.han_model()
        # 提取非种子Prefix嵌入并L2归一化
        if "Prefix" not in non_seed_embeds:
            print("⚠️ 未生成非种子Prefix嵌入，跳过模型基补全")
            return
        prefix_embeds = non_seed_embeds["Prefix"]
        prefix_embeds = F.normalize(prefix_embeds, p=2, dim=1)
        print(f"   - 非种子Prefix嵌入生成完成：{prefix_embeds.shape[0]}个向量（维度{prefix_embeds.shape[1]}）")

        # 2. 补全规则（实体类型→字段→边类型三元组）
        completion_rules = [
            {
                "ent_type": "Mnt",
                "target_col": "mnt-by",
                "edge_type_triple": ("Prefix", "prefix_to_mnt", "Mnt")
            },
            {
                "ent_type": "Netname",
                "target_col": "netname",
                "edge_type_triple": ("Prefix", "prefix_to_netname", "Netname")
            },
            {
                "ent_type": "Country",
                "target_col": "country",
                "edge_type_triple": ("Prefix", "prefix_to_country", "Country")
            },
            {
                "ent_type": "Status",
                "target_col": "status",
                "edge_type_triple": ("Prefix", "prefix_to_status", "Status")
            }
        ]

        # 3. 对每个字段执行补全
        for rule in completion_rules:
            ent_type = rule["ent_type"]
            target_col = rule["target_col"]
            edge_type_triple = rule["edge_type_triple"]

            # 跳过：候选池为空或字段不存在
            if ent_type not in self.candidate_pool or len(self.candidate_pool[ent_type]) == 0:
                print(f"⚠️ 跳过{target_col}：无候选实体")
                continue
            if target_col not in self.combined_df.columns:
                print(f"⚠️ 跳过{target_col}：数据中无此字段")
                continue

            # 准备候选实体嵌入（兼容种子嵌入缺失的情况）
            candidate_vals = [c[0] for c in self.candidate_pool[ent_type]]
            candidate_embeds = []
            for val in candidate_vals:
                if val in self.seed_embeddings[ent_type]:
                    candidate_embeds.append(self.seed_embeddings[ent_type][val])
                else:
                    # 种子嵌入缺失时，从种子图实时生成
                    if ent_type in self.seed_graph.ntypes and val in self.ent_val_to_id[ent_type]:
                        ent_id = self.ent_val_to_id[ent_type][val]
                        with torch.no_grad():
                            seed_embeds = self.han_model()
                            if ent_type in seed_embeds and ent_id < len(seed_embeds[ent_type]):
                                cand_embed = seed_embeds[ent_type][ent_id]
                                candidate_embeds.append(cand_embed)
            if len(candidate_embeds) == 0:
                print(f"⚠️ 跳过{target_col}：无候选实体嵌入")
                continue

            # 筛选待补全的非种子前缀
            mask = (self.combined_df["is_seed"] == False) & (self.combined_df[target_col].isna())
            count = 0

            for idx, row in self.combined_df[mask].iterrows():
                prefix_val = row["inet6num"]
                # 获取非种子Prefix的原始ID和连续ID
                prefix_raw_id = self.ent_val_to_id["Prefix"].get(prefix_val)
                if prefix_raw_id is None:
                    continue
                # 转换为连续ID（嵌入索引）
                if prefix_raw_id not in self.raw_to_continuous_id["Prefix"]:
                    continue
                prefix_cid = self.raw_to_continuous_id["Prefix"][prefix_raw_id]
                if prefix_cid >= len(prefix_embeds):
                    continue
                prefix_embed = prefix_embeds[prefix_cid]

                # 计算与所有候选实体的关联概率
                probs = self.calculate_association_prob(prefix_embed, candidate_embeds)
                max_prob_idx = np.argmax(probs)
                max_prob = probs[max_prob_idx]
                best_candidate = candidate_vals[max_prob_idx]

                # 满足置信度阈值且一致性校验通过
                if max_prob >= CONFIDENCE_THRESHOLD:
                    if self.check_consistency(row, ent_type, best_candidate):
                        # 补全字段并标记来源
                        self.combined_df.at[idx, target_col] = best_candidate
                        self.combined_df.at[idx, f"{target_col}_source"] = f"model_pred(prob={max_prob:.3f})"
                        count += 1
                        # 向临时图添加补全的边（使用连续ID）
                        ent_raw_id = self.ent_val_to_id[ent_type][best_candidate]
                        ent_cid = self.raw_to_continuous_id[ent_type][ent_raw_id]
                        if edge_type_triple not in self.temp_graph.canonical_etypes:
                            self.temp_graph.add_edges([prefix_cid], [ent_cid], etype=edge_type_triple)
                        else:
                            self.temp_graph.add_edges([prefix_cid], [ent_cid], etype=edge_type_triple)

            print(f"   - {target_col}：{count}条补全（置信度≥{CONFIDENCE_THRESHOLD}）")

    def update_hetero_graph(self) -> None:
        """更新异构图：合并种子图与非种子图（含补全的边）"""
        print("\n===== 更新异构图 =====")
        if self.seed_graph is None or self.temp_graph is None:
            print("⚠️ 种子图或临时图未构建，跳过异构图更新")
            return

        # 1. 合并节点（种子节点+非种子节点）
        merged_feats = {}
        for ntype in self.seed_graph.ntypes:
            # 种子节点特征
            seed_feats = self.seed_graph.nodes[ntype].data["feat"]
            # 非种子节点特征（若存在）
            if ntype in self.temp_graph.ntypes and "feat" in self.temp_graph.nodes[ntype].data:
                non_seed_feats = self.temp_graph.nodes[ntype].data["feat"]
                # 合并特征（按ID顺序，非种子ID在种子之后）
                merged_feats[ntype] = torch.cat([seed_feats, non_seed_feats], dim=0)
            else:
                merged_feats[ntype] = seed_feats

        # 2. 合并边（种子边+非种子边+补全边）
        merged_edges = {}
        for edge_type_triple in self.seed_graph.canonical_etypes:
            # 种子边
            seed_src, seed_dst = self.seed_graph.edges(etype=edge_type_triple)
            # 非种子边（若存在）
            if edge_type_triple in self.temp_graph.canonical_etypes:
                non_seed_src, non_seed_dst = self.temp_graph.edges(etype=edge_type_triple)
                # 非种子边ID偏移（避免与种子ID冲突）
                seed_node_count = self.seed_graph.num_nodes(edge_type_triple[0])
                non_seed_src_shifted = non_seed_src + seed_node_count
                non_seed_dst_shifted = non_seed_dst + self.seed_graph.num_nodes(edge_type_triple[2])
                # 合并
                merged_src = torch.cat([seed_src, non_seed_src_shifted], dim=0)
                merged_dst = torch.cat([seed_dst, non_seed_dst_shifted], dim=0)
                merged_edges[edge_type_triple] = (merged_src, merged_dst)
            else:
                merged_edges[edge_type_triple] = (seed_src, seed_dst)

        # 3. 构建更新后的异构图
        self.updated_graph = dgl.heterograph(merged_edges).to(DEVICE)
        for ntype in merged_feats.keys():
            self.updated_graph.nodes[ntype].data["feat"] = merged_feats[ntype]

        # 4. 保存更新后的图
        dgl.save_graphs(UPDATED_GRAPH_PATH, [self.updated_graph])
        print(f"✅ 异构图更新完成并保存至：{UPDATED_GRAPH_PATH}")
        print(f"   - 总节点数：{sum(self.updated_graph.num_nodes(ntype) for ntype in self.updated_graph.ntypes)}")
        print(
            f"   - 总边数：{sum(self.updated_graph.num_edges(etype) for etype in self.updated_graph.canonical_etypes)}")

    def save_results(self) -> None:
        """保存补全结果并输出统计报告"""
        print("\n===== 补全结果统计与保存 =====")
        # 补全效果统计
        completion_stats = []
        for col in ["netname", "descr", "country", "mnt-by", "status", "org", "descr_cleaned"]:
            if col in self.combined_df.columns:
                total = len(self.combined_df)
                missing = self.combined_df[col].isna().sum()
                completed = total - missing
                completion_rate = (completed / total) * 100 if total > 0 else 0
                # 按种子/非种子拆分统计
                seed_completed = self.combined_df[self.combined_df["is_seed"]][col].notna().sum()
                seed_total = self.combined_df["is_seed"].sum()
                non_seed_completed = completed - seed_completed
                non_seed_total = total - seed_total
                non_seed_completion_rate = (non_seed_completed / non_seed_total) * 100 if non_seed_total > 0 else 0
                completion_stats.append({
                    "字段": col,
                    "总记录数": total,
                    "总补全率": f"{completion_rate:.1f}%",
                    "非种子记录数": non_seed_total,
                    "非种子补全率": f"{non_seed_completion_rate:.1f}%"
                })

        # 打印统计表格
        stats_df = pd.DataFrame(completion_stats)
        print(stats_df.to_string(index=False))

        # 保存补全结果
        try:
            self.combined_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
            print(f"\n✅ 补全结果已保存至：{OUTPUT_PATH}")
        except Exception as e:
            raise RuntimeError(f"❌ 结果保存失败：{e}")

    def run(self) -> None:
        """执行完整补全流程（严格遵循论文3.4.1节步骤）"""
        print("=" * 60)
        print("📚 非种子前缀Whois信息补全流程（论文3.4.1节）")
        print("=" * 60)

        try:
            # 论文要求的流程顺序：
            # 1. 数据加载与合并
            self.load_data()
            self.merge_data()
            # 2. 非种子前缀预处理（生成33维特征）
            self.basic_preprocessing()
            # 3. 构建非种子临时异构图
            self.build_non_seed_graph()
            # 4. 构建候选实体池
            self.build_candidate_pool()
            # 5. 模型基补全（核心步骤）
            self.model_based_completion()
            # 6. 更新异构图（含补全边）
            self.update_hetero_graph()
            # 7. 保存结果
            self.save_results()

            print("\n" + "=" * 60)
            print("✅ 补全流程全部完成！")
            print("=" * 60)
        except Exception as e:
            # 异常时保存部分结果
            if self.combined_df is not None:
                partial_path = f"partial_{OUTPUT_PATH}"
                self.combined_df.to_csv(partial_path, index=False, encoding="utf-8")
                print(f"\n⚠️ 补全流程异常中断，已保存部分结果至：{partial_path}")
            raise RuntimeError(f"❌ 补全流程失败：{e}")


# ===================== 执行入口 =====================
if __name__ == "__main__":
    import os  # 新增os导入，用于文件存在性判断

    try:
        completer = WhoisCompleter()
        completer.run()
    except Exception as e:
        print(f"\n❌ 程序执行失败：{e}")
        import traceback

        traceback.print_exc()
        exit(1)