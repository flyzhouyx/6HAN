import pandas as pd
import numpy as np
import dgl
import torch  # 新增：导入PyTorch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import ipaddress
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore")

# 下载NLTK资源（首次运行需执行）
# nltk.download('punkt')
# nltk.download('stopwords')

# ===================== 配置项（优先使用GPU）=====================
INPUT_CSV = "Data/seed_prefixes_with_info.csv"
OUTPUT_GRAPH = "Data/ipv6_hetero_graph.bin"
OUTPUT_ENTITY_MAP = "Data/entity_mapping.csv"
STOP_WORDS = set(stopwords.words('english'))
MIN_KEYWORD_FREQ = 2  # 关键词最低出现频次
MIN_KEYWORD_LEN = 3  # 关键词最低长度
TOP_K_KEYWORDS = 5  # 每个descr筛选TOP5核心关键词（按TF-IDF权重）
IPV6_MAX_PREFIX_LEN = 128  # IPv6最大前缀长度（归一化用）

# 自动检测CUDA，优先使用GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  # 使用第0块GPU（多GPU可改cuda:1等）
    print(f"✅ 检测到CUDA可用，使用GPU：{torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  未检测到CUDA，使用CPU")


# ===================== 核心工具函数 =====================
def z_score_normalize(feat: np.ndarray) -> np.ndarray:
    """Z-Score标准化（匹配论文公式(1)）
    公式：x' = (x - μ) / σ，μ为均值，σ为标准差
    """
    mu = np.mean(feat, axis=0)
    sigma = np.std(feat, axis=0)
    # 避免除以0（σ=0时特征值全相同，归一化为0）
    sigma[sigma == 0] = 1e-8
    normalized_feat = (feat - mu) / sigma
    return normalized_feat


def extract_keywords_from_descr(descr_series: pd.Series) -> Tuple[List[List[str]], Dict[str, float]]:
    """
    从descr字段提取核心关键词（匹配论文：NLTK分词→过滤停用词→TF-IDF算权重→筛选核心词）
    返回：每个前缀的TOP-K关键词列表 + 全局关键词TF-IDF权重字典
    """
    # 1. 预处理：空值填充+文本清洗
    descr_clean = descr_series.fillna("").apply(lambda x: re.sub(r"[^a-zA-Z\s-]", "", x))

    # 2. NLTK分词+停用词过滤
    tokenized_corpus = []
    for descr in descr_clean:
        tokens = word_tokenize(descr.lower())  # NLTK标准分词
        filtered_tokens = [
            token for token in tokens
            if token not in STOP_WORDS
               and len(token) >= MIN_KEYWORD_LEN
               and token.isalpha()
        ]
        tokenized_corpus.append(" ".join(filtered_tokens))  # TF-IDF输入格式

    # 3. TF-IDF计算词权重
    tfidf = TfidfVectorizer(stop_words=None, min_df=MIN_KEYWORD_FREQ)
    tfidf_matrix = tfidf.fit_transform(tokenized_corpus)
    keyword_vocab = tfidf.get_feature_names_out()
    keyword_weight = dict(zip(keyword_vocab, tfidf.idf_))  # IDF作为权重（表征词的重要性）

    # 4. 筛选每个前缀的TOP-K核心关键词
    final_keywords_list = []
    for i in range(len(tokenized_corpus)):
        if not tokenized_corpus[i]:
            final_keywords_list.append([])
            continue
        # 获取当前文本的TF-IDF权重
        row = tfidf_matrix[i].toarray()[0]
        # 按权重排序取TOP-K
        kw_idx = np.argsort(row)[::-1][:TOP_K_KEYWORDS]
        top_k_kw = [keyword_vocab[idx] for idx in kw_idx if row[idx] > 0]
        final_keywords_list.append(top_k_kw)

    print(f"关键词提取完成：全局有效关键词数={len(keyword_weight)}，单文本TOP{TOP_K_KEYWORDS}核心词")
    print(f"高频关键词（IDF权重TOP5）：{dict(sorted(keyword_weight.items(), key=lambda x: x[1], reverse=True)[:5])}")
    return final_keywords_list, keyword_weight


def build_entity_mapping(df: pd.DataFrame, keyword_weight: Dict[str, float]) -> Dict[str, Dict[str, int]]:
    """构建6类实体的ID映射（严格对齐表1）"""
    entity_mapping = defaultdict(dict)
    current_id = 0

    # 1. Prefix（核心实体：IPv6前缀）
    prefixes = df["inet6num"].unique()
    for prefix in prefixes:
        entity_mapping["Prefix"][prefix] = current_id
        current_id += 1
    # 2. Mnt（管理主体标识）
    mnts = df["mnt-by"].dropna().unique()
    for mnt in mnts:
        entity_mapping["Mnt"][mnt] = current_id
        current_id += 1
    # 3. Netname（业务网络标识）
    netnames = df["netname"].dropna().unique()
    for netname in netnames:
        entity_mapping["Netname"][netname] = current_id
        current_id += 1
    # 4. Country（地域属性标识）
    countries = df["country"].dropna().unique()
    for country in countries:
        entity_mapping["Country"][country] = current_id
        current_id += 1
    # 5. Keyword（业务语义标识：仅保留有效关键词）
    valid_keywords = [kw for kw in keyword_weight.keys() if keyword_weight[kw] > 0]
    for kw in valid_keywords:
        entity_mapping["Keyword"][kw] = current_id
        current_id += 1
    # 6. Status（使用权限标识）
    statuses = df["status"].dropna().unique()
    for status in statuses:
        entity_mapping["Status"][status] = current_id
        current_id += 1

    # 打印实体统计（对齐论文表1）
    print("\n=== 实体统计（表1）===")
    for ent_type, ent2id in entity_mapping.items():
        print(f"- {ent_type}：{len(ent2id)} 个实体")
    return entity_mapping


def build_entity_features(df: pd.DataFrame, entity_mapping: Dict[str, Dict[str, int]],
                          keyword_weight: Dict[str, float]) -> Dict[str, torch.Tensor]:
    """构建6类实体特征（严格对齐表3，所有特征执行Z-Score标准化，返回GPU/CPU张量）"""
    features = {}

    # -------------------- 1. Prefix特征（33维）--------------------
    # 表3规则：32维半字节编码 + 1维前缀长度归一化
    prefix_vals = list(entity_mapping["Prefix"].keys())
    prefix_feats = []
    for prefix in prefix_vals:
        try:
            ip_part, len_part = prefix.split("/")
            # IPv6半字节编码（32维）：展开为8段×4位=32个半字节
            full_ip = ipaddress.IPv6Address(ip_part).exploded.replace(":", "")
            ip_encoding = [int(c, 16) for c in full_ip]  # 每个半字节转0-15数值
            # 前缀长度归一化（1维）：/len → [0,1]
            len_feat = int(len_part) / IPV6_MAX_PREFIX_LEN
            # 合并特征
            prefix_feat = ip_encoding + [len_feat]
            prefix_feats.append(prefix_feat)
        except Exception as e:
            print(f"⚠️ Prefix {prefix} 解析失败：{e}，填充0向量")
            prefix_feats.append([0] * 33)
    # Z-Score标准化 + 转PyTorch张量（移至指定设备）
    prefix_feats = np.array(prefix_feats, dtype=np.float32)
    features["Prefix"] = torch.tensor(z_score_normalize(prefix_feats), device=DEVICE)
    print(f"Prefix特征：{features['Prefix'].shape}（33维，已标准化，设备：{features['Prefix'].device}）")

    # -------------------- 2. Mnt特征（2维）--------------------
    # 表3规则：维护规模归一化 + 区域标识
    mnt_vals = list(entity_mapping["Mnt"].keys())
    # 统计每个Mnt管理的前缀数
    mnt_prefix_count = df["mnt-by"].value_counts().to_dict()
    max_mnt_count = max(mnt_prefix_count.values()) if mnt_prefix_count else 1
    mnt_feats = []
    for mnt in mnt_vals:
        # 维度1：维护规模归一化 = 该Mnt前缀数 / 最大维护规模
        scale_feat = mnt_prefix_count.get(mnt, 1) / max_mnt_count
        # 维度2：区域标识 = 含国家代码(JP/CN/US)为1，否则0.5
        region_feat = 1.0 if any(reg in mnt for reg in ["JP", "CN", "US"]) else 0.5
        mnt_feats.append([scale_feat, region_feat])
    # Z-Score标准化 + 转PyTorch张量
    mnt_feats = np.array(mnt_feats, dtype=np.float32)
    features["Mnt"] = torch.tensor(z_score_normalize(mnt_feats), device=DEVICE)

    # -------------------- 3. Netname特征（2维）--------------------
    # 表3规则：名称长度归一化 + NET标识
    netname_vals = list(entity_mapping["Netname"].keys())
    max_netname_len = max(len(name) for name in netname_vals) if netname_vals else 1
    netname_feats = []
    for name in netname_vals:
        # 维度1：名称长度归一化 = 字符长度 / 最大长度
        len_feat = len(name) / max_netname_len
        # 维度2：NET标识 = 含NET子串为1，否则0
        net_feat = 1.0 if "NET" in name.upper() else 0.0
        netname_feats.append([len_feat, net_feat])
    # Z-Score标准化 + 转PyTorch张量
    netname_feats = np.array(netname_feats, dtype=np.float32)
    features["Netname"] = torch.tensor(z_score_normalize(netname_feats), device=DEVICE)

    # -------------------- 4. Country特征（2维）--------------------
    # 表3规则：前缀数量占比 + 区域分类占比
    country_vals = list(entity_mapping["Country"].keys())
    country_prefix_count = df["country"].value_counts().to_dict()
    total_prefix = len(df)
    # 预定义大洲映射（对齐论文表3）
    continent_mapping = {
        "JP": "Asia", "CN": "Asia", "KR": "Asia", "SG": "Asia",
        "US": "NA", "CA": "NA", "MX": "NA",
        "GB": "EU", "DE": "EU", "FR": "EU"
    }
    # 统计各大洲前缀数
    continent_count = defaultdict(int)
    for country, cnt in country_prefix_count.items():
        continent = continent_mapping.get(country, "Other")
        continent_count[continent] += cnt
    # 构建特征
    country_feats = []
    for country in country_vals:
        # 维度1：前缀数量占比 = 该国家前缀数 / 总前缀数
        ratio_feat = country_prefix_count.get(country, 1) / total_prefix
        # 维度2：区域分类占比 = 所属大洲前缀数 / 总前缀数
        continent = continent_mapping.get(country, "Other")
        continent_feat = continent_count[continent] / total_prefix
        country_feats.append([ratio_feat, continent_feat])
    # Z-Score标准化 + 转PyTorch张量
    country_feats = np.array(country_feats, dtype=np.float32)
    features["Country"] = torch.tensor(z_score_normalize(country_feats), device=DEVICE)

    # -------------------- 5. Keyword特征（2维）--------------------
    # 表3规则：词频归一化 + 组织关联度
    kw_vals = list(entity_mapping["Keyword"].keys())
    # 统计关键词全局词频
    kw_global_freq = Counter()
    for kw_list in df["keywords"]:
        kw_global_freq.update(kw_list)
    max_kw_freq = max(kw_global_freq.values()) if kw_global_freq else 1
    # 统计关键词-Mnt共现次数
    kw_mnt_cooccur = defaultdict(int)
    total_kw_occur = defaultdict(int)
    for _, row in df.iterrows():
        mnt = row["mnt-by"]
        kws = row["keywords"]
        if pd.isna(mnt) or not kws:
            continue
        for kw in kws:
            kw_mnt_cooccur[(kw, mnt)] += 1
            total_kw_occur[kw] += 1
    # 构建特征
    kw_feats = []
    for kw in kw_vals:
        # 维度1：词频归一化 = 全局词频 / 最大词频
        freq_feat = kw_global_freq.get(kw, 1) / max_kw_freq
        # 维度2：组织关联度 = 关键词-Mnt共现次数 / 关键词总出现次数
        cooccur_total = sum(kw_mnt_cooccur[(kw, m)] for m in mnt_vals if (kw, m) in kw_mnt_cooccur)
        org_feat = cooccur_total / total_kw_occur.get(kw, 1) if total_kw_occur.get(kw, 1) > 0 else 0.0
        kw_feats.append([freq_feat, org_feat])
    # Z-Score标准化 + 转PyTorch张量
    kw_feats = np.array(kw_feats, dtype=np.float32)
    features["Keyword"] = torch.tensor(z_score_normalize(kw_feats), device=DEVICE)

    # -------------------- 6. Status特征（2维）--------------------
    # 表3规则：可移植性标识 + 分配状态标识
    status_vals = list(entity_mapping["Status"].keys())
    status_feats = []
    for status in status_vals:
        # 维度1：可移植性标识 = PORTABLE为1，否则0
        portable_feat = 1.0 if "PORTABLE" in status else 0.0
        # 维度2：分配状态标识 = ASSIGNED为1，否则0
        assigned_feat = 1.0 if "ASSIGNED" in status else 0.0
        status_feats.append([portable_feat, assigned_feat])
    # Z-Score标准化 + 转PyTorch张量
    status_feats = np.array(status_feats, dtype=np.float32)
    features["Status"] = torch.tensor(z_score_normalize(status_feats), device=DEVICE)

    # 验证特征维度（对齐表3）
    print("\n=== 实体特征维度验证（表3）===")
    feat_dim_check = {
        "Prefix": 33, "Mnt": 2, "Netname": 2,
        "Country": 2, "Keyword": 2, "Status": 2
    }
    for ent_type, exp_dim in feat_dim_check.items():
        act_dim = features[ent_type].shape[1]
        print(
            f"- {ent_type}：预期{exp_dim}维，实际{act_dim}维 → {'✅' if exp_dim == act_dim else '❌'}，设备：{features[ent_type].device}")
    return features


def build_graph_edges(df: pd.DataFrame, entity_mapping: Dict[str, Dict[str, int]], g: dgl.DGLHeteroGraph) -> None:
    """构建7类边（严格对齐表2，边权重转为GPU/CPU张量）"""

    # 辅助函数：获取实体ID
    def get_ent_id(ent_type: str, ent_val: str) -> int:
        if pd.isna(ent_val):
            return None
        return entity_mapping[ent_type].get(ent_val)

    # -------------------- 1. prefix_to_mnt（Prefix→Mnt）--------------------
    src, dst = [], []
    for _, row in df.iterrows():
        p_id = get_ent_id("Prefix", row["inet6num"])
        m_id = get_ent_id("Mnt", row["mnt-by"])
        if p_id and m_id:
            src.append(p_id)
            dst.append(m_id)
    g.add_edges(src, dst, etype=("Prefix", "prefix_to_mnt", "Mnt"))
    # 边权重转PyTorch张量（移至指定设备）
    g.edges[("Prefix", "prefix_to_mnt", "Mnt")].data["weight"] = torch.ones(len(src), dtype=torch.float32,
                                                                            device=DEVICE)
    print(f"prefix_to_mnt 边数：{len(src)}，权重设备：{g.edges[('Prefix', 'prefix_to_mnt', 'Mnt')].data['weight'].device}")

    # -------------------- 2. prefix_to_netname（Prefix→Netname）--------------------
    src, dst = [], []
    for _, row in df.iterrows():
        p_id = get_ent_id("Prefix", row["inet6num"])
        n_id = get_ent_id("Netname", row["netname"])
        if p_id and n_id:
            src.append(p_id)
            dst.append(n_id)
    g.add_edges(src, dst, etype=("Prefix", "prefix_to_netname", "Netname"))
    g.edges[("Prefix", "prefix_to_netname", "Netname")].data["weight"] = torch.ones(len(src), dtype=torch.float32,
                                                                                    device=DEVICE)
    print(f"prefix_to_netname 边数：{len(src)}")

    # -------------------- 3. prefix_to_country（Prefix→Country）--------------------
    src, dst = [], []
    for _, row in df.iterrows():
        p_id = get_ent_id("Prefix", row["inet6num"])
        c_id = get_ent_id("Country", row["country"])
        if p_id and c_id:
            src.append(p_id)
            dst.append(c_id)
    g.add_edges(src, dst, etype=("Prefix", "prefix_to_country", "Country"))
    g.edges[("Prefix", "prefix_to_country", "Country")].data["weight"] = torch.ones(len(src), dtype=torch.float32,
                                                                                    device=DEVICE)
    print(f"prefix_to_country 边数：{len(src)}")

    # -------------------- 4. prefix_to_keyword（Prefix→Keyword）--------------------
    src, dst = [], []
    for _, row in df.iterrows():
        p_id = get_ent_id("Prefix", row["inet6num"])
        kws = row["keywords"]
        if p_id and kws:
            for kw in kws:
                k_id = get_ent_id("Keyword", kw)
                if k_id:
                    src.append(p_id)
                    dst.append(k_id)
    g.add_edges(src, dst, etype=("Prefix", "prefix_to_keyword", "Keyword"))
    g.edges[("Prefix", "prefix_to_keyword", "Keyword")].data["weight"] = torch.ones(len(src), dtype=torch.float32,
                                                                                    device=DEVICE)
    print(f"prefix_to_keyword 边数：{len(src)}")

    # -------------------- 5. prefix_to_status（Prefix→Status）--------------------
    src, dst = [], []
    for _, row in df.iterrows():
        p_id = get_ent_id("Prefix", row["inet6num"])
        s_id = get_ent_id("Status", row["status"])
        if p_id and s_id:
            src.append(p_id)
            dst.append(s_id)
    g.add_edges(src, dst, etype=("Prefix", "prefix_to_status", "Status"))
    g.edges[("Prefix", "prefix_to_status", "Status")].data["weight"] = torch.ones(len(src), dtype=torch.float32,
                                                                                  device=DEVICE)
    print(f"prefix_to_status 边数：{len(src)}")

    # -------------------- 6. same_mnt_prefix（Prefix→Prefix）--------------------
    # 规则：同Mnt的前缀互加有向边
    src, dst = [], []
    mnt_groups = df.groupby("mnt-by")["inet6num"].apply(list).to_dict()
    for mnt, prefixes in mnt_groups.items():
        if pd.isna(mnt) or len(prefixes) < 2:
            continue
        prefix_ids = [get_ent_id("Prefix", p) for p in prefixes if get_ent_id("Prefix", p)]
        # 生成所有两两有向边
        for p1 in prefix_ids:
            for p2 in prefix_ids:
                if p1 != p2:
                    src.append(p1)
                    dst.append(p2)
    g.add_edges(src, dst, etype=("Prefix", "same_mnt_prefix", "Prefix"))
    g.edges[("Prefix", "same_mnt_prefix", "Prefix")].data["weight"] = torch.ones(len(src), dtype=torch.float32,
                                                                                 device=DEVICE)
    print(f"same_mnt_prefix 边数：{len(src)}")

    # -------------------- 7. same_netname_prefix（Prefix→Prefix）--------------------
    # 规则：同Netname的前缀互加有向边
    src, dst = [], []
    netname_groups = df.groupby("netname")["inet6num"].apply(list).to_dict()
    for netname, prefixes in netname_groups.items():
        if pd.isna(netname) or len(prefixes) < 2:
            continue
        prefix_ids = [get_ent_id("Prefix", p) for p in prefixes if get_ent_id("Prefix", p)]
        # 生成所有两两有向边
        for p1 in prefix_ids:
            for p2 in prefix_ids:
                if p1 != p2:
                    src.append(p1)
                    dst.append(p2)
    g.add_edges(src, dst, etype=("Prefix", "same_netname_prefix", "Prefix"))
    g.edges[("Prefix", "same_netname_prefix", "Prefix")].data["weight"] = torch.ones(len(src), dtype=torch.float32,
                                                                                     device=DEVICE)
    print(f"same_netname_prefix 边数：{len(src)}")


# ===================== 主执行流程（匹配论文Algorithm 1）=====================
if __name__ == "__main__":
    # Step 1: 加载数据
    df = pd.read_csv(INPUT_CSV)
    print(f"\n=== 加载数据 ===")
    print(f"数据行数：{len(df)}")
    print(f"核心字段：{df[['inet6num', 'netname', 'country', 'mnt-by', 'descr', 'status']].columns.tolist()}")

    # Step 2: 提取核心关键词（NLTK+TF-IDF）
    df["keywords"], keyword_weight = extract_keywords_from_descr(df["descr"])

    # Step 3: 构建实体映射
    entity_mapping = build_entity_mapping(df, keyword_weight)
    # 保存实体映射表
    mapping_records = []
    for ent_type, ent2id in entity_mapping.items():
        for ent_val, ent_id in ent2id.items():
            mapping_records.append({
                "entity_type": ent_type,
                "entity_value": ent_val,
                "entity_id": ent_id
            })
    pd.DataFrame(mapping_records).to_csv(OUTPUT_ENTITY_MAP, index=False, encoding="utf-8")
    print(f"\n实体映射表已保存至：{OUTPUT_ENTITY_MAP}")

    # Step 4: 初始化异构图（7类边）
    RELATION_TUPLES = [
        ("Prefix", "prefix_to_mnt", "Mnt"),
        ("Prefix", "prefix_to_netname", "Netname"),
        ("Prefix", "prefix_to_country", "Country"),
        ("Prefix", "prefix_to_keyword", "Keyword"),
        ("Prefix", "prefix_to_status", "Status"),
        ("Prefix", "same_mnt_prefix", "Prefix"),
        ("Prefix", "same_netname_prefix", "Prefix")
    ]
    num_nodes_dict = {ent_type: len(ent2id) for ent_type, ent2id in entity_mapping.items()}
    # 初始化图并移至GPU/CPU
    g = dgl.heterograph({rel: ([], []) for rel in RELATION_TUPLES}, num_nodes_dict=num_nodes_dict).to(DEVICE)
    print(f"\n=== 初始化异构图 ===")
    print(f"实体类型数：{len(g.ntypes)}")
    print(f"关系类型数：{len(g.canonical_etypes)}")
    print(f"图设备：{g.device}")

    # Step 5: 构建并赋值实体特征（含Z-Score标准化）
    entity_features = build_entity_features(df, entity_mapping, keyword_weight)
    for ent_type, feat in entity_features.items():
        g.nodes[ent_type].data["feat"] = feat

    # Step 6: 构建7类边
    print(f"\n=== 构建边（表2）===")
    build_graph_edges(df, entity_mapping, g)

    # Step 7: 保存异构图并验证
    # 保存前将图移回CPU（DGL保存GPU图可能有兼容性问题）
    g_save = g.cpu()
    dgl.save_graphs(OUTPUT_GRAPH, [g_save])
    print(f"\n=== 异构图构建完成 ===")
    print(f"图保存路径：{OUTPUT_GRAPH}")
    print(f"总节点数：{sum(g.num_nodes(ntype) for ntype in g.ntypes)}")
    print(f"总边数：{sum(g.num_edges(etype) for etype in g.canonical_etypes)}")
    print(f"当前计算设备：{DEVICE}")

    # 验证示例：Prefix→Mnt边的前5条
    target_etype = ("Prefix", "prefix_to_mnt", "Mnt")
    if g.num_edges(target_etype) > 0:
        src_ids, dst_ids = g.all_edges(etype=target_etype, form="uv")
        print(f"\n=== 边示例（prefix_to_mnt 前5条）===")
        for i in range(min(5, len(src_ids))):
            p_id = src_ids[i].item()
            m_id = dst_ids[i].item()
            # 反向查询实体值
            p_val = next(k for k, v in entity_mapping["Prefix"].items() if v == p_id)
            m_val = next(k for k, v in entity_mapping["Mnt"].items() if v == m_id)
            print(f"Prefix({p_id}): {p_val} → Mnt({m_id}): {m_val}")