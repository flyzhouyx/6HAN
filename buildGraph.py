import pandas as pd
import numpy as np
import dgl
import torch
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

INPUT_CSV = "Data/seed_prefixes_with_info.csv"
OUTPUT_GRAPH = "Data/ipv6_hetero_graph.bin"
OUTPUT_ENTITY_MAP = "Data/entity_mapping.csv"
STOP_WORDS = set(stopwords.words('english'))
MIN_KEYWORD_FREQ = 2
MIN_KEYWORD_LEN = 3
TOP_K_KEYWORDS = 5
IPV6_MAX_PREFIX_LEN = 128

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def z_score_normalize(feat: np.ndarray) -> np.ndarray:
    mu = np.mean(feat, axis=0)
    sigma = np.std(feat, axis=0)
    sigma[sigma == 0] = 1e-8
    normalized_feat = (feat - mu) / sigma
    return normalized_feat

def extract_keywords_from_descr(descr_series: pd.Series) -> Tuple[List[List[str]], Dict[str, float]]:
    descr_clean = descr_series.fillna("").apply(lambda x: re.sub(r"[^a-zA-Z\s-]", "", x))

    tokenized_corpus = []
    for descr in descr_clean:
        tokens = word_tokenize(descr.lower())
        filtered_tokens = [
            token for token in tokens
            if token not in STOP_WORDS
               and len(token) >= MIN_KEYWORD_LEN
               and token.isalpha()
        ]
        tokenized_corpus.append(" ".join(filtered_tokens))

    tfidf = TfidfVectorizer(stop_words=None, min_df=MIN_KEYWORD_FREQ)
    tfidf_matrix = tfidf.fit_transform(tokenized_corpus)
    keyword_vocab = tfidf.get_feature_names_out()
    keyword_weight = dict(zip(keyword_vocab, tfidf.idf_))

    final_keywords_list = []
    for i in range(len(tokenized_corpus)):
        if not tokenized_corpus[i]:
            final_keywords_list.append([])
            continue
        row = tfidf_matrix[i].toarray()[0]
        kw_idx = np.argsort(row)[::-1][:TOP_K_KEYWORDS]
        top_k_kw = [keyword_vocab[idx] for idx in kw_idx if row[idx] > 0]
        final_keywords_list.append(top_k_kw)

    return final_keywords_list, keyword_weight

def build_entity_mapping(df: pd.DataFrame, keyword_weight: Dict[str, float]) -> Dict[str, Dict[str, int]]:
    entity_mapping = defaultdict(dict)
    current_id = 0

    prefixes = df["inet6num"].unique()
    for prefix in prefixes:
        entity_mapping["Prefix"][prefix] = current_id
        current_id += 1

    mnts = df["mnt-by"].dropna().unique()
    for mnt in mnts:
        entity_mapping["Mnt"][mnt] = current_id
        current_id += 1

    netnames = df["netname"].dropna().unique()
    for netname in netnames:
        entity_mapping["Netname"][netname] = current_id
        current_id += 1

    countries = df["country"].dropna().unique()
    for country in countries:
        entity_mapping["Country"][country] = current_id
        current_id += 1

    valid_keywords = [kw for kw in keyword_weight.keys() if keyword_weight[kw] > 0]
    for kw in valid_keywords:
        entity_mapping["Keyword"][kw] = current_id
        current_id += 1

    statuses = df["status"].dropna().unique()
    for status in statuses:
        entity_mapping["Status"][status] = current_id
        current_id += 1

    return entity_mapping

def build_entity_features(df: pd.DataFrame, entity_mapping: Dict[str, Dict[str, int]],
                          keyword_weight: Dict[str, float]) -> Dict[str, torch.Tensor]:
    features = {}

    prefix_vals = list(entity_mapping["Prefix"].keys())
    prefix_feats = []
    for prefix in prefix_vals:
        try:
            ip_part, len_part = prefix.split("/")
            full_ip = ipaddress.IPv6Address(ip_part).exploded.replace(":", "")
            ip_encoding = [int(c, 16) for c in full_ip]
            len_feat = int(len_part) / IPV6_MAX_PREFIX_LEN
            prefix_feat = ip_encoding + [len_feat]
            prefix_feats.append(prefix_feat)
        except Exception as e:
            prefix_feats.append([0] * 33)
    prefix_feats = np.array(prefix_feats, dtype=np.float32)
    features["Prefix"] = torch.tensor(z_score_normalize(prefix_feats), device=DEVICE)

    mnt_vals = list(entity_mapping["Mnt"].keys())
    mnt_prefix_count = df["mnt-by"].value_counts().to_dict()
    max_mnt_count = max(mnt_prefix_count.values()) if mnt_prefix_count else 1
    mnt_feats = []
    for mnt in mnt_vals:
        scale_feat = mnt_prefix_count.get(mnt, 1) / max_mnt_count
        region_feat = 1.0 if any(reg in mnt for reg in ["JP", "CN", "US"]) else 0.5
        mnt_feats.append([scale_feat, region_feat])
    mnt_feats = np.array(mnt_feats, dtype=np.float32)
    features["Mnt"] = torch.tensor(z_score_normalize(mnt_feats), device=DEVICE)

    netname_vals = list(entity_mapping["Netname"].keys())
    max_netname_len = max(len(name) for name in netname_vals) if netname_vals else 1
    netname_feats = []
    for name in netname_vals:
        len_feat = len(name) / max_netname_len
        net_feat = 1.0 if "NET" in name.upper() else 0.0
        netname_feats.append([len_feat, net_feat])
    netname_feats = np.array(netname_feats, dtype=np.float32)
    features["Netname"] = torch.tensor(z_score_normalize(netname_feats), device=DEVICE)

    country_vals = list(entity_mapping["Country"].keys())
    country_prefix_count = df["country"].value_counts().to_dict()
    total_prefix = len(df)
    continent_mapping = {
        "JP": "Asia", "CN": "Asia", "KR": "Asia", "SG": "Asia",
        "US": "NA", "CA": "NA", "MX": "NA",
        "GB": "EU", "DE": "EU", "FR": "EU"
    }
    continent_count = defaultdict(int)
    for country, cnt in country_prefix_count.items():
        continent = continent_mapping.get(country, "Other")
        continent_count[continent] += cnt
    country_feats = []
    for country in country_vals:
        ratio_feat = country_prefix_count.get(country, 1) / total_prefix
        continent = continent_mapping.get(country, "Other")
        continent_feat = continent_count[continent] / total_prefix
        country_feats.append([ratio_feat, continent_feat])
    country_feats = np.array(country_feats, dtype=np.float32)
    features["Country"] = torch.tensor(z_score_normalize(country_feats), device=DEVICE)

    kw_vals = list(entity_mapping["Keyword"].keys())
    kw_global_freq = Counter()
    for kw_list in df["keywords"]:
        kw_global_freq.update(kw_list)
    max_kw_freq = max(kw_global_freq.values()) if kw_global_freq else 1
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
    kw_feats = []
    for kw in kw_vals:
        freq_feat = kw_global_freq.get(kw, 1) / max_kw_freq
        cooccur_total = sum(kw_mnt_cooccur[(kw, m)] for m in mnt_vals if (kw, m) in kw_mnt_cooccur)
        org_feat = cooccur_total / total_kw_occur.get(kw, 1) if total_kw_occur.get(kw, 1) > 0 else 0.0
        kw_feats.append([freq_feat, org_feat])
    kw_feats = np.array(kw_feats, dtype=np.float32)
    features["Keyword"] = torch.tensor(z_score_normalize(kw_feats), device=DEVICE)

    status_vals = list(entity_mapping["Status"].keys())
    status_feats = []
    for status in status_vals:
        portable_feat = 1.0 if "PORTABLE" in status else 0.0
        assigned_feat = 1.0 if "ASSIGNED" in status else 0.0
        status_feats.append([portable_feat, assigned_feat])
    status_feats = np.array(status_feats, dtype=np.float32)
    features["Status"] = torch.tensor(z_score_normalize(status_feats), device=DEVICE)

    return features

def build_graph_edges(df: pd.DataFrame, entity_mapping: Dict[str, Dict[str, int]], g: dgl.DGLHeteroGraph) -> None:
    def get_ent_id(ent_type: str, ent_val: str) -> int:
        if pd.isna(ent_val):
            return None
        return entity_mapping[ent_type].get(ent_val)

    src, dst = [], []
    for _, row in df.iterrows():
        p_id = get_ent_id("Prefix", row["inet6num"])
        m_id = get_ent_id("Mnt", row["mnt-by"])
        if p_id and m_id:
            src.append(p_id)
            dst.append(m_id)
    g.add_edges(src, dst, etype=("Prefix", "prefix_to_mnt", "Mnt"))
    g.edges[("Prefix", "prefix_to_mnt", "Mnt")].data["weight"] = torch.ones(len(src), dtype=torch.float32,
                                                                            device=DEVICE)

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

    src, dst = [], []
    mnt_groups = df.groupby("mnt-by")["inet6num"].apply(list).to_dict()
    for mnt, prefixes in mnt_groups.items():
        if pd.isna(mnt) or len(prefixes) < 2:
            continue
        prefix_ids = [get_ent_id("Prefix", p) for p in prefixes if get_ent_id("Prefix", p)]
        for p1 in prefix_ids:
            for p2 in prefix_ids:
                if p1 != p2:
                    src.append(p1)
                    dst.append(p2)
    g.add_edges(src, dst, etype=("Prefix", "same_mnt_prefix", "Prefix"))
    g.edges[("Prefix", "same_mnt_prefix", "Prefix")].data["weight"] = torch.ones(len(src), dtype=torch.float32,
                                                                                 device=DEVICE)

    src, dst = [], []
    netname_groups = df.groupby("netname")["inet6num"].apply(list).to_dict()
    for netname, prefixes in netname_groups.items():
        if pd.isna(netname) or len(prefixes) < 2:
            continue
        prefix_ids = [get_ent_id("Prefix", p) for p in prefixes if get_ent_id("Prefix", p)]
        for p1 in prefix_ids:
            for p2 in prefix_ids:
                if p1 != p2:
                    src.append(p1)
                    dst.append(p2)
    g.add_edges(src, dst, etype=("Prefix", "same_netname_prefix", "Prefix"))
    g.edges[("Prefix", "same_netname_prefix", "Prefix")].data["weight"] = torch.ones(len(src), dtype=torch.float32,
                                                                                     device=DEVICE)

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)

    df["keywords"], keyword_weight = extract_keywords_from_descr(df["descr"])

    entity_mapping = build_entity_mapping(df, keyword_weight)
    mapping_records = []
    for ent_type, ent2id in entity_mapping.items():
        for ent_val, ent_id in ent2id.items():
            mapping_records.append({
                "entity_type": ent_type,
                "entity_value": ent_val,
                "entity_id": ent_id
            })
    pd.DataFrame(mapping_records).to_csv(OUTPUT_ENTITY_MAP, index=False, encoding="utf-8")

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
    g = dgl.heterograph({rel: ([], []) for rel in RELATION_TUPLES}, num_nodes_dict=num_nodes_dict).to(DEVICE)

    entity_features = build_entity_features(df, entity_mapping, keyword_weight)
    for ent_type, feat in entity_features.items():
        g.nodes[ent_type].data["feat"] = feat

    build_graph_edges(df, entity_mapping, g)

    g_save = g.cpu()
    dgl.save_graphs(OUTPUT_GRAPH, [g_save])
