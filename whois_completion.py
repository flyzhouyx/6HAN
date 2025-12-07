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
import os

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 64
CONFIDENCE_THRESHOLD = 0.7
MIN_CANDIDATE_FREQ = 3
STOP_WORDS = set(stopwords.words('english'))

TRAIN_HIDDEN_DIM = 16
TRAIN_NUM_HEADS = 4

class FeatureMappingLayer(torch.nn.Module):
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
    def __init__(self, edge_types: List[str], in_dim: int = 64, hidden_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.W = torch.nn.Linear(in_dim, in_dim)
        self.a = torch.nn.Linear(2 * in_dim, 1)

        self.gat_layers = torch.nn.ModuleDict()
        for etype in edge_types:
            self.gat_layers[etype] = dgl.nn.GATConv(
                in_feats=in_dim,
                out_feats=hidden_dim,
                num_heads=num_heads,
                allow_zero_in_degree=True
            )

    def forward(self, g: dgl.DGLHeteroGraph, h: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        gat_out = {}

        for stype, etype, dtype in g.canonical_etypes:
            if etype not in self.gat_layers:
                continue
            if stype not in h or dtype not in h:
                continue

            try:
                feat_src = {stype: h[stype]}
                feat_dst = {dtype: h[dtype]}
                out = self.gat_layers[etype](g[stype, etype, dtype], (feat_src, feat_dst))
                out_tensor = out[dtype].flatten(1)
            except:
                out = self.gat_layers[etype](g[stype, etype, dtype], (h[stype], h[dtype]))
                out_tensor = out.flatten(1)

            if dtype not in gat_out:
                gat_out[dtype] = []
            gat_out[dtype].append(out_tensor)

        h_out = {}
        for ntype in g.ntypes:
            if ntype in gat_out and len(gat_out[ntype]) > 0:
                h_out[ntype] = torch.stack(gat_out[ntype], dim=0).sum(dim=0)
            else:
                h_out[ntype] = torch.zeros((g.num_nodes(ntype), self.num_heads * self.hidden_dim), device=DEVICE)

        return h_out

class HANLinkPredModel(torch.nn.Module):
    def __init__(self,
                 g: dgl.DGLHeteroGraph,
                 in_feats_dict: Dict[str, int]):
        super().__init__()
        self.g = g
        self.in_feats_dict = in_feats_dict

        self.feature_mapping = FeatureMappingLayer(in_feats_dict, out_dim=64)

        edge_types = [e[1] for e in g.canonical_etypes]
        self.node_attention = NodeAttentionLayer(
            edge_types=edge_types,
            in_dim=64,
            hidden_dim=TRAIN_HIDDEN_DIM,
            num_heads=TRAIN_NUM_HEADS
        )

        self.gat2 = torch.nn.ModuleDict()
        for stype, etype, dtype in g.canonical_etypes:
            self.gat2[etype] = dgl.nn.GATConv(
                in_feats=64,
                out_feats=64,
                num_heads=1,
                allow_zero_in_degree=True
            )

    def forward(self, input_g: Optional[dgl.DGLHeteroGraph] = None) -> Dict[str, torch.Tensor]:
        g = input_g if input_g is not None else self.g

        h = self.feature_mapping(g)

        h1 = self.node_attention(g, h)

        h2 = {}
        for stype, etype, dtype in g.canonical_etypes:
            if etype not in self.gat2:
                continue
            if stype not in h1 or dtype not in h1:
                continue

            try:
                feat_src = {stype: h1[stype][:, :64]}
                feat_dst = {dtype: h1[dtype][:, :64]}
                out = self.gat2[etype](g[stype, etype, dtype], (feat_src, feat_dst))
                out_tensor = out[dtype].squeeze(1)
            except:
                out = self.gat2[etype](g[stype, etype, dtype], (h1[stype][:, :64], h1[dtype][:, :64]))
                out_tensor = out.squeeze(1)

            if dtype not in h2:
                h2[dtype] = []
            h2[dtype].append(out_tensor)

        final_embeds = {}
        for ntype in g.ntypes:
            if ntype in h2 and len(h2[ntype]) > 0:
                final_embeds[ntype] = torch.stack(h2[ntype], dim=0).sum(dim=0)
            else:
                final_embeds[ntype] = h1[ntype][:, :64]

            final_embeds[ntype] = F.normalize(final_embeds[ntype], p=2, dim=1)

        return final_embeds

class WhoisLinkPredCompleter:
    def __init__(self,
                 non_seed_csv: str,
                 entity_map: str,
                 model_path: str,
                 seed_graph_path: str,
                 output_path: str = "completed_whois.csv",
                 updated_graph_path: str = "updated_ipv6_graph.bin"):
        self.non_seed_csv = non_seed_csv
        self.entity_map = entity_map
        self.model_path = model_path
        self.seed_graph_path = seed_graph_path
        self.output_path = output_path
        self.updated_graph_path = updated_graph_path

        self.non_seed_df = None
        self.seed_graph = None
        self.temp_graph = None
        self.updated_graph = None

        self.ent_val_to_id = defaultdict(dict)
        self.ent_id_to_val = defaultdict(dict)
        self.next_ent_id = defaultdict(int)

        self.han_model = None
        self.seed_embeds = {}
        self.non_seed_embeds = {}
        self.candidate_pool = defaultdict(list)

        self.prefix_id_map = {}
        self.other_id_maps = {
            'Country': {}, 'Keyword': {}, 'Mnt': {},
            'Netname': {}, 'Status': {}
        }

    def load_seed_resources(self):
        mapping_df = pd.read_csv(self.entity_map)
        for _, row in mapping_df.iterrows():
            etype = row['entity_type']
            eval = row['entity_value']
            eid = row['entity_id']
            self.ent_val_to_id[etype][eval] = eid
            self.ent_id_to_val[etype][eid] = eval

        for etype in self.ent_val_to_id:
            self.next_ent_id[etype] = max(self.ent_val_to_id[etype].values()) + 1 if self.ent_val_to_id[etype] else 0

        graphs, _ = dgl.load_graphs(self.seed_graph_path)
        self.seed_graph = graphs[0].to(DEVICE)

        in_feats_dict = {ntype: self.seed_graph.nodes[ntype].data['feat'].shape[1] for ntype in self.seed_graph.ntypes}
        self.han_model = HANLinkPredModel(
            g=self.seed_graph,
            in_feats_dict=in_feats_dict
        ).to(DEVICE)

        state_dict = torch.load(self.model_path, map_location=DEVICE)
        self.han_model.load_state_dict(state_dict, strict=False)

        for param in self.han_model.parameters():
            param.requires_grad = False
        self.han_model.eval()

        seed_embed_path = "Data/seed_prefix_embeds.pt"
        if os.path.exists(seed_embed_path):
            self.seed_embeds['Prefix'] = torch.load(seed_embed_path, map_location=DEVICE)
            with torch.no_grad():
                all_embeds = self.han_model()
                for ntype in all_embeds:
                    if ntype != 'Prefix':
                        self.seed_embeds[ntype] = all_embeds[ntype]
        else:
            with torch.no_grad():
                self.seed_embeds = self.han_model()
            os.makedirs("Data", exist_ok=True)
            torch.save(self.seed_embeds['Prefix'], seed_embed_path)

        self._build_candidate_pool()

    def _build_candidate_pool(self):
        candidate_types = ['Mnt', 'Netname', 'Country', 'Status']

        for etype in candidate_types:
            edge_types = [e for e in self.seed_graph.canonical_etypes if e[2] == etype]
            total_counts = Counter()

            for stype, edge_type, dtype in edge_types:
                if dtype != etype:
                    continue
                try:
                    _, dst_ids = self.seed_graph.edges(etype=edge_type)
                    total_counts.update(dst_ids.cpu().numpy())
                except:
                    continue

            for eid, freq in total_counts.most_common():
                if freq >= MIN_CANDIDATE_FREQ:
                    eval = self.ent_id_to_val[etype].get(eid)
                    if eval:
                        self.candidate_pool[etype].append((eval, eid))

    def load_and_preprocess_data(self):
        self.non_seed_df = pd.read_csv(self.non_seed_csv, encoding='utf-8', low_memory=False)

        if 'inet6num' not in self.non_seed_df.columns:
            raise ValueError(f"输入CSV必须包含'inet6num'列（IPv6前缀），当前列：{self.non_seed_df.columns.tolist()}")

        self.non_seed_df = self.non_seed_df[~self.non_seed_df['inet6num'].isna()].reset_index(drop=True)

        self._basic_preprocess()

    def _basic_preprocess(self):
        str_fields = ['netname', 'country', 'mnt-by', 'status', 'descr']
        for field in str_fields:
            if field in self.non_seed_df.columns:
                self.non_seed_df[field] = self.non_seed_df[field].astype(str).str.strip().str.upper()
                self.non_seed_df.loc[self.non_seed_df[field].isin(['NAN', 'NaN', 'nan', '']), field] = np.nan

        if 'descr_cleaned' not in self.non_seed_df.columns and 'descr' in self.non_seed_df.columns:
            self.non_seed_df['descr_cleaned'] = np.nan

            mask = self.non_seed_df['descr_cleaned'].isna() & ~self.non_seed_df['descr'].isna()
            if mask.sum() > 0:
                def clean_descr(s):
                    tokens = re.split(r'[\s-]+', s.lower())
                    return ' '.join([t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) >= 3]).upper()

                self.non_seed_df.loc[mask, 'descr_cleaned'] = self.non_seed_df.loc[mask, 'descr'].apply(clean_descr)

        self.non_seed_df['to_complete'] = ''
        total_missing = 0
        for field in ['netname', 'country', 'mnt-by', 'status']:
            if field in self.non_seed_df.columns:
                mask = self.non_seed_df[field].isna()
                self.non_seed_df.loc[mask, 'to_complete'] += f'{field},'
                missing_count = mask.sum()
                total_missing += missing_count

    def build_non_seed_graph(self):
        self.prefix_id_map.clear()
        for ntype in self.other_id_maps:
            self.other_id_maps[ntype].clear()

        for idx, row in self.non_seed_df.iterrows():
            prefix = row['inet6num']
            if pd.isna(prefix):
                continue

            if prefix not in self.prefix_id_map:
                self.prefix_id_map[prefix] = len(self.prefix_id_map)

            self._assign_other_entity_ids(row)

        graph_data = defaultdict(list)
        for idx, row in self.non_seed_df.iterrows():
            prefix = row['inet6num']
            if pd.isna(prefix) or prefix not in self.prefix_id_map:
                continue

            prefix_id = self.prefix_id_map[prefix]
            self._add_edges_for_prefix(prefix_id, row, graph_data)

        num_nodes_dict = {
            'Prefix': len(self.prefix_id_map),
            'Country': len(self.other_id_maps['Country']),
            'Keyword': len(self.other_id_maps['Keyword']),
            'Mnt': len(self.other_id_maps['Mnt']),
            'Netname': len(self.other_id_maps['Netname']),
            'Status': len(self.other_id_maps['Status'])
        }

        filtered_graph_data = {}
        for (stype, etype, dtype), edges in graph_data.items():
            if edges:
                src_ids = torch.tensor([e[0] for e in edges], dtype=torch.long, device=DEVICE)
                dst_ids = torch.tensor([e[1] for e in edges], dtype=torch.long, device=DEVICE)
                filtered_graph_data[(stype, etype, dtype)] = (src_ids, dst_ids)

        self.temp_graph = dgl.heterograph(filtered_graph_data, num_nodes_dict=num_nodes_dict, device=DEVICE)

        self._add_node_features_to_temp_graph()

    def _assign_other_entity_ids(self, row):
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
        if self.temp_graph.num_nodes('Prefix') > 0:
            prefix_feats = torch.zeros((self.temp_graph.num_nodes('Prefix'), 33), device=DEVICE)

            for prefix, pid in self.prefix_id_map.items():
                if pid >= self.temp_graph.num_nodes('Prefix'):
                    continue
                row_mask = self.non_seed_df['inet6num'] == prefix
                if row_mask.sum() > 0:
                    row = self.non_seed_df[row_mask].iloc[0]
                    prefix_feats[pid] = self._generate_prefix_feat(prefix, row)

            self.temp_graph.nodes['Prefix'].data['feat'] = prefix_feats

        for ntype in ['Country', 'Mnt', 'Netname', 'Status', 'Keyword']:
            num_nodes = self.temp_graph.num_nodes(ntype)
            if num_nodes == 0:
                continue

            feats = torch.zeros((num_nodes, 2), device=DEVICE)
            for val, eid in self.other_id_maps[ntype].items():
                if eid >= num_nodes:
                    continue

                if val in self.ent_val_to_id.get(ntype, {}):
                    seed_eid = self.ent_val_to_id[ntype][val]
                    if seed_eid < self.seed_graph.num_nodes(ntype):
                        feats[eid] = self.seed_graph.nodes[ntype].data['feat'][seed_eid]
                else:
                    feats[eid] = torch.randn(2, device=DEVICE)

            self.temp_graph.nodes[ntype].data['feat'] = feats

    def _generate_prefix_feat(self, prefix: str, row: pd.Series) -> torch.Tensor:
        feat = []

        try:
            plen = int(prefix.split('/')[-1]) if '/' in str(prefix) else 0
            feat.append(torch.tensor([plen / 128.0], dtype=torch.float32, device=DEVICE))
        except:
            feat.append(torch.tensor([0.0], dtype=torch.float32, device=DEVICE))

        feat.append(torch.tensor([1.0], dtype=torch.float32, device=DEVICE))

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

        top_countries = ['US', 'CN', 'JP', 'DE', 'UK', 'FR', 'KR', 'CA', 'AU', 'IN']
        country_onehot = torch.zeros(10, dtype=torch.float32, device=DEVICE)
        if not pd.isna(row.get('country')) and row['country'] in top_countries:
            country_onehot[top_countries.index(row['country'])] = 1.0
        feat.append(country_onehot)

        top_keywords = ['INTERNET', 'SERVICE', 'PROVIDER', 'NETWORK', 'COMMUNICATION',
                        'TECHNOLOGY', 'CORPORATION', 'ORGANIZATION', 'GOVERNMENT', 'EDUCATION']
        keyword_bow = torch.zeros(10, dtype=torch.float32, device=DEVICE)
        if not pd.isna(row.get('descr_cleaned')):
            desc = row['descr_cleaned']
            for i, kw in enumerate(top_keywords):
                if kw in desc:
                    keyword_bow[i] = 1.0
        feat.append(keyword_bow)

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
        if self.temp_graph is None or self.temp_graph.num_nodes('Prefix') == 0:
            return

        with torch.no_grad():
            self.non_seed_embeds = self.han_model(input_g=self.temp_graph)

        if 'Prefix' not in self.non_seed_embeds or len(self.non_seed_embeds['Prefix']) == 0:
            return

        prefix_embeds = self.non_seed_embeds['Prefix']

        non_seed_embed_path = "Data/non_seed_prefix_embeds.pt"
        os.makedirs("Data", exist_ok=True)
        torch.save(prefix_embeds, non_seed_embed_path)

        complete_mapping = {
            'mnt-by': ('Mnt', 'prefix_to_mnt'),
            'netname': ('Netname', 'prefix_to_netname'),
            'country': ('Country', 'prefix_to_country'),
            'status': ('Status', 'prefix_to_status')
        }

        total_completed = 0
        for field, (etype, edge_type) in complete_mapping.items():
            if field not in self.non_seed_df.columns:
                continue
            if etype not in self.candidate_pool or len(self.candidate_pool[etype]) == 0:
                continue
            completed = self._complete_single_field(field, etype, edge_type, prefix_embeds)
            total_completed += completed

    def _complete_single_field(self, field: str, etype: str, edge_type: str, prefix_embeds: torch.Tensor) -> int:
        mask = self.non_seed_df[field].isna()
        if not mask.any():
            return 0

        candidate_vals = [c[0] for c in self.candidate_pool[etype]]
        candidate_ids = [c[1] for c in self.candidate_pool[etype]]

        if etype not in self.seed_embeds or len(candidate_ids) == 0:
            return 0

        candidate_embeds = self.seed_embeds[etype][candidate_ids].to(DEVICE)
        completed = 0

        for idx in self.non_seed_df[mask].index:
            row = self.non_seed_df.iloc[idx]
            prefix = row['inet6num']

            if prefix not in self.prefix_id_map:
                continue
            prefix_id = self.prefix_id_map[prefix]
            if prefix_id >= len(prefix_embeds):
                continue

            p_embed = prefix_embeds[prefix_id:prefix_id + 1]
            dot_products = torch.matmul(p_embed, candidate_embeds.T).squeeze(0)
            probs = torch.sigmoid(dot_products).cpu().numpy()

            max_idx = np.argmax(probs)
            max_prob = probs[max_idx]

            if max_prob > CONFIDENCE_THRESHOLD:
                best_candidate = candidate_vals[max_idx]
                if self._consistency_check(row, field, best_candidate):
                    self.non_seed_df.at[idx, field] = best_candidate
                    self.non_seed_df.at[idx, f'{field}_confidence'] = float(max_prob)
                    completed += 1

        return completed

    def _consistency_check(self, row: pd.Series, field: str, candidate: str) -> bool:
        if field == 'mnt-by':
            country = row.get('country')
            if not pd.isna(country) and country != '':
                country_code = re.findall(r'^[A-Z]{2}', candidate)
                if country_code and country_code[0] != country:
                    return False

        if field == 'status':
            valid_status = ['ALLOCATED', 'ASSIGNED', 'RESERVED']
            if not any(vs in candidate for vs in valid_status):
                return False

        return True

    def update_hetero_graph(self):
        if self.temp_graph is None or self.seed_graph is None:
            return

        try:
            for stype, etype, dtype in self.temp_graph.canonical_etypes:
                edge_key = (stype, etype, dtype)
                num_edges = self.temp_graph.num_edges(edge_key)
                self.temp_graph.edges[edge_key].data['weight'] = torch.ones(num_edges, device=DEVICE)

            for stype, etype, dtype in self.seed_graph.canonical_etypes:
                edge_key = (stype, etype, dtype)
                if 'weight' in self.seed_graph.edges[edge_key].data:
                    self.seed_graph.edges[edge_key].data['weight'] = self.seed_graph.edges[edge_key].data[
                        'weight'].float()

            self.updated_graph = dgl.merge([self.seed_graph, self.temp_graph])

            os.makedirs(os.path.dirname(self.updated_graph_path), exist_ok=True)
            dgl.save_graphs(self.updated_graph_path, [self.updated_graph])

        except Exception as e:
            try:
                dgl.save_graphs(self.updated_graph_path, [self.temp_graph])
            except Exception as e2:
                pass

    def save_results(self):
        stats = []
        total_records = len(self.non_seed_df)
        for field in ['netname', 'country', 'mnt-by', 'status']:
            if field not in self.non_seed_df.columns:
                continue
            missing_before = self.non_seed_df[field].isna().sum()
            missing_after = self.non_seed_df[field].isna().sum()
            completed = missing_before - missing_after
            completion_rate = (completed / total_records) * 100 if total_records > 0 else 0
            stats.append({
                '字段': field,
                '总记录数': total_records,
                '补全数': completed,
                '补全率': f'{completion_rate:.1f}%'
            })

        try:
            self.non_seed_df.to_csv(self.output_path, index=False, encoding='utf-8')
        except Exception as e:
            pass

    def run(self):
        try:
            self.load_seed_resources()
            self.load_and_preprocess_data()
            self.build_non_seed_graph()
            self.predict_missing_edges()
            self.update_hetero_graph()
            self.save_results()
        except Exception as e:
            import traceback
            traceback.print_exc()
            if self.non_seed_df is not None:
                self.non_seed_df.to_csv(f"partial_{self.output_path}", index=False, encoding='utf-8')
            raise RuntimeError(f"补全流程失败：{str(e)}")

if __name__ == "__main__":
    config = {
        "non_seed_csv": "Data/non_seed_prefixes_with_info.csv",
        "entity_map": "Data/entity_mapping.csv",
        "model_path": "Data/6han_model.pth",
        "seed_graph_path": "Data/ipv6_hetero_graph.bin",
        "output_path": "Data/completed_whois.csv",
        "updated_graph_path": "Data/updated_ipv6_graph.bin"
    }

    completer = WhoisLinkPredCompleter(**config)
    completer.run()
