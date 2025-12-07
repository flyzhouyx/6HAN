import pandas as pd
import numpy as np
import re
import json
import math
import os
import random
import torch
import torch.nn.functional as F
import ipaddress
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
from typing import Tuple, List, Dict, Set, DefaultDict, Optional

warnings.filterwarnings('ignore')

class Config:
    ETA = 5
    TOTAL_GENERATION_BUDGET = 1000000
    SINGLE_PATTERN_LIMIT = 5000
    MAX_GENERATION_ROUNDS = 10
    DUPLICATE_CHECK_BATCH = 50000
    IPV6_NIBBLES = 32
    HEX_CHARS = '0123456789abcdefABCDEF'
    WILDCARD = '*'
    SEED_CSV_PREFIX_COL = 'inet6num'
    NON_SEED_CSV_PREFIX_COL = 'inet6num'
    EMBEDDING_DIM = 64
    TOP_K_SIMILAR = 3
    SIMILARITY_THRESHOLD = 0.1
    FORCE_TOP_K = True
    BATCH_SIZE = 500
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED_EMBEDS_PATH = "Data/seed_prefix_embeds.pt"
    NON_SEED_EMBEDS_PATH = "Data/non_seed_prefix_embeds.pt"

def preprocess_ipv6(ip: Optional[str]) -> str:
    if pd.isna(ip) or ip == '':
        return '0' * 32
    ip = str(ip).strip()
    if '/' in ip:
        ip = ip.split('/')[0]
    try:
        ip_obj = ipaddress.IPv6Address(ip)
        parts = ip_obj.exploded.split(':')
        padded_parts = [part.zfill(4) for part in parts]
        full_ip = ''.join(padded_parts)
        return full_ip.ljust(32, '0')[:32]
    except:
        return '0' * 32

def calculate_entropy(full_ips: List[str], position: int) -> float:
    if not full_ips:
        return 0.0
    char_counts = Counter([ip[position] for ip in full_ips if len(ip) == 32])
    total = sum(char_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        entropy -= p * math.log2(p) if p > 0 else 0
    return entropy

def get_variable_positions(full_ips: List[str]) -> List[int]:
    entropies = [(i, calculate_entropy(full_ips, i)) for i in range(32)]
    return [i for i, entropy in entropies if entropy > 0.1]

def create_pattern(full_ips: List[str], variable_positions: List[int]) -> str:
    if not full_ips:
        return '0' * 32
    base = list(full_ips[0])
    for pos in variable_positions:
        if pos < 32:
            base[pos] = Config.WILDCARD
    return ''.join(base)

def cluster_ips(full_ips: List[str], variable_positions: List[int], eta: int) -> List[Tuple[str, List[str]]]:
    if len(full_ips) <= eta or not variable_positions:
        current_vars = get_variable_positions(full_ips)
        pattern = create_pattern(full_ips, current_vars)
        return [(pattern, full_ips)]
    split_pos = variable_positions[0]
    clusters = defaultdict(list)
    for ip in full_ips:
        char = ip[split_pos] if split_pos < len(ip) else '0'
        clusters[char].append(ip)
    result = []
    new_vars = variable_positions[1:]
    for cluster in clusters.values():
        result.extend(cluster_ips(cluster, new_vars, eta))
    return result

def pattern_mining(addr_set: Set[str], eta: int) -> List[Dict[str, any]]:
    full_ips = [preprocess_ipv6(ip) for ip in addr_set if ip]
    if not full_ips:
        return []
    variable_positions = get_variable_positions(full_ips)
    clusters = cluster_ips(full_ips, variable_positions, eta)
    pattern_set = []
    for pattern, ips in clusters:
        if len(ips) == 1 and Config.WILDCARD not in pattern:
            continue
        original_ips = [next(ip for ip in addr_set if preprocess_ipv6(ip) == full_ip) for full_ip in ips]
        pattern_set.append({
            "pattern": pattern,
            "addresses": original_ips,
            "count": len(original_ips)
        })
    return pattern_set

def nibbles_to_ipv6(nibbles_str: str) -> Optional[str]:
    if len(nibbles_str) != 32:
        return None
    try:
        segments = [nibbles_str[i:i + 4] for i in range(0, 32, 4)]
        ipv6_str = ':'.join(segments)
        return str(ipaddress.IPv6Address(ipv6_str))
    except:
        return None

class PrefixSimilarityRetriever:
    def __init__(self,
                 seed_prefixes_csv: str,
                 non_seed_prefixes_csv: str,
                 seed_embeds_path: str = Config.SEED_EMBEDS_PATH,
                 non_seed_embeds_path: str = Config.NON_SEED_EMBEDS_PATH,
                 output_mapping_path: str = "Data/similarity_mapping.json"):

        self.seed_prefixes_csv = seed_prefixes_csv
        self.non_seed_prefixes_csv = non_seed_prefixes_csv
        self.seed_embeds_path = seed_embeds_path
        self.non_seed_embeds_path = non_seed_embeds_path
        self.output_mapping_path = output_mapping_path

        self.seed_prefixes: List[str] = []
        self.non_seed_prefixes: List[str] = []
        self.seed_embeds: Optional[torch.Tensor] = None
        self.non_seed_embeds: Optional[torch.Tensor] = None
        self.similarity_mapping: Dict[str, List[str]] = {}

    def load_prefixes(self) -> None:
        df_seed = pd.read_csv(self.seed_prefixes_csv, encoding='utf-8')
        if Config.SEED_CSV_PREFIX_COL not in df_seed.columns:
            raise ValueError(f"种子CSV无{Config.SEED_CSV_PREFIX_COL}列")

        raw_seed_prefixes = df_seed[Config.SEED_CSV_PREFIX_COL].dropna().unique().tolist()
        valid_seed_prefixes = []
        for prefix in raw_seed_prefixes:
            prefix_str = str(prefix).strip()
            if prefix_str and '/' in prefix_str:
                try:
                    ipaddress.IPv6Network(prefix_str, strict=False)
                    valid_seed_prefixes.append(prefix_str)
                except:
                    continue
        self.seed_prefixes = valid_seed_prefixes

        df_non_seed = pd.read_csv(self.non_seed_prefixes_csv, encoding='utf-8', low_memory=False)
        possible_cols = [Config.NON_SEED_CSV_PREFIX_COL, 'prefix', 'IPv6_Prefix', 'inet6num_prefix']
        prefix_col = next((col for col in possible_cols if col in df_non_seed.columns), None)
        if prefix_col is None:
            raise ValueError("非种子CSV无前缀列")

        raw_non_seed_prefixes = df_non_seed[prefix_col].dropna().unique().tolist()
        valid_non_seed_prefixes = []
        for prefix in raw_non_seed_prefixes:
            prefix_str = str(prefix).strip()
            if prefix_str:
                valid_non_seed_prefixes.append(prefix_str)
        self.non_seed_prefixes = valid_non_seed_prefixes

    def load_embeddings(self) -> None:
        self.seed_embeds = torch.load(self.seed_embeds_path, map_location=Config.DEVICE)
        if len(self.seed_embeds) != len(self.seed_prefixes):
            full_seed_embeds = torch.zeros((len(self.seed_prefixes), Config.EMBEDDING_DIM), device=Config.DEVICE)
            full_seed_embeds[:len(self.seed_embeds)] = self.seed_embeds
            seed_mean = self.seed_embeds.mean(dim=0, keepdim=True)
            full_seed_embeds[len(self.seed_embeds):] = seed_mean
            self.seed_embeds = full_seed_embeds

        self.non_seed_embeds = torch.load(self.non_seed_embeds_path, map_location=Config.DEVICE)
        if len(self.non_seed_embeds) != len(self.non_seed_prefixes):
            full_non_seed_embeds = torch.zeros((len(self.non_seed_prefixes), Config.EMBEDDING_DIM),
                                               device=Config.DEVICE)
            full_non_seed_embeds[:len(self.non_seed_embeds)] = self.non_seed_embeds
            non_seed_mean = self.non_seed_embeds.mean(dim=0, keepdim=True) if len(
                self.non_seed_embeds) > 0 else torch.zeros(1, Config.EMBEDDING_DIM, device=Config.DEVICE)
            full_non_seed_embeds[len(self.non_seed_embeds):] = non_seed_mean
            self.non_seed_embeds = full_non_seed_embeds

        self.seed_embeds = F.normalize(self.seed_embeds, p=2, dim=1)
        self.non_seed_embeds = F.normalize(self.non_seed_embeds, p=2, dim=1)

    def calculate_similarity(self) -> None:
        num_batches = (len(self.non_seed_prefixes) + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
        for batch_idx in tqdm(range(num_batches), desc="匹配相似前缀（分批次）"):
            start = batch_idx * Config.BATCH_SIZE
            end = min((batch_idx + 1) * Config.BATCH_SIZE, len(self.non_seed_prefixes))
            batch_non_seed_embeds = self.non_seed_embeds[start:end]

            similarity_matrix = F.cosine_similarity(
                batch_non_seed_embeds.unsqueeze(1),
                self.seed_embeds.unsqueeze(0),
                dim=2
            )

            for i in range(batch_non_seed_embeds.shape[0]):
                non_seed_idx = start + i
                non_seed_prefix = self.non_seed_prefixes[non_seed_idx]
                sim_scores = similarity_matrix[i].cpu().numpy()

                sorted_indices = np.argsort(sim_scores)[::-1]
                top_seed_prefixes = []

                if Config.FORCE_TOP_K:
                    top_indices = sorted_indices[:Config.TOP_K_SIMILAR]
                    top_seed_prefixes = [self.seed_prefixes[idx] for idx in top_indices if
                                         idx < len(self.seed_prefixes)]
                else:
                    for idx in sorted_indices[:Config.TOP_K_SIMILAR]:
                        if sim_scores[idx] >= Config.SIMILARITY_THRESHOLD:
                            top_seed_prefixes.append(self.seed_prefixes[idx])

                self.similarity_mapping[non_seed_prefix] = top_seed_prefixes

    def save_similarity_mapping(self) -> None:
        os.makedirs(os.path.dirname(self.output_mapping_path), exist_ok=True)
        with open(self.output_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.similarity_mapping, f, indent=2, ensure_ascii=False)

    def run(self) -> None:
        self.load_prefixes()
        self.load_embeddings()
        self.calculate_similarity()
        self.save_similarity_mapping()

class SeedPatternMiner:
    def __init__(self, matched_prefixes_path: str, output_patterns_path: str):
        self.matched_prefixes_path = matched_prefixes_path
        self.output_patterns_path = output_patterns_path
        self.matched_data: Dict[str, List[str]] = {}
        self.seed_patterns: DefaultDict[str, List[str]] = defaultdict(list)

    def load_matched_data(self) -> None:
        if not os.path.exists(self.matched_prefixes_path):
            raise FileNotFoundError(f"匹配数据不存在：{self.matched_prefixes_path}")
        with open(self.matched_prefixes_path, 'r', encoding='utf-8') as f:
            self.matched_data = json.load(f)
        self.matched_data = {p: addr for p, addr in self.matched_data.items() if len(addr) > 0}

    def mine_patterns(self) -> None:
        for prefix, addresses in tqdm(self.matched_data.items(), desc="挖掘前缀模式"):
            if len(addresses) > 10000:
                addresses = random.sample(addresses, 10000)
            patterns_info = pattern_mining(set(addresses), Config.ETA)
            patterns = [p_info["pattern"] for p_info in patterns_info]
            self.seed_patterns[prefix] = list(set(patterns))

    def save_pattern_library(self) -> None:
        output_data: Dict[str, List[Dict[str, any]]] = {}
        for prefix, patterns in self.seed_patterns.items():
            pattern_info_list = []
            for pattern in patterns:
                pattern_info_list.append({
                    "pattern": pattern,
                    "wildcard_count": pattern.count(Config.WILDCARD),
                    "prefix": prefix
                })
            output_data[prefix] = pattern_info_list

        with open(self.output_patterns_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        csv_data = []
        for prefix, patterns_info in output_data.items():
            for p_info in patterns_info:
                csv_data.append({
                    "seed_prefix": prefix,
                    "pattern": p_info["pattern"],
                    "wildcard_count": p_info["wildcard_count"],
                    "prefix_length": int(prefix.split('/')[1]) if '/' in prefix else 0
                })
        df = pd.DataFrame(csv_data)
        csv_path = self.output_patterns_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')

    def run(self) -> None:
        self.load_matched_data()
        self.mine_patterns()
        self.save_pattern_library()

class PatternMigrationGenerator:
    def __init__(self,
                 seed_patterns_path: str,
                 non_seed_prefixes_csv: str,
                 similarity_mapping_path: str,
                 output_targets_path: str):

        self.seed_patterns_path = seed_patterns_path
        self.non_seed_prefixes_csv = non_seed_prefixes_csv
        self.similarity_mapping_path = similarity_mapping_path
        self.output_targets_path = output_targets_path

        self.seed_patterns: DefaultDict[str, List[str]] = defaultdict(list)
        self.non_seed_prefixes: List[str] = []
        self.similarity_mapping: Dict[str, List[str]] = {}
        self.migrated_patterns: DefaultDict[str, List[str]] = defaultdict(list)
        self.generated_targets: Set[str] = set()
        self.generation_stats: Dict[str, any] = {
            "rounds": [],
            "total_generated": 0,
            "total_valid": 0,
            "duplicates_removed": 0
        }

    def load_seed_patterns(self) -> None:
        csv_path = self.seed_patterns_path.replace('.json', '.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.seed_patterns[row['seed_prefix']].append(row['pattern'])
        else:
            with open(self.seed_patterns_path, 'r', encoding='utf-8') as f:
                pattern_data = json.load(f)
            for seed_prefix, patterns_info in pattern_data.items():
                self.seed_patterns[seed_prefix] = [p['pattern'] for p in patterns_info]

        for prefix in self.seed_patterns:
            self.seed_patterns[prefix] = list(set(self.seed_patterns[prefix]))

    def load_non_seed_prefixes(self) -> None:
        df = pd.read_csv(self.non_seed_prefixes_csv, encoding='utf-8', low_memory=False)
        possible_cols = [Config.NON_SEED_CSV_PREFIX_COL, 'prefix', 'IPv6_Prefix']
        prefix_col = next((col for col in possible_cols if col in df.columns), None)
        if prefix_col is None:
            raise ValueError("CSV中未找到前缀列")

        raw_prefixes = df[prefix_col].dropna().unique().tolist()
        valid_prefixes = []
        for prefix in raw_prefixes:
            prefix_str = str(prefix).strip()
            if prefix_str:
                valid_prefixes.append(prefix_str)
        self.non_seed_prefixes = valid_prefixes

    def load_similarity_mapping(self) -> None:
        with open(self.similarity_mapping_path, 'r', encoding='utf-8') as f:
            self.similarity_mapping = json.load(f)
        self.similarity_mapping = {k: v for k, v in self.similarity_mapping.items()
                                   if k in self.non_seed_prefixes and len(v) > 0}

    def migrate_patterns(self) -> None:
        for non_seed_prefix in tqdm(self.similarity_mapping.keys(), desc="模式迁移"):
            try:
                if '/' not in non_seed_prefix:
                    non_seed_prefix = f"{non_seed_prefix}/64"
                non_seed_net = ipaddress.IPv6Network(non_seed_prefix, strict=False)
                non_seed_len = non_seed_net.prefixlen
                non_seed_nibble_len = non_seed_len // 4
                non_seed_full = preprocess_ipv6(str(non_seed_net.network_address))
            except:
                non_seed_nibble_len = 16
                non_seed_full = '0' * 32

            similar_seed_prefixes = self.similarity_mapping[non_seed_prefix]
            migrated_patterns = []

            for seed_prefix in similar_seed_prefixes:
                if seed_prefix not in self.seed_patterns:
                    continue
                for seed_pattern in self.seed_patterns[seed_prefix]:
                    if len(seed_pattern) != 32:
                        continue
                    migrated_pattern = list(seed_pattern)
                    for i in range(min(non_seed_nibble_len, 32)):
                        migrated_pattern[i] = non_seed_full[i]
                    migrated_patterns.append(''.join(migrated_pattern))

            self.migrated_patterns[non_seed_prefix] = list(set(migrated_patterns))

    def generate_target_addresses(self) -> None:
        valid_non_seed_count = len(self.migrated_patterns)
        if valid_non_seed_count == 0:
            for i in range(min(100, Config.TOTAL_GENERATION_BUDGET)):
                self.generated_targets.add(f"2001:db8::{i}")
            return

        total_generated = 0
        round_num = 1

        while total_generated < Config.TOTAL_GENERATION_BUDGET and round_num <= Config.MAX_GENERATION_ROUNDS:
            remaining_budget = Config.TOTAL_GENERATION_BUDGET - total_generated
            single_prefix_quota = remaining_budget // valid_non_seed_count
            if single_prefix_quota <= 0:
                single_prefix_quota = 1

            round_generated = 0
            round_duplicates = 0

            for non_seed_prefix in tqdm(self.migrated_patterns.keys(), desc=f"第{round_num}轮生成地址"):
                if round_generated >= remaining_budget:
                    break

                migrated_patterns = self.migrated_patterns[non_seed_prefix]
                if not migrated_patterns:
                    continue

                prefix_remaining = min(single_prefix_quota, remaining_budget - round_generated)
                if prefix_remaining <= 0:
                    break

                patterns_count = max(1, len(migrated_patterns))
                per_pattern_quota = prefix_remaining // patterns_count

                for pattern in migrated_patterns:
                    if prefix_remaining <= 0 or round_generated >= remaining_budget:
                        break

                    pattern_generate_count = min(
                        Config.SINGLE_PATTERN_LIMIT,
                        per_pattern_quota,
                        remaining_budget - round_generated
                    )
                    if pattern_generate_count <= 0:
                        continue

                    generated, duplicates = self._generate_from_pattern(pattern, pattern_generate_count)
                    round_generated += generated
                    round_duplicates += duplicates
                    prefix_remaining -= generated

            total_generated = len(self.generated_targets)
            self.generation_stats["rounds"].append({
                "round": round_num,
                "generated": round_generated,
                "duplicates": round_duplicates,
                "total_after_round": total_generated
            })

            round_num += 1

        final_valid: List[str] = []
        for ip in self.generated_targets:
            try:
                ipaddress.IPv6Address(ip)
                final_valid.append(ip)
            except:
                continue

        if len(final_valid) > Config.TOTAL_GENERATION_BUDGET:
            final_valid = random.sample(final_valid, Config.TOTAL_GENERATION_BUDGET)

        self.generated_targets = set(final_valid)
        self.generation_stats["total_generated"] = len(self.generated_targets)
        self.generation_stats["total_valid"] = len(final_valid)
        self.generation_stats["duplicates_removed"] = len(self.generated_targets) - len(final_valid)

    def _generate_from_pattern(self, pattern: str, count: int) -> Tuple[int, int]:
        if len(pattern) != 32:
            return 0, 0

        wildcard_positions = [i for i, c in enumerate(pattern) if c == Config.WILDCARD]
        generated_count = 0
        duplicate_count = 0

        for _ in range(count):
            generated_pattern = list(pattern)
            for pos in wildcard_positions:
                generated_pattern[pos] = random.choice(Config.HEX_CHARS[:16]).lower()
            ip_str = nibbles_to_ipv6(''.join(generated_pattern))

            if ip_str:
                if ip_str in self.generated_targets:
                    duplicate_count += 1
                else:
                    self.generated_targets.add(ip_str)
                    generated_count += 1

            if len(self.generated_targets) % Config.DUPLICATE_CHECK_BATCH == 0:
                pass

        return generated_count, duplicate_count

    def save_target_addresses(self) -> None:
        final_targets = list(self.generated_targets)
        random.shuffle(final_targets)

        with open(self.output_targets_path, 'w', encoding='utf-8') as f:
            for ip in final_targets:
                f.write(f"{ip}\n")

        stats = {
            'config': {
                'total_budget': Config.TOTAL_GENERATION_BUDGET,
                'max_rounds': Config.MAX_GENERATION_ROUNDS,
                'single_pattern_limit': Config.SINGLE_PATTERN_LIMIT,
                'duplicate_check_batch': Config.DUPLICATE_CHECK_BATCH
            },
            'generation': self.generation_stats,
            'pattern_stats': {
                'migrated_pattern_count': sum(len(p) for p in self.migrated_patterns.values()),
                'non_seed_prefix_count': len(self.migrated_patterns),
                'seed_pattern_count': sum(len(p) for p in self.seed_patterns.values())
            },
            'final_result': {
                'total_valid_addresses': len(final_targets),
                'meets_budget': len(final_targets) >= Config.TOTAL_GENERATION_BUDGET,
                'source_non_seed_csv': self.non_seed_prefixes_csv
            }
        }

        stats_path = self.output_targets_path.replace('.txt', '_detailed_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

    def run(self) -> None:
        self.load_seed_patterns()
        self.load_non_seed_prefixes()
        self.load_similarity_mapping()
        self.migrate_patterns()
        self.generate_target_addresses()
        self.save_target_addresses()

def main() -> None:
    os.makedirs("Data", exist_ok=True)

    retriever = PrefixSimilarityRetriever(
        seed_prefixes_csv="Data/seed_prefixes_with_info.csv",
        non_seed_prefixes_csv="Data/completed_whois.csv",
        seed_embeds_path="Data/seed_prefix_embeds.pt",
        non_seed_embeds_path="Data/non_seed_prefix_embeds.pt",
        output_mapping_path="Data/similarity_mapping.json"
    )
    retriever.run()

    migration_generator = PatternMigrationGenerator(
        seed_patterns_path="Data/ipv6_patterns_result.json",
        non_seed_prefixes_csv="Data/completed_whois.csv",
        similarity_mapping_path="Data/similarity_mapping.json",
        output_targets_path="Data/target_ipv6_addresses.txt"
    )
    migration_generator.run()

if __name__ == "__main__":
    main()