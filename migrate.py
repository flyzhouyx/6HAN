import pandas as pd
import json
import ipaddress
from typing import Dict, List, Optional, Tuple

# 配置项
SIMILARITY_PATH = "Data/similarity_retrieval_result.csv"  # 相似检索结果
SEED_PATTERN_PATH = "Data/ipv6_patterns_result.json"    # 种子模式库
OUTPUT_PATH = "Data/non_seed_patterns_result.json"      # 非种子模式输出路径

# 迁移参数
MAX_RETRY_RANK = 5  # 最大降级匹配rank（若rank=1无模式，最多检查到rank=5）
EXAMPLE_COUNT = 3   # 每个模式生成的示例地址数量

def get_prefix_fixed_part(ipv6_prefix: str) -> Tuple[str, int]:
    """解析IPv6前缀，返回32位固定部分字符串和固定位数L"""
    try:
        prefix_str, len_str = ipv6_prefix.split("/")
        prefix_len = int(len_str)
        full_ip = ipaddress.IPv6Address(prefix_str).exploded
        full_ip_no_colon = full_ip.replace(":", "")
        fixed_len = prefix_len // 4
        fixed_part = full_ip_no_colon[:fixed_len]
        return fixed_part, fixed_len
    except Exception as e:
        print(f"⚠️ 解析前缀[{ipv6_prefix}]失败：{e}，跳过该前缀")
        return "", 0


def match_seed_pattern(non_seed_prefix: str, similarity_df: pd.DataFrame,
                       seed_patterns: Dict[str, List[Dict]]) -> Optional[Tuple[str, List[Dict], str]]:
    """
    为非种子前缀匹配种子模式，新增返回migration_suggestion
    返回：(匹配的种子前缀, 种子模式列表, migration_suggestion)，无匹配则返回None
    """
    non_seed_similar = similarity_df[similarity_df["non_seed_prefix"] == non_seed_prefix].sort_values("rank")

    for _, row in non_seed_similar.iterrows():
        seed_prefix = row["similar_seed_prefix"]
        seed_rank = row["rank"]
        # 提取当前行的migration_suggestion
        migration_suggestion = row.get("migration_suggestion", "")

        if seed_prefix in seed_patterns and len(seed_patterns[seed_prefix]) > 0:
            print(
                f"✅ 非种子[{non_seed_prefix}]匹配到种子[{seed_prefix}]（rank={seed_rank}），共{len(seed_patterns[seed_prefix])}个模式")
            return seed_prefix, seed_patterns[seed_prefix], migration_suggestion

        if seed_rank >= MAX_RETRY_RANK:
            break

    print(f"⚠️ 非种子[{non_seed_prefix}]未匹配到有效种子模式（已尝试前{MAX_RETRY_RANK}个rank）")
    return None, None, None


def migrate_pattern(non_seed_prefix: str, seed_patterns: List[Dict], fixed_part: str, fixed_len: int) -> List[Dict]:
    """将种子模式迁移到非种子前缀：替换固定部分，生成示例地址"""
    non_seed_patterns = []

    for seed_pattern_info in seed_patterns:
        seed_pattern = seed_pattern_info["pattern"]
        seed_count = seed_pattern_info["count"]

        if len(seed_pattern) != 32:
            print(f"⚠️ 种子模式[{seed_pattern}]长度异常（非32位），跳过")
            continue

        migrated_pattern = fixed_part + seed_pattern[fixed_len:]

        example_addresses = []
        for i in range(EXAMPLE_COUNT):
            replace_val = ["1", "2", "d"][i % EXAMPLE_COUNT]
            addr_no_colon = migrated_pattern.replace("*", replace_val)
            if len(addr_no_colon) < 32:
                addr_no_colon = addr_no_colon.ljust(32, "0")
            try:
                addr_segments = [addr_no_colon[i * 4:(i + 1) * 4] for i in range(8)]
                addr_with_colon = ":".join(addr_segments)
                standard_addr = ipaddress.IPv6Address(addr_with_colon).compressed
                example_addresses.append(standard_addr)
            except Exception as e:
                print(f"⚠️ 生成示例地址失败（模式：{migrated_pattern}）：{e}")
                example_addresses.append(f"invalid_addr_{i + 1}")

        non_seed_patterns.append({
            "pattern": migrated_pattern,
            "addresses": example_addresses,
            "count": seed_count,
            "source_seed_pattern": seed_pattern
        })

    return non_seed_patterns


def load_input_data() -> Tuple[pd.DataFrame, Dict[str, List[Dict]]]:
    """读取相似检索结果和种子模式库，确保包含migration_suggestion列"""
    similarity_df = pd.read_csv(SIMILARITY_PATH).drop_duplicates(subset=["non_seed_prefix", "rank"])
    # 检查是否存在migration_suggestion列
    if "migration_suggestion" not in similarity_df.columns:
        raise ValueError("similarity_retrieval_result.csv中未找到migration_suggestion列")
    print(f"✅ 加载相似检索结果：共{len(similarity_df)}条记录，{similarity_df['non_seed_prefix'].nunique()}个非种子前缀")

    with open(SEED_PATTERN_PATH, "r", encoding="utf-8") as f:
        seed_patterns = json.load(f)
    print(
        f"✅ 加载种子模式库：共{len(seed_patterns)}个种子前缀，{sum(len(pats) for pats in seed_patterns.values())}个模式")

    return similarity_df, seed_patterns


def batch_migrate_patterns() -> Dict[str, Dict]:
    """批量为所有非种子前缀执行模式迁移，新增收集migration_suggestion"""
    similarity_df, seed_patterns = load_input_data()
    non_seed_result = {}

    unique_non_seeds = similarity_df["non_seed_prefix"].unique()
    for non_seed in unique_non_seeds:
        print(f"\n=== 处理非种子前缀：{non_seed} ===")

        fixed_part, fixed_len = get_prefix_fixed_part(non_seed)
        if not fixed_part or fixed_len == 0:
            continue

        # 获取匹配的种子模式及对应的migration_suggestion
        matched_seed, matched_patterns, migration_suggestion = match_seed_pattern(non_seed, similarity_df, seed_patterns)
        if not matched_seed or not matched_patterns:
            continue

        migrated_patterns = migrate_pattern(non_seed, matched_patterns, fixed_part, fixed_len)
        if not migrated_patterns:
            print(f"⚠️ 非种子[{non_seed}]未生成有效模式")
            continue

        # 结果中添加migration_suggestion
        non_seed_result[non_seed] = {
            "migrated_patterns": migrated_patterns,
            "source_seed_prefix": matched_seed,
            "prefix_fixed_part": fixed_part,
            "prefix_fixed_length": fixed_len,
            "migration_suggestion": migration_suggestion  # 新增字段
        }

    print(f"\n✅ 模式迁移完成：{len(non_seed_result)}个非种子前缀生成有效模式")
    return non_seed_result


def save_migrated_patterns(non_seed_result: Dict[str, Dict]):
    """保存非种子模式结果到JSON文件，包含migration_suggestion"""
    output_data = {}
    for non_seed, info in non_seed_result.items():
        patterns = [
            {k: v for k, v in pat.items() if k != "source_seed_pattern"}
            for pat in info["migrated_patterns"]
        ]
        output_data[non_seed] = {
            "patterns": patterns,
            "metadata": {
                "source_seed_prefix": info["source_seed_prefix"],
                "prefix_fixed_part": info["prefix_fixed_part"],
                "prefix_fixed_length": info["prefix_fixed_length"],
                "migration_suggestion": info["migration_suggestion"]  # 写入结果文件
            }
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 非种子模式结果已保存至：{OUTPUT_PATH}")
    print(
        f"📊 结果统计：共{len(output_data)}个非种子前缀，{sum(len(info['patterns']) for info in output_data.values())}个模式")

if __name__ == "__main__":
    non_seed_pattern_result = batch_migrate_patterns()
    if non_seed_pattern_result:
        save_migrated_patterns(non_seed_pattern_result)
    else:
        print("❌ 未生成任何非种子模式，无需保存")