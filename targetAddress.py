import json
import ipaddress
import sys
from typing import Dict, List, Tuple

# 递归深度与核心配置
sys.setrecursionlimit(1000)
PATTERN_PATH = "Data/non_seed_patterns_result.json"
OUTPUT_PATH = "Data/ipv6_target_addresses_6.txt"  # 改为TXT格式
TOTAL_BUDGET = 50_000_000
MAX_WILDCARDS = 8
HEX_CHARS = "0123456789abcdef"


# 以下函数完全复用之前的递归逻辑，无修改
def count_wildcards(pattern: str) -> int:
    if len(pattern) != 32:
        print(f"⚠️ 模式[{pattern[:20]}...]长度异常（需32位），跳过")
        return -1
    return pattern.count("*")


def recursive_generate_with_quota(current: str, remaining: int, quota: int, result: List[str]) -> None:
    if len(result) >= quota:
        return
    if remaining == 0:
        result.append(current)
        return
    for c in HEX_CHARS:
        recursive_generate_with_quota(current + c, remaining - 1, quota, result)
        if len(result) >= quota:
            break


def generate_wildcard_combinations(num_wildcards: int, quota: int) -> List[str]:
    if num_wildcards > MAX_WILDCARDS:
        print(f"⚠️ 通配符数量({num_wildcards})超过上限({MAX_WILDCARDS})，跳过")
        return []
    max_possible = min(quota, len(HEX_CHARS) ** num_wildcards)
    if max_possible <= 0:
        return []
    result = []
    recursive_generate_with_quota("", num_wildcards, max_possible, result)
    return result[:quota]


def calculate_prefix_quota(patterns_data: Dict[str, Dict]) -> int:
    num_prefixes = len(patterns_data)
    if num_prefixes == 0:
        return 0
    quota = TOTAL_BUDGET // num_prefixes
    print(f"✅ 预算分配完成：")
    print(f"   - 总预算：{TOTAL_BUDGET:,} 个地址")
    print(f"   - 非种子前缀数：{num_prefixes:,} 个")
    print(f"   - 单个前缀额度：{quota} 个地址")
    return quota


def load_non_seed_patterns() -> Dict[str, Dict]:
    try:
        with open(PATTERN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        filtered_data = {
            prefix: info for prefix, info in data.items()
            if info.get("patterns", [])
        }
        print(f"✅ 加载模式文件：原始{len(data):,}个前缀，过滤后{len(filtered_data):,}个前缀（含模式）")
        return filtered_data
    except Exception as e:
        print(f"❌ 加载模式失败：{e}")
        return {}


def pattern_to_ipv6(pattern_str: str) -> str:
    if len(pattern_str) != 32:
        raise ValueError(f"长度异常：{len(pattern_str)}位（需32位）")
    segments = [pattern_str[i * 4:(i + 1) * 4] for i in range(8)]
    return ipaddress.IPv6Address(":".join(segments)).compressed


def replace_wildcards(pattern: str, replacement: str) -> str:
    replaced = []
    rep_idx = 0
    for c in pattern:
        if c == "*" and rep_idx < len(replacement):
            replaced.append(replacement[rep_idx])
            rep_idx += 1
        else:
            replaced.append(c)
    return "".join(replaced)


def generate_addresses_for_pattern(pattern: str, remaining_quota: int) -> Tuple[List[str], int]:
    addresses = []
    num_wildcards = count_wildcards(pattern)

    if num_wildcards < 0:
        return addresses, 0
    if num_wildcards == 0:
        try:
            addr = pattern_to_ipv6(pattern)
            addresses.append(addr)
            return addresses, 1
        except Exception as e:
            print(f"⚠️ 无通配符模式错误：{e}")
            return addresses, 0

    # 取消单个模式的数量限制，仅受剩余额度和通配符最大组合数限制
    quota_for_pattern = min(remaining_quota, len(HEX_CHARS) ** num_wildcards)
    if quota_for_pattern <= 0:
        return addresses, 0

    print(f"   模式[{pattern[:20]}...]（{num_wildcards}个通配符）：需生成{quota_for_pattern}个地址")
    combinations = generate_wildcard_combinations(num_wildcards, quota_for_pattern)
    if not combinations:
        print(f"   未生成有效组合，跳过")
        return addresses, 0

    for combo in combinations:
        replaced_pattern = replace_wildcards(pattern, combo)
        try:
            addr = pattern_to_ipv6(replaced_pattern)
            addresses.append(addr)
        except Exception as e:
            continue

    consumed = len(addresses)
    print(f"   实际生成：{consumed}个地址，剩余额度：{remaining_quota - consumed}")
    return addresses, consumed


def batch_generate_and_write() -> int:
    """批量生成地址并直接写入TXT（边生成边写，减少内存占用）"""
    # 1. 加载数据并分配预算
    patterns_data = load_non_seed_patterns()
    prefix_quota = calculate_prefix_quota(patterns_data)
    if prefix_quota <= 0:
        return 0

    # 2. 初始化TXT文件（清空原有内容）
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write("")  # 清空文件
        print(f"✅ 初始化TXT文件：{OUTPUT_PATH}")
    except Exception as e:
        print(f"❌ 初始化文件失败：{e}")
        return 0

    # 3. 生成并写入地址
    total_written = 0
    prefix_idx = 0
    total_prefixes = len(patterns_data)

    for non_seed_prefix, non_seed_info in patterns_data.items():
        prefix_idx += 1
        print(f"\n=== 处理前缀 {prefix_idx}/{total_prefixes}：{non_seed_prefix} ===")

        # 遍历该前缀的模式
        patterns = non_seed_info.get("patterns", [])
        remaining_quota = prefix_quota

        for pattern_info in patterns:
            if remaining_quota <= 0:
                print(f"   前缀额度已耗尽，停止处理后续模式")
                break

            pattern = pattern_info["pattern"]
            # 生成地址
            try:
                addresses, consumed = generate_addresses_for_pattern(pattern, remaining_quota)
            except Exception as e:
                print(f"   模式处理异常：{e}，跳过")
                continue

            # 4. 实时写入TXT（每行1个地址）
            if addresses:
                with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                    for addr in addresses:
                        f.write(f"{addr}\n")  # 关键：仅写入地址，换行分隔

            # 更新统计
            total_written += consumed
            remaining_quota -= consumed

            # 5. 全局预算监控
            if total_written >= TOTAL_BUDGET:
                print(f"\n✅ 总预算已耗尽（已写入{total_written:,}个地址），停止所有处理")
                return total_written

        # 进度提示（每1000个前缀或10万条地址）
        if prefix_idx % 1000 == 0 or total_written % 100_000 == 0:
            progress = (total_written / TOTAL_BUDGET) * 100
            print(
                f"\n📊 进度：{prefix_idx:,}/{total_prefixes:,} 前缀，已写入{total_written:,}/{TOTAL_BUDGET:,} 地址（{progress:.1f}%）")

    # 最终统计
    print(f"\n=== 批量生成完成 ===")
    print(f"处理前缀数：{prefix_idx:,}/{total_prefixes:,}")
    print(f"实际写入地址数：{total_written:,}/{TOTAL_BUDGET:,}")
    print(f"预算使用率：{(total_written / TOTAL_BUDGET) * 100:.1f}%")
    return total_written

if __name__ == "__main__":
    try:
        total_written = batch_generate_and_write()
        print(f"\n✅ 目标地址生成完成！")
        print(f"📁 输出文件：{OUTPUT_PATH}")
        print(f"🔢 总地址数：{total_written:,}")
    except MemoryError:
        print(f"\n❌ 内存不足！建议：降低MAX_WILDCARDS（当前{MAX_WILDCARDS}）")
    except RecursionError:
        print(f"\n❌ 递归深度超限！建议：降低MAX_WILDCARDS（当前{MAX_WILDCARDS}）")
    except Exception as e:
        print(f"\n❌ 程序异常：{e}")
