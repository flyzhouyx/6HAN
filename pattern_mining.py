import json
import math
from collections import defaultdict
import os


def preprocess_ipv6(ip):
    """
    预处理IPv6地址：补齐省略的零并去除冒号
    将IPv6地址转换为32个半字节的十六进制字符串
    """
    # 分割地址
    parts = ip.split(':')

    # 处理::省略
    if '' in parts:
        # 找到空字符串的位置
        empty_index = parts.index('')
        # 计算需要填充的零组数
        fill_count = 8 - len(parts) + 1
        # 填充零
        parts = parts[:empty_index] + ['0000'] * fill_count + parts[empty_index + 1:]

    # 确保有8个部分
    while len(parts) < 8:
        parts.append('0000')

    # 补齐每个部分到4个字符
    padded_parts = []
    for part in parts:
        padded = part.zfill(4)
        padded_parts.append(padded)

    # 连接所有部分，得到32个半字节的字符串
    full_ip = ''.join(padded_parts)
    return full_ip


def calculate_entropy(full_ips, position):
    """
    计算指定半字节位置的信息熵
    """
    # 统计每个字符出现的频率
    char_counts = defaultdict(int)
    total = len(full_ips)

    if total == 0:
        return 0

    for ip in full_ips:
        char = ip[position]
        char_counts[char] += 1

    # 计算熵
    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def get_variable_positions(full_ips):
    """
    找出所有熵值不为0的半字节位置（可变位置）
    并按熵值从小到大排序
    """
    entropies = []
    # IPv6地址有32个半字节
    for i in range(32):
        entropy = calculate_entropy(full_ips, i)
        entropies.append((i, entropy))

    # 按熵值从小到大排序
    sorted_entropies = sorted(entropies, key=lambda x: x[1])

    # 筛选出熵值不为0的位置
    variable_positions = [i for i, entropy in sorted_entropies if entropy > 1e-9]

    return variable_positions


def create_pattern(full_ips, variable_positions):
    """
    根据一组IP和可变位置创建地址模式
    固定位置使用实际字符，可变位置使用*
    """
    if not full_ips:
        return ""

    # 以第一个IP为基础创建模式
    pattern = list(full_ips[0])

    # 标记可变位置
    for pos in variable_positions:
        pattern[pos] = '*'

    return ''.join(pattern)


def cluster_ips(full_ips, variable_positions, eta):
    """
    递归聚类IPv6地址
    """
    # 如果地址数量小于等于阈值，停止分裂
    if len(full_ips) <= eta:
        # 获取当前簇的可变位置
        current_vars = get_variable_positions(full_ips)
        # 创建模式
        pattern = create_pattern(full_ips, current_vars)
        return [(pattern, full_ips)]

    # 找到第一个可用的分割位置（熵值最低的可变位置）
    split_pos = None
    for pos in variable_positions:
        # 检查这个位置是否真的有变化
        if calculate_entropy(full_ips, pos) > 1e-9:
            split_pos = pos
            break

    # 如果没有找到可分割的位置，停止分裂
    if split_pos is None:
        pattern = create_pattern(full_ips, [])
        return [(pattern, full_ips)]

    # 根据分割位置进行聚类
    clusters = defaultdict(list)
    for ip in full_ips:
        char = ip[split_pos]
        clusters[char].append(ip)

    # 对每个子簇递归聚类
    result = []
    # 为子簇计算新的可变位置（排除当前分割位置）
    new_variable_positions = [pos for pos in variable_positions if pos != split_pos]

    for cluster in clusters.values():
        result.extend(cluster_ips(cluster, new_variable_positions, eta))

    return result


def pattern_mining(addr_set, eta):
    """
    IPv6地址模式挖掘主函数
    """
    # 预处理所有IP地址
    full_ips = [preprocess_ipv6(ip) for ip in addr_set]

    # 计算所有半字节位的熵值并排序
    variable_positions = get_variable_positions(full_ips)

    # 聚类
    clusters = cluster_ips(full_ips, variable_positions, eta)

    # 转换为模式库格式，并过滤掉仅包含一个地址且没有通配符的模式
    pattern_set = []
    for pattern, ips in clusters:
        # 将原始IP地址与模式对应
        original_ips = []
        for full_ip in ips:
            # 找到原始IP
            original_ip = next(ip for ip in addr_set if preprocess_ipv6(ip) == full_ip)
            original_ips.append(original_ip)

        pattern_info = {
            "pattern": pattern,
            "addresses": original_ips,
            "count": len(original_ips)
        }

        # 过滤条件：只有一个地址且没有通配符的模式不保留
        if not (pattern_info["count"] == 1 and '*' not in pattern_info["pattern"]):
            pattern_set.append(pattern_info)

    return pattern_set


def main():
    # 定义数据文件路径（位于同级Data文件夹下）
    data_file_path = os.path.join('Data', 'matched_prefixes_result.json')

    # 检查文件是否存在
    if not os.path.exists(data_file_path):
        print(f"错误：数据文件 {data_file_path} 不存在")
        return

    # 读取数据
    with open(data_file_path, 'r') as f:
        data = json.load(f)

    # 设置阈值η
    eta = 3

    # 对每个前缀下的地址进行模式挖掘
    all_patterns = {}
    for prefix, addresses in data.items():
        print(f"Processing prefix: {prefix} with {len(addresses)} addresses")
        patterns = pattern_mining(addresses, eta)
        all_patterns[prefix] = patterns

    # 确保Data文件夹存在
    os.makedirs('Data', exist_ok=True)

    # 保存结果到Data文件夹
    result_file_path = os.path.join('Data', 'ipv6_patterns_result.json')
    with open(result_file_path, 'w') as f:
        json.dump(all_patterns, f, indent=2)

    print(f"Pattern mining completed. Results saved to {result_file_path}")


if __name__ == "__main__":
    main()
