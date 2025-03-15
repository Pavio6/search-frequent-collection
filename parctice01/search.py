import pandas as pd
import time
import matplotlib.pyplot as plt


# 数据预处理
def load_data(file_path):
    # 指定俄语编码
    df = pd.read_csv(file_path, header=None, encoding='cp1251')
    transactions = []
    for index, row in df.iterrows():
        transaction = set(row.dropna().astype(str))  # 将非空值转为字符串并存入集合
        transactions.append(transaction)
    return transactions


# 生成候选项集
def generate_candidates(itemsets, length):
    candidates = set()
    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            union = itemsets[i].union(itemsets[j])
            if len(union) == length:
                candidates.add(frozenset(union))
    return [frozenset(c) for c in candidates]


# 计算支持度
def calculate_support(transactions, candidate):
    count = 0
    for transaction in transactions:
        if candidate.issubset(transaction):
            count += 1
    return count / len(transactions)


# Apriori算法核心
def apriori(transactions, min_support):
    items = set()
    for transaction in transactions:
        for item in transaction:
            items.add(frozenset([item]))

    frequent_itemsets = {}
    length = 1
    current_itemsets = list(items)

    while current_itemsets:
        frequent_current = []
        for itemset in current_itemsets:
            support = calculate_support(transactions, itemset)
            if support >= min_support:
                frequent_current.append((itemset, support))

        if frequent_current:
            frequent_itemsets[length] = frequent_current
            candidates = generate_candidates(
                [itemset for itemset, _ in frequent_current], length + 1)
            current_itemsets = candidates
            length += 1
        else:
            break
    return frequent_itemsets


# 主函数
def main(file_path, support_thresholds, sort_method='support'):
    transactions = load_data(file_path)

    # 实验数据存储
    performance = {'threshold': [], 'time': []}
    itemset_counts = {'threshold': [], 'length': [], 'count': []}

    for threshold in support_thresholds:
        start_time = time.time()
        frequent_itemsets = apriori(transactions, threshold)
        elapsed_time = time.time() - start_time

        # 记录性能
        performance['threshold'].append(threshold)
        performance['time'].append(elapsed_time)

        # 统计项集数量
        for length in frequent_itemsets:
            count = len(frequent_itemsets[length])
            itemset_counts['threshold'].append(threshold)
            itemset_counts['length'].append(length)
            itemset_counts['count'].append(count)

    # 可视化
    plt.figure(figsize=(12, 5))

    # 性能对比图
    plt.subplot(1, 2, 1)
    plt.plot(performance['threshold'], performance['time'], marker='o')
    plt.xlabel('Support Threshold (%)')
    plt.ylabel('Execution Time (s)')
    plt.title('Performance Comparison')

    # 频繁项集数量分布图
    plt.subplot(1, 2, 2)
    df_counts = pd.DataFrame(itemset_counts)
    for threshold in support_thresholds:
        subset = df_counts[df_counts['threshold'] == threshold]
        plt.plot(subset['length'], subset['count'], marker='o', label=f'{threshold}%')
    plt.xlabel('Itemset Length')
    plt.ylabel('Number of Frequent Itemsets')
    plt.title('Frequent Itemsets Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()


if __name__ == "__main__":
    # 参数设置
    file_path = 'C:\\Users\\93198\\Desktop\\information-processing\\notes\\baskets.csv'
    support_thresholds = [0.01, 0.03, 0.05, 0.1, 0.15]  # 支持度阈值（百分比）

    # 运行主程序
    main(file_path, support_thresholds)
