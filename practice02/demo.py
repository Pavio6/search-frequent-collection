import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import time
import matplotlib.pyplot as plt
import chardet
from matplotlib.font_manager import FontProperties

font_path = 'C:/Windows/Fonts/simhei.ttf'
font = FontProperties(fname=font_path)
plt.rcParams['font.family'] = font.get_name()
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


# 读取数据集
def read_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']

        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None


# 数据预处理
def preprocess_data(df):
    transactions = []
    for _, row in df.iterrows():
        transaction = [item for item in row if pd.notna(item)]
        transactions.append(transaction)
    return transactions


# 运行 Apriori 算法并返回结果和运行时间
def run_apriori(transactions, support_threshold):
    start_time = time.time()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=support_threshold, use_colnames=True)
    end_time = time.time()
    execution_time = end_time - start_time
    return frequent_itemsets, execution_time


# 提取关联规则
def extract_association_rules(frequent_itemsets, confidence_threshold):
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_threshold)
    return rules


# 可视化性能比较（运行时间与支持阈值）
def plot_performance(support_thresholds, execution_times):
    plt.figure(figsize=(10, 5))
    plt.plot(support_thresholds, execution_times, marker='o')
    plt.xlabel('支持阈值', fontproperties=font)
    plt.xticks(rotation=45, fontproperties=font)
    plt.ylabel('运行时间 (秒)', fontproperties=font)
    plt.title('在固定数据集上，运行时间与支持阈值的关系', fontproperties=font)
    plt.show()


# 可视化不同长度频繁项集的数量
def plot_itemset_length_counts(support_thresholds, all_itemset_length_counts):
    plt.figure(figsize=(10, 5))
    for i, itemset_counts in enumerate(all_itemset_length_counts):
        plt.plot(itemset_counts.index, itemset_counts.values, marker='o',
                 label=f'支持阈值: {support_thresholds[i]}')
    plt.xlabel('频繁项集长度', fontproperties=font)
    plt.ylabel('频繁项集数量', fontproperties=font)
    plt.title('在固定数据集上，不同支持阈值下不同长度频繁项集的数量', fontproperties=font)
    plt.legend(prop=font)
    plt.show()


# 可视化不同置信度阈值下的规则数量
def plot_rule_counts(confidence_thresholds, rule_counts):
    plt.figure(figsize=(10, 5))
    plt.plot(confidence_thresholds, rule_counts, marker='o')
    plt.xlabel('置信度阈值', fontproperties=font)
    plt.ylabel('规则数量', fontproperties=font)
    plt.title('在固定数据集上，不同置信度阈值下的规则数量', fontproperties=font)
    plt.show()


# 可视化不同置信度阈值下的性能
def plot_performance_by_confidence(confidence_thresholds, execution_times):
    plt.figure(figsize=(10, 5))
    plt.plot(confidence_thresholds, execution_times, marker='o')
    plt.xlabel('置信度阈值', fontproperties=font)
    plt.ylabel('运行时间 (秒)', fontproperties=font)
    plt.title('在固定数据集上，运行时间与置信度阈值的关系', fontproperties=font)
    plt.show()


# 主函数
def main():
    file_path = "C:\\Users\\93198\\Desktop\\information-processing\\search-frequent-collections\\parctice01\\baskets.csv"
    df = read_data(file_path)
    if df is None:
        return

    transactions = preprocess_data(df)
    support_threshold = 0.001
    confidence_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    execution_times = []
    rule_counts = []

    frequent_itemsets, _ = run_apriori(transactions, support_threshold)

    for confidence in confidence_thresholds:
        start_time = time.time()
        rules = extract_association_rules(frequent_itemsets, confidence)
        end_time = time.time()
        execution_time = end_time - start_time

        execution_times.append(execution_time)
        rule_counts.append(len(rules))

        # 输出规则列表
        print(f"\n置信度阈值: {confidence}")
        for idx, row in rules.iterrows():
            antecedent = ', '.join(list(row['antecedents']))
            consequent = ', '.join(list(row['consequents']))
            print(f"{antecedent} → {consequent} (支持度: {row['support']:.3f}, 置信度: {row['confidence']:.3f})")

    plot_performance_by_confidence(confidence_thresholds, execution_times)
    plot_rule_counts(confidence_thresholds, rule_counts)


if __name__ == "__main__":
    main()
