import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 设置中文字体，如果需要在图表中显示中文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass


def load_data(file_path):
    """
    加载数据集

    参数:
        file_path: 数据集文件路径

    返回:
        X: 特征矩阵
        y: 目标变量
    """
    # 读取数据
    try:
        data = pd.read_csv(file_path)
    except:
        try:
            # 尝试读取Excel文件
            data = pd.read_excel(file_path)
        except:
            raise ValueError("不支持的文件格式，请提供CSV或Excel文件")

    # 假设最后一列是目标变量，其余为特征
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 处理分类特征
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}

    for column in categorical_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # 处理目标变量，如果它是分类变量
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y


def train_ensemble_model(X_train, y_train, ensemble_type, n_estimators, **kwargs):
    """
    训练集成模型

    参数:
        X_train: 训练特征
        y_train: 训练标签
        ensemble_type: 集成方法类型 ('bagging', 'random_forest', 'boosting')
        n_estimators: 集成成员数量
        **kwargs: 其他参数

    返回:
        训练好的模型
    """
    if ensemble_type == 'bagging':
        # 装袋法(Bagging)
        model = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            random_state=42,
            **kwargs
        )
    elif ensemble_type == 'random_forest':
        # 随机森林(Random Forest)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            **kwargs
        )
    elif ensemble_type == 'boosting':
        # 提升法(AdaBoost)
        model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            random_state=42,
            **kwargs
        )
    else:
        raise ValueError("不支持的集成方法类型。请选择 'bagging', 'random_forest' 或 'boosting'")

    # 训练模型
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能

    参数:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签

    返回:
        包含各种性能指标的字典
    """
    # 预测
    y_pred = model.predict(X_test)

    # 计算性能指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    return metrics


def compare_ensemble_sizes(X, y, ensemble_type, size_range, **kwargs):
    """
    比较不同集成规模的性能

    参数:
        X: 特征矩阵
        y: 目标变量
        ensemble_type: 集成方法类型
        size_range: 集成规模范围，如 range(50, 101, 10)
        **kwargs: 传递给模型的其他参数

    返回:
        包含不同规模性能指标的DataFrame
    """
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 存储不同集成规模的性能结果
    results = []

    for n_estimators in size_range:
        print(f"训练集成规模为 {n_estimators} 的模型...")

        # 训练模型
        model = train_ensemble_model(X_train, y_train, ensemble_type, n_estimators, **kwargs)

        # 评估模型
        metrics = evaluate_model(model, X_test, y_test)
        metrics['n_estimators'] = n_estimators
        results.append(metrics)

    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def visualize_results(results_df, dt_metrics=None):
    """
    可视化结果

    参数:
        results_df: 包含不同集成规模性能指标的DataFrame
        dt_metrics: 决策树模型的性能指标，用于比较
    """
    # 设置图表风格
    sns.set(style="whitegrid")

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 绘制性能指标随集成规模变化的曲线
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        plt.plot(results_df['n_estimators'], results_df[metric], marker='o', label=f'{metric}')

    # 如果提供了决策树性能指标，添加到图表
    if dt_metrics is not None:
        # 创建不同的线型和颜色
        colors = ['red', 'green', 'blue', 'purple']  # 自定义颜色列表
        i = 0
        for metric in metrics:
            plt.axhline(y=dt_metrics[metric], color=colors[i],
                        linestyle='--', label=f'DT {metric}')
            i += 1

        # 添加图表标题和标签
    plt.title('Impact of Ensemble Size on Classification Performance', fontsize=14)
    plt.xlabel('Number of Ensemble Members', fontsize=12)
    plt.ylabel('Performance Metrics', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图表
    plt.savefig('ensemble_performance.png', dpi=300)

    # 显示图表
    plt.show()


def train_single_decision_tree(X, y):
    """
    训练单个决策树，用于与集成方法比较

    参数:
        X: 特征矩阵
        y: 目标变量

    返回:
        决策树的性能指标
    """
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练决策树模型
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    # 评估模型
    y_pred = dt.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    return metrics


def main():
    """
    主函数，运行集成分类实验
    """
    # 数据集路径
    data_path = "../data/adult.data.csv"  # 请替换为您的数据集路径

    # 设置实验参数
    ensemble_type = "random_forest"  # 可选: 'bagging', 'random_forest', 'boosting'
    size_range = range(50, 101, 10)  # 集成规模范围: 50, 60, 70, 80, 90, 100

    # 加载数据
    print("加载数据...")
    X, y = load_data(data_path)

    # 训练单个决策树作为基准
    print("训练单个决策树作为基准...")
    dt_metrics = train_single_decision_tree(X, y)
    print(f"决策树性能指标: {dt_metrics}")

    # 比较不同集成规模
    print("比较不同集成规模的性能...")
    results_df = compare_ensemble_sizes(X, y, ensemble_type, size_range)

    # 打印结果
    print("\n各集成规模的性能指标:")
    print(results_df)

    # 可视化结果
    print("\n生成可视化结果...")
    visualize_results(results_df, dt_metrics)

    print("\n实验完成！结果已保存为 'ensemble_performance.png'")


if __name__ == "__main__":
    main()
