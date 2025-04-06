import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import graphviz
from sklearn.tree import export_graphviz


class DecisionTreeClassifierApp:
    """
    决策树分类应用
    支持多种分裂标准和性能评估
    """

    def __init__(self):
        """初始化分类器应用"""
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tree = None
        self.criterion = None
        self.train_ratio = None

    def load_data(self, train_file, test_file=None, train_ratio=1.0):
        """
        加载数据集

        Args:
            train_file: 训练数据文件路径
            test_file: 测试数据文件路径，如果为None则从训练集分割
            train_ratio: 训练集占比
        """
        # 列名定义（根据adult.names文件中的描述）
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]

        # 读取数据
        df = pd.read_csv(train_file, header=None, names=column_names,
                         sep=r',\s* ', engine='python', na_values='?')

        # 数据清洗
        df = df.dropna()  # 删除包含缺失值的行

        # 特征工程
        # 将分类变量转换为数值
        categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                                'relationship', 'race', 'sex', 'native-country']

        for feature in categorical_features:
            df[feature] = pd.Categorical(df[feature]).codes

        # 目标变量转换
        df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

        # 分割特征和目标
        X = df.drop('income', axis=1)
        y = df['income']

        # 根据指定的训练集比例分割数据
        self.train_ratio = train_ratio
        if test_file is None:
            # 如果没有提供测试集，则从训练集中分割
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, train_size=train_ratio, random_state=42
            )
        else:
            # 如果提供了测试集，则直接使用
            test_df = pd.read_csv(test_file, header=None, names=column_names,
                                  sep=r',\s*', engine='python', na_values='?', skiprows=1)
            test_df = test_df.dropna()

            for feature in categorical_features:
                test_df[feature] = pd.Categorical(test_df[feature]).codes

            test_df['income'] = test_df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

            self.X_train = X
            self.y_train = y
            self.X_test = test_df.drop('income', axis=1)
            self.y_test = test_df['income']

        # 输出数据集信息
        print(f"训练集大小: {len(self.X_train)}，测试集大小: {len(self.X_test)}")
        print(f"训练集中正样本比例: {self.y_train.mean():.4f}")
        print(f"测试集中正样本比例: {self.y_test.mean():.4f}")

    def train(self, criterion='gini', max_depth=None):
        """
        训练决策树模型

        Args:
            criterion: 分裂标准，可选 'gini'(基尼指数), 'entropy'(信息增益/熵)
                      或 'log_loss'(等价于熵，但实现不同)
        """
        self.criterion = criterion

        # 创建并训练决策树模型，加入深度限制参数
        self.tree = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=50,  # 增加分裂所需的最小样本数，简化树结构
            random_state=42
        )
        self.tree.fit(self.X_train, self.y_train)

        # 输出树的基本信息
        print(f"决策树深度: {self.tree.get_depth()}")
        print(f"决策树叶节点数: {self.tree.get_n_leaves()}")

    def evaluate(self):
        """评估模型并返回性能指标"""
        # 在测试集上进行预测
        y_pred = self.tree.predict(self.X_test)

        # 计算性能指标
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        # 返回性能指标字典
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        # 输出性能指标
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1 Score): {f1:.4f}")

        return metrics

    def visualize_tree(self, max_depth=2):
        """
        可视化决策树

        Args:
            max_depth: 可视化的最大深度
        """
        if self.tree is None:
            print("请先训练模型！")
            return

            # 使用更大的图形尺寸和更少的深度来提高可读性
            plt.figure(figsize=(15, 25))
            plot_tree(self.tree,
                      feature_names=self.X_train.columns,
                      class_names=['<=50K', '>50K'],
                      filled=True,
                      rounded=True,
                      max_depth=max_depth,
                      fontsize=12,  # 增加字体大小
                      proportion=False,  # 不按比例显示节点大小
                      impurity=False)  # 不显示不纯度值，简化节点内容
            plt.title(f'决策树 (标准: {self.criterion}, 训练比例: {self.train_ratio:.2f})', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'decision_tree_{self.criterion}_{self.train_ratio}.png', dpi=300)
            plt.show()

    def export_tree_dot(self, max_depth=3):
        """
        导出决策树为DOT格式和PDF图形，更适合复杂树的可视化

        Args:
            max_depth: 导出的最大深度
        """
        if self.tree is None:
            print("请先训练模型！")
            return

        # 导出为DOT格式
        dot_data = export_graphviz(
            self.tree,
            max_depth=max_depth,
            feature_names=self.X_train.columns,
            class_names=['<=50K', '>50K'],
            filled=True,
            rounded=True,
            special_characters=True,
            out_file=None
        )

        # 使用graphviz渲染
        try:
            graph = graphviz.Source(dot_data)
            graph.render(f"tree_{self.criterion}_{self.train_ratio}")
            print(f"树已导出为 tree_{self.criterion}_{self.train_ratio}.pdf")
            return graph
        except Exception as e:
            print(f"导出图形失败，可能需要安装Graphviz: {e}")
            print("可以将以下内容保存为.dot文件，然后在线渲染: https://dreampuf.github.io/GraphvizOnline/")
            print(dot_data[:1000] + "..." if len(dot_data) > 1000 else dot_data)

    def visualize_simplified_tree(self, max_depth=2):
        """
        可视化简化版的决策树，只显示关键信息

        Args:
            max_depth: 可视化的最大深度
        """
        if self.tree is None:
            print("请先训练模型！")
            return

        # 绘制树
        plt.figure(figsize=(20, 12))

        # 获取特征重要性
        importances = self.tree.feature_importances_
        feature_names = self.X_train.columns

        # 在树的可视化旁边添加特征重要性条形图
        plt.subplot(1, 2, 1)
        plot_tree(self.tree,
                  feature_names=feature_names,
                  class_names=['<=50K', '>50K'],
                  filled=True,
                  rounded=True,
                  max_depth=max_depth,
                  fontsize=10,
                  impurity=False)  # 不显示不纯度值
        plt.title(f'简化决策树 (标准: {self.criterion})', fontsize=14)

        # 添加特征重要性图
        plt.subplot(1, 2, 2)
        indices = np.argsort(importances)[::-1]
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('特征重要性')
        plt.title('特征重要性排名', fontsize=14)

        plt.tight_layout()
        plt.savefig(f'simplified_tree_{self.criterion}_{self.train_ratio}.png', dpi=300)
        plt.show()


def run_experiments():
    """运行完整的实验"""
    app = DecisionTreeClassifierApp()

    # 文件路径
    train_file = 'C:\\Users\\93198\\Desktop\\test-demo\\adult.data.csv'  # 请替换为实际的文件路径

    # 存储不同训练比例的结果
    results = []

    # 不同的分裂标准
    criteria = {
        'gini': '基尼指数 (Gini)',
        'entropy': '信息增益 (Info Gain)',
        'log_loss': '增益率 (Gain Ratio)'
    }

    # 对每个分裂标准进行实验
    for criterion, criterion_name in criteria.items():
        print(f"\n======= 使用分裂标准: {criterion_name} =======")

        # 测试不同的训练集比例
        for train_ratio in [0.6, 0.7, 0.8, 0.9]:
            print(f"\n----- 训练比例: {train_ratio:.1f} -----")

            # 加载数据
            app.load_data(train_file, train_ratio=train_ratio)

            # 训练模型 - 为了可视化效果，限制树的深度为5
            app.train(criterion=criterion, max_depth=5)

            # 评估模型
            metrics = app.evaluate()

            # 可视化决策树（如果训练集比例是最高的）
            if train_ratio == 0.9:
                # 使用改进的可视化方法
                app.visualize_tree(max_depth=2)
                app.visualize_simplified_tree(max_depth=2)
                app.export_tree_dot(max_depth=3)

            # 存储结果
            results.append({
                'criterion': criterion_name,
                'train_ratio': train_ratio,
                **metrics
            })

    # 将结果转换为DataFrame
    results_df = pd.DataFrame(results)

    # 可视化性能指标
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']

    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i + 1)
        sns.lineplot(data=results_df, x='train_ratio', y=metric, hue='criterion', marker='o')
        plt.title(f'{metric.capitalize()} 随训练集比例的变化', fontsize=14)
        plt.xlabel('训练集比例', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300)
    plt.show()

    # 打印表格形式的结果
    print("\n性能指标汇总:")
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    results = run_experiments()
