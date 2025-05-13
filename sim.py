import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pyro
from pyro.infer import MCMC, NUTS
from pyro.distributions import Normal, Bernoulli
from sklearn.metrics import accuracy_score

# 生成模拟数据
def generate_synthetic_data(num_samples, num_features):
    # 生成高斯分布的特征向量
    X = np.random.multivariate_normal(mean=np.zeros(num_features), cov=np.eye(num_features), size=num_samples)
    # 生成对应的标签（二分类问题）
    y = np.random.randint(0, 2, num_samples)
    return X, y

# 核范数（Nuclear Norm）
def nuclear_norm(matrix):
    return torch.sum(torch.svd(matrix)[1])

# 弗罗贝尼乌斯范数（Frobenius Norm）
def frobenius_norm(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError(f"Matrix dimensions must match: {matrix1.shape} vs {matrix2.shape}")
    return torch.sum((matrix1 - matrix2) ** 2)

# 定义贝叶斯图模型
class BayesianGraph:
    def __init__(self, num_nodes, num_features, use_bayesian=False, use_gp=False, use_mcmc=False):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.use_bayesian = use_bayesian
        self.use_gp = use_gp
        self.use_mcmc = use_mcmc
        self.edge_means = torch.randn(num_nodes * num_nodes, requires_grad=True)  # 随机初始化均值
        self.edge_vars = torch.ones(num_nodes * num_nodes, requires_grad=True)
        self.potential_edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
        if self.use_gp:
            self.gp = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), alpha=1e-5)
        self.fixed_spatial_masks = torch.tensor([[1 if i != j else 0 for j in range(num_nodes)] for i in range(num_nodes)]).float()
        self.fixed_temporal_masks = torch.tensor([[1 for j in range(num_nodes)] for i in range(num_nodes)]).float()
        self.initial_adjacency_matrix = None  # 用于保存初始图结构
        self.edge_dropout_rate = 0.0  # 设置边的采样比例

    def set_edge_dropout_rate(self, rate):
        self.edge_dropout_rate = rate

    def construct_graph(self, temperature=1.0, training=True):
        adjacency_matrix = torch.zeros((self.num_nodes, self.num_nodes))
        if self.use_mcmc and training:
            # 定义贝叶斯模型和MCMC采样器
            def model(edge_logits):
                edge_means = pyro.sample("edge_means", Normal(torch.zeros_like(edge_logits), torch.ones_like(edge_logits)))
                for i in range(len(self.potential_edges)):
                    prob = torch.sigmoid(edge_means[i] / temperature)
                    pyro.sample(f"edge_{i}", Bernoulli(prob), obs=torch.tensor(torch.rand(1) < prob, dtype=torch.float32))
                return edge_means

            nuts_kernel = NUTS(model, step_size=0.01)
            mcmc = MCMC(nuts_kernel, num_samples=10, warmup_steps=2)
            mcmc.run(torch.zeros(len(self.potential_edges)))
            samples = mcmc.get_samples()
            edge_means_samples = samples["edge_means"]

            for i in range(len(self.potential_edges)):
                if i >= len(edge_means_samples):
                    continue
                edge_prob = torch.sigmoid(edge_means_samples[i].mean() / temperature)
                if torch.rand(1) < edge_prob:
                    src, dst = self.potential_edges[i]
                    adjacency_matrix[src, dst] = 1
        else:
            for i in range(len(self.potential_edges)):
                if self.use_bayesian:
                    edge_prob = torch.sigmoid(self.edge_means[i] / temperature)
                else:
                    edge_prob = torch.sigmoid(torch.randn(1) / temperature)
                
                # 应用边采样（用于推理阶段的dropout效果）
                if not training and torch.rand(1) < self.edge_dropout_rate:
                    continue
                
                if torch.rand(1) < edge_prob:
                    src, dst = self.potential_edges[i]
                    adjacency_matrix[src, dst] = 1
        return adjacency_matrix

    def train(self, X, y, num_iterations=1000, learning_rate=0.001):
        optimizer = torch.optim.Adam([self.edge_means, self.edge_vars], lr=learning_rate)
        loss_history = []
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            loss = self.compute_loss(X, y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        return loss_history

    def compute_loss(self, X, y):
        # 将输入数据 X 转换为合适的矩阵形式
        spatial_matrix_train = torch.tensor(X, dtype=torch.float32).reshape((self.num_nodes, -1))
        spatial_matrix_train.requires_grad_(True)
        temporal_matrix_train = torch.tensor(X, dtype=torch.float32).reshape((self.num_nodes, -1))
        temporal_matrix_train.requires_grad_(True)
        
        # 确保固定掩码矩阵的维度与输入矩阵一致
        fixed_spatial_masks = self.fixed_spatial_masks.clone().detach().requires_grad_(False)
        fixed_temporal_masks = self.fixed_temporal_masks.clone().detach().requires_grad_(False)
        
        # 核范数和弗罗贝尼乌斯范数的计算
        loss_s = nuclear_norm(spatial_matrix_train)
        loss_t = nuclear_norm(temporal_matrix_train)
        
        # 将固定掩码矩阵调整为与输入矩阵相同的维度
        fixed_spatial_masks_resized = fixed_spatial_masks.repeat(1, spatial_matrix_train.shape[1] // fixed_spatial_masks.shape[1])
        fixed_temporal_masks_resized = fixed_temporal_masks.repeat(1, temporal_matrix_train.shape[1] // fixed_temporal_masks.shape[1])
        
        frob_loss_s = frobenius_norm(fixed_spatial_masks_resized, spatial_matrix_train)
        frob_loss_t = frobenius_norm(fixed_temporal_masks_resized, temporal_matrix_train)
        add_loss = loss_s + loss_t + torch.relu(frob_loss_s - 0.1) + torch.relu(frob_loss_t - 0.1)
        
        if self.use_bayesian:
            variance_loss_spatial = torch.sum(self.edge_vars)
            variance_loss_temporal = torch.sum(self.edge_vars)
            add_loss += 0.1 * (variance_loss_spatial + variance_loss_temporal)
        
        return add_loss

    def save_graph(self, path):
        adjacency_matrix = self.construct_graph(training=False)
        torch.save(adjacency_matrix, path)
        return adjacency_matrix

    def predict(self, X):
        # 构建图结构用于预测
        adjacency_matrix = self.construct_graph(training=False)
        # 在实际应用中，这里应该包含基于图结构的预测逻辑
        # 这里仅作为示例，随机生成预测结果
        return torch.randint(0, 2, (X.shape[0],))

# 模拟实验
def simulate_experiment(args):
    # 创建保存目录
    save_dir = "sim"
    os.makedirs(save_dir, exist_ok=True)

    # 生成模拟数据
    num_features = args.num_nodes * args.num_nodes  # 确保特征数量与节点数量的平方一致
    args.num_features = num_features
    X, y = generate_synthetic_data(args.num_samples, num_features)

    # 初始化贝叶斯图模型
    graph = BayesianGraph(
        num_nodes=args.num_nodes,
        num_features=num_features,
        use_bayesian=args.use_bayesian,
        use_gp=args.use_gp,
        use_mcmc=args.use_mcmc
    )
    
    # 设置边采样比例
    graph.set_edge_dropout_rate(args.edge_dropout_rate)

    # 保存初始图结构
    initial_adjacency_matrix = graph.construct_graph()
    initial_graph_path = os.path.join(save_dir, 'initial_graph.pt')
    torch.save(initial_adjacency_matrix, initial_graph_path)
    print(f"Initial graph saved to {initial_graph_path}")

    # 绘制初始图结构的边和节点分布
    plot_edge_node_distribution(initial_adjacency_matrix, os.path.join(save_dir, 'initial_edge_node_dist.png'), 'Initial')

    # 训练模型并记录损失
    loss_history = graph.train(X, y, num_iterations=args.num_iterations, learning_rate=args.lr)

    # 使用模型进行预测并计算准确率
    y_pred = graph.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    # 保存训练后的图结构
    trained_adjacency_matrix = graph.construct_graph(training=False)
    trained_graph_path = os.path.join(save_dir, 'trained_graph.pt')
    torch.save(trained_adjacency_matrix, trained_graph_path)
    print(f"Trained graph saved to {trained_graph_path}")

    # 绘制训练后的图结构的边和节点分布
    plot_edge_node_distribution(trained_adjacency_matrix, os.path.join(save_dir, 'trained_edge_node_dist.png'), 'Trained')

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Convergence')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(save_dir, 'loss_convergence.png')
    plt.savefig(loss_plot_path)
    plt.close()

    # 绘制初始图结构热图
    plt.figure(figsize=(10, 6))
    plt.imshow(initial_adjacency_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Initial Adjacency Matrix Heatmap')
    plt.xlabel('Node')
    plt.ylabel('Node')
    initial_heatmap_path = os.path.join(save_dir, 'initial_adjacency_heatmap.png')
    plt.savefig(initial_heatmap_path)
    plt.close()

    # 绘制训练后的图结构热图
    plt.figure(figsize=(10, 6))
    plt.imshow(trained_adjacency_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Trained Adjacency Matrix Heatmap')
    plt.xlabel('Node')
    plt.ylabel('Node')
    trained_heatmap_path = os.path.join(save_dir, 'trained_adjacency_heatmap.png')
    plt.savefig(trained_heatmap_path)
    plt.close()

    # 绘制边权重的均值和方差
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(graph.edge_means)), graph.edge_means.detach().numpy())
    plt.xlabel('Edge Index')
    plt.ylabel('Mean')
    plt.title('Edge Means')

    plt.subplot(1, 2, 2)
    plt.bar(range(len(graph.edge_vars)), graph.edge_vars.detach().numpy())
    plt.xlabel('Edge Index')
    plt.ylabel('Variance')
    plt.title('Edge Variances')

    plt.tight_layout()
    stats_plot_path = os.path.join(save_dir, 'edge_stats.png')
    plt.savefig(stats_plot_path)
    plt.close()

    print(f"Plots saved to {save_dir} directory.")

# 绘制边和节点分布图
def plot_edge_node_distribution(adjacency_matrix, save_path, title_prefix):
    # 边分布统计
    edge_counts = adjacency_matrix.flatten().numpy()
    
    # 节点入度和出度统计
    in_degree = torch.sum(adjacency_matrix, dim=0).numpy()
    out_degree = torch.sum(adjacency_matrix, dim=1).numpy()

    plt.figure(figsize=(18, 6))

    # 边分布直方图
    plt.subplot(1, 3, 1)
    plt.hist(edge_counts, bins=2)
    plt.title(f'{title_prefix} Edge Distribution')
    plt.xlabel('Edge Existence')
    plt.ylabel('Count')

    # 节点入度分布
    plt.subplot(1, 3, 2)
    plt.bar(range(len(in_degree)), in_degree)
    plt.title(f'{title_prefix} Node In-Degree')
    plt.xlabel('Node Index')
    plt.ylabel('In-Degree')

    # 节点出度分布
    plt.subplot(1, 3, 3)
    plt.bar(range(len(out_degree)), out_degree)
    plt.title(f'{title_prefix} Node Out-Degree')
    plt.xlabel('Node Index')
    plt.ylabel('Out-Degree')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate experiments with different components")
    parser.add_argument('--num_samples', type=int, default=500, help='Number of synthetic data samples')
    parser.add_argument('--num_features', type=int, default=100, help='Number of features in synthetic data')
    parser.add_argument('--num_nodes', type=int, default=20, help='Number of nodes in the graph')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for sampling')
    parser.add_argument('--use_bayesian', action='store_true', help='Whether to use Bayesian distribution')
    parser.add_argument('--use_gp', action='store_true', help='Whether to use Gaussian Process')
    parser.add_argument('--use_mcmc', action='store_true', help='Whether to use MCMC sampling')
    parser.add_argument('--edge_dropout_rate', type=float, default=0.25, help='Edge dropout rate for inference')

    args = parser.parse_args()
    simulate_experiment(args)