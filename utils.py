import numpy as np
import pandas as pd
from anndata import AnnData
import networkx as nx
import scanpy as sc
from typing import Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
import torch
import random
from torch_geometric.utils import to_networkx
from sklearn.metrics import average_precision_score


def data_preparation(input_expData: Union[str, sc.AnnData, pd.DataFrame],
                     input_priorNet: Union[str, pd.DataFrame]) -> dict[str: AnnData]:
    """
    Prepare the data object for CEFCON.
    """

    print('[0] - Data loading and preprocessing...')

    ## [1] Single-cell RNA-seq data
    if isinstance(input_expData, str):
        p = Path(input_expData)
        if p.suffix == '.csv':
            adata = sc.read_csv(input_expData, first_column_names=True)
        else:  # h5ad
            adata = sc.read_h5ad(input_expData)
    elif isinstance(input_expData, sc.AnnData):
        adata = input_expData
    elif isinstance(input_expData, pd.DataFrame):
        adata = sc.AnnData(X=input_expData.iloc[:, 1:].values)  # 只保留数值部分
        adata.var_names = input_expData.columns[1:]  # 假设第一列为标签
    else:
        raise Exception("Invalid input! The input format must be '.csv' file or '.h5ad' "
                        "formatted file, or an 'AnnData' object!", input_expData)

    #adata = adata[1:, :].copy()
    # Gene symbols are uniformly handled in uppercase
    adata.var_names = adata.var_names.str.upper()
    #print(adata.var_names)
    ## [2] Prior network data
    if isinstance(input_priorNet, str):
        netData = pd.read_csv(input_priorNet, index_col=None, header=0)
    elif isinstance(input_priorNet, pd.DataFrame):
        netData = input_priorNet.copy()
    else:
        raise Exception("Invalid input!", input_priorNet)

    # make sure the genes of prior network are in the input scRNA-seq data
    netData['from'] = netData['from'].str.upper()
    netData['to'] = netData['to'].str.upper()
    netData = netData.loc[netData['from'].isin(adata.var_names.values)
                          & netData['to'].isin(adata.var_names.values), :]

    netData = netData.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)

    # Transfer into networkx object
    priori_network = nx.from_pandas_edgelist(netData, source='from', target='to', create_using=nx.DiGraph)
    priori_network_nodes = np.array(priori_network.nodes())

    # in_degree, out_degree (centrality)
    in_degree = pd.DataFrame.from_dict(nx.in_degree_centrality(priori_network),
                                       orient='index', columns=['in_degree'])
    out_degree = pd.DataFrame.from_dict(nx.out_degree_centrality(priori_network),
                                        orient='index', columns=['out_degree'])
    centrality = pd.concat([in_degree, out_degree], axis=1)
    centrality = centrality.loc[priori_network_nodes, :]

    ## [3] A mapper for node index and gene name
    idx_GeneName_map = pd.DataFrame({
        'idx': range(adata.n_vars),  # 从0开始的索引
        'geneName': adata.var_names  # 保留基因名
    }, index=adata.var_names)  # 设置索引为基因名

    # 初始化边列表
    directed_edges = []

    # 处理有向边
    for _, row in netData[netData['edge_type'] == 'directed'].iterrows():
        from_idx = idx_GeneName_map.loc[row['from'], 'idx']
        to_idx = idx_GeneName_map.loc[row['to'], 'idx']
        directed_edges.append({'from': from_idx, 'to': to_idx})
    # 处理无向边
    for _, row in netData[netData['edge_type'] == 'undirected'].iterrows():
        from_idx = idx_GeneName_map.loc[row['from'], 'idx']
        to_idx = idx_GeneName_map.loc[row['to'], 'idx']
        directed_edges.append({'from': from_idx, 'to': to_idx})
        directed_edges.append({'from': to_idx, 'to': from_idx})  # 添加反向边
    # 创建边列表的 DataFrame

    edgelist = pd.DataFrame(directed_edges)
    print(edgelist)

    # Only keep the genes that exist in both single cell data and the prior gene interaction network
    #adata = adata[:, priori_network_nodes].copy()  # Copy here to avoid implicit modification warnings
    adata = adata.copy()
    #adata.varm['centrality_prior_net'] = centrality

    #[4] - Normalizing the expression matrix...
    # 标准化
    adata.X = adata.X / adata.X.sum(axis=0) * 1e4  # 对每个基因进行标准化
    sc.pp.log1p(adata)

    adata.varm['idx_GeneName_map'] = idx_GeneName_map
    adata.uns['edgelist'] = edgelist
    #adata = add_relative_gaussian_noise(adata, mean=0, noise_factor=5)
    print(f"    n_genes × n_cells = {adata.n_vars} × {adata.n_obs}")

    return adata



def plot_grn_degree_distribution(GRN_df):
    """
    Plot degree distribution of the inferred GRN
    """
    # 创建有向图
    network = nx.from_pandas_edgelist(GRN_df, source='TF', target='Target', edge_attr='value',
                                      create_using=nx.DiGraph())

    # 移除自环
    network.remove_edges_from(nx.selfloop_edges(network))
    network = network.subgraph(max(nx.weakly_connected_components(network), key=len)).copy()

    # 计算度序列
    degree_sequence = pd.DataFrame(np.array(network.degree))
    degree_sequence.columns = ["ind", "degree"]
    degree_sequence['degree'] = degree_sequence['degree'].astype(int)
    degree_sequence = degree_sequence.loc[degree_sequence['degree'] != 0, :]
    degree_sequence = degree_sequence.set_index("ind")
    dist = degree_sequence.degree.value_counts() / degree_sequence.degree.value_counts().sum()
    dist.index = dist.index.astype(int)

    x = np.log(dist.index.values).reshape([-1, 1])
    y = np.log(dist.values).reshape([-1, 1])

    model = lr()
    model.fit(x, y)

    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    x_ = np.array([-1, 5]).reshape([-1, 1])
    y_ = model.predict(x_)

    ax.set_title("Degree Distribution (Log Scale)")
    ax.plot(x_.flatten(), y_.flatten(), c="black", alpha=0.5)

    ax.scatter(x.flatten(), y.flatten(), c="black")
    ax.text(0.45, 0.95,
            f"slope: {model.coef_[0][0]:.4g}, " + r"$R^2$: " + f"{model.score(x, y):.4g}\n" +
            f"num_of_genes: {network.number_of_nodes()}\n" +
            f"num_of_edges: {network.number_of_edges()}\n" +
            f"clustering_coefficient: {nx.average_clustering(network):.4g}\n",
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=8.5)
    ax.set_ylim([y.min() - 0.2, y.max() + 0.2])
    ax.set_xlim([-0.2, x.max() + 0.2])
    ax.set_xlabel("log k")
    ax.set_ylabel("log P(k)")
    ax.grid(None)
    print("plot!")
    plt.tight_layout()
    plt.savefig("/home/guanqiyuan/qyguan/lovepp/degree_distribution.png", dpi=300, bbox_inches='tight')  # 指定文件名和分辨率
    plt.close()  # 关闭图形以释放内存




def drop_feature(data, alpha, beta, k):
    """
    根据节点的度分布对节点特征进行随机丢弃。
    Args:
        data (object): 包含图结构的 PyG 数据对象，`data.x` 是特征矩阵。
        alpha (float): 高度节点特征丢弃的概率。
        beta (float): 低度节点特征丢弃的概率。
        k (int): 度阈值，大于 k 的节点被视为高出度节点。
    Returns:
        torch.Tensor: 丢弃后的特征矩阵。
    """
    x = data.x  # 节点特征矩阵 (num_nodes, num_features)
    degree = data.degree  # 节点度 (num_nodes,)
    # 根据度计算丢弃概率
    drop_prob = torch.where(degree > k, alpha, beta).unsqueeze(1)  # (num_nodes, 1)
    # 随机生成丢弃掩码
    drop_mask = torch.rand_like(x, device=x.device) < drop_prob  # (num_nodes, num_features)
    # 克隆并更新特征矩阵
    x = x.clone()
    x[drop_mask] = 0  # 丢弃特征

    return x



def generate_pos_mask(data, k):
    # Initialize a zero matrix of size (num_nodes, num_nodes)
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    pos_mask = torch.ones((num_nodes, num_nodes), dtype=torch.float32)

    # Iterate through edge_index to mark positive samples
    for i in range(edge_index.shape[1]):
        start_node = edge_index[0, i]  # Starting node of the edge
        end_node = edge_index[1, i]  # Ending node of the edge

        # Mark the corresponding position in the matrix as a positive sample (1)
        pos_mask[start_node, end_node] = k

    return pos_mask


def find_special_structures(data, k):
    """
    查找具有特定拓扑结构的边（如星形结构的出边）。
    Args:
        data (object): 包含图结构的 PyG 数据对象。
        k (int): 度阈值，大于 k 的节点被视为高出度节点。
    Returns:
        list: 特殊结构的边列表 [(src, dst), ...]。
        list: 高出度节点列表。
    """
    edge_index = data.edge_index  # (2, num_edges)
    num_nodes = data.num_nodes

    # 计算出度
    out_degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    out_degree.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.long))

    # 筛选高出度节点
    high_out_degree_nodes = (out_degree > k).nonzero(as_tuple=True)[0]

    # 获取高出度节点的出边
    mask = torch.isin(edge_index[0], high_out_degree_nodes)  # 筛选第一列为高出度节点的边
    special_edges = edge_index[:, mask].t().tolist()
    special_edges = torch.tensor(special_edges, dtype=torch.long, device=data.edge_index.device)
    return special_edges


def drop_edges(data, alpha, beta, special_edges):
    """
    从图的边集中按照比例随机删除特定边和普通边。

    Args:
        data (object): 包含边信息的图数据对象，`data.edge_index` 为 (2, num_edges) 的边索引张量。
        alpha (float): 特殊边删除的比例，范围 [0, 1]。
        beta (float): 普通边删除的比例，范围 [0, 1]。
        special_edges (torch.Tensor): 特殊边的集合，形状为 (num_special_edges, 2)。

    Returns:
        torch.Tensor: 过滤后的边索引张量，形状为 (2, num_remaining_edges)。
    """
    edge_index = data.edge_index  # 原始边集合，形状为 (2, num_edges)

    # 将特殊边转换为集合，便于匹配
    special_edges_set = set(map(tuple, special_edges.tolist()))

    # 转换边为列表
    edge_list = edge_index.t().tolist()

    # 标记哪些边是特殊边
    is_special_edge = torch.tensor([tuple(edge) in special_edges_set for edge in edge_list])

    # 分离特殊边和普通边
    special_edge_indices = is_special_edge.nonzero(as_tuple=True)[0]
    normal_edge_indices = (~is_special_edge).nonzero(as_tuple=True)[0]

    # 删除特殊边的随机采样
    num_special_to_drop = int(alpha * len(special_edge_indices))
    drop_special_indices = torch.randperm(len(special_edge_indices))[:num_special_to_drop]

    # 删除普通边的随机采样
    num_normal_to_drop = int(beta * len(normal_edge_indices))
    drop_normal_indices = torch.randperm(len(normal_edge_indices))[:num_normal_to_drop]

    # 构造删除的索引
    drop_indices = torch.cat([special_edge_indices[drop_special_indices], normal_edge_indices[drop_normal_indices]])

    # 创建新的边索引
    keep_mask = torch.ones(len(edge_list), dtype=torch.bool)
    keep_mask[drop_indices] = False
    new_edge_index = edge_index[:, keep_mask]

    return new_edge_index


def calculate_epr_aupr(GRN_df, label_file, label_column1, label_column2, TF_column, Target_column, value_column):
    """
    计算EPR和AUPR比率

    Parameters:
    - GRN_df: 基因调控网络DataFrame
    - label_file: 标签文件路径
    - label_column1: 标签文件中的第一个基因列
    - label_column2: 标签文件中的第二个基因列
    - TF_column: GRN_df中的TF列
    - Target_column: GRN_df中的Target列
    - value_column: GRN_df中的value列

    Returns:
    - epr: EPR比率
    - aupr_ratio: AUPR比率
    """

    # 读取标签文件
    label_df = pd.read_csv(label_file)
    TFs = set(label_df[label_column1])
    Genes = set(label_df[label_column1]) | set(label_df[label_column2])
    GRN_df1 = GRN_df
    # 筛选GRN数据
    GRN_df_filtered = GRN_df[GRN_df[TF_column].apply(lambda x: x in TFs)]
    GRN_df_filtered = GRN_df_filtered[GRN_df_filtered[Target_column].apply(lambda x: x in Genes)]

    # 创建标签集
    label_set = set(label_df[label_column1] + '|' + label_df[label_column2])
    label_set1 = set(label_df[label_column1] + label_df[label_column2])

    # 计算EPR
    GRN_df_filtered = GRN_df_filtered.iloc[:len(label_set)]
    EPR = len(set(GRN_df_filtered[TF_column] + '|' + GRN_df_filtered[Target_column]) & label_set) / (
            len(label_set) ** 2 / (len(TFs) * len(Genes) - len(TFs)))
    print(f"{label_file.split('/')[-1]} EPR:", EPR)

    # 计算AUPR
    res_d = {}
    l, p = [], []
    for item in GRN_df1.to_dict('records'):
        res_d[item[TF_column] + item[Target_column]] = item[value_column]

    for item in set(label_df[label_column1]):
        for item2 in set(label_df[label_column1]) | set(label_df[label_column2]):
            if item + item2 in label_set1:
                l.append(1)
            else:
                l.append(0)
            if item + item2 in res_d:
                p.append(res_d[item + item2])
            else:
                p.append(-1)

    aupr_ratio = average_precision_score(l, p) / np.mean(l)
    print(f"{label_file.split('/')[-1]} AUPR ratio:", aupr_ratio)

    return EPR, aupr_ratio

def parse_hidden_layers(hidden_layers_str):
    # 解析格式为 '40 40 5,16 16 2' 的字符串
    hidden_layers = []
    for layer_str in hidden_layers_str.split(','):
        dims = list(map(int, layer_str.split()))
        hidden_layers.append(dims)
    return hidden_layers



def add_relative_gaussian_noise(adata, mean, noise_factor):
    """
    Add Gaussian noise proportional to each cell's or gene's expression.
    """
    noise = np.random.normal(loc=mean, scale=noise_factor * np.abs(adata.X), size=adata.X.shape)
    adata.X = adata.X + noise
    return adata
