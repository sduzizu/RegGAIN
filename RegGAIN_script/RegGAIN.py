import pandas as pd
import torch
import random
from torch_geometric.data import Data
from utils import drop_feature, generate_pos_mask, drop_edges, find_special_structures
from Model import MixHopModel, Model
from scanpy import AnnData
from utils import data_preparation, parse_hidden_layers, calculate_epr_aupr
import argparse
import os
import numpy as np
from tqdm import tqdm
import time


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train MixHop model on a remote server")
    parser.add_argument('--exp_data', type=str, required=True, help="Path to expression data CSV")
    parser.add_argument('--prior_net', type=str, required=True, help="Path to prior network CSV")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run the model")
    parser.add_argument('--repeat', type=int, default=10, help="Number of training repetitions")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility") 
    parser.add_argument('--k', type=int, default=50, help="degree centrality of special nodes")
    parser.add_argument('--adjacency_powers', type=int, nargs='+', default=[0, 1, 2], help="List of adjacency matrix powers.")
    parser.add_argument('--first_layer_dims', type=int, nargs='+', default=[80, 80, 10], help="Dimensions of embeddings for each layer.")
    parser.add_argument('--hidden_layer_dims_list', type=str, default="40 40 5,16 16 2",
                        help="Dimensions for hidden layers. Provide as comma-separated layers (e.g., '40 40 5,16 16 2').")
    parser.add_argument('--pos', type=int, default=10, help="gamma")
    parser.add_argument('--label_STRING', type=str, help="Path to label_STRING CSV")
    parser.add_argument('--label_Non_Spec', type=str, help="Path to label_Non-Spec CSV")
    parser.add_argument('--label_Specific', type=str, help="Path to label_Specific CSV")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    if torch.cuda.is_available():
        device = torch.device("cuda:1")  
        print(f"Using device: {device}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}") 
        print(f"Current device index: {torch.cuda.current_device()}") 
    else:
        device = torch.device("cpu")
        print("Using CPU, no GPU available.")


    def get_PYG_data(adata: AnnData) -> Data:
        # edge index
        source_nodes = adata.uns['edgelist']['from'].tolist()
        target_nodes = adata.uns['edgelist']['to'].tolist()
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long).to(device)
        print(f"    n_genes × n_cells = {adata.n_vars} × {adata.n_obs}")

        df_numeric = adata.to_df().T.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Convert to tensor and move to device
        x = torch.from_numpy(df_numeric.to_numpy()).float().to(device)
        num_nodes = x.size(0)
        degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
        degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=device))

        return Data(x=x, edge_index=edge_index, degree=degree)


    def train(model: Model, data, optimizer, edge_alpha1, edge_alpha2, edge_beta1, edge_beta2, k, node_alpha1,
              node_alpha2,
              node_beta1, node_beta2, special_edges):

        model.train()
        optimizer.zero_grad()
        edge_index_1 = drop_edges(data, edge_alpha1, edge_beta1, special_edges)
        edge_index_2 = drop_edges(data, edge_alpha2, edge_beta2, special_edges)

        x_1 = drop_feature(data, node_alpha1, node_beta1, k) 
        x_2 = drop_feature(data, node_alpha2, node_beta2, k) 

        z1_out, z1_in = model(x_1, edge_index_1)
        z2_out, z2_in = model(x_2, edge_index_2)

        loss_out = model.loss(z1_out, z2_out)
        loss_in = model.loss(z1_in, z2_in)
        loss = loss_in + loss_out

        loss.backward()
        optimizer.step()

        return loss.item()


    def test(model: Model, x, edge_index):
        model.eval()  
        with torch.no_grad(): 
            z_in, z_out = model(x, edge_index)
        return z_in, z_out


    # Load and process input data
    input_expdata = pd.read_csv(args.exp_data)
    input_priorNet = pd.read_csv(args.prior_net)
    adata = data_preparation(input_expdata, input_priorNet)
    pyg_data = get_PYG_data(adata)
    num_features = pyg_data.x.size(1)
    special_edges = find_special_structures(pyg_data, args.k)
    adjacency_powers = args.adjacency_powers  
    first_layer_dims = args.first_layer_dims 
    hidden_layer_dims_list = parse_hidden_layers(args.hidden_layer_dims_list)

    # Store embeddings from each repetition
    all_z_in = []
    all_z_out = []

    for run in range(args.repeat):
        encoder_out = MixHopModel(
            edge_index=pyg_data.edge_index, 
            input_dim=num_features,
            adjacency_powers=adjacency_powers,
            first_layer_dim_per_power=first_layer_dims,  
            hidden_layer_dims_per_power_list=hidden_layer_dims_list
        ).to(device)

        encoder_in = MixHopModel(
            edge_index=pyg_data.edge_index, 
            input_dim=num_features,
            adjacency_powers=adjacency_powers,
            first_layer_dim_per_power=first_layer_dims,
            hidden_layer_dims_per_power_list=hidden_layer_dims_list
        ).to(device)

        model = Model(encoder_out=encoder_out, encoder_in=encoder_in, num_proj_hidden=64, tau=3).to(
            device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        min_loss = np.inf
        for epoch in tqdm(range(args.epochs), desc=f'Run {run + 1}', unit='epoch'):
            loss = train(model, pyg_data, optimizer, edge_alpha1=0.6, edge_alpha2=0.3,
                         edge_beta1=0.3, edge_beta2=0.3, k=args.k,
                         node_alpha1=0.5, node_alpha2=0.2, node_beta1=0.2, node_beta2=0.2, special_edges=special_edges)
            if min_loss > loss:
                min_loss = loss

        z_in, z_out = test(model, pyg_data.x, pyg_data.edge_index)
        all_z_in.append(z_in.cpu().numpy())
        all_z_out.append(z_out.cpu().numpy())


    GRN_matrix_sum = None
    num_iterations = len(all_z_in) 
    original_indices = adata.var.index.tolist()  
    pos_mask = generate_pos_mask(pyg_data, args.pos)
    for z_in, z_out in zip(all_z_in, all_z_out):

        mean_z_in = np.mean(z_in, axis=1, keepdims=True)
        std_z_in = np.std(z_in, axis=1, keepdims=True)

        mean_z_out = np.mean(z_out, axis=1, keepdims=True) 
        std_z_out = np.std(z_out, axis=1, keepdims=True) 

        normalized_z_in = (z_in - mean_z_in) / std_z_in
        normalized_z_out = (z_out - mean_z_out) / std_z_out

        GRN_matrix = np.dot(normalized_z_out, normalized_z_in.T)
        rows, cols = np.indices(GRN_matrix.shape)

        GRN_matrix = GRN_matrix * pos_mask.numpy()
        np.fill_diagonal(GRN_matrix, 0)

        if GRN_matrix_sum is None:
            GRN_matrix_sum = GRN_matrix
        else:
            GRN_matrix_sum += GRN_matrix

        GRN_df = pd.DataFrame({
            'TF': [original_indices[i] for i in rows.flatten()],
            'Target': [original_indices[j] for j in cols.flatten()],
            'value': GRN_matrix.flatten()
        })
        GRN_df['value'] = abs(GRN_df['value'])
        GRN_df = GRN_df.sort_values('value', ascending=False)

    avg_GRN = GRN_matrix_sum /num_iterations

    avg_GRN_df = pd.DataFrame(avg_GRN)

    rows, cols = np.indices(avg_GRN.shape)
    average_GRN_df = pd.DataFrame({
        'TF': [original_indices[i] for i in rows.flatten()],
        'Target': [original_indices[j] for j in cols.flatten()],
        'value': avg_GRN.flatten()
    })
    average_GRN_df['value'] = abs(average_GRN_df['value'])
    average_GRN_df = average_GRN_df.sort_values('value', ascending=False)


    # Evaluate predicted GRN against gold standard networks
    calculate_epr_aupr(average_GRN_df, args.label_STRING, 'Gene1', 'Gene2', 'TF', 'Target', 'value')
    calculate_epr_aupr(average_GRN_df, args.label_Non_Spec, 'Gene1', 'Gene2', 'TF', 'Target', 'value')
    calculate_epr_aupr(average_GRN_df, args.label_Specific, 'Gene1', 'Gene2', 'TF', 'Target', 'value')
