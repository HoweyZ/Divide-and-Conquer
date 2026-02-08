# mp/graph_transformer_encoder.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from torch_scatter import scatter_add, scatter_mean


# ====== 你提供的 GraphTransformer 实现（原样粘贴） ======
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
    return func

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(g)
        head_out = g.ndata['wV'] / (g.ndata['z'] + 1e-12)
        return head_out

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0,
                 layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, h):
        h_in1 = h
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.out_channels)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)

        if self.residual:
            h = h_in1 + h

        if self.layer_norm:
            h = self.layer_norm1(h)
        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2(h)
        if self.batch_norm:
            h = self.batch_norm2(h)

        return h


def _get_edge_index_from_complexbatch(data):

    c0 = data.cochains[0]

    if not hasattr(c0, "upper_index") or c0.upper_index is None:
        raise AttributeError("cochains[0].upper_index 不存在或为 None，无法构造 1-skeleton edge_index")

    edge_index = c0.upper_index

    if edge_index.dim() != 2:
        raise ValueError(f"upper_index 维度不对，期望 2D，实际 {edge_index.dim()}D")

    if edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()

    if edge_index.size(0) != 2:
        raise ValueError(f"upper_index shape 不对，期望 [2, E]，实际 {tuple(edge_index.shape)}")

    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()

    return edge_index



class GraphTransformerEncoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 96,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 readout: str = "mean",
                 layer_norm: bool = False,
                 batch_norm: bool = True,
                 residual: bool = True,
                 use_bias: bool = False):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        self.hidden_dim = hidden_dim
        self.readout = readout

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                  dropout=dropout, layer_norm=layer_norm,
                                  batch_norm=batch_norm, residual=residual, use_bias=use_bias)
            for _ in range(num_layers)
        ])

    def _build_dgl_graph(self, edge_index, num_nodes, device):
        src = edge_index[0].to(device)
        dst = edge_index[1].to(device)
    
        # 加反向边（无向化），GPU OK
        src2 = torch.cat([src, dst], dim=0)
        dst2 = torch.cat([dst, src], dim=0)
    
        g = dgl.graph((src2, dst2), num_nodes=num_nodes, device=device)
    
        # 可选：去自环（通常没必要；如果你想要自环反而要 add_self_loop）
        # g = dgl.remove_self_loop(g)
    
        return g



    def _pool(self, h, batch):
        if self.readout == "sum":
            return scatter_add(h, batch, dim=0)
        if self.readout == "mean":
            return scatter_mean(h, batch, dim=0)
        raise NotImplementedError(self.readout)

    def forward(self, data):
        c0 = data.cochains[0]
        x = c0.x
        batch = c0.batch
        edge_index = _get_edge_index_from_complexbatch(data)

        g = self._build_dgl_graph(edge_index, num_nodes=x.size(0), device=x.device)

        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(g, h)

        hg = self._pool(h, batch)
        return hg

    def cell_mask_forward(self, data, mask_model=None):
        # 先算节点隐表示，再对节点做 mask（兼容 SampleSelector/FeatureMask），再 pooling
        c0 = data.cochains[0]
        x = c0.x
        batch = c0.batch
        edge_index = _get_edge_index_from_complexbatch(data)

        g = self._build_dgl_graph(edge_index, num_nodes=x.size(0), device=x.device)

        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(g, h)

        if mask_model is not None:
            h = mask_model(h)

        hg = self._pool(h, batch)
        return hg

    @torch.no_grad()
    def get_mask_embeddings(self, loader, device, mask_model):
        self.eval()
        ret, y = [], []
        for data in loader:
            data = data.to(device)
            emb = self.cell_mask_forward(data=data, mask_model=mask_model)
            ret.append(emb.cpu().numpy())
            y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
