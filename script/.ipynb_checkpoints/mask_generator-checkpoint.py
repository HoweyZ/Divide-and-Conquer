# semiSupervised/mask_generator.py
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from mp.models import CIN0, SparseCIN, CINpp
from mp.graph_transformer_encoder import GraphTransformerEncoder


def _str2bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y")


class FeatureMask(nn.Module):
    def __init__(self, feature_out_dim: int):
        super().__init__()
        self.mask_encoder = nn.Linear(feature_out_dim, feature_out_dim)
        init.constant_(self.mask_encoder.weight, 1)

    def forward(self, x):
        return torch.sigmoid(self.mask_encoder(x))


class SampleSelector(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = nn.Linear(input_dim, 2)

    def gumbel_softmax(self, logits, tau=1.0, hard=True):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = F.softmax((logits + gumbel_noise) / tau, dim=-1)
        if hard:
            y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        return y

    def forward(self, x):
        logits = self.mlp(x)
        mask = self.gumbel_softmax(logits, tau=0.5, hard=True)
        binary_mask = mask[:, 1]
        masked_x = x * binary_mask.unsqueeze(-1)
        return masked_x


class MaskSimclr(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()

        use_coboundaries = _str2bool(getattr(args, "use_coboundaries", "false"))
        readout_dims = tuple(sorted(getattr(args, "readout_dims", (0, 1, 2))))

        if args.model == "graph_transformer":
            
            #in_dim = dataset.num_features_in_dim(0)
            
            in_dim = int(getattr(args, "gt_in_dim", dataset.num_features_in_dim(0)))

            # 关键修正：
            # 你的命令传了 --emb_dim 32，工程其它地方也用 args.emb_dim
            # 所以这里不要硬编码 96，直接用 args.emb_dim
            hidden_dim = int(getattr(args, "emb_dim", 512))

            self.encoder = GraphTransformerEncoder(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=args.num_layers,
                num_heads=int(getattr(args, "num_heads", 4)),
                dropout=float(getattr(args, "drop_rate", 0.0)),
                readout=getattr(args, "readout", "mean"),
                layer_norm=bool(getattr(args, "layer_norm", False)),
                batch_norm=True,
                residual=True,
                use_bias=False,
            )
            self.embedding_dim = hidden_dim

        elif args.model == "cin":
            self.encoder = CIN0(
                dataset.num_features_in_dim(0),
                dataset.num_classes,
                args.num_layers,
                args.emb_dim,
                dropout_rate=args.drop_rate,
                max_dim=dataset.max_dim,
                jump_mode=args.jump_mode,
                nonlinearity=args.nonlinearity,
                readout=args.readout,
            )
            self.embedding_dim = getattr(self.encoder, "output_dim", args.emb_dim)

        elif args.model == "sparse_cin":
            self.encoder = SparseCIN(
                dataset.num_features_in_dim(0),
                dataset.num_classes,
                args.num_layers,
                args.emb_dim,
                dropout_rate=args.drop_rate,
                max_dim=dataset.max_dim,
                jump_mode=args.jump_mode,
                nonlinearity=args.nonlinearity,
                readout=args.readout,
                final_readout=args.final_readout,
                apply_dropout_before=args.drop_position,
                use_coboundaries=use_coboundaries,
                graph_norm=args.graph_norm,
                readout_dims=readout_dims,
            )
            self.embedding_dim = getattr(self.encoder, "output_dim", args.emb_dim)

        elif args.model == "cin++":
            self.encoder = CINpp(
                dataset.num_features_in_dim(0),
                dataset.num_classes,
                args.num_layers,
                args.emb_dim,
                dropout_rate=args.drop_rate,
                max_dim=dataset.max_dim,
                jump_mode=args.jump_mode,
                nonlinearity=args.nonlinearity,
                readout=args.readout,
                final_readout=args.final_readout,
                apply_dropout_before=args.drop_position,
                use_coboundaries=use_coboundaries,
                graph_norm=args.graph_norm,
                readout_dims=readout_dims,
            )
            self.embedding_dim = getattr(self.encoder, "output_dim", args.emb_dim)

        else:
            raise ValueError(f"Unknown args.model = {args.model}")

        self.proj_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, complexBatch):
        x = self.encoder(complexBatch)
        x = self.proj_head(x)
        return x

    def cell_mask_forward(self, complexBatch, cell_mask_model):
        x = self.encoder.cell_mask_forward(data=complexBatch, mask_model=cell_mask_model)
        x = self.proj_head(x)
        return x

    @staticmethod
    def simclr_loss(x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()
        return loss

    def encode(self, complexBatch):
        """Return encoder embedding BEFORE projection head."""
        return self.encoder(complexBatch)
