# semiSupervised/parser.py
import os
import time
import argparse
from definitions import ROOT_DIR


def get_parser():
    parser = argparse.ArgumentParser(description='CWN experiment.')

    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to set')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any')
    parser.add_argument('--model', type=str, default='sparse_cin',
                        help='model, possible choices: cin, sparse_cin, cin++, graph_transformer')


    parser.add_argument('--use_coboundaries', type=str, default='True',
                        help='whether to use coboundary features for up-messages in sparse_cin (string True/False)')
    parser.add_argument('--include_down_adj', action='store_true',
                        help='whether to use lower adjacencies (i.e. CIN++ networks)')

    parser.add_argument('--indrop_rate', type=float, default=0.0)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_position', type=str, default='lin2')

    parser.add_argument('--nonlinearity', type=str, default='relu')
    parser.add_argument('--readout', type=str, default='sum')
    parser.add_argument('--final_readout', type=str, default='sum')
    parser.add_argument('--readout_dims', type=int, nargs='+', default=(0, 1, 2))
    parser.add_argument('--jump_mode', type=str, default='cat')
    parser.add_argument('--graph_norm', type=str, default='bn', choices=['bn', 'ln', 'id'])

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=32)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--dataset', type=str, default="PROTEINS")
    parser.add_argument('--max_dim', type=int, default=2)
    parser.add_argument('--max_ring_size', type=int, default=5)

    parser.add_argument('--result_folder', type=str, default=os.path.join(ROOT_DIR, 'results'))
    parser.add_argument('--exp_name', type=str, default=str(time.time()))
    parser.add_argument('--dump_curves', action='store_true')
    parser.add_argument('--untrained', action='store_true')

    parser.add_argument('--init_method', type=str, default='mean')
    parser.add_argument('--train_eval_period', type=int, default=10)

    parser.add_argument('--flow_points', type=int, default=400)
    parser.add_argument('--flow_classes', type=int, default=3)

    parser.add_argument('--train_orient', type=str, default='default')
    parser.add_argument('--test_orient', type=str, default='default')
    parser.add_argument('--fully_orient_invar', action='store_true')

    parser.add_argument('--use_edge_features', action='store_true')
    parser.add_argument('--simple_features', action='store_true')
    parser.add_argument('--early_stop', action='store_true')

    parser.add_argument('--paraid', type=int, default=0)
    parser.add_argument('--preproc_jobs', type=int, default=2)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--no_sencond_order', action="store_true")

    # ---------- stage control ----------
    parser.add_argument('--do_pretrain', action='store_true',
                        help='run supervised pretraining and save pt')
    parser.add_argument('--do_probe', action='store_true',
                        help='run gradient probe')
    parser.add_argument('--do_train', action='store_true',
                        help='run weighted training')
    parser.add_argument('--do_test', action='store_true',
                        help='run test/evaluation on target dataset')
    parser.add_argument('--ckpt_path', type=str, default='pretrained.pt',
                        help='path to save/load pretrained encoder')

    # ---------- dataset mix ----------
    parser.add_argument('--mix_datasets', type=str,
                        default='PROTEINS,MUTAG,NCI1,NCI109',
                        help='comma-separated 4 dataset names for pretrain/probe/train mix')
    parser.add_argument('--pretrain_ratio', type=float, default=0.10)
    parser.add_argument('--probe_ratio', type=float, default=0.05)

    # ---------- supervised pretrain ----------
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # ---------- probe ----------
    parser.add_argument('--probe_batches', type=int, default=20)



    parser.add_argument("--auto_heatmap_ft", type=str, default="true")
    parser.add_argument("--heatmap_ft_strength", type=float, default=1.0)
    parser.add_argument("--heatmap_ft_min", type=float, default=0.2)
    parser.add_argument("--heatmap_ft_max", type=float, default=5.0)
    
    parser.add_argument(
    "--freeze_space", type=str, default="none",
    help="Ablation: freeze one partition (none|A|B|C|D|all).")



    parser.add_argument('--topk_ratio_con', type=float, default=0.20)
    parser.add_argument('--topk_ratio_ce', type=float, default=0.20)

  
    parser.add_argument(
        '--probe_data_mode', type=str, default='target',
        choices=['target', 'mixed'],
        help="probe data distribution: 'target' uses --dataset; 'mixed' uses --mix_datasets"
    )
    parser.add_argument(
        '--heatmap_data_mode', type=str, default='target',
        choices=['target', 'mixed'],
        help="heatmap data distribution: 'target' uses --dataset; 'mixed' uses --mix_datasets"
    )
    parser.add_argument(
        '--train_data_mode', type=str, default='target',
        choices=['target', 'mixed'],
        help="weighted train data distribution: 'target' uses --dataset; 'mixed' uses --mix_datasets"
    )

   
    parser.add_argument('--overlap_pct_con', type=float, default=70.0,
                        help='percentile threshold for contrastive head selection (lower -> select more)')
    parser.add_argument('--overlap_pct_ce', type=float, default=70.0,
                        help='percentile threshold for CE head selection (lower -> select more)')

  
    parser.add_argument('--head_select_mode', type=str, default='percentile',
                        choices=['percentile', 'topk'],
                        help="how to select heads: 'percentile' or 'topk'")

    parser.add_argument('--head_select_per_layer', action='store_true', default=True,
                        help='select heads per layer (default: True)')
    parser.add_argument('--no_head_select_per_layer', action='store_false', dest='head_select_per_layer',
                        help='disable per-layer head selection')

    parser.add_argument('--overlap_auto_relax', action='store_true', default=True,
                        help='auto relax percentile thresholds if overlap is empty (default: True)')
    parser.add_argument('--no_overlap_auto_relax', action='store_false', dest='overlap_auto_relax',
                        help='disable auto relax')
    parser.add_argument('--overlap_relax_step', type=float, default=5.0)
    parser.add_argument('--overlap_relax_min', type=float, default=0.0)

    parser.add_argument('--topk_head_k_con', type=int, default=0,
                        help='topk heads per layer for contrastive (0 means use ratio)')
    parser.add_argument('--topk_head_k_ce', type=int, default=0,
                        help='topk heads per layer for CE (0 means use ratio)')
    parser.add_argument('--topk_head_ratio_con', type=float, default=0.2)
    parser.add_argument('--topk_head_ratio_ce', type=float, default=0.2)

    # ---------- heatmap (optional; simgrace_meta.py uses getattr with defaults) ----------
    parser.add_argument('--heatmap_batches', type=int, default=8,
                        help='number of batches to accumulate heatmap')
    parser.add_argument('--heatmap_ratio', type=float, default=None,
                        help='subsample ratio for heatmap loader (default: use probe_ratio)')
    parser.add_argument('--heatmap_normalize_mode', type=str, default='mean',
                        choices=['none', 'mean', 'per_head'],
                        help='normalize heatmap accumulation')
    parser.add_argument('--heatmap_binary', action='store_true', default=False,
                        help='use percentile binary mask for visualization')
    parser.add_argument('--heatmap_keep_percentile', type=float, default=85.0)
    parser.add_argument('--heatmap_percentile_mode', type=str, default='global',
                        choices=['global', 'per_row'])

    parser.add_argument('--heatmap_topk_dim_k', type=int, default=None,
                        help='keep top-k dims in each row (if set, overrides ratio)')
    parser.add_argument('--heatmap_topk_dim_ratio', type=float, default=0.8)
    parser.add_argument('--heatmap_topk_mode', type=str, default='per_row',
                        choices=['per_row', 'global'])

    # ---------- gradient weighting ----------
    parser.add_argument('--w_A', type=float, default=1.5)
    parser.add_argument('--w_B', type=float, default=1.5)
    parser.add_argument('--w_C', type=float, default=2.0)
    parser.add_argument('--w_D', type=float, default=0.5)

    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--layer_norm', action='store_true')

    # ---------- embedding visualization (optional; simgrace_meta.py uses getattr) ----------
    parser.add_argument('--emb_vis_method', type=str, default='tsne', choices=['tsne', 'umap'])
    parser.add_argument('--tsne_perplexity', type=int, default=10)
    parser.add_argument('--umap_n_neighbors', type=int, default=15)
    parser.add_argument('--umap_min_dist', type=float, default=0.1)
    parser.add_argument('--umap_metric', type=str, default='euclidean')

    return parser
