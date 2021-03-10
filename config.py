import torch
import argparse
import os

save_dir = './experiment/'

slide_dir = r"your dataset dir"
mask_folder_all = os.path.join(slide_dir, r'your foremask dir')

csv_dir = './dataset_csv/'
csv_all = os.path.join(csv_dir, r'your dataset csv')

patch_size = 512

scale_dict_checked = {'slide':8, 'bottom':8, 'mid': 4, 'ignored': 2, 'tip': 1}
scale_openslide_dict_checked = {'slide':0, 'bottom':0, 'mid': 1, 'ignored': 2, 'tip': 3}

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_split", type=int, default=-1)

    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--soft_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--lr_extractor_factor", type=float, default=1)
    parser.add_argument("--lr_fam_factor", type=float, default=1)
    parser.add_argument("--lr_agg_factor", type=float, default=1)

    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momen", type=float, default=0.9)
    parser.add_argument("--log_path", type=str, default=save_dir)
    parser.add_argument("--theme", type=str, default="")
    # parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--job_type", type=str, default='S')

    parser.add_argument("--sparse_ratio", type=float, default=10)
    parser.add_argument("--bottom2mid_ratio", type=float, default=1)
    parser.add_argument("--mid2tw_ratio", type=float, default=1)
    parser.add_argument("--gcn2fam_ratio", type=float, default=1)
    parser.add_argument("--gcn_ratio", type=float, default=1)

    parser.add_argument("--extractor", type=str, default='resnet18')
    parser.add_argument("--extractor_pretrained", type=int, default=0)
    parser.add_argument("--instance_affine", type=int, default=0)
    parser.add_argument('--ndims_rnn', default=128, type=int, help='length of hidden representation (default: 128)')
    parser.add_argument('--patch_limits', default=128, type=int, help='max num of patches in single forward of extractor')
    parser.add_argument('--fam_limits', default=8, type=int, help='max num of patches in single forward of extractor')

    parser.add_argument('--pool_type', default=None, type=str, help='what pool to replace RNN')
    parser.add_argument('--pool_attention', default=0, type=int, help='whether use attention for pooling')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import json

    args = get_args()
    print(args.__dict__)
    print(type(args.__dict__))

    with open('./args.json', 'w') as f:
        # json.dump(args.__dict__, f)
        f.write(json.dumps(args.__dict__, indent=4))

