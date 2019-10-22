import os
import argparse
import logging

logger = logging.getLogger(__name__)

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

parser = argparse.ArgumentParser()
parser.register('type', 'bool', str2bool)
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--epoch', type=int, default=50, help='Number of epoches to run')
parser.add_argument('--optimizer', type=str, default='adamax', help='optimizer, adamax or sgd')
parser.add_argument('--use_cuda', type='bool', default=True, help='use cuda or not')
parser.add_argument('--grad_clipping', type=float, default=10.0, help='maximum L2 norm for gradient clipping')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--check_per_epoch', type=int, default=8, help='batch size')
parser.add_argument('--img_size', type=int, default=256, help='batch size')
parser.add_argument('--hidden_size', type=int, default=96, help='default size for RNN layer')
parser.add_argument('--dropout_emb', type=float, default=0.4, help='dropout rate for embeddings')
parser.add_argument('--pretrained', type=str, default='', help='pretrained model path')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
args = parser.parse_args()

print(args)

if args.pretrained:
    assert all(os.path.exists(p) for p in args.pretrained.split(',')), 'Checkpoint %s does not exist.' % args.pretrained
