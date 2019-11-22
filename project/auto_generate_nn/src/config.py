import os
import argparse
import logging
import json
logger = logging.getLogger(__name__)
from shutil import copyfile

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
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--check_per_epoch', type=int, default=8, help='batch size')
parser.add_argument('--img_size', type=int, default=256, help='batch size')
parser.add_argument('--hidden_size', type=int, default=96, help='default size for RNN layer')
parser.add_argument('--number_of_labels', type=int, default=7, help='number of labels')
parser.add_argument('--dropout_emb', type=float, default=0.4, help='dropout rate for embeddings')
parser.add_argument('--pretrained', type=str, default='', help='pretrained model path')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
args = parser.parse_args()

print(args)

if args.pretrained:
    assert all(os.path.exists(p) for p in args.pretrained.split(',')), 'Checkpoint %s does not exist.' % args.pretrained


def create_logo_generator_cfg(path, width=512, heigth=512, images=64, objects=1000, noise=False, ):
    config = dict()

    config["width"] = width
    config["heigth"] = heigth
    config["images"] = images
    config["objects"] = objects


    with open(path, "w") as json_file:
        json.dump(config, json_file)


def copy_standard_config(to_path, from_path="/home/petigep/college/orak/digikep2/Digikep2_logo/Generator/Linux/config_standard.json"):
    copyfile(from_path, to_path)


def setConstanst_Value(value, from_path):
    with open(from_path, "r") as in_f:
        x = json.load(in_f)
    x["image"]["constantSetupRate"] = value

    with open(from_path, "w") as out_f:
        json.dump(x, out_f, indent=4)
