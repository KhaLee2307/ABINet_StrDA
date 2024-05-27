import os
import sys
import re
import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from utils import Config, CharsetMapper

from source.dataset import hierarchical_dataset
from source.indexer import IndexingIntermediate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    print(module_name, class_name)
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    return model


def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    #assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model


def main(opt):
    dashed_line = "-" * 80

    # load intermediate data (raw)
    select_data = list(np.load(opt.select_data))
    intermediate, _ = hierarchical_dataset(opt.adapt_data, opt, mode = "raw")

    """ model configuration """
    converter = CharsetMapper(filename=opt.config.dataset_charset_path,
                            max_length=opt.config.dataset_max_length + 1)

    opt.character = re.sub("['\u2591']", "",converter.alphanumeric)
    opt.num_class = converter.num_classes
    
    # setup model
    model = get_model(opt.config).to(device)
    # load pretrained model
    model = load(model, opt.config.model_checkpoint, device=device)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    # setup indexer (confidence approach)
    confidence = IndexingIntermediate(opt, select_data)

    # indexing
    confidence.select_confidence(intermediate, model, converter, opt)

    print(dashed_line)
        

if __name__ == "__main__":
    """ Argument """ 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapt_data",
        default="data/train/real/",
        help="path to adaptation dataset",
    )
    parser.add_argument(
        "--saved_model",
        required=True, 
        help="path to saved_model for prediction"
    )
    parser.add_argument("--batch_size_val", type=int, default=512, help="input batch size val")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    """ Data Processing """
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=128, help="the width of the input image"
    )
    """ Experiment """ 
    parser.add_argument(
        "--manual_seed", type=int, default=111, help="for random seed setting"
    )
    """ Adaptation """
    parser.add_argument(
        "--select_data",
        required=True,
        help="path to select data",
    )
    parser.add_argument("--approach", required=True, help="select indexing approach")
    parser.add_argument("--num_groups", type=int, required = True, help="number of intermediate data group")

    parser.add_argument('--model_eval', type=str, default='alignment', 
                choices=['alignment', 'vision', 'language'])
    parser.add_argument('--config', type=str, default='configs/train_abinet.yaml',
                    help='path to config file')

    opt = parser.parse_args()

    config = Config(opt.config)
    config.model_checkpoint = opt.saved_model
    config.model_eval = opt.model_eval
    config.device = device
    opt.config = config

    """ Seed and GPU setting """
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True  # It fasten training.
    cudnn.deterministic = True

    if sys.platform == "win32":
        opt.workers = 0

    opt.gpu_name = "_".join(torch.cuda.get_device_name().split())
    if sys.platform == "linux":
        opt.CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        opt.CUDA_VISIBLE_DEVICES = 0  # for convenience

    command_line_input = " ".join(sys.argv)
    print(
        f"Command line input: CUDA_VISIBLE_DEVICES={opt.CUDA_VISIBLE_DEVICES} python {command_line_input}"
    )

    main(opt)