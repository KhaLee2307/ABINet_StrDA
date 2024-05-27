import os
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

from utils import Config, Averager, FocalLoss

from source.dataset_customize import Pseudolabel_Dataset, hierarchical_dataset, get_dataloader
from source.indexer import IndexingIntermediate
from source.model_abinet import BaseVision_DomainDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    #assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    # print(model.state_dict())
    return model

def main(opt):
    dashed_line = "-" * 80

    # load target data (raw)
    select_data = list(np.load(opt.select_data))
    intermediate, _ = hierarchical_dataset(opt.adapt_data, opt, mode = "raw")

    # setup indexer (DD approach)
    discrimination = IndexingIntermediate(opt, select_data)

    # load target data
    target_data_raw = Subset(intermediate, select_data)
    target_data = Pseudolabel_Dataset(target_data_raw, np.full(len(target_data_raw), 1))

    print(dashed_line)

    # load source data
    source_data_raw, _ = hierarchical_dataset(opt.source_data, opt, mode = "raw")
    source_data = Pseudolabel_Dataset(source_data_raw, np.full(len(source_data_raw), 0))

    del target_data_raw, source_data_raw

    print(dashed_line)

    # setup model
    model = BaseVision_DomainDiscriminator(opt.config)

    if opt.infer or opt.pretrained:
        model = torch.nn.DataParallel(model).to(device)
        model = load(model, opt.config.model_checkpoint, device=device, strict = True)
    else:
        # load pretrained model (baseline)
        model = load(model, opt.config.model_checkpoint, device=device, strict = False)

        # data parallel for multi-GPU
        model = torch.nn.DataParallel(model).to(device)

    model.train()
    #print(model.state_dict())

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print(f"Trainable params num: {sum(params_num)}")

    # setup loss (not contain sigmoid function)
    # criterion = FocalLoss().to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    # get dataloader
    source_loader = get_dataloader(opt, source_data, opt.batch_size, shuffle=True)
    target_loader = get_dataloader(opt, target_data, opt.batch_size, shuffle=True)
    # set up iter dataloader
    source_loader_iter = iter(source_loader)

    del source_data, target_data

    # start training
    if (opt.infer == False):
        model.train()

        # set up optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay = 0.005)

        # set up scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=opt.lr,
                    cycle_momentum=False,
                    div_factor=20,
                    final_div_factor=1000,
                    total_steps=opt.num_epoch * (len(select_data) // opt.batch_size + 1),
                )

        # train
        train_loss_avg = Averager()

        for epoch in tqdm(range(opt.num_epoch)):

            for batch in tqdm(target_loader):

                images_target_tensor, labels_target = batch

                try:
                    images_source_tensor, labels_source = next(source_loader_iter)
                except StopIteration:
                    del source_loader_iter
                    source_loader_iter = iter(source_loader)
                    images_source_tensor, labels_source = next(source_loader_iter)

                images_tensor = torch.cat((images_source_tensor, images_target_tensor), 0)
                labels = labels_source + labels_target
                images = images_tensor.to(device)
                preds = model(images)
                loss = criterion(preds, torch.Tensor(labels).view(-1,1).to(device))

                # optimize
                model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opt.grad_clip
                )   # gradient clipping with 5 (Default)
                optimizer.step()
                train_loss_avg.add(loss)
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            valid_log = f'\n\nEpoch: {epoch + 1}/{opt.num_epoch}: '
            valid_log += f'Train_loss: {train_loss_avg.val():0.5f}, Current_lr: {lr:0.7f}\n'
            print(valid_log)
            train_loss_avg.reset()

            torch.save(
                model.state_dict(),
                f"indexing/{opt.approach}/discriminator.pth"
            )

        model.eval()

    del source_loader_iter, source_loader, target_loader

    # indexing intermediate domain
    discrimination.select_DD(intermediate, model)

    print(dashed_line)


if __name__ == "__main__":
    """ Argument """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_data",
        default="../../data/train/synth/",
        help="path to training dataset",
    )
    parser.add_argument(
        "--adapt_data",
        default="../../data/train/real/",
        help="path to adaptation dataset",
    )
    parser.add_argument(
        "--saved_model",
        default="trained_model/TRBA.pth",
        help="path to pretrained model"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--batch_size_val", type=int, default=512, help="input batch size val")
    parser.add_argument("--num_epoch", type=int, default=50, help="number of epoch")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument(
        "--grad_clip", type=float, default=20, help="gradient clipping value. default=5"
    )
    """ Data Processing """
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    """ Model Architecture """
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=128, help="the width of the input image"
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=3,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    """ Optimizer """
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="learning rate, 0.0005 for Adam",
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
    parser.add_argument("--infer", action='store_true', default=False, help='inference or not')

    parser.add_argument('--model_eval', type=str, default='alignment',
                choices=['alignment', 'vision', 'language'])
    parser.add_argument('--config', type=str, default='configs/train_abinet.yaml',
                    help='path to config file')
    parser.add_argument("--pretrained", action='store_true', default=False, help='pretrained or not')

    opt = parser.parse_args()

    config = Config(opt.config)
    config.model_checkpoint = opt.saved_model
    config.model_eval = opt.model_eval
    config.device = device
    config.charset_path = "data/DD.txt"
    opt.config = config

    """ Seed and GPU setting """
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    cudnn.benchmark = True  # it fasten training.
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
