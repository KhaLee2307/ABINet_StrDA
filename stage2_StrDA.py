
import os
import sys
import re
import time
import random
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

from utils import Config, Logger, CharsetMapper, onehot, Averager

from source.dataset import Pseudolabel_Dataset, hierarchical_dataset, get_dataloader

from test import validation
from losses import MultiLosses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _decode(logit, charset):
    """ Greed decode """
    out = F.softmax(logit, dim=2)
    pt_text, pt_scores, pt_lengths = [], [], []
    for o in out:
        text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
        text = text.split(charset.null_char)[0]  # end at end-token
        pt_text.append(text)
        pt_scores.append(o.max(dim=1)[0])
        pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
    return pt_text

def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    print(module_name, class_name)
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    return model

def postprocess(output, charset, model_eval, training = False):
    def _get_output(last_output, model_eval, training):
        if training: 
            last_output = (last_output[0][-1], last_output[1][-1], last_output[2])
        if isinstance(last_output, (tuple, list)):
            for res in last_output:
                if res['name'] == model_eval: output = res
        else: output = last_output
        return output

    def _decode(logit):
        """ Greed decode """
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    output = _get_output(output, model_eval, training)
    logits, pt_lengths = output['logits'], output['pt_lengths']
    pt_text, pt_scores, pt_lengths_ = _decode(logits)

    return pt_text, pt_scores, pt_lengths_


def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    #assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model


def process_intermediate(opt, model, charset, inter_data, adapt_list, round):
    """ Make prediction and return them """

    # get adapt_data
    data = Subset(inter_data, adapt_list)
    data = Pseudolabel_Dataset(data, adapt_list)
    dataloader = get_dataloader(opt, data, opt.batch_size_val, shuffle=False)

    del data

    model.eval()
    with torch.no_grad():
        list_adapt_data = list()
        pseudo_adapt = list()

        mean_conf = 0

        for (image_tensors, image_indexs) in tqdm(dataloader):
            batch_size = len(image_indexs)
            image = image_tensors.to(device)
            res = model(image)

            # Select max probabilty (greedy decoding) then decode index to character
            pt_texts, pt_scores, __ = postprocess(res, charset, opt.config.model_eval)

            for pred, pred_max_prob, index in zip(
                pt_texts, pt_scores, image_indexs
            ):
                pred_max_prob = pred_max_prob[:len(pred)]
                # calculate confidence score (= multiply of pred_max_prob)
                if len(pred_max_prob.cumprod(dim=0)) > 0:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
                else:
                    confidence_score = 0

                if ( 
                    "[PAD]" in pred 
                    or "[UNK]" in pred 
                    or "[SOS]" in pred 
                ): 
                    continue
                list_adapt_data.append(index)
                pseudo_adapt.append(pred)
                mean_conf += confidence_score

    mean_conf /= (len(list_adapt_data))
    mean_conf = int(mean_conf * 100) / 100

    del dataloader

    # save pseudo-labels
    with open(f'indexing/{opt.approach}/pseudolabel_{round}.txt', "w") as file:
        for string in pseudo_adapt:
            file.write(string + "\n")

    # free cache
    torch.cuda.empty_cache()
                
    return list_adapt_data, pseudo_adapt, mean_conf

           
def self_training(opt, filtered_parameters, model, criterion, charset, \
                  source_loader, valid_loader, adapting_loader, mean_conf, round = 0):

    num_iter = (opt.total_iter // opt.val_interval) // opt.num_groups * opt.val_interval

    if round == 1:
        num_iter += (opt.total_iter // opt.val_interval) % opt.num_groups * opt.val_interval

    # set up iter dataloader
    source_loader_iter = iter(source_loader)
    adapting_loader_iter = iter(adapting_loader)

    # set up optimizer
    optimizer = torch.optim.AdamW(filtered_parameters, lr=opt.lr, weight_decay = 0.005)

    # set up scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt.lr,
                cycle_momentum=False,
                div_factor=20,
                final_div_factor=1000,
                total_steps=num_iter,
            )
    
    train_loss_avg = Averager()
    source_loss_avg = Averager()
    adapting_loss_avg = Averager()
    best_score = float('-inf')
    score_descent = 0

    log = ""

    model.train()
    # training loop
    for iteration in tqdm(
        range(0, num_iter + 1),
        total=num_iter,
        position=0,
        leave=True,
    ):
        if (iteration % opt.val_interval == 0):
            # valiation part
            model.eval()
            with torch.no_grad():
                (
                    valid_loss,
                    current_score,
                    preds,
                    confidence_score,
                    labels,
                    infer_time,
                    length_of_data,
                ) = validation(model, criterion, valid_loader, charset, opt)
            model.train()

            if (current_score >= best_score):
                score_descent = 0

                best_score = current_score
            else:
                score_descent += 1

            # Log
            lr = optimizer.param_groups[0]["lr"]
            valid_log = f'\nValidation at {iteration}/{num_iter}:\n'
            valid_log += f'Train_loss: {train_loss_avg.val():0.4f}, Valid_loss: {valid_loss:0.4f}, '
            valid_log += f'Source_loss: {source_loss_avg.val():0.4f}, Adapting_loss: {adapting_loss_avg.val():0.4f},\n'
            valid_log += f'Current_lr: {lr:0.7f}, '
            valid_log += f'Current_score: {current_score:0.2f}, Best_score: {best_score:0.2f}, '
            valid_log += f'Score_descent: {score_descent}\n'
            print(valid_log)

            log += valid_log

            log += "-" * 80 +"\n"

            train_loss_avg.reset()
            source_loss_avg.reset()
            adapting_loss_avg.reset()

        if iteration == num_iter:
            log += f'Stop training at iteration: {iteration}!\n'
            break

        # training part
        """ loss of source domain """
        try:
            images_source_tensor, labels_source = next(source_loader_iter)
        except StopIteration:
            del source_loader_iter
            source_loader_iter = iter(source_loader)
            images_source_tensor, labels_source = next(source_loader_iter)

        images_source = images_source_tensor.to(device)       
        
        labels_source_index, labels_source_length= [], []
        for label_source in labels_source:
            # post-process label
            label_source = label_source.lower()
            label_source = re.sub(charset.out_of_alphanumeric, "", label_source)
            
            label_source_length = torch.tensor([len(label_source)+ 1])
            label_source_index_ = torch.tensor(charset.get_labels(label_source))
            label_source_index = onehot(label_source_index_, charset.num_classes)
            
            labels_source_index.append(label_source_index)
            labels_source_length.append(label_source_length)

        labels_source_index = torch.stack(labels_source_index).to(device)
        labels_source_length = torch.cat(labels_source_length).to(device)

        preds_source = model(images_source)

        loss_source = criterion(
            preds_source, labels_source_index, label_source_length
        )
        
        # print(_decode(labels_source_index,charset))
        # print(postprocess(preds_source, charset, opt.config.model_eval, training= True)[0])
        # print(loss_source)
        """ loss of semi """
        try:
            images_unlabel_tensor, labels_adapting = next(adapting_loader_iter)
        except StopIteration:
            del adapting_loader_iter
            adapting_loader_iter = iter(adapting_loader)
            images_unlabel_tensor, labels_adapting = next(adapting_loader_iter)
        
        images_unlabel = images_unlabel_tensor.to(device)

        labels_adapting_index, labels_adapting_length= [], []
        for label_adapting in labels_adapting:
            label_adapting = label_adapting.lower()
            label_adapting = re.sub(charset.out_of_alphanumeric, "", label_adapting)
            

            label_adapting_length = torch.tensor([len(label_adapting)+ 1])
            label_adapting_index_ = torch.tensor(charset.get_labels(label_adapting))
            label_adapting_index = onehot(label_adapting_index_, charset.num_classes)
            
            labels_adapting_index.append(label_adapting_index)
            labels_adapting_length.append(label_adapting_length)

        labels_adapting_index = torch.stack(labels_adapting_index).to(device)
        labels_adapting_length = torch.cat(labels_adapting_length).to(device)

        batch_unlabel_size = len(labels_adapting)

        preds_adapting = model(images_unlabel)

        loss_adapting = criterion(
            preds_adapting, labels_adapting_index, labels_adapting_length
        )

        loss = (10 - mean_conf) * loss_source + loss_adapting * mean_conf
        #loss = loss_source
        model.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     model.parameters(), opt.grad_clip
        # )   # gradient clipping with 5 (Default)
        optimizer.step()

        train_loss_avg.add(loss)
        source_loss_avg.add(loss_source)
        adapting_loss_avg.add(loss_adapting)

        scheduler.step()

    model.eval()

    # save model
    torch.save(
        model.state_dict(),
        f"./trained_model/{opt.approach}/StrDA_round{round}.pth"
    )

    # save log
    print(log, file= open(f'log/{opt.approach}/log_self_training_round{round}.txt', 'w'))

    del optimizer, scheduler, source_loader_iter, adapting_loader_iter

    # free cache
    torch.cuda.empty_cache()


def main(opt):
    dashed_line = "-" * 80
    main_log = ""
    opt_log = dashed_line + "\n"
    
    converter = CharsetMapper(filename=opt.config.dataset_charset_path,
                            max_length=opt.config.dataset_max_length + 1)
    opt.character = re.sub("['\u2591']", "",converter.alphanumeric)
    opt.num_class = converter.num_classes

    """ create folder for log and trained model """
    if (not os.path.exists(f'log/{opt.approach}/')):
        os.makedirs(f'log/{opt.approach}/')
    if (not os.path.exists(f'trained_model/{opt.approach}/')):
        os.makedirs(f'trained_model/{opt.approach}/')

    """ dataset preparation """
    # source data
    source_data, source_data_log = hierarchical_dataset(opt.source_data, opt)
    source_loader = get_dataloader(opt, source_data, opt.batch_size, shuffle=True)
    
    opt_log += source_data_log

    # validation data
    valid_data, valid_data_log = hierarchical_dataset(opt.valid_data, opt)
    valid_loader = get_dataloader(opt, valid_data, opt.batch_size_val, shuffle = False) # 'True' to check training progress with validation function.
    
    opt_log += valid_data_log

    # adaptation data
    intermediate,  intermediate_log= hierarchical_dataset(opt.adapt_data, opt, mode = "raw")

    opt_log += intermediate_log

    del source_data, valid_data, source_data_log, valid_data_log, intermediate_log

    """ model configuration """
    
    # setup model
    model = get_model(opt.config).to(device)
    model = load(model, opt.config.model_checkpoint, device=device)

    opt_log += "Init model\n"

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    # load pretrained model
    torch.save(
            model.state_dict(),
            f"./trained_model/{opt.approach}/StrDA_round0.pth"
        )
    opt_log += "Load pretrained model\n"

    """ setup loss """
    criterion = MultiLosses(one_hot= True, device= device)

    # filter that only require gradient descent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print(f"Trainable params num: {sum(params_num)}")
    opt_log += f"Trainable params num: {sum(params_num)}"

    del params_num

    """ final options """
    opt_log += "------------ Options -------------\n"
    args = vars(opt)
    for k, v in args.items():
        if str(k) == "character" and len(str(v)) > 500:
            opt_log += f"{str(k)}: So many characters to show all: number of characters: {len(str(v))}\n"
        else:
            opt_log += f"{str(k)}: {str(v)}\n"
    opt_log += "---------------------------------------\n"
    print(opt_log)
    main_log += opt_log
    print("Start Adapting...\n")
    main_log += "Start Adapting...\n"
    
    for round in range(opt.num_groups):

        adapt_log = ""
        print(f"\nRound {round+1}/{opt.num_groups}: \n")
        adapt_log += f"\nRound {round+1}/{opt.num_groups}: \n"

        # load best model of previous round
        # adapt_log +=  f"- Load best model of previous round ({round}). \n"
        # pretrained = torch.load(f"./trained_model/{opt.approach}/StrDA_round{round}.pth")
        # model.load_state_dict(pretrained)
        # del pretrained

        # select intermediate domain
        adapting_list = list(np.load(f'indexing/{opt.approach}/intermediate_{round + 1}.npy'))

        # assign pseudo labels
        print("- Set up Intermediate data \n")
        adapt_log += "- Set up Intermediate data \n"
        list_adapt_data, pseudo_adapt, mean_conf = process_intermediate(
                opt, model, converter, intermediate, adapting_list, round + 1
            )

        data_log = ""
        data_log += f"-- Number of apating data: {len(list_adapt_data)} \n"
        data_log += f"-- Mean of confidences: {mean_conf} \n"

        print(data_log)
        adapt_log += data_log

        # restrict adapting data
        adapting_data = Subset(intermediate, list_adapt_data)
        adapting_data = Pseudolabel_Dataset(adapting_data, pseudo_adapt)

        if opt.aug == True:
            adapting_loader = get_dataloader(opt, adapting_data, opt.batch_size, shuffle=True, mode="adapt")
        else:
            adapting_loader = get_dataloader(opt, adapting_data, opt.batch_size, shuffle=True)

        del adapting_list, adapting_data, list_adapt_data, pseudo_adapt

        # self-training
        print(dashed_line)
        print("- Seft-training...")
        adapt_log += "\n- Seft-training"

        # adjust mean_conf (round_down)
        mean_conf = int(mean_conf * 10)

        self_training_start = time.time()
        if (round >= opt.checkpoint):
            self_training(opt, filtered_parameters, model, criterion, converter, \
                        source_loader, valid_loader, adapting_loader, mean_conf, round + 1)
        self_training_end = time.time()

        print(f"Processing time: {self_training_end - self_training_start}s")
        print(f"Saved log for adapting round to: 'log/{opt.approach}/log_self_training_round{round + 1}.txt'")
        adapt_log += f"\nProcessing time: {self_training_end - self_training_start}s"
        adapt_log += f"\nSaved log for adapting round to: 'log/{opt.approach}/log_self_training_round{round + 1}.txt'"

        adapt_log += "\n" + dashed_line + "\n"
        main_log += adapt_log

        print(dashed_line)
        print(dashed_line)
        print(dashed_line)
    
    # save log
    print(main_log, file= open(f'log/{opt.approach}/log_StrDA.txt', 'w'))
    return
            

if __name__ == "__main__":
    """ Argument """ 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_data",
        default="data/train/synth/",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--valid_data",
        default="data/val/",
        help="path to validation dataset",
    )
    parser.add_argument(
        "--adapt_data",
        default="data/train/real/",
        help="path to adaptation dataset",
    )
    parser.add_argument(
        "--saved_model",
        required=True, 
        help="path to saved_model to evaluation"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--batch_size_val", type=int, default=512, help="input batch size val")
    parser.add_argument("--total_iter", type=int, default=50000, help="number of iterations to train for each round")
    parser.add_argument("--val_interval", type=int, default=500, help="Interval between each validation")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5, help="gradient clipping value. default=5"
    )
    """ Data processing """
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=128, help="the width of the input image"
    )
    parser.add_argument(
        "--NED", action="store_true", help="For Normalized edit_distance"
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
    parser.add_argument("--approach", required = True, help="select indexing approach")
    parser.add_argument("--num_groups", type=int, required = True, help="number of intermediate data group")
    parser.add_argument("--aug", action='store_true', default=False, help='augmentation or not')
    parser.add_argument("--checkpoint", type=int, default=0, help="iteration of checkpoint")

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
