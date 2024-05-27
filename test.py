import os
import sys
import time
import argparse
import re
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from nltk.metrics.distance import edit_distance

from utils import Averager, onehot, Config, CharsetMapper

from losses import MultiLosses

from source.dataset import hierarchical_dataset, AlignCollate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    print(module_name, class_name)
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    return model

def postprocess(output, charset, model_eval):
    def _get_output(last_output, model_eval):
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

    output = _get_output(output, model_eval)
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



def benchmark_all_eval(model, criterion, converter, opt):
    """evaluation with 6 benchmark evaluation datasets"""
    eval_data_list = [
        "IIIT5k_3000",
        "SVT",
        "IC13_1015",
        "IC15_2077",
        "SVTP",
        "CUTE80",
    ]
    if (opt.addition == True):
        eval_data_list = [
            "COCOv1.4",
            "Uber",
            "ArT",
            "ReCTS",
        ]
    if (opt.exception == True):
        eval_data_list = [
            "IIIT5k_3000",
            "SVT",
            "IC13_857",
            "IC15_1811",
            "SVTP",
            "CUTE80",
        ]
    if (opt.all == True):
        eval_data_list = [
            "IIIT5k_3000",
            "SVT",
            "IC13_857",
            "IC13_1015",
            "IC15_1811",
            "IC15_2077",
            "SVTP",
            "CUTE80",
        ]

    accuracy_list = []
    total_forward_time = 0
    total_eval_data_number = 0
    total_correct_number = 0
    dashed_line = "-" * 80
    print(dashed_line)
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_eval = AlignCollate(opt)
        eval_data, eval_data_log = hierarchical_dataset(
            root=eval_data_path, opt=opt
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_eval,
            pin_memory=True,
        )

        _, accuracy_by_best_model, _, _, _, infer_time, length_of_data = validation(
            model, criterion, eval_loader, converter, opt, tqdm_position=0
        )
        accuracy_list.append(f"{accuracy_by_best_model:0.2f}")
        total_forward_time += infer_time
        total_eval_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        print(f"Acc {accuracy_by_best_model:0.2f}")
        print(dashed_line)

    averaged_forward_time = total_forward_time / total_eval_data_number * 1000
    total_accuracy = total_correct_number / total_eval_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    eval_log = "accuracy: "
    for name, accuracy in zip(eval_data_list, accuracy_list):
        eval_log += f"{name}: {accuracy}\t"
    eval_log += f"total_accuracy: {total_accuracy:0.2f}\t"
    eval_log += f"averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.2f}"
    print(eval_log)

    # for convenience
    print("\t".join(accuracy_list))
    print(f"Total_accuracy:{total_accuracy:0.2f}")

    return total_accuracy, eval_data_list, accuracy_list

    
def validation(model, criterion, eval_loader, charset, opt, tqdm_position=1):
    """validation or evaluation"""
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    model.eval()

    for i, (image_tensors, labels) in tqdm(
        enumerate(eval_loader),
        total=len(eval_loader),
        position=tqdm_position,
        leave=False,
    ):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction

        labels_index, labels_length= [], []
        for label in labels:
            label = label.lower()
            label = re.sub(charset.out_of_alphanumeric, "", label)

            label_length = torch.tensor(len(label)+ 1)
            label_index_ = torch.tensor(charset.get_labels(label))
            label_index = onehot(label_index_, charset.num_classes)

            labels_index.append(label_index)
            labels_length.append(label_length)

        labels_index = torch.stack(labels_index).to(device)
        #labels_length = torch.stack(labels_length).to(device)
        start_time = time.time()
        preds = model(image)
        forward_time = time.time() - start_time
        
        cost = criterion(
            preds, labels_index.contiguous(), labels_length
        )
        # select max probabilty (greedy decoding) then decode index to character
        preds_str, preds_score , _  = postprocess(preds, charset, opt.config.model_eval)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score

        confidence_score_list = []
        for gt, prd, prd_max_prob in zip(labels, preds_str, preds_score):
            prd_max_prob = prd_max_prob[:len(prd)]
            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting. = same with ASTER
            gt = gt.lower()
            prd = prd.lower()
            alphanumeric_case_insensitve = "0123456789abcdefghijklmnopqrstuvwxyz"
            out_of_alphanumeric_case_insensitve = f"[^{alphanumeric_case_insensitve}]"
            gt = re.sub(out_of_alphanumeric_case_insensitve, "", gt)
            prd = re.sub(out_of_alphanumeric_case_insensitve, "", prd)

            if opt.NED:
                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(prd) == 0:
                    norm_ED += 0
                elif len(gt) > len(prd):
                    norm_ED += 1 - edit_distance(prd, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(prd, gt) / len(prd)

            else:
                if prd == gt:
                    n_correct += 1

            # calculate confidence score (= multiply of prd_max_prob)
            try:
                confidence_score = prd_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([EOS])
            confidence_score_list.append(confidence_score)

    if opt.NED:
        # ICDAR2019 Normalized Edit Distance. In web page, they report % of norm_ED (= norm_ED * 100).
        score = norm_ED / float(length_of_data) * 100
    else:
        score = n_correct / float(length_of_data) * 100  # accuracy

    return (
        valid_loss_avg.val(),
        score,
        preds_str,
        confidence_score_list,
        labels,
        infer_time,
        length_of_data,
    )


def test(opt):
    """model configuration"""

    converter = CharsetMapper(filename=opt.config.dataset_charset_path,
                            max_length=opt.config.dataset_max_length + 1)

    opt.num_class = converter.num_classes

    model = get_model(opt.config).to(device)
    model = torch.nn.DataParallel(model).to(device)
    model = load(model, opt.config.model_checkpoint, device=device)

    #model = torch.nn.DataParallel(model).to(device)
    
    """ setup loss """
    criterion = MultiLosses(one_hot= True, device= device)

    """ evaluation """
    model.eval()
    with torch.no_grad():
        # evaluate 6 benchmark evaluation datasets
        benchmark_all_eval(model, criterion, converter, opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_data", 
        default="data/test/benchmark/",
        help="path to evaluation dataset",
    )
    parser.add_argument("--addition", action='store_true', default=False, help='test on addition data')
    parser.add_argument("--exception", action='store_true', default=False, help='test on exception data')
    parser.add_argument("--all", action='store_true', default=False, help='test on all data')
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch_size", type=int, default=512, help="input batch size")
    parser.add_argument(
        "--saved_model", required=True, help="path to saved_model to evaluation"
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
        "--character",
        type=str,
        default="abcdefghijklmnopqrstuvwxyz0123456789",
        help="character label",
    )
    parser.add_argument(
        "--NED", action="store_true", help="For Normalized edit_distance"
    )


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

    cudnn.benchmark = True
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

    test(opt)
