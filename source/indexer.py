import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from source.dataset import Pseudolabel_Dataset, get_dataloader, AlignCollateHDGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IndexingIntermediate(object):
    def __init__(
        self, opt, select_data
    ):
        """
        Indexing Intermediate Domain data for Gradual Domain Adaptation using 2 main approach:
        - Domain Discriminator (DD)
        - Harmonic Domain Gap Estimator (HDGE)
        Each approach gives one sample a distance. Then, we sort them in ascending order.

        Parameters
        ----------
        opt: argparse.ArgumentParser().parse_args()
            argument
        select_data: list()
            the array of selected data
        """

        self.opt = opt
        self.num_groups = opt.num_groups    # the number of intermediate groups
        self.remain_data = select_data   # the number of remain data after selection steps
        self.k_number = len(select_data) // self.num_groups    # the number of data per group
        self.score = [] # the order of data after processing

        # make dir
        if (not os.path.exists(f'indexing/{self.opt.approach}/')):
            os.makedirs(f'indexing/{self.opt.approach}/')

    def save_intermediate(self):
        
        result_index = [u[0] for u in self.score]
        result_score = [u[1] for u in self.score]

        np.save(f'indexing/{self.opt.approach}/intermediate_score.npy', result_score)
        np.save(f'indexing/{self.opt.approach}/intermediate_index.npy', result_index)
        
        for iter in range(self.num_groups // 2):
            # select k_number highest score
            add_source = [u for u in result_index[:self.k_number]]
            # select k_number lowest score
            add_target = [u for u in result_index[-self.k_number:]]
            # adjust result
            result_index = np.setdiff1d(result_index, add_source + add_target)

            # save work
            source = np.array(add_source, dtype=np.int32)
            target = np.array(add_target, dtype=np.int32)
            
            np.save(f'indexing/{self.opt.approach}/intermediate_{iter + 1}.npy', source)
            np.save(f'indexing/{self.opt.approach}/intermediate_{self.num_groups - iter}.npy', target)

        if (self.num_groups % 2 != 0):
            result_index = np.array(result_index, dtype=np.int32)
            np.save(f'indexing/{self.opt.approach}/intermediate_{self.num_groups // 2 + 1}.npy', result_index)

        print("\nAll information saved at " + f'indexing/{self.opt.approach}/')
        
        return [], []
    
    def select_DD(self, intermediate, model):
        """
        Select intermediate data group for each adaptation round and save them

        Parameters
        ----------
        intermediate: torch.utils.data.Dataset
            intermediate data
        model: Model
            discriminator module
        
        Return
        ----------
        """

        print("-" * 80)
        print("Select intermediate domain")

        unlabel_data_remain = Subset(intermediate, self.remain_data)

        # assign pseudo labels by the order of sample in dataset
        unlabel_data_remain = Pseudolabel_Dataset(unlabel_data_remain, self.remain_data)
        intermediate_loader = get_dataloader(self.opt, unlabel_data_remain, self.opt.batch_size_val, shuffle=False)
        
        del unlabel_data_remain

        result = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(intermediate_loader):
                image_tensors, index_unlabel = batch
                image = image_tensors.to(device)
                
                preds = model(image)
                preds_prob = torch.sigmoid(preds).detach().cpu().squeeze().numpy().tolist()
                result.extend(list(zip(index_unlabel, preds_prob)))

            # sort result by probability
            result = sorted(result, key=lambda x: x[1])

            print(result[:20])
            print(result[-20:])

            self.score = result

        self.save_intermediate()

    def select_HDGE(self, intermediate, dis_source, dis_target):
        """
        Select intermediate data group for each adaptation round and save them

        Parameters
        ----------
        intermediate: torch.utils.data.Dataset
            intermediate data
        dis_source: Model
            discriminator of source module
        dis_target: Model
            discriminator of target module
        
        Return
        ----------
        """

        print("-" * 80)
        print("Select intermediate domain")

        unlabel_data_remain = Subset(intermediate, self.remain_data)
       
        myAlignCollate = AlignCollateHDGE(self.opt, infer=True)
        intermediate_loader = torch.utils.data.DataLoader(
            unlabel_data_remain,
            batch_size=self.opt.batch_size_val,
            shuffle=False,
            num_workers=self.opt.num_workers,
            collate_fn=myAlignCollate,
            pin_memory=False,
            drop_last=False,
        )

        del unlabel_data_remain

        dis_source = dis_source.to(device)
        dis_target = dis_target.to(device)

        source_loss = []
        target_loss = []    

        dis_source.eval()
        dis_target.eval()
        with torch.no_grad():
            for batch in tqdm(intermediate_loader):
                image_tensors = batch
                image = image_tensors.to(device)
                
                source_dis = dis_source(image)
                target_dis = dis_target(image)

                real_label = torch.ones(source_dis.size()).to(device)
                
                # calculate MSE for each sample
                source_batch_loss = torch.mean((source_dis - real_label)**2, dim=(1,2,3)).cpu().squeeze().numpy().tolist()
                target_batch_loss = torch.mean((target_dis - real_label)**2, dim=(1,2,3)).cpu().squeeze().numpy().tolist()

                source_loss.extend(source_batch_loss)
                target_loss.extend(target_batch_loss)

        np.save(f'indexing/{self.opt.approach}/source_loss.npy', source_loss)
        np.save(f'indexing/{self.opt.approach}/target_loss.npy', target_loss)

    def select_confidence(self, intermediate, model, charset, opt):
        """
        Select intermediate data group for each adaptation round and save them

        Parameters
        ----------
        intermediate: torch.utils.data.Dataset
            intermediate data
        model: Model
            discriminator module
        
        Return
        ----------
        """

        print("-" * 80)
        print("Select intermediate domain")

        unlabel_data_remain = Subset(intermediate, self.remain_data)

        # assign pseudo labels by the order of sample in dataset
        unlabel_data_remain = Pseudolabel_Dataset(unlabel_data_remain, self.remain_data)
        intermediate_loader = get_dataloader(self.opt, unlabel_data_remain, self.opt.batch_size_val, shuffle=False)
        
        del unlabel_data_remain

        result = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(intermediate_loader):
                image_tensors, index_unlabel = batch
                images = image_tensors.to(device)
                
                res = model(images)
                # Select max probabilty (greedy decoding) then decode index to character
                pt_texts, pt_scores, __ = postprocess(res, charset, opt.config.model_eval)
                preds_prob = list()
                for pred, pred_max_prob in zip(
                    pt_texts, pt_scores
                ):
                    pred_max_prob = pred_max_prob[:len(pred)]
                    if len(pred_max_prob.cumprod(dim=0)) > 0 :
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()
                    else:
                        confidence_score = 0

                    preds_prob.append(confidence_score)

                # print(index_unlabel)
                # print(preds_prob)
                result.extend(list(zip(index_unlabel, preds_prob)))

            # sort result by confidence score
            result = sorted(result, key=lambda x: x[1], reverse=True)

            print(result[:20])
            print(result[-20:])

            self.score = result

        self.save_intermediate()


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