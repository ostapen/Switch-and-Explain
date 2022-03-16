"""Wrapper for a conditional generation dataset present in 2 tab-separated columns:
source[TAB]target
"""
import logging
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from multiprocessing import Pool
import torch
from torch.utils.data import DataLoader
import string
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import random
import numpy as np
from ast import literal_eval
import ast
np.set_printoptions(threshold = 10000)
# ['eng', 'eng&spa', 'fra', 'ita', 'spa']
num2lang = {1: 'eng', 2: 'eng&spa', 5: 'spa'}
MODEL_NAME = 'xlm-roberta-base'

from .data_utils import pad_nt_matrix_roberta, pad_nt_matrix_xlnet, pad_nt_matrix_roberta_3d

tqdm.pandas()

class ClassificationData(pl.LightningDataModule):
    def __init__(self, basedir: str, tokenizer_name: str, batch_size: int, num_workers: int = 16, codeswitch: bool = False, balanced: bool = False, \
                 use_speaker_descriptions: bool=False, use_speaker_tokens: bool=False,use_eot_tokens: bool=True,load_full_control=True,\
                 get_lang_feats: bool=False, context_size: int=1, load_only_balanced: bool=False, load_control_data = False, load_description_data=False,\
                 load_list_desc = False, load_sent_desc = False, full_mtl_setup = False, load_partner_desc = False, load_triplet = False, load_finetune_bangor = False, \
                 do_social_predictions = False, age=False, language=False, order=False, gender=False, mixing=False, country=False, \
                 leave_one_out = False, fake_spk=False):
        super().__init__()
        self.basedir = basedir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        self.use_speaker_descriptions = use_speaker_descriptions
        self.load_control_data = load_control_data
        self.load_triplet = load_triplet
        self.load_finetune_bangor = load_finetune_bangor
        self.fake_spk = fake_spk
        if not load_control_data:
            load_full_control = False
        self.load_full_control = load_full_control
        self.load_list_desc = load_list_desc
        self.load_sent_desc = load_sent_desc
        self.load_partner_desc=load_partner_desc
        self.full_mtl_setup = full_mtl_setup
        self.leave_one_out = leave_one_out

        self.categorical_only = True
        if use_speaker_tokens:
            print("using speaker tokens")
            use_eot_tokens=False
            new_toks = ['[SPK{}]'.format(idx+1) for idx in range(5)]
        elif use_eot_tokens:
            new_toks = ['[EOU]', '[EOT]']
        if use_eot_tokens or use_speaker_tokens:
            self.tokenizer.add_tokens(new_toks)
        #     print(self.tokenizer.get_added_vocab())
        self.codeswitch = codeswitch
        self.balanced = balanced
        self.get_lang_ids = get_lang_feats
        self.context_size = context_size
        self.use_eot_tokens = use_eot_tokens
        self.use_speaker_tokens = use_speaker_tokens
        self.load_only_balanced = load_only_balanced
        self.load_description_data = load_description_data

        self.social_predictions = do_social_predictions
        self.gender = gender
        self.age=age
        self.language=language
        self.mixing=mixing
        self.order=order
        self.country=country


        if codeswitch:


            self.collator = BangorCollator(tokenizer_name, self.load_triplet, self.social_predictions)
        else:
            self.collator = MyCollator(tokenizer_name)

    def train_dataloader(self):
        desc = ''
        if self.load_description_data:
            desc = '_desc'
        if self.codeswitch and not self.load_control_data:
            triplet_df_path = None
            spl_ = None
            if self.balanced:
                print("loading balanced train data")
                if self.categorical_only:
                    print("loading categorical data, numerical features binned")
                    finetune = ""
                    if self.load_finetune_bangor:
                        print("loading finetuning data... assume triplet finetuning firstt")
                        #finetune = "_tune"
                        spl_ = 10000
                    elif self.load_triplet:
                        print('loading data for triplet loss')
                        #finetune = '_rep_learn'
                        spl_ = 10000
                        triplet_df_path = f"{self.basedir}/train_binned_data_balanced_triplet.csv"

                    data_path = f"{self.basedir}/train_binned_data_balanced{desc}{finetune}_with_parse.csv"
                else:
                    data_path = f"{self.basedir}/train_data_balanced_normalized_with_parse.csv"
            else:
                data_path = f"{self.basedir}/train_data_normalized.csv"
            dataset = BangorDataset(tokenizer=self.tokenizer,load_desc=self.load_description_data, load_list = self.load_list_desc, \
                                    load_sent = self.load_sent_desc, load_partner=self.load_partner_desc,load_triplet=self.load_triplet,load_finetune=self.load_finetune_bangor, \
                                    data_path=data_path, use_speaker_tokens=self.use_speaker_tokens,triplet_df_path=triplet_df_path,data_split=spl_,\
                                    use_speaker_descriptions=self.use_speaker_descriptions, get_lang_ids=self.get_lang_ids, context_size=self.context_size, classify_speaker=self.social_predictions, \
                                    gender=self.gender, age=self.age, language=self.language, mixing=self.mixing, order=self.order, country=self.country, \
                                    leave_one_out = self.leave_one_out)
        elif self.load_control_data:

            print("loading control data (train)")
            desc = ''
            if self.load_description_data:
                desc = '_desc'
            path_ = ''
            if not self.load_full_control:
                path_ = '_train'
                data_path = f"{self.basedir}/control_experiment{desc}{path_}_with_parse.csv"
            else:
                data_path = f"{self.basedir}/train_binned_data_balanced{desc}_with_parse.csv"
            dataset = BangorDataset(tokenizer=self.tokenizer,
                                    data_path=data_path,use_speaker_tokens=self.use_speaker_tokens, load_desc=self.load_description_data, load_list = self.load_list_desc, \
                                    load_sent = self.load_sent_desc, load_partner=self.load_partner_desc, gender=self.gender, age=self.age, language=self.language, mixing=self.mixing, order=self.order, country=self.country,\
                                    use_speaker_descriptions=self.use_speaker_descriptions, get_lang_ids=self.get_lang_ids, context_size=self.context_size, classify_speaker=self.social_predictions, \
                                    leave_one_out = self.leave_one_out)

        else:
            dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                            data_path=f"{self.basedir}/train_with_parse.json")

        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, collate_fn=self.collator)

    def val_dataloader(self):
        desc = ''
        spl_ = None
        if self.load_description_data:
            desc = '_desc'
        if self.codeswitch and not self.load_control_data:
            triplet_df_path = None
            if self.balanced:
                print("loading val data")
                if self.categorical_only:
                    finetune = ""

                    if self.load_finetune_bangor:
                        print("loading finetuning data... assume triplet finetuning first")
                        #finetune = "_tune"
                        spl_ = 1500
                    elif self.load_triplet:
                        print('loading data for triplet loss')
                        triplet_df_path = f"{self.basedir}/valid_binned_data_balanced_triplet.csv"
                        spl_ = 1500

                        #finetune = '_rep_learn'
                    data_path = f"{self.basedir}/valid_binned_data_balanced{desc}{finetune}_with_parse.csv"
                    #data_path = f"{self.basedir}/valid_binned_data_with_parse.csv"
                else:

                    data_path = f"{self.basedir}/valid_data_balanced_normalized_with_parse.csv"
                    #data_path = f"{self.basedir}/valid_data_normalized_with_parse.csv"
            else:
                data_path = f"{self.basedir}/valid_binned_data{desc}_with_parse.csv"
            dataset = BangorDataset(tokenizer=self.tokenizer, classify_speaker=self.full_mtl_setup,gender=self.gender, age=self.age, language=self.language, mixing=self.mixing, order=self.order, country=self.country,\
                                    data_path=data_path,use_speaker_tokens=self.use_speaker_tokens, load_desc=self.load_description_data, load_list = self.load_list_desc, \
                                    load_sent = self.load_sent_desc, load_partner=self.load_partner_desc,load_finetune=self.load_finetune_bangor,\
                                    use_speaker_descriptions=self.use_speaker_descriptions,triplet_df_path=triplet_df_path,data_split=spl_,\
                                    get_lang_ids=self.get_lang_ids, context_size=self.context_size, load_triplet=self.load_triplet, \
                                    leave_one_out = self.leave_one_out)

        elif self.load_control_data:

            print("loading control data (val)")
            desc = ''
            if self.load_description_data:
                desc = '_desc'
            path_ = ''
            if not self.load_full_control:
                path_ = '_valid'
                data_path = f"{self.basedir}/control_experiment{desc}{path_}_with_parse.csv"
            else:
                data_path = f"{self.basedir}/valid_binned_data_balanced{desc}_with_parse.csv"
            dataset = BangorDataset(tokenizer=self.tokenizer,load_desc=self.load_description_data, load_list = self.load_list_desc, \
                                    load_sent = self.load_sent_desc, load_partner=self.load_partner_desc, classify_speaker=self.full_mtl_setup, \
                                    data_path=data_path,use_speaker_tokens=self.use_speaker_tokens, \
                                    use_speaker_descriptions=self.use_speaker_descriptions, get_lang_ids=self.get_lang_ids, context_size=self.context_size, \
                                    gender=self.gender, age=self.age, language=self.language, mixing=self.mixing, order=self.order, country=self.country, \
                                    leave_one_out = self.leave_one_out)
        else:
            dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                            data_path=f"{self.basedir}/dev_with_parse.json")

        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)

    def test_dataloader(self):
        desc = ''
        if self.load_description_data:
            desc = '_desc'
        if self.codeswitch and not self.load_control_data:
            # if self.balanced:
            #     data_path = f"{self.basedir}/test_data_balanced_normalized.csv"
            # else:
            #     data_path = f"{self.basedir}/test_data_normalized.csv"
            if self.categorical_only:
                data_path1 = f"{self.basedir}/test_binned_data_balanced{desc}_with_parse.csv"
                data_path2 = f"{self.basedir}/test_binned_data{desc}_with_parse.csv"
            else:
                data_path1 = f"{self.basedir}/test_data_balanced_normalized_with_parse.csv"
                data_path2 = f"{self.basedir}/test_data_normalized_with_parse.csv"
            dataset_balanced = BangorDataset(tokenizer=self.tokenizer,use_speaker_tokens=self.use_speaker_tokens,load_desc=self.load_description_data, load_list = self.load_list_desc, \
                                             load_sent = self.load_sent_desc, load_partner=self.load_partner_desc, classify_speaker=self.full_mtl_setup, \
                                             data_path=data_path1, \
                                             use_speaker_descriptions=self.use_speaker_descriptions,  get_lang_ids=self.get_lang_ids, context_size=self.context_size, \
                                             gender=self.gender, age=self.age, language=self.language, mixing=self.mixing, order=self.order, country=self.country, \
                                             leave_one_out = self.leave_one_out)
            dataset_unbalanced = BangorDataset(tokenizer=self.tokenizer,use_speaker_tokens=self.use_speaker_tokens,load_desc=self.load_description_data, load_list = self.load_list_desc, \
                                               load_sent = self.load_sent_desc, load_partner=self.load_partner_desc, classify_speaker=self.full_mtl_setup, \
                                               data_path=data_path2, \
                                               use_speaker_descriptions=self.use_speaker_descriptions,  get_lang_ids=self.get_lang_ids, context_size=self.context_size, \
                                               gender=self.gender, age=self.age, language=self.language, mixing=self.mixing, order=self.order, country=self.country, \
                                               leave_one_out = self.leave_one_out)

            bal_loader = DataLoader(dataset=dataset_balanced, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)
            if self.load_only_balanced:
                return bal_loader
            unbal_loader = DataLoader(dataset=dataset_unbalanced, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)
            combined = CombinedLoader({'balanced': bal_loader, 'unbalanced':unbal_loader}, 'max_size_cycle')
            return combined

        elif self.load_control_data:
            print("loading control data (test)")
            desc = ''
            if self.load_description_data:
                desc = '_desc'
            path_ = ''
            if not self.load_full_control:
                path_ = '_test'
            data_path = f"{self.basedir}/control_experiment{desc}{path_}_with_parse.csv"
            print(len(pd.read_csv(data_path)))
            dataset_balanced = BangorDataset(tokenizer=self.tokenizer,use_speaker_tokens=self.use_speaker_tokens,load_desc=self.load_description_data, load_list = self.load_list_desc, \
                                             load_sent = self.load_sent_desc, load_partner=self.load_partner_desc, \
                                             data_path=data_path, control_data=self.load_control_data,classify_speaker=self.full_mtl_setup, \
                                             use_speaker_descriptions=self.use_speaker_descriptions,  get_lang_ids=self.get_lang_ids, context_size=self.context_size, \
                                             gender=self.gender, age=self.age, language=self.language, mixing=self.mixing, order=self.order, country=self.country, \
                                             leave_one_out = self.leave_one_out)
            bal_loader = DataLoader(dataset=dataset_balanced, batch_size=self.batch_size,
                                    shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)
            return bal_loader
        else:
            data_path = f"{self.basedir}/test_parse.json"
            dataset = ClassificationDataset(tokenizer=self.tokenizer,
                                            data_path=data_path)
            return DataLoader(dataset=dataset, batch_size=self.batch_size,
                              shuffle=False, num_workers=self.num_workers, collate_fn=self.collator)





class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, data_path: str) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.read_dataset()


    def read_dataset(self):
        logging.info("Reading data from {}".format(self.data_path))
        pathname, ext = os.path.splitext(self.data_path)
        data = pd.read_json(self.data_path, orient="records", lines=True)

        self.sentences, self.answer_labels, self.nt_idx_matrix = [], [], []
        logging.info(f"Reading dataset file from {self.data_path}")
        # print(data, len(data))
        for i, row in tqdm(data.iterrows(), total=len(data), desc="Reading dataset samples"):
            self.answer_labels.append(int(row["label"]))
            self.sentences.append(row["sentence"])
            self.nt_idx_matrix.append(torch.tensor(row["nt_idx_matrix"]).long())


        encoded_input = self.tokenizer(self.sentences)
        self.input_ids = encoded_input["input_ids"]
        if "token_type_ids" in encoded_input:
            self.token_type_ids = encoded_input["token_type_ids"]
        else:
            self.token_type_ids = [[0] * len(s) for s in encoded_input["input_ids"]]


    def __len__(self) -> int:
            return len(self.sentences)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return (self.input_ids[i], self.token_type_ids[i], self.nt_idx_matrix[i], self.answer_labels[i])
class MyCollator(object):
    def __init__(self, model_name):
        self.model_name = model_name
        if "xlnet" in model_name:
            self.pad_fn = pad_nt_matrix_xlnet
        elif "roberta" in model_name or 'xlm-r' in model_name:
            self.pad_fn = pad_nt_matrix_roberta

        else:
            raise NotImplementedError

    def __call__(self, batch):
        max_token_len = 0
        max_phrase_len = 0
        num_elems = len(batch)
        for i in range(num_elems):
            tokens, _, _, _, _, _ = batch[i]
            max_token_len = max(max_token_len, len(tokens))

        tokens = torch.zeros(num_elems, max_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()
        labels = torch.zeros(num_elems).long()
        nt_idx_matrix = []

        for i in range(num_elems):
            toks, _, psy, syn, soc, label = batch[i]
            # idx_matrix = torch.tensor(idx_matrix).long()
            idx_matrix = self.pad_fn(nt_idx_matrix=idx_matrix,
                                     max_nt_len=max_phrase_len,
                                     max_length=max_token_len)
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1
            nt_idx_matrix.append(idx_matrix)
            labels[i] = label

        padded_ndx_tensor = torch.stack(nt_idx_matrix, dim=0)
        return [tokens, tokens_mask, padded_ndx_tensor, labels]

class BangorDataset(ClassificationDataset):
    def __init__(self, tokenizer, data_path: str, use_speaker_descriptions: bool=False, load_list: bool=False, load_sent: bool=False, load_partner: bool=False,\
                 get_lang_ids: bool=False, context_size: int =1, use_speaker_tokens: bool=False, load_desc: bool=False, control_data: bool=False, \
                 load_triplet: bool=False, triplet_df_path = None, load_finetune:bool=False, data_split:int=None, \
                  classify_speaker: bool = False, gender=False, language=False, age=False, mixing=False, order=False, country=False, \
                 load_only_spk = False, leave_one_out:bool =False) -> None:
        self.use_speaker_descriptions = use_speaker_descriptions
        self.get_lang_ids = get_lang_ids
        self.context_size = context_size
        self.use_speaker_tokens = use_speaker_tokens
        self.load_desc = load_desc
        self.control_data = control_data
        self.load_list = load_list
        self.load_sent = load_sent
        self.load_partner = load_partner
        self.load_triplet = load_triplet
        self.load_finetune = load_finetune
        self.data_split = data_split
        self.classify_speaker = classify_speaker
        self.load_only_spk = load_only_spk
        self.leave_one_out=leave_one_out
        self.gender = gender
        self.age=age
        self.language=language
        self.mixing=mixing
        self.order=order
        self.country=country
        self.prune=False
        if any([self.gender, self.age, self.language, self.mixing, self.order, self.country]):
            self.prune=True
            if self.leave_one_out:
                self.gender = not self.gender
                self.age = not self.age
                self.language = not self.language
                self.mixing = not self.mixing
                self.order = not self.order
                self.country = not self.country


        if self.load_triplet:
            assert triplet_df_path is not None
            self.triplet_df_path = triplet_df_path
        if self.load_triplet or self.load_finetune:
            assert self.data_split is not None


        super().__init__(tokenizer, data_path)



    def map_words_to_subwords(self, spk_order, spk_ids, toks):

        break_toks = list(np.where(np.isin(np.asarray(toks), np.array(self.tok_ids)))[0])
        break_toks.append(len(toks)-1)
        sp_ids_unique = set(spk_ids)
        try:
            spk_order_dict = {s_id: spk_order.index(s_id) for s_id in sp_ids_unique}
        except:
            import pdb; pdb.set_trace()
        if self.use_speaker_tokens:
            break_toks = break_toks[1:]
        spk_masks = {}

        for idx, id in enumerate(spk_ids):
            spk_new = np.zeros(len(toks))
            if idx == 0:
                start_idx = 1
                end_idx = break_toks[idx]
            else:
                start_idx = break_toks[idx-1]+1
                if idx+1 == len(break_toks):
                    end_idx = len(toks)
                else:
                    try:
                        end_idx = break_toks[idx+1]
                    except:
                        import pdb; pdb.set_trace()
            assert len(spk_new) == len(toks)


            #spk_new[range(start_idx, end_idx)] = spk_order_dict[id]
            spk_new[range(start_idx, end_idx)] = 1
            spk_new[-1] = 0
            spk_new[0] = 0
            spk_new=torch.Tensor(spk_new.tolist())
            if id in spk_masks:
                spk_masks[id] += spk_new
            else:
                spk_masks[id] = spk_new
        sp_masks_final = []
        for s_id, s in enumerate(spk_order):
            if s == spk_ids[-1]:
                return spk_masks.get(s)
            sp_masks_final.append(spk_masks.get(s, torch.Tensor([0]*len(toks))))
        # if sum(spk_new) == 0:
        #     import pdb; pdb.set_trace()
        assert len(sp_masks_final) ==len(spk_order)


        return sp_masks_final

    # def map_words_to_subwords_desc(self, spk_order, spk_ids, toks):
    #     #break_toks = list(np.where((np.asarray(toks) == self.tok_ids[0])| (np.asarray(toks) == self.tok_ids[1]))[0])
    #     break_toks = list(np.where(np.isin(np.asarray(toks), np.array(self.tok_ids)))[0])
    #     break_toks.append(len(toks)-1)
    #     sp_ids_unique = set(spk_ids)
    #     try:
    #         spk_order_dict = {s_id: spk_order.index(s_id) for s_id in sp_ids_unique}
    #     except:
    #         import pdb; pdb.set_trace()
    #     if self.use_speaker_tokens:
    #         break_toks = break_toks[1:]
    #     spk_masks = {}
    #
    #     for idx, id in enumerate(spk_ids):
    #         spk_new = np.zeros(len(toks))
    #         if idx == 0:
    #             start_idx = 1
    #             end_idx = break_toks[idx]
    #         else:
    #             start_idx = break_toks[idx-1]+1
    #             if idx+1 == len(break_toks):
    #                 end_idx = len(toks)
    #             else:
    #                 end_idx = break_toks[idx+1]
    #         assert len(spk_new) == len(toks)
    #
    #
    #         #spk_new[range(start_idx, end_idx)] = spk_order_dict[id]
    #         spk_new[range(start_idx, end_idx)] = 1
    #         spk_new[-1] = -1
    #         spk_new[0] = -1
    #         spk_new=torch.Tensor(spk_new.tolist())
    #         if id in spk_masks:
    #             spk_masks[id] += spk_new
    #         else:
    #             spk_masks[id] = spk_new
    #     sp_masks_final = []
    #     for s_id, s in enumerate(spk_order):
    #         if s == spk_ids[-1]:
    #             return spk_masks.get(s)
    #         sp_masks_final.append(spk_masks.get(s, torch.Tensor([0]*len(toks))))
    #     # if sum(spk_new) == 0:
    #     #     import pdb; pdb.set_trace()
    #     assert len(sp_masks_final) ==len(spk_order)
    #
    #
    #     return sp_masks_final
    def map_words_to_subwords_desc(self, tok, sent):
        def check_phrase_contents(phr):
            tp = 'funct'
            if phr == '</s>':
                tp = 'spl'
            if any([j in phr for j in ['person', 'male', 'female', 'man', 'woman', 'women', 'men', 'girl', 'boy', 'girls', 'boys']]):
                tp = 'gender'
            if any([j in phr for j in ['old', 'older', 'young', 'younger', 'middle-aged', 'girl', 'boy', 'girls', 'boys']]):
                if tp != 'funct':
                    tp+= '_age'
                else:
                    tp='age'
            elif any([j in phr for j in ['first', 'second', 'third', 'fourth', 'fifth']]):
                tp = 'order'
            elif any([j in phr for j in ['country', 'countries']]):
                tp = 'country'
            elif any([j in phr for j in ['prefer', 'prefers']]):
                tp = 'language'
            elif any([j in phr for j in ['mixing', 'mixes', 'mix', 'switch','switches', 'switching']]):
                tp = 'mixing'
            if tp == '':
                import pdb; pdb.set_trace()
            return tp




        words_back = [wd for wd in self.tokenizer.convert_ids_to_tokens(tok) if wd != "<pad>"]
        phr_to_tok_map = []
        split_by = ','
        if "**" in sent:
            split_by = '**'
        sent_split_full = sent.split("</s>")
        sent_split = []
        for i_s, s in enumerate(sent_split_full):
            ss = s.split(split_by)
            ss = [se for se in ss if se not in ['', ' '] and not se.isspace()]
            if self.load_list:
                ss = [ss[i]+','  if i != len(ss)-1 else ss[i] for i in range(len(ss))]
            sent_split.extend(ss)
            if i_s <len(sent_split_full)-1:
                sent_split.append("</s>")


        # if "NIC" in sent and "JES" in sent:
        #     import pdb; pdb.set_trace()


        sent_split.insert(0, '<s>')
        sent_split.append("</s>")
        curr_wd = ""
        idxs_for_phrase = []
        curr_phr = sent_split[0]
        curr_phr_spl = curr_phr.split()


        phr_types=['funct']
        # if self.load_partner:
        #     phr_types.append('funct')
        # else:
        #     if self.load_list:
        #         phr_types.append('order')
        #     else:
        #         phr_types.append('funct')


        ord_= ['funct', 'order', 'age', 'gender', 'country', 'language', 'mixing']


        for i in range(len(words_back)):
            wd = words_back[i]

            wd = "".join([a for a in wd if a.isalnum() or a in string.punctuation])
            idxs_for_phrase.append(i)
            curr_wd+= wd
            if curr_wd == curr_phr_spl[0] or (curr_wd[:-1] == curr_phr_spl[0] and curr_wd[-1] == ',') or (curr_wd[1:] == curr_phr_spl[0] and curr_wd[0] == ','):
                #word_to_tok_map.append(idxs_for_word)


                curr_phr_spl = curr_phr_spl[1:]
                curr_wd = ""

                if len(curr_phr_spl) == 0:
                    phr_to_tok_map.append(idxs_for_phrase)
                    idxs_for_phrase = []
                    if len(phr_to_tok_map)<len(sent_split):
                        curr_phr = sent_split[len(phr_to_tok_map)]

                        if self.load_partner or self.load_sent:
                            phr_types.append(check_phrase_contents(curr_phr))
                        else:

                            phr_types.append(ord_[len(phr_to_tok_map)%7])

                        curr_phr_spl = curr_phr.split()



                #idxs_for_word = []

        try:
            assert len(phr_to_tok_map) == (len(phr_types)) == len(sent_split)
        except:
            import pdb; pdb.set_trace()
        # if "NIC" in sent and "JES" in sent:
        #     import pdb; pdb.set_trace()
        return phr_to_tok_map, sent_split, phr_types



    def read_dataset(self):
        print("------")
        print(self.data_path)
        print("------")
        data = pd.read_csv(self.data_path)
        if not self.control_data:
            # if self.load_finetune:
            #
            #     data_to_process = data.loc[self.data_split:]
            # else:
            #     if self.load_triplet:
            #         data_to_process = data.loc[2:self.data_split].reset_index(drop=True)
            #     else:

            data_to_process = data[2:]


        else:
            data_to_process = data
        # ['nt_idx_matrix_utt', 'nt_idx_matrix_speaker', 'nt_idx_matrix_full_context']
        # ['lex_nt_idx_matrix', 'socio_nt_idx_matrix', 'psych_nt_idx_matrix']





        #cols =  list(np.array(np.matrix(data.iloc[0]['features'])).flatten())
        #feats = list(data_to_process['features'].apply(lambda r: np.asarray(np.matrix(r)).reshape(-1,len(cols))))

        self.sentences, self.speaker_ids, self.cs_labels,  self.speaker_context, self.nt_idx_matrix = [],[], [], [], []
        self.speaker_ids_context = []
        self.eng2spa = []
        self.spa2eng = []
        self.speaker_masks = []
        self.utterance_masks = []
        self.speaker_and_utterance_masks = []
        self.gender_tags = []
        self.age_tags = []
        self.mix_tags = []
        self.lang_tags = []
        self.input_ids = []
        self.token_type_ids =[]



        for i, row in data_to_process.iterrows():
            conv_id = row['convId']
            self.cs_labels.append(int(row["nextWordSwitched"]))
            # spk_ids = literal_eval(row['speakerOrder'])
            # spk_ids_context = literal_eval(row['speakerIds'])
            # spk_ids_context = ['_'.join([sp, conv_id]) for sp in spk_ids_context][-1*(self.context_size + 1):]
            spk_ids = literal_eval(row['speakerOrder'])

            #spk_ids = ['_'.join([sp, conv_id]) for sp in spk_ids]
            spk_ids_context = literal_eval(row['speakerIds'])[-1*(self.context_size+1):]
            #spk_ids_context = ['_'.join([sp, conv_id]) for sp in spk_ids_context][-1*(self.context_size + 1):]
            def prune_description(d_):
                split_by = ','
                if '**' in d_:
                    split_by = '**'
                tok = self.tokenizer(d_.replace("**",''))['input_ids']
                tok = [t for t in tok if t!=6]

                phr_to_idx_map, sent_spl, phr_types = self.map_words_to_subwords_desc(tok, d_)

                keywords= ['funct', 'spl']
                if self.order or self.load_list:
                    keywords.append('order')
                if self.gender:
                    keywords.extend(['gender', 'gender_age'])
                if self.age:
                    keywords.extend(['age', 'gender_age'])
                if self.language:
                    keywords.append('language')
                if self.country:
                    keywords.append('country')
                if self.mixing:
                    keywords.append('mixing')
                # list(np.where(np.isin(np.asarray(toks), np.array(self.tok_ids)))[0])
                idxs_keys = np.where(np.isin(np.array(phr_types), np.array(keywords)))[0]
                phr_non_funct = [k for k in phr_types[:-1] if k not in ['funct', 'spl']]
                funct_keys = ['funct','spl']
                if self.load_list and not self.order:
                    funct_keys.append('order')
                idxs_keys_nonfunct = np.where(np.isin(np.array(phr_non_funct), np.array(list(set(keywords)-set(funct_keys)))))[0]

                if len(idxs_keys) == 0:
                    import pdb; pdb.set_trace()
                desc_reduced = [sent_spl[j] for j in idxs_keys[:-1]]
                phr_types_reduced = [phr_types[j] for j in idxs_keys[:-1]]


                try:
                    ids_to_keep = [phr_to_idx_map[j] for j in idxs_keys]
                except:
                    import pdb; pdb.set_trace()

                desc_reduced = desc_reduced[1:]
                phr_types_reduced = phr_types_reduced[1:]
                desc_ids_to_toss = []
                if self.load_partner:
                    funct_terms = np.where(np.array(phr_types_reduced) == 'funct')[0]
                    for funct in funct_terms:
                        if phr_types_reduced[min(funct+1, len(phr_types_reduced)-1)] == 'spl':
                            desc_ids_to_toss.append(funct)
                            desc_ids_to_toss.append(funct+1)
                            ids_to_keep[funct+1] = []
                            ids_to_keep[funct+2] = []
                    # if phr_types[idxs_keys[1]] == 'funct' and phr_types[idxs_keys[2]] == 'spl':
                    #     # no interaction terms, get rid of interaction signal
                    #     desc_reduced = desc_reduced[1:]
                    #     ids_to_keep[-1] = []
                        #ids_to_keep.pop(1)

                    if desc_reduced[0] == "</s>":
                        #desc_reduced = desc_reduced[1:]
                        desc_ids_to_toss.append(0)
                        ids_to_keep[0] = []
                        #ids_to_keep.pop(0)
                    if desc_reduced[-1] == "</s>":
                        desc_reduced = desc_reduced[:-1]
                        ids_to_keep.pop(len(ids_to_keep)-1)
                        phr_types_reduced = phr_types_reduced[:-1]
                elif self.load_list:
                    if not self.order:
                        spl = np.where(np.array(desc_reduced) == "</s>")[0]
                        idxs = [-1]
                        if len(spl) >0:
                            idxs.extend(spl.tolist())
                        for i_e, idx in enumerate(idxs):
                            prev = []
                            if idx > -1:
                                prev = ids_to_keep[:idx+1]

                            sp_id = desc_reduced[idx+1].split()[0]
                            desc_reduced[idx+1] =sp_id
                            sp_id_tok = [t for t in self.tokenizer(sp_id,add_special_tokens=False)['input_ids'] if t!=6]

                            ids_to_keep = prev+[ids_to_keep[idx+1]]+[ids_to_keep[idx+2][:len(sp_id_tok)]] + ids_to_keep[idx+3:]



                if self.load_sent or self.load_partner:
                    spl = np.where(np.array(desc_reduced) == "</s>")[0]
                    idxs = [-1]
                    if len(spl) >0:
                        idxs.extend(spl.tolist())
                    for i_e, idx in enumerate(idxs):
                        if idx+2>=len(desc_reduced) or idx +1 >=len(desc_reduced):
                            continue
                        prev = []
                        if idx > -1:
                            prev = ids_to_keep[:idx+1]

                        try:


                            first_wd_ = desc_reduced[idx+2].split()[0]
                        except:
                            import pdb; pdb.set_trace()
                        if 'is a' in desc_reduced[idx+1]:
                            if first_wd_ in ['between', 'and']:
                                desc_reduced[idx+1] = desc_reduced[idx+1].replace("is a", '')

                                ids_to_keep = prev+[ids_to_keep[idx+1]]+[ids_to_keep[idx+2][:-2]] + ids_to_keep[idx+3:]

                            elif (first_wd_ == 'from') or (self.age and not self.gender):
                                desc_reduced[idx+1] = desc_reduced[idx+1].replace("is a", 'is')
                                ids_to_keep = prev+[ids_to_keep[idx+1]]+[ids_to_keep[idx+2][:-1]] + ids_to_keep[idx+3:]
                            elif first_wd_ ==  desc_reduced[idx+1].split()[0]:
                                desc_ids_to_toss.append(idx+1)
                                ids_to_keep = prev + [ids_to_keep[idx+1]] + [[]]+ids_to_keep[idx+3:]
                max_tok = phr_to_idx_map[-1][0]
                desc_ids_to_toss = list(set(desc_ids_to_toss))
                if len(desc_ids_to_toss) > 0:
                    desc_reduced = [desc_reduced[j] for j in range(len(desc_reduced)) if j not in desc_ids_to_toss]
                    ids_to_keep = [j for j in ids_to_keep if len(j) > 0]
                if desc_reduced[-1] == "</s>":
                    desc_reduced = desc_reduced[:-1]
                    ids_to_keep.pop(len(ids_to_keep)-1)
                    phr_types_reduced = phr_types_reduced[:-1]


                # if self.load_list:
                #     tmp = []
                #     spl = np.where(np.array(desc_reduced) == "</s>")[0]
                #     s_idx = 0
                #     import pdb; pdb.set_trace()
                #     for s in spl:
                #         tmp.append(', '.join(desc_reduced[s_idx:s]))
                #         tmp.append("</s>")
                #         s_idx = s+1
                #         ids_to_keep[s] =  ids_to_keep[s][:-1]
                #     tmp.append(', '.join(desc_reduced[s_idx:]))
                #     ids_to_keep[-2] =  ids_to_keep[-2][:-1]
                #     desc_new = ' '.join(tmp)
                # else:
                # if self.gender and len(desc_reduced) == 1 or (self.order and 'order' not in phr_types_reduced):
                #     import pdb; pdb.set_trace()
                desc_new = ' '.join(desc_reduced)
                # if "are all" in desc_new:
                #     import pdb; pdb.set_trace()
                ids_to_keep = [i for i_list in ids_to_keep for i in i_list]
                ids_to_discard = np.delete(np.arange(max_tok+1), ids_to_keep)
                return desc_new, idxs_keys_nonfunct, ids_to_discard







            def get_reduced_set(ld, ctx):
                reduced = ld[-1*(ctx+1):]

                tmp_r = []
                for r in reduced:
                    if r not in tmp_r:

                        tmp_r.append(r)
                tmp_desc = ' </s> '.join(tmp_r)
                ids_to_keep = []
                ids_to_discard = []
                if self.prune:
                    tmp_desc, ids_to_keep, ids_to_discard = prune_description(tmp_desc)
                return tmp_desc, ids_to_keep,ids_to_discard


            if self.load_desc:
                type_ = 'list'
                ctx_tmp = ''
                if self.load_sent:
                    type_ = 'sentence'
                elif self.load_partner:
                    type_ = 'partner'
                    ctx_tmp = self.context_size
                # if self.classify_speaker:
                #     spk_mask = [torch.tensor(t) for t in ast.literal_eval(row['speaker_map{}_desc_speaker_{}'.format(self.context_size, type_)])]
                #     utt_mask = [torch.tensor(t) for t in ast.literal_eval(row['speaker_map{}_desc_utterance_{}'.format(self.context_size, type_)])]
                #     um_tmp = {}
                #     for s_id, utt in zip(spk_ids_context[-1*(self.context_size+1):], utt_mask):
                #         if s_id not in um_tmp:
                #             um_tmp[s_id] = utt.clone()
                #         else:
                #             um_tmp[s_id] = um_tmp[s_id] + utt.clone()
                #     utt_mask = um_tmp[spk_ids_context[-1]].long()
                #     sp_focus = spk_ids.index(spk_ids_context[-1])
                #     spk_mask = spk_mask[sp_focus].long()
                #     #utt_mask = utt_mask[sp_focus]
                #     spk_utt_mask = torch.clip(spk_mask + utt_mask, min=0, max=1).long()
                #
                #     #um_tmp = {}
                #     #utt_mask_final = []
                #     #spk_utt_mask_final = []
                #     # for s_id, utt in zip(spk_ids_context, utt_mask):
                #     #     if s_id not in um_tmp:
                #     #         um_tmp[s_id] = torch.Tensor(utt)
                #     #     else:
                #     #         um_tmp[s_id] += torch.Tensor(utt)
                #     # for idx, s_id_o in enumerate(spk_ids):
                #     #     utt_mask_final.append(um_tmp[s_id_o])
                #     #     spk_utt_mask_final.append(um_tmp[s_id_o] + spk_mask[idx])
                #     #spk_and_utt_mask = torch.clip(spk_mask + utt_mask, min=0, max=1)
                #     self.speaker_masks.append(spk_mask)
                #     self.speaker_and_utterance_masks.append(spk_utt_mask)
                #     self.utterance_masks.append(utt_mask)

                    #self.utterance_masks.append(utt_mask)
                    #self.speaker_and_utterance_masks.append(spk_and_utt_mask)
                sentences = literal_eval(row['sentence_spk'])
                sentences = sentences[-1*(min(self.context_size+1, len(sentences))):]
                sent = ' '.join(sentences)
                desc_tmp = row['{}_description{}'.format(type_, ctx_tmp)]
                if type_ != 'partner':
                    desc_list = literal_eval(desc_tmp)

                    #desc_list = sorted(list(set(desc_list[-1*(min(self.context_size+1, len(desc_list))):])))
                    #desc = ' </s> '.join(desc_list)
                    desc, ids_to_keep, ids_to_discard = get_reduced_set(desc_list, self.context_size)
                else:
                    desc = desc_tmp
                    if self.prune:
                        desc, ids_to_keep, ids_to_discard = prune_description(desc)

                    # desc_to_keep = []
                    # spk_to_mask = []
                    # for sp_ in spk_ids:
                    #     if sp_ not in spk_ids_context:
                    #         spk_to_mask.append(sp_)
                    # desc_list = desc_tmp.split("</s>")
                    # if len(spk_to_mask) > 0:
                    #     for d in desc_list:
                    #         if "all" not in d:
                    #             sp_tmp = d.split()[0]
                    #             if sp_tmp not in spk_to_mask:
                    #                 desc_list.append(d)
                    #         else:
                    #             # for sp in spk_to_mask:
                    #             #     if sp in d:
                    #             #
                    #             #         if "{},".format(sp) in d:
                    #             #             to_replace = "{},".format(sp)
                    #             #         else:
                    #             #             to_replace = sp
                    #             #         d = d.replace(to_replace, ' ')
                    #             desc_to_keep.append(d)
                    # else:
                    #     desc_to_keep = desc_list
                    # desc = "</s>".join(desc_to_keep)
                    # assert all([j not in desc for j in spk_to_mask])

                if not self.load_only_spk:
                    sent_to_app = " </s></s> ".join([desc, sent])
                else:
                    sent_to_app = desc
                #sent_to_app = row['sentence{}_speaker_desc_{}'.format(self.context_size, type_)]
                sent_to_app = sent_to_app.replace("**", '')
                self.sentences.append(sent_to_app)
                encoded_input = self.tokenizer(sent_to_app, add_special_tokens=True)['input_ids']
                encoded = [j for j in encoded_input if j!=6]
                # if "IRI" and "JAM" in sent_to_app:
                #     import pdb; pdb.set_trace()
                # if "ASH" in sent_to_app and "JAC" in sent_to_app:
                #     import pdb; pdb.set_trace()
                self.input_ids.append(encoded)
                #'nt_idx_matrix{}_desc_speaker', 'nt_idx_matrix{}_desc_utterance'
                spk_nt = torch.tensor(ast.literal_eval(row['nt_idx_matrix{}_desc_speaker_{}'.format(self.context_size, type_)]))
                spk_ = ast.literal_eval(row['nt_idx_matrix{}_desc_speaker_{}'.format(self.context_size, type_)])
                utt_nt = torch.tensor(ast.literal_eval(row['nt_idx_matrix{}_desc_utterance_{}'.format(self.context_size, type_)]))
                utt_ = ast.literal_eval(row['nt_idx_matrix{}_desc_utterance_{}'.format(self.context_size, type_)])
                if self.prune:
                    try:
                        #import pdb; pdb.set_trace()
                        spk_nt = spk_nt[ids_to_keep,:]
                        spk_nt = np.delete(spk_nt, ids_to_discard, axis=1)
                        jj = np.where(spk_nt == 1)[1]
                        mm= np.array(encoded)[np.array(jj)]
                        check = self.tokenizer.convert_ids_to_tokens(mm)
                        utt_nt = np.delete(utt_nt, ids_to_discard, axis=1)
                        if len(spk_nt) == 0:
                            spk_nt = torch.zeros((1, utt_nt.shape[1]))
                        #further_prune = torch.where(spk_nt.sum(dim=0)>1)[0]
                        # if len(further_prune) > 0:
                        #     spk_nt = spk_nt[further_prune]
                        # else:
                        #     spk_nt = []
                    except:
                        import pdb; pdb.set_trace()
                # ps = []
                # for s_ in spk_:
                #     s_ = np.asarray(s_).astype(bool)
                #     tmp = np.asarray(encoded)[s_].tolist()
                #     p = self.tokenizer.convert_ids_to_tokens(tmp)
                #     ps.append(p)
                # for s_ in utt_:
                #     s_ = np.asarray(s_).astype(bool)
                #     tmp = np.asarray(encoded)[s_].tolist()
                #     p = self.tokenizer.convert_ids_to_tokens(tmp)
                #     ps.append(p)
                # import pdb; pdb.set_trace()
                if not self.load_only_spk:
                    self.nt_idx_matrix.append((spk_nt, utt_nt))
                else:
                    self.nt_idx_matrix.append(spk_nt[:, :len(encoded)])


            # elif self.use_speaker_tokens:
            #
            #
            #     self.sentences.append(row['sentence{}_speaker'.format(self.context_size)])
            #     self.nt_idx_matrix.append(torch.tensor(ast.literal_eval(row["nt_idx_matrix{}_speaker".format(self.context_size)])))
            else:
                if not self.control_data:
                    sentences = literal_eval(row['sentences_eot'])
                else:
                    sentences = literal_eval(row['sentences_spk'])
                sentences = sentences[-1*(min(self.context_size+1, len(sentences))):]
                sent = ' '.join(sentences)
                self.sentences.append(sent)
                encoded_input = self.tokenizer(sent, add_special_tokens=True)['input_ids']
                encoded = [j for j in encoded_input if j!=6]
                self.input_ids.append(encoded)
                if not self.control_data:
                    self.nt_idx_matrix.append(torch.tensor(ast.literal_eval(row["nt_idx_matrix{}".format(self.context_size)])))
                else:
                    tmp = torch.tensor(ast.literal_eval(row['nt_idx_matrix{}_desc_utterance_{}'.format(self.context_size, 'list')]))
                    tmp = tmp[:, -1*len(encoded):]
                    self.nt_idx_matrix.append(tmp)
            # else:
            #     self.sentences.append(row['sentence'])
            #     self.nt_idx_matrix.append(torch.tensor(ast.literal_eval(row["nt_idx_matrix_utt"])))
            #     if self.use_speaker_descriptions:
            #         self.speaker_context.append(row["speakerDescriptions"])
            #         self.speaker_nt_idx_matrix.appenfd(torch.tensor(ast.literal_eval(row["nt_idx_matrix_speaker"])))
                    #self.speaker_nt_idx_matrix.append(torch.from_numpy(np.asarray(np.matrix(literal_eval(row["nt_idx_matrix_speaker"])))))
            if "token_type_ids" in encoded_input:
                self.token_type_ids.append(encoded_input["token_type_ids"])
            else:
                self.token_type_ids.append([0] * len(encoded))
            if self.classify_speaker:
                self.gender_tags.append(int(row['isFemale']))
                self.age_tags.append(int(row['isOlder']))
                self.mix_tags.append(int(row['prefersCs']))
                self.lang_tags.append(int(row['balancedBilingual']))

            self.spa2eng.append(int(row['spa2eng']))
            self.eng2spa.append(int(row['eng2spa']))

            #soc_to_append = soc[:, :-1]





            self.speaker_ids.append(spk_ids)
            self.speaker_ids_context.append(spk_ids_context)


        # if self.ƒriptions:
        #     encoded_spk_context = self.tokenizer(self.speaker_context, add_special_tokens=False)
        #     self.speaker_context_ids = encoded_spk_context['input_ids']



        print("Frac switched utts: {:.4f}".format(sum(self.cs_labels)/len(self.cs_labels)))

        #self.input_ids = encoded_input["input_ids"]
        self.added_toks = self.tokenizer.get_added_vocab()
        self.tok_ids = [self.added_toks[a] for a in self.added_toks.keys()]

        # with Pool() as P:
        #     spk_ids_mask = P.map(self.map_words_to_subwords,[[s,i] for s, i in zip(self.speaker_ids_context, self.input_ids)])
        spk_ids_mask = [None]*len(self.input_ids)
        if not self.load_desc:
            spk_ids_mask = list(map(self.map_words_to_subwords,self.speaker_ids, self.speaker_ids_context, self.input_ids))
            if self.classify_speaker:
                self.utterance_masks = spk_ids_mask


        # if "token_type_ids" in encoded_input:
        #     self.token_type_ids = encoded_input["token_type_ids"]
        # else:
        #     self.token_type_ids = [[0] * len(s) for s in encoded_input["input_ids"]]
        self.sentences_new = []
        self.labels_new = []
        # if self.load_triplet:
        #     print('preparing triplets...')
        #     triplet_df = pd.read_csv(self.triplet_df_path)
        #     print(len(triplet_df), len(data_to_process))
        #     for idx, row in triplet_df.iterrows():
        #         anchor = row['anchor']
        #         pos = row['positive']
        #         neg = row['negative']
        #         self.sentences_new.append((self.input_ids[anchor], self.input_ids[pos], self.input_ids[neg]))
        #         self.labels_new.append((self.cs_labels[anchor], self.cs_labels[pos], self.cs_labels[neg]))


    def __getitem__(self, i):
        # We’ll pad at the batch level.
        #if self.load_triplet and 'sentence-transformers' not in self.tokenizer.name_or_path:
        #    return (self.sentences_new[i], self.labels_new[i])
        utt_mask = None
        spk_mask = None
        spk_utt_mask = None
        gend = None
        age = None
        mix = None
        lang = None
        if self.classify_speaker:
            utt_mask = self.utterance_masks[i]
            if self.load_desc:
                spk_mask = self.speaker_masks[i]
                spk_utt_mask = self.speaker_and_utterance_masks[i]
            gend = self.gender_tags[i]
            age = self.age_tags[i]
            mix = self.mix_tags[i]
            lang = self.lang_tags[i]

        return (self.input_ids[i], self.nt_idx_matrix[i], self.token_type_ids[i], self.cs_labels[i], \
         self.speaker_ids[i], self.speaker_ids_context[i],self.spa2eng[i], self.eng2spa[i], utt_mask, spk_mask, spk_utt_mask, gend, age, mix, lang)
        # else:
        #     return (self.input_ids[i], self.token_type_ids[i], self.cs_labels[i],\
        #             self.speaker_ids[i], self.psycholinguistic[i], self.socio[i], self.synsem[i], [], self.nt_idx_matrix[i], self.spa2eng[i], self.eng2spa[i])



class BangorCollator(MyCollator):
    def __init__(self, model_name, load_triplet, classify_speaker):
        self.pad_fn_3d = pad_nt_matrix_roberta_3d
        self.load_triplet = load_triplet
        self.model_name = model_name
        self.classify_speaker = classify_speaker
        super().__init__(model_name)
    def __call__(self, batch):
        max_token_len = 0
        max_utt_len = 0
        max_phrase_len = 0
        max_phrase_len2 = 0
        max_spk_phrase_len = 0
        max_spk_len = 0


        tmp = 0
        num_elems = len(batch)
        for i in range(num_elems):
            # self.input_ids[i], self.nt_idx_matrix[i], self.token_type_ids[i], self.cs_labels[i], \
            #          self.speaker_ids[i], self.speaker_ids_context[i], self.speaker_ids_mask[i], self.spa2eng[i], self.eng2spa[i]
            #if not self.load_triplet or 'sentence-transformers' in self.model_name:
            tokens, nt_idx_matrix,_, _, spk_ids, spk_ids_ctx, _, _, _, _, _, _, _, _, _= batch[i]
            max_token_len = max(max_token_len, len(tokens))
            if type(nt_idx_matrix) == tuple:
                if len(nt_idx_matrix[0]) > 0:
                    max_phrase_len = max(max_phrase_len, len(nt_idx_matrix[0]))
                max_phrase_len2 = max(max_phrase_len2, len(nt_idx_matrix[1]))
            else:
                max_phrase_len = max(max_phrase_len, len(nt_idx_matrix))
            # else:
            #     tokens_list, labels_list = batch[i]
            #     for tk in tokens_list:
            #         max_token_len = max(max_token_len, len(tk))



        #if not self.load_triplet or 'sentence-transformers' in self.model_name:
        tokens = torch.full((num_elems, max_token_len), 2).long()
        tokens_mask = torch.zeros(num_elems, max_token_len).long()
        labels = torch.zeros(num_elems).long()
        genders = torch.zeros(num_elems).long()
        ages = torch.zeros(num_elems).long()
        mixes = torch.zeros(num_elems).long()
        langs = torch.zeros(num_elems).long()
        # else:
        #     tokens = torch.full((num_elems,3, max_token_len), 2).long()
        #     tokens_mask = torch.zeros(num_elems,3, max_token_len).long()
        #     labels = torch.zeros((num_elems, 3)).long()


        speaker_orders = []
        speaker_context_ids = []

        utt_masks = torch.zeros(num_elems, max_token_len).long()
        speaker_masks = torch.zeros(num_elems, max_token_len).long()
        speaker_utt_masks = torch.zeros(num_elems, max_token_len).long()





        spa2eng = torch.zeros(num_elems).long()
        eng2spa = torch.zeros(num_elems).long()


        idx_matrices = []
        spk_ids_masks = []
        active_nt_tokens = []




        for i in range(num_elems):
            # self.input_ids[i], self.nt_idx_matrix[i], self.token_type_ids[i], self.cs_labels[i], \
            #          self.speaker_ids[i], self.speaker_ids_context[i], self.speaker_ids_mask[i], self.spa2eng[i], self.eng2spa[i]
            #if not self.load_triplet or (self.load_triplet and 'sentence-transformers' in self.model_name):
            toks, nt_idx_matrix,_, label, spk_ids, spk_ids_context, sp2en,en2sp, utt_mask, spk_mask, spk_utt_mask, gend, age, mix, lang= batch[i]
            length = len(toks)
            if type(nt_idx_matrix) == tuple:
                if len(nt_idx_matrix[0]) > 0:
                    to_app = ([i for i in range(nt_idx_matrix[0].shape[0])], [i for i in range(nt_idx_matrix[1].shape[0])])
                    nt_idx_matrix1 = self.pad_fn(nt_idx_matrix=torch.Tensor(nt_idx_matrix[0]), max_nt_len=max_phrase_len,
                                                 max_length=max_token_len, phrase_start=0)
                else:
                    nt_idx_matrix1 = []
                    to_app = ([], [i for i in range(nt_idx_matrix[1].shape[0])])
                active_nt_tokens.append(to_app)


                nt_idx_matrix2 = self.pad_fn(nt_idx_matrix=torch.Tensor(nt_idx_matrix[1]), max_nt_len=max_phrase_len2,
                                             max_length=max_token_len, phrase_start=0)
                idx_matrices.append((nt_idx_matrix1, nt_idx_matrix2))



            else:
                active_nt_tokens.append([i for i in range(nt_idx_matrix.shape[0])])
                #active_nt_tokens.append(active_nt_mask)

                nt_idx_matrix = self.pad_fn(nt_idx_matrix=torch.Tensor(nt_idx_matrix), max_nt_len=max_phrase_len,
                                            max_length=max_token_len, phrase_start=0)
                idx_matrices.append(nt_idx_matrix)
            speaker_orders.append(spk_ids)
            speaker_context_ids.append(spk_ids_context)

            tokens[i, :length] = torch.LongTensor(toks)
            tokens_mask[i, :length] = 1

            if utt_mask is not None:
                utt_masks[i, :length] = torch.LongTensor(utt_mask.long())
                # masks = []
                # for um in utt_mask:
                #     tmp = torch.zeros(1, max_token_len).long()
                #     tmp[0,:length] = um
                #     masks.append(tmp)
                # utt_masks.append(masks)
                if spk_mask is not None:
                    speaker_masks[i, :length] = torch.LongTensor(spk_mask.long())
                    speaker_utt_masks[i, :length]= torch.LongTensor(spk_utt_mask.long())
                #     for list_, to_add in zip([spk_mask, spk_utt_mask], [speaker_masks, speaker_utt_masks]):
                #         tmp_list = []
                #         for l in list_:
                #             tmp = torch.zeros(1, max_token_len).long()
                #             tmp[0,:length] = l
                #             tmp_list.append(tmp)
                #         to_add.append(tmp_list)


            # spk_tokens_mask = np.zeros(max_token_len)
            # if spk_ids_mask is not None:
            #     spk_tokens_mask[:length] = spk_ids_mask
            #     spk_ids_masks.append(spk_ids_mask.tolist())
            # else:
            #     spk_ids_masks.append([None])

            labels[i] = label
            spa2eng[i] = sp2en
            eng2spa[i] = en2sp
            if gend is not None:
                genders[i] = gend
                ages[i] = age
                mixes[i] = mix
                langs[i] = lang
            # else:
            #     toks, label = batch[i]
            #     for t_idx, t in enumerate(toks):
            #         length = len(toks[t_idx])
            #         labels[i][t_idx] = label[t_idx]
            #         tokens[i][t_idx][:length] = torch.LongTensor(toks[t_idx])
            #         tokens_mask[i][t_idx][:length] = 1
        #if self.load_triplet and not 'sentence-transformers' in self.model_name:
        #    return [tokens, tokens_mask, labels]
            #idx_matrix = torch.tensor(idx_matrix)
            #n_nt, n_tokens = nt_idx_matrix.size()
            #active_nt_mask = torch.zeros(max_phrase_len)
            #active_nt_mask[:len(nt_idx_matrix)] = 1










            #idx_matrices[i: phrase_len] = torch.LongTensor(nt_idx_matrix)
            #idx_matrices[i: max_token_len] = torch.Tensor(nt_idx_matrix)

        if type(nt_idx_matrix) == tuple:
            if len(nt_idx_matrix[0]) > 0:

                padded_ndx_tensor = (torch.stack([id[0] for id in idx_matrices], dim=0), torch.stack([id[1] for id in idx_matrices], dim=0))
            else:
                padded_ndx_tensor = ([id[0] for id in idx_matrices], torch.stack([id[1] for id in idx_matrices], dim=0))
            active_nt_tokens = ([[id[0] for id in active_nt_tokens]], [[id[1] for id in active_nt_tokens]])
        else:
            padded_ndx_tensor = torch.stack(idx_matrices, dim=0)
        if utt_mask is None:
            utt_masks = None
            speaker_masks = None
            speaker_utt_masks = None
        elif spk_mask is None:
            speaker_masks = None

            speaker_utt_masks = None


        return [tokens, tokens_mask,  (padded_ndx_tensor,active_nt_tokens), speaker_orders, speaker_context_ids, spa2eng, eng2spa, utt_masks, speaker_masks, speaker_utt_masks, \
                genders, ages, mixes, langs, labels]



# if __name__ == "__main__":
#     import sys
#     dm = ClassificationData(
#         basedir=sys.argv[1], model_name=sys.argv[2], batch_size=32)
#     for (tokens, tokens_mask, nt_idx_matrix, labels) in dm.train_dataloader():
#         print(torch.tensor(tokens_mask[0].tokens).shape)
