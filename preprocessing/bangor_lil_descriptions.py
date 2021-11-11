import argparse
import csv
import json
from typing import Dict
import os
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, XLNetTokenizer, DistilBertTokenizer, XLMRobertaTokenizer, AutoModel, AutoConfig
import string
from transformers import AutoTokenizer, AutoModel

import math
from transformers.modeling_utils import SequenceSummary

from data_utils import pad_nt_matrix_roberta
from tqdm import tqdm
import ast
import torch
N_GRAM = 5
SPEECH_N_GRAM = 3
OVERLAP_PHRASES = True

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # source https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class ParsedDataset(object):
    def __init__(self, tokenizer_name):
        self.parse_trees: Dict[str, str] = {}
        self.tokenizer_name = tokenizer_name
        # if 'xlm-roberta' in tokenizer_name:
        #     self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
        #
        # elif 'roberta' in tokenizer_name:
        #     self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        # elif 'xlnet' in tokenizer_name:
        #     self.tokenizer = XLNetTokenizer.from_pretrained(tokenizer_name)
        # elif 'distilbert' in tokenizer_name:
        #     self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.model = AutoModel.from_pretrained('xlm-roberta-base')

        self.config = AutoConfig.from_pretrained('xlm-roberta-base')
        self.pooler = SequenceSummary(self.config)
        self.model.word_embeddings = self.model.resize_token_embeddings(len(self.tokenizer))
    def reset_tokenizer(self):
        if 'xlm-roberta' in self.tokenizer_name:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.tokenizer_name)

        elif 'roberta' in self.tokenizer_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_name)
        elif 'xlnet' in self.tokenizer_name:
            self.tokenizer = XLNetTokenizer.from_pretrained(self.tokenizer_name)
        elif 'distilbert' in self.tokenizer_name:
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.tokenizer_name)


    def map_words_to_subwords(self, tok, sent):
        words_back = [wd for wd in self.tokenizer.convert_ids_to_tokens(tok) if wd != "<pad>"]
        word_to_tok_map = []
        sent_split = sent.split()
        sent_split = [s for s in sent_split if s not in ['', ' '] and not s.isspace()]
        # if "NIC" in sent and "JES" in sent:
        #     import pdb; pdb.set_trace()


        sent_split.insert(0, '<s>')
        sent_split.append("</s>")
        curr_wd = ""
        idxs_for_word = []

        for i in range(len(words_back)):
            wd = words_back[i]
            wd = "".join([a for a in wd if a.isalnum() or a in string.punctuation])
            idxs_for_word.append(i)
            curr_wd+= wd
            if curr_wd == sent_split[0]:
                word_to_tok_map.append(idxs_for_word)
                curr_wd = ""
                idxs_for_word = []
                sent_split = sent_split[1:]
        try:
            assert len(word_to_tok_map) == (len(sent.split())+2)
        except:
            import pdb; pdb.set_trace()
        # if "NIC" in sent and "JES" in sent:
        #     import pdb; pdb.set_trace()
        return word_to_tok_map
    def map_to_phrases_speaker(self, sent, word_to_tok_map, split_by_sep = ','):
        # if ngram == SPEECH_N_GRAM:
        #     import pdb; pdb.set_trace()
        idx_matrices = []
        spk_descs = sent.split("</s>")
        word_start = 1
        # if "NIC" in sent and "JES" in sent:
        #     import pdb; pdb.set_trace()
        # if "ASH" and "JAC" in sent and "all" not in sent and "**" in sent:
        #     import pdb; pdb.set_trace()
        word_starts = []
        phr_starts_all = []
        spk_toks_all = []
        phr_lens_all = []

        partner_feats = False
        spk_ids = []
        for desc in spk_descs:

            if not desc.isspace() and desc != "":

                word_starts.append(word_start)

                phrase_starts = []
                sent_split_arr = np.array(desc.split())
                split_by = ","
                is_word_idx = 2
                add_word = 2
                if "**" in desc:
                    split_by = "**"
                    is_word_idx = 1
                first_phr = desc.split(split_by)[0].split()

                if "all" in first_phr:
                    #import pdb; pdb.set_trace()
                    is_word_idx = len(first_phr)
                    add_word = len(first_phr)
                    partner_feats = True
                if "all" not in first_phr:
                    spk_ids.append(first_phr[0])
                if len(spk_descs) == 1 and "all" in desc:
                    names = desc.split("are all")[0].split(', ')
                    spk_ids.extend(names)




                sent_split_tmp = [s for s in desc.split(split_by) if s not in ['', ' '] and not s.isspace()]
                try:
                    if split_by != "**":
                        is_phr = sent_split_tmp[0].split()[:2]
                        sent_split = [' ' .join(sent_split_tmp[0].split()[2:])] + sent_split_tmp[1:]
                    else:
                        add_word = len(sent_split_tmp[0].split())
                        sent_split = sent_split_tmp[1:]
                except:
                    import pdb; pdb.set_trace()
                sent_split.insert(0, '<s>')
                sent_split.append("</s>")
                sent_spl_arr = np.array(sent_split)

                spk_toks = [jj for j in word_to_tok_map[word_start:word_start+is_word_idx] for jj in j]
                word_start += add_word
                spk_toks_all.append(spk_toks)
                #word_start+= len(desc.split())+1
                lens = [len(sent_split[r].split()) if r not in [0, len(sent_split)-1] else 0 for r in range(len(sent_split))]

            # s spk is first born <.s> spk is last place

            lens.pop(0)
            #lens.pop(0)
            lens.pop(-1)
            # if "**" in desc:
            #     import pdb; pdb.set_trace()
            phr_lens_all.append(lens)
            for idx, length in enumerate(lens):
                phrase_starts.append(word_start)
                word_start += length
            word_start += 1
            phr_starts_all.append(phrase_starts)



        #eos_toks_map = [word_to_tok_map[eos][0] for eos in eos_toks]
        tok_len = sum([len(j) for j in word_to_tok_map])
        spk_masks = []
        partner_ft = None
        #if '**' in desc:
        #    import pdb; pdb.set_trace()
        for phrase_starts, spk_toks, lens in zip(phr_starts_all, spk_toks_all, phr_lens_all):
            spk_mask = np.zeros(tok_len)
            if partner_feats and partner_ft is not None:
                spk_mask = np.zeros(tok_len) + spk_masks[0]
            for i, phrase_idx in enumerate(phrase_starts):
                if phrase_idx < len(word_to_tok_map):
                    phrase = np.zeros(tok_len)
                    words = [k for h in word_to_tok_map[phrase_idx:min(phrase_idx+lens[i], len(word_to_tok_map))] for k in h]
                    phrase[words] = 1
                    phrase[spk_toks] = 1
                    spk_mask[words] = 1
                    spk_mask[spk_toks] = 1

                    #phrase[eos_toks_map] = 0
                    phrase[0] = 0
                    phrase[-1] = 0

                    if sum(phrase) == 0:
                        import pdb; pdb.set_trace()
                    idx_matrices.append(list(phrase))
            spk_mask[0] = 0
            spk_mask[-1] = 0
            spk_masks.append(spk_mask.tolist())
            if partner_feats and partner_ft is None:
                partner_ft = spk_masks[0]
            if len(idx_matrices) == 0:
                import pdb; pdb.set_trace()


        if partner_feats and len(spk_masks) > 1:
            spk_masks = spk_masks[1:]
        try:
            if not (len(spk_descs) == 1 and "all" in spk_descs[0]):
                assert len(spk_masks) == len(spk_ids)
            else:
                spk_tmp = spk_masks[0]
                spk_masks = [spk_tmp] * len(spk_ids)
        except:
            import pdb; pdb.set_trace()
        spk2mask = {sp: mask for sp, mask in zip(spk_ids, spk_masks)}
        if len(spk2mask) == 0:
            import pdb; pdb.set_trace()
        # if "NIC" in sent and "JES" in sent:
        #     import pdb; pdb.set_trace()if "NIC
        # if "IRI" and "JAM" in sent and "all" not in sent and "**" in sent:
        #      import pdb; pdb.set_trace()
        return [idx_matrices, spk2mask]

    def map_to_phrases(self, sent, word_to_tok_map, ngram = N_GRAM, eot=False):
        # if ngram == SPEECH_N_GRAM:
        #     import pdb; pdb.set_trace()
        idx_matrices = []
        sent_split = sent.split()
        sent_split.insert(0, '<s>')
        sent_split.append("</s>")
        stop_toks = ['</s>','[EOU]','[EOT]', ':']
        stop_toks += ["[SPK{}]".format(i+1) for i in range(5)]
        sent_spl_arr = np.array(sent_split)

        #eos_toks = np.where((sent_spl_arr == '</s>') | (sent_spl_arr == '[EOU]')| (sent_spl_arr == '[EOT]'))[0]
        eos_toks = list(np.where(np.isin(sent_spl_arr, np.array(stop_toks)))[0])

        eos_toks_list = sorted(list(eos_toks))
        # if len(eos_toks_list) > 1:spk_m
        #     eos_toks_list.pop(0)

        tok_len = sum([len(j) for j in word_to_tok_map])



        sent_splits_all = []
        lens = []

        if eot:
            sent_split_eot = sent.split("[EOT]")
            for idx, phr in enumerate(sent_split_eot):
                utts = phr.split("[EOU]")

                for utt in utts:
                    sent_splits_all.append(' '.join(utt))
                    len_phr = len(utt.split())
                    # if idx in [0, len(sent_split)-1]:
                    #     len_phr -= 1
                    lens.append(len_phr)
        else:
            for i_cur, eot_idx in enumerate(eos_toks_list[:-1]):
                next_idx = eos_toks_list[i_cur+1]- int(i_cur == len(eos_toks_list)-3)
                lens.append(eos_toks_list[i_cur+1] - eot_idx-1-int(i_cur == len(eos_toks_list)-3))
                utt = np.array(sent_split)[eot_idx+1: next_idx]
                if len(utt) > 0:
                    sent_splits_all.append(' '.join(utt))

        spk_splits = np.where(sent_spl_arr == ':')[0]
        spk_maps = []
        for idx_sp, sp in enumerate(spk_splits):
            eos_toks.append(sp-1)
            if idx_sp + 1 <len(spk_splits):
                end = spk_splits[idx_sp+1]-1
            else:
                end = len(word_to_tok_map)
            spk_map = np.zeros(tok_len)
            words = [k for h in word_to_tok_map[sp-1:end] for k in h]
            spk_map[words] = 1
            spk_map[-1] = 0
            spk_map[0] = 0
            spk_maps.append(spk_map.tolist())


        num_phrases = 0
        phrase_starts = []
        if eot:
            ix_max = 0
        else:
            ix_max = 1
        word_idx = sorted(eos_toks_list)[0]+1
        for idx, length in enumerate(lens):
            # do sliding window of 5 grams
            if not OVERLAP_PHRASES:
                num_loc_phrases = math.ceil(length / N_GRAM)
                idx_shift = ngram
            else:
                idx_shift = 2
                num_loc_phrases = max(math.ceil((length-ngram)/idx_shift), 0) + 1

            num_phrases += num_loc_phrases
            tmp = [word_idx]
            for i in range(num_loc_phrases-1):
                word_idx += idx_shift
                tmp.append(word_idx)
            if eot:
                word_idx = eos_toks_list[idx] +1
            else:
                word_idx = eos_toks_list[idx+1] +1
            phrase_starts.append(tmp)
        if num_phrases == 0:
            import pdb; pdb.set_trace()
        eos_toks_list = sorted(eos_toks_list)
        eos_toks_map = [ee for eos in eos_toks for ee in word_to_tok_map[eos]]


        #import pdb; pdb.set_trace()
        for phrase_idx_list in phrase_starts:
            for phrase_idx in phrase_idx_list:
                if phrase_idx < len(word_to_tok_map) and eos_toks_list[ix_max] != phrase_idx:
                    phrase = np.zeros(tok_len)
                    try:

                        words = [k for h in word_to_tok_map[phrase_idx:min(phrase_idx+ngram, len(word_to_tok_map), eos_toks_list[ix_max])] for k in h]
                    except:
                        import pdb; pdb.set_trace()
                    phrase[words] = 1
                    phrase[eos_toks_map] = 0
                    phrase[0] = 0
                    if len(eos_toks_list) == 0:
                        import pdb; pdb.set_trace()
                    eos_idx = word_to_tok_map[eos_toks_list[ix_max]][0]
                    phrase[eos_idx:] = 0
                    # if word_idx+idx_shift < eos_toks_list[0]:
                    #     word_idx += idx_shift
                    # else:
                    #     word_idx = eos_toks_list[0] + 1
                    # #word_idx = min(word_idx + N_GRAM, eos_idx+1)
                    # if word_idx in eos_toks:
                    #     while word_idx in eos_toks:
                    #         word_idx += 1
                    # if word_idx > eos_toks_list[0]:
                    if sum(phrase) == 0:
                        import pdb; pdb.set_trace()

                    idx_matrices.append(list(phrase))
            eos_toks_list.pop(0)
        if len(idx_matrices) == 0:
            import pdb; pdb.set_trace()
        return [idx_matrices, spk_maps]
    def clean_desc(self, spk_ids, spk_ids_full, partner_desc, ctx_size):
        spk_ids_context = spk_ids_full[-1*(ctx_size+1):]
        desc_list = partner_desc.split("</s>")
        desc_to_keep = []
        spk_to_mask = []
        for sp_ in spk_ids:
            if sp_ not in spk_ids_context:
                spk_to_mask.append(sp_)
        if len(spk_to_mask) > 0:
            for d in desc_list:
                if "all" not in d:
                    sp_tmp = d.split()[0]

                    if sp_tmp not in spk_to_mask or any([j in d for j in ['first', 'second', 'third', 'fourth', 'fifth']]):
                        to_keep = d
                        try:
                            tmp_spl = to_keep.split("**")
                            if len(tmp_spl) > 1:
                                if "switches" in to_keep.split("**")[1]:
                                    to_keep = to_keep.replace("is a **", '** ')
                                    to_keep = to_keep.replace(" ** and ", ' ** ')

                            desc_to_keep.append(to_keep)
                        except:
                            import pdb; pdb.set_trace()
                else:
                    # for sp in spk_to_mask:
                    #     if sp in d:
                    #
                    #         if "{},".format(sp) in d:
                    #             to_replace = "{},".format(sp)
                    #         else:
                    #             to_replace = sp
                    #         d = d.replace(to_replace, ' ')
                    desc_to_keep.append(d)
        else:
            desc_to_keep = desc_list
        return ' </s> '.join(desc_to_keep)


    def read_and_store_from_csv(self, binned_files, spk_files, control_data=False, valid=False):
        input_file_name, output_file_name = binned_files
        csv_full = pd.read_csv(input_file_name)
        if not control_data:
            csv=csv_full[2:]
        else:
            csv = csv_full.copy()
        if binned_files is not None:
            bin_in, bin_out = binned_files
        if spk_files is not None:
            spk_in, spk_out = spk_files
        csv_bin = pd.read_csv(bin_in)
        desc_types = ['partner', 'list']
        #if not control_data:
        desc_types.append('sentence')
        #desc_types.append('partner')
        #desc_types.append('list_mask')
        #for ctx_size in range(1, 11):
        if not control_data:
            ctx_max =11
        else:
            ctx_max= 4
        data_to_add = {}
        sentences_all = list(csv['sentence_spk'].apply(lambda x: ast.literal_eval(x)))
        list_descs_all = list(csv['list_description'].apply(lambda x: ast.literal_eval(x)))
        sent_descs_all = list(csv['sentence_description'].apply(lambda x: ast.literal_eval(x)))
        partner_descs_all = list(csv['partner_description'])
        spk_orders = list(csv['speakerOrder'].apply(lambda x: ast.literal_eval(x)))
        spk_context = list(csv['speakerIds'].apply(lambda x: ast.literal_eval(x)))
        list_descs_masked = []
        sent_descs_masked = []
        partner_descs_masked = []
        ## save utterances as a list.. then in data load, take utts of [-(ctx+1):], join with spk desk
        ## can save nt's separately
        for ctx_size in tqdm([1,2,3,5,7,9,10], total=7):
            """
            self.use_speaker_tokens = use_speaker_tokens
            if use_speaker_tokens:
                new_toks = ['[SPK{}]'.format(idx+1) for idx in range(5)]
            else:
                new_toks = ['[EOU]', '[EOT]']
        self.tokenizer.add_tokens(new_toks)
        
            """
            sentences = [' '.join(sent[-1*(ctx_size+1):]) for sent in sentences_all]
            def get_reduced_set(ld, ctx):
                reduced = ld[-1*(ctx+1):]
                tmp_r = []
                for r in reduced:
                    if r not in tmp_r:
                        tmp_r.append(r)
                return ' </s> '.join(tmp_r)
            for desc_type in desc_types:

                if 'list' in desc_type:
                    #descs = [' </s> '.join(sorted(list(set(ld[-1*(ctx_size+1):])))) for ld in list_descs_all]
                    descs = list(map(get_reduced_set,list_descs_all, [ctx_size]*len(list_descs_all)))
                elif 'sent' in desc_type:
                    descs = [' </s> '.join(set(ld[-1*(ctx_size+1):])) for ld in sent_descs_all]
                    #descs = [' </s> '.join(sorted(list(set(ld[-1*(ctx_size+1):])))) for ld in sent_descs_all]
                    #descs = list(map(get_reduced_set,sent_descs_all, [ctx_size]*len(sent_descs_all)))
                else:
                    descs = list(map(self.clean_desc, spk_orders, spk_context, partner_descs_all, [ctx_size]*len(partner_descs_all)))


                    csv_full['partner_description{}'.format(ctx_size)] = ['',''] + descs
                    csv_bin['partner_description{}'.format(ctx_size)] = ['', ''] + descs
                #sentences = list(csv['sentence{}_speaker_desc_{}'.format(ctx_size, desc_type)])
                #sentences_clean = []
                #sentences_to_proc_full =[[],[],[],[]]
                # if desc_type == 'list_mask':
                #     for s in sentences:
                #         spk_desc, dialogue = s.split("</s></s>")
                #         spk_phrs_clean = []
                #         spl_tmp = spk_desc.split("</s>")
                #
                #         for idx_sp, spk_phr_full in enumerate(spl_tmp):
                #
                #
                #             spk_phr = spk_phr_full.split(",")
                #             if idx_sp == (len(spl_tmp)-1):
                #                 spk_phr_g = spk_phr_full.split(",")
                #                 spk_phr_g[2] = " <mask>"
                #                 spk_phr_l = spk_phr_full.split(",")
                #                 spk_phr_a = spk_phr_full.split(",")
                #                 spk_phr_a[1] = " <mask>"
                #                 spk_phr_m = spk_phr_full.split(",")
                #
                #
                #             if 'English and Spanish' in spk_phr[-2]:
                #                 tmp = "both"
                #             elif "English nor Spanish" in spk_phr[-2]:
                #                 tmp = "neither"
                #             elif "English" in spk_phr[-2]:
                #                 tmp = "English"
                #             else:
                #                 tmp = "Spanish"
                #             spk_phr[-2] = " between English and Spanish prefers {}".format(tmp)
                #             if idx_sp == (len(spl_tmp)-1):
                #                 spk_phr_l[-2] = " between English and Spanish prefers {}".format("<mask>")
                #                 spk_phr_g[-2] = " between English and Spanish prefers {}".format(tmp)
                #                 spk_phr_a[-2] = " between English and Spanish prefers {}".format(tmp)
                #                 spk_phr_m[-2] = " between English and Spanish prefers {}".format(tmp)
                #
                #
                #             if "separates" in spk_phr[-1] and "most" in spk_phr[-1]:
                #                 tmp= "never"
                #             elif "unknown" in spk_phr[-1] and "more" in spk_phr[-1]:
                #                 tmp = "rarely"
                #             elif "mixes" in spk_phr[-1]:
                #                 tmp = "most"
                #             else:
                #                 tmp = "sometimes"
                #             spk_phr[-1] = " mixes languages {}".format(tmp)
                #             if idx_sp == (len(spl_tmp)-1):
                #                 spk_phr_m[-1] = " mixes languages {}".format("<mask>")
                #                 spk_phr_g[-1] = " mixes languages {}".format(tmp)
                #                 spk_phr_a[-1] = " mixes languages {}".format(tmp)
                #                 spk_phr_l[-1] = " mixes languages {}".format(tmp)
                #                 spk_phr_g = ', '.join(spk_phr_g)
                #                 spk_phr_l = ', '.join(spk_phr_l)
                #                 spk_phr_a = ', '.join(spk_phr_a)
                #                 spk_phr_m = ', '.join(spk_phr_m)
                #                 spk_g = " </s> ".join(spk_phrs_clean + [spk_phr_g])
                #                 spk_l = " </s> ".join(spk_phrs_clean + [spk_phr_l])
                #                 spk_a = " </s> ".join(spk_phrs_clean + [spk_phr_a])
                #                 spk_m = " </s> ".join(spk_phrs_clean + [spk_phr_m])
                #                 sentences_to_proc_full[0].append(" </s></s> ".join([spk_g, dialogue]))
                #                 sentences_to_proc_full[1].append(" </s></s> ".join([spk_l, dialogue]))
                #                 sentences_to_proc_full[2].append(" </s></s> ".join([spk_a, dialogue]))
                #                 sentences_to_proc_full[3].append(" </s></s> ".join([spk_m, dialogue]))
                #             desc_new = ', '.join(spk_phr)
                #             spk_phrs_clean.append(desc_new)
                #         desc_new_full = " </s> ".join(spk_phrs_clean)
                #
                #         sentences_clean.append(" </s></s> ".join([desc_new_full, dialogue]))
                #
                # else:
                #     sentences_clean = sentences
                #     sentences_to_proc_full = [sentences]


                ngram = [N_GRAM]*len(sentences)

                # if "mask" in desc_type:
                #     csv_full['sentence{}_speaker_desc_{}'.format(ctx_size, desc_type)] = [0,0] + sentences_clean
                #     csv_bin['sentence{}_speaker_desc_{}'.format(ctx_size, desc_type)] = [0,0] + sentences_clean
                #probes = ['gender', 'language', 'age', 'mixing']
                #for s_ix, sentences_to_proc in enumerate(sentences_to_proc_full):
                nt_idx_matrices_all = []
                spk_maps_all = []
                for idx_spl, funct, sentences_tmp in zip([0,1], [self.map_to_phrases_speaker, self.map_to_phrases], [descs, sentences]):

                    ### need to account for the commas!!!!
                        # try:
                        #     #sentences_tmp = [s.split("</s></s>")[idx_spl] for s in sentences_tmp]
                        # except:
                        #     import pdb; pdb.set_trace()
                        sentences_to_proc = sentences_tmp
                        if idx_spl == 0:
                            sentences_to_tok = [s.replace("**", '') for s in sentences_tmp]
                        else:
                            sentences_to_tok = sentences_tmp
                        toks_tmp = self.tokenizer(sentences_to_tok, add_special_tokens=True)['input_ids']
                        toks_tmp = [[t for t in tk if t!= 6] for tk in toks_tmp]
                        word_to_tok_maps = list(map(self.map_words_to_subwords,toks_tmp, sentences_to_tok))
                        mega_list = list(map(funct, sentences_to_proc, word_to_tok_maps, ngram))
                        nt_idx_matrices = [m[0] for m in mega_list]
                        spk_maps = [m[1] for m in mega_list]
                        if idx_spl == 0:
                            spk_orders = csv['speakerOrder'].apply(lambda x: ast.literal_eval(x)).tolist()

                            if control_data:
                                spk_orders =[["BOT" if 'bot' in b else "HUM" for b in orders] for orders in spk_orders]
                            assert len(spk_orders) == len(spk_maps)
                            #spk_maps = [[s_map.get(sp, []) for sp in order] for order, s_map in zip(spk_orders, spk_maps)]
                            ##for s in spk_maps:
                            #    try:
                            #        assert not all([jj == [] for jj in s])
                            ##    except:
                            #        import pdb; pdb.set_trace()
                        spk_maps_all.append(spk_maps)
                        nt_idx_matrices_all.append(nt_idx_matrices)
                        speaker_nts_final = []
                        utt_nts_final = []
                        spk_maps_final = []
                        spk_utt_maps_final = []
                        spk_ids_all = csv['speakerIds'].apply(lambda x: ast.literal_eval(x)).tolist()
                        sentences_test = [" </s></s> ".join([d,s,]).replace("**", '') for d, s in zip(descs, sentences_to_proc)]
                        tok_test = self.tokenizer(sentences_test, add_special_tokens=True)['input_ids']
                        tok_test = [[t for t in tk if t!= 6] for tk in tok_test]

                for idx_spl, col_name, to_app, to_app_spk in zip([0,1], ['nt_idx_matrix{}_desc_speaker_{}', 'nt_idx_matrix{}_desc_utterance_{}'], [speaker_nts_final, utt_nts_final], [spk_maps_final, spk_utt_maps_final]):
                    other_nt = nt_idx_matrices_all[1-idx_spl]
                    spk_maps_tmp = spk_maps_all[idx_spl]
                    # if "mask" in desc_type:
                    #     col_name += "_{}".format(probes[s_ix])


                    for i, nt_list in enumerate(nt_idx_matrices_all[idx_spl]):
                        nt_tmp_final = []
                        phrases_check = []
                        phr_check_sp = []

                        tok_len_other = len(other_nt[i][0])
                        pad_toks_num = tok_len_other
                        spk_tmp_final = []


                        for nt_ in nt_list:



                            if idx_spl == 0:
                                nt_final = nt_ + [0]*pad_toks_num
                                #to_app = speaker_nts_final
                            else:
                                nt_final = [0]*pad_toks_num + nt_
                                #to_app = utt_nts_final
                            try:
                                assert len(nt_final) == len(tok_test[i])
                            except:
                                import pdb; pdb.set_trace()
                            jj = np.asarray(nt_final)
                            jj_phr = np.where(jj == 1)[0]
                            toks_phr = list(np.array(tok_test[i])[jj_phr])

                            # if "TOM sometimes" in toks_phr or "TOM is a sometimes" in toks_phr:
                            #     import pdb; pdb.set_trace()
                            phr_check = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(toks_phr))

                            phrases_check.append(phr_check)
                            nt_tmp_final.append(nt_final)
                        # if "IRI" and "JAM" in sentences_test[i] and 'sent' in desc_type:
                        #     import pdb; pdb.set_trace()
                        # if desc_type == 'partner' and valid:
                        #     import pdb; pdb.set_trace()
                        #if idx_spl == 1:
                        #    try:
                                #assert(len(spk_maps_tmp[i]) == len(spk_ids_all[i][-1*(ctx_size +1):]))
                            #except:
                        #        import pdb; pdb.set_trace()
                        # for spk_map in spk_maps_tmp[i]:
                        #
                        #     if idx_spl == 0:
                        #         if spk_map == []:
                        #             spk_tmp_final.append(spk_map + [0]*(pad_toks_num+len(nt_idx_matrices_all[idx_spl][i][0])))
                        #         else:
                        #
                        #             spk_tmp_final.append(spk_map + [0]*pad_toks_num)
                        #
                        #     else:
                        #
                        #
                        #         spk_tmp_final.append([0]*pad_toks_num + spk_map)
                        #     jj = np.asarray(spk_tmp_final[-1])
                        #     jj_phr = np.where(jj == 1)[0]
                        #     toks_phr = list(np.array(tok_test[i])[jj_phr])
                        #     phr_check_sp.append(self.tokenizer.convert_ids_to_tokens(toks_phr))
                        #
                        #





                        to_app.append(nt_tmp_final)
                        to_app_spk.append(spk_tmp_final)


                        # if idx_spl == 1:
                        #     import pdb; pdb.set_trace()
                        # if desc_type == 'list':
                        #     import pdb; pdb.set_trace()
                        # if 'sent' in desc_type:
                        #     import pdb; pdb.set_trace()



                    if not control_data:
                        to_app = [[0], [0]] + to_app
                        to_app_spk = [[0], [0]] + to_app_spk
                    #data_to_add.append(to_app)
                    #data_to_add_spk.append(to_app_spk)
                    #data_to_add[col_name.format(ctx_size, desc_type).replace("nt_idx_matrix", "speaker_map")] = to_app_spk
                    data_to_add[col_name.format(ctx_size, desc_type)] = to_app
                    #cols_to_add.append(col_name.format(ctx_size, desc_type))
        print(data_to_add.keys())
        tmp = pd.DataFrame(data_to_add)

        #for data_, data_spk, col_, col_spk in zip(data_to_add, data_to_add_spk, cols_to_add, cols_to_add_spk):
        try:

            #csv_full[col_spk] = data_spk
            #csv_full[col_] =data_
            csv_full = pd.concat([csv_full, tmp], axis=1)
            csv_full.drop(columns=['partner_description'], inplace=True)

            if binned_files is not None:
                csv_bin = pd.concat([csv_bin, tmp], axis=1)
                csv_bin.drop(columns=['partner_description'], inplace=True)
                # csv_bin[col_spk] = data_spk
                # csv_bin[col_] =data_
        except:
            import pdb; pdb.set_trace()


        csv_full.to_csv(output_file_name, index=False)

        if binned_files is not None:
            csv_bin.to_csv(bin_out, index=False)







def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--balanced", action='store_true',
                        help="Whether to parse balanced or unbalanced set.")

    parser.add_argument("--non_overlap_phrases", action='store_true',
                        help="Whether to take only every unique ngram or overlap phrases.")



    parser.add_argument("--tokenizer_name", default='xlm-roberta-base', type=str, required=False,
                        help="Tokenizer name")
    parser.add_argument("--control_data", action='store_true',
                        help="Whether or not to ")


    args = parser.parse_args()
    parsed_data = ParsedDataset(tokenizer_name=args.tokenizer_name)
    if args.non_overlap_phrases:
        OVERLAP_PHRASES=False

    # Read input files from folder
    if args.balanced:
        bal_n = 'balanced_'
        bal = '_balanced'
    else:
        bal = ''
        bal_n = ''
    spls = ['valid', 'test', 'train']
    if args.control_data:
        spls = ['full', 'train', 'valid', 'test']

    for file_split in spls:
        val=False
        print(file_split)
        if file_split == 'valid':
            val =True
        #input_file_name = os.path.join(args.data_dir, '{}_data_{}normalized.csv'.format(file_split, bal_n))
        #output_file_name = os.path.join(args.data_dir, '{}_data_{}normalized_with_parse.csv'.format(file_split, bal_n))
        if not args.control_data:
            input_file_name_binned = os.path.join(args.data_dir, '{}_binned_data{}.csv'.format(file_split, bal))
            output_file_name_binned = os.path.join(args.data_dir, '{}_binned_data{}_desc_with_parse.csv'.format(file_split, bal))
        else:
            spl_tmp = ''

            if file_split != 'full':
                spl_tmp +=  '_'+ file_split
            input_file_name_binned = os.path.join(args.data_dir, 'control_experiment_transformers{}.csv'.format(spl_tmp))

            output_file_name_binned = os.path.join(args.data_dir, 'control_experiment_desc{}_with_parse.csv'.format(spl_tmp))
        #output_file_name_binned = os.path.join(args.data_dir, '{}_binned_data{}_with_parse.csv'.format(file_split, bal))
        control_ = ''
        if args.control_data:
            control_ = '_control'
        spk_in = os.path.join(args.data_dir, 'speaker_description{}.csv'.format(control_))
        spk_out =os.path.join(args.data_dir, 'speaker_description{}_with_parse.csv'.format(control_))
        parsed_data.read_and_store_from_csv(binned_files=(input_file_name_binned, output_file_name_binned), spk_files = (spk_in, spk_out), control_data = args.control_data, valid=val)


if __name__ == "__main__":
    main()
