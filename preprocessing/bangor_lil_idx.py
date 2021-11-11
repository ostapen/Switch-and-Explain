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
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # if 'xlm-roberta' in tokenizer_name:
        #     self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
        #
        # elif 'roberta' in tokenizer_name:
        #     self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        # elif 'xlnet' in tokenizer_name:
        #     self.tokenizer = XLNetTokenizer.from_pretrained(tokenizer_name)
        # elif 'distilbert' in tokenizer_name:
        #     self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

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
        sent_split = [s for s in sent_split if s not in ['', ' ']]

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
        return word_to_tok_map
    def map_to_phrases_speaker(self, sent, word_to_tok_map, split_by_sep = ','):
        # if ngram == SPEECH_N_GRAM:
        #     import pdb; pdb.set_trace()
        idx_matrices = []

        sent_split = [s for s in sent.split(',') if s not in ['', ' ']]

        sent_split.insert(0, '<s>')
        sent_split.append("</s>")
        sent_spl_arr = np.array(sent_split)



        #eos_toks_map = [word_to_tok_map[eos][0] for eos in eos_toks]
        tok_len = sum([len(j) for j in word_to_tok_map])
        lens = [len(sent_split[r].split()) if r not in [0, len(sent_split)-1] else 0 for r in range(len(sent_split))]
        lens.pop(0)
        lens.pop(-1)

        num_phrases = len(sent_split)-2
        phrase_starts = []
        word_idx = 1


        for idx, length in enumerate(lens):
            phrase_starts.append(word_idx)
            word_idx += length
        for i, phrase_idx in enumerate(phrase_starts):
            if phrase_idx < len(word_to_tok_map):
                phrase = np.zeros(tok_len)
                words = [k for h in word_to_tok_map[phrase_idx:min(phrase_idx+lens[i], len(word_to_tok_map))] for k in h]
                phrase[words] = 1
                #phrase[eos_toks_map] = 0
                phrase[0] = 0
                phrase[-1] = 0

                if sum(phrase) == 0:
                    import pdb; pdb.set_trace()
                idx_matrices.append(list(phrase))
        if len(idx_matrices) == 0:
            import pdb; pdb.set_trace()
        return idx_matrices

    def map_to_phrases(self, sent, word_to_tok_map, ngram = N_GRAM, eot=False):
        # if ngram == SPEECH_N_GRAM:
        #     import pdb; pdb.set_trace()
        idx_matrices = []
        sent_split = sent.split()
        sent_split.insert(0, '<s>')
        sent_split.append("</s>")
        stop_toks = ['</s>','[EOU]','[EOT]']
        stop_toks += ["[SPK{}]".format(i+1) for i in range(5)]
        sent_spl_arr = np.array(sent_split)
        #eos_toks = np.where((sent_spl_arr == '</s>') | (sent_spl_arr == '[EOU]')| (sent_spl_arr == '[EOT]'))[0]
        eos_toks = np.where(np.isin(sent_spl_arr, np.array(stop_toks)))[0]
        eos_toks_list = sorted(list(eos_toks))
        # if len(eos_toks_list) > 1:
        #     eos_toks_list.pop(0)
        eos_toks_map = [word_to_tok_map[eos][0] for eos in eos_toks]
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
            lens.append(eos_toks[0]-1)

            for i_cur, eot_idx in enumerate(eos_toks_list[:-1]):
                next_idx = eos_toks_list[i_cur+1]
                lens.append(eos_toks_list[i_cur+1] - eot_idx-1)
                utt = np.array(sent_split)[eot_idx+1: next_idx]
                sent_splits_all.append(' '.join(utt))



        num_phrases = 0
        phrase_starts = []
        # if eot:
        #     ix_max = 0
        # else:
        ix_max = 0
        word_idx = 1+ix_max
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
            #if eot:
            word_idx = eos_toks_list[idx] +1
            #else:
            #    word_idx = eos_toks_list[idx+1] +1
            phrase_starts.append(tmp)
        if num_phrases == 0:
            import pdb; pdb.set_trace()

        if eot:
            for tk in eos_toks_map[:-1]:
                phrase = np.zeros(tok_len)
                phrase[tk] = 1
                idx_matrices.append(list(phrase))
        #for phrase_idx in range(max(1, num_phrases)):
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
        return idx_matrices


    def read_and_store_from_csv(self, binned_files, spk_files, control_data=False, probe_data = False):
        input_file_name, output_file_name = binned_files
        csv_full = pd.read_csv(input_file_name)
        if not control_data and not probe_data:
            csv=csv_full[2:]
        else:
            csv = csv_full.copy()
        if binned_files is not None:
            bin_in, bin_out = binned_files
        if spk_files is not None:
            spk_in, spk_out = spk_files
        csv_bin = pd.read_csv(bin_in)
        if not control_data:
            ctx_max = 6
        else:
            ctx_max= 4
        if probe_data:
            ctx_max = 2
        for ctx_size in [1,2,3,5,7,9,10]:
            if not probe_data:
                sents = list(csv['sentences_eot'].apply(lambda x: ' '.join(ast.literal_eval(x)[-1*(ctx_size+1):])))
                new_toks_ = ['[EOU]', '[EOT]']
            else:
                sents = list(csv['sentence'])
                new_toks_ = []
            """
            self.use_speaker_tokens = use_speaker_tokens
            if use_speaker_tokens:
                new_toks = ['[SPK{}]'.format(idx+1) for idx in range(5)]
            else:
                new_toks = ['[EOU]', '[EOT]']
        self.tokenizer.add_tokens(new_toks)
            """
            #for sentences, col_name, new_toks in zip([list(csv['sentence{}'.format(ctx_size)]), list(csv['sentence{}_speaker'.format(ctx_size)])], ['nt_idx_matrix{}', 'nt_idx_matrix{}_speaker'], [['[EOU]', '[EOT]'], ['[SPK{}]'.format(idx+1) for idx in range(5)]]):
            for sentences, col_name, new_toks in zip([sents], ['nt_idx_matrix{}'], [new_toks_]):

                #speaker_descriptions = list(csv['speakerDescriptions'])
            #for sentences in sents:
                if not probe_data:
                    self.tokenizer.add_tokens(new_toks)
                    print(self.tokenizer.get_added_vocab())
                ngram = [N_GRAM]*len(sentences)
                if '[EOT]' in new_toks:
                    eot = [True]*len(sentences)
                else:
                    eot = [False]*len(sentences)

                toks = self.tokenizer(list(sentences), add_special_tokens=True)['input_ids']
                toks = [[t for t in tk if t != 6] for tk in toks]
                word_to_tok_maps = list(map(self.map_words_to_subwords,toks, sentences))
                nt_idx_matrices = list(map(self.map_to_phrases, sentences, word_to_tok_maps, ngram, eot))

                if not control_data and not probe_data:
                    nt_idx_matrices_full = [[0],[0]]
                else:
                    nt_idx_matrices_full = []
                nt_idx_matrices_full.extend(nt_idx_matrices)
                try:

                    csv_full[col_name.format(ctx_size)] = nt_idx_matrices_full
                    if binned_files is not None:
                         csv_bin[col_name.format(ctx_size)] = nt_idx_matrices_full
                except:
                    import pdb; pdb.set_trace()
                self.reset_tokenizer()
        print(self.tokenizer.get_added_vocab())
        csv_full.to_csv(output_file_name, index=False)
        if binned_files is not None:
            csv_bin.to_csv(bin_out, index=False)
        process = False
        #if 'train' in input_file_name or control_data:
        if process:
            # TRY USING THE SENTENCE-XLMR HERE TOO
            # CAN ALSO TRY MULTIHEAD ATTENTION OVER THE OTHER SPEAKER'S FEATURES TO COMBINE
            #if not control_data:
            spk_description_csv = pd.read_csv(spk_in)
            simple_descs_tmp = spk_description_csv['simple_description_list'].apply(lambda x: x.replace(',', '')).tolist()
            simple_descs = spk_description_csv['simple_description_list'].tolist()
            phrase_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')
            phrase_model = AutoModel.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')
            #
            encoded_phrases = phrase_tokenizer(simple_descs_tmp, padding=True, truncation=True, return_tensors='pt')
            toks = encoded_phrases['input_ids'].tolist()

            #toks2 = self.tokenizer(simple_descs_tmp, add_special_tokens=True)['input_ids']
            word_to_tok_maps = list(map(self.map_words_to_subwords,tqdm(toks, total=len(toks)), simple_descs_tmp))
            nt_idx_matrices = list(map(self.map_to_phrases_speaker, simple_descs, word_to_tok_maps, [',']*len(simple_descs)))
            spk_description_csv['nt_idx_matrix'] = nt_idx_matrices
            spk_description_csv.to_csv(spk_out, index=False)
            spk_id_list = spk_description_csv['speakerId'].tolist()
            max_nt_len = 0
            max_tk_len = 0
            for s_id, nt_idx_matr, tk in zip(spk_id_list, nt_idx_matrices, toks):
                max_nt_len = max(len(nt_idx_matr), max_nt_len)
                max_tk_len = max(len(tk), max_tk_len)

            spk_desc_dict = {}
            with torch.no_grad():
                model_output = phrase_model(**encoded_phrases)
            sentence_embeddings = mean_pooling(model_output, encoded_phrases['attention_mask'])
            sent_emb_mean = torch.mean(sentence_embeddings, dim=0)
            sentence_embeddings_norm = sentence_embeddings-sent_emb_mean
            for idx_spk, s_id, nt_idx_matr, tk in zip([i for i in range(sentence_embeddings.shape[0])],spk_id_list, nt_idx_matrices, toks):
                tokens_padded = torch.zeros([max_tk_len]).long()
                try:
                    tokens_padded[:len(tk)] = torch.Tensor([tk]).long()
                    attention_mask = torch.zeros([max_tk_len]).long()
                    attention_mask[:len(tk)] = 1
                    try:
                        nt_ = pad_nt_matrix_roberta(torch.Tensor(nt_idx_matr), max_nt_len, max_tk_len, phrase_start = 0)
                    except:
                        import pdb; pdb.set_trace()
                    # spk_desc_dict[s_id] = {'nt_idx_matrix': nt_.long(), 'active_nt_tokens':[i for i in range(len(nt_idx_matr))], 'attention_mask': attention_mask, 'input_ids':tokens_padded, 'hidden_states': self.model(input_ids=tokens_padded.view(1, tokens_padded.shape[0]), \
                    #                                                                                                                                                                     attention_mask = attention_mask.long().view(1, attention_mask.shape[0]), output_hidden_states=True)['hidden_states']}
                    spk_desc_dict[s_id] = {'nt_idx_matrix': nt_.long(), 'active_nt_tokens':[i for i in range(len(nt_idx_matr))], 'attention_mask': encoded_phrases['attention_mask'][idx_spk], 'input_ids':tokens_padded, 'hidden_states': model_output[0][idx_spk], 'pool':sentence_embeddings_norm[idx_spk]}
                except:

                    import pdb; pdb.set_trace()




            #spk_desc_dict= {s_id: {'nt_idx_matrix': nt_idx_matr, 'input_ids':tk, 'hidden_states': self.model(input_ids=torch.Tensor([tk]).long(), attention_mask = torch.ones(torch.Tensor([tk]).size()).long(), output_hidden_states=True)['hidden_states']}  for s_id, nt_idx_matr, tk in zip(spk_id_list, nt_idx_matrices, toks)}
            torch.save(spk_desc_dict,spk_out.replace(".csv", '_and_toks.pt'))
            phrases = spk_description_csv['simple_description_list'].apply(lambda x: x.split(',')).tolist()
            if not control_data:

                phrases_unique = []
                #max_p_len = 0
                for p in phrases:
                    p_final = []
                    for phr in p:
                        if phr not in ['', ' ']:
                            p_final.append(phr)


                    phrases_unique.extend(p_final)

                phrases_unique = sorted(list(set(phrases_unique)))
                #phrase_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')
                encoded_phrases = phrase_tokenizer(phrases_unique, padding=True, truncation=True, return_tensors='pt')
                toks_for_phrases = encoded_phrases['input_ids'].tolist()
                lens = [len(phr) for phr in toks_for_phrases]
                max_p_len = max(lens)
                word_to_tok_maps = list(map(self.map_words_to_subwords,tqdm(toks_for_phrases, total=len(toks_for_phrases)), phrases_unique))
                nt_idx_matrices_phr = list(map(self.map_to_phrases_speaker, phrases_unique, word_to_tok_maps))
                try:
                    nt_padded = [pad_nt_matrix_roberta(torch.Tensor(nt_idx_matr), 1, max_p_len, phrase_start = 0) for nt_idx_matr in nt_idx_matrices_phr]
                except:
                    import pdb; pdb.set_trace()

                ## just make an embedding layer to load up... pre extract the input ids for the embed layer

                #phrase_model = AutoModel.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')
                # Tokenize sentences

                # Compute token embeddings
                with torch.no_grad():
                    model_output = phrase_model(**encoded_phrases)

                # Perform pooling. In this case, mean pooling.
                sentence_embeddings = mean_pooling(model_output, encoded_phrases['attention_mask'])
                mean_sent_emb = torch.mean(sentence_embeddings, dim=0)
                sentence_embeddings = sentence_embeddings-mean_sent_emb
                # hidden_states= [self.model(input_ids=torch.Tensor([tk_phr]).long(), attention_mask = torch.ones(torch.Tensor([tk_phr]).size()).long(), output_hidden_states=True)['hidden_states'] \
                #         for tk_phr in toks_for_phrases]
                #pooled = [torch.zeros(1, self.config.hidden_size)]+[self.pooler(h[-1]).view(1,-1) for h in hidden_states]
                pooled = torch.cat([torch.zeros(1, self.config.hidden_size), sentence_embeddings], dim=0)
                phrase_emb = torch.nn.Embedding.from_pretrained(pooled)
                # 0 is the pad token
                phr2ix = {phr: ix+1 for ix, phr in enumerate(phrases_unique)}
                ix2phr = {v:k for k,v in phr2ix.items()}
            else:
                phrase_info_dict = torch.load(spk_out.replace("_control_with_parse.csv", '_phrase_embeddings_with_parse.pt'))
                phr2ix = phrase_info_dict['phr2ix']
                ix2phr = phrase_info_dict['ix2phr']
                phrases_unique = sorted([p for p in phr2ix.keys()])
                max_p_len = phrase_info_dict['spk2data']['MAR_zeledon8']['nt_idx_matrix'].shape[1]


            phr_tokens_emb = []
            nt_padded_spk = []
            active_nt_ = []
            for p in phrases:
                tmp_toks = []
                nt_tmp_all = []
                for phr in p:
                    if phr not in ['', ' ']:
                        try:
                            ix_phr = phr2ix[phr]
                        except:
                            import pdb; pdb.set_trace()
                        tmp_toks.append(ix_phr)
                        #nt_pad_ = nt_padded[ix_phr-1]
                active_nt_.append(torch.LongTensor([i for i in range(len(tmp_toks))]))
                og_len = len(tmp_toks)
                if len(tmp_toks)< max_p_len:
                    tmp_toks = tmp_toks + [0]*(max_p_len-len(tmp_toks))

                for i in range(len(tmp_toks)):
                    nt_tmp = torch.zeros(len(tmp_toks))
                    if i >= og_len and og_len < max_p_len:
                        nt_tmp_all.append(nt_tmp.long())
                    else:
                        nt_tmp[i] = 1
                        nt_tmp_all.append(nt_tmp.long())

                phr_tokens_emb.append(torch.LongTensor(tmp_toks))
                nt_padded_spk.append(torch.stack(nt_tmp_all, dim=0))
                #nt_padded_spk.append(nt_tmp_all)
            if not control_data:
                phrase_info_dict = {'hidden_states': model_output[0], 'phrase_words': phrases_unique, 'phr2ix': phr2ix, 'ix2phr': ix2phr, 'nt_idx_matrix':nt_padded, \
                                    'spk2data': {sp: {'input_ids': tk, 'nt_idx_matrix': nt_, 'active_nt_tokens':a_nt, 'attention_mask':torch.LongTensor(a_nt)} for sp, tk, nt_, a_nt in zip(spk_id_list, phr_tokens_emb, nt_padded_spk, active_nt_)}}
                if len(phrase_info_dict['spk2data']) == 0:
                    import pdb; pdb.set_trace()
                torch.save(phrase_info_dict, spk_out.replace("_with_parse.csv", '_phrase_embeddings_with_parse.pt'))


                #phrase_embed = torch.nn.Embedding(num_embeddings=len(phrases_unique), embedding_dim=self.config.hidden_size, padding_idx=0, )
                torch.save(phrase_emb,spk_out.replace("_with_parse.csv", '_phrase_embeddings.pt'))
            else:
                phr_dict = {'spk2data': {sp: {'input_ids': tk, 'nt_idx_matrix': nt_, 'active_nt_tokens':a_nt, 'attention_mask': torch.LongTensor(a_nt)} for sp, tk, nt_, a_nt in zip(spk_id_list, phr_tokens_emb, nt_padded_spk, active_nt_)}}
                torch.save(phr_dict, spk_out.replace("_with_parse.csv", '_phrase_embeddings_with_parse.pt'))
        #torch.save(spk_desc_dict,spk_out.replace(".csv", '.pt'))






def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--balanced", action='store_true',
                        help="Whether to parse balanced or unbalanced set.")

    parser.add_argument("--non_overlap_phrases", action='store_true',
                        help="Whether to take only every unique ngram or overlap phrases.")

    parser.add_argument("--probe_data", action='store_true',
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
    spls = ['train', 'valid', 'test']
    if args.control_data:
        spls = ['full', 'train', 'valid', 'test']
    control_ = ''
    if args.control_data:
        control_ = '_control'

    for file_split in spls:
        print(file_split)



        #input_file_name = os.path.join(args.data_dir, '{}_data_{}normalized.csv'.format(file_split, bal_n))
        #output_file_name = os.path.join(args.data_dir, '{}_data_{}normalized_with_parse.csv'.format(file_split, bal_n))
        if not args.probe_data:
            if not args.control_data:
                input_file_name_binned = os.path.join(args.data_dir, '{}_binned_data{}.csv'.format(file_split, bal))
                output_file_name_binned = os.path.join(args.data_dir, '{}_binned_data{}_with_parse.csv'.format(file_split, bal))

            else:
                spl_tmp = ''

                if file_split != 'full':
                    spl_tmp +=  '_'+ file_split
                input_file_name_binned = os.path.join(args.data_dir, 'control_experiment_transformers{}.csv'.format(spl_tmp))
                output_file_name_binned = os.path.join(args.data_dir, 'control_experiment{}_with_parse.csv'.format(spl_tmp))
            #output_file_name_binned = os.path.join(args.data_dir, '{}_binned_data{}_with_parse.csv'.format(file_split, bal))

            spk_in = os.path.join(args.data_dir, 'speaker_description{}.csv'.format(control_))
            spk_out =os.path.join(args.data_dir, 'speaker_description{}_with_parse.csv'.format(control_))
            parsed_data.read_and_store_from_csv(binned_files=(input_file_name_binned, output_file_name_binned), spk_files = (spk_in, spk_out), control_data = args.control_data)
        else:
            for probe in ['gender', 'age', 'language', 'mixing']:
                input_file_name_binned = os.path.join(args.data_dir, '{}_{}_probe.csv'.format(file_split, probe))
                output_file_name_binned = os.path.join(args.data_dir, '{}_{}_probe_with_parse.csv'.format(file_split,probe))
                spk_in = os.path.join(args.data_dir, 'speaker_description{}.csv'.format(control_))
                spk_out =os.path.join(args.data_dir, 'speaker_description{}_with_parse.csv'.format(control_))
                parsed_data.read_and_store_from_csv(binned_files=(input_file_name_binned, output_file_name_binned), spk_files = None, control_data = args.control_data, probe_data=args.probe_data)


if __name__ == "__main__":
    main()
