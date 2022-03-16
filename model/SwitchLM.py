from argparse import ArgumentParser

import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW
from transformers import AutoModel, AutoConfig
from transformers.modeling_utils import SequenceSummary
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import math
import os
from .model_utils import TimeDistributed
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import pickle as pkl


class SwitchLM(LightningModule):
    def __init__(self, hparams, vocab_size, tuning_parameters=None):
        super().__init__()

        if tuning_parameters is not None:
            """
            args_dict['context_size'] = tune.choice([1, 2, 3])
            args_dict['batch_size'] = tune.choice([16, 32])
            args_dict['lr'] = tune.loguniform(1e-5, 1e-2)
            args_dict['phrase_embeddings'] = tune.choice([True, False])
            args_dict['multihead_pool'] = tune.choice([True, False])
            args_dict['multihead_pool_over_input'] = tune.choice([True, False])
            args_dict['concatenate_speaker'] = tune.choice([True, False])
            args_dict['lamda'] = tune.uniform(0, 10)
            args_dict['lamda_spk'] = tune.uniform(0, 10)
            """
            hparams.context_size = tuning_parameters['context_size']
            hparams.batch_size = tuning_parameters['batch_size']
            hparams.lr = tuning_parameters['lr']
            hparams.phrase_embeddings = tuning_parameters['phrase_embeddings']
            hparams.multihead_pool = tuning_parameters['multihead_pool']
            hparams.multihead_pool_over_input = tuning_parameters['multihead_pool_over_input']
            hparams.concatenate_speaker = tuning_parameters['concatenate_speaker']
            hparams.lamda = tuning_parameters['lamda']
            hparams.lamda_spk = tuning_parameters['lamda_spk']
        #self.hparams = hparams
        #print(self.hparams)
        self.save_hyperparameters(hparams, 'vocab_size')
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(self.hparams.model_name)
        self.model = AutoModel.from_pretrained(self.hparams.model_name)

        if not self.hparams.no_adapter:
            if self.hparams.use_speaker_tokens:
                adapter = self.model.load_adapter(self.hparams.adapter_path_speaker, config="pfeiffer+inv")
            else:
                adapter = self.model.load_adapter(self.hparams.adapter_path, config="pfeiffer+inv")
            self.model.set_active_adapters(adapter)

        vocab_size = vocab_size



        self.model.word_embeddings = self.model.resize_token_embeddings(vocab_size)
        print(vocab_size)
        self.model.position_embeddings = self.model.embeddings.position_embeddings



        # idx 0 is reserved for padding / separators
        # only consider spk + gp other speakers into 'partners' - ie consider only 2 indices
        #self.speaker_encodings = nn.Embedding(self.hparams.speaker_emb_dim, config.hidden_size, padding_idx=0)
        if self.hparams.use_speaker_descriptions:
            if self.hparams.multihead_pool or self.hparams.multihead_pool_over_input:

                self.speaker_pool = torch.nn.MultiheadAttention(self.config.hidden_size,
                                                                dropout=0.2,
                                                                num_heads=8)
            if not self.hparams.phrase_embeddings:
                self.speaker_encodings = torch.load(self.format_str(self.hparams.speaker_descriptions))
            else:
                self.phrase_dictionary= torch.load(self.format_str(self.hparams.phrase_dictionary_path))
                self.speaker_encodings =self.phrase_dictionary['spk2data']

                self.speaker_phrase_embeddings = torch.load(self.format_str(self.hparams.phrase_embeddings_path))
                self.speaker_phrase_embeddings.weight.requires_grad = True

        # print("vocab size, xlmr, updated")
        # print(self.model.vocab_size)
        """
        have explicit tokens [SPK1], [SPK2] in the input?
        
        """



        # why sequencesummary, why not just last hidden state?
        # from transformer: This output is usually not a good summary of the semantic content of the input,
        # you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence.
        self.pooler = SequenceSummary(self.config)

        num_classifier_feats =self.config.hidden_size
        # if self.hparams.use_speaker_descriptions:
        #     num_classifier_feats = num_classifier_feats*2


        #if self.hparams.





        if not self.hparams.concatenate_speaker:
            in_d = num_classifier_feats
        else:
            in_d = num_classifier_feats*2
        self.classifier = nn.Linear(in_d, self.hparams.num_classes)

        if self.hparams.concat_speaker_logits:

            self.speaker_linear = nn.Linear(in_d*(self.hparams.context_size+1), self.hparams.num_classes)
            self.ensemble_classifier = nn.Linear(self.hparams.num_classes*2, self.hparams.num_classes)



            # map to same embed space to simply add together
        if 'speaker_trait_predictions' in self.hparams:
            if self.hparams.speaker_trait_predictions:
                self.g_classifier = nn.Linear(in_d, 2)
                self.l_classifier = nn.Linear(in_d, 2)
                self.a_classifier = nn.Linear(in_d, 2)
                self.m_classifier = nn.Linear(in_d, 2)









        if self.hparams.self_explain_ngram:
            #self.concept_store = torch.load(self.hparams.concept_store)
            self.phrase_logits = TimeDistributed(nn.Linear(in_d,
                                                            self.hparams.num_classes))
            self.speaker_phrase_logits = TimeDistributed(nn.Linear(self.config.hidden_size,
                                                                   self.hparams.num_classes))

            self.topk =  self.hparams.topk
            self.multihead_attention = torch.nn.MultiheadAttention(self.config.hidden_size,
                                                                   dropout=0.2,
                                                                   num_heads=8)
            # self.topk_gil_mlp = TimeDistributed(nn.Linear(config.hidden_size,
            #                                               self.hparams.num_classes))

            self.topk_gil_mlp = nn.Linear(self.config.hidden_size,self.hparams.num_classes)


            self.activation = nn.ReLU()

            self.lamda = self.hparams.lamda
            self.gamma = self.hparams.gamma
            self.lamda_spk = self.hparams.lamda_spk

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        if self.hparams.balanced:
            weights = None
        else:
            print("class weights")
            weights=torch.cuda.FloatTensor([1,4])
            #print(weights)
        weights=torch.cuda.FloatTensor([1,4])
        self.loss = nn.CrossEntropyLoss(weight=weights)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--balanced", action='store_true',
                            help="use balanced datasets")
        parser.add_argument("--min_lr", default=0, type=float,
                            help="Minimum learning rate.")
        parser.add_argument("--self_explain_ngram", action='store_true',
                            help="Whether to use self explain framework for ngram")
        parser.add_argument("--plateau_scheduler", action='store_true',
                            help="Use plateau scheduler or not")

        parser.add_argument("--save_val_acc", action='store_true',
                            help="Save model according to val_acc")

        parser.add_argument("--save_val_f1", action='store_true',
                            help="Save model according to val_f1")
        parser.add_argument("--speaker_trait_predictions", action='store_true',
                            help="Add additional losses over tasks to predict speaker traits from utterances ")
        parser.add_argument("--full_mtl_setup", action='store_true',
                            help="use mtl loss at val+test time")

        parser.add_argument("--only_lil", action='store_true',
                            help="Don't use concept store, only lil layer")
        parser.add_argument("--only_gil", action='store_true',
                           help="use concept store, but not lil layer")
        parser.add_argument("--eval_per_language", action='store_true',
                            help="add evaluation for english and spanish separately")
        parser.add_argument("--speaker_descriptions", type=str,default = '/projects/tir3/users/aostapen/CodeSwitch/SelfExplain/bangor_data/feature_spk_new_raw/speaker_description{}_with_parse_and_toks.pt',
                            help="path to speaker description file....")



        parser.add_argument("--concept_store", type=str,default = '/projects/tir3/users/aostapen/CodeSwitch/SelfExplain/bangor_data/feature_spk2/concept_store.pt',
                            help="path to ngram concept store")


        parser.add_argument("--add_spk_logits", action='store_true',
                            help="Add speaker logits to the sentence logits for computing predictions.")
        parser.add_argument("--multihead_pool", action='store_true',
                            help="Whether to pool using attention instead of the seqsummary (self attention w speaker desc")
        parser.add_argument("--multihead_pool_over_input", action='store_true',
                            help="Whether to pool using attention instead of the seqsummary, attention with speaker's words as (key?)")
        parser.add_argument("--concat_speaker_logits", action='store_true',
                            help="Keep speaker info separate from sentences embs, cat the logits after passing them through a linear layer")

        parser.add_argument("--concatenate_speaker", action='store_true',
                            help="Whether to concatenate speaker features to sentence embeddings (default: sum).")
        parser.add_argument("--phrase_embeddings", action='store_true',
                            help="Instead of speaker description tokens, use a phrase embedding table + finetune.")

        parser.add_argument("--phrase_embeddings_path", type=str,default = '/projects/tir3/users/aostapen/CodeSwitch/SelfExplain/bangor_data/feature_spk_new_raw/speaker_description{}_phrase_embeddings.pt',
                            help="path to phrase embeddings layer")
        parser.add_argument("--phrase_dictionary_path", type=str,default = '/projects/tir3/users/aostapen/CodeSwitch/SelfExplain/bangor_data/feature_spk_new_raw/speaker_description{}_phrase_embeddings_with_parse.pt',
                            help="path to phrase embeddings layer")


        parser.add_argument("--adapter_path", type=str,default = '/projects/tir3/users/aostapen/CodeSwitch/SelfExplain/pretrained/adapter-raw/checkpoint-8500/mlm',
                            help="path to pretrained adapter for bangor text")
        parser.add_argument("--adapter_path_speaker", type=str,default = '/projects/tir3/users/aostapen/CodeSwitch/SelfExplain/pretrained/adapter-raw-speaker/mlm',
                            help="path to pretrained adapter for bangor text")
        parser.add_argument("--no_adapter", action='store_true',
                            help="run raw XLMR")
        parser.add_argument("--speaker_trait_predictions_utt", action='store_true',
                            help="Save model according to val_f1")
        parser.add_argument("--speaker_trait_predictions_spk_utt", action='store_true',
                            help="Save model according to val_f1")
        parser.add_argument("--speaker_trait_predictions_spk", action='store_true',
                            help="Save model according to val_f1")


        parser.add_argument("--topk_output_file", default='topk_outputs.pkl', type=str,
                            help="filepath to save topk outputs to. default saves in topks/ directory under models/")
        parser.add_argument("--h_dim", type=int,
                            help="Size of the hidden dimension.", default=768)
        parser.add_argument("--n_heads", type=int,
                            help="Number of attention heads.", default=1)
        parser.add_argument("--kqv_dim", type=int,
                            help="Dimensionality of the each attention head.", default=256)
        parser.add_argument("--num_classes", type=float,
                            help="Number of classes.", default=2)
        parser.add_argument("--lr", default=3e-5, type=float,
                            help="Initial learning rate.")
        parser.add_argument("--weight_decay", default=0.01, type=float,
                            help="Weight decay rate.")
        parser.add_argument("--warmup_prop", default=0.1, type=float,
                            help="Warmup proportion.")
        parser.add_argument("--topk", default=10, type=int,
                            help="Topk GIL concepts")
        parser.add_argument("--lamda", default=0.01, type=float,
                            help="Lamda Parameter")
        parser.add_argument("--lamda_spk", default=0.03, type=float,
                            help="Lamda Parameter")
        parser.add_argument("--prepend_description", action='store_true',
                            help="prepend spk description to the dialogue")
        parser.add_argument("--gamma", default=0.01, type=float,
                            help="Gamma parameter")
        return parser
    def format_str(self, inp_file, format_str = ''):
        return inp_file.format(format_str)
    def load_control_embeddings(self):
        if not self.hparams.phrase_embeddings:
            self.speaker_encodings = torch.load(self.format_str(self.hparams.speaker_descriptions, '_control'))
        # else:
        #     self.phrase_dictionary= torch.load(self.format_str(self.hparams.phrase_dictionary_path,'_control'))
        #     self.speaker_encodings =self.phrase_dictionary['spk2data']
        #
        #     self.speaker_phrase_embeddings = torch.load(self.format_str(self.hparams.phrase_embeddings_path,'_control'))
        #     self.speaker_phrase_embeddings.weight.requires_grad = True
    def resize_token_embeddings(self, vocab_size):
        print("updating vocab size")
        self.model.resize_token_embeddings(vocab_size)
    def mean_pooling(self, model_output, attention_mask, hidden_states=False):
        # source https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual
        try:
            if not hidden_states:
                token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            else:
                token_embeddings = model_output
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().cuda()
        except:
            import pdb; pdb.set_trace()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def configure_optimizers(self):
        optimizer= AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), weight_decay=0.001,
                         eps=1e-6)
        mode = 'max'
        monitor = 'val_acc'
        # if self.mlm:
        #     mode = 'min'
        #     monitor = 'val_loss'

        #if self.hparams.plateau_scheduler:
        #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=, num_cycles=0.5, last_epoch=- 1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.5,
                                                               patience=3,
                                                               mode=mode,
                                                               verbose=True)
        # return [optimizer], [scheduler]
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': monitor
        }
        #else:
        #    return optimizer


    def forward(self, batch):
        # if self.hparams.self_explain_ngram:
        #     self.concept_store = self.concept_store.to(self.model.device)

        tokens, tokens_mask,  (padded_ndx_tensor,active_nt_tokens), speaker_orders, speaker_context_ids, spa2eng, eng2spa, utt_masks, speaker_masks, speaker_utt_masks, genders, ages, mixes, langs, labels = batch
        # step 1: encode the sentence
        sentence_cls, hidden_state, (speaker_hiddens_all, spk_attn_mask, spk_nts_all, active_nt_toks_spk, spk_toks, spks_all, spk_list_all) = self.forward_classifier(input_ids=tokens,\
                                                                    spk_ids_context=speaker_context_ids, spk_ids_order= speaker_orders, \
                                                                                            attention_mask=tokens_mask,token_type_ids=None)
        #if self.hparams.use_cs_feats:
        if self.hparams.use_speaker_descriptions:


            speaker_hiddens_all = speaker_hiddens_all.to(self.model.device)
            spk_nts_all =spk_nts_all.to(self.model.device).float()
            spk_list_all = spk_list_all.to(self.model.device)






        logits = self.classifier(sentence_cls)
        if self.hparams.concat_speaker_logits:
            spk_logits = self.speaker_linear(spk_list_all)
            logits_full = torch.cat([logits, spk_logits], dim=-1)

        else:
            logits_full = logits
        if logits_full.sum().isnan().item():
            import pdb; pdb.set_trace()
        lil_logits = None
        lil_logits_mean = None
        gil_logits, topk_indices = None, None
        acc_for_lang = None
        tpr, precision, recall, f1 = None, None, None, None
        logits_dict = {'logits': logits_full}
        if 'speaker_trait_predictions' in self.hparams:
            if self.hparams.speaker_trait_predictions and utt_masks is not None:
                logits_dict['glam'] = []
                logits_dict['glam_labels'] = [genders, langs, ages, mixes]
                pooled_seq_rep = self.mean_pooling(hidden_state, utt_masks, hidden_states=True)
                for classifier in [self.g_classifier, self.l_classifier, self.a_classifier, self.m_classifier]:
                    logits_ = classifier(pooled_seq_rep)
                    logits_dict['glam'].append(logits_)
        if self.hparams.concat_speaker_logits:
            logits_dict['logits_utt']= logits
            logits_dict['logits_spk']= spk_logits
        topks = {}
        if self.hparams.self_explain_ngram:

            lil_logits_mean = torch.Tensor([0]).cuda()
            lil_logits_spk_mean = torch.Tensor([0]).cuda()
            gil_logits = torch.Tensor([0]).cuda()
            if not self.hparams.only_lil:
                gil_logits, topk_indices = self.gil(pooled_input=sentence_cls)
                topks['topk_gil'] = topk_indices
            # lil_logits_lex = self.lil(hidden_state=lex_feats,
            #                       nt_idx_matrix=padded_lex_ndx_tensor)
            # lil_logits_psych = self.lil(hidden_state=psych_feats,
            #                           nt_idx_matrix=padded_psych_ndx_tensor)
            if not self.hparams.only_gil:
                try:
                    lil_logits, topk_utt = self.lil(hidden_state=hidden_state,
                                          nt_idx_matrix=padded_ndx_tensor, attention_mask=tokens_mask)
                except:
                    import pdb; pdb.set_trace()
                lil_logits_tmp = [lil_logits[b][active_nt_tokens[b], :] for b in range(len(tokens))]
                lil_logits_mean= torch.stack([torch.mean(logits_, dim=0) for logits_ in lil_logits_tmp], dim=0).cuda()

                logits_dict['lil_logits_utt'] = lil_logits
                logits_dict['lil_logits_utt_active_ndxs'] = active_nt_tokens
                topks['topk_utt'] = topk_utt


                #lil_logits_mean = torch.mean(lil_logits, dim=1)
                if self.hparams.use_speaker_descriptions:

                    # speaker_hiddens_all = speaker_hiddens_all.to(self.model.device)
                    # spk_nts_all =spk_nts_all.to(self.model.device).float()

                    lil_logits_sp, topk_spk = self.lil(hidden_state=speaker_hiddens_all,
                                             nt_idx_matrix=spk_nts_all, attention_mask = spk_attn_mask, speaker=True)
                    lil_logits_tmp_sp = [lil_logits_sp[b][active_nt_toks_spk[b], :] for b in range(len(spk_nts_all))]
                    #TODO
                    lil_logits_spk_mean_tmp= torch.stack([torch.mean(logits_, dim=0) for logits_ in lil_logits_tmp_sp], dim=0).cuda()
                    # #lil_logits_spk_mean = torch.mean(lil_logits_sp, dim=1)
                    #logits_dict['lil_logits_spk'] = lil_logits_sp
                    #logits_dict['lil_logits_spk_active_idxs'] = active_nt_toks_sp
                    lil_logits_sp_dict = {sp: log_ for sp, log_ in zip(spks_all, lil_logits_tmp_sp)}
                    spk_nt_act_dict = {sp: nt_ for sp, nt_ in zip(spks_all, active_nt_toks_spk)}
                    #lil_topk_sp_dict = {sp: topk_ for sp, topk_ in zip(spks_all, topk_sp)}
                    logits_dict['lil_logits_spk'] = lil_logits_sp
                    all_lil_logits = []
                    all_active_nt = []
                    lil_logits_spk_mean_dict = {sp: mn_ for sp, mn_ in zip(spks_all, lil_logits_spk_mean_tmp)}
                    lil_logits_spk_mean = []
                    topk_spk_dict = {sp: tpk for sp, tpk in zip(spks_all, topk_spk)}
                    toks_spk_dict = {sp: tok for sp, tok in zip(spks_all, spk_toks)}

                    nt_spk_dict = {sp: nt for sp, nt in zip(spks_all, spk_nts_all)}
                    topks_all = []
                    spk_toks_all = []
                    nt_all = []



                    for sp_ctx, sp_order in zip(speaker_context_ids, speaker_orders):
                        tmp_log_ = {}
                        tmp_nt = {}
                        tmp_logits_ = 0
                        tmp_topk = {}
                        tmp_tok = {}
                        tmp_full_nt = {}
                        for sp in set(sp_ctx):
                            if sp in lil_logits_sp_dict:
                                try:
                                    tmp_log_[sp] = lil_logits_sp_dict[sp]
                                except:
                                    import pdb; pdb.set_trace()
                                tmp_logits_ = tmp_logits_ + lil_logits_spk_mean_dict[sp]
                                tmp_nt[sp] = spk_nt_act_dict[sp]
                                tmp_topk[sp] = topk_spk_dict[sp]
                                tmp_tok[sp] = toks_spk_dict[sp]
                                tmp_full_nt[sp] = nt_spk_dict[sp]

                        all_lil_logits.append(tmp_log_)
                        lil_logits_spk_mean.append(tmp_logits_)
                        all_active_nt.append(tmp_nt)
                        topks_all.append(tmp_topk)
                        spk_toks_all.append(tmp_tok)
                        nt_all.append(tmp_full_nt)
                    lil_logits_spk_mean = torch.stack(lil_logits_spk_mean, dim=0)

                    logits_dict['lil_logits_spk'] = all_lil_logits
                    logits_dict['lil_logits_spk_active_idxs'] = all_active_nt
                    logits_dict['speaker_tokens'] = spk_toks_all
                    logits_dict['nt_speaker_all'] = nt_all



                    topks['topk_spk'] = topks_all
            logits_full = logits + self.lamda * lil_logits_mean + self.lamda_spk*lil_logits_spk_mean
            #self.gamma * gil_logits -- ignore the gil logits
            # if self.hparams.add_spk_logits:
            #     logits_full = logits_full + self.lamda_spk * lil_logits_spk_mean
        else:
            logits_full = logits
        # if self.hparams.threshold != 0.5:
        #     predicted_labels = torch.where(torch.nn.functional.softmax(logits_full, dim=1)[:, 1] >=self.hparams.threshold, 1, 0)
        # else:
        predicted_labels = torch.argmax(logits_full, -1)
        if labels is not None:
            acc = torch.true_divide(
                (predicted_labels == labels).sum(), labels.shape[0])
            cs_idxs = torch.where(labels == 1, True, False)
            non_cs_idxs = ~cs_idxs

            tps = (predicted_labels[cs_idxs] == 1).sum()
            tns = (predicted_labels[non_cs_idxs] == 0).sum()
            fns = (predicted_labels[cs_idxs] != 1).sum()
            fps = (predicted_labels[non_cs_idxs] != 0).sum()
            if len(cs_idxs) > 0:
                if tps+fns > 0:
                    tpr = tps/(tps + fns)
                else:
                    tpr = torch.Tensor([0]).cuda()
                if tps+fps > 0:
                    precision = tps/(tps+fps)
                else:
                    precision = torch.Tensor([0]).cuda()
                if tps+fns > 0:
                    recall = tps/(tps+fns)
                else:
                    recall = torch.Tensor([0]).cuda()
                accuracy = (tps+tns)/(tps+tns+fns+fps)
                if precision+recall > 0:
                    f1 = (2*precision*recall)/(precision+recall)
                else:
                    f1 = torch.Tensor([0]).cuda()

            if self.hparams.eval_per_language:
                acc_for_lang = {'eng': None, 'spa': None}
                #masked_select = labels.masked_select(spa2eng.bool())
                lang_to_proc = {'eng': eng2spa, 'spa':spa2eng}
                for lang, ids in lang_to_proc.items():
                    ids_lang = torch.where(ids== 1)[0]
                    labels_for_lang= labels[ids_lang]
                    if len(labels_for_lang) > 0:
                        assert labels_for_lang.sum() == len(labels_for_lang)
                        predicted_for_lang = predicted_labels[ids_lang]
                        correct_preds = torch.true_divide(
                            (predicted_for_lang == labels_for_lang).sum(), labels_for_lang.shape[0])
                        acc_for_lang[lang] = correct_preds

        else:
            acc, tpr, precision, recall, f1 = None, None, None, None, None
        # return logits, acc, {"topk_indices": topk_indices,
        #                      "lil_logits": lil_logits}
        return logits_full, acc, tpr, precision, recall, f1, acc_for_lang, topks, logits_dict

    def gil(self, pooled_input):
        batch_size = pooled_input.size(0)
        # torch.Size([32, 13318])
        inner_products = torch.mm(pooled_input, self.concept_store.T)
        # torch.Size([32, 10])
        _, topk_indices = torch.topk(inner_products, k=self.topk)

        # torch.Size([320, 768]); topkindices.view(-1).shape=[320]
        topk_concepts = torch.index_select(self.concept_store, 0, topk_indices.view(-1))
        # torch.Size([32, 10, 768])
        topk_concepts = topk_concepts.view(batch_size, self.topk, -1).contiguous()

        # concat_pooled_concepts.shape
        concat_pooled_concepts = torch.cat([pooled_input.unsqueeze(1), topk_concepts], dim=1)
        attended_concepts, _ = self.multihead_attention(query=concat_pooled_concepts,
                                                     key=concat_pooled_concepts,
                                                     value=concat_pooled_concepts)

        gil_topk_logits = self.topk_gil_mlp(attended_concepts[:,0,:])
        # print(gil_topk_logits.size())
        # gil_logits = torch.mean(gil_topk_logits, dim=1)
        return gil_topk_logits, topk_indices

    def lil(self, hidden_state, nt_idx_matrix, speaker=False, attention_mask=None):
        try:
            phrase_level_hidden = torch.bmm(nt_idx_matrix, hidden_state)
        except:
            import pdb; pdb.set_trace()
        batch_size = hidden_state.size(0)
        # torch.Size([32, 14, 768])
        phrase_level_activations = self.activation(phrase_level_hidden)

        ## maybe sequence summary isn't good??
        #pooled_seq_rep = self.pooler(hidden_state).unsqueeze(1)
        pooled_seq_rep = self.mean_pooling(hidden_state, attention_mask, hidden_states=True).unsqueeze(1)
        try:
            phrase_level_activations = phrase_level_activations - pooled_seq_rep
        except:
            import pdb; pdb.set_trace()
        #tmp = torch.abs(phrase_level_activations).sum(dim=2)
        tmp = phrase_level_activations.sum(dim=2)
        _, topk_indices = torch.topk(tmp, k=min(self.topk, nt_idx_matrix.shape[1]))
        topk_concepts = torch.stack([nt_idx_matrix[b][topk_indices[b]] for b in range(batch_size)], dim=0)
        #topk_concepts = topk_concepts.view(batch_size, self.topk, -1).contiguous()
        if not speaker:
            phrase_level_logits = self.phrase_logits(phrase_level_activations)
        else:
            phrase_level_logits = self.speaker_phrase_logits(phrase_level_activations)
        return phrase_level_logits, topk_concepts


    def forward_classifier(self, input_ids: torch.Tensor, spk_ids_order: list, spk_ids_context: list, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None):
        """Returns the pooled token
        """
        #TODO -- mask out padding value in attention!!!!

        #word_embeddings = self.model.word_embeddings(input_ids)
        #spk_embeddings = self.speaker_encodings(spk_ids)
        #input_embs = word_embeddings + spk_embeddings
        # posn embeddings are added in the model's forward
        # idx 0 is reserved for padding / separators
        # only consider spk + gp other speakers into 'partners' - ie consider only 2 indices
        # outputs = self.model(inputs_embeds=input_ids,
        #                      token_type_ids=token_type_ids,
        #                      attention_mask=attention_mask,
        #                      output_hidden_states=True)

        speaker_hiddens_all = []
        spk_nts_all = []
        active_nt_toks = []
        spks_all = []
        spk_toks = []
        spk_attens = []

        key_padding_masks = []
        spk_list_all = []
        if self.hparams.use_speaker_descriptions and not self.hparams.prepend_description:
            spk_ids_mask = None
            if type(attention_mask) == tuple:
                attention_mask = attention_mask[0]
            hidden_states = self.model(input_ids = input_ids, token_type_ids=token_type_ids,
                                       attention_mask = attention_mask,
                                       output_hidden_states=True)['hidden_states'][-1]
            if torch.sum(hidden_states).isnan():
                import pdb; pdb.set_trace()

            speaker_hiddens = {}
            #sp_embs = {s_id: self.speaker_encodings[s_id][0] for s_id in sp_ids_unique}
            #spk_feats_list = [self.speaker_encodings[s_id][0] if s_id in spk_ids_context else torch.zeros(1, self.hparams.hidden_size) for s_id in spk_ids_mask]
            #spk_feats_embs = torch.stack(spk_feats_list, dim=0)
            spk_feats_embs = torch.zeros((input_ids.shape[0], input_ids.shape[1],self.config.hidden_size)).to(self.model.device)
            for b_idx in range(len(input_ids)):
                sp_ids_unique = set(spk_ids_context[b_idx])

                sp_mask =np.asarray(spk_ids_mask[b_idx])
                try:
                    spk_order = {s_id: spk_ids_order[b_idx].index(s_id) for s_id in sp_ids_unique}
                except:
                    import pdb; pdb.set_trace()
                spk_list_tmp = torch.zeros((self.hparams.context_size+1, self.config.hidden_size)).to(self.model.device)

                for s_id in list(sp_ids_unique)[:len(np.unique(sp_mask))]:
                    tok_posns = np.where(sp_mask == spk_order[s_id])[0]
                    if len(tok_posns) == 0:
                        continue
                    #import pdb; pdb.set_trace()
                    sp_mask_tmp= torch.Tensor(spk_ids_mask[b_idx])
                    key_masks = torch.cat([torch.where(sp_mask_tmp == spk_order[s_id], 1, 0).to(self.model.device), torch.zeros(attention_mask[b_idx].shape[0]-sp_mask_tmp.shape[0]).to(self.model.device)], dim=0)
                    key_padding_masks.append(key_masks)

                    try:
                        #spk_feats_embs[b_idx,tok_posns, :] = self.speaker_encodings[s_id]['pooler_output'].to(self.model.device)
                        if not self.hparams.phrase_embeddings:
                            #speaker_hiddens_tmp = self.speaker_encodings[s_id]['hidden_states'][-1].to(self.model.device)
                            speaker_hiddens_tmp = self.speaker_encodings[s_id]['hidden_states'].unsqueeze(0).to(self.model.device)
                        else:
                            if torch.sum(self.speaker_phrase_embeddings.weight).isnan():
                                import pdb; pdb.set_trace()
                            speaker_hiddens_tmp = self.speaker_phrase_embeddings(self.speaker_encodings[s_id]['input_ids'].to(self.model.device)).unsqueeze(0)

                        #speaker_hiddens[s_id] = (speaker_hiddens_tmp, self.speaker_encodings[s_id]['nt_idx_matrix'])
                        if s_id not in spks_all:
                            speaker_hiddens_all.append(speaker_hiddens_tmp)
                            spk_nts_all.append(self.speaker_encodings[s_id]['nt_idx_matrix'])
                            active_nt_toks.append(self.speaker_encodings[s_id]['active_nt_tokens'])
                            spk_toks.append(self.speaker_encodings[s_id]['input_ids'])
                            spk_attens.append(self.speaker_encodings[s_id]['attention_mask'])
                            spks_all.append(s_id)
                        """
                        "--multihead_pool", action='store_true',
                            help="Whether to pool using attention instead of the seqsummary (self attention w speaker desc")
                        "--multihead_pool_over_input", action='store_true',
                            help="Whether to pool using attention instead of the seqsummary, attention with speaker's words as (key?)")
                        """
                        if self.hparams.multihead_pool or self.hparams.multihead_pool_over_input:


                            spk_reshape = speaker_hiddens_tmp.view(speaker_hiddens_tmp.shape[1], \
                                                                    speaker_hiddens_tmp.shape[0], speaker_hiddens_tmp.shape[2])
                            if self.hparams.multihead_pool_over_input:
                                key = hidden_states[b_idx].unsqueeze(0)
                                key = key.view(key.shape[1], key.shape[0], key.shape[2])
                                mask = (1-key_masks).to(bool).unsqueeze(0)


                                # spk_atten, attn = self.speaker_pool(query=spk_reshape,
                                #                                     key=key, value=key, key_padding_mask = (1-key_masks).to(bool))
                            else:
                                key = spk_reshape
                                mask = torch.zeros((key.shape[1], key.shape[0])).to(bool).to(self.model.device)
                            spk_atten, attn = self.speaker_pool(query=spk_reshape,
                                                         key=key, value=key, key_padding_mask = mask)
                            spk_pool = torch.sum(spk_reshape*spk_atten, dim=0)
                        else:
                            #spk_pool = self.pooler(speaker_hiddens_tmp)
                            spk_pool = self.mean_pooling(speaker_hiddens_tmp, self.speaker_encodings[s_id]['attention_mask'], hidden_states=True)
                        spk_emb = self.dropout(spk_pool)
                        spk_feats_embs[b_idx,tok_posns, :] = spk_emb
                        if torch.sum(spk_emb).isnan():
                            import pdb; pdb.set_trace()
                        spk_list_tmp[np.where(np.asarray(spk_ids_context[b_idx]) == s_id)[0]] = spk_emb
                    except:
                        import pdb; pdb.set_trace()
                    #import pdb; pdb.set_trace()
                    # spk_list_tmp2 = [spk_list_tmp[i] for i in range(len(spk_list_tmp))]
                    # spk_list_tmp2 = torch.cat(spk_list_tmp2, dim=0)
                    # spk_list_tmp3 = spk_list_tmp.reshape(1, -1)
                spk_list_all.append(spk_list_tmp.reshape(1,-1))
            spk_feats_embs = spk_feats_embs.to(self.model.device)
            spk_list_all= torch.cat(spk_list_all, dim=0).to(self.model.device)
            # if len(sp_ids_unique) != len(spks_all):
            #     import pdb; pdb.set_trace()

            try:
                spk_nts_all = torch.stack(spk_nts_all, dim=0)
            except:
                import pdb; pdb.set_trace()
            speaker_hiddens_all = torch.cat(speaker_hiddens_all, dim=0)
            try:
                spk_attens = torch.stack(spk_attens, dim=0)
            except:
                import pdb; pdb.set_trace()
            #word_embeddings = self.model.word_embeddings(input_ids)
            #inputs_embeds = spk_feats_embs + word_embeddings
            try:
                # hidden_states = self.model(inputs_embeds = inputs_embeds, token_type_ids=token_type_ids,
                #                      attention_mask=attention_mask,
                #                      output_hidden_states=True)['hidden_states']


                if not self.hparams.concat_speaker_logits:
                    if not self.hparams.concatenate_speaker:
                        hidden_states = hidden_states + spk_feats_embs
                    else:
                        try:

                            hidden_states = torch.cat([hidden_states, spk_feats_embs], dim=2)
                        except:
                            import pdb; pdb.set_trace()

            except:
                import pdb; pdb.set_trace()







        else:
            outputs = self.model(input_ids=input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)

            hidden_states = outputs["hidden_states"][-1]
        #cls_hidden_state = self.dropout(self.pooler(hidden_states[-1]))
        #cls_hidden_state = self.dropout(outputs['pooler_output'])
        #cls_hidden_state = self.dropout(self.pooler(hidden_states))
        cls_hidden_state = self.dropout(self.mean_pooling(hidden_states, attention_mask=attention_mask, hidden_states=True))
        if  torch.sum(cls_hidden_state).isnan().item():
            import pdb; pdb.set_trace()

        return cls_hidden_state, hidden_states, (speaker_hiddens_all, spk_attens, spk_nts_all, active_nt_toks, spk_toks, spks_all, spk_list_all)

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc, tpr, precision, recall, f1, acc_for_lang, topks, logits_dict = self(batch)
        loss = self.loss(logits, batch[-1])
        tb_log = {'step': self.current_epoch}
        if 'glam' in logits_dict:
            final_loss = 0.6*loss
            for log_tmp, lbls in zip(logits_dict['glam'], logits_dict['glam_labels']):
                loss_tmp = self.loss(log_tmp, lbls)
                final_loss += 0.1*loss_tmp
        else:
            final_loss = loss
        tb_log['train_loss'] = final_loss
        tb_log['train_acc'] = acc
        self.log('train_acc', acc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        if tpr is not None:
            self.log('train_tpr', tpr, on_step=True,
                     on_epoch=True, prog_bar=True, sync_dist=True)
        if precision is not None:
            self.log('train_precision', precision, on_step=True,
                     on_epoch=True, prog_bar=True, sync_dist=True)
        if recall is not None:
            self.log('train_recall', recall, on_step=True,
                     on_epoch=True, prog_bar=True, sync_dist=True)
        if f1 is not None:
            self.log('train_f1',f1, on_step=True,
                     on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', final_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)

        return {"loss": final_loss, "topk": topks, 'logits': logits_dict, 'log': tb_log}


    def validation_step(self, batch, batch_idx):
        # Load the data into variables
        logits, acc, tpr, precision, recall, f1, acc_for_lang, topks, logits_dict  = self(batch)
        if self.hparams.balanced:
            weights = None
        else:
            weights=torch.cuda.FloatTensor([1,6])
        #weights=torch.cuda.FloatTensor([1,4])
        loss_f = nn.CrossEntropyLoss(weight=weights)
        loss = loss_f(logits, batch[-1])
        final_loss = loss
        # if self.hparams.full_mtl_setup:
        #     final_loss = 0.6*loss
        #     for log_tmp, lbls in zip(logits_dict['glam'], logits_dict['glam_labels']):
        #         loss_tmp = self.loss(log_tmp, lbls)
        #         final_loss += 0.1*loss_tmp
        # else:
        #     final_loss = loss
        tb_log = {'step':self.current_epoch}
        self.log('val_loss', final_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        tb_log['val_loss'] = final_loss
        self.log('val_acc', acc, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, logger=True)
        tb_log['val_acc'] = acc
        if tpr is not None:
            self.log('val_tpr', tpr, on_step=True,
                     on_epoch=True, prog_bar=True, sync_dist=True)
        if precision is not None:
            self.log('val_precision', precision, on_step=True,
                     on_epoch=True, prog_bar=True, sync_dist=True)
        if recall is not None:
            self.log('val_recall', recall, on_step=True,
                     on_epoch=True, prog_bar=True, sync_dist=True)
        if f1 is not None:
            self.log('val_f1',f1, on_step=True,
                     on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "topk": topks, 'logits': logits_dict, "acc": acc, "f1": f1, 'log': tb_log}
    def validation_epoch_end(self, val_step_outputs):
        loss_all = torch.zeros(len(val_step_outputs))
        acc_all= torch.zeros(len(val_step_outputs))
        for idx, res_dict in enumerate(val_step_outputs):
            loss_all[idx] = res_dict['loss']
            acc_all[idx] = res_dict['acc']
        loss_mean = torch.mean(loss_all)
        acc_mean = torch.mean(acc_all)
        self.log('val_loss_mean', loss_mean, logger=True)
        self.log('val_acc_mean', acc_mean, logger=True)
        self.logger.experiment.add_scalar("Loss/Val", loss_mean)
        self.logger.experiment.add_scalar("Acc/Val", acc_mean)




    def test_step(self, batch, batch_idx):
        # Load the data into variables
        bal_batch = batch["balanced"]
        unbal_batch = batch["unbalanced"]


        results = {}
        for batch, name in zip([bal_batch, unbal_batch], ['balanced', 'unbalanced']):
            logits, acc, tpr, precision, recall, f1, acc_for_lang, topks, logits_dict = self(batch)
            tmp = {}
            #if self.hparams.balanced:
            if name == 'balanced':
                weights = None
            else:
                weights= torch.cuda.FloatTensor([1,6])
            #weights=torch.cuda.FloatTensor([1,4])
            loss_f = nn.CrossEntropyLoss(weight=weights)
            loss = loss_f(logits, batch[-1])
            # if self.hparams.full_mtl_setup:
            #     final_loss = 0.6*loss
            #     for log_tmp, lbls in zip(logits_dict['glam'], logits_dict['glam_labels']):
            #         loss_tmp = self.loss(log_tmp, lbls)
            #         final_loss += 0.1*loss_tmp
            # else:
            final_loss = loss
            # if acc_non_cs:
            #     tmp['acc-monolingual'] = acc_non_cs.item()
            # if acc_cs:
            #     tmp['acc-cs'] = acc_cs.item()
            tmp['loss'] = final_loss.item()
            tmp['accuracy'] = acc.item()
            tmp['tpr'] = tpr
            tmp['precision'] = precision
            tmp['recall'] = recall
            tmp['f1'] = f1
            if acc_for_lang is not None:
                tmp['accuracy_spa']  = acc_for_lang['spa']
                tmp['accuracy_eng'] = acc_for_lang['eng']
            tmp['topk']= topks
            tmp['logits']= logits_dict
            #tmp= {"loss": loss, "accuracy": acc, "acc-cs": acc_cs, "acc-monolingual": acc_non_cs}
            results[name] = tmp
        return results
    def test_epoch_end(self, test_step_outputs):
        results = {'balanced': {'accuracy': [], 'loss': [], 'tpr': [], 'precision': [], 'recall': [], 'f1': []}, \
                   'unbalanced': {'accuracy': [], 'loss': [], 'tpr': [], 'precision': [], 'recall': [], 'f1': []},}
        results_list= ['tpr', 'precision', 'recall', 'f1']
        # topk_names = ['topk_gil', 'topk_utt', 'topk_spk']
        # topks_all = defaultdict(list)
        if self.hparams.eval_per_language:
            for res_name, res_dict in results.items():
                res_dict['accuracy_spa'] = []
                res_dict['accuracy_eng'] = []
                results[res_name] = res_dict
            results_list.extend(['accuracy_spa', 'accuracy_eng'])
        topks_all = defaultdict(list)
        for out in test_step_outputs:
            for name in ['balanced', 'unbalanced']:
                results_for_set=out[name]
                results[name]['loss'].append(results_for_set['loss'])
                results[name]['accuracy'].append(results_for_set['accuracy'])
                if name == 'balanced':
                    topk_concepts = results_for_set['topk']
                    for key, val in topk_concepts.items():
                        topks_all[key]=val

                for res in results_list:
                    if res in results_for_set:
                        if results_for_set[res] is not None:
                            results[name][res].append(results_for_set[res].item())
        if len(topks_all.keys())> 0:
            if not os.path.exists('topks'):
                os.mkdir('topks')
            with open(os.path.join('topks', self.hparams.topk_output_file), 'wb') as f:
                pkl.dump(topks_all, f)


        final = {}
        for set_name, res_dict in results.items():
            tmp = {}
            for metric, list_ in res_dict.items():
                if len(list_) > 0:
                    try:
                        list_ = torch.tensor(list_)

                        self.log('{} - {}, avg'.format(set_name, metric), value=list_.mean().item(), on_epoch=True)
                        self.log('{} - {}, st. dev'.format(set_name, metric), value=list_.std().item(), on_epoch=True)
                        tmp[metric] = ("mean: {}".format(list_.mean().item()), "median: {}".format(list_.median().item()), \
                                       "total: {}".format(len(list_)), "standard deviation: {}".format(list_.std().item()))
                    except:
                        import pdb; pdb.set_trace()
                # else:
                #     print("wtf! no metric {} in {}".format(metric, set_name))
            final[set_name] = tmp
        print(final)

        #return final
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        tqdm_dict.pop("val_loss_step", None)
        tqdm_dict.pop("val_acc_step", None)
        tqdm_dict.pop('val_non_cs_acc_step', None)
        tqdm_dict.pop('val_cs_acc_step', None)
        return tqdm_dict


if __name__ == "__main__":
    sentences = ['This framework generates embeddings for each input sentence',
                 'Sentences are passed as a list of string.',
                 'The quick brown fox jumps over the lazy dog.']
