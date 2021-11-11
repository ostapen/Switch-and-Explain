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
from .BatchedHardTripletLoss import BatchHardTripletLoss
from .BatchSemiHardTripletLoss import BatchSemiHardTripletLoss


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x
class SwitchLMSpeaker(LightningModule):
    def __init__(self, hparams, tuning_parameters=None):

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
            hparams.ensemble_speaker_utt = tuning_parameters['ensemble_speaker_utt']
            hparams.ensemble_speaker = tuning_parameters['ensemble_speaker']
            hparams.ensemble_utterance = tuning_parameters['ensemble_utterance']
            hparams.lamda = tuning_parameters['lamda']
            hparams.lamda_spk = tuning_parameters['lamda_spk']

        #self.model = AutoModel.from_pretrained('xlm-roberta-base')
        #import pdb; pdb.set_trace()
        #adapter = self.model.load_adapter(self.hparams.adapter_path, config="pfeiffer+inv")
        #self.model.set_active_adapters(adapter)







        #self.hparams.update(hparams)
        #print(self.hparams)
        self.save_hyperparameters(hparams)
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(self.hparams.model_name)
        self.model = AutoModel.from_pretrained(self.hparams.model_name)
        # = False
        # idx 0 is reserved for padding / separators
        # only consider spk + gp other speakers into 'partners' - ie consider only 2 indices
        #self.speaker_encodings = nn.Embedding(self.hparams.speaker_emb_dim, config.hidden_size, padding_idx=0)

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







        in_d = num_classifier_feats
        if any([jj in self.hparams for jj in ['speaker_trait_predictions_utt','speaker_trait_predictions_spk_utt','speaker_trait_predictions_spk']]):
            if any([self.hparams.speaker_trait_predictions_utt,self.hparams.speaker_trait_predictions_spk_utt, self.hparams.speaker_trait_predictions_spk]):
                self.g_classifier = nn.Linear(in_d, 2)
                self.l_classifier = nn.Linear(in_d, 2)
                self.a_classifier = nn.Linear(in_d, 2)
                self.m_classifier = nn.Linear(in_d, 2)
        self.mlm = False
        if any([jj in self.hparams for jj in ['age_mlm', 'gender_mlm', 'language_mlm', 'mixing_mlm']]):
            if any([self.hparams.age_mlm, self.hparams.gender_mlm, self.hparams.language_mlm, self.hparams.mixing_mlm]):
                self.mlm= True
                self.lm_head = RobertaLMHead(self.config)
                self.lm_loss = nn.CrossEntropyLoss()

        logits_preds = 0
        if self.hparams.ensemble_speaker_utt:
            self.hparams.ensemble_speaker = True
            self.hparams.ensemble_utterance = True

        if not self.hparams.ensemble_speaker_utt:
            self.classifier = nn.Linear(in_d, self.hparams.num_classes)
            logits_preds += 2
        if self.hparams.ensemble_speaker or self.hparams.ensemble_utterance:
            if self.hparams.ensemble_speaker:
                logits_preds += 2

                self.speaker_linear = nn.Linear(in_d, self.hparams.num_classes)
            if self.hparams.ensemble_utterance:
                logits_preds += 2

                self.utterance_linear = nn.Linear(in_d, self.hparams.num_classes)
            self.ensemble_classifier = nn.Linear(logits_preds, self.hparams.num_classes)
        if self.hparams.mlm_pretrain:
            self.hparams.lm_loss = 1
            self.hparams.cs_loss = 0
        if self.hparams.alternate_tasks_n_epochs > 0:
            self.hparams.cs_loss = 1
            self.hparams.lm_loss = 0
            self.epochs_to_go = self.hparams.alternate_tasks_n_epochs




            # map to same embed space to simply add together








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
            print(weights)
        #weights=torch.cuda.FloatTensor([1,4])
        self.loss = nn.CrossEntropyLoss(weight=weights)



    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--balanced", action='store_true',
                            help="use balanced datasets")

        parser.add_argument("--gender", action='store_true',
                            help="use balanced datasets")
        parser.add_argument("--age", action='store_true',
                            help="use balanced datasets")
        parser.add_argument("--order", action='store_true',
                            help="use balanced datasets")
        parser.add_argument("--country", action='store_true',
                            help="use balanced datasets")
        parser.add_argument("--language", action='store_true',
                            help="use balanced datasets")
        parser.add_argument("--mixing", action='store_true',
                            help="use balanced datasets")
        parser.add_argument("--leave_one_out", action='store_true',
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

        parser.add_argument("--speaker_trait_predictions_utt", action='store_true',
                            help="Save model according to val_f1")
        parser.add_argument("--speaker_trait_predictions_spk_utt", action='store_true',
                            help="Save model according to val_f1")
        parser.add_argument("--speaker_trait_predictions_spk", action='store_true',
                            help="Save model according to val_f1")

        parser.add_argument("--only_lil", action='store_true',
                            help="Don't use concept store, only lil layer")
        parser.add_argument("--only_gil", action='store_true',
                           help="use concept store, but not lil layer")
        parser.add_argument("--eval_per_language", action='store_true',
                            help="add evaluation for english and spanish separately")



        parser.add_argument("--concept_store", type=str,default = '/projects/tir3/users/aostapen/CodeSwitch/SelfExplain/bangor_data/feature_spk2/concept_store.pt',
                            help="path to ngram concept store")

        parser.add_argument("--ensemble_utterance", action='store_true',
                            help="Ensemble predictions from full sentence, utterance logits, and from speaker logits")
        parser.add_argument("--ensemble_speaker_utt", action='store_true',
                            help="Ensemble predictions from utterance logits and from speaker logits, without using full sentence")


        parser.add_argument("--ensemble_speaker", action='store_true',
                            help="Ensemble predictions from full sentence and from speaker logits.")
        parser.add_argument("--multihead_pool", action='store_true',
                            help="Whether to pool using attention instead of the seqsummary (self attention w speaker desc")
        parser.add_argument("--multihead_pool_over_input", action='store_true',
                            help="Whether to pool using attention instead of the seqsummary, attention with speaker's words as (key?)")

        parser.add_argument("--mlm_pretrain", action='store_true',
                            help="pretrain mlm first")
        parser.add_argument("--cs_tune", action='store_true',
                            help="tune pretrained mlm model to the code switch prediction task")
        parser.add_argument("--alternate_tasks_n_epochs", type=int,
                            help="How many epochs to train each task", default=0)





        parser.add_argument("--adapter_path", type=str,default = '/projects/tir3/users/aostapen/CodeSwitch/SelfExplain/pretrained/adapter-raw-desc/mlm',
                            help="path to pretrained adapter for bangor text")
        parser.add_argument("--adapter_path_speaker", type=str,default = '/projects/tir3/users/aostapen/CodeSwitch/SelfExplain/pretrained/adapter-raw-speaker/checkpoint-8500/mlm',
                            help="path to pretrained adapter for bangor text")


        parser.add_argument("--list", action='store_true',
                            help="use list style descriptions")
        parser.add_argument("--sentence", action='store_true',
                            help="use sentence style descriptions")
        parser.add_argument("--partner", action='store_true',
                            help="use sentence style descriptions")

        parser.add_argument("--triplet_loss", action='store_true',
                            help="first tune model with triplet loss")
        parser.add_argument("--contrastive_loss", action='store_true',
                            help="first tune model with contrastive loss")
        parser.add_argument("--finetune_bangor", action='store_true',
                            help="finetune to bangor after learning representations")

        parser.add_argument("--full_mtl_setup", action='store_true',
                            help="use mtl loss at val+test time")


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
        parser.add_argument("--lr", default=5e-5, type=float,
                            help="Initial learning rate.")
        parser.add_argument("--weight_decay", default=0.001, type=float,
                            help="Weight decay rate.")
        parser.add_argument("--warmup_prop", default=0.1, type=float,
                            help="Warmup proportion.")
        parser.add_argument("--topk", default=10, type=int,
                            help="Topk GIL concepts")
        parser.add_argument("--lamda", default=0.01, type=float,
                            help="Lamda Parameter")
        parser.add_argument("--lm_loss", default=0.5, type=float,
                            help="Lamda Parameter")
        parser.add_argument("--cs_loss", default=0.5, type=float,
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


    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias
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
        if self.mlm:
            mode = 'min'
            monitor = 'val_loss'

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
        #if not self.hparams.triplet_loss or (self.hparams.triplet_loss and 'sentence-transformers' in self.hparams.model_name):
        tokens, tokens_mask,  (padded_ndx_tensor,active_nt_tokens), speaker_orders, speaker_context_ids, spa2eng, eng2spa, utt_masks, speaker_masks, speaker_utt_masks, genders, ages, mixes, langs, labels = batch
        # step 1: encode the sentence
        if self.mlm:
            (mlm_tokens, tokens) = tokens
            (mlm_tokens_mask, tokens_mask) = tokens_mask

        padded_ndx_tensor_spk, padded_ndx_tensor_utt = None, None
        active_nt_tokens_spk, active_nt_tokens_utt=None, None
        if not self.hparams.mlm_pretrain and type(padded_ndx_tensor) == tuple:
            padded_ndx_tensor_spk, padded_ndx_tensor_utt = padded_ndx_tensor

            active_nt_tokens_spk, active_nt_tokens_utt = active_nt_tokens
        # else:
        #     tokens, tokens_mask, labels = batch
        #     padded_ndx_tensor_spk = None
        #     padded_ndx_tensor_utt = None
        sentence_cls, spk_only_cls, utt_only_cls, hidden_state, (attentions_spk, attentions_utt) = self.forward_classifier(input_ids=tokens,attention_mask=tokens_mask,token_type_ids=None, \
                                                                                                                           speaker_nts=padded_ndx_tensor_spk, utterance_nts=padded_ndx_tensor_utt)

        #if self.hparams.use_cs_feats:

        logits_list = []
        if not self.hparams.ensemble_speaker_utt and not self.hparams.mlm_pretrain:

            logits_full = self.classifier(sentence_cls)
            logits_list.append(logits_full)
        logits_dict = {}
        if self.mlm:
            logits_dict['mlm_input'] = (tokens, tokens_mask)
            logits_dict['mlm_labels'] = mlm_tokens
            lm_head_output = self.lm_head(hidden_state)
            logits_dict['lm_head_output'] = lm_head_output
            logits_full = lm_head_output
            logits = lm_head_output
        if not self.hparams.mlm_pretrain:
            if not self.hparams.ensemble_speaker and not self.hparams.ensemble_utterance:
                logits=logits_full
            else:
                if self.hparams.ensemble_speaker:
                    logits_spk = self.speaker_linear(spk_only_cls)
                    logits_list.append(logits_spk)
                    logits_dict['logits_spk']= logits_spk
                if self.hparams.ensemble_utterance:
                    logits_utt = self.utterance_linear(utt_only_cls)
                    logits_list.append(logits_utt)
                    logits_dict['logits_utt']= logits_utt
                logits_cat = torch.cat(logits_list, dim=1)
                logits=self.ensemble_classifier(logits_cat)
            #import pdb; pdb.set_trace()
            logits_dict['logits'] = logits
            if any([jj in self.hparams for jj in ['speaker_trait_predictions_utt','speaker_trait_predictions_spk_utt','speaker_trait_predictions_spk']]):
                if any([self.hparams.speaker_trait_predictions_utt, self.hparams.speaker_trait_predictions_spk, self.hparams.speaker_trait_predictions_spk_utt]) and utt_masks is not None:
                    logits_dict['glam'] = []
                    logits_dict['glam_labels'] = [genders, langs, ages, mixes]
                    if self.hparams.speaker_trait_predictions_utt:
                        attn_mask = utt_masks
                    elif self.hparams.speaker_trait_predictions_spk:
                        attn_mask = speaker_masks
                    else:
                        attn_mask = speaker_utt_masks
                    pooled_seq_rep = self.mean_pooling(hidden_state, attn_mask, hidden_states=True)
                    for classifier in [self.g_classifier, self.l_classifier, self.a_classifier, self.m_classifier]:
                        logits_ = classifier(pooled_seq_rep)
                        logits_dict['glam'].append(logits_)
        if logits.sum().isnan().item():
            import pdb; pdb.set_trace()
        lil_logits = None
        lil_logits_mean = None
        gil_logits, topk_indices = None, None
        acc_for_lang = None
        tpr, precision, recall, f1 = None, None, None, None

        topks = {}
        if self.hparams.self_explain_ngram and not self.hparams.mlm_pretrain:

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
                                          nt_idx_matrix=padded_ndx_tensor_utt, attention_mask=attentions_utt)
                except:
                    import pdb; pdb.set_trace()

                lil_logits_tmp = [lil_logits[b][active_nt_tokens_utt[0][b], :] for b in range(len(tokens))]
                lil_logits_mean= torch.stack([torch.mean(logits_, dim=0) for logits_ in lil_logits_tmp], dim=0)

                logits_dict['lil_logits_utt'] = lil_logits
                logits_dict['lil_logits_utt_active_ndxs'] = active_nt_tokens_utt
                topks['topk_utt'] = topk_utt


                #lil_logits_mean = torch.mean(lil_logits, dim=1)
                lil_logits_sp, topk_spk = self.lil(hidden_state=hidden_state,
                                             nt_idx_matrix=padded_ndx_tensor_spk, attention_mask = attentions_spk, speaker=True)
                lil_logits_tmp_sp = [lil_logits_sp[b][active_nt_tokens_spk[0][b], :] for b in range(len(padded_ndx_tensor_spk))]
                logits_dict['lil_logits_spk_active_idxs'] = active_nt_tokens_spk
                #TODO
                lil_logits_spk_mean= torch.stack([torch.mean(logits_, dim=0) for logits_ in lil_logits_tmp_sp], dim=0)

                logits_dict['lil_logits_spk'] = lil_logits_sp
                logits_dict['speaker_tokens'] = tokens




                #lil_logits_spk_mean = torch.stack(lil_logits_spk_mean, dim=0)

                logits_dict['nt_speaker_all'] = padded_ndx_tensor_spk
                topks['topk_spk'] = topk_spk



            logits_full = logits + self.lamda * lil_logits_mean + self.lamda_spk*lil_logits_spk_mean
            #self.gamma * gil_logits -- ignore the gil logits
            # if self.hparams.add_spk_logits:
            #     logits_full = logits_full + self.lamda_spk * lil_logits_spk_mean
        else:
            logits_full = logits
        # if self.hparams.threshold != 0.5:
        #     predicted_labels = torch.where(torch.nn.functional.softmax(logits_full, dim=1)[:, 1] >=self.hparams.threshold, 1, 0)
        # else:

        if labels is not None:
            predicted_labels = torch.argmax(logits_full, -1)
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
        #elif 'sentence-transformer' in self.hparams.model_name:

        else:

            acc, tpr, precision, recall, f1= None, None, None, None, None
            acc_for_lang, topks= None, None
        logits_dict['labels'] = labels
            #logits_full = logits_list
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


    def forward_classifier(self, input_ids: torch.Tensor,  attention_mask: torch.Tensor, speaker_nts: torch.Tensor, utterance_nts: torch.Tensor, token_type_ids: torch.Tensor = None):
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
        #if not self.hparams.triplet_loss or (self.hparams.triplet_loss and 'sentence-transformer' in self.hparams.model_name):
        attentions_spk, attentions_utt = None, None
        if speaker_nts is not None:
            attentions_spk= torch.clip(speaker_nts.sum(dim=1), min=0, max=1)
            attentions_utt= torch.clip(utterance_nts.sum(dim=1), min=0, max=1)
        #attentions2 = torch.LongTensor(np.clip(nt_idx_matrix[1].sum(), [0,1]))


        if type(attention_mask) == tuple:
                attention_mask = attention_mask[0]
        outputs = self.model(input_ids=input_ids,
                                 token_type_ids=token_type_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)

        hidden_states = outputs["hidden_states"][-1]
        #cls_hidden_state = self.dropout(self.pooler(hidden_states[-1]))
        #cls_hidden_state = self.dropout(outputs['pooler_output'])
        #cls_hidden_state = self.dropout(self.pooler(hidden_states))
        cls_hidden_state = self.dropout(self.mean_pooling(hidden_states, attention_mask=attention_mask, hidden_states=True))
        spk_only_cls, utt_only_cls = None, None
        if speaker_nts is not None:
            spk_only_cls = self.dropout(self.mean_pooling(hidden_states, attention_mask=attentions_spk, hidden_states=True))
            utt_only_cls= self.dropout(self.mean_pooling(hidden_states, attention_mask=attentions_utt, hidden_states=True))
        if  torch.sum(cls_hidden_state).isnan().item():
            import pdb; pdb.set_trace()
        # else:
        #     cls_hidden_state = []
        #     spk_only_cls = None
        #     utt_only_cls = None
        #     hidden_states = []
        #     attentions_spk = None
        #     attentions_utt = None
        #     for i in range(3):
        #         attn =attention_mask[:, i, :]
        #         outputs = self.model(input_ids=input_ids[:, i, :],
        #                              token_type_ids=token_type_ids,
        #                              attention_mask=attn,
        #                              output_hidden_states=True)
        #         hs = outputs["hidden_states"][-1]
        #         hidden_states.append(hs)
        #         cls_hs = self.dropout(self.mean_pooling(hs, attention_mask=attn, hidden_states=True))
        #         cls_hidden_state.append(cls_hs)

        return cls_hidden_state, spk_only_cls, utt_only_cls, hidden_states, (attentions_spk, attentions_utt)

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        # if self.testing:
        #      self.testing=False
        tb_log = {}
        logits, acc, tpr, precision, recall, f1, acc_for_lang, topks, logits_dict = self(batch)

        if not self.hparams.triplet_loss:
            if not self.hparams.mlm_pretrain:
                loss = self.loss(logits, batch[-1])
                tb_log['cs_loss'] = loss
                self.log('train_cs_loss',loss, on_step=True, on_epoch=True, logger=True)
            # if 'glam' in logits_dict:
            #     final_loss = 0.6*loss
            #     for log_tmp, lbls in zip(logits_dict['glam'], logits_dict['glam_labels']):
            #         loss_tmp = self.loss(log_tmp, lbls)
            #         final_loss += 0.1*loss_tmp
            if self.mlm and self.hparams.lm_loss > 0:
                (tokens, tokens_mask) = logits_dict['mlm_input']
                mlm_tokens = logits_dict['mlm_labels']
                predictions= logits_dict['lm_head_output']
                lm_loss = self.lm_loss(predictions.view(-1,self.config.vocab_size),mlm_tokens.view(-1))
                tb_log['mlm_loss'] = lm_loss
                self.log('train_mlm_loss', lm_loss, on_step=True, on_epoch=True, logger=True)
                final_loss = self.hparams.lm_loss*lm_loss
                if not self.hparams.mlm_pretrain:
                    final_loss = final_loss + self.hparams.cs_loss*loss
                #loss_fct(prediction_scores.view(-1, self.config.vocab_size)
                #logits_dict['mlm_input'] = (tokens, tokens_mask)
                #logits_dict['mlm_labels'] = mlm_tokens
            else:
                final_loss = loss
        #elif 'sentence-transformers' not in self.hparams.model_name:
        #    loss = self.loss(logits[0], logits[1], logits[2])
        else:

            final_loss= self.loss(logits, logits_dict['labels'])
            tb_log['triplet_loss'] = final_loss
        self.log('train_loss', final_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if acc is not None:
            tb_log['accuracy'] = acc
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

        if final_loss.isnan():
            import pdb; pdb.set_trace()
        tb_log['step'] = self.current_epoch
        return {"loss": final_loss, "topk": topks, 'logits': logits_dict, 'log': tb_log, "acc": acc}
    def training_epoch_end(self, outputs) -> None:
        if self.hparams.alternate_tasks_n_epochs > 0:
            if self.epochs_to_go > 0:
                self.epochs_to_go -=1
            else:
                if self.hparams.cs_loss == 0:
                    self.hparams.cs_loss = 1
                    self.hparams.lm_loss = 0
                else:
                    self.hparams.lm_loss = 1
                    self.hparams.cs_loss = 0
                self.epochs_to_go = self.hparams.alternate_tasks_n_epochs
        loss_all = torch.zeros(len(outputs))
        loss_cs = torch.zeros(len(outputs))
        loss_lm = torch.zeros(len(outputs))

        acc_all= torch.zeros(len(outputs))
        for idx, res_dict in enumerate(outputs):
            loss_all[idx] = res_dict['loss']
            if res_dict['acc'] is not None:
                acc_all[idx] = res_dict['acc']
            if 'mlm_loss' in res_dict:
                loss_lm[idx] = res_dict['mlm_loss']
            if 'cs_loss' in res_dict:
                loss_cs[idx] = res_dict['cs_loss']
        loss_mean = torch.mean(loss_all)
        tb_log = {'train_loss_mean':loss_mean}
        if 'mlm_loss' in res_dict:
            mean_lm_loss = torch.mean(loss_lm)
            self.log('train_lm_loss_mean', mean_lm_loss, logger=True, on_epoch=True)
        if 'cs_loss' in res_dict:
            mean_cs_loss = torch.mean(loss_cs)
            self.log('train_cs_loss_mean', mean_cs_loss, logger=True, on_epoch=True)
        if not self.hparams.triplet_loss and not self.hparams.mlm_pretrain:

            acc_mean = torch.mean(acc_all)
            self.log('train_acc_mean', acc_mean, logger=True, on_epoch=True)
            tb_log['train_acc_mean'] = acc_mean
            self.logger.experiment.add_scalar("Acc/Train", acc_mean)
        self.logger.experiment.add_scalar("Loss/Train", loss_mean)

        self.log('train_loss_mean', loss_mean, logger=True, on_epoch=True)
        #return {'loss': loss_mean, 'log':tb_log}


    def validation_step(self, batch, batch_idx):
        # Load the data into variables
        tb_log = {}
        logits, acc, tpr, precision, recall, f1, acc_for_lang, topks, logits_dict  = self(batch)
        if self.hparams.balanced:
            weights = None
        else:
            weights=torch.cuda.FloatTensor([1,4])
        #weights=torch.cuda.FloatTensor([1,4])
        if not self.hparams.triplet_loss:
            if not self.hparams.mlm_pretrain:
                loss_f = nn.CrossEntropyLoss(weight=weights)

                loss = loss_f(logits, batch[-1])
                tb_log['cs_loss'] = loss
                self.log('val_cs_loss',loss, on_step=True, on_epoch=True, logger=True)
            # if 'glam' in logits_dict:
            #     final_loss = 0.6*loss
            #     for log_tmp, lbls in zip(logits_dict['glam'], logits_dict['glam_labels']):
            #         loss_tmp = self.loss(log_tmp, lbls)
            #         final_loss += 0.1*loss_tmp
            if self.mlm and self.hparams.lm_loss > 0:

                loss_f2 = nn.CrossEntropyLoss()
                (tokens, tokens_mask) = logits_dict['mlm_input']
                mlm_tokens = logits_dict['mlm_labels']
                predictions= logits_dict['lm_head_output']
                lm_loss = loss_f2(predictions.view(-1,self.config.vocab_size),mlm_tokens.view(-1))
                tb_log['mlm_loss'] = lm_loss
                self.log('val_mlm_loss', lm_loss, on_step=True, on_epoch=True, logger=True)
                final_loss = self.hparams.lm_loss*lm_loss
                if not self.hparams.mlm_pretrain:
                    final_loss = final_loss + self.hparams.cs_loss*loss
            else:
                final_loss=loss

        #elif 'sentence-transformers' in self.hparams.model_name:
        else:
            loss_f = BatchSemiHardTripletLoss(self.model)
            final_loss = loss_f(logits, logits_dict['labels'])
            tb_log['triplet_loss'] = final_loss
        #else:
        #    loss_f = nn.TripletMarginLoss()
        #    loss = loss_f(logits[0], logits[1], logits[2])
        self.log('val_loss', final_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        if acc is not None:
            self.log('val_acc', acc, on_step=True, on_epoch=True,
                     prog_bar=True, sync_dist=True, logger=True)
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
        tb_log['val_acc'] = acc
        tb_log['self'] = self.current_epoch
        if final_loss.isnan():
            import pdb; pdb.set_trace()
        return {"loss": final_loss, "topk": topks, 'logits': logits_dict, "acc": acc, "f1": f1, "log": tb_log}
    def validation_epoch_end(self, val_step_outputs):
        loss_all = torch.zeros(len(val_step_outputs))
        loss_lm = torch.zeros(len(val_step_outputs))
        loss_cs = torch.zeros(len(val_step_outputs))
        acc_all= torch.zeros(len(val_step_outputs))
        for idx, res_dict in enumerate(val_step_outputs):
            loss_all[idx] = res_dict['loss']
            if res_dict['acc'] is not None:
                acc_all[idx] = res_dict['acc']
            if 'mlm_loss' in res_dict:
                loss_lm[idx] = res_dict['mlm_loss']
            if 'cs_loss' in res_dict:
                loss_cs[idx] = res_dict['cs_loss']
        if 'mlm_loss' in res_dict:
            mean_lm_loss = torch.mean(loss_lm)
            self.log('val_lm_loss_mean', mean_lm_loss, logger=True, on_epoch=True)
        if 'cs_loss' in res_dict:
            mean_cs_loss = torch.mean(loss_cs)
            self.log('val_cs_loss_mean', mean_cs_loss, logger=True, on_epoch=True)
        loss_mean = torch.mean(loss_all)
        tb_log = {'val_loss_mean':loss_mean}
        if not self.hparams.triplet_loss and not self.hparams.mlm_pretrain:
            acc_mean = torch.mean(acc_all)
            self.log('val_acc_mean', acc_mean, logger=True, on_epoch=True)
            tb_log['val_acc_mean'] = acc_mean
            self.logger.experiment.add_scalar("Acc/Val", acc_mean)
        self.logger.experiment.add_scalar("Loss/Val", loss_mean)

        self.log('val_loss_mean', loss_mean, logger=True, on_epoch=True)




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
                weights= torch.cuda.FloatTensor([1,4])
            #weights=torch.cuda.FloatTensor([1,4])
            if not self.hparams.triplet_loss:
                if not self.hparams.mlm_pretrain:
                    loss_f = nn.CrossEntropyLoss(weight=weights)

                    loss = loss_f(logits, batch[-1])
                    tmp['cs_loss'] = loss.item()
                # if 'glam' in logits_dict:
                #     final_loss = 0.6*loss
                #     for log_tmp, lbls in zip(logits_dict['glam'], logits_dict['glam_labels']):
                #         loss_tmp = self.loss(log_tmp, lbls)
                #         final_loss += 0.1*loss_tmp
                if self.mlm and self.hparams.lm_loss > 0:
                    loss_f2 = nn.CrossEntropyLoss()
                    (tokens, tokens_mask) = logits_dict['mlm_input']
                    mlm_tokens = logits_dict['mlm_labels']
                    predictions= logits_dict['lm_head_output']
                    lm_loss = loss_f2(predictions.view(-1,self.config.vocab_size),mlm_tokens.view(-1))
                    tmp['cs_loss'] = lm_loss.item()
                    final_loss =self.hparams.lm_loss*lm_loss
                    if not self.hparams.mlm_pretrain:
                        final_loss = final_loss + self.hparams.cs_loss*loss
                else:
                    final_loss = loss

            #elif 'sentence-transformers' in self.hparams.model_name:
            else:
                loss_f = BatchSemiHardTripletLoss(self.model)
                final_loss = loss_f(logits, logits_dict['labels'])
            if final_loss.isnan():
                import pdb; pdb.set_trace()
            tmp['loss'] = final_loss.item()

            if not self.hparams.mlm_pretrain:
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
        results = {'balanced': {'accuracy': [], 'loss': [], 'tpr': [], 'cs_loss': [], 'mlm_loss': [], 'precision': [], 'recall': [], 'f1': []}, \
                   'unbalanced': {'accuracy': [], 'loss': [], 'cs_loss': [], 'mlm_loss': [], 'tpr': [], 'precision': [], 'recall': [], 'f1': []},}
        results_list= ['tpr', 'precision', 'recall', 'f1']
        # topk_names = ['topk_gil', 'topk_utt', 'topk_spk']
        # topks_all = defaultdict(list)
        if self.hparams.eval_per_language and not self.hparams.mlm_pretrain:
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
                if 'cs_loss' in results_for_set:
                    results[name]['cs_loss'].append(results_for_set['cs_loss'])
                if 'mlm_loss' in results_for_set:
                    results[name]['mlm_loss'].append(results_for_set['mlm_loss'])
                if not self.hparams.mlm_pretrain:
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
