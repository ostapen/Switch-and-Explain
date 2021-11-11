from operator import itemgetter

import torch
import numpy as np
from pytorch_lightning import Trainer
from tqdm import tqdm
import pandas as pd
import resource
import random

from argparse import ArgumentParser
import pytorch_lightning as pl

from transformers import AutoTokenizer
from pytorch_lightning.core.lightning import LightningModule

import pickle as pkl
from torch.nn import CrossEntropyLoss
from model.data import ClassificationData
from model.data_mlm import ClassificationDataMLM
from model.SwitchLM import SwitchLM
from model.SwitchLMSpeaker import SwitchLMSpeaker
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import json
DEVICE='cpu'
if torch.cuda.is_available():
    DEVICE='cuda'
OVERLAPPING=True
PHR2IX = None
IX2PHR = None
GLOBAL_PHR_EMBS = None

class SwitchLMForEval(LightningModule):
    def __init__(self, ckpt, control=False, threshold=-1, do_only_eval=False, prepend_description=False, seed=18, random=False):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.use_phrase_embs = False
        self.use_speaker_toks = False
        self.use_speaker_descriptions = False
        self.load_sent = False
        self.load_partner = False
        self.load_list = False
        self.random=random
        self.accelerator = None
        if not prepend_description:

            print("using SwitchLM basic")
            model = SwitchLM.load_from_checkpoint(ckpt)
            self.use_phrase_embs = model.hparams.phrase_embeddings
            self.use_speaker_toks = model.hparams.use_speaker_tokens
            self.use_speaker_descriptions = model.hparams.use_speaker_descriptions
            if not self.use_speaker_toks:
                new_toks = ['[EOU]', '[EOT]']
            else:
                new_toks = ['[SPK{}]'.format(idx+1) for idx in range(5)]
            tokenizer.add_tokens(new_toks)
        else:
            model = SwitchLMSpeaker.load_from_checkpoint(ckpt)
            self.load_list = model.hparams.list
            self.load_sent = model.hparams.sentence
            self.load_partner = model.hparams.partner


            self.ensemble_speaker_utt = model.hparams.ensemble_speaker_utt
            self.ensemble_utterance = model.hparams.ensemble_utterance
            self.ensemble_speaker = model.hparams.ensemble_speaker
        self.testing_file = False
        self.do_only_eval = do_only_eval
        self.seed = seed
        self.control = control
        self.threshold=threshold
        if threshold != -1:
            print("Using threshold, {}".format(threshold))

        if control:

            print("control")
            if not prepend_description:
                model.load_control_embeddings()
            self.testing_file=True

        model.eval()
        self.model = model.to(DEVICE)
        self.pair_ids = None
        self.prepend_description=prepend_description


        self.context_size = model.hparams.context_size



        self.tokenizer=tokenizer
    def set_pair_ids(self, pair_ids):
        self.pair_ids = pair_ids
    def map_to_words(self, toks):
        try:
            words= [wd for wd in self.tokenizer.convert_ids_to_tokens(toks) if wd!= "<pad>"]
            words= self.tokenizer.convert_tokens_to_string(words)
        except:
            import pdb; pdb.set_trace()

        #words = ' '.join(words)
        words = words.replace('</s>', '')
        words = words.replace('<s>', '')
        return words
    def forward(self, batch):

        return self.model(batch)
    def test_step(self, batch, batch_idx):
        if len(batch)==2:
            batch = batch['unbalanced']
        #import pdb; pdb.set_trace()

        tokens, tokens_mask,  (padded_ndx_tensor,active_nt_tokens), speaker_orders, speaker_context_ids, spa2eng, eng2spa, utt_masks, speaker_masks, speaker_utt_masks, genders, ages, mixes, langs, labels = batch

        logits, acc, tpr, precision, recall, f1, acc_for_lang, topks, logits_dict = self(batch)

        weights = None
        loss_f = CrossEntropyLoss(weight=weights)
        def get_out_info(logits):

            probs = torch.nn.functional.softmax(logits, dim=1).cpu().data.numpy()[:, 1]

            return probs
        if self.threshold != -1:
            pred_probs= get_out_info(logits)
            pred = int((pred_probs >= self.threshold))
        else:
            pred = torch.argmax(logits, -1)

        # if acc_non_cs:
        #     tmp['acc-monolingual'] = acc_non_cs.item()
        # if acc_cs:
        #     tmp['acc-cs'] = acc_cs.item()

        loss = loss_f(logits, batch[-1])


        if len(logits.shape) <2:
            import pdb; pdb.set_trace()
        # if type(padded_ndx_tensor) == tuple:
        #     (spk, utt) = padded_ndx_tensor
        #     padded_ndx_tensor = (spk.to('cpu'), utt.to('cpu'))
        # else:
        #     padded_ndx_tensor = a
        # if type(topks) == tuple:
        #     (spk, utt) = topks
        #     topks = (spk.to('cpu'), utt.to('cpu'))
        # if type(topks) == dict:
        #     for l, item in topks.items():
        #         topks[l] = item.to('cpu')
        # for l, item in logits_dict.items():
        #     if type(item) != list:
        #         if type(item) == tuple:
        #             (spk, utt) = item
        #             item = (spk.to('cpu'), utt.to('cpu'))
        #             logits_dict[l] = item
        #         else:
        #             logits_dict[l] = item.to('cpu')
        #
        # import pdb; pdb.set_trace()
        # if type(tokens) == tuple:
        #     (spk, utt) = tokens
        #     tokens = (spk.to('cpu'), utt.to('cpu'))
        # else:
        #     tokens = tokens.to('cpu')
        # if self.accelerator =='dp':
        #     #loss = torch.stack([loss.unsqueeze(0), torch.Tensor([-1]).cuda()], dim=0)
        #     for l, item in logits_dict.items():
        #         try:
        #             if type(item) != list:
        #                 if type(item) == tuple:
        #                     (spk, utt) = item
        #                     # spk = torch.cat([spk, torch.zeros(spk.shape).cuda()], dim=0)
        #                     # utt = torch.cat([utt, torch.zeros(utt.shape).cuda()], dim=0)
        #                     spk = spk.tolist()
        #                     utt = utt.tolist()
        #                     logits_dict[l] = (spk, utt)
        #
        #                 else:
        #                     logits_dict[l] = item.tolist()
        #                     #logits_dict[l] = torch.cat([item, torch.zeros(item.shape).cuda()], dim=0)
        #         except:
        #             import pdb; pdb.set_trace()
        #     if type(tokens) == tuple:
        #         (spk, utt) = tokens
        #         spk = spk.tolist()
        #         utt = utt.tolist()
        #         # spk = torch.cat([spk, torch.zeros(spk.shape).cuda()], dim=0)
        #         # utt = torch.cat([utt, torch.zeros(utt.shape).cuda()], dim=0)
        #         tokens = (spk, utt)
        #     else:
        #         tokens = tokens.tolist()
        #         #tokens = torch.cat([tokens, torch.zeros(tokens.shape).cuda()], dim=0)
        #     if type(padded_ndx_tensor) == tuple:
        #         (spk, utt) = padded_ndx_tensor
        #         #spk = torch.cat([spk, torch.zeros(spk.shape).cuda()], dim=0)
        #         #utt = torch.cat([utt, torch.zeros(utt.shape).cuda()], dim=0)
        #         spk = spk.tolist()
        #         utt = utt.tolist()
        #         padded_ndx_tensor = (spk, utt)
        #     else:
        #         padded_ndx_tensor = padded_ndx_tensor.tolist()
        #         #padded_ndx_tensor = torch.cat([padded_ndx_tensor, torch.zeros(padded_ndx_tensor.shape).cuda()], dim=0)
        #





        return {'loss': loss,'batch_id':batch_idx, 'prediction': pred, 'f1':f1, 'precision': precision, 'recall':recall, 'tokens': tokens, 'padded_ndx_tensor':padded_ndx_tensor, 'accuracy':acc, \
                    'labels': labels, 'batch_len': len(batch), 'topk': topks, 'logits': logits_dict, 'speaker_context': speaker_context_ids}

        # return {'loss': loss.to('cpu'), 'prediction': pred.to('cpu'), 'f1':f1.to('cpu'), 'precision': precision.to('cpu'), 'recall':recall.to('cpu'), 'tokens': tokens, 'padded_ndx_tensor':padded_ndx_tensor, 'accuracy':acc.to('cpu'), \
        #         'labels': labels.to('cpu'), 'batch_len': len(batch), 'topk': topks, 'logits': logits_dict, 'speaker_context': speaker_context_ids}

    def test_epoch_end(self, test_step_outputs):
        i = 0
        predicted_labels, true_labels, gil_overall, lil_overall_sf_spk, lil_overall_act_spk = [], [], [], [], []
        lil_overall_sf_utt, lil_overall_act_utt = [], []
        f1s = []


        #['utterance', 'speaker']

        sentences = []
        precisions = []
        recalls = []

        total_evaluated = 0.
        spk_contexts_all = []
        for idx, out in tqdm(enumerate(test_step_outputs), total=len(test_step_outputs)):
            topks = out['topk']
            logits_dict = out['logits']



            tokens = out['tokens']
            padded_ndx_tensor = out['padded_ndx_tensor']
            pred = out['prediction']
            # acc = out['accuracy']
            # f1 = out['f1']
            # recall = out['recall']
            # precision = out['precision']
            labels = out['labels']
            f1s.append(out['f1'].detach().item())
            recalls.append(out['recall'].detach().item())
            precisions.append(out['precision'].detach().item())
            batch_len = out['batch_len']
            spk_contexts_all.append(out['speaker_context'])




            logits = logits_dict['logits']
            ## TODO --- UPDATE THIS!!!
            if not self.do_only_eval:

            # gil_interpretations = gil_interpret(concept_map=concept_map,
            #                                     list_of_interpret_dict=topks)
                lil_interpretations_sf,  lil_interpretations_act, sentences_all = lil_interpret(tokenizer=self.tokenizer, logits=logits, active_nt_tokens=logits_dict['lil_logits_utt_active_ndxs'],
                                                                                                topks=topks,
                                                                                                input_tokens = tokens, logits_dict=logits_dict,
                                                                                                padded_ndx_tensor=padded_ndx_tensor, use_phrase_embs=self.use_phrase_embs, prepend_description = self.prepend_description)
                lil_overall_sf_utt.append('\n'.join(lil_interpretations_sf['utterance']))
                lil_overall_act_utt.append('\n'.join(lil_interpretations_act['utterance']))
                if 'speaker' in lil_interpretations_sf:
                    lil_overall_sf_spk.append('\n'.join(lil_interpretations_sf['speaker']))
                    lil_overall_act_spk.append('\n'.join(lil_interpretations_act['speaker']))

            else:
                sentences_all = self.map_to_words(tokens.to('cpu').tolist())
            sentences.append(sentences_all)
            # accs.append(acc.item())
            # f1s.append(f1.item())
            # recalls.append(recall.item())
            # precisions.append(precision.item())
            # batch_predicted_labels = torch.argmax(logits, -1)
            predicted_labels.extend(pred.tolist())


            true_labels.extend(labels.tolist())
            #gil_overall.extend(gil_interpretations)
            #lil_overall.extend(lil_interpretations_sf)


            # print(gil_interpretations)
            # print(lil_interpretations_sf)
            # print(lil_interpretations_act)
            #print(sentences_all[0])
            total_evaluated += batch_len
            #total_correct += (acc.item() * batch_len)
            # print(
            #     f"Accuracy = {round((total_correct * 100) / (total_evaluated), 2)}, Batch accuracy = {round(acc.item(), 2)}")
            #i += tokens.size(0)
            # print(f"Accuracy = {round((total_correct * 100) / (total_evaluated), 2)}")
            # print(f"Accuracy = {round(np.array(accs).mean(), 2)}")
        true_labels = torch.Tensor(true_labels).cuda()
        predicted_labels = torch.Tensor(predicted_labels).cuda()
        acc = torch.true_divide(
            (predicted_labels == true_labels).sum(), true_labels.shape[0])
        cs_idxs = torch.where(true_labels == 1, True, False)
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
        desc = ''
        fold = 'interpretations'
        if self.random:
            fold += '_random'
        if not os.path.exists(fold):
            os.mkdir(fold)

        if not self.prepend_description:
            if self.use_speaker_toks:
                desc += '_using_spk_toks'

            else:
                desc += '_using_eot_eou'
            if self.use_speaker_descriptions:
                desc += '_using_speaker_descriptions'
        else:
            desc +='_prepend_desc'
            if self.ensemble_speaker_utt:
                desc += '_ensemble_spk_utt'
            else:

                if self.ensemble_utterance:
                    desc += '_ensemble_utt'
                if self.ensemble_speaker:
                    desc += '_ensemble_spk'
            if self.load_list:
                desc += '_list'
            elif self.load_sent:
                desc += "_sent"
            elif self.load_partner:
                desc += '_partner'

        desc += '_using_context_{}'.format(self.context_size)
        if not self.prepend_description:
            if 'speaker_trait_predictions' in self.model.hparams:
                if self.model.hparams.speaker_trait_predictions:
                    desc += "_spk_trait_pred"

        else:
            if any([jj in self.model.hparams for jj in ['speaker_trait_predictions_spk', 'speaker_trait_predictions_utt', 'speaker_trait_predictions_spk_utt']]):
                if self.model.hparams.speaker_trait_predictions_spk:
                    desc += "_spk_trait_pred_spk"
                if self.model.hparams.speaker_trait_predictions_utt:
                    desc += "_spk_trait_pred_utt"

                if self.model.hparams.speaker_trait_predictions_spk_utt:
                    desc += "_spk_trait_pred_spk_utt"
            if any([jj in self.model.hparams for jj in ['age', 'gender', 'language', 'mixing', 'country', 'order']]):
                exl = '_only'
                if 'leave_one_out' in self.model.hparams:
                    if self.model.hparams.leave_one_out:
                        exl= '_leave_out'


                desc+=exl
                if self.model.hparams.age:
                    desc += '_age'
                if self.model.hparams.gender:
                    desc += '_gender'
                if self.model.hparams.language:
                    desc += '_language'
                if self.model.hparams.mixing:
                    desc += '_mixing'
                if self.model.hparams.country:
                    desc += '_country'
                if self.model.hparams.order:
                    desc += '_order'
            # if any([jj in self.model.hparams for jj in ['age_mlm', 'gender_mlm', 'language_mlm', 'mixing_mlm']]):
            #     if self.model.hparams.age_mlm:
            #         desc += '_age_mlm'
            #     if self.model.hparams.gender_mlm:
            #         desc += '_gender_mlm'
            #     if self.model.hparams.language_mlm:
            #         desc += '_language_mlm'
            #     if self.model.hparams.mixing_mlm:
            #         desc += '_mixing_mlm'


        self_exp = ""
        if self.model.hparams.self_explain_ngram:
            self_exp = "_self_explain"
        if not self.do_only_eval:
            fname = '{}{}{}_interpretations_raw_seed_{}{}.csv'
        else:
            fname = '{}{}{}_eval_only_seed_{}{}.csv'
        control_ = ''
        if self.control:
            control_ = 'control_'
        if self.testing_file:
            fname = fname.format(control_, 'test', desc, self.seed, self_exp)
        else:
            fname = fname.format(control_, 'valid', desc, self.seed, self_exp)


        fname = os.path.join(fold, fname)
        data_dict = {"predicted_labels": predicted_labels.cpu().tolist(), "true_labels": true_labels.cpu().tolist(),\
         "full_sentence":sentences}
        if not self.do_only_eval:
            data_dict["lil_interpretations_softmax"]= lil_overall_sf_utt
            data_dict["lil_interpretations_activation"] =lil_overall_act_utt
            if len(gil_overall) > 0:
                data_dict["gil_interpretations"]= gil_overall
            if len(lil_overall_sf_spk) > 0:
                data_dict["spk_lil_interpretations_activation"] = lil_overall_act_spk
                data_dict["spk_lil_interpretations_softmax"] = lil_overall_sf_spk
                data_dict["speaker_context"]=spk_contexts_all
        print("Fraction code-switching {:.4f}".format(sum(true_labels)/len(true_labels)))
        print("Accuracy {:.4f}".format(acc*100))
        true = np.asarray(true_labels.tolist())
        pred= np.asarray(predicted_labels.tolist())
        #print("Mean F1 {:.4f}".format(np.mean(f1s)))
        print("F1 {:.4f}".format(f1_score(true, pred)))
        print("F1, neg class {:.4f}".format(f1_score(1-true, 1-pred)))
        print("Recall, {:.4f}".format(recall_score(true, pred)))
        print("Precision {:.4f}".format(precision_score(true, pred)))

        #print("F1 {:.4f}".format(f1))
        #print("Precision {:.4f}".format(np.mean(precisions)))
        #print("Recall {:.4f}".format(np.mean(recalls)))
        if self.control:
            data_dict['pairId'] = self.pair_ids

        #print(len(data_dict['spk_lil_interpretations_activation']))
        print(fname)
        pd.DataFrame(data_dict).to_csv(fname, sep="\t", index=False)


def load_model(ckpt, batch_size, dataset_basedir, balanced=True, control=False, threshold=-1, do_only_eval=False, prepend_description=False, seed=18, ngpus = 1):
    if balanced:
        print('loading balanced val set')
    rand=False
    if 'random' in dataset_basedir:
        rand = True
    model = SwitchLMForEval(ckpt, control=control, threshold=threshold, do_only_eval=do_only_eval, prepend_description=prepend_description, seed=seed, random=rand)
    accelerator = None
    if ngpus >1:
        accelerator = 'dp'
        model.accelerator = accelerator
    trainer = Trainer(gpus=ngpus, accelerator=accelerator)
    reg_data = True
    age, language, gend, mix, cntry, ord = False, False, False, False, False, False
    leave_one = False
    if any([jj in model.model.hparams for jj in ['age', 'language', 'gender', 'mixing', 'country', 'order']]):
        print('getting prune params')
        age = model.model.hparams.age
        language = model.model.hparams.language
        gend = model.model.hparams.gender
        mix = model.model.hparams.mixing
        cntry = model.model.hparams.country
        ord = model.model.hparams.order
        if 'leave_one_out' in model.model.hparams:
            leave_one = model.model.hparams.leave_one_out

        # if any([model.model.hparams.age_mlm, model.model.hparams.gender_mlm, model.model.hparams.language_mlm, model.model.hparams.mixing_mlm]):
        #     dm = ClassificationDataMLM(basedir=dataset_basedir, tokenizer_name='xlm-roberta-base', context_size=model.context_size, \
        #                                batch_size=batch_size, codeswitch=True, num_workers=10, balanced=balanced, \
        #                                load_sent_desc=model.load_sent, load_list_desc=model.load_list, load_partner_desc=model.load_partner, \
        #                                load_description_data=model.prepend_description, \
        #                                 load_only_balanced=balanced, load_control_data=control)
        #     #model.model.hparams.age_mlm = False
        #     #model.model.hparams.gender_mlm = False
        #     #model.model.hparams.language_mlm = False
        #     #model.model.hparams.mixing_mlm = False
        #     #model.model.hparams.mlm = False
        #     model.model.mlm = False
        #     print(model.model.mlm)

                                       # ,\
                                       # language_mlm=model.model.hparams.language_mlm, age_mlm=model.model.hparams.age_mlm, gender_mlm=model.model.hparams.gender_mlm, mixing_mlm=model.model.hparams.mixing_mlm)
            #reg_data = False
    #if reg_data:
    dm = ClassificationData(basedir=dataset_basedir, tokenizer_name='xlm-roberta-base', use_speaker_tokens=model.use_speaker_toks, \
                            use_speaker_descriptions=model.use_speaker_descriptions,load_sent_desc=model.load_sent, load_list_desc=model.load_list, load_partner_desc=model.load_partner, \
                            batch_size=batch_size, codeswitch=True, num_workers=16, balanced=balanced, load_description_data=model.prepend_description,\
                            context_size=model.context_size, load_only_balanced=balanced, load_control_data=control, age=age, language=language, mixing=mix, order=ord, \
                            gender=gend, country=cntry, leave_one_out=leave_one)
    desc = ''
    if prepend_description:
        desc = '_desc'
    #pair_ids = pd.read_csv(os.path.join(dataset_basedir, 'control_experiment{}_with_parse.csv'.format(desc)))['pairId'].tolist()
    #model.set_pair_ids(pair_ids)
    #print(len(pair_ids), len(model.pair_ids))
    #assert model.pair_ids is not None

    return model,trainer, dm


def load_examples(file_name, context_size, use_speaker_descriptions):
    samples = pd.read_csv(file_name)
    spk = []

    #if use_speaker_descriptions:
        #spk = samples['speakerDescriptions'].tolist()
    utt = samples['sentence{}'.format(context_size)].tolist()




    return spk, utt




def gil_interpret(concept_map, list_of_interpret_dict):
    batch_concepts = []
    for topk_concepts in list_of_interpret_dict["topk_gil"]:
        concepts = [concept_map[x] for x in topk_concepts.tolist()][:10]
        batch_concepts.append(concepts)
    return batch_concepts


def lil_interpret(tokenizer, logits, logits_dict, topks, input_tokens, padded_ndx_tensor, active_nt_tokens, use_phrase_embs=False, prepend_description=False):
    if prepend_description:
        padded_ndx_tensor_spk, padded_ndx_tensor_utt = padded_ndx_tensor
    else:
        padded_ndx_tensor_utt = padded_ndx_tensor
        active_nt_tokens_utt = active_nt_tokens

    def map_to_words(toks):
        #toks = [t for t in toks if t!=6]
        if len(toks) >1:
            if toks[0] == toks[1]:
                toks = toks[1:]

        try:
            words= [wd for wd in tokenizer.convert_ids_to_tokens(toks) if wd!= "<pad>"]
            words= tokenizer.convert_tokens_to_string(words)
        except:
            import pdb; pdb.set_trace()
        #words = ' '.join(words)
        words = words.replace('</s>', ' ')
        words = words.replace('<s>', ' ')
        return words
    def map_to_words_phr(toks):
        assert IX2PHR is not None
        if len(toks) >1:
            if toks[0] == toks[1]:
                toks = toks[1:]

        words = []
        try:
            for idx in toks:
                if idx not in IX2PHR:
                    print(idx)
                words.append(IX2PHR.get(idx, str(idx)))
        except:
            import pdb; pdb.set_trace()

        words = ' '.join(words)
        words = words.replace('</s>', '')
        words = words.replace('<s>', '')
        return words

    # lil_logits_tmp = [lil_logits[b][active_nt_tokens[b], :] for b in range(len(batch))]
    # lil_logits_mean= torch.stack([torch.mean(logits_, dim=0) for logits_ in lil_logits_tmp], dim=0).cuda()
    try:
        sf_logits = torch.softmax(logits, dim=-1).tolist()

        lil_sf_utt_logits = [torch.softmax(logits_dict["lil_logits_utt"][b][active_nt_tokens[b], :], dim=-1) for b in range(len(active_nt_tokens))]
    #lil_sf_utt_logits = torch.softmax(lil_logits_tmp, dim=-1).tolist()
    except:
        import pdb; pdb.set_trace()
    lil_sf_logits_all = [lil_sf_utt_logits]
    topks_all = [topks['topk_utt']]
    speaker_data = {}




    phrases_all = [[[input_tokens[k].masked_select(padded_ndx_tensor_utt[k][j].bool()) for j in range(len(padded_ndx_tensor_utt[k]))] for k in range(len(padded_ndx_tensor_utt))]]
    #phrases_all = [torch.masked_select(input_tokens[0], padded_ndx_tensor_utt[j].bool()) for j in range(len(padded_ndx_tensor_utt))]
    if 'lil_logits_spk' in logits_dict:
        active_nt_tokens_sp = logits_dict['lil_logits_spk_active_idxs']
        logits_sp = logits_dict["lil_logits_spk"]

        if prepend_description:
            lil_sp_sf_logits = [torch.softmax(logits_sp[b][active_nt_tokens_sp[b], :], dim=-1) for b in range(len(active_nt_tokens_sp))]
        else:
            lil_sp_sf_logits = []

            for b_idx, log_sp, active_nt in zip([i for i in range(len(logits_sp))], logits_sp, active_nt_tokens_sp):
                tmp_sf = {}
                for sp in log_sp.keys():
                    act_nt = active_nt[sp]
                    try:
                        softm = torch.softmax(log_sp[sp][act_nt, :], dim=-1)
                    except:
                        import pdb; pdb.set_trace()
                    tmp_sf[sp] = softm
                lil_sp_sf_logits.append(tmp_sf)

        #lil_sp_sf_logits = [torch.softmax(logits_dict["lil_logits_spk"][b][active_nt_tokens_sp[b], :], dim=-1) for b in range(active_nt_tokens.shape[0])]
        #lil_sp_sf_logits = torch.softmax(lil_logits_tmp_spk, dim=-1).tolist()
        #lil_sp_sf_logits = torch.xfdsoftmax(logits_dict["lil_logits_spk"], dim=-1).tolist()
        #lil_sf_logits_all.append(lil_sp_sf_logits)
        #topks_all.append(topks['topk_spk'])
        phrases_spk_all = logits_dict['speaker_tokens']
        nt_spk_all = logits_dict['nt_speaker_all']
        if not prepend_description:



            phrases_spk = [{sp: [phrases_spk_all[k][sp].masked_select(phrase.bool()) for phrase in phrase_list if torch.sum(phrase) > 0] for sp, phrase_list in nt_spk_all[k].items()} for k in range(len(phrases_spk_all))]
        else:
            phrases_spk = [[[input_tokens[k].masked_select(padded_ndx_tensor_spk[k][j].bool()) for j in range(len(padded_ndx_tensor_spk[k]))] for k in range(len(padded_ndx_tensor_spk))]]
        # for sp_dict in phrases_spk:
        #
        #     for sp, phrses in sp_dict.items():
        #         for ph in phrses:
        #             try:
        #             if ph == torch.Tensor([0]):
        #                 import pdb; pdb.set_trace()
        #print(phrases_spk)
        #print(phrases_spk_all)
        speaker_data['phrases'] = phrases_spk

        speaker_data['topk'] = topks['topk_spk']
        speaker_data['sf_logits'] = lil_sp_sf_logits
        speaker_data['logits'] = logits_sp
        speaker_data['tokens'] = phrases_spk_all
        #phrases_spk = [[[input_tokens[k].masked_select(padded_spk_ndx_tensor[k][j].bool()) for j in range(len(padded_spk_ndx_tensor[k]))] for k in range(len(padded_ndx_tensor))]]
        #phrases_all.append(phrases_spk)

    sentences_all = map_to_words(input_tokens[0])
    # if "IRI" in sentences_all and "JAM" in sentences_all:
    #     import pdb; pdb.set_trace()


    lil_outputs_sf = {}
    lil_outputs_act = {}
    lil_outputs_sp = {}
    for lil_sf_logits,  phrase_list, name in zip(lil_sf_logits_all, phrases_all, ['utterance']):


        for idx, (sf_item, lil_sf_item, nt_topks, toks, phrases) in enumerate(zip(sf_logits, lil_sf_logits, topks_all, input_tokens, phrase_list)):
            #dev_sample = dev_samples[current_idx + idx]
            lil_dict = {}
            try:
                argmax_sf, _ = max(enumerate(sf_item), key=itemgetter(1))
            except:
                import pdb; pdb.set_trace()

            def get_phrase_relevance(lil_sf_item, phrases, output_dict, use_phrase=False):

                for phrase_idx, phrase in enumerate(phrases):
                    try:
                        phrase_logits = lil_sf_item[phrase_idx]
                    except:
                        import pdb; pdb.set_trace()

                    try:
                        argmax_loc, _ = max(enumerate(phrase_logits), key=itemgetter(1))
                    except:
                        import pdb; pdb.set_trace()
                    if argmax_loc == argmax_sf:
                        sign_ = 1
                    else:
                        sign_ = -1
                    relevance_score = torch.abs(phrase_logits[argmax_sf] - sf_item[argmax_sf])*sign_

                    if not use_phrase:
                        phrase_words = list(map(map_to_words, [phrase.to('cpu').tolist()]))[0]
                    else:
                        phrase_words = list(map(map_to_words_phr, [phrase.to('cpu').tolist()]))[0]
                    phrase_words = phrase_words.replace("<s>", '')
                    phrase_words = phrase_words.replace("</s>", '')
                    output_dict[phrase_words] = relevance_score
                return output_dict
            def topk_act(nt_topks, toks, use_phrase=False):
                lil_list_act = []
                for phrase_nt in nt_topks:
                    try:
                        phrase = toks.masked_select(phrase_nt.bool())
                    except:
                        import pdb; pdb.set_trace()
                    if not use_phrase:
                        phrase_words = list(map(map_to_words, [phrase.to('cpu').tolist()]))[0]
                    else:
                        phrase_words = list(map(map_to_words_phr, [phrase.to('cpu').tolist()]))[0]
                    lil_list_act.append(phrase_words)
                return lil_list_act
            if prepend_description:
                lil_dict = get_phrase_relevance(lil_sf_item[idx], phrases, lil_dict)
            else:
                lil_dict = get_phrase_relevance(lil_sf_item, phrases, lil_dict)
            def sort_phrase_relevances(phr_dict):
                names = [v for v in phr_dict.keys()]
                vals = np.asarray([phr_dict[v].detach().item() for v in names])
                neg_vals = np.where(vals < 0)[0]
                pos_vals = np.where(vals > 0)[0]
                pos_vals_idx_sorted = np.argsort(vals[pos_vals])
                #pos_vals_idx_sorted = pos_vals[pos_vals_sorted]
                neg_vals_idx_sorted = np.argsort(vals[neg_vals])
                idxs_sorted_all = pos_vals[pos_vals_idx_sorted].tolist() + neg_vals[neg_vals_idx_sorted].tolist()
                phr_rel = [' :: '.join([names[j], str(np.round(vals[j], decimals=5))]) for j in idxs_sorted_all]
                return phr_rel



            #lil_dict = [' ; '.join([k, str(v.detach().item())]) for k, v in sorted(lil_dict.items(), key=lambda item: item[1])[::-1]]
            lil_dict = sort_phrase_relevances(lil_dict)
            spk_phrases_all = []
            spk_lil_acts_all = []
            if len(speaker_data) > 0:
                logits_dict = speaker_data['sf_logits'][idx]
                phrases_dict = speaker_data['phrases'][idx]
                topks_sp = speaker_data['topk'][idx]
                toks_dict = speaker_data['tokens'][idx]


                if not prepend_description:
                    for sp in logits_dict.keys():

                        spk_lil = get_phrase_relevance(logits_dict[sp], phrases_dict[sp], {}, use_phrase=use_phrase_embs)
                        phr_rel = [' ; '.join([sp, k, str(v.detach().item())]) for k, v in sorted(spk_lil.items(), key=lambda item: item[1])[::-1]]
                        spk_phrases_all.extend(phr_rel)
                        lil_act_sp = topk_act(topks_sp[sp], toks_dict[sp], use_phrase=use_phrase_embs)
                        spk_lil_acts_all.extend(lil_act_sp)
                else:
                    if logits_dict.shape[0] == 1:
                        logits_dict = logits_dict[0]
                        phrases_dict=phrases_dict[0]


                    spk_lil = get_phrase_relevance(logits_dict, phrases_dict, {})
                    #phr_rel = [' ; '.join([sp, k, str(v.detach().item())]) for k, v in sorted(spk_lil.items(), key=lambda item: item[1])[::-1]]
                    phr_rel = sort_phrase_relevances(spk_lil)
                    spk_phrases_all.extend(phr_rel)
                    lil_act_sp = topk_act(topks_sp, toks_dict)
                    spk_lil_acts_all.extend(lil_act_sp)


                lil_outputs_act['speaker'] = spk_lil_acts_all
                lil_outputs_sf['speaker'] = spk_phrases_all


            lil_outputs_sf[name] = lil_dict

            # now use topks from the raw activation scores


            lil_list_act = topk_act(nt_topks, toks)

            lil_outputs_act[name] = lil_list_act
            #return lil_dict, lil_outputs_act

    return lil_outputs_sf, lil_outputs_act, sentences_all


def load_concept_map(concept_map_path):
    concept_map = {}
    with open(concept_map_path, 'r') as open_file:
        concept_map_str = json.loads(open_file.read())
    for key, value in concept_map_str.items():
        concept_map[int(key)] = value
    return concept_map
def load_phr_dict(phr_dict_path):
    tmp = torch.load(phr_dict_path)
    GLOBAL_PHR_EMBS = tmp
    return tmp['phr2ix'], tmp['ix2phr']

if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--concept_map', type=str, default='bangor_data/feature_spk2/concept_idx.json')
    parser.add_argument('--data_dir', type=str, default="bangor_data/feature_spk_new_raw")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=18)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument("--balanced", action='store_true', help="load balanced dataset")
    parser.add_argument("--prepend_description", action='store_true', help="load balanced dataset")
    parser.add_argument("--do_only_eval", action='store_true', help="don't save interpretations, just eval")
    parser.add_argument("--control_data", action='store_true', help="load control dataset (common amigos)")
    #parser.add_argument("--age_mlm", action='store_true', help="load control dataset (common amigos)")
    #parser.add_argument("--gender_mlm", action='store_true', help="load control dataset (common amigos)")
    #parser.add_argument("--language_mlm", action='store_true', help="load control dataset (common amigos)")
    #parser.add_argument("--mixing_mlm", action='store_true', help="load control dataset (common amigos)")
    parser.add_argument("--threshold", type=float, required=False, default=-1,help="set decision threshold, default just takes the highest log likelihood for prediction")
    parser.add_argument('--phrase_embeddings_path', type=str, default="bangor_data/feature_spk_new_raw/speaker_description_phrase_embeddings_with_parse.pt")

    args = parser.parse_args()
    print(args.balanced)
    model, trainer, dm = load_model(args.ckpt, batch_size = args.batch_size, prepend_description=args.prepend_description,seed=args.seed, \
                                    dataset_basedir=args.data_dir, balanced=args.balanced, control=args.control_data, threshold=args.threshold, do_only_eval = args.do_only_eval, ngpus=args.gpus)

    #concept_map = load_concept_map(concept_map_path=args.concept_map)
    # if 'non_overlapped' in args.data_dir:
    #     print("using non overlapped nt-idx")
    #     OVERLAPPING=False
    # elif 'overlapped' in args.data_dir:
    #     print("using overlapped nt-idx")
    #     OVERLAPPING=True
    SEED = args.seed
    np.random.seed(SEED)
    random.seed(SEED)
    pl.utilities.seed.seed_everything(SEED)
    pl.seed_everything(SEED)
    PHR2IX, IX2PHR = load_phr_dict(args.phrase_embeddings_path)
    test_file_list = [False, True]
    if args.control_data:
        test_file_list = [True]
    for testing in test_file_list:
        model.testing_file=testing
        if not testing:

            trainer.test(test_dataloaders=dm.val_dataloader(), model=model)
        else:
            trainer.test(test_dataloaders=dm.test_dataloader(), model=model)
        # eval(model,
        #      dm.val_dataloader(),
        #      concept_map=concept_map,
        #      dev_file=os.path.join(args.data_dir, filename),
        #      paths_output_loc=args.paths_output_loc,
        #      use_full_context=args.use_full_context, use_speaker_descriptions=args.use_speaker_descriptions)
