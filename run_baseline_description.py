import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import random
import numpy as np
import pytorch_lightning as pl
import logging
import os
from argparse import ArgumentParser
import resource
from model.data import ClassificationData
from pytorch_lightning.loggers import TensorBoardLogger
import torch


from model.SwitchLM import SwitchLM
from model.SwitchLMSpeaker_TripletLoss import SwitchLMSpeakerLoss
from model.SwitchLMSpeaker import SwitchLMSpeaker
from model.data_mlm import ClassificationDataMLM
from interpret_bangor import SwitchLMForEval

def get_train_steps(dm):
  total_devices = args.num_gpus * args.num_nodes
  train_batches = len(dm.train_dataloader()) // total_devices
  accumulate_grad_batches = args.accumulate_grad_batches
  if args.accumulate_grad_batches is None:
      accumulate_grad_batches = 1
  return (args.max_epochs * train_batches) // accumulate_grad_batches




rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
# init: important to make sure every node initializes the same weights



# argparser
parser = ArgumentParser()
parser.add_argument('--num_gpus', type=int)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--clip_grad', type=float, default=1.0)
parser.add_argument("--dataset_basedir", help="Base directory where the dataset is located.", type=str)
parser.add_argument("--model_name", default='xlm-roberta-base', help="Model to use.")
parser.add_argument("--load_descriptions", action='store_true', help="prepend spk desc to each dialogue")
parser.add_argument("--age_mlm", action='store_true', help="prepend spk desc to each dialogue")
parser.add_argument("--gender_mlm", action='store_true', help="prepend spk desc to each dialogue")
parser.add_argument("--language_mlm", action='store_true', help="prepend spk desc to each dialogue")
parser.add_argument("--mixing_mlm", action='store_true', help="prepend spk desc to each dialogue")
parser.add_argument("--use_full_context", action='store_true', help="early fusion of speaker context")
parser.add_argument("--context_size", default=1,type=int, help="number of sentences prior for context")
parser.add_argument("--monitor", default="acc",type=str, help="monitor loss, acc, or f1")
parser.add_argument("--control", action='store_true', help="load control data for finetuning")
parser.add_argument("--ckpt", default="",type=str, help="ckpt from which to load and finetune a model")
parser.add_argument("--tensorboard_dir", default="tb_logs",type=str, help="tensorboard directory for logs")
parser.add_argument("--spk_control", action='store_true', help="use 'fake' speaker descriptions as a control")
parser.add_argument("--seed", default=18,type=int, help="seed for model run")




parser.add_argument("--use_speaker_descriptions", action='store_true', help='prepend a sentence of speaker context to the previous and current utterances')

parser = pl.Trainer.add_argparse_args(parser)


parser = SwitchLMSpeaker.add_model_specific_args(parser)

args = parser.parse_args()

SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
pl.utilities.seed.seed_everything(SEED)
pytorch_lightning.seed_everything(SEED)


args.num_gpus = len(str(args.gpus).split(","))


logging.basicConfig(level=logging.INFO)
balanced = args.balanced
use_speaker_descriptions = args.use_speaker_descriptions
eval_sep_languages = args.eval_per_language
speaker_trait_predictions = any([args.speaker_trait_predictions_utt,args.speaker_trait_predictions_spk,args.speaker_trait_predictions_spk_utt])
# Step 1: Init Data
logging.info("Loading the data module for Bangor")
# dm = ClassificationData(basedir=args.dataset_basedir, tokenizer_name=args.model_name,\
#                         batch_size=args.batch_size, codeswitch=True, num_workers=args.num_workers, balanced=balanced,\
#                        use_speaker_descriptions=use_speaker_descriptions,\
#                         get_lang_feats=eval_sep_languages, use_full_context=args.use_full_context)
if any([args.age_mlm, args.gender_mlm, args.language_mlm, args.mixing_mlm]):
    dm = ClassificationDataMLM(basedir=args.dataset_basedir, tokenizer_name=args.model_name, context_size=args.context_size,load_description_data=True, \
                               batch_size=args.batch_size, codeswitch=True, num_workers=args.num_workers, balanced=balanced, use_eot_tokens=False, mlm_pretrain=args.mlm_pretrain, \
                               load_control_data=args.control, load_full_control=False, load_list_desc =args.list, load_sent_desc = args.sentence, \
                               load_partner_desc=args.partner, full_mtl_setup=args.full_mtl_setup, language=args.language, age=args.age, gender=args.gender, mixing=args.mixing)
else:
    dm = ClassificationData(basedir=args.dataset_basedir, tokenizer_name=args.model_name, context_size=args.context_size,load_description_data=True,  \
                            batch_size=args.batch_size, codeswitch=True, num_workers=args.num_workers, balanced=balanced, use_speaker_tokens=False,use_eot_tokens=False, use_speaker_descriptions=False,\
                            load_control_data=args.control, load_full_control=False, load_list_desc =args.list, load_sent_desc = args.sentence, \
                            do_social_predictions=speaker_trait_predictions, load_partner_desc=args.partner, load_triplet=args.triplet_loss,  full_mtl_setup=args.full_mtl_setup, \
                            language=args.language, age=args.age, gender=args.gender, mixing=args.mixing, country=args.country,order=args.order, leave_one_out=args.leave_one_out, fake_spk=args.spk_control)

# Step 2: Init Model
model_tmp = None
if not args.control and not args.finetune_bangor:

    logging.info("Initializing the model")
    if args.contrastive_loss or args.triplet_loss:
        model = SwitchLMSpeakerLoss(hparams=args)
    else:
        model = SwitchLMSpeaker(hparams=args)

else:
    logging.info("loading from previous checkpoint")

    model_tmp = SwitchLMSpeaker.load_from_checkpoint(args.ckpt)
    model = SwitchLMSpeaker(hparams=args)
    model.model.embeddings = model_tmp.model.embeddings
    # if args.finetune_bangor:
    #     model.hparams.finetune_bangor = True
    #     model.hparams.triplet_loss = False
    #     model.loss = torch.nn.CrossEntropyLoss()
#import pdb; pdb.set_trace():q

#args.vocab_size = len(dm.tokenizer)


model.hparams.warmup_steps = int(get_train_steps(dm) * model.hparams.warmup_prop)
#lr_monitor = LearningRateMonitor(logging_interval='step')
monit = args.monitor
if args.triplet_loss or args.contrastive_loss:
    monit = 'loss'
monitor = 'val_{}'.format(monit)
print("Monitor, ", monit)
fname = '{epoch}-{step}-{val_' + monit+'_epoch:.4f}'


# Step 3: Start
# if 'non_overlapped' in args.dataset_basedir:
#     print("using non overlapped nt-idx")
#     desc = 'non_overlapped'
# elif 'overlapped' in args.dataset_basedir:
#     print("using overlapped nt-idx")
#     desc = 'overlapped'
# if args.use_full_context:
#     desc += '_full_context'
#     args.use_speaker_descriptions=False
desc = ''
if 'random' in args.dataset_basedir:
    desc += '_random'
if args.self_explain_ngram:
    desc += '_self_explain'
if args.load_descriptions:
    desc += '_load_descriptions'
    if args.ensemble_speaker_utt:
        desc += '_ensemble_spk_utt'
    else:

        if args.ensemble_utterance:
            desc += '_ensemble_utt'
        if args.ensemble_speaker:
            desc += '_ensemble_spk'
    if args.list:
        desc += '_list'
    elif args.sentence:
        desc += '_sentence'
    elif args.partner:
        desc += '_partner'
if any([args.gender, args.age, args.language, args.mixing, args.order, args.country]):
    exl = '_only'
    if args.leave_one_out:
        exl = '_leave_out'
    if args.gender:
        desc+="_gender"
    if args.age:
        desc+="_age"
    if args.language:
        desc+="_language"
    if args.mixing:
        desc+="_mixing"
    if args.order:
        desc+="_order"
    if args.country:
        desc+="_country"
    desc += exl

if model_tmp is not None:
    if any([model_tmp.hparams.gender, model_tmp.hparams.age, model_tmp.hparams.language, model_tmp.hparams.mixing, model_tmp.hparams.order, model_tmp.hparams.country]):
        exl = '_only'
        if 'leave_one_out' in model_tmp.hparams:
            if model_tmp.hparams.leave_one_out:
                exl= '_leave_out'


        desc+=exl
        if model_tmp.hparams.gender:
            desc+="_gender"
        if model_tmp.hparams.age:
            desc+="_age"
        if model_tmp.hparams.language:
            desc+="_language"
        if model_tmp.hparams.mixing:
            desc+="_mixing"
        if model_tmp.hparams.order:
            desc+="_order"
        if model_tmp.hparams.country:
            desc+="_country"
if args.control:
    desc = 'finetune_control_'+desc
if args.triplet_loss:
    desc = 'triplet_tune_' + desc
if args.finetune_bangor:
    desc += '_finetune_bangor_post_triplet'
desc += '_ctx_size_'+str(args.context_size)
if speaker_trait_predictions:
    desc += '_spk_trait_aux'
if args.speaker_trait_predictions_utt:
    desc += "_utt"
elif args.speaker_trait_predictions_spk:
    desc += "_spk"
elif args.speaker_trait_predictions_spk_utt:
    desc += "_spk_utt"

if args.full_mtl_setup:
    desc+= "_full_mtl"
# if args.threshold != 0.5:
#     desc += "threshold_{}".format(args.threshold)
# if args.age_mlm:
#     desc += "_age_mlm"
# if args.gender_mlm:
#     desc += "_gender_mlm"
# if args.language_mlm:
#     desc += "_language_mlm"
# if args.mixing_mlm:
#     desc += "_mixing_mlm"
desc += "_"+monitor
mode='max'

if monit == 'loss':
    mode = 'min'
print(mode)
desc += str(args.seed)
logger = TensorBoardLogger(args.tensorboard_dir, name=desc)
logging.info("Starting the training")
checkpoint_callback = ModelCheckpoint(
    os.path.join(os.getcwd(), 'chkpts{}'.format(desc)),
    save_top_k=3,
    verbose=True,
    monitor='{}_epoch'.format(monitor), save_weights_only=True,
    mode=mode
)
accelerator = None
if args.gpus > 1:
    accelerator='dp'

accumulate_grad_batches=args.accumulate_grad_batches
if accumulate_grad_batches is None:
    accumulate_grad_batches = 1
# if any([args.age_mlm, args.language_mlm, args.gender_mlm, args.mixing_mlm]):
#     accumulate_grad_batches =
trainer = pl.Trainer.from_argparse_args(args, log_every_n_steps=25, accumulate_grad_batches = accumulate_grad_batches, precision=16, callbacks=[checkpoint_callback], accelerator=accelerator, val_check_interval=0.5, gradient_clip_val=args.clip_grad, track_grad_norm=2, stochastic_weight_avg=True, logger=logger)
trainer.fit(model, dm)
if not args.control and not args.triplet_loss and not args.finetune_bangor:
    trainer.test(ckpt_path='best', test_dataloaders= dm.test_dataloader())
print("evaluating/interpreting results... ")
rand='random' in args.dataset_basedir
best= SwitchLMForEval(checkpoint_callback.best_model_path, control=False, threshold=-1, do_only_eval=False, prepend_description=True, seed=SEED, random=rand)
trainer2 = pl.Trainer(gpus=1, accelerator=None)
if any([args.age_mlm, args.gender_mlm, args.language_mlm, args.mixing_mlm]):
    dm2 = ClassificationDataMLM(basedir=args.dataset_basedir, tokenizer_name=args.model_name, context_size=args.context_size,load_description_data=True, \
                               batch_size=1, codeswitch=True, num_workers=args.num_workers, balanced=False, use_eot_tokens=False, mlm_pretrain=args.mlm_pretrain, \
                               load_control_data=args.control, load_full_control=False, load_list_desc =args.list, load_sent_desc = args.sentence, \
                               load_partner_desc=args.partner, full_mtl_setup=args.full_mtl_setup, language=args.language, age=args.age, gender=args.gender, mixing=args.mixing)
else:
    dm2 = ClassificationData(basedir=args.dataset_basedir, tokenizer_name=args.model_name, context_size=args.context_size,load_description_data=True, \
                            batch_size=1, codeswitch=True, num_workers=args.num_workers, balanced=False, use_speaker_tokens=False,use_eot_tokens=False, use_speaker_descriptions=False, \
                            load_control_data=args.control, load_full_control=False, load_list_desc =args.list, load_sent_desc = args.sentence, \
                            do_social_predictions=speaker_trait_predictions, load_partner_desc=args.partner, load_triplet=args.triplet_loss,  full_mtl_setup=args.full_mtl_setup, \
                            language=args.language, age=args.age, gender=args.gender, mixing=args.mixing, country=args.country,order=args.order, leave_one_out=args.leave_one_out, fake_spk=args.spk_control)

test_file_list = [False, True]
# if args.control_data:
#     test_file_list = [True]
for testing in test_file_list:
    best.testing_file=testing
    if not testing:

        trainer2.test(test_dataloaders=dm2.val_dataloader(), model=best)
    else:
        trainer2.test(test_dataloaders=dm2.test_dataloader(), model=best)
