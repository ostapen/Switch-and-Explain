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
from interpret_bangor import SwitchLMForEval



from model.SwitchLM import SwitchLM
#from model.SwitchLMSpeaker import SwitchLMSpeaker

def get_train_steps(dm):
  total_devices = args.num_gpus * args.num_nodes
  train_batches = len(dm.train_dataloader()) // total_devices
  return (args.max_epochs * train_batches) // args.accumulate_grad_batches




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
parser.add_argument("--use_speaker_tokens", action='store_true', help="prepend [SPK1], etc to each utterance")

parser.add_argument("--use_full_context", action='store_true', help="early fusion of speaker context")
parser.add_argument("--context_size", default=1,type=int, help="number of sentences prior for context")

parser.add_argument("--monitor", default="acc",type=str, help="monitor loss, acc, or f1")
parser.add_argument("--seed", default=18,type=int, help="seed for model run")
parser.add_argument("--control", action='store_true', help="load control data for finetuning")
parser.add_argument("--ckpt", default="",type=str, help="ckpt from which to load and finetune a model")
parser.add_argument("--tensorboard_dir", default="tb_logs_baseline",type=str, help="tensorboard directory for logs")


parser.add_argument("--use_speaker_descriptions", action='store_true', help='prepend a sentence of speaker context to the previous and current utterances')

parser = pl.Trainer.add_argparse_args(parser)


parser = SwitchLM.add_model_specific_args(parser)
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

# Step 1: Init Data
logging.info("Loading the data module for Bangor")
# dm = ClassificationData(basedir=args.dataset_basedir, tokenizer_name=args.model_name,\
#                         batch_size=args.batch_size, codeswitch=True, num_workers=args.num_workers, balanced=balanced,\
#                        use_speaker_descriptions=use_speaker_descriptions,\
#                         get_lang_feats=eval_sep_languages, use_full_context=args.use_full_context)
dm = ClassificationData(basedir=args.dataset_basedir, tokenizer_name=args.model_name, context_size=args.context_size,load_description_data=False,  \
                        batch_size=args.batch_size, codeswitch=True, num_workers=args.num_workers, balanced=balanced, use_speaker_tokens=args.use_speaker_tokens, \
                        load_control_data=args.control, load_full_control=False, do_social_predictions=args.speaker_trait_predictions, full_mtl_setup=args.full_mtl_setup)

# Step 2: Init Model
#import pdb; pdb.set_trace():q
if not args.control:

    logging.info("Initializing the model")
    model = SwitchLM(hparams=args, vocab_size=len(dm.tokenizer))
else:
    logging.info("loading from previous checkpoint")
    model = SwitchLM.load_from_checkpoint(args.ckpt)
#args.vocab_size = len(dm.tokenizer)

model.hparams.warmup_steps = int(get_train_steps(dm) * model.hparams.warmup_prop)
#lr_monitor = LearningRateMonitor(logging_interval='step')
print(model.model.vocab_size)
monitor = 'val_{}'.format(args.monitor)
print("Monitor, ", monitor)
fname = '{epoch}-{step}-{val_' + args.monitor+'_epoch:.4f}'


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
if args.use_speaker_descriptions:
    desc += '_speaker_descriptions'
    if args.phrase_embeddings:
        desc += '_with_phrase'
    if args.concatenate_speaker:
        desc += '_cat_speaker'
    if args.multihead_pool:
        desc+= '_multihead_pool'
    if args.multihead_pool_over_input:
        desc+= '_multihead_pool_over_input'
    if args.concat_speaker_logits:
        desc += '_concat_speaker_logits'

# if args.threshold != 0.5:
#     desc += "threshold_{}".format(args.threshold)

if args.use_speaker_tokens:
    desc += '_spk_tokens'
else:
    desc += '_eot_eou'
if args.no_adapter:
    desc += '_no_adapter'
desc += '_ctx_size_'+str(args.context_size)
if args.speaker_trait_predictions:
    desc += '_spk_trait_aux'
if args.full_mtl_setup:
    desc+= "_full_mtl"
desc += "_"+monitor
mode='max'
if args.monitor == 'loss':
    mode = 'min'

desc += str(args.seed)
logging.info("Starting the training")
checkpoint_callback = ModelCheckpoint(
    os.path.join(os.getcwd(), 'chkpts{}'.format(desc)),
    save_top_k=3,
    verbose=True,
    monitor='{}_epoch'.format(monitor), save_weights_only=True,
    mode=mode
)
logger = TensorBoardLogger(args.tensorboard_dir, name=desc)
accelerator = None
if args.gpus > 1:
    accelerator='dp'
trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], val_check_interval=0.5, accelerator=accelerator, gradient_clip_val=args.clip_grad, track_grad_norm=2, logger=logger)
trainer.fit(model, dm)
if not args.control:
    trainer.test(ckpt_path='best')

print("evaluating/interpreting results... ")
rand='random' in args.dataset_basedir
best= SwitchLMForEval(checkpoint_callback.best_model_path, control=False, threshold=-1, do_only_eval=False, prepend_description=False, seed=SEED, random=rand)
dm2 = ClassificationData(basedir=args.dataset_basedir, tokenizer_name=args.model_name, context_size=args.context_size,load_description_data=False, \
                         batch_size=1, codeswitch=True, num_workers=args.num_workers, balanced=False, use_speaker_tokens=args.use_speaker_tokens, \
                         load_control_data=args.control, load_full_control=False, do_social_predictions=args.speaker_trait_predictions, full_mtl_setup=args.full_mtl_setup)
trainer2 = pl.Trainer(gpus=1, accelerator=None)
test_file_list = [False, True]
# if args.control_data:
#     test_file_list = [True]
for testing in test_file_list:
    best.testing_file=testing
    if not testing:

        trainer2.test(test_dataloaders=dm2.val_dataloader(), model=best)
    else:
        trainer2.test(test_dataloaders=dm2.test_dataloader(), model=best)