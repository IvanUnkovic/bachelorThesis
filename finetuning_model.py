import argparse
import torch.nn as nn
import pytorch_lightning as pl

from typing import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from rockfish.model.model import Rockfish
from rockfish.model.datasets import RFDataModule

#function used for removing a specific layer from a decoder/encoder
def remove_layer(model, index, isdecoder, isencoder):
    if isdecoder:
        decoder = list(model.children())[7]
        modulelistD = list(decoder.children())[0]
        encoder = list(model.children())[5]
        modulelistE = list(encoder.children())[0]
        #removing certain layers
        #key_to_remove = list(modulelistD._modules.keys())[11]
        #modulelistD._modules.pop(key_to_remove)
        #key_to_remove = list(modulelistE._modules.keys())[11]
        #modulelistE._modules.pop(key_to_remove)

def load_for_finetune(path, ft_lr, freeze, replace_classification):
    #model loaded from checkpoint
    model = Rockfish.load_from_checkpoint(path, strict=False, track_metrics=True)
    #adaptation of learning rate
    model.hparams.lr = ft_lr

    #if freeze set to true, freeze all layers except fc_mod which is the classification head 
    if freeze:
        for name, param in model.named_parameters():
            if name.startswith('fc_mod'):
                continue
            param.requires_grad = False
    #if replace_classification set to true, replace existing fc_mod with a randomly initialized one 
    if replace_classification:
        model.fc_mod = nn.Linear(model.hparams.features, 1)
    return model

#function used for creating trainer defaults
def get_trainer_defaults() -> Dict[str, Any]:
    trainer_defaults = {}

    model_checkpoint = ModelCheckpoint(monitor='f1-score',
                                       save_top_k=3,
                                       mode='max')
    trainer_defaults['callbacks'] = [model_checkpoint]

    wandb = WandbLogger(project='dna-mod', log_model=True, save_dir='wandb')
    trainer_defaults['logger'] = wandb

    return trainer_defaults

def finetune_main(args: argparse.Namespace) -> None:
    #model is loaded from a checkpoint and prepared for possible fine-tuning
    model = load_for_finetune(path=args.ckpt_path, ft_lr=args.lr, freeze=args.freeze_layers, replace_classification=args.replace_classification_head)
    #removing a layer from a model
    remove_layer(model, args.ind, args.dec, args.enc)
    print(model)
    #data module is initializes
    data_module = RFDataModule(train_data=args.train_data, train_labels=args.train_labels,
                               val_data=args.val_data, val_labels=args.val_labels,
                               train_batch_size=args.train_batch_size,
                               val_batch_size=args.val_batch_size)
    trainer_defaults = get_trainer_defaults()
    trainer = pl.Trainer(strategy='ddp',
                         accelerator='gpu',
                         devices=args.gpus,
                         precision=16,
                         gradient_clip_val=1.0,
                         val_check_interval=args.val_check_interval,
                         logger=trainer_defaults['logger'],
                         callbacks=trainer_defaults['callbacks'])
    #fine-tuning is performed
    trainer.fit(model=model, datamodule=data_module)
    #checkpoint is saved
    checkpoint_name = "something.ckpt"
    trainer.save_checkpoint(checkpoint_name)

#parser is defined
def add_finetuning_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint of model that will be fine-tuned')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training dataset in .rf format')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to training labels in .npy format')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation dataset in .rf format')
    parser.add_argument('--val_labels', type=str, required=True, help='Path to validation labels in .npy format')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate value')
    parser.add_argument('--gpus', type=int, nargs='+', default=None, help='GPU cards that will be used for fine-tuning, accepts more than one in format "0 1"')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Batch size that will be used for training step')
    parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size that will be used for validation step')
    parser.add_argument('--val_check_interval', type=int, default=5000, help='Number of training steps between two validation steps')
    parser.add_argument('--freeze_layers', action='store_true', help='Flag indicating whether layers other than classification head should be frozen during fine-tuning')
    parser.add_argument('--replace_classification_head', action='store_true', help='Flag indicating whether current should be replaced by a randomly initialized one prior to fine-tuning')
    parser.add_argument('--ind', type=int, default=11, help='Index of a layer in the signal encoder or alignment decoder that is going to be removed from the model')
    parser.add_argument('--dec', action='store_true', help='Flag indicating whether the layer is going to be removed from a decoder')
    parser.add_argument('--enc', action='store_true', help='Flag indicating whether the layer is going to be removed from a decoder') 
    return parser.parse_args()

if __name__ == '__main__':
    args = add_finetuning_args(argparse.ArgumentParser())
    finetune_main(args)
