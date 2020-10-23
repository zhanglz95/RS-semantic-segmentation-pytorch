import argparse
from pathlib import Path
import logging
import json
from torch.utils.data import random_split, DataLoader

import dataset as D
import model as M

from trainer import Trainer


def train(config_path):
    # logging config
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    # file handle
    fh = logging.FileHandler('log.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    # stream handle
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(fh)

    if(not Path(config_path).exists()):
        logger.error(f"The config path \'{config_path}\' is not exists.")
        return
    logger.info(f"Start training from config file \'{config_path}\'.")
    # load configs from .json
    configs = json.load(open(config_path))
    # initial train loader and test loader
    train_dataset = getattr(D, configs['dataset_name'])(**configs['dataset_configs'])

    val_percent = configs['val_percent']
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val
    train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=configs['batchsize'], shuffle=True, num_workers=8, pin_memory=True)
    if configs['hasVal']:
        val_loader = DataLoader(val_dataset, batch_size=configs['batchsize'], shuffle=True, num_workers=8, pin_memory=True)
    else:
        val_loader = None
    
    #initial model
    model = getattr(M, configs['model'])(**configs['model_configs'])

    trainer = Trainer(
        configs,
        model, 
        train_loader,
        val_loader,
        logger
    )

    trainer.train()


def main():
    parser = argparse.ArgumentParser(description='RS Semantic Segmentation for Training...')

    parser.add_argument('-c', '--config', default='./configs/unet_vaihingen_config.json',
                        type=str, help='Path to the config file.')
    parser.add_argument('-c_dir', '--config_dir', default=None,
                        type=str, help='Dir Path to config files if want to train with batch config files.')

    args = parser.parse_args()

    # train with batch files
    if args.config_dir:
        config_dir = Path(args.config_dir)
        config_paths = config_dir.glob('*.json')
        for config in config_paths:
            train(config)
    # train with single files
    else:
        train(args.config)
if __name__ == "__main__":
    main()