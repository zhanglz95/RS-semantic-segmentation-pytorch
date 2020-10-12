import torch
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import loss as L
import optimizer as O
from eval import eval_net

class Trainer():
    def __init__(self, configs, model, train_loader, val_loader, logger):
        self.model = model
        self.configs = configs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        # general name
        self.model_name = self.configs['model']
        self.dataset_name = self.configs['dataset_name']

        # general config
        self.num_classes = self.configs['model_configs']['num_classes']
        self.epochs = self.configs['epochs']
        self.lr = self.configs['optimizer_configs']['lr']
        self.bs = self.configs['batchsize']
        self.sc = self.configs['dataset_configs']['scale']
        self.val_interval = self.configs['val_interval']

        # Device
        if self.configs['use_gpu']:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        # resume checkpoint
        self.resume_path = self.configs['resume_path']
        if(self.resume_path):
            self.resume_checkpoint(self.configs['resume_path'])

        # tensorboard config
        self.tb_writer = SummaryWriter(comment=f'MODEL_{self.model_name}_LR_{self.lr}_BS_{self.bs}_SCALE_{self.sc}',
                                       log_dir=self.configs['tb_dir'])
        self.global_step = 0

        # loss function
        self.criterion = getattr(L, self.configs['loss'])

        # optimizer
        self.optimizer = getattr(O, self.configs['optimizer'])(self.model.parameters(), **self.configs['optimizer_configs'])

        # scheduler
        if(self.configs['scheduler'] == ''):
            # self.scheduler = scheduler above
            pass
        else:
            # default scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min' if self.num_classes > 1 else 'max', patience=2)

        # checkpoint save path
        self.cp_path = self.configs['checkpoint_path']
        if not Path(self.cp_path).exists():
            os.makedirs(self.cp_path)

    def train(self):
        for epoch in range(self.epochs):
            # train one epoch
            self.model.train()
            
            with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.epochs}', unit='batch') as pbar:
                for imgs, true_masks in self.train_loader:
                    self.global_step += 1
                    imgs = imgs.to(device=self.device, dtype=torch.float32)
                    mask_type = torch.float32 if self.num_classes == 1 else torch.long
                    true_masks = true_masks.to(device=self.device, dtype=mask_type)

                    pred_masks = self.model(imgs)
                    loss = self.criterion(pred_masks, true_masks)
                    self.tb_writer.add_scalar('loss/train', loss.item(), self.global_step)

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.optimizer.step()

                    pbar.update()
                    if self.global_step % self.val_interval == 0:
                        val_score = eval_net(self.model, self.val_loader, self.num_classes, self.device)
                        self.scheduler.step(val_score)
                        self.tb_writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)

                        self.logger.info('Validation Score: {}'.format(val_score))
                        self.tb_writer.add_scalar('score/test', val_score, self.global_step)

                        self.tb_writer.add_images('images', imgs, self.global_step)
                        if self.num_classes == 1:
                            self.tb_writer.add_images('masks/true', true_masks, self.global_step)
                            self.tb_writer.add_images('masks/pred', torch.sigmoid(pred_masks) > 0.5, self.global_step)
                        self.save_checkpoint(f'{self.model_name}_{self.dataset_name}', self.global_step)

    def save_checkpoint(self, name, epoch):
        filename = Path(self.cp_path) / f'{name}_{epoch}.pth'
        torch.save(self.model.state_dict(), filename)

    def resume_checkpoint(self, resume_path):
        self.model.load_state_dict(torch.load(resume_path))