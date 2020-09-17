import time
from datetime import datetime
from glob import glob

import torch
import os

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from metrics.meter import AverageMeter, ClassificationMeter


class Fitter:

    def __init__(self, model, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        self.model = model

        self.device = config.device
        self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.writer = SummaryWriter()  # tensorboard writer
        self.log(f'Fitter prepared. Device is {self.device}')

    def get_model(self):
        return self.model

    def fit(self, train_loader, val_loader):
        for epoch in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f"\n{timestamp}\n LR: {lr}")
            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            self.writer.add_scalar("Loss/Train", summary_loss.avg, self.epoch)

            t = time.time()
            summary_loss, summary_precision, summary_recall, summary_f1 = self.validation(val_loader)
            self.writer.add_scalar("Loss/Validation", summary_loss.avg, self.epoch)
            self.writer.add_scalar("Metrics/Precision", summary_precision, self.epoch)
            self.writer.add_scalar("Metrics/Recall", summary_recall, self.epoch)
            self.writer.add_scalar("Metrics/F1", summary_f1, self.epoch)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}\n'
                + f"Precision: {summary_precision}, Recall: {summary_recall}, F1: {summary_f1}")
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)
            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        classMeter = ClassificationMeter()
        summary_precision, summary_recall, summary_f1 = 0, 0, 0
        t = time.time()
        for step, (images, targets) in tqdm(enumerate(val_loader)):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = images.to(self.device)
                targets = targets.to(self.device)
                pred = self.model(images)
                loss = self.criterion(pred, targets)
                precision, recall, f1 = classMeter.computePRF1(pred.argmax(dim=1), targets)
                summary_precision += precision
                summary_recall += recall
                summary_f1 += f1
                summary_loss.update(loss.detach().item(), self.config.batch_size)

        summary_precision /= len(val_loader)
        summary_recall /= len(val_loader)
        summary_f1 /= len(val_loader)
        return summary_loss, summary_precision, summary_recall, summary_f1

    def train_one_epoch(self, train_loader) -> AverageMeter:
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()

        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}'
                    )
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                loss = self.criterion(self.model(images), targets)
                loss.backward()
                summary_loss.update(loss.detach().item(), self.config.batch_size)

                self.optimizer.step()

                if self.config.step_scheduler:
                    self.scheduler.step()
        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
