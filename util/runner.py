"""
    Runner for training and testing models.
    Tianyu Liu
"""
import json
import sys
import os
import logging
import random
from pathlib import Path

import numpy as np

import torch

from torch.optim import AdamW

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import time
from os.path import join
from datetime import datetime

import util
from util.tensorize_coref import CorefDataProcessor, coref_collate_fn
from util.tensorize_ner import NERDataProcessor, ner_collate_fn
from util.tensorize_ere import EREDataProcessor, ere_collate_fn

from metrics import CorefEvaluator, MentionEvaluator

from models.model_coref import CorefWrapper
from models.model_ner import NERWrapper
from models.model_ere import EREWrapper

import transformers
transformers.utils.logging.set_verbosity_error()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__file__)


class Runner:
    def __init__(self, config_file, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed
        # Set up seed
        if seed:
            util.set_seed(seed)
        # Set up config
        self.config = util.initialize_config(config_name, config_file=config_file)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up device
        self.device = 'cpu' if gpu_id is None else gpu_id
        # Use mixed precision training
        self.use_amp = self.config['use_amp']

        # Set up data
        if self.config['task'].lower() == 'coref':
            self.data = CorefDataProcessor(self.config)
            self.collate_fn = coref_collate_fn
            self.wrapper_class_fn = CorefWrapper
        elif self.config['task'].lower() == 'ner':
            self.data = NERDataProcessor(self.config)
            self.collate_fn = ner_collate_fn
            self.wrapper_class_fn = NERWrapper
        elif self.config['task'].lower() == 'ere':
            self.data = EREDataProcessor(self.config)
            self.collate_fn = ere_collate_fn
            self.wrapper_class_fn = EREWrapper


    def initialize_model(self, saved_suffix=None, continue_training=False):
        wrapper = self.wrapper_class_fn(self.config, self.device, pretrained_path=saved_suffix)
        start_epoch = 0
        if saved_suffix:
            wrapper, start_epoch = self.load_model_checkpoint(wrapper, saved_suffix, continue_training=continue_training)   
        return wrapper, start_epoch


    def train(self, wrapper, continued=False, start_epoch=0):
        logger.info('Config:')
        for name, value in self.config.items():
            logger.info('%s: %s' % (name, value))
        logger.info('Model parameters:')

        wrapper.parallel_preparation_training()

        epochs, grad_accum = self.config['num_epochs'], self.config['gradient_accumulation_steps']
        batch_size = self.config['batch_size']

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        examples_seen, examples_unseen = self.data.get_test_tensor_examples()
        stored_info = self.data.get_stored_info()

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // (grad_accum * batch_size)
        if not continued:
            self.optimizer = self.get_optimizer(wrapper)
            self.scheduler = self.get_scheduler(self.optimizer, total_update_steps)

        # Get model parameters for grad clipping
        plm_param, task_param = wrapper.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)
        logger.info('Starting step: %d' % self.scheduler._step_count)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1, max_f1_test = 0, 0
        best_ckpt_step_count = 0
        start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        trainloader = DataLoader(
            examples_train, batch_size=batch_size, shuffle=True, 
            num_workers=0,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

        for epo in range(start_epoch, epochs):
            logger.info("*******************EPOCH %d*******************" % epo)
            for doc_key, example in trainloader:
                # Forward pass
                wrapper.train()
                example_gpu = {}
                for k, v in example.items():
                    if v is not None:
                        example_gpu[k] = v.to(self.device)
                with torch.cuda.amp.autocast(
                    enabled=self.use_amp, dtype=torch.bfloat16
                ):
                    loss = wrapper(**example_gpu) / grad_accum
                # Backward; accumulate gradients and clip by grad norm
                loss.backward()
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if self.config['max_grad_norm']:
                        norm_plm = torch.nn.utils.clip_grad_norm_(
                            plm_param,
                            self.config['max_grad_norm'],
                            error_if_nonfinite=False
                        )
                        norm_task = torch.nn.utils.clip_grad_norm_(
                            task_param,
                            self.config['max_grad_norm'],
                            error_if_nonfinite=False
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if self.scheduler._step_count % self.config['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / self.config['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info(
                            'Step %d: avg loss %.2f; steps/sec %.2f' %
                            (self.scheduler._step_count, avg_loss,
                            self.config['report_frequency'] / (end_time - start_time))
                        )
                        start_time = end_time

                    # Evaluate
                    if self.scheduler._step_count % self.config['eval_frequency'] == 0:
                        logger.info('Dev')

                        f1, _ = self.evaluate(
                            wrapper, examples_dev, stored_info, self.scheduler._step_count
                        )
                        logger.info('Test')
                        f1_test = 0.
                        if f1 > max_f1:
                            max_f1 = max(max_f1, f1)
                            max_f1_test = 0. 
                            self.save_model_checkpoint(
                                wrapper, self.optimizer, self.scheduler, self.scheduler._step_count, epo
                            )
                            best_ckpt_step_count = self.scheduler._step_count

                        logger.info('Eval max f1: %.2f' % max_f1)
                        logger.info('Test max f1: %.2f' % max_f1_test)
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % self.scheduler._step_count)
        logger.info('**********Testing**********')

        def save_metrics2file(path: str, filename: str, metrics):
            metrics_path = (
                    Path(path) / filename
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving metrics to {metrics_path}")
            with metrics_path.open("w") as f:
                json.dump(
                    {"test_metrics": metrics}, f,
                )
        
        # load best model
        wrapper, _ = self.load_model_checkpoint(wrapper, suffix=self.name_suffix + "_" + best_ckpt_step_count) 
        logger.info('**********Test**********')
        _, metrics = self.evaluate(wrapper, examples_test, stored_info, -1, predict=False)
        save_metrics2file(self.config['log_dir'], 'metrics.json', metrics)
        logger.info('**********Seen**********')
        _, seen_metrics = self.evaluate(wrapper, examples_seen, stored_info, -1, predict=False)
        save_metrics2file(self.config['log_dir'], 'metricsSeen.json', seen_metrics)
        logger.info('**********Unseen**********')
        _, unseen_metrics = self.evaluate(wrapper, examples_unseen, stored_info, -1, predict=False)
        save_metrics2file(self.config['log_dir'], 'metricsUnseen.json', unseen_metrics)
        logger.info('**********Training**********')
        _, _ = self.evaluate(wrapper, examples_train, stored_info, -1, predict=False)

        return

    def evaluate(
        self, wrapper, tensor_examples, stored_info, step, predict=False
    ):
        # use different evaluator for different task
        # should return two values: f1, metrics
        # f1 is used for model selection, the higher the better
        raise NotImplementedError()


    def get_optimizer(self, wrapper):
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        plm_param, task_param = wrapper.get_params(named=True)

        grouped_param = [
            {
                'params': [p for n, p in plm_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['plm_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in plm_param if any(nd in n for nd in no_decay)],
                'lr': self.config['plm_learning_rate'],
                'weight_decay': 0.0
            }, {
                'params': [p for n, p in task_param],
                'lr': self.config['task_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        if self.config["optimizer"].lower() == 'adamw':
            opt_class = AdamW

        logger.info(opt_class)
        optimizer = opt_class(
            grouped_param,
            lr=self.config['plm_learning_rate'],
            eps=self.config['adam_eps'],
            fused=True
        )
        return optimizer

    def get_scheduler(self, optimizer, total_update_steps):
        # Only warm up plm lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        lr_lambda_plm = util.get_scheduler_lambda(
            self.config['plm_scheduler'], warmup_steps, total_update_steps)
        lr_lambda_task = util.get_scheduler_lambda(
            self.config['task_scheduler'], 0, total_update_steps)

        scheduler = LambdaLR(optimizer, [
            lr_lambda_plm, # parameters with decay
            lr_lambda_plm, # parameters without decay
            lr_lambda_task
        ])
        return scheduler

    def save_model_checkpoint(self, wrapper, optimizer, scheduler, step, current_epoch):
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save({
            'current_epoch': current_epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, path_ckpt)
        wrapper.model.save_pretrained(self.config['log_dir'])
        logger.info('Saved model, optmizer, scheduler to %s' % path_ckpt)
        return

    def load_model_checkpoint(self, wrapper, suffix, continue_training=False):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')

        if not os.path.exists(path_ckpt):
            wrapper.model = wrapper.model.from_pretrained(suffix)
            print("loaded suffix model")
        else:
            wrapper.model = wrapper.model.from_pretrained(self.config['log_dir'])
            print("loaded log_dir model")
        if continue_training:
            checkpoint = torch.load(path_ckpt, map_location=torch.device('cpu'))
            self.optimizer = self.get_optimizer(wrapper)

            epochs, grad_accum = self.config['num_epochs'], self.config['gradient_accumulation_steps']
            batch_size = self.config['batch_size']
            total_update_steps = len(self.data.get_tensor_examples()[0]) *\
                                    epochs // (grad_accum * batch_size)
            self.scheduler = self.get_scheduler(self.optimizer, total_update_steps)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            current_epoch = checkpoint['current_epoch']

            logger.info('Loaded model, optmizer, scheduler from %s' % path_ckpt)
            return wrapper, current_epoch
        else:
            return wrapper, -1