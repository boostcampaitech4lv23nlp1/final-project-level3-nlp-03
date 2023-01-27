from tqdm.auto import tqdm
from typing import *
import torch
import wandb
import numpy as np
import os
import time
import gc
import torch.nn as nn
from omegaconf import DictConfig
from utils import wandb_setting
from utils import check_dir
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, DataCollatorWithPadding, default_data_collator

class BaselineTrainer():
    """
    훈련과정입니다.
    """
    def __init__(
        self,
        config: Optional[Union[DictConfig, dict]] = None,
        model: Union[PreTrainedModel, nn.Module] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset:Optional[Dataset] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        compute_metrics = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        self.config = config

        self.device = torch.device('cuda')
        self._move_model_to_device(model, self.device)
        self.model = model
        
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        
        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers

        self.save_dir = config.save_dir
        self.best_model_epoch, self.val_loss_values, self.val_score_values = [], [], []

    def loop(self):
        self.is_wandb = wandb_setting(self.config)
        self.save_dir = check_dir(self.config.save_dir)
        self.compute_metrics._set_save_dir(self.save_dir)
        for epoch in range(self.config.train.max_epoch):
            standard_time = time.time()
            self._train_epoch()
            self._eval_epoch(epoch)
            if self.is_wandb:
                wandb.log({'epoch' : epoch, 'runtime(Min)' : (time.time() - standard_time) / 60})
        torch.cuda.empty_cache()
        gc.collect()
        del self.train_dataset, self.val_dataset
        best_model_name = self.select_best_model() if self.config.train.max_epoch > 1 else None
        if self.config.do_predict and best_model_name:
            self._predcit(best_model_name)
            print('='*50, 'Inference Complete!', '='*50, sep='\n\n')
            
    def predict(self, best_model=None):
        self.compute_metrics.make()
        self.compute_metrics._set_save_dir(self.save_dir)
        assert best_model, "path를 입력해주세요."
        self._predcit(best_model)
    
    def _train_epoch(self):
        gc.collect()
        self.model.train()
        epoch_loss = 0
        steps = 0
        pbar = tqdm(self._get_train_dataloader())
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            steps += 1
            inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'token_type_ids': batch['token_type_ids'].to(self.device),
                    'bbox': batch["bbox"].to(self.device),
                    'image': batch["image"].to(self.device),
                }
            start_logits, end_logits = self.model(**inputs)

            ignored_index = start_logits.size(1)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)

            # seq_len 길이만큼 boundary를 설정하여 seq_len 밖으로 벗어날 경우 벗어난 값을 최소값인 0(cls 토큰)으로 설정해줌
            start_positions.clamp(0, ignored_index)
            end_positions.clamp(0, ignored_index)

            # 각 start, end의 loss 평균
            criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
            loss = (criterion(start_logits, start_positions) + criterion(end_logits, end_positions)) / 2
                
            loss.backward()
            epoch_loss += loss.detach().cpu().numpy().item()
            
            self.optimizer.step()
            
            pbar.set_postfix({
                'loss' : epoch_loss / steps,
                'lr' : self.optimizer.param_groups[0]['lr'],
            })
            if self.is_wandb:
                wandb.log({'train_loss':epoch_loss/steps})
            # print(time.time()- standard)
        if self.lr_scheduler:
            self.lr_scheduler.step()
        pbar.close()

    def _eval_epoch(self, epoch):
        if epoch == 0: self.compute_metrics.make()
        val_loss = 0
        val_steps = 0
        pbar = tqdm(self._get_val_dataloader())
        start_logits_all, end_logits_all = [],[]
        with torch.no_grad():
            self.model.eval()
            for valid_batch in tqdm(pbar):
                val_steps += 1
                inputs = {
                        'input_ids': valid_batch['input_ids'].to(self.device),
                        'attention_mask': valid_batch['attention_mask'].to(self.device),
                        'token_type_ids': valid_batch['token_type_ids'].to(self.device),
                        'bbox': valid_batch["bbox"].to(self.device),
                        'image': valid_batch["image"].to(self.device),
                    }

                start_logits, end_logits = self.model(**inputs)

                ignored_index = start_logits.size(1)
                start_positions = valid_batch['start_positions'].to(self.device)
                end_positions = valid_batch['end_positions'].to(self.device)

                # seq_len 길이만큼 boundary를 설정하여 seq_len 밖으로 벗어날 경우 벗어난 값을 최소값인 0(cls 토큰)으로 설정해줌
                start_positions.clamp(0, ignored_index)
                end_positions.clamp(0, ignored_index)

                # 각 start, end의 loss 평균
                criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)
                loss = (criterion(start_logits, start_positions) + criterion(end_logits, end_positions)) / 2
                val_loss += loss.detach().cpu().numpy().item()

                start_logits_all.append(start_logits.detach().cpu().numpy())
                end_logits_all.append(end_logits.detach().cpu().numpy())
            pbar.close()
            
            val_loss /= val_steps

            start_logits_all = np.concatenate(start_logits_all)
            end_logits_all = np.concatenate(end_logits_all)

            anls_score = self.compute_metrics.compute_val(
                start_logits_all,
                end_logits_all,
                '0' + str(epoch+1) if epoch < 9 else epoch + 1
            )
            
            print(f"Epoch [{epoch+1}/{self.config.train.max_epoch}] Val_loss :", val_loss)
            print(f"Epoch [{epoch+1}/{self.config.train.max_epoch}] ANLS :", anls_score)
            
            if self.is_wandb:
                wandb.log({'epoch' : epoch+1, 'val_loss' : val_loss})
                wandb.log({'epoch' : epoch+1, 'ANLS' : anls_score})
                
            epoch = '0' + str(epoch+1) if epoch < 9 else epoch + 1
            torch.save(self.model.state_dict(), f'save/{self.save_dir}/epoch:{epoch}_model.pt')
            print('save checkpoint!')

        self.best_model_epoch.append(f'save/{self.save_dir}/epoch:{epoch}_model.pt')
        self.val_loss_values.append(val_loss)
        self.val_score_values.append(anls_score)

    def _predcit(self, best_model_name):
        checkpoint = torch.load(best_model_name)
        self.model.load_state_dict(checkpoint)
        self._move_model_to_device(self.model, self.device)
        self.model.eval()
        
        pbar = tqdm(self._get_test_dataloader())
        start_logits_all, end_logits_all = [], []
        with torch.no_grad():
            for test_batch in tqdm(pbar):
                inputs = {
                        'input_ids': test_batch['input_ids'].to(self.device),
                        'attention_mask': test_batch['attention_mask'].to(self.device),
                        'token_type_ids': test_batch['token_type_ids'].to(self.device),
                        'bbox': test_batch["bbox"].to(self.device),
                        'image': test_batch["image"].to(self.device),
                    }

                start_logits, end_logits = self.model(**inputs)

                start_logits_all.append(start_logits.detach().cpu().numpy())
                end_logits_all.append(end_logits.detach().cpu().numpy())
            pbar.close()

        start_logits_all = np.concatenate(start_logits_all)
        end_logits_all = np.concatenate(end_logits_all)
        self.compute_metrics.compute_test(start_logits_all, end_logits_all)

    def _move_model_to_device(self, model, device):
        model = model.to(device)

    def _get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: train 데이터셋을 넣어주세요.")
        
        train_dataset = self.train_dataset
        batch_size = self.config.train.batch_size
        sampler = RandomSampler(train_dataset)
        collate_fn = self.data_collator

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=self.config.train.num_workers
        )
        
    def _get_val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("Trainer: val 데이터셋을 넣어주세요.")
        
        val_dataset = self.val_dataset
        batch_size = self.config.train.batch_size
        sampler = SequentialSampler(val_dataset)
        collate_fn = self.data_collator

        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=self.config.train.num_workers
        )

    def _get_test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Trainer: test 데이터셋을 넣어주세요.")
        
        test_dataset = self.test_dataset
        batch_size = self.config.train.test_batch_size
        sampler = SequentialSampler(test_dataset)
        collate_fn = self.data_collator

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=self.config.train.num_workers
        )

    def select_best_model(self):
        # loss 기준
        best_model = self.best_model_epoch[np.array(self.val_loss_values).argmin()]
        # score 기준
        # best_model = self.best_model_epoch[np.array(self.val_score_values).argmax()]
        os.rename(best_model, best_model.split('.pt')[0] + '_best.pt')
        
        return best_model.split('.pt')[0] + '_best.pt'