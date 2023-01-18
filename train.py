import torch
import argparse
from omegaconf import OmegaConf

from transformers import AutoTokenizer
from datasets import load_dataset

import data_process  as Data_process
import trainer as Trainer
import model as Model

import torch.optim as optim
import utils.loss as Criterion
import utils.metric as Metric

from utils.wandb_setting import wandb_setting
from utils.seed_setting import seed_setting

def main(config):
    seed_setting(config.train.seed)
    torch.cuda.empty_cache()
    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    
    print('='*50,f'현재 적용되고 있는 전처리 클래스는 {config.data.preprocess}입니다.', '='*50, sep='\n\n')
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, use_fast=True)
    train_data = load_dataset(config.data.train_load, split='train')
    val_data = load_dataset(config.data.val_load, split='val')
    test_data = load_dataset(config.data.test_load, split='test')

    preprocess = getattr(Data_process, config.data.preprocess)(tokenizer, 512, 128)
    
    train_dataset = train_data.map(preprocess.train,
                remove_columns=train_data.column_names,
                num_proc=4,
                batched=True,
                )

    val_dataset = val_data.map(preprocess.val,
                remove_columns=val_data.column_names,
                num_proc=4,
                batched=True,
                )
    
    # test_dataset = test_data.map(preprocess.test,
    #             remove_columns=test_data.column_names,
    #             num_proc=2,
    #             batched=True,
    #             # cache_file_name=True
    #             )
    
    # valid_dataset = valid_dataset.remove_columns(["question_id", "offset_mapping"])
    # train_dataset.set_format("torch"), val_dataset.set_format("torch")
    # data_collator = DataCollatorWithPadding(tokenizer)

    # train_dataloader = DataLoader(train_dataset, batch_size= config.train.batch_size, collate_fn=data_collator, pin_memory=True, shuffle=True)
    # valid_dataloader = DataLoader(val_dataset, batch_size= config.train.batch_size, collate_fn=data_collator, pin_memory=True, shuffle=False)
    
    # metric = getattr(Metric, config.model.metric_class)(
    #     dataset = val_dataset,
    #     raw_data = val_data,
    #     n_best_size = config.train.n_best_size,
    #     max_answer_length = config.train.max_answer_length,
    #     save_dir = config.save_dir,
    #     mode = 'train',
    #     tokenizer = tokenizer
    # )
    # # 모델 아키텍처를 불러옵니다.
    # print('='*50,f'현재 적용되고 있는 모델 클래스는 {config.model.model_class}입니다.', '='*50, sep='\n\n')
    # model = getattr(Model, config.model.model_class)(
    #     model_name = config.model.model_name,
    #     num_labels=2,
    #     dropout_rate = config.train.dropout_rate,
    #     )
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline')
    args, _ = parser.parse_known_args()
    ## ex) python3 train.py --config baseline
    
    config = OmegaConf.load(f'./configs/{args.config}.yaml')
    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')

    main(config)