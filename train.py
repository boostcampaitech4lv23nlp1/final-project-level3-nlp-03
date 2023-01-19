import torch
import argparse
from omegaconf import OmegaConf

from transformers import AutoTokenizer
from datasets import load_dataset
from utils.seed_setting import seed_setting

import data_process  as Data_process
import trainer as Trainer
import model as Model

import torch.optim as optim
import utils.metric as Metric


def main(config):
    seed_setting(config.train.seed)
    torch.cuda.empty_cache()
    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, use_fast=True)
    train_data = load_dataset(config.data.train_load, split='train')
    val_data = load_dataset(config.data.val_load, split='val')
    test_data = load_dataset(config.data.test_load, split='test')

    print('='*50,f'현재 적용되고 있는 전처리 클래스는 {config.data.preprocess}입니다.', '='*50, sep='\n\n')
    preprocess = getattr(Data_process, config.data.preprocess)(tokenizer, max_length=512, stride=128)
    train_dataset = train_data.map(preprocess.train, remove_columns=train_data.column_names, num_proc=4, batched=True)
    val_dataset = val_data.map(preprocess.val, remove_columns=val_data.column_names, num_proc=2, batched=True)
    test_dataset = test_data.map(preprocess.test, remove_columns=test_data.column_names, num_proc=2, batched=True)
    
    print('='*50,f'현재 적용되고 있는 메트릭 클래스는 {config.model.metric_class}입니다.', '='*50, sep='\n\n')
    compute_metrics = getattr(Metric, config.model.metric_class)(
        val_data = val_data,
        val_dataset = val_dataset,
        test_data = test_data,
        test_dataset = test_dataset,
        n_best_size = config.train.n_best_size,
        max_answer_length = config.train.max_answer_length
        )
    train_dataset.set_format("torch"), val_dataset.set_format("torch"), test_dataset.set_format("torch")
    
    print('='*50,f'현재 적용되고 있는 모델 클래스는 {config.model.model_class}입니다.', '='*50, sep='\n\n')
    model = getattr(Model, config.model.model_class)(
        model_name = config.model.model_name,
        num_labels=2,
        dropout_rate = config.train.dropout_rate,
        )
    
    optimizer = getattr(optim, config.model.optimizer)(model.parameters(), lr=config.train.learning_rate)
    lr_scheduler = None
    
    print('='*50,f'현재 적용되고 있는 트레이너는 {config.model.trainer_class}입니다.', '='*50, sep='\n\n')
    trainer = getattr(Trainer, config.model.trainer_class)(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer,lr_scheduler)
    )
    
    trainer.loop()
    
    ## trainer.predict(path) 오직 predict (inference만 하고싶은 경우)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline')
    args, _ = parser.parse_known_args()
    ## ex) python3 train.py --config baseline
    
    config = OmegaConf.load(f'./configs/{args.config}.yaml')
    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')

    main(config)