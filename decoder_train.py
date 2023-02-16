from utils.seed_setting import seed_setting
from datasets import load_dataset
from typing import *

from transformers import AutoTokenizer, AutoModel, BartConfig
from omegaconf import OmegaConf
import torch
import argparse
import torch.optim as optim
import data_process as Data_process
import utils.metric as Metric
import model as Model
import trainer as Trainer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def collate(data):
    return {
            'input_ids': torch.stack([d['input_ids'] for d in data],dim=0),
            'attention_mask': torch.stack([d['attention_mask'] for d in data],dim=0),
            'token_type_ids': torch.stack([d['token_type_ids'] for d in data],dim=0),
            'bbox': torch.stack([d['bbox'] for d in data],dim=0),
            'image': torch.cat([d['image'] for d in data],dim=0),
            'labels' : torch.stack([d['labels'] for d in data], dim=0),
            'decoder_input_ids' : torch.stack([d['decoder_input_ids'] for d in data], dim=0),
}

def main(config):
    seed_setting(config.train.seed)
    torch.cuda.empty_cache()
    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": sorted(["<sep/>", "<s_question>", "</s_question>", "<s_answer>", "</s_answer>"])}) # add q+a token
    tokenizer.eos_token_id=2
    train_data = load_dataset(config.data.train_load, split='train')
    val_data = load_dataset(config.data.val_load, split='val')
    test_data = load_dataset(config.data.test_load, split='test')
    ##
    train_data = train_data.select(range(134))
    val_data = val_data.select(range(27))
    test_data = test_data.select(range(4))
    ##
    print('='*50,f'현재 적용되고 있는 전처리 클래스는 {config.data.preprocess}입니다.', '='*50, sep='\n\n')
    preprocess = getattr(Data_process, config.data.preprocess)(tokenizer, max_length=config.train.max_length, stride=config.train.stride, boundary = config.data.boundary)

    train_dataset = train_data.map(preprocess.train, remove_columns=train_data.column_names, batched=True, num_proc=8)
    val_dataset = val_data.map(preprocess.val, remove_columns=val_data.column_names, batched=True, num_proc=8)
    test_dataset = test_data.map(preprocess.test, remove_columns=test_data.column_names, batched=True, num_proc=8)
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

    auth_token = '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'
    encoder = AutoModel.from_pretrained('hundredeuk2/bm_mrc', use_auth_token = auth_token,
                                        num_labels = 2)
    decoder = getattr(Model, config.model.model_class).from_pretrained("facebook/bart-large",
        config = BartConfig(
                    decoder_layers = 4,
                    is_decoder=True,
                    is_encoder_decoder=False,
                    add_cross_attention=True,
                    # max_position_embeddings = 1536,
                    # vocab_size=len(tokenizer),
                    scale_embedding=True,
                    add_final_layer_norm=True),
                    ignore_mismatched_sizes = True,
    )
    
    optimizer = getattr(optim, config.model.optimizer)(decoder.parameters(), lr=config.train.learning_rate)
    lr_scheduler = None
    
    print('='*50,f'현재 적용되고 있는 트레이너는 {config.model.trainer_class}입니다.', '='*50, sep='\n\n')
    trainer = getattr(Trainer, config.model.trainer_class)(
        config=config,
        encoder=encoder,
        decoder = decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator = collate,
        optimizers=(optimizer,lr_scheduler),
        tokenizer = tokenizer
    )
    
    trainer.loop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='decoder')
    args, _ = parser.parse_known_args()
    ## ex) python3 train.py --config baseline

    config = OmegaConf.load(f'./configs/{args.config}.yaml')
    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')

    main(config)