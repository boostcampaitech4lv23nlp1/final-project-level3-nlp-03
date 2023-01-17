import wandb
from omegaconf import DictConfig, OmegaConf

def disentangle(container:dict, config:DictConfig):
    """
    중첩이 되어있는 다중 딕셔너리를 하나의 딕셔너리로 풀어주는 코드입니다.
    """
    for key, value in config.items():
        if type(value) == DictConfig:
            disentangle(container, value)
        else:
            container[key] = value

    return container

def wandb_setting(config:DictConfig, sweep:bool=False):
    """
    wandb 설정을 yaml 파일에 넣어주세요
    ex)
    wandb:
        entity: nlp6
        project: test-project
        group: test-group
        experiment: test-experiment
    """

    assert config.get('wandb'), "Please write wandb configuration in config.yaml file"
    
    config_w = disentangle(dict(), config)
    wandb.login()
    
    print(OmegaConf.to_yaml(config))

    wandb.init(entity=config.wandb.entity,
                project=config.wandb.project,
                group=config.wandb.group,
                name=config.wandb.experiment,
                mode="online" if config.wandb.online else 'offline',
                config=config_w)