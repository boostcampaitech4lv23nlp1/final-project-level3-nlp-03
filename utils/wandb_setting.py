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
    
    try:
        config_w = disentangle(dict(), config)
        wandb.login()
        print(OmegaConf.to_yaml(config))
        
        if config.wandb.use_wandb:
            wandb.init(entity=config.wandb.entity,
                    project=config.wandb.project,
                    group=config.wandb.group,
                    name=config.wandb.experiment,
                    mode="online" if config.wandb.online else 'offline',
                    config=config_w)
            return True
        else:
            print('='*20, 'use_wandb = False이므로 wandb를 사용하지 않습니다.', '='*20, sep='\n')
            return False
    except:
        print('='*20, '에러가 발생하여 완디비 설정을 off하고 진행합니다.', '='*20, sep='\n')
        return False