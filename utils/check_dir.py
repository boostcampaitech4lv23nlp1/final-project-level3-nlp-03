import os
from time import time

def check_dir(save_directory):
    if not os.path.isdir(f'save/{save_directory}'):
        print('모델을 저장할 디렉토리를 생성합니다.')
        os.makedirs(f'save/{save_directory}')
        if os.path.isdir(f'save/{save_directory}'):
            print('생성완료')

        return save_directory

    elif os.path.isdir(f'save/{save_directory}') and not bool((os.listdir(f'save/{save_directory}'))):
        return save_directory

    elif os.path.isdir(f'save/{save_directory}') and bool(os.listdir(f'save/{save_directory}')):
        print('='*50,f'파일이 덮여씌여지는 것을 방지하기 위해 새로운 디렉토리를 생성합니다. 필요없는 모델은 확인 후 삭제해주세요.', '='*50, sep='\n\n')
        number = 1
        if number < 10:
            number = '0' + str(number)
        next_directory = f'{save_directory}_{number}'

        # next_directory가 없을때까지 다음 번호를 부여하면서 체크
        while next_directory in os.listdir('save'):
            number += 1
            if number < 10:
                number = '0' + str(number)
            next_directory = f'{next_directory}_{number}'

        os.makedirs(f'save/{next_directory}')
        if os.path.isdir(f'save/{next_directory}'):
            print('생성완료')
    
        return next_directory