"""기본 정보 대비 일출·일몰만 추가하여 같은 설정으로 학습·검증·시험한다."""
from run_external_experiment import main

if __name__=='__main__':
    main(feature_set='solar')
