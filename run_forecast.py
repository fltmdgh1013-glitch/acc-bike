"""Reproduce coursework results from the original CSV."""
import argparse
from solution import main as train
from analyze_forecast import main as analyze
from sequence_models import run
from compare_sequences import main as compare
from evaluate_timing import main as evaluate_timing
from korean_labels import export_korean_csvs


def main(include_external=False, include_air=False):
    train()
    run('RNN')
    run('Transformer')
    analyze()
    compare()
    evaluate_timing()
    export_korean_csvs()
    if include_external or include_air:
        from run_external_experiment import main as run_external
        from analyze_external_experiment import main as analyze_external
        run_external()
        analyze_external()
    if include_air:
        from run_air_quality_experiment import main as run_air
        from analyze_air_quality_experiment import main as analyze_air
        run_air()
        analyze_air()
        run_air(include_calendar=False)
        analyze_air(include_calendar=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-external', action='store_true', help='외부 달력 정보 추가 전후 실험도 실행')
    parser.add_argument('--with-air', action='store_true', help='달력 실험에 이어 미세먼지 추가 실험도 실행')
    args = parser.parse_args()
    main(include_external=args.with_external, include_air=args.with_air)
