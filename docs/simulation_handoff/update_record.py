"""Build the manually maintained research handoff. Chat extraction is retired."""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--pending-answer')
    args = parser.parse_args()
    if args.pending_answer:
        parser.error('대화 원문 추출은 중단했습니다. research_record.tex를 실험 단위로 갱신하세요.')
    if not args.compile:
        print('연구 시도·조건·결과·판단을 research_record.tex에 갱신한 뒤 --compile로 빌드합니다.')
        return
    executable = shutil.which('xelatex') or '/Library/TeX/texbin/xelatex'
    for _ in range(2):
        result = subprocess.run([executable, '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                                cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout[-5000:])
            raise SystemExit(result.returncode)
    for name in ['연구진행_인계자료.pdf', '대화_문답기록.pdf']:
        shutil.copy2(ROOT / 'main.pdf', ROOT / name)
    print('연구 진행·인계자료 PDF를 갱신했습니다.')

if __name__ == '__main__':
    main()
