"""공정 데이터 처리용 공용 함수 모음 (process_utils.py)실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""


def calc_defect_rate(defect_count, total_count):
    """불량률(%)을 계산한다."""
    if total_count == 0:
        return 0.0
    return defect_count / total_count * 100
