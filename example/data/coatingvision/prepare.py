# -*- coding: utf-8 -*-
"""
CoatingVision 전극 코팅 결함 데이터 -> 교재 실습용 가공본 생성 스크립트
=======================================================================
원본 : Sampath, V. "CoatingVision: A Defect Dataset for Coating Manufacturing"
       figshare, DOI 10.6084/m9.figshare.29260121 (CC BY 4.0)
       CoatingVision.zip / 94,296,797 bytes

이 스크립트가 만드는 것
  1) labels_original.csv          원본 classification/labels.csv 를 그대로 복사(2,227행)
  2) coating_labels_tidy.csv      파일명 파싱으로 공정변수 컬럼을 붙인 tidy 표(2,227행)
  3) images_curated/              클래스 균형을 맞춘 큐레이션 이미지 54장(원본 파일명 유지)
  4) masks_curated/               위 54장에 대응하는 세그멘테이션 마스크 PNG
  5) detection_labels_curated/    위 54장 중 YOLO 박스가 존재하는 것만 .txt 복사
  6) curated_index.csv            큐레이션 서브셋 색인(선별 사유 포함)

사용법
  python prepare.py --zip  C:/path/to/CoatingVision.zip --out .
  python prepare.py --src  C:/path/to/extracted_root  --out .
      (--src 는 classification/ detection/ segmentation/ 이 들어있는 폴더)

필요 패키지: pandas, numpy, pillow
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------
LABELS = ["Surface_Crack", "Delamination", "Pinhole", "unclassified"]
DEFECT_LABELS = ["Surface_Crack", "Delamination", "Pinhole"]  # unclassified 제외

# 파일명 문법:  R<run>-<gap>um-<position>_frame_<frame>_patch_<patch>.png
#   예) R1-1100um-top-to-bottom-center-1_frame_220_patch_7.png
NAME_RE = re.compile(
    r"^R(?P<run>\d+)-(?P<coating_gap_um>\d+)um-(?P<position>.+?)"
    r"_frame_(?P<frame>\d+)_patch_(?P<patch>\d+)\.png$"
)
STEM_RE = re.compile(r"_frame_\d+_patch_\d+\.png$")

# 큐레이션 서브셋 선별 기준 (아래 select_curated_subset 참조)
MIN_STRATUM_SIZE = 20  # 이 개수 이상 확보된 라벨조합만 후보 계층으로 사용
PER_STRATUM = 6        # 계층당 뽑을 장수  -> 9계층 x 6장 = 54장
# 난수는 쓰지 않는다. 전 과정이 결정론적 정렬로만 이뤄져 몇 번을 돌려도 결과가 같다.


# --------------------------------------------------------------------------
# 입력 준비
# --------------------------------------------------------------------------
def resolve_source(zip_path, src_path):
    """zip 이 주어지면 임시 폴더에 풀고, 그 경로를 돌려준다."""
    if src_path:
        return src_path, None
    if not zip_path:
        sys.exit("오류: --zip 또는 --src 중 하나는 반드시 지정해야 합니다.")
    tmp = tempfile.mkdtemp(prefix="coatingvision_")
    print(f"[1/7] ZIP 해제 중 -> {tmp}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    return tmp, tmp


def check_layout(src):
    need = [
        "classification/labels.csv",
        "classification/images",
        "detection/labels",
        "segmentation/masks",
    ]
    for rel in need:
        if not os.path.exists(os.path.join(src, rel)):
            sys.exit(f"오류: 원본 구조가 예상과 다릅니다. 없음 -> {rel}")


# --------------------------------------------------------------------------
# tidy 표 만들기
# --------------------------------------------------------------------------
def build_tidy(src):
    """원본 labels.csv 를 읽어 공정변수/그룹/마스크 통계를 붙인 tidy DataFrame 반환."""
    df = pd.read_csv(os.path.join(src, "classification", "labels.csv"))

    # --- (a) 파일명 파싱 -> 공정변수 ---------------------------------------
    parsed = df["original_file_name"].str.extract(NAME_RE)
    if parsed.isna().any().any():
        bad = df.loc[parsed.isna().any(axis=1), "original_file_name"].tolist()[:5]
        sys.exit(f"오류: 파일명 문법에 맞지 않는 항목이 있습니다: {bad}")

    df["run"] = parsed["run"].astype(int)
    df["coating_gap_um"] = parsed["coating_gap_um"].astype(int)
    df["position"] = parsed["position"]
    df["frame"] = parsed["frame"].astype(int)
    df["patch"] = parsed["patch"].astype(int)

    # --- (b) 그룹 컬럼 (데이터 누수 방지용) --------------------------------
    # video_id : 하나의 코팅 영상(런+갭+위치). 같은 영상의 패치는 서로 매우 닮아 있다.
    # group_frame : 같은 프레임에서 잘라낸 패치 묶음. 가장 강한 누수 원인.
    df["video_id"] = df["original_file_name"].str.replace(STEM_RE, "", regex=True)
    df["group_frame"] = df["video_id"] + "_frame_" + df["frame"].astype(str)

    # --- (c) 라벨 요약 -----------------------------------------------------
    df["n_defect_types"] = df[DEFECT_LABELS].sum(axis=1)
    df["any_defect"] = (df["n_defect_types"] > 0).astype(int)
    # 라벨조합. '0010' 처럼 쓰면 read_csv 가 정수 10 으로 바꿔버려 앞자리 0 이
    # 사라진다. 하이픈으로 이어 붙여 항상 문자열로 읽히게 한다. 예 '1-0-1-0'
    df["label_combo"] = df[LABELS].astype(str).agg("-".join, axis=1)
    # 사람이 읽는 결함 조합명. 예 'Surface_Crack+Pinhole', 결함 없으면 'none'
    df["defect_set"] = [
        "+".join([c for c in LABELS if r[c] == 1]) or "none"
        for _, r in df[LABELS].iterrows()
    ]
    # 아무 라벨도 1이 아닌 행 = 결함도 unclassified 도 아님(정상 패치)
    df["is_unlabeled"] = (df[LABELS].sum(axis=1) == 0).astype(int)

    # --- (d) 디텍션 라벨 유무/박스 수 --------------------------------------
    # 원본에 data.yaml / classes.txt 가 없어 YOLO class id 의 이름이 명시돼 있지
    # 않다. 실측 교차검증 결과 id 0 은 Pinhole 양성 이미지 519장과 정확히 1:1,
    # id 1 은 unclassified 양성 78장과 정확히 1:1로 대응한다(불일치 0건).
    # 따라서 아래 매핑은 '데이터로 추론한 값'이며 원저자 공식 문서가 아니다.
    DET_ID_TO_NAME = {"0": "Pinhole", "1": "unclassified"}

    det_dir = os.path.join(src, "detection", "labels")
    n_boxes, det_ids, det_names = [], [], []
    for fn in df["file_name"]:
        p = os.path.join(det_dir, os.path.splitext(fn)[0] + ".txt")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            ids = sorted({ln.split()[0] for ln in lines})
            n_boxes.append(len(lines))
            det_ids.append("|".join(ids))
            det_names.append("|".join(DET_ID_TO_NAME.get(i, "id" + i) for i in ids))
        else:
            # 라벨 파일 자체가 없음 = YOLO 관례상 '박스 없음'.
            # 빈 문자열을 쓰면 read_csv 가 NaN 으로 읽어 초보자가 헷갈리므로 'none'.
            n_boxes.append(0)
            det_ids.append("none")
            det_names.append("none")
    df["n_boxes"] = n_boxes
    df["det_class_ids"] = det_ids
    df["det_class_names"] = det_names
    df["has_detection_label"] = (df["n_boxes"] > 0).astype(int)

    # --- (e) 마스크 면적 ---------------------------------------------------
    mask_dir = os.path.join(src, "segmentation", "masks")
    px = []
    for fn in df["file_name"]:
        p = os.path.join(mask_dir, os.path.splitext(fn)[0] + ".png")
        arr = np.array(Image.open(p).convert("L"))
        px.append(int((arr > 0).sum()))
    df["mask_pixels"] = px
    df["mask_area_frac"] = (df["mask_pixels"] / (640 * 480)).round(6)

    cols = [
        "file_name", "original_file_name",
        "run", "coating_gap_um", "position", "video_id", "frame", "patch", "group_frame",
        "Surface_Crack", "Delamination", "Pinhole", "unclassified",
        "n_defect_types", "any_defect", "label_combo", "defect_set", "is_unlabeled",
        "has_detection_label", "n_boxes", "det_class_ids", "det_class_names",
        "mask_pixels", "mask_area_frac",
    ]
    return df[cols]


# --------------------------------------------------------------------------
# 큐레이션 서브셋 선별
# --------------------------------------------------------------------------
def select_curated_subset(df):
    """
    선별 기준 (재현 가능, 난수 의존 없음)
    ------------------------------------------------------------------
    1) 계층(stratum) = 4개 라벨의 정확한 조합 문자열(label_combo).
       멀티라벨이므로 '단일 클래스 균형'이 성립하지 않는다. 따라서
       조합 자체를 계층으로 삼아 희귀 조합까지 교재에 노출시킨다.
    2) 전체에서 MIN_STRATUM_SIZE(=20)장 이상 확보된 조합만 후보로 채택.
       (표본이 6장도 안정적으로 안 나오는 초희귀 조합은 제외)
    3) 각 계층에서 PER_STRATUM(=6)장을 뽑되,
       - 코팅갭(600~1100um)이 최대한 골고루 섞이도록 갭을 라운드로빈으로 순회
       - 이미 뽑은 group_frame(같은 프레임)은 재사용하지 않는다
         -> 서브셋 안에서 프레임 중복으로 인한 누수/중복학습 방지
       - 동일 조건에서는 (coating_gap_um, video_id, frame, patch) 오름차순으로
         결정론적으로 선택
    4) 결과: 9계층 x 6장 = 54장 (요구 범위 40~60장 충족)
    """
    counts = df["label_combo"].value_counts()
    strata = sorted(
        [c for c in counts.index if counts[c] >= MIN_STRATUM_SIZE],
        key=lambda c: (-counts[c], c),
    )

    picked, used_frames, taken = [], set(), set()

    def take(pool, combo, chosen, enforce_unique_frame):
        """코팅갭을 한 바퀴 순회하며 갭당 최대 1장씩 뽑는다. 뽑았으면 True."""
        progressed = False
        for gap in sorted(pool["coating_gap_um"].unique()):
            if len(chosen) >= PER_STRATUM:
                break
            cand = pool[(pool["coating_gap_um"] == gap) & (~pool["file_name"].isin(taken))]
            if enforce_unique_frame:
                cand = cand[~cand["group_frame"].isin(used_frames)]
            if len(cand) == 0:
                continue
            row = cand.iloc[0]
            taken.add(row["file_name"])
            used_frames.add(row["group_frame"])
            r = row.to_dict()
            r["stratum"] = combo
            r["pick_reason"] = (
                f"combo={combo} gap={row['coating_gap_um']}um "
                f"video={row['video_id']} frame={row['frame']} "
                f"unique_frame={enforce_unique_frame}"
            )
            chosen.append(r)
            progressed = True
        return progressed

    for combo in strata:
        pool = df[df["label_combo"] == combo].sort_values(
            ["coating_gap_um", "video_id", "frame", "patch"], kind="mergesort"
        )
        chosen = []
        # 1단계: 프레임 중복 금지를 지키면서 갭을 골고루 순회
        while len(chosen) < PER_STRATUM and take(pool, combo, chosen, True):
            pass
        # 2단계: 그래도 모자랄 때만 프레임 중복 허용(사유를 pick_reason 에 기록)
        while len(chosen) < PER_STRATUM and take(pool, combo, chosen, False):
            pass
        picked.extend(chosen)

    return pd.DataFrame(picked)


def copy_subset(sub, src, out):
    img_out = os.path.join(out, "images_curated")
    msk_out = os.path.join(out, "masks_curated")
    det_out = os.path.join(out, "detection_labels_curated")
    for d in (img_out, msk_out, det_out):
        os.makedirs(d, exist_ok=True)

    n_det = 0
    for _, r in sub.iterrows():
        fn = r["file_name"]                       # 원본 파일명 그대로 유지
        stem = os.path.splitext(fn)[0]
        shutil.copy2(os.path.join(src, "classification", "images", fn),
                     os.path.join(img_out, fn))
        shutil.copy2(os.path.join(src, "segmentation", "masks", stem + ".png"),
                     os.path.join(msk_out, stem + ".png"))
        dp = os.path.join(src, "detection", "labels", stem + ".txt")
        if os.path.exists(dp):
            shutil.copy2(dp, os.path.join(det_out, stem + ".txt"))
            n_det += 1
    return img_out, msk_out, det_out, n_det


def folder_mb(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            tot += os.path.getsize(os.path.join(root, f))
    return tot / 1e6


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_path", default=None)
    ap.add_argument("--src", dest="src_path", default=None)
    ap.add_argument("--out", dest="out", default=".")
    args = ap.parse_args()

    src, tmp = resolve_source(args.zip_path, args.src_path)
    check_layout(src)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    print("[2/7] 원본 labels.csv 복사")
    shutil.copy2(os.path.join(src, "classification", "labels.csv"),
                 os.path.join(out, "labels_original.csv"))

    print("[3/7] tidy 표 생성(파일명 파싱 + 마스크 면적 계산)")
    tidy = build_tidy(src)
    tidy_path = os.path.join(out, "coating_labels_tidy.csv")
    tidy.to_csv(tidy_path, index=False, encoding="utf-8-sig")
    print(f"      -> {tidy_path}  ({len(tidy)}행 x {tidy.shape[1]}열)")

    print("[4/7] 큐레이션 서브셋 선별")
    sub = select_curated_subset(tidy)
    print(f"      -> {len(sub)}장 / 계층 {sub['stratum'].nunique()}개")

    print("[5/7] 이미지·마스크·박스 복사")
    img_out, msk_out, det_out, n_det = copy_subset(sub, src, out)

    print("[6/7] 큐레이션 색인 저장")
    idx_cols = ["file_name", "original_file_name", "stratum", "defect_set", "pick_reason",
                "run", "coating_gap_um", "position", "video_id", "frame", "patch",
                "group_frame"] + LABELS + ["n_boxes", "det_class_names",
                                           "mask_pixels", "mask_area_frac"]
    sub[idx_cols].to_csv(os.path.join(out, "curated_index.csv"),
                         index=False, encoding="utf-8-sig")

    print("[7/7] 요약")
    print(f"      images_curated : {len(os.listdir(img_out))}장  {folder_mb(img_out):.2f} MB")
    print(f"      masks_curated  : {len(os.listdir(msk_out))}장  {folder_mb(msk_out):.2f} MB")
    print(f"      detection_labels_curated : {n_det}개")
    print(f"      배포폴더 전체   : {folder_mb(out):.2f} MB")

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
