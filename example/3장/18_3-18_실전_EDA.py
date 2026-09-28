# -*- coding: utf-8 -*-
"""3장 예제 — 3.18 실전 EDA — 파일명에서 공정 변수를 캐낸다 (필수, 25분)

교재 출처 : manuscript/31_ch3_eda_fe.md
실행 방법 : example 폴더에서  python "3장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 15_3-15_연습문제.py, 16_3-16_실전_정제_①_현장_엑셀_워크북_해체하기.py, 17_3-17_실전_정제_②_통계_규칙이_못_잡는_물리적_불가값.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --- 실행 준비 (교재에서는 앞 절에서 이미 실행한 코드) --------------------
import pandas as pd

# --------------------------------------------------------------------
# [이 절에서 쓸 데이터 — 코팅 결함 이미지 라벨]
# --------------------------------------------------------------------
lab = pd.read_csv('data/coatingvision/labels_original.csv')
print("모양:", lab.shape)
print(lab.head(3).to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   모양: (2227, 6)
#                                       original_file_name   file_name  Surface_Crack  Delamination  Pinhole  unclassified
#                    R7-700um-middle_frame_489_patch_1.png image_1.jpg              1             0        0             0
#   R1-1100um-top-to-bottom-center-1_frame_220_patch_7.png image_2.jpg              1             0        0             0
#      R1-1100um-top-to-bottom-center_frame_85_patch_1.png image_3.jpg              1             0        1             0

# --------------------------------------------------------------------
# [먼저 라벨 구조부터 — 멀티라벨이라는 낯선 형식]
# --------------------------------------------------------------------
labels = ['Surface_Crack', 'Delamination', 'Pinhole', 'unclassified']
print(pd.DataFrame({'양성수': lab[labels].sum(),
                    '비율(%)': (lab[labels].mean() * 100).round(1)}).to_string())
print()
print("한 장에 붙은 라벨 수:")
print(lab[labels].sum(axis=1).value_counts().sort_index().to_string())

# 교재 실행 결과 ------------------------------------------------------
#                   양성수  비율(%)
#   Surface_Crack  1947   87.4
#   Delamination    203    9.1
#   Pinhole         519   23.3
#   unclassified     78    3.5
#   한 장에 붙은 라벨 수:
#   0      44
#   1    1657
#   2     488
#   3      38

# --------------------------------------------------------------------
# [정규식 5분 — 문자열에서 규칙을 뽑아내는 도구]
# --------------------------------------------------------------------
pat = (r'^R(?P<run>\d+)-(?P<gap>\d+)um-(?P<position>.+?)'
       r'_frame_(?P<frame>\d+)_patch_(?P<patch>\d+)\.png$')

ext = lab['original_file_name'].str.extract(pat)
print("추출 실패:", int(ext['run'].isna().sum()), "건")
print(ext.head(3).to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   추출 실패: 0 건
#   run  gap               position frame patch
#     7  700                 middle   489     1
#     1 1100 top-to-bottom-center-1   220     7
#     1 1100   top-to-bottom-center    85     1

# --------------------------------------------------------------------
# [정규식 5분 — 문자열에서 규칙을 뽑아내는 도구]
# --------------------------------------------------------------------
lab['run'] = ext['run'].astype(int)
lab['gap'] = ext['gap'].astype(int)
lab['frame'] = ext['frame'].astype(int)
lab['position'] = ext['position'].str.replace('top-to-bottom-center', 'TBC')
lab['video_id'] = ('R' + lab['run'].astype(str) + '-'
                   + lab['gap'].astype(str) + '-' + lab['position'])
lab['frame_id'] = lab['video_id'] + '_f' + lab['frame'].astype(str)

print("런:", sorted(lab['run'].unique().tolist()))
print("코팅 갭:", sorted(lab['gap'].unique().tolist()))
print("영상:", lab['video_id'].nunique(), "개 · 프레임:", lab['frame_id'].nunique(),
      "개 · 패치:", len(lab), "개")

# 교재 실행 결과 ------------------------------------------------------
#   런: [1, 7]
#   코팅 갭: [600, 700, 800, 900, 1000, 1100]
#   영상: 8 개 · 프레임: 367 개 · 패치: 2227 개

# --------------------------------------------------------------------
# [갭별 결함률 크로스탭 — 그리고 즉시 따라붙는 경고]
# --------------------------------------------------------------------
tab = lab.groupby('gap')[labels].mean().mul(100).round(1)
tab.insert(0, '패치수', lab.groupby('gap').size())
print(tab.to_string())

# 교재 실행 결과 ------------------------------------------------------
#         패치수  Surface_Crack  Delamination  Pinhole  unclassified
#   gap                                                          
#   600   306           59.2           0.0     38.2          10.1
#   700   407           82.8          10.8     24.3           4.4
#   800   303           81.5          16.5     28.4           2.6
#   900   304           95.7           1.6     12.8           4.3
#   1000  230           98.3           9.1     20.9           0.9
#   1100  677           98.2          12.3     19.2           0.9

# --------------------------------------------------------------------
# [갭별 결함률 크로스탭 — 그리고 즉시 따라붙는 경고]
# --------------------------------------------------------------------
print(pd.crosstab(lab['gap'], lab['video_id']).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   video_id  R1-1000-TBC  R1-1100-TBC  R1-1100-TBC-1  R1-600-TBC  R1-700-TBC  R1-800-TBC  R1-900-TBC  R7-700-middle
#   gap                                                                                                             
#   600                 0            0              0         306           0           0           0              0
#   700                 0            0              0           0         275           0           0            132
#   800                 0            0              0           0           0         303           0              0
#   900                 0            0              0           0           0           0         304              0
#   1000              230            0              0           0           0           0           0              0
#   1100                0          404            273           0           0           0           0              0

# --------------------------------------------------------------------
# [결정적 반례 — 같은 700 µm, 89.6배 차이]
# --------------------------------------------------------------------
g7 = lab[lab['gap'] == 700]
out = g7.groupby('run')['Delamination'].agg(패치수='size', 박리='sum', 검출률='mean')
out['검출률'] = (out['검출률'] * 100).round(2)
print(out.to_string())

# 교재 실행 결과 ------------------------------------------------------
#        패치수  박리    검출률
#   run                
#   1    275   1   0.36
#   7    132  43  32.58

# --------------------------------------------------------------------
# [행은 독립이 아니다 — 패치·프레임·영상의 3층 구조]
# --------------------------------------------------------------------
print("프레임당 패치 수 — 중앙값:", int(lab.groupby('frame_id').size().median()),
      "· 최대:", int(lab.groupby('frame_id').size().max()))

# 교재 실행 결과 ------------------------------------------------------
#   프레임당 패치 수 — 중앙값: 7 · 최대: 9

# --------------------------------------------------------------------
# [결함을 눈으로 보기]
# --------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

idx = pd.read_csv('data/coatingvision/curated_index.csv')
pick = ['image_1019.jpg', 'image_1501.jpg', 'image_1729.jpg']
sel = idx[idx['file_name'].isin(pick)].set_index('file_name').loc[pick]
print(sel[['run', 'coating_gap_um', 'frame', 'patch', 'defect_set']].to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, fn in zip(axes, pick):
    img = mpimg.imread('data/coatingvision/images_curated/' + fn)
    ax.imshow(img)
    ax.set_title(f"{fn}\n{sel.loc[fn, 'defect_set']}", fontsize=9)
    ax.axis('off')
    print(fn, "→ 배열 크기:", img.shape)
plt.tight_layout()
plt.show()

# 교재 실행 결과 ------------------------------------------------------
#                   run  coating_gap_um  frame  patch                          defect_set
#   file_name                                                                            
#   image_1019.jpg    1             600    372      4               Surface_Crack+Pinhole
#   image_1501.jpg    1            1100     56      2  Surface_Crack+Delamination+Pinhole
#   image_1729.jpg    7             700    566      0                        Delamination
#   image_1019.jpg → 배열 크기: (480, 640, 3)
#   image_1501.jpg → 배열 크기: (480, 640, 3)
#   image_1729.jpg → 배열 크기: (480, 640, 3)

