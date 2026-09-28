# -*- coding: utf-8 -*-
"""3장 예제 — 3.16 실전 정제 ① 현장 엑셀 워크북 해체하기 (필수, 35분)

교재 출처 : manuscript/31_ch3_eda_fe.md
실행 방법 : example 폴더에서  python "3장/<이 파일>"
            (교재 코드의 data/ 상대경로를 그대로 쓰기 위해 작업 디렉터리는 example)
선행 필요 : 이 절의 코드는 앞 절에서 만든 변수를 이어 쓴다.
            단독 실행이 필요하면 같은 장의 _장전체_실행본.py 를 쓰거나
            아래 절을 먼저 실행하라 — 13_3-13_차원_축소_개요.py, 14_3-14_EDA_리포트_정리.py, 15_3-15_연습문제.py
실행 상태 : 단독 실행 확인됨(이 파일만 돌려도 끝까지 실행된다).
"""

# --------------------------------------------------------------------
# [첫걸음 — 엑셀은 '격자'로 먼저 연다]
# --------------------------------------------------------------------
import pandas as pd
import numpy as np

XL = 'data/wmg/raw/Half-cell (Cathode) Electrochemical Performance.xlsx'

xls = pd.ExcelFile(XL)
print("시트 개수:", len(xls.sheet_names))
print(xls.sheet_names[:5], "...", xls.sheet_names[-1])

# 교재 실행 결과 ------------------------------------------------------
#   시트 개수: 19
#   ['Table', 'Group1', 'Group2', 'Group3', 'Group4'] ... Group18

# --------------------------------------------------------------------
# [첫걸음 — 엑셀은 '격자'로 먼저 연다]
# --------------------------------------------------------------------
grid = pd.read_excel(XL, sheet_name='Group1', header=None)
print("격자 모양:", grid.shape)
print("0행:", grid.iloc[0].tolist())

# 교재 실행 결과 ------------------------------------------------------
#   격자 모양: (151, 7)
#   0행: ['Electrode and cell details', 'Cathode cell ID', 'DD001', 'DD002', 'DD056', 'Mean ', 'Standard deviation']

# --------------------------------------------------------------------
# [첫걸음 — 엑셀은 '격자'로 먼저 연다]
# --------------------------------------------------------------------
print(grid.iloc[[20, 32, 80], 0:6].to_string())

# 교재 실행 결과 ------------------------------------------------------
#                                         0                       1    2           3           4           5
#   20                                  NaN  Coating thickness (um)   45          45          46   45.333333
#   32                Capacity Charge (mAh)                 At C/20  NaN     3.61307    3.698358    3.655714
#   80  Gravimetric Charge Capacity (mAh/g)                 At C/20    0  182.911073  183.087603  121.999559

# --------------------------------------------------------------------
# [첫걸음 — 엑셀은 '격자'로 먼저 연다]
# --------------------------------------------------------------------
g1 = pd.read_excel(XL, sheet_name='Group1')
g2 = pd.read_excel(XL, sheet_name='Group2')
print("Group1 컬럼:", list(g1.columns))
print("Group2 컬럼:", list(g2.columns))
print("그냥 concat:", pd.concat([g1, g2], ignore_index=True).shape)

# 교재 실행 결과 ------------------------------------------------------
#   Group1 컬럼: ['Electrode and cell details', 'Cathode cell ID', 'DD001', 'DD002', 'DD056', 'Mean ', 'Standard deviation']
#   Group2 컬럼: ['Electrode and cell details', 'Cathode cell ID', 'DD032', 'DD033', 'DD034', 'Mean ', 'Standard deviation']
#   그냥 concat: (300, 10)

# --------------------------------------------------------------------
# [(섹션, 항목, 셀ID) 3중키 파서 직접 짜기]
# --------------------------------------------------------------------
def read_group_sheet(path, sheet):
    """시트 한 장 → (조건, 섹션, 항목, 셀ID, 값) 다섯 열의 tidy 표."""
    g = pd.read_excel(path, sheet_name=sheet, header=None)
    cell_ids = [g.iat[0, c] for c in (2, 3, 4)]   # 0행의 2~4열 = 셀 ID 세 개
    g[0] = g[0].ffill()                           # 병합으로 비어 있는 섹션명 채우기
    recs = []
    for _, row in g.iterrows():
        if pd.isna(row[1]):                       # 항목명이 없는 빈 줄은 건너뛴다
            continue
        for k, c in enumerate((2, 3, 4)):
            recs.append({'group': sheet, 'section': row[0], 'label': row[1],
                         'cell_id': cell_ids[k], 'value': row[c]})
    return pd.DataFrame(recs)

# --------------------------------------------------------------------
# [(섹션, 항목, 셀ID) 3중키 파서 직접 짜기]
# --------------------------------------------------------------------
t1 = read_group_sheet(XL, 'Group1')
print("Group1 tidy:", t1.shape)
print(t1[t1['label'] == 'At C/20'].head(6).to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   Group1 tidy: (426, 5)
#    group                  section   label cell_id     value
#   Group1    Capacity Charge (mAh) At C/20   DD001       NaN
#   Group1    Capacity Charge (mAh) At C/20   DD002   3.61307
#   Group1    Capacity Charge (mAh) At C/20   DD056  3.698358
#   Group1 Capacity Discharge (mAh) At C/20   DD001  3.266397
#   Group1 Capacity Discharge (mAh) At C/20   DD002  3.284771
#   Group1 Capacity Discharge (mAh) At C/20   DD056  3.345534

# --------------------------------------------------------------------
# [(섹션, 항목, 셀ID) 3중키 파서 직접 짜기]
# --------------------------------------------------------------------
sheets = ['Group%d' % i for i in range(1, 19)]
tidy = pd.concat([read_group_sheet(XL, s) for s in sheets], ignore_index=True)
tidy['value_num'] = pd.to_numeric(tidy['value'], errors='coerce')

print("tidy 전체:", tidy.shape)
print("셀 개수:", tidy['cell_id'].nunique(), " 섹션 개수:", tidy['section'].nunique())
print("숫자 변환 실패:", int(tidy['value_num'].isna().sum()),
      "건 (그중 원래 빈칸", int(tidy['value'].isna().sum()), "건)")

# 교재 실행 결과 ------------------------------------------------------
#   tidy 전체: (7680, 6)
#   셀 개수: 54  섹션 개수: 11
#   숫자 변환 실패: 691 건 (그중 원래 빈칸 152 건)

# --------------------------------------------------------------------
# [(섹션, 항목, 셀ID) 3중키 파서 직접 짜기]
# --------------------------------------------------------------------
sel = tidy[(tidy['section'] == 'Gravimetric Discharge Capacity (mAh/g)') &
           (tidy['label'].isin(['At C/20', 'At 5C', 'At 10C']))]
wide = sel.pivot_table(index='cell_id', columns='label', values='value_num')
print(wide.shape)
print(wide.head(4).round(2))

# 교재 실행 결과 ------------------------------------------------------
#   (54, 3)
#   label    At 10C   At 5C  At C/20
#   cell_id                         
#   DD001     72.04  124.59   165.49
#   DD002     74.85  124.28   166.29
#   DD004     29.48  112.40   165.50
#   DD005     31.89  108.33   164.87

# --------------------------------------------------------------------
# [(섹션, 항목, 셀ID) 3중키 파서 직접 짜기]
# --------------------------------------------------------------------
ref = pd.read_csv('data/wmg/wmg_cells_54.csv').set_index('cell_id')
diff = (wide['At C/20'] - ref['grav_dis_c20_mahg']).abs()
print("배포 CSV와의 최대 차이:", float(diff.max()))
print("일치한 셀:", int((diff < 1e-9).sum()), "/", len(diff))

# 교재 실행 결과 ------------------------------------------------------
#   배포 CSV와의 최대 차이: 2.842170943040401e-14
#   일치한 셀: 54 / 54

# --------------------------------------------------------------------
# [원본의 계산 오류를 스스로 찾아내기 — 변동계수 스캔]
# --------------------------------------------------------------------
st = (tidy.groupby(['group', 'section', 'label'])['value_num']
          .agg(['mean', 'std', 'count']))
st = st[(st['count'] == 3) & (st['mean'].abs() > 0)]   # 3반복이 온전한 항목만
st['cv'] = st['std'] / st['mean'].abs()

print("검사 대상:", len(st), "개 (조건 × 항목)")
print("CV 중앙값:", round(float(st['cv'].median()), 4))
print(st['cv'].sort_values(ascending=False).head(5).round(3).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   검사 대상: 1903 개 (조건 × 항목)
#   CV 중앙값: 0.0065
#   group    section                                  label  
#   Group13  Volumetric Discharge Capacity (mAh/cm3)  At 10C     1.078
#            Capacity Discharge (mAh)                 At 10C     1.078
#            Gravimetric Discharge Capacity (mAh/g)   At 10C     1.077
#   Group1   Volumetric Charge Capacity (mAh/cm3)     At C/20    0.866
#            Gravimetric Charge Capacity (mAh/g)      At C/20    0.866

# --------------------------------------------------------------------
# [원본의 계산 오류를 스스로 찾아내기 — 변동계수 스캔]
# --------------------------------------------------------------------
grid = pd.read_excel(XL, sheet_name='Group1', header=None)
sec = grid[0].ffill()
row = grid[(sec == 'Gravimetric Charge Capacity (mAh/g)') & (grid[1] == 'At C/20')].iloc[0]

vals = row[[2, 3, 4, 5]]
vals.index = ['DD001', 'DD002', 'DD056', '워크북 Mean']
print(vals.astype(float).round(4).to_string())

v3 = vals[['DD001', 'DD002', 'DD056']].astype(float)
print("0을 그대로 둔 평균      :", round(float(v3.mean()), 4))
print("0을 결측으로 되돌린 평균:", round(float(v3.replace(0, np.nan).mean()), 4))

# 교재 실행 결과 ------------------------------------------------------
#   DD001         0.0000
#   DD002       182.9111
#   DD056       183.0876
#   워크북 Mean    121.9996
#   0을 그대로 둔 평균      : 121.9996
#   0을 결측으로 되돌린 평균: 182.9993

# --------------------------------------------------------------------
# [원본의 계산 오류를 스스로 찾아내기 — 변동계수 스캔]
# --------------------------------------------------------------------
row2 = grid[(sec == 'Capacity Charge (mAh)') & (grid[1] == 'At C/20')].iloc[0]
v2 = row2[[2, 3, 4, 5]]
v2.index = ['DD001', 'DD002', 'DD056', '워크북 Mean']
print(v2.astype(float).round(4).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   DD001          NaN
#   DD002       3.6131
#   DD056       3.6984
#   워크북 Mean    3.6557

# --------------------------------------------------------------------
# [병합 2단 헤더와 엑셀 날짜 — 두 번째 워크북]
# --------------------------------------------------------------------
CAL = 'data/wmg/raw/Intermediate measurements during calendering.xlsx'
print(pd.ExcelFile(CAL).sheet_names)

c0 = pd.read_excel(CAL, sheet_name='Cathode')
print("모양:", c0.shape)
print("컬럼 앞 7개:", list(c0.columns)[:7])

# 교재 실행 결과 ------------------------------------------------------
#   ['Cathode', 'Cathode-Intermediate']
#   모양: (40, 36)
#   컬럼 앞 7개: ['Unnamed: 0', 'Mean values', 'Calendering conditions', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Calendering date']

# --------------------------------------------------------------------
# [병합 2단 헤더와 엑셀 날짜 — 두 번째 워크북]
# --------------------------------------------------------------------
c = pd.read_excel(CAL, sheet_name='Cathode', header=[0, 1])
print("모양:", c.shape)
for col in list(c.columns)[:7]:
    print("  ", col)

# 교재 실행 결과 ------------------------------------------------------
#   모양: (39, 36)
#      ('Unnamed: 0_level_0', 'No')
#      ('Mean values', 'Electrode ID (P-porous, M-medium, D-dense)')
#      ('Calendering conditions', 'Target coating weight (GSM)')
#      ('Calendering conditions', 'Roll temperature (oC)')
#      ('Calendering conditions', 'Target density (g/cm3)')
#      ('Calendering conditions', 'Calculated target porosity (%)')
#      ('Calendering date', 'Unnamed: 6_level_1')

# --------------------------------------------------------------------
# [병합 2단 헤더와 엑셀 날짜 — 두 번째 워크북]
# --------------------------------------------------------------------
def flatten(col):
    top, bot = str(col[0]), str(col[1])
    top = '' if top.startswith('Unnamed') else top.strip()
    bot = '' if bot.startswith('Unnamed') else bot.strip()
    return (top + ' | ' + bot).strip(' |')

c.columns = [flatten(x) for x in c.columns]
for col in list(c.columns)[:7]:
    print("  ", col)

# 교재 실행 결과 ------------------------------------------------------
#      No
#      Mean values | Electrode ID (P-porous, M-medium, D-dense)
#      Calendering conditions | Target coating weight (GSM)
#      Calendering conditions | Roll temperature (oC)
#      Calendering conditions | Target density (g/cm3)
#      Calendering conditions | Calculated target porosity (%)
#      Calendering date

# --------------------------------------------------------------------
# [병합 2단 헤더와 엑셀 날짜 — 두 번째 워크북]
# --------------------------------------------------------------------
c = c.rename(columns={
    'Calendering conditions | Target coating weight (GSM)': 'gsm',
    'Calendering conditions | Roll temperature (oC)': 'roll_temp',
    'Calendering date': 'cal_date'})

# --------------------------------------------------------------------
# [병합 2단 헤더와 엑셀 날짜 — 두 번째 워크북]
# --------------------------------------------------------------------
print("행 수:", len(c), " / 설계상 조건 수: 18")
print(c[['No', 'gsm', 'roll_temp']].iloc[16:24].to_string())
print("목표 코팅중량 고유값:",
      sorted(pd.to_numeric(c['gsm'], errors='coerce').dropna().unique().tolist()))
print("39행 그대로 낸 평균:",
      round(float(pd.to_numeric(c['gsm'], errors='coerce').mean()), 2))

# 교재 실행 결과 ------------------------------------------------------
#   행 수: 39  / 설계상 조건 수: 18
#        No                          gsm              roll_temp
#   16   17                       182.73                    145
#   17   18                       182.73                    145
#   18  NaN                          NaN                    NaN
#   19  NaN       Calendering conditions                    NaN
#   20   No  Target coating weight (GSM)  Roll temperature (oC)
#   21    1                          1.5                     85
#   22    2                          1.5                     85
#   23    3                          1.5                     85
#   목표 코팅중량 고유값: [1.32, 1.5, 122.48, 182.73]
#   39행 그대로 낸 평균: 77.01

# --------------------------------------------------------------------
# [병합 2단 헤더와 엑셀 날짜 — 두 번째 워크북]
# --------------------------------------------------------------------
cond = c.iloc[:18].copy()
print("조건 표:", cond[['No', 'gsm', 'roll_temp', 'cal_date']].shape)
print("목표 코팅중량 고유값:", sorted(cond['gsm'].unique().tolist()))

# 교재 실행 결과 ------------------------------------------------------
#   조건 표: (18, 4)
#   목표 코팅중량 고유값: [122.48, 182.73]

# --------------------------------------------------------------------
# [병합 2단 헤더와 엑셀 날짜 — 두 번째 워크북]
# --------------------------------------------------------------------
print("dtype:", cond['cal_date'].dtype)
print("첫 값:", repr(cond['cal_date'].iloc[0]))

cond['cal_date'] = pd.to_datetime(cond['cal_date'])
print("변환 후 dtype:", cond['cal_date'].dtype)

serial = (cond['cal_date'] - pd.Timestamp('1899-12-30')).dt.days
print("엑셀 serial:", sorted(serial.unique().tolist()))

# 교재 실행 결과 ------------------------------------------------------
#   dtype: object
#   첫 값: datetime.datetime(2021, 10, 15, 0, 0)
#   변환 후 dtype: datetime64[us]
#   엑셀 serial: [44484, 44487, 44496]

# --------------------------------------------------------------------
# [병합 2단 헤더와 엑셀 날짜 — 두 번째 워크북]
# --------------------------------------------------------------------
print(pd.crosstab(cond['cal_date'].dt.date, cond['roll_temp']).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   roll_temp   85   120  145
#   cal_date                 
#   2021-10-15    6    0    0
#   2021-10-18    0    6    0
#   2021-10-27    0    0    6

# --------------------------------------------------------------------
# [NaN이 아닌 실패값 — 0.0009 mAh/g 일곱 셀]
# --------------------------------------------------------------------
cells = pd.read_csv('data/wmg/wmg_cells_54.csv')
y = 'grav_dis_10c_mahg'
print("결측:", int(cells[y].isna().sum()), "건")
print(cells[y].sort_values().head(9).round(4).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   결측: 0 건
#   27    0.0009
#   29    0.0009
#   46    0.0009
#   28    0.0009
#   47    0.0009
#   45    0.0009
#   37    0.0009
#   36    3.0044
#   38    7.4846

# --------------------------------------------------------------------
# [NaN이 아닌 실패값 — 0.0009 mAh/g 일곱 셀]
# --------------------------------------------------------------------
fail = cells[y] < 1.0
print("1 mAh/g 미만:", int(fail.sum()), "셀")
print(cells.loc[fail, ['cell_id', 'coat_weight_level', 'density_level',
                       'roll_temp_c', y]].to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   1 mAh/g 미만: 7 셀
#   cell_id coat_weight_level density_level  roll_temp_c  grav_dis_10c_mahg
#     DD059                 H             P         85.0           0.000887
#     DD027                 H             P         85.0           0.000892
#     DD028                 H             P         85.0           0.000890
#     DD051                 H             P        120.0           0.000895
#     DD029                 H             P        145.0           0.000893
#     DD030                 H             P        145.0           0.000891
#     DD031                 H             P        145.0           0.000892

# --------------------------------------------------------------------
# [NaN이 아닌 실패값 — 0.0009 mAh/g 일곱 셀]
# --------------------------------------------------------------------
print("평균 (54셀 전부):", round(float(cells[y].mean()), 2))
print("평균 (실패 7셀 제외):", round(float(cells.loc[~fail, y].mean()), 2))
print("---")
print("전체 54셀 :", round(float(cells['porosity_pct'].corr(cells[y])), 4))
ok = cells[~fail]
print("실패 제외 47셀:", round(float(ok['porosity_pct'].corr(ok[y])), 4))

# 교재 실행 결과 ------------------------------------------------------
#   평균 (54셀 전부): 42.04
#   평균 (실패 7셀 제외): 48.31
#   ---
#   전체 54셀 : -0.1154
#   실패 제외 47셀: 0.0812

# --------------------------------------------------------------------
# [이론 한계를 넘는 값 — DD020을 어떻게 판정할 것인가]
# --------------------------------------------------------------------
Y = 'grav_dis_c20_mahg'
print(cells[Y].describe().round(2).to_string())

# 교재 실행 결과 ------------------------------------------------------
#   count     54.00
#   mean     166.28
#   std        8.90
#   min      153.30
#   25%      165.04
#   50%      165.69
#   75%      166.86
#   max      226.80

# --------------------------------------------------------------------
# [이론 한계를 넘는 값 — DD020을 어떻게 판정할 것인가]
# --------------------------------------------------------------------
print(cells.nlargest(3, Y)[['cell_id', 'group', Y]].round(2).to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   cell_id  group  grav_dis_c20_mahg
#     DD020      3             226.80
#     DD036      5             169.34
#     DD039      6             168.72

# --------------------------------------------------------------------
# [이론 한계를 넘는 값 — DD020을 어떻게 판정할 것인가]
# --------------------------------------------------------------------
print(cells[cells['group'] == 3][['cell_id', 'coat_weight_level', 'roll_temp_c',
                                  'density_level', Y]].round(2).to_string(index=False))

# 교재 실행 결과 ------------------------------------------------------
#   cell_id coat_weight_level  roll_temp_c density_level  grav_dis_c20_mahg
#     DD020                 L         85.0             D             226.80
#     DD021                 L         85.0             D             166.39
#     DD022                 L         85.0             D             167.51

# --------------------------------------------------------------------
# [이론 한계를 넘는 값 — DD020을 어떻게 판정할 것인가]
# --------------------------------------------------------------------
sd = cells.groupby('group')[Y].std()
print("group 3 의 SD :", round(float(sd.loc[3]), 2))
print("나머지 17조건 SD 중앙값:", round(float(sd.drop(3).median()), 2))
print("배율:", round(float(sd.loc[3] / sd.drop(3).median()), 1), "배")

# 교재 실행 결과 ------------------------------------------------------
#   group 3 의 SD : 34.56
#   나머지 17조건 SD 중앙값: 0.82
#   배율: 42.0 배

# --------------------------------------------------------------------
# [이론 한계를 넘는 값 — DD020을 어떻게 판정할 것인가]
# --------------------------------------------------------------------
asi = [c for c in cells.columns if c.startswith('asi_')]
print("ASI 열:", len(asi), "개")
print("DD020 의 ASI 결측:",
      int(cells.loc[cells['cell_id'] == 'DD020', asi].isna().sum(axis=1).iloc[0]), "개")
print("나머지 53셀의 ASI 결측 합계:",
      int(cells.loc[cells['cell_id'] != 'DD020', asi].isna().sum().sum()), "개")

# 교재 실행 결과 ------------------------------------------------------
#   ASI 열: 6 개
#   DD020 의 ASI 결측: 6 개
#   나머지 53셀의 ASI 결측 합계: 0 개

# --------------------------------------------------------------------
# [이 절의 정제 규약 — clean_wmg()]
# --------------------------------------------------------------------
def clean_wmg(path):
    """WMG 하프셀 표 정제 — 실패값·불가값을 결측으로 되돌리고 플래그를 남긴다."""
    d = pd.read_csv(path)
    # 1) 10C 시험 실패(0.0009 mAh/g)를 결측으로 되돌리되, 사실을 플래그로 보존
    d['flag_10c_fail'] = (d['grav_dis_10c_mahg'] < 1.0).astype(int)
    d.loc[d['flag_10c_fail'] == 1, 'grav_dis_10c_mahg'] = np.nan
    # 2) 이론 한계 초과는 지우지 않고 플래그만 단다
    d['flag_over_theory'] = (d['grav_dis_c20_mahg'] > 200).astype(int)
    return d

w = clean_wmg('data/wmg/wmg_cells_54.csv')
print("모양:", w.shape)
print("10C 실패 플래그:", int(w['flag_10c_fail'].sum()), "셀")
print("이론 초과 플래그:", int(w['flag_over_theory'].sum()), "셀")
print("정제 후 10C 평균:", round(float(w['grav_dis_10c_mahg'].mean()), 2))
print("정제 후 10C 결측:", int(w['grav_dis_10c_mahg'].isna().sum()), "건")

# 교재 실행 결과 ------------------------------------------------------
#   모양: (54, 60)
#   10C 실패 플래그: 7 셀
#   이론 초과 플래그: 1 셀
#   정제 후 10C 평균: 48.31
#   정제 후 10C 결측: 7 건

