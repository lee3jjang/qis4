import numpy as np
import pandas as pd
from typing import Tuple

def get_cf(cf: pd.DataFrame, pdgr_cd: str, cf_type: str) -> Tuple[pd.Series, pd.Series]:
    n = 7 if pdgr_cd in ['25', '26'] else 5
    if not set(['AY_YM', 'BASE_1', 'BASE_2', 'BASE_3', 'BASE_4', 'BASE_5', 'BASE_6', 'BASE_7']).issubset(cf.columns):
        raise Exception('cf 필수 컬럼 누락 오류')

    if len(cf) != n:
        raise Exception('cf 입력 크기 오류')

    cf_arr = cf.sort_values(by='AY_YM')[['BASE_1', 'BASE_2', 'BASE_3', 'BASE_4', 'BASE_5', 'BASE_6', 'BASE_7']].to_numpy()

    pay_cf_all = []
    for i in range(n-1, 0, -1):
        pay_cf = 0
        for j in range(i):
            pay_cf += cf_arr[(n-1)-j, (n-1-i)+j+1]-cf_arr[(n-1)-j, (n-1-i)+j]
        pay_cf = max(pay_cf, 0)
        pay_cf_all.append(pay_cf)
    pay_cf_all = np.array(pay_cf_all)
    pay_cf_rate = pd.Series(pay_cf_all/pay_cf_all.sum())
    
    if cf_type == '보험금':
        cf_t = pd.Series(np.arange(0.5, n-0.5))
        return (cf_t, pay_cf_rate)



def clsf_crd_grd(data: pd.DataFrame, reins_crd_grd: pd.DataFrame) -> pd.Series:
    """CRD_GRD 가공

    Args:
        data (pd.DataFrame): 일반 출재, 계약/보상 기초데이터
        reins_crd_grd (pd.DataFrame): 재보험자별 신용등급

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        pd.Series: CRD_GRD 결과

    Example:
        일반_출재_미경과보험료 = pd.read_excel(FILE_PATH / '일반_출재_미경과보험료.xlsx', dtype={'RRNR_DAT_DVCD': str, 'RRNR_CTC_BZ_DVCD': str, 'ARC_INPL_CD': str, 'T02_RN_RINSC_CD': str})
        재보험자_국내신용등급 = pd.read_excel(FILE_PATH / '재보험자_국내신용등급.xlsx', dtype={'재보험사코드': str}) \
            .rename(columns = {'재보험사코드': 'T02_RN_RINSC_CD', '국내신용등급': 'CRD_GRD'})
        일반_출재_미경과보험료['CRD_GRD'] = clsf_crd_grd(일반_출재_미경과보험료, 재보험자_국내신용등급)
    """

    # 컬럼 존재성 검사
    if not set(['T02_RN_RINSC_CD']).issubset(data.columns):
        raise Exception('data 필수 컬럼 누락 오류')
    if not set(['T02_RN_RINSC_CD', 'CRD_GRD']).issubset(reins_crd_grd.columns):
        raise Exception('reins_crd_grd 필수 컬럼 누락 오류')

    crd_grd = data.merge(reins_crd_grd, on='T02_RN_RINSC_CD', how='left') \
        .assign(CRD_GRD = lambda x: x['CRD_GRD'].fillna('무등급'))

    return crd_grd['CRD_GRD']

def clsf_rrnr_dvcd(data: pd.DataFrame, ret_type: str) -> pd.Series:
    """RRNR_DVCD 가공

    Args:
        data (pd.DataFrame): 일반 원수/출재, 계약/보상 기초데이터
        ret_type (str): "원수" 또는 "출재"

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        pd.Series: RRNR_DVCD 결과

    Example:
        >>> 일반_원수_미경과보험료 = pd.read_excel(FILE_PATH / '일반_원수_미경과보험료.xlsx', dtype={'RRNR_DAT_DVCD': str, 'RRNR_CTC_BZ_DVCD': str, 'ARC_INPL_CD': str})
        >>> 일반_원수_미경과보험료['RRNR_DVCD'] = clsf_rrnr_dvcd(일반_원수_미경과보험료, '원수')
    """

    if ret_type == '원수':
        data['RRNR_DVCD'] = data['RRNR_DAT_DVCD'].map(lambda x: {'01': '01', '02': '02', '03': '02'}.get(x, '#'))
        if '#' in data['RRNR_DVCD'].values:
            raise Exception('전처리 누락 오류')
    elif ret_type == '출재':
        data['RRNR_DVCD'] = '03'
    else:
        raise Exception('보유구분 입력 오류')

    return data['RRNR_DVCD']


def clsf_dmfr_dvcd(data: pd.DataFrame) -> pd.Series:
    """DMFR_DVCD 가공 (※ RRNR_DVCD 선행 필수)

    Args:
        data (pd.DataFrame): 일반 원수/출재, 계약/보상 기초데이터

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        pd.Series: DMFR_DVCD 결과

    Example:
        >>> 일반_원수_미경과보험료 = pd.read_excel(FILE_PATH / '일반_원수_미경과보험료.xlsx', dtype={'RRNR_DAT_DVCD': str, 'RRNR_CTC_BZ_DVCD': str, 'ARC_INPL_CD': str})
        >>> 일반_원수_미경과보험료['RRNR_DVCD'] = clsf_rrnr_dvcd(일반_원수_미경과보험료, '원수')
        >>> 일반_원수_미경과보험료['DMFR_DVCD'] = clsf_dmfr_dvcd(일반_원수_미경과보험료)
    """

    # 컬럼 존재성 검사
    if not set(['RRNR_DVCD', 'ARC_INPL_CD']).issubset(data.columns):
        raise Exception('data 필수 컬럼 누락 오류')
    if 'RRNR_CTC_BZ_DVCD' in data.columns:
        dmfr_col = 'RRNR_CTC_BZ_DVCD'
    elif 'RRNR_DMFR_DVCD' in data.columns:
        dmfr_col = 'RRNR_DMFR_DVCD'
    else:
        raise Exception('data 필수 컬럼 누락 오류')

    dmfr_dvcd = data \
        .assign(DMFR_DVCD = lambda x: np.where((x['ARC_INPL_CD'] == '1069010') & (x['RRNR_DVCD'] == '02'), '02', x[dmfr_col])) \
        .assign(DMFR_DVCD = lambda x: np.where(x['DMFR_DVCD'] == '2', '02', '01'))

    return dmfr_dvcd['DMFR_DVCD']


def clsf_boz_cd(data: pd.DataFrame, prd_info: pd.DataFrame) -> pd.Series:
    """BOZ_CD 가공

    Args:
        data (pd.DataFrame): 일반 원수/출재, 계약/보상 기초데이터
        prd_info (pd.DataFrame): 상품정보

    Returns:
        pd.Series: BOZ_CD 결과

    Example:
        >>> 일반_상품정보 = pd.read_excel(FILE_PATH / '일반_상품정보.xlsx', dtype={'PDC_CD': str, 'PDGR_CD': str})
        >>> 일반_원수_미경과보험료 = pd.read_excel(FILE_PATH / '일반_원수_미경과보험료.xlsx', dtype={'RRNR_DAT_DVCD': str, 'RRNR_CTC_BZ_DVCD': str, 'ARC_INPL_CD': str})
        >>> 일반_원수_미경과보험료['BOZ_CD'] = clsf_boz_cd(일반_원수_미경과보험료, 일반_상품정보)
    """

    # 상품군코드 -> 보종코드 mapper
    pdgr_boz_mapper = {
        '23': 'A001', '24': 'A002', '27': 'A003', '31': 'A004',
        '25': 'A005', '26': 'A006', '28': 'A007', '30': 'A010',
        '29': 'A010', '34': 'A011'
    }

    # 컬럼 존재성 검사
    if not set(['ARC_INPL_CD', 'NTNL_CTRY_CD']).issubset(data.columns):
        raise Exception('data 필수 컬럼 누락 오류')
    if not set(['PDC_CD', 'PDGR_CD']).issubset(prd_info.columns):
        raise Exception('prd_info 필수 컬럼 누락 오류')

    boz_cd = data \
        .assign(PDC_CD = lambda x: x['ARC_INPL_CD'].str.slice(0,5)) \
        .merge(prd_info, on='PDC_CD', how='left') \
        .assign(BOZ_CD = lambda x: x['PDGR_CD'].map(lambda y: pdgr_boz_mapper.get(y, '#'))) \
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'] =='10607', 'A008', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x['PDGR_CD'] == '30', 'A010', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'].isin(['10902', '10903']) & (x.NTNL_CTRY_CD == 'KR'), 'A009', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'].isin(['10902', '10903']) & (x.NTNL_CTRY_CD != 'KR'), 'A010', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x.PDC_CD.isin(['10011', '10013', '10016', '10017', '10900', '10901']), 'A003', x['BOZ_CD']))

    ## 아래코드 삭제예정(임시) ##
    boz_cd = boz_cd.assign(BOZ_CD = lambda x: np.where(x.PDC_CD.isin([
        '13411', '13414', '13415', '13418', '13413', '13416',
        '13814', '13412', '13512', '13606', '13409', '13510']), 'A010', x['BOZ_CD']))

    # 전처리 누락여부 검사
    if len(boz_cd.query('BOZ_CD == "#"')) != 0:
        print(boz_cd)
        raise Exception('전처리 누락 오류')

    return boz_cd['BOZ_CD']


if __name__ == '__main__':
    pass