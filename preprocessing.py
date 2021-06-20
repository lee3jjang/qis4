import numpy as np
import pandas as pd
from typing import Tuple

def get_loss_adj(cf_t: pd.Series, cf_rate: pd.Series, crd_grd: str, int_rate: pd.DataFrame, fwd_pd: pd.DataFrame) -> float:
    """손실조정율 계산

    Args:
        cf_t (pd.Series): 현금흐름시점
        cf_rate (pd.Series): 현금흐름비중
        crd_grd (str): 거래상대방 신용등급
        int_rate (pd.DataFrame): 할인율
        fwd_pd (pd.DataFrame): 신용등급별 부도율

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        float: 손실조정율

    Example:
        >>> 일반_보험금진전추이 = pd.read_excel(FILE_PATH / '일반_보험금진전추이.xlsx')
        >>> cf_t, cf_rate = get_cf(일반_보험금진전추이.query('PDGR_CD == "26"'), pdgr_cd='26', cf_type='보험료')
        >>> get_loss_adj(cf_t, cf_rate, crd_grd='무등급', int_rate=할인율.query('KICS_SCEN_NO == 1'), fwd_pd=선도부도율)
    """

    # 컬럼 존재성 검사
    if not set(['MAT_TERM', 'SPOT_RATE']).issubset(int_rate.columns):
        raise Exception('int_rate 필수 컬럼 누락 오류')
    if not set(['GRADE', 'YEAR', 'FWD_PD']).issubset(fwd_pd.columns):
        raise Exception('fwd_pd 필수 컬럼 누락 오류')
    if not int_rate['MAT_TERM'].is_unique:
        raise Exception('MAT_TERM 유일성 오류')
    if not fwd_pd.query('GRADE == @crd_grd')['YEAR'].is_unique:
        raise Exception(f'FWD_PD 유일성 오류(GRADE == {crd_grd}')
    if crd_grd == '무등급':
        crd_grd = 'B'

    t = np.round(cf_t*12).astype(int)
    x = int_rate.set_index('MAT_TERM').loc[t, 'SPOT_RATE'].reset_index()
    m, r = x['MAT_TERM'], x['SPOT_RATE']
    disc_rate = (cf_rate/(1+r)**cf_t).reset_index(drop=True)
    fwd_pd = fwd_pd.query('GRADE == @crd_grd').set_index('YEAR').loc[np.arange(len(cf_t))+1, 'FWD_PD'].reset_index(drop=True)
    
    lgd = 0.5
    loss = lgd*cf_rate[::-1].cumsum()[::-1]
    loss_adj = np.sum(loss*fwd_pd*disc_rate)
    return loss_adj


def get_disc_factor(cf_t: pd.Series, cf_rate: pd.Series, int_rate: pd.DataFrame) -> float:
    """할인요소계산

    Args:
        cf_t (pd.Series): 지급시점
        cf_rate (pd.Series): 현금흐름비중
        int_rate (pd.DataFrame): 할인율

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        float: 할인요소

    Example:
        >>> 일반_보험금진전추이 = pd.read_excel(FILE_PATH / '일반_보험금진전추이.xlsx')
        >>> 할인율 = pd.read_excel(FILE_PATH / '할인율.xlsx')
        >>> cf_t, cf_rate = get_cf(일반_보험금진전추이.query('PDGR_CD == "25"'), pdgr_cd="25", cf_type="보험료")
        >>> get_disc_factor(cf_t, cf_rate, 할인율.query('KICS_SCEN_NO == 1'))
    """

    # 컬럼 존재성 검사
    if not set(['MAT_TERM', 'SPOT_RATE']).issubset(int_rate.columns):
        raise Exception('int_rate 필수 컬럼 누락 오류')
    if not int_rate['MAT_TERM'].is_unique:
        raise Exception('MAT_TERM 유일성 오류')

    t = np.round(cf_t*12).astype(int)
    x = int_rate.set_index('MAT_TERM').loc[t, 'SPOT_RATE'].reset_index()
    m = x['MAT_TERM']
    r = x['SPOT_RATE']
    disc_fac = np.sum(cf_rate/(1+r)**cf_t)

    return disc_fac


def get_cf(cf: pd.DataFrame, pdgr_cd: str, cf_type: str) -> Tuple[pd.Series, pd.Series]:
    """보험금/보험료 현금흐름비중 계산

    Args:
        cf (pd.DataFrame): 보험금 진전추이
        pdgr_cd (str): 상품군코드
        cf_type (str): "보험금" 또는 "보험료"

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        Tuple[pd.Series, pd.Series]: (지급시점, 현금흐름비중)

    Example:
        >>> 일반_보험금진전추이 = pd.read_excel(FILE_PATH / '일반_보험금진전추이.xlsx')
        >>> pdgr_cd = '26'
        >>> cf_type = '보험료'
        >>> cf_t, cf_rate = get_cf(일반_보험금진전추이.query('PDGR_CD == @pdgr_cd'), pdgr_cd, cf_type)
    """
    n = 7 if pdgr_cd in ['25', '26'] else 5
    if not set(['AY', 'BASE_1', 'BASE_2', 'BASE_3', 'BASE_4', 'BASE_5', 'BASE_6', 'BASE_7']).issubset(cf.columns):
        raise Exception('cf 필수 컬럼 누락 오류')

    if len(cf) != n:
        raise Exception('cf 입력 크기 오류')

    cf_arr = cf.sort_values(by='AY')[['BASE_1', 'BASE_2', 'BASE_3', 'BASE_4', 'BASE_5', 'BASE_6', 'BASE_7']].to_numpy()

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
        cf_t = pd.Series(np.arange(n-1)+0.5)
        return (cf_t, pay_cf_rate)
    elif cf_type == '보험료':
        cf_t = pd.Series(np.arange(n-1)+0.2929)
        pay_cf_cum_rate = pay_cf_rate.cumsum()
        one_minus_adj_rate = np.zeros(n-1)
        adj_rate = np.zeros(n-1)
        for i in range(n-2):
            adj_rate[i] = pay_cf_cum_rate[i]+0.2071*(pay_cf_cum_rate[i+1]-pay_cf_cum_rate[i])
        adj_rate[-1] = 1
        one_minus_adj_rate[0] = adj_rate[0]
        for i in range(1, n-1):
            one_minus_adj_rate[i] = adj_rate[i]-adj_rate[i-1]
        one_minus_adj_rate = pd.Series(one_minus_adj_rate)
        return (cf_t, one_minus_adj_rate)

    else:
        raise Exception('cf_type 입력 오류')


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
        >>> 일반_출재_미경과보험료 = pd.read_excel(FILE_PATH / '일반_출재_미경과보험료.xlsx', dtype={'RRNR_DAT_DVCD': str, 'RRNR_CTC_BZ_DVCD': str, 'ARC_INPL_CD': str, 'T02_RN_RINSC_CD': str})
        >>> 재보험자_국내신용등급 = pd.read_excel(FILE_PATH / '재보험자_국내신용등급.xlsx', dtype={'재보험사코드': str}) \
                .rename(columns = {'재보험사코드': 'T02_RN_RINSC_CD', '국내신용등급': 'CRD_GRD'})
        >>> 일반_출재_미경과보험료['CRD_GRD'] = clsf_crd_grd(일반_출재_미경과보험료, 재보험자_국내신용등급)
    """

    # 컬럼 존재성 검사
    if not set(['T02_RN_RINSC_CD']).issubset(data.columns):
        raise Exception('data 필수 컬럼 누락 오류')
    if not set(['T02_RN_RINSC_CD', 'CRD_GRD']).issubset(reins_crd_grd.columns):
        raise Exception('reins_crd_grd 필수 컬럼 누락 오류')
    if set(['CRD_GRD']).issubset(data.columns):
        data.drop('CRD_GRD', axis=1, inplace=True)

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


def clsf_pdgr_cd(data: pd.DataFrame, prd_info: pd.DataFrame) -> pd.Series:
    """PDGR_CD 가공 (원래 PDGR_CD랑 조금 다름)

    Args:
        data (pd.DataFrame): 일반 원수/출재, 계약/보상 기초데이터
        prd_info (pd.DataFrame): 상품정보

    Returns:
        pd.Series: BOZ_CD 결과

    Example:
        >>> 일반_상품정보 = pd.read_excel(FILE_PATH / '일반_상품정보.xlsx', dtype={'PDC_CD': str, 'PDGR_CD': str})
        >>> 일반_원수_미경과보험료 = pd.read_excel(FILE_PATH / '일반_원수_미경과보험료.xlsx', dtype={'RRNR_DAT_DVCD': str, 'RRNR_CTC_BZ_DVCD': str, 'ARC_INPL_CD': str})
        >>> 일반_원수_미경과보험료['PDGR_CD'] = clsf_pdgr_cd(일반_원수_미경과보험료, 일반_상품정보)
    """
    # 컬럼 존재성 검사
    if not set(['ARC_INPL_CD']).issubset(data.columns):
        raise Exception('data 필수 컬럼 누락 오류')
    if not set(['PDC_CD', 'PDGR_CD']).issubset(prd_info.columns):
        raise Exception('prd_info 필수 컬럼 누락 오류')
    if set(['PDGR_CD']).issubset(data.columns):
        data.drop('PDGR_CD', axis=1, inplace=True)

    pdgr_cd = data \
        .assign(PDC_CD = lambda x: x['ARC_INPL_CD'].str.slice(0,5)) \
        .merge(prd_info, on='PDC_CD', how='left') \
        .assign(PDGR_CD = lambda x: x['PDGR_CD'].fillna('#')) \
        .assign(PDGR_CD = lambda x: np.where(x['PDGR_CD'] == '34', '29', x['PDGR_CD'])) \
        .assign(PDGR_CD = lambda x: np.where(x.PDC_CD.isin(['10011', '10013', '10016', '10017', '10900', '10901']), '27', x['PDGR_CD']))

    ## 아래코드 삭제예정(임시) ##
    pdgr_cd = pdgr_cd.assign(PDGR_CD = lambda x: np.where(x.PDC_CD.isin([
        '13411', '13414', '13415', '13418', '13413', '13416',
        '13814', '13412', '13512', '13606', '13409', '13510', '10580']), '29', x['PDGR_CD']))

    # 전처리 누락여부 검사
    if len(pdgr_cd.query('PDGR_CD == "#"')) != 0:
        print(pdgr_cd.query('PDGR_CD == "#"'))
        raise Exception('전처리 누락 오류')

    return pdgr_cd['PDGR_CD']


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
    if not set(['ARC_INPL_CD', 'DMFR_DVCD']).issubset(data.columns):
        raise Exception('data 필수 컬럼 누락 오류')
    if not set(['PDC_CD', 'PDGR_CD']).issubset(prd_info.columns):
        raise Exception('prd_info 필수 컬럼 누락 오류')
    if set(['PDGR_CD']).issubset(data.columns):
        data.drop('PDGR_CD', axis=1, inplace=True)

    boz_cd = data \
        .assign(PDC_CD = lambda x: x['ARC_INPL_CD'].str.slice(0,5)) \
        .merge(prd_info, on='PDC_CD', how='left') \
        .assign(BOZ_CD = lambda x: x['PDGR_CD'].map(lambda y: pdgr_boz_mapper.get(y, '#'))) \
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'] == '10607', 'A008', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x['PDGR_CD'] == '30', 'A010', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'].isin(['10902', '10903']) & (x.DMFR_DVCD == '01'), 'A009', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'].isin(['10902', '10903']) & (x.DMFR_DVCD != '01'), 'A010', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x.PDC_CD.isin(['10011', '10013', '10016', '10017', '10900', '10901']), 'A003', x['BOZ_CD']))

    ## 아래코드 삭제예정(임시) ##
    boz_cd = boz_cd.assign(BOZ_CD = lambda x: np.where(x.PDC_CD.isin([
        '13411', '13414', '13415', '13418', '13413', '13416',
        '13814', '13412', '13512', '13606', '13409', '13510', '10580']), 'A010', x['BOZ_CD']))

    # 전처리 누락여부 검사
    if len(boz_cd.query('BOZ_CD == "#"')) != 0:
        print(boz_cd.query('BOZ_CD == "#"'))
        raise Exception('전처리 누락 오류')

    return boz_cd['BOZ_CD']


if __name__ == '__main__':
    pass