import numpy as np
import pandas as pd
from typing import Tuple


def clsf_tty_cd_grp(data: pd.DataFrame) -> pd.Series:
    """TTY_CD_GRP 가공

    Args:
        data (pd.DataFrame): 재보험특약정보

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        pd.Series: TTY_CD_GRP 결과

    Examples:
        >>> 일반_특약정보 = pd.read_excel(FILE_PATH / f'일반_특약정보_{BASE_YYMM}.xlsx', dtype={'RRNR_TTY_CD': str, 'TTY_YR': str})
        >>> 일반_특약정보['TTY_CD_GRP'] = clsf_tty_cd_grp(일반_특약정보)
    """
    # 컬럼 존재성 검사
    if not set(['TTY_CD_NM']).issubset(data.columns):
        raise Exception('data 필수 컬럼 누락 오류')

    tty_cd_grp = data.copy()
    tty_cd_grp['TTY_CD_GRP'] = "#"
    tty_cd_grp.loc[lambda x: x['TTY_CD_NM'].str.contains('기술보험'), 'TTY_CD_GRP'] = '기술보험특약'
    tty_cd_grp.loc[lambda x: x['TTY_CD_NM'].str.contains('해외PST'), 'TTY_CD_GRP'] = '해외PST'
    tty_cd_grp.loc[lambda x: x['TTY_CD_NM'].str.contains('근재보험'), 'TTY_CD_GRP'] = '근재보험특약'
    tty_cd_grp.loc[lambda x: x['TTY_CD_NM'].str.contains('배상책임보험'), 'TTY_CD_GRP'] = '배상책임보험특약'
    tty_cd_grp.loc[lambda x: x['TTY_CD_NM'].str.contains('재물보험|패키지', regex=True), 'TTY_CD_GRP'] = '재물보험특약'

    # 전처리 누락여부 검사
    if len(tty_cd_grp.query('TTY_CD_GRP == "#"')) != 0:
        print(tty_cd_grp.query('TTY_CD_GRP == "#"'))
        raise Exception('전처리 누락 오류')

    return tty_cd_grp['TTY_CD_GRP']


def clsf_p_np_dvcd(data: pd.DataFrame) -> pd.Series:

    # 컬럼 존재성 검사
    if not set(['RRNR_DVCD']).issubset(data.columns):
        raise Exception('data 필수 컬럼 누락 오류')
    if 'INRI_RN_KND_CD' in data.columns:
        rn_knd_col = 'INRI_RN_KND_CD'
    elif 'RN_KND_CD' in data.columns:
        rn_knd_col = 'RN_KND_CD'
    else:
        raise Exception('data 필수 컬럼 누락 오류')

    # 출수재종류코드 -> 비례비비례 mapper
    p_np_mapper = {'TP': 'P', 'TN': 'N', 'FP': 'P', 'FN': 'N', 'AP': 'P', 'TJ': 'P'}

    p_np_dvcd = data \
        .assign(P_NP_DVCD = lambda x: x[rn_knd_col].apply(lambda y: p_np_mapper.get(y, 'P'))) \
        .assign(P_NP_DVCD = lambda x: np.where(x['RRNR_DVCD']=="01", "#", x['P_NP_DVCD']))

    return p_np_dvcd['P_NP_DVCD']




def clsf_cntr_catg_cd(data: pd.DataFrame, cntr_grp_info: pd.DataFrame) -> pd.Series:
    """CNTR_CATG_CD 가공

    Args:
        data (pd.DataFrame): 일반 원수/출재, 계약/보상 기초데이터
        cntr_grp_info (pd.DataFrame): 국가그룹정보

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        pd.Series: CNTR_CATG_CD 결과

    Example:
        >>> 일반_원수_미경과보험료 = pd.read_excel(FILE_PATH / '일반_원수_미경과보험료.xlsx', dtype={'RRNR_DAT_DVCD': str, 'RRNR_CTC_BZ_DVCD': str, 'ARC_INPL_CD': str})
        >>> 국가그룹 = pd.read_excel(FILE_PATH / '국가그룹.xlsx', dtype={'CNTR_CATG_CD': str})
        >>> 일반_원수_미경과보험료['RRNR_DVCD'] = clsf_rrnr_dvcd(일반_원수_미경과보험료_가공, '원수')
        >>> 일반_원수_미경과보험료['DMFR_DVCD'] = clsf_dmfr_dvcd(일반_원수_미경과보험료)
        >>> 일반_원수_미경과보험료['CNTR_CATG_CD'] = clsf_cntr_catg_cd(일반_원수_미경과보험료, 국가그룹)
    """

    # 컬럼 존재성 검사
    if not set(['NTNL_CTRY_CD', 'ARC_INPL_CD', 'DMFR_DVCD']).issubset(data.columns):
        raise Exception('data 필수 컬럼 누락 오류')
    if not set(['CNTR_CD', 'CNTR_CATG_CD']).issubset(cntr_grp_info.columns):
        raise Exception('cntr_grp_info 필수 컬럼 누락 오류')
    if not cntr_grp_info['CNTR_CD'].is_unique:
        raise Exception('CNTR_CD 유일성 오류')
    if set(['CNTR_CD', 'CNTR_CATG_CD']).issubset(data.columns):
        data.drop(['CNTR_CD', 'CNTR_CATG_CD'], axis=1, inplace=True)

    catr_catg_cd = data \
        .merge(cntr_grp_info, left_on='NTNL_CTRY_CD', right_on='CNTR_CD', how='left') \
        .assign(CNTR_CATG_CD = lambda x: x['CNTR_CATG_CD'].fillna('#')) \
        .assign(CNTR_CATG_CD = lambda x: np.where(x['DMFR_DVCD'] == '01', '01', np.where(x['ARC_INPL_CD'].str.slice(0,5)=='10900', '04', x['CNTR_CATG_CD'])))
    
    # 전처리 누락여부 검사
    if len(catr_catg_cd.query('CNTR_CATG_CD == "#"')) != 0:
        print(catr_catg_cd.query('CNTR_CATG_CD == "#"'))
        raise Exception('전처리 누락 오류')

    return catr_catg_cd['CNTR_CATG_CD']


def get_loss_adj_rate_all(cf: pd.DataFrame, int_rate: pd.DataFrame, fwd_pd: pd.DataFrame) -> pd.DataFrame:
    """손실조정율 테이블 생성

    Args:
        cf (pd.DataFrame): 보험금 진전추이
        int_rate (pd.DataFrame): 할인율
        fwd_pd (pd.DataFrame): 선도부도율

    Returns:
        pd.DataFrame: 손실조정율 테이블
    
    Example:
        >>> 일반_보험금진전추이 = pd.read_excel(FILE_PATH / '일반_보험금진전추이.xlsx', dtype={'PDGR_CD': str, 'AY': str})
        >>> 선도부도율 = pd.read_excel(FILE_PATH / '선도부도율.xlsx')
        >>> 할인율 = pd.read_excel(FILE_PATH / '할인율.xlsx')
        >>> 손실조정율 = get_disc_factor_all(일반_보험금진전추이, 할인율.query('KICS_SCEN_NO == 1'), 선도부도율)
    """

    loss_adj_rate_all = []
    for crd_grd in np.append(fwd_pd['GRADE'].unique(), '무등급'):
        for pdgr_cd in cf['PDGR_CD'].unique():
            for cf_type in ['보험료', '보험금']:
                cf_t, cf_rate = get_cf(cf.query('PDGR_CD == @pdgr_cd'), pdgr_cd=pdgr_cd, cf_type=cf_type)
                loss_adj_rate = get_loss_adj_rate(cf_t, cf_rate, crd_grd, int_rate, fwd_pd)
                loss_adj_rate_all.append([pdgr_cd, crd_grd, loss_adj_rate, cf_type])
    loss_adj_rate_df = pd.DataFrame(loss_adj_rate_all, columns=['PDGR_CD', 'CRD_GRD', 'LOSS_ADJ_RATE', 'PRM_RSV'])
    loss_adj_rate_df = loss_adj_rate_df.pivot_table(index=['PDGR_CD', 'CRD_GRD'], columns='PRM_RSV', values='LOSS_ADJ_RATE', aggfunc=np.sum).reset_index()
    loss_adj_rate_df.columns.name = None
    loss_adj_rate_df = loss_adj_rate_df.rename(columns={'보험금': 'LOSS_ADJ_RATE_RSV', '보험료': 'LOSS_ADJ_RATE_PRM'})
    loss_adj_rate_df.insert(0, 'RRNR_DVCD', '03')
    loss_adj_rate_df = loss_adj_rate_df.loc[lambda x: x['PDGR_CD'].isin(['23', '24', '25', '26', '27', '28', '29', '31'])].reset_index(drop=True)
    return loss_adj_rate_df

def get_loss_adj_rate(cf_t: pd.Series, cf_rate: pd.Series, crd_grd: str, int_rate: pd.DataFrame, fwd_pd: pd.DataFrame) -> float:
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
        >>> get_loss_adj_rate(cf_t, cf_rate, crd_grd='무등급', int_rate=할인율.query('KICS_SCEN_NO == 1'), fwd_pd=선도부도율)
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
    loss_adj_rate = np.sum(loss*fwd_pd*disc_rate)
    return loss_adj_rate


def get_disc_factor_all(cf: pd.DataFrame, int_rate: pd.DataFrame) -> pd.DataFrame:
    """할인요소 테이블 생성

    Args:
        cf (pd.DataFrame): 보험금 진전추이
        int_rate (pd.DataFrame): 할인율

    Returns:
        pd.DataFrame: 할인요소 테이블

    Example:
        >>> 일반_보험금진전추이 = pd.read_excel(FILE_PATH / '일반_보험금진전추이.xlsx', dtype={'PDGR_CD': str, 'AY': str})
        >>> 할인율 = pd.read_excel(FILE_PATH / '할인율.xlsx')
        >>> 일반_할인요소 = get_disc_factor_all(일반_보험금진전추이, 할인율.query('KICS_SCEN_NO == 1'))
    """
    disc_fac_all = []
    for pdgr_cd in cf['PDGR_CD'].unique():
        for cf_type in ['보험료', '보험금']:
            cf_t, cf_rate = get_cf(cf.query('PDGR_CD == @pdgr_cd'), pdgr_cd=pdgr_cd, cf_type=cf_type)
            disc_fac = get_disc_factor(cf_t, cf_rate, int_rate)
            disc_fac_all.append([pdgr_cd, disc_fac, cf_type])
    disc_fac_df = pd.DataFrame(disc_fac_all, columns=['PDGR_CD', 'DISC_FAC', 'PRM_RSV'])
    disc_fac_df = disc_fac_df.pivot_table(index='PDGR_CD', columns='PRM_RSV', values='DISC_FAC', aggfunc=np.sum).reset_index()
    disc_fac_df.columns.name = None
    disc_fac_df = disc_fac_df.rename(columns={'보험금': 'DISC_FAC_RSV', '보험료': 'DISC_FAC_PRM'})
    return disc_fac_df


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


def get_loss_adj_rate_all_c(cf: pd.DataFrame, int_rate: pd.DataFrame, fwd_pd: pd.DataFrame) -> pd.DataFrame:
    """손실조정율 테이블 생성

    Args:
        cf (pd.DataFrame): 보험금 진전추이
        int_rate (pd.DataFrame): 할인율
        fwd_pd (pd.DataFrame): 선도부도율

    Returns:
        pd.DataFrame: 손실조정율 테이블
    
    Example:
        >>> 일반_보험금진전추이 = pd.read_excel(FILE_PATH / '일반_보험금진전추이.xlsx', dtype={'PDGR_CD': str, 'AY': str})
        >>> 선도부도율 = pd.read_excel(FILE_PATH / '선도부도율.xlsx')
        >>> 할인율 = pd.read_excel(FILE_PATH / '할인율.xlsx')
        >>> 손실조정율 = get_disc_factor_all(일반_보험금진전추이, 할인율.query('KICS_SCEN_NO == 1'), 선도부도율)
    """

    loss_adj_rate_all = []
    for crd_grd in np.append(fwd_pd['GRADE'].unique(), '무등급'):
        for boz_cd in cf['BOZ_CD'].unique():
            for cf_type in ['보험료', '보험금']:
                cf_t, cf_rate = get_cf_c(cf.query('BOZ_CD == @boz_cd'), boz_cd=boz_cd, cf_type=cf_type)
                loss_adj_rate = get_loss_adj_rate(cf_t, cf_rate, crd_grd, int_rate, fwd_pd)
                loss_adj_rate_all.append([boz_cd, crd_grd, loss_adj_rate, cf_type])
    loss_adj_rate_df = pd.DataFrame(loss_adj_rate_all, columns=['BOZ_CD', 'CRD_GRD', 'LOSS_ADJ_RATE', 'PRM_RSV'])
    loss_adj_rate_df = loss_adj_rate_df.pivot_table(index=['BOZ_CD', 'CRD_GRD'], columns='PRM_RSV', values='LOSS_ADJ_RATE', aggfunc=np.sum).reset_index()
    loss_adj_rate_df.columns.name = None
    loss_adj_rate_df = loss_adj_rate_df.rename(columns={'보험금': 'LOSS_ADJ_RATE_RSV', '보험료': 'LOSS_ADJ_RATE_PRM'})
    loss_adj_rate_df.insert(0, 'RRNR_DVCD', '03')
    return loss_adj_rate_df


def get_loss_adj_rate_c(cf_t: pd.Series, cf_rate: pd.Series, crd_grd: str, int_rate: pd.DataFrame, fwd_pd: pd.DataFrame) -> float:
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
        >>> get_loss_adj_rate(cf_t, cf_rate, crd_grd='무등급', int_rate=할인율.query('KICS_SCEN_NO == 1'), fwd_pd=선도부도율)
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
    loss_adj_rate = np.sum(loss*fwd_pd*disc_rate)
    return loss_adj_rate


def get_disc_factor_all_c(cf: pd.DataFrame, int_rate: pd.DataFrame) -> pd.DataFrame:
    
    disc_fac_all = []
    for boz_cd in cf['BOZ_CD'].unique():
        for cf_type in ['보험료', '보험금']:
            cf_t, cf_rate = get_cf_c(cf.query('BOZ_CD == @boz_cd'), boz_cd=boz_cd, cf_type=cf_type)
            disc_fac = get_disc_factor(cf_t, cf_rate, int_rate)
            disc_fac_all.append([boz_cd, disc_fac, cf_type])
    disc_fac_df = pd.DataFrame(disc_fac_all, columns=['BOZ_CD', 'DISC_FAC', 'PRM_RSV'])
    disc_fac_df = disc_fac_df.pivot_table(index='BOZ_CD', columns='PRM_RSV', values='DISC_FAC', aggfunc=np.sum).reset_index()
    disc_fac_df.columns.name = None
    disc_fac_df = disc_fac_df.rename(columns={'보험금': 'DISC_FAC_RSV', '보험료': 'DISC_FAC_PRM'})
    return disc_fac_df


def get_disc_factor_c(cf_t: pd.Series, cf_rate: pd.Series, int_rate: pd.DataFrame) -> float:

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


def get_cf_c(cf: pd.DataFrame, boz_cd: str, cf_type: str) -> Tuple[pd.Series, pd.Series]:
    
    n = 7
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
        .assign(PDGR_CD = lambda x: np.where(x['PDC_CD'] == '10420', '25', x['PDGR_CD'])) \
        .assign(PDGR_CD = lambda x: np.where(x['PDC_CD'] == '13804', '26', x['PDGR_CD'])) \
        .assign(PDGR_CD = lambda x: np.where(x['PDGR_CD'].isin(['30','34']), '29', x['PDGR_CD'])) \
        .assign(PDGR_CD = lambda x: np.where(x['PDC_CD'].isin(['10902', '10903']), '29', x['PDGR_CD'])) \
        .assign(PDGR_CD = lambda x: np.where(x.PDC_CD.isin(['10011', '10013', '10016', '10017', '10580', '10900', '10901']), '27', x['PDGR_CD']))
        
    # 전처리 누락여부 검사
    if len(pdgr_cd.query('PDGR_CD == "#"')) != 0:
        print(pdgr_cd.query('PDGR_CD == "#"'))
        raise Exception('전처리 누락 오류')

    return pdgr_cd['PDGR_CD']

def clsf_boz_cd_c(data: pd.DataFrame, pdgr_info: pd.DataFrame) -> pd.Series:
    """자동차 BOZ_CD 가공

    Args:
        data (pd.DataFrame): 자동차 원수/출재, 계약/보상 기초데이터
        pdgr_info (pd.DataFrame): [description]

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        pd.Series: BOZ_CD 결과

    Example:
        >>> 자동차_상품군정보 = pd.read_excel(FILE_PATH / '자동차_상품군정보.xlsx', dtype={'PDGR_CD': str, 'BSC_CVR_CD': str})
        >>> 자동차_원수_미경과보험료 = pd.read_excel(FILE_PATH / f'자동차_원수_미경과보험료_{BASE_YYMM}.xlsx', dtype={'BSC_CVR_CD': str, 'PDGR_CD': str, 'PDC_CD': str, 'INER_CHN_DVCD': str})
        >>> clsf_boz_cd_c(자동차_원수_미경과보험료, 자동차_상품군정보)
    """

    # 컬럼 존재성 검사
    if not set(['PDGR_CD', 'BSC_CVR_CD']).issubset(data.columns):
        raise Exception('data 필수 컬럼 누락 오류')
    if not set(['PDGR_CD', 'BSC_CVR_CD', 'BOZ_CD']).issubset(pdgr_info.columns):
        raise Exception('prd_info 필수 컬럼 누락 오류')
    if set(['BOZ_CD']).issubset(data.columns):
        data.drop('BOZ_CD', axis=1, inplace=True)
    if len(pdgr_info) != len(pdgr_info[['PDGR_CD', 'BSC_CVR_CD']].drop_duplicates()):
        raise Exception('pdgr_info 키값 유일성 오류')

    boz_cd = data.merge(pdgr_info, on=['PDGR_CD', 'BSC_CVR_CD'], how='left') \
        .assign(BOZ_CD = lambda x: x['BOZ_CD'].fillna('B007')) # 검토 필요

    return boz_cd['BOZ_CD']


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
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'] == '10420', 'A005', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'] == '13804', 'A006', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'] == '10607', 'A008', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'].isin(['10902', '10903']) & (x.DMFR_DVCD == '01'), 'A009', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x['PDC_CD'].isin(['10902', '10903']) & (x.DMFR_DVCD != '01'), 'A010', x['BOZ_CD'])) \
        .assign(BOZ_CD = lambda x: np.where(x.PDC_CD.isin(['10011', '10013', '10016', '10017', '10580', '10900', '10901']), 'A003', x['BOZ_CD']))

    # 전처리 누락여부 검사
    if len(boz_cd.query('BOZ_CD == "#"')) != 0:
        print(boz_cd.query('BOZ_CD == "#"'))
        raise Exception('전처리 누락 오류')

    return boz_cd['BOZ_CD']


if __name__ == '__main__':
    pass