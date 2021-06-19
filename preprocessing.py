import numpy as np
import pandas as pd

def clsf_rrnr_dvcd(data: pd.DataFrame, ret_type: str) -> pd.Series:
    """[summary]

    Args:
        data (pd.DataFrame): 일반 원수/출재, 계약/보상 기초데이터
        ret_type (str): 원수 또는 출재

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