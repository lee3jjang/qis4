import numpy as np
import pandas as pd

def clsf_boz_cd(data: pd.DataFrame, prd_info: pd.DataFrame) -> pd.Series:
    """각 row 별 boz_cd를 맵핑

    Args:
        data (pd.DataFrame): 처리대상 테이블 (필수컬럼 : ['ARC_INPL_CD', 'NTNL_CTRY_CD'])
        prd_info (pd.DataFrame): 상품정보 (필수컬럼 : ['PDC_CD', 'PDGR_CD'])

    Returns:
        pd.Series: boz_cd 맵핑 결과

    Example:
        >>> data = pd.DataFrame([
                    ['1090201', 'KR'],
                    ['1060701', 'KR'],
                    ['1040010', 'GB'],
                ], columns = ['ARC_INPL_CD', 'NTNL_CTRY_CD'])
        >>> prd_info = pd.DataFrame([
                    ['10902', '29'],
                    ['10607', '28'],
                    ['10400', '23'],
                ], columns = ['PDC_CD', 'PDGR_CD'])
        >>> clsf_boz_cd(data, prd_info)
        0    A009
        1    A008
        2    A001
        Name: BOZ_CD, dtype: object
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

    boz_cd = data.assign(PDC_CD = lambda x: x['ARC_INPL_CD'].str.slice(0,5)) \
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