/*
	BSC_CVR_CD : 대인1=01, 대인2=02, 대물=03, 자기신체/자동차상해=04, 무보험=05, 자차=06, 기계장치=07, 다른자동차운전=08, SOS=09, 기본담보=10
	PDC_CD :  프로미카개인용=20001, 다이렉트개인용=20014, 프로미카업무용=20016, 다이렉트개인용=20054, 다이렉트업무용=20023, 프로미카영업용=20025, 다이렉트영업용=20027, 다이렉트업무용=20055, 이륜자동차보험=20028, 건설기계영업용=20058, 영업용(공동)=20042, 업무용(공동)=20041, 개인용(공동)=20040, 취급업자보험=20035, 다이렉트이륜차=20057 등
	PDGR_CD : 개인용=08, 업무용=09, 영업용=10, 이륜차=11, 외화표시=12, 운전자=13, 취급업자=14, 모터바이크=15, 농기계=16, 운전면허교습생=17, FREEDRIVER=18, 개인용공동=19, 업무용공동=20, 영업용공동=21, 이륜차공동=22
	INER_CHN_DVCD : 직판=01, 방카=02, 교차=03, GA=04, 홈쇼핑=05, 카드=06, POM=07, CM=08, 미디어=09, TM=10, 직급=11, 교육용설계사=12, 교육용대리점=13, CM(완결형)=14, 홈쇼핑(광고비)=15, 표준=99
	SL_CVR_CD : 대물배상책임(의무)=200003, 대물배상책임=200004, 대인Ⅰ배상책임=200001, 대인Ⅱ배상책임=200002, 자기차량손해(차대차)=200137, 자기차량손해(차대차이외)=200138, 대물배상책임=200004, 자동차상해=200006 등
	USTK_DVCD : 일반=1, 공동=2
*/


/**********************************************************/
/*            Ⅰ. 자동차_원수_미경과보험료 ('19.12 기준)            */
/**********************************************************/

-- 1. (전체) 1조 8,050억
SELECT SUM(LTPD_URND_PRM)/1E8 AS LTPD_URND_PRM
	FROM FDMTAG02001
	WHERE CLG_YM = SUBSTR(:EP_PARM02,1,6);

-- 2. [기본담보별] (대물) 6,747억, (자차) 3,575억, (대인1) 3,218억, (대인2) 3,071억 (자신/자상) 699억 등
SELECT BSC_CVR_CD, SUM(LTPD_URND_PRM)/1E8 AS LTPD_URND_PRM
	FROM FDMTAG02001
	WHERE CLG_YM = SUBSTR(:EP_PARM02,1,6)
	GROUP BY BSC_CVR_CD
	ORDER BY 2 DESC;

-- 3. [상품군별] (개인용) 1조 2,815억, (업무용) 3,536억, (영업용) 1,265억 (이륜차) 252억 (영업용공동) 61억
SELECT PDGR_CD, SUM(LTPD_URND_PRM)/1E8 AS LTPD_URND_PRM
	FROM FDMTAG02001
	WHERE CLG_YM = SUBSTR(:EP_PARM02,1,6)
	GROUP BY PDGR_CD
	ORDER BY 2 DESC;

-- 4. [채널별] (GA) 5,895억, (직판) 5,219억, (표준) 4,112억, (CM) 2,650억, (교차) 159억
SELECT INER_CHN_DVCD, SUM(LTPD_URND_PRM)/1E8 AS LTPD_URND_PRM
	FROM FDMTAG02001
	WHERE CLG_YM = SUBSTR(:EP_PARM02,1,6)
	GROUP BY INER_CHN_DVCD
	ORDER BY 2 DESC;

-- 5. [상품별] (프로미카개인용) 6,488억, (다이렉트개인용) 3,940억, (프로미카업무용) 2,474억, (다이렉트개인용) 2,385억, (다이렉트업무용) 825억, (프로미카영업용) 732억, (다이렉트영업용) 434억, (다이렉트업무용) 238억, (이륜자동차보험) 228억, (건설기계영업용) 98억, (영업용(공동)) 61억, (업무용(공동)) 41억, (개인용(공동)) 36억, (취급업자보험) 29억, (다이렉트이륜차) 25억 등
--SELECT * FROM FBMTEX00013 WHERE ASUM_INPD_DVCD='2' AND BSE_YM = '201812'; -- 상품코드
SELECT PDC_CD, SUM(LTPD_URND_PRM)/1E8 AS LTPD_URND_PRM
	FROM FDMTAG02001
	WHERE CLG_YM = SUBSTR(:EP_PARM02,1,6)
	GROUP BY PDC_CD
	ORDER BY 2 DESC;

-- 6. [판매담보별] (대물(의무)) 5,191억, (대인1) 3,218억, (대인2) 3,061억, (자차(차대차)) 1,962억, (자차(차대차외)) 1,572억, (대물) 1,545억, (자상) 555억 등
--SELECT * FROM FDMTTP01002 WHERE INPD_DVCD = '2'; -- 판매담보코드
SELECT SL_CVR_CD, SUM(LTPD_URND_PRM)/1E8 AS LTPD_URND_PRM
	FROM FDMTAG02001
	WHERE CLG_YM = SUBSTR(:EP_PARM02,1,6)
	GROUP BY SL_CVR_CD
	ORDER BY 2 DESC;

-- 7. [인수구분별] (일반) 1조 7,904억, (공동) 146억
SELECT USTK_DVCD, SUM(LTPD_URND_PRM)/1E8 AS LTPD_URND_PRM
	FROM FDMTAG02001
	WHERE CLG_YM = SUBSTR(:EP_PARM02,1,6)
	GROUP BY USTK_DVCD
	ORDER BY 2 DESC;
