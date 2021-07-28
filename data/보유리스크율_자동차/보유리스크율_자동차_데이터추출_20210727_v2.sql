﻿-- 자동차_원수_직전3년연간경과보험료 (2021.07.28)
SELECT /*+ FULL(A) PARALLEL(A 8) */
       SUBSTR(CLG_YM,1,4) AS FY, BSC_CVR_CD, PDGR_CD,
		   SUM(ELP_PRM) AS OGL_ELP_PRM
	FROM FDMTAG02001 A
  WHERE CLG_YM BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(:ET_PARM02, 'YYYYMMDD'),-35),'YYYYMM') AND SUBSTR(:ET_PARM02,1,6)
  GROUP BY SUBSTR(CLG_YM,1,4), BSC_CVR_CD, PDGR_CD;

-- 자동차_비례출재_직전3년연간경과보험료 (2021.07.28)
SELECT /*+ FULL(A) PARALLEL(A 8) */
       SUBSTR(CLG_YM,1,4) AS FY, BSC_CVR_CD, PDGR_CD,
		   SUM(RN_ELP_PRM) AS P_RN_ELP_PRM
	FROM FDMTAG02002 A
  WHERE CLG_YM BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(:ET_PARM02, 'YYYYMMDD'),-35),'YYYYMM') AND SUBSTR(:ET_PARM02,1,6)
  GROUP BY SUBSTR(CLG_YM,1,4), BSC_CVR_CD, PDGR_CD;

-- 자동차_원수_직전3년연간손해액 (2021.07.28)
SELECT SUBSTR(CLG_YM,1,4) AS FY, BSC_CVR_CD, PDGR_CD,
       SUM(PYN_BNF+LTPD_OST_BNF-PTRM_OST_BNF+LSAT) AS OGL_LOSS
	FROM FDMTTC02001
	WHERE CLG_YM BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(:ET_PARM02, 'YYYYMMDD'),-35),'YYYYMM') AND SUBSTR(:ET_PARM02,1,6)
	GROUP BY SUBSTR(CLG_YM,1,4), BSC_CVR_CD, PDGR_CD;

-- 자동차_비례출재_직전3년연간손해액 (2021.07.28)
SELECT SUBSTR(CLG_YM,1,4) AS FY, BSC_CVR_CD, PDGR_CD,
       SUM(RN_PYN_BNF+RN_LTPD_OST_BNF-RN_PTRM_OST_BNF+RN_LSAT) AS P_RN_LOSS
	FROM FDMTTC02002
	WHERE CLG_YM BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(:ET_PARM02, 'YYYYMMDD'),-35),'YYYYMM') AND SUBSTR(:ET_PARM02,1,6)
	GROUP BY SUBSTR(CLG_YM,1,4), BSC_CVR_CD, PDGR_CD;

-- 자동차_특약보종별_직전1년경과보험료
SELECT /*+ FULL(C) FULL(D) PARALLEL(C 8) PARALLEL(D 8) */
       C.UY, C.BSC_CVR_CD, C.PDGR_CD,
       SUM(OGL_ELP_PRM) AS OGL_ELP_PRM_1YR,
       SUM(RN_ELP_PRM) AS RN_ELP_PRM_1YR
	FROM (
		SELECT /*+ FULL(A) PARALLEL(A 8) */
		       SUBSTR(UY_YM,1,4) AS UY, BSC_CVR_CD, PDGR_CD,
			 		 SUM(ELP_PRM) AS OGL_ELP_PRM
			FROM FDMTAG02001 A
			WHERE CLG_YM BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(:ET_PARM02, 'YYYYMMDD'),-11),'YYYYMM') AND SUBSTR(:ET_PARM02,1,6)
				AND PLNO IN (SELECT PLNO FROM FDMTAG02002)
			GROUP BY SUBSTR(UY_YM,1,4), BSC_CVR_CD, PDGR_CD
	) C INNER JOIN (
		SELECT /*+ FULL(B) PARALLEL(B 8) */
		       SUBSTR(UY_YM,1,4) AS UY, BSC_CVR_CD, PDGR_CD,
	         SUM(RN_ELP_PRM) AS RN_ELP_PRM
			FROM FDMTAG02002 B
			WHERE CLG_YM BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(:ET_PARM02, 'YYYYMMDD'),-11),'YYYYMM') AND SUBSTR(:ET_PARM02,1,6)
			GROUP BY SUBSTR(UY_YM,1,4), BSC_CVR_CD, PDGR_CD
	) D
		ON C.UY = D.UY
		AND C.BSC_CVR_CD = D.BSC_CVR_CD
		AND C.PDGR_CD = D.PDGR_CD
	GROUP BY C.UY, C.BSC_CVR_CD, C.PDGR_CD;


-- 자동차_특약보종별_직전1년손해액
SELECT /*+ FULL(C) FULL(D) PARALLEL(C 8) PARALLEL(D 8) */
       C.UY, C.BSC_CVR_CD, C.PDGR_CD,
       SUM(OGL_LOSS) AS OGL_LOSS_1YR,
       SUM(RN_LOSS) AS RN_LOSS_1YR
	FROM (
		SELECT /*+ FULL(A) PARALLEL(A 8) */
		       SUBSTR(UY_YM,1,4) AS UY, BSC_CVR_CD, PDGR_CD,
			 		 SUM(PYN_BNF+LTPD_OST_BNF-PTRM_OST_BNF+LSAT) AS OGL_LOSS
			FROM FDMTTC02001 A
			WHERE CLG_YM BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(:ET_PARM02, 'YYYYMMDD'),-11),'YYYYMM') AND SUBSTR(:ET_PARM02,1,6)
				AND PLNO IN (SELECT PLNO FROM FDMTTC02002)
			GROUP BY SUBSTR(UY_YM,1,4), BSC_CVR_CD, PDGR_CD
	) C INNER JOIN (
		SELECT /*+ FULL(B) PARALLEL(B 8) */
		       SUBSTR(UY_YM,1,4) AS UY, BSC_CVR_CD, PDGR_CD,
	         SUM(RN_PYN_BNF+RN_LTPD_OST_BNF-RN_PTRM_OST_BNF+RN_LSAT) AS RN_LOSS
			FROM FDMTTC02002 B
			WHERE CLG_YM BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(:ET_PARM02, 'YYYYMMDD'),-11),'YYYYMM') AND SUBSTR(:ET_PARM02,1,6)
			GROUP BY SUBSTR(UY_YM,1,4), BSC_CVR_CD, PDGR_CD
	) D
		ON C.UY = D.UY
		AND C.BSC_CVR_CD = D.BSC_CVR_CD
		AND C.PDGR_CD = D.PDGR_CD
	GROUP BY C.UY, C.BSC_CVR_CD, C.PDGR_CD;


-- 자동차_특약보종별_경과보험료
SELECT /*+ FULL(C) FULL(D) PARALLEL(C 8) PARALLEL(D 8) */
       C.UY, C.BSC_CVR_CD, C.PDGR_CD,
       SUM(OGL_ELP_PRM) AS OGL_ELP_PRM,
       SUM(RN_ELP_PRM) AS RN_ELP_PRM
	FROM (
		SELECT /*+ FULL(A) PARALLEL(A 8) */
		       SUBSTR(UY_YM,1,4) AS UY, BSC_CVR_CD, PDGR_CD,
			 		 SUM(ELP_PRM) AS OGL_ELP_PRM
			FROM FDMTAG02001 A
			WHERE CLG_YM <= SUBSTR(:ET_PARM02,1,6)
				AND PLNO IN (SELECT PLNO FROM FDMTAG02002)
			GROUP BY SUBSTR(UY_YM,1,4), BSC_CVR_CD, PDGR_CD
	) C INNER JOIN (
		SELECT /*+ FULL(B) PARALLEL(B 8) */
		       SUBSTR(UY_YM,1,4) AS UY, BSC_CVR_CD, PDGR_CD,
	         SUM(RN_ELP_PRM) AS RN_ELP_PRM
			FROM FDMTAG02002 B
			WHERE CLG_YM <= SUBSTR(:ET_PARM02,1,6)
			GROUP BY SUBSTR(UY_YM,1,4), BSC_CVR_CD, PDGR_CD
	) D
		ON C.UY = D.UY
		AND C.BSC_CVR_CD = D.BSC_CVR_CD
		AND C.PDGR_CD = D.PDGR_CD
	GROUP BY C.UY, C.BSC_CVR_CD, C.PDGR_CD;

-- 자동차_특약보종별_손해액
SELECT /*+ FULL(C) FULL(D) PARALLEL(C 8) PARALLEL(D 8) */ C.BSC_CVR_CD, C.PDGR_CD,
       SUM(OGL_LOSS) AS OGL_LOSS,
       SUM(RN_LOSS) AS RN_LOSS
	FROM (
		SELECT /*+ FULL(A) PARALLEL(A 8) */
		       BSC_CVR_CD, PDGR_CD,
			 		 SUM(PYN_BNF+LTPD_OST_BNF-PTRM_OST_BNF+LSAT) AS OGL_LOSS
			FROM FDMTTC02001 A
			WHERE CLG_YM <= SUBSTR(:ET_PARM02,1,6)
				AND PLNO IN (SELECT PLNO FROM FDMTTC02002)
			GROUP BY BSC_CVR_CD, PDGR_CD
	) C INNER JOIN (
		SELECT /*+ FULL(B) PARALLEL(B 8) */
		       BSC_CVR_CD, PDGR_CD,
	         SUM(RN_PYN_BNF+RN_LTPD_OST_BNF-RN_PTRM_OST_BNF+RN_LSAT) AS RN_LOSS
			FROM FDMTTC02002 B
			WHERE CLG_YM <= SUBSTR(:ET_PARM02,1,6)
			GROUP BY BSC_CVR_CD, PDGR_CD
	) D
		ON C.BSC_CVR_CD = D.BSC_CVR_CD
		AND C.PDGR_CD = D.PDGR_CD
	GROUP BY C.BSC_CVR_CD, C.PDGR_CD;

------------------------------------------------------------------------------------------------------------