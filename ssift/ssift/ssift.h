/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		ssift.h
Description:	��������ȡ���洢
Author:			���
Version:		0.1
Date:			�������
History:�޸���ʷ��¼�б� ÿ���޸ļ�¼Ӧ�����޸����ڡ��޸��߼��޸����ݼ�����
*****************************************************************************/
#ifndef SSIFT_H
#define SSIFT_H

#include "cxcore.h"

//̽��������ʱ���õ��Ľṹ��
struct detection_data
{
	int r;				//���������ڵ���
	int c;				//���������ڵ���
	int octv;			//������������
	int intvl;			//���������ڲ�
	double subintvl;	//������㷽���ϵ�������ƫ����
	double scl_octv;	//������������ĳ߶�
};

struct feature;

/************************************��������*********************************/

/** ��˹������ÿ���ڵĲ��� */
#define SSIFT_INTVLS			3

/** ��0��ĳ�ʼ�߶� */
#define SSIFT_SIGMA				1.6

/** �Աȶ���ֵ */
#define SSIFT_CONTR_THR			0.04

/** ������Ե��Ӧʱ�ı�ֵ */
#define SSIFT_CURV_THR			10

/** �Ƿ���ͼ���С */
#define SSIFT_IMG_DBL			1

/** ����ͼ��ĳ߶� */
#define SSIFT_INIT_SIGMA		0.5

/** ͼ��ı߽� */
#define SSIFT_IMG_BORDER		5

/** ���������� */
#define SSIFT_MAX_INTERP_STEPS	5

/** ����ֱ��ͼ��ά�� */
#define SSIFT_ORI_HIST_BINS		12

/** �ؼ��㷽��������ݶȵ�ģֵ�ӳɵ�ϵ�� */
#define SSIFT_ORI_SIG_FCTR		1.5

/** �ؼ��㷽������У�����뾶 */
#define SSIFT_ORI_RADIUS		3.0 * SSIFT_ORI_SIG_FCTR

/** �ؼ��㷽������У��Է���ֱ��ͼ��ƽ������ */
#define SSIFT_ORI_SMOOTH_PASSES 2

/** �ؼ��㷽�����ʱ����������ֵ80%���ݶȷ��� */
#define SSIFT_ORI_PEAK_RATIO	0.8

/** ���ڹ�һ����������ӣ����������е�Ԫ����ֵ */
#define SSIFT_DESCR_MAG_THR		0.2

/** ������������������֮��ת����ص�ϵ�� */
#define SSIFT_INT_DESCR_FCTR	512.0

/** ��feature�ṹ��feature_data��Ա�ĵ�ַȡ���� */
#define feat_detection_data(f) ( (struct detection_data*)(f->feature_data) )

/************************************����ԭ��*********************************/

/*****************************************************************************
Function:		// ssift_features
Description:	// ��ȡͼ��������
Calls:			// _ssift_features
Called By:		// imgFeat.main
Table Accessed: // ��
Table Updated:	// ��
Input:			// @img		����ȡ�������ͼ��
				// @feat	��ȡ����������洢����
Output:			// ��ȡ��������������
Return:			// ��������ȡʧ��	������������������ 
Others:			// ��
*****************************************************************************/
extern int ssift_features(IplImage* img,struct feature** feat);

/*****************************************************************************
Function:		// _ssift_features
Description:	// ��ȡͼ��������
Calls:			// create_init_img��build_gauss_pyr��
				// build_dog_pyr��scale_space_extrema��
				// calc_feature_scales��compute_descriptors��
				// cvReleaseMemStorage��release_pyr��
				// adjust_for_img_dbl��cvReleaseImage��
				// log��cvCreateMemStorage��
Called By:		// ssift_features
Table Accessed: // ��
Table Updated:	// ��
Input:			// @img			����ȡ�������ͼ��
				// @feat		��ȡ����������洢����
				// @intvls		�߶ȿռ���ÿ���Ϊ���ٲ�
				// @sigma		��ʼͼ��ĳ߶ȣ���˹ƽ���ĦҲ�����
				// @contr_thr	�Աȶ���ֵ
				// @curv_thr	������ԵЧӦʱ���õı�ֵ
				// @img_dbl		�Ƿ���ͼ��Ĳ���
Output:			// ��ȡ��������������
Return:			// ��������ȡʧ��	������������������ 
Others:			// ��
*****************************************************************************/
extern int _ssift_features( IplImage* img, struct feature** feat, int intvls,
						   double sigma, double contr_thr, int curv_thr,
						   int img_dbl );

#endif 
