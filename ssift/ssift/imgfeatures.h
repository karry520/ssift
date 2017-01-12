/*****************************************************************************
Copyright: 2016-2017, Likaiyun
File name: imgfeatures.h
Description: ��������Ҫ�ṹ
Author: ���
Version: 0.1
Date: �������
History: �޸���ʷ��¼�б� ÿ���޸ļ�¼Ӧ�����޸����ڡ��޸��߼��޸����ݼ�����
*****************************************************************************/
#ifndef IMGFEATURES_H
#define IMGFEATURES_H

#include "cxcore.h"

/************************************�궨��**********************************/

//��������ʾ����ɫ
#define FEATURE_LIULI_COLOR CV_RGB(255,0,255)

//��������ά��
#define FEATURE_DIMENSION 12

enum feature_match_type
{
	FEATURE_FWD_MATCH,
	FEATURE_BCK_MATCH,
	FEATURE_MDL_MATCH,
};
/************************************�ṹ��**********************************/

//������ṹ��
struct feature
{
	double x;							//x����
	double y;							//y����
	double scale;						//�߶�
	double orientation;					//����
	int dimension;						//ά��
	double descr[FEATURE_DIMENSION];	//������
	struct feature* fwd_match;     /**< matching feature from forward image */
	struct feature* bck_match;     /**< matching feature from backmward image */
	struct feature* mdl_match;     /**< matching feature from model */
	CvPoint2D64f mdl_pt;           /**< location in model */
	CvPoint2D64f img_point;				//������
	void* feature_data;					//�Զ�������
};

/*****************************************************************************
Function:		// draw_liuli_features
Description:	// ��ͼ���л���������
Calls:			// draw_liuli_feature
Called By:		// imgFeat.main
Table Accessed: // ��
Table Updated:	// ��
Input:			// @img		����ȡ�������ͼ��
				// @feat	�洢�����������
				// @n		����������
Output:			// ��
Return:			// ��
Others:			// ��
*****************************************************************************/
extern void draw_liuli_features(IplImage* img,struct feature* feat,int n);
extern double descr_dist_sq( struct feature* f1, struct feature* f2 );
#endif