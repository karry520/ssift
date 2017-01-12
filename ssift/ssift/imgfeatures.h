/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		imgfeatures.h
Description:	��������Ҫ�ṹ
Author:			���
Version:		0.1
Date:			�������
History:�޸���ʷ��¼�б� ÿ���޸ļ�¼Ӧ�����޸����ڡ��޸��߼��޸����ݼ�����
*****************************************************************************/
#ifndef IMGFEATURES_H
#define IMGFEATURES_H

#include "cxcore.h"

/************************************�궨��**********************************/

//��������ʾ����ɫ
#define FEATURE_LIULI_COLOR CV_RGB(255,0,255)

//��������ά��
#define FEATURE_DIMENSION 12

//ƥ������
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
	struct feature* fwd_match;			//
	struct feature* bck_match;			//
	struct feature* mdl_match;			//
	CvPoint2D64f mdl_pt;				//
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

/*****************************************************************************
Function:		// descr_dist_sq
Description:	// ������������֮��ľ���
Calls:			// 
Called By:		// kdtree_bbf_knn��
Table Accessed: // ��
Table Updated:	// ��
Input:			// @f1	������1
				// @f2	������2
				// @n	����֮��ľ���
Output:			// ��
Return:			// ��
Others:			// ��
*****************************************************************************/

extern double descr_dist_sq( struct feature* f1, struct feature* f2 );

#endif