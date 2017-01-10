/*****************************************************************************
Copyright: 2016-2017, Likaiyun
File name: imgfeatures.h
Description: ��������Ҫ�ṹ
Author: ���
Version: 0.1
Date: �������
History: �޸���ʷ��¼�б� ÿ���޸ļ�¼Ӧ�����޸����ڡ��޸��߼��޸����ݼ�����
*****************************************************************************/

#include "cxcore.h"

/************************************�궨��**********************************/

//��������ʾ����ɫ
#define FEATURE_LIULI_COLOR CV_RGB(255,0,255)

//��������ά��
#define FEATURE_DIMENSION 12

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
