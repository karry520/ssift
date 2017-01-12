/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		imgfeatures.cpp
Description:	��������Ҫ�ṹ
Author:			���
Version:		0.1
Date:			�������
History:�޸���ʷ��¼�б� ÿ���޸ļ�¼Ӧ�����޸����ڡ��޸��߼��޸����ݼ�����
*****************************************************************************/

#include "utils.h"
#include "imgfeatures.h"

#include <cxcore.h>

#include <math.h>

/*********************************���غ�������*******************************/

static void draw_liuli_feature(IplImage* ,struct  feature* ,CvScalar );

/****************************��������imgfeatures.h***************************/

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
void draw_liuli_features(IplImage* img,struct feature* feat,int n)
{
	CvScalar color = CV_RGB( 255, 255, 255);
	int i;
	
	if(img->nChannels > 1)
		color = FEATURE_LIULI_COLOR;
	for ( i = 0;  i < n;  i++)
	{
		draw_liuli_feature( img, feat + i, color);
	}
}
/*****************************************************************************
Function:		// draw_liuli_feature
Description:	// ��ͼ���л���������
Calls:			// cvLine��cvRound��cvPoint
Called By:		// draw_liuli_features
Table Accessed: // ��
Table Updated:	// ��
Input:			// @img		����ȡ�������ͼ��
				// @feat	�洢�����������
				// @n		��������ɫ 
Output:			// ��
Return:			// ��
Others:			// ��
*****************************************************************************/
static void draw_liuli_feature(IplImage* img,struct  feature* feat,CvScalar color)
{
	int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
	double scl, ori;
	double scale = 5.0;
	double hscale = 0.75;
	CvPoint start, end, h1, h2;

	start_x = cvRound( feat->x );
	start_y = cvRound( feat->y );
	scl = feat->scale;
	ori = feat->orientation;
	len = cvRound( scl * scale );
	hlen = cvRound( scl * hscale );
	blen = len - hlen;
	end_x = cvRound( len *  cos( ori ) ) + start_x;
	end_y = cvRound( len * -sin( ori ) ) + start_y;
	h1_x = cvRound( blen *  cos( ori + CV_PI / 18.0 ) ) + start_x;
	h1_y = cvRound( blen * -sin( ori + CV_PI / 18.0 ) ) + start_y;
	h2_x = cvRound( blen *  cos( ori - CV_PI / 18.0 ) ) + start_x;
	h2_y = cvRound( blen * -sin( ori - CV_PI / 18.0 ) ) + start_y;
	start = cvPoint( start_x, start_y );
	end = cvPoint( end_x, end_y );
	h1 = cvPoint( h1_x, h1_y );
	h2 = cvPoint( h2_x, h2_y );

	cvLine( img, start, end, color, 1, 8, 0 );
	cvLine( img, end, h1, color, 1, 8, 0 );
	cvLine( img, end, h2, color, 1, 8, 0 );
}

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

double descr_dist_sq( struct feature* f1, struct feature* f2 )
{
	double diff, dsq = 0;
	double* descr1, * descr2;
	int i, d;

	d = f1->dimension;
	if( f2->dimension != d )
		return DBL_MAX;
	descr1 = f1->descr;
	descr2 = f2->descr;

	for( i = 0; i < d; i++ )
	{
		diff = descr1[i] - descr2[i];
		dsq += diff*diff;
	}
	return dsq;
}