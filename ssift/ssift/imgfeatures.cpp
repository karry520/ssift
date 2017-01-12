/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		imgfeatures.cpp
Description:	特征点重要结构
Author:			李开运
Version:		0.1
Date:			完成日期
History:修改历史记录列表， 每条修改记录应包括修改日期、修改者及修改内容简述。
*****************************************************************************/

#include "utils.h"
#include "imgfeatures.h"

#include <cxcore.h>

#include <math.h>

/*********************************本地函数申明*******************************/

static void draw_liuli_feature(IplImage* ,struct  feature* ,CvScalar );

/****************************函数定义imgfeatures.h***************************/

/*****************************************************************************
Function:		// draw_liuli_features
Description:	// 在图像中画出特征点
Calls:			// draw_liuli_feature
Called By:		// imgFeat.main
Table Accessed: // 无
Table Updated:	// 无
Input:			// @img		被提取特征点的图像
				// @feat	存储特征点的序列
				// @n		特征点数量
Output:			// 无
Return:			// 无
Others:			// 无
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
Description:	// 在图像中画出特征点
Calls:			// cvLine、cvRound、cvPoint
Called By:		// draw_liuli_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @img		被提取特征点的图像
				// @feat	存储特征点的序列
				// @n		特征点颜色 
Output:			// 无
Return:			// 无
Others:			// 无
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
Description:	// 计算两特征点之间的距离
Calls:			// 
Called By:		// kdtree_bbf_knn、
Table Accessed: // 无
Table Updated:	// 无
Input:			// @f1	特征点1
				// @f2	特征点2
				// @n	两点之间的距离
Output:			// 无
Return:			// 无
Others:			// 无
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