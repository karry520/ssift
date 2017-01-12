/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		imgfeatures.h
Description:	特征点重要结构
Author:			李开运
Version:		0.1
Date:			完成日期
History:修改历史记录列表， 每条修改记录应包括修改日期、修改者及修改内容简述。
*****************************************************************************/
#ifndef IMGFEATURES_H
#define IMGFEATURES_H

#include "cxcore.h"

/************************************宏定义**********************************/

//特征点显示的颜色
#define FEATURE_LIULI_COLOR CV_RGB(255,0,255)

//描述向量维数
#define FEATURE_DIMENSION 12

//匹配类型
enum feature_match_type
{
	FEATURE_FWD_MATCH,
	FEATURE_BCK_MATCH,
	FEATURE_MDL_MATCH,
};
/************************************结构体**********************************/

//特征点结构体
struct feature
{
	double x;							//x坐标
	double y;							//y坐标
	double scale;						//尺度
	double orientation;					//方向
	int dimension;						//维数
	double descr[FEATURE_DIMENSION];	//描述符
	struct feature* fwd_match;			//
	struct feature* bck_match;			//
	struct feature* mdl_match;			//
	CvPoint2D64f mdl_pt;				//
	CvPoint2D64f img_point;				//点坐标
	void* feature_data;					//自定义数据
};

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

extern void draw_liuli_features(IplImage* img,struct feature* feat,int n);

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

extern double descr_dist_sq( struct feature* f1, struct feature* f2 );

#endif