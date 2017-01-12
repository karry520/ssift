/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		ssift.h
Description:	特征点提取、存储
Author:			李开运
Version:		0.1
Date:			完成日期
History:修改历史记录列表， 每条修改记录应包括修改日期、修改者及修改内容简述。
*****************************************************************************/
#ifndef SSIFT_H
#define SSIFT_H

#include "cxcore.h"

//探测特征点时所用到的结构体
struct detection_data
{
	int r;				//特征点所在的行
	int c;				//特征点所在的列
	int octv;			//特征点所在组
	int intvl;			//特征点所在层
	double subintvl;	//特征点层方向上的亚像素偏移量
	double scl_octv;	//特征点所在组的尺度
};

struct feature;

/************************************常量定义*********************************/

/** 高斯金字塔每组内的层数 */
#define SSIFT_INTVLS			3

/** 第0层的初始尺度 */
#define SSIFT_SIGMA				1.6

/** 对比度阈值 */
#define SSIFT_CONTR_THR			0.04

/** 消除边缘响应时的比值 */
#define SSIFT_CURV_THR			10

/** 是否倍增图像大小 */
#define SSIFT_IMG_DBL			1

/** 输入图像的尺度 */
#define SSIFT_INIT_SIGMA		0.5

/** 图像的边界 */
#define SSIFT_IMG_BORDER		5

/** 最大迭代次数 */
#define SSIFT_MAX_INTERP_STEPS	5

/** 方向直方图的维数 */
#define SSIFT_ORI_HIST_BINS		12

/** 关键点方向分配中梯度的模值加成的系数 */
#define SSIFT_ORI_SIG_FCTR		1.5

/** 关键点方向分配中，领域半径 */
#define SSIFT_ORI_RADIUS		3.0 * SSIFT_ORI_SIG_FCTR

/** 关键点方向分配中，对方向直方图的平滑次数 */
#define SSIFT_ORI_SMOOTH_PASSES 2

/** 关键点方向分配时，保留最大峰值80%的梯度方向 */
#define SSIFT_ORI_PEAK_RATIO	0.8

/** 对于归一化后的描述子，对其向量中的元素阈值 */
#define SSIFT_DESCR_MAG_THR		0.2

/** 浮点型与整型描述子之间转换相关的系数 */
#define SSIFT_INT_DESCR_FCTR	512.0

/** 将feature结构中feature_data成员的地址取出来 */
#define feat_detection_data(f) ( (struct detection_data*)(f->feature_data) )

/************************************函数原型*********************************/

/*****************************************************************************
Function:		// ssift_features
Description:	// 提取图像特征点
Calls:			// _ssift_features
Called By:		// imgFeat.main
Table Accessed: // 无
Table Updated:	// 无
Input:			// @img		被提取特征点的图像
				// @feat	提取到的特征点存储其中
Output:			// 提取到的特征点数量
Return:			// 其它：提取失败	正整数：特征点数量 
Others:			// 无
*****************************************************************************/
extern int ssift_features(IplImage* img,struct feature** feat);

/*****************************************************************************
Function:		// _ssift_features
Description:	// 提取图像特征点
Calls:			// create_init_img、build_gauss_pyr、
				// build_dog_pyr、scale_space_extrema、
				// calc_feature_scales、compute_descriptors、
				// cvReleaseMemStorage、release_pyr、
				// adjust_for_img_dbl、cvReleaseImage、
				// log、cvCreateMemStorage、
Called By:		// ssift_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @img			被提取特征点的图像
				// @feat		提取到的特征点存储其中
				// @intvls		尺度空间中每组分为多少层
				// @sigma		初始图像的尺度（高斯平滑的σ参数）
				// @contr_thr	对比度阈值
				// @curv_thr	消除边缘效应时所用的比值
				// @img_dbl		是否倍增图像的参数
Output:			// 提取到的特征点数量
Return:			// 其它：提取失败	正整数：特征点数量 
Others:			// 无
*****************************************************************************/
extern int _ssift_features( IplImage* img, struct feature** feat, int intvls,
						   double sigma, double contr_thr, int curv_thr,
						   int img_dbl );

#endif 
