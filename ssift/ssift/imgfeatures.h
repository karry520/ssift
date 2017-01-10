#include "cxcore.h"
//特征点显示的颜色
#define FEATURE_LIULI_COLOR CV_RGB(255,0,255)

//描述向量维数
#define FEATURE_DIMENSION 12

//特征点结构体
struct feature
{
	double x;
	double y;
	double scale;
	double orientation;
	int dimension;
	double descr[FEATURE_DIMENSION];
	CvPoint2D64f img_point;
	void* feature_data;
};

/**
 *画出图片中的特征点
 */

extern void draw_liuli_features(IplImage* img,struct feature* feat,int n);
