#include "cxcore.h"
//��������ʾ����ɫ
#define FEATURE_LIULI_COLOR CV_RGB(255,0,255)

//��������ά��
#define FEATURE_DIMENSION 12

//������ṹ��
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
 *����ͼƬ�е�������
 */

extern void draw_liuli_features(IplImage* img,struct feature* feat,int n);
