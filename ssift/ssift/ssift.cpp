/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		ssift
Description:	提取图片中的特征点
Author:			李开运
Version:		0.1
Date:			完成日期
History:		修改历史记录列表， 每条修改记录应包括修改日期、修改者及修改内容简述。
*****************************************************************************/

#include "ssift.h"
#include "imgfeatures.h"
#include "utils.h"

#include <cxcore.h>
#include <cv.h>

/*****************************本地函数原型**********************************/ 

//初始化输入图像
static IplImage* create_init_img( IplImage*, int, double );
//将输入图像转换为32位灰度图,并进行归一化
static IplImage* convert_to_gray32( IplImage* );
//构建高斯金字塔
static IplImage*** build_gauss_pyr( IplImage*, int, int, double );
//图像降采样
static IplImage* downsample( IplImage* );
//构建高斯差分金字塔
static IplImage*** build_dog_pyr( IplImage***, int, int );
//尺度空间极值检测
static CvSeq* scale_space_extrema( IplImage***, int, int, double, int, CvMemStorage*);
//探测像素点是否是邻域范围内的极值点
static int is_extremum( IplImage***, int, int, int, int );
//插值实现极值点精确定位
static struct feature* interp_extremum( IplImage***, int, int, int, int, int, double);
//进行一次极值点插值，计算x，y，σ方向(层方向)上的子像素偏移
static void interp_step( IplImage***, int, int, int, int, double*, double*, double* );
//在DoG金字塔中计算像素的x方向、y方向以及尺度方向上的偏导
static CvMat* deriv_3D( IplImage***, int, int, int, int );
//像素点的3*3海森矩阵
static CvMat* hessian_3D( IplImage***, int, int, int, int );
//计算被插值点的对比度
static double interp_contr( IplImage***, int, int, int, int, double, double, double );
//创建一个feature类型的结构体
static struct feature* new_feature( void );
//判断某点是否边缘点
static int is_too_edge_like( IplImage*, int, int, int );
//计算特征点序列中每个特征点的尺度
static void calc_feature_scales( CvSeq*, double, int );
//将特征点序列中每个特征点的坐标减半
static void adjust_for_img_dbl( CvSeq* );
//计算特征点的梯度直方图
static void calc_feature_oris( CvSeq*, IplImage*** );
//对像素点进行直方统计
static double* ori_hist( IplImage*, int, int, int, int, double );
//计算指定点的梯度的幅值magnitude和方向orientation
static int calc_grad_mag_ori( IplImage*, int, int, double*, double* );
//对梯度方向直方图进行高斯平滑
static void smooth_ori_hist( double*, int );
//关键点主方向
static double dominant_ori( double*, int );
//添加辅方向
static void add_good_ori_features( CvSeq*, double*, int, double, struct feature* );
//copy feature类型的结构体
static struct feature* clone_feature( struct feature* );
//计算特征描述子向量
static void compute_descriptors( CvSeq*, IplImage***);
//计算特征点附近区域的方向直方图
static double*** descr_hist( IplImage*, int, int, double, double, int, int );
static void interp_hist_entry( double***, double, double, double, double, int, int);
//将关键点邻域统计梯度梯度直方图转化为描述子向量
static void hist_to_descr( double***, int, int, struct feature* );
//归一化描述子向量
static void normalize_descr( struct feature* );
//特征点按尺度的降序排列时用到的比较函数
static int feature_cmp( void*, void*, void* );
//释放方向直方图存储空间
static void release_descr_hist( double****, int );
//释放图像金字塔存储空间
static void release_pyr( IplImage****, int, int );


/*****************************函数声明 ssift.h*******************************/ 

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

int ssift_features( IplImage* img, struct feature** feat )
{
	return _ssift_features( img, feat, SSIFT_INTVLS, SSIFT_SIGMA, SSIFT_CONTR_THR,
		SSIFT_CURV_THR, SSIFT_IMG_DBL);
}

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
int _ssift_features( IplImage* img, struct feature** feat, int intvls,
					double sigma, double contr_thr, int curv_thr,
					int img_dbl )
{
	//定义变量
	IplImage* init_img;
	IplImage*** gauss_pyr, *** dog_pyr;
	CvMemStorage* storage;
	CvSeq* features;
	int octvs,i,n = 0;

	//检查参数
	if( ! img )
		fatal_error( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

	if( ! feat )
		fatal_error( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

	//第零步：处理图片、构建尺度空间
	init_img = create_init_img( img, img_dbl, sigma );
	octvs = log( MIN( init_img->width, init_img->height ) ) / log(2) - 2;
	gauss_pyr = build_gauss_pyr( init_img, octvs, intvls, sigma );
	dog_pyr = build_dog_pyr( gauss_pyr, octvs, intvls );

	//第一步：极值检测 
	storage = cvCreateMemStorage( 0 );
	features = scale_space_extrema( dog_pyr, octvs, intvls, contr_thr,
		curv_thr, storage );
	calc_feature_scales( features, sigma, intvls );
	if( img_dbl )
		adjust_for_img_dbl( features );

	//第二步：特征向量
	compute_descriptors( features, gauss_pyr );

	//释放内存空间
	cvReleaseMemStorage( &storage );
	cvReleaseImage( &init_img );
	release_pyr( &gauss_pyr, octvs, intvls + 3 );
	release_pyr( &dog_pyr, octvs, intvls + 2 );

	return n;
}


/*****************************本地函数声明*******************************/ 

/*****************************************************************************
Function:		// create_init_img
Description:	// 图像初始化处理
Calls:			// convert_to_gray32、cvCreateImage、cvSmooth、cvResize、cvReleaseImage
Called By:		// _ssift_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @img		输入的图像
				// @img_dbl	输入图像是否倍增
				// @sigma	图像初始尺度
Output:			// 无
Return:			// 初始化后的图像 
Others:			// 其它说明
*****************************************************************************/
static IplImage* create_init_img( IplImage* img, int img_dbl, double sigma )
{
	IplImage* gray, * dbl;
	float sig_diff;

	//将图像转化为32位灰度图
	gray = convert_to_gray32( img );
	if( img_dbl )
	{
		sig_diff = sqrt( sigma * sigma - SSIFT_INIT_SIGMA * SSIFT_INIT_SIGMA * 4 );
		dbl = cvCreateImage( cvSize( img->width*2, img->height*2 ),
			IPL_DEPTH_32F, 1 );
		cvResize( gray, dbl, CV_INTER_CUBIC );
		cvSmooth( dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
		cvReleaseImage( &gray );
		return dbl;
	}
	else
	{
		sig_diff = sqrt( sigma * sigma - SSIFT_INIT_SIGMA * SSIFT_INIT_SIGMA );
		cvSmooth( gray, gray, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
		return gray;
	}
}



/*****************************************************************************
Function:		// convert_to_gray32
Description:	// 将图像转化为32位灰度图
Calls:			// cvCreateImage、cvClone、cvCvtColor、cvConvertScale
Called By:		// create_init_img
Table Accessed: // 无
Table Updated:	// 无
Input:			// 需要转化的图像指针
Output:			// 无
Return:			// 转化后的32位灰度图
Others:			// 其它说明
*****************************************************************************/
static IplImage* convert_to_gray32( IplImage* img )
{
	IplImage* gray8, * gray32;

	gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
	if( img->nChannels == 1 )
		gray8 = (IplImage*)cvClone( img );
	else
	{
		gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
		cvCvtColor( img, gray8, CV_BGR2GRAY );
	}
	cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

	cvReleaseImage( &gray8 );
	return gray32;
}


/*****************************************************************************
Function:		// build_gauss_pyr
Description:	// 构建高斯图像金字塔
Calls:			// pow、sqrt、cvCloneImage、downsample、cvSmooth、cvCreateImage
Called By:		// _ssift_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @base	初始化后的32位灰度图像 
				// @octvs	金字塔分多少组
				// @intvls	每组多少层
				// @sigma	图像初始尺度
Output:			// 无
Return:			// 高斯图像金字塔指针
Others:			// 其它说明
*****************************************************************************/
static IplImage*** build_gauss_pyr( IplImage* base, int octvs,
								   int intvls, double sigma )
{
	IplImage*** gauss_pyr;
	double* sig = (double*)calloc( intvls + 3, sizeof(double));
	double sig_total, sig_prev, k;
	int i, o;

	gauss_pyr = (IplImage***)calloc( octvs, sizeof( IplImage** ) );
	for( i = 0; i < octvs; i++ )
		gauss_pyr[i] = (IplImage**)calloc( intvls + 3, sizeof( IplImage* ) );

	sig[0] = sigma;
	k = pow( 2.0, 1.0 / intvls );
	for( i = 1; i < intvls + 3; i++ )
	{
		sig_prev = pow( k, i - 1 ) * sigma;
		sig_total = sig_prev * k;
		sig[i] = sqrt( sig_total * sig_total - sig_prev * sig_prev );
	}

	for( o = 0; o < octvs; o++ )
		for( i = 0; i < intvls + 3; i++ )
		{
			if( o == 0  &&  i == 0 )
				gauss_pyr[o][i] = cvCloneImage(base);

			/* base of new octvave is halved image from end of previous octave */
			else if( i == 0 )
				gauss_pyr[o][i] = downsample( gauss_pyr[o-1][intvls] );

			/* blur the current octave's last image to create the next one */
			else
			{
				gauss_pyr[o][i] = cvCreateImage( cvGetSize(gauss_pyr[o][i-1]),
					IPL_DEPTH_32F, 1 );
				cvSmooth( gauss_pyr[o][i-1], gauss_pyr[o][i],
					CV_GAUSSIAN, 0, 0, sig[i], sig[i] );
			}
		}

		free( sig );
		return gauss_pyr;
}


/*****************************************************************************
Function:		// downsample
Description:	// 图像降采样
Calls:			// cvCreateImage、cvResize
Called By:		// build_gauss_pyr
Table Accessed: // 无
Table Updated:	// 无
Input:			// 需要降采样的图像指针
Output:			// 无
Return:			// 降采样后的图像
Others:			// 其它说明
*****************************************************************************/
static IplImage* downsample( IplImage* img )
{
	IplImage* smaller = cvCreateImage( cvSize(img->width / 2, img->height / 2),
		img->depth, img->nChannels );
	cvResize( img, smaller, CV_INTER_NN );

	return smaller;
}


/*****************************************************************************
Function:		// build_dog_pyr
Description:	// 构建高斯差分金字塔 
Calls:			// cvCreateImage
Called By:		// _ssift_features、cvSub
Table Accessed: // 被访问的表（此项仅对于牵扯到数据库操作的程序）
Table Updated:	// 被修改的表（此项仅对于牵扯到数据库操作的程序）
Input:			// 输入参数说明，包括每个参数的作// 用、取值说明及参数间关系
Output:			// 对输出参数的说明
Return:			// 函数返回值的说明
Others:			// 其它说明
*****************************************************************************/
static IplImage*** build_dog_pyr( IplImage*** gauss_pyr, int octvs, int intvls )
{
	IplImage*** dog_pyr;
	int i, o;

	dog_pyr = (IplImage***)calloc( octvs, sizeof( IplImage** ) );
	for( i = 0; i < octvs; i++ )
		dog_pyr[i] = (IplImage**)calloc( intvls + 2, sizeof(IplImage*) );

	for( o = 0; o < octvs; o++ )
		for( i = 0; i < intvls + 2; i++ )
		{
			dog_pyr[o][i] = cvCreateImage( cvGetSize(gauss_pyr[o][i]),
				IPL_DEPTH_32F, 1 );
			cvSub( gauss_pyr[o][i+1], gauss_pyr[o][i], dog_pyr[o][i], NULL );
		}

		return dog_pyr;
}

/*****************************************************************************
Function:		// scale_space_extrema
Description:	// 尺度空间极值检测，将检测到的极值点加入到CvSeq序列中
Calls:			// cvCreateSeq、is_extremum、is_too_edge_like、cvSeqPush
Called By:		// _ssift_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @dog_pyr		高斯差分金字塔
				// @octvs		组数
				// @intvls		层数
				// @contr_thr	对比度阈值，去除不稳定特征
				// @curv_thr	消除边缘响应时所用的比值
Output:			// 无
Return:			// 找到的特征点存储序列
Others:			// 其它说明
*****************************************************************************/
static CvSeq* scale_space_extrema( IplImage*** dog_pyr, int octvs, int intvls,
								   double contr_thr, int curv_thr,
								   CvMemStorage* storage )
{
	CvSeq* features;
	double prelim_contr_thr = 0.5 * contr_thr / intvls;
	struct feature* feat;
	struct detection_data* ddata;
	int o, i, r, c;

	features = cvCreateSeq( 0, sizeof(CvSeq), sizeof(struct feature), storage );
	for( o = 0; o < octvs; o++ )
		for( i = 1; i <= intvls; i++ )
			for(r = SSIFT_IMG_BORDER; r < dog_pyr[o][0]->height-SSIFT_IMG_BORDER; r++)
				for(c = SSIFT_IMG_BORDER; c < dog_pyr[o][0]->width-SSIFT_IMG_BORDER; c++)
					/* perform preliminary check on contrast */
					if( ABS( pixval32f( dog_pyr[o][i], r, c ) ) > prelim_contr_thr )
						if( is_extremum( dog_pyr, o, i, r, c ) )
						{
							feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
							if( feat )
							{
								ddata = feat_detection_data( feat );
								if( ! is_too_edge_like( dog_pyr[ddata->octv][ddata->intvl],
									ddata->r, ddata->c, curv_thr ) )
								{
									cvSeqPush( features, feat );
								}
								else
									free( ddata );
								free( feat );
							}
						}

	return features;
}