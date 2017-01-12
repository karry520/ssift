/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		ssift.cpp
Description:	特征点提取、存储
Author:			李开运
Version:		0.1
Date:			完成日期
History:修改历史记录列表， 每条修改记录应包括修改日期、修改者及修改内容简述。
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
static double* descr_hist( IplImage*, int, int, double);
static void interp_hist_entry( double***, double, double, double, double, int, int);
//将关键点邻域统计梯度梯度直方图转化为描述子向量
static void hist_to_descr( double*,struct feature* );
//归一化描述子向量
static void normalize_descr( struct feature* );
//特征点按尺度的降序排列时用到的比较函数
static int feature_cmp( void*, void*, void* );
//释放方向直方图存储空间
static void release_descr_hist( double* );
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
	n = features->total;
	calc_feature_scales( features, sigma, intvls );
	if( img_dbl )
		adjust_for_img_dbl( features );

	//第二步：特征向量
	//compute_descriptors( features, gauss_pyr );
	calc_feature_oris( features, gauss_pyr );

	cvSeqSort( features, (CvCmpFunc)feature_cmp, NULL );
	n = features->total;
	*feat = (feature*)calloc( n, sizeof(struct feature) );
	*feat = (feature*)cvCvtSeqToArray( features, *feat, CV_WHOLE_SEQ );
	for( i = 0; i < n; i++ )
	{
		free( (*feat)[i].feature_data );
		(*feat)[i].feature_data = NULL;
	}
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
					/* 像素点的对比度大于阈值 */
					if( ABS( pixval32f( dog_pyr[o][i], r, c ) ) > prelim_contr_thr )
						if( is_extremum( dog_pyr, o, i, r, c ) )
						{
							/* 极值点精确定位 */
							feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
							if( feat )
							{
								ddata = feat_detection_data( feat );
								/* 消除边缘响应点 */
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

/*****************************************************************************
Function:		// is_extremum
Description:	// 判断给定点是否是极值点（是否是周边26个点中的最大最小）
Calls:			// pixval32f、
Called By:		// scale_space_extrema
Table Accessed: // 无
Table Updated:	// 无
Input:			// @dog_pyr		高斯差分金字塔
				// @octvs		所在组数
				// @intvls		所在层数
				// @r			所在行
				// @c			所在列
Output:			// 无
Return:			// 最大或最小值返回0 其它返回非0
Others:			// 其它说明
*****************************************************************************/
static int is_extremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
	float val = pixval32f( dog_pyr[octv][intvl], r, c );
	int i, j, k;

	/* 检查是否是最大值 */
	if( val > 0 )
	{
		for( i = -1; i <= 1; i++ )
			for( j = -1; j <= 1; j++ )
				for( k = -1; k <= 1; k++ )
					if( val < pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
						return 0;
	}

	/* 检查是否是最小值 */
	else
	{
		for( i = -1; i <= 1; i++ )
			for( j = -1; j <= 1; j++ )
				for( k = -1; k <= 1; k++ )
					if( val > pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
						return 0;
	}

	return 1;
}



/*****************************************************************************
Function:		// interp_extremum
Description:	// 插值拟合，极值点精确定位
Calls:			// interp_step、interp_contr
Called By:		// scale_space_extrema
Table Accessed: // 无
Table Updated:	// 无
Input:			// @dog_pyr		高斯差分金字塔
				// @octvs		所在组数
				// @intvls		所在层数
				// @r			所在行
				// @c			所在列
				// @intvls		金字塔每组的导数
				// @contr_thr	对比度阈值，去除不稳定特征
Output:			// 无
Return:			// 返回精确定位后的特征点结构体
Others:			// 其它说明
*****************************************************************************/
static struct feature* interp_extremum( IplImage*** dog_pyr, int octv, int intvl,
										int r, int c, int intvls, double contr_thr )
{
	struct feature* feat;
	struct detection_data* ddata;
	double xi, xr, xc, contr;
	int i = 0;

	while( i < SSIFT_MAX_INTERP_STEPS )
	{
		interp_step( dog_pyr, octv, intvl, r, c, &xi, &xr, &xc );
		if( ABS( xi ) < 0.5  &&  ABS( xr ) < 0.5  &&  ABS( xc ) < 0.5 )
			break;

		c += cvRound( xc );
		r += cvRound( xr );
		intvl += cvRound( xi );

		if( intvl < 1  ||
			intvl > intvls  ||
			c < SSIFT_IMG_BORDER  ||
			r < SSIFT_IMG_BORDER  ||
			c >= dog_pyr[octv][0]->width - SSIFT_IMG_BORDER  ||
			r >= dog_pyr[octv][0]->height - SSIFT_IMG_BORDER )
		{
			return NULL;
		}

		i++;
	}

	/* ensure convergence of interpolation */
	if( i >= SSIFT_MAX_INTERP_STEPS )
		return NULL;

	contr = interp_contr( dog_pyr, octv, intvl, r, c, xi, xr, xc );
	if( ABS( contr ) < contr_thr / intvls )
		return NULL;

	feat = new_feature();
	ddata = feat_detection_data( feat );
	feat->img_point.x = feat->x = ( c + xc ) * pow( 2.0, octv );
	feat->img_point.y = feat->y = ( r + xr ) * pow( 2.0, octv );
	ddata->r = r;
	ddata->c = c;
	ddata->octv = octv;
	ddata->intvl = intvl;
	ddata->subintvl = xi;

	return feat;
}

/*****************************************************************************
Function:		// interp_step
Description:	// 每一步拟合的操作
Calls:			// deriv_3D、hessian_3D、cvCreateMat、cvInvert、cvGEMM、cvReleaseMat
Called By:		// interp_extremum
Table Accessed: // 无
Table Updated:	// 无
Input:			// @dog_pyr		高斯差分金字塔
				// @octvs		所在组数
				// @intvls		所在层数
				// @r			所在行
				// @c			所在列
				// @xi			像素点层方向上的偏移量
				// @xr			像素点行方向上的偏移量
				// @xc			像素点列方向上的偏移量
Output:			// 无
Return:			// 返回三维方向上的偏移量
Others:			// 其它说明
*****************************************************************************/

static void interp_step( IplImage*** dog_pyr, int octv, int intvl, int r, int c,
						 double* xi, double* xr, double* xc )
{
	CvMat* dD, * H, * H_inv, X;
	double x[3] = { 0 };

	dD = deriv_3D( dog_pyr, octv, intvl, r, c );
	H = hessian_3D( dog_pyr, octv, intvl, r, c );
	H_inv = cvCreateMat( 3, 3, CV_64FC1 );
	cvInvert( H, H_inv, CV_SVD );
	cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
	cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 );

	cvReleaseMat( &dD );
	cvReleaseMat( &H );
	cvReleaseMat( &H_inv );

	*xi = x[2];
	*xr = x[1];
	*xc = x[0];
}

/*****************************************************************************
Function:		// deriv_3D
Description:	// 每一步拟合的操作计算高斯差像素点在行、列、层三个方向上的偏导
Calls:			// pixval32f、cvCreateMat、cvmSet
Called By:		// interp_step
Table Accessed: // 无
Table Updated:	// 无
Input:			// @dog_pyr		高斯差分金字塔
				// @octvs		所在组数
				// @intvls		所在层数
				// @r			所在行
				// @c			所在列
Output:			// 无
Return:			// 返回由像素点在行、列、层三个方向上的偏导构成的矩阵
Others:			// 其它说明
*****************************************************************************/
static CvMat* deriv_3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
	CvMat* dI;
	double dx, dy, ds;

	dx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) -
		pixval32f( dog_pyr[octv][intvl], r, c-1 ) ) / 2.0;
	dy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) -
		pixval32f( dog_pyr[octv][intvl], r-1, c ) ) / 2.0;
	ds = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) -
		pixval32f( dog_pyr[octv][intvl-1], r, c ) ) / 2.0;

	dI = cvCreateMat( 3, 1, CV_64FC1 );
	cvmSet( dI, 0, 0, dx );
	cvmSet( dI, 1, 0, dy );
	cvmSet( dI, 2, 0, ds );

	return dI;
}

/*****************************************************************************
Function:		// hessian_3D
Description:	// 计算像素点的3*3的海森矩阵
Calls:			// pixval32f、cvCreateMat、cvmSet
Called By:		// interp_step
Table Accessed: // 无
Table Updated:	// 无
Input:			// @dog_pyr		高斯差分金字塔
				// @octvs		所在组数
				// @intvls		所在层数
				// @r			所在行
				// @c			所在列
Output:			// 无
Return:			// 返回由像素点的海森矩阵
Others:			// 其它说明
*****************************************************************************/
static CvMat* hessian_3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
	CvMat* H;
	double v, dxx, dyy, dss, dxy, dxs, dys;

	v = pixval32f( dog_pyr[octv][intvl], r, c );
	dxx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) + 
			pixval32f( dog_pyr[octv][intvl], r, c-1 ) - 2 * v );
	dyy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) +
			pixval32f( dog_pyr[octv][intvl], r-1, c ) - 2 * v );
	dss = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) +
			pixval32f( dog_pyr[octv][intvl-1], r, c ) - 2 * v );
	dxy = ( pixval32f( dog_pyr[octv][intvl], r+1, c+1 ) -
			pixval32f( dog_pyr[octv][intvl], r+1, c-1 ) -
			pixval32f( dog_pyr[octv][intvl], r-1, c+1 ) +
			pixval32f( dog_pyr[octv][intvl], r-1, c-1 ) ) / 4.0;
	dxs = ( pixval32f( dog_pyr[octv][intvl+1], r, c+1 ) -
			pixval32f( dog_pyr[octv][intvl+1], r, c-1 ) -
			pixval32f( dog_pyr[octv][intvl-1], r, c+1 ) +
			pixval32f( dog_pyr[octv][intvl-1], r, c-1 ) ) / 4.0;
	dys = ( pixval32f( dog_pyr[octv][intvl+1], r+1, c ) -
			pixval32f( dog_pyr[octv][intvl+1], r-1, c ) -
			pixval32f( dog_pyr[octv][intvl-1], r+1, c ) +
			pixval32f( dog_pyr[octv][intvl-1], r-1, c ) ) / 4.0;

	H = cvCreateMat( 3, 3, CV_64FC1 );
	cvmSet( H, 0, 0, dxx );
	cvmSet( H, 0, 1, dxy );
	cvmSet( H, 0, 2, dxs );
	cvmSet( H, 1, 0, dxy );
	cvmSet( H, 1, 1, dyy );
	cvmSet( H, 1, 2, dys );
	cvmSet( H, 2, 0, dxs );
	cvmSet( H, 2, 1, dys );
	cvmSet( H, 2, 2, dss );

	return H;
}

/*****************************************************************************
Function:		// interp_contr
Description:	// 计算被插值后像素点的对比度
Calls:			// cvInitMatHeader、deriv_3D、cvGEMM、cvReleaseMat、pixval32f
Called By:		// scale_space_extrema
Table Accessed: // 无
Table Updated:	// 无
Input:			// @dog_pyr		高斯差分金字塔
				// @octvs		所在组数
				// @intvls		所在层数
				// @r			所在行
				// @c			所在列
				// @xi			像素点层方向上的偏移量
				// @xr			像素点行方向上的偏移量
				// @xc			像素点列方向上的偏移量
Output:			// 无
Return:			// 返回由像素点的海森矩阵
Others:			// 其它说明
*****************************************************************************/
static double interp_contr( IplImage*** dog_pyr, int octv, int intvl, int r,
							int c, double xi, double xr, double xc )
{
	CvMat* dD, X, T;
	double t[1], x[3] = { xc, xr, xi };

	cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
	cvInitMatHeader( &T, 1, 1, CV_64FC1, t, CV_AUTOSTEP );
	dD = deriv_3D( dog_pyr, octv, intvl, r, c );
	cvGEMM( dD, &X, 1, NULL, 0, &T,  CV_GEMM_A_T );
	cvReleaseMat( &dD );

	return pixval32f( dog_pyr[octv][intvl], r, c ) + t[0] * 0.5;
}

/*****************************************************************************
Function:		// new_feature
Description:	// 创建一个新的特征点(feature)结构体
Calls:			// cvInitMatHeader、deriv_3D、cvGEMM、cvReleaseMat、pixval32f
Called By:		// scale_space_extrema
Table Accessed: // 无
Table Updated:	// 无
Input:			// 无
Output:			// 无
Return:			// 返回已分配存储空间的结构体首地址
Others:			// 其它说明
*****************************************************************************/
static struct feature* new_feature( void )
{
	struct feature* feat;
	struct detection_data* ddata;

	feat = (feature*)malloc( sizeof( struct feature ) );
	memset( feat, 0, sizeof( struct feature ) );
	ddata = (detection_data*)malloc( sizeof( struct detection_data ) );
	memset( ddata, 0, sizeof( struct detection_data ) );
	feat->feature_data = ddata;

	return feat;
}

/*****************************************************************************
Function:		// is_too_edge_like
Description:	// 消除不稳定的边缘响应点
Calls:			// pixval32f、
Called By:		// scale_space_extrema
Table Accessed: // 无
Table Updated:	// 无
Input:			// @dog_pyr		高斯差分金字塔
				// @r			所在行
				// @c			所在列
				// @curv_thr	主曲率比值
Output:			// 
Return:			// 0：非边缘点 1：边缘点
Others:			// 其它说明
*****************************************************************************/
static int is_too_edge_like( IplImage* dog_img, int r, int c, int curv_thr )
{
	double d, dxx, dyy, dxy, tr, det;

	/* 主曲率与海森矩阵的特征值成正比 */
	d = pixval32f(dog_img, r, c);
	dxx = pixval32f( dog_img, r, c+1 ) + pixval32f( dog_img, r, c-1 ) - 2 * d;
	dyy = pixval32f( dog_img, r+1, c ) + pixval32f( dog_img, r-1, c ) - 2 * d;
	dxy = ( pixval32f(dog_img, r+1, c+1) - pixval32f(dog_img, r+1, c-1) -
			pixval32f(dog_img, r-1, c+1) + pixval32f(dog_img, r-1, c-1) ) / 4.0;
	tr = dxx + dyy;
	det = dxx * dyy - dxy * dxy;

	/* 与主曲率有不同的符号 */
	if( det <= 0 )
		return 1;
	/* 通过比值判断 */
	if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
		return 0;
	return 1;
}

/*****************************************************************************
Function:		// calc_feature_scales
Description:	// 计算特征点的尺度
Calls:			// CV_GET_SEQ_ELEM
Called By:		// _ssift_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @features	特征点序列
				// @sigma		图像初始尺度
				// @intvls		每组分intvls层
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static void calc_feature_scales( CvSeq* features, double sigma, int intvls )
{
	struct feature* feat;
	struct detection_data* ddata;
	double intvl;
	int i, n;

	n = features->total;
	for( i = 0; i < n; i++ )
	{
		feat = CV_GET_SEQ_ELEM( struct feature, features, i );
		ddata = feat_detection_data( feat );
		intvl = ddata->intvl + ddata->subintvl;
		feat->scale = sigma * pow( 2.0, ddata->octv + intvl / intvls );
		ddata->scl_octv = sigma * pow( 2.0, intvl / intvls );
	}
}

/*****************************************************************************
Function:		// adjust_for_img_dbl
Description:	// 缩小1倍图片的参数
Calls:			// CV_GET_SEQ_ELEM
Called By:		// _ssift_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @features	特征点序列
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static void adjust_for_img_dbl( CvSeq* features )
{
	struct feature* feat;
	int i, n;

  	n = features->total;
  	for( i = 0; i < n; i++ )
    {
    	//从序列中获取一个feature类型的元素
      	feat = CV_GET_SEQ_ELEM( struct feature, features, i );
      	feat->x /= 2.0;
      	feat->y /= 2.0;
		feat->scale /= 2.0;
		feat->img_point.x /= 2.0;
		feat->img_point.y /= 2.0;
    }
}
/*****************************************************************************
Function:		// calc_feature_oris
Description:	// 计算图像特征点的方向和描述符
Calls:			// cvSeqPopFront、ori_hist、dominant_ori、add_good_ori_features
Called By:		// _ssift_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @features	特征点序列
				// @gauss_pyr	高斯差分金字塔
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static void calc_feature_oris( CvSeq* features, IplImage*** gauss_pyr )
{
	struct feature* feat;
	struct detection_data* ddata;
	double* hist;
	double omax;
	int i, j, n = features->total;

	for( i = 0; i < n; i++ )
	{
		feat = (feature*)malloc( sizeof( struct feature ) );
		cvSeqPopFront( features, feat );
		ddata = feat_detection_data( feat );
		hist = ori_hist( gauss_pyr[ddata->octv][ddata->intvl],
						ddata->r, ddata->c, SSIFT_ORI_HIST_BINS,
						cvRound( SSIFT_ORI_RADIUS * ddata->scl_octv ),
						SSIFT_ORI_SIG_FCTR * ddata->scl_octv );
		for( j = 0; j < SSIFT_ORI_SMOOTH_PASSES; j++ )
			smooth_ori_hist( hist, SSIFT_ORI_HIST_BINS );
		omax = dominant_ori( hist, SSIFT_ORI_HIST_BINS );
		add_good_ori_features( features, hist, SSIFT_ORI_HIST_BINS,
								omax * SSIFT_ORI_PEAK_RATIO, feat );
		free( ddata );
		free( feat );
		free( hist );
	}
}



/*****************************************************************************
Function:		// ori_hist
Description:	// 计算特征点方向直方图
Calls:			// calc_grad_mag_ori、cvRound
Called By:		// calc_feature_oris
Table Accessed: // 无
Table Updated:	// 无
Input:			// @features	特征点序列
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static double* ori_hist( IplImage* img, int r, int c, int n, int rad, double sigma)
{
	double* hist;
	double mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
	int bin, i, j;

	hist = (double*)calloc( n, sizeof( double ) );
	exp_denom = 2.0 * sigma * sigma;
	for( i = -rad; i <= rad; i++ )
		for( j = -( cvRound( sqrt( rad*rad - i*i ) ) ); j <= ( cvRound( sqrt( rad*rad - i*i ) ) ); j++ )
			if( calc_grad_mag_ori( img, r + i, c + j, &mag, &ori ) )
			{
				w = exp( -( i*i + j*j ) / exp_denom );
				bin = cvRound( n * ( ori + CV_PI ) / PI2 );
				bin = ( bin < n )? bin : 0;
				hist[bin] += w * mag;
			}

	return hist;
}

/*****************************************************************************
Function:		// calc_grad_mag_ori
Description:	// 计算像素点的模值和方向
Calls:			// pixval32f、sqrt、atan2
Called By:		// calc_feature_oris
Table Accessed: // 无
Table Updated:	// 无
Input:			// @img		特征点序列
				// @r		特征点行
				// @c		特征点列
				// @mag		将计算后的模值存入其中
				// @ori		将计算后的方向存入其中
Output:			// 
Return:			// 计算成功返回 1		其它返回 0
Others:			// 其它说明
*****************************************************************************/
static int calc_grad_mag_ori( IplImage* img, int r, int c, double* mag, double* ori )
{
	double dx, dy;

	if( r > 0  &&  r < img->height - 1  &&  c > 0  &&  c < img->width - 1 )
	{
		dx = pixval32f( img, r, c+1 ) - pixval32f( img, r, c-1 );
		dy = pixval32f( img, r-1, c ) - pixval32f( img, r+1, c );
		*mag = sqrt( dx*dx + dy*dy );
		*ori = atan2( dy, dx );
		return 1;
	}

	else
		return 0;
}

/*****************************************************************************
Function:		// smooth_ori_hist
Description:	// 高斯平滑方向直方图
Calls:			// 无
Called By:		// calc_feature_oris
Table Accessed: // 无
Table Updated:	// 无
Input:			// @hist	方向直方图 
				// n		直方图的维数
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static void smooth_ori_hist( double* hist, int n )
{
	double prev, tmp, h0 = hist[0];
	int i;

	prev = hist[n-1];
	for( i = 0; i < n; i++ )
	{
		tmp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] + 
			0.25 * ( ( i+1 == n )? h0 : hist[i+1] );
		prev = tmp;
	}
}

/*****************************************************************************
Function:		// dominant_ori
Description:	// 计算直方图中最大的值 
Calls:			// 无
Called By:		// calc_feature_oris
Table Accessed: // 无
Table Updated:	// 无
Input:			// @hist	直方图 
				// @n		直方图的维数
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static double dominant_ori( double* hist, int n )
{
	double omax;
	int maxbin, i;

	omax = hist[0];
	maxbin = 0;
	for( i = 1; i < n; i++ )
		if( hist[i] > omax )
		{
			omax = hist[i];
			maxbin = i;
		}
	return omax;
}

/*
	插值	
*/
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )

/*****************************************************************************
Function:		// add_good_ori_features
Description:	// 将好的特征点方向及描述子添加至特征点序列中
Calls:			// interp_hist_peak、clone_feature、hist_to_descr
Called By:		// calc_feature_oris
Table Accessed: // 无
Table Updated:	// 无
Input:			// @features	特征点序列
				// @hist		直方图
				// @n			直方图的维数
				// @mag_thr		模值阈值
				// @feat		特征点结构
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static void add_good_ori_features( CvSeq* features, double* hist, int n,
								   double mag_thr, struct feature* feat )
{
	struct feature* new_feat;
	double bin, PI2 = CV_PI * 2.0;
	int l, r, i;

	for( i = 0; i < n; i++ )
	{
		l = ( i == 0 )? n - 1 : i-1;
		r = ( i + 1 ) % n;

		if( hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thr )
		{
			bin = i + interp_hist_peak( hist[l], hist[i], hist[r] );
			bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;
			new_feat = clone_feature( feat );
			new_feat->orientation = ( ( PI2 * bin ) / n ) - CV_PI;

			hist_to_descr(hist,new_feat);

			cvSeqPush( features, new_feat );
			free( new_feat );
		}
	}
}

/*****************************************************************************
Function:		// hist_to_descr
Description:	// 将直方图作为向量存入特征点结构体中
Calls:			// normalize_descr
Called By:		// add_good_ori_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @hist	直方图 
				// @feat	特征点结构体
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static void hist_to_descr( double* hist,struct feature* feat)
{
	int int_val;
	for (int i = 0; i < SSIFT_ORI_HIST_BINS; i++)
	{
		feat->descr[i] = hist[i];
	}
	feat->dimension = SSIFT_ORI_HIST_BINS;
	normalize_descr( feat );
	for( int i = 0; i < SSIFT_ORI_HIST_BINS; i++ )
		if( feat->descr[i] > SSIFT_DESCR_MAG_THR )
			feat->descr[i] = SSIFT_DESCR_MAG_THR;
	normalize_descr( feat );
	for( int i = 0; i < SSIFT_ORI_HIST_BINS; i++ )
	{
		int_val = SSIFT_INT_DESCR_FCTR * feat->descr[i];
		feat->descr[i] = MIN( 255, int_val );
	}
}
/*****************************************************************************
Function:		// normalize_descr、
Description:	// 归一化
Calls:			// 无
Called By:		// hist_to_descr
Table Accessed: // 无
Table Updated:	// 无
Input:			// @feat	特征点结构体
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static void normalize_descr( struct feature* feat )
{
	double cur, len_inv, len_sq = 0.0;
	int i, d = feat->dimension;

	for( i = 0; i < d; i++ )
	{
		cur = feat->descr[i];
		len_sq += cur*cur;
	}
	len_inv = 1.0 / sqrt( len_sq );
	for( i = 0; i < d; i++ )
		feat->descr[i] *= len_inv;
}

/*****************************************************************************
Function:		// clone_feature
Description:	// 复制一个特征点
Calls:			// 无
Called By:		// add_good_ori_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @feat	特征点结构体
Output:			// 
Return:			// 复制后的特征点指针
Others:			// 其它说明
*****************************************************************************/
static struct feature* clone_feature( struct feature* feat )
{
	struct feature* new_feat;
	struct detection_data* ddata;

	new_feat = new_feature();
	ddata = feat_detection_data( new_feat );
	memcpy( new_feat, feat, sizeof( struct feature ) );
	memcpy( ddata, feat_detection_data(feat), sizeof( struct detection_data ) );
	new_feat->feature_data = ddata;

	return new_feat;
}

/*****************************************************************************
Function:		// release_pyr
Description:	// 释放金字塔空间
Calls:			// cvReleaseImage
Called By:		// _ssift_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @pyr		需要释放的金字塔指针 
				// @octvs	金字塔分为多少组
				// @n		每组的图像数
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static void release_pyr( IplImage**** pyr, int octvs, int n )
{
	int i, j;
	for( i = 0; i < octvs; i++ )
	{
		for( j = 0; j < n; j++ )
			cvReleaseImage( &(*pyr)[i][j] );
		free( (*pyr)[i] );
	}
	free( *pyr );
	*pyr = NULL;
}

/*****************************************************************************
Function:		// feature_cmp
Description:	// 比较特征点的尺度排序用
Calls:			// 无
Called By:		// _ssift_features
Table Accessed: // 无
Table Updated:	// 无
Input:			// @feat1		特征点1
				// @feat2		特征点2
				// @param		比较时回调函数
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
static int feature_cmp( void* feat1, void* feat2, void* param )
{
	struct feature* f1 = (struct feature*) feat1;
	struct feature* f2 = (struct feature*) feat2;

	if( f1->scale < f2->scale )
		return 1;
	if( f1->scale > f2->scale )
		return -1;
	return 0;
}
