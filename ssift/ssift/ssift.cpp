/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		ssift
Description:	��ȡͼƬ�е�������
Author:			���
Version:		0.1
Date:			�������
History:		�޸���ʷ��¼�б� ÿ���޸ļ�¼Ӧ�����޸����ڡ��޸��߼��޸����ݼ�����
*****************************************************************************/

#include "ssift.h"
#include "imgfeatures.h"
#include "utils.h"

#include <cxcore.h>
#include <cv.h>

/*****************************���غ���ԭ��**********************************/ 

//��ʼ������ͼ��
static IplImage* create_init_img( IplImage*, int, double );
//������ͼ��ת��Ϊ32λ�Ҷ�ͼ,�����й�һ��
static IplImage* convert_to_gray32( IplImage* );
//������˹������
static IplImage*** build_gauss_pyr( IplImage*, int, int, double );
//ͼ�񽵲���
static IplImage* downsample( IplImage* );
//������˹��ֽ�����
static IplImage*** build_dog_pyr( IplImage***, int, int );
//�߶ȿռ伫ֵ���
static CvSeq* scale_space_extrema( IplImage***, int, int, double, int, CvMemStorage*);
//̽�����ص��Ƿ�������Χ�ڵļ�ֵ��
static int is_extremum( IplImage***, int, int, int, int );
//��ֵʵ�ּ�ֵ�㾫ȷ��λ
static struct feature* interp_extremum( IplImage***, int, int, int, int, int, double);
//����һ�μ�ֵ���ֵ������x��y���ҷ���(�㷽��)�ϵ�������ƫ��
static void interp_step( IplImage***, int, int, int, int, double*, double*, double* );
//��DoG�������м������ص�x����y�����Լ��߶ȷ����ϵ�ƫ��
static CvMat* deriv_3D( IplImage***, int, int, int, int );
//���ص��3*3��ɭ����
static CvMat* hessian_3D( IplImage***, int, int, int, int );
//���㱻��ֵ��ĶԱȶ�
static double interp_contr( IplImage***, int, int, int, int, double, double, double );
//����һ��feature���͵Ľṹ��
static struct feature* new_feature( void );
//�ж�ĳ���Ƿ��Ե��
static int is_too_edge_like( IplImage*, int, int, int );
//����������������ÿ��������ĳ߶�
static void calc_feature_scales( CvSeq*, double, int );
//��������������ÿ����������������
static void adjust_for_img_dbl( CvSeq* );
//������������ݶ�ֱ��ͼ
static void calc_feature_oris( CvSeq*, IplImage*** );
//�����ص����ֱ��ͳ��
static double* ori_hist( IplImage*, int, int, int, int, double );
//����ָ������ݶȵķ�ֵmagnitude�ͷ���orientation
static int calc_grad_mag_ori( IplImage*, int, int, double*, double* );
//���ݶȷ���ֱ��ͼ���и�˹ƽ��
static void smooth_ori_hist( double*, int );
//�ؼ���������
static double dominant_ori( double*, int );
//��Ӹ�����
static void add_good_ori_features( CvSeq*, double*, int, double, struct feature* );
//copy feature���͵Ľṹ��
static struct feature* clone_feature( struct feature* );
//������������������
static void compute_descriptors( CvSeq*, IplImage***);
//���������㸽������ķ���ֱ��ͼ
static double*** descr_hist( IplImage*, int, int, double, double, int, int );
static void interp_hist_entry( double***, double, double, double, double, int, int);
//���ؼ�������ͳ���ݶ��ݶ�ֱ��ͼת��Ϊ����������
static void hist_to_descr( double***, int, int, struct feature* );
//��һ������������
static void normalize_descr( struct feature* );
//�����㰴�߶ȵĽ�������ʱ�õ��ıȽϺ���
static int feature_cmp( void*, void*, void* );
//�ͷŷ���ֱ��ͼ�洢�ռ�
static void release_descr_hist( double****, int );
//�ͷ�ͼ��������洢�ռ�
static void release_pyr( IplImage****, int, int );


/*****************************�������� ssift.h*******************************/ 

/*****************************************************************************
Function:		// ssift_features
Description:	// ��ȡͼ��������
Calls:			// _ssift_features
Called By:		// imgFeat.main
Table Accessed: // ��
Table Updated:	// ��
Input:			// @img		����ȡ�������ͼ��
				// @feat	��ȡ����������洢����
Output:			// ��ȡ��������������
Return:			// ��������ȡʧ��	������������������ 
Others:			// ��
*****************************************************************************/

int ssift_features( IplImage* img, struct feature** feat )
{
	return _ssift_features( img, feat, SSIFT_INTVLS, SSIFT_SIGMA, SSIFT_CONTR_THR,
		SSIFT_CURV_THR, SSIFT_IMG_DBL);
}

/*****************************************************************************
Function:		// _ssift_features
Description:	// ��ȡͼ��������
Calls:			// create_init_img��build_gauss_pyr��
				// build_dog_pyr��scale_space_extrema��
				// calc_feature_scales��compute_descriptors��
				// cvReleaseMemStorage��release_pyr��
				// adjust_for_img_dbl��cvReleaseImage��
				// log��cvCreateMemStorage��
Called By:		// ssift_features
Table Accessed: // ��
Table Updated:	// ��
Input:			// @img			����ȡ�������ͼ��
				// @feat		��ȡ����������洢����
				// @intvls		�߶ȿռ���ÿ���Ϊ���ٲ�
				// @sigma		��ʼͼ��ĳ߶ȣ���˹ƽ���ĦҲ�����
				// @contr_thr	�Աȶ���ֵ
				// @curv_thr	������ԵЧӦʱ���õı�ֵ
				// @img_dbl		�Ƿ���ͼ��Ĳ���
Output:			// ��ȡ��������������
Return:			// ��������ȡʧ��	������������������ 
Others:			// ��
*****************************************************************************/
int _ssift_features( IplImage* img, struct feature** feat, int intvls,
					double sigma, double contr_thr, int curv_thr,
					int img_dbl )
{
	//�������
	IplImage* init_img;
	IplImage*** gauss_pyr, *** dog_pyr;
	CvMemStorage* storage;
	CvSeq* features;
	int octvs,i,n = 0;

	//������
	if( ! img )
		fatal_error( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

	if( ! feat )
		fatal_error( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

	//���㲽������ͼƬ�������߶ȿռ�
	init_img = create_init_img( img, img_dbl, sigma );
	octvs = log( MIN( init_img->width, init_img->height ) ) / log(2) - 2;
	gauss_pyr = build_gauss_pyr( init_img, octvs, intvls, sigma );
	dog_pyr = build_dog_pyr( gauss_pyr, octvs, intvls );

	//��һ������ֵ��� 
	storage = cvCreateMemStorage( 0 );
	features = scale_space_extrema( dog_pyr, octvs, intvls, contr_thr,
		curv_thr, storage );
	calc_feature_scales( features, sigma, intvls );
	if( img_dbl )
		adjust_for_img_dbl( features );

	//�ڶ�������������
	compute_descriptors( features, gauss_pyr );

	//�ͷ��ڴ�ռ�
	cvReleaseMemStorage( &storage );
	cvReleaseImage( &init_img );
	release_pyr( &gauss_pyr, octvs, intvls + 3 );
	release_pyr( &dog_pyr, octvs, intvls + 2 );

	return n;
}


/*****************************���غ�������*******************************/ 

/*****************************************************************************
Function:		// create_init_img
Description:	// ͼ���ʼ������
Calls:			// convert_to_gray32��cvCreateImage��cvSmooth��cvResize��cvReleaseImage
Called By:		// _ssift_features
Table Accessed: // ��
Table Updated:	// ��
Input:			// @img		�����ͼ��
				// @img_dbl	����ͼ���Ƿ���
				// @sigma	ͼ���ʼ�߶�
Output:			// ��
Return:			// ��ʼ�����ͼ�� 
Others:			// ����˵��
*****************************************************************************/
static IplImage* create_init_img( IplImage* img, int img_dbl, double sigma )
{
	IplImage* gray, * dbl;
	float sig_diff;

	//��ͼ��ת��Ϊ32λ�Ҷ�ͼ
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
Description:	// ��ͼ��ת��Ϊ32λ�Ҷ�ͼ
Calls:			// cvCreateImage��cvClone��cvCvtColor��cvConvertScale
Called By:		// create_init_img
Table Accessed: // ��
Table Updated:	// ��
Input:			// ��Ҫת����ͼ��ָ��
Output:			// ��
Return:			// ת�����32λ�Ҷ�ͼ
Others:			// ����˵��
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
Description:	// ������˹ͼ�������
Calls:			// pow��sqrt��cvCloneImage��downsample��cvSmooth��cvCreateImage
Called By:		// _ssift_features
Table Accessed: // ��
Table Updated:	// ��
Input:			// @base	��ʼ�����32λ�Ҷ�ͼ�� 
				// @octvs	�������ֶ�����
				// @intvls	ÿ����ٲ�
				// @sigma	ͼ���ʼ�߶�
Output:			// ��
Return:			// ��˹ͼ�������ָ��
Others:			// ����˵��
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
Description:	// ͼ�񽵲���
Calls:			// cvCreateImage��cvResize
Called By:		// build_gauss_pyr
Table Accessed: // ��
Table Updated:	// ��
Input:			// ��Ҫ��������ͼ��ָ��
Output:			// ��
Return:			// ���������ͼ��
Others:			// ����˵��
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
Description:	// ������˹��ֽ����� 
Calls:			// cvCreateImage
Called By:		// _ssift_features��cvSub
Table Accessed: // �����ʵı����������ǣ�������ݿ�����ĳ���
Table Updated:	// ���޸ĵı����������ǣ�������ݿ�����ĳ���
Input:			// �������˵��������ÿ����������// �á�ȡֵ˵�����������ϵ
Output:			// �����������˵��
Return:			// ��������ֵ��˵��
Others:			// ����˵��
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
Description:	// �߶ȿռ伫ֵ��⣬����⵽�ļ�ֵ����뵽CvSeq������
Calls:			// cvCreateSeq��is_extremum��is_too_edge_like��cvSeqPush
Called By:		// _ssift_features
Table Accessed: // ��
Table Updated:	// ��
Input:			// @dog_pyr		��˹��ֽ�����
				// @octvs		����
				// @intvls		����
				// @contr_thr	�Աȶ���ֵ��ȥ�����ȶ�����
				// @curv_thr	������Ե��Ӧʱ���õı�ֵ
Output:			// ��
Return:			// �ҵ���������洢����
Others:			// ����˵��
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
					/* ���ص�ĶԱȶȴ�����ֵ */
					if( ABS( pixval32f( dog_pyr[o][i], r, c ) ) > prelim_contr_thr )
						if( is_extremum( dog_pyr, o, i, r, c ) )
						{
							/* ��ֵ�㾫ȷ��λ */
							feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
							if( feat )
							{
								ddata = feat_detection_data( feat );
								/* ������Ե��Ӧ�� */
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
Description:	// �жϸ������Ƿ��Ǽ�ֵ�㣨�Ƿ����ܱ�26�����е������С��
Calls:			// pixval32f��
Called By:		// scale_space_extrema
Table Accessed: // ��
Table Updated:	// ��
Input:			// @dog_pyr		��˹��ֽ�����
				// @octvs		��������
				// @intvls		���ڲ���
				// @r			������
				// @c			������
Output:			// ��
Return:			// ������Сֵ����0 �������ط�0
Others:			// ����˵��
*****************************************************************************/
static int is_extremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
	float val = pixval32f( dog_pyr[octv][intvl], r, c );
	int i, j, k;

	/* ����Ƿ������ֵ */
	if( val > 0 )
	{
		for( i = -1; i <= 1; i++ )
			for( j = -1; j <= 1; j++ )
				for( k = -1; k <= 1; k++ )
					if( val < pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
						return 0;
	}

	/* ����Ƿ�����Сֵ */
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
Description:	// ��ֵ��ϣ���ֵ�㾫ȷ��λ
Calls:			// interp_step��interp_contr
Called By:		// scale_space_extrema
Table Accessed: // ��
Table Updated:	// ��
Input:			// @dog_pyr		��˹��ֽ�����
				// @octvs		��������
				// @intvls		���ڲ���
				// @r			������
				// @c			������
				// @intvls		������ÿ��ĵ���
				// @contr_thr	�Աȶ���ֵ��ȥ�����ȶ�����
Output:			// ��
Return:			// ���ؾ�ȷ��λ���������ṹ��
Others:			// ����˵��
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
Description:	// ÿһ����ϵĲ���
Calls:			// deriv_3D��hessian_3D��cvCreateMat��cvInvert��cvGEMM��cvReleaseMat
Called By:		// interp_extremum
Table Accessed: // ��
Table Updated:	// ��
Input:			// @dog_pyr		��˹��ֽ�����
				// @octvs		��������
				// @intvls		���ڲ���
				// @r			������
				// @c			������
				// @xi			���ص�㷽���ϵ�ƫ����
				// @xr			���ص��з����ϵ�ƫ����
				// @xc			���ص��з����ϵ�ƫ����
Output:			// ��
Return:			// ������ά�����ϵ�ƫ����
Others:			// ����˵��
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
Description:	// ÿһ����ϵĲ��������˹�����ص����С��С������������ϵ�ƫ��
Calls:			// pixval32f��cvCreateMat��cvmSet
Called By:		// interp_step
Table Accessed: // ��
Table Updated:	// ��
Input:			// @dog_pyr		��˹��ֽ�����
				// @octvs		��������
				// @intvls		���ڲ���
				// @r			������
				// @c			������
Output:			// ��
Return:			// ���������ص����С��С������������ϵ�ƫ�����ɵľ���
Others:			// ����˵��
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
Description:	// �������ص��3*3�ĺ�ɭ����
Calls:			// pixval32f��cvCreateMat��cvmSet
Called By:		// interp_step
Table Accessed: // ��
Table Updated:	// ��
Input:			// @dog_pyr		��˹��ֽ�����
				// @octvs		��������
				// @intvls		���ڲ���
				// @r			������
				// @c			������
Output:			// ��
Return:			// ���������ص�ĺ�ɭ����
Others:			// ����˵��
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
Description:	// ���㱻��ֵ�����ص�ĶԱȶ�
Calls:			// cvInitMatHeader��deriv_3D��cvGEMM��cvReleaseMat��pixval32f
Called By:		// scale_space_extrema
Table Accessed: // ��
Table Updated:	// ��
Input:			// @dog_pyr		��˹��ֽ�����
				// @octvs		��������
				// @intvls		���ڲ���
				// @r			������
				// @c			������
				// @xi			���ص�㷽���ϵ�ƫ����
				// @xr			���ص��з����ϵ�ƫ����
				// @xc			���ص��з����ϵ�ƫ����
Output:			// ��
Return:			// ���������ص�ĺ�ɭ����
Others:			// ����˵��
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
Description:	// ����һ���µ�������(feature)�ṹ��
Calls:			// cvInitMatHeader��deriv_3D��cvGEMM��cvReleaseMat��pixval32f
Called By:		// scale_space_extrema
Table Accessed: // ��
Table Updated:	// ��
Input:			// ��
Output:			// ��
Return:			// �����ѷ���洢�ռ�Ľṹ���׵�ַ
Others:			// ����˵��
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
Description:	// �������ȶ��ı�Ե��Ӧ��
Calls:			// pixval32f��
Called By:		// scale_space_extrema
Table Accessed: // ��
Table Updated:	// ��
Input:			// @dog_pyr		��˹��ֽ�����
				// @r			������
				// @c			������
				// @curv_thr	�����ʱ�ֵ
Output:			// 
Return:			// 0���Ǳ�Ե�� 1����Ե��
Others:			// ����˵��
*****************************************************************************/
static int is_too_edge_like( IplImage* dog_img, int r, int c, int curv_thr )
{
	double d, dxx, dyy, dxy, tr, det;

	/* �������뺣ɭ���������ֵ������ */
	d = pixval32f(dog_img, r, c);
	dxx = pixval32f( dog_img, r, c+1 ) + pixval32f( dog_img, r, c-1 ) - 2 * d;
	dyy = pixval32f( dog_img, r+1, c ) + pixval32f( dog_img, r-1, c ) - 2 * d;
	dxy = ( pixval32f(dog_img, r+1, c+1) - pixval32f(dog_img, r+1, c-1) -
			pixval32f(dog_img, r-1, c+1) + pixval32f(dog_img, r-1, c-1) ) / 4.0;
	tr = dxx + dyy;
	det = dxx * dyy - dxy * dxy;

	/* ���������в�ͬ�ķ��� */
	if( det <= 0 )
		return 1;
	/* ͨ����ֵ�ж� */
	if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
		return 0;
	return 1;
}
