/**
 *
 * */
#include "utils.h"
#include "imgfeatures.h"

#include <cxcore.h>

#include <math.h>
static void draw_liuli_feature(IplImage* ,struct  feature* ,CvScalar );

static void draw_liuli_features(IplImage* img,struct feature* feat,int n)
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

static void draw_liuli_feature(IplImage* img,struct  feature* feat,CvScalar color)
{
	int len, hlen, blen, start_x, start_y, end_x, end_y, h1_x, h1_y, h2_x, h2_y;
	double scl, ori;
	double scale = 5.0;
	double hscale = 0.75;
	CvPoint start, end, h1, h2;

	/* compute points for an arrow scaled and rotated by feat's scl and ori */
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