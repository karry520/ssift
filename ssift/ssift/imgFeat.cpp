/*
This program detects image features using SIFT keypoints. For more info,
refer to:

Lowe, D. Distinctive image features from scale-invariant keypoints.
International Journal of Computer Vision, 60, 2 (2004), pp.91--110.

Copyright (C) 2006-2010  Rob Hess <hess@eecs.oregonstate.edu>

Note: The SIFT algorithm is patented in the United States and cannot be
used in commercial products without a license from the University of
British Columbia.  For more information, refer to the file LICENSE.ubc
that accompanied this distribution.

Version: 1.1.2-20100521
*/

#include "ssift.h"
#include "imgfeatures.h"
#include "utils.h"

#include <highgui.h>

#include <stdio.h>

/******************************** Globals ************************************/

char* img_file_name = "beaver.png";
char* out_img_name = NULL;
int display = 1;
int intvls = SSIFT_INTVLS;
double sigma = SSIFT_SIGMA;
double contr_thr = SSIFT_CONTR_THR;
int curv_thr = SSIFT_CURV_THR;
int img_dbl = SSIFT_IMG_DBL;



/********************************** Main *************************************/

int main( void )
{
	IplImage* img;
	struct feature* features;
	int n = 0;

	img = cvLoadImage( img_file_name, 1 );

	n = _ssift_features( img, &features, intvls, sigma, contr_thr, curv_thr,
		img_dbl );

	draw_liuli_features( img, features, n );
	cvNamedWindow( img_file_name, 1 );
	cvShowImage( img_file_name, img );
	cvWaitKey( 0 );


	return 0;
}
