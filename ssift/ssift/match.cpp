
/*
Detects SIFT features in two images and finds matches between them.

Copyright (C) 2006-2010  Rob Hess <hess@eecs.oregonstate.edu>

@version 1.1.2-20100521
*/

#include "ssift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <stdio.h>


/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

/******************************** Globals ************************************/

char img1_file[] = "beaver.png";
char img2_file[] = "beaver_xform.png";

/********************************** Main *************************************/

double time0 = 0,time1 = 0,time2 = 0;


int main( int argc, char** argv )
{
	IplImage* img1, * img2, * stacked;
	struct feature* feat1, * feat2, * feat;
	struct feature** nbrs;
	struct kd_node* kd_root;
	CvPoint pt1, pt2;
	double d0, d1;
	int n1, n2, k, i, m = 0;

    struct feature **inliers;
    int n_inliers;

	img1 = cvLoadImage( img1_file, 1 );
	if( ! img1 )
		fatal_error( "unable to load image from %s", img1_file );
	img2 = cvLoadImage( img2_file, 1 );
	if( ! img2 )
		fatal_error( "unable to load image from %s", img2_file );
	stacked = stack_imgs( img1, img2 );

	fprintf( stderr, "Finding features in %s...\n", img1_file );
	time0 = (double)cvGetTickCount();
	n1 = ssift_features( img1, &feat1 );
	time0 = ((double)cvGetTickCount()) - time0;
	printf( "img1 ssift = %gus\n", time0/(cvGetTickFrequency()) );
	
	fprintf( stderr, "Finding features in %s...\n", img2_file );
	time1 = (double)cvGetTickCount();
	n2 = ssift_features( img2, &feat2 );
	time1 = ((double)cvGetTickCount()) - time1;
	printf( "img2 ssift = %gus\n", time1/(cvGetTickFrequency()) );
	time2 = (double)cvGetTickCount();
	kd_root = kdtree_build( feat2, n2 );
	for( i = 0; i < n1; i++ )
	{
		feat = feat1 + i;
		k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
		if( k == 2 )
		{
			d0 = descr_dist_sq( feat, nbrs[0] );
			d1 = descr_dist_sq( feat, nbrs[1] );
			if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
			{
				m++;
				feat1[i].fwd_match = nbrs[0];
			}
		}
		free( nbrs );
	}

	{
		CvMat* H;
		H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
			homog_xfer_err, 3.0, &inliers,&n_inliers );
		time2 = ((double)cvGetTickCount()) - time2;
		printf( "match = %gus\n", time2/(cvGetTickFrequency()) );
		if( H )
		{

			int i;
			for( i = 0; i<n_inliers; i++)
			{
				feat = inliers[i];
				pt1 = cvPoint(cvRound(feat->x), cvRound(feat->y));
				pt2 = cvPoint(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));
      
				pt2.y += img1->height;
				cvLine(stacked,pt1,pt2,CV_RGB(255,0,255),1,8,0);
			}

		}
	}

	fprintf( stderr, "Found %d total matches\n", m );
	cvNamedWindow( "Matches", CV_WINDOW_NORMAL );
	cvShowImage( "Matches", stacked );

	cvWaitKey( 0 );	

	cvReleaseImage( &stacked );
	cvReleaseImage( &img1 );
	cvReleaseImage( &img2 );
	kdtree_release( kd_root );
	free( feat1 );
	free( feat2 );
	return 0;
}
