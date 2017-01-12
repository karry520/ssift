/*****************************************************************************
Copyright: 2016-2017, Likaiyun
File name: kdtree.h
Description: 提取图片中的特征点
Author: 李开运
Version: 0.1
Date: 完成日期
History: 修改历史记录列表， 每条修改记录应包括修改日期、修改者及修改内容简述。
*****************************************************************************/

#ifndef KDTREE_H
#define KDTREE_H

#include "cxcore.h"

struct kd_node
{
	int ki;
	double kv;
	int leaf;
	struct feature* features;
	int n;
	struct kd_node* kd_left;
	struct kd_node* kd_right;
};

extern struct kd_node* kdtree_build( struct feature* features, int n );
extern int kdtree_bbf_knn( struct kd_node* kd_root, struct feature* feat,
						  int k, struct feature*** nbrs, int max_nn_chks );
extern int kdtree_bbf_spatial_knn( struct kd_node* kd_root,
								struct feature* feat, int k,
								struct feature*** nbrs, int max_nn_chks,
								CvRect rect, int model );
extern void kdtree_release( struct kd_node* kd_root );
#endif 