/*****************************************************************************
Copyright: 2016-2017, Likaiyun
File name: kdtree.h
Description: ��ȡͼƬ�е�������
Author: ���
Version: 0.1
Date: �������
History: �޸���ʷ��¼�б� ÿ���޸ļ�¼Ӧ�����޸����ڡ��޸��߼��޸����ݼ�����
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