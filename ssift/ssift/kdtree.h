/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		kdtree.h
Description:	kd-tree-bbf�㷨
Author:			���
Version:		0.1
Date:			�������
History:�޸���ʷ��¼�б� ÿ���޸ļ�¼Ӧ�����޸����ڡ��޸��߼��޸����ݼ�����
*****************************************************************************/

#ifndef KDTREE_H
#define KDTREE_H

#include "cxcore.h"
/************************************�ṹ��**********************************/

//kd_tree��Ҫ���ݽṹ
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
/************************************��������*********************************/

/*****************************************************************************
Function:		// kdtree_build
Description:	// ���������㼯�Ͻ���k-d��
Calls:			// ��
Called By:		// matchs.main
Table Accessed: // ��
Table Updated:	// ��
Input:			// @featrues	�����㼯
				// @n			���������
Output:			// 
Return:			// ���س�ʼ����kd_tree�����ڵ㣨kd_node��
Others:			// ����˵��
*****************************************************************************/
extern struct kd_node* kdtree_build( struct feature* features, int n );

/*****************************************************************************
Function:		// kdtree_bbf_knn
Description:	// BBF�㷨Ѱ��k����������
Calls:			// ��
Called By:		// matchs.main
Table Accessed: // ��
Table Updated:	// ��
Input:			// @kd_root		k-d�������
				// @feat		Ŀ������������������
				// @k			��������
				// @nbrs		k������������
				// @max_nn_chks	�����������
Output:			// 
Return:			// �洢��nbrs�еĽ��ڸ�����-1��ʾʧ��
Others:			// ����˵��
*****************************************************************************/
extern int kdtree_bbf_knn( struct kd_node* kd_root, struct feature* feat,
						  int k, struct feature*** nbrs, int max_nn_chks );

/*****************************************************************************
Function:		// kdtree_bbf_knn
Description:	// BBF�㷨Ѱ��k����������
Calls:			// ��
Called By:		// matchs.main
Table Accessed: // ��
Table Updated:	// ��
Input:			// @kd_root		k-d�������
				// @feat		Ŀ������������������
				// @k			��������
				// @nbrs		k������������
				// @max_nn_chks	�����������
Output:			// 
Return:			// �洢��nbrs�еĽ��ڸ�����-1��ʾʧ��
Others:			// ����˵��
*****************************************************************************/
extern int kdtree_bbf_spatial_knn( struct kd_node* kd_root,
								struct feature* feat, int k,
								struct feature*** nbrs, int max_nn_chks,
								CvRect rect, int model );

/*****************************************************************************
Function:		// kdtree_release
Description:	// �ͷſռ�
Calls:			// ��
Called By:		// matchs.main 
Table Accessed: // ��
Table Updated:	// ��
Input:			// @kd_root		k-d�������
Output:			// 
Return:			// ��
Others:			// ����˵��
*****************************************************************************/
extern void kdtree_release( struct kd_node* kd_root );

#endif 