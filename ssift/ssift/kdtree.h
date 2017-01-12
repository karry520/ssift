/*****************************************************************************
Copyright:		2016-2017, Likaiyun
File name:		kdtree.h
Description:	kd-tree-bbf算法
Author:			李开运
Version:		0.1
Date:			完成日期
History:修改历史记录列表， 每条修改记录应包括修改日期、修改者及修改内容简述。
*****************************************************************************/

#ifndef KDTREE_H
#define KDTREE_H

#include "cxcore.h"
/************************************结构体**********************************/

//kd_tree重要数据结构
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
/************************************函数申明*********************************/

/*****************************************************************************
Function:		// kdtree_build
Description:	// 根据特征点集合建立k-d树
Calls:			// 无
Called By:		// matchs.main
Table Accessed: // 无
Table Updated:	// 无
Input:			// @featrues	特征点集
				// @n			特征点个数
Output:			// 
Return:			// 返回初始化后kd_tree树根节点（kd_node）
Others:			// 其它说明
*****************************************************************************/
extern struct kd_node* kdtree_build( struct feature* features, int n );

/*****************************************************************************
Function:		// kdtree_bbf_knn
Description:	// BBF算法寻找k近邻特征点
Calls:			// 无
Called By:		// matchs.main
Table Accessed: // 无
Table Updated:	// 无
Input:			// @kd_root		k-d树根结点
				// @feat		目标特征点描述子向量
				// @k			近邻数量
				// @nbrs		k个近邻特征点
				// @max_nn_chks	最大搜索次数
Output:			// 
Return:			// 存储在nbrs中的近邻个数，-1表示失败
Others:			// 其它说明
*****************************************************************************/
extern int kdtree_bbf_knn( struct kd_node* kd_root, struct feature* feat,
						  int k, struct feature*** nbrs, int max_nn_chks );

/*****************************************************************************
Function:		// kdtree_bbf_knn
Description:	// BBF算法寻找k近邻特征点
Calls:			// 无
Called By:		// matchs.main
Table Accessed: // 无
Table Updated:	// 无
Input:			// @kd_root		k-d树根结点
				// @feat		目标特征点描述子向量
				// @k			近邻数量
				// @nbrs		k个近邻特征点
				// @max_nn_chks	最大搜索次数
Output:			// 
Return:			// 存储在nbrs中的近邻个数，-1表示失败
Others:			// 其它说明
*****************************************************************************/
extern int kdtree_bbf_spatial_knn( struct kd_node* kd_root,
								struct feature* feat, int k,
								struct feature*** nbrs, int max_nn_chks,
								CvRect rect, int model );

/*****************************************************************************
Function:		// kdtree_release
Description:	// 释放空间
Calls:			// 无
Called By:		// matchs.main 
Table Accessed: // 无
Table Updated:	// 无
Input:			// @kd_root		k-d树根结点
Output:			// 
Return:			// 无
Others:			// 其它说明
*****************************************************************************/
extern void kdtree_release( struct kd_node* kd_root );

#endif 