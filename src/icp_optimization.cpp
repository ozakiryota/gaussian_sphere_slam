#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/tf.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>
#include <tf/transform_broadcaster.h>

class ICP{
	private:
		ros::NodeHandle nh;
		ros::NodeHandle nhPrivate;
		/*subscribe*/
		ros::Subscriber sub_pc;
		ros::Subscriber sub_odom;
		/*publish*/
		ros::Publisher pub_odom;
		tf::TransformBroadcaster tf_broadcaster;
		/*viewer*/
		pcl::visualization::PCLVisualizer viewer{"icp"};
		/*cloud*/
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_now {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_last {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed {new pcl::PointCloud<pcl::PointXYZ>};
		/*odom*/
		nav_msgs::Odometry odom_icp;
		nav_msgs::Odometry odom_now;
		nav_msgs::Odometry odom_last;
		/*flags*/
		bool first_callback_odom = true;
		/*time*/
		ros::Time time_start;
		/*parameters*/
		double pc_range;
		double iterations;
		double trans_epsilon;
		double fit_epsilon;
	public:
		ICP();
		void InitializeOdom(nav_msgs::Odometry& odom);
		void CallbackPC(const sensor_msgs::PointCloud2ConstPtr& msg);
		void CallbackOdom(const nav_msgs::OdometryConstPtr& msg);
		void Compute(void);
		void Transformation(void);
		Eigen::Quaternionf QuatMsgToEigen(geometry_msgs::Quaternion q_msg);
		geometry_msgs::Quaternion QuatEigenToMsg(Eigen::Quaternionf q_eigen);
		void Visualization(void);
		void Publication(void);
};

ICP::ICP()
	:nhPrivate("~")
{
	sub_pc = nh.subscribe("/velodyne_points", 1, &ICP::CallbackPC, this);
	sub_pose = nh.subscribe("/", 1, &ICP::CallbackOdom, this);
	pub_odom = nh.advertise<nav_msgs::Odometry>("/icp_odometry", 1);
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addCoordinateSystem(0.5, "axis");
	viewer.setCameraPosition(0.0, 0.0, 80.0, 0.0, 0.0, 0.0);
	InitializeOdom(odom_icp);
	InitializeOdom(odom_now);
	InitializeOdom(odom_last);

	nhPrivate.param("pc_range", pc_range, {100.0});
	nhPrivate.param("iterations", iterations, {100});
	nhPrivate.param("trans_epsilon", trans_epsilon, {1.0e-8});
	nhPrivate.param("fit_epsilon", fit_epsilon, {1.0e-8});
	std::cout << "pc_range = " << pc_range << std::endl;
	std::cout << "iterations = " << iterations << std::endl;
	std::cout << "trans_epsilon = " << trans_epsilon << std::endl;
	std::cout << "fit_epsilon = " << fit_epsilon << std::endl;
}

void ICP::InitializeOdom(nav_msgs::Odometry& odom)
{
	odom.header.frame_id = "/odom";
	odom.child_frame_id = "/icp_odometry";
	odom.pose.pose.position.x = 0.0;
	odom.pose.pose.position.y = 0.0;
	odom.pose.pose.position.z = 0.0;
	odom.pose.pose.orientation.x = 0.0;
	odom.pose.pose.orientation.y = 0.0;
	odom.pose.pose.orientation.z = 0.0;
	odom.pose.pose.orientation.w = 1.0;
}

void ICP::CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	std::cout << "CALLBACK PC" << std::endl;
	
	if(cloud_last->points.empty()){
		pcl::fromROSMsg(*msg, *cloud_now);
		*cloud_last = *cloud_now;
	}
	else{
		*cloud_last = *cloud_now;
		pcl::fromROSMsg(*msg, *cloud_now);
	}
	std::cout << "msg->header.stamp.toSec()-time_start.toSec() = " << msg->header.stamp.toSec()-time_start.toSec() << std::endl;

	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(cloud_now);
	pass.setFilterFieldName("x");
	pass.setFilterLimits(-pc_range, pc_range);
	pass.filter(*cloud_now);
	pass.setInputCloud(cloud_now);
	pass.setFilterFieldName("y");
	pass.setFilterLimits(-pc_range, pc_range);
	pass.filter(*cloud_now);
}

void ICP::CallbackOdom(const nav_msgs::OdometryConstPtr& msg)
{
	std::cout << "CALLBACK ODOM" << std::endl;

	if(first_callback_odom){
		odom_now = *msg;
		odom_last = odom_now;
		time_start = ros::Time::now();
	}
	else{
		odom_last = odom_now;
		odom_now = *msg;
	}
	std::cout << "msg->header.stamp.toSec()-time_start.toSec() = " << msg->header.stamp.toSec()-time_start.toSec() << std::endl;

	first_callback_odom = false;
}

void ICP::Compute(void)
{
	std::cout << "COMPUTE" << std::endl;
	Transformation();
	Visualization();
	Publication();
}

void ICP::Transformation(void)
{
	/*set parameters*/
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	// icp.setMaxCorrespondenceDistance(0.5);
	// icp.setMaximumIterations(iterations);
	// icp.setTransformationEpsilon(trans_epsilon);
	// icp.setEuclideanFitnessEpsilon(fit_epsilon);
	icp.setInputSource(cloud_now);
	icp.setInputTarget(cloud_last);

	/*initial guess*/
	Eigen::Quaternionf q_pose_now = QuatMsgToEigen(odom_now.pose.pose.orientation);
	Eigen::Quaternionf q_pose_last = QuatMsgToEigen(odom_last.pose.pose.orientation);
	Eigen::Quaternionf q_relative_rotation = q_pose_last.inverse()*q_pose_now;
	q_relative_rotation.normalize();
	Eigen::Quaternionf q_global_move(
		0.0,
		odom_now.pose.pose.position.x - odom_last.pose.pose.position.x,
		odom_now.pose.pose.position.y - odom_last.pose.pose.position.y,
		odom_now.pose.pose.position.z - odom_last.pose.pose.position.z);
	Eigen::Quaternionf q_local_move = q_pose_last.inverse()*q_global_move*q_pose_last;
	Eigen::Translation3f init_translation(q_local_move.x(), q_local_move.y(), q_local_move.z());
	Eigen::AngleAxisf init_rotation(q_relative_rotation);
	Eigen::Matrix4f init_guess = (init_translation*init_rotation).matrix();

	/*align*/
	icp.align(*cloud_transformed, init_guess);
	// icp.align(*cloud_transformed);

	/*print*/
	std::cout << "Iterative Closest Point has converged:" << (bool)icp.hasConverged() 
		<< std::endl << " score: " << icp.getFitnessScore () << std::endl;
	std::cout << "icp.getFinalTransformation()" << std::endl << icp.getFinalTransformation() << std::endl;
	std::cout << "init_guess" << std::endl << init_guess << std::endl;

	/*convert to /odom*/
	Eigen::Matrix4f m_transformation = icp.getFinalTransformation();
	Eigen::Matrix3f m_rot = m_transformation.block(0, 0, 3, 3);
	Eigen::Quaternionf q_rot(m_rot);
	q_rot.normalize();
	Eigen::Quaternionf q_pose = QuatMsgToEigen(odom_icp.pose.pose.orientation);
	odom_icp.pose.pose.orientation = QuatEigenToMsg((q_pose*q_rot).normalized());

	Eigen::Quaternionf q_trans(
		0.0,
		m_transformation(0, 3),
		m_transformation(1, 3),
		m_transformation(2, 3)
	);
	q_trans = q_pose*q_trans*q_pose.inverse();
	std::cout << q_trans.x() << std::endl;
	std::cout << q_trans.y() << std::endl;
	std::cout << q_trans.z() << std::endl;
	std::cout << q_trans.w() << std::endl;
	odom_icp.pose.pose.position.x += q_trans.x();
	odom_icp.pose.pose.position.y += q_trans.y();
	odom_icp.pose.pose.position.z += q_trans.z();
}

Eigen::Quaternionf ICP::QuatMsgToEigen(geometry_msgs::Quaternion q_msg)
{
	Eigen::Quaternionf q_eigen(
		(float)q_msg.w,
		(float)q_msg.x,
		(float)q_msg.y,
		(float)q_msg.z
	);
	q_eigen.normalize();
	return q_eigen;
}

geometry_msgs::Quaternion ICP::QuatEigenToMsg(Eigen::Quaternionf q_eigen)
{
	geometry_msgs::Quaternion q_msg;
	q_msg.x = (double)q_eigen.x();
	q_msg.y = (double)q_eigen.y();
	q_msg.z = (double)q_eigen.z();
	q_msg.w = (double)q_eigen.w();
	return q_msg;
}

void ICP::Visualization(void)
{
	viewer.removeAllPointClouds();

	viewer.addPointCloud<pcl::PointXYZ>(cloud_now, "cloud_now");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud_now");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "cloud_now");

	viewer.addPointCloud<pcl::PointXYZ>(cloud_last, "cloud_last");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "cloud_last");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "cloud_last");

	viewer.addPointCloud<pcl::PointXYZ>(cloud_transformed, "cloud_transformed");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "cloud_transformed");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "cloud_transformed");

	viewer.spinOnce();
}

void ICP::Publication(void)
{
	/*publish*/
	// odom_icp.header.stamp = ros::Time::now();
	odom_icp.header.stamp = odom_now.header.stamp;
	pub_odom.publish(odom_icp);
	/*tf broadcast*/
    geometry_msgs::TransformStamped transform;
	transform.header.stamp = ros::Time::now();
	transform.header.frame_id = "/odom";
	transform.child_frame_id = "/icp_odometry";
	transform.transform.translation.x = odom_icp.pose.pose.position.x;
	transform.transform.translation.y = odom_icp.pose.pose.position.y;
	transform.transform.translation.z = odom_icp.pose.pose.position.z;
	transform.transform.rotation = odom_icp.pose.pose.orientation;
	tf_broadcaster.sendTransform(transform);
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "icp");
	
	ICP icp;

	// ros::spin();

	ros::Rate loop_rate(20);
	while(ros::ok()){
		std::cout << "----------" << std::endl;
		ros::spinOnce();
		double time_start = ros::Time::now().toSec();
		icp.Compute();
		std::cout << "computation time: " << ros::Time::now().toSec() - time_start  << "[s]" << std::endl;
		loop_rate.sleep();
	}
}
