#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/tf.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/passthrough.h>
/* #include <tf/transform_broadcaster.h> */

class GICPOptimization{
	private:
		ros::NodeHandle nh;
		ros::NodeHandle nhPrivate;
		/*subscribe*/
		ros::Subscriber sub_pc;
		ros::Subscriber sub_pose;
		/*publish*/
		ros::Publisher pub_pc;
		// tf::TransformBroadcaster tf_broadcaster;
		/*viewer*/
		pcl::visualization::PCLVisualizer viewer{"gicp"};
		/*cloud*/
		pcl::PointCloud<pcl::PointNormal>::Ptr map {new pcl::PointCloud<pcl::PointNormal>};
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud {new pcl::PointCloud<pcl::PointNormal>};
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_transformed {new pcl::PointCloud<pcl::PointNormal>};
		pcl::PointCloud<pcl::PointNormal>::Ptr map_filtered {new pcl::PointCloud<pcl::PointNormal>};
		/*flags*/
		bool first_callback_pose = true;
		/*time*/
		// ros::Time time_start;
		/*parameters*/
		double pc_range;
		int iterations;
		double correspond_dist;
		double trans_epsilon;
		double fit_epsilon;
	public:
		GICPOptimization();
		void CallbackPC(const sensor_msgs::PointCloud2ConstPtr& msg);
		void CallbackPose(const geometry_msgs::PoseStampedConstPtr& msg);
		void PCFilter(pcl::PointCloud<pcl::PointNormal>::Ptr pc, pcl::PointCloud<pcl::PointNormal>::Ptr pc_out, std::vector<double> range);
		bool Transformation(geometry_msgs::PoseStamped pose);
		void Visualization(void);
		void Publication(void);
		Eigen::Quaternionf QuatMsgToEigen(geometry_msgs::Quaternion q_msg);
		geometry_msgs::Quaternion QuatEigenToMsg(Eigen::Quaternionf q_eigen);
};

GICPOptimization::GICPOptimization()
	:nhPrivate("~")
{
	sub_pc = nh.subscribe("/normals", 1, &GICPOptimization::CallbackPC, this);
	sub_pose = nh.subscribe("/wall_ekf_slam/pose", 1, &GICPOptimization::CallbackPose, this);
	pub_pc = nh.advertise<sensor_msgs::PointCloud2>("/map", 1);
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addCoordinateSystem(0.5, "axis");
	viewer.setCameraPosition(0.0, 0.0, 80.0, 0.0, 0.0, 0.0);

	nhPrivate.param("pc_range", pc_range, {100.0});
	nhPrivate.param("iterations", iterations, 100);
	nhPrivate.param("correspond_dist", correspond_dist, {0.1});
	nhPrivate.param("trans_epsilon", trans_epsilon, {1e-8});
	nhPrivate.param("fit_epsilon", fit_epsilon, {1.0e-8});
	std::cout << "pc_range = " << pc_range << std::endl;
	std::cout << "iterations = " << iterations << std::endl;
	std::cout << "correspond_dist = " << correspond_dist << std::endl;
	std::cout << "trans_epsilon = " << trans_epsilon << std::endl;
	std::cout << "fit_epsilon = " << fit_epsilon << std::endl;
}

void GICPOptimization::CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	std::cout << "CALLBACK PC" << std::endl;
	
	pcl::fromROSMsg(*msg, *cloud);
}

void GICPOptimization::CallbackPose(const geometry_msgs::PoseStampedConstPtr& msg)
{
	std::cout << "CALLBACK POSE" << std::endl;

	bool has_converged;
	if(first_callback_pose || map->points.empty()){
		*map = *cloud;
	}
	else{
		has_converged = Transformation(*msg);
	}

	*map += *cloud_transformed;
	// map->header = cloud->header;
	map->header.stamp = cloud->header.stamp;
	map->header.frame_id = msg->header.frame_id;

	first_callback_pose = false;

	Publication();
	Visualization();

	std::cout << "cloud->points.size() = " << cloud->points.size() << std::endl;
	std::cout << "map->points.size() = " << map->points.size() << std::endl;
	if(!has_converged)	exit(1);
}

void GICPOptimization::PCFilter(pcl::PointCloud<pcl::PointNormal>::Ptr pc_in, pcl::PointCloud<pcl::PointNormal>::Ptr pc_out, std::vector<double> range)
{
	pcl::PassThrough<pcl::PointNormal> pass;
	pass.setInputCloud(pc_in);
	pass.setFilterFieldName("x");
	pass.setFilterLimits(range[0], range[1]);
	pass.filter(*pc_out);
	pass.setInputCloud(pc_out);
	pass.setFilterFieldName("y");
	pass.setFilterLimits(range[2], range[3]);
	pass.filter(*pc_out);
}

bool GICPOptimization::Transformation(geometry_msgs::PoseStamped pose)
{
	
	/*filter pc*/
	std::vector<double> range_map{
		pose.pose.position.x - pc_range, 
		pose.pose.position.x + pc_range, 
		pose.pose.position.y - pc_range, 
		pose.pose.position.y + pc_range
	};
	PCFilter(cloud, cloud, std::vector<double> {-pc_range, pc_range, -pc_range, pc_range});
	PCFilter(map, map_filtered, range_map);

	/*set parameters*/
	pcl::GeneralizedIterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> gicp;
	gicp.setMaximumIterations(iterations);
	gicp.setMaxCorrespondenceDistance(correspond_dist);
	gicp.setTransformationEpsilon(trans_epsilon);
	gicp.setEuclideanFitnessEpsilon(fit_epsilon);
	gicp.setInputSource(cloud);
	gicp.setInputTarget(map_filtered);

	/*initial guess*/
	Eigen::Translation3f init_translation(
		pose.pose.position.x,
		pose.pose.position.y,
		pose.pose.position.z
	);
	Eigen::AngleAxisf init_rotation(
		QuatMsgToEigen(pose.pose.orientation)
	);
	Eigen::Matrix4f init_guess = (init_translation*init_rotation).matrix();

	/*align*/
	gicp.align(*cloud_transformed, init_guess);
	// gicp.computeTransformation(*cloud, init_guess);
	// gicp.align(*cloud_transformed);

	/*print*/
	std::cout << "Iterative Closest Point has converged:" << (bool)gicp.hasConverged() << std::endl;
	std::cout << "score: " << gicp.getFitnessScore() << std::endl;
	std::cout << "gicp.getFinalTransformation()" << std::endl << gicp.getFinalTransformation() << std::endl;
	std::cout << "init_guess" << std::endl << init_guess << std::endl;

	/*input*/
	Eigen::Matrix4f m_transformation = gicp.getFinalTransformation();
	Eigen::Matrix3f m_rot = m_transformation.block(0, 0, 3, 3);
	Eigen::Quaternionf q_rot(m_rot);
	q_rot.normalize();

	return gicp.hasConverged();
}

Eigen::Quaternionf GICPOptimization::QuatMsgToEigen(geometry_msgs::Quaternion q_msg)
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

geometry_msgs::Quaternion GICPOptimization::QuatEigenToMsg(Eigen::Quaternionf q_eigen)
{
	geometry_msgs::Quaternion q_msg;
	q_msg.x = (double)q_eigen.x();
	q_msg.y = (double)q_eigen.y();
	q_msg.z = (double)q_eigen.z();
	q_msg.w = (double)q_eigen.w();
	return q_msg;
}

void GICPOptimization::Visualization(void)
{
	viewer.removeAllPointClouds();

	viewer.addPointCloud<pcl::PointNormal>(cloud, "cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "cloud");

	viewer.addPointCloud<pcl::PointNormal>(map, "map");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "map");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.5, "map");

	viewer.addPointCloud<pcl::PointNormal>(map_filtered, "map_filtered");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "map_filtered");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "map_filtered");

	viewer.addPointCloud<pcl::PointNormal>(cloud_transformed, "cloud_transformed");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "cloud_transformed");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "cloud_transformed");

	viewer.spinOnce();
}

void GICPOptimization::Publication(void)
{
	/*publish*/
	sensor_msgs::PointCloud2 pc_pub;
	pcl::toROSMsg(*map, pc_pub);
	pub_pc.publish(pc_pub);
		
	/*tf broadcast*/
    /* geometry_msgs::TransformStamped transform; */
	/* transform.header.stamp = ros::Time::now(); */
	/* transform.header.frame_id = "/odom"; */
	/* transform.child_frame_id = "/icp_odometry"; */
	/* transform.transform.translation.x = odom_icp.pose.pose.position.x; */
	/* transform.transform.translation.y = odom_icp.pose.pose.position.y; */
	/* transform.transform.translation.z = odom_icp.pose.pose.position.z; */
	/* transform.transform.rotation = odom_icp.pose.pose.orientation; */
	/* tf_broadcaster.sendTransform(transform); */
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "gicp");
	
	GICPOptimization gicp_optimization;

	ros::spin();
}
