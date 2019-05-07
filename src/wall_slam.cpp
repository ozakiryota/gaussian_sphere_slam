#include <ros/ros.h>
#include <geometry_msgs/Quaternion.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64MultiArray.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <pcl_conversions/pcl_conversions.h>
#include <tf/tf.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
// #include <Eigen/Core>
// #include <Eigen/LU>

class WallSLAM{
	private:
		ros::NodeHandle nh;
		/*subscribe*/
		ros::Subscriber sub_inipose;
		ros::Subscriber sub_bias;
		ros::Subscriber sub_imu;
		ros::Subscriber sub_odom;
		/*publish*/
		ros::Publisher pub_pose;
		/*const*/
		const int num_state = 3;
		const int size_robot_state = 6;	//X, Y, Z, R, P, Y (Global)
		const int size_wall_state = 3;	//x, y, z (Local)
		/*objects*/
		tf::Quaternion q_pose;
		tf::Quaternion q_pose_last_at_slamcallback;
		Eigen::MatrixXd X;
		Eigen::MatrixXd P;
		sensor_msgs::Imu bias;
		tf::Quaternion q_slam_now;
		tf::Quaternion q_slam_last;
		/*flags*/
		bool inipose_is_available = false;
		bool bias_is_available = false;
		bool first_callback_imu = true;
		bool first_callback_odom = true;
		/*counter*/
		int count_rpy_walls = 0;
		int count_slam = 0;
		/*time*/
		ros::Time time_imu_now;
		ros::Time time_imu_last;
		ros::Time time_odom_now;
		ros::Time time_odom_last;
	public:
		WallSLAM();
		void CallbackInipose(const geometry_msgs::QuaternionConstPtr& msg);
		void CallbackBias(const sensor_msgs::ImuConstPtr& msg);
		void CallbackIMU(const sensor_msgs::ImuConstPtr& msg);
		void PredictionIMU(sensor_msgs::Imu imu, double dt);
		void CallbackOdom(const nav_msgs::OdometryConstPtr& msg);
		void PredictionOdom(nav_msgs::Odometry odom, double dt);
		float PiToPi(double angle);
		void Publication();
};

WallSLAM::WallSLAM()
{
	sub_inipose = nh.subscribe("/initial_pose", 1, &WallSLAM::CallbackInipose, this);
	sub_bias = nh.subscribe("/imu_bias", 1, &WallSLAM::CallbackBias, this);
	sub_imu = nh.subscribe("/imu/data", 1, &WallSLAM::CallbackIMU, this);
	sub_odom = nh.subscribe("/odom", 1, &WallSLAM::CallbackOdom, this);
	pub_pose = nh.advertise<geometry_msgs::PoseStamped>("/pose_wall_slam", 1);
	q_pose = tf::Quaternion(0.0, 0.0, 0.0, 1.0);
	X = Eigen::MatrixXd::Constant(num_state, 1, 0.0);
	P = 1.0e-10*Eigen::MatrixXd::Identity(num_state, num_state);
}

void WallSLAM::CallbackInipose(const geometry_msgs::QuaternionConstPtr& msg)
{
	if(!inipose_is_available){
		quaternionMsgToTF(*msg, q_pose);
		q_pose.normalize();
		q_pose_last_at_slamcallback = q_pose;
		tf::Matrix3x3(q_pose).getRPY(X(0, 0), X(1, 0), X(2, 0));
		inipose_is_available = true;
		std::cout << "inipose_is_available = " << inipose_is_available << std::endl;
		std::cout << "initial pose = " << std::endl << X << std::endl;
	}
}

void WallSLAM::CallbackBias(const sensor_msgs::ImuConstPtr& msg)
{
	if(!bias_is_available){
		bias = *msg;
		bias_is_available = true;
	}
}

void WallSLAM::CallbackIMU(const sensor_msgs::ImuConstPtr& msg)
{
	time_imu_now = ros::Time::now();
	double dt;
	try{
		dt = (time_imu_now - time_imu_last).toSec();
	}
	catch(std::runtime_error& ex) {
		ROS_ERROR("Exception: [%s]", ex.what());
	}
	time_imu_last = time_imu_now;
	if(first_callback_imu)	dt = 0.0;
	else if(inipose_is_available)	PredictionIMU(*msg, dt);
	
	Publication();

	first_callback_imu = false;
}

void WallSLAM::PredictionIMU(sensor_msgs::Imu imu, double dt)
{
	double x = X(0, 0);
	double y = X(1, 0);
	double z = X(2, 0);
	double r_ = X(3, 0);
	double p_ = X(4, 0);
	double y_ = X(5, 0);

	double delta_r = imu.angular_velocity.x*dt;
	double delta_p = imu.angular_velocity.y*dt;
	double delta_y = imu.angular_velocity.z*dt;
	if(bias_is_available){
		delta_r -= bias.angular_velocity.x*dt;
		delta_p -= bias.angular_velocity.y*dt;
		delta_y -= bias.angular_velocity.z*dt;
	}
	// delta_r = PiToPi(delta_r);
	// delta_p = PiToPi(delta_p);
	// delta_y = PiToPi(delta_y);
	
	int num_wall = (X.rows() - size_robot_state)/size_wall_state;

	/*F*/
	Eigen::MatrixXd F(X.rows(), 1);
	F.block(0, 0, size_robot_state, size_robot_state) <<
		x,
		y,
		z,
		r_ + (delta_r + sin(r_)*tan(p_)*delta_p + cos(r_)*tan(p_)*delta_y),
		p_ + (cos(r_)*delta_p - sin(r_)*delta_y),
		y_ + (sin(r_)/cos(p_)*delta_p + cos(r_)/cos(p_)*delta_y);

	Eigen::Matrix3d Rot;
	Rot <<	cos(delta_p)*cos(delta_y),	cos(delta_p)*sin(delta_y),	-sin(delta_p),
			sin(delta_r)*sin(delta_p)*cos(delta_y) - cos(delta_r)*sin(delta_y),	sin(delta_r)*sin(delta_p)*sin(delta_y) + cos(delta_r)*cos(delta_y),	sin(delta_r)*cos(delta_p),
			cos(delta_r)*sin(delta_p)*cos(delta_y) + sin(delta_r)*sin(delta_y),	cos(delta_r)*sin(delta_p)*sin(delta_y) - sin(delta_r)*cos(delta_y),	cos(delta_r)*cos(delta_p);
	for(int i=0;i<num_wall;i++)	F.block(size_robot_state + i*size_wall_state, 0, size_wall_state, 1) = Rot*X.block(size_robot_state + i*size_wall_state, 0, size_wall_state, 1);

	/*jF*/
	Eigen::MatrixXd jF(X.rows(), X.rows());
	/*xyz*/
	jF.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
	jF.block(0, 3, 3, 3) = Eigen::Matrix3d::Zero();
	jF.block(0, size_robot_state, 3, num_wall*size_wall_state) = Eigen::MatrixXd::Zero(3, num_wall*size_wall_state);
	/*rpy*/
	jF.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
	jF(3, 3) = 1 + (cos(r_)*tan(p_)*delta_p - sin(r_)*tan(p_)*delta_y);
	jF(3, 4) = (sin(r_)/cos(p_)/cos(p_)*delta_p + cos(r_)/cos(p_)/cos(p_)*delta_y);
	jF(3, 5) = 0;
	jF(4, 3) = (-sin(r_)*delta_p - cos(r_)*delta_y);
	jF(4, 4) = 1;
	jF(4, 5) = 0;
	jF(5, 3) = (cos(r_)/cos(p_)*delta_p - sin(r_)/cos(p_)*delta_y);
	jF(5, 4) = (sin(r_)*sin(p_)/cos(p_)/cos(p_)*delta_p + cos(r_)*sin(p_)/cos(p_)/cos(p_)*delta_y);
	jF(5, 5) = 1;
	jF.block(3, size_robot_state, 3, num_wall*size_wall_state) = Eigen::MatrixXd::Zero(3, num_wall*size_wall_state);
	/*wall-xyz*/
	jF.block(size_robot_state, 0, num_wall*size_wall_state, 3) = Eigen::MatrixXd::Zero(num_wall*size_wall_state, 3);
	jF.block(size_robot_state, 3, num_wall*size_wall_state, 3) = Eigen::MatrixXd::Zero(num_wall*size_wall_state, 3);
	for(int i=0;i<num_wall;i++){
		jF.block(size_robot_state + i*size_wall_state, size_robot_state, size_wall_state, i*size_wall_state) = Eigen::MatrixXd::Zero(size_wall_state, i*size_wall_state);
		jF.block(size_robot_state + i*size_wall_state, size_robot_state + i*size_wall_state, size_wall_state, size_wall_state) = Rot;
		jF.block(size_robot_state + i*size_wall_state, size_robot_state + (i+1)*size_wall_state, size_wall_state, i*size_wall_state) = Eigen::MatrixXd::Zero(size_wall_state, i*size_wall_state);
	}
	
	const double sigma = 1.0e-1;
	Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(num_state, num_state);
	// Eigen::MatrixXd Q(num_state, num_state);
	// Q <<	1.0e-1,	0, 0,
	//  		0,	1.0e-1,	0,
	// 		0,	0,	5.0e+5;
	
	X = F;
	for(int i=0;i<3;i++){
		if(X(i, 0)>M_PI)	X(i, 0) -= 2.0*M_PI;
		if(X(i, 0)<-M_PI)	X(i, 0) += 2.0*M_PI;
	}
	q_pose = tf::createQuaternionFromRPY(X(0, 0), X(1, 0), X(2, 0));

	// tf::Quaternion q_relative_rotation = tf::createQuaternionFromRPY(delta_r, delta_p, delta_y);
	// q_pose = q_pose*q_relative_rotation;
	// q_pose.normalize();
	// tf::Matrix3x3(q_pose).getRPY(X(0, 0), X(1, 0), X(2, 0));

	P = jF*P*jF.transpose() + Q;
}

void WallSLAM::CallbackOdom(const nav_msgs::OdometryConstPtr& msg)
{
	time_odom_now = ros::Time::now();
	double dt;
	try{
		dt = (time_odom_now - time_odom_last).toSec();
	}
	catch(std::runtime_error& ex) {
		ROS_ERROR("Exception: [%s]", ex.what());
	}
	time_odom_last = time_odom_now;
	if(first_callback_odom)	dt = 0.0;
	else if(inipose_is_available)	PredictionOdom(*msg, dt);
	
	Publication();

	first_callback_odom = false;
}

void WallSLAM::PredictionOdom(nav_msgs::Odometry odom, double dt)
{
	Eigen::Vector3d delta_local = {odom.twist.twist.linear.x*dt, 0, 0};
}

void WallSLAM::Publication(void)
{
	geometry_msgs::PoseStamped pose_out;
	q_pose.normalize();
	quaternionTFToMsg(q_pose, pose_out.pose.orientation);
	pose_out.header.frame_id = "/odom";
	pose_out.header.stamp = ros::Time::now();
	pub_pose.publish(pose_out);
}

float WallSLAM::PiToPi(double angle)
{
	return fmod(angle + M_PI, 2*M_PI) - M_PI;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "wall_slam");
	std::cout << "E.K.F. POSE" << std::endl;
	
	WallSLAM wall_slam;
	ros::spin();
}
