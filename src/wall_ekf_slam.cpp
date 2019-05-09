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

class WallEKFSLAM{
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
		WallEKFSLAM();
		void CallbackInipose(const geometry_msgs::QuaternionConstPtr& msg);
		void CallbackBias(const sensor_msgs::ImuConstPtr& msg);
		void CallbackIMU(const sensor_msgs::ImuConstPtr& msg);
		void PredictionIMU(sensor_msgs::Imu imu, double dt);
		void CallbackOdom(const nav_msgs::OdometryConstPtr& msg);
		void PredictionOdom(nav_msgs::Odometry odom, double dt);
		void Publication();
		geometry_msgs::PoseStamped StateVectorToPoseStamped(void);
		float PiToPi(double angle);
};

WallEKFSLAM::WallEKFSLAM()
{
	sub_inipose = nh.subscribe("/initial_pose", 1, &WallEKFSLAM::CallbackInipose, this);
	sub_bias = nh.subscribe("/imu_bias", 1, &WallEKFSLAM::CallbackBias, this);
	sub_imu = nh.subscribe("/imu/data", 1, &WallEKFSLAM::CallbackIMU, this);
	sub_odom = nh.subscribe("/tinypower/odom", 1, &WallEKFSLAM::CallbackOdom, this);
	pub_pose = nh.advertise<geometry_msgs::PoseStamped>("/pose_wall_ekf_slam", 1);
	q_pose = tf::Quaternion(0.0, 0.0, 0.0, 1.0);
	X = Eigen::MatrixXd::Constant(size_robot_state, 1, 0.0);
	P = 1.0e-10*Eigen::MatrixXd::Identity(size_robot_state, size_robot_state);
}

void WallEKFSLAM::CallbackInipose(const geometry_msgs::QuaternionConstPtr& msg)
{
	if(!inipose_is_available){
		quaternionMsgToTF(*msg, q_pose);
		q_pose.normalize();
		q_pose_last_at_slamcallback = q_pose;
		tf::Matrix3x3(q_pose).getRPY(X(3, 0), X(4, 0), X(5, 0));
		inipose_is_available = true;
		std::cout << "inipose_is_available = " << inipose_is_available << std::endl;
		std::cout << "initial pose = " << std::endl << X << std::endl;
	}
}

void WallEKFSLAM::CallbackBias(const sensor_msgs::ImuConstPtr& msg)
{
	if(!bias_is_available){
		bias = *msg;
		bias_is_available = true;
	}
}

void WallEKFSLAM::CallbackIMU(const sensor_msgs::ImuConstPtr& msg)
{
	std::cout << "Callback IMU" << std::endl;

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

void WallEKFSLAM::PredictionIMU(sensor_msgs::Imu imu, double dt)
{
	std::cout << "PredictionIMU" << std::endl;
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
	Eigen::Vector3d Drpy = {delta_r, delta_p, delta_y};
	
	int num_wall = (X.rows() - size_robot_state)/size_wall_state;

	Eigen::Matrix3d Rot_rpy;	//normal rotation
	Rot_rpy <<	1,	sin(r_)*tan(p_),	cos(r_)*tan(p_),
				0,	cos(r_),			-sin(r_),
				0,	sin(r_)/cos(p_),	cos(r_)/cos(p_);

	Eigen::Matrix3d Rot_xyz_inv;	//inverse rotation
	Rot_xyz_inv <<	cos(delta_p)*cos(delta_y),	cos(delta_p)*sin(delta_y),	-sin(delta_p),
					sin(delta_r)*sin(delta_p)*cos(delta_y) - cos(delta_r)*sin(delta_y),	sin(delta_r)*sin(delta_p)*sin(delta_y) + cos(delta_r)*cos(delta_y),	sin(delta_r)*cos(delta_p),
					cos(delta_r)*sin(delta_p)*cos(delta_y) + sin(delta_r)*sin(delta_y),	cos(delta_r)*sin(delta_p)*sin(delta_y) - sin(delta_r)*cos(delta_y),	cos(delta_r)*cos(delta_p);

	/*F*/
	Eigen::MatrixXd F(X.rows(), 1);
	/* F.block(0, 0, size_robot_state, 1) <<	x, */
	/* 										y, */
	/* 										z, */
	/* 										r_ + (delta_r + sin(r_)*tan(p_)*delta_p + cos(r_)*tan(p_)*delta_y), */
	/* 										p_ + (cos(r_)*delta_p - sin(r_)*delta_y), */
	/* 										y_ + (sin(r_)/cos(p_)*delta_p + cos(r_)/cos(p_)*delta_y); */
	F.block(0, 0, 3, 1) = X.block(0, 0, 3, 1);
	F.block(3, 0, 3, 1) = X.block(3, 0, 3, 1) + Rot_rpy*Drpy;
	for(int i=0;i<num_wall;i++)	F.block(size_robot_state + i*size_wall_state, 0, size_wall_state, 1) = Rot_xyz_inv*X.block(size_robot_state + i*size_wall_state, 0, size_wall_state, 1);

	/*jF*/
	Eigen::MatrixXd jF(X.rows(), X.rows());
	/*jF-xyz*/
	jF.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
	jF.block(0, 3, 3, 3) = Eigen::Matrix3d::Zero();
	jF.block(0, size_robot_state, 3, num_wall*size_wall_state) = Eigen::MatrixXd::Zero(3, num_wall*size_wall_state);
	/*jF-rpy*/
	jF.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
	jF(3, 3) = 1 + cos(r_)*tan(p_)*delta_p - sin(r_)*tan(p_)*delta_y;
	jF(3, 4) = sin(r_)/cos(p_)/cos(p_)*delta_p + cos(r_)/cos(p_)/cos(p_)*delta_y;
	jF(3, 5) = 0;
	jF(4, 3) = -sin(r_)*delta_p - cos(r_)*delta_y;
	jF(4, 4) = 1;
	jF(4, 5) = 0;
	jF(5, 3) = cos(r_)/cos(p_)*delta_p - sin(r_)/cos(p_)*delta_y;
	jF(5, 4) = sin(r_)*sin(p_)/cos(p_)/cos(p_)*delta_p + cos(r_)*sin(p_)/cos(p_)/cos(p_)*delta_y;
	jF(5, 5) = 1;
	jF.block(3, size_robot_state, 3, num_wall*size_wall_state) = Eigen::MatrixXd::Zero(3, num_wall*size_wall_state);
	/*jF-wall_xyz*/
	jF.block(size_robot_state, 0, num_wall*size_wall_state, 3) = Eigen::MatrixXd::Zero(num_wall*size_wall_state, 3);
	jF.block(size_robot_state, 3, num_wall*size_wall_state, 3) = Eigen::MatrixXd::Zero(num_wall*size_wall_state, 3);
	Eigen::MatrixXd jWall = Eigen::MatrixXd::Zero(num_wall*size_wall_state, num_wall*size_wall_state);
	for(int i=0;i<num_wall;i++)	jWall.block(i*size_wall_state, i*size_wall_state, size_wall_state, size_wall_state) = Rot_xyz_inv;
	jF.block(size_robot_state, size_robot_state, num_wall*size_wall_state, num_wall*size_wall_state) = jWall;
	
	/*Q*/
	const double sigma = 1.0e-1;
	Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(X.rows(), X.rows());
	
	/*Update*/
	X = F;
	P = jF*P*jF.transpose() + Q;

	std::cout << "X =" << std::endl << X << std::endl;
	std::cout << "P =" << std::endl << P << std::endl;
	std::cout << "jF =" << std::endl << jF << std::endl;
}

void WallEKFSLAM::CallbackOdom(const nav_msgs::OdometryConstPtr& msg)
{
	std::cout << "Callback Odom" << std::endl;

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

void WallEKFSLAM::PredictionOdom(nav_msgs::Odometry odom, double dt)
{
	std::cout << "Prediction Odom" << std::endl;

	double x = X(0, 0);
	double y = X(1, 0);
	double z = X(2, 0);
	double r_ = X(3, 0);
	double p_ = X(4, 0);
	double y_ = X(5, 0);
	Eigen::Vector3d Dxyz = {odom.twist.twist.linear.x*dt, 0, 0};

	int num_wall = (X.rows() - size_robot_state)/size_wall_state;

	Eigen::Matrix3d Rot_xyz;	//normal rotation
	Rot_xyz <<	cos(p_)*cos(y_),	sin(r_)*sin(p_)*cos(y_) - cos(r_)*sin(y_),	cos(r_)*sin(p_)*cos(y_) + sin(r_)*sin(y_),
				cos(p_)*sin(y_),	sin(r_)*sin(p_)*sin(y_) + cos(r_)*cos(y_),	cos(r_)*sin(p_)*sin(y_) - sin(r_)*cos(y_),
				-sin(p_),			sin(r_)*cos(p_),							cos(r_)*cos(p_);

	/*F*/
	Eigen::MatrixXd F(X.rows(), 1);
	F.block(0, 0, 3, 1) = X.block(0, 0, 3, 1) + Rot_xyz*Dxyz;
	F.block(3, 0, 3, 1) = X.block(3, 0, 3, 1);
	for(int i=0;i<num_wall;i++){
		Eigen::Vector3d wall_xyz = X.block(size_robot_state + i*size_wall_state, 0, 3, 1);
		F.block(size_robot_state + i*size_wall_state, 0, 3, 1) = wall_xyz - wall_xyz.dot(Dxyz)/wall_xyz.dot(wall_xyz)*wall_xyz;
	}

	/*jF*/
	Eigen::MatrixXd jF(X.rows(), X.rows());
	/*xyz*/
	jF.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
	jF(0, 3) = Dxyz(1)*(cos(r_)*sin(p_)*cos(y_) + sin(r_)*sin(y_)) + Dxyz(2)*(-sin(r_)*sin(p_)*cos(y_) + cos(r_)*sin(y_));
	jF(0, 4) = Dxyz(0)*(-sin(p_)*cos(y_)) + Dxyz(1)*(sin(r_)*cos(p_)*cos(y_)) + Dxyz(2)*(cos(r_)*cos(p_)*cos(y_));
	jF(0, 5) = Dxyz(0)*(-cos(p_)*sin(y_)) + Dxyz(1)*(-sin(r_)*sin(p_)*sin(y_) - cos(r_)*cos(y_)) + Dxyz(2)*(-cos(r_)*sin(p_)*sin(y_) + sin(r_)*cos(y_));
	jF(1, 3) = Dxyz(1)*(cos(r_)*sin(p_)*sin(y_) - sin(r_)*cos(y_)) + Dxyz(2)*(-sin(r_)*sin(p_)*sin(y_) - cos(r_)*cos(y_));
	jF(1, 4) = Dxyz(0)*(-sin(p_)*sin(y_)) + Dxyz(1)*(sin(r_)*cos(p_)*sin(y_)) + Dxyz(2)*(cos(r_)*cos(p_)*sin(y_));
	jF(1, 5) = Dxyz(0)*(cos(p_)*cos(y_)) + Dxyz(1)*(sin(r_)*sin(p_)*cos(y_) - cos(r_)*sin(y_)) + Dxyz(2)*(cos(r_)*sin(p_)*cos(y_) + sin(r_)*sin(y_));
	jF(2, 3) = Dxyz(1)*(cos(r_)*cos(p_)) + Dxyz(2)*(-sin(r_)*cos(p_)) ;
	jF(2, 4) = Dxyz(0)*(-cos(p_)) + Dxyz(1)*(-sin(r_)*sin(p_)) + Dxyz(2)*(-cos(r_)*sin(p_)) ;
	jF(2, 5) = 0;
	jF.block(0, size_robot_state, 3, num_wall*size_wall_state) = Eigen::MatrixXd::Zero(3, num_wall*size_wall_state);
	/*rpy*/
	jF.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
	jF.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity();
	jF.block(3, size_robot_state, 3, num_wall*size_wall_state) = Eigen::MatrixXd::Zero(3, num_wall*size_wall_state);
	/*wall-xyz*/
	jF.block(size_robot_state, 0, num_wall*size_wall_state, 3) = Eigen::MatrixXd::Zero(num_wall*size_wall_state, 3);
	jF.block(size_robot_state, 3, num_wall*size_wall_state, 3) = Eigen::MatrixXd::Zero(num_wall*size_wall_state, 3);
	for(int i=0;i<num_wall;i++){
		Eigen::Vector3d wall_xyz = X.block(size_robot_state + i*size_wall_state, 0, 3, 1);
		double d2 = wall_xyz(0)*wall_xyz(0) + wall_xyz(1)*wall_xyz(1) + wall_xyz(2)*wall_xyz(2);

		jF.block(size_robot_state + i*size_wall_state, size_robot_state, size_wall_state, i*size_wall_state) = Eigen::MatrixXd::Zero(size_wall_state, i*size_wall_state);
		jF(size_robot_state + i*size_wall_state, size_robot_state + i*size_wall_state) = 1 - (2*Dxyz(0)*wall_xyz(0)/d2 - 2*Dxyz(0)*wall_xyz(0)*wall_xyz(0)*wall_xyz(0)/(d2*d2));
		jF(size_robot_state + i*size_wall_state, size_robot_state + i*size_wall_state + 1) = 2*Dxyz(0)*wall_xyz(0)*wall_xyz(0)*wall_xyz(1)/(d2*d2);
		jF(size_robot_state + i*size_wall_state, size_robot_state + i*size_wall_state + 2) = 2*Dxyz(0)*wall_xyz(0)*wall_xyz(0)*wall_xyz(2)/(d2*d2);
		jF(size_robot_state + i*size_wall_state + 1, size_robot_state + i*size_wall_state) = -Dxyz(0)*wall_xyz(1)/d2 + 2*Dxyz(0)*wall_xyz(0)*wall_xyz(0)*wall_xyz(1)/(d2*d2);
		jF(size_robot_state + i*size_wall_state + 1, size_robot_state + i*size_wall_state + 1) = 1 - Dxyz(0)*wall_xyz(0)/d2 + 2*Dxyz(0)*wall_xyz(0)*wall_xyz(1)*wall_xyz(1)/(d2*d2);
		jF(size_robot_state + i*size_wall_state + 1, size_robot_state + i*size_wall_state + 2) = 2*Dxyz(0)*wall_xyz(0)*wall_xyz(1)*wall_xyz(2)/(d2*d2);
		jF(size_robot_state + i*size_wall_state + 2, size_robot_state + i*size_wall_state) = -Dxyz(0)*wall_xyz(2)/d2 + 2*Dxyz(0)*wall_xyz(0)*wall_xyz(0)*wall_xyz(2)/(d2*d2);
		jF(size_robot_state + i*size_wall_state + 2, size_robot_state + i*size_wall_state + 1) = 2*Dxyz(0)*wall_xyz(0)*wall_xyz(1)*wall_xyz(2)/(d2*d2);
		jF(size_robot_state + i*size_wall_state + 2, size_robot_state + i*size_wall_state + 2) = 1 - Dxyz(0)*wall_xyz(0)/d2 + 2*Dxyz(0)*wall_xyz(0)*wall_xyz(2)*wall_xyz(2)/(d2*d2);
		jF.block(size_robot_state + i*size_wall_state, size_robot_state + (i+1)*size_wall_state, size_wall_state, i*size_wall_state) = Eigen::MatrixXd::Zero(size_wall_state, i*size_wall_state);
	}

	/*Q*/
	const double sigma = 1.0e-1;
	Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(X.rows(), X.rows());
	
	/*Update*/
	X = F;
	P = jF*P*jF.transpose() + Q;

	std::cout << "Dxyz =" << std::endl << Dxyz << std::endl;
}

void WallEKFSLAM::Publication(void)
{
	std::cout << "Publication" << std::endl;

	geometry_msgs::PoseStamped pose_pub = StateVectorToPoseStamped();
	pose_pub.header.frame_id = "/odom";
	// pose_pub.header.stamp = ros::Time::now();
	pose_pub.header.stamp = time_imu_now;
	pub_pose.publish(pose_pub);
}

geometry_msgs::PoseStamped WallEKFSLAM::StateVectorToPoseStamped(void)
{
	geometry_msgs::PoseStamped pose;
	pose.pose.position.x = X(0, 0);
	pose.pose.position.y = X(1, 0);
	pose.pose.position.z = X(2, 0);
	tf::Quaternion q_orientation = tf::createQuaternionFromRPY(X(3, 0), X(4, 0), X(5, 0));
	pose.pose.orientation.x = q_orientation.x();
	pose.pose.orientation.y = q_orientation.y();
	pose.pose.orientation.z = q_orientation.z();
	pose.pose.orientation.w = q_orientation.w();

	return pose;
}

float WallEKFSLAM::PiToPi(double angle)
{
	return fmod(angle + M_PI, 2*M_PI) - M_PI;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "wall_ekf_slam");
	std::cout << "E.K.F. POSE" << std::endl;
	
	WallEKFSLAM wall_ekf_slam;
	ros::spin();
}
