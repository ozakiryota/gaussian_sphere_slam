#include <ros/ros.h>
#include <geometry_msgs/Quaternion.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
// #include <Eigen/Core>
// #include <Eigen/LU>
#include <std_msgs/Float64MultiArray.h>

class WallEKFSLAM{
	private:
		/*node handle*/
		ros::NodeHandle nh;
		/*subscribe*/
		ros::Subscriber sub_inipose;
		ros::Subscriber sub_bias;
		ros::Subscriber sub_imu;
		ros::Subscriber sub_odom;
		ros::Subscriber sub_dgaussiansphere_obs;
		/*publish*/
		tf::TransformBroadcaster tf_broadcaster;
		ros::Publisher pub_pose;
		ros::Publisher pub_dgaussiansphere_est;	////visualize
		ros::Publisher pub_marker;		////visualize
		ros::Publisher pub_markerarray;	//visualize
		ros::Publisher pub_posearray;	//visualize
		ros::Publisher pub_variance;	//visualize
		/*const*/
		const int size_robot_state = 6;	//X, Y, Z, R, P, Y (Global)
		const int size_wall_state = 3;	//x, y, z (Local)
		/*struct*/
		struct LMInfo{
			Eigen::Vector3d Ng;
			Eigen::VectorXd Xini;
			geometry_msgs::Pose origin;
			bool was_observed_in_this_scan;
			bool is_inward;	//from global origin
			int count_match = 0;
			int count_nomatch = 0;
			double observed_range[3][2] = {};	//[x, y, z][min, max] in wall frame
			double probable_range[3][2] = {};	//[x, y, z][negative, positive]
			bool reached_edge[3][2];	//[x, y, z][negative, positive] in wall frame
			bool available;
			std::vector<bool> list_lm_observed_simul;
			bool was_merged = false;
			bool was_erased = false;
		};
		struct ObsInfo{
			int matched_lm_id = -1;
			Eigen::VectorXd H;
			Eigen::MatrixXd jH;
			Eigen::VectorXd Y;
			Eigen::MatrixXd S;
		};
		/*class*/
		class RemoveUnavailableLM{
			private:
				Eigen::VectorXd X_;
				Eigen::MatrixXd P_;
				int size_robot_state_;
				int size_wall_state_;
				std::vector<int> available_lm_indices;
				std::vector<int> unavailable_lm_indices;
				std::vector<LMInfo> list_lm_info_;
			public:
				RemoveUnavailableLM(const Eigen::VectorXd& X, const Eigen::MatrixXd& P, int size_robot_state, int size_wall_state, std::vector<LMInfo> list_lm_info);
				void InputAvailableLMIndex(int lm_index);
				void InputUnavailableLMIndex(int lm_index);
				void Remove(Eigen::VectorXd& X, Eigen::MatrixXd& P, std::vector<LMInfo>& list_lm_info);
				void Recover(Eigen::VectorXd& X, Eigen::MatrixXd& P, std::vector<LMInfo>& list_lm_info);
		};
		/*objects*/
		Eigen::VectorXd X;
		Eigen::MatrixXd P;
		sensor_msgs::Imu bias;
		pcl::PointCloud<pcl::InterestPoint>::Ptr d_gaussian_sphere {new pcl::PointCloud<pcl::InterestPoint>};
		std::vector<LMInfo> list_lm_info;
		std::vector<LMInfo> list_erased_lm_info;
		/*flags*/
		bool inipose_is_available = false;
		bool bias_is_available = false;
		bool first_callback_imu = true;
		bool first_callback_odom = true;
		/*counter*/
		int counter_imu = 0;
		/*time*/
		ros::Time time_imu_now;
		ros::Time time_imu_last;
		ros::Time time_odom_now;
		ros::Time time_odom_last;
		/*visualization*/
		visualization_msgs::Marker matching_lines;
		visualization_msgs::MarkerArray planes;
	public:
		WallEKFSLAM();
		void SetUpVisualizationMarkerLineList(visualization_msgs::Marker& marker);	//visualization
		void CallbackInipose(const geometry_msgs::QuaternionConstPtr& msg);
		void CallbackBias(const sensor_msgs::ImuConstPtr& msg);
		void CallbackIMU(const sensor_msgs::ImuConstPtr& msg);
		void PredictionIMU(sensor_msgs::Imu imu, double dt);
		void ObservationIMU(sensor_msgs::Imu imu);
		void CallbackOdom(const nav_msgs::OdometryConstPtr& msg);
		void PredictionOdom(nav_msgs::Odometry odom, double dt);
		void CallbackDGaussianSphere(const sensor_msgs::PointCloud2ConstPtr &msg);
		void SearchCorrespondObsID(std::vector<ObsInfo>& list_obs_info, int lm_id);
		void Innovation(int lm_id, const Eigen::Vector3d& Z, Eigen::VectorXd& H, Eigen::MatrixXd& jH, Eigen::VectorXd& Y, Eigen::MatrixXd& S);
		void PushBackLMInfo(const Eigen::Vector3d& Nl);
		/* tf::Quaternion GetRotationQuaternionBetweenVectors(const Eigen::Vector3d& Origin, const Eigen::Vector3d& Target); */
		bool CheckNormalIsInward(const Eigen::Vector3d& Ng);
		void JudgeWallsCanBeObserbed(void);
		void PushBackMarkerMatchingLines(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2);	//visualization
		void ObservationUpdate(const Eigen::VectorXd& Z, const Eigen::VectorXd& H, const Eigen::MatrixXd& jH, const Eigen::VectorXd& Diag_sigma);
		void UpdateLMInfo(int lm_id);
		void PushBackMarkerPlanes(LMInfo lm_info);	//visualization
		void EraseLM(int index);
		Eigen::Vector3d PlaneGlobalToLocal(const Eigen::Vector3d& Ng);
		Eigen::Vector3d PlaneLocalToGlobal(const Eigen::Vector3d& Nl);
		Eigen::Vector3d PointLocalToGlobal(const Eigen::Vector3d& Pl);
		Eigen::Vector3d PointGlobalToWallFrame(const Eigen::Vector3d& Pg, geometry_msgs::Pose origin);
		Eigen::Vector3d PointWallFrameToGlobal(const Eigen::Vector3d& Pl, geometry_msgs::Pose origin);
		void Publication();
		geometry_msgs::PoseStamped StateVectorToPoseStamped(void);
		pcl::PointCloud<pcl::PointXYZ> StateVectorToPC(void);
		Eigen::Matrix3d GetRotationXYZMatrix(const Eigen::Vector3d& RPY, bool inverse);
		void VectorVStack(Eigen::VectorXd& A, const Eigen::VectorXd& B);
		void MatrixVStack(Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
		geometry_msgs::Quaternion QuatEigenToMsg(Eigen::Quaterniond q_eigen);
		double PiToPi(double angle);
};

WallEKFSLAM::WallEKFSLAM()
{
	sub_inipose = nh.subscribe("/initial_orientation", 1, &WallEKFSLAM::CallbackInipose, this);
	sub_bias = nh.subscribe("/imu/bias", 1, &WallEKFSLAM::CallbackBias, this);
	sub_imu = nh.subscribe("/imu/data", 1, &WallEKFSLAM::CallbackIMU, this);
	sub_odom = nh.subscribe("/tinypower/odom", 1, &WallEKFSLAM::CallbackOdom, this);
	sub_dgaussiansphere_obs = nh.subscribe("/d_gaussian_sphere_obs", 1, &WallEKFSLAM::CallbackDGaussianSphere, this);
	pub_pose = nh.advertise<geometry_msgs::PoseStamped>("/wall_ekf_slam/pose", 1);
	pub_posearray = nh.advertise<geometry_msgs::PoseArray>("/wall_origins", 1);
	pub_dgaussiansphere_est = nh.advertise<sensor_msgs::PointCloud2>("/d_gaussian_sphere_est", 1);
	pub_marker = nh.advertise<visualization_msgs::Marker>("matching_lines", 1);
	pub_markerarray = nh.advertise<visualization_msgs::MarkerArray>("planes", 1);
	pub_variance = nh.advertise<std_msgs::Float64MultiArray>("variance", 1);
	X = Eigen::VectorXd::Zero(size_robot_state);
	P = Eigen::MatrixXd::Identity(size_robot_state, size_robot_state);
	SetUpVisualizationMarkerLineList(matching_lines);
}

void WallEKFSLAM::SetUpVisualizationMarkerLineList(visualization_msgs::Marker& marker)
{
	marker.header.frame_id = "/odom";
	marker.ns = "matching_lines";
	marker.id = 0;
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose.orientation.x = 0.0;
	marker.pose.orientation.y = 0.0;
	marker.pose.orientation.z = 0.0;
	marker.pose.orientation.w = 1.0;
	marker.type = visualization_msgs::Marker::LINE_LIST;
	marker.scale.x = 0.2;
	marker.color.r = 0.0;
	marker.color.g = 0.0;
	marker.color.b = 1.0;
	marker.color.a = 1.0;
}

void WallEKFSLAM::CallbackInipose(const geometry_msgs::QuaternionConstPtr& msg)
{
	if(!inipose_is_available){
		tf::Quaternion q_pose;
		quaternionMsgToTF(*msg, q_pose);
		tf::Matrix3x3(q_pose).getRPY(X(3), X(4), X(5));
		inipose_is_available = true;
		std::cout << "inipose_is_available = " << inipose_is_available << std::endl;
		std::cout << "initial robot state = " << std::endl << X.segment(0, size_robot_state) << std::endl;
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
	/* std::cout << "Callback IMU" << std::endl; */

	time_imu_now = msg->header.stamp;
	double dt;
	try{
		dt = (time_imu_now - time_imu_last).toSec();
	}
	catch(std::runtime_error& ex) {
		ROS_ERROR("Exception: [%s]", ex.what());
	}
	time_imu_last = time_imu_now;
	if(first_callback_imu)	dt = 0.0;
	else if(inipose_is_available){
		/*angular velocity*/
		PredictionIMU(*msg, dt);
		/*linear acceleration*/
		counter_imu++;
		const int rate = 100;
		// if(counter_imu==rate)	ObservationIMU(*msg);
		counter_imu %= rate;	
	}
	
	Publication();

	first_callback_imu = false;
}

void WallEKFSLAM::PredictionIMU(sensor_msgs::Imu imu, double dt)
{
	// RemoveUnavailableLM remover(X, P, size_robot_state, size_wall_state, list_lm_info);
	// for(size_t i=0;i<list_lm_info.size();i++){
	// 	if(list_lm_info[i].available)	remover.InputAvailableLMIndex(i);
	// 	else	remover.InputUnavailableLMIndex(i);
	// }
	// remover.Remove(X, P, list_lm_info);

	/* std::cout << "PredictionIMU" << std::endl; */
	double x = X(0);
	double y = X(1);
	double z = X(2);
	double r_ = X(3);
	double p_ = X(4);
	double y_ = X(5);

	double delta_r = imu.angular_velocity.x*dt;
	double delta_p = imu.angular_velocity.y*dt;
	double delta_y = imu.angular_velocity.z*dt;
	if(bias_is_available){
		delta_r -= bias.angular_velocity.x*dt;
		delta_p -= bias.angular_velocity.y*dt;
		delta_y -= bias.angular_velocity.z*dt;
	}
	Eigen::Vector3d Drpy = {delta_r, delta_p, delta_y};
	
	int num_wall = (X.size() - size_robot_state)/size_wall_state;

	Eigen::Matrix3d Rot_rpy;	//normal rotation
	Rot_rpy <<	1,	sin(r_)*tan(p_),	cos(r_)*tan(p_),
				0,	cos(r_),			-sin(r_),
				0,	sin(r_)/cos(p_),	cos(r_)/cos(p_);

	/*F*/
	Eigen::VectorXd F(X.size());
	F.segment(0, 3) = X.segment(0, 3);
	F.segment(3, 3) = X.segment(3, 3) + Rot_rpy*Drpy;
	for(int i=3;i<6;i++)	F(i) = PiToPi(F(i));
	F.segment(size_robot_state, num_wall*size_wall_state) = X.segment(size_robot_state, num_wall*size_wall_state);

	/*jF*/
	Eigen::MatrixXd jF = Eigen::MatrixXd::Zero(X.size(), X.size());
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
	jF.block(size_robot_state, size_robot_state, num_wall*size_wall_state, num_wall*size_wall_state) = Eigen::MatrixXd::Identity(num_wall*size_wall_state, num_wall*size_wall_state);
	
	/*Q*/
	const double sigma = 1.0e-4;
	Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(X.size(), X.size());
	Q.block(0, 0, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
	Q.block(size_robot_state, size_robot_state, num_wall*size_wall_state, num_wall*size_wall_state) = Eigen::MatrixXd::Zero(num_wall*size_wall_state, num_wall*size_wall_state);
	
	/*Update*/
	X = F;
	P = jF*P*jF.transpose() + Q;
	
	// remover.Recover(X, P, list_lm_info);

	/* std::cout << "X =" << std::endl << X << std::endl; */
	/* std::cout << "P =" << std::endl << P << std::endl; */
	/* std::cout << "jF =" << std::endl << jF << std::endl; */
}

void WallEKFSLAM::ObservationIMU(sensor_msgs::Imu imu)
{
	std::cout << "Observation IMU---------------" << std::endl;

	Eigen::Vector3d RPY = X.segment(3, 3);
	/*Z*/
	Eigen::Vector3d Z(
		-imu.linear_acceleration.x,
		-imu.linear_acceleration.y,
		-imu.linear_acceleration.z
	);

	/*H*/
	Eigen::Vector3d G(0, 0, -9.80665);
	Eigen::Vector3d H = GetRotationXYZMatrix(RPY, true)*G;
	/*jH*/
	Eigen::MatrixXd jH = Eigen::MatrixXd::Zero(Z.size(), X.size());
	/*dH/d(RPY)*/
	jH(0, 3) = 0;
	jH(0, 4) = G(0)*( -sin(RPY(1))*cos(RPY(2)) ) + G(1)*( -sin(RPY(1))*sin(RPY(2)) ) + G(2)*( -cos(RPY(1)) );
	jH(0, 5) = G(0)*( -cos(RPY(1))*sin(RPY(2)) ) + G(1)*( cos(RPY(1))*cos(RPY(2)) );
	jH(1, 3) = G(0)*( cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)) ) + G(1)*( cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) - sin(RPY(0))*cos(RPY(2)) ) + G(2)*( cos(RPY(0))*cos(RPY(1)) );
	jH(1, 4) = G(0)*( sin(RPY(0))*cos(RPY(1))*cos(RPY(2)) ) + G(1)*( sin(RPY(0))*cos(RPY(1))*sin(RPY(2)) ) + G(2)*( -sin(RPY(0))*sin(RPY(1)) );
	jH(1, 5) = G(0)*( -sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) - cos(RPY(0))*cos(RPY(2)) ) + G(1)*( sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) - cos(RPY(0))*sin(RPY(2)) );
	jH(2, 3) = G(0)*( -sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) + cos(RPY(0))*sin(RPY(2)) ) + G(1)*( -sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) - cos(RPY(0))*cos(RPY(2)) ) + G(2)*( -sin(RPY(0))*cos(RPY(1)) );
	jH(2, 4) = G(0)*( cos(RPY(0))*cos(RPY(1))*cos(RPY(2)) ) + G(1)*( cos(RPY(0))*cos(RPY(1))*sin(RPY(2)) ) + G(2)*( -cos(RPY(0))*sin(RPY(1)) );
	jH(2, 5) = G(0)*( -cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) + sin(RPY(0))*cos(RPY(2)) ) + G(1)*( cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)) );
	/*R*/
	const double sigma = 1.0e-1;
	Eigen::MatrixXd R = sigma*Eigen::MatrixXd::Identity(Z.size(), Z.size());
	/*Y, S, K, I*/
	Eigen::VectorXd Y = Z - H;
	Eigen::MatrixXd S = jH*P*jH.transpose() + R;
	Eigen::MatrixXd K = P*jH.transpose()*S.inverse();
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(X.size(), X.size());
	/*update*/
	X = X + K*Y;
	P = (I - K*jH)*P;
}

void WallEKFSLAM::CallbackOdom(const nav_msgs::OdometryConstPtr& msg)
{
	/* std::cout << "Callback Odom" << std::endl; */

	time_odom_now = msg->header.stamp;
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
	/* std::cout << "Prediction Odom" << std::endl; */
	// RemoveUnavailableLM remover(X, P, size_robot_state, size_wall_state, list_lm_info);
	// for(size_t i=0;i<list_lm_info.size();i++){
	// 	if(list_lm_info[i].available)	remover.InputAvailableLMIndex(i);
	// 	else	remover.InputUnavailableLMIndex(i);
	// }
	// remover.Remove(X, P, list_lm_info);

	double x = X(0);
	double y = X(1);
	double z = X(2);
	double r_ = X(3);
	double p_ = X(4);
	double y_ = X(5);
	Eigen::Vector3d Dxyz = {odom.twist.twist.linear.x*dt, 0, 0};

	int num_wall = (X.size() - size_robot_state)/size_wall_state;

	/*F*/
	Eigen::VectorXd F(X.size());
	F.segment(0, 3) = X.segment(0, 3) + GetRotationXYZMatrix(X.segment(3, 3), false)*Dxyz;
	F.segment(3, 3) = X.segment(3, 3);
	F.segment(size_robot_state, num_wall*size_wall_state) = X.segment(size_robot_state, num_wall*size_wall_state);

	/*jF*/
	Eigen::MatrixXd jF = Eigen::MatrixXd::Zero(X.size(), X.size());
	/*jF-xyz*/
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
	/*jF-rpy*/
	jF.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
	jF.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity();
	jF.block(3, size_robot_state, 3, num_wall*size_wall_state) = Eigen::MatrixXd::Zero(3, num_wall*size_wall_state);
	/*jF-wall_xyz*/
	jF.block(size_robot_state, size_robot_state, num_wall*size_wall_state, num_wall*size_wall_state) = Eigen::MatrixXd::Identity(num_wall*size_wall_state, num_wall*size_wall_state);

	/*Q*/
	const double sigma = 1.0e-4;
	Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(X.size(), X.size());
	Q.block(3, 3, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
	Q.block(size_robot_state, size_robot_state, num_wall*size_wall_state, num_wall*size_wall_state) = Eigen::MatrixXd::Zero(num_wall*size_wall_state, num_wall*size_wall_state);
	
	/* std::cout << "X =" << std::endl << X << std::endl; */
	/* std::cout << "P =" << std::endl << P << std::endl; */
	/* std::cout << "jF =" << std::endl << jF << std::endl; */
	/* std::cout << "F =" << std::endl << F << std::endl; */
	
	/*Update*/
	X = F;
	P = jF*P*jF.transpose() + Q;

	// remover.Recover(X, P, list_lm_info);
}

void WallEKFSLAM::CallbackDGaussianSphere(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	std::cout << "Callback D-Gaussian Sphere" << std::endl;
	std::cout << "num_wall = " << (X.size() - size_robot_state)/size_wall_state << std::endl;
	
	pcl::fromROSMsg(*msg, *d_gaussian_sphere);
	std::cout << "d_gaussian_sphere->points.size() = " << d_gaussian_sphere->points.size() << std::endl;
	for(size_t i=0;i<d_gaussian_sphere->points.size();i++)	std::cout << "d_gaussian_sphere->points[" << i << "].strength = " << d_gaussian_sphere->points[i].strength << std::endl;

	std::vector<ObsInfo> list_obs_info(d_gaussian_sphere->points.size());

	matching_lines.points.clear();	//visualization
	planes.markers.clear();	//visualization

	JudgeWallsCanBeObserbed();
	// RemoveUnavailableLM remover(X, P, size_robot_state, size_wall_state, list_lm_info);
	// for(size_t i=0;i<list_lm_info.size();i++){
	// 	if(list_lm_info[i].available)	remover.InputAvailableLMIndex(i);
	// 	else	remover.InputUnavailableLMIndex(i);
	// }
	// remover.Remove(X, P, list_lm_info);

	int num_wall = (X.size() - size_robot_state)/size_wall_state;
	/*matching*/
	for(size_t i=0;i<list_lm_info.size();i++){
		if(list_lm_info[i].available)	SearchCorrespondObsID(list_obs_info, i);
	}
	// for(int i=0;i<num_wall;i++){
	// 	SearchCorrespondObsID(list_obs_info, i);
	// }
	/*new landmark or update*/
	Eigen::VectorXd Xnew(0);
	Eigen::VectorXd Zstacked(0);
	Eigen::VectorXd Hstacked(0);
	Eigen::MatrixXd jHstacked(0, 0);
	Eigen::VectorXd Diag_sigma(0);

	for(size_t i=0;i<list_obs_info.size();i++){
		int lm_id = list_obs_info[i].matched_lm_id;
		Eigen::Vector3d Nl(
			d_gaussian_sphere->points[i].x,
			d_gaussian_sphere->points[i].y,
			d_gaussian_sphere->points[i].z
		);
		if(lm_id==-1){	//new landmark
			VectorVStack(Xnew, PlaneLocalToGlobal(Nl));
			PushBackLMInfo(Nl);
		}
		else{
			/*update LM info*/
			UpdateLMInfo(lm_id);
			/*judge in maching time*/
			const int threshold_count_match = 5;
			if(list_lm_info[lm_id].count_match>threshold_count_match)	list_lm_info[lm_id].available = true;
			else	list_lm_info[lm_id].available = false;
			/*stack*/
			if(list_lm_info[lm_id].available){
				PushBackMarkerMatchingLines(X.segment(size_robot_state + lm_id*size_wall_state, size_wall_state), PointLocalToGlobal(Nl));
				VectorVStack(Zstacked, Nl);
				VectorVStack(Hstacked, list_obs_info[i].H);
				MatrixVStack(jHstacked, list_obs_info[i].jH);
				double tmp_sigma = 0.2*100/(double)d_gaussian_sphere->points[i].strength;
				/* tmp_sigma *= 0.1*list_lm_info[lm_id].count_match; */
				VectorVStack(Diag_sigma, Eigen::Vector3d(tmp_sigma, tmp_sigma, tmp_sigma));
				std::cout << "tmp_sigma = " << tmp_sigma << std::endl;
				
				/*test*/
				// Innovation(lm_id, Nl, list_obs_info[i].H, list_obs_info[i].jH, list_obs_info[i].Y, list_obs_info[i].S);
				// ObservationUpdate(Nl, list_obs_info[i].H, list_obs_info[i].jH);
			}
		}
	}
	/*arrange LM info*/
	const double tolerance = 2.0;
	for(int i=0;i<list_lm_info.size();i++){
		list_lm_info[i].list_lm_observed_simul.resize(list_lm_info.size(), false);	//keeps valuses and inputs "false" into new memories
		/*update unmached lm info*/
		if(!list_lm_info[i].was_observed_in_this_scan){
			/*count no-match*/
			list_lm_info[i].count_nomatch++;
			const int threshold_count_match = 5;
			const int threshold_count_nomatch = 1000;
			if(list_lm_info[i].count_match<threshold_count_match && list_lm_info[i].count_nomatch>threshold_count_nomatch)	list_lm_info[i].was_erased = true;
			/*observed range*/
			Eigen::Vector3d Position_in_wall_frame = PointGlobalToWallFrame(X.segment(0, 3), list_lm_info[i].origin);
			if(Position_in_wall_frame(0)<list_lm_info[i].observed_range[0][1]){
				for(int j=1;j<3;j++){
					if(Position_in_wall_frame(j) < list_lm_info[i].observed_range[j][0]-tolerance)	list_lm_info[i].reached_edge[j][0] = true;
					if(Position_in_wall_frame(j) > list_lm_info[i].observed_range[j][1]+tolerance)	list_lm_info[i].reached_edge[j][1] = true;
				}
			}
		}
		/*make list of LM watched at the same time*/
		else{
			for(int j=0;j<list_lm_info.size();j++){
				if(list_lm_info[j].was_observed_in_this_scan)	list_lm_info[i].list_lm_observed_simul[j] = true;
			}
		}
		PushBackMarkerPlanes(list_lm_info[i]);
		/*reset*/
		list_lm_info[i].was_observed_in_this_scan = false;
	}
	std::cout << "test1" << std::endl;
	/*update*/
	if(Zstacked.size()>0 && inipose_is_available)	ObservationUpdate(Zstacked, Hstacked, jHstacked, Diag_sigma);
	// remover.Recover(X, P, list_lm_info);
	/*Registration of new walls*/
	X.conservativeResize(X.size() + Xnew.size());
	X.segment(X.size() - Xnew.size(), Xnew.size()) = Xnew;
	Eigen::MatrixXd Ptmp = P;
	const double sigma = 0.25;
	P = sigma*Eigen::MatrixXd::Identity(X.size(), X.size());
	P.block(0, 0, Ptmp.rows(), Ptmp.cols()) = Ptmp;
	std::cout << "test2" << std::endl;
	/*delete marged LM*/
	for(size_t i=0;i<list_lm_info.size();){
		if(list_lm_info[i].was_merged || list_lm_info[i].was_erased)	EraseLM(i);
		else i++;
	}
	std::cout << "test3" << std::endl;
	for(size_t i=0;i<list_erased_lm_info.size();i++)	PushBackMarkerPlanes(list_erased_lm_info[i]);

	Publication();
}

void WallEKFSLAM::SearchCorrespondObsID(std::vector<ObsInfo>& list_obs_info, int lm_id)
{
	Eigen::VectorXd H_correspond;
	Eigen::MatrixXd jH_correspond;
	Eigen::VectorXd Y_correspond;
	Eigen::MatrixXd S_correspond;

	const double threshold_mahalanobis_dist = 0.36;	//chi-square distribution
	double min_mahalanobis_dist = threshold_mahalanobis_dist;
	// const double threshold_euclidean_dist = 0.15;	//test
	const double threshold_euclidean_dist = 0.2;	//test
	double min_euclidean_dist = threshold_euclidean_dist;	//test
	int correspond_id = -1;
	/*search*/
	for(size_t i=0;i<d_gaussian_sphere->points.size();i++){
		Eigen::Vector3d Zi(
			d_gaussian_sphere->points[i].x,
			d_gaussian_sphere->points[i].y,
			d_gaussian_sphere->points[i].z
		);

		Eigen::VectorXd Hi;
		Eigen::MatrixXd jHi;
		Eigen::VectorXd Yi;
		Eigen::MatrixXd Si;
		Innovation(lm_id, Zi, Hi, jHi, Yi, Si);

		double mahalanobis_dist = Yi.transpose()*Si.inverse()*Yi;
		double euclidean_dist = Yi.norm();	//test
		/* std::cout << "mahalanobis_dist = " << mahalanobis_dist << std::endl; */
		if(std::isnan(mahalanobis_dist)){	//test
			std::cout << "mahalanobis_dist is NAN" << std::endl;
			// exit(1);
		}

		if(euclidean_dist<min_euclidean_dist){	//test
			min_euclidean_dist = euclidean_dist;
			correspond_id = i;
			H_correspond = Hi;
			jH_correspond = jHi;
			Y_correspond = Yi;
			S_correspond = Si;
		}
		// if(!std::isnan(mahalanobis_dist) && mahalanobis_dist<min_mahalanobis_dist){
		// 	min_mahalanobis_dist = mahalanobis_dist;
		// 	correspond_id = i;
		// 	H_correspond = Hi;
		// 	jH_correspond = jHi;
		// 	Y_correspond = Yi;
		// 	S_correspond = Si;
		// }
	}
	std::cout << "mahalanobis_dist = " << Y_correspond.transpose()*S_correspond.inverse()*Y_correspond << std::endl;
	/*input*/
	if(correspond_id!=-1){
		if(list_obs_info[correspond_id].matched_lm_id==-1){
			list_obs_info[correspond_id].matched_lm_id = lm_id;
			list_obs_info[correspond_id].H = H_correspond;
			list_obs_info[correspond_id].jH = jH_correspond;
			list_obs_info[correspond_id].Y = Y_correspond;
			list_obs_info[correspond_id].S = S_correspond;
		}
		else{
			int id1 = list_obs_info[correspond_id].matched_lm_id;
			int id2 = lm_id;
			if(!list_lm_info[id1].list_lm_observed_simul[id2]){
				std::cout << "merged!" << std::endl;
				list_lm_info[id2].was_merged = true;
				/*convert observed range in id2-frame to id1-frame*/
				double observed_range_of_id2_in_id1_frame[3][2] = {};
				Eigen::Vector3d Negative(
					list_lm_info[id2].observed_range[0][0],
					list_lm_info[id2].observed_range[1][0],
					list_lm_info[id2].observed_range[2][0]
				);
				Eigen::Vector3d Positive(
					list_lm_info[id2].observed_range[0][1],
					list_lm_info[id2].observed_range[1][1],
					list_lm_info[id2].observed_range[2][1]
				);
				Negative = PointGlobalToWallFrame( PointWallFrameToGlobal(Negative, list_lm_info[id2].origin), list_lm_info[id1].origin );	//negative vector of id2 in id1 frame
				Positive = PointGlobalToWallFrame( PointWallFrameToGlobal(Positive, list_lm_info[id2].origin), list_lm_info[id1].origin );	//positive vector of id2 in id1 frame
				for(int j=0;j<3;j++){
					if(Negative(j)<Positive(j)){
						observed_range_of_id2_in_id1_frame[j][0] = Negative(j);
						observed_range_of_id2_in_id1_frame[j][1] = Positive(j);
					}
					else{
						observed_range_of_id2_in_id1_frame[j][0] = Positive(j);
						observed_range_of_id2_in_id1_frame[j][1] = Negative(j);
					}
				}
				/*merge observed range*/
				for(int j=0;j<3;j++){
					if(observed_range_of_id2_in_id1_frame[j][0] < list_lm_info[id1].observed_range[j][0]){
						list_lm_info[id1].observed_range[j][0] = observed_range_of_id2_in_id1_frame[j][0];
						list_lm_info[id1].reached_edge[j][0] = list_lm_info[id2].reached_edge[j][0];
					}
					if(observed_range_of_id2_in_id1_frame[j][1] > list_lm_info[id1].observed_range[j][1]){
						list_lm_info[id1].observed_range[j][1] = observed_range_of_id2_in_id1_frame[j][1];
						list_lm_info[id1].reached_edge[j][1] = list_lm_info[id2].reached_edge[j][1];
					}
				}
			}
			else if(min_euclidean_dist<list_obs_info[correspond_id].Y.norm()){
				list_obs_info[correspond_id].matched_lm_id = lm_id;
				list_obs_info[correspond_id].H = H_correspond;
				list_obs_info[correspond_id].jH = jH_correspond;
				list_obs_info[correspond_id].Y = Y_correspond;
				list_obs_info[correspond_id].S = S_correspond;
			}
		}
	}
}

void WallEKFSLAM::Innovation(int lm_id, const Eigen::Vector3d& Z, Eigen::VectorXd& H, Eigen::MatrixXd& jH, Eigen::VectorXd& Y, Eigen::MatrixXd& S)
{
	Eigen::Vector3d Ng = X.segment(size_robot_state + lm_id*size_wall_state, size_wall_state);
	Eigen::Vector3d RPY = X.segment(3, 3);
	double d2 = Ng.dot(Ng);
	/*H*/
	H = PlaneGlobalToLocal(Ng);
	/*jH*/
	jH = Eigen::MatrixXd::Zero(Z.size(), X.size());
	/*dH/d(XYZ)*/
	Eigen::Vector3d rotN = GetRotationXYZMatrix(RPY, true)*Ng;
	for(int j=0;j<Z.size();j++){
		for(int k=0;k<3;k++)	jH(j, k) = -Ng(k)/d2*rotN(j);
	}
	/*dH/d(RPY)*/
	Eigen::Vector3d delN = Ng - Ng.dot(X.segment(0, 3))/d2*Ng;
	jH(0, 3) = 0;
	jH(0, 4) = (-sin(RPY(1))*cos(RPY(2)))*delN(0) + (-sin(RPY(1))*sin(RPY(2)))*delN(1) + (-cos(RPY(1)))*delN(2);
	jH(0, 5) = (-cos(RPY(1))*sin(RPY(2)))*delN(0) + (cos(RPY(1))*cos(RPY(2)))*delN(1);
	jH(1, 3) = (cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)))*delN(0) + (cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) - sin(RPY(0))*cos(RPY(2)))*delN(1) + (cos(RPY(0))*cos(RPY(1)))*delN(2);
	jH(1, 4) = (sin(RPY(0))*cos(RPY(1))*cos(RPY(2)))*delN(0) + (sin(RPY(0))*cos(RPY(1))*sin(RPY(2)))*delN(1) + (-sin(RPY(0))*sin(RPY(1)))*delN(2);
	jH(1, 5) = (-sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) - cos(RPY(0))*cos(RPY(2)))*delN(0) + (sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) - cos(RPY(0))*sin(RPY(2)))*delN(1);
	jH(2, 3) = (-sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) + cos(RPY(0))*sin(RPY(2)))*delN(0) + (-sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) - cos(RPY(0))*cos(RPY(2)))*delN(1) + (-sin(RPY(0))*cos(RPY(1)))*delN(2);
	jH(2, 4) = (cos(RPY(0))*cos(RPY(1))*cos(RPY(2)))*delN(0) + (cos(RPY(0))*cos(RPY(1))*sin(RPY(2)))*delN(1) + (-cos(RPY(0))*sin(RPY(1)))*delN(2);
	jH(2, 5) = (-cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) + sin(RPY(0))*cos(RPY(2)))*delN(0) + (cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)))*delN(1);
	/*dH/d(Wall)*/
	Eigen::Matrix3d Tmp;
	for(int j=0;j<Z.size();j++){
		for(int k=0;k<size_wall_state;k++){
			if(j==k)	Tmp(j, k) = 1 - ((Ng.dot(X.segment(0, 3)) + Ng(j)*X(k))/d2 - Ng(j)*Ng.dot(X.segment(0, 3))/(d2*d2)*2*Ng(k));
			else	Tmp(j, k) = -(Ng(j)*X(k)/d2 - Ng(j)*Ng.dot(X.segment(0, 3))/(d2*d2)*2*Ng(k));
		}
	}
	jH.block(0, size_robot_state + lm_id*size_wall_state, Z.size(), size_wall_state) = GetRotationXYZMatrix(RPY, true)*Tmp;
	/*R*/
	const double sigma = 1.0e-2;
	Eigen::MatrixXd R = sigma*Eigen::MatrixXd::Identity(Z.size(), Z.size());
	/*Y, S*/
	Y = Z - H;
	S = jH*P*jH.transpose() + R;
}

void WallEKFSLAM::PushBackLMInfo(const Eigen::Vector3d& Nl)
{
	/*origin-position*/
	Eigen::Vector3d Pg = PointLocalToGlobal(Nl);
	/*origin-orientation*/
	std::vector<Eigen::Vector3d> Axes_wall_local(3);	//vectors of xyz axes
	Axes_wall_local[0] = -Nl;
	Axes_wall_local[2] = Eigen::Vector3d(0,0,1) - Eigen::Vector3d(0,0,1).dot(Nl)/Nl.dot(Nl)*Nl;
	Axes_wall_local[1] = -Axes_wall_local[0].cross(Axes_wall_local[2]);	//this cross product needs x(-1)

	Eigen::Matrix3d Axes_wall_global;
	for(int i=0;i<3;i++){
		Eigen::Vector3d Axis_wall_global = GetRotationXYZMatrix(X.segment(3, 3), false)*Axes_wall_local[i];
		Axis_wall_global /= Axis_wall_global.norm();
		Axes_wall_global.block(0, i, 3, 1) = Axis_wall_global;
	}
	Eigen::Quaterniond q_orientation(Axes_wall_global);
	q_orientation.normalize();

	/*push back*/
	LMInfo tmp;
	tmp.Ng = PlaneLocalToGlobal(Nl);
	tmp.Xini = X.segment(0, size_robot_state);
	tmp.origin.position.x = Pg(0);
	tmp.origin.position.y = Pg(1);
	tmp.origin.position.z = Pg(2);
	tmp.origin.orientation = QuatEigenToMsg(q_orientation);
	tmp.observed_range[0][1] = Nl.norm();
	for(int j=1;j<3;j++){	//y,z
		tmp.reached_edge[j][0] = false;
		tmp.reached_edge[j][1] = false;
	}
	tmp.is_inward = CheckNormalIsInward(PlaneLocalToGlobal(Nl));
	tmp.was_observed_in_this_scan = true;
	tmp.count_match = 0;
	list_lm_info.push_back(tmp);
}

bool WallEKFSLAM::CheckNormalIsInward(const Eigen::Vector3d& Ng)
{
	Eigen::Vector3d VerticalPosition = X.segment(0, 3).dot(Ng)/Ng.dot(Ng)*Ng;
	double dot = VerticalPosition.dot(Ng);
	if(dot<0)	return true;
	else{
		double dist_wall = Ng.norm();
		double dist_robot = VerticalPosition.norm();
		if(dist_robot<dist_wall)	return true;
		else	return false;
	}
}

void WallEKFSLAM::JudgeWallsCanBeObserbed(void)
{
	const double small_tolerance = 1.0;
	const double large_tolerance = 5.0;
	
	for(size_t i=0;i<list_lm_info.size();i++){
		Eigen::Vector3d Ng = X.segment(size_robot_state+i*size_wall_state, size_wall_state);
		/*judge in direction of normal*/
		if(list_lm_info[i].is_inward!=CheckNormalIsInward(Ng))	list_lm_info[i].available = false;
		else if(list_lm_info[i].was_merged || list_lm_info[i].was_erased)	list_lm_info[i].available = false;	//test
		else{
			/*set probable range*/
			for(int j=1;j<3;j++){	//y,z
				if(list_lm_info[i].reached_edge[j][0])	list_lm_info[i].probable_range[j][0] = list_lm_info[i].observed_range[j][0] - small_tolerance;
				else	list_lm_info[i].probable_range[j][0] = list_lm_info[i].observed_range[j][0] - large_tolerance;
				if(list_lm_info[i].reached_edge[j][1])	list_lm_info[i].probable_range[j][1] = list_lm_info[i].observed_range[j][1] + small_tolerance;
				else	list_lm_info[i].probable_range[j][1] = list_lm_info[i].observed_range[j][1] + large_tolerance;
			}
			/*judge in distance*/
			Eigen::Vector3d Position_in_wall_frame = PointGlobalToWallFrame(X.segment(0, 3), list_lm_info[i].origin);
			if(Position_in_wall_frame(1)<list_lm_info[i].probable_range[1][0] || Position_in_wall_frame(1)>list_lm_info[i].probable_range[1][1] || Position_in_wall_frame(2)<list_lm_info[i].probable_range[2][0] || Position_in_wall_frame(2)>list_lm_info[i].probable_range[2][1])	list_lm_info[i].available = false;
			else	list_lm_info[i].available = true;
		}
	}
}

void WallEKFSLAM::PushBackMarkerMatchingLines(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2)
{
	geometry_msgs::Point tmp_p1;
	tmp_p1.x = P1[0];
	tmp_p1.y = P1[1];
	tmp_p1.z = P1[2];
	geometry_msgs::Point tmp_p2;
	tmp_p2.x = P2[0];
	tmp_p2.y = P2[1];
	tmp_p2.z = P2[2];
	matching_lines.points.push_back(tmp_p1);
	matching_lines.points.push_back(tmp_p2);
}

void WallEKFSLAM::ObservationUpdate(const Eigen::VectorXd& Z, const Eigen::VectorXd& H, const Eigen::MatrixXd& jH, const Eigen::VectorXd& Diag_sigma)
{
	Eigen::VectorXd Y = Z - H;
	// const double sigma = 1.0e-1;
	// const double sigma = 1.2e-1;	//using floor
	const double sigma = 1.0e-1;	//test
	// Eigen::MatrixXd R = sigma*Eigen::MatrixXd::Identity(Z.size(), Z.size());
	Eigen::MatrixXd R = Diag_sigma.asDiagonal();
	Eigen::MatrixXd S = jH*P*jH.transpose() + R;
	Eigen::MatrixXd K = P*jH.transpose()*S.inverse();
	X = X + K*Y;
	for(int i=3;i<6;i++)	X(i) = PiToPi(X(i));
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(X.size(), X.size());
	P = (I - K*jH)*P;
}

Eigen::Vector3d WallEKFSLAM::PlaneGlobalToLocal(const Eigen::Vector3d& Ng)
{
	Eigen::Vector3d DeltaVertical = X.segment(0, 3).dot(Ng)/Ng.dot(Ng)*Ng;
	Eigen::Vector3d delL = Ng - DeltaVertical;
	Eigen::Vector3d Nl = GetRotationXYZMatrix(X.segment(3, 3), true)*delL;
	return Nl;
}

void WallEKFSLAM::UpdateLMInfo(int lm_id)
{
	Eigen::Vector3d Ng = X.segment(size_robot_state + lm_id*size_wall_state, size_wall_state);
	/*count*/
	list_lm_info[lm_id].was_observed_in_this_scan = true;
	list_lm_info[lm_id].count_match += 1;
	/*observed range*/
	Eigen::Vector3d Position_in_wall_frame = PointGlobalToWallFrame(X.segment(0, 3), list_lm_info[lm_id].origin);
	for(int j=0;j<3;j++){
		if(Position_in_wall_frame(j) < list_lm_info[lm_id].observed_range[j][0]){
			list_lm_info[lm_id].observed_range[j][0] = Position_in_wall_frame(j);
			list_lm_info[lm_id].reached_edge[j][0] = false;
		}
		if(Position_in_wall_frame(j) > list_lm_info[lm_id].observed_range[j][1]){
			list_lm_info[lm_id].observed_range[j][1] = Position_in_wall_frame(j);
			list_lm_info[lm_id].reached_edge[j][1] = false;
		}
	}
	/*origin-position*/
	Eigen::Vector3d DeltaVertical = list_lm_info[lm_id].Xini.segment(0, 3).dot(Ng)/Ng.dot(Ng)*Ng;
	Eigen::Vector3d delL = Ng - DeltaVertical;
	Eigen::Vector3d Nl = GetRotationXYZMatrix(list_lm_info[lm_id].Xini.segment(3, 3), true)*delL;
	Eigen::Vector3d Pg = GetRotationXYZMatrix(list_lm_info[lm_id].Xini.segment(3, 3), false)*Nl + list_lm_info[lm_id].Xini.segment(0, 3);
	/*origin-orientation*/
	tf::Quaternion q_origin_orientation_old;
	quaternionMsgToTF(list_lm_info[lm_id].origin.orientation, q_origin_orientation_old);
	double theta = acos(list_lm_info[lm_id].Ng.dot(Ng)/list_lm_info[lm_id].Ng.norm()/Ng.norm());
	if(std::isnan(theta))	theta = 0.0;
	Eigen::Vector3d Axis = list_lm_info[lm_id].Ng.cross(Ng);
	Axis.normalize();
	tf::Quaternion q_rotation(sin(theta/2.0)*Axis(0), sin(theta/2.0)*Axis(1), sin(theta/2.0)*Axis(2), cos(theta/2.0));
	q_rotation.normalize();
	/*input*/
	list_lm_info[lm_id].Ng = Ng;
	list_lm_info[lm_id].origin.position.x = Pg(0);
	list_lm_info[lm_id].origin.position.y = Pg(1);
	list_lm_info[lm_id].origin.position.z = Pg(2);
	quaternionTFToMsg((q_rotation*q_origin_orientation_old).normalized(), list_lm_info[lm_id].origin.orientation);
}

void WallEKFSLAM::PushBackMarkerPlanes(LMInfo lm_info)
{
	const double thickness = 0.1;
	double width = lm_info.observed_range[1][1] - lm_info.observed_range[1][0];
	double height = lm_info.observed_range[2][1] - lm_info.observed_range[2][0];
	/* double width = lm_info.probable_range[1][1] - lm_info.probable_range[1][0];		//test */
	/* double height = lm_info.probable_range[2][1] - lm_info.probable_range[2][0];	//test */
	tf::Quaternion q_origin_orientation;
	quaternionMsgToTF(lm_info.origin.orientation, q_origin_orientation);
	tf::Quaternion q_bias(
		0.0,
		(lm_info.observed_range[1][1] + lm_info.observed_range[1][0])/2.0,
		(lm_info.observed_range[2][1] + lm_info.observed_range[2][0])/2.0,
		0.0
	);
	q_bias = q_origin_orientation*q_bias*q_origin_orientation.inverse();

	visualization_msgs::Marker tmp;
	tmp.header.frame_id = "/odom";
	tmp.header.stamp = time_imu_now;
	tmp.ns = "planes";
	tmp.id = planes.markers.size();
	tmp.action = visualization_msgs::Marker::ADD;
	tmp.pose = lm_info.origin;
	tmp.pose.position.x += q_bias.x();
	tmp.pose.position.y += q_bias.y();
	tmp.pose.position.z += q_bias.z();
	tmp.type = visualization_msgs::Marker::CUBE;
	tmp.scale.x = thickness;
	tmp.scale.y = width + 0.5;
	tmp.scale.z = height + 0.5;
	if(lm_info.was_observed_in_this_scan){
		tmp.color.r = 1.0;
		tmp.color.g = 0.0;
		tmp.color.b = 0.0;
		tmp.color.a = 0.9;
	}
	else if(lm_info.was_erased){
		tmp.color.r = 1.0;
		tmp.color.g = 1.0;
		tmp.color.b = 1.0;
		tmp.color.a = 0.9;
	}
	else if(lm_info.was_merged){
		tmp.color.r = 1.0;
		tmp.color.g = 1.0;
		tmp.color.b = 0.0;
		tmp.color.a = 0.5;
	}
	else if(lm_info.available){
		tmp.color.r = 0.0;
		tmp.color.g = 1.0;
		tmp.color.b = 0.0;
		tmp.color.a = 0.9;
	}
	else{
		tmp.color.r = 0.0;
		tmp.color.g = 0.0;
		tmp.color.b = 1.0;
		tmp.color.a = 0.9;
	}
	if(fabs(lm_info.origin.orientation.x)>0.5 || fabs(lm_info.origin.orientation.y)>0.5){	//floor
		/* tmp.color.r = 0.5; */
		/* tmp.color.g = 0.5; */
		/* tmp.color.b = 0.5; */
		tmp.color.a = 0.2;
	}

	planes.markers.push_back(tmp);
}

void WallEKFSLAM::EraseLM(int index)
{
	/*keep*/
	list_erased_lm_info.push_back(list_lm_info[index]);
	/*list*/
	list_lm_info.erase(list_lm_info.begin() + index);
	/*delmit point*/
	int delimit_point = size_robot_state + index*size_wall_state;
	int delimit_point_ = size_robot_state + (index+1)*size_wall_state;
	/*X*/
	Eigen::VectorXd tmp_X = X;
	X.resize(X.size() - size_wall_state);
	X.segment(0, delimit_point) = tmp_X.segment(0, delimit_point);
	X.segment(delimit_point, X.size() - delimit_point) = tmp_X.segment(delimit_point_, tmp_X.size() - delimit_point_);
	/*P*/
	Eigen::MatrixXd tmp_P = P;
	P.resize(P.cols() - size_wall_state, P.rows() - size_wall_state);
	P.block(0, 0, delimit_point, delimit_point) = tmp_P.block(0, 0, delimit_point, delimit_point);
	P.block(0, delimit_point, delimit_point, P.cols()-delimit_point) = tmp_P.block(0, delimit_point_, delimit_point_, P.cols()-delimit_point_);
	P.block(delimit_point, 0, P.rows()-delimit_point, delimit_point) = tmp_P.block(delimit_point_, 0, P.rows()-delimit_point_, delimit_point_);
	P.block(delimit_point, delimit_point, P.rows()-delimit_point, P.cols()-delimit_point) = tmp_P.block(delimit_point_, delimit_point_, P.rows()-delimit_point_, P.cols()-delimit_point_);
}

Eigen::Vector3d WallEKFSLAM::PlaneLocalToGlobal(const Eigen::Vector3d& Nl)
{
	Eigen::Vector3d rotL = GetRotationXYZMatrix(X.segment(3, 3), false)*Nl;
	Eigen::Vector3d DeltaVertical = X.segment(0, 3).dot(rotL)/rotL.dot(rotL)*rotL;
	Eigen::Vector3d Ng = rotL + DeltaVertical;
	return Ng;
}

Eigen::Vector3d WallEKFSLAM::PointLocalToGlobal(const Eigen::Vector3d& Pl)
{
	Eigen::Vector3d Pg = GetRotationXYZMatrix(X.segment(3, 3), false)*Pl + X.segment(0, 3);
	return Pg;
}

Eigen::Vector3d WallEKFSLAM::PointGlobalToWallFrame(const Eigen::Vector3d& Pg, geometry_msgs::Pose origin)
{
	/*linear & rotation*/
	tf::Quaternion q_pg(
		Pg(0) - origin.position.x,
		Pg(1) - origin.position.y,
		Pg(2) - origin.position.z,
		0.0
	);
	tf::Quaternion q_origin_orientation;
	quaternionMsgToTF(origin.orientation, q_origin_orientation);
	tf::Quaternion q_pl = q_origin_orientation.inverse()*q_pg*q_origin_orientation;

	Eigen::Vector3d Pl = Eigen::Vector3d(q_pl.x(), q_pl.y(), q_pl.z());

	return Pl;
}

Eigen::Vector3d WallEKFSLAM::PointWallFrameToGlobal(const Eigen::Vector3d& Pl, geometry_msgs::Pose origin)
{
	/*rotation*/
	tf::Quaternion q_pl(
		Pl(0),
		Pl(1),
		Pl(2),
		0.0
	);
	tf::Quaternion q_origin_orientation;
	quaternionMsgToTF(origin.orientation, q_origin_orientation);
	tf::Quaternion q_pg = q_origin_orientation*q_pl*q_origin_orientation.inverse();
	/*linear*/
	Eigen::Vector3d Pg(
		q_pg.x() + origin.position.x,
		q_pg.y() + origin.position.y,
		q_pg.z() + origin.position.z
	);

	return Pg;
}

void WallEKFSLAM::Publication(void)
{
	/* std::cout << "Publication" << std::endl; */

	for(int i=3;i<6;i++){	//test
		if(fabs(X(i))>M_PI){
			std::cout << "+PI -PI error" << std::endl;
			std::cout << "X(" << i << ") = " << X(i) << std::endl;
			exit(1);
		}
	}
	for(size_t i=0;i<X.size();i++){	//test
		if(std::isnan(X(i))){
			std::cout << "NAN error" << std::endl;
			std::cout << "X(" << i << ") = " << X(i) << std::endl;
			exit(1);
		}
	}

	/*pose*/
	geometry_msgs::PoseStamped pose_pub = StateVectorToPoseStamped();
	pose_pub.header.frame_id = "/odom";
	// pose_pub.header.stamp = ros::Time::now();
	pose_pub.header.stamp = time_imu_now;
	pub_pose.publish(pose_pub);

	/*tf broadcast*/
    geometry_msgs::TransformStamped transform;
	transform.header.stamp = pose_pub.header.stamp;
	transform.header.frame_id = "/odom";
	transform.child_frame_id = "/velodyne";
	transform.transform.translation.x = pose_pub.pose.position.x;
	transform.transform.translation.y = pose_pub.pose.position.y;
	transform.transform.translation.z = pose_pub.pose.position.z;
	transform.transform.rotation = pose_pub.pose.orientation;
	tf_broadcaster.sendTransform(transform);

	/*pc*/
	sensor_msgs::PointCloud2 pc_pub;
	pcl::toROSMsg(StateVectorToPC(), pc_pub);
	pc_pub.header.frame_id = "/odom";
	pc_pub.header.stamp = time_imu_now;
	pub_dgaussiansphere_est.publish(pc_pub);

	/*visualization marker*/
	matching_lines.header.stamp = time_imu_now;
	// pub_marker.publish(matching_lines);

	/*walls*/
	geometry_msgs::PoseArray wall_origins;
	wall_origins.header.frame_id = "/odom";
	wall_origins.header.stamp = time_imu_now;
	/* for(size_t i=0;i<list_lm_info.size();i++)	if(list_lm_info[i].available)	wall_origins.poses.push_back(list_lm_info[i].origin); */
	for(size_t i=0;i<list_lm_info.size();i++){
		wall_origins.poses.push_back(list_lm_info[i].origin);
	}
	pub_posearray.publish(wall_origins);
	pub_markerarray.publish(planes);

	/*variance*/
	std_msgs::Float64MultiArray variance_pub;
	for(int i=0;i<P.cols();i++)	variance_pub.data.push_back(P(i, i));
	pub_variance.publish(variance_pub);
}

geometry_msgs::PoseStamped WallEKFSLAM::StateVectorToPoseStamped(void)
{
	geometry_msgs::PoseStamped pose;
	pose.pose.position.x = X(0);
	pose.pose.position.y = X(1);
	pose.pose.position.z = X(2);
	tf::Quaternion q_orientation = tf::createQuaternionFromRPY(X(3), X(4), X(5));
	pose.pose.orientation.x = q_orientation.x();
	pose.pose.orientation.y = q_orientation.y();
	pose.pose.orientation.z = q_orientation.z();
	pose.pose.orientation.w = q_orientation.w();

	return pose;
}

pcl::PointCloud<pcl::PointXYZ> WallEKFSLAM::StateVectorToPC(void)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr pc (new pcl::PointCloud<pcl::PointXYZ>);
	int num_wall = (X.size() - size_robot_state)/size_wall_state;
	for(int i=0;i<num_wall;i++){
		if(list_lm_info[i].available){
			pcl::PointXYZ tmp;
			tmp.x = X(size_robot_state + i*size_wall_state);
			tmp.y = X(size_robot_state + i*size_wall_state + 1);
			tmp.z = X(size_robot_state + i*size_wall_state + 2);
			pc->points.push_back(tmp);
		}
	}
	return *pc;
}

Eigen::Matrix3d WallEKFSLAM::GetRotationXYZMatrix(const Eigen::Vector3d& RPY, bool inverse)
{
	Eigen::Matrix3d Rot_xyz;
	Rot_xyz <<
		cos(RPY(1))*cos(RPY(2)),	sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) - cos(RPY(0))*sin(RPY(2)),	cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)),
		cos(RPY(1))*sin(RPY(2)),	sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) + cos(RPY(0))*cos(RPY(2)),	cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) - sin(RPY(0))*cos(RPY(2)),
		-sin(RPY(1)),				sin(RPY(0))*cos(RPY(1)),										cos(RPY(0))*cos(RPY(1));
	
	Eigen::Matrix3d Rot_xyz_inv;
	Rot_xyz_inv <<
		cos(RPY(1))*cos(RPY(2)),										cos(RPY(1))*sin(RPY(2)),										-sin(RPY(1)),
		sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) - cos(RPY(0))*sin(RPY(2)),	sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) + cos(RPY(0))*cos(RPY(2)),	sin(RPY(0))*cos(RPY(1)),
		cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)),	cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) - sin(RPY(0))*cos(RPY(2)),	cos(RPY(0))*cos(RPY(1));

	if(!inverse)	return Rot_xyz;
	else	return Rot_xyz_inv;	//=Rot_xyz.transpose()
}

void WallEKFSLAM::VectorVStack(Eigen::VectorXd& A, const Eigen::VectorXd& B)
{
	A.conservativeResize(A.rows() + B.rows());
	A.segment(A.size() - B.size(), B.size()) = B;
}

void WallEKFSLAM::MatrixVStack(Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
	A.conservativeResize(A.rows() + B.rows(), B.cols());
	A.block(A.rows() - B.rows(), 0, B.rows(), B.cols()) = B;
}

geometry_msgs::Quaternion WallEKFSLAM::QuatEigenToMsg(Eigen::Quaterniond q_eigen)
{
	geometry_msgs::Quaternion q_msg;
	q_msg.x = q_eigen.x();
	q_msg.y = q_eigen.y();
	q_msg.z = q_eigen.z();
	q_msg.w = q_eigen.w();
	return q_msg;
}

double WallEKFSLAM::PiToPi(double angle)
{
	/* return fmod(angle + M_PI, 2*M_PI) - M_PI; */
	return atan2(sin(angle), cos(angle)); 
}

WallEKFSLAM::RemoveUnavailableLM::RemoveUnavailableLM(const Eigen::VectorXd& X, const Eigen::MatrixXd& P, int size_robot_state, int size_wall_state, std::vector<LMInfo> list_lm_info)
{
	X_ = X;
	P_ = P;
	size_robot_state_ = size_robot_state;
	size_wall_state_ = size_wall_state;
	list_lm_info_ = list_lm_info;
}
void WallEKFSLAM::RemoveUnavailableLM::InputAvailableLMIndex(int lm_index)
{
	available_lm_indices.push_back(lm_index);
}
void WallEKFSLAM::RemoveUnavailableLM::InputUnavailableLMIndex(int lm_index)
{
	unavailable_lm_indices.push_back(lm_index);
}
void WallEKFSLAM::RemoveUnavailableLM::Remove(Eigen::VectorXd& X, Eigen::MatrixXd& P, std::vector<LMInfo>& list_lm_info)
{
	if(unavailable_lm_indices.size()!=0){
		// std::cout << "X = " << std::endl << X << std::endl;
		// std::cout << "P = " << std::endl << P << std::endl;
		// for(size_t i=0;i<unavailable_lm_indices.size();i++)	std::cout << "unavailable_lm_indices[i] = " << unavailable_lm_indices[i]  << std::endl;
		X = X_;
		P = P_;
		for(size_t i=0;i<unavailable_lm_indices.size();i++){
			/*delmit point*/
			int index = unavailable_lm_indices[i] - i;
			int delimit_point = size_robot_state_ + index*size_wall_state_;
			int delimit_point_ = size_robot_state_ + (index+1)*size_wall_state_;
			/*X*/
			Eigen::VectorXd tmp_X = X;
			X.resize(X.size() - size_wall_state_);
			X.segment(0, delimit_point) = tmp_X.segment(0, delimit_point);
			X.segment(delimit_point, X.size() - delimit_point) = tmp_X.segment(delimit_point_, tmp_X.size() - delimit_point_);
			/*P*/
			Eigen::MatrixXd tmp_P = P;
			P.resize(P.cols() - size_wall_state_, P.rows() - size_wall_state_);
			P.block(0, 0, delimit_point, delimit_point) = tmp_P.block(0, 0, delimit_point, delimit_point);
			P.block(0, delimit_point, delimit_point, P.cols()-delimit_point) = tmp_P.block(0, delimit_point_, delimit_point_, P.cols()-delimit_point_);
			P.block(delimit_point, 0, P.rows()-delimit_point, delimit_point) = tmp_P.block(delimit_point_, 0, P.rows()-delimit_point_, delimit_point_);
			P.block(delimit_point, delimit_point, P.rows()-delimit_point, P.cols()-delimit_point) = tmp_P.block(delimit_point_, delimit_point_, P.rows()-delimit_point_, P.cols()-delimit_point_);
			/*LM list*/
			list_lm_info.erase(list_lm_info.begin() + index);
		}
		// std::cout << "removed" << std::endl;
		// std::cout << "X = " << std::endl << X << std::endl;
		// std::cout << "P = " << std::endl << P << std::endl;
	}
}
void WallEKFSLAM::RemoveUnavailableLM::Recover(Eigen::VectorXd& X, Eigen::MatrixXd& P, std::vector<LMInfo>& list_lm_info)
{
	if(unavailable_lm_indices.size()!=0){
		std::cout << "X = " << std::endl << X << std::endl;
		std::cout << "P = " << std::endl << P << std::endl;
		for(size_t i=0;i<available_lm_indices.size();i++)	std::cout << "available_lm_indices[i] = " << available_lm_indices[i]  << std::endl;

		Eigen::VectorXd tmp_X = X_;
		Eigen::MatrixXd tmp_P = P_;

		/*robot*/
		tmp_X.segment(0, size_robot_state_) = X.segment(0, size_robot_state_);
		tmp_P.block(0, 0, size_robot_state_, size_robot_state_) = P.block(0, 0, size_robot_state_, size_robot_state_);
		/*LM*/
		for(size_t i=0;i<available_lm_indices.size();i++){
			tmp_X.segment(size_robot_state_ + available_lm_indices[i]*size_wall_state_, size_wall_state_) = X.segment(size_robot_state_ + i*size_wall_state_, size_wall_state_);
			for(size_t j=0;j<available_lm_indices.size();j++){
				tmp_P.block(size_robot_state_ + available_lm_indices[i]*size_wall_state_, size_robot_state_ + available_lm_indices[j]*size_wall_state_, size_wall_state_, size_wall_state_) = P.block(size_robot_state_ + i*size_wall_state_, size_robot_state_ + j*size_wall_state_, size_wall_state_, size_wall_state_);
				// tmp_P(available_lm_indices[i], available_lm_indices[j]) = P(i, j);
			}
		}

		X = tmp_X;
		P = tmp_P;
		std::cout << "recovered" << std::endl;
		std::cout << "X = " << std::endl << X << std::endl;
		std::cout << "P = " << std::endl << P << std::endl;
		for(size_t i=0;i<unavailable_lm_indices.size();i++){
			int index = unavailable_lm_indices[i];
			list_lm_info.insert(list_lm_info.begin() + index, list_lm_info_[index]);
		}
	}
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "wall_ekf_slam");
	std::cout << "E.K.F. POSE" << std::endl;
	
	WallEKFSLAM wall_ekf_slam;
	ros::spin();
}
