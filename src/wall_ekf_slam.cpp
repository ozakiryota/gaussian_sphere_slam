#include <ros/ros.h>
#include <geometry_msgs/Quaternion.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
// #include <Eigen/Core>
// #include <Eigen/LU>

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
		ros::Publisher pub_dgaussiansphere_est;
		ros::Publisher pub_marker;
		/*const*/
		const int size_robot_state = 6;	//X, Y, Z, R, P, Y (Global)
		const int size_wall_state = 3;	//x, y, z (Local)
		/*struct*/
		struct WallInfo{
			bool is_inward;
			int count_match;
			double observable_area[3][2];	//[x, y, z][min, max]
			double unobservable_area[3][2];	//[x, y, z][min, max]
			bool available;
		};
		/*objects*/
		Eigen::VectorXd X;
		Eigen::MatrixXd P;
		sensor_msgs::Imu bias;
		pcl::PointCloud<pcl::InterestPoint>::Ptr d_gaussian_sphere {new pcl::PointCloud<pcl::InterestPoint>};
		std::vector<WallInfo> list_wall_info;
		/*flags*/
		bool inipose_is_available = false;
		bool bias_is_available = false;
		bool first_callback_imu = true;
		bool first_callback_odom = true;
		/*time*/
		ros::Time time_imu_now;
		ros::Time time_imu_last;
		ros::Time time_odom_now;
		ros::Time time_odom_last;
		/*visualization*/
		visualization_msgs::Marker matching_lines;
	public:
		WallEKFSLAM();
		void SetUpVisualizationMarker(visualization_msgs::Marker& marker);	//visualization
		void CallbackInipose(const geometry_msgs::QuaternionConstPtr& msg);
		void CallbackBias(const sensor_msgs::ImuConstPtr& msg);
		void CallbackIMU(const sensor_msgs::ImuConstPtr& msg);
		void PredictionIMU(sensor_msgs::Imu imu, double dt);
		void CallbackOdom(const nav_msgs::OdometryConstPtr& msg);
		void PredictionOdom(nav_msgs::Odometry odom, double dt);
		void CallbackDGaussianSphere(const sensor_msgs::PointCloud2ConstPtr &msg);
		int SearchCorrespondWallID(const Eigen::VectorXd& Zi, Eigen::VectorXd& Hi, Eigen::MatrixXd& jHi, Eigen::VectorXd& Yi, Eigen::MatrixXd& Si);
		void PushBackWallInfo(const Eigen::Vector3d& Ng);
		bool CheckNormalIsInward(const Eigen::Vector3d& Ng);
		void JudgeWallsCanBeObserbed(void);
		void ObservationUpdate(const Eigen::VectorXd& Z, const Eigen::VectorXd& H, const Eigen::MatrixXd& jH);
		Eigen::Vector3d PlaneGlobalToLocal(const Eigen::Vector3d& Ng);
		Eigen::Vector3d PlaneLocalToGlobal(const Eigen::Vector3d& Nl);
		void PushBackMatchingLine(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2);	//visualization
		void Publication();
		geometry_msgs::PoseStamped StateVectorToPoseStamped(void);
		pcl::PointCloud<pcl::PointXYZ> StateVectorToPC(void);
		Eigen::Matrix3d GetRotationXYZMatrix(const Eigen::Vector3d& RPY, bool inverse);
		void VectorVStack(Eigen::VectorXd& A, const Eigen::VectorXd& B);
		void MatrixVStack(Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
		double PiToPi(double angle);
};

WallEKFSLAM::WallEKFSLAM()
{
	sub_inipose = nh.subscribe("/initial_orientation", 1, &WallEKFSLAM::CallbackInipose, this);
	sub_bias = nh.subscribe("/imu/bias", 1, &WallEKFSLAM::CallbackBias, this);
	sub_imu = nh.subscribe("/imu/data", 1, &WallEKFSLAM::CallbackIMU, this);
	sub_odom = nh.subscribe("/tinypower/odom", 1, &WallEKFSLAM::CallbackOdom, this);
	sub_dgaussiansphere_obs = nh.subscribe("/d_gaussian_sphere_obs", 1, &WallEKFSLAM::CallbackDGaussianSphere, this);
	pub_pose = nh.advertise<geometry_msgs::PoseStamped>("/pose_wall_ekf_slam", 1);
	pub_dgaussiansphere_est = nh.advertise<sensor_msgs::PointCloud2>("/d_gaussian_sphere_est", 1);
	pub_marker = nh.advertise<visualization_msgs::Marker>("matching_lines", 1);
	X = Eigen::VectorXd::Zero(size_robot_state);
	P = Eigen::MatrixXd::Identity(size_robot_state, size_robot_state);
	SetUpVisualizationMarker(matching_lines);
}

void WallEKFSLAM::SetUpVisualizationMarker(visualization_msgs::Marker& marker)
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

	/* std::cout << "X =" << std::endl << X << std::endl; */
	/* std::cout << "P =" << std::endl << P << std::endl; */
	/* std::cout << "jF =" << std::endl << jF << std::endl; */
}

void WallEKFSLAM::CallbackOdom(const nav_msgs::OdometryConstPtr& msg)
{
	/* std::cout << "Callback Odom" << std::endl; */

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
	else	PredictionOdom(*msg, dt);
	
	Publication();

	first_callback_odom = false;
}

void WallEKFSLAM::PredictionOdom(nav_msgs::Odometry odom, double dt)
{
	/* std::cout << "Prediction Odom" << std::endl; */

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
}

void WallEKFSLAM::CallbackDGaussianSphere(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	std::cout << "Callback D-Gaussian Sphere" << std::endl;
	std::cout << "num_wall = " << (X.size() - size_robot_state)/size_wall_state << std::endl;
	
	pcl::fromROSMsg(*msg, *d_gaussian_sphere);
	std::cout << "d_gaussian_sphere->points.size() = " << d_gaussian_sphere->points.size() << std::endl;
	for(size_t i=0;i<d_gaussian_sphere->points.size();i++)	std::cout << "d_gaussian_sphere->points[" << i << "].strength = " << d_gaussian_sphere->points[i].strength << std::endl;
	Eigen::VectorXd Xnew(0);

	Eigen::VectorXd Zstacked(0);
	Eigen::VectorXd Hstacked(0);
	Eigen::MatrixXd jHstacked(0, 0);

	matching_lines.points.clear();

	JudgeWallsCanBeObserbed();

	for(size_t i=0;i<d_gaussian_sphere->points.size();i++){
		Eigen::Vector3d Zi(
			d_gaussian_sphere->points[i].x,
			d_gaussian_sphere->points[i].y,
			d_gaussian_sphere->points[i].z
		);
		/* std::cout << "Zi =" << std::endl << Zi << std::endl; */
		Eigen::VectorXd Hi;
		Eigen::MatrixXd jHi;
		Eigen::VectorXd Yi;
		Eigen::MatrixXd Si;
		int correspond_id = SearchCorrespondWallID(Zi, Hi, jHi, Yi, Si);
		/* correspond_id = -1;	//test */
		if(correspond_id==-1){
			// Xnew.conservativeResize(Xnew.size() + size_wall_state);
			// Xnew.segment(Xnew.size() - size_wall_state, size_wall_state) = PlaneLocalToGlobal(Zi);
			VectorVStack(Xnew, PlaneLocalToGlobal(Zi));
			PushBackWallInfo(PlaneLocalToGlobal(Zi));
		}
		else{
			// PushBackMatchingLine(X.segment(size_robot_state + correspond_id*size_wall_state, size_wall_state), GetRotationXYZMatrix(X.segment(3, 3), false)*Zi + X.segment(0, 3));

			// std::cout << "P =" << std::endl << P << std::endl;
			// std::cout << "jHi =" << std::endl << jHi << std::endl;
			// std::cout << "Si =" << std::endl << Si << std::endl;
			// std::cout << "Si.inverse() =" << std::endl << Si.inverse() << std::endl;

			// Eigen::MatrixXd Ki = P*jHi.transpose()*Si.inverse();
			// X = X + Ki*Yi;
			// for(int i=3;i<6;i++)	X(i) = PiToPi(X(i));
			// Eigen::MatrixXd I = Eigen::MatrixXd::Identity(X.size(), X.size());
			// P = (I - Ki*jHi)*P;

			list_wall_info[correspond_id].count_match += 1;
			const int threshold_count_match = 3;
			if(list_wall_info[correspond_id].count_match>threshold_count_match)	list_wall_info[correspond_id].available = true;
			else	list_wall_info[correspond_id].available = false;

			if(list_wall_info[correspond_id].available){
				PushBackMatchingLine(X.segment(size_robot_state + correspond_id*size_wall_state, size_wall_state), GetRotationXYZMatrix(X.segment(3, 3), false)*Zi + X.segment(0, 3));
				VectorVStack(Zstacked, Zi);
				VectorVStack(Hstacked, Hi);
				MatrixVStack(jHstacked, jHi);
			}

			// std::cout << "correspond_id =" << correspond_id << std::endl;
			// std::cout << "Yi =" << std::endl << Yi << std::endl;
			// std::cout << "Ki*Yi =" << std::endl << Ki*Yi << std::endl;
		}
	}
	/*update*/
	if(Zstacked.size()>0)	ObservationUpdate(Zstacked, Hstacked, jHstacked);
	/*Registration of new walls*/
	X.conservativeResize(X.size() + Xnew.size());
	X.segment(X.size() - Xnew.size(), Xnew.size()) = Xnew;
	Eigen::MatrixXd Ptmp = P;
	P = Eigen::MatrixXd::Identity(X.size(), X.size());
	P.block(0, 0, Ptmp.rows(), Ptmp.cols()) = Ptmp;

	Publication();
}

int WallEKFSLAM::SearchCorrespondWallID(const Eigen::VectorXd& Zi, Eigen::VectorXd& Hi, Eigen::MatrixXd& jHi, Eigen::VectorXd& Yi, Eigen::MatrixXd& Si)
{
	/* std::cout << "Zi = " << Zi << std::endl; */
	int num_wall = (X.size() - size_robot_state)/size_wall_state;

	const double threshold_mahalanobis_dist = 0.36;	//chi-square distribution
	double min_mahalanobis_dist = threshold_mahalanobis_dist;
	// const double threshold_euclidean_dist = 0.15;	//test
	const double threshold_euclidean_dist = 0.2;	//test
	double min_euclidean_dist = threshold_euclidean_dist;	//test
	int correspond_id = -1;
	for(int i=0;i<num_wall;i++){
		if(list_wall_info[i].available){
			Eigen::Vector3d Ng = X.segment(size_robot_state + i*size_wall_state, size_wall_state);
			Eigen::Vector3d RPY = X.segment(3, 3);
			double d2 = Ng.dot(Ng);
			/*H*/
			Eigen::Vector3d H = PlaneGlobalToLocal(Ng);
			/*jH*/
			Eigen::MatrixXd jH = Eigen::MatrixXd::Zero(Zi.size(), X.size());
			/*dH/d(XYZ)*/
			Eigen::Vector3d rotN = GetRotationXYZMatrix(RPY, true)*Ng;
			for(int j=0;j<Zi.size();j++){
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
			for(int j=0;j<Zi.size();j++){
				for(int k=0;k<size_wall_state;k++){
					if(j==k)	Tmp(j, k) = 1 - ((Ng.dot(X.segment(0, 3)) + Ng(j)*X(k))/d2 - Ng(j)*Ng.dot(X.segment(0, 3))/(d2*d2)*2*Ng(k));
					else	Tmp(j, k) = -(Ng(j)*X(k)/d2 - Ng(j)*Ng.dot(X.segment(0, 3))/(d2*d2)*2*Ng(k));
				}
			}
			jH.block(0, size_robot_state + i*size_wall_state, Zi.size(), size_wall_state) = GetRotationXYZMatrix(RPY, true)*Tmp;
			/*R*/
			const double sigma = 1.0e-2;
			Eigen::MatrixXd R = sigma*Eigen::MatrixXd::Identity(Zi.size(), Zi.size());
			/*Y, S*/
			Eigen::VectorXd Y = Zi - H;
			Eigen::MatrixXd S = jH*P*jH.transpose() + R;

			double mahalanobis_dist = Y.transpose()*S.inverse()*Y;
			double euclidean_dist = Y.norm();	//test
			/* std::cout << "mahalanobis_dist = " << mahalanobis_dist << std::endl; */
			/* std::cout << "H = " << H << std::endl; */
			// std::cout << "euclidean_dist = " << euclidean_dist << std::endl;
			if(std::isnan(mahalanobis_dist)){	//test
				/* std::cout << "mahalanobis_dist is NAN" << std::endl; */
				/* std::cout << "X =" << std::endl << X << std::endl; */
				/* std::cout << "P =" << std::endl << P << std::endl; */
				/* std::cout << "jH =" << std::endl << jH << std::endl; */
				/* std::cout << "S =" << std::endl << S << std::endl; */
				/* std::cout << "S.inverse() =" << std::endl << S.inverse() << std::endl; */
				/* std::cout << "Y =" << std::endl << Y << std::endl; */
				// exit(1);
			}
			if(euclidean_dist<min_euclidean_dist && list_wall_info[i].is_inward==CheckNormalIsInward(X.segment(size_robot_state+i*size_wall_state, size_wall_state))){	//test
				min_euclidean_dist = euclidean_dist;
				correspond_id = i;
				Hi = H;
				jHi = jH;
				Yi = Y;
				Si = S;
			}
			// if(!std::isnan(mahalanobis_dist) && mahalanobis_dist<min_mahalanobis_dist){
			// 	min_mahalanobis_dist = mahalanobis_dist;
			// 	correspond_id = i;
			// 	jHi = jH;
			// 	Yi = Y;
			// 	Si = S;
			// }
			/* std::cout << "jH =" << std::endl << jH << std::endl; */
		}
	}

	return correspond_id;
}

void WallEKFSLAM::PushBackWallInfo(const Eigen::Vector3d& Ng)
{
	WallInfo tmp;
	tmp.is_inward = CheckNormalIsInward(Ng);
	tmp.count_match = 0;
	list_wall_info.push_back(tmp);
}

bool WallEKFSLAM::CheckNormalIsInward(const Eigen::Vector3d& Ng)
{
	double dist_wall = Ng.norm();
	Eigen::Vector3d VerticalPosition = X.segment(0, 3).dot(Ng)/Ng.dot(Ng)*Ng;
	double dist_robot = VerticalPosition.norm();
	if(dist_robot<dist_wall)	return true;
	else	return false;
}

void WallEKFSLAM::JudgeWallsCanBeObserbed(void)
{
	int num_wall = (X.size() - size_robot_state)/size_wall_state;
	for(int i=0;i<num_wall;i++){
		Eigen::Vector3d Ng = X.segment(size_robot_state+i*size_wall_state, size_wall_state);
		if(list_wall_info[i].is_inward!=CheckNormalIsInward(Ng))	list_wall_info[i].available = false;
		else	list_wall_info[i].available = true;
	}
}

void WallEKFSLAM::ObservationUpdate(const Eigen::VectorXd& Z, const Eigen::VectorXd& H, const Eigen::MatrixXd& jH)
{
	Eigen::VectorXd Y = Z - H;
	const double sigma = 1.0e-1;
	Eigen::MatrixXd R = sigma*Eigen::MatrixXd::Identity(Z.size(), Z.size());
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

Eigen::Vector3d WallEKFSLAM::PlaneLocalToGlobal(const Eigen::Vector3d& Nl)
{
	Eigen::Vector3d rotL = GetRotationXYZMatrix(X.segment(3, 3), false)*Nl;
	Eigen::Vector3d DeltaVertical = X.segment(0, 3).dot(rotL)/rotL.dot(rotL)*rotL;
	Eigen::Vector3d Ng = rotL + DeltaVertical;
	return Ng;
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
	pub_marker.publish(matching_lines);
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
		if(list_wall_info[i].available){
			pcl::PointXYZ tmp;
			tmp.x = X(size_robot_state + i*size_wall_state);
			tmp.y = X(size_robot_state + i*size_wall_state + 1);
			tmp.z = X(size_robot_state + i*size_wall_state + 2);
			pc->points.push_back(tmp);
		}
	}
	return *pc;
}

void WallEKFSLAM::PushBackMatchingLine(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2)
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

double WallEKFSLAM::PiToPi(double angle)
{
	/* return fmod(angle + M_PI, 2*M_PI) - M_PI; */
	return atan2(sin(angle), cos(angle)); 
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "wall_ekf_slam");
	std::cout << "E.K.F. POSE" << std::endl;
	
	WallEKFSLAM wall_ekf_slam;
	ros::spin();
}
