#include <ros/ros.h>
#include <geometry_msgs/Quaternion.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
/* #include <std_msgs/Float64MultiArray.h> */
#include <pcl_conversions/pcl_conversions.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
/* #include <pcl/common/transforms.h> */
/* #include <pcl/kdtree/kdtree_flann.h> */
/* #include <pcl/visualization/cloud_viewer.h> */
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
		ros::Subscriber sub_dgaussiansphere_obs;
		/*publish*/
		tf::TransformBroadcaster tf_broadcaster;
		ros::Publisher pub_pose;
		ros::Publisher pub_dgaussiansphere_est;
		/*const*/
		const int size_robot_state = 6;	//X, Y, Z, R, P, Y (Global)
		const int size_wall_state = 3;	//x, y, z (Local)
		/*objects*/
		Eigen::VectorXd X;
		Eigen::MatrixXd P;
		sensor_msgs::Imu bias;
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
	public:
		WallEKFSLAM();
		void CallbackInipose(const geometry_msgs::QuaternionConstPtr& msg);
		void CallbackBias(const sensor_msgs::ImuConstPtr& msg);
		void CallbackIMU(const sensor_msgs::ImuConstPtr& msg);
		void PredictionIMU(sensor_msgs::Imu imu, double dt);
		void CallbackOdom(const nav_msgs::OdometryConstPtr& msg);
		void PredictionOdom(nav_msgs::Odometry odom, double dt);
		void CallbackDGaussianSphere(const sensor_msgs::PointCloud2ConstPtr &msg);
		int SearchCorrespondWallID(Eigen::VectorXd Zi, Eigen::MatrixXd& jHi, Eigen::VectorXd& Yi, Eigen::MatrixXd& Si);
		Eigen::Vector3d PlaneGlobalToLocal(Eigen::Vector3d G);
		Eigen::Vector3d PlaneLocalToGlobal(Eigen::Vector3d L);
		Eigen::Matrix3d GetRotationXYZMatrix(Eigen::Vector3d RPY, bool inverse);
		void Publication();
		geometry_msgs::PoseStamped StateVectorToPoseStamped(void);
		pcl::PointCloud<pcl::PointXYZ> StateVectorToPC(void);
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
	X = Eigen::VectorXd::Zero(size_robot_state);
	P = Eigen::MatrixXd::Identity(size_robot_state, size_robot_state);

	/*test*/
	/* Eigen::VectorXd XYZ(9); */
	/* XYZ << 0, 2, 0, 2, 0, 0, 0, 0 ,2; */
	/* X.conservativeResize(X.size() + XYZ.size()); */
	/* X.segment(X.size() - XYZ.size(), XYZ.size()) = XYZ; */
	/* P = Eigen::MatrixXd::Identity(X.size(), X.size()); */
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
	else if(bias_is_available)	PredictionIMU(*msg, dt);
	
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
	F.segment(size_robot_state, num_wall*size_wall_state) = X.segment(size_robot_state, num_wall*size_wall_state);

	/*jF*/
	Eigen::MatrixXd jF(X.size(), X.size());
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
	const double sigma = 1.0e-1;
	Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(X.size(), X.size());
	
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
	Eigen::MatrixXd jF(X.size(), X.size());
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
	const double sigma = 1.0e-1;
	Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(X.size(), X.size());
	
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
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr d_gaussian_sphere {new pcl::PointCloud<pcl::PointXYZ>};
	pcl::fromROSMsg(*msg, *d_gaussian_sphere);
	std::cout << "d_gaussian_sphere->points.size() = " << d_gaussian_sphere->points.size() << std::endl;
	Eigen::VectorXd Xnew(0);
	for(size_t i=0;i<d_gaussian_sphere->points.size();i++){
		Eigen::Vector3d Zi(
			d_gaussian_sphere->points[i].x,
			d_gaussian_sphere->points[i].y,
			d_gaussian_sphere->points[i].z
		);
		Eigen::MatrixXd jHi;
		Eigen::VectorXd Yi;
		Eigen::MatrixXd Si;
		int correspond_id = SearchCorrespondWallID(Zi, jHi, Yi, Si);
		if(correspond_id==-1){
			Xnew.conservativeResize(Xnew.size() + size_wall_state);
			Xnew.segment(Xnew.size() - size_wall_state, size_wall_state) = PlaneLocalToGlobal(Zi);
		}
		else{
			Eigen::MatrixXd Ki = P*jHi.transpose()*Si.inverse();
			X = X + Ki*Yi;
			Eigen::MatrixXd I = Eigen::MatrixXd::Identity(X.size(), X.size());
			P = (I - Ki*jHi)*P;

			std::cout << "correspond_id =" << correspond_id << std::endl;
			/* std::cout << "Yi =" << std::endl << Yi << std::endl; */
			/* std::cout << "Ki*Yi =" << std::endl << Ki*Yi << std::endl; */
		}
	}
	X.conservativeResize(X.size() + Xnew.size());
	X.segment(X.size() - Xnew.size(), Xnew.size()) = Xnew;
	Eigen::MatrixXd Ptmp = P;
	P = Eigen::MatrixXd::Identity(X.size(), X.size());
	P.block(0, 0, Ptmp.rows(), Ptmp.cols()) = Ptmp;

	Publication();
}

int WallEKFSLAM::SearchCorrespondWallID(Eigen::VectorXd Zi, Eigen::MatrixXd& jHi, Eigen::VectorXd& Yi, Eigen::MatrixXd& Si)
{
	int num_wall = (X.size() - size_robot_state)/size_wall_state;

	const double threshold_mahalanobis = 2.0;
	double min_mahalanobis = threshold_mahalanobis;
	int correspond_id = -1;
	for(int i=0;i<num_wall;i++){
		Eigen::Vector3d N = X.segment(size_robot_state + i*size_wall_state, size_wall_state);
		Eigen::Vector3d RPY = X.segment(3, 3);
		double d2 = N.dot(N);
		/*H*/
		Eigen::Vector3d H = PlaneLocalToGlobal(Zi);
		/*jH*/
		Eigen::MatrixXd jH = Eigen::MatrixXd::Zero(Zi.size(), X.size());
		/*dH/d(XYZ)*/
		Eigen::Vector3d rotN = GetRotationXYZMatrix(RPY, true)*N;
		for(int j=0;j<Zi.size();j++){
			for(int k=0;k<3;k++)	jH(j, k) = -N(k)/d2*rotN(j);
		}
		/*dH/d(RPY)*/
cos(RPY(1))*cos(RPY(2)),										cos(RPY(1))*sin(RPY(2)),										-sin(RPY(1)),
sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) - cos(RPY(0))*sin(RPY(2)),	sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) + cos(RPY(0))*cos(RPY(2)),	sin(RPY(0))*cos(RPY(1)),
cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)),	cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) - sin(RPY(0))*cos(RPY(2)),	cos(RPY(0))*cos(RPY(1));

		Eigen::Vector3d delN = N - N.dot(X.segment(0, 3))/d2*N;
		jH(0, 3) = 0;
		jH(0, 4) = (-sin(RPY(1))*cos(RPY(2)))*delN(0) + (-sin(RPY(1))*sin(RPY(2)))*delN(1) + (-cos(RPY(1)))*delN(2);
		jH(0, 5) = (-cos(RPY(1))*sin(RPY(2)))*delN(0) + (cos(RPY(1))*cos(RPY(2)))*delN(1);
		jH(1, 3) = (cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)))*delN(0) + (cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) - sin(RPY(0))*cos(RPY(2)))*delN(1) + (cos(RPY(0))*cos(RPY(1)))*delN(2);
		jH(1, 4) = (sin(RPY(0))*cos(RPY(1))*cos(RPY(2)))*delN(0) + (sin(RPY(0))*cos(RPY(1))*sin(RPY(2)))*delN(1) - (sin(RPY(0))*sin(RPY(1)))*delN(2);
		jH(1, 5) = (-sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) - cos(RPY(0))*cos(RPY(2)))*delN(0) + (sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) - cos(RPY(0))*sin(RPY(2)))*delN(1);
		jH(2, 3) = (-sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) + cos(RPY(0))*sin(RPY(2)))*delN(0) + (-sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) - cos(RPY(0))*cos(RPY(2)))*delN(1) + (-sin(RPY(0))*cos(RPY(1)))*delN(2);
		jH(2, 4) = (cos(RPY(0))*cos(RPY(1))*cos(RPY(2)))*delN(0) + (cos(RPY(0))*cos(RPY(1))*sin(RPY(2)))*delN(1) + (-cos(RPY(0))*sin(RPY(1)))*delN(2);
		jH(2, 5) = (-cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) + sin(RPY(0))*cos(RPY(2)))*delN(0) + (cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)))*delN(1);
		/*dH/d(Wall)*/
		Eigen::Matrix3d Tmp;
		for(int j=0;j<Zi.size();j++){
			for(int k=0;k<size_wall_state;k++){
				if(j==k)	Tmp(j, k) = 1 - ((N.dot(X.segment(0, 3)) + N(j)*X(j))/d2 - N(j)*N.dot(X.segment(0, 3))/(d2*d2)*2*N(k));
				else	Tmp(j, k) = -(N(j)*X(k)/d2 - N(j)*N.dot(X.segment(0, 3))/(d2*d2)*2*N(k));
			}
		}
		jH.block(0, size_robot_state + i*size_wall_state, Zi.size(), size_wall_state) = GetRotationXYZMatrix(RPY, true)*Tmp;
		/*R*/
		const double sigma = 1.0e-1;
		Eigen::MatrixXd R = sigma*Eigen::MatrixXd::Identity(Zi.size(), Zi.size());
		/*Y, S*/
		Eigen::VectorXd Y = Zi - H;
		Eigen::MatrixXd S = jH*P*jH.transpose() + R;

		double mahalanobis_dist = Y.transpose()*S.inverse()*Y;
		std::cout << "mahalanobis_dist =" << mahalanobis_dist << std::endl;
		if(mahalanobis_dist<min_mahalanobis){
			correspond_id = i;
			jHi = jH;
			Yi = Y;
			Si = S;
		}
		/* std::cout << "jH =" << std::endl << jH << std::endl; */
	}

	return correspond_id;
}

Eigen::Vector3d WallEKFSLAM::PlaneGlobalToLocal(Eigen::Vector3d G)
{
	Eigen::Vector3d DeltaVertical = X.segment(0, 3).dot(G)/G.dot(G)*G;
	Eigen::Vector3d delL = G - DeltaVertical;
	Eigen::Vector3d L = GetRotationXYZMatrix(X.segment(3, 3), true)*delL;
	return L;
}

Eigen::Vector3d WallEKFSLAM::PlaneLocalToGlobal(Eigen::Vector3d L)
{
	Eigen::Vector3d rotL = GetRotationXYZMatrix(X.segment(3, 3), false)*L;
	Eigen::Vector3d DeltaVertical = X.segment(0, 3).dot(rotL)/rotL.dot(rotL)*rotL;
	Eigen::Vector3d G = rotL + DeltaVertical;
	return G;
}

Eigen::Matrix3d WallEKFSLAM::GetRotationXYZMatrix(Eigen::Vector3d RPY, bool inverse)
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

void WallEKFSLAM::Publication(void)
{
	/* std::cout << "Publication" << std::endl; */

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
		pcl::PointXYZ tmp;
		tmp.x = X(size_robot_state + i*size_wall_state);
		tmp.y = X(size_robot_state + i*size_wall_state + 1);
		tmp.z = X(size_robot_state + i*size_wall_state + 2);
		pc->points.push_back(tmp);
	}
	return *pc;
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
