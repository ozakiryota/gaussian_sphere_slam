#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <omp.h>

class DGaussianSphere{
	private:
		/*node handle*/
		ros::NodeHandle nh;
		ros::NodeHandle nhPrivate;
		/*subscribe*/
		ros::Subscriber sub_pc;
		/*publish*/
		ros::Publisher pub_pc;
		/*pcl*/
		pcl::visualization::PCLVisualizer viewer {"Normal Estimation Multi Thread"};
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointNormal>::Ptr normals {new pcl::PointCloud<pcl::PointNormal>};
		pcl::PointCloud<pcl::PointNormal>::Ptr normals_extracted {new pcl::PointCloud<pcl::PointNormal>};
		/*objects*/
		Eigen::Vector3f Gvector{0.0, 0.0, -1.0};	//tmp
		/*parameters*/
		int skip;
		double search_radius_ratio;
		bool mode_remove_ground;
	public:
		DGaussianSphere();
		void CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg);
		void Computation(void);
		double Getdepth(pcl::PointXYZ point);
		std::vector<int> KdtreeSearch(pcl::PointXYZ searchpoint, double search_radius);
		bool JudgeFlatness(const Eigen::Vector4f& plane_parameters, std::vector<int> indices);
		double AngleBetweenVectors(const Eigen::Vector3f& V1, const Eigen::Vector3f& V2);
		double ComputeFittingError(const Eigen::Vector4f& N, std::vector<int> indices);
		void Visualization(void);
		void Publication(void);
};

DGaussianSphere::DGaussianSphere()
	:nhPrivate("~")
{
	sub_pc = nh.subscribe("/velodyne_points", 1, &DGaussianSphere::CallbackPC, this);
	pub_pc = nh.advertise<sensor_msgs::PointCloud2>("/normals", 1);
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addCoordinateSystem(0.8, "axis");
	viewer.setCameraPosition(-30.0, 0.0, 10.0, 0.0, 0.0, 1.0);

	nhPrivate.param("skip", skip, 3);
	nhPrivate.param("search_radius_ratio", search_radius_ratio, 0.09);
	nhPrivate.param("mode_remove_ground", mode_remove_ground, false);
	std::cout << "skip = " << skip << std::endl;
	std::cout << "search_radius_ratio = " << search_radius_ratio << std::endl;
	std::cout << "mode_remove_ground = " << (bool)mode_remove_ground << std::endl;
}

void DGaussianSphere::CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	/* std::cout << "CALLBACK PC" << std::endl; */

	pcl::fromROSMsg(*msg, *cloud);
	std::cout << "cloud->points.size() = " << cloud->points.size() << std::endl;
	normals->points.clear();
	normals->points.resize((cloud->points.size()-1)/skip + 1);

	kdtree.setInputCloud(cloud);
	Computation();

	Publication();
	Visualization();
}

void DGaussianSphere::Computation(void)
{
	std::cout << "omp_get_max_threads() = " << omp_get_max_threads() << std::endl;

	double time_start = ros::Time::now().toSec();

	std::vector<bool> extract_indices((cloud->points.size()-1)/skip + 1, false);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for(size_t i=0;i<cloud->points.size();i+=skip){
		size_t normal_index = i/skip;
		/*search neighbor points*/
		double laser_distance = Getdepth(cloud->points[i]);
		double search_radius = search_radius_ratio*laser_distance;
		std::vector<int> indices = KdtreeSearch(cloud->points[i], search_radius);
		/*compute normal*/
		float curvature;
		Eigen::Vector4f plane_parameters;
		pcl::computePointNormal(*cloud, indices, plane_parameters, curvature);
		/*judge*/
		extract_indices[normal_index] = JudgeFlatness(plane_parameters, indices);
		/*input*/
		normals->points[normal_index].x = cloud->points[i].x;
		normals->points[normal_index].y = cloud->points[i].y;
		normals->points[normal_index].z = cloud->points[i].z;
		normals->points[normal_index].normal_x = plane_parameters[0];
		normals->points[normal_index].normal_y = plane_parameters[1];
		normals->points[normal_index].normal_z = plane_parameters[2];
		normals->points[normal_index].curvature = curvature;
		flipNormalTowardsViewpoint(cloud->points[i], 0.0, 0.0, 0.0, normals->points[normal_index].normal_x, normals->points[normal_index].normal_y, normals->points[normal_index].normal_z);
	}
	for(size_t i=0;i<normals->points.size();){
		if(!extract_indices[i]){
			std::cout << "remove unsused normal" << std::endl;
			normals->points.erase(normals->points.begin() + i);
			extract_indices.erase(extract_indices.begin() + i);
		}
		else	i++;
	}

	std::cout << "computation time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
}

double DGaussianSphere::Getdepth(pcl::PointXYZ point)
{
	double depth = sqrt(
		point.x*point.x
		+ point.y*point.y
		+ point.z*point.z
	);
	return depth;
}

std::vector<int> DGaussianSphere::KdtreeSearch(pcl::PointXYZ searchpoint, double search_radius)
{
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
	if(kdtree.radiusSearch(searchpoint, search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance)<=0)	std::cout << "kdtree error" << std::endl;
	return pointIdxRadiusSearch; 
}

bool DGaussianSphere::JudgeFlatness(const Eigen::Vector4f& plane_parameters, std::vector<int> indices)
{
	/*threshold*/
	const size_t threshold_num_neighborpoints = 10;
	const double threshold_angle = 30.0;	//[deg]
	const double threshold_fitting_error = 0.01;	//[m]

	/*number of neighbor-points*/
	if(indices.size() < threshold_num_neighborpoints)	return false;
	/*nan*/
	if(std::isnan(plane_parameters(0)) || std::isnan(plane_parameters(1)) || std::isnan(plane_parameters(2)))	return false;
	/*angle*/
	if(mode_remove_ground){
		if(fabs(AngleBetweenVectors(plane_parameters.segment(0, 3), Gvector)-M_PI/2.0)>threshold_angle/180.0*M_PI)	return false;
	}
	/*fitting error*/
	if(ComputeFittingError(plane_parameters, indices) > threshold_fitting_error)	return false;
	/*pass*/
	return true;
}

double DGaussianSphere::AngleBetweenVectors(const Eigen::Vector3f& V1, const Eigen::Vector3f& V2)
{
	double angle = acos(V1.dot(V2)/V1.norm()/V2.norm());
	return angle;
}

double DGaussianSphere::ComputeFittingError(const Eigen::Vector4f& N, std::vector<int> indices)
{
	double ave_fitting_error = 0.0;
	for(size_t i=0;i<indices.size();++i){
		Eigen::Vector3f P(
			cloud->points[indices[i]].x,
			cloud->points[indices[i]].y,
			cloud->points[indices[i]].z
		);
		ave_fitting_error += fabs(N.segment(0, 3).dot(P) + N(3))/N.segment(0, 3).norm();
	}
	ave_fitting_error /= (double)indices.size();
	return ave_fitting_error;
}

void DGaussianSphere::Visualization(void)
{
	viewer.removeAllPointClouds();

	viewer.addPointCloud(cloud, "cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
	
	viewer.addPointCloudNormals<pcl::PointNormal>(normals, 1, 0.5, "normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "normals");

	viewer.spinOnce();
}

void DGaussianSphere::Publication(void)
{
	normals->header.stamp = cloud->header.stamp;
	normals->header.frame_id = cloud->header.frame_id;

	sensor_msgs::PointCloud2 pc;
	pcl::toROSMsg(*normals, pc);
	pub_pc.publish(pc);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "normal_estimation_multi_thread");
	std::cout << "Normal Estimation Multi Thread" << std::endl;
	
	DGaussianSphere normal_estimation_multi_thread;

	ros::spin();
}
