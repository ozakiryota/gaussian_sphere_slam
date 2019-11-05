#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
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
		ros::Publisher pub_gausspc;
		ros::Publisher pub_pc;
		ros::Publisher pub_nc;
		/*pcl*/
		pcl::visualization::PCLVisualizer viewer {"Normal Estimation Multi Thread"};
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointNormal>::Ptr normals {new pcl::PointCloud<pcl::PointNormal>};
		pcl::PointCloud<pcl::PointNormal>::Ptr normals_extracted {new pcl::PointCloud<pcl::PointNormal>};
		pcl::PointCloud<pcl::PointXYZ>::Ptr d_gaussian_sphere {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::InterestPoint>::Ptr d_gaussian_sphere_clustered {new pcl::PointCloud<pcl::InterestPoint>};
		pcl::PointCloud<pcl::PointNormal>::Ptr d_gaussian_sphere_clustered_n {new pcl::PointCloud<pcl::PointNormal>};
		/*objects*/
		Eigen::Vector3f Gvector{0.0, 0.0, -1.0};	//tmp
		/*parameters*/
		int skip;
		double search_radius_ratio;
		bool mode_remove_ground;
		bool mode_open_viewer;
		bool mode_clustering;
		bool mode_decimate_points;
		int decimated_size;
		double cluster_distance;
		int min_num_cluster_belongings;
		/*time counter*/
	 	double avg_computation_time = 0.0;
		int counter = 0;
	public:
		DGaussianSphere();
		void CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg);
		void ClearPC(void);
		void Computation(void);
		double Getdepth(pcl::PointXYZ point);
		std::vector<int> KdtreeSearch(pcl::PointXYZ searchpoint, double search_radius);
		bool JudgeForSelecting(const Eigen::Vector4f& plane_parameters, std::vector<int> indices);
		double AngleBetweenVectors(const Eigen::Vector3f& V1, const Eigen::Vector3f& V2);
		double ComputeFittingError(const Eigen::Vector4f& N, std::vector<int> indices);
		void DecimatePC(void);
		void ClusterDGauss(void);
		void Visualization(void);
		void Publication(void);
};

DGaussianSphere::DGaussianSphere()
	:nhPrivate("~")
{
	sub_pc = nh.subscribe("/velodyne_points", 1, &DGaussianSphere::CallbackPC, this);
	pub_gausspc = nh.advertise<sensor_msgs::PointCloud2>("/d_gaussian_sphere", 1);
	pub_pc = nh.advertise<sensor_msgs::PointCloud2>("/d_gaussian_sphere_obs", 1);
	pub_nc = nh.advertise<sensor_msgs::PointCloud2>("/normals", 1);
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addCoordinateSystem(0.8, "axis");
	// viewer.setCameraPosition(-30.0, 0.0, 10.0, 0.0, 0.0, 1.0);
	viewer.setCameraPosition(0.0, 0.0, 35.0, 0.0, 0.0, 0.0);

	nhPrivate.param("skip", skip, 3);
	nhPrivate.param("search_radius_ratio", search_radius_ratio, 0.09);
	nhPrivate.param("mode_remove_ground", mode_remove_ground, false);
	nhPrivate.param("mode_open_viewer", mode_open_viewer, true);
	nhPrivate.param("mode_clustering", mode_clustering, true);
	nhPrivate.param("mode_decimate_points", mode_decimate_points, true);
	nhPrivate.param("decimated_size", decimated_size, 800);
	nhPrivate.param("cluster_distance", cluster_distance, 0.1);
	nhPrivate.param("min_num_cluster_belongings", min_num_cluster_belongings, 30);
	std::cout << "skip = " << skip << std::endl;
	std::cout << "search_radius_ratio = " << search_radius_ratio << std::endl;
	std::cout << "mode_remove_ground = " << (bool)mode_remove_ground << std::endl;
	std::cout << "mode_open_viewer = " << (bool)mode_open_viewer << std::endl;
	std::cout << "mode_clustering = " << (bool)mode_clustering << std::endl;
	std::cout << "mode_decimate_points = " << (bool)mode_decimate_points << std::endl;
	std::cout << "decimated_size = " << decimated_size << std::endl;
	std::cout << "cluster_distance = " << cluster_distance << std::endl;
	std::cout << "min_num_cluster_belongings = " << min_num_cluster_belongings << std::endl;

	if(!mode_open_viewer)	viewer.close();
}

void DGaussianSphere::CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	/* std::cout << "CALLBACK PC" << std::endl; */

	double time_start = ros::Time::now().toSec();

	pcl::fromROSMsg(*msg, *cloud);
	std::cout << "==========" << std::endl;
	std::cout << "cloud->points.size() = " << cloud->points.size() << std::endl;
	ClearPC();

	kdtree.setInputCloud(cloud);
	Computation();
	if(mode_decimate_points && d_gaussian_sphere->points.size()>decimated_size)	DecimatePC();
	if(mode_clustering)	ClusterDGauss();

	double tmp_time = ros::Time::now().toSec() - time_start;
	counter++;
	avg_computation_time = (avg_computation_time*(counter - 1) + tmp_time)/counter;
	std::cout << "Feature extraction: avg_computation_time [s] = " << avg_computation_time << std::endl;

	Publication();
	if(mode_open_viewer)	Visualization();
}

void DGaussianSphere::ClearPC(void)
{
	normals->points.clear();
	normals_extracted->points.clear();
	d_gaussian_sphere->points.clear();
	d_gaussian_sphere_clustered->points.clear();
	d_gaussian_sphere_clustered_n->points.clear();
}

void DGaussianSphere::Computation(void)
{
	std::cout << "omp_get_max_threads() = " << omp_get_max_threads() << std::endl;

	double time_start = ros::Time::now().toSec();

	const double min_search_radius = 0.2;
	normals->points.resize((cloud->points.size()-1)/skip + 1);
	std::vector<bool> extract_indices((cloud->points.size()-1)/skip + 1, false);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for(size_t i=0;i<cloud->points.size();i+=skip){
		size_t normal_index = i/skip;
		/*search neighbor points*/
		double laser_distance = Getdepth(cloud->points[i]);
		double search_radius = search_radius_ratio*laser_distance;
		if(search_radius<min_search_radius)	search_radius = min_search_radius;
		std::vector<int> indices = KdtreeSearch(cloud->points[i], search_radius);
		/*compute normal*/
		float curvature;
		Eigen::Vector4f plane_parameters;
		pcl::computePointNormal(*cloud, indices, plane_parameters, curvature);
		/*judge*/
		extract_indices[normal_index] = JudgeForSelecting(plane_parameters, indices);
		/*input*/
		normals->points[normal_index].x = cloud->points[i].x;
		normals->points[normal_index].y = cloud->points[i].y;
		normals->points[normal_index].z = cloud->points[i].z;
		normals->points[normal_index].data_n[0] = plane_parameters(0);
		normals->points[normal_index].data_n[1] = plane_parameters(1);
		normals->points[normal_index].data_n[2] = plane_parameters(2);
		normals->points[normal_index].data_n[3] = plane_parameters(3);
		/* normals->points[normal_index].normal_x = plane_parameters[0]; */
		/* normals->points[normal_index].normal_y = plane_parameters[1]; */
		/* normals->points[normal_index].normal_z = plane_parameters[2]; */
		normals->points[normal_index].curvature = curvature;
		flipNormalTowardsViewpoint(cloud->points[i], 0.0, 0.0, 0.0, normals->points[normal_index].normal_x, normals->points[normal_index].normal_y, normals->points[normal_index].normal_z);
	}
	for(size_t i=0;i<normals->points.size();i++){
		if(extract_indices[i]){
			/*extracted normals*/
			normals_extracted->points.push_back(normals->points[i]);
			/*d-gaussian shpere*/
			pcl::PointXYZ tmp;
			tmp.x = -fabs(normals->points[i].data_n[3])*normals->points[i].data_n[0];
			tmp.y = -fabs(normals->points[i].data_n[3])*normals->points[i].data_n[1];
			tmp.z = -fabs(normals->points[i].data_n[3])*normals->points[i].data_n[2];
			d_gaussian_sphere->points.push_back(tmp);
			/* std::cout << "remove unsused normal" << std::endl; */
			/* normals->points.erase(normals->points.begin() + i); */
			/* extract_indices.erase(extract_indices.begin() + i); */
		}
	}

	std::cout << "normal estimation time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
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

bool DGaussianSphere::JudgeForSelecting(const Eigen::Vector4f& plane_parameters, std::vector<int> indices)
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

void DGaussianSphere::DecimatePC(void)
{
	std::cout << "d_gaussian_sphere->points.size() : " << d_gaussian_sphere->points.size() << " -> " << decimated_size << std::endl;

	double sparse_step = d_gaussian_sphere->points.size()/(double)decimated_size;
	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_pc (new pcl::PointCloud<pcl::PointXYZ>);
	tmp_pc->points.resize(decimated_size);

	for(int i=0;i<decimated_size;++i){
		int original_index = i*sparse_step;
		tmp_pc->points[i] = d_gaussian_sphere->points[original_index];
	}
	d_gaussian_sphere = tmp_pc;
}

void DGaussianSphere::ClusterDGauss(void)
{
	double time_start = ros::Time::now().toSec();

	/* const double cluster_distance = 0.1; */
	/* const int min_num_cluster_belongings = 30;	//indoor */
	/* const int min_num_cluster_belongings = 40;	//outside */
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(d_gaussian_sphere);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
	ece.setClusterTolerance(cluster_distance);
	ece.setMinClusterSize(min_num_cluster_belongings);
	ece.setMaxClusterSize(d_gaussian_sphere->points.size());
	ece.setSearchMethod(tree);
	ece.setInputCloud(d_gaussian_sphere);
	ece.extract(cluster_indices);

	// std::cout << "cluster_indices.size() = " << cluster_indices.size() << std::endl;

	pcl::ExtractIndices<pcl::PointXYZ> ei;
	ei.setInputCloud(d_gaussian_sphere);
	ei.setNegative(false);
	for(size_t i=0;i<cluster_indices.size();i++){
		/*extract*/
		pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_clustered_points (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointIndices::Ptr tmp_clustered_indices (new pcl::PointIndices);
		*tmp_clustered_indices = cluster_indices[i];
		ei.setIndices(tmp_clustered_indices);
		ei.filter(*tmp_clustered_points);
		/*compute centroid*/
		Eigen::Vector4f xyz_centroid;
		pcl::compute3DCentroid(*tmp_clustered_points, xyz_centroid);
		/*input*/
		pcl::InterestPoint tmp_centroid;
		tmp_centroid.x = xyz_centroid[0];
		tmp_centroid.y = xyz_centroid[1];
		tmp_centroid.z = xyz_centroid[2];
		tmp_centroid.strength = tmp_clustered_indices->indices.size();
		d_gaussian_sphere_clustered->points.push_back(tmp_centroid);
		/*for Visualization*/
		pcl::PointNormal tmp_centroid_n;
		tmp_centroid_n.x = 0.0;
		tmp_centroid_n.y = 0.0;
		tmp_centroid_n.z = 0.0;
		tmp_centroid_n.normal_x = xyz_centroid[0];
		tmp_centroid_n.normal_y = xyz_centroid[1];
		tmp_centroid_n.normal_z = xyz_centroid[2];
		d_gaussian_sphere_clustered_n->points.push_back(tmp_centroid_n);
	}

	std::cout << "clustering time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
	std::cout << "d_gaussian_sphere_clustered->points.size() = " << d_gaussian_sphere_clustered->points.size() << std::endl;
}

void DGaussianSphere::Visualization(void)
{
	viewer.removeAllPointClouds();

	/*cloud*/
	viewer.addPointCloud(cloud, "cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
	/*normals*/
	viewer.addPointCloudNormals<pcl::PointNormal>(normals, 1, 0.5, "normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "normals");
	/*extracted normals*/
	viewer.addPointCloudNormals<pcl::PointNormal>(normals_extracted, 1, 0.5, "normals_extracted");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 1.0, "normals_extracted");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "normals_extracted");
	/*d-gaussian sphere*/
	viewer.addPointCloud(d_gaussian_sphere, "d_gaussian_sphere");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.8, "d_gaussian_sphere");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "d_gaussian_sphere");
	/*d-gaussian sphere n*/
	viewer.addPointCloudNormals<pcl::PointNormal>(d_gaussian_sphere_clustered_n, 1, 1.0, "d_gaussian_sphere_clustered_n");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "d_gaussian_sphere_clustered_n");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "d_gaussian_sphere_clustered_n");
	
	viewer.spinOnce();
}

void DGaussianSphere::Publication(void)
{
	/*gausspc*/
	d_gaussian_sphere->header.stamp = cloud->header.stamp;
	d_gaussian_sphere->header.frame_id = cloud->header.frame_id;
	sensor_msgs::PointCloud2 gausspc_pub;
	pcl::toROSMsg(*d_gaussian_sphere, gausspc_pub);
	pub_gausspc.publish(gausspc_pub);
	/*pc*/
	d_gaussian_sphere_clustered->header.stamp = cloud->header.stamp;
	d_gaussian_sphere_clustered->header.frame_id = cloud->header.frame_id;
	sensor_msgs::PointCloud2 pc_pub;
	pcl::toROSMsg(*d_gaussian_sphere_clustered, pc_pub);
	pub_pc.publish(pc_pub);
	/*nc*/
	normals->header.stamp = cloud->header.stamp;
	normals->header.frame_id = cloud->header.frame_id;
	sensor_msgs::PointCloud2 nc_pub;
	pcl::toROSMsg(*normals, nc_pub);
	pub_nc.publish(nc_pub);	
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "normal_estimation_multi_thread");
	std::cout << "Normal Estimation Multi Thread" << std::endl;
	
	DGaussianSphere normal_estimation_multi_thread;

	ros::spin();
}
