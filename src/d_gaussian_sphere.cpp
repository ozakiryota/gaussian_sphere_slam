#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <tf/tf.h>
#include <thread>
// #include <omp.h>

class DGaussianSphere{
	private:
		/*node handle*/
		ros::NodeHandle nh;
		/*subscribe*/
		ros::Subscriber sub_pc;
		/*publish*/
		ros::Publisher pub_pc;
		/*pcl*/
		pcl::visualization::PCLVisualizer viewer{"D-Gaussian Spheres"};
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		pcl::PointNormal g_vector;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointNormal>::Ptr normals {new pcl::PointCloud<pcl::PointNormal>};
		pcl::PointCloud<pcl::PointNormal>::Ptr normals_extracted {new pcl::PointCloud<pcl::PointNormal>};
		pcl::PointCloud<pcl::PointXYZ>::Ptr d_gaussian_sphere {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointXYZ>::Ptr d_gaussian_sphere_clustered {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointNormal>::Ptr d_gaussian_sphere_clustered_n {new pcl::PointCloud<pcl::PointNormal>};
		/*objects*/
		ros::Time time_pub;
		/*flags*/
		const bool mode_depth_is_ignored = false;
		const bool mode_floor_is_used = true;
	public:
		DGaussianSphere();
		void CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg);
		void ClearPoints(void);
		std::vector<int> KdtreeSearch(pcl::PointXYZ searchpoint, double search_radius);
		void MultiThreadComputation(void);
		void DecimatePoints(size_t limited_num_points);
		double AngleBetweenVectors(pcl::PointNormal v1, pcl::PointNormal v2);
		double ComputeSquareError(Eigen::Vector4f plane_parameters, std::vector<int> indices);
		void ClusterDGauss(void);
		void Visualization(void);
		void Publication(void);
	protected:
		class FittingWalls_{
			private:
				pcl::PointCloud<pcl::PointNormal>::Ptr normals_ {new pcl::PointCloud<pcl::PointNormal>};
				pcl::PointCloud<pcl::PointNormal>::Ptr normals_extracted_ {new pcl::PointCloud<pcl::PointNormal>};
				pcl::PointCloud<pcl::PointXYZ>::Ptr gaussian_sphere_ {new pcl::PointCloud<pcl::PointXYZ>};
				pcl::PointCloud<pcl::PointXYZ>::Ptr d_gaussian_sphere_ {new pcl::PointCloud<pcl::PointXYZ>};
			public:
				void Compute(DGaussianSphere &mainclass, size_t i_start, size_t i_end);
				void Merge(DGaussianSphere &mainclass);
		};
};

DGaussianSphere::DGaussianSphere()
{
	sub_pc = nh.subscribe("/velodyne_points", 1, &DGaussianSphere::CallbackPC, this);
	pub_pc = nh.advertise<sensor_msgs::PointCloud2>("/d_gaussian_sphere_obs", 1);
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addCoordinateSystem(0.8, "axis");
	// viewer.setCameraPosition(0.0, 0.0, 50.0, 0.0, 0.0, 0.0);
	viewer.setCameraPosition(-30.0, 0.0, 10.0, 0.0, 0.0, 1.0);

	/*test*/
	g_vector.x = 0.0;
	g_vector.y = 0.0;
	g_vector.z = 0.0;
	g_vector.normal_x = 0.0;
	g_vector.normal_y = 0.0;
	g_vector.normal_z = -1.0;
}

void DGaussianSphere::CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	std::cout << "--------------------------" << std::endl;
	std::cout << "CALLBACK PC" << std::endl;
	double t_start_callback_pc = ros::Time::now().toSec();

	pcl::fromROSMsg(*msg, *cloud);
	time_pub = msg->header.stamp;
	ClearPoints();
	kdtree.setInputCloud(cloud);
	MultiThreadComputation();
	
	std::cout << "d_gaussian_sphere->points.size() = " << d_gaussian_sphere->points.size() << std::endl;
	const size_t limited_num_points = 800;
	if(d_gaussian_sphere->points.size()>limited_num_points)	DecimatePoints(limited_num_points);
	ClusterDGauss();
	Publication();
	Visualization();
	
	std::cout << "time for CallbackPC[s] = " << ros::Time::now().toSec() - t_start_callback_pc << std::endl;
}

void DGaussianSphere::ClearPoints(void)
{
	// std::cout << "CLEAR POINTS" << std::endl;
	normals->points.clear();
	normals_extracted->points.clear();
	d_gaussian_sphere->points.clear();
	d_gaussian_sphere_clustered->points.clear();
	d_gaussian_sphere_clustered_n->points.clear();
}

void DGaussianSphere::MultiThreadComputation(void)
{
	const int num_threads = std::thread::hardware_concurrency();
	std::vector<std::thread> threads_fittingwalls;
	std::vector<FittingWalls_> objects;
	for(int i=0;i<num_threads;i++){
		FittingWalls_ tmp_object;
		objects.push_back(tmp_object);
	}
	double t_start_normal_est = ros::Time::now().toSec();
	for(int i=0;i<num_threads;i++){
		threads_fittingwalls.push_back(
			std::thread([i, num_threads, &objects, this]{
				objects[i].Compute(*this, i*cloud->points.size()/num_threads, (i+1)*cloud->points.size()/num_threads);
			})
		);
	}
	for(std::thread &th : threads_fittingwalls)	th.join();
	for(int i=0;i<num_threads;i++)	objects[i].Merge(*this);
	std::cout << "time for normal estimation[s] = " << ros::Time::now().toSec() - t_start_normal_est << std::endl;
}

void DGaussianSphere::DecimatePoints(size_t limited_num_points)
{
	/*simple*/
	double sparse_step = d_gaussian_sphere->points.size()/(double)limited_num_points;
	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud {new pcl::PointCloud<pcl::PointXYZ>};
	for(double a=0.0;a<d_gaussian_sphere->points.size();a+=sparse_step)	tmp_cloud->points.push_back(d_gaussian_sphere->points[a]);
	d_gaussian_sphere = tmp_cloud;

	// #<{(|multi threads|)}>#
	// double sparse_step = d_gaussian_sphere->points.size()/(double)limited_num_points;
	// class DecimatingPoints{
	// 	private:
	// 	public:
	// 		pcl::PointCloud<pcl::PointXYZ>::Ptr pc_decimated {new pcl::PointCloud<pcl::PointXYZ>};
	// 		void PushBack(pcl::PointCloud<pcl::PointXYZ> pc, int start, int end, double step){
	// 			for(double k=start;k<end;k+=step)	pc_decimated->points.push_back(pc.points[k]);
	// 		};
	// };
	// std::vector<DecimatingPoints> decimating_objects;
	// for(int i=0;i<num_threads;i++){
	// 	DecimatingPoints tmp_decimating_object;
	// 	decimating_objects.push_back(tmp_decimating_object);
	// }
	// std::vector<std::thread> decimating_threads;
	// for(int i=0;i<num_threads;i++){
	// 	decimating_threads.push_back(
	// 		std::thread([i, num_threads, &decimating_objects, this, sparse_step]{
	// 			decimating_objects[i].PushBack(*d_gaussian_sphere, i*d_gaussian_sphere->points.size()/num_threads, (i+1)*d_gaussian_sphere->points.size()/num_threads, sparse_step);
	// 		})
	// 	);
	// }
	// for(std::thread &th : decimating_threads)	th.join();
	// d_gaussian_sphere->points.clear();
	// for(int i=0;i<num_threads;i++)	*d_gaussian_sphere += *decimating_objects[i].pc_decimated;

	std::cout << "-> d_gaussian_sphere->points.size() = " << d_gaussian_sphere->points.size() << std::endl;
}

void DGaussianSphere::FittingWalls_::Compute(DGaussianSphere &mainclass, size_t i_start, size_t i_end)
{
	// std::cout << "NORMAL ESTIMATION" << std::endl;

	const size_t skip_step = 3;
	for(size_t i=i_start;i<i_end;i+=skip_step){
		/*search neighbor points*/
		std::vector<int> indices;
		double laser_distance = sqrt(mainclass.cloud->points[i].x*mainclass.cloud->points[i].x + mainclass.cloud->points[i].y*mainclass.cloud->points[i].y + mainclass.cloud->points[i].z*mainclass.cloud->points[i].z);
		const double search_radius_min = 0.3;
		const double ratio = 0.09;
		double search_radius = ratio*laser_distance;
		if(search_radius<search_radius_min)	search_radius = search_radius_min;
		indices = mainclass.KdtreeSearch(mainclass.cloud->points[i], search_radius);
		/*compute normal*/
		float curvature;
		Eigen::Vector4f plane_parameters;
		pcl::computePointNormal(*mainclass.cloud, indices, plane_parameters, curvature);
		/*create tmp object*/
		pcl::PointNormal tmp_normal;
		tmp_normal.x = mainclass.cloud->points[i].x;
		tmp_normal.y = mainclass.cloud->points[i].y;
		tmp_normal.z = mainclass.cloud->points[i].z;
		tmp_normal.normal_x = plane_parameters[0];
		tmp_normal.normal_y = plane_parameters[1];
		tmp_normal.normal_z = plane_parameters[2];
		tmp_normal.curvature = curvature;
		flipNormalTowardsViewpoint(tmp_normal, 0.0, 0.0, 0.0, tmp_normal.normal_x, tmp_normal.normal_y, tmp_normal.normal_z);
		normals_->points.push_back(tmp_normal);
		/*judge num*/
		// const size_t threshold_num_neighborpoints_dgauss = 5;
		// const size_t threshold_num_neighborpoints_dgauss = 20;
		const size_t threshold_num_neighborpoints_dgauss = 10;
		if(indices.size()<threshold_num_neighborpoints_dgauss)	continue;
		/*delete nan*/
		if(std::isnan(plane_parameters[0]) || std::isnan(plane_parameters[1]) || std::isnan(plane_parameters[2]))	continue;
		/*judge angle*/
		if(!mainclass.mode_floor_is_used){
			const double threshold_angle = 30.0;	//[deg]
			if(fabs(mainclass.AngleBetweenVectors(tmp_normal, mainclass.g_vector)-M_PI/2.0)>threshold_angle/180.0*M_PI)	continue;
		}
		/*judge fitting error*/
		const double threshold_fitting_error = 0.01;	//[m]
		if(mainclass.ComputeSquareError(plane_parameters, indices)>threshold_fitting_error)	continue;
		/*input*/
		normals_extracted_->points.push_back(tmp_normal);
		pcl::PointXYZ tmp_point;
		tmp_point.x = -plane_parameters[3]*plane_parameters[0];
		tmp_point.y = -plane_parameters[3]*plane_parameters[1];
		tmp_point.z = -plane_parameters[3]*plane_parameters[2];
		if(mainclass.mode_depth_is_ignored){
			tmp_point.x = -tmp_normal.normal_x;	//test
			tmp_point.y = -tmp_normal.normal_y;	//test
			tmp_point.z = -tmp_normal.normal_z;	//test
		}
		d_gaussian_sphere_->points.push_back(tmp_point);
	}
}

void DGaussianSphere::FittingWalls_::Merge(DGaussianSphere &mainclass)
{
	*mainclass.normals += *normals_;
	*mainclass.normals_extracted += *normals_extracted_;
	*mainclass.d_gaussian_sphere += *d_gaussian_sphere_;
}

std::vector<int> DGaussianSphere::KdtreeSearch(pcl::PointXYZ searchpoint, double search_radius)
{
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
	if(kdtree.radiusSearch(searchpoint, search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance)<=0)	std::cout << "kdtree error" << std::endl;
	return pointIdxRadiusSearch; 
}

double DGaussianSphere::AngleBetweenVectors(pcl::PointNormal v1, pcl::PointNormal v2)
{
	double dot_product = v1.normal_x*v2.normal_x + v1.normal_y*v2.normal_y + v1.normal_z*v2.normal_z;
	double v1_norm = sqrt(v1.normal_x*v1.normal_x + v1.normal_y*v1.normal_y + v1.normal_z*v1.normal_z);
	double v2_norm = sqrt(v2.normal_x*v2.normal_x + v2.normal_y*v2.normal_y + v2.normal_z*v2.normal_z);
	double angle = acos(dot_product/(v1_norm*v2_norm));
	return angle;
}

double DGaussianSphere::ComputeSquareError(Eigen::Vector4f plane_parameters, std::vector<int> indices)
{
	double sum_square_error = 0.0;
	for(size_t i=0;i<indices.size();i++){
		sum_square_error +=		fabs(plane_parameters[0]*cloud->points[indices[i]].x
									+plane_parameters[1]*cloud->points[indices[i]].y
									+plane_parameters[2]*cloud->points[indices[i]].z
									+plane_parameters[3])
								/sqrt(plane_parameters[0]*plane_parameters[0]
									+plane_parameters[1]*plane_parameters[1]
									+plane_parameters[2]*plane_parameters[2])
								/(double)indices.size();
	}
	return sum_square_error;
}

void DGaussianSphere::ClusterDGauss(void)
{
	// std::cout << "POINT CLUSTER" << std::endl;

	// const double cluster_distance = 0.1;
	const double cluster_distance = 0.1;
	// const int min_num_cluster_belongings = 20;
	const int min_num_cluster_belongings = 30;
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
		pcl::PointXYZ tmp_centroid;
		tmp_centroid.x = xyz_centroid[0];
		tmp_centroid.y = xyz_centroid[1];
		tmp_centroid.z = xyz_centroid[2];
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
}

void DGaussianSphere::Visualization(void)
{
	viewer.removeAllPointClouds();

	viewer.addPointCloud(cloud, "cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
	
	viewer.addPointCloudNormals<pcl::PointNormal>(normals, 1, 0.5, "normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "normals");

	viewer.addPointCloudNormals<pcl::PointNormal>(normals_extracted, 1, 0.5, "normals_extracted");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 1.0, "normals_extracted");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "normals_extracted");

	viewer.addPointCloud(d_gaussian_sphere, "d_gaussian_sphere");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.8, "d_gaussian_sphere");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "d_gaussian_sphere");
	
	viewer.addPointCloudNormals<pcl::PointNormal>(d_gaussian_sphere_clustered_n, 1, 1.0, "d_gaussian_sphere_clustered_n");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.8, 0.0, 1.0, "d_gaussian_sphere_clustered_n");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "d_gaussian_sphere_clustered_n");

	viewer.spinOnce();
}

void DGaussianSphere::Publication(void)
{
	sensor_msgs::PointCloud2 pc_pub;
	pcl::toROSMsg(*d_gaussian_sphere_clustered, pc_pub);
	pc_pub.header.frame_id = cloud->header.frame_id;
	pc_pub.header.stamp = time_pub;
	pub_pc.publish(pc_pub);	
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "d_gaussian_sphere");
	
	DGaussianSphere d_gaussian_sphere;

	ros::spin();
}
