#ifndef SFM_TRAIN_H
#define SFM_TRAIN_H

//#define CERES_FOUND true
#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <iostream>
#include <opencv2/core.hpp>
//#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/calib3d.hpp>      // optflow::
#include <opencv2/video.hpp>

#include <opencv2/sfm.hpp>
//#include <opencv2/sfm/simple_pipeline.hpp>
#include <opencv2/sfm/triangulation.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
//#include <vector>


using namespace std; 
using namespace cv;
using namespace cv::sfm;
using namespace cv::xfeatures2d;
using namespace cv::cuda;

class SFM_Reconstruction 
{
private:
    
    unsigned long Match_find_SIFT( vector< KeyPoint >,
                                   vector< KeyPoint >,
                                   Mat,
                                   Mat,
                                   vector< KeyPoint > *,
                                   vector< KeyPoint > *,
                                   vector< DMatch > *);
        // Keypoints
    //unsigned long key_num = 10;  // Num keypoint
    //Ptr<FeatureDetector> detector = ORB::create(static_cast<int>(key_num), 1.2f, 8, 31, 0, 4, ORB::HARRIS_SCORE, 31);   // HARRIS_SCORE, FAST_SCORE
    //Ptr<SURF> detectorSURF = cv::xfeatures2d::SURF::create(static_cast<double>(key_num), 4, 3, true, false);   // cv::xfeatures2d::
    //Ptr<SURF> detectorSURF = cv::xfeatures2d::SURF::create(100);
    Ptr< SIFT > detectorSIFT = cv::xfeatures2d::SIFT::create();             // 0, 4, 0.04, 10, 1.6
    vector< KeyPoint > keypoints[2];                                    // Key points
    Mat descriptors[2];                           // Descriptors key points
    unsigned long numKeypoints;                                         // Number key points
    vector< DMatch > good_matches;                                      // Good matches between frames
    vector< KeyPoint > good_points[2];                      // Good points satisfying the threshold
    vector< Point2f > points[2];                                 // KeyPoints -> Points(x, y)
    
    Mat frame4;
    
        //          now            default   my_CPU
    int numLevels = 5;          // 5;        1
    double pyrScale = 0.4;      // 0.5;      0.9
    bool fastPyramids = false;
    int winSize_FOF = 12;       // 13;       12
    int numIters = 2;           // 10;       2
    int polyN = 7;              // 5;        7
    double polySigma = 1.7;     // 1.1;      1.7
    int flags = 0;
    Ptr< cuda::FarnebackOpticalFlow > FOF = cuda::FarnebackOpticalFlow::create( numLevels, 
                                                                                pyrScale,
                                                                                fastPyramids,
                                                                                winSize_FOF,
                                                                                numIters,
                                                                                polyN,
                                                                                polySigma,
                                                                                flags );
    int winSize = 13;
    int maxLevel = 3;
    int iters = 3;
    bool useInitialFlow = false;
    Ptr< cuda::DensePyrLKOpticalFlow > DPLKOF = cuda::DensePyrLKOpticalFlow::create( Size(winSize, winSize), 
                                                                                     maxLevel, 
                                                                                     iters,
                                                                                     useInitialFlow );
    double alpha = 0.197;
    double gamma = 50.0;
    double scale_factor = 0.8;
    int inner_iterations = 5;
    int outer_iterations = 150;
    int solver_iterations = 10;
    Ptr< cuda::BroxOpticalFlow > BOF = cuda::BroxOpticalFlow::create( alpha,
                                                                      gamma, 
                                                                      scale_factor,
                                                                      inner_iterations, 
                                                                      outer_iterations, 
                                                                      solver_iterations );
    double tau = 0.25;
    double lambda = 0.15;
    double theta = 0.3;
    int nscales = 5;
    int warps = 5;
    double epsilon = 0.01;
    int iterations = 300;
    double scaleStep = 0.8;
    double gamma_OFD = 0.0;
    bool useInitialFlow_OFD = false;
    Ptr< cuda::OpticalFlowDual_TVL1 > OFD = cuda::OpticalFlowDual_TVL1::create( tau,
                                                                                lambda, 
                                                                                theta,
                                                                                nscales,
                                                                                warps,
                                                                                epsilon,
                                                                                iterations, 
                                                                                scaleStep,
                                                                                gamma_OFD,
                                                                                useInitialFlow );
    GpuMat gpu_fg1, gpu_fg2, gpu_flow;
    
public:
    
        // Camera 
    Mat frame[2];
        // Essential & Fundamental & Camera matrix
    Mat E = Mat( 3, 3, CV_32FC1);
    Mat R, r, t;
    Mat F = Mat( 3, 3, CV_32FC1);
    Matx33d K[2];
        // 3D points
    Mat points3D;
    vector < Scalar > points3D_BGR;
    Mat valid_mask;
    
        // Optical flow
    Mat flow, frameFlow;
    
        // SFM camera
    /*cv::sfm::libmv_CameraIntrinsicsOptions camera { SFM_DISTORTION_MODEL_DIVISION, 
                                                    intrinsic(0, 0), 
                                                    intrinsic(1, 1), 
                                                    intrinsic(0, 2), 
                                                    intrinsic(1, 2),
                                                    distCoeffs(0, 0),
                                                    distCoeffs(0, 1),
                                                    distCoeffs(0, 4),
                                                    distCoeffs(0, 2),
                                                    distCoeffs(0, 3) };*/

    SFM_Reconstruction() = default;
    void Reconstruct3D( Mat *, Mat *, Matx33d );    // Put old frame then new frame and K matrix
    void Reconstruct3DopticFlow( Mat *, Mat *, Matx33d );
    void Reconstruct3Dstereo( Mat *, Mat *, Matx33d, Matx33d );
    void opticalFlow( Mat *, Mat *, int, int );
    void destroyWinSFM();
    void drawKeyPoints( Mat *, vector< KeyPoint > * );
    
};

#endif // SFM_TRAIN_H
