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

//#include <vector>


using namespace std; 
using namespace cv;
using namespace cv::sfm;
using namespace cv::xfeatures2d;

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
    vector< KeyPoint > keypoints1_SIFT, keypoints2_SIFT;                // Key points
    Mat descriptors1_SIFT, descriptors2_SIFT;                           // Descriptors key points
    unsigned long numKeypoints;                                         // Number key points
    vector< DMatch > good_matches;                                      // Good matches between frames
    vector< KeyPoint > good_points1, good_points2;                      // Good points satisfying the threshold
    vector< Point2f > points1, points2;                                 // KeyPoints -> Points(x, y)
    
    Mat frame4;
    
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
