#include <string>
#include <iostream>
#include <fstream>
#include <getopt.h>     // Для long_opts[]
#include <utility>      // std::move()
//#include <vector>
#include <set>
//#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>

#include <Eigen/Eigen>

#include <zcm/zcm.h>
#include <zcm/zcm-cpp.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/impl/point_types.hpp>
//#include <pcl/console/parse.h>
//#include <pcl/features/normal_3d.h>
#include <pcl/visualization/common/common.h>
//#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/cloud_viewer.h>

#include <boost/thread/thread.hpp>

#include "ZcmCameraBaslerJpegFrame.hpp"
#include "Header/sfm_train.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace zcm;
using namespace pcl;

struct Args
{
    string inlog = "";
    string params = "";
    
    bool parse(int argc, char *argv[])
    {
        // Default sets
        const char *optstring = "i:p:h";
        struct option long_opts[] = {
            { "input_zcm",  required_argument,  0,  'i' },
            { "params",     required_argument,  0,  'p' },
            { "help",       no_argument,        0,  'h' },
            { 0, 0, 0}
        };
        
        int c;
        while ( (c = getopt_long( argc, argv, optstring, long_opts, 0 ) ) >= 0 )
        {
            switch(c)
            {
                case 'i': inlog     = string( optarg ); break;
                case 'p': params     = string( optarg ); break;
                case 'h': default: usage(); return false;
            };
        }
        
        if (inlog == "") 
        {
                    std::cerr << " --- Please specify logfile input" << std::endl;
                    return false;
        }
        if (params == "") 
        {
                    std::cerr << " --- Please specify folder with the parameters camera" << std::endl;
                    return false;
        }
        
        return true;
    }
    
    void usage()
    {
        cout << "usage: log2stereo [options]" << endl
             << "" << endl
             << " Convert zcm log file to stereo image file" << endl
             << "Example:" << endl
             << "    Reconst_plane -i zcm.log -p parameters/folder/" << endl
             << "" << endl
             << "Options:" << endl
             << "" << endl
             << "  -h, --help                           Shows this help text and exits" << endl
             << "  -i, --input_zcm=logfile              Input log to convert" << endl
             << "  -p, --params=parameters/folder       Folder of parameters" << endl
             << endl << endl;
    }
};


int main(int argc, char *argv[]) //int argc, char *argv[]
{
// --- Input options
    Args args;
    if ( !args.parse(argc, argv) ) return 1;
    
    string input_filename = args.inlog;
    cout << " --- Input_ZCM_file: \t\t\t" << input_filename << endl;
    string parametersDir = args.params;
    cout << " --- Folder with the parameters: \t" << parametersDir << endl;
    
// --- Read zcm log file
    LogFile *zcm_log;
    zcm_log = new LogFile( input_filename, "r" );
    if ( !zcm_log->good() )
    {
        cout << " --- Bad zcm log: " << input_filename << endl;
        exit(0);
    }
// --- Get left & right same channels from thread
    set < string > zcm_list;
    Mat img[2];
    long temp_t_samp = 0;
    bool Lflag = false, Rflag = false;
    cout << "Time: " << endl;
    while ( true )
    {
        const LogEvent *event = zcm_log->readNextEvent();
        if ( !event ) break;
        ZcmCameraBaslerJpegFrame zcm_msg;
        long tts = 0;
        if ( event->channel == "FLZcmCameraBaslerJpegFrame" )
        {
            tts = zcm_msg.service.u_timestamp;
            cout << "L " << tts << endl;
            zcm_msg.decode( event->data, 0, static_cast<unsigned>(event->datalen) );
            img[0] = imdecode( zcm_msg.jpeg, IMREAD_COLOR);
            imwrite( "videoZCM_1908212042_L.jpg", img[0]);
            Lflag = true;
            
            if ( (temp_t_samp == tts) && (Lflag) && (Rflag) ) break;
            else temp_t_samp = tts;
            Rflag = false;
        }
        else if ( event->channel == "FRZcmCameraBaslerJpegFrame" )
        {
            tts = zcm_msg.service.u_timestamp;
            cout << "R " << tts << endl;
            zcm_msg.decode( event->data, 0, static_cast<unsigned>(event->datalen) );
            img[1] = imdecode( zcm_msg.jpeg, IMREAD_COLOR);
            imwrite( "videoZCM_1908212042_R.jpg", img[1]);
            Rflag = true;
            
            if ( (temp_t_samp == tts) && (Lflag) && (Rflag) ) break;
            else temp_t_samp = tts;
            Lflag = false;
        }
        //zcm_list.insert( event->channel );
    }
    cout << " --- Same left & right files saved " << endl;
//    cout << "zcm_list: " << endl;
//    for ( auto i : zcm_list )
//        cout << "\t" << i << endl;
    
    Size imageSize = Size( img[0].cols, img[0].rows );
    
    
// --- Read camera internal settings
    cout << endl << " --- --- READ camera options" << endl;
    Matx < double, 3, 3 > mtx[2];
    Matx < double, 1, 5 > dist[2];
    Matx < double, 3, 4 > projection[2];
    Matx < double, 3, 3 > rectification[2];
    Rect ROI[2];
    
// --- Left camera options
    cout << " --- LEFT camera" << endl;
        // Camera matrix
    fstream file_params( parametersDir + "mtx.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "mtx.csv" << endl;
        exit(0);
    }
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++)
            file_params >> mtx[0](i, j);
    cout << "mtxL = " << endl << mtx[0] << endl;
    file_params.close();
        // Distortion matrix
    file_params.open( parametersDir + "dist.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "dist.csv" << endl;
        exit(0);
    }
    for ( int j = 0; j < 5; j++ )
        file_params >> dist[0](0, j);
    cout << "distL = " << endl << dist[0] << endl;
    file_params.close();
        // Projection matrix
    file_params.open( parametersDir + "to_right/22500061/leftProjection.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/leftProjection.csv" << endl;
        exit(0);
    }
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 4; j++)
            file_params >> projection[0](i, j);
    cout << "projectionL = " << endl << projection[0] << endl;
    file_params.close();
        // Rectification matrix
    file_params.open( parametersDir + "to_right/22500061/leftRectification.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/leftRectification.csv" << endl;
        exit(0);
    }
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++)
            file_params >> rectification[0](i, j);
    cout << "rectificationL = " << endl << rectification[0] << endl;
    file_params.close();
        // ROI mask
    file_params.open( parametersDir + "to_right/22500061/leftROI.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/leftROI.csv" << endl;
        exit(0);
    }
    double a1,a2,a3,a4;
    file_params >> a1 >> a2 >> a3 >> a4;
    ROI[0].x        = int(a1);
    ROI[0].y        = int(a2);
    ROI[0].width    = int(a3);
    ROI[0].height   = int(a4);
    cout << "ROIL = " << endl << ROI[0] << endl;
    file_params.close();
    
// --- Right camera options
    cout << endl << " --- RIGHT camera" << endl;
        // Camera matrix
    file_params.open( parametersDir + "to_right/22500061/mtxR.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/mtxR.csv" << endl;
        exit(0);
    }
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++)
            file_params >> mtx[1](i, j);
    cout << "mtxR = " << endl << mtx[1] << endl;
    file_params.close();
        // Distortion matrix
    file_params.open( parametersDir + "to_right/22500061/distR.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/distR.csv" << endl;
        exit(0);
    }
    for ( int j = 0; j < 5; j++ )
        file_params >> dist[1](0, j);
    cout << "distR = " << endl << dist[1] << endl;
    file_params.close();
        // Projection matrix
    file_params.open( parametersDir + "to_right/22500061/rightProjection.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/rightProjection.csv" << endl;
        exit(0);
    }
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 4; j++)
            file_params >> projection[1](i, j);
    cout << "projectionR = " << endl << projection[1] << endl;
    file_params.close();
        // Rectification matrix
    file_params.open( parametersDir + "to_right/22500061/rightRectification.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/rightRectification.csv" << endl;
        exit(0);
    }
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++)
            file_params >> rectification[1](i, j);
    cout << "rectificationR = " << endl << rectification[1] << endl;
    file_params.close();
        // ROI mask
    file_params.open( parametersDir + "to_right/22500061/rightROI.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/rightROI.csv" << endl;
        exit(0);
    }
    file_params >> a1 >> a2 >> a3 >> a4;
    ROI[1].x        = int(a1);
    ROI[1].y        = int(a2);
    ROI[1].width    = int(a3);
    ROI[1].height   = int(a4);
    cout << "ROIL = " << endl << ROI[1] << endl;
    file_params.close();
    cout << " --- --- END READ camera options" << endl;
    
// --- Stereo rectify
    cout << endl << " --- --- STEREO RECTIFY" << endl;
    Mat rmap[2][2];
    Mat imgRemap[2];
    Mat K[2], R, t, Rct[2], P[2], Q;
    Rect validRoi[2];

// --- SFM
    mtx[0](0, 2) = 772;
    mtx[0](1, 2) = 1050;
    
    SFM_Reconstruction sterio_sfm;
    sterio_sfm.Reconstruct3D( &img[1], &img[0], mtx[0] );
    //sterio_sfm.Reconstruct3DopticFlow( &img[1], &img[0], mtx[0] );
    
    sterio_sfm.R.copyTo( R );
    sterio_sfm.t.copyTo( t );            
    
//    ZcmCameraCalibratingParams zcm_calib;
//    float K1[3][3];
//    cout << "K1= " << endl;
//    for ( int i = 0; i < 3; i++ )
//    {
//        for ( int j = 0; j < 3; j++)
//            cout << zcm_calib.cam_mtx[j][i] << "\t\t";
//        cout << endl;
//    }   
//    decomposeProjectionMatrix( projection[0], 
//                               K[0], R[0], T[0] );
//    cout << "K_L= " << endl << K[0] << endl
//         << "R_L= " << endl << R[0] << endl
//         << "T_L= " << endl << T[0] << endl;
//    decomposeProjectionMatrix( projection[1], 
//                               K[1], R[1], T[1] );
//    cout << "K_R= " << endl << K[1] << endl
//         << "R_R= " << endl << R[1] << endl
//         << "T_R= " << endl << T[1] << endl;
    
    stereoRectify( mtx[0], dist[0], 
                   mtx[1], dist[1], 
                   imageSize, 
                   R.inv(), t, 
                   Rct[0], Rct[1], P[0], P[1], Q,   // output
                   CALIB_ZERO_DISPARITY, -1, 
                   imageSize, 
                   &validRoi[0], 
                   &validRoi[1] );
    for (unsigned i = 0; i < 2; i++)
    {
        initUndistortRectifyMap( mtx[i], dist[i], 
                                 Rct[i],  // Rct[i], rectification[i]
                                 P[i],     // P[i], projection[i]
                                 imageSize, 
                                 CV_32FC1, 
                                 rmap[i][0], rmap[i][1] );
        remap( img[i], 
               imgRemap[i], 
               rmap[i][0], 
               rmap[i][1], 
               INTER_LINEAR );
    }
    imwrite( "Remap_frame_L.jpg", imgRemap[0] );
    imwrite( "Remap_frame_R.jpg", imgRemap[1] );
        // Combine two images
    Mat frameLR = Mat( imgRemap[0].rows, imgRemap[0].cols + imgRemap[1].cols, imgRemap[0].type() );
    Rect r1(0, 0, imgRemap[0].cols, imgRemap[0].rows);
    Rect r2(imgRemap[0].cols, 0, imgRemap[1].cols, imgRemap[1].rows);
    putText( imgRemap[0], "L", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
    putText( imgRemap[1], "R", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
    imgRemap[0].copyTo(frameLR( r1 ));
    imgRemap[1].copyTo(frameLR( r2 ));
    for( int i = 0; i < frameLR.rows; i += 100 )
        for( int j = 0; j < frameLR.cols; j++ )
            frameLR.at< Vec3b >(i, j)[2] = 255;
    imwrite( "Remap_frame_LR.jpg", frameLR );
    cout << " --- Same left & right rectify files saved" << endl;
    
// --- DEPTH MAP
    Ptr < StereoSGBM > sbm = StereoSGBM::create( 0,                         // minDisparity
                                                 256,                       // numDisparities must be divisible by 16
                                                 17,                        // blockSize
                                                 0,                         // P1
                                                 0,                         // P2
                                                 1,                         // disp12MaxDiff
                                                 0,                         // prefilterCap
                                                 10,                         // uniquenessRatio
                                                 100,                         // speckleWindowSize
                                                 32,                         // speckleRange
                                                 StereoSGBM::MODE_SGBM );   // mode
    Mat imgGrey[2]; 
    cvtColor( imgRemap[0], imgGrey[0], COLOR_BGR2GRAY);
    cvtColor( imgRemap[1], imgGrey[1], COLOR_BGR2GRAY);
    
//    Mat imgLine[2];
//    vector< Vec4i > lines[2];
//    Canny( imgRemap[0], imgGrey[0], 10, 50, 3, false );
//    Canny( imgRemap[1], imgGrey[1], 10, 50, 3, false );
//    imwrite( "Canny_frame_L.jpg", imgGrey[0] );
//    imwrite( "Canny_frame_R.jpg", imgGrey[1] );
//    HoughLinesP( imgGrey[0], lines[0], 1, CV_PI/180, 80, 30, 10 );
//    cvtColor( imgGrey[0], imgLine[0], COLOR_GRAY2BGR);
//    imgLine[0] *= 0;
//    for( size_t i = 0; i < lines[0].size(); i++ )
//    {
//        line( imgLine[0], Point(lines[0][i][0], lines[0][i][1]),
//        Point( lines[0][i][2], lines[0][i][3]), Scalar(0,0,255), 3, 8 );
//    }
//    HoughLinesP( imgGrey[1], lines[1], 1, CV_PI/180, 80, 30, 10 );
//    cvtColor( imgGrey[1], imgLine[1], COLOR_GRAY2BGR);
//    imgLine[1] *= 0;
//    for( size_t i = 0; i < lines[1].size(); i++ )
//    {
//        line( imgLine[1], Point(lines[1][i][0], lines[1][i][1]),
//        Point( lines[1][i][2], lines[1][i][3]), Scalar(0,0,255), 3, 8 );
//    }
//    imwrite( "Hough_frame_L.jpg", imgLine[0] );
//    imwrite( "Hough_frame_R.jpg", imgLine[1] );
    
        // Calculate
    Mat imgDisp_bm;
    sbm->compute( imgGrey[0], imgGrey[1], imgDisp_bm );
    //sbm->compute( imgLine[0], imgLine[1], imgDisp_bm );
        // Nomalization
    double minVal; double maxVal;
    minMaxLoc( imgDisp_bm, &minVal, &maxVal );
    Mat imgDispNorm_bm;
    imgDisp_bm.convertTo( imgDispNorm_bm, CV_8UC1, 255/(maxVal - minVal) );
    imwrite( "BM.jpeg", imgDispNorm_bm );
    cout << " --- ImgDispBM files saved" << endl;
    
// --- reconstruct 3D points
    Mat points3D;
    reprojectImageTo3D( imgDisp_bm, points3D, Q, false );
    //cout << "Point3D = " << endl << points3D << endl;
    FileStorage Stereo_3D;
    Stereo_3D.open( "Stereo_3D.txt", FileStorage::WRITE );
    Stereo_3D << "Q" << Q;
    Stereo_3D << "imgDispNorm_bm" << imgDispNorm_bm;
    //Stereo_3D << "Point3D" << points3D;
    Stereo_3D.release();
    cout << " --- --- END STEREO RECTIFY" << endl;
    
// --- 3D visual
    cout << endl << " --- --- 3D VIZUALIZATION" << endl;
    PointCloud < PointXYZRGB > cloud;
    PointCloud < PointXYZRGB > ::Ptr cloud2 ( new PointCloud < PointXYZRGB > );
    boost::shared_ptr < visualization::PCLVisualizer > viewer ( new visualization::PCLVisualizer ("3D Viewer") );
        
        // Calculate non zero elements in disparate map & creaye 3D points cloud
    unsigned noZero = 0; 
    for (size_t i = 0; i < imgDispNorm_bm.total(); i++ ) {
            if ( imgDispNorm_bm.at< uchar >(int(i)) ) noZero++;
        }
    cloud.height = 1;
    cloud.width = static_cast< unsigned int >( noZero );  // points3D.cols * points3D.rows
    cloud.is_dense = false;
    cloud.points.resize( cloud.width * cloud.height );
    
        // Calculate X Y Z R G B values 3D point & mul Q
    Vector4d vec_tmp;
    Matrix4d Q_32F, Rx, Ry, Rz;
    Q_32F << Q.at<double>(0, 0), Q.at<double>(0, 1), Q.at<double>(0, 2), Q.at<double>(0, 3),
             Q.at<double>(1, 0), Q.at<double>(1, 1), Q.at<double>(1, 2), Q.at<double>(1, 3),
             Q.at<double>(2, 0), Q.at<double>(2, 1), Q.at<double>(2, 2), Q.at<double>(2, 3),
             Q.at<double>(3, 0), Q.at<double>(3, 1), Q.at<double>(3, 2), Q.at<double>(3, 3);
    double fx = 24;
    Rx << 1,  0,                    0,                   0,
          0,  cos(fx * CV_PI / 180),  sin(fx * CV_PI / 180), 0,
          0,  -sin(fx * CV_PI / 180), cos(fx * CV_PI / 180), 0,
          0,  0,                    0,                   1;
    double fy = 0;
    Ry << cos(fy * CV_PI / 180),  0, -sin(fy * CV_PI / 180), 0,
          0,                    1, 0,                    0,
          sin(fy * CV_PI / 180),  0, cos(fy * CV_PI / 180),  0,
          0,                    0, 0,                    1;
    double fz = 7;
    Rz << cos(fz * CV_PI / 180),  sin(fz * CV_PI / 180), 0, 0,
          -sin(fz * CV_PI / 180), cos(fz * CV_PI / 180), 0, 0,
          0,                    0,                   1, 0,
          0,                    0,                   0, 1;
    
    unsigned nk = 0;
    float lp = 800, wp = 200, hp = 20;
    for(int y = 0; y < imgDisp_bm.rows; ++y) 
    {
        for(int x = 0; x < imgDisp_bm.cols; ++x) 
        {
            if ( imgDispNorm_bm.at< uchar >(y, x) )
            {
                vec_tmp(0) = x;
                vec_tmp(1) = y;
                vec_tmp(2) = double( imgDispNorm_bm.at< uchar >(y, x) );
                vec_tmp(3) = 1;
                vec_tmp = Rx * Ry * Rz * Q_32F * vec_tmp;
                vec_tmp /= vec_tmp(3);
                if ( ((vec_tmp(0) < double(-wp/4)) && (vec_tmp(0) > double(wp/4))) || 
                     (vec_tmp(1) > double(hp + 3)) ||
                     (vec_tmp(2) > 200) ) 
                {
                    vec_tmp(0) = 0;
                    vec_tmp(1) = 0;
                    vec_tmp(2) = 0;
                }
                cloud.points[nk].x = float(vec_tmp(0));
                cloud.points[nk].y = float(vec_tmp(1));
                cloud.points[nk].z = float(vec_tmp(2));
                cloud.points[nk].r = imgRemap[0].at< Vec3b >(y, x)(2);
                cloud.points[nk].g = imgRemap[0].at< Vec3b >(y, x)(1);
                cloud.points[nk].b = imgRemap[0].at< Vec3b >(y, x)(0);
                nk++;
            }
        }
    }    
        // Save & load 3D points fixle
    pcl::io::savePCDFileASCII ("Reconstruct_cloud.pcd", cloud);
    pcl::io::loadPCDFile("Reconstruct_cloud.pcd", *cloud2);  // test_pcd.pcd
    cout << " --- 3D points cloud saved" << endl;
        // Visualization 3D points
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(5.0, "global");
    viewer->addPointCloud< pcl::PointXYZRGB >( cloud2, "sample cloud0", 0 );
    viewer->setPointCloudRenderingProperties ( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud0" );
    
// Plane 
    cloud.height = 1;
    cloud.width = unsigned(lp * wp);
    cloud.is_dense = false;
    cloud.points.resize( cloud.width * cloud.height );
    nk = 0;
    for ( unsigned i = 0; i < lp; i++ )
    {
        for ( unsigned j = 0; j < wp; j++ )
        {
            cloud.points[nk].x = float( (-wp/2 + j) /4 + j%4 );
            cloud.points[nk].y = hp;
            cloud.points[nk].z = float( i/8 + j%8 );
            cloud.points[nk].r = 0;
            cloud.points[nk].g = 100;
            cloud.points[nk].b = 255;
            nk++;
        }
    }
    pcl::io::savePCDFileASCII ("Reconstruct_cloud.pcd", cloud);
    pcl::io::loadPCDFile("Reconstruct_cloud.pcd", *cloud2);
    viewer->addPointCloud< pcl::PointXYZRGB >( cloud2, "plane", 0 );
    viewer->setPointCloudRenderingProperties ( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "plane" );
    
    viewer->spin();
    cout << " --- --- END 3D VIZUALIZATION" << endl;
    
    return 0;
}
