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

#define PCL_VISUAL true // Includes visualization: true, false
#define WHT_VISUAL 1 // Visualize points using a depth map or SFM: 0 - DISP, 1 - SFM

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
    cout << " --- --- END READ camera options" << endl;
// --- END Read camera internal settings
    
    
// --- Stereo rectify
    cout << endl << " --- --- STEREO RECTIFY" << endl;
    Mat rmap[2][2];
    Mat imgRemap[2];
    Mat K[2], R, t, Rct[2], P[2], Q;
    Rect validRoi[2];

// --- SFM
    mtx[0](0, 2) = 772;
    mtx[0](1, 2) = 1047;
    
    SFM_Reconstruction stereo_sfm;
    //stereo_sfm.Reconstruct3D( &img[1], &img[0], mtx[0] );
    stereo_sfm.Reconstruct3DopticFlow( &img[1], &img[0], mtx[0] );
    
    stereo_sfm.R.copyTo( R );
    stereo_sfm.t.copyTo( t );            
    
    stereoRectify( mtx[0], dist[0], 
                   mtx[1], dist[1], 
                   imageSize, 
                   R.inv(), t,                // R.inv()
                   Rct[0], Rct[1], P[0], P[1], Q,   // output
                   CALIB_ZERO_DISPARITY, -1, 
                   imageSize, 
                   &validRoi[0], 
                   &validRoi[1] );
    for (unsigned i = 0; i < 2; i++)
    {
        initUndistortRectifyMap( mtx[i], dist[i], 
                                 Rct[i],    // Rct[i], rectification[i]
                                 P[i],      // P[i], projection[i]
                                 imageSize, 
                                 CV_32FC1, 
                                 rmap[i][0], rmap[i][1] );
        remap( img[i], 
               imgRemap[i], 
               rmap[i][0], 
               rmap[i][1], 
               INTER_LINEAR );
    }
    
        // Combine two images & save 
    Mat frameLR = Mat( imgRemap[0].rows, imgRemap[0].cols + imgRemap[1].cols, imgRemap[0].type() );
    Rect r1(0, 0, imgRemap[0].cols, imgRemap[0].rows);
    Rect r2(imgRemap[0].cols, 0, imgRemap[1].cols, imgRemap[1].rows);
    imgRemap[0].copyTo(frameLR( r1 ));
    imgRemap[1].copyTo(frameLR( r2 ));
    for( int i = 0; i < frameLR.rows; i += 100 )
        for( int j = 0; j < frameLR.cols; j++ )
            frameLR.at< Vec3b >(i, j)[2] = 255;
    putText( frameLR, "L", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
    putText( frameLR, "R", Point(imgRemap[0].cols + 5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
    imwrite( "Remap_frame_LR.jpg", frameLR );
    cout << " --- Same left & right rectify files saved Remap_frame_LR.jpg" << endl;
// --- END SFM
    
// --- DEPTH MAP
    Ptr < StereoSGBM > sbm = StereoSGBM::create( 0,                         // minDisparity
                                                 96,                       // numDisparities must be divisible by 16
                                                 17,                        // blockSize
                                                 0,                         // P1
                                                 2048,                         // P2              0
                                                 0,                         // disp12MaxDiff       1
                                                 0,                         // prefilterCap
                                                 0,                         // uniquenessRatio      10
                                                 0,                         // speckleWindowSize    100
                                                 0,                         // speckleRange          32
                                                 StereoSGBM::MODE_SGBM_3WAY );   // mode MODE_SGBM
    resize( imgRemap[0], imgRemap[0], Size(1024,720), 0, 0, INTER_LINEAR );
    resize( imgRemap[1], imgRemap[1], Size(1024,720), 0, 0, INTER_LINEAR );
    imwrite( "Remap_frame_L.jpg", imgRemap[0] );
    imwrite( "Remap_frame_R.jpg", imgRemap[1] );
    Mat imgGrey[2]; 
    cvtColor( imgRemap[0], imgGrey[0], COLOR_BGR2GRAY);
    cvtColor( imgRemap[1], imgGrey[1], COLOR_BGR2GRAY);
    
        // Calculate
    Mat imgDisp_bm;
    sbm->compute( imgGrey[0], imgGrey[1], imgDisp_bm );
    //sbm->compute( imgLine[0], imgLine[1], imgDisp_bm );
    //imwrite( "imgDisp_bm.png", imgDisp_bm );
    
        // Nomalization
    double minVal; double maxVal;
    minMaxLoc( imgDisp_bm, &minVal, &maxVal );
    Mat imgDispNorm_bm;
    imgDisp_bm.convertTo( imgDispNorm_bm, CV_8UC1, 255/(maxVal - minVal) );
    Mat imgDisp_color;
    applyColorMap( imgDispNorm_bm, imgDisp_color, COLORMAP_RAINBOW );   // COLORMAP_HOT
    imwrite( "BM.jpeg", imgDisp_color );
    cout << " --- ImgDispBM files saved BM.jpeg" << endl;
    
        // Reprojects a disparity image to 3D space
    Mat points3D;
    reprojectImageTo3D( imgDisp_bm, points3D, Q, false );   // imgDispNorm_bm imgDisp_bm
    //cout << "Point3D = " << endl << points3D << endl;
    FileStorage Stereo_3D;
    Stereo_3D.open( "Stereo_3D.txt", FileStorage::WRITE );
    Stereo_3D << "Q" << Q;
    Stereo_3D << "imgDispNorm_bm" << imgDisp_bm;
    Stereo_3D << "Point3D" << points3D;
    Stereo_3D.release();
// --- END DEPTH MAP
    
    Mat R_inv, r_inv;
    R_inv = stereo_sfm.R.inv();
    Rodrigues( R_inv, r_inv );
    FileStorage Reconst_RESULT;
    Reconst_RESULT.open("Reconst_RESULT.txt", FileStorage::WRITE);
    Reconst_RESULT << "mtxL" << mtx[0];
    Reconst_RESULT << "mtxR" << mtx[1];
    Reconst_RESULT << "distL" << dist[0];
    Reconst_RESULT << "distR" << dist[1];
    Reconst_RESULT << "leftRectification" << Rct[0];
    Reconst_RESULT << "rightRectification" << Rct[1];
    Reconst_RESULT << "leftProjection" << P[0];
    Reconst_RESULT << "rightProjection" << P[1];
    Reconst_RESULT << "Q" << Q;                      // Disp_matrix
    Reconst_RESULT << "leftROI" << validRoi[0];
    Reconst_RESULT << "rightROI" << validRoi[1];
    Reconst_RESULT << "R" << stereo_sfm.R;
    Reconst_RESULT << "R_inv" << R_inv;
    Reconst_RESULT << "r" << stereo_sfm.r;
    Reconst_RESULT << "r_inv" << r_inv;
    Reconst_RESULT << "t" << stereo_sfm.t;
    Reconst_RESULT << "E" << stereo_sfm.E;
    Reconst_RESULT << "F" << stereo_sfm.F;
    //Reconst_RESULT << "points3D" << stereo_sfm.points3D;
    //Reconst_RESULT << "valid_mask" << stereo_sfm.valid_mask;
    Reconst_RESULT.release();
    cout << " --- Result of reconstruct written into file: Reconst_RESULT.txt" << endl;
    cout << " --- --- END STEREO RECTIFY" << endl;
// --- END Stereo rectify
    
    
// --- 3D visual
#if PCL_VISUAL 
    cout << endl << " --- --- 3D VIZUALIZATION" << endl;
    PointCloud < PointXYZRGB > ::Ptr cloud ( new PointCloud < PointXYZRGB > );
    boost::shared_ptr < visualization::PCLVisualizer > viewer ( new visualization::PCLVisualizer ("3D Viewer") );
        
        // Calculate non zero elements in disparate map & creaye 3D points cloud
    cloud->height = 1;
#if ( WHT_VISUAL == 0 )
    unsigned noZero = 0;
    vector < Scalar > points3D_BGR;
    for ( int y = 0; y < imgDispNorm_bm.rows; y++ ) 
    {
        for ( int x = 0; x < imgDispNorm_bm.cols; x++ ) 
        {
            if ( imgDispNorm_bm.at< uchar >(y, x) )
            {
                noZero++;
                points3D_BGR.push_back( Scalar( imgRemap[0].at< Vec3b >(y, x)[0],
                                                imgRemap[0].at< Vec3b >(y, x)[1],
                                                imgRemap[0].at< Vec3b >(y, x)[2] ) );
            }
        }
    }
    cloud->width = static_cast< unsigned int >( noZero );  // points3D.cols * points3D.rows
#elif ( WHT_VISUAL == 1 )
    cloud->width = static_cast< unsigned int >( stereo_sfm.points3D.cols );
#endif    
    cloud->is_dense = false;
    cloud->points.resize( cloud->width * cloud->height );
    
        // Calculate X Y Z R G B values 3D point & mul Q
    Matrix4d  Rx, Ry, Rz;
//    Matrix4d Q_32F;
//    Q_32F << Q.at<double>(0, 0), Q.at<double>(0, 1), Q.at<double>(0, 2), Q.at<double>(0, 3),
//             Q.at<double>(1, 0), Q.at<double>(1, 1), Q.at<double>(1, 2), Q.at<double>(1, 3),
//             Q.at<double>(2, 0), Q.at<double>(2, 1), Q.at<double>(2, 2), Q.at<double>(2, 3),
//             Q.at<double>(3, 0), Q.at<double>(3, 1), Q.at<double>(3, 2), Q.at<double>(3, 3);
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
    
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        Vector4d vec_tmp;
#if ( WHT_VISUAL == 0 )
        vec_tmp << points3D.at< double >(0, static_cast<int>(i)),
                   points3D.at< double >(1, static_cast<int>(i)),
                   points3D.at< double >(2, static_cast<int>(i)),
                   points3D.at< double >(3, static_cast<int>(i));
        cloud->points[i].r = static_cast< uint8_t >( points3D_BGR.at(i)[2] );
        cloud->points[i].g = static_cast< uint8_t >( points3D_BGR.at(i)[1] );
        cloud->points[i].b = static_cast< uint8_t >( points3D_BGR.at(i)[0] );
#elif ( WHT_VISUAL == 1 )
        vec_tmp << stereo_sfm.points3D.at< double >(0, static_cast<int>(i)),
                   stereo_sfm.points3D.at< double >(1, static_cast<int>(i)),
                   stereo_sfm.points3D.at< double >(2, static_cast<int>(i)),
                   stereo_sfm.points3D.at< double >(3, static_cast<int>(i));
        cloud->points[i].r = static_cast< uint8_t >( stereo_sfm.points3D_BGR.at(i)[2] );
        cloud->points[i].g = static_cast< uint8_t >( stereo_sfm.points3D_BGR.at(i)[1] );
        cloud->points[i].b = static_cast< uint8_t >( stereo_sfm.points3D_BGR.at(i)[0] );
#endif    
        //vec_tmp = Rx * Ry * Rz  * vec_tmp;
        if ( (abs(vec_tmp(0)) < 100) &&
             (abs(vec_tmp(1)) < 100) && 
             (abs(vec_tmp(2)) < 100) )
        {
            cloud->points[i].x = float(vec_tmp(0));
            cloud->points[i].y = float(vec_tmp(1));
            cloud->points[i].z = float(vec_tmp(2));
        }
        else
        {
            cloud->points[i].x = 0;
            cloud->points[i].y = 0;
            cloud->points[i].z = 0;
        }
    }
    
        // Save & load 3D points fixle
    pcl::io::savePCDFileASCII ("Reconstruct_cloud.pcd", *cloud);    // test_pcd.pcd
    cout << " --- 3D points cloud saved" << endl;
        // Visualization 3D points
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(5.0, "global");
    viewer->addPointCloud< pcl::PointXYZRGB >( cloud, "sample cloud0", 0 );
    viewer->setPointCloudRenderingProperties ( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud0" );
    
// Plane 
//    float lp = 800, wp = 200, hp = 20;
//    cloud->height = 1;
//    cloud->width = unsigned(lp * wp);
//    cloud->is_dense = false;
//    cloud->points.resize( cloud->width * cloud->height );
//    unsigned nk = 0;
//    for ( unsigned i = 0; i < lp; i++ )
//    {
//        for ( unsigned j = 0; j < wp; j++ )
//        {
//            cloud->points[nk].x = float( (-wp/2 + j) /4 + j%4 );
//            cloud->points[nk].y = hp;
//            cloud->points[nk].z = float( i/8 + j%8 );
//            cloud->points[nk].r = 0;
//            cloud->points[nk].g = 100;
//            cloud->points[nk].b = 255;
//            nk++;
//        }
//    }
//    viewer->addPointCloud< pcl::PointXYZRGB >( cloud, "plane", 0 );
//    viewer->setPointCloudRenderingProperties ( pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "plane" );
    
    viewer->spin();
    cout << " --- --- END 3D VIZUALIZATION" << endl;
#endif
// --- END 3D visual
    
    return 0;
}






/*


        // 3D points from dispmap
//    for(int y = 0; y < imgDisp_bm.rows; ++y) 
//    {
//        for(int x = 0; x < imgDisp_bm.cols; ++x) 
//        {
//            if ( imgDispNorm_bm.at< uchar >(y, x) )
//            {
//                Vector4d vec_tmp;
//                vec_tmp(0) = x;
//                vec_tmp(1) = y;
//                vec_tmp(2) = double( imgDispNorm_bm.at< uchar >(y, x) );
//                vec_tmp(3) = 1;
//                vec_tmp = Rx * Ry * Rz * Q_32F * vec_tmp;
//                vec_tmp /= vec_tmp(3);
//                if ( ((vec_tmp(0) < double(-wp/4)) && (vec_tmp(0) > double(wp/4))) || 
//                     (vec_tmp(1) > double(hp + 3)) ||
//                     (vec_tmp(2) > 200) ) 
//                {
//                    vec_tmp(0) = 0;
//                    vec_tmp(1) = 0;
//                    vec_tmp(2) = 0;
//                }
//                cloud->points[nk].x = float(vec_tmp(0));
//                cloud->points[nk].y = float(vec_tmp(1));
//                cloud->points[nk].z = float(vec_tmp(2));
//                cloud->points[nk].r = imgRemap[0].at< Vec3b >(y, x)(2);
//                cloud->points[nk].g = imgRemap[0].at< Vec3b >(y, x)(1);
//                cloud->points[nk].b = imgRemap[0].at< Vec3b >(y, x)(0);
//                nk++;
//            }
//        }
//    }

  
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
    
    
*/ 
