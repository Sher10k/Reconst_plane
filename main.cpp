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

#include "ZcmCameraBaslerJpegFrame.hpp"
#include "Header/sfm_train.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace zcm;

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
    SFM_Reconstruction sterio_sfm( &img[0] );
    sterio_sfm.Reconstruction3D( &img[1], &img[0], mtx[0] );
    //sterio_sfm.Reconstruction3DopticFlow( &img[1], &img[0], mtx[0] );
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
                   Rct[0], Rct[1], P[0], P[1], Q, 
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
    cout << " --- --- END STEREO RECTIFY" << endl;
    

    
    
    return 0;
}
