#include <string>
#include <iostream>
#include <getopt.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>

#include <Eigen/Eigen>

#include <zcm/zcm.h>
#include <zcm/zcm-cpp.hpp>

#include "ZcmCameraBaslerJpegFrame.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace zcm;

struct Args
{
    string inlog = "";
    
    bool parse(int argc, char *argv[])
    {
        // Default sets
        const char *optstring = "i:h";
        struct option long_opts[] = {
            { "input",  required_argument,  0,  'i' },
            { "help",       no_argument,        0,  'h' },
            { 0, 0, 0}
        };
        
        int c;
        while ( (c = getopt_long( argc, argv, optstring, long_opts, 0 ) ) >= 0 )
        {
            switch(c)
            {
                case 'i': inlog     = string( optarg ); break;
                case 'h': default: usage(); return false;
            };
        }
        
        if (inlog == "") 
        {
                    std::cerr << "Please specify logfile input" << std::endl;
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
             << "    Reconst_plane -i zcm.log " << endl
             << "" << endl
             << "Options:" << endl
             << "" << endl
             << "  -h, --help              Shows this help text and exits" << endl
             << "  -i, --input=logfile     Input log to convert" << endl
             << endl << endl;
    }
};

int main(int argc, char *argv[]) //int argc, char *argv[]
{
    Args args;
    if (!args.parse(argc, argv)) return 1;
    
    string input_filename = args.inlog;
    cout << "input_filename: " << input_filename << endl;
    
    LogFile *zcm_log;
    
    
    
    
    return 0;
}
