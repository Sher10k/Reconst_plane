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

class ReadStream {
private:
        //Приватное поле - файл
    ifstream f;
public:
        //Конструктор, открывающий файл
    ReadStream(const char *FileName)
    {
        f.open(FileName); 
        if(!f.is_open()) cout<<"Файл не открыт";
    }
        //Метод чтения массива символов-байт с указанной позиции
        //который возвращает этот массив
    char *read(int position, int count)
    {
        if(!f.is_open()) return 0;
        f.seekg(position);
        if(f.eof()) return 0;
        char *buffer=new char[count];
        f.read(buffer,count);
        return buffer;
    }
        //Деструктор, закрывающий файл
    ~ReadStream()
    {
        f.close(); 
    }
};


int main(int argc, char *argv[]) //int argc, char *argv[]
{
        // Input options
    Args args;
    if ( !args.parse(argc, argv) ) return 1;
    
    string input_filename = args.inlog;
    cout << " --- Input_ZCM_file: \t\t\t" << input_filename << endl;
    string parametersDir = args.params;
    cout << " --- Folder with the parameters: \t" << parametersDir << endl;
    
        // Read zcm log file
    LogFile *zcm_log;
    zcm_log = new LogFile( input_filename, "r" );
    if ( !zcm_log->good() )
    {
        cout << " --- Bad zcm log: " << input_filename << endl;
        exit(0);
    }
        // Get left & right same channels from thread
    set < string > zcm_list;
    Mat imgL, imgR;
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
            imgL = imdecode( zcm_msg.jpeg, IMREAD_COLOR);
            imwrite( "videoZCM_1908212042_L.jpg", imgL);
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
            imgR = imdecode( zcm_msg.jpeg, IMREAD_COLOR);
            imwrite( "videoZCM_1908212042_R.jpg", imgR);
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
    
    fstream file_params( parametersDir + "mtx.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "mtx.csv" << endl;
        exit(0);
    }
    
    double d1;
    file_params >> d1;
    cout << "d1: " << d1 << endl;
    
    
    file_params.close();
    
    return 0;
}
