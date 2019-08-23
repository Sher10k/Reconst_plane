
#include "Header/sfm_train.h"

//SFM_Reconstruction::SFM_Reconstruction(VideoCapture *data_CAP)
//{
//    SFM_Reconstruction::setParam(data_CAP);
//}

SFM_Reconstruction::SFM_Reconstruction( Mat *frame )
{
    SFM_Reconstruction::setParam( frame );
}

void SFM_Reconstruction::setParam( Mat *frame)
{
    width_frame = frame->cols;
    height_frame = frame->rows;
    frame1 = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);
    frame2 = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);
    frame4 = Mat::zeros(Size(2 * width_frame, height_frame), CV_8UC3);
    numKeypoints = 0;
}

void SFM_Reconstruction::Reconstruction3D(Mat *data_frame1, Mat *data_frame2, Matx33d K)
{
    //points3D *= 0;
    points3D.release();
    points3D_BGR.clear();
    R *= 0;
    t *= 0;
//--- STEP 1 --- Detection and calculation
    keypoints1_SIFT.clear();
    keypoints2_SIFT.clear();
    descriptors1_SIFT *= 0;
    descriptors2_SIFT *= 0;
    
    int Xc = 751, Yc = 250;
    Mat maskL = Mat::zeros( Size( width_frame, height_frame ), CV_8UC1 );
    Mat maskR = maskL.clone();
    for ( int i = 0; i < frame2.rows; i++ )
    {
        for ( int j = 0; j < frame2.cols; j++ ) 
        {
            if ( j < Xc ) maskL.at< char >(i, j) = 1;
            else maskR.at< char >(i, j) = 1;
        }
    }
    vector< KeyPoint > kpL, kpR;
    Mat dpL, dpR;
    dpL *= 0;
    dpR *= 0;
    
    if (!data_frame1->empty()) {
        data_frame1->copyTo(frame1);
//        kpL.clear();
//        kpR.clear();
        
//        detectorSIFT->detectAndCompute( frame1, maskL, kpL, dpL );
//        detectorSIFT->detectAndCompute( frame1, maskR, kpR, dpR );
//        keypoints1_SIFT.reserve( kpL.size() + kpR.size() );
//        keypoints1_SIFT.insert( keypoints1_SIFT.end(), kpL.begin(), kpL.end() );
//        keypoints1_SIFT.insert( keypoints1_SIFT.end(), kpR.begin(), kpR.end() );
        
        detectorSIFT->detectAndCompute(frame1, cv::noArray(), keypoints1_SIFT, descriptors1_SIFT);
    }
    if (!data_frame2->empty()) {
        data_frame2->copyTo(frame2);
//        kpL.clear();
//        kpR.clear();
//        dpL *= 0;
//        dpR *= 0;
        
//        detectorSIFT->detectAndCompute( frame2, maskL, kpL, dpL );
//        detectorSIFT->detectAndCompute( frame2, maskR, kpR, dpR );
//        keypoints1_SIFT.reserve( kpL.size() + kpR.size() );
//        keypoints1_SIFT.insert( keypoints1_SIFT.end(), kpL.begin(), kpL.end() );
//        keypoints1_SIFT.insert( keypoints1_SIFT.end(), kpR.begin(), kpR.end() );
//        kpL.clear();
//        kpR.clear();
        
        detectorSIFT->detectAndCompute(frame2, cv::noArray(), keypoints2_SIFT, descriptors2_SIFT);
    }
//--- STEP 2 --- Matcher keypoints
    good_matches.clear();
    good_points1.clear();
    good_points2.clear();
    points1.clear();
    points2.clear();
    numKeypoints = 0;
    if ((keypoints1_SIFT.size() != 0) && (keypoints2_SIFT.size() != 0))
    {
        numKeypoints = Match_find_SIFT(keypoints1_SIFT,
                                       keypoints2_SIFT,
                                       descriptors1_SIFT,
                                       descriptors2_SIFT,
                                       &good_points1,
                                       &good_points2,
                                       &good_matches);
        points1.resize(numKeypoints);
        points2.resize(numKeypoints);
        KeyPoint::convert(good_points1, points1);     // Convert from KeyPoint to Point2f
        KeyPoint::convert(good_points2, points2);
//--- STEP 3 --- Find essential matrix
        if (numKeypoints > 7)
        {
            Essen_mask = Mat::ones( Size( 1, int(numKeypoints)), Essen_mask.type());
            E = findEssentialMat(points1, points2, K, RANSAC, 0.999, 3.0, Essen_mask);
            F = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99, Essen_mask);
            correctMatches(F, points1, points2, points1, points2);
            
//--- STEP 4 --- Decompose essential matrix
            Mat p3D;
            p3D *= 0;
            recoverPose(E, points1, points2, K, R, t, 10, Essen_mask, p3D);
            int numNoZeroMask = countNonZero(Essen_mask);
            points3D = Mat::zeros( 4, numNoZeroMask, p3D.type() );
            
            int k = 0;
            for (int i = 0; i < p3D.cols; i++)
            {
                if ( Essen_mask.at< uchar >(i) == 1 )
                {
                    for (int j = 0; j < points3D.rows; j++)
                    {
                        points3D.at< double >(j, k) = p3D.at< double >(j, k) / p3D.at< double >(3, k);
                    }
                    points3D_BGR.push_back( Scalar( (data_frame1->at<Vec3b>(points1.at(static_cast<size_t>(i)))[0] + 
                                                     data_frame2->at<Vec3b>(points2.at(static_cast<size_t>(i)))[0] ) / 2,
                                                    (data_frame1->at<Vec3b>(points1.at(static_cast<size_t>(i)))[1] + 
                                                     data_frame2->at<Vec3b>(points2.at(static_cast<size_t>(i)))[1] ) / 2,
                                                    (data_frame1->at<Vec3b>(points1.at(static_cast<size_t>(i)))[2] + 
                                                     data_frame2->at<Vec3b>(points2.at(static_cast<size_t>(i)))[2] ) / 2 ) );
                    k++;
                }
            }
            
            FileStorage SFM_Result;
            SFM_Result.open("SFM_Result.txt", FileStorage::WRITE);
            SFM_Result << "E" << E;
            SFM_Result << "F" << F;
            //SFM_Result << "points1" << points1;
            //SFM_Result << "points2" << points2;
            SFM_Result << "K" << K;
            SFM_Result << "R" << R;
            SFM_Result << "t" << t;
            SFM_Result << "points3D" << points3D;
            SFM_Result << "Essen_mask" << Essen_mask;
            SFM_Result.release();
            cout << " --- SFM_Result written into file: SFM_Result.txt" << endl;
            
            drawMatches(frame1, good_points1, frame2, good_points2, good_matches, frame4);
            //imshow("SFM-result", frame4);
            imwrite("SFM-result.jpg", frame4);
            waitKey(10);
        } 
        else 
        {
            if (!data_frame1->empty()) drawKeyPoints(&frame1, &keypoints1_SIFT);
            if (!data_frame2->empty()) drawKeyPoints(&frame2, &keypoints2_SIFT);
            Rect r1(0, 0, frame1.cols, frame1.rows);
            Rect r2(frame2.cols, 0, frame2.cols, frame2.rows);
            frame1.copyTo(frame4( r1 ));
            frame2.copyTo(frame4( r2 ));
            imshow("SFM-result", frame4);
            waitKey(10);
            cout << " --- Not enough keypoints" << endl;
        }
    } 
    else 
    {
        if (!data_frame1->empty()) drawKeyPoints(&frame1, &keypoints1_SIFT);
        if (!data_frame2->empty()) drawKeyPoints(&frame2, &keypoints2_SIFT);
        Rect r1(0, 0, frame1.cols, frame1.rows);
        Rect r2(frame2.cols, 0, frame2.cols, frame2.rows);
        frame1.copyTo(frame4( r1 ));
        frame2.copyTo(frame4( r2 ));
        imshow("SFM-result", frame4);
        waitKey(10);
        cout <<" --- No keypoints found in one of the frames" << endl;
    }
}

void SFM_Reconstruction::Reconstruction3DopticFlow(Mat *data_frame1, Mat *data_frame2, Matx33d K)
{
    R *= 0;
    t *= 0;
    Mat frame = Mat::zeros(Size(width_frame, height_frame), CV_8UC3);           // data_frame2->type()
    flow = Mat::zeros(Size(width_frame, height_frame), CV_32FC2);
    if ((!data_frame1->empty()) && (!data_frame2->empty()))
    {
//--- STEP 1 --- Calculate optical flow -------------------------------------//
        points1.clear();
        points2.clear();
        numKeypoints = 0;
        Mat fg1, fg2;
        cvtColor( *data_frame1, fg1, COLOR_BGR2GRAY );
        cvtColor( *data_frame2, fg2, COLOR_BGR2GRAY );
//        imshow( "fg1", fg1 );
//        imshow( "fg2", fg2 );
//        waitKey(20);
        //calcOpticalFlowFarneback( fg1, fg2, flow, 0.9, 1, 12, 2, 8, 1.7, 0 );    // OPTFLOW_FARNEBACK_GAUSSIAN
        //calcOpticalFlowFarneback( fg1, fg2, flow, 0.6, 4, 5, 2, 3, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN );
        //optflow::calcOpticalFlowSparseToDense( fg1, fg2, flow, 4, 128, 0.01f, true, 500.0f, 1.5f);
        optflow::calcOpticalFlowSparseToDense( fg1, fg2, flow, 4, 128, 0.01f, true, 500.0f, 1.5f);
        
        int win = 3;
        for (int y = 0; y < frame.rows; y += win) {
            for (int x = 0; x < frame.cols; x += win) {
                // get the flow from y, x position * 3 for better visibility
                const Point2f flowatxy = flow.at<Point2f>(y, x) * 1;
                // draw line at flow direction
                line(frame, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 200, 0));
                // draw initial point
                circle(frame, Point(x, y), 1, Scalar(0, 0, 0), -1);
                points1.push_back(Point2f(x, y));
                points2.push_back(Point2f((x + flowatxy.x), (y + flowatxy.y)));
                numKeypoints++;
            }
        }
        //imshow("RGB", points3D_BGR);
        //waitKey(10);
        
//--- STEP 2 --- Find essential matrix --------------------------------------//
        Essen_mask = Mat::ones( Size( 1, int(numKeypoints)), Essen_mask.type());
        E = findEssentialMat( points1, points2, K, RANSAC, 0.999, 3.0, Essen_mask );
        F = findFundamentalMat( points1, points2, FM_RANSAC, 3, 0.99);
//        correctMatches(F, points1, points2, points1, points2);
        
//--- STEP 3 --- Calculation of 3d points -----------------------------------//
        char maskf = 1;
        Mat p3D;
        p3D.release();
        points3D.release();
        points3D_BGR.clear();
        
        recoverPose(E, points1, points2, K, R, t, 15, Essen_mask, p3D);
        
        int k = 0;
        if ( maskf == 0) 
        {
            points3D = p3D.clone();
            for (int i = 0; i < points3D.cols; i++)
            {
                for (int j = 0; j < points3D.rows; j++) points3D.at< double >(j, i) /= points3D.at< double >(3, i);
                points3D_BGR.push_back( Scalar( (data_frame1->at<Vec3b>(points1.at(static_cast<size_t>(i)))[0] + 
                                                 data_frame2->at<Vec3b>(points2.at(static_cast<size_t>(i)))[0] ) / 2,
                                                (data_frame1->at<Vec3b>(points1.at(static_cast<size_t>(i)))[1] + 
                                                 data_frame2->at<Vec3b>(points2.at(static_cast<size_t>(i)))[1] ) / 2,
                                                (data_frame1->at<Vec3b>(points1.at(static_cast<size_t>(i)))[2] + 
                                                 data_frame2->at<Vec3b>(points2.at(static_cast<size_t>(i)))[2] ) / 2 ) );
            }
        }
        else 
        {
            int numNoZeroMask = countNonZero(Essen_mask);
            points3D = Mat::zeros( 4, numNoZeroMask, p3D.type() );
            for (int i = 0; i < p3D.cols; i++)
            {
                if ( Essen_mask.at< uchar >(i) == 1 )
                {
                    for (int j = 0; j < points3D.rows; j++)
                    {
                        points3D.at< double >(j, k) = p3D.at< double >(j, i) / p3D.at< double >(3, i);
                    }
                    points3D_BGR.push_back( Scalar( (data_frame1->at<Vec3b>(points1.at(static_cast<size_t>(i)))[0] + 
                                                     data_frame2->at<Vec3b>(points2.at(static_cast<size_t>(i)))[0] ) / 2,
                                                    (data_frame1->at<Vec3b>(points1.at(static_cast<size_t>(i)))[1] + 
                                                     data_frame2->at<Vec3b>(points2.at(static_cast<size_t>(i)))[1] ) / 2,
                                                    (data_frame1->at<Vec3b>(points1.at(static_cast<size_t>(i)))[2] + 
                                                     data_frame2->at<Vec3b>(points2.at(static_cast<size_t>(i)))[2] ) / 2 ) );
                    k++;
                }
            }
        }
        
        FileStorage SFM_Result;
        SFM_Result.open("SFM_Result_opticflow.txt", FileStorage::WRITE);
        SFM_Result << "E" << E;
        SFM_Result << "F" << F;
        //SFM_Result << "points1" << points1;
        //SFM_Result << "points2" << points2;
        SFM_Result << "K" << K;
        SFM_Result << "R" << R;
        SFM_Result << "t" << t;
        SFM_Result << "points3D" << points3D;
        SFM_Result << "Essen_mask" << Essen_mask;
        SFM_Result.release();
        cout << " --- SFM_Result written into file: SFM_Result_opticflow.txt" << endl;
        
        //resize( frame, frame, Size(640, 480), 0, 0, INTER_LINEAR );
        //imshow("SFM-result", frame);
        imwrite("SFM-result.jpg", frame);
        waitKey(10);
    }
    else
    {   
        cout << " --- Not enough frames" << endl;
    }
}


void SFM_Reconstruction::opticalFlow(Mat *f, Mat *fc, int win, int vecS)
{
    (*f).copyTo(img2Original);
    //img2Original = *f;
    flow = Mat(img2Original.cols, img2Original.rows, CV_32FC2);
    cvtColor(*f, frameGREY, COLOR_BGR2GRAY);
    cvtColor(*fc, frameCacheGREY, COLOR_BGR2GRAY);
    //calcOpticalFlowFarneback(frameGREY, frameCacheGREY, flow, 0.9, 1, 12, 2, 8, 1.7, 0);//OPTFLOW_FARNEBACK_GAUSSIAN
    optflow::calcOpticalFlowSparseToDense(frameGREY, frameCacheGREY, flow, 8, 128, 0.05f, true, 500.0f, 1.5f);
    
    
    //cv::normalize(flow, flow, 1, 0, NORM_L2, -1, noArray());
    
    for (int y = 0; y < img2Original.rows; y += win) {
        for (int x = 0; x < img2Original.cols; x += win) {
            // get the flow from y, x position * 3 for better visibility
            const Point2f flowatxy = flow.at<Point2f>(y, x) * vecS;
            // draw line at flow direction
            line(img2Original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255, 200, 0));
            // draw initial point
            circle(img2Original, Point(x, y), 1, Scalar(0, 0, 0), -1);
        }
    }
}

void SFM_Reconstruction::destroyWinSFM()
{
    namedWindow("SFM-result", WINDOW_AUTOSIZE);
    destroyWindow("SFM-result");
    namedWindow("OpticalFlow", WINDOW_AUTOSIZE);
    destroyWindow("OpticalFlow");
}

// SIFT Find matches between points
unsigned long SFM_Reconstruction::Match_find_SIFT( vector< KeyPoint > kpf1,
                                                   vector< KeyPoint > kpf2,
                                                   Mat dpf1,
                                                   Mat dpf2,
                                                   vector< KeyPoint > *gp1,
                                                   vector< KeyPoint > *gp2,
                                                   vector< DMatch > *gm)
{
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();
    vector<cv::DMatch> matches;
    cv::DMatch dist1, dist2;
 
    matcher->match(dpf1, dpf2, matches, cv::noArray());     // Matches key points of the frame with key points of the frame2

    // Sort match
    unsigned long temp = 0;
    unsigned long kn = static_cast<unsigned long>(matches.size());
    for (unsigned long i = 0; i < kn - 1; i++) { //key_num - 1
        dist1 = matches[i];
        for (unsigned long j = i + 1; j < kn; j++) {
            dist2 = matches[j];
            if (dist2.distance < dist1.distance) {
                dist1 = matches[j];
                temp = j;
            }
            if (j == kn - 1) {
                matches[temp] = matches[i];
                matches[i] = dist1;
                break;
            }
        }
    }
    //-- Вычисление максимального и минимального расстояния среди всех дескрипторов в пространстве признаков
    float max_dist = 0, min_dist = 100;
    for(size_t i = 0; i < matches.size(); i++ ) {
        float dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    //cout << "-- Max dist : " << max_dist << endl;
    //cout << "-- Min dist : " << min_dist << endl;
    // Selection of key points on both frames satisfying the threshold
    temp = 0;
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i].distance <  10 * min_dist) {   // threshold
            int new_i = static_cast<int>( gp1->size() );
            gp1->push_back( kpf1[ static_cast<unsigned long>( matches[i].queryIdx ) ] );    // queryIdx
            gp2->push_back( kpf2[ static_cast<unsigned long>( matches[i].trainIdx ) ] );    // trainIdx
            gm->push_back( cv::DMatch(new_i, new_i, 0) );
            temp++;
        }
    }
    
    cout << " --- Number of similar keypoints found : " << temp << endl;
    return temp;
}

void SFM_Reconstruction::drawKeyPoints( Mat *f, 
                                        vector< KeyPoint > *kp)
{
    vector< Point2f > drawPoint;
    KeyPoint::convert(*kp, drawPoint);
    RNG rng(12345);
    for (size_t i = 0; i < drawPoint.size(); i++) {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        circle(*f, drawPoint[i], 3, color, 1, LINE_8, 0);
        //putText( *f, to_string(i), drawPoint[i], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 2);
    }
}
