//#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance);

int main(int argc, char** argv)
{
    double vis_mult = 1.0;
    double sigma = 1.5;
    double lambda = 8000.0;
    int wsize = 3;
    int max_disp = 96;

    float scale = 0.5f;

    Mat imgl,imgr;

    VideoCapture cap1("udpsrc port=5002 ! application/x-rtp,media=video,payload=26,encoding-name=JPEG,framerate=30/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink " ,CAP_GSTREAMER);
    VideoCapture cap2("udpsrc port=5001 ! application/x-rtp,media=video,payload=26,encoding-name=JPEG,framerate=30/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink " ,CAP_GSTREAMER);

    if(!cap1.isOpened())
    {
        printf("Capture failure\n");
        return 1;
    }
    if(!cap2.isOpened())
    {
        printf("Capture failure\n");
        return 1;
    }

    const char* intrinsic_filename = "./intrinsics.yml";
    const char* extrinsic_filename = "./extrinsics.yml";

    FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", intrinsic_filename);
        return -1;
    }

    Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;

    M1 *= scale;
    M2 *= scale;

    fs.open(extrinsic_filename, CV_STORAGE_READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file %s\n", extrinsic_filename);
        return -1;
    }

    Rect roi1, roi2;
    Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;

    /*if(!grabber.getFrames(imgl,imgr))
    {
        cout << "blad kamer" << endl;
        return -1;
    }*/
    //while(!grabber.getFrames(imgl,imgr));
    cap1 >> imgl;
    cap2 >> imgr;

    if (scale != 1.f)
    {
        /*Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(imgl, temp1, Size(), scale, scale, method);
        imgl = temp1;
        resize(imgr, temp2, Size(), scale, scale, method);
        imgr = temp2;*/
    }

    Size img_size = imgl.size();
    

    Mat map11, map12, map21, map22, Q;

    stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);


    int numDispTrack = 6;
    int blockSizeTrack = 1;
    int cn = imgl.channels();

    int blockSize = blockSizeTrack*2 + 1;//nieparzysta
    int p1 = 8*cn*blockSize*blockSize;
    int p2 = 32*cn*blockSize*blockSize;
    int minDisp = 0;
    int numOfDisp = numDispTrack*16; // podzielne przez 16
    int uniqRat = 5;
    int speckWinSize = 100;
    int speckRange = 32;
    int dispMaxDiff = 17;
    int preFilCap = 63;
    
    Ptr<StereoSGBM> sgbm = StereoSGBM::create();

    sgbm->setMode(StereoSGBM::MODE_SGBM);
    sgbm->setPreFilterCap(preFilCap);
    sgbm->setBlockSize(blockSize);
    sgbm->setP1(p1);
    sgbm->setP2(p2);
    sgbm->setMinDisparity(minDisp);
    sgbm->setNumDisparities(numOfDisp);
    sgbm->setUniquenessRatio(uniqRat);
    sgbm->setSpeckleWindowSize(speckWinSize);
    sgbm->setSpeckleRange(speckRange);
    sgbm->setDisp12MaxDiff(dispMaxDiff);

    Mat img1r, img2r;
    Mat left_disp;
    Mat filtered_disp;
    Rect ROI;
    Ptr<DisparityWLSFilter> wls_filter;


    ROI = computeROI(imgl.size(),sgbm);

    wls_filter = createDisparityWLSFilterGeneric(false);
    wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*wsize));
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);

    Mat raw_disp_vis;
    Mat filtered_disp_vis;

    char key;

    namedWindow("raw disparity", WINDOW_NORMAL);
    resizeWindow("raw disparity", 640, 480);
    namedWindow("filtered disparity", WINDOW_NORMAL);
    resizeWindow("filtered disparity", 640, 480);
    while(key != 27)
    {
        cap1 >> imgl;
        cap2 >> imgr;

        if(imgl.empty())  
            break;
        if(imgr.empty())  
            break;

        if (scale != 1.f)
        {
            /*Mat temp1, temp2;
            int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
            resize(imgl, temp1, Size(), scale, scale, method);
            imgl = temp1;
            resize(imgr, temp2, Size(), scale, scale, method);
            imgr = temp2;*/
        }

        remap(imgl, img1r, map11, map12, INTER_LINEAR);
        remap(imgr, img2r, map21, map22, INTER_LINEAR);

        imgl = img1r;
        imgr = img2r;

        sgbm->compute(imgl,imgr,left_disp);
        wls_filter->filter(left_disp,imgl,filtered_disp,Mat(),ROI);

        getDisparityVis(left_disp,raw_disp_vis,vis_mult);
        
        imshow("raw disparity", raw_disp_vis);

        getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
        
        imshow("filtered disparity", filtered_disp_vis);

        key = waitKey(30);
    }

    return 0;
}

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance)
{
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size/2;
    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width + minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}
