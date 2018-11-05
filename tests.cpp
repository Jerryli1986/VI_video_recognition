#include "freenect-playback-wrapper.h"
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <sys/stat.h>
#include "opencv2/opencv.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>

using namespace cv::ml;
using namespace cv;
using namespace std;
Mat translateImg(Mat &img, int offsetx, int offsety);
int main(int argc, char * argv[])
{
// import the video frame
//FreenectPlaybackWrapper wrap("C:/Users/Daisy/Desktop/Set2/Set2");
	
cv::Mat currentRGB;
cv::Mat currentDepth;
cv::Mat currentThreshold;
cv::Mat img1;
cv::Mat img2;
cv::Mat Results(cv::Size(640, 640), CV_8UC3);  ///"B:/elephen.jpg"
currentRGB = imread("B:/Dog_145.png", CV_LOAD_IMAGE_UNCHANGED);//CV_LOAD_IMAGE_COLOR

// Create the RGB and Depth Windows
cv::namedWindow("RGB", CV_WINDOW_AUTOSIZE);
cv::namedWindow("img1", CV_WINDOW_AUTOSIZE);
cv::namedWindow("img2", CV_WINDOW_AUTOSIZE);

	 // resize(chop)
    //resize(currentRGB, img1, Size(640, 480));
    Rect roi(150, 70, 350, 250);
	//rectangle(currentRGB, roi, (100, 255, 100), 2, 8, 0);
	//img1.copyTo(img2);
	//translateImg(img2, -30, 20);
	
	cv::Mat RGBchannels[4];
	vector<Mat> RGBDchannels;
	cv::Mat imgtosave;
	cv::Mat imgcrop;
	split(currentRGB, RGBchannels);
	// normalize
	normalize(RGBchannels[0], RGBchannels[0], 0, 255, NORM_MINMAX, -1, Mat());
	normalize(RGBchannels[1], RGBchannels[1], 0, 255, NORM_MINMAX, -1, Mat());
	normalize(RGBchannels[2], RGBchannels[2], 0, 255, NORM_MINMAX, -1, Mat());
	normalize(RGBchannels[3], RGBchannels[3], 0, 255, NORM_MINMAX, -1, Mat());
	//Histogram Equalization
	//It is a method that improves the contrast in an image, in order to stretch out the intensity range
	equalizeHist(RGBchannels[0], RGBchannels[0]);
	equalizeHist(RGBchannels[1], RGBchannels[1]);
	equalizeHist(RGBchannels[2], RGBchannels[2]);
	equalizeHist(RGBchannels[3], RGBchannels[3]);
	// merge channels
	RGBDchannels.push_back(RGBchannels[0]);
	RGBDchannels.push_back(RGBchannels[1]);
	RGBDchannels.push_back(RGBchannels[2]);
	translateImg(RGBchannels[3], -35, 30);
	RGBDchannels.push_back(RGBchannels[3]);
	merge(RGBDchannels, imgcrop);
	
	// crop image and resize
	resize(imgcrop(roi), imgtosave, Size(640, 480));
	imwrite("B:/imgsave1.png", imgtosave);
	//cv::imshow("Depth", currentDepth);
	cv::imshow("RGB", currentRGB);
	cv::imshow("img1", imgtosave(roi));
	cv::imshow("img2", imgtosave);
	cv::waitKey(0);


return 0;
}

Mat translateImg(Mat &img, int offsetx, int offsety) {
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	warpAffine(img, img, trans_mat, img.size());
	return img;
}
/* https://stackoverflow.com/questions/19068085/shift-image-content-with-opencv/26766505#26766505 */