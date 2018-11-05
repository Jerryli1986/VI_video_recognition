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

/////**************Input parameter**************///////
string Videopath = "C:/Users/Daisy/Desktop/Set2/Set2";

string Modelpath = "B:/SVMRBF_4channels.yml";
string savefolder = "B:/predictimage/";

//**********************************************///


HOGDescriptor hog(
	Size(640, 480), //winSize
	Size(8, 8), //blocksize
	Size(8, 8), //blockStride,
	Size(8, 8), //cellSize,
	9, //nbins,
	1, //derivAper,
	-1, //winSigma,
	0, //histogramNormType,
	0.2, //L2HysThresh,
	0,//gammal correction,
	64,//nlevels=64
	1);

Ptr<SVM> svm= SVM::create();
std::vector<std::vector<float> > imgHOG;
vector<float> descriptors;
Mat src;
int thresh = 80;
RNG rng(12345);
int m = 1;
string doc ;
int emptyframecount = 0;
string labels[] = { "Baby","Dog","Dinosaur","Coffee_Tin","Mug",
"Car","Camera","Keyboard","Koala","Blackberry",
"Diet_Coke_Bottle","Duck","Dragon","Android" };

// declare functions
int detectobj(Mat & in_img, Mat & out_img1, Mat & out_img2);
int saveimg(Mat & img, int detectobj, string classname, int m);
void imgtext(Mat & frame, int response);
Mat translateImg(Mat &img, int offsetx, int offsety);
void imgpreprocess(Mat & RGBimg, Mat & Thresimg, Mat & imgcrop);
int main(int argc, char * argv[])
{
	
	FreenectPlaybackWrapper wrap(Videopath);

	cv::Mat currentRGB;
	cv::Mat currentDepth;
	cv::Mat currentThreshold;
	cv::Mat currentContour;
	cv::Mat Results(cv::Size(640, 640), CV_8UC3);
	// Create the RGB and Depth Windows
	cv::namedWindow("Results", WINDOW_NORMAL);

	char key = '0';

	uint8_t status = 255;
	
	// svm loading
	svm = SVM::load(Modelpath);
	while (key != 27 && status != 0)
	{
		status = wrap.GetNextFrame();
		if (status & State::UPDATED_RGB)
			currentRGB = wrap.RGB;
		if (status & State::UPDATED_DEPTH)
		{
			//preprocessing
			currentDepth = wrap.Depth;
			threshold(currentDepth, currentThreshold, 80, 255, THRESH_BINARY);
			currentThreshold.copyTo(src);
			//using contour to detect object
			currentContour = Mat::zeros(currentDepth.rows, currentDepth.cols, currentDepth.type());

			int de = detectobj(src, currentContour, currentRGB);
			if (de == 1)
			{
				cv::Mat imgtosave;
				cv::Mat imggrey;
				imgpreprocess(currentRGB, src, imgtosave);
				
				// testing

				hog.compute(imggrey, descriptors);
				int response = svm->predict(descriptors);

				imgtext(currentRGB, response);
				doc = savefolder + to_string(response) + "/";
				m = saveimg(currentRGB, de, labels[response], m);
			}
			
		}

		// Show the images in one windows
		cv::Mat RGB_mat;
		cv::Mat Depth_mat;
		cv::Mat Threshold_mat;
		cv::Mat Contour_mat;
		cv::Mat Gray_mat;

		currentRGB.copyTo(Results(cv::Rect(0, 0, 640, 480)));

		resize(currentDepth, Depth_mat, Size(currentDepth.cols / 3, currentDepth.rows / 3));
		cvtColor(Depth_mat, Depth_mat, COLOR_GRAY2BGR);
		Depth_mat.copyTo(Results(cv::Rect(0, 480, Depth_mat.cols, Depth_mat.rows)));

		resize(currentThreshold, Threshold_mat, Size(currentThreshold.cols / 3, currentThreshold.rows / 3));
		cvtColor(Threshold_mat, Threshold_mat, COLOR_GRAY2BGR);
		Threshold_mat.copyTo(Results(cv::Rect(currentDepth.cols / 3, 480, currentThreshold.cols / 3, currentThreshold.rows / 3)));

		resize(currentContour, Contour_mat, Size(currentContour.cols / 3, currentContour.rows / 3));
		cvtColor(Contour_mat, Contour_mat, COLOR_GRAY2BGR);
		Contour_mat.copyTo(Results(cv::Rect(currentDepth.cols / 3 + currentThreshold.cols / 3, 480, currentContour.cols / 3, currentContour.rows / 3)));

		cv::imshow("Results", Results);
		// Check for keyboard input
		key = cv::waitKey(10);
	}
	return 0;
}

int detectobj(Mat & in_img, Mat & out_img1, Mat & out_img2)
{
	int N = 0;

	////// contour
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	/// Detect edges using canny
	Canny(in_img, canny_output, thresh, thresh * 2, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	/// Get the moments
	vector<Moments> mu(contours.size());
	///  Get the mass centers:
	vector<Point2d> mc(contours.size());
	/// centre of image
	vector<Point2d> img_c(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);

		mu[i] = moments(contours[i], false);
		mc[i] = Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);

		img_c[i] = Point2d(320, 240);
		if (area > 10 && area < 150 && norm(mc[i] - img_c[i]) < 150)
		{
			Scalar color = Scalar(rng.uniform(0,  255), rng.uniform(0,255), rng.uniform(0, 255));
			drawContours(out_img1, contours, i, color, 2, 8, hierarchy, 0, Point());
			rectangle(out_img2, boundingRect(contours[i]), Scalar(0, 255, 0), 2, 8, 0);
			N++;
		}
	}
	if (N > 0)
		return 1;
	else
		return 0;
}

int saveimg(Mat & img, int detectobj, string classname, int m)
{
	if (detectobj == 1)
	{
		m++;
		cv::imwrite(doc + classname + "_" + to_string(m) + ".png", img);
	}
	return m;
}

void imgtext(Mat & frame, int response)
{
	switch (response) {
	case 0:
	{
		putText(frame, "Baby", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 1:
	{
		putText(frame, "Dog", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 2:
	{
		putText(frame, "Dinosaur", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 3:
	{
		putText(frame, "Coffee_Tin", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 4:
	{
		putText(frame, "Mug", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 5:
	{
		putText(frame, "Car", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 6:
	{
		putText(frame, "Camera", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 7:
	{
		putText(frame, "Keyboard", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 8:
	{
		putText(frame, "Koala", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 9:
	{
		putText(frame, "Blackberry", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 10:
	{
		putText(frame, "Coke_Bottle", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 11:
	{
		putText(frame, "Duck", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 12:
	{
		putText(frame, "Dragon", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	case 13:
	{
		putText(frame, "Android", Point2f(100, 200), FONT_HERSHEY_COMPLEX_SMALL, 2, Scalar(151, 220, 100), 1, CV_AA);
		break;
	}
	default: printf(" ");
	}
}

void imgpreprocess(Mat & RGBimg, Mat & Thresimg, Mat & imgtosave)
{
	//
	cv::Mat RGBchannels[3];
	vector<Mat> RGBDchannels;
	cv::Mat imgcrop;
	split(RGBimg, RGBchannels);
	//normalize
	normalize(RGBchannels[0], RGBchannels[0], 0, 255, NORM_MINMAX, -1, Mat());
	normalize(RGBchannels[1], RGBchannels[1], 0, 255, NORM_MINMAX, -1, Mat());
	normalize(RGBchannels[2], RGBchannels[2], 0, 255, NORM_MINMAX, -1, Mat());
	normalize(Thresimg, Thresimg, 0, 255, NORM_MINMAX, -1, Mat());
	//Histogram Equalization
	equalizeHist(RGBchannels[0], RGBchannels[0]);
	equalizeHist(RGBchannels[1], RGBchannels[1]);
	equalizeHist(RGBchannels[2], RGBchannels[2]);
	equalizeHist(Thresimg, Thresimg);
	// merge channels
	RGBDchannels.push_back(RGBchannels[0]);
	RGBDchannels.push_back(RGBchannels[1]);
	RGBDchannels.push_back(RGBchannels[2]);
	translateImg(Thresimg, -35, 30);
	RGBDchannels.push_back(Thresimg);
	merge(RGBDchannels, imgcrop);
	// crop image and resize
	Rect roi(150, 70, 350, 250);
	resize(imgcrop(roi), imgtosave, Size(640, 480));
}
Mat translateImg(Mat &img, int offsetx, int offsety) {
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	warpAffine(img, img, trans_mat, img.size());
	return img;
}
/* https://stackoverflow.com/questions/19068085/shift-image-content-with-opencv/26766505#26766505 */

