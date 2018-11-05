#include "freenect-playback-wrapper.h"
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <sys/stat.h>


using namespace cv::ml;
using namespace cv;
using namespace std;

//**************parameter********************//
string videopath = "C:/Users/Daisy/Desktop/Set1/Set1";

//*******************************************//

struct DataSet { std::string filename;int label;};

int thresh = 80;
RNG rng(12345);
int m=1;
int k = 0;
string doc= "B:/classimage/" + to_string(k) + "/";
int emptyframecount = 0;
vector<DataSet> dataList;

// declare functions
int detectobj(Mat & in_img, Mat & out_img1, Mat & out_img2);
int saveimg(Mat & img, int detectobj, string classname, int m);
vector<DataSet> datalists();
Mat translateImg(Mat &img, int offsetx, int offsety);
void imgpreprocess(Mat & RGBimg, Mat & Thresimg, Mat & imgcrop);
int s=0;

string labels[] ={"Baby","Dog","Dinosaur","Coffee_Tin","Mug",
                 "Car","Camera","Keyboard","Koala","Blackberry",
	             "Diet_Coke_Bottle","Duck","Dragon","Android"};


int main(int argc, char * argv[])
{  
	 // import the video frame
	FreenectPlaybackWrapper wrap(videopath);
	Mat src;
	cv::Mat currentRGB;
	cv::Mat currentDepth;
	cv::Mat currentThreshold;
	cv::Mat currentContour;
	cv::Mat Results(cv::Size(640, 640), CV_8UC3);
	
	Rect roi(150, 100, 350, 250);
	// Create the RGB and Depth Windows
	cv::namedWindow("Results", WINDOW_NORMAL);
	char key = '0';
	uint8_t status = 255;
	// generate training data and save in the folder "classimage"
	
	while (key != 27 && status != 0)
	{
		status = wrap.GetNextFrame();
		if (status & State::UPDATED_RGB)
			currentRGB = wrap.RGB;
		// depth image
		if (status & State::UPDATED_DEPTH)
		{
			//depthpreprocessing
			currentDepth = wrap.Depth;
			
			threshold(currentDepth, currentThreshold,80,255, THRESH_BINARY);
			currentThreshold.copyTo(src);
			currentContour = Mat::zeros(currentDepth.rows, currentDepth.cols, currentDepth.type());
			int de = detectobj(src, currentContour,currentRGB);

			if (de==1 )
			{
				cv::Mat imgtosave;
				imgpreprocess(currentRGB, src, imgtosave);
				m = saveimg(imgtosave, de, labels[k-1], m);
				emptyframecount=0;
			}
			else
				emptyframecount++;
           // create next class training data folder path
			if (emptyframecount>=50 && s==0)
			{
				 doc = "B:/classimage/" + to_string(k) + "/";
				 k++;
			}
			if (emptyframecount < 50)
			{
				s = 0;
			}
			else
			{
				s = emptyframecount;
			}
		}
		
		cv::Mat RGB_mat;
		cv::Mat Depth_mat;
		cv::Mat Threshold_mat;
		cv::Mat Contour_mat;

		currentRGB.copyTo(Results(cv::Rect(0, 0, 640, 480)));
		
		resize(currentDepth, Depth_mat, Size(currentDepth.cols / 3, currentDepth.rows / 3));
		cvtColor(Depth_mat, Depth_mat, COLOR_GRAY2BGR);
		Depth_mat.copyTo(Results(cv::Rect(0, 480, Depth_mat.cols, Depth_mat.rows)));

		resize(currentThreshold, Threshold_mat, Size(currentThreshold.cols / 3, currentThreshold.rows / 3));
		cvtColor(Threshold_mat, Threshold_mat, COLOR_GRAY2BGR);
		Threshold_mat.copyTo(Results(cv::Rect( currentDepth.cols / 3, 480, currentThreshold.cols / 3, currentThreshold.rows / 3)));

		
		resize(currentContour, Contour_mat, Size(currentContour.cols / 3, currentContour.rows / 3));
		cvtColor(Contour_mat, Contour_mat, COLOR_GRAY2BGR);
		Contour_mat.copyTo(Results(cv::Rect(currentDepth.cols / 3 + currentThreshold.cols / 3, 480, currentContour.cols / 3, currentContour.rows / 3)));
		
		cv::imshow("Results", Results);
		key = cv::waitKey(2);
	}
	
	return 0;
}


int detectobj(Mat & in_img, Mat & out_img1,Mat & out_img2)
{
	int N=0;
	
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
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(out_img1, contours, i, color, 2, 8, hierarchy, 0, Point());
			//rectangle(out_img2, boundingRect(contours[i]), Scalar(0, 255, 0), 2, 8, 0);
			N++;
		}
	}
	if (N > 0)
		return 1;
	else
		return 0;
}

int saveimg(Mat & img, int detectobj,string classname,int m)
{
	if (detectobj == 1)
	{
		m++;

		cv::imwrite(doc + classname+"_"+to_string(m) + ".png", img);
	}
	return m;
}


vector<DataSet> datalists() {
	for (int j = 0; j < 14;j++)
	{
		std::vector<cv::String> filenames;
		string folderName;
		folderName = "B:/classimage/" + to_string(j);
		cv::String folder = folderName.c_str(); // converting from std::string->cv::String
		cv::glob(folder, filenames);
		for (int jj = 0; jj < filenames.size();jj++)
		{
			DataSet tempDataset;
			tempDataset.filename = static_cast<std::string>(filenames[jj]);
			tempDataset.label = j;
			dataList.push_back(tempDataset);
		}
	};
	return dataList;

};

void imgpreprocess(Mat & RGBimg,Mat & Thresimg,Mat & imgtosave)
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
