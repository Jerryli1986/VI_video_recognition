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

//**************parameter********************//
string modelpath = "B:/SVMRBF_1channels.yml";

//*******************************************//

struct DataSet { std::string filename;int label; };
Mat src;
int thresh = 80;
RNG rng(12345);
int m = 1;
int k = 0;
string doc = "B:/classimage/" + to_string(k) + "/";
int emptyframecount = 0;
vector<DataSet> dataList;
typedef std::vector<std::string> stringvec;
// declare functions
//int detectobj(Mat & img, Mat & drawing);
//int saveimg(Mat & img, int detectobj, string classname, int m);
vector<DataSet> datalists();
void computeHOG(vector<Mat> &inputCells, vector<vector<float> > &outputHOG);
void ConvertVectortoMatrix(std::vector<std::vector<float> > &ipHOG, Mat & opMat);
void ConvertlabeltoMat(vector<int>  &iplabel, Mat &oplabel);
void SVMtrain(Mat &trainMat, Mat &trainLabels);

void read_directory(const std::string& name, stringvec& v);

int s = 0;
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

string labels[] = { "Baby","Dog","Dinosaur","Coffee_Tin","Mug",
"Car","Camera","Keyboard","Koala","Blackberry",
"Diet_Coke_Bottle","Duck","Dragon","Android" };

int main(int argc, char * argv[])
{
	
	vector<Mat> trainCells;
	vector<Mat> testCells;
	vector<int> trainLabels;
	vector<int> testLabels;
	vector<DataSet> datalist = datalists();
	for (int l = 0;l < datalist.size();l++)
	{
		Mat img = imread(datalist[l].filename, 0);
		trainCells.push_back(img);
		trainLabels.push_back(datalist[l].label);
	};

	std::vector<std::vector<float> > trainHOG;
	computeHOG(trainCells, trainHOG);

	int descriptor_size = trainHOG[0].size();
	
	Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
   ConvertVectortoMatrix(trainHOG,trainMat);

	Mat labelsMat(datalist.size(), 1, CV_32SC1);
	ConvertlabeltoMat(trainLabels, labelsMat);
	// Train the SVM
	SVMtrain(trainMat, labelsMat);


	return 0;
}

vector<DataSet> datalists() {
	for (int j = 0; j < 14;j++)
	{
		std::vector<std::string> filenames;
		string folderName;
		folderName = "B:\\classimage\\" + to_string(j);
		read_directory(folderName, filenames);

		for (int jj = 0; jj < filenames.size();jj++)
		{
			DataSet tempDataset;
			tempDataset.filename = "B:/classimage/" + to_string(j)+"\/"+filenames[jj];
			tempDataset.label = j;
			dataList.push_back(tempDataset);
		} 
	};
	return dataList;

};
//  get the name of all files in folder
void read_directory(const std::string& name, stringvec& v)
{
	std::string pattern(name);
	pattern.append("\\*.jpg");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

void computeHOG(vector<Mat> &inputCells, vector<vector<float> > &outputHOG) {

	for (int y = 0; y<inputCells.size(); y++) {
		vector<float> descriptors;
		hog.compute(inputCells[y], descriptors);
		outputHOG.push_back(descriptors);
	}
}

void ConvertVectortoMatrix(std::vector<std::vector<float> > &ipHOG,Mat &opMat)
{
	
	int descriptor_size = ipHOG[0].size();
	
	for (int i = 0; i<ipHOG.size(); i++) {
		for (int j = 0; j<descriptor_size; j++) {
			opMat.at<float>(i, j) = ipHOG[i][j];
		}
	}
};

void ConvertlabeltoMat(vector<int>  &iplabel, Mat &oplabel)
{

	for (int i = 0; i<iplabel.size(); i++) 
	{
	
	  oplabel.at<int>(i, 0) = iplabel[i];
	}
};

void SVMtrain(Mat &trainMat, Mat &labelsMat) {
	Ptr<SVM> svm = SVM::create();
	svm->setGamma(0.001);
	svm->setC(100);
	svm->setKernel(SVM::RBF);
	svm->setType(SVM::C_SVC);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainMat, ROW_SAMPLE, labelsMat);
	svm->save(modelpath);
}