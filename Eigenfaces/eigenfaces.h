#pragma once
#ifndef _EIGENFACE__DEF_
#define _EIGENFACE__DEF_
//includes
#include <iostream>
#include <opencv2/opencv.hpp>
//namespaces
using namespace std;
using namespace cv;
//functions
Mat train(vector<Mat> faces);
int test(Mat &candidate);

#endif
