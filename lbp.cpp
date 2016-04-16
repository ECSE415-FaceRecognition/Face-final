// Project

//includes
#include "lbp.h"
#include "util.h"

#include <opencv2/opencv.hpp> 
#include <opencv2/nonfree/nonfree.hpp> 

#include <fstream>
#include <iomanip>
#include "tinydir.h"

#undef max

//namespaces
using namespace cv;
using namespace std;


double computePixelLBP(const Mat input);
Mat computeLBP(const Mat input);
Mat computeImageLBP(const Mat input, int patchNumber);
vector<Mat> getSpatialPyramidHistogram(const Mat input, int levels);

void lbp_train(std::vector<std::vector<std::string> > const& people, std::vector<std::vector<LBPData> > &histograms, int levels) {
	/* train */
	for (unsigned int i=0; i < people.size(); i++) {
		std::vector<LBPData> each_person;
		for (unsigned int j = 0; j < people[i].size(); j++) {
			std::string name = people[i][j];
			// open image
			cv::Mat im = cv::imread(name);

			//convert to greyScale
			cv::cvtColor(im, im, CV_RGB2GRAY);

			//compute spatial pyramid histogram for number of level
			LBPData tmp;
			tmp.hist = getSpatialPyramidHistogram(im, levels);
			tmp.name = name;

			each_person.push_back(tmp);
		}
		
		histograms.push_back(each_person);
		std::cout << "Testing Person: " << i << std::endl;
	}
}

string lbp_test(vector<Mat> test_person, vector<string> const& people, vector<vector<LBPData> > &histograms, int levels) {	
	
	/* person 1 */
	double best = std::numeric_limits<double>::max();
	string guess;
	for (unsigned int i = 0; i<histograms.size(); i++) {
		for (unsigned int j = 0; j<histograms[i].size(); j++) {
            /* calculate modified chi squared distance */
			vector<double>levelDistances(levels);
			
            /* compare all histograms on a per level basis */
			for (int lvl = 0; lvl < levels; lvl++){
				levelDistances[lvl] = compareHist(test_person[lvl], histograms[i][j].hist[lvl], CV_COMP_CHISQR);
			}

			/* compute sum */
			double sum = 0;
			for (int s = 1; s < levels; s++){
				sum = sum + levelDistances[s] / (pow(2, (levels - 1 - s + 1)));
			}
			
            /* compute final distance */
			double diff = levelDistances[0] / (pow(2, (levels - 1))) + sum;

			if (diff < best) {
				best = diff;
				guess = histograms[i][j].name;
			}
		}
	}

    /* find proper person to return. I avoid returning file names since lengths and the tilts/pans
     * differ too much 
     */
	for (string person : people) {
		if (guess.find(person) != string::npos) {
			return person;
		}
	}
	return "NotFound";
}

//functions by Linus
//access HPID by pose - return path of images and annotations based on tilt and pan angle -- 1st vector = image, 2nd vector = annotations
vector<vector<string> > get_image_Path_hpid(string tilt, string pan){
	//define variables
	vector<vector<string> > output(2);
	vector <string> names;
	vector <string> annotation;
	//define filename looking for
	string lookingForJPG = tilt + pan + ".jpg";
	string lookingForTXT = tilt + pan + ".txt";

	//go trough all IDs
	for (int i = 1; i < 16; i++) {
		//set path to current person
		stringstream idNumber;
		idNumber << std::setfill('0') << std::setw(2) << i;
		string path = string(POSE_DIR) + "Person" + idNumber.str() + "/";

		//get image name for all specified pose
		tinydir_dir dir;
		tinydir_open(&dir, path.c_str());
		while (dir.has_next)
		{
			tinydir_file file;
			tinydir_readfile(&dir, &file);
			if (file.is_reg)
			{
				if (file.name[0] != '.') {
					//only get jpgs with the desired angles		
					if (string(file.name).find(lookingForJPG) != string::npos)
					{
						names.push_back(path + string(file.name));
					}
					if (string(file.name).find(lookingForTXT) != string::npos)
					{
						annotation.push_back(path + string(file.name));
					}

				}
			}
			tinydir_next(&dir);
		}
	}
	//return found filenames
	output[0] = names;
	output[1] = annotation;
	return output;
}

//get annotation for all images of a certain pose
vector <Rect> get_Rect_Image_hpid(string tilt, string pan){
	vector <Rect> output;
	//get Path of image files
	vector<vector<string> > imagePath = get_image_Path_hpid(tilt, pan);

	//iterate through every name
	for (unsigned int i = 0; i < imagePath[1].size(); i++){
		//load annotation file and extract center points
		ifstream file(imagePath[1][i]);
		string str;
		string annotationFile;
		int count = 0;
		int centerX;
		int centerY;
		while (getline(file, str))
		{
			if (count == 3){
				centerX = stoi(str);
				//check if outside of boundary, if so shift
				if (50 > centerX){
					centerX = 50;
				}
				if (334 < centerX){
					centerX = 334;
				}

			}
			if (count == 4){
				centerY = stoi(str);
				//check if outside of boundary, if so shift
				if (50 > centerY){
					centerY = 50;
				}
				if (238 < centerY){
					centerY = 238;
				}
			}
			count++;
		}
		//get rectangle
		Rect faceRect(centerX - 50, centerY - 50, 100, 100);
		output.push_back(faceRect);
	}
	return output;
}

//get all images of a certain pose
vector <Mat> get_Image_hpid(string tilt, string pan){

	vector <Mat> output;
	//get Path of image files
	vector<vector<string> > imagePath = get_image_Path_hpid(tilt, pan);

	//iterate through every name
	for (unsigned int i = 0; i < imagePath[0].size(); i++){
		//load image
		Mat currentImage = imread(imagePath[0][i]);

		//convert to grayscale and push into array
		cvtColor(currentImage, currentImage, CV_RGB2GRAY);
		output.push_back(currentImage);
	}
	return output;
}

//display image for all 65 poses based on ID and series
Mat displayPoseImages(int id, int series){
	//create output
	Mat stichedImage;

	//define tilt and pan for valid images
	vector<string> tilt = { "+30", "+15", "+0", "-15", "-30" };
	vector <string> pan = { "+90", "+75", "+60", "+45", "+30", "+15", "+0", "-15", "-30", "-45", "-60", "-75", "-90" };

	//for all tilts
	for (unsigned int i = 0; i < tilt.size(); i++) {
		//for all pans
		Mat imageRow;
		for (unsigned int j = 0; j < pan.size(); j++) {
			//get images for this pose
			//load images for this angle
			vector <Mat> loadedPoses = get_Image_hpid(tilt[i], pan[j]);

			//load rectangle for this angle
			vector <Rect> loadedRect = get_Rect_Image_hpid(tilt[i], pan[j]);

			//find array position for person and series
			int position = (id)*series - 1;

			//extract image and rectangel
			Mat image = loadedPoses[position];
			Rect rectFace = loadedRect[position];

			//draw rectangle on image
			rectangle(image, rectFace, 255, 2, 8, 0);

			//append horizontally
			if (j == 0){
				imageRow = image;
			}
			else{
				hconcat(imageRow, image, imageRow);
			}
		}
		//append vertically
		stichedImage.push_back(imageRow);
	}

	//resize image
	int resizeFactor = 3;
	resize(stichedImage, stichedImage, Size(stichedImage.cols / 3, stichedImage.rows / 3));
	//return 		
	return stichedImage;
}

//get array of size: 21 poses x # of images containing all training images mapped into coarse pose groups
vector <vector<Mat> > getPoseEstimationTrainImages(){
	//create pose estimation training array
	vector <vector<Mat> > poseTrainingImages;

	//define tilt and pan to map poses into coarse pose classes
	vector <vector<int> > tilt(3);
	vector <vector<int> > pan(7);
	tilt[0] = { 60, 70 };
	tilt[1] = { 80, 90, 100 };
	tilt[2] = { 110, 120 };
	pan[0] = { 0, 10 };
	pan[1] = { 20, 30, 40 };
	pan[2] = { 50, 60, 70 };
	pan[3] = { 80, 90, 100 };
	pan[4] = { 110, 120, 130 };
	pan[5] = { 140, 150, 160 };
	pan[6] = { 170, 180 };

	//for all coarse tilts
	int count = 0;
	for (unsigned int i = 0; i < tilt.size(); i++) {

		//for all coarse pans
		for (unsigned int k = 0; k < pan.size(); k++) {
			vector<Mat> poseImages;

			//for all fine tilts
			for (unsigned int j = 0; j < tilt[i].size(); j++) {

				//for all fine pans
				for (unsigned int m = 0; m < pan[k].size(); m++) {

					//get name list
					vector<string> names = getQmulNames();

					//iterate through every name to get the specific pose of every person
					//error in qmul function, set to 10 for now!
					//for (int n = 0; n < names.size(); n++) {
					for (int n = 0; n < 10; n++) {

						//load file
						Mat image = imread(get_image_qmul(names[n], tilt[i][j], pan[k][m]));

						//convert to greyscale
						cvtColor(image, image, CV_RGB2GRAY);

						//pushback into poseImages vector
						poseImages.push_back(image);
					}
				}
			}
			//pushback into final vector
			poseTrainingImages.push_back(poseImages);
		}
	}
	return poseTrainingImages;
}

//get array of size: 21 poses x # of images containing all testing images mapped into coarse pose groups
vector <vector<Mat> > getPoseEstimationTestingImages(){
	//create pose estimation testing array
	vector <vector<Mat> > poseTestingImages;

	//define tilt and pan to map poses into coarse pose classes
	vector <vector<string> > tilt(3);
	tilt[0] = { "-30" };
	tilt[1] = { "-15", "+0", "+15" };
	tilt[2] = { "+30" };
	vector <string> pan = { "-90", "-60", "-30", "+0", "+30", "+60", "+90" };

	//for all coarse tilts
	for (unsigned int i = 0; i < tilt.size(); i++) {
		//for all pans
		for (unsigned int j = 0; j < pan.size(); j++) {
			vector<Mat> sameposeTestImages;

			//for all fine tilts
			for (unsigned int k = 0; k < tilt[i].size(); k++) {
				//load images for this angle
				vector <Mat> loadedPoses = get_Image_hpid(tilt[i][k], pan[j]);

				//load rectangle for this angle
				vector <Rect> loadedRect = get_Rect_Image_hpid(tilt[i][k], pan[j]);

				//append every image in this array to poseTestImages
				for (unsigned int n = 0; n < loadedPoses.size(); n++) {
					Mat currentImg = loadedPoses[n];

					//crop image using rect
					currentImg = currentImg(loadedRect[n]);

					//pushback
					sameposeTestImages.push_back(loadedPoses[n]);
				}
			}
			//push back
			poseTestingImages.push_back(sameposeTestImages);
		}
	}
	//return 		
	return poseTestingImages;
}

//LBP computation functions
vector<Mat> getSpatialPyramidHistogram(const Mat input, int levels){
	vector<Mat> spatialHistogram(levels);
	//compute histogram for each level
	for (int i = 1; i < levels + 1; i++){
		//calculate histogram of current level
		Mat currentLevelHist = computeImageLBP(input, i);

		//normalize
		normalize(currentLevelHist, currentLevelHist, 1, NORM_L2);

		//concatenate to complete Spatial Pyramid Histogram
		spatialHistogram[i - 1].push_back(currentLevelHist);
	}

	//return full spatial pyramid histogram
	return spatialHistogram;
}


Mat computeImageLBP(const Mat input, int patchNumber){
	Mat imageHistogram;
	vector <Mat> imageSplit(patchNumber*patchNumber);
	//find size of images
	int patchWidth = input.cols / patchNumber;
	int patchHeight = input.rows / patchNumber;

	//split image
	int k = 0;
	for (int y = 0; y < patchNumber; y++){
		for (int x = 0; x < patchNumber; x++){
			imageSplit[k] = input(Rect(x*patchWidth, y*patchHeight, patchWidth, patchHeight));
			k++;
		}
	}

	//compute histogram for each image and concate to image Histogram;

	for (int i = 0; i < patchNumber*patchNumber; i++){
		//calculate local histogram
		Mat currentHist = computeLBP(imageSplit[i]);

		//normalize
		normalize(currentHist, currentHist, 1, NORM_L2);

		//concatenate to complete Image Histogram
		imageHistogram.push_back(currentHist);
	}

	//return image Histogram
	return imageHistogram;

}


Mat computeLBP(const Mat input){
	//compute LBP for every pixel in input matrix
	Mat LBPimage(input.size(), CV_64F);

	for (int x = 1; x < input.cols - 1; x++){
		for (int y = 1; y < input.rows - 1; y++){
			//extract surrounding 3x3 matrix for every pixel
			Mat pixelNeighbors = input(Rect(x - 1, y - 1, 3, 3));
			LBPimage.at<double>(y, x) = computePixelLBP(pixelNeighbors);
		}
	}

	//show image
	LBPimage.convertTo(LBPimage, CV_8U);
	//imshow("LBP image", LBPimage);
	//waitKey(0);

	//compute histogram
	Mat histogram;
	int nbins = 59; // 59 bins
	int histSize[] = { nbins }; //one dimension
	float range[] = { 0, 255 }; //up to 255 value
	const float *ranges[] = { range };
	int channels[] = { 0 };
	calcHist(&LBPimage, 1, channels, Mat(), histogram, 1, histSize, ranges);

	//return histogram
	return histogram;
}


double computePixelLBP(const Mat input){
	//compute LBP value for pixel
	Mat pixel;

	input.convertTo(pixel, CV_32S);
	//only for testing
	//Mat pixel = (Mat_<int>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);	
	//testing end

	int center = pixel.at<int>(1, 1);
	vector<int> LBPvec;

	//check every pixel and create 8bit array, starting at left top corner of matrix
	LBPvec.push_back(!(center < pixel.at<int>(0, 0)));
	LBPvec.push_back(!(center < pixel.at<int>(0, 1)));
	LBPvec.push_back(!(center < pixel.at<int>(0, 2)));
	LBPvec.push_back(!(center < pixel.at<int>(1, 2)));
	LBPvec.push_back(!(center < pixel.at<int>(2, 2)));
	LBPvec.push_back(!(center < pixel.at<int>(2, 1)));
	LBPvec.push_back(!(center < pixel.at<int>(2, 0)));
	LBPvec.push_back(!(center < pixel.at<int>(1, 0)));

	//check if there are more than two 0-1 or 1-0 transitions
	int transitions = 0;
	//check for every element but last
	for (unsigned int i = 0; i < LBPvec.size() - 1; i++){
		if (LBPvec[i + 1] - LBPvec[i] != 0)
			transitions = transitions + 1;
	}
	//check for first and last element
	if (LBPvec[0] - LBPvec[LBPvec.size() - 1] != 0){
		transitions = transitions + 1;
	}
	//compute LVP value
	double LVPvalue = 0;
	//if transitions are 2 or less, compute the LBP value, otherwise LVPvalue remains 0
	if (transitions <= 2){
		for (unsigned int i = 0; i < LBPvec.size(); i++){
			if (LBPvec[i] == 1){
				LVPvalue = LVPvalue + pow(2, (double)i);
			}
		}
	}
	//cout << LVPvalue << "\n";
	//return LVP value
	return LVPvalue;

}

//compute confusion matrix for LBP for a certain number of levels
Mat getLBPConfusionMatrix(int levels){
	//LBP Histograms for Pose Estimation
	levels = levels + 1;

	//create pose Estimation training dataset
	vector <vector<Mat> > poseEstimationTrainingImages = getPoseEstimationTrainImages();

	//create pose Estimation testing dataset
	vector <vector<Mat> > poseEstimationTestingImages = getPoseEstimationTestingImages();
	cout << "All images loaded! \n";
	//create LBP spatial pyramid histograms for all training images
	vector <vector<vector<Mat> >> trainingHistograms(21);
	//run through all images in the 21 poses and create histogram	
	for (unsigned int i = 0; i < 21; i++){
		//for (unsigned int j = 0; j < poseEstimationTrainingImages[i].size(); j++){
			//for faster run
		for (int j = 0; j < 5; j++){
			trainingHistograms[i].push_back(getSpatialPyramidHistogram(poseEstimationTrainingImages[i][j], levels));
		}
	}
	cout << "Training Histogram DONE! \n";
	//create LBP spatial pyramid histograms for all testing images
	vector <vector<vector<Mat> >> testingHistograms(21);
	//run through all images in the 21 poses and create histogram	
	for (unsigned int i = 0; i < 21; i++){
		//for (unsigned int j = 0; j < poseEstimationTestingImages[i].size(); j++){
			//for faster run
		for (int j = 0; j < 5; j++){
			testingHistograms[i].push_back(getSpatialPyramidHistogram(poseEstimationTestingImages[i][j], levels));
		}
	}
	cout << "Testing Histograms DONE! \n";

	//find pose and create confusion matrix
	int numberOfPoses = 21;
	Mat confusionMatrix(21, 21, CV_64F, Scalar(0));

	//go through all poses
	for (int i = 0; i<numberOfPoses; i++) {
		//go through all test images for that pose
		for (unsigned int j = 0; j < testingHistograms[i].size(); j++) {
			vector<Mat> currentTestHistogram = testingHistograms[i][j];

			//matched pose for current image
			int matchedPose;

			//set inital distance to infinity
			double smallestDistance = numeric_limits<double>::infinity();

			//for each trained subject, compare each of the 21 poses with all of the images
			for (int k = 0; k < numberOfPoses; k++) {
				for (unsigned int u = 0; u < trainingHistograms[k].size(); u++) {
					//find distance
					vector<double>levelDistances(levels);
					//compare all histograms on a per level basis
					for (int lvl = 0; lvl < levels; lvl++){
						levelDistances[lvl] = compareHist(currentTestHistogram[lvl], trainingHistograms[k][u][lvl], CV_COMP_CHISQR);
					}
					//calculate weighted distance
					//compute sum
					double sum = 0;
					for (int s = 1; s < levels; s++){
						sum = sum + levelDistances[s] / (pow(2, (levels - 1 - s + 1)));
					}
					//compute final distance
					double distance = levelDistances[0] / (pow(2, (levels - 1)));

					//check if this is the smallest distance yet
					if (distance < smallestDistance){
						//enter new distance
						smallestDistance = distance;
						//set as matched pose
						matchedPose = k;
					}
				}
			}
			//add for this image the result to confusion matrix - real poses are column, matched are row
			confusionMatrix.at<double>(matchedPose, i) = confusionMatrix.at<double>(matchedPose, i) + 1;
		}
	}
	//normalize confusion matrix
	//go through every row
	for (int row = 0; row < confusionMatrix.rows; row++) {
		//sum up every item in this row
		double sum = 0;
		for (int item = 0; item < confusionMatrix.cols; item++) {
			sum = sum + confusionMatrix.at<double>(row, item);
		}
		//devide every element in this row by this number sum is not zero
		if (sum != 0){
			confusionMatrix.row(row) = confusionMatrix.row(row) / sum;
		}
	}
	//display confusion matrix
	return confusionMatrix;
}

void linus_main()
{
	//show head pose data base
	Mat headPoseImage = displayPoseImages(15, 2);
	imshow("Head Pose Example", headPoseImage);
	waitKey(0);

	//compute confusion matrix - set level = 2
	Mat LBPConfusionMatrix = getLBPConfusionMatrix(2);
	cout << LBPConfusionMatrix;

	//wait to close console
	getchar();
}


