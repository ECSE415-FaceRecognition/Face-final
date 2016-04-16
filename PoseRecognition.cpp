// Project 5 - LBP for Pose Estimation
// Code written by Linus Lehnert -260482309

//includes
#include "stdafx.h"
#include <opencv2/opencv.hpp> 
#include <opencv2/nonfree/nonfree.hpp> 

#include <fstream>
#include <iomanip>
#include "tinydir.h"


//namespaces
using namespace cv;
using namespace std;

//function prototypes
double computePixelLBP(const Mat input);
Mat computeLBP(const Mat input);
Mat computeImageLBP(const Mat input, int patchNumber);

//set path for folder locations
const char QMUL_DIR[] = "QMUL/";
const char POSE_DIR[] = "HeadPoseImageDatabase/";

//functions by Jit 

//get  image path for QMUL database
string get_image_qmul(string person, int tilt, int pose) {
	stringstream s, tilt_ss, pose_ss;
	tilt_ss << setfill('0') << setw(3) << tilt;
	pose_ss << setfill('0') << setw(3) << pose;

	s << QMUL_DIR << person << "/" << person.substr(0, person.size() - 4) << "_" << tilt_ss.str() << "_" << pose_ss.str() << ".ras";
	return s.str();
}

//get QMUL name list
vector<string> getQmulNames(){
	tinydir_dir dir;
	tinydir_open(&dir, QMUL_DIR);

	std::vector<std::string> people;

	// populate peopls with everyone from the QMUL dataset
	while (dir.has_next)
	{
		tinydir_file file;
		tinydir_readfile(&dir, &file);
		if (file.is_dir)
		{
			if (file.name[0] != '.') {
				people.push_back(file.name);
			}
		}
		tinydir_next(&dir);
	}
	return people;
}

//end of function by Jit

//access HPID by pose - return path of images and annotations based on tilt and pan angle -- 1st vector = image, 2nd vector = annotations
vector<vector<string>> get_image_Path_hpid(string tilt, string pan){
	//define variables
	vector<vector<string>> output(2);
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
	vector<vector<string>> imagePath = get_image_Path_hpid(tilt, pan);

	//iterate through every name
	for (int i = 0; i < imagePath[1].size(); i++){
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
	vector<vector<string>> imagePath = get_image_Path_hpid(tilt, pan);

	//iterate through every name
	for (int i = 0; i < imagePath[0].size(); i++){
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
	for (int i = 0; i < tilt.size(); i++) {
		//for all pans
		Mat imageRow;
		for (int j = 0; j < pan.size(); j++) {
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
	int resizeFactor = 1;
	resize(stichedImage, stichedImage, Size(stichedImage.cols / resizeFactor, stichedImage.rows / resizeFactor));
	//return 		
	return stichedImage;
}

////display image for given subject
Mat displayQmulImages(string subject){
	//create output
	Mat stichedImage;

	//define tilt and pan for images images
	vector<int> tilt = { 60, 70, 80, 90, 100, 110, 120 };
	vector <int> pan = { 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180 };

	//for all tilts
	for (int i = 0; i < tilt.size(); i++) {
		//for all pans
		Mat imageRow;
		for (int j = 0; j < pan.size(); j++) {
			//get images for this pose
			//load images for this angle
			Mat image = imread(get_image_qmul(subject, tilt[i], pan[j]));

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
	int resizeFactor = 1;
	resize(stichedImage, stichedImage, Size(stichedImage.cols / resizeFactor, stichedImage.rows / resizeFactor));
	//return 		
	return stichedImage;

}

//get array of size: 21 poses x # of images containing all training images mapped into coarse pose groups
vector <vector<Mat>> getPoseEstimationTrainImages(){
	//create pose estimation training array
	vector <vector<Mat>> poseTrainingImages;

	//define tilt and pan to map poses into coarse pose classes
	vector <vector<int>> tilt(3);
	vector <vector<int>> pan(7);
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
	for (int i = 0; i < tilt.size(); i++) {

		//for all coarse pans
		for (int k = 0; k < pan.size(); k++) {
			vector<Mat> poseImages;

			//for all fine tilts
			for (int j = 0; j < tilt[i].size(); j++) {

				//for all fine pans
				for (int m = 0; m < pan[k].size(); m++) {

					//get name list
					vector<string> names = getQmulNames();

					//iterate through every name to get the specific pose of every person
					//for (int n = 0; n < 10; n++) {
					for (int n = 0; n < names.size(); n++) {
					

						//load file
						Mat image = imread(get_image_qmul(names[n], tilt[i][j], pan[k][m]));
						cout << get_image_qmul(names[n], tilt[i][j], pan[k][m]) << "\n";
						//convert to greyscale
						cvtColor(image, image, CV_RGB2GRAY);

						//pushback into poseImages vector
						poseImages.push_back(image);

						//imshow("Test", image);
						//waitKey(0);
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
vector <vector<Mat>> getPoseEstimationTestingImages(){
	//create pose estimation testing array
	vector <vector<Mat>> poseTestingImages;

	//define tilt and pan to map poses into coarse pose classes
	vector <vector<string>> tilt(3);
	tilt[0] = { "+30" };
	tilt[1] = { "+15", "+0", "-15" };
	tilt[2] = { "-30" };
	vector <string> pan = { "+90", "+60", "+30", "+0", "-30", "-60", "-90" };

	//for all coarse tilts
	for (int i = 0; i < tilt.size(); i++) {
		//for all pans
		for (int j = 0; j < pan.size(); j++) {
			vector<Mat> sameposeTestImages;

			//for all fine tilts
			for (int k = 0; k < tilt[i].size(); k++) {
				//load images for this angle
				vector <Mat> loadedPoses = get_Image_hpid(tilt[i][k], pan[j]);

				//load rectangle for this angle
				vector <Rect> loadedRect = get_Rect_Image_hpid(tilt[i][k], pan[j]);

				//append every image in this array to poseTestImages
				for (int n = 0; n < loadedPoses.size(); n++) {
					Mat currentImg = loadedPoses[n];
					//imshow("Image", currentImg);
					//waitKey(0);
					//crop image using rect
					currentImg = currentImg(loadedRect[n]);

					//pushback
					sameposeTestImages.push_back(currentImg);
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
	for (int i = 0; i < LBPvec.size() - 1; i++){
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
		for (int i = 0; i < LBPvec.size(); i++){
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
	vector <vector<Mat>> poseEstimationTrainingImages = getPoseEstimationTrainImages();

	//create pose Estimation testing dataset
	vector <vector<Mat>> poseEstimationTestingImages = getPoseEstimationTestingImages();
	
	cout << "All images loaded! \n";
	
	//create LBP spatial pyramid histograms for all training images
	vector <vector<vector<Mat>>> trainingHistograms(21);
	//run through all images in the 21 poses and create histogram	
	for (int i = 0; i < 21; i++){
		for (int j = 0; j < poseEstimationTrainingImages[i].size(); j++){
			//for faster run
			//for (int j = 0; j < 5; j++){
			trainingHistograms[i].push_back(getSpatialPyramidHistogram(poseEstimationTrainingImages[i][j], levels));
			cout << "Pose" << i << "Image:" << j << "Done \n";
		}
	}
	cout << "Training Histogram DONE! \n";
	//create LBP spatial pyramid histograms for all testing images
	vector <vector<vector<Mat>>> testingHistograms(21);
	//run through all images in the 21 poses and create histogram	
	for (int i = 0; i < 21; i++){
		for (int j = 0; j < poseEstimationTestingImages[i].size(); j++){
			//for faster run
			//for (int j = 0; j < 5; j++){
			testingHistograms[i].push_back(getSpatialPyramidHistogram(poseEstimationTestingImages[i][j], levels));
			cout << "Pose" << i << "Image:" << j << "Done \n";
		}
	}
	cout << "Testing Histograms DONE! \n";

	//find pose and create confusion matrix
	int numberOfPoses = 21;
	Mat confusionMatrix(21, 21, CV_64F, Scalar(0));

	//go through all 21 poses
	for (int i = 0; i<numberOfPoses; i++) {
		//go through all test images for that pose
		for (int j = 0; j < testingHistograms[i].size(); j++) {
			vector<Mat> currentTestHistogram = testingHistograms[i][j];

			//matched pose for current image
			int matchedPose;

			//set inital distance to infinity
			double smallestDistance = numeric_limits<double>::infinity();

			//for each trained subject, compare each of the 21 poses with all of the images
			for (int k = 0; k < numberOfPoses; k++) {
				for (int u = 0; u < trainingHistograms[k].size(); u++) {
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
					double distance = sum + levelDistances[0] / (pow(2, (levels - 1)));

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


//compute confusion matrix for BOW for a certain number of codewords
Mat getBowConfusionMatrix(int numCodewords){

	// load openCV non free
	initModule_nonfree();

	//init codebook for all 21 poses
	vector<Mat> codeBook(21);

	// init training keypoints
	vector<vector<vector <KeyPoint>>> trainSiftKeys;

	//create training histograms
	vector <vector<Mat>> bowTrainHistograms(21);

	//create pose Estimation training dataset
	vector <vector<Mat>> poseEstimationTrainingImages = getPoseEstimationTrainImages();

	//create pose Estimation testing dataset
	vector <vector<Mat>> poseEstimationTestingImages = getPoseEstimationTestingImages();


	//run through all images in the 21 poses and get Sift descriptor	
	for (int i = 0; i < 21; i++){

		//create object SIFT_detector
		Ptr<FeatureDetector> SIFT_detector = FeatureDetector::create("SIFT");

		//create SIFT descriptor extractor object
		Ptr<DescriptorExtractor> SIFT_extractor = DescriptorExtractor::create("SIFT");

		//Mat object called D for storing all SIFT descriptors
		Mat siftDescript;
		vector<vector<KeyPoint>> currentPoseSift;
		//do SIFT on every image 
		for (int j = 0; j < poseEstimationTrainingImages[i].size(); j++){

			//init keypoints
			vector<KeyPoint> currentSifts;

			//detect keypoints and put into siftKeys
			SIFT_detector->detect(poseEstimationTrainingImages[i][j], currentSifts);

			//pushpack keypoints
			currentPoseSift.push_back(currentSifts);

			//compute SIFT descriptor of remaining keypoints
			Mat currentDescriptor;
			SIFT_extractor->compute(poseEstimationTrainingImages[i][j], currentSifts, currentDescriptor);

			siftDescript.push_back(currentDescriptor);
		}
		//pushback keypoints
		trainSiftKeys.push_back(currentPoseSift);

		//compute codebook
		cout << "Building Codebook........" << i << "!\n";

		//create bag of words trainer object
		BOWKMeansTrainer bagTrainer = BOWKMeansTrainer(numCodewords, TermCriteria(), 1, KMEANS_PP_CENTERS);

		//add all descriptors
		bagTrainer.add(siftDescript);

		//compute the codebook using k means clustering and store into codeBook
		codeBook[i] = bagTrainer.cluster();
		cout << "DONE!\n";
	}

	//got now all 21 codebooks!

	//run through all images in the 21 poses and get Sift descriptor	
	for (int i = 0; i < 21; i++){
		//delete for real
		//for (int i = 0; i < 5; i++){
		//end delete
		//create object SIFT_detector
		Ptr<FeatureDetector> SIFT_detector = FeatureDetector::create("SIFT");

		//create SIFT descriptor extractor object
		Ptr<DescriptorExtractor> SIFT_extractor = DescriptorExtractor::create("SIFT");

		//create matcher
		Ptr<DescriptorMatcher> theMatcher = DescriptorMatcher::create("BruteForce");

		//create BOW extractor
		Ptr<BOWImgDescriptorExtractor> theBower(new BOWImgDescriptorExtractor(SIFT_extractor, theMatcher));

		//set codebook
		theBower->setVocabulary(codeBook[i]);

		//compute histogram for every image
		for (int j = 0; j < poseEstimationTrainingImages[i].size(); j++){
			//compute Histogram
			Mat bowHistogram;
			theBower->compute2(poseEstimationTrainingImages[i][j], trainSiftKeys[i][j], bowHistogram);

			//normalize bow histogram
			normalize(bowHistogram, bowHistogram, 1, NORM_L2);

			//store histogram into imageDescriptors
			bowTrainHistograms[i].push_back(bowHistogram);

			//output message
			cout << "Histogram for Category: " << (i) << " - Image: " << (j) << " finished! \n";
		}
	}
	//finished training function, got all histograms for training in 21 poses

	//start testing set generation
	vector<vector<vector<Mat>>> bowTestHistograms;


	//run through all 21 codebooks
	for (int i = 0; i < 21; i++){
		//for each codebook

		//create objects
		Ptr<FeatureDetector> SIFT_detectorTest = FeatureDetector::create("SIFT");
		Ptr<DescriptorExtractor> SIFT_extractorTest = DescriptorExtractor::create("SIFT");
		Ptr<DescriptorMatcher> theMatcherTest = DescriptorMatcher::create("BruteForce");
		Ptr<BOWImgDescriptorExtractor> theBowerTest(new BOWImgDescriptorExtractor(SIFT_extractorTest, theMatcherTest));

		//Set codebook of the bag of wors descriptor extractor object
		theBowerTest->setVocabulary(codeBook[i]);
		vector <vector<Mat>>bowTestHistogramsCodeBook;
		//loop for each test image possible
		//loop all categories
		for (int i = 0; i < 21; i++){

			//image histograms
			vector<Mat>bowTrainImageHist;

			for (int j = 0; j < poseEstimationTestingImages[i].size(); j++){

				//detect SIFT keypoints
				vector<KeyPoint> testKeyPoints;
				SIFT_detectorTest->detect(poseEstimationTestingImages[i][j], testKeyPoints);

				//compute Bag of words histogram representation
				Mat curTestBowHist;
				theBowerTest->compute2(poseEstimationTestingImages[i][j], testKeyPoints, curTestBowHist);

				//output message
				cout << "Pose" << i << "Image:" << j << "Done \n";

				//pushback histogram
				bowTrainImageHist.push_back(curTestBowHist);
			}
			//pushback into codebook
			bowTestHistogramsCodeBook.push_back(bowTrainImageHist);
		}
		//pushback into codebook histogram
		bowTestHistograms.push_back(bowTestHistogramsCodeBook);
	}

	//Training completed message
	cout << "TRAINING COMPLETED!\n";

	//testing done
	//start comparison of training and testing images------------------

	//find pose and create confusion matrix
	int numberOfPoses = 21;
	Mat confusionMatrix(21, 21, CV_64F, Scalar(0));

	//go through all 21 poses
	for (int i = 0; i<numberOfPoses; i++) {
		//go through all test images for that pose
		for (int j = 0; j < bowTrainHistograms[i].size(); j++) {
			//set current training histogram
			Mat currentTestHistogram = bowTrainHistograms[i][j];

			//matched pose for current image
			int matchedPose;

			//set inital distance to infinity
			double smallestDistance = numeric_limits<double>::infinity();

			//for each trained subject, compare each of the 21 codesbook with the each 21 poses with all of the images
			//for each bose
			for (int x = 0; x < bowTestHistograms.size(); x++) {
				//for each codebook
				for (int k = 0; k < bowTestHistograms[x].size(); k++) {
					//for each image
					for (int u = 0; u < bowTestHistograms[x][k].size(); u++) {
						//find distance
						double distance = compareHist(currentTestHistogram, bowTestHistograms[x][k][u], CV_COMP_CHISQR);

						//check if this is the smallest distance yet
						if (distance < smallestDistance){
							//enter new distance
							smallestDistance = distance;
							//set as matched pose
							matchedPose = x;
						}
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



void main()
{
	//show qmul database image
	Mat qmulImage = displayQmulImages("YongminYGrey");
	imshow("Qmul Example", qmulImage);
	//imwrite("QMUL EXAMPLE.png", qmulImage);
	waitKey(0);
	

	//show head pose database image
	Mat headPoseImage = displayPoseImages(15, 2);
	imshow("Head Pose Example", headPoseImage);
	//imwrite("Head Pose Example.png", headPoseImage);
	waitKey(0);

	//compute confusion matrix - set level = 4
	Mat LBPConfusionMatrix = getLBPConfusionMatrix(4);
	cout << LBPConfusionMatrix;

	//compute confusion matrix for BOW
	Mat BowConfusionMatrix = getBowConfusionMatrix(300);
	cout << BowConfusionMatrix;
	

	//wait to close console
	getchar();
}



