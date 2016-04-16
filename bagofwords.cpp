#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

const char QMUL_DIR[] = "QMUL/";

#include "tinydir.h"

using namespace cv;
using namespace std;

#include <iostream>
#include <fstream>
using namespace std;



template <typename T>
void seven_fold_cv(std::vector<T> &people, std::vector<std::vector<T> > &folds) {

	/* randomize the people vector */
	std::random_shuffle(people.begin(), people.end());

	/* split people into the 7 subsamples */
	int i = people.size();
	while (i > 0) {
		i--;
		folds[i%NUM_FOLDS].push_back(people[i]);
	}
}

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

string get_image_qmul(string person, int tilt, int pose) {
	stringstream s, tilt_ss, pose_ss;
	tilt_ss << setfill('0') << setw(3) << tilt;
	pose_ss << setfill('0') << setw(3) << pose;

	s << QMUL_DIR << person << "/" << person.substr(0, person.size() - 4) << "_" << tilt_ss.str() << "_" << pose_ss.str() << ".ras";
	return s.str();
}

vector<vector<string> > open_all_qmul_by_person(vector<string> people) {
	vector<vector<string> > names;
	for (unsigned int person = 0; person<people.size(); person++) {
		std::cout << people[person] << std::endl;
		vector<string> tmp;
		for (int tilt = 60; tilt <= 120; tilt += 10) {
			for (int pose = 0; pose <= 180; pose += 10) {
				tmp.push_back(get_image_qmul(people[person], tilt, pose));
			}
		}
		names.push_back(tmp);
	}
	return names;
}


struct BOWData {
	cv::Mat image;
	std::string name;
};
struct ProbData {
	cv::Mat mean;
	cv::Mat covar;
	std::string name;
};

struct ImgDec {
	Mat descriptor;
	string name;
};

const int NUM_FOLDS = 7;

/* Function prototypes */
void Train(std::vector<std::vector<BOWData>> &training_images, Mat &codeBook, vector<vector<ImgDec>> &imageDescriptors, const int numCodewords, Mat& D, Ptr<DescriptorExtractor> extractor, vector<vector<vector<KeyPoint>>> &imageKeypoints);
void Test(std::vector<std::vector<BOWData>> &testing_images, const Mat codeBook, vector<vector<ImgDec>> const& imageDescriptors, int num, vector<string> people);
void find_all_keypoints(std::vector<std::vector<BOWData>> &training_images, vector<vector<vector<KeyPoint>>> &imageKeypoints, Mat &D, Ptr<DescriptorExtractor> extractor);


//std::string do_lbp_prob_match(std::vector<std::vector<BOWData> > histograms, BOWData test, std::vector<std::string> const& people_tmp) {
//
//	int sz = histograms[0].size();
//	std::vector<probdata> gaussians;
//	std::vector<cv::mat> all_histogram_of_person;
//	/* fill `gaussians with the covar and mean of all images of one person */
//	for (auto &person : histograms) {
//		all_histogram_of_person.clear();
//		/* iterate through all images of a person */
//		for (auto &image : person) {
//			// std::cout << image.name << std::endl;
//			cv::mat tmp;
//			for (auto &level_hist : image.hist) {
//				tmp.push_back(level_hist);
//			}
//			all_histogram_of_person.push_back(tmp);
//		}
//
//		/* fill probdata struct */
//		probdata tmp;
//		tmp.name = person[0].name;
//		cv::calccovarmatrix(all_histogram_of_person, tmp.covar, tmp.mean, cv_covar_normal, 5);
//
//		/* store covar and mean of this person */
//		gaussians.push_back(tmp);
//	}
//
//	std::vector<cv::mat> test_vector;
//	probdata tmp;
//
//	cv::mat person;
//	tmp.name = test.name;
//	for (auto &level : test.hist) {
//		person.push_back(level);
//	}
//
//	//std::cout << "size of histogram" << person.size() << std::endl;
//
//	//    gaussians.push_back(tmp);
//
//	auto sz_covar = gaussians[0].covar.size();
//	auto sz_mean = gaussians[0].mean.size();
//	for (auto &gaussian : gaussians) {
//		if (gaussian.mean.size() != sz_mean) {
//			std::cout << gaussian.name << " has a strange mean size" << std::endl;
//		}
//		if (gaussian.covar.size() != sz_covar) {
//			std::cout << gaussian.name << " has a strange covar size" << std::endl;
//		}
//	}
//
//
//	// todo diagonalize a matrix ?? http://cs229.stanford.edu/section/gaussians.pdf ??
//	// todo compute gaussian difference
//
//	/*
//	* max(p(i|s)) => probability of i (a person) given s (a multivariate gaussian of a testing
//	* image)
//	*
//	* => how to get p(s|i) using covar and mean? (
//	* https://en.wikipedia.org/wiki/multivariate_normal_distribution#non-degenerate_case)
//	*/
//
//	cv::mat best; std::string best_name;
//	for (int i = 0; i < gaussians.size(); i++) {
//		cv::mat diff = (person - gaussians[i].mean);
//		cv::mat exponent = diff.t() * gaussians[i].covar.inv(cv::decomp_svd) * diff;
//
//		exponent = cv::abs(exponent);
//
//		if (best.empty() || (exponent.at<double>(0, 0) < best.at<double>(0, 0))) {
//			best = exponent;
//			best_name = gaussians[i].name;
//		}
//
//		//std::cout << "exponent is " <<  exponent << " for " << gaussians[i].name  <<  std::endl;
//		//std::cout << "result is   " <<  result << " for " << gaussians[i].name  <<  std::endl;
//	}
//
//	//std::cout << "actual: " << tmp.name << std::endl;
//	//std::cout << "guess: " << best_name << std::endl;
//
//	/*
//	* => use same method to get p(s) from testing image
//	* p(i|s) = p(s|i) / p(s)
//	* p(i|s) = p(s|i) / sum(...)
//	*/
//
//	for (auto &person : people_tmp) {
//		if (best_name.find(person) != std::string::npos) {
//			//std::cout << person << std::endl;
//			return person;
//		}
//	}
//
//	return "error";
//}



void main(void)
{
	/* Initialize OpenCV nonfree module */
	initModule_nonfree();

	/* Put the full path of the Caltech 101 folder here */
	const string datasetPath = "C:/Users/skanet1/vision/BagOfWords/dataset/Caltech 101";

	/* Set the number of training and testing images per category */
	const int numTrainingData = 40;
	const int numTestingData = 2;

	/* Set the number of codewords*/
	//const int numCodewords = 100; 
	int n_codewords[] = { /*10, 20 50, 100,  200, */ 300 /*, 400, 500, 600, 700, 800, 900, 1000*/ };
	/* Load the dataset by instantiating the helper class */


	auto names = getQmulNames();
	auto image_names = open_all_qmul_by_person(names);

	//image_names.resize(10);
	//for (int i = 0; i < image_names.size(); i++) {
	//	image_names[i].resize(14);
	//}

	std::vector<std::vector<BOWData>> all_images(image_names.size());

	int i = 0;
	//for (vector<string> &person : image_names) {
	//	for (string &image : person) {
	for (int i = 0; i < image_names.size(); i++) {
		std::cout << image_names[i][0] << std::endl;
		for (int j = 0; j < image_names[i].size(); j++) {
			BOWData tmp;
			tmp.name = image_names[i][j];
			tmp.image = imread(image_names[i][j]);
			all_images[i].push_back(tmp);
		}
		//i++;
	}

	std::vector<std::vector<BOWData> > folds;

	folds.resize(NUM_FOLDS);
	for (int i = 0; i < all_images.size(); i++) {
		srand(0xDEADBEEF);	// make test deterministic
		seven_fold_cv(all_images[i], folds);
	}


	int x = 0;
	int which_fold;
	std::vector<std::vector<BOWData>> training_images;
	std::vector<std::vector<BOWData>> testing_images;

	training_images.clear();
	for (int fold = 0; fold < (NUM_FOLDS - 1); fold++) {
		which_fold = (fold + x) % NUM_FOLDS;
		training_images.push_back(folds[which_fold]);
	}
	which_fold = 6;
	testing_images.push_back(folds[which_fold]);

	int guessed_correct = 0;

	///* run recognition for testing subsample */
	//for (int i = 0; i < testing_images.size(); i++) {
	//	/* get name and historam laters */
	//	std::vector<cv::Mat> test_person = testing_images[i].image;  //  TODO set as reference
	//	std::string test_name = testing_images[i].name;
	//	//std::string guessed = lbp_test(test_person, people_tmp, training_images, level);
	//	std::string guessed;
	//	/* determine if lbp guessed properly. test_name is a file name, so we simply
	//	search in the file name for the guessed person */
	//	if (test_name.find(guessed) != std::string::npos) {
	//		guessed_correct++;
	//		//std::cout << "guessed " << guessed_correct << "correct" << std::endl;
	//	}
	//}

	/* log the recognition rate */


	/* Variable definition */
	Mat codeBook;
	vector<vector<ImgDec>> imageDescriptors;
	vector<vector<vector<KeyPoint>>> imageKeypoints;

	imageDescriptors.resize(training_images.size());

	for (unsigned int i = 0; i<training_images.size(); i++) {
		imageDescriptors[i].resize(training_images[i].size());
	}

	Mat D;
	Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor;

	find_all_keypoints(training_images, imageKeypoints, D, extractor);
	for (int i = 0; i < sizeof(n_codewords); i++) {
		/* Training */
		std::cout << "Training " << n_codewords[i] << std::endl;
		Train(training_images, codeBook, imageDescriptors, n_codewords[i], D, extractor, imageKeypoints);

		/* Testing */
		std::cout << "Testing " << n_codewords[i] << std::endl;
		Test(testing_images, codeBook, imageDescriptors, n_codewords[i], names);
		std::cout << "Done running with " << n_codewords[i] << " codewords" << std::endl;
	}
	std::system("pause");
}

/* get keypoints */

void find_all_keypoints(std::vector<std::vector<BOWData>> &training_images, vector<vector<vector<KeyPoint>>> &imageKeypoints, Mat &D, Ptr<DescriptorExtractor> extractor) {
	Ptr<FeatureDetector> detector = new SiftFeatureDetector;
	vector<cv::KeyPoint> keypoints;

	imageKeypoints.resize(training_images.size());
	for (unsigned int cat = 0; cat < training_images.size(); cat++) {
		imageKeypoints[cat].resize(training_images[cat].size());
		for (unsigned int im = 0; im < training_images[cat].size(); im++) {
			// Get a reference to the rectangle and image

			Mat image = training_images[cat][im].image;
			Mat tmp;

			// detect keypoints
			detector->detect(image, keypoints);

			// compute SIFT features
			extractor->compute(image, keypoints, tmp);

			imageKeypoints[cat][im] = keypoints;
			D.push_back(tmp);
		}
	}
}

/* Train BoW */
void Train(std::vector<std::vector<BOWData>> &training_images, Mat &codeBook, vector<vector<ImgDec>> &imageDescriptors, const int numCodewords, Mat& D, Ptr<DescriptorExtractor> extractor, vector<vector<vector<KeyPoint>>> &imageKeypoints)
{
	Ptr<DescriptorMatcher> matcher = new BFMatcher;
	Ptr<BOWImgDescriptorExtractor> descriptor_extractor = new ::BOWImgDescriptorExtractor(extractor, matcher);

	BOWKMeansTrainer trainer(numCodewords);

	// Add descriptors to trainer
	trainer.add(D);
	codeBook = trainer.cluster();

	//std::cout << "Build Codebook" << std::endl;

	// Set Vocabulary
	descriptor_extractor->setVocabulary(codeBook);

	//std::cout << "Finding Bag of Words for images" << std::endl;
	//std::cout << "Testing for " << Dataset.trainingImages.size() << " Images" << std::endl;
	for (unsigned int cat = 0; cat < training_images.size(); cat++) {
		for (unsigned int im = 0; im < imageDescriptors[cat].size(); im++) {
			Mat const& img = training_images[cat][im].image;
			Mat out;
			vector<KeyPoint> &kpts = imageKeypoints[cat][im];
			descriptor_extractor->compute2(img, kpts, out);
			imageDescriptors[cat][im].descriptor = out;
			imageDescriptors[cat][im].name = training_images[cat][im].name;
		}
	}
}

/* Test BoW */

//void Test(std::vector<std::vector<BOWData>> &testing_images, const Mat codeBook, vector<vector<Mat>> const& imageDescriptors, int num)
void Test(std::vector<std::vector<BOWData>> &testing_images, const Mat codeBook, vector<vector<ImgDec>> const& imageDescriptors, int num, vector<string> people)
{
	Ptr<FeatureDetector> detector = new SiftFeatureDetector;
	Ptr<DescriptorExtractor> extractor = new SiftDescriptorExtractor;
	Ptr<DescriptorMatcher> matcher = new BFMatcher;

	Ptr<BOWImgDescriptorExtractor> descriptor_extractor = new BOWImgDescriptorExtractor(extractor, matcher);
	descriptor_extractor->setVocabulary(codeBook);

	vector<cv::KeyPoint> keypoints;
	int total_correct = 0, total = 0;
	//std::cout << "Test size: " << Dataset.testImages.size() << std::endl;
	for (unsigned int cat = 0; cat < testing_images.size(); cat++) {
		//std::cout << "Internal Size: " << Dataset.testImages[cat].size() << std::endl;
		for (unsigned int im = 0; im < testing_images[cat].size(); im++) {
			// Get a reference to the rectangle and image

			Mat image = testing_images[cat][im].image;
			Mat bag;
			// detect keypoints
			detector->detect(image, keypoints);

			descriptor_extractor->compute2(image, keypoints, bag);

			double min = DBL_MAX;
			int category = -1;
			int image_index = -1;
			// TODO, maybe wer're looping across the wrong variables
			for (unsigned int i = 0; i < imageDescriptors.size(); i++) {
				for (unsigned int j = 0; j < imageDescriptors[i].size(); j++) {
					double d = norm(bag, imageDescriptors[i][j].descriptor);/*, CV_COMP_CHISQR);*/
					if (d < min) {
						//std::cout << "Better Match match in category: " << i << std::endl;
						//cout << bag;
						min = d;
						category = i;
						image_index = j;
						//std::cout << "Testing image: " << testing_images[cat][im].name << " guessed image" << imageDescriptors[category][image_index].name << std::endl;
					}
				}

			}

			std::ostringstream os;

			// TODO Uncomment for part 2
			//std::cout << "Testing image: " << testing_images[cat][im].name << " guessed image" << imageDescriptors[category][image_index].name << std::endl;
			//std::cout << "Testing image: " << bag;

			//os << "test_image_" << cat << "_" << im << "_codewords_" << num << "_actual_" << testing_images[cat][0].name << "_guessed_ " << training_images[category][0].name << ".jpg";
			//imwrite(os.str(), image);
			//imshow(os.str(), image);
			//std::cout << "Best match in category: " << category << std::endl;

			int matched = 0;
			int unmatched = 0;

			for (int i = 0; i < people.size(); i++) {
				if (testing_images[cat][im].name.find(people[i]) != string::npos 
					&& imageDescriptors[category][image_index].name.find(people[i]) != string::npos) {
					total_correct++;
					if (matched < 2){
						if (matched == 1){
							ofstream myfile;
							myfile.open("Match1.txt");
							myfile << imageDescriptors[category][image_index].descriptor;
							myfile.close();
							matched++;
						}
						else {
							ofstream myfile;
							myfile.open("Match2.txt");
							myfile << imageDescriptors[category][image_index].descriptor;
							myfile.close();
							matched++;
						}
						
					}
					
				}
				else {
					if (unmatched < 2){	
						if (unmatched == 1){
							ofstream myfile;
							myfile.open("Notmatch1.txt");
							myfile << imageDescriptors[category][image_index].descriptor;
							myfile.close();
							unmatched++;
						}
						else {
							ofstream myfile;
							myfile.open("Notmatch2.txt");
							myfile << imageDescriptors[category][image_index].descriptor;
							myfile.close();
							unmatched++;
						}

					}
				}
			}
			total++;
		}
	}
	std::cout << "correctly guessed " << total_correct << " out of " << total << " images" << std::endl;
	std::cout << "rate was " << (double)total_correct / (double)total << std::endl;
}

