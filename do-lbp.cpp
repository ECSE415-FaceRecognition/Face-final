#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "lbp.h"
#include "tinydir.h"
#include "util.h"

// allow numeric limits to work
#undef max

void do_lbp_face_recognition(std::vector<std::string> const& people);
void seven_fold_cv(std::vector<std::string> &people, std::vector<std::vector<cv::string> > &folds);
void qmul_all_images_of_person(std::string person);

int main()
{
	//load image and set directory - just for testing
	tinydir_dir dir;
	tinydir_open(&dir, QMUL_DIR);

	std::vector<std::string> people = getQmulNames();

	do_lbp_face_recognition(people);

    return 0;
}

void do_lbp_face_recognition(std::vector<std::string> const& people_tmp) {

	/* N people by X images per person */
	std::vector<std::vector<LBPData> > histograms;

	std::vector<std::vector<cv::string> > image_names = open_all_qmul_by_person(people_tmp);
	std::vector<std::vector<LBPData> > folds;

    /* open a file for logging */
	std::ofstream lbp_face_log;
	lbp_face_log.open("lbp_face_log.txt");

	/* get lbp histrograms of all images.
     * when needed, simply pass a copy to the data 
     */
	lbp_train(image_names, histograms, MAX_LEVELS);

	/* split into 7 training sets, preserving the order of people */
	folds.resize(NUM_FOLDS);
	for (int i = 0; i < histograms.size(); i++) {
		srand(0xDEADBEEF);	// make test deterministic
		seven_fold_cv(histograms[i], folds);
	}

    /* set of folds to use for training */
	std::vector<std::vector<LBPData> > training_images;
    /* set of folds to use for testing */
	std::vector<LBPData> testing_images;
	
    /* calculate LBP over levels up to MAX_LEVELS */
	for (int level = 1; level <= MAX_LEVELS; level++) {
		double average_rate = 0;
        /* variable to keep track of fold */
		int which_fold;
		/* create a set of training images from 6 of the 7 folds */
		for (int x = 0; x < NUM_FOLDS; x++) {
			training_images.clear();
			for (int fold=0; fold < (NUM_FOLDS-1); fold++) {
				which_fold = (fold + x) % NUM_FOLDS;
				training_images.push_back(folds[which_fold]);
			}
			which_fold = (NUM_FOLDS-1 + x) % NUM_FOLDS;
			testing_images = folds[which_fold];

			int guessed_correct = 0;

			/* run recognition for testing subsample */
			for (int i=0; i < testing_images.size(); i++) {
                /* get name and historam laters */
				std::vector<cv::Mat> test_person = testing_images[i].hist;  //  TODO set as reference
				std::string test_name = testing_images[i].name;
				std::string guessed = lbp_test(test_person, people_tmp, training_images, level);
			
				/* determine if lbp guessed properly. test_name is a file name, so we simply
					search in the file name for the guessed person */
				if (test_name.find(guessed) != std::string::npos) {
					guessed_correct++;
					//std::cout << "guessed " << guessed_correct << "correct" << std::endl;
				}
			}

			/* log the recognition rate */
			double rate = (((double)guessed_correct) / ((double)folds[(NUM_FOLDS - 1 + x) % NUM_FOLDS].size()));
			lbp_face_log << "For level " << level << " iteration " << x << "guessed" << guessed_correct << " of " << folds[(NUM_FOLDS - 1 + x) % NUM_FOLDS].size() 
				<< "with a rate of " << rate << std::endl;
			average_rate += rate;
		}
		std::cout << "Overall rate was " << average_rate / NUM_FOLDS << " for level " << level << std::endl;
	}
	system("pause");
}

/* open all qmul images of a particular person, as specified in the assignment
 * description
 */
void qmul_all_images_of_person(std::string person) {
	std::vector<std::string> tmp;
	tmp.push_back(person);

	auto open_test = open_all_qmul_by_person(tmp);

	const int index = 0;
	for (int i=0; i < open_test[index].size(); i++) {
		cv::Mat img = cv::imread(open_test[index][i]);
		cv::imshow(open_test[index][i], img);
	}

	cv::waitKey();

}
