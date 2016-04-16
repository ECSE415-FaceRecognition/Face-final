#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>


#include <opencv2/opencv.hpp>

#include "eigenfaces.h"
#include "tinydir.h"


// allow numeric limits to work
#undef max

const char QMUL_DIR[] = "QMUL/";
const char POSE_DIR[] = "HeadPoseImageDatabase/"; 

std::string tilts[] = { "-90", "-60", "-30", "-15", "0", "+15", "+30", "+60", "+90" };
std::string pans[] = { "-90", "-75", "-60", "-45", "-30", "-15", "0", "+15", "+30", "+45", "+60", "+75", "+90" };


void eigenface_std(vector<Mat> &faces, Mat &test) {

	vector<int> labels;

	for (int i = 0; i < faces.size(); i++) {
		labels.push_back(i);
	}

	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->train(faces, labels);
	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(test);
	cout << "label predicted = " << predictedLabel << endl;
}


std::string get_image_pose(int id, int serie, int number, std::string tilt, std::string pan) {
	std::stringstream s, id_ss, number_ss;

	id_ss << std::setfill('0') << std::setw(2) << id;
	number_ss << std::setfill('0') << std::setw(2) << number;

	s << POSE_DIR << "Person" << id_ss.str() << "/" << "person" << id_ss.str() << serie << number_ss.str() << tilt << pan << ".jpg";
	return s.str();
}

std::string get_image_qmul(std::string person, int tilt, int angle) {
	std::stringstream s, tilt_ss, angle_ss;
	tilt_ss << std::setfill('0') << std::setw(3) << tilt;
	angle_ss << std::setfill('0') << std::setw(3) << angle;

	s << QMUL_DIR << person << "/" << person.substr(0, person.size() - 4) << "_" << tilt_ss.str() << "_" << angle_ss.str() << ".ras";
	return s.str();
}

std::vector<std::string> open_all_poses() {
	std::vector<std::string> names;
	for (int id = 0; id <= 15; id++) {
		for (int serie = 1; serie <= 2; serie++) {
			for (int number = 0; number <= 92; number++) {
				for (int til = 0; til <= 9; til++) {
					for (int pan = 0; pan <= 9; pan++) {
						std::string pose_name = get_image_pose(1, 1, 00, "-90", "+0");
						names.push_back(pose_name);
					}
				}
			}
		}
	}
	return names;
}

vector<vector<string> > open_all_qmul_by_person(vector<string> people) {
	vector<vector<string> > names;
	for (int person = 0; person<people.size(); person++) {
		vector<string> tmp;
		for (int tilt = 60; tilt <= 120; tilt += 10) {
			for (int pose = 0; pose <= 180; pose += 10) {
				tmp.push_back(get_image_qmul(people[person], tilt, pose));
			}
		}
		names.push_back(tmp);
	}
	cout << "size of names = " << names.size() << endl;
	cout << "size of namesnames = " << names[0].size() << endl;
	return names;
}


int main()
{
	//load image and set directory - just for testing

	std::vector<std::string> poses = open_all_poses();

	/*for (int i = 0; i < 100; i++) {
		int index = rand() % poses.size();
		cv::Mat im = cv::imread(poses[index]);
		if (im.empty()) {
			std::cout << "error at: " << poses[index] << std::endl;
		}
	}*/
	//imshow(pose_name , img);
	//cv::waitKey();
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

	tinydir_close(&dir);

	vector<vector<string>> images = open_all_qmul_by_person(people);

	vector<Mat> faces;
	/* train */
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			cout << "i = " << i << endl;
			std::string name = images[i][j];
			// open image
			cv::Mat im = cv::imread(name);
			cv::cvtColor(im, im, CV_RGB2GRAY);
			faces.push_back(im);
		}
	}

	//std::string name = get_image_qmul(people[0], 60, 0);
	//std::cout << "testing: " << name << std::endl;
	//// open image
	//cv::Mat im = cv::imread(name);
	//cv::cvtColor(im, im, CV_RGB2GRAY);

	//eigenface_std(faces, im);
	Mat eigen = train(faces);
	/* testing */

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) { //images[0].size()
			std::string name = images[i][j];
			// open image
			cv::Mat im2 = cv::imread(name);
			cv::cvtColor(im2, im2, CV_RGB2GRAY);
			imwrite("test_im.png", im2);
			//waitKey(0);
			int result = test(im2);

			imwrite("actual.png", faces[result]);
			//waitKey(0);
			//            faces.push_back(im);
		}
	}

	//int test_person = 1;

	//std::string name = get_image_qmul(people[0], 60, 0);
	//std::cout << "testing: " << name << std::endl;
	//// open image
	//cv::Mat im = cv::imread(name);

	////convert to greyScale
	//cv::cvtColor(im, im, CV_RGB2GRAY);
	//imshow("test_im", im);
	//waitKey(0);
	//int result = test(im);
	//cout << "index = " << result << endl;
	
//	result.convertTo(result, CV_8UC1);
	//cout << "size of final face = " << result.size() << endl;
	/*imshow("actual", faces[result]);
	waitKey(0);*/

	faces.clear();

	//compute spatial pyramid histogram for number of level
	//tests.push_back(getSpatialPyramidHistogram(im, 1));


	/* person 1 */
	/*double best = std::numeric_limits<double>::max();
	int person = -1;
	for (int i = 0; i<histograms.size(); i++) {
		double diff = cv::compareHist(histograms[i], tests[0], CV_COMP_CHISQR);

		std::cout << "Difference was " << diff << ", opposed to best: " << best << std::endl;

		if (diff < best) {
			best = diff;
			person = i;
		}

	}

	std::cout << "The Person was most likely: " << people[person] << std::endl;
	if (person != test_person) {
		std::cout << "This was wrong" << std::endl;
	}*/

	std::system("pause");


}