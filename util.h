#pragma once

const char QMUL_DIR[] = "QMUL/";
const char POSE_DIR[] = "HeadPoseImageDatabase/";

#include <algorithm>
#include <string>
#include <vector>

const int NUM_FOLDS = 7;
const int MAX_LEVELS = 2;

std::string get_image_hpid(int id, int serie, int number, std::string tilt, std::string pan);
std::string get_image_qmul(std::string person, int tilt, int pose);

std::vector<std::vector<std::string> > open_all_qmul_by_person(std::vector<std::string> people);
std::vector<std::vector<std::string> > open_all_qmul_by_pose(std::vector<std::string> people);
std::vector<std::string> getQmulNames();

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
