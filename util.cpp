#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

#include "util.h"

#include "tinydir.h"

using namespace std;

string tilts[] = {"-90", "-60", "-30", "-15", "0", "+15", "+30", "+60", "+90"};
string pans[] = {"-90", "-75", "-60", "-45", "-30", "-15", "0", "+15", "+30", "+45", "+60", "+75", "+90"};

string get_image_hpid(int id, int serie, int number, string tilt, string pan) {
	stringstream s, id_ss, number_ss;
	
	id_ss << setfill('0') << setw(2) << id;
	number_ss << setfill('0') << setw(2) << number;

	s << POSE_DIR << "Person" << id_ss.str() << "/" << "person" << id_ss.str() << serie << number_ss.str() << tilt << pan << ".jpg";
	return s.str();
}

string get_image_qmul(string person, int tilt, int pose) {
	stringstream s, tilt_ss, pose_ss;
	tilt_ss << setfill('0') << setw(3) << tilt;
	pose_ss << setfill('0') << setw(3) << pose;

	s << QMUL_DIR << person << "/" << person.substr(0, person.size() - 4) << "_"  << tilt_ss.str() << "_" << pose_ss.str() << ".ras";
	return s.str();
}

vector<vector<string> > open_all_qmul_by_person(vector<string> people) {
	vector<vector<string> > names;
	for (int person=0; person<people.size(); person++) {
		vector<string> tmp;
		for (int tilt=60; tilt<=120; tilt += 10) {
			for (int pose=0; pose <= 180; pose += 10) {
				tmp.push_back(get_image_qmul(people[person], tilt, pose));	
			}
		}
		names.push_back(tmp);
	}
	return names;
}

vector<vector<string> > open_all_qmul_by_pose(vector<string> people) {
	vector<vector<string> > poses;
	for (int pose=0; pose <= 180; pose += 10) {
		std::cout << pose << std::endl;
		vector<string> tmp;
		for (int tilt=60; tilt<=120; tilt += 10) {
			for (int person=0; person<people.size(); person++) {
				tmp.push_back(get_image_qmul(people[person], tilt, pose));	
			}
		}
		poses.push_back(tmp);
	}
	return poses;
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