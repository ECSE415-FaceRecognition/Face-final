#include "eigenfaces.h"

Mat generate_flat_diff(vector<Mat> faces);
void sort_mat(const Mat &input, Mat &sorted, const Mat &indices);

Mat mean_face;
Mat coeff;
Mat eigenfaces;
//double num_eig = 20;

Mat train(vector<Mat> faces) {
    cout<<"Training"<<endl;
    Mat flat_images = Mat::zeros(faces.size(), faces[0].rows*faces[0].cols, CV_64F);
    
    //Flatten the images
    int col_count = 0;
    for(int i = 0; i<faces.size(); i++)
    {
        Mat temp = Mat::zeros(faces[0].rows, faces[0].cols, CV_64F);
        col_count = 0;
        faces[i].copyTo(temp);

        //Convert input images;
        temp.convertTo(temp, CV_64FC1);
        
        //Now reshape the face and copy to temp mat;
        temp.reshape(0,1);
        for(int j = 0; j<faces[i].rows*faces[i].cols; j++)
        {
            flat_images.at<double>(i, col_count) = temp.at<double>(0,col_count);
            col_count = col_count + 1;
        }
    }
    
    //  Compute the mean face
    mean_face = Mat::zeros(1, flat_images.cols, CV_64FC1);
    for(int i = 0; i<mean_face.cols; i++)
    {
        mean_face.at<double>(0, i) = mean(flat_images.col(i))[0];
    }
    
    //  Now compute diff_faces (face[i] - mean)
    Mat diff_faces = Mat::zeros(flat_images.rows, flat_images.cols, CV_64FC1);
    for(int j = 0; j<faces.size(); j++)
    {
        diff_faces.row(j) = flat_images.row(j)-mean_face;
    }
    //cout<<"Mean face size = "<<mean_face.size()<<endl;

    Mat covar_mat = diff_faces*diff_faces.t();
    //cout<<covar_mat.size()<<endl;
    
    //  Now compute the eigenvector and eigenvalues
    Mat eigenval, eigenvec;
    eigen(covar_mat, eigenval, eigenvec);
    
    //cout<<eigenvec.size()<<endl;
    //cout<<diff_faces.size()<<endl;
    eigenfaces = diff_faces.t()*eigenvec;
    
    
    //  Now project the image onto the eigenface matrix
    cout <<"eigenfaces"<< eigenfaces.size()<<endl;
    coeff = Mat::zeros(faces.size(), faces.size(), CV_64FC1);
    for(int i = 0; i<faces.size();i++)
    {
        for(int j = 0; j<eigenfaces.cols; j++)
        {
			coeff.at<double>(i, j) = diff_faces.row(i).dot(eigenfaces.col(j).t());
        }
    }

	//return coeffs
	return coeff;
}

int test(Mat &candidate){
    cout<<"Now testing"<<endl;
	Mat flat_candidate;

	candidate.copyTo(flat_candidate);
	//notify that testing has begun
    cout <<"TEST"<<endl;
	
	flat_candidate = (flat_candidate.reshape(0, 1)); //Flatten the input candidate image
    cout<<"eigenfaces size"<<eigenfaces.size()<<endl;
	cout << "mean_face size" << mean_face.size() << endl;
	cout << "flat_candidate size" << mean_face.size() << endl;
    
	flat_candidate.convertTo(flat_candidate, CV_64FC1);
	flat_candidate = flat_candidate - mean_face;
	flat_candidate.convertTo(flat_candidate, CV_8UC1);

	///uncomment to display diff_face test
 //   Mat temp;
 //   temp = flat_candidate.reshape(0, 100);
 //   namedWindow("Display diff face", WINDOW_AUTOSIZE);   // Create a window for display.
 //   imshow("Display diff face", temp);
 //   cv::waitKey(0);
	flat_candidate.convertTo(flat_candidate, CV_64FC1);
	flat_candidate = flat_candidate.t();

    //Now Project data on eigenfaces
    Mat test_coefs = Mat::zeros(1, eigenfaces.cols, CV_64FC1);

    for (int i = 0; i < eigenfaces.cols; i++) {
        test_coefs.at<double>(0,i) = flat_candidate.dot(eigenfaces.col(i));
    }

	//find best match
	double min = DBL_MAX;
    int min_id = -1;
    for (int i = 0; i < coeff.rows; i++) {
        Mat temp4 = coeff.row(i);
		//cout << norm(temp4, test_coefs, NORM_L2) << endl;
        if (norm(temp4, test_coefs, NORM_L2) < min) {
            min = norm(temp4, test_coefs, NORM_L2);
            min_id = i;
        }
    }
    if (min_id == -1) {
        std::cout << "Error" << std::endl;
    }

	//return label of best match
    return min_id;

}