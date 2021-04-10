#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<vector>

using namespace std;
using namespace cv;

void rotateTest(const Mat& img, Mat& dest, const RotatedRect r) {
	Point center = Point(r.center.x, r.center.y);

	double angle = (r.size.width > r.size.height) ? 90 + r.angle : r.angle;

	Mat rotMat = getRotationMatrix2D(Point2f(center.x, center.y), angle, 1);
	warpAffine(img, dest, rotMat, img.size(), INTER_CUBIC);
}



void standardize(const Mat& img, Mat& dest) {
	
	Mat tmp;
	threshold(img, tmp, 130, 255, THRESH_BINARY_INV);

	
	medianBlur(tmp, tmp, 3);

	vector<vector<Point>> contours;
	
	findContours(tmp, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	
	RotatedRect r = minAreaRect(contours[0]);

	
	Mat img2, tmp2;
	rotateTest(img, img2, r);
	rotateTest(tmp, tmp2, r);



	findContours(tmp2, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	dest = img2(boundingRect(contours[0]));
}

void createTheMaskOfTheNonGrayCells(const Mat& solver, Mat& dest) {

	Mat gray_cells;


	inRange(solver, 100, 200, gray_cells);



	medianBlur(gray_cells, gray_cells, 5);

	Mat struct_element = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
	erode(gray_cells, gray_cells, struct_element);

	dest = 255 - gray_cells;
}

int main() {

	Mat solver0 = imread("test_solver.png", IMREAD_GRAYSCALE);

	Mat solver;
	standardize(solver0, solver);
	 
	Mat nongray;
	createTheMaskOfTheNonGrayCells(solver, nongray);


	for (int i = 0; i <= 17; ++i) {
		Mat img0 = imread("test_" + to_string(i) + ".png", IMREAD_GRAYSCALE);

		Mat img, tmp;
		standardize(img0, img);

	
		resize(img, tmp, nongray.size(), INTER_NEAREST);



		Mat thimg;
		threshold(tmp, thimg, 150, 250, THRESH_BINARY_INV);

		thimg.setTo(0, nongray);


		imshow("correct answers", thimg);
		imshow("solver", solver);
		waitKey(0);

	 
		dilate(thimg, thimg, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));


		vector<vector<Point>> contours;
		
		findContours(thimg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		
		Mat canvas;
		cvtColor(img, canvas, COLOR_GRAY2BGR);
		double count = 0;
		
		for (auto c : contours) {
			
			Rect r = boundingRect(c);

			
			rectangle(canvas, r, Scalar(255, 0, 0), 3);

			imshow("original", img);

			imshow("answer", canvas);

			char columns = (char)((r.x - 62) / 48 + 'A');
			int rows = (r.y - 43) / 42 + 1;
			count = rows ? count + 1 : count;
			cout << columns << rows << endl;

			waitKey(0);
		}
		double percentage = count / 10 * 100;
		Mat result = canvas.clone();
		if (percentage >= 50.0)
			putText(result, "Passed: " + to_string(int(percentage)) + "%", Point(30, 400), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 0), 2);

		else
			putText(result, "Failed: " + to_string(int(percentage)) + "%", Point(30, 400), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2);
		imshow("result", result);
		waitKey(0);
		destroyWindow("result");
	}



	return 0;
}