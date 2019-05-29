#include<iostream>
#include<opencv.hpp>
#include<opencv2/face/facemark.hpp>
#include<highgui.hpp>
#include<fstream>
#include<face.hpp>
#include<core.hpp>
#include<face/facerec.hpp>
#include<face/facemark_train.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;
using namespace ml;

int main()
{
	VideoCapture cap(0);
	Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
	CascadeClassifier det;
	det.load("haarcascade_frontalface_alt2.xml");
	model->read("model(alpha).xml");
	Mat sb;
	vector<Rect>facess;
	for  (;1; )
	{
		vector<double>facevalue;
		vector<int>rejectlevel;
		cap >> sb;
		Mat sbc = sb.clone();
		cvtColor(sbc, sbc, COLOR_BGR2GRAY);
		det.detectMultiScale(sbc, facess, 1.1, 2, 0 | CASCADE_FIND_BIGGEST_OBJECT, Size(30, 30));
		if (facess.size()>0)
			for (int i = 0; i < facess.size(); i++)
			{
				Mat oneface = sbc(Rect(facess[i]));
				resize(oneface, oneface, Size(192, 192));
				int code = model->predict(oneface);
				//cout << "afdhsfdjhgdw";
				rectangle(sb, facess[i], Scalar(0, 255, 0), 10);
				putText(sb, to_string(code), Point(facess[i].x, facess[i].y + 12), 3, 1, Scalar(0, 0, 255));
			}
		imshow("det", sb);
		waitKey(10);
		
	}

}