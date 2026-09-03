#include "stdafx.h"
#include <stdio.h>
#include "Augmentation.h"

using namespace cv;
using namespace std;

// Utility: extract filename without extension
string getBaseName(const string &path)
{
	size_t slash = path.find_last_of("/\\");
	size_t dot = path.find_last_of(".");
	return path.substr(slash + 1, dot - slash - 1);
}

// Utility: extract directory
string getDirectory(const string &path)
{
	size_t slash = path.find_last_of("/\\");
	return path.substr(0, slash + 1);
}

// Random helper
float randFloat(float a, float b)
{
	return a + static_cast<float>(rand()) / RAND_MAX * (b - a);
}

void writeAugmentedImage(const string &InputImgPath)
{
	Mat img = imread(InputImgPath);
	if (img.empty())
	{
		cout << "Error loading image\n";
		return;
	}

	resize(img, img, Size(224, 224));

	string base = getBaseName(InputImgPath);
	string dir = getDirectory(InputImgPath);

	// ---------- 1. Horizontal Flip ----------
	Mat flipImg;
	flip(img, flipImg, 1);
	imwrite(dir + base + "_1.jpg", flipImg);

	// ---------- 2. Rotation ----------
	float angle = randFloat(-20, 20);
	Point2f center(112, 112);
	Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
	Mat rotImg;
	warpAffine(img, rotImg, rotMat, Size(224, 224));
	imwrite(dir + base + "_2.jpg", rotImg);

	// ---------- 3. Color Jitter ----------
	Mat jitter;
	img.convertTo(jitter, -1, randFloat(0.8, 1.2), randFloat(-20, 20)); // contrast + brightness

	Mat hsv;
	cvtColor(jitter, hsv, COLOR_BGR2HSV);

	vector<Mat> channels;
	split(hsv, channels);

	channels[1] *= randFloat(0.8, 1.2); // saturation
	channels[2] *= randFloat(0.8, 1.2); // value

	merge(channels, hsv);
	cvtColor(hsv, jitter, COLOR_HSV2BGR);

	imwrite(dir + base + "_3.jpg", jitter);
	// ---------- 4. Gaussian Noise ----------
	Mat noise = img.clone();
	Mat gauss(img.size(), CV_16SC3);
	randn(gauss, 0, 15);

	Mat temp;
	noise.convertTo(temp, CV_16SC3);
	addWeighted(temp, 1.0, gauss, 1.0, 0, temp);
	temp.convertTo(noise, CV_8UC3);

	imwrite(dir + base + "_4.jpg", noise);
	// ---------- 5. MixUp ----------
	Mat mixup;
	Mat img2;
	flip(img, img2, 1); // use flipped version as second image

	float alpha = randFloat(0.1, 0.3);
	addWeighted(img, alpha, img2, 1 - alpha, 0, mixup);

	imwrite(dir + base + "_5.jpg", mixup);
	// ---------- 6. Slight Blur (camera variation) ----------
	Mat blurImg;
	int ksize = (rand() % 2) * 2 + 5; // 3 or 5
	GaussianBlur(img, blurImg, Size(ksize, ksize), 0);
	imwrite(dir + base + "_6.jpg", blurImg);


	// ---------- 7. Small Affine Warp (facial variation) ----------
	Mat warpImg;

	// Define 3 points from original image
	Point2f srcTri[3];
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(223, 0);
	srcTri[2] = Point2f(0, 223);

	// Slightly perturb destination points
	float shift = 10.0f; // small deformation
	Point2f dstTri[3];
	dstTri[0] = Point2f(randFloat(-shift, shift), randFloat(-shift, shift));
	dstTri[1] = Point2f(223 + randFloat(-shift, shift), randFloat(-shift, shift));
	dstTri[2] = Point2f(randFloat(-shift, shift), 223 + randFloat(-shift, shift));

	Mat warpMat = getAffineTransform(srcTri, dstTri);
	warpAffine(img, warpImg, warpMat, Size(224, 224), INTER_LINEAR, BORDER_REFLECT);

	imwrite(dir + base + "_7.jpg", warpImg);

	// ---------- 8. Illumination Gradient (Improved - Smooth & Subtle) ----------
	Mat illumImg = img.clone();

	// Create smooth gradient mask
	Mat gradient(224, 224, CV_32FC1);

	bool horizontal = rand() % 5;

	// Define smooth intensity range (VERY IMPORTANT)
	float start = randFloat(0.85f, 1.45f);   // near normal
	float end = randFloat(1.45f, 1.650f);   // slight bright/dark variation

	for (int i = 0; i < 224; i++)
	{
		for (int j = 0; j < 224; j++)
		{
			float t = horizontal ? (float)j / 223.0f : (float)i / 223.0f;
			gradient.at<float>(i, j) = start + t * (end - start);
		}
	}

	// OPTIONAL: smooth the gradient further
	GaussianBlur(gradient, gradient, Size(31, 31), 0);

	// Apply to image
	Mat illumFloat;
	illumImg.convertTo(illumFloat, CV_32FC3);

	vector<Mat> ch;
	split(illumFloat, ch);

	for (int c = 0; c < 3; c++)
	{
		multiply(ch[c], gradient, ch[c]);
	}

	merge(ch, illumFloat);
	illumFloat.convertTo(illumImg, CV_8UC3);

	imwrite(dir + base + "_8.jpg", illumImg);
}