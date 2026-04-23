package com.facerec.test;

import java.util.Arrays;
import java.util.Vector;

import com.facerec.util.ImageFile;
import com.google.gson.Gson;
import com.facerec.util.ImageUtil;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.javacv.VideoInputFrameGrabber;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_objdetect.*;

public class Main {

	private static MatVector extractFeatures(Mat img, Vector<Integer> left, Vector<Integer> top) {
		MatVector output = new MatVector();
		return null;
	}

	private static void findContoursBasic(Mat img) {
		MatVector mv = new MatVector();
		findContours(img, mv, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	}
	
	final private static String XML_FILE = "haarcascade_frontalface_default.xml";
	final private static String SampleFiles = "data/facereg";
	private static FrameGrabber grabber = null;
	
    public static void main(String[] args) {
		Mat mat = imread("data/training/luoyafei/IMG_20161114_131335.jpg");
		if (mat.empty()) {
			System.err.println("无法读取图像");
			return;
		}
		Mat resizeMat = new Mat();
		resize(mat, resizeMat, new Size(720, 1280));
		
		detectFace(resizeMat);
		
		waitKey(0);
		destroyAllWindows();
	}
    
    private static void detectFace(Mat src) {
    	CascadeClassifier cascade = new CascadeClassifier(XML_FILE);
    	
    	if (cascade.empty()) {
    		System.err.println("无法加载级联分类器: " + XML_FILE);
    		return;
    	}
    	
    	Mat gray = new Mat();
    	if (src.channels() > 1) {
    		cvtColor(src, gray, COLOR_BGR2GRAY);
    	} else {
    		gray = src.clone();
    	}
    	
    	equalizeHist(gray, gray);
    	
    	RectVector faces = new RectVector();
    	cascade.detectMultiScale(gray, faces);
    	
    	long total_Faces = faces.size();
    	
    	for(long i = 0; i < total_Faces; i++) {
    		Rect r = faces.get(i);
    		rectangle(src, 
    			new Point(r.x(), r.y()), 
    			new Point(r.x() + r.width(), r.y() + r.height()),
    			Scalar.RED,
    			2,
    			LINE_AA,
    			0);
    	}
    	
    	imshow("Result", src);
    }
}
