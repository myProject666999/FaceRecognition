package com.facerec.test;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.CHAIN_APPROX_SIMPLE;

import static org.bytedeco.opencv.global.opencv_imgproc.RETR_LIST;
import static org.bytedeco.opencv.global.opencv_imgproc.THRESH_BINARY;
import static org.bytedeco.opencv.global.opencv_imgproc.blur;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.findContours;
import static org.bytedeco.opencv.global.opencv_imgproc.threshold;

public class MotionDetector {
    public static void main(String[] args) throws Exception {
        OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        grabber.start();

        Mat frame = converter.convert(grabber.grab());
        Mat image = null;
        Mat prevImage = null;
        Mat diff = null;

        CanvasFrame canvasFrame = new CanvasFrame("Motion Detector");
        canvasFrame.setCanvasSize(frame.cols(), frame.rows());

        while (canvasFrame.isVisible() && (frame = converter.convert(grabber.grab())) != null) {
            org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur(frame, frame, new Size(9, 9), 2.0);
            
            if (image == null) {
                image = new Mat(frame.rows(), frame.cols(), CV_8UC1);
                cvtColor(frame, image, COLOR_BGR2GRAY);
            } else {
                prevImage = image.clone();
                image = new Mat(frame.rows(), frame.cols(), CV_8UC1);
                cvtColor(frame, image, COLOR_BGR2GRAY);
            }

            if (diff == null) {
                diff = new Mat(frame.rows(), frame.cols(), CV_8UC1);
            }

            if (prevImage != null) {
                Mat absDiff = new Mat();
                org.bytedeco.opencv.global.opencv_core.absdiff(image, prevImage, absDiff);
                threshold(absDiff, diff, 64, 255, THRESH_BINARY);

                canvasFrame.showImage(converter.convert(diff));

                MatVector contours = new MatVector();
                Mat hierarchy = new Mat();
                findContours(diff, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

                long contourCount = contours.size();
                if (contourCount > 0) {
                    System.out.println("Motion detected! Contours: " + contourCount);
                }
            }
        }
        grabber.stop();
        canvasFrame.dispose();
    }
}
