package com.facerec.test;

import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.TM_CCORR_NORMED;
import static org.bytedeco.opencv.global.opencv_imgproc.matchTemplate;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;

public class TemplateMatch {

	private Mat matchSample = null;

    public void setMatchSample(String filename) {
        matchSample = imread(filename);
    }

    
    public boolean performMatchTemplate(Mat source) {
    	if (matchSample == null || matchSample.empty() || source.empty()) {
    		return false;
    	}
    	
        Mat result = new Mat();
        
        org.bytedeco.opencv.global.opencv_imgproc.matchTemplate(source, this.matchSample, result, TM_CCORR_NORMED);
        
        Point maxLoc = new Point();
        Point minLoc = new Point();
        DoublePointer minVal = new DoublePointer(1);
        DoublePointer maxVal = new DoublePointer(1);
        
        minMaxLoc(result, minVal, maxVal, minLoc, maxLoc, null);
        
        boolean resultBool = maxVal.get(0) > 0.97f;
        
        return resultBool;
    }
}
