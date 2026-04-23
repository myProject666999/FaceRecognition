package com.facerec.test;

import org.bytedeco.javacv.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;

import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_highgui.destroyAllWindows;
import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_highgui.waitKey;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.TM_CCORR_NORMED;
import static org.bytedeco.opencv.global.opencv_imgproc.matchTemplate;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

public class TemplateMatching {

    public static void main(String[] args) throws Exception {
    	String fileUrl_1 = "data/luoyafei.jpg";
    	String fileUrl_2 = "data/luoluo.jpg";
    	int width = 500;
    	int height = 500;

        Mat src = imread(fileUrl_1);
        Mat tmp = imread(fileUrl_2);
        
        if (src.empty() || tmp.empty()) {
        	System.err.println("无法读取图像文件");
        	return;
        }

        Mat result = new Mat();
        
        matchTemplate(src, tmp, result, TM_CCORR_NORMED);

        DoublePointer min_val_d = new DoublePointer(1);
        DoublePointer max_val_d = new DoublePointer(1);
        Point minLoc = new Point();
        Point maxLoc = new Point();
        
        minMaxLoc(result, min_val_d, max_val_d, minLoc, maxLoc, null);
        
        System.out.println("Max correlation: " + max_val_d.get(0));

        Point point = new Point();
        point.x(maxLoc.x() + tmp.cols());
        point.y(maxLoc.y() + tmp.rows());

        rectangle(src, maxLoc, point, Scalar.RED, 2, 8, 0);
        
        Rect rect = new Rect();
        rect.x(maxLoc.x());
        rect.y(maxLoc.y());
        rect.width(tmp.cols() + width);
        rect.height(tmp.rows() + height);
        
        if (maxLoc.x() + rect.width() <= src.cols() && maxLoc.y() + rect.height() <= src.rows()) {
            Mat imageNew = new Mat(src, rect);
            imwrite("template_result.jpg", imageNew);
        }

        imshow("Template Matching", src);
        waitKey(0);
        destroyAllWindows();
    }
}
