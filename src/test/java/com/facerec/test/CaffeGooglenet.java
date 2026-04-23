package com.facerec.test;

import static org.bytedeco.opencv.global.opencv_core.minMaxLoc;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;

public class CaffeGooglenet {

    public static void getMaxClass(Mat probMat, Point classId, double[] classProb) {
        Mat reshaped = probMat.reshape(1, 1);
        minMaxLoc(reshaped, null, classProb, null, classId, null);
    }

    public static List<String> readClassNames() {
        String filename = "synset_words.txt";
        List<String> classNames = null;

        try (BufferedReader br = new BufferedReader(new FileReader(new File(filename)))) {
            classNames = new ArrayList<String>();
            String name = null;
            while ((name = br.readLine()) != null) {
                classNames.add(name.substring(name.indexOf(' ')+1));
            }
        } catch (IOException ex) {
            System.err.println("File with classes labels not found " + filename);
            System.exit(-1);
        }
        return classNames;
    }

    public static void main(String[] args) throws Exception {
        String modelTxt = "bvlc_googlenet.prototxt";
        String modelBin = "bvlc_googlenet.caffemodel";
        String imageFile = (args.length > 0) ? args[0] : "space_shuttle.jpg";

        Net net = new Net();
        
        System.err.println("注意: Caffe importer在OpenCV 4.x中已被移除。");
        System.err.println("建议使用: Net.readNetFromCaffe() 或 readNetFromTensorflow() 等方法。");
        System.err.println("此测试文件需要更新以使用新的API。");

        Mat img = imread(imageFile);

        if (img.empty()) {
            System.err.println("Can't read image from the file: " + imageFile);
            System.exit(-1);
        }

        resize(img, img, new Size(224, 224));
        
        System.out.println("测试框架已更新，但Caffe模型加载需要新的API。");
        System.out.println("请确保使用 readNetFromCaffe(modelTxt, modelBin) 方法加载模型。");
    }
}
