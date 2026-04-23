package com.facerec.test;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import java.util.HashMap;

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
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.javacv.FrameGrabber.Exception;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_highgui.destroyAllWindows;
import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_highgui.waitKey;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_PLAIN;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.equalizeHist;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class OpenCVFaceRecognizer {
	final private static String trainingDir = "data/facereg";
	private static HashMap<Integer, String> faceMap = new HashMap<>();
	
    public static void main(String[] args) throws Exception {
    	initMap();
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

        String trainingDir = "data/facereg";

        CascadeClassifier face_cascade = new CascadeClassifier("haarcascade_frontalface_default.xml");
        FaceRecognizer lbphFaceRecognizer = LBPHFaceRecognizer.create();
        
        OpenCVFrameGrabber grabber = null;
        try {
            grabber = OpenCVFrameGrabber.createDefault(0);
            grabber.start();
        } catch (Exception e) {
            System.err.println("Failed start the grabber.");
        }

        Frame videoFrame = null;
        Mat videoMat = new Mat();
        while (true) {
            videoFrame = grabber.grab();
            videoMat = converterToMat.convert(videoFrame);
            Mat videoMatGray = new Mat();
            cvtColor(videoMat, videoMatGray, COLOR_BGRA2GRAY);
            equalizeHist(videoMatGray, videoMatGray);

            Point p = new Point();
            RectVector faces = new RectVector();
            face_cascade.detectMultiScale(videoMatGray, faces);

            for (int i = 0; i < faces.size(); i++) {
                Rect face_i = faces.get(i);

                Mat face = new Mat(videoMatGray, face_i);
                
                resize(face, face, new Size(200, 200));

                String faceName = compareFace(face);
                rectangle(videoMat, face_i, new Scalar(0, 255, 0, 1));

                String box_text = "MingZi:" + faceName;
                int pos_x = Math.max(face_i.tl().x() - 10, 0);
                int pos_y = Math.max(face_i.tl().y() - 10, 0);
                putText(videoMat, box_text, new Point(pos_x, pos_y),
                        FONT_HERSHEY_PLAIN, 1.0, new Scalar(0, 255, 0, 2.0));
            }
            imshow("face_recognizer", videoMat);

            char key = (char) waitKey(20);
            if (key == 27) {
                destroyAllWindows();
                break;
            }
        }
    }
    
    private static String compareFace(Mat testImage) {

        File root = new File(trainingDir);

        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        File[] imageFiles = root.listFiles(imgFilter);
        
        if (imageFiles == null || imageFiles.length == 0) {
            System.err.println("警告: 样本目录为空或不存在: " + trainingDir);
            return "unknown";
        }

        MatVector images = new MatVector(imageFiles.length);
        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        
        IntBuffer labelsBuf = labels.getIntBuffer();

        int counter = 0;

        for (File image : imageFiles) {
            Mat img = imread(image.getAbsolutePath(), IMREAD_GRAYSCALE);
            String num = image.getName().replaceAll("\\D", "");
            int label = Integer.parseInt(num);

            images.put(counter, img);

            labelsBuf.put(counter, label);

            counter++;
        }

        FaceRecognizer faceRecognizer = EigenFaceRecognizer.create();

        faceRecognizer.train(images, labels);

        IntPointer label = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);
        faceRecognizer.predict(testImage, label, confidence);
        int predictedLabel = label.get(0);
        System.out.println("Predicted label: " + faceMap.get(predictedLabel));
        return  faceMap.get(predictedLabel);
    }
    
    
    private static void initMap() {
    	File root = new File(trainingDir);
        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        File[] imageFiles = root.listFiles(imgFilter);
        
        if (imageFiles == null) {
            return;
        }

        for (File image : imageFiles) {
            String num = image.getName().replaceAll("\\D", "");
            String name = image.getName().split("\\d")[0];
            int label = Integer.parseInt(num);
            faceMap.put(label, name);
        }
    }
    
}
