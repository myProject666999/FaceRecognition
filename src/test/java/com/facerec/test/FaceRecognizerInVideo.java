package com.facerec.test;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.FrameGrabber.Exception;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

import static org.bytedeco.opencv.global.opencv_highgui.destroyAllWindows;
import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_highgui.waitKey;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_PLAIN;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.equalizeHist;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

public class FaceRecognizerInVideo {

    public static void main(String[] args) throws Exception {
        OpenCVFrameConverter.ToMat converterToMat = new OpenCVFrameConverter.ToMat();

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

            RectVector faces = new RectVector();
            face_cascade.detectMultiScale(videoMatGray, faces);

            for (int i = 0; i < faces.size(); i++) {
                Rect face_i = faces.get(i);

                Mat face = new Mat(videoMatGray, face_i);

                IntPointer label = new IntPointer(1);
                DoublePointer confidence = new DoublePointer(1);
                lbphFaceRecognizer.predict(face, label, confidence);
                int prediction = label.get(0);

                rectangle(videoMat, face_i, new Scalar(0, 255, 0, 1));

                String box_text = "Prediction = " + prediction;
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

}
