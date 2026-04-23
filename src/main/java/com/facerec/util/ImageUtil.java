package com.facerec.util;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import com.google.gson.Gson;
import com.google.gson.JsonIOException;
import com.google.gson.JsonSyntaxException;

public class ImageUtil {

	private static final String XML_FILE = "haarcascade_frontalface_default.xml";
	private static final Gson gson = new Gson();

	private static CascadeClassifier cascade = new CascadeClassifier(XML_FILE);

	public static boolean dealSampleFaceImage(Mat src, String fileUrl) {
		if (cascade.empty()) {
			System.err.println("错误: 无法加载级联分类器: " + XML_FILE);
			return false;
		}
		
		Mat imgTemp = src.clone();
		RectVector faces = new RectVector();
		
		cascade.detectMultiScale(src, faces);
		
		long total_Faces = faces.size();
		if (total_Faces == 0)
			return false;
		
		for (long i = 0; i < total_Faces; i++) {
			Rect r = faces.get(i);
			Mat faceROI = new Mat(imgTemp, r);
			Mat resizedFace = new Mat();
			resize(faceROI, resizedFace, new Size(200, 200));
			return imwrite(fileUrl, resizedFace);
		}
		return false;
	}
	
	@SuppressWarnings("resource")
	public static void saveFileJson(ImageFile imageFile) {
		String filesContent = gson.toJson(imageFile);
		File file = new File("files.json");
System.out.println(imageFile.toString());
		try {
			FileOutputStream fileOutput = new FileOutputStream(file);
			fileOutput.write(filesContent.getBytes());
		} catch(IOException e) {}
	}
	
	@SuppressWarnings("finally")
	public static ImageFile getImageFile() {
		ImageFile file = null;
		try {
			file =  gson.fromJson(new FileReader("files.json"), ImageFile.class);
		} catch (JsonSyntaxException | JsonIOException | FileNotFoundException e) {
System.out.println("files.json 文件加载失败！没有找到！");
			file = new ImageFile();			
		} finally {
			return file;
		}
	}
	public static String getNameInNumber(String nb) {
		
		return "老王";
	}
}
