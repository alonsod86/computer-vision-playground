package es.alonso.javacv.examples;
import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_objdetect.*;

import com.googlecode.javacv.CanvasFrame;
import com.googlecode.javacv.FrameGrabber.Exception;
import com.googlecode.javacv.OpenCVFrameGrabber;
import com.googlecode.javacv.cpp.opencv_core.CvMemStorage;
import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.CvSeq;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_objdetect.CvHaarClassifierCascade;

/**
 * JavaCV WebCam face detector using HaarCascadeClassifier
 * 
 * It uses the same principle in FaceDetector.java
 * 
 * @author alonsod86
 *
 */
public class WebCam {
	public static final String XML_FILE = "resources/haarcascade_frontalface_default.xml";
	public static final String XML_TEST = "resources/haarcascade_upperbody.xml";
	final private static int WEBCAM_DEVICE_INDEX = 0;
	
	//Define classifier 
	static CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(XML_FILE));

	static CvMemStorage storage = CvMemStorage.create();

	public static void main(String[] args) throws Exception {
		int captureWidth = 640;
		int captureHeight = 480;

		// The available FrameGrabber classes include OpenCVFrameGrabber (opencv_videoio),
		// DC1394FrameGrabber, FlyCaptureFrameGrabber, OpenKinectFrameGrabber,
		// PS3EyeFrameGrabber, VideoInputFrameGrabber, and FFmpegFrameGrabber.
		OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(WEBCAM_DEVICE_INDEX);
		grabber.setImageWidth(captureWidth);
		grabber.setImageHeight(captureHeight);
		grabber.start();

		// A really nice hardware accelerated component for our preview...
		CanvasFrame cFrame = new CanvasFrame("Capture Preview", CanvasFrame.getDefaultGamma() / grabber.getGamma());

		IplImage capturedFrame = null;

		// While we are capturing...
		while ((capturedFrame = grabber.grab()) != null)
		{
			if (cFrame.isVisible()) {
				// Show our frame in the preview
				cFrame.showImage(detect(capturedFrame));
			}
		}
		
		cFrame.dispose();
		grabber.stop();

	}

	//Detect for face using classifier XML file 
	public static IplImage detect(IplImage src){
		//Detect objects
		CvSeq sign = cvHaarDetectObjects(
				src,
				cascade,
				storage,
				1.5,
				1,
				CV_HAAR_DO_CANNY_PRUNING);

		cvClearMemStorage(storage);

		int total_Faces = sign.total();		

		//Draw rectangles around detected objects
		for(int i = 0; i < total_Faces; i++){
			CvRect r = new CvRect(cvGetSeqElem(sign, i));
			cvRectangle (
					src,
					cvPoint(r.x(), r.y()),
					cvPoint(r.width() + r.x(), r.height() + r.y()),
					CvScalar.RED,
					2,
					CV_AA,
					0);

		}

		return src;
	}			
}
