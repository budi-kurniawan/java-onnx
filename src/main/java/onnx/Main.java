package onnx;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.TensorInfo;

public class Main {
	public static void main(String[] args) {
		String modelPath = "./src/main/resources/tinyyolov2-7.onnx";
	    try (OrtEnvironment env = OrtEnvironment.getEnvironment();
	    		OrtSession.SessionOptions options = new SessionOptions()) {
	    	try (OrtSession session = env.createSession(modelPath, options)) {
	    		Map<String, NodeInfo> inputInfoList = session.getInputInfo();
	    		NodeInfo input = inputInfoList.get("data_0");
	    		TensorInfo inputInfo = (TensorInfo) input.getInfo();
	    		int[] expectedInputDimensions = new int[] {1, 3, 224, 224};

	    		Map<String, NodeInfo> outputInfoList = session.getOutputInfo();
	    		NodeInfo output = outputInfoList.get("softmaxout_1");
	    		TensorInfo outputInfo = (TensorInfo) output.getInfo();
	    		int[] expectedOutputDimensions = new int[] {1, 1000, 1, 1};
	    	}	
	      } catch (OrtException e) {
	    	  e.printStackTrace();
	    	  
	      }
	}

}
