package onnxsnapstart;

import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OnnxTensor;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.Arrays;
import java.util.Collections;

import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.RequestHandler;
import com.amazonaws.services.lambda.runtime.events.APIGatewayProxyRequestEvent;
import com.amazonaws.services.lambda.runtime.events.APIGatewayProxyResponseEvent;

/**
 * Handler for Onnx predictions on Lambda function.
 */
public class App implements RequestHandler<APIGatewayProxyRequestEvent, APIGatewayProxyResponseEvent> {

    // Onnx session with preloaded model which will be reused between invocations and will be
    // initialized as part of snapshot creation
    private static OrtSession onnxSession;

    // Returns Onnx session with preloaded model. Reuses existing session if exists.
    private static OrtSession getOnnxSession() {
        String modelPath = "inception_v3.onnx";
        if (onnxSession==null) {
          System.out.println("Start model load");
          try (OrtEnvironment env = OrtEnvironment.getEnvironment("createSessionFromPath");
            OrtSession.SessionOptions options = new SessionOptions()) {
          try {
            OrtSession session = env.createSession(modelPath, options);
            Map<String, NodeInfo> inputInfoList = session.getInputInfo();
            Map<String, NodeInfo> outputInfoList = session.getOutputInfo();
            System.out.println(inputInfoList);
            System.out.println(outputInfoList);
            onnxSession = session;
            return onnxSession;
          }
          catch(OrtException exc) {
            exc.printStackTrace();
          }
        }
        }
        return onnxSession;
    }

    // This code runs during snapshot initialization. In the normal lambda that would run in init phase.
    static {
        System.out.println("Start model init");
        getOnnxSession();
        System.out.println("Finished model init");
    }

    // Main handler for the Lambda
    public APIGatewayProxyResponseEvent handleRequest(final APIGatewayProxyRequestEvent input, final Context context) {
        Map<String, String> headers = new HashMap<>();
        headers.put("Content-Type", "application/json");
        headers.put("X-Custom-Header", "application/json");


        float[][][][] testData = new float[1][3][299][299];

        try (OrtEnvironment env = OrtEnvironment.getEnvironment("createSessionFromPath")) {
            OnnxTensor test = OnnxTensor.createTensor(env, testData);
            OrtSession session = getOnnxSession();
            String inputName = session.getInputNames().iterator().next();
            Result output = session.run(Collections.singletonMap(inputName, test));
            System.out.println(output);
        }
        catch(OrtException exc) {
            exc.printStackTrace();
        }


        APIGatewayProxyResponseEvent response = new APIGatewayProxyResponseEvent().withHeaders(headers);
        String output = String.format("{ \"message\": \"made prediction\" }");

        return response
                .withStatusCode(200)
                .withBody(output);
    }
}
