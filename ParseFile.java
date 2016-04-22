
import java.io.File;
import java.io.IOException;
import java.util.Enumeration;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class ParseFile {

    private static final String ARFF_TEST_FILE_PATH = "C:\\Users\\Eugene\\Documents\\PMP\\Machine Learning\\Lab1\\testingD.arff";
    private static final String ARFF_TRAINING_FILE_PATH = "C:\\Users\\Eugene\\Documents\\PMP\\Machine Learning\\Lab1\\training_subsetD.arff";
	
	//private static final String ARFF_TRAINING_FILE_PATH = "C:\\Users\\Eugene\\Documents\\PMP\\Machine Learning\\Lab1\\Simple\\train.arff";
	//private static final String ARFF_TEST_FILE_PATH = "C:\\Users\\Eugene\\Documents\\PMP\\Machine Learning\\Lab1\\Simple\\test.arff";

    public static void main(String[] args) throws Exception {
        Instances trainingInstances = LoadFile(ARFF_TRAINING_FILE_PATH);
        
        System.out.println("Building ID3 from training data...");
        
        ID3Chi id3 = new ID3Chi(0.99);
        id3.buildClassifier(trainingInstances);
        
        System.out.println("Completed ID3 tree training");
        System.out.println(id3.toString());

        Instances testInstances = LoadFile(ARFF_TEST_FILE_PATH);
        
        System.out.println("Running ID3 on test data");

        int hit = 0;
        for(Enumeration<Instance> e = testInstances.enumerateInstances(); e.hasMoreElements();){
        	Instance i = e.nextElement();
        	double d = id3.classifyInstance(i);
        	String c = id3.convertClassValueToString(testInstances.classAttribute(), d);
        	if (d == i.classValue()) {
        		hit++;
        	}
            //System.out.println("Instance:" + i + " has class: " + c);
        }
        System.out.println("Accuracy on test data is: " + (double)hit*100/testInstances.numInstances() + "%");

        //*
        System.out.println("Running ID3 on training data");

        int hit2 = 0;
        for(Enumeration<Instance> e = trainingInstances.enumerateInstances(); e.hasMoreElements();){
        	Instance i = e.nextElement();
        	double d = id3.classifyInstance(i);
        	String c = id3.convertClassValueToString(testInstances.classAttribute(), d);
        	if (d == i.classValue()) {
        		hit2++;
        	}
            //System.out.println("Instance:" + i + " has class: " + id3.convertClassValueToString(trainingInstances.classAttribute(), d));
        }
        System.out.println("Accuracy on training data is: " + (double)hit2*100/trainingInstances.numInstances() + "%");
        /**/
    }
    
    private static Instances LoadFile(String path) throws IOException {
        ArffLoader arffTestLoader = new ArffLoader();

        File datasetFile = new File(path);
        arffTestLoader.setFile(datasetFile);

        Instances attributes = arffTestLoader.getStructure();
        Instances dataInstances = arffTestLoader.getDataSet();
        
        for (Enumeration<Attribute> a = attributes.enumerateAttributes(); a.hasMoreElements();){
        	Attribute attr = a.nextElement();
        	System.out.println("Attribute: " + attr.index() + ": type: " + attr.type() + ": " + attr);
        }

        /*
        for(Enumeration<Instance> e = dataInstances.enumerateInstances(); e.hasMoreElements();){
            System.out.println("Instance:" + e.nextElement());
        }
        */
        
        dataInstances.setClassIndex(dataInstances.numAttributes()-1);
        
        return dataInstances;
    }
}
