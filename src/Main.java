import java.io.File;
import java.io.IOException;

import weka.core.*;
import weka.core.converters.*;
import weka.classifiers.Classifier;
import weka.classifiers.trees.*;
import weka.filters.*;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.trees.SimpleCart;

public class Main {	
	public static void printInstances(Instances instances){
	    for (int i=0;i< instances.numAttributes();i++){
	    	System.out.print(""+instances.attribute(i).name()+"\t");
	    }
	    System.out.println("");
	    for (int i=0; i< instances.numInstances(); i++){
	    	Instance instance = instances.instance(i);
	    	for (int j=0;j<instance.numAttributes();j++){
	    		if (instance.attribute(j).isString())
	    			System.out.print(""+instance.stringValue(j)+"\t");
	    		else if (instance.attribute(j).isNumeric())
	    			System.out.print(""+instance.value(j)+"\t");
	    		
	    	}
	    	System.out.println("");
	    }
	}
	
	public static void testClassifierAndPrintResults(Classifier classifier, Instances testSet) throws Exception{
	    int numFalsePositives = 0;
	    int numTruePositives = 0;
	    int numFalseNegatives = 0;
	    int numTrueNegatives = 0;
	    for (int i = 0; i < testSet.numInstances(); i++) {
	    	Instance instance = testSet.instance(i);
	    	double pred = classifier.classifyInstance(instance);
	    	String className = testSet.classAttribute().value((int)pred);
    		String actualClassName = testSet.classAttribute().value((int)instance.classValue());
	    	if (className.equals("spam")){
	    		if (actualClassName.equals("spam")){
	    			numTruePositives++;
	    		}else{
	    			numFalsePositives++;
	    		}
	    	}else{
	    		if (actualClassName.equals("spam")){
	    			numFalseNegatives++;
	    		}else{
	    			numTrueNegatives++;
	    		}
	    	}
	    }
    	System.out.println("Confusion Matrix");
    	System.out.println(""+numTruePositives+"\t"+numFalsePositives);
    	System.out.println(""+numFalseNegatives+"\t"+numTrueNegatives);
    	System.out.println("Accuracy: " + ((double)(numTruePositives+numTrueNegatives))/testSet.numInstances());
	}
	
	public static final double TR_PERCENT = 0.9;
	
	public static void main(String [] args) throws Exception{
	    TextDirectoryLoader loader = new TextDirectoryLoader();
	    loader.setDirectory(new File("./dataset"));
	    Instances dataRaw = loader.getDataSet();
	    System.out.println("data loaded. size: "+dataRaw.numInstances());
	    
	    StringToWordVector filter = new StringToWordVector();
		filter.setInputFormat(dataRaw);
	    Instances dataFiltered = Filter.useFilter(dataRaw, filter);
	    	    
	    dataFiltered.randomize(new java.util.Random(0));

	    System.out.println("training set: " + TR_PERCENT*100 + "%");
	    System.out.println("test set: " + (100-TR_PERCENT*100) + "%");
	    
	    //memisah training dan test set
	    int trainingSetSize = (int) (dataFiltered.numInstances()*TR_PERCENT);
	    int testSetSize = (int) (dataFiltered.numInstances()-trainingSetSize);
	    Instances trainingSet = new Instances(dataFiltered,0,trainingSetSize);
	    Instances testSet = new Instances(dataFiltered,trainingSetSize,testSetSize);

	    //coba hanya dengan CART
	    System.out.println("mencoba SimpleCart...");
	    SimpleCart simpleCart = new SimpleCart();
	    long time_start_train_2 = System.currentTimeMillis();
	    simpleCart.buildClassifier(trainingSet);
	    System.out.printf("Training time with " + trainingSet.numInstances() + " + instances : %.3fsec%n",
	    	(System.currentTimeMillis() - time_start_train_2)/1000.0D
	    );
	    testClassifierAndPrintResults(simpleCart, testSet);

	    //formalisasi
	    WordVectorFormalizer formalizerFilter = new WordVectorFormalizer();
	    formalizerFilter.setInputFormat(dataFiltered);
	    
	    //coba dengan formalisasi
	    System.out.println("mencoba SimpleCart + formalisasi...");
	    SimpleCart simpleFormalizedCart = new SimpleCart();
	    long time_start_train_1 = System.currentTimeMillis();
	    simpleFormalizedCart.buildClassifier(formalizerFilter.doFilter(trainingSet));
	    System.out.printf("Training time with " + trainingSet.numInstances() + " + instances : %.3fsec%n",
	    	(System.currentTimeMillis() - time_start_train_1)/1000.0D
	    );
	    testClassifierAndPrintResults(simpleFormalizedCart, formalizerFilter.doFilter(testSet));

	    	    
	}
}
