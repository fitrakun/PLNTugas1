import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import IndonesianNLP.IndonesianSentenceFormalization;
import IndonesianNLP.IndonesianStemmer;


public class WordVectorStemmer {

	private Instances outputFormat = null;
	Map<String,String> stemmedWordsMapping = null;
	Set<String> stemmedWords = null;

	public boolean setInputFormat(Instances instanceInfo){
		stemmedWords = new HashSet<String>();
		stemmedWordsMapping = new HashMap<String,String>();
		IndonesianStemmer is = new IndonesianStemmer();
		for (int i=0; i<instanceInfo.numAttributes();i++){
			if (instanceInfo.attribute(i).equals(instanceInfo.classAttribute())){
				//do nothing
			}else {
				String originalWord = instanceInfo.attribute(i).name();
				String stemmedWord = is.stem(originalWord);
				if (!stemmedWords.contains(stemmedWord))
					stemmedWords.add(stemmedWord);
				stemmedWordsMapping.put(originalWord, stemmedWord);
			}
		}
		FastVector attInfo = new FastVector(stemmedWords.size());
		for (String s : stemmedWords){
			attInfo.addElement(new Attribute(s));
		}
		attInfo.addElement(instanceInfo.classAttribute());
		outputFormat = new Instances(instanceInfo.relationName()+"_formalized",attInfo,0);
		if (outputFormat != null)
			return true;
		else return false;
	}
		
	public Instances getOutputFormat(){
		return outputFormat;
		
	}
	
	public Instances doFilter(Instances input){
		Instances output = new Instances(outputFormat, input.numInstances());
		for (int i=0;i<input.numInstances();i++){
			Instance instance = input.instance(i);
			Instance outputInstance = new SparseInstance(0);
			for (int j=0;j < instance.numAttributes(); j++){
				if (input.attribute(i).equals(input.classAttribute())){
					outputInstance.setClassValue(instance.classValue());
				}else {
					String originalWord = instance.attribute(j).name();
					if (stemmedWordsMapping.containsKey(originalWord)){
						String formalWord = stemmedWordsMapping.get(originalWord);
					}
				}
			}
			output.add(outputInstance);
		}
		return output;
	}
}
