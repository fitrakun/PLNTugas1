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
import weka.filters.Filter;

import IndonesianNLP.IndonesianSentenceFormalization;


public class WordVectorFormalizer {

	private Instances outputFormat = null;
	Map<String,String> formalWordsMapping = null;
	Set<String> formalWords = null;

	public boolean setInputFormat(Instances instanceInfo){
		formalWords = new HashSet<String>();
		formalWordsMapping = new HashMap<String,String>();
		IndonesianSentenceFormalization isf = new IndonesianSentenceFormalization();
		for (int i=0; i<instanceInfo.numAttributes();i++){
			if (instanceInfo.attribute(i).equals(instanceInfo.classAttribute())){
				//do nothing
			}else {
				String originalWord = instanceInfo.attribute(i).name();
				String formalWord = isf.formalizeWord(originalWord);
				if (!formalWords.contains(formalWord))
					formalWords.add(formalWord);
				formalWordsMapping.put(originalWord, formalWord);
			}
		}
		FastVector attInfo = new FastVector(formalWords.size());
		for (String s : formalWords){
			attInfo.addElement(new Attribute(s));
		}
		attInfo.addElement(instanceInfo.classAttribute());
		outputFormat = new Instances(instanceInfo.relationName()+"_formalized",attInfo,0);
		outputFormat.setClassIndex(outputFormat.numAttributes()-1);
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
			Instance outputInstance = new SparseInstance(instance.numAttributes());
			outputInstance.setDataset(output);
			output.add(outputInstance);
			for (int j=0;j < instance.numAttributes(); j++){
				if (input.attribute(i).equals(input.classAttribute())){
					outputInstance.setClassValue(instance.stringValue(instance.classAttribute()));
				}else {
					String originalWord = instance.attribute(j).name();
					if (formalWordsMapping.containsKey(originalWord)){
						String formalWord = formalWordsMapping.get(originalWord);
						if (outputInstance.isMissing(output.attribute(formalWord))){
							outputInstance.setValue(output.attribute(formalWord), instance.value(i));
						}else{
							outputInstance.setValue(output.attribute(formalWord), outputInstance.value(i)+instance.value(i));
						}
					}
				}
			}
			for (int j=0;j < outputInstance.numAttributes(); j++){
				if (outputInstance.isMissing(j)){
					outputInstance.setValue(j, 0);
				}
			}
		}
		return output;
	}
}
