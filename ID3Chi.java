/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * Copied original implementation of Id3 by 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 6404 $ 
*/

//package weka.classifiers.trees;

import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;

/**
 * <!-- globalinfo-start --> Class for constructing an unpruned decision tree
 * based on the ID3 algorithm. Can only deal with nominal attributes. No missing
 * values allowed. Empty leaves may result in unclassified instances. For more
 * information see: <br/>
 * <br/>
 * R. Quinlan (1986). Induction of decision trees. Machine Learning.
 * 1(1):81-106.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;article{Quinlan1986,
 *    author = {R. Quinlan},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {81-106},
 *    title = {Induction of decision trees},
 *    volume = {1},
 *    year = {1986}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 *  -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 *
 */
public class ID3Chi extends Classifier {

	/** for serialization */
	static final long serialVersionUID = -2693678647096322561L;

	/** The node's successors. */
	private ID3Chi[] m_Successors;

	/** Attribute used for splitting. */
	private Attribute m_Attribute;

	/** Class value if node is leaf. */
	private double m_ClassValue;

	/** Class distribution if node is leaf. */
	private double[] m_Distribution;

	/** Class attribute of dataset. */
	private Attribute m_ClassAttribute;
	
	private double m_Ratio;

	/** Confidence level 0.95 for 95%, 0.99 for 99%, etc */
	private double m_confidenceLevel;

	static final int MinNumberOfExpectedValuesForChi = 5;
	static final double MaxPercentageOfSmallCategoriesForChi = 0.2;
	
	public ID3Chi(double confidenceLevel) {
		m_confidenceLevel = confidenceLevel;	
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);

		// Use proportional distribution and token classification approach from
		// ID3 paper
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		
		// don't allow missing class values
		//result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Builds ID3Chi decision tree classifier.
	 *
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if classifier can't be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		makeTree(data);
	}

	/**
	 * Method for building an ID3Chi tree.
	 *
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if decision tree can't be built successfully
	 */
	private void makeTree(Instances data) throws Exception {

		// Check if no instances have reached this node.
		/*
		if (data.numInstances() == 0) {
			m_Attribute = null;
			m_ClassValue = Instance.missingValue();
			m_Distribution = new double[data.numClasses()];
			return;
		}
		/**/
		if (data.numInstances() == 0) {
			SetNullDistribution(data);
		}

		// Compute attribute with maximum information gain.
		double[] infoGains = new double[data.numAttributes()];
		Enumeration attEnum = data.enumerateAttributes();
		double entropyOfAllData = computeEntropy(data);

		while (attEnum.hasMoreElements()) {
			Attribute att = (Attribute) attEnum.nextElement();
			infoGains[att.index()] = computeInfoGain(data, att, entropyOfAllData);
		}
		m_Attribute = data.attribute(Utils.maxIndex(infoGains));

		double chiSquare = computeChiSquare(data, m_Attribute);

		int degreesOfFreedom = m_Attribute.numValues() - 1;
		ChiSquaredDistribution chi = new ChiSquaredDistribution(degreesOfFreedom);
		double threshold = chi.inverseCumulativeProbability(m_confidenceLevel);

		// Make leaf if information gain is zero.
		// Otherwise create successors.
		if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
			MakeALeaf(data);
		} else {
			// Discard unknown values for selected attribute
			//data.deleteWithMissing(m_Attribute);
			Instances[] subset = splitData(data, m_Attribute);

			if (CheckIfCanApplyChiSquare(subset) && (chiSquare <= threshold)) {
				MakeALeaf(data);
				return;
			}

			m_Successors = new ID3Chi[m_Attribute.numValues()];
			for (int j = 0; j < m_Attribute.numValues(); j++) {
				m_Successors[j] = new ID3Chi(this.m_confidenceLevel);
				m_Successors[j].m_Ratio = (double)subset[j].numInstances()/(double)data.numInstances();
				m_Successors[j].makeTree(subset[j]);
			}
		}
	}

	private void MakeALeaf(Instances data) {

		data.deleteWithMissing(m_Attribute);
		
		if (data.numInstances() == 0) {
			SetNullDistribution(data);
			return;
		}		
		
		m_Distribution = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			m_Distribution[(int) inst.classValue()]++;
		}
		Utils.normalize(m_Distribution);
		m_ClassValue = Utils.maxIndex(m_Distribution);
		m_ClassAttribute = data.classAttribute();
		
		// set m_Attribute to null to mark this node as a leaf
		m_Attribute = null;
	}
	
	private void SetNullDistribution(Instances data) {
		m_Attribute = null;
		m_ClassValue = Instance.missingValue();
		
		// TODO: think if it's better to keep all the distributions equal to 0
		m_Distribution = new double[data.numClasses()];
		for (int i=0; i < m_Distribution.length; i++) {
			m_Distribution[i] = 1.0 / (double) data.numClasses();
		}		
	}

	private boolean CheckIfCanApplyChiSquare(Instances[] subset) {

		int splitLessThan5 = 0;
		for (int j = 0; j < subset.length; j++) {
			if (subset[j].numInstances() < MinNumberOfExpectedValuesForChi) {
				splitLessThan5++;
			}
		}

		return (splitLessThan5 / subset.length) < MaxPercentageOfSmallCategoriesForChi;
	}

	/**
	 * Classifies a given test instance using the decision tree.
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return the classification
	 * @throws NoSupportForMissingValuesException
	 *             if instance has missing values
	 */
	public double classifyInstance(Instance instance) {

		double [] tokenDistribution = classifyInstanceWithToken(instance, 1.0);
		return Utils.maxIndex(tokenDistribution);
	}

	private double [] classifyInstanceWithToken(Instance instance, double token) {

		int numClasses = instance.numClasses();
		double [] tokenDistribution = new double[numClasses];
		if (m_Attribute == null) {
			for (int j = 0; j < numClasses; j++) {
				tokenDistribution[j] = token * m_Distribution[j];			
			}
		} else {
			// for attribute values get token distribution
			if (instance.isMissing(m_Attribute)) {
				for (int j = 0; j < m_Attribute.numValues(); j++) {
					double [] dist = m_Successors[j].classifyInstanceWithToken(instance, token * m_Successors[j].m_Ratio);
					for (int i = 0; i < numClasses; i++) {
						tokenDistribution[i] += dist[i];			
					}				
				}
			} else {
				int idx = (int)instance.value(m_Attribute);
				tokenDistribution = m_Successors[idx].classifyInstanceWithToken(instance, token * m_Successors[idx].m_Ratio);				
			}
		}
		
		return tokenDistribution;
	}
	
	public String convertClassValueToString(Attribute classAttribute, double classValue) {
		return classAttribute.value((int) classValue);
	}

	/**
	 * Computes class distribution for instance using decision tree.
	 *
	 * @param instance
	 *            the instance for which distribution is to be computed
	 * @return the class distribution for the given instance
	 * @throws NoSupportForMissingValuesException
	 *             if instance has missing values
	 */
	public double[] distributionForInstance(Instance instance) {

		if (m_Attribute == null) {
			return m_Distribution;
		} else {
			return m_Successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
		}
	}

	/**
	 * Prints the decision tree using the private toString method from below.
	 *
	 * @return a textual description of the classifier
	 */
	public String toString() {

		if ((m_Distribution == null) && (m_Successors == null)) {
			return "ID3Chi: No model built yet.";
		}
		return "ID3Chi\n\n" + toString(0);
	}

	/**
	 * Computes Chi-Square function for an attribute.
	 *
	 * @param data
	 *            the data for which info gain is to be computed
	 * @param att
	 *            the attribute
	 * @return the chi-square for the given attribute and data
	 * @throws Exception
	 *             if computation fails
	 */
	private double computeChiSquare(Instances data, Attribute att) throws Exception {

		double chiSquare = 0;
		double[] classCounts = GetClassCounts(data);
		Instances[] subset = splitData(data, att);
		for (int j = 0; j < att.numValues(); j++) {
			if (subset[j].numInstances() > 0) {
				chiSquare += computeChiSquareForSubset(subset[j], att, classCounts, data.numInstances());
			}
		}
		return chiSquare;
	}

	/**
	 * Computes Chi-Square element for given subset.
	 *
	 * @param subset
	 *            the data for which info gain is to be computed
	 * @param att
	 *            the attribute
	 * @setClassCounts class counts for initial set of instances
	 * @setNumInstances number of instances for set of data
	 * @return the chi-square for the given attribute and data
	 * @throws Exception
	 *             if computation fails
	 */
	private double computeChiSquareForSubset(Instances subset, Attribute att, double[] setClassCounts,
			double setNumInstances) {

		double[] subsetClassCounts = GetClassCounts(subset);
		double result = 0;
		double d = subset.numInstances() / setNumInstances;
		for (int j = 0; j < subset.numClasses(); j++) {
			double ciNew = setClassCounts[j] * d;
			if (ciNew > 0) {
				result += Math.pow(subsetClassCounts[j] - ciNew, 2) / ciNew;
			}
		}
		return result;
	}

	/**
	 * Computes information gain for an attribute.
	 *
	 * @param data
	 *            the data for which info gain is to be computed
	 * @param att
	 *            the attribute
	 * @param entropyOfAllData
	 *            entropy of data set
	 * @return the information gain for the given attribute and data
	 * @throws Exception
	 *             if computation fails
	 */
	private double computeInfoGain(Instances data, Attribute att, double entropyOfAllData) throws Exception {

		double infoGain = entropyOfAllData;
		Instances[] subset = splitData(data, att);
		
		int numUnknown = subset[att.numValues()].numInstances();		
		if (numUnknown == data.numInstances()) {
			return 0;
		}
		
		double[] classCountsUnknownData = GetClassCounts(subset[att.numValues()]);
		
		for (int j = 0; j < att.numValues(); j++) {
			if (subset[j].numInstances() > 0) {
				double ratio = (double)subset[j].numInstances()/(double)data.numInstances();
				infoGain -= (((double) subset[j].numInstances() + (double)numUnknown * ratio) / (double) data.numInstances())
						* computeEntropyWithUnknowns(subset[j], subset[att.numValues()], classCountsUnknownData, ratio);
			}
		}
		return infoGain;
	}

	/**
	 * Computes the entropy of a dataset.
	 * 
	 * @param data
	 *            the data for which entropy is to be computed
	 * @return the entropy of the data's class distribution
	 * @throws Exception
	 *             if computation fails
	 */
	private double computeEntropy(Instances data) throws Exception {

		double[] classCounts = GetClassCounts(data);
		double entropy = 0;
		for (int j = 0; j < data.numClasses(); j++) {
			if (classCounts[j] > 0) {
				entropy -= classCounts[j] * Utils.log2(classCounts[j]);
			}
		}
		entropy /= (double) data.numInstances();
		return entropy + Utils.log2(data.numInstances());
	}

	private double computeEntropyWithUnknowns(Instances data, Instances unknownData, double[] classCountsUnknownData, double ratio) throws Exception {

		double[] classCounts = GetClassCounts(data);
		double entropy = 0;
		for (int j = 0; j < data.numClasses(); j++) {
			double p = classCounts[j] + classCountsUnknownData[j] * ratio;
			if (p > 0) {
				entropy -= p * Utils.log2(p);
			}
		}
		entropy /= (double) data.numInstances();
		return entropy + Utils.log2(data.numInstances());
	}
	
	private double[] GetClassCounts(Instances data) {

		double[] classCounts = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			classCounts[(int) inst.classValue()]++;
		}
		return classCounts;
	}

	/**
	 * Splits a dataset according to the values of a nominal attribute.
	 *
	 * @param data
	 *            the data which is to be split
	 * @param att
	 *            the attribute to be used for splitting
	 * @return the sets of instances produced by the split
	 */
	private Instances[] splitData(Instances data, Attribute att) {

		// [att.numValues()] is location for "unknown" values
		Instances[] subset = new Instances[att.numValues()+1];
		for (int j = 0; j <= att.numValues(); j++) {
			subset[j] = new Instances(data, data.numInstances());
		}
		
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			if (inst.isMissing(att)) {
				subset[att.numValues()].add(inst);
			} else {
				subset[(int) inst.value(att)].add(inst);
			}
		}
		for (int i = 0; i < subset.length; i++) {
			subset[i].compactify();
		}
		return subset;
	}

	/**
	 * Outputs a tree at a certain level.
	 *
	 * @param level
	 *            the level at which the tree is to be printed
	 * @return the tree as string at the given level
	 */
	private String toString(int level) {

		StringBuffer text = new StringBuffer();

		if (m_Attribute == null) {
			if (Instance.isMissingValue(m_ClassValue)) {
				text.append(": null");
			} else {
				text.append(": " + m_ClassAttribute.value((int) m_ClassValue));
			}
		} else {
			for (int j = 0; j < m_Attribute.numValues(); j++) {
				text.append("\n");
				for (int i = 0; i < level; i++) {
					text.append("|  ");
				}
				text.append(m_Attribute.name() + " = " + m_Attribute.value(j));
				text.append(m_Successors[j].toString(level + 1));
			}
		}
		return text.toString();
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 6404 $");
	}

	/**
	 * Main method.
	 *
	 * @param args
	 *            the options for the classifier
	 */
	/*
	public static void main(String[] args) {
		runClassifier(new ID3Chi(), args);
	}
	/**/
}
