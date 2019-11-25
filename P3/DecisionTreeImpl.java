import java.util.ArrayList;
import java.util.List;
/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 */
public class DecisionTreeImpl {
	public DecTreeNode root;
	public List<List<Integer>> trainData;
	public int maxPerLeaf;
	public int maxDepth;
	public int numAttr;
	public int depth;

	// Build a decision tree given a training set
    DecisionTreeImpl(List<List<Integer>> trainDataSet, int mPerLeaf, int mDepth) {
        this.trainData = trainDataSet;
        this.maxPerLeaf = mPerLeaf;
        this.maxDepth = mDepth;
        this.depth = 0;
        if (this.trainData.size() > 0) this.numAttr = trainDataSet.get(0).size() - 1;
        this.root = buildTree();
    }
    
	// Build a decision tree given a training set with depth 
	DecisionTreeImpl(List<List<Integer>> trainDataSet, int mPerLeaf, int mDepth , int depth) {
		this.trainData = trainDataSet;
		this.maxPerLeaf = mPerLeaf;
		this.maxDepth = mDepth;
		this.depth = depth;
		if (this.trainData.size() > 0) this.numAttr = trainDataSet.get(0).size() - 1;
		this.root = buildTree();
	}
	
    //calculate the class's entropy
    private static double classentro(List<List<Integer>> data) {
    	int oneNumber = 0;
        int zeroNumber = 0;     
        int sizex = data.size();
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).get(data.get(0).size() - 1) != 0) {
            	oneNumber++;          	
            }else {
            	zeroNumber++;
            }     	
        }
        double x = (double) zeroNumber / sizex;
        double y = (double) oneNumber / sizex;
        return calculateH(x, y);
    }
    
    // Calculates the entropy of H(x,y)
    private static double calculateH (double i, double j) {
    	double dataEntropy = 0.0;
        if (i == 0 || j == 0) {
        	return 0.0;
        } else {
        	dataEntropy = (-i) * (Math.log(i) / Math.log(2)) + (-j) * (Math.log(j) / Math.log(2));
        }
        return dataEntropy;
    }
    
    // 2-D array to store best threshold and best information gain 
    private static double[][] bestThresholdAndGain(List<List<Integer>> data) {
    	//Because there are nine columns are attributes
    	// we use the first line to store the best threshold and second line to store the best info gain 
        double table[][] = new double[2][9];

        for (int x = 0; x <= 8; x++) {
        	int bestarr = -1;
            double best = -1;
            for (int y = 1; y <= 9; y++) {
            	int countS = 0;
                int countB = 0;            
                int s0 = 0;
                int s1 = 0;
                int b0 = 0;
                int b1 = 0;  
                double entropy = 0.0;
                for (int z = 0; z < data.size(); z++) {
                    if (data.get(z).get(x) > y) {
                        countB++;
                        if (data.get(z).get(9) != 0)
                        	b1++;
                        else
                            b0++;
                    } else {
                        countS++;
                        if (data.get(z).get(9) != 0)
                        	s1++;
                        else
                            s0++;
                    }
                }
                if (countB == 0) {
                	entropy = calculateH((double) s0 / countS, (double) s1 / countS);
                } else if (countS == 0) {
                	entropy = calculateH((double) b0 / countB, (double) b1 / countB);
                } else {
                	entropy = ((double) countS / data.size() * calculateH((double) s0 / countS, (double) s1 / countS) 
                			+ (double) countB / data.size() * calculateH((double) b0 / countB, (double) b1 / countB));
                }                
                double gain = classentro(data) - entropy;
                if (gain > best) {
                    best = gain;
                    bestarr = y;
                }
            }            
            table[0][x] = (double) bestarr;
            table[1][x] = best; 
        }
        return table;
    }
    
    // find the best attribute through the table 
    private static int bestAttribute(List<List<Integer>> data) {
    	double[][] newTable = bestThresholdAndGain(data);
        double best = 0.0;
        int attribute = -1;
        for (int i = 0; i <= 8; i++) {    
            double temp = newTable[1][i];
            if (temp > best) {
                best = temp;
                attribute = i;
            }
        }
        return attribute;
    }
    
    // find the best information gain through the table 
    private static double bestInfoGain(List<List<Integer>> data) {
        double[][] newTable = bestThresholdAndGain(data);
        double bestInfoGain = 0;
        for (int i = 0; i <= 8; i++) {
            double tem = newTable[1][i];
            if (tem > bestInfoGain)
            	bestInfoGain = tem;
        }
        return bestInfoGain;
    }
   
    // Find the best threshold of an attribute
    private static int bestThreshold(List<List<Integer>> data, int attr) {
    	return  (int) bestThresholdAndGain(data)[0][attr];
    }
    
	private DecTreeNode buildTree() {
		// TODO: add code here	
		int attribute = 0;
		int classLable = 1;
		int threshold = 0;
		if (trainData.size() != 0) {
			attribute = bestAttribute(trainData);
			if (attribute != -1) {
				threshold = bestThreshold(trainData, attribute);
			}
		}else {
			attribute = 0;
		}
		if (depth == maxDepth || trainData.size() <= maxPerLeaf || bestInfoGain(trainData) == 0) {
			int oneNumber = 0;
			int zeroNumber = 0;
            for (List<Integer> row : trainData) {
                if (row.get(row.size() - 1) == 1) {
                	oneNumber++;
                } else if (row.get(row.size() - 1) == 0) {
                	zeroNumber++;
                }
            }
            if (zeroNumber != oneNumber) {
            	classLable = ((zeroNumber - oneNumber) > 0) ? 0 : 1;
            } else {
            	classLable = 1;
            }
            DecTreeNode node = new DecTreeNode(classLable, 0, 0);
            node.right = null;
            node.left = null;
            return node;
        }
		
		ArrayList<List<Integer>> right = new ArrayList<List<Integer>>();
		ArrayList<List<Integer>> left = new ArrayList<List<Integer>>();    
        for (List<Integer> row : trainData) {
            if (row.get(attribute) > threshold) {
            	right.add(row);
            } else {
            	left.add(row);
            }
        }

        DecTreeNode node = new DecTreeNode(classLable, attribute, threshold);
        DecisionTreeImpl RightTree = new DecisionTreeImpl(right, maxPerLeaf, maxDepth, depth + 1);
        node.right = RightTree.root;
        DecisionTreeImpl leftTree = new DecisionTreeImpl(left, maxPerLeaf, maxDepth, depth + 1);
        node.left = leftTree.root;        
        return node;
	}
	
	public int classify(List<Integer> instance) {
		DecTreeNode current = root;
		return classifyHelper(instance, current);
	}
	
	private int classifyHelper(List<Integer> instance, DecTreeNode node) {
        if (node.isLeaf()) {
            return node.classLabel;
        } else {
        	int threshold = node.threshold;
        	int attribute = node.attribute;         
            if (instance.get(attribute) > threshold) {
            	return classifyHelper(instance, node.right);
            } else {
            	return classifyHelper(instance, node.left);
            }
        }
    }
	
	// Print the decision tree in the specified format
	public void printTree() {
		printTreeNode("", this.root);
	}

	public void printTreeNode(String prefixStr, DecTreeNode node) {
		String printStr = prefixStr + "X_" + node.attribute;
		System.out.print(printStr + " <= " + String.format("%d", node.threshold));
		if(node.left.isLeaf()) {
			System.out.println(" : " + String.valueOf(node.left.classLabel));
		}
		else {
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.left);
		}
		System.out.print(printStr + " > " + String.format("%d", node.threshold));
		if(node.right.isLeaf()) {
			System.out.println(" : " + String.valueOf(node.right.classLabel));
		}
		else {
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.right);
		}
	}
	
	public double printTest(List<List<Integer>> testDataSet) {
		int numEqual = 0;
		int numTotal = 0;
		for (int i = 0; i < testDataSet.size(); i ++)
		{
			int prediction = classify(testDataSet.get(i));
			int groundTruth = testDataSet.get(i).get(testDataSet.get(i).size() - 1);
			System.out.println(prediction);
			if (groundTruth == prediction) {
				numEqual++;
			}
			numTotal++;
		}
		double accuracy = numEqual*100.0 / (double)numTotal;
		System.out.println(String.format("%.2f", accuracy) + "%");
		return accuracy;
	}
}
