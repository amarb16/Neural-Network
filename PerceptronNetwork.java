import java.util.Arrays;

/**
 * Represents a network of perceptrons. (Thus, each perceptron is essentially independent.)
 * Of course, this required a completed Perceptron class in order to operate.
 */
public class PerceptronNetwork {
    
    private Perceptron[] perceptrons;
    
    /**
     * Create a perceptron network with the specified parameters.
     */
    public PerceptronNetwork(int numInputs, int numOutputs, double learningRate) {
        this.perceptrons = new Perceptron[numOutputs];
        
        for(int pID = 0; pID < perceptrons.length; pID++)
            perceptrons[pID] = new Perceptron(numInputs, 0.1, learningRate);
    }
    
    /**
     * Edit this method as you see fit, and then call it with:
     * PerceptronNetwork.makeHardcoded()
     * to create a perceptron network with weights hardcoded as you would like,
     * for testing by hand.
     */
    public static PerceptronNetwork makeHardcoded() {
        PerceptronNetwork net = new PerceptronNetwork(2, 2, 1);
        
        // Just making two identical perceptrons here
        Perceptron p1 = Perceptron.makeHardcoded();
        Perceptron p2 = Perceptron.makeHardcoded();
        
        net.perceptrons = new Perceptron[] {p1, p2};
        
        return net;
    }
    
    /**
     * Computes the output the given network would obtain on the given example's inputs.
     * Simply calls computeOutput in the Perceptron class, on each perceptron in the network.
     */
    public double[] computeOutput(double[] inputs) {
        double[] outputs = new double[perceptrons.length];
        
        for(int pID = 0; pID < perceptrons.length; pID++)
            outputs[pID] = perceptrons[pID].computeOutput(inputs);
            
        return outputs;
    }
    
    /**
     * Updates the weights on each perceptron in the network, according to the given example.
     * Simply calls train1Example in the Perceptron class, on each perceptron in the network.
     */
    public void train1Example(Example ex) {
        for(int pID = 0; pID < perceptrons.length; pID++)
            perceptrons[pID].train1Example(ex.inputs, ex.outputs[pID]);
    }
    
    /**
     * Updates the weights on each perceptron in the network, according to the given set of examples, 
     * trained for the given number of epochs.
     * Simply calls trainEpochs in the Perceptron class, on each perceptron in the network.
     */
    public void trainEpochs(Example[] examples, int numEpochs) {
        for(int pID = 0; pID < perceptrons.length; pID++)
            perceptrons[pID].trainEpochs(examples, pID, numEpochs);
    }
    
    /**
     * Computes the output of this network on each of the given examples,
     * and determines the average error on this output.
     * 
     * @param outputLevel Set to 0 to print only the average error, 1 to also identify each example by name and print its error,
     * or 2 to list each full example and print its error.
     */
    public void runTestSet(Example[] examples, int outputLevel) {
        double errorSum = 0;
        double[] output;
        for(int exID = 0; exID < examples.length; exID++) {
            output = computeOutput(examples[exID].inputs);
            
            for(int outputID = 0; outputID < perceptrons.length; outputID++)
                errorSum += Math.abs(examples[exID].outputs[outputID] - output[outputID]);
                
            if (outputLevel == 1)
                System.out.println("Example " + examples[exID].name + ": " + Arrays.toString(output));
            else if (outputLevel == 2)
                System.out.println("Example " + examples[exID] + ": " + Arrays.toString(output));
        }
        System.out.println("Average output node error: " + (errorSum / (examples.length * examples[0].outputs.length)));
    }
    
    /**
     * Provides a String representation of this network.
     */
    public String toString() {
        return Arrays.toString(perceptrons);
    }
    
    /**
     * Tests the network code of this class. Really just creates a network of two perceptrons that
     * are exactly the same.
     */
    public static void testPNet() {
        PerceptronNetwork net = PerceptronNetwork.makeHardcoded();
        System.out.println(net);
        
        Example ex1 = new Example(new double[] {-0.5, 0.3}, new double[] {0.6, 0.6});
        Example ex2 = new Example(new double[] {0.2, -0.4}, new double[] {0.8, 0.8});
        
        net.trainEpochs(new Example[] {ex1, ex2}, 1);
        System.out.println(net);
    }
    
    /**
     * Creates and trains a network to learn AND.
     */
    public static void learnAnd(int numEpochs) {
        Example ex1 = new Example(new double[] {0, 0}, new double[] {0});
        Example ex2 = new Example(new double[] {0, 1}, new double[] {0});
        Example ex3 = new Example(new double[] {1, 0}, new double[] {0});
        Example ex4 = new Example(new double[] {1, 1}, new double[] {1});
        
        learnBool(new Example[] {ex1, ex2, ex3, ex4}, numEpochs);
    }
    
    /**
     * Creates and trains a network to learn OR.
     */
    public static void learnOr(int numEpochs) {
        Example ex1 = new Example(new double[] {0, 0}, new double[] {0});
        Example ex2 = new Example(new double[] {0, 1}, new double[] {1});
        Example ex3 = new Example(new double[] {1, 0}, new double[] {1});
        Example ex4 = new Example(new double[] {1, 1}, new double[] {1});
        
        learnBool(new Example[] {ex1, ex2, ex3, ex4}, numEpochs);
    }
    
    /**
     * Creates and trains a network to learn XOR.
     * Will fail, of course, since XOR is not linearly separable.
     */
    public static void learnXor(int numEpochs) {
        Example ex1 = new Example(new double[] {0, 0}, new double[] {0});
        Example ex2 = new Example(new double[] {0, 1}, new double[] {1});
        Example ex3 = new Example(new double[] {1, 0}, new double[] {1});
        Example ex4 = new Example(new double[] {1, 1}, new double[] {0});
        
        learnBool(new Example[] {ex1, ex2, ex3, ex4}, numEpochs);
    }
    
    /**
     * A helper method for the boolean function learning examples above.
     */
    private static void learnBool(Example[] examples, int numEpochs) {
        PerceptronNetwork net = new PerceptronNetwork(2, 1, 0.5);
        
        net.trainEpochs(examples, numEpochs);
        System.out.println(net);
        net.runTestSet(examples, 2);
    }
    
}