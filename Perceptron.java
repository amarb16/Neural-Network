import java.util.Random;
import java.util.Arrays;

public class Perceptron {
    
    private double learningRate;
    
    /**
     * Since the bias input is always -1, in my opinion it's easier to have the bias weight
     * be totally separate, rather than part of the regular weights array.
     * But it's also easy to forget about the bias weights sometimes, 
     * so be careful! I recommend adding notes to yourself like "BIAS" 
     * for each function that handles weights, to make sure you handle 
     * the bias weights properly.
     */
    private double biasWeight;
    
    /**
     * weights[j] is the weight from input j into this unit.
     * Note that weights[0] is a regular weight. The bias weight is stored in the biasWeight field.
     */
    private double[] weights;
    
    public Perceptron(int numInputs, double weightRange, double learningRate) {
        this.weights = new double[numInputs];
        RandomWeightGenerator wtGen = new RandomWeightGenerator(weightRange);
        for(int j = 0; j < weights.length; j++)
            weights[j] = wtGen.generateWeight();
            
        this.biasWeight = wtGen.generateWeight();
        this.learningRate = learningRate;
    }
    
    /**
     * Get a string representation of this perceptron.
     */
    public String toString() {
        return "{" + biasWeight + ", " + Arrays.toString(weights) + "}";
    }
    
    /**
     *  Edit this method as you see fit, and then call it with:
     *  Perceptron.makeHardcoded()
     *  to create a neuron with weights hardcoded as you would like,
     *  for testing by hand.
     */
    public static Perceptron makeHardcoded() {
        Perceptron p = new Perceptron(2, 0, 1);
        p.biasWeight = 0.3;
        p.weights = new double[] {0.1, -0.2};
        return p;
    }
    
    /**
     * Determine the output this perceptron gives on the given inputs.
     */
    public double computeOutput(double[] inputs) {
        double calc = biasWeight * -1;
        for(int i = 0; i < weights.length; i++){
            calc += weights[i]*inputs[i];
        }
        return 1 / (1 + Math.exp(-calc));
    }
    
    /**
     * Update the weights based on correctOutput for the given inputs.
     */
    public void train1Example(double[] inputs, double correctOutput) {
        double g = computeOutput(inputs);
        double err = correctOutput - g;
        biasWeight = biasWeight + learningRate * err * g * (1-g)* -1;
        for(int i = 0; i < weights.length; i++){
            weights[i] = weights[i] + learningRate * err * g * (1-g)* inputs[i];
        }
    }
    
    /**
     * Update the weights by training on the given examples.
     * 
     * @param examples An array of Example objects.
     * examples[0] is the Example object at index 0.
     * examples[0].inputs is the array of doubles representing the inputs (but not the bias).
     * examples[0].inputs[0] is the first (non-bias) input.
     * examples[0].outputs is the array of doubles representing the ouputs.
     * examples[0].outputs[perceptronID] is the correct output for this perceptron.
     * 
     * @param perceptronID The identification of this perceptron, so we know which example output
     * to train on.
     */
    //this is a higer level method
    public void trainEpochs(Example[] examples, int perceptronID, int numEpochs) {
        for( int n = 0; n < numEpochs; n++ ){//for each taining example, consider the input a
            for(int p = 0; p < examples.length; p++) //each perceptron i with corresponding target output fi(a)
            {
                //train1Example(examples[0].inputs[0], -1);
                train1Example(examples[p].inputs, examples[p].outputs[perceptronID]);
                //update Wj,i according to the weight update rule, we already did that in Train1Example
            }
        }
    }
    
    public static void testPerceptronOutput() {
        Perceptron p = Perceptron.makeHardcoded();
        System.out.println(p);
        System.out.println("Output: " + p.computeOutput(new double[] {-0.5, 0.3}));
    }
    
    public static void testPerceptronLearning() {
        Perceptron p = Perceptron.makeHardcoded();
        System.out.println(p);
        
        p.train1Example(new double[] {-0.5, 0.3}, 0.6);
        System.out.println(p);
    }
    
    public static void testPerceptronLearning2() {
        Perceptron p = Perceptron.makeHardcoded();
        System.out.println(p);
        
        Example ex1 = new Example(new double[] {-0.5, 0.3}, new double[] {0.6});
        Example ex2 = new Example(new double[] {0.2, -0.4}, new double[] {0.8});
        
        p.trainEpochs(new Example[] {ex1, ex2}, 0, 1);
        System.out.println(p);
    }
    
}