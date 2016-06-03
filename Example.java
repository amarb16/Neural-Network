import java.util.Arrays;

/**
 * A representation of an example.
 * Fields are public since this is just a container class.
 * @author Steven Bogaerts
 */
public class Example {
    
    /**
     * The inputs to the network.
     */
    public double[] inputs;
    
    /**
     * The expected outputs, or null if unknown.
     */
    public double[] outputs;
    
    /**
     * The name of this Example, if applicable. (empty string otherwise)
     */
    public String name;
    
    /**
     * Create an Example with the given inputs.
     * Outputs will be null, and name will be the empty string.
     */
    public Example(double[] inputs) {
        this(inputs, null, "");
    }
    
    /**
     * Create an Example with the given inputs and outputs.
     * name will be the empty string.
     */
    public Example(double[] inputs, double[] outputs) {
        this(inputs, outputs, "");
    }
    
    /**
     * Create an Example with the given inputs, outputs, and name.
     */
    public Example(double[] inputs, double[] outputs, String name) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.name = name;
    }
    
    public String toString() {
        return name + ": " + Arrays.toString(inputs) + " --> " + Arrays.toString(outputs);
    }
    
}