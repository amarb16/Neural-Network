import java.util.Random;

/**
 * A weight generator.
 */
public class RandomWeightGenerator {
    
    private double range;
    private Random rand;
    
    /**
     * Creates a weight generator giving values in the range
     * -range to range.
     */
    public RandomWeightGenerator(double range) {
        this.range = range;
        this.rand = new Random();
    }
    
    /**
     * Returns the generated weight.
     */
    public double generateWeight() {
        return range * (rand.nextDouble() * 2 - 1);
    }
    
}
