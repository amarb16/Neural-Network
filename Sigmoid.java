public class Sigmoid {
    
    /**
     * This method can be called with Sigmoid.sig(_____);
     */
    public static double sig(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }
    
}