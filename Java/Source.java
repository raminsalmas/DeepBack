import org.tensorflow.*;
import org.tensorflow.keras.*;
import org.tensorflow.keras.layers.*;
import org.tensorflow.keras.optimizers.*;
import org.tensorflow.keras.callbacks.EarlyStopping;

import javax.imageio.plugins.jpeg.JPEGImageReadParam;
import java.io.*;
import java.util.*;
import org.tensorflow.Tensor;

public class MyModel {

    private String path = "./Models/";
    private String run = "prediction";
    private Model model = null;

    public MyModel() {
        // Initialize paths and settings
    }

    // Build the neural network model
    public void buildModel() {
        Input inputs_s1 = Input.create(new Shape(1));
        Layer features = Dense.create(1000).setActivation("relu").build().apply(inputs_s1);
        features = Dense.create(800).setActivation("relu").build().apply(features);
        features = Dense.create(500).setActivation("relu").build().apply(features);
        features = Dense.create(150).setActivation("relu").build().apply(features);
        features = Dense.create(100).setActivation("relu").build().apply(features);
        features = Dense.create(150).setActivation("relu").build().apply(features);
        features = Dense.create(500).setActivation("relu").build().apply(features);
        features = Dense.create(800).setActivation("relu").build().apply(features);
        features = Dense.create(1000).setActivation("relu").build().apply(features);
        Layer features_2 = Dense.create(1).build().apply(features);
        Layer features_3 = Dense.create(1).build().apply(features);
        this.model = new Model(inputs_s1, new Layer[]{features_2, features_3});
    }

    // Compile the model
    public void compileModel() {
        Optimizer opt = new Adam();
        this.model.compile(opt, "mean_squared_error");
    }

    // Train the model
    public History trainModel(Tensor X, Tensor FEX, Tensor BEX, Tensor Xv, Tensor FEX_v, Tensor BEX_v) {
        EarlyStopping es = new EarlyStopping().monitor("val_loss").mode(EarlyStopping.Mode.MIN).patience(100);
        History history = this.model.fit(new Tensor[]{X}, new Tensor[]{FEX, BEX}, 1000, 0, 0, null,
                Arrays.asList(Xv), Arrays.asList(FEX_v, BEX_v), null, null, 2, null, null);
        return history;
    }

    // Evaluate the model
    public double[] evaluateModel(Tensor Xv, Tensor FEX_v, Tensor BEX_v) {
        return this.model.evaluate(new Tensor[]{Xv}, new Tensor[]{FEX_v, BEX_v});
    }

    // Save the trained model
    public void saveModel(String filename) {
        this.model.save(this.path + filename + ".h5");
    }

    // Predict using the model
    public Tensor predict(Tensor inputs) {
        return this.model.predict(inputs);
    }

    // Calculate metrics for evaluation
    public double[] calculateMetrics(Tensor predicted, Tensor expected) {
        double[] metrics = new double[2];
        // Calculate R2 score and mean squared error here
        return metrics;
    }

    // Save results to a file
    public void saveResults(String filename, double cor, double err) {
        try (FileWriter writer = new FileWriter(this.path + filename, true)) {
            writer.write(String.format("%.6f, %.6f\n", cor, err));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Plot training and validation loss
    public void plotLoss(History history) {
        // Plot the loss here
    }

    // Plot RFU comparison with expected values
    public void plotRFU(double[] EXP, double[] RFU_BF, double error, double R) {
        // Plot RFU comparison here
    }

    // Calculate BackExcitation values
    public double backExc(double RFU, double BEX, double FEX) {
        return (RFU - FEX) / (BEX - FEX);
    }

    // Process inputs and calculate RFU_BF
    public double[] processInputs(double[] input) {
        double[] RFU = input;
        double[][] EXP = new double[0][]; // Load EXP data from file
        double[][] data = new double[0][]; // Predict using the model
        double[] FEX_f = data[0];
        double[] BEX_f = data[1];
        double[] RFU_BF = new double[0]; // Calculate RFU_BF
        return new double[]{RFU_BF, EXP};
    }

    // Main function
    public static void main(String[] args) {
        double[][] dataset = new double[0][]; // Load dataset from file
        double[][] validation_ = new double[0][]; // Load validation data from file

        // Process validation data
        double[] Xv = Arrays.copyOfRange(validation_[0], 0, 7);
        double[] Yv = Arrays.copyOfRange(validation_[0], 7, 9);
        double[] FEX_v = Arrays.copyOfRange(Yv, 0, 1);
        double[] BEX_v = Arrays.copyOfRange(Yv, 1, 2);

        // Shuffle dataset
        Random rand = new Random();
        for (int i = dataset.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            double[] temp = dataset[i];
            dataset[i] = dataset[j];
            dataset[j] = temp;
        }

        double[] X = Arrays.copyOfRange(dataset[0], 0, 7);
        double[] Y = Arrays.copyOfRange(dataset[0], 7, 9);
        double[] FEX = Arrays.copyOfRange(Y, 0, 1);
        double[] BEX = Arrays.copyOfRange(Y, 1, 2);

        // Initialize the model
        MyModel myModel = new MyModel();
        myModel.buildModel();
        myModel.compileModel();

        // Train the model and get history
        History history = myModel.trainModel(Tensor.create(X), Tensor.create(FEX), Tensor.create(BEX),
                Tensor.create(Xv), Tensor.create(FEX_v), Tensor.create(BEX_v));
        double[] pp = myModel.evaluateModel(Tensor.create(Xv), Tensor.create(FEX_v), Tensor.create(BEX_v));

        // Plot loss history
        myModel.plotLoss(history);

        // Calculate metrics and save results
        double[] metrics = myModel.calculateMetrics(Tensor.create(pp), history);
        myModel.saveResults("cor_3.txt", metrics[0], metrics[1]);

        // Save trained model
        myModel.saveModel("model_BK_FX_new_no_alpha_1_systems_Cor_" + metrics[0] + "_" + metrics[1] + "_" + metrics[0] + "_" + metrics[0]);

        // Process input data, calculate RFU_BF, and plot
        double[] input =
