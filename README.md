// Импортируем необходимые классы
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.optimizers.Adam;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.api.iterator.Iterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
// Uploading the data

// Neural Network configuration
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(123)
        .iterations(100)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .learningRate(0.01)
        .weightInit(WeightInit.XAVIER)
        .updater(new Adam(0.01))
        .list(3)
        .layer(0, new DenseLayer.Builder()
                .nIn(numInputFeatures)
                .nOut(128)
                .activation(Activation.RELU)
                .build())
        .layer(1, new DenseLayer.Builder()
                .nIn(128)
                .nOut(64)
                .activation(Activation.RELU)
                .build())
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(64)
                .nOut(numOutputClasses)
                .build())
                   .layer(1, new DenseLayer.Builder()
                .nIn(128)
                .nOut(64)
                .activation(Activation.RELU)
                .build())
        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMIN)
                .nIn(64)
                .nOut(numOutputClasses)
                .build())
                   .layer(3, new DenseLayer.Builder()
                .nIn(128)
                .nOut(64)
                .activation(Activation.RELU)
                .build())
        .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.POSITIVELOGLIKELIHOOD)
                .activation(Activation.Swish)
                .nIn(64)
                .nOut(numOutputClasses)
                .build())
        .pretrain(false)
        .backprop(true)
        .build();

// Creating a neural network
MultiLayerNetwork model = new MultiLayerNetwork(conf);
model.init();

// Model Training
model.fit(dataIterator);

// Using a trained model for forecasting
//  using the model
