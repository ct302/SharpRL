using System;
using System.Linq;

namespace SharpRL.Agents
{
    /// <summary>
    /// Simple neural network interface for Q-function approximation
    /// This would ideally integrate with SharpGrad
    /// </summary>
    public interface INeuralNetwork
    {
        double[] Forward(double[] input);
        void Train(double[] input, double[] target);
        void CopyWeightsFrom(INeuralNetwork other);
        void Save(string path);
        void Load(string path);
    }
    
    /// <summary>
    /// Basic feedforward neural network implementation
    /// Like the brain of your AI coach - processes game state and outputs play values
    /// </summary>
    public class SimpleNeuralNetwork : INeuralNetwork
    {
        private readonly int inputSize;
        private readonly int outputSize;
        private readonly int[] hiddenSizes;
        private readonly double learningRate;
        private readonly Random random;
        
        // Network layers
        private double[][] weights;
        private double[][] biases;
        private double[][] activations;
        
        public SimpleNeuralNetwork(int inputSize, int outputSize, int[] hiddenSizes, double learningRate = 0.001)
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.hiddenSizes = hiddenSizes;
            this.learningRate = learningRate;
            this.random = new Random();
            
            InitializeNetwork();
        }
        
        private void InitializeNetwork()
        {
            int totalLayers = hiddenSizes.Length + 1;
            weights = new double[totalLayers][];
            biases = new double[totalLayers][];
            activations = new double[totalLayers + 1][];
            
            // Initialize weights and biases with Xavier/He initialization
            int prevSize = inputSize;
            for (int i = 0; i < totalLayers; i++)
            {
                int currSize = (i < hiddenSizes.Length) ? hiddenSizes[i] : outputSize;
                
                weights[i] = new double[prevSize * currSize];
                biases[i] = new double[currSize];
                
                // Xavier initialization
                double stdDev = Math.Sqrt(2.0 / prevSize);
                for (int j = 0; j < weights[i].Length; j++)
                {
                    weights[i][j] = random.NextGaussian(0, stdDev);
                }
                
                prevSize = currSize;
            }
        }
        
        public double[] Forward(double[] input)
        {
            activations[0] = input;
            
            for (int layer = 0; layer < weights.Length; layer++)
            {
                int inputDim = activations[layer].Length;
                int outputDim = biases[layer].Length;
                double[] output = new double[outputDim];
                
                // Linear transformation: y = Wx + b
                for (int i = 0; i < outputDim; i++)
                {
                    double sum = biases[layer][i];
                    for (int j = 0; j < inputDim; j++)
                    {
                        sum += weights[layer][i * inputDim + j] * activations[layer][j];
                    }
                    
                    // ReLU activation for hidden layers, linear for output
                    output[i] = (layer < weights.Length - 1) ? Math.Max(0, sum) : sum;
                }
                
                activations[layer + 1] = output;
            }
            
            return activations[activations.Length - 1];
        }
        
        public void Train(double[] input, double[] target)
        {
            // Forward pass
            double[] output = Forward(input);
            
            // Compute loss gradient (MSE)
            double[] outputGrad = new double[output.Length];
            for (int i = 0; i < output.Length; i++)
            {
                outputGrad[i] = 2 * (output[i] - target[i]) / output.Length;
            }
            
            // Backward pass (simplified - would be much cleaner with SharpGrad)
            double[][] gradients = new double[weights.Length][];
            for (int layer = weights.Length - 1; layer >= 0; layer--)
            {
                int inputDim = (layer > 0) ? biases[layer - 1].Length : inputSize;
                int outputDim = biases[layer].Length;
                
                // Update weights and biases
                for (int i = 0; i < outputDim; i++)
                {
                    // Update bias
                    biases[layer][i] -= learningRate * outputGrad[i];
                    
                    // Update weights
                    for (int j = 0; j < inputDim; j++)
                    {
                        double activation = (layer > 0) ? activations[layer][j] : input[j];
                        weights[layer][i * inputDim + j] -= learningRate * outputGrad[i] * activation;
                    }
                }
                
                // Propagate gradient to previous layer
                if (layer > 0)
                {
                    double[] prevGrad = new double[inputDim];
                    for (int j = 0; j < inputDim; j++)
                    {
                        double sum = 0;
                        for (int i = 0; i < outputDim; i++)
                        {
                            sum += weights[layer][i * inputDim + j] * outputGrad[i];
                        }
                        // ReLU derivative
                        prevGrad[j] = (activations[layer][j] > 0) ? sum : 0;
                    }
                    outputGrad = prevGrad;
                }
            }
        }
        
        public void CopyWeightsFrom(INeuralNetwork other)
        {
            if (other is SimpleNeuralNetwork snn)
            {
                for (int i = 0; i < weights.Length; i++)
                {
                    Array.Copy(snn.weights[i], weights[i], weights[i].Length);
                    Array.Copy(snn.biases[i], biases[i], biases[i].Length);
                }
            }
        }
        
        public void Save(string path)
        {
            // Save network architecture and weights
            var saveData = new
            {
                InputSize = inputSize,
                OutputSize = outputSize,
                HiddenSizes = hiddenSizes,
                LearningRate = learningRate,
                Weights = weights.Select(w => w.ToList()).ToList(),
                Biases = biases.Select(b => b.ToList()).ToList()
            };
            
            string json = System.Text.Json.JsonSerializer.Serialize(saveData, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            System.IO.File.WriteAllText(path, json);
        }
        
        public void Load(string path)
        {
            if (!System.IO.File.Exists(path))
            {
                throw new System.IO.FileNotFoundException($"Model file not found: {path}");
            }
            
            string json = System.IO.File.ReadAllText(path);
            using var doc = System.Text.Json.JsonDocument.Parse(json);
            var root = doc.RootElement;
            
            // Load weights
            var weightsElement = root.GetProperty("Weights");
            for (int i = 0; i < weights.Length && i < weightsElement.GetArrayLength(); i++)
            {
                var layerWeights = weightsElement[i];
                int j = 0;
                foreach (var weight in layerWeights.EnumerateArray())
                {
                    if (j < weights[i].Length)
                    {
                        weights[i][j] = weight.GetDouble();
                        j++;
                    }
                }
            }
            
            // Load biases
            var biasesElement = root.GetProperty("Biases");
            for (int i = 0; i < biases.Length && i < biasesElement.GetArrayLength(); i++)
            {
                var layerBiases = biasesElement[i];
                int j = 0;
                foreach (var bias in layerBiases.EnumerateArray())
                {
                    if (j < biases[i].Length)
                    {
                        biases[i][j] = bias.GetDouble();
                        j++;
                    }
                }
            }
        }
    }
    
    // Extension for Gaussian random numbers
    public static class RandomExtensions
    {
        public static double NextGaussian(this Random random, double mean = 0, double stdDev = 1)
        {
            // Box-Muller transform
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }
    }
}