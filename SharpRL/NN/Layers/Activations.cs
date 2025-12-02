using System;
using System.Collections.Generic;
using System.Linq;
using SharpRL.AutoGrad;

namespace SharpRL.NN.Layers
{
    /// <summary>
    /// Sequential container for stacking layers
    /// </summary>
    public class Sequential : Module
    {
        private List<Module> layers;

        public Sequential(params Module[] layers)
        {
            this.layers = layers.ToList();
        }

        public void Add(Module layer)
        {
            layers.Add(layer);
        }

        public override Tensor Forward(Tensor input)
        {
            var output = input;
            foreach (var layer in layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        public override List<Tensor> Parameters()
        {
            var parameters = new List<Tensor>();
            foreach (var layer in layers)
            {
                parameters.AddRange(layer.Parameters());
            }
            return parameters;
        }

        public override Module Train()
        {
            base.Train();
            foreach (var layer in layers)
            {
                layer.Train();
            }
            return this;
        }

        public override Module Eval()
        {
            base.Eval();
            foreach (var layer in layers)
            {
                layer.Eval();
            }
            return this;
        }
    }

    /// <summary>
    /// ReLU activation layer
    /// </summary>
    public class ReLU : Module
    {
        public override Tensor Forward(Tensor input)
        {
            return input.ReLU();
        }

        public override List<Tensor> Parameters()
        {
            return new List<Tensor>();
        }
    }

    /// <summary>
    /// Sigmoid activation layer
    /// </summary>
    public class Sigmoid : Module
    {
        public override Tensor Forward(Tensor input)
        {
            return input.Sigmoid();
        }

        public override List<Tensor> Parameters()
        {
            return new List<Tensor>();
        }
    }

    /// <summary>
    /// Tanh activation layer
    /// </summary>
    public class Tanh : Module
    {
        public override Tensor Forward(Tensor input)
        {
            return input.Tanh();
        }

        public override List<Tensor> Parameters()
        {
            return new List<Tensor>();
        }
    }

    /// <summary>
    /// LogSoftmax activation layer with proper gradient support
    /// log(softmax(x)) = x - log(sum(exp(x)))
    /// </summary>
    public class LogSoftmax : Module
    {
        public override Tensor Forward(Tensor input)
        {
            float[] output = new float[input.Size];
            
            if (input.Shape.Length == 2)
            {
                // Batch processing
                int batchSize = input.Shape[0];
                int features = input.Shape[1];
                
                for (int b = 0; b < batchSize; b++)
                {
                    // Find max for numerical stability
                    float maxVal = float.NegativeInfinity;
                    for (int i = 0; i < features; i++)
                    {
                        maxVal = Math.Max(maxVal, input.Data[b * features + i]);
                    }
                    
                    // Compute log sum exp
                    float sumExp = 0;
                    for (int i = 0; i < features; i++)
                    {
                        sumExp += (float)Math.Exp(input.Data[b * features + i] - maxVal);
                    }
                    float logSumExp = (float)Math.Log(sumExp) + maxVal;
                    
                    // Compute log softmax
                    for (int i = 0; i < features; i++)
                    {
                        output[b * features + i] = input.Data[b * features + i] - logSumExp;
                    }
                }
            }
            else
            {
                // Single sample
                float maxVal = input.Data.Max();
                float sumExp = 0;
                for (int i = 0; i < input.Size; i++)
                {
                    sumExp += (float)Math.Exp(input.Data[i] - maxVal);
                }
                float logSumExp = (float)Math.Log(sumExp) + maxVal;
                
                for (int i = 0; i < input.Size; i++)
                {
                    output[i] = input.Data[i] - logSumExp;
                }
            }
            
            var result = new Tensor(output, input.Shape, input.RequiresGrad);
            
            if (input.RequiresGrad)
            {
                result.GradFn = new LogSoftmaxFunction(input, output);
                result.IsLeaf = false;
            }
            
            return result;
        }
        
        public override List<Tensor> Parameters()
        {
            return new List<Tensor>();
        }
    }

    /// <summary>
    /// LogSoftmax backward function
    /// </summary>
    internal class LogSoftmaxFunction : Function
    {
        private float[] softmaxOutput;

        public LogSoftmaxFunction(Tensor input, float[] logSoftmaxOutput)
        {
            Inputs = new List<Tensor> { input };
            // Store softmax values (exp of log softmax)
            softmaxOutput = new float[logSoftmaxOutput.Length];
            for (int i = 0; i < logSoftmaxOutput.Length; i++)
            {
                softmaxOutput[i] = (float)Math.Exp(logSoftmaxOutput[i]);
            }
        }

        public override void Backward(Tensor grad)
        {
            var input = Inputs[0];
            if (!input.RequiresGrad)
                return;

            if (input.Grad == null)
                input.Grad = Tensor.Zeros(input.Shape);

            if (input.Shape.Length == 2)
            {
                int batchSize = input.Shape[0];
                int features = input.Shape[1];

                for (int b = 0; b < batchSize; b++)
                {
                    // Sum of upstream gradients for this sample
                    float gradSum = 0;
                    for (int i = 0; i < features; i++)
                    {
                        gradSum += grad.Data[b * features + i];
                    }

                    // ∂L/∂x_i = ∂L/∂y_i - softmax_i * sum(∂L/∂y_j)
                    for (int i = 0; i < features; i++)
                    {
                        int idx = b * features + i;
                        input.Grad.Data[idx] += grad.Data[idx] - softmaxOutput[idx] * gradSum;
                    }
                }
            }
            else
            {
                float gradSum = 0;
                for (int i = 0; i < input.Size; i++)
                {
                    gradSum += grad.Data[i];
                }

                for (int i = 0; i < input.Size; i++)
                {
                    input.Grad.Data[i] += grad.Data[i] - softmaxOutput[i] * gradSum;
                }
            }
        }
    }

    /// <summary>
    /// Softmax activation layer
    /// </summary>
    public class Softmax : Module
    {
        public override Tensor Forward(Tensor input)
        {
            float[] output = new float[input.Size];
            
            if (input.Shape.Length == 2)
            {
                int batchSize = input.Shape[0];
                int features = input.Shape[1];
                
                for (int b = 0; b < batchSize; b++)
                {
                    float maxVal = float.NegativeInfinity;
                    for (int i = 0; i < features; i++)
                    {
                        maxVal = Math.Max(maxVal, input.Data[b * features + i]);
                    }
                    
                    float sumExp = 0;
                    for (int i = 0; i < features; i++)
                    {
                        output[b * features + i] = (float)Math.Exp(input.Data[b * features + i] - maxVal);
                        sumExp += output[b * features + i];
                    }
                    
                    for (int i = 0; i < features; i++)
                    {
                        output[b * features + i] /= sumExp;
                    }
                }
            }
            else
            {
                float maxVal = input.Data.Max();
                float sumExp = 0;
                for (int i = 0; i < input.Size; i++)
                {
                    output[i] = (float)Math.Exp(input.Data[i] - maxVal);
                    sumExp += output[i];
                }
                for (int i = 0; i < input.Size; i++)
                {
                    output[i] /= sumExp;
                }
            }
            
            var result = new Tensor(output, input.Shape, input.RequiresGrad);
            
            if (input.RequiresGrad)
            {
                result.GradFn = new SoftmaxFunction(input, output);
                result.IsLeaf = false;
            }
            
            return result;
        }
        
        public override List<Tensor> Parameters()
        {
            return new List<Tensor>();
        }
    }

    /// <summary>
    /// Softmax backward function
    /// </summary>
    internal class SoftmaxFunction : Function
    {
        private float[] softmaxOutput;

        public SoftmaxFunction(Tensor input, float[] output)
        {
            Inputs = new List<Tensor> { input };
            softmaxOutput = output.ToArray();
        }

        public override void Backward(Tensor grad)
        {
            var input = Inputs[0];
            if (!input.RequiresGrad)
                return;

            if (input.Grad == null)
                input.Grad = Tensor.Zeros(input.Shape);

            if (input.Shape.Length == 2)
            {
                int batchSize = input.Shape[0];
                int features = input.Shape[1];

                for (int b = 0; b < batchSize; b++)
                {
                    // Jacobian-vector product for softmax
                    float dot = 0;
                    for (int i = 0; i < features; i++)
                    {
                        dot += grad.Data[b * features + i] * softmaxOutput[b * features + i];
                    }

                    for (int i = 0; i < features; i++)
                    {
                        int idx = b * features + i;
                        input.Grad.Data[idx] += softmaxOutput[idx] * (grad.Data[idx] - dot);
                    }
                }
            }
            else
            {
                float dot = 0;
                for (int i = 0; i < input.Size; i++)
                {
                    dot += grad.Data[i] * softmaxOutput[i];
                }

                for (int i = 0; i < input.Size; i++)
                {
                    input.Grad.Data[i] += softmaxOutput[i] * (grad.Data[i] - dot);
                }
            }
        }
    }

    /// <summary>
    /// Dropout layer for regularization
    /// </summary>
    public class Dropout : Module
    {
        private float p;
        private Random random;

        public Dropout(float p = 0.5f)
        {
            this.p = p;
            this.random = new Random();
        }

        public override Tensor Forward(Tensor input)
        {
            if (!IsTraining || p == 0)
            {
                return input;
            }

            float[] mask = new float[input.Size];
            float scale = 1f / (1f - p);

            for (int i = 0; i < input.Size; i++)
            {
                mask[i] = random.NextDouble() > p ? scale : 0f;
            }

            var maskTensor = new Tensor(mask, input.Shape, false);
            return input * maskTensor;
        }

        public override List<Tensor> Parameters()
        {
            return new List<Tensor>();
        }
    }
}
