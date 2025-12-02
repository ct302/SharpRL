using System;
using System.Collections.Generic;
using System.Linq;
using SharpRL.AutoGrad;

namespace SharpRL.NN.Loss
{
    /// <summary>
    /// Mean Squared Error Loss: (1/n) * Σ(predicted - target)²
    /// </summary>
    public class MSELoss
    {
        public Tensor Forward(Tensor predicted, Tensor target)
        {
            if (!predicted.Shape.SequenceEqual(target.Shape))
                throw new ArgumentException("Predicted and target must have same shape");

            // Compute (predicted - target)² for each element
            float[] diff = new float[predicted.Size];
            float sum = 0;
            for (int i = 0; i < predicted.Size; i++)
            {
                float d = predicted.Data[i] - target.Data[i];
                diff[i] = d * d;
                sum += diff[i];
            }

            // Return scalar mean
            float mean = sum / predicted.Size;
            var output = new Tensor(new float[] { mean }, new int[] { 1 }, predicted.RequiresGrad);
            
            if (predicted.RequiresGrad)
            {
                output.GradFn = new MSEFunction(predicted, target);
                output.IsLeaf = false;
            }

            return output;
        }
    }

    internal class MSEFunction : Function
    {
        public MSEFunction(Tensor predicted, Tensor target)
        {
            Inputs = new List<Tensor> { predicted, target };
        }

        public override void Backward(Tensor grad)
        {
            var predicted = Inputs[0];
            var target = Inputs[1];

            if (predicted.RequiresGrad)
            {
                if (predicted.Grad == null)
                    predicted.Grad = Tensor.Zeros(predicted.Shape);

                // ∂MSE/∂predicted = 2(predicted - target) / n
                // grad.Data[0] is the upstream gradient (scalar)
                float scale = 2f * grad.Data[0] / predicted.Size;
                for (int i = 0; i < predicted.Size; i++)
                {
                    predicted.Grad.Data[i] += scale * (predicted.Data[i] - target.Data[i]);
                }
            }
        }
    }

    /// <summary>
    /// Cross Entropy Loss for classification
    /// </summary>
    public class CrossEntropyLoss
    {
        public Tensor Forward(Tensor logits, Tensor target)
        {
            if (logits.Shape.Length != 2)
                throw new ArgumentException("Logits must be 2D (batch_size, num_classes)");

            int batchSize = logits.Shape[0];
            int numClasses = logits.Shape[1];

            float totalLoss = 0;

            for (int b = 0; b < batchSize; b++)
            {
                // Get logits for this sample
                float maxLogit = float.NegativeInfinity;
                for (int c = 0; c < numClasses; c++)
                {
                    maxLogit = Math.Max(maxLogit, logits.Data[b * numClasses + c]);
                }

                // Compute softmax denominator
                float sumExp = 0;
                for (int c = 0; c < numClasses; c++)
                {
                    sumExp += (float)Math.Exp(logits.Data[b * numClasses + c] - maxLogit);
                }

                // Get target class
                int targetClass = (int)target.Data[b];

                // Compute cross entropy: -log(p[target])
                float logProb = logits.Data[b * numClasses + targetClass] - maxLogit - (float)Math.Log(sumExp);
                totalLoss -= logProb;
            }

            float meanLoss = totalLoss / batchSize;
            var output = new Tensor(new float[] { meanLoss }, new int[] { 1 }, logits.RequiresGrad);
            
            if (logits.RequiresGrad)
            {
                output.GradFn = new CrossEntropyFunction(logits, target);
                output.IsLeaf = false;
            }

            return output;
        }
    }

    internal class CrossEntropyFunction : Function
    {
        public CrossEntropyFunction(Tensor logits, Tensor target)
        {
            Inputs = new List<Tensor> { logits, target };
        }

        public override void Backward(Tensor grad)
        {
            var logits = Inputs[0];
            var target = Inputs[1];

            if (!logits.RequiresGrad)
                return;

            if (logits.Grad == null)
                logits.Grad = Tensor.Zeros(logits.Shape);

            int batchSize = logits.Shape[0];
            int numClasses = logits.Shape[1];
            float scale = grad.Data[0] / batchSize;

            for (int b = 0; b < batchSize; b++)
            {
                // Compute softmax for this sample
                float maxLogit = float.NegativeInfinity;
                for (int c = 0; c < numClasses; c++)
                {
                    maxLogit = Math.Max(maxLogit, logits.Data[b * numClasses + c]);
                }

                float sumExp = 0;
                float[] probs = new float[numClasses];
                for (int c = 0; c < numClasses; c++)
                {
                    probs[c] = (float)Math.Exp(logits.Data[b * numClasses + c] - maxLogit);
                    sumExp += probs[c];
                }
                for (int c = 0; c < numClasses; c++)
                {
                    probs[c] /= sumExp;
                }

                int targetClass = (int)target.Data[b];

                // Gradient: softmax - one_hot(target)
                for (int c = 0; c < numClasses; c++)
                {
                    float gradValue = probs[c];
                    if (c == targetClass)
                        gradValue -= 1.0f;

                    logits.Grad.Data[b * numClasses + c] += scale * gradValue;
                }
            }
        }
    }

    /// <summary>
    /// Smooth L1 Loss (Huber Loss) - robust to outliers
    /// </summary>
    public class SmoothL1Loss
    {
        private float beta;

        public SmoothL1Loss(float beta = 1.0f)
        {
            this.beta = beta;
        }

        public Tensor Forward(Tensor predicted, Tensor target)
        {
            if (!predicted.Shape.SequenceEqual(target.Shape))
                throw new ArgumentException("Predicted and target must have same shape");

            float sum = 0;
            for (int i = 0; i < predicted.Size; i++)
            {
                float diff = Math.Abs(predicted.Data[i] - target.Data[i]);
                if (diff < beta)
                {
                    sum += 0.5f * diff * diff / beta;
                }
                else
                {
                    sum += diff - 0.5f * beta;
                }
            }

            float mean = sum / predicted.Size;
            var output = new Tensor(new float[] { mean }, new int[] { 1 }, predicted.RequiresGrad);
            
            if (predicted.RequiresGrad)
            {
                output.GradFn = new SmoothL1Function(predicted, target, beta);
                output.IsLeaf = false;
            }

            return output;
        }
    }

    internal class SmoothL1Function : Function
    {
        private float beta;

        public SmoothL1Function(Tensor predicted, Tensor target, float beta)
        {
            Inputs = new List<Tensor> { predicted, target };
            this.beta = beta;
        }

        public override void Backward(Tensor grad)
        {
            var predicted = Inputs[0];
            var target = Inputs[1];

            if (!predicted.RequiresGrad)
                return;

            if (predicted.Grad == null)
                predicted.Grad = Tensor.Zeros(predicted.Shape);

            // grad.Data[0] is the upstream scalar gradient
            float scale = grad.Data[0] / predicted.Size;

            for (int i = 0; i < predicted.Size; i++)
            {
                float diff = predicted.Data[i] - target.Data[i];
                float absDiff = Math.Abs(diff);

                float gradValue;
                if (absDiff < beta)
                {
                    gradValue = diff / beta;
                }
                else
                {
                    gradValue = diff > 0 ? 1f : -1f;
                }

                predicted.Grad.Data[i] += scale * gradValue;
            }
        }
    }
}
