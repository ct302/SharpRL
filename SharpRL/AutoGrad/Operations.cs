using System;
using System.Linq;

namespace SharpRL.AutoGrad
{
    /// <summary>
    /// Addition operation with automatic differentiation
    /// </summary>
    public class AddFunction : Function
    {
        public Tensor Forward(Tensor a, Tensor b)
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Tensors must have same shape for addition");

            float[] result = new float[a.Size];
            for (int i = 0; i < a.Size; i++)
            {
                result[i] = a.Data[i] + b.Data[i];
            }

            var output = new Tensor(result, a.Shape, a.RequiresGrad || b.RequiresGrad);
            
            if (output.RequiresGrad)
            {
                Inputs = new List<Tensor> { a, b };
                output.GradFn = this;
                output.IsLeaf = false; // Created by operation, not a leaf
            }
            
            return output;
        }

        public override void Backward(Tensor grad)
        {
            // ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
            // Only accumulate gradients for leaf tensors
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[0].Grad!.Data[i] += grad.Data[i];
                }
            }
            
            if (Inputs[1].RequiresGrad)
            {
                if (Inputs[1].Grad == null)
                    Inputs[1].Grad = Tensor.Zeros(Inputs[1].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[1].Grad!.Data[i] += grad.Data[i];
                }
            }
        }
    }

    /// <summary>
    /// Subtraction operation with automatic differentiation
    /// </summary>
    public class SubtractFunction : Function
    {
        public Tensor Forward(Tensor a, Tensor b)
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Tensors must have same shape for subtraction");

            float[] result = new float[a.Size];
            for (int i = 0; i < a.Size; i++)
            {
                result[i] = a.Data[i] - b.Data[i];
            }

            var output = new Tensor(result, a.Shape, a.RequiresGrad || b.RequiresGrad);
            
            if (output.RequiresGrad)
            {
                Inputs = new List<Tensor> { a, b };
                output.GradFn = this;
                output.IsLeaf = false; // Created by operation, not a leaf
            }
            
            return output;
        }

        public override void Backward(Tensor grad)
        {
            // ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[0].Grad!.Data[i] += grad.Data[i];
                }
            }
            
            if (Inputs[1].RequiresGrad)
            {
                if (Inputs[1].Grad == null)
                    Inputs[1].Grad = Tensor.Zeros(Inputs[1].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[1].Grad!.Data[i] -= grad.Data[i];
                }
            }
        }
    }

    /// <summary>
    /// Element-wise multiplication with automatic differentiation
    /// </summary>
    public class MultiplyFunction : Function
    {
        public Tensor Forward(Tensor a, Tensor b)
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Tensors must have same shape for multiplication");

            float[] result = new float[a.Size];
            for (int i = 0; i < a.Size; i++)
            {
                result[i] = a.Data[i] * b.Data[i];
            }

            var output = new Tensor(result, a.Shape, a.RequiresGrad || b.RequiresGrad);
            
            if (output.RequiresGrad)
            {
                Inputs = new List<Tensor> { a, b };
                output.GradFn = this;
                output.IsLeaf = false; // Created by operation, not a leaf
            }
            
            return output;
        }

        public override void Backward(Tensor grad)
        {
            // ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[0].Grad!.Data[i] += grad.Data[i] * Inputs[1].Data[i];
                }
            }
            
            if (Inputs[1].RequiresGrad)
            {
                if (Inputs[1].Grad == null)
                    Inputs[1].Grad = Tensor.Zeros(Inputs[1].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[1].Grad!.Data[i] += grad.Data[i] * Inputs[0].Data[i];
                }
            }
        }
    }

    /// <summary>
    /// Element-wise division with automatic differentiation
    /// </summary>
    public class DivideFunction : Function
    {
        public Tensor Forward(Tensor a, Tensor b)
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Tensors must have same shape for division");

            float[] result = new float[a.Size];
            for (int i = 0; i < a.Size; i++)
            {
                result[i] = a.Data[i] / b.Data[i];
            }

            var output = new Tensor(result, a.Shape, a.RequiresGrad || b.RequiresGrad);
            
            if (output.RequiresGrad)
            {
                Inputs = new List<Tensor> { a, b };
                output.GradFn = this;
                output.IsLeaf = false; // Created by operation, not a leaf
            }
            
            return output;
        }

        public override void Backward(Tensor grad)
        {
            // ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[0].Grad!.Data[i] += grad.Data[i] / Inputs[1].Data[i];
                }
            }
            
            if (Inputs[1].RequiresGrad)
            {
                if (Inputs[1].Grad == null)
                    Inputs[1].Grad = Tensor.Zeros(Inputs[1].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[1].Grad!.Data[i] -= grad.Data[i] * Inputs[0].Data[i] / (Inputs[1].Data[i] * Inputs[1].Data[i]);
                }
            }
        }
    }

    /// <summary>
    /// Scalar multiplication with automatic differentiation
    /// </summary>
    public class ScalarMultiplyFunction : Function
    {
        private float scalar;

        public ScalarMultiplyFunction(Tensor input, float scalar)
        {
            Inputs = new List<Tensor> { input };
            this.scalar = scalar;
        }

        public override void Backward(Tensor grad)
        {
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[0].Grad!.Data[i] += grad.Data[i] * scalar;
                }
            }
        }
    }

    /// <summary>
    /// Matrix multiplication with automatic differentiation
    /// Like combining player stats with team strategies
    /// </summary>
    public class MatMulFunction : Function
    {
        private int[] aShape = null!;
        private int[] bShape = null!;

        public Tensor Forward(Tensor a, Tensor b)
        {
            if (a.Shape.Length != 2 || b.Shape.Length != 2)
                throw new ArgumentException("MatMul requires 2D tensors");
            
            if (a.Shape[1] != b.Shape[0])
                throw new ArgumentException($"Shape mismatch: [{a.Shape[0]},{a.Shape[1]}] @ [{b.Shape[0]},{b.Shape[1]}]");

            int m = a.Shape[0];
            int n = a.Shape[1];
            int p = b.Shape[1];
            
            float[] result = new float[m * p];
            
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += a.Data[i * n + k] * b.Data[k * p + j];
                    }
                    result[i * p + j] = sum;
                }
            }
            
            var output = new Tensor(result, new int[] { m, p }, a.RequiresGrad || b.RequiresGrad);
            
            if (output.RequiresGrad)
            {
                Inputs = new List<Tensor> { a, b };
                aShape = a.Shape;
                bShape = b.Shape;
                output.GradFn = this;
                output.IsLeaf = false; // Created by operation, not a leaf
            }
            
            return output;
        }

        public override void Backward(Tensor grad)
        {
            // ∂(A@B)/∂A = grad @ B^T
            // ∂(A@B)/∂B = A^T @ grad
            
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(aShape);
                
                // grad @ B^T
                for (int i = 0; i < aShape[0]; i++)
                {
                    for (int j = 0; j < aShape[1]; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < bShape[1]; k++)
                        {
                            sum += grad.Data[i * bShape[1] + k] * Inputs[1].Data[j * bShape[1] + k];
                        }
                        Inputs[0].Grad!.Data[i * aShape[1] + j] += sum;
                    }
                }
            }
            
            if (Inputs[1].RequiresGrad)
            {
                if (Inputs[1].Grad == null)
                    Inputs[1].Grad = Tensor.Zeros(bShape);
                
                // A^T @ grad
                for (int i = 0; i < bShape[0]; i++)
                {
                    for (int j = 0; j < bShape[1]; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < aShape[0]; k++)
                        {
                            sum += Inputs[0].Data[k * aShape[1] + i] * grad.Data[k * bShape[1] + j];
                        }
                        Inputs[1].Grad!.Data[i * bShape[1] + j] += sum;
                    }
                }
            }
        }
    }

    /// <summary>
    /// ReLU activation function: max(0, x)
    /// Like keeping only positive yardage plays
    /// </summary>
    public class ReLUFunction : Function
    {
        private bool[] mask = null!;

        public Tensor Forward(Tensor input)
        {
            float[] result = new float[input.Size];
            mask = new bool[input.Size];
            
            for (int i = 0; i < input.Size; i++)
            {
                result[i] = Math.Max(0, input.Data[i]);
                mask[i] = input.Data[i] > 0;
            }
            
            var output = new Tensor(result, input.Shape, input.RequiresGrad);
            
            if (input.RequiresGrad)
            {
                Inputs = new List<Tensor> { input };
                output.GradFn = this;
                output.IsLeaf = false; // Created by operation, not a leaf
            }
            
            return output;
        }

        public override void Backward(Tensor grad)
        {
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    if (mask[i])
                    {
                        Inputs[0].Grad!.Data[i] += grad.Data[i];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Sigmoid activation: 1 / (1 + e^-x)
    /// Like calculating win probability
    /// </summary>
    public class SigmoidFunction : Function
    {
        private float[] output = null!;

        public Tensor Forward(Tensor input)
        {
            output = new float[input.Size];
            
            for (int i = 0; i < input.Size; i++)
            {
                output[i] = 1f / (1f + (float)Math.Exp(-input.Data[i]));
            }
            
            var result = new Tensor(output, input.Shape, input.RequiresGrad);
            
            if (input.RequiresGrad)
            {
                Inputs = new List<Tensor> { input };
                result.GradFn = this;
                result.IsLeaf = false; // Created by operation, not a leaf
            }
            
            return result;
        }

        public override void Backward(Tensor grad)
        {
            // ∂sigmoid/∂x = sigmoid * (1 - sigmoid)
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[0].Grad!.Data[i] += grad.Data[i] * output[i] * (1 - output[i]);
                }
            }
        }
    }

    /// <summary>
    /// Tanh activation function
    /// </summary>
    public class TanhFunction : Function
    {
        private float[] output = null!;

        public Tensor Forward(Tensor input)
        {
            output = new float[input.Size];
            
            for (int i = 0; i < input.Size; i++)
            {
                output[i] = (float)Math.Tanh(input.Data[i]);
            }
            
            var result = new Tensor(output, input.Shape, input.RequiresGrad);
            
            if (input.RequiresGrad)
            {
                Inputs = new List<Tensor> { input };
                result.GradFn = this;
                result.IsLeaf = false; // Created by operation, not a leaf
            }
            
            return result;
        }

        public override void Backward(Tensor grad)
        {
            // ∂tanh/∂x = 1 - tanh²
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[0].Grad!.Data[i] += grad.Data[i] * (1 - output[i] * output[i]);
                }
            }
        }
    }

    /// <summary>
    /// Mean operation
    /// </summary>
    public class MeanFunction : Function
    {
        public MeanFunction(Tensor input)
        {
            Inputs = new List<Tensor> { input };
        }

        public override void Backward(Tensor grad)
        {
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                float gradValue = grad.Data[0] / Inputs[0].Size;
                for (int i = 0; i < Inputs[0].Size; i++)
                {
                    Inputs[0].Grad!.Data[i] += gradValue;
                }
            }
        }
    }

    /// <summary>
    /// Sum operation
    /// </summary>
    public class SumFunction : Function
    {
        public SumFunction(Tensor input)
        {
            Inputs = new List<Tensor> { input };
        }

        public override void Backward(Tensor grad)
        {
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                for (int i = 0; i < Inputs[0].Size; i++)
                {
                    Inputs[0].Grad!.Data[i] += grad.Data[0];
                }
            }
        }
    }

    /// <summary>
    /// Transpose operation
    /// </summary>
    public class TransposeFunction : Function
    {
        public TransposeFunction(Tensor input)
        {
            Inputs = new List<Tensor> { input };
        }

        public override void Backward(Tensor grad)
        {
            if (Inputs[0].RequiresGrad)
            {
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(Inputs[0].Shape);
                
                // Transpose the gradient back
                int rows = grad.Shape[0];
                int cols = grad.Shape[1];
                
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        Inputs[0].Grad!.Data[j * rows + i] += grad.Data[i * cols + j];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Reshape operation
    /// </summary>
    public class ReshapeFunction : Function
    {
        private int[] originalShape;

        public ReshapeFunction(Tensor input, int[] originalShape)
        {
            Inputs = new List<Tensor> { input };
            this.originalShape = originalShape;
        }

        public override void Backward(Tensor grad)
        {
            if (Inputs[0].RequiresGrad)
            {
                // Just reshape the gradient back to original shape
                if (Inputs[0].Grad == null)
                    Inputs[0].Grad = Tensor.Zeros(originalShape);
                
                for (int i = 0; i < grad.Size; i++)
                {
                    Inputs[0].Grad!.Data[i] += grad.Data[i];
                }
            }
        }
    }

    /// <summary>
    /// Broadcast bias from (outFeatures,) to (batchSize, outFeatures)
    /// Gradient is summed across batch dimension back to original bias
    /// </summary>
    public class BroadcastBiasFunction : Function
    {
        private int batchSize;
        private int outFeatures;

        public BroadcastBiasFunction(Tensor bias, int batchSize, int outFeatures)
        {
            Inputs = new List<Tensor> { bias };
            this.batchSize = batchSize;
            this.outFeatures = outFeatures;
        }

        public override void Backward(Tensor grad)
        {
            var bias = Inputs[0];
            if (!bias.RequiresGrad)
                return;

            if (bias.Grad == null)
                bias.Grad = Tensor.Zeros(bias.Shape);

            // Sum gradients across batch dimension
            // grad shape: (batchSize, outFeatures)
            // bias shape: (outFeatures,)
            for (int j = 0; j < outFeatures; j++)
            {
                float sum = 0;
                for (int i = 0; i < batchSize; i++)
                {
                    sum += grad.Data[i * outFeatures + j];
                }
                bias.Grad.Data[j] += sum;
            }
        }
    }

    /// <summary>
    /// Gather operation - extracts values at specific indices along dimension 1
    /// Used in DQN to extract Q-values for taken actions
    /// Input: [batchSize, numActions], Indices: [batchSize], Output: [batchSize, 1]
    /// </summary>
    public class GatherFunction : Function
    {
        private int[] indices = null!;
        private int numActions;

        public static Tensor Apply(Tensor input, int[] indices)
        {
            if (input.Shape.Length != 2)
                throw new ArgumentException("Gather requires 2D input tensor");
            if (indices.Length != input.Shape[0])
                throw new ArgumentException("Indices length must match batch size");

            int batchSize = input.Shape[0];
            int numActions = input.Shape[1];

            float[] result = new float[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                result[i] = input.Data[i * numActions + indices[i]];
            }

            var output = new Tensor(result, new int[] { batchSize, 1 }, input.RequiresGrad);

            if (input.RequiresGrad)
            {
                var fn = new GatherFunction();
                fn.Inputs = new List<Tensor> { input };
                fn.indices = indices;
                fn.numActions = numActions;
                output.GradFn = fn;
                output.IsLeaf = false;
            }

            return output;
        }

        public override void Backward(Tensor grad)
        {
            var input = Inputs[0];
            if (!input.RequiresGrad)
                return;

            if (input.Grad == null)
                input.Grad = Tensor.Zeros(input.Shape);

            int batchSize = input.Shape[0];

            // Scatter gradient back to original positions
            for (int i = 0; i < batchSize; i++)
            {
                input.Grad.Data[i * numActions + indices[i]] += grad.Data[i];
            }
        }
    }
}
