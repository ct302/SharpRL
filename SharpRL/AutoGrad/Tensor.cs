using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpRL.AutoGrad
{
    /// <summary>
    /// Multi-dimensional tensor with automatic differentiation support.
    /// The core data structure for neural networks in SharpRL.
    /// 
    /// NFL ANALOGY:
    /// Think of a Tensor like game statistics that flow through your playbook:
    /// - Data = The raw stats (yards, completions, etc.)
    /// - Grad = How much each stat affects the final score
    /// - RequiresGrad = Whether we're tracking this stat for analysis
    /// - GradFn = The play that generated these stats
    /// </summary>
    public class Tensor
    {
        #region Fields and Properties

        /// <summary>
        /// The underlying data (like raw game stats)
        /// </summary>
        public float[] Data { get; private set; }

        /// <summary>
        /// Shape of the tensor (like [games, quarters, stats])
        /// </summary>
        public int[] Shape { get; private set; }

        /// <summary>
        /// Strides for efficient indexing
        /// </summary>
        public int[] Strides { get; private set; }

        /// <summary>
        /// Total number of elements
        /// </summary>
        public int Size { get; private set; }

        /// <summary>
        /// Gradient tensor (how much this affects the loss)
        /// </summary>
        public Tensor? Grad { get; set; }

        /// <summary>
        /// Whether to track gradients for this tensor
        /// </summary>
        public bool RequiresGrad { get; set; }

        /// <summary>
        /// The function that created this tensor (for backprop)
        /// </summary>
        public Function? GradFn { get; set; }

        /// <summary>
        /// Whether this is a leaf tensor (created by user, not by operation)
        /// Only leaf tensors accumulate gradients
        /// </summary>
        public bool IsLeaf { get; set; }

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a tensor from data and shape
        /// </summary>
        public Tensor(float[] data, int[] shape, bool requiresGrad = false)
        {
            // Input validation
            if (data == null)
                throw new ArgumentNullException(nameof(data), "Data array cannot be null");
            if (shape == null)
                throw new ArgumentNullException(nameof(shape), "Shape array cannot be null");
            if (shape.Length == 0)
                throw new ArgumentException("Shape must have at least one dimension", nameof(shape));
            if (shape.Any(d => d <= 0))
                throw new ArgumentException("All shape dimensions must be positive", nameof(shape));
            if (data.Length == 0)
                throw new ArgumentException("Data array cannot be empty", nameof(data));
            
            int expectedSize = shape.Aggregate(1, (a, b) => a * b);
            if (data.Length != expectedSize)
                throw new ArgumentException($"Data length ({data.Length}) must match shape dimensions ({expectedSize})", nameof(data));

            // Validate no NaN or infinity in data
            if (data.Any(f => float.IsNaN(f) || float.IsInfinity(f)))
                throw new ArgumentException("Data contains NaN or Infinity values", nameof(data));

            Data = data;
            Shape = shape;
            RequiresGrad = requiresGrad;
            Size = data.Length;
            Strides = ComputeStrides(shape);
            IsLeaf = true; // User-created tensors are leaves
            // Note: Grad is initialized to null - Backward methods handle allocation
        }

        /// <summary>
        /// Creates a tensor filled with zeros (like an empty stat sheet)
        /// </summary>
        public static Tensor Zeros(int[] shape, bool requiresGrad = false)
        {
            if (shape == null)
                throw new ArgumentNullException(nameof(shape), "Shape array cannot be null");
            if (shape.Length == 0)
                throw new ArgumentException("Shape must have at least one dimension", nameof(shape));
            if (shape.Any(d => d <= 0))
                throw new ArgumentException("All shape dimensions must be positive", nameof(shape));
            
            int size = shape.Aggregate(1, (a, b) => a * b);
            return new Tensor(new float[size], shape, requiresGrad);
        }

        /// <summary>
        /// Creates a tensor filled with ones
        /// </summary>
        public static Tensor Ones(int[] shape, bool requiresGrad = false)
        {
            if (shape == null)
                throw new ArgumentNullException(nameof(shape), "Shape array cannot be null");
            if (shape.Length == 0)
                throw new ArgumentException("Shape must have at least one dimension", nameof(shape));
            if (shape.Any(d => d <= 0))
                throw new ArgumentException("All shape dimensions must be positive", nameof(shape));
            
            int size = shape.Aggregate(1, (a, b) => a * b);
            float[] data = Enumerable.Repeat(1f, size).ToArray();
            return new Tensor(data, shape, requiresGrad);
        }

        /// <summary>
        /// Creates a tensor with random normal values (He initialization)
        /// Like randomizing starting positions in Madden
        /// </summary>
        public static Tensor Randn(int[] shape, bool requiresGrad = false, float scale = 1.0f)
        {
            if (shape == null)
                throw new ArgumentNullException(nameof(shape), "Shape array cannot be null");
            if (shape.Length == 0)
                throw new ArgumentException("Shape must have at least one dimension", nameof(shape));
            if (shape.Any(d => d <= 0))
                throw new ArgumentException("All shape dimensions must be positive", nameof(shape));
            if (scale <= 0)
                throw new ArgumentException("Scale must be positive", nameof(scale));
            if (float.IsNaN(scale) || float.IsInfinity(scale))
                throw new ArgumentException("Scale cannot be NaN or Infinity", nameof(scale));
            
            int size = shape.Aggregate(1, (a, b) => a * b);
            Random random = new Random();
            float[] data = new float[size];
            
            // Box-Muller transform for normal distribution
            for (int i = 0; i < size; i += 2)
            {
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                data[i] = (float)(randStdNormal * scale);
                
                if (i + 1 < size)
                {
                    double randStdNormal2 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    data[i + 1] = (float)(randStdNormal2 * scale);
                }
            }
            
            return new Tensor(data, shape, requiresGrad);
        }

        /// <summary>
        /// Creates a tensor with uniform random values
        /// </summary>
        public static Tensor Uniform(int[] shape, float min = 0f, float max = 1f, bool requiresGrad = false)
        {
            if (shape == null)
                throw new ArgumentNullException(nameof(shape), "Shape array cannot be null");
            if (shape.Length == 0)
                throw new ArgumentException("Shape must have at least one dimension", nameof(shape));
            if (shape.Any(d => d <= 0))
                throw new ArgumentException("All shape dimensions must be positive", nameof(shape));
            if (float.IsNaN(min) || float.IsInfinity(min))
                throw new ArgumentException("Min cannot be NaN or Infinity", nameof(min));
            if (float.IsNaN(max) || float.IsInfinity(max))
                throw new ArgumentException("Max cannot be NaN or Infinity", nameof(max));
            if (min >= max)
                throw new ArgumentException("Min must be less than max", nameof(min));
            
            int size = shape.Aggregate(1, (a, b) => a * b);
            Random random = new Random();
            float[] data = new float[size];
            float range = max - min;
            
            for (int i = 0; i < size; i++)
            {
                data[i] = (float)(random.NextDouble() * range + min);
            }
            
            return new Tensor(data, shape, requiresGrad);
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Computes strides for multi-dimensional indexing
        /// </summary>
        private int[] ComputeStrides(int[] shape)
        {
            int[] strides = new int[shape.Length];
            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }

        /// <summary>
        /// Reshapes the tensor (like reorganizing stats into different views)
        /// </summary>
        public Tensor Reshape(params int[] newShape)
        {
            // Allow -1 for auto-calculation of one dimension
            int autoIndex = -1;
            int knownSize = 1;
            
            for (int i = 0; i < newShape.Length; i++)
            {
                if (newShape[i] == -1)
                {
                    if (autoIndex != -1)
                        throw new ArgumentException("Only one dimension can be -1");
                    autoIndex = i;
                }
                else
                {
                    knownSize *= newShape[i];
                }
            }
            
            if (autoIndex != -1)
            {
                newShape[autoIndex] = Size / knownSize;
            }
            
            int newSize = newShape.Aggregate(1, (a, b) => a * b);
            if (newSize != Size)
                throw new ArgumentException($"Cannot reshape tensor of size {Size} to {newSize}");

            var result = new Tensor(Data.ToArray(), newShape, RequiresGrad);
            if (RequiresGrad)
            {
                result.GradFn = new ReshapeFunction(this, Shape);
            }
            return result;
        }

        /// <summary>
        /// Transposes a 2D tensor
        /// </summary>
        public Tensor T()
        {
            if (Shape.Length != 2)
                throw new InvalidOperationException("Transpose only works on 2D tensors");
            
            float[] transposed = new float[Size];
            int rows = Shape[0];
            int cols = Shape[1];
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    transposed[j * rows + i] = Data[i * cols + j];
                }
            }
            
            var result = new Tensor(transposed, new int[] { cols, rows }, RequiresGrad);
            if (RequiresGrad)
            {
                result.GradFn = new TransposeFunction(this);
            }
            return result;
        }

        /// <summary>
        /// Creates a copy of the tensor
        /// </summary>
        public Tensor Clone()
        {
            return new Tensor(Data.ToArray(), Shape.ToArray(), RequiresGrad);
        }

        /// <summary>
        /// Detaches tensor from computation graph
        /// </summary>
        public Tensor Detach()
        {
            return new Tensor(Data.ToArray(), Shape.ToArray(), false);
        }

        public override string ToString()
        {
            if (Shape.Length == 1)
            {
                return $"Tensor([{string.Join(", ", Data.Take(Math.Min(10, Data.Length)))}...], shape=[{string.Join(",", Shape)}])";
            }
            return $"Tensor(shape=[{string.Join(",", Shape)}], requiresGrad={RequiresGrad})";
        }

        #endregion

        #region Backpropagation

        /// <summary>
        /// Performs backpropagation (like reviewing game film to see what led to the score)
        /// </summary>
        public void Backward()
        {
            if (!RequiresGrad)
                throw new InvalidOperationException("Cannot call backward on tensor that doesn't require grad");

            // Initialize gradient to ones (∂Loss/∂Loss = 1)
            if (Grad == null)
            {
                Grad = Ones(Shape);
            }

            // Build topological order
            var topo = new List<Tensor>();
            var visited = new HashSet<Tensor>();
            BuildTopo(this, visited, topo);

            // Backpropagate in reverse order
            foreach (var tensor in topo.AsEnumerable().Reverse())
            {
                if (tensor.GradFn != null && tensor.Grad != null)
                {
                    tensor.GradFn.Backward(tensor.Grad);
                    
                    // Clear gradient for non-leaf tensors after passing it backward
                    if (!tensor.IsLeaf)
                    {
                        tensor.Grad = null;
                    }
                }
            }
        }

        /// <summary>
        /// Builds topological ordering for backprop
        /// </summary>
        private void BuildTopo(Tensor tensor, HashSet<Tensor> visited, List<Tensor> topo)
        {
            if (!visited.Contains(tensor))
            {
                visited.Add(tensor);
                if (tensor.GradFn != null)
                {
                    foreach (var input in tensor.GradFn.Inputs)
                    {
                        BuildTopo(input, visited, topo);
                    }
                }
                topo.Add(tensor);
            }
        }

        /// <summary>
        /// Clears the gradient (like resetting stats for a new game)
        /// </summary>
        public void ZeroGrad()
        {
            if (Grad != null)
            {
                for (int i = 0; i < Grad.Data.Length; i++)
                {
                    Grad.Data[i] = 0f;
                }
            }
        }

        #endregion

        #region Operator Overloading

        /// <summary>
        /// Element-wise addition
        /// </summary>
        public static Tensor operator +(Tensor a, Tensor b)
        {
            return new AddFunction().Forward(a, b);
        }

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        public static Tensor operator -(Tensor a, Tensor b)
        {
            return new SubtractFunction().Forward(a, b);
        }

        /// <summary>
        /// Element-wise multiplication
        /// </summary>
        public static Tensor operator *(Tensor a, Tensor b)
        {
            return new MultiplyFunction().Forward(a, b);
        }

        /// <summary>
        /// Element-wise division
        /// </summary>
        public static Tensor operator /(Tensor a, Tensor b)
        {
            return new DivideFunction().Forward(a, b);
        }

        /// <summary>
        /// Scalar multiplication
        /// </summary>
        public static Tensor operator *(Tensor a, float scalar)
        {
            float[] result = a.Data.Select(x => x * scalar).ToArray();
            var output = new Tensor(result, a.Shape, a.RequiresGrad);
            
            if (a.RequiresGrad)
            {
                output.GradFn = new ScalarMultiplyFunction(a, scalar);
            }
            
            return output;
        }

        /// <summary>
        /// Scalar multiplication (commutative)
        /// </summary>
        public static Tensor operator *(float scalar, Tensor a)
        {
            return a * scalar;
        }

        #endregion

        #region Tensor Operations

        /// <summary>
        /// Matrix multiplication (like combining player stats with team coefficients)
        /// </summary>
        public Tensor MatMul(Tensor other)
        {
            return new MatMulFunction().Forward(this, other);
        }

        /// <summary>
        /// ReLU activation: max(0, x) (like keeping only positive plays)
        /// </summary>
        public Tensor ReLU()
        {
            return new ReLUFunction().Forward(this);
        }

        /// <summary>
        /// Sigmoid activation: 1 / (1 + e^-x) (probability of success)
        /// </summary>
        public Tensor Sigmoid()
        {
            return new SigmoidFunction().Forward(this);
        }

        /// <summary>
        /// Tanh activation: (e^x - e^-x) / (e^x + e^-x)
        /// </summary>
        public Tensor Tanh()
        {
            return new TanhFunction().Forward(this);
        }

        /// <summary>
        /// Computes mean of all elements
        /// </summary>
        public Tensor Mean()
        {
            float sum = Data.Sum();
            float mean = sum / Size;
            
            var output = new Tensor(new float[] { mean }, new int[] { 1 }, RequiresGrad);
            
            if (RequiresGrad)
            {
                output.GradFn = new MeanFunction(this);
            }
            
            return output;
        }

        /// <summary>
        /// Computes sum of all elements
        /// </summary>
        public Tensor Sum()
        {
            float sum = Data.Sum();
            
            var output = new Tensor(new float[] { sum }, new int[] { 1 }, RequiresGrad);
            
            if (RequiresGrad)
            {
                output.GradFn = new SumFunction(this);
            }
            
            return output;
        }

        /// <summary>
        /// Gets a single element (for scalar tensors)
        /// </summary>
        public float Item()
        {
            if (Size != 1)
                throw new InvalidOperationException("Item() only works on scalar tensors");
            return Data[0];
        }

        #endregion
    }

    /// <summary>
    /// Base class for all automatic differentiation functions
    /// </summary>
    public abstract class Function
    {
        public List<Tensor> Inputs { get; protected set; } = new List<Tensor>();
        public abstract void Backward(Tensor grad);
    }
}
