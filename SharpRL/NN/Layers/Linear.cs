using System;
using System.Collections.Generic;
using SharpRL.AutoGrad;

namespace SharpRL.NN.Layers
{
    /// <summary>
    /// Fully connected (linear) layer: y = xW^T + b
    /// 
    /// NFL ANALOGY:
    /// Like offensive line formations:
    /// - Input = Players coming to the line
    /// - Weights = Blocking assignments for each player
    /// - Bias = Base formation adjustment
    /// - Output = Resulting protection/gaps created
    /// </summary>
    public class Linear : Module
    {
        /// <summary>
        /// Weight matrix (the blocking schemes)
        /// </summary>
        public Tensor Weight { get; private set; }

        /// <summary>
        /// Bias vector (formation adjustments) - null if bias disabled
        /// </summary>
        public Tensor? Bias { get; private set; }

        private int inFeatures;
        private int outFeatures;
        private bool useBias;

        /// <summary>
        /// Creates a linear layer
        /// </summary>
        public Linear(int inFeatures, int outFeatures, bool bias = true)
        {
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
            this.useBias = bias;

            // He initialization for ReLU networks
            float scale = (float)Math.Sqrt(2.0 / inFeatures);
            Weight = Tensor.Randn(new int[] { outFeatures, inFeatures }, requiresGrad: true, scale: scale);

            if (useBias)
            {
                Bias = Tensor.Zeros(new int[] { outFeatures }, requiresGrad: true);
            }
        }

        /// <summary>
        /// Forward pass: output = input @ W^T + b
        /// </summary>
        public override Tensor Forward(Tensor input)
        {
            // Handle both 1D and 2D inputs
            bool is1D = input.Shape.Length == 1;
            if (is1D)
            {
                input = input.Reshape(1, -1);
            }

            // input: (batch_size, in_features)
            // Weight: (out_features, in_features)
            // Need: input @ Weight^T = (batch_size, out_features)

            var output = input.MatMul(Weight.T());

            if (useBias)
            {
                // Broadcast bias across batch dimension
                output = output + BroadcastBias(output.Shape[0]);
            }

            if (is1D)
            {
                output = output.Reshape(-1);
            }

            return output;
        }

        private Tensor BroadcastBias(int batchSize)
        {
            if (Bias == null)
                throw new InvalidOperationException("Bias is null - this method should only be called when useBias is true");
            
            float[] broadcasted = new float[batchSize * outFeatures];
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < outFeatures; j++)
                {
                    broadcasted[i * outFeatures + j] = Bias.Data[j];
                }
            }
            
            var output = new Tensor(broadcasted, new int[] { batchSize, outFeatures }, Bias.RequiresGrad);
            
            // CRITICAL: Link gradient back to original bias tensor
            if (Bias.RequiresGrad)
            {
                output.GradFn = new BroadcastBiasFunction(Bias, batchSize, outFeatures);
                output.IsLeaf = false;
            }
            
            return output;
        }

        public override List<Tensor> Parameters()
        {
            var parameters = new List<Tensor> { Weight };
            if (useBias && Bias != null)
            {
                parameters.Add(Bias);
            }
            return parameters;
        }

        public override string ToString()
        {
            return $"Linear(in_features={inFeatures}, out_features={outFeatures}, bias={useBias})";
        }
    }
}
