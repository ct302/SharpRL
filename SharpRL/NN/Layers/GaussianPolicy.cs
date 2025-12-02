using System;
using System.Collections.Generic;
using SharpRL.AutoGrad;

namespace SharpRL.NN.Layers
{
    /// <summary>
    /// Gaussian Policy Network for continuous action spaces
    /// Outputs mean and log standard deviation for each action dimension
    /// 
    /// NFL ANALOGY:
    /// Instead of picking play #1, #2, or #3 (discrete), this lets you call plays with
    /// continuous adjustments - like "run play #1 but adjust blocking angle by 23.5 degrees"
    /// 
    /// The Gaussian distribution is like having a confidence interval on your play call:
    /// - Mean = Your intended play adjustment
    /// - Std = How much variation/exploration you allow
    /// - Log Std = Keeps std positive (you can't have negative confidence)
    /// </summary>
    public class GaussianPolicy : Module
    {
        private readonly Linear meanLayer;
        private readonly int actionDim;
        private readonly float actionScale;
        private float logStd;  // Learnable log standard deviation (shared across all actions)
        private readonly float logStdMin;
        private readonly float logStdMax;
        private readonly Random random;

        /// <summary>
        /// Create a Gaussian policy network
        /// </summary>
        public GaussianPolicy(
            int inputDim, 
            int actionDim,
            float actionScale = 1.0f,
            float logStdMin = -20f,
            float logStdMax = 2f,
            int? seed = null)
        {
            this.actionDim = actionDim;
            this.actionScale = actionScale;
            this.logStdMin = logStdMin;
            this.logStdMax = logStdMax;
            this.random = seed.HasValue ? new Random(seed.Value) : new Random();

            meanLayer = new Linear(inputDim, actionDim);
            logStd = 0f;
        }

        public override List<Tensor> Parameters()
        {
            return meanLayer.Parameters();
        }

        /// <summary>
        /// Sample an action from the Gaussian distribution
        /// </summary>
        public float[] Sample(float[] state, bool deterministic = false)
        {
            var stateTensor = new Tensor(state, new int[] { 1, state.Length });
            var meanTensor = meanLayer.Forward(stateTensor);
            

            if (deterministic)
            {
                // Deterministic: just return the mean scaled by actionScale
                var deterministicActions = new float[actionDim];
                for (int i = 0; i < actionDim; i++)
                {
                    deterministicActions[i] = meanTensor.Data[i] * actionScale;
                }
                return deterministicActions;
            }

            // Stochastic: sample from Gaussian distribution
            var std = MathF.Exp(Math.Clamp(logStd, logStdMin, logStdMax));
            var actions = new float[actionDim];
            
            for (int i = 0; i < actionDim; i++)
            {
                // Box-Muller transform for Gaussian sampling
                var u1 = (float)random.NextDouble();
                var u2 = (float)random.NextDouble();
                var gaussianSample = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
                
                actions[i] = (meanTensor.Data[i] + gaussianSample * std) * actionScale;
            }
            
            return actions;
        }

        /// <summary>
        /// Compute log probability of an action under the current policy
        /// </summary>
        public float LogProb(float[] state, float[] action)
        {
            var stateTensor = new Tensor(state, new int[] { 1, state.Length });
            var meanTensor = meanLayer.Forward(stateTensor);
            var std = MathF.Exp(Math.Clamp(logStd, logStdMin, logStdMax));

            // Compute log probability for Gaussian: log p(a|s) = -0.5 * ((a-μ)/σ)^2 - log(σ) - 0.5*log(2π)
            float logProb = 0f;
            for (int i = 0; i < actionDim; i++)
            {
                var normalizedAction = action[i] / actionScale;
                var diff = normalizedAction - meanTensor.Data[i];
                logProb += -0.5f * (diff * diff) / (std * std) - MathF.Log(std) - 0.5f * MathF.Log(2f * MathF.PI);
            }

            return logProb;
        }

        /// <summary>
        /// Get entropy of the policy distribution
        /// Higher entropy = more exploration
        /// </summary>
        public float Entropy()
        {
            var std = MathF.Exp(Math.Clamp(logStd, logStdMin, logStdMax));
            // Entropy of Gaussian: 0.5 * log(2πeσ²) for each dimension
            return actionDim * (0.5f * MathF.Log(2f * MathF.PI * MathF.E * std * std));
        }

        /// <summary>
        /// Update log standard deviation (for learning exploration)
        /// </summary>
        public void UpdateLogStd(float delta)
        {
            logStd = Math.Clamp(logStd + delta, logStdMin, logStdMax);
        }

        /// <summary>
        /// Get current standard deviation value
        /// </summary>
        public float GetStd()
        {
            return MathF.Exp(Math.Clamp(logStd, logStdMin, logStdMax));
        }

        public override Tensor Forward(Tensor input)
        {
            return meanLayer.Forward(input);
        }
    }
}
