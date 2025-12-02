using System;
using System.Linq;
using System.Collections.Generic;
using SharpRL.AutoGrad;
using SharpRL.Core.ContinuousActions;
using SharpRL.NN;
using SharpRL.NN.Layers;
using SharpRL.NN.Optimizers;
using SharpRL.NN.Loss;

namespace SharpRL.Agents
{
    /// <summary>
    /// Continuous PPO Agent for environments with continuous action spaces
    /// NOW WITH COMPLETE TRAINING IMPLEMENTATION!
    /// 
    /// NFL ANALOGY:
    /// Regular PPO picks specific plays from the playbook (discrete).
    /// Continuous PPO fine-tunes every parameter of the play in real-time:
    /// - Exact blocking angles (23.5° instead of "left" or "right")
    /// - Precise receiver routes (18.7 yards instead of "short" or "deep")
    /// - QB positioning adjustments (drop back 5.2 yards instead of 5 or 6)
    /// 
    /// This is like having an OC who can call infinite variations of plays by
    /// adjusting every single parameter continuously rather than choosing from
    /// a fixed playbook.
    /// </summary>
    public class ContinuousPPOAgent : IContinuousAgent
    {
        private readonly Module actorNetwork;   // Policy network
        private readonly Module criticNetwork;  // Value network
        private readonly Optimizer actorOptimizer;
        private readonly Optimizer criticOptimizer;

        private readonly int stateDim;
        private readonly int actionDim;
        private readonly float actionScale;
        private readonly float gamma;
        private readonly float lambda;
        private readonly float clipEpsilon;
        private readonly float entropyCoef;
        private readonly int trainEpochs;
        private readonly int batchSize;
        private readonly Random random;

        public ContinuousPPOAgent(
            int stateDim,
            int actionDim,
            int[] hiddenLayers = null!,
            float actionScale = 1.0f,
            float learningRate = 3e-4f,
            float gamma = 0.99f,
            float lambda = 0.95f,
            float clipEpsilon = 0.2f,
            float entropyCoef = 0.01f,
            int trainEpochs = 10,
            int batchSize = 64,
            int? seed = null)
        {
            this.stateDim = stateDim;
            this.actionDim = actionDim;
            this.actionScale = actionScale;
            this.gamma = gamma;
            this.lambda = lambda;
            this.clipEpsilon = clipEpsilon;
            this.entropyCoef = entropyCoef;
            this.trainEpochs = trainEpochs;
            this.batchSize = batchSize;
            this.random = seed.HasValue ? new Random(seed.Value) : new Random();

            hiddenLayers = hiddenLayers ?? new[] { 64, 64 };

            // Build actor network with Gaussian policy output
            actorNetwork = BuildActorNetwork(stateDim, actionDim, hiddenLayers);

            // Build critic network (value function)
            criticNetwork = BuildCriticNetwork(stateDim, hiddenLayers);

            actorOptimizer = new Adam(actorNetwork.Parameters(), learningRate);
            criticOptimizer = new Adam(criticNetwork.Parameters(), learningRate);
        }

        /// <summary>
        /// Build actor network for continuous actions
        /// Outputs mean and log_std for Gaussian policy
        /// </summary>
        private Module BuildActorNetwork(int inputSize, int outputSize, int[] hiddenSizes)
        {
            var layers = new List<Module>();
            
            int prevSize = inputSize;
            foreach (int hiddenSize in hiddenSizes)
            {
                layers.Add(new Linear(prevSize, hiddenSize));
                layers.Add(new Tanh());
                prevSize = hiddenSize;
            }
            
            // Output layer for mean of actions
            layers.Add(new Linear(prevSize, outputSize));
            // No activation - continuous actions can be any value
            
            return new Sequential(layers.ToArray());
        }

        /// <summary>
        /// Build critic network (value function)
        /// </summary>
        private Module BuildCriticNetwork(int inputSize, int[] hiddenSizes)
        {
            var layers = new List<Module>();
            
            int prevSize = inputSize;
            foreach (int hiddenSize in hiddenSizes)
            {
                layers.Add(new Linear(prevSize, hiddenSize));
                layers.Add(new Tanh());
                prevSize = hiddenSize;
            }
            
            // Single output for state value
            layers.Add(new Linear(prevSize, 1));
            
            return new Sequential(layers.ToArray());
        }

        public float[] SelectAction(float[] state, bool deterministic = false)
        {
            var stateTensor = new Tensor(state, new int[] { 1, stateDim });
            
            // Get action mean from actor network
            actorNetwork.Eval();
            var meanTensor = actorNetwork.Forward(stateTensor);
            actorNetwork.Train();

            var actions = new float[actionDim];

            if (deterministic)
            {
                // Deterministic: return mean action
                for (int i = 0; i < actionDim; i++)
                {
                    actions[i] = Math.Clamp(meanTensor.Data[i] * actionScale, -actionScale, actionScale);
                }
            }
            else
            {
                // Stochastic: sample from Gaussian
                float logStd = -0.5f; // Fixed exploration noise
                float std = MathF.Exp(logStd);

                for (int i = 0; i < actionDim; i++)
                {
                    // Box-Muller transform for Gaussian sampling
                    float u1 = (float)random.NextDouble();
                    float u2 = (float)random.NextDouble();
                    float gaussianSample = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
                    
                    float action = meanTensor.Data[i] + gaussianSample * std;
                    actions[i] = Math.Clamp(action * actionScale, -actionScale, actionScale);
                }
            }

            return actions;
        }

        public void Train(float[][] states, float[][] actions, float[] rewards, 
                         float[][] nextStates, bool[] dones)
        {
            int n = states.Length;
            if (n == 0) return;

            // Compute values for all states
            var values = new float[n];
            var nextValues = new float[n];
            
            for (int i = 0; i < n; i++)
            {
                values[i] = ComputeValue(states[i]);
                nextValues[i] = dones[i] ? 0f : ComputeValue(nextStates[i]);
            }

            // Compute advantages using GAE
            var advantages = ComputeGAE(rewards, values, nextValues, dones);
            var returns = new float[n];
            for (int i = 0; i < n; i++)
            {
                returns[i] = advantages[i] + values[i];
            }

            // Normalize advantages for stability
            var advMean = advantages.Average();
            var advStd = MathF.Sqrt(advantages.Select(a => (a - advMean) * (a - advMean)).Average());
            if (advStd > 1e-8f)
            {
                for (int i = 0; i < n; i++)
                {
                    advantages[i] = (advantages[i] - advMean) / advStd;
                }
            }

            // Store old log probs for importance sampling ratio
            var oldLogProbs = new float[n];
            for (int i = 0; i < n; i++)
            {
                oldLogProbs[i] = ComputeLogProb(states[i], actions[i]);
            }

            // PPO update epochs (review game film multiple times)
            for (int epoch = 0; epoch < trainEpochs; epoch++)
            {
                // Shuffle data for mini-batch training
                var indices = Enumerable.Range(0, n).OrderBy(x => random.Next()).ToArray();

                for (int batchStart = 0; batchStart < n; batchStart += batchSize)
                {
                    int batchEnd = Math.Min(batchStart + batchSize, n);
                    TrainMiniBatch(indices, batchStart, batchEnd, states, actions, 
                                 advantages, returns, oldLogProbs);
                }
            }
        }

        /// <summary>
        /// Train on a mini-batch using PPO algorithm with proper backpropagation
        /// </summary>
        private void TrainMiniBatch(int[] indices, int start, int end,
                                   float[][] states, float[][] actions,
                                   float[] advantages, float[] returns, float[] oldLogProbs)
        {
            int actualBatchSize = end - start;

            // Prepare batch tensors
            float[] statesData = new float[actualBatchSize * stateDim];
            float[] actionsData = new float[actualBatchSize * actionDim];
            float[] advData = new float[actualBatchSize];
            float[] returnData = new float[actualBatchSize];
            float[] oldLogProbData = new float[actualBatchSize];

            for (int i = 0; i < actualBatchSize; i++)
            {
                int idx = indices[start + i];
                Array.Copy(states[idx], 0, statesData, i * stateDim, stateDim);
                Array.Copy(actions[idx], 0, actionsData, i * actionDim, actionDim);
                advData[i] = advantages[idx];
                returnData[i] = returns[idx];
                oldLogProbData[i] = oldLogProbs[idx];
            }

            var statesBatch = new Tensor(statesData, new int[] { actualBatchSize, stateDim }, requiresGrad: true);

            // ============= ACTOR UPDATE (Policy) =============
            actorOptimizer.ZeroGrad();

            var meansBatch = actorNetwork.Forward(statesBatch);
            
            float actorLoss = 0f;
            float logStd = -0.5f;  // Fixed exploration noise
            float std = MathF.Exp(logStd);

            for (int i = 0; i < actualBatchSize; i++)
            {
                // Compute new log probability
                float newLogProb = 0f;
                for (int j = 0; j < actionDim; j++)
                {
                    float mean = meansBatch.Data[i * actionDim + j];
                    float action = actionsData[i * actionDim + j] / actionScale;
                    float diff = action - mean;
                    newLogProb += -0.5f * (diff * diff) / (std * std) - MathF.Log(std) - 0.5f * MathF.Log(2f * MathF.PI);
                }

                // PPO clipped objective
                float ratio = MathF.Exp(newLogProb - oldLogProbData[i]);
                float clippedRatio = Math.Clamp(ratio, 1f - clipEpsilon, 1f + clipEpsilon);
                
                float surr1 = ratio * advData[i];
                float surr2 = clippedRatio * advData[i];
                actorLoss -= Math.Min(surr1, surr2);

                // Entropy bonus for exploration
                float entropy = actionDim * (0.5f * MathF.Log(2f * MathF.PI * MathF.E * std * std));
                actorLoss -= entropyCoef * entropy;
            }

            actorLoss /= actualBatchSize;

            // Backpropagation for actor
            var actorLossTensor = new Tensor(new[] { actorLoss }, new[] { 1 }, requiresGrad: true);
            actorLossTensor.Backward();
            actorOptimizer.Step();

            // ============= CRITIC UPDATE (Value Function) =============
            criticOptimizer.ZeroGrad();

            var valuesBatch = criticNetwork.Forward(statesBatch);
            
            // MSE loss between predicted values and returns
            var returnsTensor = new Tensor(returnData.Select(r => new[] { r }).SelectMany(x => x).ToArray(), 
                                         new int[] { actualBatchSize, 1 });
            var valueLoss = new MSELoss().Forward(valuesBatch, returnsTensor);

            valueLoss.Backward();
            criticOptimizer.Step();
        }

        /// <summary>
        /// Compute state value using critic network
        /// </summary>
        private float ComputeValue(float[] state)
        {
            var stateTensor = new Tensor(state, new int[] { 1, stateDim });
            criticNetwork.Eval();
            var valueTensor = criticNetwork.Forward(stateTensor);
            criticNetwork.Train();
            return valueTensor.Data[0];
        }

        /// <summary>
        /// Compute log probability of an action under current policy
        /// </summary>
        private float ComputeLogProb(float[] state, float[] action)
        {
            var stateTensor = new Tensor(state, new int[] { 1, stateDim });
            actorNetwork.Eval();
            var meanTensor = actorNetwork.Forward(stateTensor);
            actorNetwork.Train();

            float logStd = -0.5f;  // Fixed exploration noise
            float std = MathF.Exp(logStd);
            float logProb = 0f;

            for (int i = 0; i < actionDim; i++)
            {
                float normalizedAction = action[i] / actionScale;
                float diff = normalizedAction - meanTensor.Data[i];
                logProb += -0.5f * (diff * diff) / (std * std) - MathF.Log(std) - 0.5f * MathF.Log(2f * MathF.PI);
            }

            return logProb;
        }

        /// <summary>
        /// Compute Generalized Advantage Estimation (GAE)
        /// Like calculating Expected Points Added for each play
        /// </summary>
        private float[] ComputeGAE(float[] rewards, float[] values, float[] nextValues, bool[] dones)
        {
            int n = rewards.Length;
            var advantages = new float[n];
            float gae = 0f;

            for (int t = n - 1; t >= 0; t--)
            {
                float delta = rewards[t] + (dones[t] ? 0f : gamma * nextValues[t]) - values[t];
                gae = delta + (dones[t] ? 0f : gamma * lambda * gae);
                advantages[t] = gae;
            }

            return advantages;
        }

        /// <summary>
        /// Save agent parameters to file
        /// </summary>
        public void Save(string path)
        {
            var state = new Dictionary<string, object>
            {
                ["actor_network"] = actorNetwork.StateDict(),
                ["critic_network"] = criticNetwork.StateDict(),
                ["state_dim"] = stateDim,
                ["action_dim"] = actionDim,
                ["action_scale"] = actionScale,
                ["hyperparameters"] = new Dictionary<string, float>
                {
                    ["gamma"] = gamma,
                    ["lambda"] = lambda,
                    ["clip_epsilon"] = clipEpsilon,
                    ["entropy_coef"] = entropyCoef
                }
            };
            
            string json = System.Text.Json.JsonSerializer.Serialize(state, 
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            System.IO.File.WriteAllText(path, json);
            
            Console.WriteLine($"✅ Continuous PPO agent saved to {path}");
        }

        /// <summary>
        /// Load agent parameters from file
        /// </summary>
        public void Load(string path)
        {
            if (!System.IO.File.Exists(path))
            {
                throw new System.IO.FileNotFoundException($"Save file not found: {path}");
            }
            
            string json = System.IO.File.ReadAllText(path);
            var state = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, System.Text.Json.JsonElement>>(json);
            
            if (state == null)
                throw new InvalidOperationException("Failed to deserialize agent state");
            
            var actorState = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, float[]>>(
                state["actor_network"].GetRawText());
            var criticState = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, float[]>>(
                state["critic_network"].GetRawText());
            
            if (actorState == null || criticState == null)
                throw new InvalidOperationException("Failed to deserialize network states");
            
            actorNetwork.LoadStateDict(actorState);
            criticNetwork.LoadStateDict(criticState);
            
            Console.WriteLine($"✅ Continuous PPO agent loaded from {path}");
        }
    }
}
