using System;
using System.Collections.Generic;
using System.Linq;
using SharpRL.Core;
using SharpRL.AutoGrad;
using SharpRL.NN;
using SharpRL.NN.Layers;
using SharpRL.NN.Loss;
using SharpRL.NN.Optimizers;

namespace SharpRL.Agents
{
    /// <summary>
    /// Proximal Policy Optimization (PPO) Agent
    /// The Tom Brady of RL algorithms - reliable, consistent, and championship-proven
    /// </summary>
    public class PPOAgent : IAgent<float[], int>
    {
        // Networks
        private Module actorNetwork;
        private Module criticNetwork;
        private Optimizer actorOptimizer;
        private Optimizer criticOptimizer;
        
        // Hyperparameters
        private readonly int stateSize;
        private readonly int actionSize;
        private readonly float learningRate;
        private readonly float discountFactor;
        private readonly float gaeLength;
        private readonly float clipEpsilon;
        private readonly float entropyCoeff;
        private readonly float valueCoeff;
        private readonly int ppoEpochs;
        private readonly int miniBatchSize;
        
        // Experience buffer for current trajectory
        private List<PPOExperience> trajectoryBuffer;
        private readonly Random random;
        
        public PPOAgent(
            int stateSize,
            int actionSize,
            int[] hiddenLayers = null!,
            float learningRate = 0.0003f,
            float discountFactor = 0.99f,
            float gaeLength = 0.95f,
            float clipEpsilon = 0.2f,
            float entropyCoeff = 0.01f,
            float valueCoeff = 0.5f,
            int ppoEpochs = 4,
            int miniBatchSize = 64,
            int? seed = null)
        {
            this.stateSize = stateSize;
            this.actionSize = actionSize;
            this.learningRate = learningRate;
            this.discountFactor = discountFactor;
            this.gaeLength = gaeLength;
            this.clipEpsilon = clipEpsilon;
            this.entropyCoeff = entropyCoeff;
            this.valueCoeff = valueCoeff;
            this.ppoEpochs = ppoEpochs;
            this.miniBatchSize = miniBatchSize;
            
            hiddenLayers = hiddenLayers ?? new[] { 64, 64 };
            
            // Build actor-critic networks
            actorNetwork = BuildActorNetwork(stateSize, actionSize, hiddenLayers);
            criticNetwork = BuildCriticNetwork(stateSize, hiddenLayers);
            
            // Setup optimizers
            actorOptimizer = new Adam(actorNetwork.Parameters(), learningRate);
            criticOptimizer = new Adam(criticNetwork.Parameters(), learningRate);
            
            trajectoryBuffer = new List<PPOExperience>();
            random = seed.HasValue ? new Random(seed.Value) : new Random();
        }
        
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
            
            layers.Add(new Linear(prevSize, outputSize));
            layers.Add(new LogSoftmax());
            
            return new Sequential(layers.ToArray());
        }
        
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
            
            layers.Add(new Linear(prevSize, 1));
            
            return new Sequential(layers.ToArray());
        }
        
        public int SelectAction(float[] state, bool explore = true)
        {
            var stateTensor = new Tensor(state, new int[] { 1, stateSize });
            
            actorNetwork.Eval();
            var logProbs = actorNetwork.Forward(stateTensor);
            actorNetwork.Train();
            
            // Convert log probs to probabilities
            float[] probs = new float[actionSize];
            float maxLogProb = logProbs.Data.Max();
            for (int i = 0; i < actionSize; i++)
            {
                probs[i] = (float)Math.Exp(logProbs.Data[i] - maxLogProb);
            }
            
            float sum = probs.Sum();
            for (int i = 0; i < actionSize; i++)
            {
                probs[i] /= sum;
            }
            
            if (explore)
            {
                return SampleFromDistribution(probs);
            }
            else
            {
                return Array.IndexOf(probs, probs.Max());
            }
        }
        
        private int SampleFromDistribution(float[] probs)
        {
            float randomValue = (float)random.NextDouble();
            float cumulative = 0;
            
            for (int i = 0; i < probs.Length; i++)
            {
                cumulative += probs[i];
                if (randomValue <= cumulative)
                {
                    return i;
                }
            }
            
            return probs.Length - 1;
        }
        
        public void Update(float[] state, int action, double reward, float[] nextState, bool done)
        {
            var stateTensor = new Tensor(state, new int[] { 1, stateSize });
            
            actorNetwork.Eval();
            var logProbs = actorNetwork.Forward(stateTensor);
            float oldLogProb = logProbs.Data[action];
            
            criticNetwork.Eval();
            var value = criticNetwork.Forward(stateTensor);
            float stateValue = value.Data[0];
            
            actorNetwork.Train();
            criticNetwork.Train();
            
            trajectoryBuffer.Add(new PPOExperience
            {
                State = state,
                Action = action,
                Reward = (float)reward,
                NextState = nextState,
                Done = done,
                OldLogProb = oldLogProb,
                Value = stateValue
            });
            
            if (done || trajectoryBuffer.Count >= 2048)
            {
                TrainOnTrajectory();
                trajectoryBuffer.Clear();
            }
        }
        
        private void TrainOnTrajectory()
        {
            if (trajectoryBuffer.Count == 0)
                return;
            
            var advantages = ComputeGAE();
            var returns = new float[trajectoryBuffer.Count];
            
            for (int i = 0; i < trajectoryBuffer.Count; i++)
            {
                returns[i] = advantages[i] + trajectoryBuffer[i].Value;
            }
            
            // Normalize advantages
            float advMean = advantages.Average();
            float advStd = (float)Math.Sqrt(advantages.Select(a => (a - advMean) * (a - advMean)).Average() + 1e-8f);
            for (int i = 0; i < advantages.Length; i++)
            {
                advantages[i] = (advantages[i] - advMean) / advStd;
            }
            
            for (int epoch = 0; epoch < ppoEpochs; epoch++)
            {
                var indices = Enumerable.Range(0, trajectoryBuffer.Count).ToList();
                Shuffle(indices);
                
                for (int start = 0; start < trajectoryBuffer.Count; start += miniBatchSize)
                {
                    int end = Math.Min(start + miniBatchSize, trajectoryBuffer.Count);
                    var batchIndices = indices.GetRange(start, end - start);
                    
                    TrainMiniBatch(batchIndices, advantages, returns);
                }
            }
        }
        
        private void TrainMiniBatch(List<int> indices, float[] advantages, float[] returns)
        {
            int batchSize = indices.Count;
            
            float[] statesData = new float[batchSize * stateSize];
            int[] actions = new int[batchSize];
            float[] oldLogProbs = new float[batchSize];
            float[] batchAdvantages = new float[batchSize];
            float[] batchReturns = new float[batchSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                int idx = indices[i];
                Array.Copy(trajectoryBuffer[idx].State, 0, statesData, i * stateSize, stateSize);
                actions[i] = trajectoryBuffer[idx].Action;
                oldLogProbs[i] = trajectoryBuffer[idx].OldLogProb;
                batchAdvantages[i] = advantages[idx];
                batchReturns[i] = returns[idx];
            }
            
            var statesTensor = new Tensor(statesData, new int[] { batchSize, stateSize });
            
            // ===== ACTOR UPDATE =====
            actorOptimizer.ZeroGrad();
            var logProbs = actorNetwork.Forward(statesTensor);
            
            // Build policy loss through tensor operations
            float[] policyLossData = new float[batchSize];
            float[] entropyData = new float[batchSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                float newLogProb = logProbs.Data[i * actionSize + actions[i]];
                float ratio = (float)Math.Exp(newLogProb - oldLogProbs[i]);
                
                float surr1 = ratio * batchAdvantages[i];
                float surr2 = Math.Clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon) * batchAdvantages[i];
                policyLossData[i] = -Math.Min(surr1, surr2);
                
                // Entropy
                float entropy = 0;
                for (int a = 0; a < actionSize; a++)
                {
                    float prob = (float)Math.Exp(logProbs.Data[i * actionSize + a]);
                    if (prob > 1e-8f)
                    {
                        entropy -= prob * logProbs.Data[i * actionSize + a];
                    }
                }
                entropyData[i] = entropy;
            }
            
            // Mean policy loss minus entropy bonus
            float meanPolicyLoss = policyLossData.Average();
            float meanEntropy = entropyData.Average();
            float totalActorLoss = meanPolicyLoss - entropyCoeff * meanEntropy;
            
            var actorLoss = new Tensor(new float[] { totalActorLoss }, new int[] { 1 }, logProbs.RequiresGrad);
            if (logProbs.RequiresGrad)
            {
                actorLoss.GradFn = new PPOActorLossFunction(logProbs, actions, oldLogProbs, batchAdvantages, clipEpsilon, entropyCoeff, actionSize);
                actorLoss.IsLeaf = false;
            }
            
            actorLoss.Backward();
            actorOptimizer.Step();
            
            // ===== CRITIC UPDATE =====
            criticOptimizer.ZeroGrad();
            var values = criticNetwork.Forward(statesTensor);
            
            var returnsTensor = new Tensor(batchReturns, new int[] { batchSize, 1 });
            var valueLoss = new MSELoss().Forward(values, returnsTensor);
            
            valueLoss.Backward();
            criticOptimizer.Step();
        }
        
        private float[] ComputeGAE()
        {
            int n = trajectoryBuffer.Count;
            float[] advantages = new float[n];
            float lastAdvantage = 0;
            
            for (int t = n - 1; t >= 0; t--)
            {
                float delta;
                if (t == n - 1)
                {
                    if (trajectoryBuffer[t].Done)
                    {
                        delta = trajectoryBuffer[t].Reward - trajectoryBuffer[t].Value;
                    }
                    else
                    {
                        var nextStateTensor = new Tensor(trajectoryBuffer[t].NextState, new int[] { 1, stateSize });
                        criticNetwork.Eval();
                        var nextValue = criticNetwork.Forward(nextStateTensor);
                        criticNetwork.Train();
                        delta = trajectoryBuffer[t].Reward + discountFactor * nextValue.Data[0] - trajectoryBuffer[t].Value;
                    }
                }
                else
                {
                    delta = trajectoryBuffer[t].Reward + discountFactor * trajectoryBuffer[t + 1].Value - trajectoryBuffer[t].Value;
                }
                
                lastAdvantage = delta + discountFactor * gaeLength * lastAdvantage;
                advantages[t] = lastAdvantage;
            }
            
            return advantages;
        }
        
        private void Shuffle<T>(List<T> list)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }
        
        public void Train()
        {
            // PPO doesn't use epsilon decay
        }
        
        public void Save(string path)
        {
            var state = new Dictionary<string, object>
            {
                ["actor_network"] = actorNetwork.StateDict(),
                ["critic_network"] = criticNetwork.StateDict(),
                ["state_size"] = stateSize,
                ["action_size"] = actionSize
            };
            
            string json = System.Text.Json.JsonSerializer.Serialize(state, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            System.IO.File.WriteAllText(path, json);
        }
        
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
        }
        
        private class PPOExperience
        {
            public required float[] State { get; set; }
            public int Action { get; set; }
            public float Reward { get; set; }
            public required float[] NextState { get; set; }
            public bool Done { get; set; }
            public float OldLogProb { get; set; }
            public float Value { get; set; }
        }
    }
    
    /// <summary>
    /// PPO Actor Loss backward function
    /// </summary>
    internal class PPOActorLossFunction : Function
    {
        private int[] actions;
        private float[] oldLogProbs;
        private float[] advantages;
        private float clipEpsilon;
        private float entropyCoeff;
        private int actionSize;

        public PPOActorLossFunction(Tensor logProbs, int[] actions, float[] oldLogProbs, 
                                     float[] advantages, float clipEpsilon, float entropyCoeff, int actionSize)
        {
            Inputs = new List<Tensor> { logProbs };
            this.actions = actions;
            this.oldLogProbs = oldLogProbs;
            this.advantages = advantages;
            this.clipEpsilon = clipEpsilon;
            this.entropyCoeff = entropyCoeff;
            this.actionSize = actionSize;
        }

        public override void Backward(Tensor grad)
        {
            var logProbs = Inputs[0];
            if (!logProbs.RequiresGrad)
                return;

            if (logProbs.Grad == null)
                logProbs.Grad = Tensor.Zeros(logProbs.Shape);

            int batchSize = logProbs.Shape[0];
            float scale = grad.Data[0] / batchSize;

            for (int i = 0; i < batchSize; i++)
            {
                float newLogProb = logProbs.Data[i * actionSize + actions[i]];
                float ratio = (float)Math.Exp(newLogProb - oldLogProbs[i]);
                
                // Policy gradient
                float surr1 = ratio * advantages[i];
                float surr2 = Math.Clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon) * advantages[i];
                
                // Gradient of clipped objective
                float policyGrad;
                if (surr1 < surr2)
                {
                    // Using surr1, gradient flows through ratio
                    policyGrad = -ratio * advantages[i];
                }
                else
                {
                    // Using surr2, check if clipped
                    if (ratio < 1 - clipEpsilon || ratio > 1 + clipEpsilon)
                    {
                        policyGrad = 0; // Clipped, no gradient
                    }
                    else
                    {
                        policyGrad = -ratio * advantages[i];
                    }
                }
                
                // Apply gradient to selected action's log prob
                logProbs.Grad.Data[i * actionSize + actions[i]] += scale * policyGrad;
                
                // Entropy gradient: -entropyCoeff * d/dlogp[-sum(p * logp)]
                // = entropyCoeff * sum((1 + logp) * dp/dlogp)
                // For softmax: dp_j/dlogp_i = p_i * (delta_ij - p_j)
                for (int a = 0; a < actionSize; a++)
                {
                    float prob = (float)Math.Exp(logProbs.Data[i * actionSize + a]);
                    float entropyGrad = entropyCoeff * (1 + logProbs.Data[i * actionSize + a]) * prob;
                    logProbs.Grad.Data[i * actionSize + a] += scale * entropyGrad;
                }
            }
        }
    }

    /// <summary>
    /// PPO Policy Loss backward (without entropy)
    /// </summary>
    internal class PPOPolicyLossFunction : Function
    {
        private int[] actions;
        private float[] oldLogProbs;
        private float[] advantages;
        private float clipEpsilon;
        private int actionSize;

        public PPOPolicyLossFunction(Tensor logProbs, int[] actions, float[] oldLogProbs, 
                                      float[] advantages, float clipEpsilon, int actionSize)
        {
            Inputs = new List<Tensor> { logProbs };
            this.actions = actions;
            this.oldLogProbs = oldLogProbs;
            this.advantages = advantages;
            this.clipEpsilon = clipEpsilon;
            this.actionSize = actionSize;
        }

        public override void Backward(Tensor grad)
        {
            var logProbs = Inputs[0];
            if (!logProbs.RequiresGrad)
                return;

            if (logProbs.Grad == null)
                logProbs.Grad = Tensor.Zeros(logProbs.Shape);

            int batchSize = logProbs.Shape[0];

            for (int i = 0; i < batchSize; i++)
            {
                float newLogProb = logProbs.Data[i * actionSize + actions[i]];
                float ratio = (float)Math.Exp(newLogProb - oldLogProbs[i]);
                
                float surr1 = ratio * advantages[i];
                float surr2 = Math.Clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon) * advantages[i];
                
                float policyGrad;
                if (surr1 < surr2)
                {
                    policyGrad = -ratio * advantages[i] * grad.Data[i];
                }
                else
                {
                    if (ratio < 1 - clipEpsilon || ratio > 1 + clipEpsilon)
                    {
                        policyGrad = 0;
                    }
                    else
                    {
                        policyGrad = -ratio * advantages[i] * grad.Data[i];
                    }
                }
                
                logProbs.Grad.Data[i * actionSize + actions[i]] += policyGrad;
            }
        }
    }
}
