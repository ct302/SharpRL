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
    /// Advantage Actor-Critic (A2C) Agent
    /// The reliable workhorse between DQN and PPO - simpler than PPO, more powerful than DQN
    /// 
    /// NFL ANALOGY:
    /// A2C is like having a veteran offensive coordinator (actor) and defensive coordinator (critic):
    /// - Actor Network = OC who calls the plays based on field position
    /// - Critic Network = DC who evaluates field position value
    /// - Single-step updates = Adjusting after every play, not waiting for game film
    /// - Advantage = How much better this play was vs expected (EPA - Expected Points Added)
    /// - Entropy bonus = Keeping the playbook unpredictable
    /// 
    /// WHY A2C:
    /// - Simpler than PPO (no clipping, no trajectory buffer, no mini-batches)
    /// - More stable than vanilla Policy Gradient (uses critic baseline)
    /// - Faster training than PPO (synchronous, immediate updates)
    /// - Great for environments where you want quick adaptation
    /// </summary>
    public class A2CAgent : IAgent<float[], int>
    {
        // Networks
        private Module actorNetwork;      // Policy network (play caller)
        private Module criticNetwork;     // Value network (position evaluator)
        private Optimizer actorOptimizer;
        private Optimizer criticOptimizer;
        
        // Hyperparameters
        private readonly int stateSize;
        private readonly int actionSize;
        private readonly float learningRate;
        private readonly float discountFactor;      // γ - future reward discount
        private readonly float entropyCoeff;        // Exploration bonus
        private readonly float valueCoeff;          // Value loss weight
        private readonly int updateFrequency;       // Update every N steps
        
        // Experience buffer for batched updates
        private List<A2CExperience> experienceBuffer;
        private readonly Random random;
        
        public A2CAgent(
            int stateSize,
            int actionSize,
            int[] hiddenLayers = null!,
            float learningRate = 0.001f,
            float discountFactor = 0.99f,
            float entropyCoeff = 0.01f,
            float valueCoeff = 0.5f,
            int updateFrequency = 5,
            int? seed = null)
        {
            this.stateSize = stateSize;
            this.actionSize = actionSize;
            this.learningRate = learningRate;
            this.discountFactor = discountFactor;
            this.entropyCoeff = entropyCoeff;
            this.valueCoeff = valueCoeff;
            this.updateFrequency = updateFrequency;
            
            hiddenLayers = hiddenLayers ?? new[] { 64, 64 };
            
            // Build actor-critic networks
            actorNetwork = BuildActorNetwork(stateSize, actionSize, hiddenLayers);
            criticNetwork = BuildCriticNetwork(stateSize, hiddenLayers);
            
            // Setup optimizers (both networks learn simultaneously)
            actorOptimizer = new Adam(actorNetwork.Parameters(), learningRate);
            criticOptimizer = new Adam(criticNetwork.Parameters(), learningRate);
            
            experienceBuffer = new List<A2CExperience>();
            random = seed.HasValue ? new Random(seed.Value) : new Random();
        }
        
        /// <summary>
        /// Build actor network (policy) - outputs log probabilities
        /// NFL ANALOGY: The offensive coordinator's decision-making process
        /// </summary>
        private Module BuildActorNetwork(int inputSize, int outputSize, int[] hiddenSizes)
        {
            var layers = new List<Module>();
            
            int prevSize = inputSize;
            foreach (int hiddenSize in hiddenSizes)
            {
                layers.Add(new Linear(prevSize, hiddenSize));
                layers.Add(new Tanh());  // Tanh works well for policy networks
                prevSize = hiddenSize;
            }
            
            // Output layer with log softmax for action probabilities
            layers.Add(new Linear(prevSize, outputSize));
            layers.Add(new LogSoftmax());
            
            return new Sequential(layers.ToArray());
        }
        
        /// <summary>
        /// Build critic network (value function) - outputs state value
        /// NFL ANALOGY: The defensive coordinator's evaluation of field position
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
        
        /// <summary>
        /// Select action based on current policy
        /// </summary>
        public int SelectAction(float[] state, bool explore = true)
        {
            var stateTensor = new Tensor(state, new int[] { 1, stateSize });
            
            // Get action probabilities from actor
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
            
            // Normalize
            float sum = probs.Sum();
            for (int i = 0; i < actionSize; i++)
            {
                probs[i] /= sum;
            }
            
            if (explore)
            {
                // Sample from distribution (like calling plays based on success probability)
                return SampleFromDistribution(probs);
            }
            else
            {
                // Greedy: Choose highest probability action
                return Array.IndexOf(probs, probs.Max());
            }
        }
        
        /// <summary>
        /// Sample action from probability distribution
        /// </summary>
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
        
        /// <summary>
        /// Store experience and train when buffer is full
        /// A2C uses synchronous updates - train after N experiences
        /// </summary>
        public void Update(float[] state, int action, double reward, float[] nextState, bool done)
        {
            // Get state value for advantage calculation
            var stateTensor = new Tensor(state, new int[] { 1, stateSize });
            
            criticNetwork.Eval();
            var stateValue = criticNetwork.Forward(stateTensor);
            float value = stateValue.Data[0];
            criticNetwork.Train();
            
            // Get action log probability
            actorNetwork.Eval();
            var logProbs = actorNetwork.Forward(stateTensor);
            float logProb = logProbs.Data[action];
            actorNetwork.Train();
            
            // Store experience
            experienceBuffer.Add(new A2CExperience
            {
                State = state,
                Action = action,
                Reward = (float)reward,
                NextState = nextState,
                Done = done,
                Value = value,
                LogProb = logProb
            });
            
            // Train when buffer reaches update frequency
            if (experienceBuffer.Count >= updateFrequency || done)
            {
                TrainOnBatch();
                experienceBuffer.Clear();
            }
        }
        
        /// <summary>
        /// Train on collected batch of experiences
        /// NFL ANALOGY: Like reviewing the last few plays and adjusting strategy
        /// </summary>
        private void TrainOnBatch()
        {
            if (experienceBuffer.Count == 0)
                return;
            
            int batchSize = experienceBuffer.Count;
            
            // Calculate returns and advantages
            float[] returns = new float[batchSize];
            float[] advantages = new float[batchSize];
            
            // Bootstrap from next state if not terminal
            float nextValue = 0;
            if (!experienceBuffer[batchSize - 1].Done)
            {
                var nextStateTensor = new Tensor(
                    experienceBuffer[batchSize - 1].NextState, 
                    new int[] { 1, stateSize }
                );
                criticNetwork.Eval();
                var nextValueTensor = criticNetwork.Forward(nextStateTensor);
                nextValue = nextValueTensor.Data[0];
                criticNetwork.Train();
            }
            
            // Calculate returns backward (like calculating final score from current position)
            float runningReturn = nextValue;
            for (int i = batchSize - 1; i >= 0; i--)
            {
                runningReturn = experienceBuffer[i].Reward + discountFactor * runningReturn;
                returns[i] = runningReturn;
                
                // Advantage = Return - Value (how much better than expected)
                // NFL ANALOGY: Actual points gained - Expected points (EPA)
                advantages[i] = returns[i] - experienceBuffer[i].Value;
            }
            
            // Normalize advantages (like adjusting for game context)
            float advMean = advantages.Average();
            float advStd = (float)Math.Sqrt(advantages.Select(a => (a - advMean) * (a - advMean)).Average());
            if (advStd > 1e-8f)
            {
                for (int i = 0; i < advantages.Length; i++)
                {
                    advantages[i] = (advantages[i] - advMean) / advStd;
                }
            }
            
            // Prepare batch tensors
            float[] statesData = new float[batchSize * stateSize];
            int[] actions = new int[batchSize];
            float[] batchReturns = new float[batchSize];
            float[] batchAdvantages = new float[batchSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                Array.Copy(experienceBuffer[i].State, 0, statesData, i * stateSize, stateSize);
                actions[i] = experienceBuffer[i].Action;
                batchReturns[i] = returns[i];
                batchAdvantages[i] = advantages[i];
            }
            
            var statesTensor = new Tensor(statesData, new int[] { batchSize, stateSize }, requiresGrad: true);
            
            // === ACTOR UPDATE (Policy Gradient) ===
            actorOptimizer.ZeroGrad();
            var logProbs = actorNetwork.Forward(statesTensor);
            
            float actorLoss = 0;
            float entropyBonus = 0;
            
            for (int i = 0; i < batchSize; i++)
            {
                // Policy gradient: -log(π(a|s)) * Advantage
                // NFL ANALOGY: Reinforce plays that worked better than expected
                float logProb = logProbs.Data[i * actionSize + actions[i]];
                actorLoss -= logProb * batchAdvantages[i];
                
                // Entropy bonus for exploration
                // NFL ANALOGY: Bonus for keeping playbook unpredictable
                for (int a = 0; a < actionSize; a++)
                {
                    float prob = (float)Math.Exp(logProbs.Data[i * actionSize + a]);
                    if (prob > 1e-8f)
                    {
                        entropyBonus -= prob * logProbs.Data[i * actionSize + a];
                    }
                }
            }
            
            // Total actor loss = policy loss - entropy bonus
            actorLoss = (actorLoss - entropyCoeff * entropyBonus) / batchSize;
            
            var actorLossTensor = new Tensor(new float[] { actorLoss }, new int[] { 1 }, requiresGrad: true);
            actorLossTensor.Backward();
            actorOptimizer.Step();
            
            // === CRITIC UPDATE (Value Function) ===
            criticOptimizer.ZeroGrad();
            var values = criticNetwork.Forward(statesTensor);
            
            // MSE loss between predicted values and actual returns
            // NFL ANALOGY: Improve our field position evaluation accuracy
            var returnsTensor = new Tensor(batchReturns, new int[] { batchSize, 1 });
            var criticLoss = new MSELoss().Forward(values, returnsTensor);
            
            // Scale by value coefficient
            var scaledCriticLoss = criticLoss * valueCoeff;
            
            scaledCriticLoss.Backward();
            criticOptimizer.Step();
        }
        
        /// <summary>
        /// A2C doesn't need additional training step (updates happen in Update())
        /// </summary>
        public void Train()
        {
            // A2C updates synchronously in Update() method
            // No epsilon decay or other per-episode operations needed
        }
        
        /// <summary>
        /// Save agent to disk
        /// </summary>
        public void Save(string path)
        {
            var state = new Dictionary<string, object>
            {
                ["actor_network"] = actorNetwork.StateDict(),
                ["critic_network"] = criticNetwork.StateDict(),
                ["state_size"] = stateSize,
                ["action_size"] = actionSize,
                ["hyperparameters"] = new Dictionary<string, float>
                {
                    ["learning_rate"] = learningRate,
                    ["discount_factor"] = discountFactor,
                    ["entropy_coeff"] = entropyCoeff,
                    ["value_coeff"] = valueCoeff
                }
            };
            
            string json = System.Text.Json.JsonSerializer.Serialize(state, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            System.IO.File.WriteAllText(path, json);
        }
        
        /// <summary>
        /// Load agent from disk
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
        }
        
        /// <summary>
        /// Get current experience buffer size
        /// </summary>
        public int GetBufferSize() => experienceBuffer.Count;
        
        /// <summary>
        /// Experience tuple for A2C
        /// </summary>
        private class A2CExperience
        {
            public float[] State { get; set; } = null!;
            public int Action { get; set; }
            public float Reward { get; set; }
            public float[] NextState { get; set; } = null!;
            public bool Done { get; set; }
            public float Value { get; set; }
            public float LogProb { get; set; }
        }
    }
}
