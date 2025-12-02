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
    /// Deep Q-Network Agent with integrated SharpGrad tensor system
    /// Supports both standard DQN and Double DQN (reduces overestimation bias)
    /// 
    /// NFL ANALOGY:
    /// DQN is like having an AI offensive coordinator:
    /// - Q-Network = Main playbook being refined each game
    /// - Target Network = Stable reference playbook (updated periodically)
    /// - Replay Buffer = Game film library for studying
    /// - Epsilon-greedy = Mixing proven plays with trick plays
    /// 
    /// DOUBLE DQN:
    /// - Standard DQN can overestimate Q-values (too optimistic about plays)
    /// - Double DQN: One coach picks the play, another coach evaluates it
    /// - This separation prevents overconfidence and improves learning stability
    /// </summary>
    public class DQNAgent : IAgent<float[], int>
    {
        private Module qNetwork;        // Main network (current playbook)
        private Module targetNetwork;   // Target network (stable playbook)
        private ReplayBuffer<float[], int> replayBuffer;
        private Optimizer optimizer;
        private SmoothL1Loss lossFunction;
        
        private readonly int stateSize;
        private readonly int actionSize;
        private readonly float discountFactor;
        private float epsilon;
        private readonly float epsilonDecay;
        private readonly float epsilonMin;
        private readonly int batchSize;
        private readonly int targetUpdateFrequency;
        private readonly bool useDoubleDQN;
        private int stepCount = 0;
        private readonly Random random;

        public DQNAgent(
            int stateSize,
            int actionSize,
            int[] hiddenLayers = null!,
            float learningRate = 0.0001f,
            float discountFactor = 0.99f,
            float epsilon = 1.0f,
            float epsilonDecay = 0.995f,
            float epsilonMin = 0.01f,
            int batchSize = 32,
            int bufferSize = 10000,
            int targetUpdateFreq = 100,
            bool useDoubleDQN = false,
            int? seed = null)
        {
            this.stateSize = stateSize;
            this.actionSize = actionSize;
            this.discountFactor = discountFactor;
            this.epsilon = epsilon;
            this.epsilonDecay = epsilonDecay;
            this.epsilonMin = epsilonMin;
            this.batchSize = batchSize;
            this.targetUpdateFrequency = targetUpdateFreq;
            this.useDoubleDQN = useDoubleDQN;
            
            // Default architecture if not specified
            hiddenLayers = hiddenLayers ?? new[] { 128, 64 };
            
            // Build Q-networks (like designing the playbook structure)
            qNetwork = BuildNetwork(stateSize, actionSize, hiddenLayers);
            targetNetwork = BuildNetwork(stateSize, actionSize, hiddenLayers);
            
            // Initialize target network with same weights
            CopyWeights(qNetwork, targetNetwork);
            
            // Setup optimizer and loss
            optimizer = new Adam(qNetwork.Parameters(), learningRate);
            lossFunction = new SmoothL1Loss();
            
            // Initialize replay buffer
            replayBuffer = new ReplayBuffer<float[], int>(bufferSize, seed);
            random = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        /// <summary>
        /// Builds a neural network for Q-function approximation
        /// </summary>
        private Module BuildNetwork(int inputSize, int outputSize, int[] hiddenSizes)
        {
            var layers = new List<Module>();
            
            int prevSize = inputSize;
            foreach (int hiddenSize in hiddenSizes)
            {
                layers.Add(new Linear(prevSize, hiddenSize));
                layers.Add(new ReLU());
                prevSize = hiddenSize;
            }
            
            // Output layer (Q-values for each action)
            layers.Add(new Linear(prevSize, outputSize));
            
            return new Sequential(layers.ToArray());
        }

        /// <summary>
        /// Copies weights from source to target network
        /// Like updating the reference playbook with proven plays
        /// </summary>
        private void CopyWeights(Module source, Module target)
        {
            var sourceParams = source.Parameters();
            var targetParams = target.Parameters();
            
            for (int i = 0; i < sourceParams.Count; i++)
            {
                Array.Copy(sourceParams[i].Data, targetParams[i].Data, sourceParams[i].Size);
            }
        }

        /// <summary>
        /// Select action using epsilon-greedy policy
        /// </summary>
        public int SelectAction(float[] state, bool explore = true)
        {
            // Epsilon-greedy exploration (try new plays vs proven plays)
            if (explore && random.NextDouble() < epsilon)
            {
                return random.Next(actionSize);
            }
            
            // Get Q-values from network
            var stateTensor = new Tensor(state, new int[] { 1, stateSize });
            qNetwork.Eval();
            var qValues = qNetwork.Forward(stateTensor);
            qNetwork.Train();
            
            // Find action with highest Q-value
            int bestAction = 0;
            float bestValue = qValues.Data[0];
            for (int i = 1; i < actionSize; i++)
            {
                if (qValues.Data[i] > bestValue)
                {
                    bestValue = qValues.Data[i];
                    bestAction = i;
                }
            }
            
            return bestAction;
        }

        /// <summary>
        /// Store experience and train the network
        /// </summary>
        public void Update(float[] state, int action, double reward, float[] nextState, bool done)
        {
            // Store experience in replay buffer (save game film)
            replayBuffer.Add(state, action, (float)reward, nextState, done);
            
            // Only train if we have enough samples
            if (replayBuffer.Count < batchSize)
            {
                return;
            }
            
            // Sample batch from replay buffer
            var batch = replayBuffer.Sample(batchSize);
            
            // Prepare batch tensors
            float[] statesData = new float[batchSize * stateSize];
            float[] nextStatesData = new float[batchSize * stateSize];
            float[] rewards = new float[batchSize];
            int[] actions = new int[batchSize];
            bool[] dones = new bool[batchSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                var exp = batch[i];
                Array.Copy(exp.State, 0, statesData, i * stateSize, stateSize);
                Array.Copy(exp.NextState, 0, nextStatesData, i * stateSize, stateSize);
                rewards[i] = (float)exp.Reward;
                actions[i] = exp.Action;
                dones[i] = exp.Done;
            }
            
            var statesTensor = new Tensor(statesData, new int[] { batchSize, stateSize });
            var nextStatesTensor = new Tensor(nextStatesData, new int[] { batchSize, stateSize });
            
            // Get current Q-values (for Double DQN action selection - no gradients needed)
            qNetwork.Eval();
            var currentQValues = qNetwork.Forward(statesTensor);
            qNetwork.Train();
            
            // Get next Q-values from target network (stable reference)
            targetNetwork.Eval();
            var nextQValues = targetNetwork.Forward(nextStatesTensor);
            
            // Compute loss and backpropagate
            optimizer.ZeroGrad();
            
            // Forward pass to get current Q-values (with gradient tracking)
            var predictedQValues = qNetwork.Forward(statesTensor);
            
            // Extract Q-values for taken actions using Gather
            // This ensures loss is computed ONLY on taken actions with correct gradient scaling
            var selectedQValues = GatherFunction.Apply(predictedQValues, actions);
            
            // Build target values for taken actions only
            float[] targetValues = new float[batchSize];
            
            // Update targets for actions taken using Bellman equation
            for (int i = 0; i < batchSize; i++)
            {
                float targetValue;
                if (dones[i])
                {
                    targetValue = rewards[i];
                }
                else
                {
                    if (useDoubleDQN)
                    {
                        // Double DQN: select with Q-network, evaluate with target network
                        int bestAction = 0;
                        float maxCurrentQ = currentQValues.Data[i * actionSize];
                        for (int a = 1; a < actionSize; a++)
                        {
                            if (currentQValues.Data[i * actionSize + a] > maxCurrentQ)
                            {
                                maxCurrentQ = currentQValues.Data[i * actionSize + a];
                                bestAction = a;
                            }
                        }
                        float nextQValue = nextQValues.Data[i * actionSize + bestAction];
                        targetValue = rewards[i] + discountFactor * nextQValue;
                    }
                    else
                    {
                        // Standard DQN: max Q-value from target network
                        float maxNextQ = nextQValues.Data[i * actionSize];
                        for (int a = 1; a < actionSize; a++)
                        {
                            maxNextQ = (float)Math.Max(maxNextQ, nextQValues.Data[i * actionSize + a]);
                        }
                        targetValue = rewards[i] + discountFactor * maxNextQ;
                    }
                }
                
                targetValues[i] = targetValue;
            }
            
            var targetTensor = new Tensor(targetValues, new int[] { batchSize, 1 });
            targetTensor.RequiresGrad = false;  // Targets don't need gradients
            
            var loss = lossFunction.Forward(selectedQValues, targetTensor);
            loss.Backward();
            optimizer.Step();
            
            stepCount++;
            
            // Update target network periodically (update reference playbook)
            if (stepCount % targetUpdateFrequency == 0)
            {
                CopyWeights(qNetwork, targetNetwork);
            }
            
            // Decay epsilon at end of each episode
            if (done)
            {
                epsilon = Math.Max(epsilonMin, epsilon * epsilonDecay);
            }
        }

        /// <summary>
        /// Legacy method - epsilon now decays automatically in Update()
        /// </summary>
        [Obsolete("Epsilon now decays automatically in Update(). This method is kept for compatibility.")]
        public void Train()
        {
            // No-op: epsilon decay moved to Update()
        }

        /// <summary>
        /// Save agent to disk
        /// </summary>
        public void Save(string path)
        {
            var state = new Dictionary<string, object>
            {
                ["q_network"] = qNetwork.StateDict(),
                ["target_network"] = targetNetwork.StateDict(),
                ["epsilon"] = epsilon,
                ["step_count"] = stepCount,
                ["state_size"] = stateSize,
                ["action_size"] = actionSize,
                ["discount_factor"] = discountFactor,
                ["epsilon_decay"] = epsilonDecay,
                ["epsilon_min"] = epsilonMin,
                ["batch_size"] = batchSize,
                ["target_update_frequency"] = targetUpdateFrequency
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
            
            // Load parameters
            epsilon = state["epsilon"].GetSingle();
            stepCount = state["step_count"].GetInt32();
            
            // Load network weights
            var qNetworkState = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, float[]>>(
                state["q_network"].GetRawText());
            var targetNetworkState = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, float[]>>(
                state["target_network"].GetRawText());
            
            if (qNetworkState == null || targetNetworkState == null)
                throw new InvalidOperationException("Failed to deserialize network states");
            
            qNetwork.LoadStateDict(qNetworkState);
            targetNetwork.LoadStateDict(targetNetworkState);
        }

        public float GetEpsilon() => epsilon;
        public int GetBufferSize() => replayBuffer.Count;
        
        // Helper methods for testing/debugging
        public Module GetQNetwork() => qNetwork;
        public bool CanTrain() => replayBuffer.Count >= batchSize;
        public int BufferCount => replayBuffer.Count;
        public float Epsilon => epsilon;
        
        public float[] GetQValues(float[] state)
        {
            var stateTensor = new Tensor(state, new int[] { 1, stateSize });
            qNetwork.Eval();
            var qValues = qNetwork.Forward(stateTensor);
            qNetwork.Train();
            
            float[] result = new float[actionSize];
            Array.Copy(qValues.Data, result, actionSize);
            return result;
        }
        
        public void Remember(float[] state, int action, double reward, float[] nextState, bool done)
        {
            replayBuffer.Add(state, action, (float)reward, nextState, done);
        }
    }
}
