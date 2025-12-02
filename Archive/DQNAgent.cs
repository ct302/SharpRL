using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpRL.Agents
{
    /// <summary>
    /// Deep Q-Network Agent - uses neural networks to approximate Q-values
    /// Like having an AI offensive coordinator that learns complex patterns
    /// </summary>
    public class DQNAgent : IAgent<double[], int>
    {
        private readonly INeuralNetwork qNetwork;        // Main network (current playbook)
        private readonly INeuralNetwork targetNetwork;   // Target network (stable reference playbook)
        private readonly ReplayBuffer<double[], int> replayBuffer;
        private readonly int actionSpace;
        private readonly double learningRate;
        private readonly double discountFactor;
        private double epsilon;
        private readonly double epsilonDecay;
        private readonly double epsilonMin;
        private readonly int batchSize;
        private readonly int targetUpdateFrequency;
        private int stepCount = 0;
        private readonly Random random;
        
        public DQNAgent(
            int stateSize,
            int actionSpace,
            int[] hiddenLayers = null,
            double learningRate = 0.001,
            double discountFactor = 0.99,
            double epsilon = 1.0,
            double epsilonDecay = 0.995,
            double epsilonMin = 0.01,
            int batchSize = 32,
            int bufferSize = 10000,
            int targetUpdateFreq = 100,
            int? seed = null)
        {
            this.actionSpace = actionSpace;
            this.learningRate = learningRate;
            this.discountFactor = discountFactor;
            this.epsilon = epsilon;
            this.epsilonDecay = epsilonDecay;
            this.epsilonMin = epsilonMin;
            this.batchSize = batchSize;
            this.targetUpdateFrequency = targetUpdateFreq;
            
            // Default architecture if not specified
            hiddenLayers = hiddenLayers ?? new[] { 128, 64 };
            
            // Build networks (would integrate with SharpGrad here)
            qNetwork = new SimpleNeuralNetwork(stateSize, actionSpace, hiddenLayers, learningRate);
            targetNetwork = new SimpleNeuralNetwork(stateSize, actionSpace, hiddenLayers, learningRate);
            
            // Initialize target network with same weights
            targetNetwork.CopyWeightsFrom(qNetwork);
            
            replayBuffer = new ReplayBuffer<double[], int>(bufferSize, seed);
            random = seed.HasValue ? new Random(seed.Value) : new Random();
        }
        
        public int SelectAction(double[] state, bool explore = true)
        {
            // Epsilon-greedy exploration
            if (explore && random.NextDouble() < epsilon)
            {
                return random.Next(actionSpace);
            }
            
            // Get Q-values from network
            double[] qValues = qNetwork.Forward(state);
            
            // Return action with highest Q-value
            return Array.IndexOf(qValues, qValues.Max());
        }
        
        public void Update(double[] state, int action, double reward, double[] nextState, bool done)
        {
            // Store experience in replay buffer
            replayBuffer.Add(state, action, reward, nextState, done);
            
            // Only train if we have enough samples
            if (replayBuffer.Count < batchSize)
            {
                return;
            }
            
            // Sample batch from replay buffer
            var batch = replayBuffer.Sample(batchSize);
            
            // Prepare training data
            var states = batch.Select(e => e.State).ToArray();
            var actions = batch.Select(e => e.Action).ToArray();
            var rewards = batch.Select(e => e.Reward).ToArray();
            var nextStates = batch.Select(e => e.NextState).ToArray();
            var dones = batch.Select(e => e.Done).ToArray();
            
            // Compute targets using target network
            double[][] targets = new double[batchSize][];
            
            for (int i = 0; i < batchSize; i++)
            {
                // Get current Q-values
                targets[i] = qNetwork.Forward(states[i]);
                
                if (dones[i])
                {
                    // Terminal state - no future reward
                    targets[i][actions[i]] = rewards[i];
                }
                else
                {
                    // Bellman equation: Q(s,a) = r + γ·max(Q(s',a'))
                    double[] nextQValues = targetNetwork.Forward(nextStates[i]);
                    double maxNextQ = nextQValues.Max();
                    targets[i][actions[i]] = rewards[i] + discountFactor * maxNextQ;
                }
            }
            
            // Train network on batch
            for (int i = 0; i < batchSize; i++)
            {
                qNetwork.Train(states[i], targets[i]);
            }
            
            stepCount++;
            
            // Update target network periodically
            if (stepCount % targetUpdateFrequency == 0)
            {
                targetNetwork.CopyWeightsFrom(qNetwork);
            }
        }
        
        public void Train()
        {
            // Decay epsilon
            epsilon = Math.Max(epsilonMin, epsilon * epsilonDecay);
        }
        
        public void Save(string path)
        {
            // Save the Q-network weights
            qNetwork.Save(path);
            
            // Save agent metadata
            var metaPath = path.Replace(".json", "_meta.json");
            var metadata = new Dictionary<string, object>
            {
                ["ActionSpace"] = actionSpace,
                ["LearningRate"] = learningRate,
                ["DiscountFactor"] = discountFactor,
                ["Epsilon"] = epsilon,
                ["EpsilonDecay"] = epsilonDecay,
                ["EpsilonMin"] = epsilonMin,
                ["BatchSize"] = batchSize,
                ["StepCount"] = stepCount
            };
            
            string json = System.Text.Json.JsonSerializer.Serialize(metadata, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            System.IO.File.WriteAllText(metaPath, json);
        }
        
        public void Load(string path)
        {
            // Load Q-network weights
            qNetwork.Load(path);
            targetNetwork.CopyWeightsFrom(qNetwork);
            
            // Load agent metadata
            var metaPath = path.Replace(".json", "_meta.json");
            if (System.IO.File.Exists(metaPath))
            {
                string json = System.IO.File.ReadAllText(metaPath);
                var metadata = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, System.Text.Json.JsonElement>>(json);
                
                epsilon = metadata["Epsilon"].GetDouble();
                stepCount = metadata["StepCount"].GetInt32();
            }
        }
    }
}