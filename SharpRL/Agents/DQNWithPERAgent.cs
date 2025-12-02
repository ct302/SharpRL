using System;
using System.Linq;
using SharpRL.AutoGrad;
using SharpRL.NN;
using SharpRL.NN.Layers;
using SharpRL.NN.Optimizers;
using SharpRL.NN.Loss;
using SharpRL.Core.ReplayBuffers;

namespace SharpRL.Agents
{
    /// <summary>
    /// DQN with Prioritized Experience Replay (PER)
    /// 
    /// NFL ANALOGY:
    /// Regular DQN = studying random game film
    /// DQN with PER = focusing on games where you made the biggest mistakes
    /// 
    /// The scout who learns from his biggest draft busts will improve faster!
    /// 
    /// KEY INNOVATION:
    /// Instead of sampling uniformly from replay buffer, we sample based on TD-error.
    /// Transitions where we were most "surprised" (high TD-error) are sampled more often.
    /// 
    /// PERFORMANCE GAINS:
    /// - 2-3x faster learning
    /// - Better sample efficiency  
    /// - Higher final performance
    /// - More stable training
    /// 
    /// TECHNICAL DETAILS:
    /// 1. Store transitions with priority = |TD-error| + ε
    /// 2. Sample with probability ∝ priority^α
    /// 3. Correct bias with importance sampling weights
    /// 4. Update priorities after each training step
    /// 
    /// COMPARED TO REGULAR DQN:
    /// Same algorithm, just smarter experience replay!
    /// </summary>
    public class DQNWithPERAgent
    {
        private readonly Module qNetwork;
        private readonly Module targetNetwork;
        private readonly PrioritizedReplayBuffer replayBuffer;
        private readonly Optimizer optimizer;
        
        private readonly int stateDim;
        private readonly int actionCount;
        private readonly float gamma;
        private readonly float tau;
        private readonly float epsilon;
        private readonly float epsilonDecay;
        private readonly float epsilonMin;
        private float currentEpsilon;
        
        private readonly Random random;
        private int updateCounter;
        private readonly int targetUpdateFreq;

        /// <summary>
        /// Create a DQN agent with Prioritized Experience Replay
        /// </summary>
        public DQNWithPERAgent(
            int stateDim,
            int actionCount,
            int[] hiddenLayers = null!,
            int bufferSize = 100000,
            float learningRate = 1e-3f,
            float gamma = 0.99f,
            float tau = 0.005f,
            float epsilon = 1.0f,
            float epsilonDecay = 0.995f,
            float epsilonMin = 0.01f,
            int targetUpdateFreq = 4,
            float alpha = 0.6f,
            float beta = 0.4f,
            float betaIncrement = 0.001f,
            int? seed = null)
        {
            this.stateDim = stateDim;
            this.actionCount = actionCount;
            this.gamma = gamma;
            this.tau = tau;
            this.epsilon = epsilon;
            this.epsilonDecay = epsilonDecay;
            this.epsilonMin = epsilonMin;
            this.currentEpsilon = epsilon;
            this.targetUpdateFreq = targetUpdateFreq;
            this.updateCounter = 0;
            this.random = seed.HasValue ? new Random(seed.Value) : new Random();

            hiddenLayers = hiddenLayers ?? new[] { 128, 128 };

            qNetwork = BuildNetwork(stateDim, actionCount, hiddenLayers);
            targetNetwork = BuildNetwork(stateDim, actionCount, hiddenLayers);
            CopyWeights(qNetwork, targetNetwork);

            optimizer = new Adam(qNetwork.Parameters(), learningRate);
            replayBuffer = new PrioritizedReplayBuffer(
                bufferSize, 
                alpha, 
                beta, 
                betaIncrement,
                seed: seed
            );
        }

        private Module BuildNetwork(int inputSize, int outputSize, int[] hiddenSizes)
        {
            var layers = new System.Collections.Generic.List<Module>();
            
            int prevSize = inputSize;
            foreach (int hiddenSize in hiddenSizes)
            {
                layers.Add(new Linear(prevSize, hiddenSize));
                layers.Add(new ReLU());
                prevSize = hiddenSize;
            }
            
            layers.Add(new Linear(prevSize, outputSize));
            return new Sequential(layers.ToArray());
        }

        public int SelectAction(float[] state, bool explore = true)
        {
            if (explore && random.NextDouble() < currentEpsilon)
                return random.Next(actionCount);

            var stateTensor = new Tensor(state, new[] { 1, stateDim });
            var qValues = qNetwork.Forward(stateTensor);
            
            int bestAction = 0;
            float maxQ = qValues.Data[0];
            
            for (int a = 1; a < actionCount; a++)
            {
                if (qValues.Data[a] > maxQ)
                {
                    maxQ = qValues.Data[a];
                    bestAction = a;
                }
            }
            
            return bestAction;
        }

        public void Store(float[] state, int action, float reward, float[] nextState, bool done)
        {
            // Convert action to one-hot
            var actionOneHot = new float[actionCount];
            actionOneHot[action] = 1.0f;
            
            replayBuffer.Add(state, actionOneHot, reward, nextState, done);
        }

        /// <summary>
        /// Train on a batch using prioritized sampling
        /// Key difference: We use importance sampling weights and update priorities
        /// </summary>
        public void Train(int batchSize = 32)
        {
            if (replayBuffer.Size < batchSize)
                return;

            // Sample batch with priorities
            var (states, actions, rewards, nextStates, dones, indices, weights) = 
                replayBuffer.Sample(batchSize);

            // Convert to tensors
            var statesTensor = new Tensor(
                states.SelectMany(s => s).ToArray(), new[] { batchSize, stateDim });
            var nextStatesTensor = new Tensor(
                nextStates.SelectMany(s => s).ToArray(), new[] { batchSize, stateDim });
            var rewardsTensor = new Tensor(rewards, new[] { batchSize, 1 });
            var donesTensor = new Tensor(
                dones.Select(d => d ? 1f : 0f).ToArray(), new[] { batchSize, 1 });
            var weightsTensor = new Tensor(weights, new[] { batchSize, 1 });

            // Compute current Q-values
            var currentQ = qNetwork.Forward(statesTensor);
            
            // Get Q-values for taken actions
            var currentQValues = new float[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                for (int a = 0; a < actionCount; a++)
                {
                    if (actions[i][a] > 0.5f)  // one-hot encoded
                    {
                        currentQValues[i] = currentQ.Data[i * actionCount + a];
                        break;
                    }
                }
            }
            var currentQTensor = new Tensor(currentQValues, new[] { batchSize, 1 });

            // Compute target Q-values
            var nextQ = targetNetwork.Forward(nextStatesTensor);
            var maxNextQ = new float[batchSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                float maxQ = nextQ.Data[i * actionCount];
                for (int a = 1; a < actionCount; a++)
                {
                    maxQ = Math.Max(maxQ, nextQ.Data[i * actionCount + a]);
                }
                maxNextQ[i] = maxQ;
            }
            var maxNextQTensor = new Tensor(maxNextQ, new[] { batchSize, 1 });

            // Compute targets: r + γ * max Q(s',a') * (1 - done)
            var targets = new float[batchSize];
            var tdErrors = new float[batchSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                targets[i] = rewards[i] + gamma * maxNextQ[i] * (1 - donesTensor.Data[i]);
                tdErrors[i] = Math.Abs(targets[i] - currentQValues[i]);
            }
            var targetsTensor = new Tensor(targets, new[] { batchSize, 1 });

            // Compute weighted MSE loss
            // Loss = mean(weight * (predicted - target)^2)
            var diff = new float[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                diff[i] = currentQValues[i] - targets[i];
                diff[i] = weights[i] * diff[i] * diff[i];
            }
            
            float loss = diff.Average();
            var lossTensor = new Tensor(new[] { loss }, new[] { 1 });

            // Backpropagation
            optimizer.ZeroGrad();
            lossTensor.Backward();
            optimizer.Step();

            // 🔥 KEY DIFFERENCE: Update priorities based on TD-errors
            replayBuffer.UpdatePriorities(indices, tdErrors);

            // Update target network
            updateCounter++;
            if (updateCounter % targetUpdateFreq == 0)
            {
                SoftUpdateTargetNetwork();
            }

            // Decay epsilon
            currentEpsilon = Math.Max(epsilonMin, currentEpsilon * epsilonDecay);
        }

        private void CopyWeights(Module source, Module target)
        {
            var sourceParams = source.Parameters();
            var targetParams = target.Parameters();
            
            for (int i = 0; i < sourceParams.Count; i++)
            {
                Array.Copy(sourceParams[i].Data, targetParams[i].Data, sourceParams[i].Data.Length);
            }
        }

        private void SoftUpdateTargetNetwork()
        {
            var mainParams = qNetwork.Parameters();
            var targetParams = targetNetwork.Parameters();
            
            for (int i = 0; i < mainParams.Count; i++)
            {
                for (int j = 0; j < mainParams[i].Data.Length; j++)
                {
                    targetParams[i].Data[j] = tau * mainParams[i].Data[j] + (1 - tau) * targetParams[i].Data[j];
                }
            }
        }

        public float GetEpsilon() => currentEpsilon;

        /// <summary>
        /// Get priority buffer statistics for monitoring
        /// </summary>
        public (float minPriority, float maxPriority, float avgPriority, float beta) GetPriorityStats()
        {
            return replayBuffer.GetStats();
        }
    }
}
