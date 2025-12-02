using System;
using System.Linq;

namespace SharpRL.Core.ReplayBuffers
{
    /// <summary>
    /// Prioritized Experience Replay Buffer
    /// 
    /// NFL ANALOGY:
    /// Regular replay buffer = watching random game film
    /// Prioritized replay = focusing on games where you made the biggest mistakes
    /// 
    /// The idea: Learn more from surprising/unexpected outcomes (high TD-error).
    /// 
    /// MATH BREAKDOWN:
    /// 
    /// Priority Calculation:
    /// p_i = (|TD-error| + ε)^α
    /// 
    /// Where:
    /// - |TD-error|: How wrong we were (surprise factor)
    /// - ε: Small constant (0.01) to ensure non-zero priority
    /// - α: Exponent (0.6) controlling how much to prioritize
    ///      α=0: uniform sampling (no prioritization)
    ///      α=1: full prioritization (greedy)
    /// 
    /// Sampling Probability:
    /// P(i) = p_i^α / Σ(p_j^α)
    /// 
    /// Importance Sampling Weight:
    /// w_i = (1/N * 1/P(i))^β
    /// 
    /// Where:
    /// - N: Buffer size
    /// - β: Bias correction (0→1 over training)
    ///      β=0: no correction (biased updates)
    ///      β=1: full correction (unbiased)
    /// 
    /// WHY IMPORTANCE SAMPLING?
    /// Prioritized sampling changes the distribution, creating bias.
    /// We correct this by weighting updates: high-priority samples get lower weights.
    /// 
    /// PERFORMANCE GAINS:
    /// - 2-3x faster learning on Atari games
    /// - More sample efficient (learn from mistakes)
    /// - Better final performance
    /// </summary>
    public class PrioritizedReplayBuffer
    {
        private readonly SumTree sumTree;
        private readonly float[][] states;
        private readonly float[][] actions;
        private readonly float[] rewards;
        private readonly float[][] nextStates;
        private readonly bool[] dones;
        
        private readonly int capacity;
        private readonly float alpha;      // Priority exponent
        private readonly float beta;       // Importance sampling weight
        private readonly float betaIncrement;
        private readonly float epsilon;    // Small constant for numerical stability
        private float currentBeta;
        
        private readonly Random random;

        /// <summary>
        /// Create a new prioritized replay buffer
        /// </summary>
        /// <param name="capacity">Maximum number of transitions to store</param>
        /// <param name="alpha">Priority exponent (0=uniform, 1=full prioritization). Default: 0.6</param>
        /// <param name="beta">Initial importance sampling weight (0=no correction, 1=full). Default: 0.4</param>
        /// <param name="betaIncrement">How much to increase beta per sample. Default: 0.001</param>
        /// <param name="epsilon">Small constant added to priorities. Default: 0.01</param>
        /// <param name="seed">Random seed</param>
        public PrioritizedReplayBuffer(
            int capacity,
            float alpha = 0.6f,
            float beta = 0.4f,
            float betaIncrement = 0.001f,
            float epsilon = 0.01f,
            int? seed = null)
        {
            this.capacity = capacity;
            this.alpha = alpha;
            this.beta = beta;
            this.betaIncrement = betaIncrement;
            this.epsilon = epsilon;
            this.currentBeta = beta;
            
            sumTree = new SumTree(capacity);
            
            states = new float[capacity][];
            actions = new float[capacity][];
            rewards = new float[capacity];
            nextStates = new float[capacity][];
            dones = new bool[capacity];
            
            random = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        /// <summary>
        /// Current number of transitions in the buffer
        /// </summary>
        public int Size => sumTree.Count;

        /// <summary>
        /// Add a transition with maximum priority (will be sampled soon)
        /// New experiences are important until we learn their true TD-error
        /// </summary>
        public void Add(float[] state, float[] action, float reward, float[] nextState, bool done)
        {
            // Get max priority or use 1.0 if buffer is empty
            float maxPriority = sumTree.Count > 0 
                ? GetMaxPriority() 
                : 1.0f;
            
            int index = sumTree.Add(maxPriority);
            
            states[index] = state;
            actions[index] = action;
            rewards[index] = reward;
            nextStates[index] = nextState;
            dones[index] = done;
        }

        /// <summary>
        /// Sample a batch of transitions based on priorities
        /// Returns: (states, actions, rewards, nextStates, dones, indices, weights)
        /// </summary>
        public (float[][] states, float[][] actions, float[] rewards, float[][] nextStates, bool[] dones, int[] indices, float[] weights) 
            Sample(int batchSize)
        {
            if (Size < batchSize)
                throw new InvalidOperationException($"Not enough samples. Need {batchSize}, have {Size}");

            var batchStates = new float[batchSize][];
            var batchActions = new float[batchSize][];
            var batchRewards = new float[batchSize];
            var batchNextStates = new float[batchSize][];
            var batchDones = new bool[batchSize];
            var batchIndices = new int[batchSize];
            var batchWeights = new float[batchSize];

            // Divide priority range into segments
            float prioritySegment = sumTree.Total / batchSize;
            
            // Calculate min probability for normalization
            float minProbability = GetMinPriority() / sumTree.Total;
            float maxWeight = (float)Math.Pow(Size * minProbability, -currentBeta);

            for (int i = 0; i < batchSize; i++)
            {
                // Sample from segment i
                float a = prioritySegment * i;
                float b = prioritySegment * (i + 1);
                float value = a + (float)random.NextDouble() * (b - a);
                
                int index = sumTree.Sample(value);
                float priority = sumTree.GetPriority(index);
                
                // Calculate sampling probability
                float probability = priority / sumTree.Total;
                
                // Calculate importance sampling weight
                float weight = (float)Math.Pow(Size * probability, -currentBeta);
                weight /= maxWeight;  // Normalize by max weight
                
                batchStates[i] = states[index];
                batchActions[i] = actions[index];
                batchRewards[i] = rewards[index];
                batchNextStates[i] = nextStates[index];
                batchDones[i] = dones[index];
                batchIndices[i] = index;
                batchWeights[i] = weight;
            }

            // Anneal beta towards 1.0 (full correction)
            currentBeta = Math.Min(1.0f, currentBeta + betaIncrement);

            return (batchStates, batchActions, batchRewards, batchNextStates, batchDones, batchIndices, batchWeights);
        }

        /// <summary>
        /// Update priorities based on TD-errors
        /// Call this after computing loss/TD-error for sampled batch
        /// </summary>
        /// <param name="indices">Indices from Sample()</param>
        /// <param name="tdErrors">TD-errors for each transition</param>
        public void UpdatePriorities(int[] indices, float[] tdErrors)
        {
            if (indices.Length != tdErrors.Length)
                throw new ArgumentException("Indices and TD-errors must have same length");

            for (int i = 0; i < indices.Length; i++)
            {
                // Priority = (|TD-error| + ε)^α
                float priority = (float)Math.Pow(Math.Abs(tdErrors[i]) + epsilon, alpha);
                sumTree.UpdatePriority(indices[i], priority);
            }
        }

        private float GetMaxPriority()
        {
            float maxPriority = 0;
            for (int i = 0; i < sumTree.Count; i++)
            {
                maxPriority = Math.Max(maxPriority, sumTree.GetPriority(i));
            }
            return maxPriority;
        }

        private float GetMinPriority()
        {
            float minPriority = float.MaxValue;
            for (int i = 0; i < sumTree.Count; i++)
            {
                minPriority = Math.Min(minPriority, sumTree.GetPriority(i));
            }
            return minPriority;
        }

        /// <summary>
        /// Get statistics about the buffer priorities (for monitoring)
        /// </summary>
        public (float min, float max, float avg, float currentBeta) GetStats()
        {
            if (Size == 0)
                return (0, 0, 0, currentBeta);

            float min = GetMinPriority();
            float max = GetMaxPriority();
            float avg = sumTree.Total / Size;

            return (min, max, avg, currentBeta);
        }
    }
}
