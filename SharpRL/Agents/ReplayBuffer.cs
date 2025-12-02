using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpRL.Core
{
    /// <summary>
    /// Experience tuple - like game film of a single play
    /// </summary>
    public class Experience<TState, TAction>
    {
        public TState State { get; set; }
        public TAction Action { get; set; }
        public double Reward { get; set; }
        public TState NextState { get; set; }
        public bool Done { get; set; }
        
        public Experience(TState state, TAction action, double reward, TState nextState, bool done)
        {
            State = state;
            Action = action;
            Reward = reward;
            NextState = nextState;
            Done = done;
        }
    }
    
    /// <summary>
    /// Experience Replay Buffer - like a film room with game footage
    /// Stores past plays to learn from
    /// </summary>
    public class ReplayBuffer<TState, TAction>
    {
        private readonly Queue<Experience<TState, TAction>> buffer;
        private readonly int maxSize;
        private readonly Random random;
        
        public int Count => buffer.Count;
        public bool IsFull => buffer.Count >= maxSize;
        
        public ReplayBuffer(int maxSize, int? seed = null)
        {
            this.maxSize = maxSize;
            buffer = new Queue<Experience<TState, TAction>>(maxSize);
            random = seed.HasValue ? new Random(seed.Value) : new Random();
        }
        
        /// <summary>
        /// Add experience to buffer (save the game film)
        /// </summary>
        public void Add(Experience<TState, TAction> experience)
        {
            if (buffer.Count >= maxSize)
            {
                buffer.Dequeue(); // Remove oldest
            }
            buffer.Enqueue(experience);
        }
        
        /// <summary>
        /// Add experience using individual components
        /// </summary>
        public void Add(TState state, TAction action, double reward, TState nextState, bool done)
        {
            Add(new Experience<TState, TAction>(state, action, reward, nextState, done));
        }
        
        /// <summary>
        /// Sample batch of experiences (pull random game film for study)
        /// </summary>
        public List<Experience<TState, TAction>> Sample(int batchSize)
        {
            if (buffer.Count < batchSize)
            {
                throw new InvalidOperationException($"Not enough experiences. Have {buffer.Count}, need {batchSize}");
            }
            
            var bufferList = buffer.ToList();
            var batch = new List<Experience<TState, TAction>>(batchSize);
            
            for (int i = 0; i < batchSize; i++)
            {
                int idx = random.Next(bufferList.Count);
                batch.Add(bufferList[idx]);
            }
            
            return batch;
        }
        
        /// <summary>
        /// Clear all experiences
        /// </summary>
        public void Clear()
        {
            buffer.Clear();
        }
    }
}