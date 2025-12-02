using System;
using System.Collections.Generic;

namespace SharpRL.Core
{
    /// <summary>
    /// Core environment interface - like the NFL field where all action happens
    /// </summary>
    public interface IEnvironment<TState, TAction>
    {
        /// <summary>
        /// Current state of the environment (field position, game situation)
        /// </summary>
        TState CurrentState { get; }
        
        /// <summary>
        /// Whether the episode is complete (game over, touchdown, turnover)
        /// </summary>
        bool IsDone { get; }
        
        /// <summary>
        /// Reset environment to initial state (new game/drive)
        /// </summary>
        TState Reset();
        
        /// <summary>
        /// Execute action and return (newState, reward, done, info)
        /// Like calling a play and seeing the result
        /// </summary>
        StepResult<TState> Step(TAction action);
        
        /// <summary>
        /// Get all valid actions from current state (available plays in playbook)
        /// </summary>
        IEnumerable<TAction> GetValidActions();
        
        /// <summary>
        /// Render/visualize the current state
        /// </summary>
        void Render();
        
        /// <summary>
        /// Get observation space dimensions
        /// </summary>
        int[] ObservationShape { get; }
        
        /// <summary>
        /// Get action space dimensions
        /// </summary>
        int ActionSpaceSize { get; }
    }
    
    /// <summary>
    /// Result from environment step - like the outcome of a play
    /// </summary>
    public class StepResult<TState>
    {
        public TState NextState { get; set; }
        public float Reward { get; set; }
        public bool Done { get; set; }
        public Dictionary<string, object> Info { get; set; }
        
        public StepResult(TState nextState, float reward, bool done, Dictionary<string, object> info = null!)
        {
            NextState = nextState;
            Reward = reward;
            Done = done;
            Info = info ?? new Dictionary<string, object>();
        }
        
        /// <summary>
        /// Deconstruct method to enable tuple-style deconstruction
        /// Example: var (nextState, reward, done) = env.Step(action);
        /// </summary>
        public void Deconstruct(out TState nextState, out float reward, out bool done)
        {
            nextState = NextState;
            reward = Reward;
            done = Done;
        }
    }
}