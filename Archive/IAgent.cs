using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpRL.Core
{
    /// <summary>
    /// Base agent interface - the player/coach making decisions
    /// </summary>
    public interface IAgent<TState, TAction>
    {
        /// <summary>
        /// Select action based on current state (call the play)
        /// </summary>
        TAction SelectAction(TState state, bool explore = true);
        
        /// <summary>
        /// Update agent after observing transition (learn from the play result)
        /// </summary>
        void Update(TState state, TAction action, double reward, TState nextState, bool done);
        
        /// <summary>
        /// Train the agent on a batch of experiences
        /// </summary>
        void Train();
        
        /// <summary>
        /// Save agent to disk
        /// </summary>
        void Save(string path);
        
        /// <summary>
        /// Load agent from disk
        /// </summary>
        void Load(string path);
    }
    
    /// <summary>
    /// Policy interface - the playbook/strategy
    /// </summary>
    public interface IPolicy<TState, TAction>
    {
        /// <summary>
        /// Get action probabilities for a state (play selection probabilities)
        /// </summary>
        Dictionary<TAction, double> GetActionProbabilities(TState state);
        
        /// <summary>
        /// Sample action from policy (pick a play from the playbook)
        /// </summary>
        TAction SampleAction(TState state);
        
        /// <summary>
        /// Get best action (greedy) - the "money play"
        /// </summary>
        TAction GetBestAction(TState state);
    }
    
    /// <summary>
    /// Value function interface - evaluates how good a state/action is
    /// Like QB rating or Expected Points Added (EPA)
    /// </summary>
    public interface IValueFunction<TState, TAction>
    {
        /// <summary>
        /// Get state value V(s) - how good is this field position?
        /// </summary>
        double GetStateValue(TState state);
        
        /// <summary>
        /// Get action value Q(s,a) - how good is this play call in this situation?
        /// </summary>
        double GetActionValue(TState state, TAction action);
        
        /// <summary>
        /// Update value estimates
        /// </summary>
        void Update(TState state, TAction action, double target);
    }
}