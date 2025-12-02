using System;
using System.Collections.Generic;
using System.Linq;
using SharpRL.Core;

namespace SharpRL.Agents
{
    /// <summary>
    /// Q-Learning agent with context awareness for dynamic decision-making strategies.
    /// 
    /// NFL ANALOGY:
    /// Like having separate playbooks for different game situations:
    /// - Normal offense playbook (moving the ball efficiently)
    /// - Two-minute drill playbook (urgency, clock management)
    /// - Goal-line playbook (power runs, play-action)
    /// 
    /// The agent learns different Q-values for the SAME field position based on context,
    /// allowing optimal strategies to emerge for each situation.
    /// 
    /// EXAMPLE USE CASES:
    /// - GridWorld: Safe vs Danger contexts (flee enemies vs reach goal)
    /// - Shopping Agent: Normal vs Holiday Season contexts (standard vs bulk ordering)
    /// - Trading Bot: Bull Market vs Bear Market contexts (aggressive vs defensive)
    /// - Game AI: Winning vs Losing contexts (conservative vs risky plays)
    /// </summary>
    public class ContextAwareQLearningAgent<TState, TAction> : IAgent<TState, TAction>
        where TState : IEquatable<TState>
        where TAction : IEquatable<TAction>
    {
        // The underlying Q-Learning agent using contextual states
        private readonly QLearningAgent<ContextualState<TState>, TAction> baseAgent;
        
        // Function to determine which context applies to a given physical state
        // NFL ANALOGY: Like the coaching staff reading the game situation
        private readonly Func<TState, IContext> contextDetector;
        
        // Heuristic strategies for unexplored states per context
        // NFL ANALOGY: Instinctive plays when you haven't practiced this situation
        // - Danger context → Run away from threat
        // - Normal context → Move toward goal
        private readonly Dictionary<IContext, Func<TState, TAction>> contextHeuristics;
        
        // Action provider function
        private readonly Func<TState, IEnumerable<TAction>> getActions;
        
        // Tracks which contextual states are unexplored (all Q-values are 0)
        private readonly HashSet<ContextualState<TState>> unexploredStates;
        
        public ContextAwareQLearningAgent(
            Func<TState, IContext> contextDetector,
            Func<TState, IEnumerable<TAction>> getActionsFunc,
            Dictionary<IContext, Func<TState, TAction>> heuristics = null!,
            double learningRate = 0.1,
            double discountFactor = 0.99,
            double epsilon = 1.0,
            double epsilonDecay = 0.995,
            double epsilonMin = 0.01,
            int? seed = null)
        {
            this.contextDetector = contextDetector ?? throw new ArgumentNullException(nameof(contextDetector));
            this.getActions = getActionsFunc ?? throw new ArgumentNullException(nameof(getActionsFunc));
            this.contextHeuristics = heuristics ?? new Dictionary<IContext, Func<TState, TAction>>();
            
            unexploredStates = new HashSet<ContextualState<TState>>();
            
            // Create wrapper function for actions that works with contextual states
            Func<ContextualState<TState>, IEnumerable<TAction>> contextualGetActions = 
                (contextualState) => getActionsFunc(contextualState.PhysicalState);
            
            // Initialize the base agent with contextual state wrapper
            baseAgent = new QLearningAgent<ContextualState<TState>, TAction>(
                learningRate: learningRate,
                discountFactor: discountFactor,
                epsilon: epsilon,
                epsilonDecay: epsilonDecay,
                epsilonMin: epsilonMin,
                getActionsFunc: contextualGetActions,
                seed: seed
            );
        }
        
        /// <summary>
        /// Register a heuristic strategy for a specific context.
        /// Used when the agent encounters an unexplored state in this context.
        /// 
        /// NFL ANALOGY:
        /// Teaching the QB instinctive reactions before they've practiced:
        /// - "If blitz is coming and you haven't seen this coverage, throw hot route"
        /// - "If winning by 20 in 4th quarter, run the ball"
        /// </summary>
        public void RegisterHeuristic(IContext context, Func<TState, TAction> heuristic)
        {
            contextHeuristics[context] = heuristic ?? throw new ArgumentNullException(nameof(heuristic));
        }
        
        /// <summary>
        /// Select an action for the given physical state.
        /// Automatically detects context and uses appropriate strategy.
        /// </summary>
        public TAction SelectAction(TState state, bool explore = true)
        {
            // Detect the current context (like reading the game situation)
            IContext currentContext = contextDetector(state);
            
            // Create the composite contextual state
            var contextualState = new ContextualState<TState>(state, currentContext);
            
            // Check if this contextual state is unexplored
            bool isUnexplored = IsStateUnexplored(contextualState);
            
            if (isUnexplored && contextHeuristics.TryGetValue(currentContext, out var heuristic))
            {
                // Use heuristic for unexplored states
                // NFL ANALOGY: Falling back on instinct when you haven't practiced this exact scenario
                return heuristic(state);
            }
            
            // Use learned Q-values (standard exploration/exploitation)
            return baseAgent.SelectAction(contextualState, explore);
        }
        
        /// <summary>
        /// Update Q-values based on experience.
        /// Automatically handles context detection for current and next states.
        /// </summary>
        public void Update(TState state, TAction action, double reward, TState nextState, bool done)
        {
            // Detect contexts for both states
            IContext currentContext = contextDetector(state);
            IContext nextContext = contextDetector(nextState);
            
            // Create contextual states
            var contextualState = new ContextualState<TState>(state, currentContext);
            var contextualNextState = new ContextualState<TState>(nextState, nextContext);
            
            // Mark this state as explored (has been updated at least once)
            unexploredStates.Remove(contextualState);
            
            // Delegate to base agent with contextual states
            baseAgent.Update(contextualState, action, reward, contextualNextState, done);
        }
        
        /// <summary>
        /// Decay exploration rate (epsilon)
        /// </summary>
        public void Train()
        {
            baseAgent.Train();
        }
        
        /// <summary>
        /// Check if a contextual state is unexplored (never been updated)
        /// </summary>
        private bool IsStateUnexplored(ContextualState<TState> contextualState)
        {
            // A state is unexplored if it's in our unexplored set OR
            // if the base agent has never seen it (Q-table doesn't contain any actions for it)
            if (unexploredStates.Contains(contextualState))
                return true;
            
            // Check if base agent has any Q-values for this state
            // If Q-table size hasn't changed, it might be unexplored
            int sizeBefore = baseAgent.GetQTableSize();
            
            // Quick heuristic: if we've never updated this state, it's unexplored
            // We track this by adding to unexploredStates when first seen
            // and removing when first updated
            
            // Add to unexplored set if we haven't seen it before
            if (!unexploredStates.Contains(contextualState))
            {
                // Assume it's unexplored on first sight
                unexploredStates.Add(contextualState);
                return true;
            }
            
            return false;
        }
        
        /// <summary>
        /// Save agent state to file
        /// </summary>
        public void Save(string path)
        {
            baseAgent.Save(path);
            // TODO: Also save context detector and heuristics if needed for full restoration
        }
        
        /// <summary>
        /// Load agent state from file
        /// </summary>
        public void Load(string path)
        {
            baseAgent.Load(path);
            unexploredStates.Clear(); // All loaded states are considered explored
        }
        
        /// <summary>
        /// Get current exploration rate (epsilon)
        /// </summary>
        public double GetEpsilon() => baseAgent.GetEpsilon();
        
        /// <summary>
        /// Get the size of the Q-table (number of state-action pairs)
        /// </summary>
        public int GetQTableSize() => baseAgent.GetQTableSize();
        
        /// <summary>
        /// Get the number of unexplored contextual states
        /// </summary>
        public int GetUnexploredStateCount() => unexploredStates.Count;
        
        /// <summary>
        /// Manually mark a state as explored (useful for pre-training or transfer learning)
        /// </summary>
        public void MarkStateAsExplored(TState state, IContext context)
        {
            var contextualState = new ContextualState<TState>(state, context);
            unexploredStates.Remove(contextualState);
        }
        
        /// <summary>
        /// Manually mark a state as unexplored (useful for curriculum learning)
        /// </summary>
        public void MarkStateAsUnexplored(TState state, IContext context)
        {
            var contextualState = new ContextualState<TState>(state, context);
            unexploredStates.Add(contextualState);
        }
        
        /// <summary>
        /// Get all registered contexts that have heuristics
        /// </summary>
        public IEnumerable<IContext> GetContextsWithHeuristics() => contextHeuristics.Keys;
    }
}
