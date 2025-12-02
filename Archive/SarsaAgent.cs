using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpRL.Agents
{
    /// <summary>
    /// SARSA Agent - State-Action-Reward-State-Action
    /// On-policy TD learning - learns from the actual plays being called
    /// Like a QB who learns from their actual decisions, not hypothetical best plays
    /// </summary>
    public class SarsaAgent<TState, TAction> : IAgent<TState, TAction>
        where TState : IEquatable<TState>
        where TAction : IEquatable<TAction>
    {
        private readonly Dictionary<(TState, TAction), double> qTable;
        private readonly double learningRate;      // α - learning speed
        private readonly double discountFactor;     // γ - future reward importance
        private double epsilon;                     // ε - exploration rate
        private readonly double epsilonDecay;
        private readonly double epsilonMin;
        private readonly Random random;
        private readonly Func<TState, IEnumerable<TAction>> getActions;
        
        // SARSA specific - remember last action for next update
        private TAction lastAction;
        
        public SarsaAgent(
            double learningRate = 0.1,
            double discountFactor = 0.99,
            double epsilon = 1.0,
            double epsilonDecay = 0.995,
            double epsilonMin = 0.01,
            Func<TState, IEnumerable<TAction>> getActionsFunc = null,
            int? seed = null)
        {
            qTable = new Dictionary<(TState, TAction), double>();
            this.learningRate = learningRate;
            this.discountFactor = discountFactor;
            this.epsilon = epsilon;
            this.epsilonDecay = epsilonDecay;
            this.epsilonMin = epsilonMin;
            this.getActions = getActionsFunc;
            random = seed.HasValue ? new Random(seed.Value) : new Random();
        }
        
        public TAction SelectAction(TState state, bool explore = true)
        {
            var actions = getActions?.Invoke(state)?.ToList() ?? GetKnownActions(state).ToList();
            
            if (!actions.Any())
            {
                throw new InvalidOperationException("No actions available for current state");
            }
            
            TAction selectedAction;
            
            // Epsilon-greedy policy
            if (explore && random.NextDouble() < epsilon)
            {
                selectedAction = actions[random.Next(actions.Count)];
            }
            else
            {
                selectedAction = GetBestAction(state, actions);
            }
            
            lastAction = selectedAction;
            return selectedAction;
        }
        
        /// <summary>
        /// SARSA Update: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
        /// 
        /// Key difference from Q-Learning:
        /// - Q-Learning uses max Q(s',a') (off-policy - learns optimal)
        /// - SARSA uses actual Q(s',a') (on-policy - learns from actual behavior)
        /// 
        /// Like learning from the plays you actually call, not the theoretical best
        /// </summary>
        public void Update(TState state, TAction action, double reward, TState nextState, bool done)
        {
            double oldQValue = GetQValue(state, action);
            double nextQValue = 0;
            
            if (!done)
            {
                // SARSA: Use the actual next action that will be taken
                // This makes it more conservative than Q-Learning
                TAction nextAction = SelectAction(nextState, true);
                nextQValue = GetQValue(nextState, nextAction);
            }
            
            // TD Target
            double tdTarget = reward + discountFactor * nextQValue;
            
            // TD Error
            double tdError = tdTarget - oldQValue;
            
            // Update Q-value
            double newQValue = oldQValue + learningRate * tdError;
            SetQValue(state, action, newQValue);
        }
        
        private TAction GetBestAction(TState state, IEnumerable<TAction> actions)
        {
            TAction bestAction = actions.First();
            double bestValue = GetQValue(state, bestAction);
            
            foreach (var action in actions.Skip(1))
            {
                double value = GetQValue(state, action);
                if (value > bestValue)
                {
                    bestValue = value;
                    bestAction = action;
                }
            }
            
            return bestAction;
        }
        
        private double GetQValue(TState state, TAction action)
        {
            return qTable.TryGetValue((state, action), out var value) ? value : 0.0;
        }
        
        private void SetQValue(TState state, TAction action, double value)
        {
            qTable[(state, action)] = value;
        }
        
        private IEnumerable<TAction> GetKnownActions(TState state)
        {
            return qTable.Keys
                .Where(key => key.Item1.Equals(state))
                .Select(key => key.Item2)
                .Distinct();
        }
        
        public void Train()
        {
            epsilon = Math.Max(epsilonMin, epsilon * epsilonDecay);
        }
        
        public void Save(string path)
        {
            // Similar to QLearning agent - save Q-table and parameters
            var saveData = new Dictionary<string, object>
            {
                ["QTable"] = qTable.Select(kvp => new
                {
                    StateHash = kvp.Key.Item1.GetHashCode(),
                    State = kvp.Key.Item1.ToString(),
                    ActionHash = kvp.Key.Item2.GetHashCode(),
                    Action = kvp.Key.Item2.ToString(),
                    Value = kvp.Value
                }).ToList(),
                ["LearningRate"] = learningRate,
                ["DiscountFactor"] = discountFactor,
                ["Epsilon"] = epsilon,
                ["EpsilonDecay"] = epsilonDecay,
                ["EpsilonMin"] = epsilonMin
            };
            
            string json = System.Text.Json.JsonSerializer.Serialize(saveData, new System.Text.Json.JsonSerializerOptions 
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
            var saveData = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, System.Text.Json.JsonElement>>(json);
            
            // Load hyperparameters
            epsilon = saveData["Epsilon"].GetDouble();
            
            // Note: Full implementation would restore Q-table
        }
        
        public double GetEpsilon() => epsilon;
        public int GetQTableSize() => qTable.Count;
    }
}