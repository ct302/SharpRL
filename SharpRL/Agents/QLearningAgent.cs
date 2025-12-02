using System;
using System.Collections.Generic;
using System.Linq;
using SharpRL.Core;

namespace SharpRL.Agents
{
    /// <summary>
    /// Q-Learning Agent - learns optimal action values through experience
    /// Like a QB learning which plays work best in different situations
    /// </summary>
    public class QLearningAgent<TState, TAction> : IAgent<TState, TAction>
        where TState : IEquatable<TState>
        where TAction : IEquatable<TAction>
    {
        private readonly Dictionary<(TState, TAction), double> qTable;
        private readonly double learningRate;      // α - how quickly we update (rookie vs veteran learning speed)
        private readonly double discountFactor;     // γ - how much we value future rewards (planning ahead)
        private double epsilon;                     // ε - exploration rate (trying new plays vs sticking to playbook)
        private readonly double epsilonDecay;       // How quickly we reduce exploration
        private readonly double epsilonMin;         // Minimum exploration rate
        private readonly Random random;
        private readonly Func<TState, IEnumerable<TAction>> getActions;
        
        public QLearningAgent(
            double learningRate = 0.1,
            double discountFactor = 0.99,
            double epsilon = 1.0,
            double epsilonDecay = 0.995,
            double epsilonMin = 0.01,
            Func<TState, IEnumerable<TAction>> getActionsFunc = null!,
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
        
        /// <summary>
        /// Select action using epsilon-greedy policy
        /// Like deciding whether to call the safe play or try something risky
        /// </summary>
        public TAction SelectAction(TState state, bool explore = true)
        {
            var actions = getActions?.Invoke(state)?.ToList() ?? GetKnownActions(state).ToList();
            
            if (!actions.Any())
            {
                throw new InvalidOperationException("No actions available for current state");
            }
            
            // Exploration vs Exploitation (trying new plays vs using proven ones)
            if (explore && random.NextDouble() < epsilon)
            {
                // Explore - pick random action (call an unexpected play)
                return actions[random.Next(actions.Count)];
            }
            else
            {
                // Exploit - pick best known action (call the money play)
                return GetBestAction(state, actions);
            }
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
        
        /// <summary>
        /// Q-Learning update rule:
        /// Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        /// 
        /// Like updating your playbook rating:
        /// NewRating = OldRating + LearningSpeed * (ActualResult + FutureValue - OldRating)
        /// </summary>
        public void Update(TState state, TAction action, double reward, TState nextState, bool done)
        {
            double oldQValue = GetQValue(state, action);
            double nextMaxQ = 0;
            
            if (!done)
            {
                // Find best value for next state (best play call from new position)
                var nextActions = getActions?.Invoke(nextState) ?? GetKnownActions(nextState);
                if (nextActions.Any())
                {
                    nextMaxQ = nextActions.Max(a => GetQValue(nextState, a));
                }
            }
            
            // TD Target: immediate reward + discounted future value
            // Like: YardsGained + ExpectedFuturePoints
            double tdTarget = reward + discountFactor * nextMaxQ;
            
            // TD Error: difference between expected and actual
            // Like: ActualEPA - ExpectedEPA
            double tdError = tdTarget - oldQValue;
            
            // Update Q-value
            double newQValue = oldQValue + learningRate * tdError;
            SetQValue(state, action, newQValue);
        }
        
        public void Train()
        {
            // Decay epsilon after training step
            epsilon = Math.Max(epsilonMin, epsilon * epsilonDecay);
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
        
        public void Save(string path)
        {
            // Create save data with serializable format
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
            
            if (saveData == null)
                throw new InvalidOperationException("Failed to deserialize agent state");
            
            // Load hyperparameters
            epsilon = saveData["Epsilon"].GetDouble();
            
            // Note: For a complete implementation, you'd need a way to deserialize states and actions
            // This would typically require a custom converter or factory method
            // For now, this shows the structure - actual implementation depends on concrete types
        }
        
        public double GetEpsilon() => epsilon;
        public int GetQTableSize() => qTable.Count;
    }
}