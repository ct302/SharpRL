using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SharpRL.Core;

namespace SharpRL.Training
{
    /// <summary>
    /// Unified training infrastructure for all RL agents
    /// 
    /// NFL ANALOGY:
    /// The Trainer is like the head coach managing the season:
    /// - Episodes = Games in the season
    /// - Checkpoints = Saving the playbook after key wins
    /// - Metrics = Season statistics tracking
    /// - Callbacks = Adjustments based on performance
    /// </summary>
    public class Trainer<TState, TAction>
        where TAction : notnull
    {
        private readonly IAgent<TState, TAction> agent;
        private readonly IEnvironment<TState, TAction> environment;
        private readonly TrainingConfig config;
        private readonly List<ICallback<TState, TAction>> callbacks;
        private readonly MetricsTracker metrics;
        private readonly Stopwatch stopwatch;
        
        public Trainer(
            IAgent<TState, TAction> agent,
            IEnvironment<TState, TAction> environment,
            TrainingConfig config = null!)
        {
            this.agent = agent;
            this.environment = environment;
            this.config = config ?? new TrainingConfig();
            this.callbacks = new List<ICallback<TState, TAction>>();
            this.metrics = new MetricsTracker();
            this.stopwatch = new Stopwatch();
            
            // Add default callbacks
            if (this.config.LoggingEnabled)
            {
                AddCallback(new LoggingCallback<TState, TAction>(this.config.LogInterval));
            }
            
            if (this.config.CheckpointEnabled)
            {
                AddCallback(new CheckpointCallback<TState, TAction>(
                    this.config.CheckpointDir, 
                    this.config.CheckpointInterval));
            }
            
            if (this.config.EarlyStopping)
            {
                AddCallback(new EarlyStoppingCallback<TState, TAction>(
                    this.config.EarlyStoppingPatience,
                    this.config.EarlyStoppingMinDelta));
            }
        }
        
        /// <summary>
        /// Add a callback for training events
        /// </summary>
        public void AddCallback(ICallback<TState, TAction> callback)
        {
            callbacks.Add(callback);
        }
        
        /// <summary>
        /// Main training loop
        /// Like running through a full season
        /// </summary>
        public TrainingResult Train()
        {
            Console.WriteLine("╔═══════════════════════════════════════════════════════════╗");
            Console.WriteLine("║                  SharpRL Training Started 🏈               ║");
            Console.WriteLine("╚═══════════════════════════════════════════════════════════╝\n");
            
            stopwatch.Start();
            
            // Training loop
            for (int episode = 0; episode < config.NumEpisodes; episode++)
            {
                // Reset environment for new episode
                TState state = environment.Reset();
                float episodeReward = 0;
                int steps = 0;
                bool done = false;
                
                // Episode start callbacks
                foreach (var callback in callbacks)
                {
                    callback.OnEpisodeStart(episode, state);
                }
                
                // Episode loop
                while (!done && steps < config.MaxStepsPerEpisode)
                {
                    // Select action
                    TAction action = agent.SelectAction(state, explore: true);
                    
                    // Step callbacks
                    foreach (var callback in callbacks)
                    {
                        callback.OnStep(episode, steps, state, action);
                    }
                    
                    // Execute action
                    var stepResult = environment.Step(action);
                    TState nextState = stepResult.NextState;
                    float reward = (float)stepResult.Reward;
                    done = stepResult.Done;
                    
                    // Update agent
                    agent.Update(state, action, reward, nextState, done);
                    
                    // Track metrics
                    episodeReward += reward;
                    state = nextState;
                    steps++;
                }
                
                // Train agent (decay epsilon, etc.)
                agent.Train();
                
                // Track episode metrics
                metrics.RecordEpisode(episode, episodeReward, steps);
                
                // Episode end callbacks
                foreach (var callback in callbacks)
                {
                    callback.OnEpisodeEnd(episode, episodeReward, steps);
                }
                
                // Check for early stopping
                if (callbacks.OfType<EarlyStoppingCallback<TState, TAction>>().Any(c => c.ShouldStop))
                {
                    Console.WriteLine($"\n🛑 Early stopping triggered at episode {episode}");
                    break;
                }
                
                // Evaluation episodes
                if ((episode + 1) % config.EvalInterval == 0)
                {
                    float evalReward = Evaluate(config.EvalEpisodes);
                    metrics.RecordEvaluation(episode, evalReward);
                    Console.WriteLine($"📊 Evaluation at episode {episode}: Average Reward = {evalReward:F2}");
                }
            }
            
            stopwatch.Stop();
            
            // Training complete
            Console.WriteLine("\n╔═══════════════════════════════════════════════════════════╗");
            Console.WriteLine("║                 Training Complete! 🏆                      ║");
            Console.WriteLine("╚═══════════════════════════════════════════════════════════╝");
            
            // Save final model
            if (!string.IsNullOrEmpty(config.SavePath))
            {
                agent.Save(config.SavePath);
                Console.WriteLine($"✅ Model saved to: {config.SavePath}");
            }
            
            // Generate training report
            var result = new TrainingResult
            {
                TotalEpisodes = metrics.TotalEpisodes,
                TotalSteps = metrics.TotalSteps,
                BestReward = metrics.BestReward,
                AverageReward = metrics.GetAverageReward(100),
                TrainingTime = stopwatch.Elapsed,
                Metrics = metrics
            };
            
            PrintSummary(result);
            
            return result;
        }
        
        /// <summary>
        /// Evaluate agent performance without exploration
        /// Like playoff games where you use your best strategies
        /// </summary>
        public float Evaluate(int numEpisodes = 10)
        {
            float totalReward = 0;
            
            for (int episode = 0; episode < numEpisodes; episode++)
            {
                TState state = environment.Reset();
                float episodeReward = 0;
                int steps = 0;
                bool done = false;
                
                while (!done && steps < config.MaxStepsPerEpisode)
                {
                    // No exploration during evaluation
                    TAction action = agent.SelectAction(state, explore: false);
                    var result = environment.Step(action);
                    
                    episodeReward += (float)result.Reward;
                    state = result.NextState;
                    done = result.Done;
                    steps++;
                }
                
                totalReward += episodeReward;
            }
            
            return totalReward / numEpisodes;
        }
        
        /// <summary>
        /// Print training summary
        /// </summary>
        private void PrintSummary(TrainingResult result)
        {
            Console.WriteLine("\n📈 Training Summary:");
            Console.WriteLine($"  Total Episodes: {result.TotalEpisodes:N0}");
            Console.WriteLine($"  Total Steps: {result.TotalSteps:N0}");
            Console.WriteLine($"  Best Reward: {result.BestReward:F2}");
            Console.WriteLine($"  Avg Reward (last 100): {result.AverageReward:F2}");
            Console.WriteLine($"  Training Time: {result.TrainingTime:hh\\:mm\\:ss}");
            Console.WriteLine($"  Episodes/Second: {result.TotalEpisodes / result.TrainingTime.TotalSeconds:F1}");
        }
    }
    
    /// <summary>
    /// Training configuration
    /// </summary>
    public class TrainingConfig
    {
        public int NumEpisodes { get; set; } = 1000;
        public int MaxStepsPerEpisode { get; set; } = 1000;
        public int EvalInterval { get; set; } = 100;
        public int EvalEpisodes { get; set; } = 10;
        public bool LoggingEnabled { get; set; } = true;
        public int LogInterval { get; set; } = 10;
        public bool CheckpointEnabled { get; set; } = true;
        public string CheckpointDir { get; set; } = "./checkpoints";
        public int CheckpointInterval { get; set; } = 100;
        public bool EarlyStopping { get; set; } = true;
        public int EarlyStoppingPatience { get; set; } = 50;
        public float EarlyStoppingMinDelta { get; set; } = 0.01f;
        public string SavePath { get; set; } = "./models/final_model.json";
    }
    
    /// <summary>
    /// Training result summary
    /// </summary>
    public class TrainingResult
    {
        public int TotalEpisodes { get; set; }
        public int TotalSteps { get; set; }
        public float BestReward { get; set; }
        public float AverageReward { get; set; }
        public TimeSpan TrainingTime { get; set; }
        public MetricsTracker Metrics { get; set; } = null!;
    }
    
    /// <summary>
    /// Metrics tracking for training
    /// Like keeping season statistics
    /// </summary>
    public class MetricsTracker
    {
        private List<float> episodeRewards = new List<float>();
        private List<int> episodeSteps = new List<int>();
        private List<(int episode, float reward)> evaluations = new List<(int, float)>();
        
        public int TotalEpisodes => episodeRewards.Count;
        public int TotalSteps => episodeSteps.Sum();
        public float BestReward => episodeRewards.Any() ? episodeRewards.Max() : 0;
        
        public void RecordEpisode(int episode, float reward, int steps)
        {
            episodeRewards.Add(reward);
            episodeSteps.Add(steps);
        }
        
        public void RecordEvaluation(int episode, float reward)
        {
            evaluations.Add((episode, reward));
        }
        
        public float GetAverageReward(int lastN)
        {
            if (episodeRewards.Count == 0) return 0;
            int count = Math.Min(lastN, episodeRewards.Count);
            return episodeRewards.TakeLast(count).Average();
        }
        
        public float GetRewardStd(int lastN)
        {
            if (episodeRewards.Count < 2) return 0;
            int count = Math.Min(lastN, episodeRewards.Count);
            var recent = episodeRewards.TakeLast(count).ToList();
            float mean = recent.Average();
            return (float)Math.Sqrt(recent.Select(r => (r - mean) * (r - mean)).Average());
        }
        
        public void SaveToCsv(string path)
        {
            using (var writer = new StreamWriter(path))
            {
                writer.WriteLine("Episode,Reward,Steps");
                for (int i = 0; i < episodeRewards.Count; i++)
                {
                    writer.WriteLine($"{i},{episodeRewards[i]},{episodeSteps[i]}");
                }
            }
        }
    }
    
    /// <summary>
    /// Base interface for training callbacks
    /// </summary>
    public interface ICallback<TState, TAction>
        where TAction : notnull
    {
        void OnEpisodeStart(int episode, TState initialState);
        void OnStep(int episode, int step, TState state, TAction action);
        void OnEpisodeEnd(int episode, float totalReward, int steps);
    }
    
    /// <summary>
    /// Logging callback for training progress
    /// </summary>
    public class LoggingCallback<TState, TAction> : ICallback<TState, TAction>
        where TAction : notnull
    {
        private int logInterval;
        private Stopwatch episodeTimer = new Stopwatch();
        
        public LoggingCallback(int logInterval = 10)
        {
            this.logInterval = logInterval;
        }
        
        public void OnEpisodeStart(int episode, TState initialState)
        {
            episodeTimer.Restart();
        }
        
        public void OnStep(int episode, int step, TState state, TAction action)
        {
            // No per-step logging by default
        }
        
        public void OnEpisodeEnd(int episode, float totalReward, int steps)
        {
            episodeTimer.Stop();
            
            if ((episode + 1) % logInterval == 0)
            {
                Console.WriteLine($"Episode {episode + 1} | " +
                                $"Reward: {totalReward:F2} | " +
                                $"Steps: {steps} | " +
                                $"Time: {episodeTimer.ElapsedMilliseconds}ms");
            }
        }
    }
    
    /// <summary>
    /// Checkpoint callback for saving models periodically
    /// </summary>
    public class CheckpointCallback<TState, TAction> : ICallback<TState, TAction>
        where TAction : notnull
    {
        private string checkpointDir;
        private int checkpointInterval;
        private IAgent<TState, TAction>? agent;
        
        public CheckpointCallback(string checkpointDir, int checkpointInterval = 100)
        {
            this.checkpointDir = checkpointDir;
            this.checkpointInterval = checkpointInterval;
            
            // Create checkpoint directory if it doesn't exist
            Directory.CreateDirectory(checkpointDir);
        }
        
        public void SetAgent(IAgent<TState, TAction> agent)
        {
            this.agent = agent;
        }
        
        public void OnEpisodeStart(int episode, TState initialState) { }
        
        public void OnStep(int episode, int step, TState state, TAction action) { }
        
        public void OnEpisodeEnd(int episode, float totalReward, int steps)
        {
            if ((episode + 1) % checkpointInterval == 0 && agent != null)
            {
                string path = Path.Combine(checkpointDir, $"checkpoint_episode_{episode + 1}.json");
                agent.Save(path);
                Console.WriteLine($"💾 Checkpoint saved: {path}");
            }
        }
    }
    
    /// <summary>
    /// Early stopping callback to prevent overfitting
    /// </summary>
    public class EarlyStoppingCallback<TState, TAction> : ICallback<TState, TAction>
        where TAction : notnull
    {
        private int patience;
        private float minDelta;
        private float bestReward = float.NegativeInfinity;
        private int patienceCounter = 0;
        
        public bool ShouldStop { get; private set; } = false;
        
        public EarlyStoppingCallback(int patience = 50, float minDelta = 0.01f)
        {
            this.patience = patience;
            this.minDelta = minDelta;
        }
        
        public void OnEpisodeStart(int episode, TState initialState) { }
        
        public void OnStep(int episode, int step, TState state, TAction action) { }
        
        public void OnEpisodeEnd(int episode, float totalReward, int steps)
        {
            if (totalReward > bestReward + minDelta)
            {
                bestReward = totalReward;
                patienceCounter = 0;
            }
            else
            {
                patienceCounter++;
                if (patienceCounter >= patience)
                {
                    ShouldStop = true;
                }
            }
        }
    }
}
