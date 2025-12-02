using System;
using SharpRL.Agents;
using SharpRL.Environments;

namespace SharpRL.Examples
{
    /// <summary>
    /// Demonstrates continuous control with PPO on the Pendulum environment
    /// 
    /// NFL ANALOGY:
    /// Training a QB to maintain perfect throwing mechanics through continuous micro-adjustments.
    /// The agent learns smooth, precise control rather than binary "good/bad" decisions.
    /// 
    /// This example shows:
    /// 1. Creating a continuous environment (Pendulum)
    /// 2. Creating a continuous agent (PPO with Gaussian policy)
    /// 3. Training with experience collection and batch updates
    /// 4. Testing the trained agent
    /// </summary>
    public class ContinuousPPOExample
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("=== Continuous PPO Training Example ===");
            Console.WriteLine("Environment: Pendulum-v0 (continuous control)");
            Console.WriteLine();

            // Create environment
            var env = new PendulumEnvironment(seed: 42);
            Console.WriteLine($"State Size: {env.StateSize}");
            Console.WriteLine($"Action Size: {env.ActionSize} (continuous)");
            Console.WriteLine();

            // Create agent
            var agent = new ContinuousPPOAgent(
                stateDim: env.StateSize,
                actionDim: env.ActionSize,
                actionScale: 2.0f,  // Max torque = 2.0
                learningRate: 3e-4f,
                gamma: 0.99f,
                lambda: 0.95f,
                clipEpsilon: 0.2f,
                entropyCoef: 0.01f,
                trainEpochs: 10,
                batchSize: 64
            );

            Console.WriteLine("Training agent...");
            Console.WriteLine();

            // Training parameters
            int numEpisodes = 100;
            int maxSteps = 200;
            int updateFrequency = 10;  // Update every 10 episodes

            // Storage for training batch
            var states = new System.Collections.Generic.List<float[]>();
            var actions = new System.Collections.Generic.List<float[]>();
            var rewards = new System.Collections.Generic.List<float>();
            var nextStates = new System.Collections.Generic.List<float[]>();
            var dones = new System.Collections.Generic.List<bool>();

            for (int episode = 0; episode < numEpisodes; episode++)
            {
                var state = env.Reset();
                float episodeReward = 0f;

                for (int step = 0; step < maxSteps; step++)
                {
                    // Select action with exploration
                    var action = agent.SelectAction(state, deterministic: false);

                    // Take step in environment
                    var (nextState, reward, done) = env.Step(action);

                    // Store experience
                    states.Add(state);
                    actions.Add(action);
                    rewards.Add(reward);
                    nextStates.Add(nextState);
                    dones.Add(done);

                    episodeReward += reward;
                    state = nextState;

                    if (done) break;
                }

                // Train agent periodically
                if ((episode + 1) % updateFrequency == 0 && states.Count > 0)
                {
                    agent.Train(
                        states.ToArray(),
                        actions.ToArray(),
                        rewards.ToArray(),
                        nextStates.ToArray(),
                        dones.ToArray()
                    );

                    // Clear buffers
                    states.Clear();
                    actions.Clear();
                    rewards.Clear();
                    nextStates.Clear();
                    dones.Clear();
                }

                if ((episode + 1) % 10 == 0)
                {
                    Console.WriteLine($"Episode {episode + 1}/{numEpisodes}, Reward: {episodeReward:F2}");
                }
            }

            Console.WriteLine();
            Console.WriteLine("Training complete!");
            Console.WriteLine();

            // Test trained agent
            Console.WriteLine("Testing trained agent (deterministic policy)...");
            Console.WriteLine();

            int numTestEpisodes = 5;
            for (int episode = 0; episode < numTestEpisodes; episode++)
            {
                var state = env.Reset();
                float episodeReward = 0f;

                Console.WriteLine($"Test Episode {episode + 1}:");

                for (int step = 0; step < maxSteps; step++)
                {
                    // Select action deterministically (no exploration)
                    var action = agent.SelectAction(state, deterministic: true);

                    // Take step
                    var (nextState, reward, done) = env.Step(action);
                    
                    episodeReward += reward;
                    state = nextState;

                    if (step % 50 == 0)
                    {
                        env.Render();
                    }

                    if (done) break;
                }

                Console.WriteLine($"  Total Reward: {episodeReward:F2}");
                Console.WriteLine();
            }

            Console.WriteLine("Example complete!");
            Console.WriteLine();
            Console.WriteLine("NFL Summary:");
            Console.WriteLine("The agent learned to apply smooth, continuous torque adjustments");
            Console.WriteLine("to keep the pendulum balanced - like a QB maintaining perfect form");
            Console.WriteLine("through micro-adjustments rather than big corrections.");
        }
    }
}
