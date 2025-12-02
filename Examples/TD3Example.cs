using System;
using SharpRL.Agents;
using SharpRL.Environments;

namespace SharpRL.Examples
{
    /// <summary>
    /// TD3 (Twin Delayed DDPG) Example on Pendulum Environment
    /// 
    /// NFL ANALOGY:
    /// TD3 is like having TWO independent scouts evaluate every play and ALWAYS
    /// going with the more conservative estimate. This prevents the overconfidence
    /// that ruined many promising seasons (looking at you, DDPG).
    /// 
    /// Unlike PPO which learns from fresh game film, TD3 learns from a massive
    /// archive of past plays (replay buffer), making it super sample-efficient.
    /// </summary>
    public class TD3Example
    {
        public static void RunTD3PendulumTraining()
        {
            Console.WriteLine("=== TD3 (Twin Delayed DDPG) on Pendulum ===");
            Console.WriteLine("State-of-the-art continuous control with three innovations:");
            Console.WriteLine("1. Twin Critics - Two Q-networks, use minimum (conservative)");
            Console.WriteLine("2. Delayed Policy - Update actor less often than critics");
            Console.WriteLine("3. Target Smoothing - Add noise to target actions");
            Console.WriteLine();

            // Create environment
            var env = new PendulumEnvironment();

            // Create TD3 agent
            var agent = new TD3Agent(
                stateDim: 3,              // [cos(θ), sin(θ), θ_dot]
                actionDim: 1,             // Torque
                hiddenLayers: new[] { 256, 256 },
                actionScale: 2.0f,        // Torque range: [-2, 2]
                bufferSize: 100000,       // Large replay buffer
                learningRate: 3e-4f,      // Standard learning rate
                gamma: 0.99f,             // Strong future discounting
                tau: 0.005f,              // Slow target updates
                policyNoise: 0.2f,        // 0.2 * 2.0 = 0.4 exploration noise
                targetNoise: 0.2f,        // 0.2 * 2.0 = 0.4 target smoothing
                noiseClip: 0.5f,          // Clip to 1.0
                policyDelay: 2,           // Update actor every 2 critic updates
                seed: 42
            );

            // Training parameters
            int numEpisodes = 100;
            int maxSteps = 200;
            int startTraining = 1000;      // Collect data before training
            int trainEvery = 50;            // Train frequency
            int trainsPerStep = 1;          // Trains per timestep when training

            Console.WriteLine($"Training for {numEpisodes} episodes...");
            Console.WriteLine($"Collecting {startTraining} transitions before training");
            Console.WriteLine($"Training every {trainEvery} steps");
            Console.WriteLine();

            float[] rewardHistory = new float[numEpisodes];
            int totalSteps = 0;

            for (int episode = 0; episode < numEpisodes; episode++)
            {
                var state = env.Reset();
                float episodeReward = 0;
                bool done = false;
                int steps = 0;

                while (!done && steps < maxSteps)
                {
                    // Select action with exploration noise during training
                    var action = agent.SelectAction(state, addNoise: true);

                    // Take step in environment
                    var (nextState, reward, isDone) = env.Step(action);

                    // Store transition
                    agent.Store(state, action, reward, nextState, isDone);
                    totalSteps++;

                    // Train if we have enough data
                    if (totalSteps > startTraining && totalSteps % trainEvery == 0)
                    {
                        for (int i = 0; i < trainsPerStep; i++)
                        {
                            agent.Train(batchSize: 64);
                        }
                    }

                    state = nextState;
                    episodeReward += reward;
                    done = isDone;
                    steps++;
                }

                rewardHistory[episode] = episodeReward;

                // Print progress every 10 episodes
                if ((episode + 1) % 10 == 0)
                {
                    float avgReward = 0;
                    for (int i = Math.Max(0, episode - 9); i <= episode; i++)
                    {
                        avgReward += rewardHistory[i];
                    }
                    avgReward /= Math.Min(10, episode + 1);

                    Console.WriteLine($"Episode {episode + 1}/{numEpisodes} | " +
                                    $"Total Steps: {totalSteps} | " +
                                    $"Avg Reward (last 10): {avgReward:F2}");
                }
            }

            Console.WriteLine();
            Console.WriteLine("=== Training Complete! ===");
            Console.WriteLine();

            // Test the trained agent (deterministic, no noise)
            Console.WriteLine("Testing trained agent (10 episodes, deterministic)...");
            float testRewardSum = 0;

            for (int episode = 0; episode < 10; episode++)
            {
                var state = env.Reset();
                float episodeReward = 0;
                bool done = false;
                int steps = 0;

                while (!done && steps < maxSteps)
                {
                    // Select action WITHOUT exploration noise
                    var action = agent.SelectAction(state, addNoise: false);

                    var (nextState, reward, isDone) = env.Step(action);

                    state = nextState;
                    episodeReward += reward;
                    done = isDone;
                    steps++;
                }

                testRewardSum += episodeReward;
                Console.WriteLine($"Test Episode {episode + 1}: Reward = {episodeReward:F2}");
            }

            float avgTestReward = testRewardSum / 10;
            Console.WriteLine();
            Console.WriteLine($"Average Test Reward: {avgTestReward:F2}");
            Console.WriteLine();

            // Explain the results
            Console.WriteLine("=== Understanding TD3 Results ===");
            Console.WriteLine();
            Console.WriteLine("TD3 ADVANTAGES:");
            Console.WriteLine("• Twin critics prevent overestimation → more stable learning");
            Console.WriteLine("• Replay buffer → 10-20x more sample efficient than PPO");
            Console.WriteLine("• Delayed updates → actor doesn't chase moving targets");
            Console.WriteLine("• Target smoothing → robust to Q-function errors");
            Console.WriteLine("• Deterministic policy → consistent behavior at test time");
            Console.WriteLine();

            Console.WriteLine("WHEN TO USE TD3:");
            Console.WriteLine("• Continuous control (robotics, vehicles)");
            Console.WriteLine("• Limited data / expensive interactions");
            Console.WriteLine("• Need deterministic policies");
            Console.WriteLine("• Dense reward tasks");
            Console.WriteLine();

            Console.WriteLine("TD3 vs PPO:");
            Console.WriteLine("• TD3: Off-policy, more sample efficient");
            Console.WriteLine("• PPO: On-policy, simpler, better with sparse rewards");
            Console.WriteLine("• TD3: Better for complex dynamics");
            Console.WriteLine("• PPO: Better for exploration-heavy tasks");
            Console.WriteLine();

            Console.WriteLine("KEY HYPERPARAMETERS:");
            Console.WriteLine($"• Policy Delay: {2} (update actor every N critic updates)");
            Console.WriteLine($"• Target Noise: {0.2f} * {2.0f} = {0.4f} (smoothing)");
            Console.WriteLine($"• Policy Noise: {0.2f} * {2.0f} = {0.4f} (exploration)");
            Console.WriteLine($"• Tau: {0.005f} (soft update rate)");
            Console.WriteLine();

            Console.WriteLine("The twin critics act like having two scouts who BOTH must");
            Console.WriteLine("approve a play. We always listen to the MORE CONSERVATIVE one,");
            Console.WriteLine("preventing the overconfidence that plagued single-critic methods!");
            Console.WriteLine();
        }

        public static void RunQuickDemo()
        {
            Console.WriteLine("=== TD3 Quick Demo ===");
            Console.WriteLine();

            var env = new PendulumEnvironment();
            var agent = new TD3Agent(
                stateDim: 3,
                actionDim: 1,
                hiddenLayers: new[] { 128, 128 },
                actionScale: 2.0f
            );

            // Quick training (25 episodes)
            for (int episode = 0; episode < 25; episode++)
            {
                var state = env.Reset();
                float episodeReward = 0;
                bool done = false;
                int steps = 0;

                while (!done && steps < 200)
                {
                    var action = agent.SelectAction(state, addNoise: true);
                    var (nextState, reward, isDone) = env.Step(action);

                    agent.Store(state, action, reward, nextState, isDone);

                    // Start training after 500 steps
                    if (episode * 200 + steps > 500 && steps % 10 == 0)
                    {
                        agent.Train(batchSize: 32);
                    }

                    state = nextState;
                    episodeReward += reward;
                    done = isDone;
                    steps++;
                }

                if ((episode + 1) % 5 == 0)
                {
                    Console.WriteLine($"Episode {episode + 1}: Reward = {episodeReward:F2}");
                }
            }

            Console.WriteLine();
            Console.WriteLine("Quick demo complete! TD3 is learning to balance the pendulum.");
            Console.WriteLine("Run full training for better results!");
        }
    }
}
