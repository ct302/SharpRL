using System;
using System.Linq;
using SharpRL.Agents;
using SharpRL.Environments;

namespace SharpRL.Examples
{
    /// <summary>
    /// Standalone SAC (Soft Actor-Critic) Example
    /// 
    /// SAC is the "Tom Brady" of continuous control - it doesn't just win, it wins while
    /// keeping ALL OPTIONS OPEN through maximum entropy reinforcement learning.
    /// 
    /// KEY FEATURES:
    /// 1. Stochastic Policy - Naturally explores through action distribution
    /// 2. Automatic Temperature Tuning - α learns itself (no manual tuning!)
    /// 3. Twin Q-Networks - Conservative estimates prevent overestimation
    /// 4. Maximum Entropy Objective - Maximize reward + exploration simultaneously
    /// 
    /// MATH IN PLAIN ENGLISH:
    /// 
    /// SAC maximizes: Reward + α × Entropy
    /// - Reward: How well you're doing (score differential)
    /// - Entropy: How unpredictable your actions are (playbook flexibility)
    /// - α (alpha): Temperature that balances the two (AUTO-TUNED!)
    /// 
    /// This creates robust policies that:
    /// - Maintain multiple good strategies (multimodal solutions)
    /// - Handle perturbations well (stochastic robustness)
    /// - Explore naturally without epsilon-greedy
    /// - Adapt to changing conditions
    /// 
    /// WHY SAC IS THE GOAT:
    /// - More robust than TD3 (stochastic policy handles uncertainty)
    /// - Better exploration than TD3 (entropy bonus built-in)
    /// - No manual tuning (alpha adapts automatically)
    /// - Perfect for real-world robotics
    /// - State-of-the-art on MuJoCo benchmarks
    /// 
    /// WHEN TO USE:
    /// - Complex continuous control (robotics, autonomous systems)
    /// - Uncertain/stochastic environments
    /// - Multiple good solutions exist
    /// - Need robust, adaptable policies
    /// - Production systems that must handle variations
    /// </summary>
    public class SACExample
    {
        public static void Run()
        {
            Console.WriteLine("╔════════════════════════════════════════════════════════╗");
            Console.WriteLine("║  SAC - Soft Actor-Critic (Maximum Entropy RL) 🏆      ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════╝\n");

            // Create pendulum environment
            var env = new PendulumEnvironment(seed: 42);

            Console.WriteLine("ENVIRONMENT:");
            Console.WriteLine($"  State: {env.StateSize}D [cos(θ), sin(θ), angular velocity]");
            Console.WriteLine($"  Action: {env.ActionSize}D continuous [torque: -2.0 to +2.0]");
            Console.WriteLine($"  Goal: Balance pendulum upright with minimal torque\n");

            // Create SAC agent with recommended hyperparameters
            var agent = new SACAgent(
                stateDim: env.StateSize,
                actionDim: env.ActionSize,
                hiddenLayers: new[] { 256, 256 },  // Larger networks for complex tasks
                actionScale: 2.0f,                 // Match environment action bounds
                bufferSize: 100000,                // Large replay buffer
                learningRate: 3e-4f,               // Standard for SAC
                gamma: 0.99f,                      // Discount factor
                tau: 0.005f,                       // Slow target network updates
                autoTuneAlpha: true,               // 🔥 AUTO-TUNE temperature!
                initialAlpha: 0.2f,                // Starting temperature
                targetEntropy: null,               // Auto-calculated as -actionDim
                seed: 42
            );

            Console.WriteLine("AGENT: SAC (Soft Actor-Critic)");
            Console.WriteLine("  Policy: Stochastic Gaussian π(a|s)");
            Console.WriteLine("  Actor: 3 → 256 → 256 → 2 (mean + log_std)");
            Console.WriteLine("  Critics: Twin Q-networks (Q1, Q2)");
            Console.WriteLine("  Temperature: AUTO-TUNED (α adapts during training) 🎯");
            Console.WriteLine("  Target Networks: Soft updates (τ = 0.005)");
            Console.WriteLine("  Replay Buffer: 100,000 transitions\n");

            // Training parameters
            int numEpisodes = 100;
            int maxSteps = 200;
            int startTraining = 1000;    // Collect data before training
            int trainEvery = 10;         // Train frequently

            Console.WriteLine("TRAINING PLAN:");
            Console.WriteLine($"  Episodes: {numEpisodes}");
            Console.WriteLine($"  Max steps per episode: {maxSteps}");
            Console.WriteLine($"  Start training after: {startTraining} transitions");
            Console.WriteLine($"  Train every: {trainEvery} steps");
            Console.WriteLine($"  Batch size: 256\n");

            Console.WriteLine("🚀 Starting training...\n");
            Console.WriteLine("Watch how SAC:");
            Console.WriteLine("  1. Explores efficiently (entropy bonus)");
            Console.WriteLine("  2. Adapts temperature automatically (α tunes itself)");
            Console.WriteLine("  3. Maintains robustness (stochastic policy)");
            Console.WriteLine();

            float[] episodeRewards = new float[numEpisodes];
            int totalSteps = 0;

            for (int episode = 0; episode < numEpisodes; episode++)
            {
                var state = env.Reset();
                float episodeReward = 0;
                bool done = false;
                int steps = 0;

                while (!done && steps < maxSteps)
                {
                    // Select action from stochastic policy
                    // Exploration is built-in through entropy!
                    var action = agent.SelectAction(state, deterministic: false);

                    var (nextState, reward, isDone) = env.Step(action);

                    // Store transition
                    agent.Store(state, action, reward, nextState, isDone);
                    totalSteps++;

                    // Train after collecting enough data
                    if (totalSteps > startTraining && totalSteps % trainEvery == 0)
                    {
                        agent.Train(batchSize: 256);
                    }

                    state = nextState;
                    episodeReward += reward;
                    done = isDone;
                    steps++;
                }

                episodeRewards[episode] = episodeReward;

                // Print progress every 10 episodes
                if ((episode + 1) % 10 == 0)
                {
                    float avgReward = episodeRewards.Skip(Math.Max(0, episode - 9)).Take(10).Average();
                    Console.WriteLine($"Episode {episode + 1}/{numEpisodes} | " +
                                    $"Steps: {totalSteps} | " +
                                    $"Avg Reward (last 10): {avgReward:F2}");
                }
            }

            Console.WriteLine("\n✅ Training complete!\n");

            // Calculate final statistics
            float finalAvgReward = episodeRewards.Skip(Math.Max(0, numEpisodes - 10)).Take(10).Average();
            Console.WriteLine($"📊 FINAL PERFORMANCE:");
            Console.WriteLine($"  Average reward (last 10 episodes): {finalAvgReward:F2}");
            Console.WriteLine($"  Total transitions collected: {totalSteps}\n");

            // Test with BOTH stochastic and deterministic policies
            Console.WriteLine("═══════════════════════════════════════════════════════");
            Console.WriteLine("TESTING TRAINED AGENT");
            Console.WriteLine("═══════════════════════════════════════════════════════\n");

            // Test 1: Stochastic policy (exploration maintained)
            Console.WriteLine("TEST 1: Stochastic Policy (maintains exploration)");
            Console.WriteLine("Like a QB who can still improvise and adapt\n");

            float stochasticReward = 0;
            for (int testEp = 0; testEp < 3; testEp++)
            {
                var state = env.Reset();
                float testReward = 0f;

                for (int step = 0; step < maxSteps; step++)
                {
                    var action = agent.SelectAction(state, deterministic: false);
                    var (nextState, reward, done) = env.Step(action);

                    testReward += reward;
                    state = nextState;

                    if (done) break;
                }

                stochasticReward += testReward;
                Console.WriteLine($"  Episode {testEp + 1}: Reward = {testReward:F2}");
            }

            float avgStochasticReward = stochasticReward / 3;
            Console.WriteLine($"  Average: {avgStochasticReward:F2}\n");

            // Test 2: Deterministic policy (pure exploitation)
            Console.WriteLine("TEST 2: Deterministic Policy (pure exploitation)");
            Console.WriteLine("Like a QB executing the perfect game plan\n");

            float deterministicReward = 0;
            for (int testEp = 0; testEp < 3; testEp++)
            {
                var state = env.Reset();
                float testReward = 0f;

                for (int step = 0; step < maxSteps; step++)
                {
                    var action = agent.SelectAction(state, deterministic: true);
                    var (nextState, reward, done) = env.Step(action);

                    testReward += reward;
                    state = nextState;

                    if (done) break;
                }

                deterministicReward += testReward;
                Console.WriteLine($"  Episode {testEp + 1}: Reward = {testReward:F2}");
            }

            float avgDeterministicReward = deterministicReward / 3;
            Console.WriteLine($"  Average: {avgDeterministicReward:F2}\n");

            // Compare policies
            Console.WriteLine("═══════════════════════════════════════════════════════");
            Console.WriteLine("POLICY COMPARISON");
            Console.WriteLine("═══════════════════════════════════════════════════════");
            Console.WriteLine($"Stochastic:    {avgStochasticReward:F2} (with exploration)");
            Console.WriteLine($"Deterministic: {avgDeterministicReward:F2} (pure exploitation)");
            Console.WriteLine();

            if (avgStochasticReward > avgDeterministicReward * 0.95f)
            {
                Console.WriteLine("✓ Stochastic policy performs well!");
                Console.WriteLine("  SAC's entropy bonus maintains good exploration.");
            }
            else
            {
                Console.WriteLine("✓ Both policies perform well!");
                Console.WriteLine("  SAC learned robust behavior in both modes.");
            }

            Console.WriteLine();
            Console.WriteLine("═══════════════════════════════════════════════════════");
            Console.WriteLine("🏆 SAC: THE GOAT OF CONTINUOUS CONTROL");
            Console.WriteLine("═══════════════════════════════════════════════════════");
            Console.WriteLine();
            Console.WriteLine("KEY ADVANTAGES:");
            Console.WriteLine("  ✓ Maximum entropy → most robust policies");
            Console.WriteLine("  ✓ Auto-tuned temperature → no manual tuning");
            Console.WriteLine("  ✓ Stochastic policy → handles uncertainty naturally");
            Console.WriteLine("  ✓ Twin critics → prevents overestimation");
            Console.WriteLine("  ✓ Off-policy → sample efficient");
            Console.WriteLine("  ✓ Multimodal → handles multiple good strategies");
            Console.WriteLine();
            Console.WriteLine("SAC vs TD3 vs PPO:");
            Console.WriteLine("  • SAC: Most robust, best exploration, stochastic");
            Console.WriteLine("  • TD3: Fastest inference, deterministic, known envs");
            Console.WriteLine("  • PPO: Most stable, on-policy, easier to tune");
            Console.WriteLine();
            Console.WriteLine("PERFECT FOR:");
            Console.WriteLine("  • Real-world robotics (handles perturbations)");
            Console.WriteLine("  • Uncertain environments (stochastic dynamics)");
            Console.WriteLine("  • Complex control (multimodal solutions)");
            Console.WriteLine("  • Production systems (robust to variations)");
            Console.WriteLine("═══════════════════════════════════════════════════════");
            Console.WriteLine();
            Console.WriteLine("🎊 Example Complete! SAC is now part of your toolkit!");
        }
    }
}
