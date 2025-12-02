using System;
using System.Linq;
using System.Collections.Generic;
using SharpRL.AutoGrad;
using SharpRL.Core;
using SharpRL.Core.ContinuousActions;
using SharpRL.NN;
using SharpRL.NN.Layers;
using SharpRL.NN.Optimizers;
using SharpRL.NN.Loss;

namespace SharpRL.Agents
{
    /// <summary>
    /// SAC (Soft Actor-Critic) Agent
    /// Maximum Entropy Reinforcement Learning for Continuous Control
    /// 
    /// NFL ANALOGY:
    /// If TD3 is like a coach with a FIXED playbook (deterministic), SAC is like a coach who
    /// keeps the playbook FLEXIBLE and ADAPTABLE (stochastic). SAC doesn't just want to win -
    /// it wants to win while maintaining MAXIMUM OPTIONS. Think of Bill Belichick: he doesn't
    /// just find one winning strategy, he maintains multiple ways to win so opponents can't
    /// predict his plays. This "entropy" (unpredictability) makes SAC incredibly robust.
    /// 
    /// THE MAXIMUM ENTROPY FRAMEWORK:
    /// Instead of just maximizing reward, SAC maximizes: Reward + α × Entropy
    /// 
    /// Where:
    /// - Reward: How well you're doing (score differential)
    /// - Entropy: How unpredictable/diverse your actions are (playbook flexibility)
    /// - α (alpha): Temperature parameter that balances the two (auto-tuned!)
    /// 
    /// FOUR KEY INNOVATIONS:
    /// 
    /// 1. Stochastic Policy with Entropy Bonus
    ///    - Policy outputs a DISTRIBUTION over actions (Gaussian)
    ///    - Agent is rewarded for keeping options open
    ///    - Naturally explores without needing epsilon-greedy
    ///    - Like keeping multiple plays ready instead of committing to one
    ///    
    /// 2. Automatic Temperature Tuning
    ///    - α parameter learned automatically during training
    ///    - No manual tuning needed - agent finds right balance
    ///    - Early: High α → explore more (find all plays)
    ///    - Late: Low α → exploit more (use best plays)
    ///    
    /// 3. Twin Q-Networks (from TD3)
    ///    - Two independent critics for conservative estimates
    ///    - Prevents overestimation bias
    ///    - Like having two scouts verify each play's value
    ///    
    /// 4. Off-Policy Learning with Replay Buffer
    ///    - Learns from past experiences (sample efficient)
    ///    - Can reuse data multiple times
    ///    - Like studying game film from previous seasons
    /// 
    /// MATH BREAKDOWN (in plain English):
    /// 
    /// The SAC objective is:
    /// J(π) = E[∑(r_t + α·H(π(·|s_t)))]
    /// 
    /// Where:
    /// - J(π): Total value of policy π (how good is our playbook)
    /// - E[...]: Expected value (average over many games)
    /// - r_t: Reward at time t (points scored this play)
    /// - α: Temperature (how much we value flexibility)
    /// - H(π(·|s_t)): Entropy of policy at state s_t (how unpredictable we are)
    /// 
    /// In simpler terms: "Pick actions that score points AND keep options open"
    /// 
    /// The Policy Loss is:
    /// L_π = E[α·log π(a|s) - Q(s,a)]
    /// 
    /// Breaking it down:
    /// - log π(a|s): Negative log probability of action a in state s (lower = more likely action)
    /// - Q(s,a): Expected future reward (how good is this action long-term)
    /// - α·log π(a|s): Entropy penalty (cost of predictability)
    /// - We MINIMIZE this loss, which means:
    ///   * Maximize Q(s,a) - pick high-value actions
    ///   * Minimize α·log π(a|s) - stay unpredictable
    /// 
    /// The Temperature Update:
    /// α_loss = -α · (log π(a|s) + H_target)
    /// 
    /// Where:
    /// - H_target: Target entropy (usually -dim(action_space))
    /// - log π(a|s): Current policy entropy
    /// - If entropy too low → increase α → encourage more exploration
    /// - If entropy too high → decrease α → focus on exploitation
    /// 
    /// WHEN TO USE SAC:
    /// - Complex continuous control (robotics, autonomous systems)
    /// - When you need robust policies (handles perturbations well)
    /// - When exploration is critical (sparse rewards)
    /// - When sample efficiency matters (learns from replay buffer)
    /// - Production systems that need to adapt to changing conditions
    /// 
    /// SAC vs TD3 vs PPO:
    /// - SAC: Stochastic + entropy → most robust, handles exploration best
    /// - TD3: Deterministic + twin critics → fastest inference, good for known environments
    /// - PPO: On-policy + trust region → most stable training, good for beginners
    /// 
    /// SAC Performance:
    /// - Often matches or beats TD3 on continuous control benchmarks
    /// - More stable than TD3 (stochastic policy smooths gradients)
    /// - Better exploration than TD3 (entropy bonus)
    /// - Can handle multimodal reward landscapes (multiple good strategies)
    /// </summary>
    public class SACAgent : IContinuousAgent
    {
        // Actor network (stochastic policy)
        private readonly Module actorNetwork;           // Outputs mean and log_std        
        // Twin critic networks
        private readonly Module critic1Network;         // First Q-network
        private readonly Module critic2Network;         // Second Q-network (twin)
        
        // Target networks (slowly updated copies for stability)
        private readonly Module targetCritic1Network;
        private readonly Module targetCritic2Network;
        
        // Temperature parameter (entropy coefficient)
        private Tensor logAlpha;                        // Log of temperature (learned)
        private float alpha;                            // Current temperature value
        private readonly float targetEntropy;           // Target entropy for auto-tuning
        
        // Optimizers
        private readonly Optimizer actorOptimizer;
        private readonly Optimizer critic1Optimizer;
        private readonly Optimizer critic2Optimizer;
        private readonly Optimizer? alphaOptimizer;
        
        // Experience replay
        private readonly ReplayBuffer<float[], float[]> replayBuffer;
        
        // Hyperparameters
        private readonly int stateDim;
        private readonly int actionDim;
        private readonly float actionScale;             // Scale actions to environment bounds
        private readonly float gamma;                   // Discount factor
        private readonly float tau;                     // Soft target update rate
        private readonly float learningRate;
        private readonly bool autoTuneAlpha;            // Whether to learn alpha
        
        // Constants for numerical stability
        private const float LOG_STD_MIN = -20f;
        private const float LOG_STD_MAX = 2f;
        private const float EPSILON = 1e-6f;
        
        private readonly Random random;

        /// <summary>
        /// Create a new SAC agent
        /// </summary>
        /// <param name="stateDim">Dimension of state space</param>
        /// <param name="actionDim">Dimension of action space</param>
        /// <param name="hiddenLayers">Hidden layer sizes (default: [256, 256])</param>
        /// <param name="actionScale">Scale factor for actions (default: 1.0)</param>
        /// <param name="bufferSize">Size of replay buffer (default: 100000)</param>
        /// <param name="learningRate">Learning rate for all networks (default: 3e-4)</param>
        /// <param name="gamma">Discount factor (default: 0.99)</param>
        /// <param name="tau">Target network update rate (default: 0.005)</param>
        /// <param name="autoTuneAlpha">Auto-tune temperature parameter (default: true)</param>
        /// <param name="initialAlpha">Initial temperature value (default: 0.2)</param>
        /// <param name="targetEntropy">Target entropy (default: -actionDim, auto-calculated if null)</param>
        /// <param name="seed">Random seed for reproducibility</param>
        public SACAgent(
            int stateDim,
            int actionDim,
            int[] hiddenLayers = null!,
            float actionScale = 1.0f,
            int bufferSize = 100000,
            float learningRate = 3e-4f,
            float gamma = 0.99f,
            float tau = 0.005f,
            bool autoTuneAlpha = true,
            float initialAlpha = 0.2f,
            float? targetEntropy = null,
            int? seed = null)
        {
            this.stateDim = stateDim;
            this.actionDim = actionDim;
            this.actionScale = actionScale;
            this.gamma = gamma;
            this.tau = tau;
            this.learningRate = learningRate;
            this.autoTuneAlpha = autoTuneAlpha;
            this.random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            // Target entropy: heuristic from SAC paper is -dim(action_space)
            this.targetEntropy = targetEntropy ?? -actionDim;

            hiddenLayers = hiddenLayers ?? new[] { 256, 256 };

            // Build networks
            actorNetwork = BuildActorNetwork(stateDim, actionDim, hiddenLayers);
            critic1Network = BuildCriticNetwork(stateDim, actionDim, hiddenLayers);
            critic2Network = BuildCriticNetwork(stateDim, actionDim, hiddenLayers);

            // Build target networks
            targetCritic1Network = BuildCriticNetwork(stateDim, actionDim, hiddenLayers);
            targetCritic2Network = BuildCriticNetwork(stateDim, actionDim, hiddenLayers);

            // Copy weights to targets
            CopyNetworkWeights(critic1Network, targetCritic1Network);
            CopyNetworkWeights(critic2Network, targetCritic2Network);

            // Initialize temperature
            logAlpha = new Tensor(new[] { (float)Math.Log(initialAlpha) }, new[] { 1 });
            logAlpha.RequiresGrad = autoTuneAlpha;
            alpha = initialAlpha;

            // Create optimizers
            actorOptimizer = new Adam(actorNetwork.Parameters(), learningRate);
            critic1Optimizer = new Adam(critic1Network.Parameters(), learningRate);
            critic2Optimizer = new Adam(critic2Network.Parameters(), learningRate);
            
            if (autoTuneAlpha)
            {
                alphaOptimizer = new Adam(new List<Tensor> { logAlpha }, learningRate);
            }

            replayBuffer = new ReplayBuffer<float[], float[]>(bufferSize);
        }

        /// <summary>
        /// Build actor network that outputs mean and log_std for Gaussian policy
        /// Output shape: [batch, actionDim * 2] where first half is mean, second half is log_std
        /// </summary>
        private Module BuildActorNetwork(int inputSize, int actionDim, int[] hiddenSizes)
        {
            var layers = new List<Module>();
            
            int prevSize = inputSize;
            foreach (int hiddenSize in hiddenSizes)
            {
                layers.Add(new Linear(prevSize, hiddenSize));
                layers.Add(new ReLU());
                prevSize = hiddenSize;
            }
            
            // Output both mean and log_std
            layers.Add(new Linear(prevSize, actionDim * 2));
            
            return new Sequential(layers.ToArray());
        }


        /// <summary>
        /// Build critic network that takes state and action as input
        /// </summary>
        private Module BuildCriticNetwork(int stateDim, int actionDim, int[] hiddenSizes)
        {
            var layers = new List<Module>();
            
            int inputSize = stateDim + actionDim;
            
            layers.Add(new Linear(inputSize, hiddenSizes[0]));
            layers.Add(new ReLU());
            
            for (int i = 1; i < hiddenSizes.Length; i++)
            {
                layers.Add(new Linear(hiddenSizes[i - 1], hiddenSizes[i]));
                layers.Add(new ReLU());
            }
            
            layers.Add(new Linear(hiddenSizes[hiddenSizes.Length - 1], 1));
            
            return new Sequential(layers.ToArray());
        }

        private void CopyNetworkWeights(Module source, Module target)
        {
            var sourceParams = source.Parameters();
            var targetParams = target.Parameters();
            
            if (sourceParams.Count != targetParams.Count)
                throw new InvalidOperationException("Network architectures don't match!");
            
            for (int i = 0; i < sourceParams.Count; i++)
            {
                Array.Copy(sourceParams[i].Data, targetParams[i].Data, sourceParams[i].Data.Length);
            }
        }


        private void SoftUpdateTargetNetwork(Module main, Module target)
        {
            var mainParams = main.Parameters();
            var targetParams = target.Parameters();
            
            for (int i = 0; i < mainParams.Count; i++)
            {
                for (int j = 0; j < mainParams[i].Data.Length; j++)
                {
                    targetParams[i].Data[j] = tau * mainParams[i].Data[j] + (1 - tau) * targetParams[i].Data[j];
                }
            }
        }

        /// <summary>
        /// Sample action from policy
        /// Uses reparameterization trick: a = tanh(μ + σ * ε) where ε ~ N(0,1)
        /// </summary>
        public float[] SelectAction(float[] state, bool deterministic = false)
        {
            if (state.Length != stateDim)
                throw new ArgumentException($"State size {state.Length} doesn't match expected {stateDim}");

            var stateTensor = new Tensor(state, new[] { 1, stateDim });
            var output = actorNetwork.Forward(stateTensor);
            
            // Split output into mean and log_std
            var mean = new float[actionDim];
            var logStd = new float[actionDim];
            
            for (int i = 0; i < actionDim; i++)
            {
                mean[i] = output.Data[i];
                logStd[i] = Math.Clamp(output.Data[actionDim + i], LOG_STD_MIN, LOG_STD_MAX);
            }


            float[] action;
            if (deterministic)
            {
                // Use mean action (no randomness)
                action = mean;
            }
            else
            {
                // Sample from Gaussian distribution
                var std = logStd.Select(ls => (float)Math.Exp(ls)).ToArray();
                action = new float[actionDim];
                
                for (int i = 0; i < actionDim; i++)
                {
                    // Sample from N(0,1)
                    float u1 = (float)random.NextDouble();
                    float u2 = (float)random.NextDouble();
                    float z = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                    
                    // Reparameterization: a = μ + σ * z
                    action[i] = mean[i] + std[i] * z;
                }
            }

            // Apply tanh squashing and scale
            for (int i = 0; i < actionDim; i++)
            {
                action[i] = (float)Math.Tanh(action[i]) * actionScale;
            }

            return action;
        }


        /// <summary>
        /// Sample action and compute log probability (used during training)
        /// Returns: (action, log_prob)
        /// </summary>
        private (Tensor action, Tensor logProb) SampleAction(Tensor states)
        {
            int batchSize = states.Shape[0];
            var output = actorNetwork.Forward(states);
            
            // Split into mean and log_std
            var meanData = new float[batchSize * actionDim];
            var logStdData = new float[batchSize * actionDim];
            
            for (int b = 0; b < batchSize; b++)
            {
                for (int a = 0; a < actionDim; a++)
                {
                    int idx = b * actionDim * 2 + a;
                    meanData[b * actionDim + a] = output.Data[idx];
                    logStdData[b * actionDim + a] = Math.Clamp(
                        output.Data[idx + actionDim], 
                        LOG_STD_MIN, 
                        LOG_STD_MAX
                    );
                }
            }

            var mean = new Tensor(meanData, new[] { batchSize, actionDim }, requiresGrad: true);
            mean.GradFn = new SplitFunction(output, 0, actionDim);
            
            var logStd = new Tensor(logStdData, new[] { batchSize, actionDim }, requiresGrad: true);
            logStd.GradFn = new SplitFunction(output, actionDim, actionDim);
            
            var std = TensorExp(logStd);


            // Sample from Gaussian using reparameterization trick
            var epsilon = SampleGaussian(batchSize, actionDim);
            var preTanhAction = TensorAdd(mean, TensorMultiply(std, epsilon));
            
            // Apply tanh squashing
            var action = TensorTanh(preTanhAction);
            
            // Compute log probability with tanh correction
            // log π(a|s) = log π(u|s) - Σ log(1 - tanh²(u))
            var logProb = ComputeLogProb(preTanhAction, mean, std, output);

            return (action, logProb);
        }

        /// <summary>
        /// Compute log probability of action under Gaussian policy with tanh squashing
        /// log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
        /// 
        /// Breaking it down:
        /// - log N(u|μ,σ): Log probability of pre-tanh action u under Gaussian
        /// - Σ log(1 - tanh²(u)): Correction for tanh transformation (change of variables)
        /// </summary>
        private Tensor ComputeLogProb(Tensor preTanhAction, Tensor mean, Tensor std, Tensor networkOutput)
        {
            int batchSize = preTanhAction.Shape[0];
            
            // Compute log probability of Gaussian: log N(u|μ,σ) = -0.5 * ((u-μ)/σ)² - log(σ) - 0.5*log(2π)
            var logProbData = new float[batchSize];
            
            for (int b = 0; b < batchSize; b++)
            {
                float logProb = 0f;
                
                for (int a = 0; a < actionDim; a++)
                {
                    int idx = b * actionDim + a;
                    float u = preTanhAction.Data[idx];
                    float mu = mean.Data[idx];
                    float sigma = std.Data[idx];
                    
                    // Gaussian log probability
                    float gaussianLogProb = -0.5f * (float)Math.Pow((u - mu) / sigma, 2) 
                                          - (float)Math.Log(sigma) 
                                          - 0.5f * (float)Math.Log(2 * Math.PI);
                    
                    // Tanh correction: log(1 - tanh²(u))
                    float tanhU = (float)Math.Tanh(u);
                    float tanhCorrection = (float)Math.Log(1 - tanhU * tanhU + EPSILON);
                    
                    logProb += gaussianLogProb - tanhCorrection;
                }
                
                logProbData[b] = logProb;
            }
            
            var result = new Tensor(logProbData, new[] { batchSize, 1 }, requiresGrad: true);
            result.GradFn = new LogProbFunction(networkOutput, preTanhAction, mean, std);
            return result;
        }

        public void Store(float[] state, float[] action, float reward, float[] nextState, bool done)
        {
            replayBuffer.Add(state, action, reward, nextState, done);
        }


        /// <summary>
        /// Train the agent using a batch of experiences
        /// </summary>
        public void Train(int batchSize = 256)
        {
            if (replayBuffer.Count < batchSize)
                return;

            var batch = replayBuffer.Sample(batchSize);
            
            // Convert to tensors
            var states = new Tensor(
                batch.SelectMany(exp => exp.State).ToArray(), new[] { batchSize, stateDim });
            var actions = new Tensor(
                batch.SelectMany(exp => exp.Action).ToArray(), new[] { batchSize, actionDim });
            var rewards = new Tensor(
                batch.Select(exp => (float)exp.Reward).ToArray(), new[] { batchSize, 1 });
            var nextStates = new Tensor(
                batch.SelectMany(exp => exp.NextState).ToArray(), new[] { batchSize, stateDim });
            var dones = new Tensor(
                batch.Select(exp => exp.Done ? 1.0f : 0.0f).ToArray(), new[] { batchSize, 1 });

            // ============================================================
            // STEP 1: Update Critics (Q-functions)
            // ============================================================
            
            // Sample next actions from current policy
            var (nextActions, nextLogProbs) = SampleAction(nextStates);
            
            // Scale actions for critic input
            var scaledNextActions = TensorScale(nextActions, actionScale);
            
            // Compute target Q-values using twin critics (take minimum)
            var nextStateActions = ConcatenateTensors(nextStates, scaledNextActions);
            var targetQ1 = targetCritic1Network.Forward(nextStateActions);
            var targetQ2 = targetCritic2Network.Forward(nextStateActions);
            var minTargetQ = MinTensors(targetQ1, targetQ2);
            
            // Compute target: r + γ * (1 - done) * (min_Q - α * log_prob)
            // The "- α * log_prob" term is the entropy bonus!
            var entropyTerm = TensorScale(nextLogProbs, alpha);
            var targetValue = TensorSubtract(minTargetQ, entropyTerm);
            var target = ComputeTDTarget(rewards, targetValue, dones);

            // Update Critic 1
            var scaledActions = TensorScale(actions, actionScale);
            var stateActions = ConcatenateTensors(states, scaledActions);
            var currentQ1 = critic1Network.Forward(stateActions);
            var critic1Loss = new MSELoss().Forward(currentQ1, target);
            
            critic1Optimizer.ZeroGrad();
            critic1Loss.Backward();
            critic1Optimizer.Step();

            // Update Critic 2
            var currentQ2 = critic2Network.Forward(stateActions);
            var critic2Loss = new MSELoss().Forward(currentQ2, target);
            
            critic2Optimizer.ZeroGrad();
            critic2Loss.Backward();
            critic2Optimizer.Step();


            // ============================================================
            // STEP 2: Update Actor (Policy)
            // ============================================================
            
            // Sample actions from current policy
            var (policyActions, policyLogProbs) = SampleAction(states);
            var scaledPolicyActions = TensorScale(policyActions, actionScale);
            
            // Compute Q-values for sampled actions
            var statePolicyActions = ConcatenateTensors(states, scaledPolicyActions);
            var q1 = critic1Network.Forward(statePolicyActions);
            var q2 = critic2Network.Forward(statePolicyActions);
            var minQ = MinTensors(q1, q2);
            
            // Actor loss: E[α * log π(a|s) - Q(s,a)]
            // We want to maximize Q and entropy, so we minimize this
            var logProbMean = TensorMean(policyLogProbs);
            var qMean = TensorMean(minQ);
            
            // Compute: alpha * logProb - Q (both as tensors to maintain grad)
            var alphaLogProb = TensorScale(logProbMean, alpha);
            var actorLoss = TensorSubtract(alphaLogProb, qMean);
            
            actorOptimizer.ZeroGrad();
            actorLoss.Backward();
            actorOptimizer.Step();

            // ============================================================
            // STEP 3: Update Temperature (Alpha)
            // ============================================================
            
            if (autoTuneAlpha)
            {
                // Alpha loss: -α * (log π(a|s) + target_entropy)
                // If entropy is below target → decrease log_alpha → decrease alpha → less entropy penalty
                // If entropy is above target → increase log_alpha → increase alpha → more entropy penalty
                var entropyDiff = logProbMean.Data[0] + targetEntropy;
                
                // Create loss tensor with gradient tracking
                var alphaLoss = new Tensor(new[] { -logAlpha.Data[0] * entropyDiff }, new[] { 1 }, requiresGrad: true);
                alphaLoss.GradFn = new IdentityFunction(logAlpha); // Connect to logAlpha for gradient flow
                
                alphaOptimizer!.ZeroGrad();
                alphaLoss.Backward();
                alphaOptimizer.Step();
                
                // Update alpha from log_alpha
                alpha = (float)Math.Exp(logAlpha.Data[0]);
            }

            // ============================================================
            // STEP 4: Soft Update Target Networks
            // ============================================================
            
            SoftUpdateTargetNetwork(critic1Network, targetCritic1Network);
            SoftUpdateTargetNetwork(critic2Network, targetCritic2Network);
        }

        // ============================================================
        // Helper Functions for Tensor Operations
        // ============================================================

        private Tensor SampleGaussian(int batchSize, int dim)
        {
            var data = new float[batchSize * dim];
            for (int i = 0; i < data.Length; i++)
            {
                // Box-Muller transform
                float u1 = (float)random.NextDouble();
                float u2 = (float)random.NextDouble();
                data[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
            }
            return new Tensor(data, new[] { batchSize, dim });
        }


        private Tensor TensorExp(Tensor t)
        {
            var data = t.Data.Select(x => (float)Math.Exp(x)).ToArray();
            return new Tensor(data, t.Shape);
        }

        private Tensor TensorTanh(Tensor t)
        {
            var data = t.Data.Select(x => (float)Math.Tanh(x)).ToArray();
            return new Tensor(data, t.Shape);
        }

        private Tensor TensorAdd(Tensor t1, Tensor t2)
        {
            var data = new float[t1.Data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = t1.Data[i] + t2.Data[i];
            }
            return new Tensor(data, t1.Shape);
        }

        private Tensor TensorSubtract(Tensor t1, Tensor t2)
        {
            var data = new float[t1.Data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = t1.Data[i] - t2.Data[i];
            }
            var result = new Tensor(data, t1.Shape, requiresGrad: t1.RequiresGrad || t2.RequiresGrad);
            if (t1.RequiresGrad || t2.RequiresGrad)
            {
                result.GradFn = new SubtractFunction(t1, t2);
            }
            return result;
        }

        private Tensor TensorMultiply(Tensor t1, Tensor t2)
        {
            var data = new float[t1.Data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = t1.Data[i] * t2.Data[i];
            }
            return new Tensor(data, t1.Shape);
        }


        private Tensor TensorScale(Tensor t, float scale)
        {
            var data = t.Data.Select(x => x * scale).ToArray();
            var result = new Tensor(data, t.Shape, requiresGrad: t.RequiresGrad);
            if (t.RequiresGrad)
            {
                result.GradFn = new ScaleFunction(t, scale);
            }
            return result;
        }

        private Tensor MinTensors(Tensor t1, Tensor t2)
        {
            var minData = new float[t1.Data.Length];
            for (int i = 0; i < t1.Data.Length; i++)
            {
                minData[i] = Math.Min(t1.Data[i], t2.Data[i]);
            }
            return new Tensor(minData, t1.Shape);
        }

        private Tensor TensorMean(Tensor t)
        {
            var mean = t.Data.Average();
            var result = new Tensor(new[] { mean }, new[] { 1 }, requiresGrad: t.RequiresGrad);
            if (t.RequiresGrad)
            {
                result.GradFn = new MeanFunction(t);
            }
            return result;
        }

        private Tensor ComputeTDTarget(Tensor rewards, Tensor targetQ, Tensor dones)
        {
            var targetData = new float[rewards.Data.Length];
            for (int i = 0; i < rewards.Data.Length; i++)
            {
                targetData[i] = rewards.Data[i] + gamma * (1 - dones.Data[i]) * targetQ.Data[i];
            }
            return new Tensor(targetData, rewards.Shape);
        }

        private Tensor ConcatenateTensors(Tensor states, Tensor actions)
        {
            int batchSize = states.Shape[0];
            int totalDim = stateDim + actionDim;
            var concatData = new float[batchSize * totalDim];

            for (int i = 0; i < batchSize; i++)
            {
                Array.Copy(states.Data, i * stateDim, concatData, i * totalDim, stateDim);
                Array.Copy(actions.Data, i * actionDim, concatData, i * totalDim + stateDim, actionDim);
            }

            return new Tensor(concatData, new[] { batchSize, totalDim });
        }

        /// <summary>
        /// Train on batch of collected experience (IContinuousAgent interface implementation)
        /// This is a wrapper that converts the batch format to match the Store/Train pattern
        /// </summary>
        public void Train(float[][] states, float[][] actions, float[] rewards, float[][] nextStates, bool[] dones)
        {
            // Store all experiences in replay buffer
            for (int i = 0; i < states.Length; i++)
            {
                Store(states[i], actions[i], rewards[i], nextStates[i], dones[i]);
            }
            
            // Train from replay buffer
            Train(batchSize: Math.Min(256, replayBuffer.Count));
        }

        // ============================================================
        // Gradient Functions for Custom Operations
        // ============================================================
        
        private class MeanFunction : Function
        {
            private readonly Tensor input;
            
            public MeanFunction(Tensor input)
            {
                this.input = input;
            }
            
            public override void Backward(Tensor grad)
            {
                // Gradient of mean: distribute gradient equally to all elements
                if (input.Grad == null)
                {
                    input.Grad = new Tensor(new float[input.Data.Length], input.Shape);
                }
                
                float meanGrad = grad.Data[0] / input.Data.Length;
                for (int i = 0; i < input.Data.Length; i++)
                {
                    input.Grad.Data[i] += meanGrad;
                }
                
                // Continue backprop
                if (input.GradFn != null)
                {
                    input.GradFn.Backward(input.Grad);
                }
            }
        }
        
        private class IdentityFunction : Function
        {
            private readonly Tensor input;
            
            public IdentityFunction(Tensor input)
            {
                this.input = input;
            }
            
            public override void Backward(Tensor grad)
            {
                // Pass gradient through unchanged
                if (input.Grad == null)
                {
                    input.Grad = new Tensor(new float[input.Data.Length], input.Shape);
                }
                
                for (int i = 0; i < input.Data.Length; i++)
                {
                    input.Grad.Data[i] += grad.Data[i];
                }
                
                // Continue backprop
                if (input.GradFn != null)
                {
                    input.GradFn.Backward(input.Grad);
                }
            }
        }
        
        private class ScaleFunction : Function
        {
            private readonly Tensor input;
            private readonly float scale;
            
            public ScaleFunction(Tensor input, float scale)
            {
                this.input = input;
                this.scale = scale;
            }
            
            public override void Backward(Tensor grad)
            {
                // Gradient of scale: multiply gradient by scale
                if (input.Grad == null)
                {
                    input.Grad = new Tensor(new float[input.Data.Length], input.Shape);
                }
                
                for (int i = 0; i < input.Data.Length; i++)
                {
                    input.Grad.Data[i] += grad.Data[i] * scale;
                }
                
                // Continue backprop
                if (input.GradFn != null)
                {
                    input.GradFn.Backward(input.Grad);
                }
            }
        }
        
        private class SubtractFunction : Function
        {
            private readonly Tensor input1;
            private readonly Tensor input2;
            
            public SubtractFunction(Tensor input1, Tensor input2)
            {
                this.input1 = input1;
                this.input2 = input2;
            }
            
            public override void Backward(Tensor grad)
            {
                // Gradient of subtract: t1 - t2
                // d(t1 - t2)/dt1 = 1, d(t1 - t2)/dt2 = -1
                
                if (input1.RequiresGrad)
                {
                    if (input1.Grad == null)
                    {
                        input1.Grad = new Tensor(new float[input1.Data.Length], input1.Shape);
                    }
                    
                    for (int i = 0; i < input1.Data.Length; i++)
                    {
                        input1.Grad.Data[i] += grad.Data[i];
                    }
                    
                    if (input1.GradFn != null)
                    {
                        input1.GradFn.Backward(input1.Grad);
                    }
                }
                
                if (input2.RequiresGrad)
                {
                    if (input2.Grad == null)
                    {
                        input2.Grad = new Tensor(new float[input2.Data.Length], input2.Shape);
                    }
                    
                    for (int i = 0; i < input2.Data.Length; i++)
                    {
                        input2.Grad.Data[i] -= grad.Data[i]; // Note the minus!
                    }
                    
                    if (input2.GradFn != null)
                    {
                        input2.GradFn.Backward(input2.Grad);
                    }
                }
            }
        }
        
        private class SplitFunction : Function
        {
            private readonly Tensor networkOutput;
            private readonly int startIdx;
            private readonly int length;
            
            public SplitFunction(Tensor networkOutput, int startIdx, int length)
            {
                this.networkOutput = networkOutput;
                this.startIdx = startIdx;
                this.length = length;
            }
            
            public override void Backward(Tensor grad)
            {
                // Backpropagate gradient to the portion of network output that was split
                if (networkOutput.Grad == null)
                {
                    networkOutput.Grad = new Tensor(new float[networkOutput.Data.Length], networkOutput.Shape);
                }
                
                int batchSize = grad.Shape[0];
                for (int b = 0; b < batchSize; b++)
                {
                    for (int i = 0; i < length; i++)
                    {
                        int outputIdx = b * (length * 2) + startIdx + i;
                        int gradIdx = b * length + i;
                        networkOutput.Grad.Data[outputIdx] += grad.Data[gradIdx];
                    }
                }
                
                if (networkOutput.GradFn != null)
                {
                    networkOutput.GradFn.Backward(networkOutput.Grad);
                }
            }
        }
        
        private class LogProbFunction : Function
        {
            private readonly Tensor networkOutput;
            private readonly Tensor preTanhAction;
            private readonly Tensor mean;
            private readonly Tensor std;
            
            public LogProbFunction(Tensor networkOutput, Tensor preTanhAction, Tensor mean, Tensor std)
            {
                this.networkOutput = networkOutput;
                this.preTanhAction = preTanhAction;
                this.mean = mean;
                this.std = std;
            }
            
            public override void Backward(Tensor grad)
            {
                // Compute proper gradients of log probability w.r.t. mean and log_std
                if (networkOutput.Grad == null)
                {
                    networkOutput.Grad = new Tensor(new float[networkOutput.Data.Length], networkOutput.Shape);
                }
                
                int batchSize = grad.Shape[0];
                int actionDim = mean.Shape[1];
                const float EPSILON = 1e-6f;
                
                for (int b = 0; b < batchSize; b++)
                {
                    float upstreamGrad = grad.Data[b];
                    
                    for (int a = 0; a < actionDim; a++)
                    {
                        float u = preTanhAction.Data[b * actionDim + a];
                        float mu = mean.Data[b * actionDim + a];
                        float sigma = std.Data[b * actionDim + a];
                        
                        // Compute tanh correction derivative: d(log(1 - tanh²(u)))/d(u)
                        float tanhU = (float)Math.Tanh(u);
                        float tanhCorrectionDeriv = -2.0f * tanhU / (1.0f - tanhU * tanhU + EPSILON);
                        
                        // Gradient w.r.t. mean (μ)
                        // d(log_prob)/d(μ) = (u - μ)/σ² + d(tanh_correction)/d(u) * d(u)/d(μ)
                        // Since u = μ + σ*ε, we have d(u)/d(μ) = 1
                        float meanGrad = (u - mu) / (sigma * sigma) + tanhCorrectionDeriv;
                        
                        // Gradient w.r.t. log_std (log σ)
                        // d(log_prob)/d(log_σ) = -1 + ((u-μ)/σ)² + (u-μ)/σ * d(tanh_correction)/d(u)
                        // Since u = μ + σ*ε, we have d(u)/d(log_σ) = σ*ε = u - μ
                        float diff = (u - mu) / sigma;
                        float logStdGrad = -1.0f + diff * diff + diff * tanhCorrectionDeriv;
                        
                        // Apply upstream gradient and add to network output gradient
                        int meanIdx = b * (actionDim * 2) + a;
                        int stdIdx = b * (actionDim * 2) + actionDim + a;
                        
                        networkOutput.Grad.Data[meanIdx] += upstreamGrad * meanGrad;
                        networkOutput.Grad.Data[stdIdx] += upstreamGrad * logStdGrad;
                    }
                }
                
                if (networkOutput.GradFn != null)
                {
                    networkOutput.GradFn.Backward(networkOutput.Grad);
                }
            }
        }

        public int BufferCount => replayBuffer.Count;

        public void Save(string path)
        {
            throw new NotImplementedException("Save functionality coming soon!");
        }

        public void Load(string path)
        {
            throw new NotImplementedException("Load functionality coming soon!");
        }
    }
}
