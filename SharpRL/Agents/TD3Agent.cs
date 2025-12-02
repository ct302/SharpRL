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
    /// TD3 (Twin Delayed Deep Deterministic Policy Gradient) Agent
    /// State-of-the-art continuous control algorithm
    /// 
    /// NFL ANALOGY:
    /// If DDPG is like having one scout evaluate plays, TD3 is like having TWO independent 
    /// scouts evaluate the same play and taking the MORE CONSERVATIVE estimate. This prevents
    /// overconfidence - like when one scout thinks a receiver is wide open but another scout
    /// sees the safety lurking. TD3 listens to the pessimistic scout, which leads to more
    /// reliable long-term strategy.
    /// 
    /// THREE KEY INNOVATIONS:
    /// 1. Twin Critics (Clipped Double Q-Learning)
    ///    - Two independent Q-networks evaluate actions
    ///    - Use MINIMUM of both estimates (conservative)
    ///    - Reduces overestimation bias that plagued DDPG
    ///    
    /// 2. Delayed Policy Updates
    ///    - Update actor less frequently than critics (e.g., every 2 steps)
    ///    - Let critics stabilize before changing policy
    ///    - Like letting scouts finish analysis before changing playbook
    ///    
    /// 3. Target Policy Smoothing
    ///    - Add small noise to target actions
    ///    - Prevents exploiting Q-function errors
    ///    - Smooths value estimates across similar actions
    /// 
    /// WHEN TO USE TD3:
    /// - Continuous control tasks (robotics, autonomous vehicles)
    /// - When sample efficiency matters (fewer training steps)
    /// - When stability is critical (production systems)
    /// - When you need deterministic policies (no randomness at test time)
    /// 
    /// TD3 vs PPO:
    /// - TD3: Off-policy (learns from replay buffer), more sample efficient
    /// - PPO: On-policy (learns from fresh data), more stable on some tasks
    /// - TD3: Better for tasks with dense rewards
    /// - PPO: Better for tasks with sparse rewards
    /// </summary>
    public class TD3Agent : IContinuousAgent
    {
        // Main networks
        private readonly Module actorNetwork;           // Deterministic policy
        private readonly Module critic1Network;         // First Q-network
        private readonly Module critic2Network;         // Second Q-network (twin)
        
        // Target networks (slowly updated copies for stability)
        private readonly Module targetActorNetwork;
        private readonly Module targetCritic1Network;
        private readonly Module targetCritic2Network;
        
        // Optimizers
        private readonly Optimizer actorOptimizer;
        private readonly Optimizer critic1Optimizer;
        private readonly Optimizer critic2Optimizer;
        
        // Experience replay
        private readonly ReplayBuffer<float[], float[]> replayBuffer;
        
        // Hyperparameters
        private readonly int stateDim;
        private readonly int actionDim;
        private readonly float actionScale;         // Scale actions to environment bounds
        private readonly float gamma;               // Discount factor
        private readonly float tau;                 // Soft target update rate
        private readonly float policyNoise;         // Exploration noise (training)
        private readonly float targetNoise;         // Target policy smoothing noise
        private readonly float noiseClip;           // Clip target noise
        private readonly int policyDelay;           // Delay policy updates
        private readonly float learningRate;
        
        // Training state
        private int updateCounter;                  // Track updates for delayed policy
        private readonly Random random;

        /// <summary>
        /// Create a new TD3 agent
        /// </summary>
        public TD3Agent(
            int stateDim,
            int actionDim,
            int[] hiddenLayers = null!,
            float actionScale = 1.0f,
            int bufferSize = 100000,
            float learningRate = 3e-4f,
            float gamma = 0.99f,
            float tau = 0.005f,
            float policyNoise = 0.1f,
            float targetNoise = 0.2f,
            float noiseClip = 0.5f,
            int policyDelay = 2,
            int? seed = null)
        {
            this.stateDim = stateDim;
            this.actionDim = actionDim;
            this.actionScale = actionScale;
            this.gamma = gamma;
            this.tau = tau;
            this.policyNoise = policyNoise * actionScale;
            this.targetNoise = targetNoise * actionScale;
            this.noiseClip = noiseClip * actionScale;
            this.policyDelay = policyDelay;
            this.learningRate = learningRate;
            this.random = seed.HasValue ? new Random(seed.Value) : new Random();
            this.updateCounter = 0;

            hiddenLayers = hiddenLayers ?? new[] { 256, 256 };

            actorNetwork = BuildActorNetwork(stateDim, actionDim, hiddenLayers);
            critic1Network = BuildCriticNetwork(stateDim, actionDim, hiddenLayers);
            critic2Network = BuildCriticNetwork(stateDim, actionDim, hiddenLayers);

            targetActorNetwork = BuildActorNetwork(stateDim, actionDim, hiddenLayers);
            targetCritic1Network = BuildCriticNetwork(stateDim, actionDim, hiddenLayers);
            targetCritic2Network = BuildCriticNetwork(stateDim, actionDim, hiddenLayers);

            CopyNetworkWeights(actorNetwork, targetActorNetwork);
            CopyNetworkWeights(critic1Network, targetCritic1Network);
            CopyNetworkWeights(critic2Network, targetCritic2Network);

            actorOptimizer = new Adam(actorNetwork.Parameters(), learningRate);
            critic1Optimizer = new Adam(critic1Network.Parameters(), learningRate);
            critic2Optimizer = new Adam(critic2Network.Parameters(), learningRate);

            replayBuffer = new ReplayBuffer<float[], float[]>(bufferSize);
        }

        private Module BuildActorNetwork(int inputSize, int outputSize, int[] hiddenSizes)
        {
            var layers = new List<Module>();
            
            int prevSize = inputSize;
            foreach (int hiddenSize in hiddenSizes)
            {
                layers.Add(new Linear(prevSize, hiddenSize));
                layers.Add(new ReLU());
                prevSize = hiddenSize;
            }
            
            layers.Add(new Linear(prevSize, outputSize));
            layers.Add(new Tanh());
            
            return new Sequential(layers.ToArray());
        }

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

        public float[] SelectAction(float[] state, bool addNoise = true)
        {
            if (state.Length != stateDim)
                throw new ArgumentException($"State size {state.Length} doesn't match expected {stateDim}");

            var stateTensor = new Tensor(state, new[] { 1, stateDim });
            var actionTensor = actorNetwork.Forward(stateTensor);
            var action = actionTensor.Data.Take(actionDim).ToArray();

            if (addNoise)
            {
                for (int i = 0; i < action.Length; i++)
                {
                    float noise = (float)(random.NextDouble() * 2 - 1) * policyNoise;
                    action[i] = Math.Clamp(action[i] + noise, -1f, 1f);
                }
            }

            for (int i = 0; i < action.Length; i++)
            {
                action[i] *= actionScale;
            }

            return action;
        }

        public void Store(float[] state, float[] action, float reward, float[] nextState, bool done)
        {
            replayBuffer.Add(state, action, reward, nextState, done);
        }

        public void Train(int batchSize = 64)
        {
            if (replayBuffer.Count < batchSize)
                return;

            updateCounter++;

            var batch = replayBuffer.Sample(batchSize);
            
            // Extract states, actions, rewards, next states, and dones from batch
            var states = new Tensor(batch.SelectMany(e => e.State).ToArray(), new[] { batchSize, stateDim });
            var actions = new Tensor(batch.SelectMany(e => e.Action).ToArray(), new[] { batchSize, actionDim });
            var rewards = new Tensor(batch.Select(e => (float)e.Reward).ToArray(), new[] { batchSize, 1 });
            var nextStates = new Tensor(batch.SelectMany(e => e.NextState).ToArray(), new[] { batchSize, stateDim });
            var dones = new Tensor(batch.Select(e => e.Done ? 1f : 0f).ToArray(), new[] { batchSize, 1 });

            var targetActions = targetActorNetwork.Forward(nextStates);
            var targetActionsWithNoise = AddClippedNoise(targetActions);

            var targetStateActions = ConcatenateTensors(nextStates, targetActionsWithNoise);
            var targetQ1 = targetCritic1Network.Forward(targetStateActions);
            var targetQ2 = targetCritic2Network.Forward(targetStateActions);
            var targetQ = MinTensors(targetQ1, targetQ2);

            var target = ComputeTDTarget(rewards, targetQ, dones);

            var stateActions = ConcatenateTensors(states, actions);
            var currentQ1 = critic1Network.Forward(stateActions);
            var critic1Loss = new MSELoss().Forward(currentQ1, target);
            
            critic1Optimizer.ZeroGrad();
            critic1Loss.Backward();
            critic1Optimizer.Step();

            var currentQ2 = critic2Network.Forward(stateActions);
            var critic2Loss = new MSELoss().Forward(currentQ2, target);
            
            critic2Optimizer.ZeroGrad();
            critic2Loss.Backward();
            critic2Optimizer.Step();

            if (updateCounter % policyDelay == 0)
            {
                var newActions = actorNetwork.Forward(states);
                var stateNewActions = ConcatenateTensors(states, newActions);
                var actorQ = critic1Network.Forward(stateNewActions);
                
                var actorQMean = TensorMean(actorQ);
                var actorLoss = TensorScale(actorQMean, -1f);
                
                actorOptimizer.ZeroGrad();
                actorLoss.Backward();
                actorOptimizer.Step();

                SoftUpdateTargetNetwork(actorNetwork, targetActorNetwork);
                SoftUpdateTargetNetwork(critic1Network, targetCritic1Network);
                SoftUpdateTargetNetwork(critic2Network, targetCritic2Network);
            }
        }

        private Tensor AddClippedNoise(Tensor actions)
        {
            var noisyData = new float[actions.Data.Length];
            for (int i = 0; i < actions.Data.Length; i++)
            {
                float noise = (float)(random.NextDouble() * 2 - 1) * targetNoise;
                noise = Math.Clamp(noise, -noiseClip, noiseClip);
                noisyData[i] = Math.Clamp(actions.Data[i] + noise, -1f, 1f);
            }
            return new Tensor(noisyData, actions.Shape);
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

        public void Save(string path)
        {
            throw new NotImplementedException("Save functionality coming soon!");
        }

        public void Load(string path)
        {
            throw new NotImplementedException("Load functionality coming soon!");
        }

        // Gradient-preserving helper functions
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

        // Gradient function classes
        private class MeanFunction : Function
        {
            private readonly Tensor input;
            
            public MeanFunction(Tensor input) { this.input = input; }
            
            public override void Backward(Tensor grad)
            {
                if (input.Grad == null)
                    input.Grad = new Tensor(new float[input.Data.Length], input.Shape);
                
                float meanGrad = grad.Data[0] / input.Data.Length;
                for (int i = 0; i < input.Data.Length; i++)
                    input.Grad.Data[i] += meanGrad;
                
                input.GradFn?.Backward(input.Grad);
            }
        }
        
        private class ScaleFunction : Function
        {
            private readonly Tensor input;
            private readonly float scale;
            
            public ScaleFunction(Tensor input, float scale) { this.input = input; this.scale = scale; }
            
            public override void Backward(Tensor grad)
            {
                if (input.Grad == null)
                    input.Grad = new Tensor(new float[input.Data.Length], input.Shape);
                
                for (int i = 0; i < input.Data.Length; i++)
                    input.Grad.Data[i] += grad.Data[i] * scale;
                
                input.GradFn?.Backward(input.Grad);
            }
        }
    }
}
