using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;
using SharpRL.Agents;
using SharpRL.Environments;
using SharpRL.Core;

namespace SharpRL.Tests.Debug
{
    public class MinimalDQNTest
    {
        private readonly ITestOutputHelper _output;

        public MinimalDQNTest(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void MinimalDQN_SingleEpisode_LogsGradients()
        {
            _output.WriteLine("=== MINIMAL DQN GRADIENT DIAGNOSTIC ===\n");

            // Setup
            var env = new CartPoleEnvironment();
            int stateDim = 4;
            int actionDim = 2;
            int hiddenDim = 64;
            float learningRate = 0.001f;
            float gamma = 0.99f;
            float epsilon = 1.0f;
            float epsilonDecay = 0.995f;
            float epsilonMin = 0.01f;
            int bufferSize = 10000;
            int batchSize = 32;
            int targetUpdateFreq = 100;

            var agent = new DQNAgent(
                stateSize: stateDim,
                actionSize: actionDim,
                hiddenLayers: new[] { hiddenDim },
                learningRate: learningRate,
                discountFactor: gamma,
                epsilon: epsilon,
                epsilonDecay: epsilonDecay,
                epsilonMin: epsilonMin,
                batchSize: batchSize,
                bufferSize: bufferSize,
                targetUpdateFreq: targetUpdateFreq,
                useDoubleDQN: false,
                seed: 42
            );
            
            var state = env.Reset();
            
            _output.WriteLine($"Initial State: [{string.Join(", ", state.Select(x => x.ToString("F4")))}]");
            _output.WriteLine($"Initial Epsilon: {agent.Epsilon:F4}");
            _output.WriteLine($"Learning Rate: {learningRate}");
            _output.WriteLine($"Batch Size: {batchSize}\n");

            // Capture initial weights
            var initialWeights = CaptureWeights(agent);
            _output.WriteLine("=== INITIAL WEIGHTS ===");
            LogWeights(initialWeights, "Initial");

            // Run single episode with detailed logging
            int step = 0;
            double totalReward = 0;
            bool done = false;

            while (!done && step < 500)
            {
                // Get action
                int action = agent.SelectAction(state);
                
                // Environment step
                var (nextState, reward, terminated) = env.Step(action);
                done = terminated;
                totalReward += reward;

                // Store transition
                agent.Remember(state, action, reward, nextState, done);

                // Try to learn (will only work if buffer has enough samples)
                if (agent.CanTrain())
                {
                    _output.WriteLine($"\n=== TRAINING STEP {step} (Buffer Size: {agent.BufferCount}) ===");
                    
                    // Capture weights before training
                    var beforeWeights = CaptureWeights(agent);
                    
                    // Get Q-values before training
                    var qValuesBefore = agent.GetQValues(state);
                    _output.WriteLine($"Q-Values Before: [{string.Join(", ", qValuesBefore.Select(x => x.ToString("F4")))}]");
                    
                    // Train 
                    agent.Update(state, action, reward, nextState, done);
                    
                    // Get Q-values after training
                    var qValuesAfter = agent.GetQValues(state);
                    _output.WriteLine($"Q-Values After:  [{string.Join(", ", qValuesAfter.Select(x => x.ToString("F4")))}]");
                    
                    // Calculate Q-value change
                    double qValueChange = 0;
                    for (int i = 0; i < qValuesBefore.Length; i++)
                    {
                        qValueChange += Math.Abs(qValuesAfter[i] - qValuesBefore[i]);
                    }
                    _output.WriteLine($"Total Q-Value Change: {qValueChange:F6}");
                    
                    // Capture weights after training
                    var afterWeights = CaptureWeights(agent);
                    
                    // Calculate weight changes
                    double totalWeightChange = 0;
                    for (int i = 0; i < beforeWeights.Count; i++)
                    {
                        double layerChange = 0;
                        for (int j = 0; j < beforeWeights[i].Length; j++)
                        {
                            double diff = Math.Abs(afterWeights[i][j] - beforeWeights[i][j]);
                            layerChange += diff;
                        }
                        totalWeightChange += layerChange;
                        _output.WriteLine($"Layer {i} Weight Change: {layerChange:E6} (avg: {layerChange / beforeWeights[i].Length:E6})");
                    }
                    
                    _output.WriteLine($"Total Weight Change: {totalWeightChange:E6}");
                    
                    // Log expectations
                    if (totalWeightChange < 1e-8)
                    {
                        _output.WriteLine("⚠️ WARNING: Weights barely changed! Possible issues:");
                        _output.WriteLine("  - Gradients are zero (gradient flow broken)");
                        _output.WriteLine("  - Gradients are too small (scaling issue)");
                        _output.WriteLine("  - Optimizer not applying updates");
                    }
                    else if (totalWeightChange < 1e-4)
                    {
                        _output.WriteLine("⚠️ WARNING: Weight changes are very small. Possible gradient scaling issue.");
                    }
                    else
                    {
                        _output.WriteLine("✓ Weights are updating normally");
                    }
                    
                    if (qValueChange < 1e-6)
                    {
                        _output.WriteLine("⚠️ WARNING: Q-values not changing! Network is not learning.");
                    }
                }

                state = nextState;
                step++;
            }

            _output.WriteLine($"\n=== EPISODE COMPLETE ===");
            _output.WriteLine($"Total Steps: {step}");
            _output.WriteLine($"Total Reward: {totalReward}");
            _output.WriteLine($"Buffer Size: {agent.BufferCount}");
            
            // Compare initial vs final weights
            var finalWeights = CaptureWeights(agent);
            _output.WriteLine("\n=== FINAL WEIGHT COMPARISON ===");
            double overallChange = 0;
            for (int i = 0; i < initialWeights.Count; i++)
            {
                double layerChange = 0;
                for (int j = 0; j < initialWeights[i].Length; j++)
                {
                    layerChange += Math.Abs(finalWeights[i][j] - initialWeights[i][j]);
                }
                overallChange += layerChange;
                _output.WriteLine($"Layer {i} Total Change: {layerChange:E6}");
            }
            _output.WriteLine($"Overall Weight Change: {overallChange:E6}");
            
            // Assertions
            Assert.True(agent.BufferCount >= batchSize, "Buffer should have enough samples");
            Assert.True(overallChange > 1e-6, $"Weights should change after training! Got: {overallChange:E6}");
        }

        private System.Collections.Generic.List<double[]> CaptureWeights(DQNAgent agent)
        {
            var weights = new System.Collections.Generic.List<double[]>();
            
            // Access the Q-network parameters
            var qNetwork = agent.GetQNetwork();
            var parameters = qNetwork.Parameters();
            
            foreach (var param in parameters)
            {
                var data = param.Data;
                var flatWeights = new double[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    flatWeights[i] = data[i];
                }
                weights.Add(flatWeights);
            }
            
            return weights;
        }

        private void LogWeights(System.Collections.Generic.List<double[]> weights, string label)
        {
            for (int i = 0; i < weights.Count; i++)
            {
                var w = weights[i];
                double mean = w.Average();
                double std = Math.Sqrt(w.Select(x => Math.Pow(x - mean, 2)).Average());
                double min = w.Min();
                double max = w.Max();
                
                _output.WriteLine($"{label} Layer {i}: mean={mean:F6}, std={std:F6}, min={min:F6}, max={max:F6}");
            }
        }
    }
}
