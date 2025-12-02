using SharpRL.Agents;
using SharpRL.Environments;
using Xunit;
using Xunit.Abstractions;

namespace SharpRL.Tests.Debug;

/// <summary>
/// Diagnostic test to track DQN learning dynamics
/// </summary>
public class DQNDiagnostics
{
    private readonly ITestOutputHelper _output;

    public DQNDiagnostics(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void DQN_DetailedDiagnostics()
    {
        var env = new CartPoleEnvironment(seed: 42);
        var agent = new DQNAgent(
            stateSize: 4,
            actionSize: 2,
            hiddenLayers: new[] { 64, 64 },
            learningRate: 1e-4f,
            batchSize: 64,
            bufferSize: 10000,
            targetUpdateFreq: 100,
            discountFactor: 0.95f,
            seed: 42);

        _output.WriteLine("=== DQN DIAGNOSTIC TEST ===\n");
        _output.WriteLine($"Initial Epsilon: {agent.Epsilon:F4}");
        
        // Track initial Q-values for a sample state
        var sampleState = env.Reset();
        var initialQValues = agent.GetQValues(sampleState);
        _output.WriteLine($"Initial Q-values: [{string.Join(", ", initialQValues.Select(q => q.ToString("F4")))}]\n");
        
        for (int episode = 0; episode < 200; episode++)
        {
            var state = env.Reset();
            float episodeReward = 0;
            int steps = 0;
            
            while (!env.IsDone && steps < 500)
            {
                var action = agent.SelectAction(state, explore: true);
                var result = env.Step(action);
                agent.Update(state, action, result.Reward, result.NextState, result.Done);
                
                episodeReward += result.Reward;
                state = result.NextState;
                steps++;
            }
            
            // Log every 20 episodes
            if ((episode + 1) % 20 == 0)
            {
                var currentQValues = agent.GetQValues(sampleState);
                _output.WriteLine($"Episode {episode + 1}:");
                _output.WriteLine($"  Reward: {episodeReward:F2}");
                _output.WriteLine($"  Epsilon: {agent.Epsilon:F4}");
                _output.WriteLine($"  Buffer Size: {agent.BufferCount}");
                _output.WriteLine($"  Q-values: [{string.Join(", ", currentQValues.Select(q => q.ToString("F4")))}]");
                _output.WriteLine($"  Q-value change: {currentQValues.Sum() - initialQValues.Sum():F4}");
                _output.WriteLine("");
            }
        }
        
        var finalQValues = agent.GetQValues(sampleState);
        _output.WriteLine("=== FINAL STATE ===");
        _output.WriteLine($"Initial Q-values: [{string.Join(", ", initialQValues.Select(q => q.ToString("F4")))}]");
        _output.WriteLine($"Final Q-values: [{string.Join(", ", finalQValues.Select(q => q.ToString("F4")))}]");
        _output.WriteLine($"Total Q-value change: {finalQValues.Sum() - initialQValues.Sum():F4}");
    }
}
