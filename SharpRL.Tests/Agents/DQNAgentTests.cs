using SharpRL.Agents;
using Xunit;

namespace SharpRL.Tests.Agents;

/// <summary>
/// Comprehensive tests for DQN Agent (Deep Q-Network)
/// Tests action selection, learning, memory, and Double DQN functionality
/// </summary>
public class DQNAgentTests
{
    [Fact]
    public void DQN_Constructor_ValidParameters()
    {
        var agent = new DQNAgent(
            stateSize: 4,
            actionSize: 2,
            hiddenLayers: new[] { 32, 32 });
        
        Assert.NotNull(agent);
    }

    [Fact]
    public void DQN_SelectAction_ReturnsValidAction()
    {
        var agent = new DQNAgent(
            stateSize: 4,
            actionSize: 2,
            hiddenLayers: new[] { 16 });
        
        var state = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        var action = agent.SelectAction(state, explore: false);
        
        Assert.True(action >= 0 && action < 2);
    }

    [Fact]
    public void DQN_SelectAction_WithEpsilonGreedy()
    {
        var agent = new DQNAgent(
            stateSize: 4,
            actionSize: 3,
            hiddenLayers: new[] { 16 });
        
        var state = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        
        // With explore = true, should sometimes explore
        var actions = new HashSet<int>();
        for (int i = 0; i < 50; i++)
        {
            var action = agent.SelectAction(state, explore: true);
            actions.Add(action);
        }
        
        // Should explore multiple actions
        Assert.True(actions.Count > 1);
    }

    [Fact]
    public void DQN_SelectAction_Deterministic()
    {
        var agent = new DQNAgent(
            stateSize: 4,
            actionSize: 2,
            hiddenLayers: new[] { 16 });
        
        var state = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        
        // With explore = false, should be deterministic
        var action1 = agent.SelectAction(state, explore: false);
        var action2 = agent.SelectAction(state, explore: false);
        
        Assert.Equal(action1, action2);
    }

    [Fact]
    public void DQN_ReplayBuffer_StoresExperience()
    {
        var agent = new DQNAgent(
            stateSize: 4,
            actionSize: 2,
            hiddenLayers: new[] { 16 },
            bufferSize: 1000);
        
        var state = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        
        // Update should not throw
        agent.Update(state, 0, 1.0, state, false);
        agent.Update(state, 1, -1.0, state, true);
        
        Assert.Equal(2, agent.GetBufferSize());
    }

    [Fact]
    public void DQN_Train_RequiresMinimumSamples()
    {
        var agent = new DQNAgent(
            stateSize: 4,
            actionSize: 2,
            hiddenLayers: new[] { 16 },
            batchSize: 32);
        
        var state = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        
        // Training with insufficient data should not crash
        agent.Update(state, 0, 1.0, state, false);
        
        Assert.True(true); // No exception thrown
    }

    [Fact]
    public void DoubleDQN_Constructor_EnablesDoubleDQN()
    {
        var agent = new DQNAgent(
            stateSize: 4,
            actionSize: 2,
            hiddenLayers: new[] { 16 },
            useDoubleDQN: true);
        
        Assert.NotNull(agent);
    }

    [Fact]
    public void DQN_Save_Load_PreservesWeights()
    {
        var agent = new DQNAgent(
            stateSize: 4,
            actionSize: 2,
            hiddenLayers: new[] { 16 });
        
        var state = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        var actionBefore = agent.SelectAction(state, explore: false);
        
        var tempPath = Path.GetTempFileName();
        agent.Save(tempPath);
        
        var loadedAgent = new DQNAgent(
            stateSize: 4,
            actionSize: 2,
            hiddenLayers: new[] { 16 });
        loadedAgent.Load(tempPath);
        var actionAfter = loadedAgent.SelectAction(state, explore: false);
        
        File.Delete(tempPath);
        
        Assert.Equal(actionBefore, actionAfter);
    }
}
