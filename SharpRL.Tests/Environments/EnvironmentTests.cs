using SharpRL.Environments;
using Xunit;

namespace SharpRL.Tests.Environments;

/// <summary>
/// Comprehensive tests for CartPole environment
/// </summary>
public class CartPoleEnvironmentTests
{
    [Fact]
    public void CartPole_Reset_ReturnsValidState()
    {
        var env = new CartPoleEnvironment();
        var state = env.Reset();
        
        Assert.Equal(4, state.Length);
        Assert.All(state, val => Assert.False(float.IsNaN(val)));
    }

    [Fact]
    public void CartPole_Step_ValidTransitions()
    {
        var env = new CartPoleEnvironment();
        env.Reset();
        
        var result = env.Step(0);
        
        Assert.Equal(4, result.NextState.Length);
        Assert.True(result.Reward == 1.0f || result.Reward == 0.0f);
    }

    [Fact]
    public void CartPole_Reset_AfterDone_Restarts()
    {
        var env = new CartPoleEnvironment();
        env.Reset();
        
        while (!env.IsDone)
        {
            env.Step(0);
        }
        
        var newState = env.Reset();
        Assert.False(env.IsDone);
        Assert.Equal(4, newState.Length);
    }
}

/// <summary>
/// Tests for MountainCar environment
/// </summary>
public class MountainCarEnvironmentTests
{
    [Fact]
    public void MountainCar_Reset_ReturnsValidState()
    {
        var env = new MountainCarEnvironment();
        var state = env.Reset();
        
        Assert.Equal(2, state.Length);
    }

    [Fact]
    public void MountainCar_Step_ValidActions()
    {
        var env = new MountainCarEnvironment();
        env.Reset();
        
        for (int action = 0; action < 3; action++)
        {
            env.Reset();
            var result = env.Step(action);
            
            Assert.Equal(2, result.NextState.Length);
            Assert.True(result.Reward <= 0);
        }
    }
}

/// <summary>
/// Tests for Pendulum environment
/// </summary>
public class PendulumEnvironmentTests
{
    [Fact]
    public void Pendulum_Reset_ReturnsValidState()
    {
        var env = new PendulumEnvironment();
        var state = env.Reset();
        
        Assert.Equal(3, state.Length);
    }

    [Fact]
    public void Pendulum_Step_ContinuousActions()
    {
        var env = new PendulumEnvironment();
        env.Reset();
        
        var result = env.Step(new[] { 0.0f });
        
        Assert.Equal(3, result.NextState.Length);
        Assert.True(result.Reward <= 0);
    }
}
