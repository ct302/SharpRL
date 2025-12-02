# SharpRL Testing & Benchmarking Suite

## Overview
Comprehensive unit tests and benchmarks for validating SharpRL library functionality and performance.

## Test Structure

### Core Tests (`SharpRL.Tests/Core/`)
- **TensorTests.cs** - Tensor operations, shapes, arithmetic
- **AutogradTests.cs** - Gradient computation, backpropagation

### Agent Tests (`SharpRL.Tests/Agents/`)  
- **DQNAgentTests.cs** - DQN functionality, memory, learning
- Additional agent tests to be added

### Environment Tests (`SharpRL.Tests/Environments/`)
- **EnvironmentTests.cs** - CartPole, MountainCar, Pendulum validation

### Benchmarks (`SharpRL.Tests/Benchmarks/`)
- **CartPoleBenchmarks.cs** - DQN & PPO on CartPole
- **PendulumBenchmarks.cs** - TD3, SAC & PPO on Pendulum

## Running Tests

```bash
# Run all tests
dotnet test SharpRL.Tests/SharpRL.Tests.csproj

# Run specific category
dotnet test --filter "FullyQualifiedName~Benchmarks"

# Run with verbose output
dotnet test -v detailed
```

## Benchmark Success Criteria

### CartPole (Discrete Actions)
- **DQN**: Avg reward >= 150 over final 100 episodes
- **PPO**: Avg reward >= 150 over final 100 episodes
- **Episodes**: 200-300 training episodes
- **Goal**: Keep pole balanced (500 max steps)

### Pendulum (Continuous Actions)
- **TD3**: Avg reward >= -400 over final 50 episodes
- **SAC**: Avg reward >= -400 over final 50 episodes  
- **PPO**: Avg reward >= -400 over final 50 episodes
- **Episodes**: 100-150 training episodes
- **Goal**: Swing up and balance (200 steps per episode)

## API Reference for Testing

### DQNAgent Constructor
```csharp
new DQNAgent(
    stateSize: 4,          // NOT stateDim
    actionSize: 2,         // NOT actionCount
    hiddenLayers: new[] { 64, 64 },  // Optional
    learningRate: 1e-3f,
    batchSize: 64,
    bufferSize: 10000)
```

### DQN Training Pattern
```csharp
var state = env.Reset();
while (!env.IsDone)
{
    var action = agent.SelectAction(state, trainMode: true);
    var (nextState, reward, done) = env.Step(action);
    
    replayBuffer.Add(state, action, reward, nextState, done);
    if (replayBuffer.Count >= batchSize)
    {
        agent.Train(replayBuffer.Sample(batchSize));
    }
    
    state = nextState;
}
```

### PPOAgent Training Pattern
```csharp
// Collect trajectories
var states = new List<float[]>();
var actions = new List<int>();
var rewards = new List<float>();
var nextStates = new List<float[]>();
var dones = new List<bool>();

var state = env.Reset();
while (!env.IsDone)
{
    var action = agent.SelectAction(state);
    var (nextState, reward, done) = env.Step(action);
    
    states.Add(state);
    actions.Add(action);
    rewards.Add(reward);
    nextStates.Add(nextState);
    dones.Add(done);
    
    state = nextState;
}

// Train on full trajectory
agent.Train(
    states.ToArray(),
    actions.ToArray(),
    rewards.ToArray(),
    nextStates.ToArray(),
    dones.ToArray());
```

## Current Status

### ✅ Completed
- Test project structure
- Core tensor tests
- Environment validation tests
- Benchmark framework (CartPole, Pendulum)

### ⚠️ In Progress
- Fixing API mismatches in agent tests
- Completing all 11 algorithm benchmarks
- Network layer unit tests

### 📋 To Do
- Replay buffer comprehensive tests
- Training infrastructure tests
- Performance regression tests
- Integration tests for full training loops

## Known Issues

1. **Test API Mismatches**: Some tests use incorrect parameter names
   - Fix: Use correct agent constructors (see API Reference above)
   
2. **Tensor Grad Access**: Tests try to use `tensor.Grad[0]` instead of `tensor.Grad.Data[0]`
   - Fix: Access gradients via Data property

3. **MatMul Signature**: Tests assume 2-arg MatMul, actual may differ
   - Fix: Check Tensor.MatMul signature in library

## Next Steps

1. Run `dotnet test` to identify remaining compilation errors
2. Fix agent test APIs to match actual library signatures
3. Complete all 11 algorithm benchmarks
4. Add integration tests for full training workflows
5. Document expected performance baselines

## Performance Baselines (To Be Established)

Once tests compile and run, establish baselines for:
- Training speed (episodes/second)
- Memory usage during training
- Convergence rates for each algorithm
- Final performance metrics on standard environments

---

**Testing Philosophy**: Tests should validate both correctness (does it work?) and performance (does it work well?). Benchmarks compare against established RL baselines like Stable-Baselines3.
