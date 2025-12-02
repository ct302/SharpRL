using System;
using System.Collections.Generic;
using SharpRL.Core;

namespace SharpRL.Environments
{
    /// <summary>
    /// Classic MountainCar-v0 Environment for discrete control
    /// 
    /// NFL ANALOGY:
    /// Imagine a punt returner stuck at the bottom of a valley (the 10-yard line).
    /// The returner isn't strong enough to run straight up the hill to the goal.
    /// Instead, they must rock back and forth, building momentum like a pendulum,
    /// until they have enough speed to crest the hill and reach the goal!
    /// 
    /// GOAL: Reach the flag at the top of the hill (position ≥ 0.5) using momentum.
    /// 
    /// STATE: [position, velocity]
    /// ACTION: 0 = Accelerate Left, 1 = No Acceleration, 2 = Accelerate Right
    /// REWARD: -1 for every step (encourages reaching goal quickly)
    /// DONE: Position ≥ 0.5 (reached flag) or 200 steps reached
    /// 
    /// KEY INSIGHT: Direct action won't work! Must build momentum by going backwards first.
    /// SUCCESS CRITERIA: Reach the flag consistently (position ≥ 0.5)
    /// </summary>
    public class MountainCarEnvironment : IEnvironment<float[], int>
    {
        // Environment constants (based on OpenAI Gym MountainCar-v0)
        private const float MinPosition = -1.2f;
        private const float MaxPosition = 0.6f;
        private const float MaxSpeed = 0.07f;
        private const float GoalPosition = 0.5f;
        private const float GoalVelocity = 0f;
        private const float Force = 0.001f;        private const float Gravity = 0.0025f;
        private const int MaxSteps = 200;

        // State variables
        private float position;
        private float velocity;
        private int stepCount;
        private float[] currentState;
        private bool isDone;
        private Random random;

        public float[] CurrentState => currentState;
        public bool IsDone => isDone;
        public int[] ObservationShape => new int[] { 2 };
        public int ActionSpaceSize => 3;  // Left, None, Right

        // Convenience properties
        public int StateSize => 2;  // [position, velocity]
        public int ActionCount => 3;

        public MountainCarEnvironment(int? seed = null)
        {
            random = seed.HasValue ? new Random(seed.Value) : new Random();
            currentState = new float[2];
            Reset();
        }

        public float[] Reset()
        {
            // Start at random position in valley with zero velocity
            position = ((float)random.NextDouble() * 0.6f - 0.6f);  // Range: [-1.2, -0.6]
            velocity = 0f;
            stepCount = 0;
            isDone = false;
            UpdateState();
            return currentState;
        }

        public StepResult<float[]> Step(int action)
        {
            if (isDone)
            {
                throw new InvalidOperationException("Episode is done. Call Reset() to start a new episode.");
            }

            // Apply force based on action (0 = left, 1 = none, 2 = right)
            float force = 0f;
            if (action == 0) force = -Force;
            else if (action == 2) force = Force;

            // Update velocity with force and gravity
            // Gravity component depends on the slope: sin(3 * position)
            velocity += force + Gravity * MathF.Cos(3 * position);
            velocity = Math.Clamp(velocity, -MaxSpeed, MaxSpeed);

            // Update position
            position += velocity;
            position = Math.Clamp(position, MinPosition, MaxPosition);

            // If hit left bound, reset velocity
            if (position == MinPosition && velocity < 0)
            {
                velocity = 0;
            }

            stepCount++;

            // Check if goal reached or max steps
            bool goalReached = position >= GoalPosition;
            bool maxStepsReached = stepCount >= MaxSteps;
            isDone = goalReached || maxStepsReached;

            // Reward: -1 for each step (encourages efficiency)
            float reward = -1.0f;
            if (goalReached) reward = 0.0f;  // No penalty on success step

            UpdateState();
            return new StepResult<float[]>(currentState, reward, isDone);
        }

        private void UpdateState()
        {
            currentState[0] = position;
            currentState[1] = velocity;
        }

        public IEnumerable<int> GetValidActions()
        {
            yield return 0;  // Accelerate left
            yield return 1;  // No acceleration
            yield return 2;  // Accelerate right
        }

        public void Render()
        {
            // Simple ASCII visualization
            int carPos = (int)((position - MinPosition) / (MaxPosition - MinPosition) * 60);
            carPos = Math.Clamp(carPos, 0, 59);

            // Draw mountain profile (simplified sine curve)
            Console.WriteLine("\nMountain Car:");
            Console.Write("[");
            for (int i = 0; i < 60; i++)
            {
                if (i == carPos)
                {
                    Console.Write("🚗");  // Car
                }
                else if (i > 50)
                {
                    Console.Write("🚩");  // Flag at goal
                    break;
                }
                else
                {
                    // Draw mountain slope
                    float pos = MinPosition + i * (MaxPosition - MinPosition) / 60f;
                    float height = MathF.Sin(3 * pos);
                    Console.Write(height > 0 ? "^" : "_");
                }
            }
            Console.WriteLine("]");
            Console.WriteLine($"Step: {stepCount}/200, Position: {position:F3}, Velocity: {velocity:F4}");
            
            if (position >= GoalPosition)
            {
                Console.WriteLine("🎉 GOAL REACHED! 🎉");
            }
        }

        /// <summary>
        /// Get the height of the mountain at a given position (for visualization)
        /// </summary>
        public float GetHeight(float pos)
        {
            return MathF.Sin(3 * pos);
        }

        /// <summary>
        /// Check if the car has reached the goal
        /// </summary>
        public bool IsGoalReached()
        {
            return position >= GoalPosition;
        }
    }
}
