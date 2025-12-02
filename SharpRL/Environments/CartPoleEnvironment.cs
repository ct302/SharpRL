using System;
using System.Collections.Generic;
using SharpRL.Core;

namespace SharpRL.Environments
{
    /// <summary>
    /// Classic CartPole-v1 Environment for discrete control
    /// 
    /// NFL ANALOGY:
    /// Imagine a running back (the cart) trying to balance a football on a pole while running.
    /// The RB can only move left or right, and must keep the ball balanced.
    /// If the ball tilts too far or the RB runs out of bounds, it's a turnover!
    /// 
    /// GOAL: Keep the pole balanced upright by moving the cart left or right.
    /// 
    /// STATE: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    /// ACTION: 0 = Push Left, 1 = Push Right
    /// REWARD: +1 for every step the pole stays upright
    /// DONE: Pole angle > ±12°, cart position > ±2.4, or 500 steps reached
    /// 
    /// SUCCESS CRITERIA: Average reward of 195+ over 100 consecutive episodes
    /// </summary>
    public class CartPoleEnvironment : IEnvironment<float[], int>
    {
        // Physics constants (based on OpenAI Gym CartPole-v1)
        private const float Gravity = 9.8f;
        private const float MassCart = 1.0f;
        private const float MassPole = 0.1f;
        private const float TotalMass = MassCart + MassPole;
        private const float Length = 0.5f;  // Half-pole length
        private const float PoleMassLength = MassPole * Length;
        private const float ForceMag = 10.0f;
        private const float Tau = 0.02f;  // Time step (50 Hz)

        // Environment boundaries
        private const float ThetaThresholdRadians = 12f * MathF.PI / 180f;  // ±12 degrees
        private const float XThreshold = 2.4f;  // ±2.4 units
        private const int MaxSteps = 500;

        // State variables
        private float x;          // Cart position
        private float xDot;       // Cart velocity
        private float theta;      // Pole angle (0 = upright)
        private float thetaDot;   // Pole angular velocity
        private int stepCount;
        private float[] currentState;
        private bool isDone;
        private Random random;

        public float[] CurrentState => currentState;
        public bool IsDone => isDone;
        public int[] ObservationShape => new int[] { 4 };
        public int ActionSpaceSize => 2;  // Left or Right

        // Convenience properties
        public int StateSize => 4;  // [x, x_dot, theta, theta_dot]
        public int ActionCount => 2;

        public CartPoleEnvironment(int? seed = null)
        {
            random = seed.HasValue ? new Random(seed.Value) : new Random();
            currentState = new float[4];
            Reset();
        }

        public float[] Reset()
        {
            // Random initial state near equilibrium (small perturbations)
            x = ((float)random.NextDouble() * 2f - 1f) * 0.05f;
            xDot = ((float)random.NextDouble() * 2f - 1f) * 0.05f;
            theta = ((float)random.NextDouble() * 2f - 1f) * 0.05f;
            thetaDot = ((float)random.NextDouble() * 2f - 1f) * 0.05f;
            
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

            // Apply force based on action (0 = left, 1 = right)
            float force = action == 1 ? ForceMag : -ForceMag;
            // Physics simulation using semi-implicit Euler method
            // These equations come from the classic cart-pole dynamics
            
            float cosTheta = MathF.Cos(theta);
            float sinTheta = MathF.Sin(theta);

            // Temporary variable for readability
            float temp = (force + PoleMassLength * thetaDot * thetaDot * sinTheta) / TotalMass;
            
            // Angular acceleration of pole
            float thetaAcc = (Gravity * sinTheta - cosTheta * temp) / 
                           (Length * (4.0f / 3.0f - MassPole * cosTheta * cosTheta / TotalMass));
            
            // Linear acceleration of cart
            float xAcc = temp - PoleMassLength * thetaAcc * cosTheta / TotalMass;

            // Update velocities (Euler integration)
            xDot += Tau * xAcc;
            thetaDot += Tau * thetaAcc;

            // Update positions
            x += Tau * xDot;
            theta += Tau * thetaDot;

            stepCount++;

            // Check termination conditions
            bool outOfBounds = x < -XThreshold || x > XThreshold;
            bool poleAngleTooLarge = theta < -ThetaThresholdRadians || theta > ThetaThresholdRadians;
            bool maxStepsReached = stepCount >= MaxSteps;

            isDone = outOfBounds || poleAngleTooLarge || maxStepsReached;

            // CRITICAL: Reward is +1 for staying balanced, even on terminal step
            // Terminal states aren't "bad" - they're just endpoints
            // The agent already gets 0 future reward from terminal states via done flag
            float reward = 1.0f;

            UpdateState();
            return new StepResult<float[]>(currentState, reward, isDone);
        }

        private void UpdateState()
        {
            currentState[0] = x;
            currentState[1] = xDot;
            currentState[2] = theta;
            currentState[3] = thetaDot;
        }

        public IEnumerable<int> GetValidActions()
        {
            yield return 0;  // Push left
            yield return 1;  // Push right
        }

        public void Render()
        {
            // Simple console visualization
            int cartPos = (int)((x + XThreshold) / (2 * XThreshold) * 40);
            cartPos = Math.Clamp(cartPos, 0, 40);

            // Draw track
            Console.Write("|");
            for (int i = 0; i < 40; i++)
            {
                if (i == cartPos)
                    Console.Write("█");  // Cart
                else
                    Console.Write("-");
            }
            Console.WriteLine("|");

            // Draw pole (simplified)
            float angleDegrees = theta * 180f / MathF.PI;
            int polePos = cartPos + (int)(10 * MathF.Sin(theta));
            polePos = Math.Clamp(polePos, 0, 40);

            Console.Write(" ");
            for (int i = 0; i < polePos; i++)
                Console.Write(" ");
            Console.WriteLine(theta > 0 ? "/" : "\\");

            Console.WriteLine($"Step: {stepCount}, Angle: {angleDegrees:F1}°, Position: {x:F2}");
        }

        /// <summary>
        /// Check if the current episode is considered "solved"
        /// CartPole-v1 is solved when average reward is 195+ over 100 episodes
        /// </summary>
        public bool IsSolved(List<double> recentRewards)
        {
            if (recentRewards.Count < 100)
                return false;

            double avgReward = 0;
            for (int i = recentRewards.Count - 100; i < recentRewards.Count; i++)
            {
                avgReward += recentRewards[i];
            }
            avgReward /= 100;

            return avgReward >= 195.0;
        }
    }
}
