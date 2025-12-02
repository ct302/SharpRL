using System;
using System.Collections.Generic;
using SharpRL.Core;

namespace SharpRL.Environments
{
    /// <summary>
    /// Classic Pendulum Environment for continuous control
    /// 
    /// NFL ANALOGY:
    /// Imagine a QB trying to keep perfect throwing mechanics (the pendulum at vertical).
    /// The QB can apply small muscle adjustments (continuous torque) to maintain form.
    /// If mechanics drift too far, it costs energy and reduces throwing accuracy.
    /// 
    /// GOAL: Keep the pendulum upright using minimal continuous control force.
    /// This tests smooth, precise control rather than binary decisions.
    /// 
    /// STATE: [cos(θ), sin(θ), angular_velocity] - angle and speed of pendulum
    /// ACTION: [torque] - continuous force applied (-2.0 to +2.0)
    /// REWARD: -(θ² + 0.1*θ_dot² + 0.001*torque²) - penalizes deviation and effort
    /// </summary>
    public class PendulumEnvironment : IEnvironment<float[], float[]>
    {
        private const float MaxSpeed = 8f;
        private const float MaxTorque = 2f;
        private const float Dt = 0.05f;  // Time step
        private const float G = 10f;     // Gravity
        private const float M = 1f;      // Mass
        private const float L = 1f;      // Length

        private float theta;      // Angle (0 = upright, π = hanging down)
        private float thetaDot;   // Angular velocity
        private float[] currentState;
        private bool isDone;
        private Random random;

        public float[] CurrentState => currentState;
        public bool IsDone => isDone;
        public int[] ObservationShape => new int[] { 3 };
        public int ActionSpaceSize => 1;
        
        // Convenience properties for continuous control
        public int StateSize => 3;  // [cos(θ), sin(θ), angular_velocity]
        public int ActionSize => 1; // [torque]

        public PendulumEnvironment(int? seed = null)
        {
            random = seed.HasValue ? new Random(seed.Value) : new Random();
            currentState = new float[3];
            Reset();
        }

        public float[] Reset()
        {
            // Random initial angle and velocity
            theta = ((float)random.NextDouble() * 2f - 1f) * MathF.PI;
            thetaDot = ((float)random.NextDouble() * 2f - 1f) * 1f;
            isDone = false;
            UpdateState();
            return currentState;
        }

        public StepResult<float[]> Step(float[] action)
        {
            float torque = Math.Clamp(action[0], -MaxTorque, MaxTorque);

            // Physics: θ'' = -3g/(2L) * sin(θ) + 3/(mL²) * torque
            float thetaAcc = -3f * G / (2f * L) * MathF.Sin(theta) + 3f / (M * L * L) * torque;
            
            // Update velocity and angle
            thetaDot += thetaAcc * Dt;
            thetaDot = Math.Clamp(thetaDot, -MaxSpeed, MaxSpeed);
            theta += thetaDot * Dt;

            // Normalize angle to [-π, π]
            theta = NormalizeAngle(theta);

            // Compute reward: penalize deviation from upright and control effort
            float reward = -(theta * theta + 0.1f * thetaDot * thetaDot + 0.001f * torque * torque);

            // Pendulum never "done" - continuous control problem
            isDone = false;

            UpdateState();
            return new StepResult<float[]>(currentState, reward, isDone);
        }

        private void UpdateState()
        {
            // State is [cos(θ), sin(θ), angular_velocity]
            // This representation avoids discontinuity at ±π
            currentState[0] = MathF.Cos(theta);
            currentState[1] = MathF.Sin(theta);
            currentState[2] = thetaDot;
        }

        private float NormalizeAngle(float angle)
        {
            // Normalize to [-π, π]
            while (angle > MathF.PI) angle -= 2f * MathF.PI;
            while (angle < -MathF.PI) angle += 2f * MathF.PI;
            return angle;
        }

        public IEnumerable<float[]> GetValidActions()
        {
            // Continuous action space - infinite valid actions
            // Return some sample actions for demonstration
            yield return new float[] { -MaxTorque };
            yield return new float[] { 0f };
            yield return new float[] { MaxTorque };
        }

        public void Render()
        {
            // Console rendering of pendulum state
            float angleDegrees = theta * 180f / MathF.PI;
            Console.WriteLine($"Pendulum: θ={angleDegrees:F1}°, θ'={thetaDot:F2} rad/s");
        }
    }
}
