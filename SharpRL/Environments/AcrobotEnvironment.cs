using System;
using System.Collections.Generic;
using SharpRL.Core;

namespace SharpRL.Environments
{
    /// <summary>
    /// Classic Acrobot-v1 Environment - two-link underactuated robot
    /// Goal: Swing the tip above a target height using only elbow torque
    /// </summary>
    public class AcrobotEnvironment : IEnvironment<float[], int>
    {
        private const float Link1Length = 1.0f;
        private const float Link2Length = 1.0f;
        private const float Link1Mass = 1.0f;
        private const float Link2Mass = 1.0f;
        private const float Link1COM = 0.5f;
        private const float Link2COM = 0.5f;
        private const float Link1Inertia = 1.0f;
        private const float Link2Inertia = 1.0f;
        private const float Gravity = 9.8f;
        private const float Dt = 0.2f;
        private const float MaxVel1 = 4.0f * MathF.PI;
        private const float MaxVel2 = 9.0f * MathF.PI;
        private const float AvailTorque = 1.0f;
        private const int MaxSteps = 500;

        private float theta1;
        private float theta2;
        private float theta1Dot;
        private float theta2Dot;
        private int stepCount;
        private float[] currentState;
        private bool isDone;
        private Random random;

        public float[] CurrentState => currentState;
        public bool IsDone => isDone;
        public int[] ObservationShape => new int[] { 6 };
        public int ActionSpaceSize => 3;
        public int StateSize => 6;
        public int ActionCount => 3;

        public AcrobotEnvironment(int? seed = null)
        {
            random = seed.HasValue ? new Random(seed.Value) : new Random();
            currentState = new float[6];
            Reset();
        }

        public float[] Reset()
        {
            theta1 = ((float)random.NextDouble() * 0.2f - 0.1f);
            theta2 = ((float)random.NextDouble() * 0.2f - 0.1f);
            theta1Dot = ((float)random.NextDouble() * 0.2f - 0.1f);
            theta2Dot = ((float)random.NextDouble() * 0.2f - 0.1f);
            stepCount = 0;
            isDone = false;
            UpdateState();
            return currentState;
        }

        public StepResult<float[]> Step(int action)
        {
            if (isDone)
                throw new InvalidOperationException("Episode done. Call Reset().");

            float torque = (action - 1) * AvailTorque;
            
            // RK4 integration for accurate dynamics
            float[] state = new float[] { theta1, theta2, theta1Dot, theta2Dot };
            float[] k1 = Dynamics(state, torque);
            
            float[] state2 = new float[4];
            for (int i = 0; i < 4; i++) state2[i] = state[i] + 0.5f * Dt * k1[i];
            float[] k2 = Dynamics(state2, torque);
            
            float[] state3 = new float[4];
            for (int i = 0; i < 4; i++) state3[i] = state[i] + 0.5f * Dt * k2[i];
            float[] k3 = Dynamics(state3, torque);
            
            float[] state4 = new float[4];
            for (int i = 0; i < 4; i++) state4[i] = state[i] + Dt * k3[i];
            float[] k4 = Dynamics(state4, torque);

            for (int i = 0; i < 4; i++)
                state[i] += (Dt / 6.0f) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);

            theta1 = WrapAngle(state[0]);
            theta2 = WrapAngle(state[1]);
            theta1Dot = Math.Clamp(state[2], -MaxVel1, MaxVel1);
            theta2Dot = Math.Clamp(state[3], -MaxVel2, MaxVel2);
            stepCount++;

            isDone = IsGoalReached() || stepCount >= MaxSteps;
            float reward = -1.0f;
            UpdateState();
            return new StepResult<float[]>(currentState, reward, isDone);
        }

        private float[] Dynamics(float[] state, float torque)
        {
            float th1 = state[0], th2 = state[1], dth1 = state[2], dth2 = state[3];
            float s1 = MathF.Sin(th1), s2 = MathF.Sin(th2);
            float s12 = MathF.Sin(th1 + th2), c2 = MathF.Cos(th2);

            float m11 = Link1Inertia + Link2Inertia + Link2Mass * Link1Length * Link1Length + 
                       2 * Link2Mass * Link1Length * Link2COM * c2;
            float m12 = Link2Inertia + Link2Mass * Link1Length * Link2COM * c2;
            float m22 = Link2Inertia;
            float det = m11 * m22 - m12 * m12;

            float h = -Link2Mass * Link1Length * Link2COM * s2;
            float c1 = h * (2 * dth1 * dth2 + dth2 * dth2);
            float c2_ = -h * dth1 * dth1;

            float phi1 = (Link1Mass * Link1COM + Link2Mass * Link1Length) * Gravity * s1 + 
                        Link2Mass * Link2COM * Gravity * s12;
            float phi2 = Link2Mass * Link2COM * Gravity * s12;

            float tau1 = 0, tau2 = torque;
            float ddth1 = (m22 * (tau1 - c1 - phi1) - m12 * (tau2 - c2_ - phi2)) / det;
            float ddth2 = (-m12 * (tau1 - c1 - phi1) + m11 * (tau2 - c2_ - phi2)) / det;

            return new float[] { dth1, dth2, ddth1, ddth2 };
        }

        private bool IsGoalReached()
        {
            float x = Link1Length * MathF.Sin(theta1) + Link2Length * MathF.Sin(theta1 + theta2);
            float y = -Link1Length * MathF.Cos(theta1) - Link2Length * MathF.Cos(theta1 + theta2);
            return y > 1.0f;
        }

        private float WrapAngle(float angle)
        {
            while (angle > MathF.PI) angle -= 2 * MathF.PI;
            while (angle < -MathF.PI) angle += 2 * MathF.PI;
            return angle;
        }

        private void UpdateState()
        {
            currentState[0] = MathF.Cos(theta1);
            currentState[1] = MathF.Sin(theta1);
            currentState[2] = MathF.Cos(theta2);
            currentState[3] = MathF.Sin(theta2);
            currentState[4] = theta1Dot;
            currentState[5] = theta2Dot;
        }

        public IEnumerable<int> GetValidActions()
        {
            yield return 0;
            yield return 1;
            yield return 2;
        }

        public void Render()
        {
            float x1 = Link1Length * MathF.Sin(theta1);
            float y1 = -Link1Length * MathF.Cos(theta1);
            float x2 = x1 + Link2Length * MathF.Sin(theta1 + theta2);
            float y2 = y1 - Link2Length * MathF.Cos(theta1 + theta2);

            Console.WriteLine($"\nAcrobot: θ1={theta1*180/MathF.PI:F1}° θ2={theta2*180/MathF.PI:F1}°");
            Console.WriteLine($"Tip: ({x2:F2}, {y2:F2}) Step: {stepCount}/{MaxSteps}");
            if (IsGoalReached()) Console.WriteLine("🎉 GOAL! 🎉");
        }
    }
}
