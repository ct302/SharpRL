using System;
using System.Collections.Generic;
using SharpRL.Core;

namespace SharpRL.Environments
{
    public class MountainCarContinuousEnvironment : IEnvironment<float[], float[]>
    {
        private const float MinPosition = -1.2f;
        private const float MaxPosition = 0.6f;
        private const float MaxSpeed = 0.07f;
        private const float GoalPosition = 0.45f;
        private const float Power = 0.0015f;
        private const int MaxSteps = 999;

        private float position;
        private float velocity;
        private int stepCount;
        private float[] currentState;
        private bool isDone;
        private Random random;

        public float[] CurrentState => currentState;
        public bool IsDone => isDone;
        public int[] ObservationShape => new int[] { 2 };
        public int ActionSpaceSize => 1;
        public int StateSize => 2;
        public int ActionSize => 1;

        public MountainCarContinuousEnvironment(int? seed = null)
        {
            random = seed.HasValue ? new Random(seed.Value) : new Random();
            currentState = new float[2];
            Reset();
        }

        public float[] Reset()
        {
            position = ((float)random.NextDouble() * 0.6f - 0.6f);
            velocity = 0f;
            stepCount = 0;
            isDone = false;
            UpdateState();
            return currentState;
        }

        public StepResult<float[]> Step(float[] action)
        {
            if (isDone)
                throw new InvalidOperationException("Episode is done. Call Reset().");

            float force = Math.Clamp(action[0], -1f, 1f);
            velocity += force * Power - MathF.Cos(3 * position) * 0.0025f;
            velocity = Math.Clamp(velocity, -MaxSpeed, MaxSpeed);
            position += velocity;
            position = Math.Clamp(position, MinPosition, MaxPosition);

            if (position == MinPosition && velocity < 0)
                velocity = 0;

            stepCount++;
            bool goalReached = position >= GoalPosition;
            isDone = goalReached || stepCount >= MaxSteps;

            float reward = goalReached ? 100.0f : -0.1f * force * force;
            UpdateState();
            return new StepResult<float[]>(currentState, reward, isDone);
        }

        private void UpdateState()
        {
            currentState[0] = position;
            currentState[1] = velocity;
        }

        public IEnumerable<float[]> GetValidActions()
        {
            yield return new float[] { -1.0f };
            yield return new float[] { 0.0f };
            yield return new float[] { 1.0f };
        }

        public void Render()
        {
            int carPos = (int)((position - MinPosition) / (MaxPosition - MinPosition) * 60);
            carPos = Math.Clamp(carPos, 0, 59);
            Console.WriteLine("\nMountain Car (Continuous):");
            Console.Write("[");
            for (int i = 0; i < 60; i++)
            {
                if (i == carPos) Console.Write("🚗");
                else if (i > 50) { Console.Write("🚩"); break; }
                else Console.Write(MathF.Sin(3 * (MinPosition + i * (MaxPosition - MinPosition) / 60f)) > 0 ? "^" : "_");
            }
            Console.WriteLine("]");
            Console.WriteLine($"Step: {stepCount}/{MaxSteps}, Pos: {position:F3}, Vel: {velocity:F4}");
            if (position >= GoalPosition) Console.WriteLine("🎉 GOAL! 🎉");
        }

        public bool IsGoalReached() => position >= GoalPosition;
        public float GetPotentialEnergy() => MathF.Sin(3 * position);
    }
}
