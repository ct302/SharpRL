using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpRL.Core;

namespace SharpRL.Environments
{
    /// <summary>
    /// GridWorld environment - classic RL testing ground
    /// Like a simplified football field with grid positions
    /// </summary>
    public class GridWorld : IEnvironment<(int x, int y), int>
    {
        // Actions: 0=Up, 1=Right, 2=Down, 3=Left (like play directions)
        public const int UP = 0;
        public const int RIGHT = 1;
        public const int DOWN = 2;
        public const int LEFT = 3;
        
        private readonly int width;
        private readonly int height;
        private readonly (int x, int y) startPos;
        private readonly (int x, int y) goalPos;
        private readonly HashSet<(int x, int y)> obstacles;
        private readonly Dictionary<(int x, int y), double> rewards;
        private (int x, int y) currentPos;
        private bool isDone;
        
        public (int x, int y) CurrentState => currentPos;
        public bool IsDone => isDone;
        public int[] ObservationShape => new[] { 2 }; // x, y coordinates
        public int ActionSpaceSize => 4;
        
        public GridWorld(
            int width = 5,
            int height = 5,
            (int x, int y)? start = null,
            (int x, int y)? goal = null,
            HashSet<(int x, int y)> obstacles = null!,
            Dictionary<(int x, int y), double> rewards = null!)
        {
            this.width = width;
            this.height = height;
            this.startPos = start ?? (0, 0);
            this.goalPos = goal ?? (width - 1, height - 1);
            this.obstacles = obstacles ?? new HashSet<(int x, int y)>();
            this.rewards = rewards ?? new Dictionary<(int x, int y), double>();
            
            // Default rewards
            if (!this.rewards.ContainsKey(goalPos))
            {
                this.rewards[goalPos] = 10.0; // Touchdown!
            }
            
            Reset();
        }
        
        public (int x, int y) Reset()
        {
            currentPos = startPos;
            isDone = false;
            return currentPos;
        }
        
        public StepResult<(int x, int y)> Step(int action)
        {
            if (isDone)
            {
                throw new InvalidOperationException("Episode is done. Call Reset()");
            }
            
            // Calculate new position based on action
            var newPos = action switch
            {
                UP => (currentPos.x, currentPos.y - 1),
                RIGHT => (currentPos.x + 1, currentPos.y),
                DOWN => (currentPos.x, currentPos.y + 1),
                LEFT => (currentPos.x - 1, currentPos.y),
                _ => currentPos
            };
            
            // Check boundaries and obstacles
            if (IsValidPosition(newPos))
            {
                currentPos = newPos;
            }
            // else stay in place (like running into the sideline)
            
            // Calculate reward
            float reward = rewards.TryGetValue(currentPos, out var r) ? (float)r : -0.1f; // Small penalty for each step
            
            // Check if done
            isDone = currentPos.Equals(goalPos);
            
            // Create info dictionary
            var info = new Dictionary<string, object>
            {
                ["position"] = currentPos,
                ["steps_taken"] = 1
            };
            
            return new StepResult<(int x, int y)>(currentPos, reward, isDone, info);
        }
        
        public IEnumerable<int> GetValidActions()
        {
            var validActions = new List<int>();
            
            // Check each direction
            if (IsValidPosition((currentPos.x, currentPos.y - 1))) validActions.Add(UP);
            if (IsValidPosition((currentPos.x + 1, currentPos.y))) validActions.Add(RIGHT);
            if (IsValidPosition((currentPos.x, currentPos.y + 1))) validActions.Add(DOWN);
            if (IsValidPosition((currentPos.x - 1, currentPos.y))) validActions.Add(LEFT);
            
            return validActions;
        }
        
        private bool IsValidPosition((int x, int y) pos)
        {
            return pos.x >= 0 && pos.x < width &&
                   pos.y >= 0 && pos.y < height &&
                   !obstacles.Contains(pos);
        }
        
        public void Render()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"\n=== GridWorld {width}x{height} ===");
            
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var pos = (x, y);
                    if (pos.Equals(currentPos))
                        sb.Append("A "); // Agent
                    else if (pos.Equals(goalPos))
                        sb.Append("G "); // Goal
                    else if (obstacles.Contains(pos))
                        sb.Append("# "); // Obstacle
                    else if (rewards.ContainsKey(pos))
                        sb.Append($"{rewards[pos]:F0} ");
                    else
                        sb.Append(". ");
                }
                sb.AppendLine();
            }
            
            Console.WriteLine(sb.ToString());
        }
    }
}