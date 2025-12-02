using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpRL.Environments
{
    /// <summary>
    /// Enhanced GridWorld with enemy support for context-aware Q-Learning
    /// 
    /// NFL ANALOGY:
    /// - Grid = Football field
    /// - Agent = Running back
    /// - Goal = End zone
    /// - Enemies = Defensive players
    /// - Context = Defensive pressure (open field vs crowded)
    /// </summary>
    public class GridWorldWithEnemies
    {
        // Actions: 0=Up, 1=Right, 2=Down, 3=Left
        public const int UP = 0;
        public const int RIGHT = 1;
        public const int DOWN = 2;
        public const int LEFT = 3;
        
        private readonly int width;
        private readonly int height;
        private readonly (int x, int y) startPos;
        private readonly (int x, int y) goalPos;
        private readonly HashSet<(int x, int y)> obstacles;
        private (int x, int y) currentPos;
        private List<(int x, int y)> enemyPositions;
        private readonly Random random;
        private bool isDone;
        private int stepCount;
        
        public int Width => width;
        public int Height => height;
        public (int x, int y) CurrentState => currentPos;
        public (int x, int y) GoalPos => goalPos;
        public IReadOnlyList<(int x, int y)> EnemyPositions => enemyPositions;
        public bool IsDone => isDone;
        public int ActionSpaceSize => 4;
        
        public GridWorldWithEnemies(
            int width = 8,
            int height = 8,
            (int x, int y)? start = null,
            (int x, int y)? goal = null,
            List<(int x, int y)> initialEnemies = null,
            HashSet<(int x, int y)> obstacles = null,
            int? seed = null)
        {
            this.width = width;
            this.height = height;
            this.startPos = start ?? (0, 0);
            this.goalPos = goal ?? (width - 1, height - 1);
            this.obstacles = obstacles ?? new HashSet<(int x, int y)>();
            this.enemyPositions = initialEnemies ?? new List<(int x, int y)>();
            this.random = seed.HasValue ? new Random(seed.Value) : new Random();
            
            Reset();
        }
        
        public (int x, int y) Reset()
        {
            currentPos = startPos;
            isDone = false;
            stepCount = 0;
            return currentPos;
        }
        
        /// <summary>
        /// Execute an action and return (reward, nextState, done)
        /// </summary>
        public (double reward, (int x, int y) nextState, bool done) Step(int action)
        {
            if (isDone)
            {
                throw new InvalidOperationException("Episode is done. Call Reset()");
            }
            
            stepCount++;
            
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
            // If invalid, stay in place (collision with wall/obstacle)
            
            // Move enemies randomly
            MoveEnemiesRandomly();
            
            // Calculate reward and check terminal conditions
            double reward;
            
            // Check if reached goal
            if (currentPos.Equals(goalPos))
            {
                reward = 100.0;
                isDone = true;
                return (reward, currentPos, isDone);
            }
            
            // Check if hit obstacle (should not happen, but defensive)
            if (obstacles.Contains(currentPos))
            {
                reward = -100.0;
                isDone = true;
                return (reward, currentPos, isDone);
            }
            
            // Check if caught by enemy
            if (enemyPositions.Contains(currentPos))
            {
                reward = -100.0;
                isDone = true;
                return (reward, currentPos, isDone);
            }
            
            // Context-aware step penalty
            // In danger: small penalty (survival > speed)
            // In safe: larger penalty (speed matters)
            bool inDanger = IsInDanger(currentPos, dangerThreshold: 3);
            reward = inDanger ? -0.1 : -1.0;
            
            return (reward, currentPos, isDone);
        }
        
        /// <summary>
        /// Get all valid actions from current position
        /// </summary>
        public IEnumerable<int> GetValidActions((int x, int y) state)
        {
            var actions = new List<int>();
            
            if (IsValidPosition((state.x, state.y - 1))) actions.Add(UP);
            if (IsValidPosition((state.x + 1, state.y))) actions.Add(RIGHT);
            if (IsValidPosition((state.x, state.y + 1))) actions.Add(DOWN);
            if (IsValidPosition((state.x - 1, state.y))) actions.Add(LEFT);
            
            // Always return all 4 actions (even if some hit walls)
            // This is more common for Q-Learning
            return new[] { UP, RIGHT, DOWN, LEFT };
        }
        
        /// <summary>
        /// Check if position is in danger (enemy within threshold distance)
        /// </summary>
        public bool IsInDanger((int x, int y) position, int dangerThreshold = 3)
        {
            foreach (var enemy in enemyPositions)
            {
                if (GetManhattanDistance(position, enemy) <= dangerThreshold)
                    return true;
            }
            return false;
        }
        
        /// <summary>
        /// Get Manhattan distance between two positions
        /// </summary>
        public int GetManhattanDistance((int x, int y) pos1, (int x, int y) pos2)
        {
            return Math.Abs(pos1.x - pos2.x) + Math.Abs(pos1.y - pos2.y);
        }
        
        /// <summary>
        /// Heuristic: Move toward goal (for safe context)
        /// </summary>
        public int GetHeuristicMoveTowardsGoal((int x, int y) position)
        {
            int dx = goalPos.x - position.x;
            int dy = goalPos.y - position.y;
            
            // Move along axis with greater distance
            if (Math.Abs(dy) > Math.Abs(dx))
                return dy > 0 ? DOWN : UP;
            else if (dx != 0)
                return dx > 0 ? RIGHT : LEFT;
            else
                return dy > 0 ? DOWN : UP; // Default
        }
        
        /// <summary>
        /// Heuristic: Flee from nearest enemy (for danger context)
        /// </summary>
        public int GetHeuristicFlee((int x, int y) position)
        {
            if (enemyPositions.Count == 0)
                return GetHeuristicMoveTowardsGoal(position);
            
            // Find nearest enemy
            var nearestEnemy = enemyPositions[0];
            int minDistance = GetManhattanDistance(position, nearestEnemy);
            
            foreach (var enemy in enemyPositions)
            {
                int dist = GetManhattanDistance(position, enemy);
                if (dist < minDistance)
                {
                    minDistance = dist;
                    nearestEnemy = enemy;
                }
            }
            
            // Flee opposite direction from enemy
            int dx = position.x - nearestEnemy.x;
            int dy = position.y - nearestEnemy.y;
            
            // Move away along axis with greater threat
            if (Math.Abs(dy) > Math.Abs(dx))
                return dy > 0 ? DOWN : UP; // Move away from enemy vertically
            else
                return dx > 0 ? RIGHT : LEFT; // Move away from enemy horizontally
        }
        
        /// <summary>
        /// Move all enemies randomly (one step each)
        /// </summary>
        private void MoveEnemiesRandomly()
        {
            for (int i = 0; i < enemyPositions.Count; i++)
            {
                var enemy = enemyPositions[i];
                int randomAction = random.Next(0, 4);
                
                var newPos = randomAction switch
                {
                    UP => (enemy.x, enemy.y - 1),
                    RIGHT => (enemy.x + 1, enemy.y),
                    DOWN => (enemy.x, enemy.y + 1),
                    LEFT => (enemy.x - 1, enemy.y),
                    _ => enemy
                };
                
                // Only move if valid position
                if (IsValidPosition(newPos))
                {
                    enemyPositions[i] = newPos;
                }
            }
        }
        
        private bool IsValidPosition((int x, int y) pos)
        {
            return pos.x >= 0 && pos.x < width &&
                   pos.y >= 0 && pos.y < height &&
                   !obstacles.Contains(pos);
        }
        
        /// <summary>
        /// Render the current grid state to console
        /// </summary>
        public void Render()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"\n=== GridWorld {width}x{height} | Step {stepCount} ===");
            
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var pos = (x, y);
                    
                    if (pos.Equals(currentPos))
                        sb.Append("A "); // Agent
                    else if (enemyPositions.Contains(pos))
                        sb.Append("E "); // Enemy
                    else if (pos.Equals(goalPos))
                        sb.Append("G "); // Goal
                    else if (obstacles.Contains(pos))
                        sb.Append("# "); // Obstacle
                    else
                        sb.Append(". ");
                }
                sb.AppendLine();
            }
            
            // Show danger status
            bool inDanger = IsInDanger(currentPos, 3);
            sb.AppendLine($"Context: {(inDanger ? "⚠️ DANGER" : "✅ SAFE")}");
            sb.AppendLine($"Position: ({currentPos.x}, {currentPos.y})");
            
            Console.WriteLine(sb.ToString());
        }
        
        /// <summary>
        /// Add a new enemy at specified position
        /// </summary>
        public void AddEnemy((int x, int y) position)
        {
            if (IsValidPosition(position) && !enemyPositions.Contains(position))
            {
                enemyPositions.Add(position);
            }
        }
    }
}
