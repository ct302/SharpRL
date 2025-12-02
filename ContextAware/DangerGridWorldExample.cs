using System;
using System.Collections.Generic;
using SharpRL.Core;
using SharpRL.Agents;

namespace SharpRL.Examples
{
    /// <summary>
    /// Example: GridWorld agent with Danger context awareness
    /// 
    /// SCENARIO:
    /// - Agent navigates an 8x8 grid to reach a goal
    /// - Enemies patrol the grid randomly
    /// - Two contexts: Safe (far from enemies) and Danger (near enemies)
    /// 
    /// LEARNING OBJECTIVES:
    /// - In SAFE context: Learn to reach goal efficiently (prioritize speed)
    /// - In DANGER context: Learn to avoid enemies (prioritize survival)
    /// 
    /// HEURISTICS:
    /// - Safe + Unexplored: Move toward goal
    /// - Danger + Unexplored: Flee from nearest enemy
    /// </summary>
    public class DangerGridWorldExample
    {
        // Example environment state
        public class GridState : IEquatable<GridState>
        {
            public int Position { get; }
            public int Width { get; }
            
            public GridState(int position, int width)
            {
                Position = position;
                Width = width;
            }
            
            public int Row => Position / Width;
            public int Col => Position % Width;
            
            public bool Equals(GridState other)
            {
                return other != null && Position == other.Position && Width == other.Width;
            }
            
            public override bool Equals(object obj) => obj is GridState state && Equals(state);
            public override int GetHashCode() => Position.GetHashCode();
            public override string ToString() => $"Pos{Position} ({Row},{Col})";
        }
        
        // Actions: Up, Down, Left, Right
        public enum GridAction { Up = 0, Down = 1, Left = 2, Right = 3 }
        
        public static void RunExample()
        {
            const int gridWidth = 8;
            const int gridHeight = 8;
            const int goalPosition = 63; // Bottom-right corner
            const int dangerThreshold = 3; // Manhattan distance for danger detection
            
            // Enemy positions (will move randomly in real implementation)
            var enemyPositions = new List<int> { 43, 58 };
            
            // Context detector: Check if any enemy is within danger threshold
            Func<GridState, IContext> contextDetector = (state) =>
            {
                foreach (var enemyPos in enemyPositions)
                {
                    int manhattanDist = GetManhattanDistance(state.Position, enemyPos, gridWidth);
                    if (manhattanDist <= dangerThreshold)
                        return SimpleContext.Danger;
                }
                return SimpleContext.Normal;
            };
            
            // Action provider
            Func<GridState, IEnumerable<GridAction>> getActions = (state) =>
            {
                return new[] { GridAction.Up, GridAction.Down, GridAction.Left, GridAction.Right };
            };
            
            // Heuristic: Move toward goal when safe
            Func<GridState, GridAction> safeHeuristic = (state) =>
            {
                int goalRow = goalPosition / gridWidth;
                int goalCol = goalPosition % gridWidth;
                int dy = goalRow - state.Row;
                int dx = goalCol - state.Col;
                
                // Move along axis with greater distance
                if (Math.Abs(dy) > Math.Abs(dx))
                    return dy > 0 ? GridAction.Down : GridAction.Up;
                else
                    return dx > 0 ? GridAction.Right : GridAction.Left;
            };
            
            // Heuristic: Flee from nearest enemy when in danger
            Func<GridState, GridAction> dangerHeuristic = (state) =>
            {
                // Find nearest enemy
                int nearestEnemy = enemyPositions[0];
                int minDist = GetManhattanDistance(state.Position, nearestEnemy, gridWidth);
                
                foreach (var enemyPos in enemyPositions)
                {
                    int dist = GetManhattanDistance(state.Position, enemyPos, gridWidth);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestEnemy = enemyPos;
                    }
                }
                
                // Flee away from enemy
                int enemyRow = nearestEnemy / gridWidth;
                int enemyCol = nearestEnemy % gridWidth;
                int dy = enemyRow - state.Row;
                int dx = enemyCol - state.Col;
                
                // Move away along axis with greater threat
                if (Math.Abs(dy) > Math.Abs(dx))
                    return dy > 0 ? GridAction.Up : GridAction.Down; // Opposite direction
                else
                    return dx > 0 ? GridAction.Left : GridAction.Right; // Opposite direction
            };
            
            // Create heuristics dictionary
            var heuristics = new Dictionary<IContext, Func<GridState, GridAction>>
            {
                { SimpleContext.Normal, safeHeuristic },
                { SimpleContext.Danger, dangerHeuristic }
            };
            
            // Create the context-aware Q-Learning agent
            var agent = new ContextAwareQLearningAgent<GridState, GridAction>(
                contextDetector: contextDetector,
                getActionsFunc: getActions,
                heuristics: heuristics,
                learningRate: 0.1,
                discountFactor: 0.99,
                epsilon: 1.0,
                epsilonDecay: 0.0001,
                epsilonMin: 0.01
            );
            
            Console.WriteLine("=== Context-Aware Q-Learning Agent Created ===");
            Console.WriteLine($"Grid Size: {gridWidth}x{gridHeight}");
            Console.WriteLine($"Goal Position: {goalPosition}");
            Console.WriteLine($"Enemy Positions: {string.Join(", ", enemyPositions)}");
            Console.WriteLine($"Danger Threshold: {dangerThreshold} (Manhattan distance)");
            Console.WriteLine("\nAgent will learn:");
            Console.WriteLine("  - SAFE context: Efficient pathfinding to goal");
            Console.WriteLine("  - DANGER context: Evasive maneuvers to avoid enemies");
            Console.WriteLine("\nFor unexplored states, agent uses instinctive heuristics:");
            Console.WriteLine("  - SAFE: Move toward goal");
            Console.WriteLine("  - DANGER: Flee from nearest enemy");
            Console.WriteLine("\n=== Training would begin here ===");
            Console.WriteLine("(Integrate with your existing GridWorldEnvironment)");
            
            // Example of action selection
            var testState = new GridState(position: 35, width: gridWidth);
            var testContext = contextDetector(testState);
            var action = agent.SelectAction(testState, explore: false);
            
            Console.WriteLine($"\nExample: State {testState.Position} → Context: {testContext.Name} → Action: {action}");
        }
        
        private static int GetManhattanDistance(int pos1, int pos2, int width)
        {
            int row1 = pos1 / width;
            int col1 = pos1 % width;
            int row2 = pos2 / width;
            int col2 = pos2 % width;
            return Math.Abs(row1 - row2) + Math.Abs(col1 - col2);
        }
    }
}
