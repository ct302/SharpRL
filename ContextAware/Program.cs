using System;
using SharpRL.Examples;

/// <summary>
/// SharpRL - Context-Aware Q-Learning Integration Demo
/// 
/// This demonstrates the "danger" feature fully integrated into SharpRL.
/// Run this to see your context-aware agent in action!
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        try
        {
            // Run the complete integration demo
            DangerContextDemo.RunCompleteDemo();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n❌ Error: {ex.Message}");
            Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            Console.WriteLine("\nPress ENTER to exit...");
            Console.ReadLine();
        }
    }
}
