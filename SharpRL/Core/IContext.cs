using System;

namespace SharpRL.Core
{
    /// <summary>
    /// Represents an operational context that affects agent decision-making.
    /// Think of it like game situations in football: Normal offense vs Two-minute drill vs Goal-line stand
    /// Each context can have completely different optimal strategies for the same physical state.
    /// </summary>
    public interface IContext : IEquatable<IContext>
    {
        /// <summary>
        /// Unique identifier for this context (0-based indexing recommended)
        /// </summary>
        int Id { get; }
        
        /// <summary>
        /// Human-readable name for debugging and logging
        /// </summary>
        string Name { get; }
        
        /// <summary>
        /// Optional: Priority level for context resolution when multiple contexts apply
        /// Higher values = higher priority (like danger overriding normal operations)
        /// </summary>
        int Priority { get; }
    }
    
    /// <summary>
    /// Simple built-in context implementation for common use cases
    /// </summary>
    public sealed class SimpleContext : IContext
    {
        public int Id { get; }
        public string Name { get; }
        public int Priority { get; }
        
        public SimpleContext(int id, string name, int priority = 0)
        {
            Id = id;
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Priority = priority;
        }
        
        public bool Equals(IContext? other)
        {
            return other != null && Id == other.Id;
        }
        
        public override bool Equals(object? obj)
        {
            return obj is IContext context && Equals(context);
        }
        
        public override int GetHashCode()
        {
            return Id.GetHashCode();
        }
        
        public override string ToString() => $"{Name} (ID: {Id})";
        
        // Pre-defined common contexts (like your Danger/Safe example)
        public static readonly IContext Normal = new SimpleContext(0, "Normal", priority: 0);
        public static readonly IContext Danger = new SimpleContext(1, "Danger", priority: 10);
        public static readonly IContext Critical = new SimpleContext(2, "Critical", priority: 20);
    }
}
