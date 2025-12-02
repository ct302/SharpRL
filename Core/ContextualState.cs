using System;

namespace SharpRL.Core
{
    /// <summary>
    /// Wraps a physical state with a context to create a composite state representation.
    /// 
    /// NFL ANALOGY:
    /// Physical State = Field position (e.g., "Own 25-yard line")
    /// Context = Game situation (e.g., "Two-minute drill")
    /// Contextual State = "Own 25-yard line during two-minute drill"
    /// 
    /// The agent learns different Q-values for the same field position based on context.
    /// </summary>
    /// <typeparam name="TState">The underlying physical state type</typeparam>
    public readonly struct ContextualState<TState> : IEquatable<ContextualState<TState>>
        where TState : IEquatable<TState>
    {
        /// <summary>
        /// The underlying physical state (position, board state, etc.)
        /// </summary>
        public TState PhysicalState { get; }
        
        /// <summary>
        /// The operational context affecting decision-making
        /// </summary>
        public IContext Context { get; }
        
        public ContextualState(TState physicalState, IContext context)
        {
            PhysicalState = physicalState ?? throw new ArgumentNullException(nameof(physicalState));
            Context = context ?? throw new ArgumentNullException(nameof(context));
        }
        
        public bool Equals(ContextualState<TState> other)
        {
            return PhysicalState.Equals(other.PhysicalState) && 
                   Context.Equals(other.Context);
        }
        
        public override bool Equals(object obj)
        {
            return obj is ContextualState<TState> state && Equals(state);
        }
        
        public override int GetHashCode()
        {
            unchecked
            {
                // Combine hash codes using prime number multiplication
                // NFL ANALOGY: Like creating a unique play code from down + distance + field position
                int hash = 17;
                hash = hash * 31 + (PhysicalState?.GetHashCode() ?? 0);
                hash = hash * 31 + (Context?.GetHashCode() ?? 0);
                return hash;
            }
        }
        
        public override string ToString()
        {
            return $"[{Context.Name}] {PhysicalState}";
        }
        
        public static bool operator ==(ContextualState<TState> left, ContextualState<TState> right)
        {
            return left.Equals(right);
        }
        
        public static bool operator !=(ContextualState<TState> left, ContextualState<TState> right)
        {
            return !left.Equals(right);
        }
        
        /// <summary>
        /// Deconstruct for pattern matching and tuple assignment
        /// Example: var (state, context) = contextualState;
        /// </summary>
        public void Deconstruct(out TState physicalState, out IContext context)
        {
            physicalState = PhysicalState;
            context = Context;
        }
    }
}
