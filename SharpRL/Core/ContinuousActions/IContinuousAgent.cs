using System;

namespace SharpRL.Core.ContinuousActions
{
    /// <summary>
    /// Interface for agents that handle continuous action spaces
    /// 
    /// NFL ANALOGY:
    /// Discrete actions = choosing specific play calls from playbook (#1, #2, #3)
    /// Continuous actions = adjusting play parameters on the fly (blocking angle: 23.5°, 
    ///                      receiver route depth: 18.7 yards, QB drop back: 5.2 yards)
    /// 
    /// Continuous agents work in spaces where actions have infinite precision rather than
    /// discrete choices. Think adjusting a slider vs pushing a button.
    /// </summary>
    public interface IContinuousAgent
    {
        /// <summary>
        /// Select a continuous action given a state
        /// </summary>
        /// <param name="state">Current state observation</param>
        /// <param name="deterministic">If true, return mean action without exploration noise</param>
        /// <returns>Continuous action vector</returns>
        float[] SelectAction(float[] state, bool deterministic = false);

        /// <summary>
        /// Train the agent on collected experience
        /// </summary>
        /// <param name="states">Batch of states</param>
        /// <param name="actions">Batch of actions taken</param>
        /// <param name="rewards">Batch of rewards received</param>
        /// <param name="nextStates">Batch of next states</param>
        /// <param name="dones">Batch of done flags</param>
        void Train(float[][] states, float[][] actions, float[] rewards, float[][] nextStates, bool[] dones);

        /// <summary>
        /// Save agent parameters to file
        /// </summary>
        void Save(string path);

        /// <summary>
        /// Load agent parameters from file
        /// </summary>
        void Load(string path);
    }
}
