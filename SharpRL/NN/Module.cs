using System.Collections.Generic;
using SharpRL.AutoGrad;

namespace SharpRL.NN
{
    /// <summary>
    /// Base class for all neural network modules (layers, models, etc.).
    /// 
    /// NFL ANALOGY:
    /// Think of Module as a coaching unit:
    /// - Parameters = The playbook (learned strategies)
    /// - Forward = Execute the play
    /// - Train/Eval = Practice vs Game mode
    /// </summary>
    public abstract class Module
    {
        /// <summary>
        /// Whether the module is in training mode
        /// </summary>
        protected bool IsTraining { get; private set; } = true;

        /// <summary>
        /// Sets the module to training mode (like practice)
        /// </summary>
        public virtual Module Train()
        {
            IsTraining = true;
            return this;
        }

        /// <summary>
        /// Sets the module to evaluation mode (like game time)
        /// </summary>
        public virtual Module Eval()
        {
            IsTraining = false;
            return this;
        }

        /// <summary>
        /// Returns all trainable parameters (the playbook)
        /// </summary>
        public abstract List<Tensor> Parameters();

        /// <summary>
        /// Forward pass computation (execute the play)
        /// </summary>
        public abstract Tensor Forward(Tensor input);

        /// <summary>
        /// Clears gradients for all parameters (reset for new play)
        /// </summary>
        public virtual void ZeroGrad()
        {
            foreach (var param in Parameters())
            {
                param.ZeroGrad();
            }
        }

        /// <summary>
        /// Save module state
        /// </summary>
        public virtual Dictionary<string, float[]> StateDict()
        {
            var state = new Dictionary<string, float[]>();
            var parameters = Parameters();
            for (int i = 0; i < parameters.Count; i++)
            {
                state[$"param_{i}"] = parameters[i].Data.ToArray();
            }
            return state;
        }

        /// <summary>
        /// Load module state
        /// </summary>
        public virtual void LoadStateDict(Dictionary<string, float[]> state)
        {
            var parameters = Parameters();
            for (int i = 0; i < parameters.Count; i++)
            {
                if (state.TryGetValue($"param_{i}", out var data))
                {
                    Array.Copy(data, parameters[i].Data, data.Length);
                }
            }
        }
    }
}
