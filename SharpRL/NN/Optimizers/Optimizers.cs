using System;
using System.Collections.Generic;
using SharpRL.AutoGrad;

namespace SharpRL.NN.Optimizers
{
    /// <summary>
    /// Base class for all optimizers
    /// </summary>
    public abstract class Optimizer
    {
        protected List<Tensor> parameters;
        protected float learningRate;

        public Optimizer(List<Tensor> parameters, float learningRate)
        {
            this.parameters = parameters;
            this.learningRate = learningRate;
        }

        /// <summary>
        /// Clears gradients for all parameters
        /// </summary>
        public virtual void ZeroGrad()
        {
            foreach (var param in parameters)
            {
                param.ZeroGrad();
            }
        }

        /// <summary>
        /// Updates parameters based on gradients
        /// </summary>
        public abstract void Step();
    }

    /// <summary>
    /// Stochastic Gradient Descent (SGD) optimizer
    /// 
    /// NFL ANALOGY:
    /// Like adjusting your playbook after each game:
    /// - Learning rate = How much you change based on one game
    /// - Momentum = Keeping successful plays from previous games
    /// </summary>
    public class SGD : Optimizer
    {
        private float momentum;
        private Dictionary<Tensor, float[]>? velocities;

        public SGD(List<Tensor> parameters, float learningRate = 0.01f, float momentum = 0f) 
            : base(parameters, learningRate)
        {
            this.momentum = momentum;
            if (momentum > 0)
            {
                velocities = new Dictionary<Tensor, float[]>();
                foreach (var param in parameters)
                {
                    velocities[param] = new float[param.Size];
                }
            }
        }

        public override void Step()
        {
            foreach (var param in parameters)
            {
                if (param.Grad == null)
                    continue;

                if (momentum > 0 && velocities != null)
                {
                    // v = momentum * v - lr * grad
                    // param = param + v
                    var velocity = velocities[param];
                    for (int i = 0; i < param.Size; i++)
                    {
                        velocity[i] = momentum * velocity[i] - learningRate * param.Grad.Data[i];
                        param.Data[i] += velocity[i];
                    }
                }
                else
                {
                    // param = param - lr * grad
                    for (int i = 0; i < param.Size; i++)
                    {
                        param.Data[i] -= learningRate * param.Grad.Data[i];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Adam optimizer (Adaptive Moment Estimation)
    /// 
    /// NFL ANALOGY:
    /// Like having both offensive and defensive coordinators:
    /// - First moment (mean) = Average direction plays are working
    /// - Second moment (variance) = How consistent each play is
    /// - Adapts learning rate per parameter based on history
    /// </summary>
    public class Adam : Optimizer
    {
        private float beta1;
        private float beta2;
        private float epsilon;
        private int step;
        private Dictionary<Tensor, float[]> m; // First moment
        private Dictionary<Tensor, float[]> v; // Second moment

        public Adam(List<Tensor> parameters, float learningRate = 0.001f, 
                   float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
            : base(parameters, learningRate)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
            this.step = 0;

            m = new Dictionary<Tensor, float[]>();
            v = new Dictionary<Tensor, float[]>();

            foreach (var param in parameters)
            {
                m[param] = new float[param.Size];
                v[param] = new float[param.Size];
            }
        }

        public override void Step()
        {
            step++;

            foreach (var param in parameters)
            {
                if (param.Grad == null)
                    continue;

                var m_param = m[param];
                var v_param = v[param];

                for (int i = 0; i < param.Size; i++)
                {
                    float grad = param.Grad.Data[i];

                    // Update biased first moment
                    m_param[i] = beta1 * m_param[i] + (1 - beta1) * grad;

                    // Update biased second moment
                    v_param[i] = beta2 * v_param[i] + (1 - beta2) * grad * grad;

                    // Bias correction
                    float m_hat = m_param[i] / (1 - (float)Math.Pow(beta1, step));
                    float v_hat = v_param[i] / (1 - (float)Math.Pow(beta2, step));

                    // Update parameter
                    param.Data[i] -= learningRate * m_hat / ((float)Math.Sqrt(v_hat) + epsilon);
                }
            }
        }
    }

    /// <summary>
    /// RMSprop optimizer
    /// Like Adam but without the first moment tracking
    /// </summary>
    public class RMSprop : Optimizer
    {
        private float alpha;
        private float epsilon;
        private Dictionary<Tensor, float[]> v;

        public RMSprop(List<Tensor> parameters, float learningRate = 0.01f, 
                      float alpha = 0.99f, float epsilon = 1e-8f)
            : base(parameters, learningRate)
        {
            this.alpha = alpha;
            this.epsilon = epsilon;

            v = new Dictionary<Tensor, float[]>();
            foreach (var param in parameters)
            {
                v[param] = new float[param.Size];
            }
        }

        public override void Step()
        {
            foreach (var param in parameters)
            {
                if (param.Grad == null)
                    continue;

                var v_param = v[param];

                for (int i = 0; i < param.Size; i++)
                {
                    float grad = param.Grad.Data[i];

                    // Update running average of squared gradients
                    v_param[i] = alpha * v_param[i] + (1 - alpha) * grad * grad;

                    // Update parameter
                    param.Data[i] -= learningRate * grad / ((float)Math.Sqrt(v_param[i]) + epsilon);
                }
            }
        }
    }
}
