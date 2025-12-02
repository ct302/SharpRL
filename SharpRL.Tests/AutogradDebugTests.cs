using System;
using Xunit;
using Xunit.Abstractions;
using SharpRL.AutoGrad;
using SharpRL.NN.Layers;
using SharpRL.NN.Optimizers;
using SharpRL.NN.Loss;
using System.Linq;

namespace SharpRL.Tests
{
    /// <summary>
    /// Minimal tests to verify autograd gradient flow at each level
    /// </summary>
    public class AutogradDebugTests
    {
        private readonly ITestOutputHelper _output;

        public AutogradDebugTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void Test01_BasicGradient_Multiplication()
        {
            _output.WriteLine("=== TEST 1: Basic Multiplication Gradient ===");
            var x = new Tensor(new float[] { 3f }, new int[] { 1 }, requiresGrad: true);
            var two = new Tensor(new float[] { 2f }, new int[] { 1 }, requiresGrad: false);
            var y = x * two;
            _output.WriteLine($"x = {x.Data[0]}, y = x * 2 = {y.Data[0]}");
            _output.WriteLine($"y.RequiresGrad = {y.RequiresGrad}, y.GradFn = {y.GradFn?.GetType().Name ?? "null"}");
            y.Backward();
            _output.WriteLine($"x.Grad = {x.Grad?.Data[0]}");
            Assert.NotNull(x.Grad);
            Assert.Equal(2f, x.Grad!.Data[0], 4);
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test02_BasicGradient_MatMul()
        {
            _output.WriteLine("=== TEST 2: Matrix Multiplication Gradient ===");
            var x = new Tensor(new float[] { 1f, 2f }, new int[] { 1, 2 }, requiresGrad: true);
            var W = new Tensor(new float[] { 3f, 4f }, new int[] { 2, 1 }, requiresGrad: true);
            var y = x.MatMul(W);
            _output.WriteLine($"y = {y.Data[0]} (expect 11)");
            y.Backward();
            _output.WriteLine($"x.Grad = [{string.Join(", ", x.Grad?.Data ?? Array.Empty<float>())}] (expect [3, 4])");
            _output.WriteLine($"W.Grad = [{string.Join(", ", W.Grad?.Data ?? Array.Empty<float>())}] (expect [1, 2])");
            Assert.NotNull(x.Grad);
            Assert.NotNull(W.Grad);
            Assert.Equal(3f, x.Grad!.Data[0], 4);
            Assert.Equal(4f, x.Grad!.Data[1], 4);
            Assert.Equal(1f, W.Grad!.Data[0], 4);
            Assert.Equal(2f, W.Grad!.Data[1], 4);
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test03_BasicGradient_ReLU()
        {
            _output.WriteLine("=== TEST 3: ReLU Gradient ===");
            var x = new Tensor(new float[] { -1f, 2f }, new int[] { 2 }, requiresGrad: true);
            var y = x.ReLU();
            var loss = y.Sum();
            loss.Backward();
            _output.WriteLine($"x.Grad = [{string.Join(", ", x.Grad?.Data ?? Array.Empty<float>())}] (expect [0, 1])");
            Assert.NotNull(x.Grad);
            Assert.Equal(0f, x.Grad!.Data[0], 4);
            Assert.Equal(1f, x.Grad!.Data[1], 4);
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test04_BasicGradient_Mean()
        {
            _output.WriteLine("=== TEST 4: Mean Gradient ===");
            var x = new Tensor(new float[] { 1f, 2f, 3f, 4f }, new int[] { 4 }, requiresGrad: true);
            var y = x.Mean();
            y.Backward();
            _output.WriteLine($"x.Grad = [{string.Join(", ", x.Grad?.Data ?? Array.Empty<float>())}] (all should be 0.25)");
            Assert.NotNull(x.Grad);
            Assert.True(x.Grad!.Data.All(g => Math.Abs(g - 0.25f) < 0.001f));
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test05_LinearLayer_WeightGradient()
        {
            _output.WriteLine("=== TEST 5: Linear Layer Weight Gradient ===");
            var linear = new Linear(2, 1, bias: false);
            linear.Weight.Data[0] = 1f;
            linear.Weight.Data[1] = 1f;
            var x = new Tensor(new float[] { 2f, 3f }, new int[] { 1, 2 }, requiresGrad: false);
            var y = linear.Forward(x);
            _output.WriteLine($"y = {y.Data[0]} (expect 5)");
            y.Backward();
            _output.WriteLine($"Weight.Grad = [{string.Join(", ", linear.Weight.Grad?.Data ?? Array.Empty<float>())}] (expect [2, 3])");
            Assert.NotNull(linear.Weight.Grad);
            Assert.Equal(2f, linear.Weight.Grad!.Data[0], 4);
            Assert.Equal(3f, linear.Weight.Grad!.Data[1], 4);
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test06_LinearLayer_BiasGradient()
        {
            _output.WriteLine("=== TEST 6: Linear Layer Bias Gradient ===");
            var linear = new Linear(2, 2, bias: true);
            linear.Weight.Data[0] = 1f; linear.Weight.Data[1] = 0f;
            linear.Weight.Data[2] = 0f; linear.Weight.Data[3] = 1f;
            linear.Bias!.Data[0] = 0.5f;
            linear.Bias!.Data[1] = 0.5f;
            var x = new Tensor(new float[] { 2f, 3f }, new int[] { 1, 2 }, requiresGrad: false);
            var y = linear.Forward(x);
            var loss = y.Mean();
            loss.Backward();
            _output.WriteLine($"Bias.Grad = [{string.Join(", ", linear.Bias!.Grad?.Data ?? Array.Empty<float>())}]");
            Assert.NotNull(linear.Bias!.Grad);
            Assert.True(linear.Bias!.Grad!.Data.Any(g => g != 0), "Bias gradients should be non-zero");
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test07_MultiLayerNetwork_Gradient()
        {
            _output.WriteLine("=== TEST 7: Multi-Layer Network Gradient ===");
            var layer1 = new Linear(2, 3, bias: true);
            var layer2 = new Linear(3, 1, bias: true);
            var x = new Tensor(new float[] { 1f, 2f }, new int[] { 1, 2 }, requiresGrad: false);
            var h1 = layer1.Forward(x);
            var h2 = h1.ReLU();
            var y = layer2.Forward(h2);
            y.Backward();
            _output.WriteLine($"Layer1 Weight.Grad sum = {layer1.Weight.Grad?.Data.Sum():F4}");
            _output.WriteLine($"Layer1 Bias.Grad sum = {layer1.Bias?.Grad?.Data.Sum():F4}");
            _output.WriteLine($"Layer2 Weight.Grad sum = {layer2.Weight.Grad?.Data.Sum():F4}");
            _output.WriteLine($"Layer2 Bias.Grad sum = {layer2.Bias?.Grad?.Data.Sum():F4}");
            Assert.True(layer1.Weight.Grad!.Data.Any(g => g != 0), "Layer1 weight needs grads");
            Assert.True(layer1.Bias!.Grad!.Data.Any(g => g != 0), "Layer1 bias needs grads");
            Assert.True(layer2.Weight.Grad!.Data.Any(g => g != 0), "Layer2 weight needs grads");
            Assert.True(layer2.Bias!.Grad!.Data.Any(g => g != 0), "Layer2 bias needs grads");
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test08_SGD_ParameterUpdate()
        {
            _output.WriteLine("=== TEST 8: SGD Parameter Update ===");
            var layer = new Linear(2, 1, bias: true);
            var optimizer = new SGD(layer.Parameters(), learningRate: 0.1f);
            var initialWeight0 = layer.Weight.Data[0];
            var initialBias0 = layer.Bias!.Data[0];
            var x = new Tensor(new float[] { 1f, 1f }, new int[] { 1, 2 }, requiresGrad: false);
            var y = layer.Forward(x);
            var loss = y.Mean() * -1;
            loss.Backward();
            optimizer.Step();
            var newWeight0 = layer.Weight.Data[0];
            var newBias0 = layer.Bias!.Data[0];
            _output.WriteLine($"Weight change = {newWeight0 - initialWeight0}");
            _output.WriteLine($"Bias change = {newBias0 - initialBias0}");
            Assert.NotEqual(initialWeight0, newWeight0);
            Assert.NotEqual(initialBias0, newBias0);
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test09_Adam_ParameterUpdate()
        {
            _output.WriteLine("=== TEST 9: Adam Parameter Update ===");
            var layer = new Linear(2, 1, bias: true);
            var optimizer = new Adam(layer.Parameters(), learningRate: 0.01f);
            var initialWeight0 = layer.Weight.Data[0];
            var initialBias0 = layer.Bias!.Data[0];
            var x = new Tensor(new float[] { 1f, 1f }, new int[] { 1, 2 }, requiresGrad: false);
            var y = layer.Forward(x);
            var loss = y.Mean();
            loss.Backward();
            optimizer.Step();
            Assert.NotEqual(initialWeight0, layer.Weight.Data[0]);
            Assert.NotEqual(initialBias0, layer.Bias!.Data[0]);
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test10_BatchedInput_Gradients()
        {
            _output.WriteLine("=== TEST 10: Batched Input Gradients ===");
            var layer = new Linear(4, 2, bias: true);
            var x = new Tensor(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, new int[] { 3, 4 }, requiresGrad: false);
            var y = layer.Forward(x);
            var loss = y.Mean();
            loss.Backward();
            Assert.NotNull(layer.Weight.Grad);
            Assert.NotNull(layer.Bias!.Grad);
            Assert.True(layer.Weight.Grad!.Data.Any(g => g != 0), "Weight grads needed");
            Assert.True(layer.Bias!.Grad!.Data.Any(g => g != 0), "Bias grads needed");
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test11_LearningLoop_MSELoss()
        {
            _output.WriteLine("=== TEST 11: Learning Loop with MSELoss Class ===");
            var layer = new Linear(1, 1, bias: true);
            var optimizer = new SGD(layer.Parameters(), learningRate: 0.1f);
            var mseLoss = new MSELoss();
            var input = new Tensor(new float[] { 1f }, new int[] { 1, 1 }, requiresGrad: false);
            var target = new Tensor(new float[] { 5f }, new int[] { 1, 1 }, requiresGrad: false);
            _output.WriteLine($"Target: 5, Initial output: {layer.Forward(input).Data[0]:F4}");
            float initialLoss = float.MaxValue;
            float finalLoss = float.MaxValue;
            for (int epoch = 0; epoch < 100; epoch++)
            {
                optimizer.ZeroGrad();
                var output = layer.Forward(input);
                var loss = mseLoss.Forward(output, target);
                if (epoch == 0) initialLoss = loss.Data[0];
                if (epoch == 99) finalLoss = loss.Data[0];
                loss.Backward();
                optimizer.Step();
                if (epoch % 25 == 0)
                    _output.WriteLine($"Epoch {epoch}: output={output.Data[0]:F4}, loss={loss.Data[0]:F6}, W.Grad={layer.Weight.Grad?.Data[0]:F4}");
            }
            var finalOutput = layer.Forward(input).Data[0];
            _output.WriteLine($"Final: output={finalOutput:F4}, loss reduced {(1-finalLoss/initialLoss)*100:F1}%");
            Assert.True(finalLoss < initialLoss * 0.1f, "Loss should decrease by 90%+");
            Assert.True(Math.Abs(finalOutput - 5f) < 0.5f, "Should converge to target");
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test12_DQN_StyleTraining()
        {
            _output.WriteLine("=== TEST 12: DQN-Style Training ===");
            var layer1 = new Linear(4, 8, bias: true);
            var layer2 = new Linear(8, 2, bias: true);
            var allParams = layer1.Parameters().Concat(layer2.Parameters()).ToList();
            var optimizer = new Adam(allParams, learningRate: 0.01f);
            var mseLoss = new MSELoss();
            var state = new Tensor(new float[] { 0.1f, 0.2f, 0.3f, 0.4f }, new int[] { 1, 4 }, requiresGrad: false);
            var targetQ = new Tensor(new float[] { 1.0f, 0.5f }, new int[] { 1, 2 }, requiresGrad: false);
            _output.WriteLine($"Target Q: [{targetQ.Data[0]}, {targetQ.Data[1]}]");
            float initialLoss = float.MaxValue;
            float finalLoss = float.MaxValue;
            for (int epoch = 0; epoch < 200; epoch++)
            {
                optimizer.ZeroGrad();
                var h1 = layer1.Forward(state);
                var h2 = h1.ReLU();
                var qValues = layer2.Forward(h2);
                var loss = mseLoss.Forward(qValues, targetQ);
                if (epoch == 0) initialLoss = loss.Data[0];
                if (epoch == 199) finalLoss = loss.Data[0];
                loss.Backward();
                optimizer.Step();
                if (epoch % 50 == 0)
                    _output.WriteLine($"Epoch {epoch}: Q=[{qValues.Data[0]:F3},{qValues.Data[1]:F3}], loss={loss.Data[0]:F5}");
            }
            var h1f = layer1.Forward(state); var h2f = h1f.ReLU(); var qf = layer2.Forward(h2f);
            _output.WriteLine($"Final Q: [{qf.Data[0]:F3},{qf.Data[1]:F3}], loss reduced {(1-finalLoss/initialLoss)*100:F1}%");
            Assert.True(finalLoss < initialLoss * 0.1f, "Loss should drop 90%+");
            Assert.True(Math.Abs(qf.Data[0] - 1.0f) < 0.3f, "Q[0] should approach 1.0");
            Assert.True(Math.Abs(qf.Data[1] - 0.5f) < 0.3f, "Q[1] should approach 0.5");
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test13_SmoothL1Loss_Training()
        {
            _output.WriteLine("=== TEST 13: Smooth L1 (Huber) Loss Training ===");
            var layer = new Linear(2, 1, bias: true);
            var optimizer = new Adam(layer.Parameters(), learningRate: 0.01f);
            var smoothL1 = new SmoothL1Loss();
            var input = new Tensor(new float[] { 1f, 2f }, new int[] { 1, 2 }, requiresGrad: false);
            var target = new Tensor(new float[] { 3f }, new int[] { 1, 1 }, requiresGrad: false);
            float initialLoss = float.MaxValue;
            float finalLoss = float.MaxValue;
            for (int epoch = 0; epoch < 100; epoch++)
            {
                optimizer.ZeroGrad();
                var output = layer.Forward(input);
                var loss = smoothL1.Forward(output, target);
                if (epoch == 0) initialLoss = loss.Data[0];
                if (epoch == 99) finalLoss = loss.Data[0];
                loss.Backward();
                optimizer.Step();
            }
            var finalOutput = layer.Forward(input).Data[0];
            _output.WriteLine($"Final output: {finalOutput:F4}, target: 3.0");
            _output.WriteLine($"Loss reduced {(1-finalLoss/initialLoss)*100:F1}%");
            Assert.True(finalLoss < initialLoss * 0.1f, "Loss should drop significantly");
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test14_CrossEntropyLoss_Training()
        {
            _output.WriteLine("=== TEST 14: Cross Entropy Loss Training ===");
            var layer = new Linear(4, 3, bias: true);  // 3 classes
            var optimizer = new Adam(layer.Parameters(), learningRate: 0.05f);
            var ceLoss = new CrossEntropyLoss();
            var input = new Tensor(new float[] { 0.5f, 0.3f, 0.2f, 0.1f }, new int[] { 1, 4 }, requiresGrad: false);
            var targetClass = new Tensor(new float[] { 1f }, new int[] { 1 }, requiresGrad: false);  // Class 1
            float initialLoss = float.MaxValue;
            float finalLoss = float.MaxValue;
            for (int epoch = 0; epoch < 100; epoch++)
            {
                optimizer.ZeroGrad();
                var logits = layer.Forward(input);
                var loss = ceLoss.Forward(logits, targetClass);
                if (epoch == 0) initialLoss = loss.Data[0];
                if (epoch == 99) finalLoss = loss.Data[0];
                loss.Backward();
                optimizer.Step();
            }
            var finalLogits = layer.Forward(input);
            int predictedClass = 0;
            for (int i = 1; i < 3; i++)
                if (finalLogits.Data[i] > finalLogits.Data[predictedClass]) predictedClass = i;
            _output.WriteLine($"Logits: [{string.Join(", ", finalLogits.Data.Select(d => d.ToString("F3")))}]");
            _output.WriteLine($"Predicted class: {predictedClass}, Target: 1");
            _output.WriteLine($"Loss reduced {(1-finalLoss/initialLoss)*100:F1}%");
            Assert.True(finalLoss < initialLoss, "Loss should decrease");
            Assert.Equal(1, predictedClass);  // Should predict correct class
            _output.WriteLine("✓ PASSED");
        }

        [Fact]
        public void Test15_GradientClipping()
        {
            _output.WriteLine("=== TEST 15: Gradient Magnitude Check ===");
            var layer = new Linear(10, 5, bias: true);
            var x = new Tensor(Enumerable.Range(0, 10).Select(i => (float)i).ToArray(), new int[] { 1, 10 }, requiresGrad: false);
            var y = layer.Forward(x);
            var loss = y.Mean();
            loss.Backward();
            var maxGrad = layer.Weight.Grad!.Data.Max(Math.Abs);
            var gradSum = layer.Weight.Grad!.Data.Sum();
            _output.WriteLine($"Max gradient magnitude: {maxGrad:F4}");
            _output.WriteLine($"Gradient sum: {gradSum:F4}");
            Assert.True(maxGrad < 100, "Gradients should not explode");
            Assert.True(!float.IsNaN(maxGrad) && !float.IsInfinity(maxGrad), "Gradients should be finite");
            _output.WriteLine("✓ PASSED");
        }
    }
}
