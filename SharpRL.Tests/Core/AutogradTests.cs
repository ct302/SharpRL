using SharpRL.AutoGrad;
using Xunit;

namespace SharpRL.Tests.Core;

public class AutogradTests
{
    [Fact]
    public void Backward_SimpleAddition_CorrectGradients()
    {
        var a = new Tensor(new float[] { 2 }, new[] { 1 }, requiresGrad: true);
        var b = new Tensor(new float[] { 3 }, new[] { 1 }, requiresGrad: true);
        
        var c = a + b;
        c.Backward();
        
        Assert.Equal(1f, a.Grad!.Data[0]);
        Assert.Equal(1f, b.Grad!.Data[0]);
    }

    [Fact]
    public void Backward_Multiplication_CorrectGradients()
    {
        var a = new Tensor(new float[] { 2 }, new[] { 1 }, requiresGrad: true);
        var b = new Tensor(new float[] { 3 }, new[] { 1 }, requiresGrad: true);
        
        var c = a * b;
        c.Backward();
        
        Assert.Equal(3f, a.Grad!.Data[0]); // dc/da = b
        Assert.Equal(2f, b.Grad!.Data[0]); // dc/db = a
    }

    [Fact]
    public void Backward_ChainRule_CorrectGradients()
    {
        var x = new Tensor(new float[] { 2 }, new[] { 1 }, requiresGrad: true);
        
        var y = x * x; // y = x^2
        var z = y + y; // z = 2y = 2x^2
        z.Backward();
        
        // dz/dx = 4x = 8
        Assert.Equal(8f, x.Grad!.Data[0], precision: 2);
    }

    [Fact]
    public void ZeroGrad_ClearsGradients()
    {
        var tensor = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        tensor.Grad = new Tensor(new float[] { 5, 5, 5 }, new[] { 3 });
        
        tensor.ZeroGrad();
        
        Assert.All(tensor.Grad.Data, g => Assert.Equal(0f, g));
    }

    [Fact]
    public void Backward_MultipleOperations()
    {
        var a = new Tensor(new float[] { 3 }, new[] { 1 }, requiresGrad: true);
        var b = new Tensor(new float[] { 2 }, new[] { 1 }, requiresGrad: true);
        
        var c = a + b;
        var d = c * a;
        d.Backward();
        
        // d = (a + b) * a = a^2 + a*b
        // dd/da = 2a + b = 6 + 2 = 8
        Assert.Equal(8f, a.Grad!.Data[0], precision: 2);
        // dd/db = a = 3
        Assert.Equal(3f, b.Grad!.Data[0], precision: 2);
    }

    [Fact]
    public void Backward_NonLeafTensor_DoesNotAccumulateGrad()
    {
        var a = new Tensor(new float[] { 2 }, new[] { 1 }, requiresGrad: true);
        var b = a + a;
        b.Backward();
        
        Assert.NotNull(a.Grad);
        // Intermediate tensors don't accumulate gradients
        Assert.Null(b.Grad);
    }
}
