using SharpRL.AutoGrad;
using Xunit;

namespace SharpRL.Tests.Core;

public class TensorTests
{
    [Fact]
    public void Tensor_Creation_WithData()
    {
        var data = new float[] { 1, 2, 3, 4 };
        var tensor = new Tensor(data, new[] { 2, 2 });
        
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(2, tensor.Shape[1]);
        Assert.Equal(4, tensor.Data.Length);
    }

    [Fact]
    public void Tensor_Addition_CorrectResult()
    {
        var t1 = new Tensor(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var t2 = new Tensor(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });
        
        var result = t1 + t2;
        
        Assert.Equal(6f, result.Data[0]);
        Assert.Equal(8f, result.Data[1]);
        Assert.Equal(10f, result.Data[2]);
        Assert.Equal(12f, result.Data[3]);
    }

    [Fact]
    public void Tensor_Subtraction_CorrectResult()
    {
        var t1 = new Tensor(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });
        var t2 = new Tensor(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        
        var result = t1 - t2;
        
        Assert.Equal(4f, result.Data[0]);
        Assert.Equal(4f, result.Data[1]);
        Assert.Equal(4f, result.Data[2]);
        Assert.Equal(4f, result.Data[3]);
    }

    [Fact]
    public void Tensor_Multiplication_ElementWise()
    {
        var t1 = new Tensor(new float[] { 2, 3 }, new[] { 2, 1 });
        var t2 = new Tensor(new float[] { 4, 5 }, new[] { 2, 1 });
        
        var result = t1 * t2;
        
        Assert.Equal(8f, result.Data[0]);
        Assert.Equal(15f, result.Data[1]);
    }

    [Fact]
    public void Tensor_MatMul_CorrectShape()
    {
        var t1 = new Tensor(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var t2 = new Tensor(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });
        
        var result = t1.MatMul(t2);
        
        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(2, result.Shape[1]);
    }

    [Fact]
    public void Tensor_Reshape_CorrectShape()
    {
        var tensor = new Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var reshaped = tensor.Reshape(3, 2);
        
        Assert.Equal(3, reshaped.Shape[0]);
        Assert.Equal(2, reshaped.Shape[1]);
        Assert.Equal(6, reshaped.Data.Length);
    }

    [Fact]
    public void Tensor_Mean_CorrectValue()
    {
        var tensor = new Tensor(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var mean = tensor.Mean();
        
        Assert.Equal(2.5f, mean.Data[0], precision: 2);
    }

    [Fact]
    public void Tensor_RequiresGrad_Tracking()
    {
        var tensor = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 }, requiresGrad: true);
        
        Assert.True(tensor.RequiresGrad);
        // Grad is initialized lazily during backward, not at construction
        Assert.Null(tensor.Grad);
    }

    [Fact]
    public void Tensor_Transpose_2D()
    {
        var tensor = new Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var transposed = tensor.T();
        
        Assert.Equal(3, transposed.Shape[0]);
        Assert.Equal(2, transposed.Shape[1]);
    }

    [Fact]
    public void Tensor_Clone_IndependentCopy()
    {
        var original = new Tensor(new float[] { 1, 2, 3 }, new[] { 3 });
        var clone = original.Clone();
        
        clone.Data[0] = 999;
        
        Assert.Equal(1f, original.Data[0]);
        Assert.Equal(999f, clone.Data[0]);
    }
}
