# Explain:
> 该部分诠释了模型的一些代码问题，如下为讲解顺序

1. DenseNet
2. ResNet
# DenseNet
## 1. 为什么定义神经网络结构的时候需要继承nn.Module?
> torch.nn.Module模块使所有神经网络模块的基类，故您的模型也需要继承基类。并实现init和forward方法 

> 简而言之，nn.Module定义init(初始化),forward(前向传播)方法，你的模型继承并重写这两个方法。

>补充：

1. 需要继承nn.Module类，并实现forward方法，只要在nn.Module的子类中定义forward方法，backward函数就会被自动实现（利用autograd机制）
2. 一般把网络中可学习参数的层放在构造函数中__init__()，没有可学习参数的层如Relu层可以放在构造函数中，也可以不放在构造函数中（在forward函数中使用nn.Functional）
3. forward 方法中可以使用各种张量操作、函数等，来定义神经网络的前向传播过程。
4. 基于 nn.Module 构建的神经网络模型中，只支持使用 mini-batch 的数据输入方式。这意味着模型的输入应该是一个张量，其形状通常是 (batch_size, channels, height, width)，其中 batch_size 是一个 mini-batch 中样本的数量，channels 是图像的通道数，height 和 width 是图像的高度和宽度。这种输入方式符合卷积神经网络（CNN）等模型的设计，而不是单个样本的输入。
-----

## 2.为什么要super(DenseLayer,self).__init__()?
> 重写子类的__init__后不再自动继承父类的__init__，必须使用super把父类的拿过来

>简而言之，super加载了父类的一些属性和方法来使用。

详细参考：[为什么继承类时构造函数需要super(X, self).__init__()](https://blog.csdn.net/Evenrose/article/details/133814183?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171092109316800226513028%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=171092109316800226513028&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-4-133814183-null-null.142^v99^pc_search_result_base6&utm_term=%20%20%20%20%23%20nn.Module%E7%9A%84%E5%AD%90%E7%B1%BB%E5%87%BD%E6%95%B0%E5%BF%85%E9%A1%BB%E5%9C%A8%E6%9E%84%E9%80%A0%E5%87%BD%E6%95%B0%E4%B8%AD%E6%89%A7%E8%A1%8C%E7%88%B6%E7%B1%BB%E7%9A%84%E6%9E%84%E9%80%A0%E5%87%BD%E6%95%B0%20%20%20%20%20%20%20%20%20super%28LeNet%2C%20self%29.__init__%28%29%20%20%23%20%E7%AD%89%E4%BB%B7%E4%B8%8Enn.Module.__init__%28%29&spm=1018.2226.3001.4187)


# ResNet

## 1.make_layer具体的维度变换涉及self.inplanes,inplanes,width(！！很难理解)
> 总之，实例block时,只需传入上一个block的输入维度，和输出维度(inplanes*block.expansion)便可。具体block里会用固定的width解决通道问题。


