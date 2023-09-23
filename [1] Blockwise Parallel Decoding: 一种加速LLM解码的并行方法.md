# [1] Blockwise Parallel Decoding: 一种加速LLM解码的并行方法

## 背景

Blockwise Parallel Decoding for Deep Autoregressive Models 是发表在NIPS 2018的工作，
![](https://pica.zhimg.com/80/v2-930bae18194627b4053684b522eba277_720w.png?source=d16d100b)


Transformer模型中的self attention 虽然在训练阶段序列中元素是并行计算的，但是在模型做解码（decoding）时，由于采用自回归(auto-regressive)解码的方式，如果要生成长度为L 的序列，需要进行L次采样(sampling/decoding)。 
本文提出了blockwise parallel decoding，说白了，就是一次生成多个token，这样我们只需要 **<L** 次采样就能生成序列。

注意，模型的解码算法有很多，常用的包括greedy、beam search、top-k samping等，详情可以看这篇[HuggingFace的博客](https://huggingface.co/blog/how-to-generate)


本文提出的blockwise parallel decoding仅针对greedy decoding做加速，不适合其他decoding算法。

简单回顾下greedy decoding，以机器翻译任务为例，假设输入序列是 $x=(x_1,...,x_n)$ ，目标序列是 $y=(y_1,...,y_m)$ ，自回归模型 $p(y|x)$ ，

$$log\ p(y|x)=\sum_{j=0}^{m-1}log(y_{j+1}|y_{\leq j},x) $$

模型预测的序列是 $y^{*}=argmax\ p(y|x)$.

如果使用greedy decoding, $`\hat{y}_{j+1}=argmax p(y_{j+1}|\hat{y}_{\leq j}, x)`$.

## Blockwise Parallel Decoding

为了描述方便，Blockwise Parallel Decoding就简称BPD了, 对于上面的模型$`p`$, 由于它是预测下一个token的，记为$`p_1`$, 假设我们还有$`K-1`$个模型，每个模型负责预测第$`k`$个token。


以下图为例，$`K=3`$, $`p_1`$负责预测下一个token, $`p_2`$负责预测后面第2个token, $`p_3`$负责预测第三个 token。这样三个模型并行预测，我们一次就可以生成3个token。

![](https://picx.zhimg.com/80/v2-895be03d7fcdbd2d2762ed91465e6a02_720w.jpg?source=d16d100b)

BPD在做推理(inference)时分了三个阶段：
* 预测(predict), 利用$`K`$个模型一次生成$`K`$个初始 token
* 验证(verify), 利用$`p_1`$重新预测$`K`$个token，这一步看下面的图解释
* 接受(accept), 对于预测阶段得到的$`K`$个初始token，选择前$`k`$个作为真正的预测结果

整个过程可以用下面这幅图表示：
![](https://picx.zhimg.com/80/v2-504b31a8a40e839b4f15bceef86aa644_720w.png?source=d16d100b)

在预测阶段，$`K`$个模型可以并行预测，在验证阶段，对于$`K`$个预测结果，组成batch，实现合适的attention mask，$p_1$能够一次性预测$`K`$个token做验证。

综上，在**理想情况下**（每次接受K个预测结果）可以把生成长度$`m`$的序列所需要的解码次数降低为$`\frac{2m}{K}`$. 并且这样得到的结果和greedy decoding完全一致！

毕竟理想很丰满，现实很骨感，实际情况中如果想更快，还可以做近似，比如在验证阶段，只要预测阶段得到的token在$`p_1`$的top N中就行，这样可以让接受的序列更长，减少decoding次数。
