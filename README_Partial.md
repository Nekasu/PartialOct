训练代码如下：
```bash
echo "Start Time: $(date)" > output2.txt && python ./train.py | tee -a output2.txt
```

# 对于PartConv代码的说明

在[Partial Convolution原始论文](https://link.springer.com/10.1007/978-3-030-01252-6_6)中, 有计算过程如下：

$$
\begin{equation}
    x' = \begin{cases}
        W^T(X\odot M)\frac{\text{sum}(I)}{\text{sum}(M)} + b, &\quad \text{if sum}(M)>0\\
        0&\quad \text{otherwise}
    \end{cases}
\end{equation}
$$

但是这一步在实现时较为困难, 所以改成了以下等价表达式：
$$
\begin{equation}
    \begin{aligned}
        &x' = W^T(X\odot M)\cdot r + b, \quad \text{if sum}(M)>0\\
        &\text{其中}\quad r = \begin{cases}
            \frac{\text{sum}(I)}{\text{sum}(M)},&\quad \text{if sum}(M)>0\\
            0,&\quad \text{otherwise}
        \end{cases}
    \end{aligned}
\end{equation}
$$

这样就可以将复杂的卷积操作变成较为简单的系数$r$的操作


故而在forward函数中, 有第三步如下, 其中`torch.where`语句为系数`r`的操作语句：

```python
# 3. 计算系数矩阵 sum(I)/sum(M)
        # 3. 计算系数矩阵 sum(I)/sum(M)
        with torch.no_grad():
            self.sum_mask: torch.Tensor = self.conv_1(self.in_mask) # 利用一个其中参数全为1的卷积核与输入掩膜相乘, 算出每个像素的sum(M)
            self.sum_I: int = self.kernel_size[0] * self.kernel_size[-1] * 1# sum(I)是一个固定值, 从数值上与卷积核的窗口大小相等
            if (self.out.shape == self.sum_mask.shape):
                self.ratio: torch.Tensor = torch.where(self.sum_mask<=0, torch.tensor(0.0), self.sum_I/self.sum_mask)
                print(self.ratio)
            else:
                print("error!")
```

