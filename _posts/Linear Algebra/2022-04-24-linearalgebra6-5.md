---
layout: single
title:  "6.5 Least Squares Problems"
categories: [Linear Algebra]
tag: [Linear Algebra, Least Squares Problems]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---









이번 포스트에서는 least-squares problems에 대해 알아보도록 하겠습니다.



<br/>



### 1) Least-Squares Problems



<br/>

#### (1) Least-Squares Problems

<br/>



다음의 linear system을 생각해봅시다.


$$
A\boldsymbol{x} = \boldsymbol{b}
$$




다음 linear system은 $A, \boldsymbol{b}$에 따라 3가지 종류의 solution이 존재할 수 있습니다. solution이 없거나(inconsistent), solution이 하나만 존재하거나, solution이 무수히 많이 존재하는 경우입니다. 이 중에서 solution이 존재하지 않는 경우를 생각해봅시다. 

만약 위 system이 solution이 존재하지 않는 경우, 다음으로 생각해볼 수 있는 것은 


$$
A\boldsymbol x \approx \boldsymbol b
$$


를 만족하는 $\boldsymbol x$를 찾는 것입니다. **즉 $A\boldsymbol{x}$와 $\boldsymbol{b}$가 가장 비슷하게 만들어주는 $\boldsymbol{x}$를 solution으로 대체하는 것입니다.** 그렇다면 $A\boldsymbol x$와 $\boldsymbol b$가 비슷하다는 것을 어떻게 정의할까요? **바로 distance를 이용합니다.**


$$
\|A\boldsymbol x - \boldsymbol b \|
$$


즉, $A\boldsymbol x$와 $\boldsymbol b$의 distance가 가장 가까워지는 $\boldsymbol x$를 solution으로 대체할 수 있습니다.

**The least squares problem은 다음 distance**


$$
\|A\boldsymbol x - \boldsymbol b \|
$$


**를 최소화하는 $\boldsymbol{x}$를 찾는 문제입니다.**





<br/>

#### (2) Least-squares solution

<br/>

위의 least squares problem의 solution은 다음과 같이 정의합니다.



<br/>

**Definition : Least-squares solution**



If $A$ is $m\times n$ matrix and $\boldsymbol b$ is in $\mathbb R^n$, a least-squares solution of $A\boldsymbol x = \boldsymbol{b}$ is an $\hat{\boldsymbol{x}}$ in $\mathbb R^n$ such that


$$
\|\boldsymbol{b}-A\hat{\boldsymbol{x}}\| \leq \|\boldsymbol{b}-A\boldsymbol{x}\|
$$


for all $\boldsymbol{x} \in \mathbb R^n$



즉 $\|\boldsymbol{b}-A\boldsymbol{x}\|$를 최소화시키는 $\boldsymbol{x}$가 least-squares solution이 됩니다. 



만약 


$$
A\boldsymbol{x} = \boldsymbol{b}
$$


가 solution이 존재한다면, least squares solution은 위 system의 solution이 됩니다. 이는 해당 system의 solution이


$$
\|\boldsymbol{b} -A\boldsymbol{x}\|=0
$$


을 만족하기 때문입니다. 



지금까지 least squares problem과 solution에 대해 알아봤습니다. 그렇다면 least-squares solution을 어떻게 찾을 수 있을까요?



<br/>

#### (3) How to find the solution

<br/>



least-squares solution을 찾는 방법에서의 핵심은 $A\boldsymbol{x}$의 의미를 파악하는 것입니다.



$A\boldsymbol{x}$는 $A$의 column의 linear combination으로 표현된 벡터입니다. 즉 모든 $\boldsymbol x \in \mathbb R^n$에 대해서


$$
A\boldsymbol{x} \in ColA
$$


입니다. $A\boldsymbol{x} =\boldsymbol{b}$가 solution이 없는 경우는


$$
\boldsymbol{b} \notin ColA
$$


인 것을 뜻합니다. 따라서, $A\boldsymbol{x} = \boldsymbol{b}$의 least squares solution을 찾는 것, 즉


$$
\|\boldsymbol{b}-A\hat{\boldsymbol{x}}\| \leq \|\boldsymbol{b}-A\boldsymbol{x}\|
$$


를 만족시키는 $\hat{\boldsymbol{x}}$를 찾는 것은 $ColA$에서 $\boldsymbol{b}$와 가장 가까운 벡터를 만들어주는 $\hat{\boldsymbol{x}}$를 찾는 것과 같습니다. 

$ColA$ 또한 $\mathbb R^m$의 subspace이므로, $\boldsymbol{b}$와 $ColA$와 가장 가까운 벡터는 projection을 통해 만들어줄 수 있습니다. 즉


$$
A\boldsymbol{x} = \hat{\boldsymbol{b}} = proj_{ColA}\boldsymbol{b}
$$


를 만족하는 $\boldsymbol{x}$가 위 least-squares problem의 solution이 됩니다. 



여기까지 least-squares problem of $A\boldsymbol{x}=\boldsymbol{b}$를 푸는 것은


$$
A\boldsymbol{x}=proj_{ColA}\boldsymbol{b} = \hat{\boldsymbol{b}}
$$


를 푸는 것과 같다는 것을 밝혔습니다. 다음의 식을 조금 더 자세히 살펴보도록 합시다. 


$$
\hat{\boldsymbol{b}} =proj_{ColA}\boldsymbol{b}
$$


일 때, 


$$
(\boldsymbol{b} - \hat{\boldsymbol{b}}) \perp \hat{\boldsymbol{b}}
$$


를 만족합니다. $\hat{\boldsymbol{b}}$는 $A$의 column의 linear combination으로 이루어져 있기 때문에 $A$를


$$
A = \begin{bmatrix} \boldsymbol{a_1} & ... & \boldsymbol{a_n} \end{bmatrix}
$$


으로 정의하면


$$
\boldsymbol{a_j} \cdot (\boldsymbol{b}-\hat{\boldsymbol{b}}) = 0 \ \ for \ \ j=1,...,n
$$


을 만족합니다. 이를 조금 더 정리하면


$$
\boldsymbol{a_j}^T(\boldsymbol{b}-\hat{\boldsymbol{b}}) =0
$$


을 만족합니다. $j=1, ..., n$에 대해 성립하므로 이를 matrix와 vector의 곱으로 바꾸면


$$
A^T(\boldsymbol{b}-\hat{\boldsymbol{b}}) =0
$$


을 만족합니다. 그런데, $\hat{\boldsymbol{b}}=A\hat{\boldsymbol{x}}$이므로


$$
A^T(\boldsymbol{b}-A\hat{\boldsymbol{x}}) = 0
$$


이 되어


$$
A^TA\hat{\boldsymbol{x}}=A^T\boldsymbol{b}
$$


를 만족합니다. 즉 $A\boldsymbol{x}=\boldsymbol{b}$의 least squares problem을 푸는 것은


$$
A^TA{\boldsymbol{x}}=A^T\boldsymbol{b}
$$


를 푸는 문제와 같게 됩니다. 여기서 만약 $A^TA$가 invertible하다면 least-squares solution은


$$
\boldsymbol{x} = (A^TA)^{-1}A^T\boldsymbol{b}
$$


로 unique하게 존재합니다.



$A^TA$의 invertibility와 least-squares solution에 관련된 정리는 다음과 같습니다.



<br/>

**Theorem**



Let $A$ be an $m\times n$ matrix. The following statements are logically equivalent



1. The equation $A\boldsymbol{x}=\boldsymbol{b}$ has a unique least-squares solution for each $\boldsymbol{b}$ in $\mathbb R^n$
2. The columns of $A$ are linearly independent
3. The matrix $A^TA$ is invertible



When these statements are ture, the least-squares solution of $A\boldsymbol{x}=\boldsymbol{b}$, $\hat{\boldsymbol{x}}$ is given by


$$
\hat{\boldsymbol{x}} = (A^TA)^{-1}A^T\boldsymbol{b}
$$




<br/>

*example*


$$
A=\begin{bmatrix} 4 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix}, \ \ \boldsymbol{b} = \begin{bmatrix} 2 \\ 0 \\ 11 \end{bmatrix}
$$


일 때, $A\boldsymbol x = \boldsymbol b$의 least-squares solution은 


$$
A^TA\boldsymbol{x} =A^T\boldsymbol{b}
$$


를 푸는 것과 같습니다. 따라서 $A^TA$와 $A^T\boldsymbol{b}$를 구하면


$$
A^TA = \begin{bmatrix} 4 & 0 & 1 \\ 0 & 2 & 1\end{bmatrix}\begin{bmatrix}4 & 0 \\ 0 & 2 \\ 1 & 1\end{bmatrix} = \begin{bmatrix}17 & 1 \\ 1 & 5\end{bmatrix}, \ \ A^T\boldsymbol{b} = \begin{bmatrix}4 & 0 & 1 \\ 0 & 2 & 1\end{bmatrix}\begin{bmatrix}2 \\ 0 \\ 11 \end{bmatrix} = \begin{bmatrix}19 \\ 11\end{bmatrix}
$$


 해당 matrix의 determinat가 0이 아니므로($17\times 5 - 1 \neq 0$) 해당 matrix는 invertible합니다. 따라서


$$
\boldsymbol{x} = (A^TA)^{-1}A^T\boldsymbol{b} = \frac{1}{84}\begin{bmatrix}5 & -1 \\ -1 & 17\end{bmatrix}\begin{bmatrix}19 \\ 11\end{bmatrix} = \frac{1}{84}\begin{bmatrix}84 \\ 168\end{bmatrix} = \begin{bmatrix}1 \\ 2\end{bmatrix}
$$


가 됩니다.



<br/>



지금까지 least-squares solution에 대해서 알아보았습니다. 다음 포스트에서는 inner product space에 대해 알아보도록 하겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



<br/>



### Appendix : Proof of Theorem



<br/>



**Theorem**



Let $A$ be an $m\times n$ matrix. The following statements are logically equivalent



1. The equation $A\boldsymbol{x}=\boldsymbol{b}$ has a unique least-squares solution for each $\boldsymbol{b}$ in $\mathbb R^n$
2. The columns of $A$ are linearly independent
3. The matrix $A^TA$ is invertible



</br>



* **proof**



1번 명제와 3번 명제는 동치입니다. 이는 $A\boldsymbol{x}=\boldsymbol{b}$의 least-squares problem을 푸는 것은


$$
A^TA\boldsymbol{x} = A^T\boldsymbol{b}
$$


를 푸는 것과 같기 때문입니다. 만약 $A^TA$가 invertible하면위 solution은


$$
\boldsymbol x = (A^TA)^{-1}A^T\boldsymbol b
$$


로 unique하게 존재합니다. 반대의 경우에도, 위 equation이 unique한 solution을 가지면 invertible matrix theorem에 의해 $A^TA$가 invertible 합니다. 



다음으로, 2번과 3번이 동치임을 밝혀 위 정리를 증명하겠습니다. 



$A^TA$가 invertible하려면, $A^TA$의 column이 linearly independent합니다. 즉 다음의


$$
A^TA\boldsymbol{x} =0
$$


의 equation이 trivial solution만을 가져야 합니다. 여기서 양변에 $\boldsymbol{x}^T$를 곱하면


$$
\boldsymbol{x}^TA^TA\boldsymbol{x}=0
$$


을 만족합니다. 이 때 좌변의 식은


$$
\boldsymbol{x}^TA^TA\boldsymbol{x} = (A\boldsymbol{x})^T(A\boldsymbol{x})= \|A\boldsymbol{x}\|^2 =0
$$


이므로 $A^TA\boldsymbol{x}=0$을 만족하는 $\boldsymbol{x}$는


$$
A\boldsymbol{x}=0
$$




을 만족해야 합니다. 이 때, $A$의 column이 linearly independent하므로 위 식을 만족하는 $\boldsymbol{x}$는


$$
\boldsymbol{x}=0
$$


밖에 존재하지 않습니다. 즉 


$$
A^TA\boldsymbol{x}=0
$$


은 trivial solution밖에 가지지 않으므로, $A^TA$의 column이 linearly independent하고, 따라서 $A^TA$가 invertible합니다. 반대 과정 역시 같은 논리로 증명이 가능합니다. 

