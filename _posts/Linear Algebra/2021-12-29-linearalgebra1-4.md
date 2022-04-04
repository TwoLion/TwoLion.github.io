---
layout: single
title:  "1.4 Matrix Equation"
categories: [Linear Algebra]
tag: [Linear Algebra]
toc: true
author_profile: true #프로필 생략 여부
use_math: true

---











이번 포스트에서는 linear system을 표현하는 방법 중 하나인 matrix equation에 대해 알아보겠습니다.

<br/>



### 1. Matrix Equation

<br/>



#### 1) matrix $\times$ vector



Matrix equation를 정의하기 위해서, matrix와 vector의 곱에 대해서 먼저 정의를 합니다.



<br/>



* **Definition**

If $A$ is an $m \times n$ matrix, with columns $\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_n}$, and if $\boldsymbol{x}$ is in $\mathbb{R}^n$, then the product of $A$ and $x$, denoted by $A\boldsymbol{x}$, is the linear combination of the columns of $A$ using the corresponding entries in $\boldsymbol{x}$ as weights


$$
A\boldsymbol{x} = \begin{bmatrix}\boldsymbol{a_1} & \boldsymbol{a_2} & ... & \boldsymbol{a_n} \end{bmatrix}
\begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_n  \end{bmatrix} 
= x_1\boldsymbol{a_1} + x_2\boldsymbol{a_2} + \cdots + x_n\boldsymbol{a_n}
$$


$A\boldsymbol{x}$ is defined only if the number of columns of $A$ equals the number of entries in $\boldsymbol{x}$



matrix와 vector의 곱은 matrix의 column의 linear combination으로 표현되고, column에 해당하는 weight이 vector $\boldsymbol{x}$ 의 성분입니다.

따라서, $A\boldsymbol{x}$가 정의되기 위해서는 $A$의 column 개수와 $\boldsymbol{x}$의 성분 개수가 같아야 합니다. 

<br/>



example 1)


$$
\begin{bmatrix}1& 2& -1 \\0 & 5 & -3\end{bmatrix} \begin{bmatrix}4 \\ 3\\ 7 \end{bmatrix} =4\begin{bmatrix}1\\0\end{bmatrix} + 3\begin{bmatrix}2\\5\end{bmatrix} + 7\begin{bmatrix}-1\\-3\end{bmatrix} = \begin{bmatrix}3\\-6\end{bmatrix}
$$

<br/>




example 2)



$\boldsymbol{v_1}$, $\boldsymbol{v_2}$, $\boldsymbol{v_3}$ are in $\mathbb{R}^m$, and linear combination $3\boldsymbol{v_1}-5\boldsymbol{v_2}+7\boldsymbol{v_3}$ can be represented as


$$
3\boldsymbol{v_1}-5\boldsymbol{v_2}+7\boldsymbol{v_3} = \begin{bmatrix}\boldsymbol{v_1} & \boldsymbol{v_2} & \boldsymbol{v_3}\end{bmatrix}\begin{bmatrix}3 \\ -5 \\ 7\end{bmatrix}
$$




이와 같이 matrix와 vector 곱이 matrix column의 linear combination으로 나타내는 것을 알면, linear system을 matrix와 vector의 곱으로 나타낼 수 있습니다.



<br/>



#### 2) Matrix Equation

<br/>



Matrix equation은 다음과 같이 정의됩니다. 

<br/>



* **Definition : Matrix Equation**



$A$ is $m \times n$ matrix, with columns $\boldsymbol{a_1}, \boldsymbol{a_2}, ..., \boldsymbol{a_n}$, and $\boldsymbol{b}$ is in $\mathbb{R}^m$, and matrix equation is represented as


$$
A\boldsymbol{x} = \boldsymbol{b}
$$




where $\boldsymbol{x}$ is in $\mathbb{R}^n$





$A\boldsymbol{x}$을 $A$의 column의 linear combination으로 표현하면 위의 matrix equation은


$$
x_1\boldsymbol{a_1} + x_2\boldsymbol{a_2} + \cdots + x_n\boldsymbol{a_n} = \boldsymbol{b}
$$


로 표현됩니다. 따라서, 이 equation의 solution은 다음의 augmented matrix를 가지는 linear system의 solution과 동일합니다.


$$
\begin{bmatrix}\boldsymbol{a_1} & \boldsymbol{a_2} & \cdots & \boldsymbol{a_n} & \boldsymbol{b} \end{bmatrix}
$$




따라서, 하나의 linear system을



1. augmented matrix
2. vector equation
3. matrix equation

세 가지 방법으로 표현할 수 있습니다.





다음 포스트에서는 linear independence에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글로 남겨주세요!