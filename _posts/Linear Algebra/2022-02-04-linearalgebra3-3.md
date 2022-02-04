---
layout: single
title:  "3.3 Cramer's Rule"
categories: [Linear Algebra]
tag: [Linear Algebra, Determinant, Cramer's rule]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---







이번 포스트에서는 Cramer's rule에 대해서 알아보겠습니다.



<br/>

### 1) Cramer's Rule



<br/>



Cramer's rule을 정의하기 위해서는 하나의 notation 정의가 필요합니다.

<br/>



* **Definition**



for any $n \times n$ matrix $A$ and any $\boldsymbol{b}$ in $\mathbb R^n$, let $A_i(\boldsymbol b)$ be the matrix obtained from $A$ by replaceing column $i$ by the vector $\boldsymbol{b}$


$$
A_i(\boldsymbol{b})=\begin{bmatrix}\boldsymbol{a_1} & \boldsymbol{a_2} & ...& \boldsymbol{a_{i-1}} & \boldsymbol{b} & \boldsymbol{a_{i+1}} & ... & \boldsymbol{a_n}  \end{bmatrix}
$$
 

즉  $A_i(\boldsymbol b)$ 는 matrix $A$의 $i$ 번째 column 대신 $\boldsymbol{b}$를 넣은 새로운 matrix입니다.



<br/>



* **Theorem : Cramer's Rule**



Let $A$ be an invertible $n \times n$ matrix. For any $\boldsymbol{b}$ in $\mathbb R ^n$, the unique solution $\boldsymbol{x}$ of $A\boldsymbol{x}=\boldsymbol{b}$ has entries given by


$$
x_i = \frac{detA_i(\boldsymbol{b})}{detA}, \ \ i=1, 2, ..., n
$$


Cramer's rule을 이용하면, determinant를 이용하여 linear system의 solution을 구할 수 있습니다. 

(증명은 appendix 참고)

<br/>



*example*


$$
\begin{aligned}
3x_1-2x_2 &= 6 \\
-5x_1+4x_2 &= 8
\end{aligned}
$$


위 linear system을 matrix equation으로 바꾸면


$$
\begin{bmatrix} 3 & -2 \\ -5 & 4 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 6 \\ 8 \end{bmatrix}
$$


Cramer's rule을 적용하기 위해 $detA_i(\boldsymbol{b})$와 $detA$를 게산하면


$$
detA_1(\boldsymbol{b}) =\begin{vmatrix}6 & -2 \\ 8 & 4 \end{vmatrix} =40 \\
detA_2(\boldsymbol{b}) =\begin{vmatrix}3 & 6 \\ -5 & 8 \end{vmatrix} = 54 \\
detA = \begin{vmatrix}3 & -2 \\ -5 & 4 \end{vmatrix} = 2
$$


Cramer's rule을 적용하면


$$
x_1 = \frac{detA_1(\boldsymbol{b})}{detA} = 20 \\
x_2 = \frac{detA_2(\boldsymbol{b})}{detA}=27
$$


따라서 위 linear system의 solution은


$$
\boldsymbol{x} = \begin{bmatrix} 20 \\ 27\end{bmatrix}
$$


입니다.





<br/>

### 2) A formula for $A^{-1}$

<br/>



Cramer's rule을 적용하여 $A^{-1}$를 찾을 수 있습니다.



<br/>

#### 1) FInding $A^{-1}$

<br/>



Let $A$ be an invertible $n\times n$ matrix. Then, the $j$th column of $A^{-1}$ is a vector $\boldsymbol{x}$ that satisfies


$$
A\boldsymbol{x} = \boldsymbol{e_j}
$$


where $\boldsymbol{e_j}$ is the $j$th column of identity matrix.



Inverse의 정의에 의해


$$
AA^{-1} = I
$$


를 만족합니다. 따라서 위 matrix 식의 결과를 각 column별로 살펴보면
$$
A\boldsymbol{x} = \boldsymbol{e_j}
$$
을 만족하는 $\boldsymbol{x}$가 $A^{-1}$의 $j$th column이 되는 것을 알 수 있습니다.



위 방정식의 solution을 구할 때, Cramer's rule을 이용하면


$$
x_{ij} = \frac{detA_i({\boldsymbol{e_j})}}{detA}
$$
가 되고, $x_{ij}$는 $A^{-1}$의 $(i, j)$ entry가 됩니다.



$A_i(\boldsymbol{e_j})$를 살펴보면


$$
\begin{aligned}

A_i(\boldsymbol{e_j}) &= \begin{bmatrix}\boldsymbol{a_1} & ... & \boldsymbol{a_{i-1}} & \boldsymbol{e_j} & \boldsymbol{a_{i+1}} & ... & \boldsymbol{a_n} \end{bmatrix} \\ \\
&=\begin{bmatrix} & & & 0 & & & \\  & & & 0 & & & \\  & & & \vdots & & &  \\\boldsymbol{a_1} & ... & \boldsymbol{a_{i-1}} & 1 & \boldsymbol{a_{i+1}} & ... & \boldsymbol{a_n} \\  & & & \vdots & & &  \\  & & & 0 & & &  \end{bmatrix}


\end{aligned}
$$


입니다. 따라서 $detA_i(\boldsymbol{e_j})$를 co-factor expansion을 이용하여 구할 때 $i$ 번째 column을 기준으로 구하게 됩니다. $A_i(\boldsymbol{e_j})$의 $(j, i)$ 위치에서의 co-factor는 $A_i(\boldsymbol{e_j})$에서 $j$ 번째 row, $i$ 번째 column을 제외한 matrix의 determinant입니다.

 해당 determinant는 $A$에서 $j$ 번째 row, $i$ 번째 column을 제외한 matrix의 determinant와 같습니다. 즉 $A_i(\boldsymbol{e_j})$의 $(i, j)$ 위치에 해당하는 cofactor는 $A$의 $(i, j)$ 위치에 해당하는 cofactor와 일치합니다. 따라서


$$
detA_i(\boldsymbol{e_j}) = C_{ji} 
$$
입니다.



$A^{-1}$의 $(i, j)$ entry에 해당하는 값을 알았으니, $A^{-1}$를 표현하면


$$
A^{-1}= \frac{1}{detA}\begin{bmatrix}C_{11} & C_{21} & ... & C_{n1} \\
C_{12} & C_{22} & ... & C_{n2} \\ 
\vdots & \vdots & \ddots & \vdots \\
C_{1n} & C_{2n} & ... & C_{nn}\end{bmatrix}
$$


이 됩니다. 여기서 $\frac{1}{detA}$를 제외한 matrix 부분


$$
\begin{bmatrix}C_{11} & C_{21} & ... & C_{n1} \\
C_{12} & C_{22} & ... & C_{n2} \\ 
\vdots & \vdots & \ddots & \vdots \\
C_{1n} & C_{2n} & ... & C_{nn}\end{bmatrix}
$$


를 **adjugate of** $A$라고 하고, $adjA$로 표시합니다.


$$
A^{-1} = \frac{1}{detA}adjA
$$






<br/>

*example*


$$
A = \begin{bmatrix} 2 & 1 & 3 \\ 1 & -1 & 1 \\ 1 & 4 & 2\end{bmatrix}
$$


$A$의 determinant를 첫 번째 행을 기준으로 co-factor expansion을 이용하여 구해보면


$$
detA = 2 \cdot(-1)^{1+1}\begin{vmatrix}-1 & 1 \\ 4 & 2\end{vmatrix} +(-1)^{1+2}\begin{vmatrix}1 & 1 \\ 1 & 2\end{vmatrix} + 3\cdot(-1)^{1+3} \begin{vmatrix}1 & -1 \\ 1 & 4\end{vmatrix} = 2 
$$


determinant가 0이 아니므로, $A$는 invertible matrix입니다. 각 entry에 해당하는 cofactor를 구해보면


$$
C_{11}=3, C_{21}=-10, C_{31}=4 \\C_{12}=1, C_{22}=1, C_{32}=-1 \\C_{13}=5, C_{23}=7, C_{33}=-3
$$


이를 이용하여 $A^{-1}$를 구하면


$$
A^{-1}=\frac{1}{detA}adjA = \frac{1}{2}\begin{bmatrix}3 & -10 & 4 \\ 1 & 1& -1 \\ 5 & 7 & -3 \end{bmatrix}
$$


가 됩니다.



<br/>



지금까지 Cramer's rule에 대해서 알아보았습니다. 다음 포스트에서는 Vector space와 subspace에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!





<br/>

#### Appendix : Proof of Theorem



<br/>



**Theorem : Cramer's Rule**





Let $A$ be an invertible $n \times n$ matrix. For any $\boldsymbol{b}$ in $\mathbb R ^n$, the unique solution $\boldsymbol{x}$ of $A\boldsymbol{x}=\boldsymbol{b}$ has entries given by


$$
x_i = \frac{detA_i(\boldsymbol{b})}{detA}, \ \ i=1, 2, ..., n
$$




* **Proof**



Let


$$
A = \begin{bmatrix} a_{11} & a_{12} & ... & a_{1n} \\ a_{21} & a_{22} & ... & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{bmatrix} = \begin{bmatrix}\boldsymbol{a_1} & \boldsymbol{a_2} & ... & \boldsymbol{a_n} \end{bmatrix}, \ \ \boldsymbol{b}=\begin{bmatrix}b_1 \\ b_2 \\ \vdots \\ b_n\end{bmatrix}
$$


$A\boldsymbol{x}=\boldsymbol{b}$를 linear system으로 표현하면 다음과 같이 표현됩니다.




$$
\begin{aligned}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n&=b_1 \cdots 1. \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n&=b_2 \cdots 2. \\
\vdots \\

a_{n1}x_1 + a_{n2}x_2 + \cdots + a_{nn}x_n&=b_n \cdots n. 


\end{aligned}
$$


첫 번째 식부터 n 번째 식까지 오른쪽에 번호를 통하여 나타내었습니다. 



여기서 $i$번 째 식의 양변에 $C_{ij}$를 곱해주겠습니다. ($i=1, 2, ... ,n$)


$$
\begin{aligned}

a_{11}C_{1j}x_1 + a_{12}C_{1j}x_2 + \cdots + a_{1n}C_{1j}x_n&=C_{1j}b_1 \cdots 1. \\
a_{21}C_{2j}x_1 + a_{22}C_{2j}x_2 + \cdots + a_{2n}C_{2j}x_n&=C_{2j}b_2 \cdots 2. \\
\vdots \\

a_{n1}C_{nj}x_1 + a_{n2}C_{nj}x_2 + \cdots + a_{nn}C_{nj}x_n&=C_{nj}b_n \cdots n.

\end{aligned}
$$






이 후, 1번 식부터 n번 식 모두를 더하고 $x_1, x_2, ..., x_n$에 대해서 정리를 하면


$$
p_1x_1+p_2x_2+\cdots+p_nx_n = b'
$$
where


$$
p_i = a_{1i}C_{1j}+a_{2i}C_{2j}+\cdots+a_{ni}C_{nj} \ \ (i=1, 2, ..., n) \\
b' =b_1C_{1j}+b_2C_{2j}+...b_nC_{nj}
$$


가 됩니다.



$p_i$에 대해서 살펴보겠습니다.



만약 $i= j$라면


$$
p_i = a_{1i}C_{1i}+a_{2i}C_{2i}+\cdots+a_{ni}C_{ni} = detA
$$


가 됩니다. $i$ 번째 column을 기준으로 co-factor expansion을 한 determinant입니다. 



만약 $i\neq j$라면


$$
p_i = a_{1i}C_{1j}+a_{2i}C_{2j}+\cdots+a_{ni}C_{nj} = detA_{j}(\boldsymbol{a_i})
$$


가 됩니다. A의 $j$ 번째 column을 $\boldsymbol{a_i}$로 바꾼 matrix의 $j$ 번째 column을 기준으로 co-factor expansion을 한 결과가 됩니다. 

여기서, $j\neq i$이므로, $A_{j}(\boldsymbol{a_i})$에서 $i$ 번째, $j$ 번째 column이 $\boldsymbol{a_i}$입니다.

 $A_{j}(\boldsymbol{a_i})$에서 똑같은 column이 존재하기 때문에, $A_{j}(\boldsymbol{a_i})$의 determinant가 0입니다.


$$
detA_{j}(\boldsymbol{a_i})=0
$$
마지막으로 $\boldsymbol{b}'$을 살펴보면


$$
b' =b_1C_{1j}+b_2C_{2j}+...b_nC_{nj} = detA_j(\boldsymbol{b})
$$


인 것을 알 수 있습니다.



이를 이용하여 식을 정리하면


$$
p_1x_1+p_2x_2+\cdots+p_nx_n = b' \\
\Rightarrow p_jx_j= b' \\
\Rightarrow (detA) x_j = detA_j(\boldsymbol{b})
$$




이 성립하고, $A$가 invertible하므로 


$$
x_j = \frac{ detA_j(\boldsymbol{b})}{detA}
$$


결과를 얻을 수 있습니다.



