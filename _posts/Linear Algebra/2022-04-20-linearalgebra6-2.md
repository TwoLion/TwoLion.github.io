---
layout: single
title:  "6.2 Orthogonal Set"
categories: [Linear Algebra]
tag: [Linear Algebra, orthogonal set]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---





이번 포스트에서는 orthogonal set에 대해서 알아보겠습니다.



<br/>

### 1) Orthogonal set

<br/>

**Definition : Orthogonal set**



A set of vectors $\{\boldsymbol{u_1}, ..., \boldsymbol{u_p}\}$ in $\mathbb R^n$ is an orthogonal set if each pair of distinct vectors from the set is orthogonal


$$
\boldsymbol{u_i} \cdot \boldsymbol{u_j} =0 \ \ for \ \ all \ \ i\neq j
$$


set에 속한 벡터들간 orthogonal하면, orthogonal set이라고 합니다. 



<br/>

*example*


$$
\boldsymbol{u_1} = \begin{bmatrix}3 \\ 1 \\ 1 \end{bmatrix}, \ \ \boldsymbol{u_2} = \begin{bmatrix}-1 \\ 2 \\ 1 \end{bmatrix}, \ \ \boldsymbol{u_3} = \begin{bmatrix}-\frac{1}{2} \\ -2 \\ \frac{7}{2}\end{bmatrix}
$$


일 때


$$
\boldsymbol{u_1} \cdot \boldsymbol{u_2} = \boldsymbol{u_1}\cdot \boldsymbol{u_3} = \boldsymbol{u_2}\cdot \boldsymbol{u_3}=0
$$


을 만족하기 때문에, 


$$
\{\boldsymbol{u_1}, \boldsymbol{u_2}, \boldsymbol{u_3}\}
$$


는 orthogonal set입니다.



<br/>

**Theorem**



If $S=\{\boldsymbol{u_1}, ..., \boldsymbol{u_p}\}$ is an orthogonal set of nonzero vectors in $\mathbb R^n$, then $S$ is linearly independent and hence is a basis for the subspace spanned by $S$.



Orthogonal set의 특징입니다. Orthogonal set은 linearly independent set이 되고, 따라서 orthogonal set $S$로 span한 subspace의 basis는 자동적으로 $S$가 됩니다. 



<br/>

**Definition : Orthogonal basis**



An orthogonal basis for a subspace $W$ of $\mathbb R^n$ is a basis for $W$ that is also an orthogonal set



subspace $W$의 basis를 만족하면서 orthogonal set 조건을 만족하는 set을 orthogonal basis라고 합니다.



orthogonal basis를 알면, $W$에 속한 벡터를 basis의 linear combination으로 표현할 때 각 벡터의 coefficient를 효과적으로 찾을 수 있습니다.



<br/>

**Theorem**



Let $\{\boldsymbol{u_1}, ..., \boldsymbol{u_p}\}$ be an orthogonal basis for a subspace $W$ of $\mathbb R^n$. For each $\boldsymbol{y}$ in $W$, the weights in the linear combination


$$
\boldsymbol{y}=c_1\boldsymbol{u_1}+\cdots + c_p\boldsymbol{u_p}
$$


and 


$$
c_j = \frac{\boldsymbol{u_j}\cdot \boldsymbol{y}}{\boldsymbol{u_j}\cdot\boldsymbol{u_j}},  \ \ \ j=1, ..., p
$$




일반적인 basis를 이용했다면, linear combination의 coefficient를 구하기 위해서는 linear system을 풀어야 coefficient를 구할 수 있었습니다. 하지만, 만약 basis가 **orthogonal basis라면,  orthogonal의 성질과 inner product를 이용하여 linear combination의 coefficients를 쉽게 구할 수 있습니다. **





<br/>

*example*


$$
\boldsymbol{u_1} = \begin{bmatrix}3 \\ 1 \\ 1 \end{bmatrix}, \boldsymbol{u_2} = \begin{bmatrix}-1 \\ 2 \\ 1 \end{bmatrix}, 
\boldsymbol{u_3}=\begin{bmatrix}-\frac{1}{2} \\ -2 \\ \frac{7}{2} \end{bmatrix}, \boldsymbol{y}=\begin{bmatrix}6 \\ 1\\ -8 \end{bmatrix}
$$

$\{\boldsymbol{u_1}, \boldsymbol{u_2}, \boldsymbol{u_3}\}$가 orthogonal set이므로, linearly independent하고, $\mathbb R^3$의 basis가 됩니다. 따라서 $\boldsymbol{y}$를 $\boldsymbol{u_1}, \boldsymbol{u_2}, \boldsymbol{u_3}$의 linear combination으로 표현이 가능합니다. 


$$
\boldsymbol{y}=c_1\boldsymbol{u_1}+c_2\boldsymbol{u_2}+c_3\boldsymbol{u_3}
$$


만약 $\{\boldsymbol{u_1}, \boldsymbol{u_2}, \boldsymbol{u_3}\}$가 orthogonal하지 않았다면, $c_1, c_2, c_3$를 equation을 직접 풀어서 구해야 합니다. 하지만 현재 $\{\boldsymbol{u_1}, \boldsymbol{u_2}, \boldsymbol{u_3}\}$가 orthogonal하므로


$$
\boldsymbol{y}\cdot\boldsymbol{u_1} = c_1\boldsymbol{u_1}\cdot\boldsymbol{u_1} \\
\boldsymbol{y}\cdot\boldsymbol{u_2} = c_2\boldsymbol{u_2}\cdot\boldsymbol{u_2} \\
\boldsymbol{y}\cdot\boldsymbol{u_3} = c_3\boldsymbol{u_3}\cdot\boldsymbol{u_3}
$$


가 성립합니다 따라서


$$
c_1 = \frac{\boldsymbol{y}\cdot \boldsymbol{u_1}}{\boldsymbol{u_1}\cdot \boldsymbol{u_1}} =1 , c_2 = \frac{\boldsymbol{y}\cdot \boldsymbol{u_2}}{\boldsymbol{u_2}\cdot \boldsymbol{u_2}} =-2, c_3=\frac{\boldsymbol{y}\cdot\boldsymbol{u_3}}{\boldsymbol{u_3}\cdot\boldsymbol{u_3}} =-2
$$


가 되고, 


$$
\boldsymbol{y}=\boldsymbol{u_1}-2\boldsymbol{u_2}-2\boldsymbol{u_3}
$$


임을 알 수 있습니다.



<br/>

### 2) Orthonormal set

<br/>



**Definition : Orthonormal set**



A set $\{\boldsymbol{u_1}, ..., \boldsymbol{u_p}\}$ is an orthonormal set if it is an orthogonal set of unit vectors



**orthogonal set 조건을 만족하면서 set에 속한 각 벡터의 length가 1인 경우, orthonormal set이라고 합니다.** 





<br/>

Standard basis $\{\boldsymbol{e_1}, ..., \boldsymbol{e_n}\}$ for $\mathbb R^n$


$$
\boldsymbol{e_1}=\begin{bmatrix}1 \\ 0 \\ \vdots  \\ 0 \end{bmatrix}, \boldsymbol{e_2}=\begin{bmatrix} 0 \\ 1 \\  \vdots \\ 0 \end{bmatrix}, ..., \boldsymbol{e_n}=\begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1\end{bmatrix}
$$


일 때, 각 벡터들끼리 orthogonal하고 length가 1이므로, standard basis는 orthonormal set입니다.





<br/>

*example*


$$
\boldsymbol{u_1} = \begin{bmatrix}\frac{3}{\sqrt {11}} \\ \frac{1}{\sqrt{11}} \\ \frac{1}{\sqrt{11}} \end{bmatrix}, \boldsymbol{u_2} = \begin{bmatrix}-\frac{1}{\sqrt{6}} \\ \frac{2}{\sqrt{6}} \\ \frac{1}{\sqrt6} \end{bmatrix}, 
\boldsymbol{u_3}=\begin{bmatrix}-\frac{1}{\sqrt{66}} \\ -\frac{2}{\sqrt{66}} \\ \frac{7}{\sqrt{66}} \end{bmatrix},
$$


위의 예시에서 length를 1로 normalizing시킨 $\boldsymbol{u_1}, \boldsymbol{u_2}, \boldsymbol{u_3}$입니다. 따라서 각 벡터들끼리 orthogonal하고 length가 1이므로 


$$
\{\boldsymbol{u_1}, \boldsymbol{u_2}, \boldsymbol{u_3}\}
$$


는 orthonormal set입니다.





<br/>

**Theorem**



An $m\times n$ matrix $U$ has orthonormal columns if and only if $U^TU=I$



$U^TU=I$가 나오는 matrix의 특징은 $U$의 column이 orthogonal하다는 것입니다. 이를 $n\times n$ matrix에서 생각하면 


$$
U^TU=I
$$


가 성립되기 때문에, $U^T = U^{-1}$임을 알 수 있습니다.





<br/>

**Theorem**



Let $U$ be an $m\times n$ matrix with orthonormal columns, and let $\boldsymbol{x}, \boldsymbol{y}$ be in $\mathbb R^n$



1. $\|U\boldsymbol{x}\|=\|\boldsymbol{x}\|$
2. $(U\boldsymbol{x})\cdot(U\boldsymbol{y})=\boldsymbol{x}\cdot\boldsymbol{y}$
3. $(U\boldsymbol{x})\cdot(U\boldsymbol{y})=0$ if and only if $\boldsymbol{x}\cdot\boldsymbol{y}=0$



$U$를 standard matrix로 가지는 linear transformation $T_U$를 생각해봅시다. 다음 정리로 인해, **$T_U$로 인해 transform된 벡터는 transform되기 이전의 벡터와 크기가 같고, inner product값이 같습니다.**







Orthonormal 개념을 $n\times n$ square matrix에 적용시킨 matrix가 orthogonal matrix입니다.

<br/>

**Definition : Orthogonal matrix**



An orthogonal matrix is a square matrix $U$ such that $U^{-1}=U^T$



Such a matrix has orthonormal columns


$$
UU^T = U^TU = I
$$


$U$의 inverse가 $U^T$가 되는 square matrix를 orthogonal matrix라고 합니다. 



<br/>

**Theorem**



An orthogonal matrix have orthonormal rows



orthogonal matrix는 orthogonal한 column을 가지고 있습니다. 그리고 inverse가 해당 matrix의 transpose인 것을 생각하면, 해당 matrix는 orthonormal한 row를 가지고 있는 것을 알 수 있습니다.(자세한 증명은 appendix 참고)



<br/>



*example*


$$
\boldsymbol{u_1} = \begin{bmatrix}\frac{3}{\sqrt {11}} \\ \frac{1}{\sqrt{11}} \\ \frac{1}{\sqrt{11}} \end{bmatrix}, \boldsymbol{u_2} = \begin{bmatrix}-\frac{1}{\sqrt{6}} \\ \frac{2}{\sqrt{6}} \\ \frac{1}{\sqrt6} \end{bmatrix}, 
\boldsymbol{u_3}=\begin{bmatrix}-\frac{1}{\sqrt{66}} \\ -\frac{2}{\sqrt{66}} \\ \frac{7}{\sqrt{66}} \end{bmatrix},
$$


다음 vector로 이루어진 set $\{\boldsymbol{u_1}, \boldsymbol{u_2}, \boldsymbol{u_3}\}$은 orthonormal set입니다. 따라서 $\boldsymbol{u_1}, \boldsymbol{u_2}, \boldsymbol{u_3}$를 column으로 하는 matrix


$$
U = \begin{bmatrix} \boldsymbol{u_1} & \boldsymbol{u_2} & \boldsymbol{u_3} \end{bmatrix}
$$


은 orthogonal matrix입니다. 즉


$$
U^TU = UU^T = I
$$


를 만족합니다.



<br/>



지금까지 orthogonal set에 대해 알아보았습니다. 다음 포스트에서는 orthogonal projection에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



<br/>



### Appendix : Proof of theorem



<br/>

**Theorem**



If $S=\{\boldsymbol{u_1}, ..., \boldsymbol{u_p}\}$ is an orthogonal set of nonzero vectors in $\mathbb R^n$, then $S$ is linearly independent and hence is a basis for the subspace spanned by $S$.



<br/>

* **Proof**



$S$가 linearly independent한지 확인해보기 위해 다음 equation


$$
c_1\boldsymbol{u_1} + \cdots c_p\boldsymbol{u_p}=0
$$


을 생각해봅시다. 양변에 $\boldsymbol{u_j}$를 내적하면


$$
c_j\boldsymbol{u_j} \cdot \boldsymbol{u_j} = 0 \iff c_j\|\boldsymbol u_j\|^2 =0
$$


 임을 알 수 있습니다. $\boldsymbol{u_j}$는 nonzero vector이므로, 


$$
c_j=0
$$


임을 뜻합니다. 따라서 $j=1,...,p$에 대해서 $c_j=0$이므로


$$
c_1\boldsymbol{u_1} + \cdots c_p\boldsymbol{u_p}=0
$$


은 trivial solution을 가집니다. 즉 $S$는 linearly independent합니다.



<br/>

**Theorem**



Let $\{\boldsymbol{u_1}, ..., \boldsymbol{u_p}\}$ be an orthogonal basis for a subspace $W$ of $\mathbb R^n$. For each $\boldsymbol{y}$ in $W$, the weights in the linear combination


$$
\boldsymbol{y}=c_1\boldsymbol{u_1}+\cdots + c_p\boldsymbol{u_p}
$$


and 


$$
c_j = \frac{\boldsymbol{u_j}\cdot \boldsymbol{y}}{\boldsymbol{u_j}\cdot\boldsymbol{u_j}},  \ \ \ j=1, ..., p
$$



<br/>

* **Proof**


$$
\boldsymbol{y}=c_1\boldsymbol{u_1}+\cdots + c_p\boldsymbol{u_p}
$$


의 양변에 $\boldsymbol{u_j}$를 inner product하면


$$
\boldsymbol{u_j}\cdot \boldsymbol{y} = c_j\boldsymbol{u_j}\cdot \boldsymbol{u_j}
$$


이 됩니다. basis의 원소 $\boldsymbol{u_j}$이므로 $\boldsymbol{u_j}\neq0$이므로


$$
c_j=\frac{\boldsymbol{u_j}\cdot \boldsymbol y}{\boldsymbol{u_j} \cdot \boldsymbol{u_j}}
$$


가 됩니다.



<br/>

**Theorem**



An $m\times n$ matrix $U$ has orthonormal columns if and only if $U^TU=I$



<br/>



* **Proof**



<br/>




$$
U = \begin{bmatrix}\boldsymbol{u_1} & \cdots & \boldsymbol{u_n} \end{bmatrix}
$$


일 때, 


$$
U^T = \begin{bmatrix}\boldsymbol{u_1}^T \\ \vdots \\ \boldsymbol{u_n}^T \end{bmatrix}
$$


입니다. 두 matrix를 곱하면


$$
U^TU = \begin{bmatrix}\boldsymbol{u_1}^T \\ \vdots \\ \boldsymbol{u_n}^T \end{bmatrix} \begin{bmatrix}\boldsymbol{u_1} & \cdots & \boldsymbol{u_n} \end{bmatrix} 
$$


가 되는데 이 때, $U^TU$ entry는


$$
(U^TU)_{ii} = \boldsymbol{u_i}^T\boldsymbol{u_i} = \boldsymbol{u_i}\cdot  \boldsymbol{u_i} = 1 \\
(U^TU)_{ij} = \boldsymbol{u_i}^T\boldsymbol{u_j} = \boldsymbol{u_j}\cdot  \boldsymbol{u_i} = 0
$$


가 됩니다. 이는 $U$의 column이 orthonormal하기 때문입니다. 따라서


$$
U^TU = I
$$


가 됩니다. 



반대로, $U^T U=I$이면


$$
(U^TU)_{ii} = \boldsymbol{u_i}^T\boldsymbol{u_i} = \boldsymbol{u_i}\cdot  \boldsymbol{u_i} = 1 \\
(U^TU)_{ij} = \boldsymbol{u_i}^T\boldsymbol{u_j} = \boldsymbol{u_j}\cdot  \boldsymbol{u_i} = 0
$$


가 성립하기 때문에, $U$의 column이 orthonormal한 것을 알 수 있습니다.



<br/>

**Theorem**



Let $U$ be an $m\times n$ matrix with orthonormal columns, and let $\boldsymbol{x}, \boldsymbol{y}$ be in $\mathbb R^n$



1. $\|U\boldsymbol{x}\|=\|\boldsymbol{x}\|$
2. $(U\boldsymbol{x})\cdot(U\boldsymbol{y})=\boldsymbol{x}\cdot\boldsymbol{y}$
3. $(U\boldsymbol{x})\cdot(U\boldsymbol{y})=0$ if and only if $\boldsymbol{x}\cdot\boldsymbol{y}=0$



<br/>

* **Proof**



<br/>

Proof of 1



$U$가 orthonormal한 column을 가지고 있으므로


$$
U^TU=I
$$


를 만족합니다. 따라서


$$
\|U\boldsymbol{x}\|^2 = (U\boldsymbol x)^T (U\boldsymbol x) = \boldsymbol x^T U^TU\boldsymbol x = \boldsymbol{x}^T\boldsymbol{x} = \|\boldsymbol{x}\|^2 
$$


을 만족합니다.



<br/>

Proof of 2



마찬가지로


$$
(U\boldsymbol{x})\cdot(U\boldsymbol{y}) = \boldsymbol{y}^TU^TU\boldsymbol x = \boldsymbol y^T \boldsymbol x = \boldsymbol x \cdot \boldsymbol y
$$


를 만족합니다.



<br/>



Proof of 3



2번에서 


$$
(U\boldsymbol{x})\cdot(U\boldsymbol{y})=\boldsymbol{x}\cdot\boldsymbol{y}
$$


이므로,


$$
(U\boldsymbol{x})\cdot(U\boldsymbol{y})=0
$$


인 것과 동치는


$$
\boldsymbol{x}\cdot\boldsymbol{y}=0
$$


과 같습니다.



<br/>

**Theorem**



An orthogonal matrix have orthonormal rows



<br/>

* **Proof**



Orthogonal matrix $U$는


$$
U^TU=UU^T=I
$$


를 만족합니다. 여기서 $V=U^T$로 정의하면 위의 식은


$$
VV^T=V^TV = I
$$


를 만족합니다. 즉 $V$ 역시 orthogonal matrix입니다. 이 때, $V$의 column은 $U$의 row와 같기 때문에, $U$의 row 또한 orthonormal합니다.



