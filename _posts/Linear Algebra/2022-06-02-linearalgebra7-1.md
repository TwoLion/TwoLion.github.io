---
layout: single
title:  "7.1 Diagonalization of Symmetric Matrices"
categories: [Linear Algebra]
tag: [Linear Algebra, Spectral Decomposition]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---



이번 포스트에서는  Symmetric matrix의 특별한 성질에 대해서 알아보도록 하겠습니다.





<br/>

### 1) Symmetric Matrix



<br/>

#### (1) Property of Symmetric Matrix



<br/>



**Definition : Symmetric Matrix**



A matrix $A$ is symmetric if $A=A^T$





$A$와 $A^T$가 같은 matrix를 symmetric matrix라고 합니다.





<br/>

**Theorem**



If $A$ is symmetric, then any two eigenvectors from different eigenspaces are orthogonal



일반적인 square matrix인 경우 다른 eigenspace에 속한 eigenvector는 linearly independent를 만족합니다. symmetric matrix는 linearly independent에 추가하여, orthogonal 조건까지 만족합니다.



이전 포스트에서, eigenvector와 eigenvalue를 이용하여 matrix를 diagonalize하는 방법에 대해 배웠습니다. $A=PDP^{-1}$로 바꿀 때, $P$의 column이 eigenvector로 구성됩니다. Symmetric matrix의 경우 서로 다른 eigenspace끼리는 orthogonal하기 때문에, $P$의 column이 orthogonal(또는 orthonormal)하도록 설정할 수 있습니다. **즉, Symmetric matrix를 diagonalize할 때, $P$가 orthogonal matrix로 diagonalization을 진행할 수 있습니다.**

이처럼, $P$가 orthogonal matrix으로 diagonalization이 가능한 경우, 해당 matrix는 **Orthogonally diagonalizable**이라고 합니다.



<br/>

**Definiton : Orthogonally diagonalizable**



An $n\times n$ matrix $A$ is said to be orthogonally diagonalizable if there are an orthogonal matrix $P$ and a diagonal matrix $D$ such that


$$
A=PDP^T
$$




<br/>

**Theorem**



An $n \times n$ matrix $A$ is orthogonally diagonalizalbe if and only if $A$ is symmetric matrix





위 정리를 통해 orthogonally diagonalizable한 matrix는 symmetric matrix뿐임을 알 수 있습니다.





<br/>

#### (2) Spectral Decomposition



<br/>

**Theorem : The Spectral Theorem for Symmetric Matrix**



An $n\times n$ symmetric matrix $A$ has the following properties



1. $A$ has $n$ real eigenvalues, counting multiplicities.
2. The dimension of the eigenspace for each eigenvalue $\lambda$ equals the multiplicity of $\lambda$ as a root of the characteristic equation.
3. The eigenspaces are mutually orthogonal, in the sense that eigenvectors corresponding to different eigenvalues are orthogonal.
4. $A$ is orthogonally diagonalizable.



$n\times n$ Symmetric matrix $A$는 다음 4가지 성질을 가지고 있습니다. 첫 째로, $A$는 중복도를 포함하여 $n$개의 실수 eigenvalue를 가집니다. 두 번째, 각각의 eigenvalue에 해당하는 eigenspace의 dimension은 characteristic equation에서 해당 eigenvalue가 가지는 중복도와 동일합니다. 두 번째 성질에 의해, symmetric matrix는 diagonalizable합니다. 이에 세 번째 성질을 추가하면, $A$는 orthogonally diagonalizable합니다.



이를 이용하여 symmetric matrix $A$는 여러개의 matrix로 분해가 가능합니다.





<br/>

**Spectral Decomposition**



Suppose $A$ is $n\times n$ symmetric matrix and $A=PDP^{-1}$, where the columns of $P$ are orthonormal eigenvectors  $\boldsymbol{u}_1, ..., \boldsymbol{u}_n$ of $A$ and the corresponding eigenvalues $\lambda_1, ..., \lambda_n$ are in the diagonal matrix $D$. Then,


$$
\begin{aligned}

A =PDP^T &= \begin{bmatrix}\boldsymbol{u}_1& ... & \boldsymbol{u}_n\end{bmatrix}\begin{bmatrix}\lambda_1 & \cdots & 0 \\\vdots &\ddots & \vdots \\ 0 & \cdots & \lambda_n\end{bmatrix}\begin{bmatrix}\boldsymbol{u}_1^T \\ \vdots \\ \boldsymbol{u}_n^T\end{bmatrix} \\
&=\begin{bmatrix}\lambda_1\boldsymbol{u}_1 & ... & \lambda_n\boldsymbol{u}_n\end{bmatrix}\begin{bmatrix}\boldsymbol{u}_1^T \\ \vdots \\ \boldsymbol{u}_n^T\end{bmatrix} \\
&=\lambda_1\boldsymbol{u}_1\boldsymbol{u}_1^T + \cdots + \lambda_n \boldsymbol{u}_n\boldsymbol{u}_n^T


\end{aligned}
$$


This representation of $A$ is called a **spectral decomposition** of $A$



Symmetric matrix는 eigenvalue와 이에 해당하는 orthonormal한 eigenvector들로 표현이 가능합니다. 다음 식에서 확인할 수 있는 점은 다음과 같습니다. 

첫 번째, $\lambda_k \boldsymbol{u}_k\boldsymbol{u}_k^T$은 rank가 1인 matrix입니다. 이는 rank의 정의와 해당 matrix product를 표현하면 쉽게 이해할 수 있습니다. 

두 번째, $\boldsymbol{u}_k\boldsymbol{u}_k^T$ matrix는 $\boldsymbol{x}\in \mathbb R^n$을 $\{\boldsymbol{u}_k\}$가 basis인 subspace로 orthogonal projection시키는 matrix입니다. 즉


$$
(\boldsymbol{u}_k\boldsymbol{u}_k^T)\boldsymbol{x}
$$


는 orthogonal projection of $\boldsymbol{x}$ onto the subspace spanned by $\boldsymbol{u}_k$입니다.





<br/>

*example*


$$
A = \begin{bmatrix}7 & 2 \\ 2 & 4\end{bmatrix} = \begin{bmatrix}\frac{2}{\sqrt5} & -\frac{1}{\sqrt 5} \\ \frac{1}{\sqrt 5} & \frac{2}{\sqrt 5}\end{bmatrix}\begin{bmatrix}8 & 0 \\ 0 & 3\end{bmatrix}\begin{bmatrix}\frac{2}{\sqrt5} & \frac{1}{\sqrt 5} \\ -\frac{1}{\sqrt 5} & \frac{2}{\sqrt 5}\end{bmatrix}
$$


다음 $A$의 spectral decomposition은


$$
A=8\begin{bmatrix}\frac{2}{\sqrt 5} \\ \frac{1}{\sqrt 5} \end{bmatrix}\begin{bmatrix}\frac{2}{\sqrt 5} & \frac{1}{\sqrt 5}\end{bmatrix} + 3 \begin{bmatrix}-\frac{1}{\sqrt 5} \\ \frac{2}{\sqrt 5}\end{bmatrix}\begin{bmatrix}-\frac{1}{\sqrt 5} & \frac{2}{\sqrt 5}\end{bmatrix}
$$


입니다.



<br/>

지금까지 Symmetric matrix의 특별한 성질과 spectral decomposition에 대해 알아보았습니다. 다음 포스트에서는 quadratic form에 대해 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!





<br/>

### Appendix : Proof of Theorem

<br/>



**Theorem**



If $A$ is symmetric, then any two eigenvectors from different eigenspaces are orthogonal



<br/>

* **Proof**



$A$의 eigenvalue $\lambda_1, \lambda_2$에 대해 이에 해당하는 eigenspace에 속한 벡터를 각각 $\boldsymbol v_1, \boldsymbol v_2$라고 가정해봅시다. 그럼


$$
A\boldsymbol{v}_1 = \lambda_1\boldsymbol{v}_1 \\
A\boldsymbol{v}_2 = \lambda_2\boldsymbol{v}_2
$$




를 만족합니다. 첫 번째 식에 $\boldsymbol{v_2}$을 내적하면


$$
A\boldsymbol{v}_1\cdot \boldsymbol{v}_2 = \boldsymbol{v}_2^TA\boldsymbol{v}_1 = \lambda_1\boldsymbol{v}_1\cdot\boldsymbol{v}_2
$$


여기서 $A$는 symmetric matrix이기 때문에 위 식은 다음 식으로 변형됩니다.


$$
A\boldsymbol{v}_1 \cdot \boldsymbol{v}_2 = \boldsymbol{v}_2^TA^T\boldsymbol{v}_1 = \boldsymbol{v}_1\cdot(A\boldsymbol{v}_2) = \lambda_2\boldsymbol{v}_1\cdot\boldsymbol{v}_2
$$


따라서


$$
\lambda_1\boldsymbol{v}_1\cdot\boldsymbol{v}_2 = \lambda_2\boldsymbol{v}_1\cdot\boldsymbol{v}_2
$$


가 성립합니다. 우변을 좌변으로 옮겨주면


$$
(\lambda_1-\lambda_2)\boldsymbol{v}_1\cdot\boldsymbol{v}_2 = 0
$$


을 만족합니다. 현재 $\lambda_1, \lambda_2$는 다르므로, 양변을 나누어주면


$$
\boldsymbol{v}_1\cdot \boldsymbol{v}_2 =0
$$


을 만족하여, $\boldsymbol{v}_1, \boldsymbol{v}_2$는 orthogonal합니다.



<br/>

**Theorem**



An $n \times n$ matrix $A$ is orthogonally diagonalizable if and only if $A$ is symmetric matrix



<br/>



* **Proof**



$A$가 symmetric matrix인 경우 서로 다른 eigenspace에 해당하는 eigenvector끼리 orthogonal한 성질을 이용하여 orthogonally diagonalizable한 것을 밝힐 수 있습니다.(만약 eigenspace의 dimension이 2 이상인 경우 Gram-Schumit process를 통해 orthonormal한 basis를 만들 수 있습니다.) 

반대의 경우 또한, $A$가 orthogonally diagonalizable하면


$$
A=PDP^T
$$


로 표현이 가능한데 이 때


$$
A^T =(PDP^T)^T = PDP^T
$$


가 되어 symmetric matrix임을 알 수 있습니다.



<br/>

**Theorem : The Spectral Theorem for Symmetric Matrix**



An $n\times n$ symmetric matrix $A$ has the following properties



1. $A$ has $n$ real eigenvalues, counting multiplicities.
2. The dimension of the eigenspace for each eigenvalue $\lambda$ equals the multiplicity of $\lambda$ as a root of the characteristic equation.
3. The eigenspaces are mutually orthogonal, in the sense that eigenvectors corresponding to different eigenvalues are orthogonal.
4. $A$ is orthogonally diagonalizable.



<br/>

* **Proof**



<br/>

* Proof of 1.



1번 성질을 밝히기 위해서 symmetric matrix가 가지는 몇 가지 성질을 이용해야 합니다. 첫 번째로



$\boldsymbol x$ 벡터가  $\mathbb C^n$에 속하고, $q = \bar{\boldsymbol x}^TA\boldsymbol x$일 때, $q=\bar q$, 즉 $q$는 실숫값을 가집니다. 이는 


$$
\bar{q} = \overline{\bar {\boldsymbol x}^TA\boldsymbol x} = \bar{\bar{\boldsymbol x}}^T\overline{A{\boldsymbol x}} = \boldsymbol x^T A \bar{\boldsymbol x} = (\boldsymbol x^T A \bar{\boldsymbol x})^T = \bar{\boldsymbol x}^TA^T\boldsymbol x = \bar{\boldsymbol x}^TA\boldsymbol x = q
$$


를 통해 알 수 있습니다. (conjugate vector와 matrix의 성질과, 현재 $q$가 $\mathbb C$에 속한 값이므로 $q=q^T$가 성립함을 이용해서 밝힐 수 있습니다. )

위 성질을 이용하면
$$
A\boldsymbol{x} =\lambda\boldsymbol{x}
$$


 를 만족하는 $\boldsymbol{x} \in \mathbb C^n$인 nonzero vector가 존재하면, 이 때 $\lambda$는 실숫값을 가지고, $\boldsymbol{x}$의 real part가 $A$의 eigenvector가 됩니다. 이는


$$
\bar{\boldsymbol{x}}^TA\boldsymbol{x} = \bar{\boldsymbol{x}}^T\lambda\boldsymbol{x} = \lambda \bar{\boldsymbol{x}}^T\boldsymbol{x}
$$


에서, 위 값이 실숫값을 가지기 때문에, 위 식과 위 식의 conjugate는 같습니다. 즉


$$
\lambda\bar{\boldsymbol{x}}^T\boldsymbol{x=}\overline{\lambda\bar{\boldsymbol{x}}^T\boldsymbol{x}} =\bar{\lambda} \boldsymbol{x}^T \bar{\boldsymbol{x}}
$$


을 만족합니다. 여기서, $\boldsymbol{x}$는 nonzero vector이고, $\bar{\boldsymbol{x}}^T\boldsymbol{x}=\boldsymbol{x}^T\bar{\boldsymbol{x}}$이므로 위 식이 성립되기 위해서는


$$
\lambda = \bar{\lambda}
$$


를 만족해야 합니다. 즉 $\lambda$는 실숫값을 가져야 합니다. 해당 $\lambda$에 대응하는 eigenvector는 다음 식


$$
A\boldsymbol{x} = \lambda\boldsymbol{x}
$$


를 만족해야 합니다. 이 때 $\boldsymbol{x} = \boldsymbol{x}_1 + i\boldsymbol{x}_2$로, real part와 imaginary part로 나누면


$$
A\boldsymbol{x} = A\boldsymbol{x}_1 + iA\boldsymbol{x}_2 = \lambda\boldsymbol{x}_1 + i\lambda\boldsymbol{x}_2, \ \ \boldsymbol{x}_1, \boldsymbol{x_2} \in \mathbb R^n
$$


와 같이 나타낼 수 있습니다. 여기서 양 변의 real part인


$$
A\boldsymbol{x}_1 = \lambda \boldsymbol{x}_1
$$


을 만족하므로, $\boldsymbol{x}_1$은 $\lambda$에 해당하는 eigenvector가 됩니다.



따라서, symmetric matrix $A$의 eigenvalue는 무조건 실숫값을 가집니다.



<br/>

* Proof of 3



3번 째 성질의 경우 앞선 theorem을 이용하여 밝힐 수 있습니다.



<br/>

* proof of 4



4번 째 성질을 밝히기 위해서는 Scur factorization에 대해 알아야 합니다. 



<br/>

**Schur factorization**



Let $A$ be an $n\times n$ matrix with $n$ real eigenvalues, counting multiplicities, denoted by $\lambda_1, ..., \lambda_n$. Then, there are orthogonal matrix $P$ and upper triangluar matrix $R$ such that 


$$
A = PRP^T
$$


(일반적인 경우 $\lambda_1, ...\lambda_n$이 실수일 필요는 없으며, 만약 matrix나 vector가 복소수까지 다룬다면 해당 factorization은 orthogonal matrix $P$가 unitary matrix($P^{-1}=\bar P$)로 변경됩니다.)



Schur factorization은  $n \times n$ matrix가 중복도를 포함하여 $n$개의 실수 eigenvalue를 가질 때, orthogonal matrix와 upper triangular matrix로 분해해주는 정리입니다. 



첫 번째 성질에서 $A$가 symmetric이면


$$
A^T = PR^TP^T = PRP^T = A
$$


를 만족하기 때문에


$$
R^T =R
$$


를 만족하게 되어, $R$은 diagonal matrix가 됩니다. 



정리하면, Schur factorization을 이용하여, $A$를 orthogonal matrix $P$와 upper triangular matrix $R$로 분해가 가능한데, 이 때, $A$가 symmetric이므로, $R$은 diagonal matrix가 되어 $A$는 orthogonally diagonalizable합니다.





<br/>

* Proof of 2



4번 정리를 이용하면 쉽게 해결할 수 있습니다. 4번에서 symmetric matrix는 orthogonally diagonalizable하므로


$$
A=PDP^T = \lambda_1\boldsymbol{u}_1\boldsymbol{u}_1^T + \cdots + \lambda_n \boldsymbol{u}_n\boldsymbol{u}_n^T
$$


와 같이 작성할 수 있습니다. 이 때 $\lambda_1=\cdots= \lambda_k=\lambda$이면 위 식은


$$
A=\lambda\boldsymbol{u}_1\boldsymbol{u}_1^T + \cdots + \lambda\boldsymbol u_k\boldsymbol u
_k^T + \cdots +\lambda_n \boldsymbol{u}_n\boldsymbol{u}_n^T
$$


다음과 같이 작성이 가능합니다. 여기서 $\{\boldsymbol{u}_1, ..., \boldsymbol{u}_k\}$는 $\lambda$에 해당하는 eigenvector이고, orthogonal하기 때문에, $\lambda$에 대응되는 eigenspace의 basis가 됩니다. 즉 $A$의 characteristic equation의 solution $\lambda$의 중복도와 $\lambda$의 eigenspace의 dimension이 같게 됩니다. 
