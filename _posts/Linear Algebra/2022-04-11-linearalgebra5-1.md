---
layout: single
title:  "5.1 Eigenvectors and Eigenvalues"
categories: [Linear Algebra]
tag: [Linear Algebra, Eigenvector, Eigenvalue]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---





이번 포스트에서는 eigenvector와 eigenvalue에 대해 알아보겠습니다.



<br/>

### 1) Eigenvecotrs and Eigenvalues

<br/>



**Definition : Eigenvecotrs and Eigenvalues**



Let $A$ be an $n \times n$ square matrix.



The eigenvector of $A$ is a non-zero vector $\boldsymbol{x}$ such that 


$$
A\boldsymbol{x} = \lambda \boldsymbol{x}
$$


for some scalar $\lambda$. 

A scalar $\lambda$ is called an eigenvalue of $A$ if there is a nontrivial solution $\boldsymbol{x}$ of $A\boldsymbol{x}=\lambda\boldsymbol{x}$

$\boldsymbol x$ is called an eigenvector corresponding to $\lambda$





Eigenvector와 eigenvalue의 정의에서의 point는 다음과 같습니다. 첫 번째로, **eigenvector와 eigenvalue는 square matrix에서 정의됩니다.** 두 번째로, 다음 equation


$$
A\boldsymbol{x} = \lambda\boldsymbol x
$$


가 **non-trivial solution이 존재할 때, 이 때 $\lambda$를 $A$의 eigenvalue라고 합니다.** 마지막으로, **위 equation의 solution 중 zero vector를 제외한 $\boldsymbol{x}$를 eigenvector corresponding to $\lambda$**라고 합니다.



위 정의를 생각해보면, 다음


$$
A\boldsymbol{x} = \lambda\boldsymbol{x}
$$


가 nontrivial solution이 존재해야 eigenvalue와 eigenvector를 정의할 수 있습니다. 우변의 항을 좌변으로 넘겨 $\boldsymbol{x}$로 묶으면


$$
(A-\lambda I)\boldsymbol x = 0
$$


위 equation이 non-trivial solution을 가져야 eigenvector와 eigenvalue를 정의할 수 있습니다. 



Eigenvector와 eigenvalue를 정의하기 위해서는 위 equation의 solution을 구해야 합니다. 즉, eigenvector corresponding to $\lambda$는


$$
Nul(A-\lambda I)
$$


다음의 null space에서 zero vector를 제외한 vector가 됩니다. 여기서, **zero vector를 포함한 다음 $Nul(A-\lambda I)$를 eigenspace of $A$ corresponding $\lambda$**라고 합니다. 즉 $\lambda$에 대한 eigenspace에서 zero vector를 제외한 나머지 vector들이 $\lambda$에 대한 eigenvector입니다.



<br/>

*example*


$$
A=\begin{bmatrix} 1 & 6 \\ 5 & 2\end{bmatrix}, \ \ \boldsymbol{u} = \begin{bmatrix}6 \\ -5\end{bmatrix}, \ \ \boldsymbol{v} = \begin{bmatrix} 3 \\ -2 \end{bmatrix}
$$


에 대해서


$$
A\boldsymbol{u} = \begin{bmatrix} -24 \\ 20 \end{bmatrix} = -4\begin{bmatrix} 6 \\ -5 \end{bmatrix}=-4\boldsymbol{u} \\

A\boldsymbol{v} = \begin{bmatrix} -9 \\ 11 \end{bmatrix} \neq k\boldsymbol{v}
$$


$A\boldsymbol{u} = -4\boldsymbol{u}$를 만족하기 때문에, $-4$는 $A$의 eigenvalue이고, 이 때 $\boldsymbol{u}$는 eigenvector of $A$ corresponding to $-4$가 됩니다. 이 때, $\boldsymbol{u}$의 scalar multiple 역시 위 식을 만족하기 때문에, eigenvector가 되어, -4에 대한 모든 eigenvector와 zero vector를 모은 집합


$$
Nul(A+4I) = \{k\boldsymbol{u} \mid k \in \mathbb R\}
$$


을 eigenspace of $A$ corresponding to $-4$라고 합니다. 

한편, $A\boldsymbol{v}\neq k\boldsymbol{v}$이기 때문에, $\boldsymbol{v}$는 eigenvector가 되지 않습니다.





  <br/>

*example*


$$
A = \begin{bmatrix}3 & -2 \\ 1 & 0\end{bmatrix}, \ \ \lambda=2
$$


$A$의 eigenvalue가 $\lambda=2$일 때, $2$에 해당하는 eigenvector를 찾아봅시다. eigenvector는 다음 조건을 만족해야 합니다.


$$
A\boldsymbol{x} = 2\boldsymbol{x}
$$


이  식을 정리하면


$$
(A-2I)\boldsymbol{x} =0 \\
\Rightarrow \begin{bmatrix}1 & -2 \\ 1 & -2 \end{bmatrix}\boldsymbol{x} = 0 \\
\Rightarrow \boldsymbol{x} = k\begin{bmatrix}2 \\ 1\end{bmatrix}, \ k \in \mathbb R
$$


가 됩니다. 



  <br/>

*example*


$$
A=\begin{bmatrix} 4 & -1 & 6 \\ 2 & 1 & 6 \\ 2 & -1 & 8 \end{bmatrix}, \ \ \lambda =2
$$


$A$의 eigenvalue 중 하나가 $\lambda=2$입니다. 이 때 $\lambda=2$에 해당하는 eigenspace는 


$$
A\boldsymbol{x} = 2 \boldsymbol{x}
$$


를 만족하는 solution 집합입니다. 이는


$$
(A-2I)\boldsymbol{x} = 0
$$


를 만족하는 solution 집합이고 이는


$$
Nul(A-2I)
$$


와 같습니다. 이를 구하기 위해 위 matrix equation의 augmented matrix를 이용하면


$$
\begin{bmatrix} 2 & -1 & 6 & 0 \\ 2 & -1 & 6 & 0 \\ 2 & -1 & 6 & 0 \end{bmatrix} \sim
\begin{bmatrix} 1 & -\frac{1}{2} & 3 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$


이 되어


$$
\boldsymbol{x} = x_2\begin{bmatrix}\frac{1}{2} \\ 1 \\ 0 \end{bmatrix} +x_3 \begin{bmatrix} -3 \\ 0 \\ 1 \end{bmatrix}, \ \ \ x_2, \ x_3 \ : \ free
$$


가 되어


$$
Nul(A-2I) = Span\{\begin{bmatrix}\frac{1}{2} \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} -3 \\ 0 \\ 1 \end{bmatrix}\}
$$


이 됩니다. 따라서 위 eigenspace의 basis는


$$
 \{\begin{bmatrix}\frac{1}{2} \\ 1 \\ 0 \end{bmatrix} ,\begin{bmatrix} -3 \\ 0 \\ 1 \end{bmatrix}\}
$$


입니다.





<br/>

### 2. Properties of eigenvalues and eigenvectors

<br/>



Eigenvalue와 eigenvector의 성질에 대해 알아보겠습니다.



<br/>



**Theorem**



The eigenvalues of a triangular matrix are the entries on its main diagonal



Triangular matrix의 경우 eigenvalue를 바로 구할 수 있습니다. **대각 성분이 eigenvalue가 됩니다.**



<br/>



**Theorem**



If $\boldsymbol{v_1}, ... , \boldsymbol{v_r}$ are eigenvectors that correspond to distinct eigenvalues $\lambda_1, ..., \lambda_r$ of an $n \times n $ matrix $A$, then the set $\{\boldsymbol{v_1}, ... , \boldsymbol{v_r}\}$ are linearliy independent.



**서로 다른 eigenvalue로부터 나온 eigenvector들은 linearly independent합니다.**



<br/>



**Theorem**



$A$ is not invertible if and only if one of eigenvalues of $A$ is 0



0인 Eigenvalue를 적어도 하나 가진다면 $A$는 invertible하지 않습니다. 즉, **invertible matrix의 eigenvalue는 0을 포함해서는 안됩니다.**



다음 정리들의 증명은 appendix를 참고하시기 바랍니다.



<br/>



**Solution of $\boldsymbol{x}_{k+1}=A\boldsymbol{x}_k$**



Eigenvector와 eigenvalue를 알고 있다면 다음 문제


$$
\boldsymbol{x}_{k+1}=A\boldsymbol{x}_k
$$


의 solution을 쉽게 구할 수 있습니다. 



$\boldsymbol{x}_1$ 을 $\lambda$에 해당하는 eigenvector라고 하면


$$
\begin{aligned}

\boldsymbol{x_2}&=A\boldsymbol{x_1} =\lambda \boldsymbol{x}_1 \\
\boldsymbol{x_3} &= A\boldsymbol{x_2} = \lambda^2\boldsymbol{x}_1 \\ 
\vdots \\
\boldsymbol{x_k} &= A\boldsymbol{x_{k-1}} = \lambda^{k-1}\boldsymbol{x}_1


\end{aligned}
$$


임을 알 수 있습니다.



지금까지 eigenvector와 eigenvalue에 대해 알아보았습니다. 다음 포스트에서는 characteristic equation에 대해 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



<br/>

### Appendix : Proof of theorem

<br/>





**Theorem**



The eigenvalues of a triangular matrix are the entries on its main diagonal



<br/>

* **Proof**



$n \times n$ upper triangular matrix $A$ 를 다음과 같이 정의하면


$$
A = \begin{bmatrix}a_1 & * &\cdots & * \\ 0 & a_2 & \cdots & * \\ \vdots & \vdots & \ddots & \vdots \\0 & 0 & \cdots & a_n \end{bmatrix}
$$
이 후 eigenvalue의 정의에 따라


$$
A\boldsymbol{x} =\lambda\boldsymbol{x}
$$


가 non-trivial solution을 가져야 합니다. 이는


$$
(A-\lambda I)\boldsymbol{x} =0
$$


이 non-trivial solution을 가져야 하는 것과 같습니다. 위 $A-\lambda  I$는 


$$
A -\lambda_I= \begin{bmatrix}a_1-\lambda & * &\cdots & * \\ 0 & a_2-\lambda & \cdots & * \\ \vdots & \vdots & \ddots & \vdots \\0 & 0 & \cdots & a_n-\lambda \end{bmatrix}
$$


와 같이 정의되며, 위 matrix가 invertible하지 말아야 하므로, determinant가 0이 되어야 합니다. 이는 


$$
(a_1-\lambda)(a_2-\lambda)\cdots(a_n-\lambda)=0
$$


을 만족해야 합니다. 즉 $A$의 eigenvalue는 $A$의 diagonal entries입니다. 이는 lower triangular matrix에서도 똑같이 적용됩니다.



<br/>



**Theorem**



If $\boldsymbol{v_1}, ... , \boldsymbol{v_r}$ are eigenvectors that correspond to distinct eigenvalues $\lambda_1, ..., \lambda_r$ of an $n \times n $ matrix $A$, then the set $\{\boldsymbol{v_1}, ... , \boldsymbol{v_r}\}$ are linearliy independent.



<br/>

* **Proof**



$\{\boldsymbol{v_1}, ... , \boldsymbol{v_r}\}$이 linearly dependent하다고 가정해봅시다. 그렇다면


$$
c_1\boldsymbol{v_1} + \cdots + c_r\boldsymbol{v_r} =0
$$


가 non trivial solution을 만족합니다. (즉 적어도 하나의 $c_i$가 0이 아니면서 위 식을 만족합니다.) 이 때, 어떤 한 벡터는 그 벡터의 index보다 적은 index를 가지는 벡터들의 linear combination으로 표현이 됩니다. 즉 


$$
\boldsymbol{v_j} = d_1\boldsymbol{v_1} +\cdots + d_{j-1}\boldsymbol{v_{j-1}}
$$


을 만족하는 $\boldsymbol{v_j}$가 적어도 하나 존재합니다. 이를 만족하는 $j$ 중 가장 작은 index를 $p$라고 하면




$$
\boldsymbol{v_p} = d_1\boldsymbol{v_1} + \cdots + d_{p-1}\boldsymbol{v_{p-1}}, \ \cdots(1)
$$


를 만족하면서, index가 가장 작기 때문에,  $\{\boldsymbol{v_1}, \cdots , \boldsymbol{v_{p-1}}\}$는 linearly independent합니다.  (1)의 양변에 $A$를 곱하면


$$
A\boldsymbol{v_p} = d_1A\boldsymbol{v_1} + \cdots + d_{p-1}A\boldsymbol{v_{p-1}}
$$


인데, $\boldsymbol{v_i}$는 $\lambda_i$에 해당하는 eigenvector이므로


$$
\lambda_r\boldsymbol{v_p} = d_1\lambda_1\boldsymbol{v_1} + \cdots + d_{p-1}\lambda_{p-1}\boldsymbol{v_{p-1}} \ \cdots (2)
$$


이 됩니다.



한편, (1) 양변에 $\lambda_p$을 곱하면


$$
\lambda_p\boldsymbol{v_p} = d_1\lambda_p\boldsymbol{v_1} + \cdots + d_{p-1}\lambda_{p}\boldsymbol{v_{p-1}} \ \cdots (3)p
$$


을 만족합니다. (2)식과 (3)식을 빼면


$$
0=d_1(\lambda_1-\lambda_p)\boldsymbol{v_1} + \cdots + d_{p-1}(\lambda_{p-1}-\lambda_p)\boldsymbol{v_{p-1}}
$$


가 성립합니다. 여기서 $\lambda_j$는 모두 다른 값이기 때문에, 앞의 coefficient 중 적어도 하나는 0이 아닙니다. 따라서


$$
c_1\boldsymbol{v_1} + \cdots + c_{p-1}\boldsymbol{v_{p-1}} =0
$$
 

의 non-trivial solution이 존재합니다. 즉  $\{\boldsymbol{v_1}, \cdots , \boldsymbol{v_{p-1}}\}$은 linearly dependent합니다. 여기서 모순이 발생하여 가정인 $\{\boldsymbol{v_1}, ... , \boldsymbol{v_r}\}$이 linearly dependent가 틀린 가정이 됩니다. 따라서


$$
\{\boldsymbol{v_1}, ... , \boldsymbol{v_r}\}
$$


은 linearly independent합니다. 



<br/>



**Theorem**



$A$ is not invertible if and only if one of eigenvalues of $A$ is 0

<br/>

* **Proof**



만약 $A$의 eigenvalue가 0이라면


$$
A\boldsymbol{x} = 0\cdot\boldsymbol{x}=0
$$


이 non-trivial solution을 가진다는 것을 뜻합니다. 따라서 $A$는 not invertible합니다.



