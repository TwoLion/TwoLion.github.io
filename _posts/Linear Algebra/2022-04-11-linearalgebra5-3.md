---
layout: single
title:  "5.3 Diagonalization"
categories: [Linear Algebra]
tag: [Linear Algebra, Diagonalization]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---





이번 포스트에서는 diagonalization에 대해서 알아보겠습니다.



<br/>



### 1) Diagonalization



<br/>



#### (1) Efficiency of diagonal matrix



<br/>



Diagonal matrix는 계산상의 이점을 가지고 있습니다.



<br/>



*example*


$$
D = \begin{bmatrix}2 & 0 \\ 0 & 3\end{bmatrix}
$$


일 때,


$$
D^2 = \begin{bmatrix}2^2 & 0 \\ 0 & 3^2\end{bmatrix}, \ \ \ D^3 = \begin{bmatrix}2^3 & 0 \\ 0 & 3^3\end{bmatrix}, \ \ \ D^k = \begin{bmatrix}2^k & 0 \\ 0 & 3^k\end{bmatrix}
$$


임을 알 수 있습니다.



<br/>



*example*


$$
D =\begin{bmatrix}2 & 0 & 0\\ 0 & 3 & 0 \\ 0 & 0 & 1\end{bmatrix}
$$


일 때,


$$
D^{-1} =\begin{bmatrix}\frac{1}{2} & 0 & 0\\ 0 & \frac{1}{3} & 0 \\ 0 & 0 & 1\end{bmatrix}
$$


인 것을 알 수 있습니다. 



<br/>



*example*


$$
A = \begin{bmatrix}7 & 2 \\ -4 & 1\end{bmatrix}, \ \ P =  \begin{bmatrix}1 & 1 \\ -1 & -2\end{bmatrix}, \ \ D= \begin{bmatrix}5 & 0 \\ 0 & 3\end{bmatrix}
$$


 $A$와 $D$는 similar합니다. 즉,


$$
A = PDP^{-1}
$$


입니다. 여기서


$$
A^2 = PDP^{-1}PDP^{-1}=PD^2P^{-1}, \\
A^3 = PD^2P^{-1}PDP^{-1}=PD^3P^{-1}, \\
\vdots\\
A^k=PD^kP^{-1} = \begin{bmatrix}1 & 1 \\ -1 & -2\end{bmatrix} \begin{bmatrix}5^k & 0 \\ 0 & 3^k\end{bmatrix}\begin{bmatrix}1 & 1 \\ -1 & -2\end{bmatrix} = \begin{bmatrix} 2\cdot 5^k -3^k & 5^k -3^k \\ 2\cdot 3^k - 2\cdot 5^k & 2\cdot 3^k - 5^k\end{bmatrix}
$$


인 것을 알 수 있습니다.



위 예시들과 같이 diagonal matrix가 가지는 계산상의 이점이 많습니다. 따라서 $A$에 대해 설명할 때 $A$ 대신 $A$와 similar한 diagonal matrix $D$를 이용하여 설명하는 경우가 많습니다. $A$를 $PDP^{-1}$로 바꾸는 과정을 diagonaization이라고 합니다.



<br/>



#### (2) Diagonalization



<br/>



**Definition : Diagonalization**



A square matrix $A$ is diagonalizable if $A$ is similar to a diagonal matrix



$A=PDP^{-1}$ for some invertible matrix $P$ and some diagonal matrix $D$



어떤 matrix가 diagonalizable하다는 것은 $A$와 diagonal matrix $D$가 similar하다는 것을 뜻합니다.



어떤 matrix가 diagonalizable 여부를 확인할 수 있는 정리는 다음과 같습니다.



<br/>



**Theorem : The Diagonalization Theorem**



An $n \times n $ matrix $A$ is diagonalizable if and only if $A$ has $n$ linearly independent eigenvectors. In fact, $A=PDP^{-1}$ with $D$ a diagonal matrix, columns of $P$ are $n$ linearly independent eigenvectors of $A$, and diagonal entries of $D$ are eigenvalues of $A$ that correspond, respectively, to the eigenvectors in $P$.

In other words, $A$ is diagonalizable if and only if there are enough eigenvectors to form a basis of $\mathbb R^n$ - eigenvector basis for $\mathbb R^n$



$n \times n $ matrix $A$와 $D$가 similar하기 위한 조건은 linearly independent한 eigenvector가 $n$개가 되면 됩니다. 이 때, $P, D$를 특정할 수 있는데, $D$의 diagonal entry는 $A$의 eigenvalue가, $P$의 각 column은 $D$의 diagonal entry의 같은 index 위치의 eigenvalue에 해당하는 eigenvector가 됩니다. 



<br/>

*example* 


$$
A=\begin{bmatrix} 1 & 3 & 3 \\ -3 & -5 & -3 \\ 3 & 3 & 1 \end{bmatrix}
$$




$A$의 diagonalizable 여부를 확인하기 위해서 먼저 $A$의 eigenvalue와 eigenvector를 구해야 합니다.


$$
\det(A-\lambda I) = \begin{vmatrix}1-\lambda & 3 & 3 \\ -3 & -5-\lambda & -3 \\ 3 & 3 & 1-\lambda\end{vmatrix}
$$


위 matrix의 determinant를 계산하기가 어렵기 때문에, row operation을 통해 계산을 쉽도록 변형 후 determinant를 구하면


$$

\begin{vmatrix}1-\lambda & 3 & 3 \\ -3 & -5-\lambda & -3 \\ 3 & 3 & 1-\lambda\end{vmatrix} =  \begin{vmatrix}1-\lambda & 3 & 3 \\ -3 & -5-\lambda & -3 \\ 0 & -2-\lambda & -2-\lambda\end{vmatrix}
$$





$$
= (-1)^{2}(1-\lambda)\begin{vmatrix}-5-\lambda & -3 \\ -2-\lambda & -2-\lambda \end{vmatrix} + (-1)^{3}(-3)\begin{vmatrix}3 & 3  \\ -2-\lambda & -2-\lambda\end{vmatrix} \\

=(1-\lambda)((-5-\lambda)(-2-\lambda)+3(-2-\lambda)) +3(3(-2-\lambda)-3(-2-\lambda)) \\
=(1-\lambda)(\lambda+2)(\lambda+2) = 0
$$



$$
\lambda = -2 , 1
$$


임을 알 수 있고, $\lambda=-2$의 multiplicity는 2입니다.

이제 eigenvalue를 계산하였으니 각각의 eigenvalue에 대한 eigenvector를 구해보면



$\lambda=1$인 경우


$$
A-I =\begin{bmatrix} 0 & 3 & 3 \\ -3 & -6 & -3 \\ 3 & 3 & 0 \end{bmatrix}
$$


이 되어 $(A-I)\boldsymbol x=0$의 solution을 augmented matrix를 통해 구하면


$$
\begin{bmatrix} 1 & 3 & 3 & 0 \\ -3 & -5 & -3 & 0 \\ 3 & 3 & 1 & 0 \end{bmatrix} \sim \begin{bmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$


이 되어


$$
\boldsymbol{x} = x_3\begin{bmatrix}1 \\ -1 \\ 1 \end{bmatrix}, \ \ \ x_3 \ \ is \ \ free
$$


가 됩니다. 



$\lambda = -2$인 경우


$$
A+2I =\begin{bmatrix} 3 & 3 & 3 \\ -3 & -3 & -3 \\ 3 & 3 & 3 \end{bmatrix}
$$


이 되어 $(A+2I)\boldsymbol{x} = 0$의 solution은


$$
\begin{bmatrix} 3 & 3 & 3 & 0 \\ -3 & -3 & -3 & 0 \\ 3 & 3 & 3 & 0 \end{bmatrix} \sim \begin{bmatrix} 1 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$


이 되어


$$
\boldsymbol{x} = x_2\begin{bmatrix}-1 \\ 1 \\ 0 \end{bmatrix} +x_3\begin{bmatrix}-1 \\ 0 \\ 1 \end{bmatrix} \ \ \ x_2, x_3 \ \ is \ \ free
$$


가 됩니다. 



$\lambda=1$일 때에 대응하는 eigenspace의 basis 원소의 개수가 하나, $\lambda=-2$일 때 대응하는 eigenspace의 basis 원소의 개수가 2개입니다. 이 때 서로 다른 eigenvalue에 해당하는 eigenvector는 linearly independent하므로, 위 matrix는 diagonalizable하고, $P, D$는 다음과 같습니다.


$$
P = \begin{bmatrix}1 & -1 & -1 \\ -1 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix}, \ \ D=\begin{bmatrix}1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 2 \end{bmatrix}
$$




<br/>

*example* 


$$
A=\begin{bmatrix} 2 & 4 & 3 \\ -4 & -6 & -3 \\ 3 & 3 & 1 \end{bmatrix}
$$


다음 matrix가 diagonalizable 여부를 확인하기 위해 먼저 eigenvalue를 구해보면


$$
\det(A-\lambda I) = \begin{vmatrix}2-\lambda & 4 & 3 \\ -4 & -6-\lambda & -3 \\ 3 & 3 & 1-\lambda\end{vmatrix}
$$



$$
\begin{vmatrix}2-\lambda & 4 & 3 \\ -4 & -6-\lambda & -3 \\ 3 & 3 & 1-\lambda\end{vmatrix} = \begin{vmatrix}2-\lambda & 4 & 3 \\ -2-\lambda & -2-\lambda & 0 \\ 3 & 3 & 1-\lambda\end{vmatrix}
$$

$$
= 3\begin{vmatrix}-2-\lambda & -2-\lambda \\ 3 & 3 \end{vmatrix} + (1-\lambda) \begin{vmatrix} 2-\lambda & 4 \\-2-\lambda & -2-\lambda \end{vmatrix} \\
= (1-\lambda)(\lambda+2)^2=0
$$


이 되어


$$
\lambda = 1, -2
$$


이고, $\lambda=-2$의 multiplicity는 2입니다.



이 때, $\lambda=-2$일 때의 eigenvector를 구해보면


$$
A+2I=\begin{bmatrix} 4 & 4 & 3 \\ -4 & -4 & -3 \\ 3 & 3 & 3 \end{bmatrix}
$$


이 되어 $(A+2I)\boldsymbol{x}=0$의 solution은


$$
\begin{bmatrix} 4 & 4 & 3 & 0 \\ -4 & -4 & -3 & 0 \\ 3 & 3 & 3 & 0 \end{bmatrix} \sim \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$


이 되어


$$
\boldsymbol{x} = x_3\begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix}, x_3 \ \ is \ \ free
$$


가 되고,



$\lambda=1$일 때의 eigenvector를 구하면


$$
A-I =\begin{bmatrix} 1 & 4 & 3 \\ -4 & -7 & -3 \\ 3 & 3 & 0 \end{bmatrix}
$$


가 되어,


$$
\begin{bmatrix} 1 & 4 & 3 & 0 \\ -4 & -7 & -3 & 0 \\ 3 & 3 & 0 & 0  \end{bmatrix} \sim \begin{bmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0  \end{bmatrix}
$$

$$
\boldsymbol{x} = x_3\begin{bmatrix} 1 \\ -1 \\ 1 \end{bmatrix}, x_3 \ \ is \ \ free
$$


가 됩니다. 지금 각각의 eigenvalue에 해당하는 eigenspace의 dimension이 1이므로, 총 합이 3이 되지 않아, 해당 matrix는 diagonalizable하지 않습니다.



이를 통해 모든 matrix가 diagonalizable하지 않다는 것을 알 수 있습니다. 



다음 정리는 matrix의 diagonalizable 여부를 확인할 수 있는 정리입니다.



<br/>

**Theorem**



An $n \times n $ matrix with $n$ distinct eigenvalues is diagonalizable



$n$개의 서로 다른 eigenvalue를 가진 $n \times n$ matrix는 diagonalizable합니다. 이는 서로 다른 eigenvalue에 해당하는 eigenvector는 linearly independent한 것을 이용하면 쉽게 확인할 수 있습니다.



<br/>

**Theorem**



Let $A$ be an $n \times n$ matrix whose distinct eigenvalues are $\lambda_1, \lambda_2, ... , \lambda_p$



1. For $1\leq k \leq p$, the dimension of the eigenspace for $\lambda_k$ is less than or equal to the multiplicity of the eigenvalue $\lambda_k$
2. The matrix $A$ is diagonaliable if and only if the sume of the dimensions of the eigenspaces equals $n$. This happens if and only if
   1. The characteristic polynomial factors completely into linear factors
   2. The dimension of the eigenspace for each $\lambda_k$ equals the multiplicity of $\lambda_k$
3. If $A$ is diagonalizable and $B_k$ is a basis for the eigenspace corresponding to $\lambda_k$ for each $k$ , then the total collection of vectors in the sets $B_1, ..., B_p$ forms an eigenvector basis for $\mathbb R^n$



이 정리에서는 eigenvalue의 multiplicity에 따라 diagonalizable 여부를 확인할 수 있는 정리입니다. 

만약 특정 eigenvalue의 multiplicity가 $k$인 경우, 해당 eigenvalue의 eigenspace의 dimension이 $k$를 만족할 때, 또한 이러한 조건을 모든 eigenvalue에 대해 성립할 때, 해당 matrix는 diagonalizable하게 됩니다. 만약 특정 eigenvalue의 multiplicity가 eigenspace의 dimension보다 클 때, 해당 matrix는 diagonalizable하지 않습니다.

위의 예시를 다시보면, 첫 번째의 예시는 $\lambda=1$일 때 eigenspace dimension이 1이었고, $\lambda=-2$일 때 eigenspace dimension이 2로 각각의 multiplicity와 같았기 때문에 diagonalizable하였고, 두 번째 예시는 $\lambda=-2$일 때의 multiplicity는 2였으나 eigenspace의 dimension이 1이었기 때문에 diagonalization이 불가능하였습니다.



<br/>



*example*


$$
A = \begin{bmatrix} 5 & 0 & 0 & 0 \\ 0 & 5 & 0 & 0 \\ 1 & 4 & -3 & 0 \\ -1 & -2 & 0 & -3 \end{bmatrix}
$$


해당 matrix가 diagonalizable 여부를 확인하기 위해 eigenvalue를 구하면 위 matrix는 triangular matrix이므로


$$
\lambda =5, -3
$$


이 되고, 각각의 multiplicity가 2가 됩니다.



$\lambda=5$일 때의 eigenvector를 구하기 위해  $(A-5I)\boldsymbol x = 0$을 풀면


$$
\begin{bmatrix} 5 & 0 & 0 & 0 & 0\\ 0 & 5 & 0 & 0 & 0 \\ 1 & 4 & -3 & 0 & 0 \\ -1 & -2 & 0 & -3 & 0 \end{bmatrix} \sim \begin{bmatrix} 1 & 0 & 8 & 16 & 0\\ 0 & 1 & -4 & -4 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}
$$


이 되어


$$
\boldsymbol{x} = x_3 \begin{bmatrix}-8 \\ 4 \\ 1 \\ 0 \end{bmatrix} + x_4 \begin{bmatrix} -16 \\ 4 \\ 0 \\ 1\end{bmatrix}, \ \ x_3, x_4 \ \ is \ \ free
$$




$\lambda=-3$일 때의 eigenvector를 구하기 위해 $(A+3I)\boldsymbol{x} =0$을 풀면


$$
\begin{bmatrix} 8 & 0 & 0 & 0 & 0\\ 0 & 8 & 0 & 0 & 0 \\ 1 & 4 & 0 & 0 & 0 \\ -1 & -2 & 0 & 0 & 0 \end{bmatrix} \sim \begin{bmatrix} 1 & 0 & 0 & 0 & 0\\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}
$$


이 되어


$$
\boldsymbol{x} = x_3 \begin{bmatrix}0 \\ 0 \\ 1 \\ 0 \end{bmatrix} + x_4 \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1\end{bmatrix}, \ \ x_3, x_4 \ \ is \ \ free
$$


이 됩니다. 이 때 각 eigenvalue의 multiplicity와 eigenspace의 dimension이 2로 같기 때문에 해당 matrix는 diagonalizable하고


$$
A=PDP^{-1}, \\
P = \begin{bmatrix} -8 & -16 & 0 & 0 \\ 4 & 4 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix}, 
D=\begin{bmatrix} 5 & 0 & 0 & 0 \\ 0 & 5 & 0 & 0 \\ 0 & 0 & -3 & 0 \\ 0 & 0 & 0 & -3 \end{bmatrix}
$$


가 됩니다.



<br/>

지금까지 diagonalization에 대해서 알아보았습니다. 다음 포스트에서는 complex eigenvalue에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



<br/>



### Appendix : Proof of Theorem



<br/>



**Theorem : The Diagonalization Theorem**



An $n \times n $ matrix $A$ is diagonalizable if and only if $A$ has $n$ linearly independent eigenvectors. In fact, $A=PDP^{-1}$ with $D$ a diagonal matrix, columns of $P$ are $n$ linearly independent eigenvectors of $A$, and diagonal entries of $D$ are eigenvalues of $A$ that correspond, respectively, to the eigenvectors in $P$.

In other words, $A$ is diagonalizable if and only if there are enough eigenvectors to form a basis of $\mathbb R^n$ - eigenvector basis for $\mathbb R^n$





<br/>



* **Proof**



$\Rightarrow$


$$
A=PDP^{-1} \iff AP=PD
$$


임을 밝히면 됩니다. 



$P$ : $n\times n$ matrix with column : $\boldsymbol{v_1}, ..., \boldsymbol{v_n}$, $D$ : diagonal matrix with $\lambda_1, ..., \lambda_n$이라고 합시다.

이 경우


$$
AP = A\begin{bmatrix} \boldsymbol{v_1} & ...  & \boldsymbol{v_n} \end{bmatrix} = \begin{bmatrix} A\boldsymbol{v_1} & ...  & A\boldsymbol{v_n} \end{bmatrix}
$$


가 되고


$$
PD = \begin{bmatrix} \boldsymbol{v_1} & ...  & \boldsymbol{v_n} \end{bmatrix}\begin{bmatrix} \lambda_1 & ...  & 0 \\ \vdots & \ddots & \vdots \\0 & \cdots & \lambda_n \end{bmatrix} = \begin{bmatrix} \lambda_1\boldsymbol{v_1} & ...  & \lambda_n\boldsymbol{v_n} \end{bmatrix}
$$




이 됩니다.


$$
AP=PD \\ \begin{bmatrix} A\boldsymbol{v_1} & ...  & A\boldsymbol{v_n} \end{bmatrix} = \begin{bmatrix} \lambda_1\boldsymbol{v_1} & ...  & \lambda_n\boldsymbol{v_n} \end{bmatrix}
$$


이 되고, 이는


$$
A\boldsymbol{v_j} = \lambda_j\boldsymbol {v_j}
$$


 이므로, $D$의 diagonal entry는 eigenvalue, $P$의 column은 해당 위치의 eigenvalue에 해당하는 eigenvector가 됩니다. ㅇ





$\Leftarrow$



$\{\boldsymbol{v_1}, ..., \boldsymbol{v_n}\}$이 linearly independent한 eigenvectors고, 각각의 eigenvector에 해당하는 eigenvalue를 $\lambda_1, ..., \lambda_n$라 하면


$$
P =\begin{bmatrix} \boldsymbol{v_1} & ...  & \boldsymbol{v_n} \end{bmatrix}, \ \ D=\begin{bmatrix} \lambda_1 & ...  & 0 \\ \vdots & \ddots & \vdots \\0 & \cdots & \lambda_n \end{bmatrix}
$$


으로 $P, D$를 정의하면


$$
PD = \begin{bmatrix} \lambda_1\boldsymbol{v_1} & ...  & \lambda_n\boldsymbol{v_n} \end{bmatrix} = \begin{bmatrix} A\boldsymbol{v_1} & ...  & A\boldsymbol{v_n} \end{bmatrix} = AP
$$


가 되어 


$$
PD = AP \\
A = PDP^{-1}
$$


가 성립하여 diagonalizable합니다. 





<br/>

**Theorem**



An $n \times n $ matrix with $n$ distinct eigenvalues is diagonalizable





<br/>



* **Proof**



$n$개의 서로 다른 eigenvalue에 해당하는 eigenvector는 linearly independent합니다. 따라서 $n$개의 linearly independent한 eigenvector를 가지고 있기 때문에, diagonalizable합니다.
