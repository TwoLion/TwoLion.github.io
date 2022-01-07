---
layout: single
title:  "2.4 Partitioned Matrix"
categories: [Linear Algebra]
tag: [Linear Algebra, Matrix, Partitioned matrix]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---



이번 포스트에서는 partitioned matrix에 대해 알아보겠습니다.





<br/>

### 1) Partitioned matrix



<br/>

Matrix의 row와 column을 쪼개어서, matrix 내부에 matrix가 존재하도록 partitioned matrix를 생각해볼 수 있습니다. 



<br/>

*example*


$$
A = \begin{bmatrix} 3 & 0 & 1 & 5 & 9 & -2 \\ -5 & 2& 4 &0 &-3 & 1 \\ -8 & -6 &3 & 1& 7 &-4\end{bmatrix}
= \begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}
$$


$A$를 4개의 matrix $A_{11}, A_{12}, A_{21}, A_{22}$로 나눌 수 있습니다. $A$ matrix에 수직선(partition할 열 구분), 수평선(partition할 행 구분)을 이용하여 partitioned matrix를 만들 수 있습니다. 


$$
A_{11}=\begin{bmatrix} 3 & 0 & 1 & 5 \\ -5 & 2 & 4 & 0\end{bmatrix}, \
A_{12}=\begin{bmatrix} 9 & -2 \\ -3 & 1 \end{bmatrix} \\
A_{21}=\begin{bmatrix} -8 & -6 & 3 & 1\end{bmatrix}, \
A_{21}=\begin{bmatrix} 7 & 4\end{bmatrix}
$$


 $A_{11}, A_{12}, A_{21}, A_{22}$는 4th column과 5th column 사이에 수직선을 그어 분리하고, 2th row와 3th row 사이 수평선을 그어 분리한 matrix입니다.

이 때, $A$를 $A_{11}, A_{12}, A_{21}, A_{22}$로 바꾸어 표현한 matrix를 **partitioned matrix**라고 하고, $A_{11}, A_{12}, A_{21}, A_{22}$를 **submatrix**라고 합니다.





<br/>

### 2) Operations of Partitioned Matrix



<br/>



#### (1) Addition and Scalar Multiple

<br/>



Partitioned matrix의 addition을 하려면, 같은 위치에 존재하는 submatrix의 size가 같아야 합니다. 그리고 결과는 같은 위치에 존재하는 submatrix의 합으로 나타납니다.

Scalar multiple의 경우 submatrix 각각에 scalar multiple을 하여 구할 수 있습니다.







<br/>

*example*


$$
A=\begin{bmatrix}1 & 0 & 3 &2 \\ 0 & -1 & 2 & 4 \\ 0 & 0 & 1 &3\end{bmatrix}=\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}, \\ 
B=\begin{bmatrix}3 & 0 & -1 & 0 \\ 1 & 2 & -4 & 4 \\ 0 & 0 & 1 & 1 \end{bmatrix}=\begin{bmatrix}B_{11} & B_{12} \\ B_{21} & B_{22} \end{bmatrix}
$$


이 때


$$
A_{11}=\begin{bmatrix}1 & 0 \\ 0 & -1 \end{bmatrix}, \ A_{12}=\begin{bmatrix}3 & 2 \\ 2 & 4 \end{bmatrix} \\
A_{21}=\begin{bmatrix}0 & 0 \end{bmatrix}, \ A_{22}=\begin{bmatrix}1 & 3 \end{bmatrix}
$$

$$
B_{11}=\begin{bmatrix}3 & 0 \\ 1 & 2 \end{bmatrix}, \ B_{12}=\begin{bmatrix}-1 & 0 \\ -4 & 4 \end{bmatrix} \\
B_{21}=\begin{bmatrix}0 & 0 \end{bmatrix}, \ B_{22}=\begin{bmatrix}1 & 1 \end{bmatrix}
$$


와 같이 partitioned matrix로 만들었을 때, $A_{ij}$와 $B_{ij}$의 size가 모든 $i,\ j$에 대해서 같기 때문에, $A+B$를


$$
A+B=\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}+\begin{bmatrix}B_{11} & B_{12} \\ B_{21} & B_{22} \end{bmatrix} = \begin{bmatrix}A_{11}+B_{11} & A_{12}+B_{12} \\ A_{21}+B_{21} & A_{22}+B_{22} \end{bmatrix}
$$


와 같이 나타낼 수 있으며, 합의 결과는
$$
A+B=\begin{bmatrix}4 & 0 & 2 &2 \\ 1 & 1 & -2 & 8 \\ 0 & 0 & 2 &4\end{bmatrix}
$$


가 됩니다.



scalar multiple의 경우 모든 entry에 scalar $r$을 곱하기 때문에, submatrix 각각에 $r$을 곱한 것과 같은 결과를 얻습니다.


$$
rA=r\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}=\begin{bmatrix}rA_{11} & rA_{12} \\ rA_{21} & rA_{22} \end{bmatrix}
$$


<br/>



#### (2) Multiplication



<br/>



Partitioned matrix끼리의 곱셈을 진행할 때는 두 가지 조건이 필요합니다. 



1. Partitioned matrix의 entry를 matrix가 아닌 숫자로 생각하였을 때, 일반적인 matrix 곱 조건이 성립해야 합니다.
2. 실제로 partitioned matrix의 multiplication을 진행할 때, submatrix끼리의 곱 조건이 성립해야 합니다.



위 조건이 만족되었을 때, partitoned matrix의 곱은 submatrix를 숫자로 생각하였을 때의 partitioned matrix의 곱을 진행하고, 그 결과 각 위치에 존재하는 submatrix끼리의 곱을 진행하여 얻을 수 있습니다. 



<br/>



*example*


$$
A=\begin{bmatrix}1 & 0 & 3 &2 \\ 0 & -1 & 2 & 4 \\ 0 & 0 & 1 &3\end{bmatrix}=\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}, \\ 
B=\begin{bmatrix}1 & 2 \\ -1 & 0 \\ 1 & 0 \\ 0 & 1\end{bmatrix}=\begin{bmatrix}B_1 \\ B_2\end{bmatrix}
$$



$$
A_{11}=\begin{bmatrix}1 & 0 \\ 0 & -1 \end{bmatrix}, \ A_{12}=\begin{bmatrix}3 & 2 \\ 2 & 4 \end{bmatrix} \\
A_{21}=\begin{bmatrix}0 & 0 \end{bmatrix}, \ A_{22}=\begin{bmatrix}1 & 3 \end{bmatrix}
$$

$$
B_1=\begin{bmatrix}1 & 2 \\ -1 & 0\end{bmatrix}, \ B_2=\begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix}
$$




이 때,


$$
AB =\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}\begin{bmatrix}B_1 \\ B_2  \end{bmatrix}
$$


가 됩니다. 여기서, submatrix를 숫자로 생각하였을 때, $2 \times 2$ matrix와 $2 \times 1$ matrix의 곱이므로 곱이 성립을 합니다. 따라서 곱을 진행하면


$$
AB = \begin{bmatrix}A_{11}B_1+A_{12}B_2 \\A_{21}B_1+A_{22}B_2 \end{bmatrix}
$$


가 됩니다.



여기서, $A_{ij}$는 $2 \times 2$ matrix, $B_k$ 또한 $2 \times 2$ matrix이므로 각각의 submatrix끼리의 곱이 성립합니다. 따라서 위를 계산해주면


$$
A_{11}B_1=\begin{bmatrix} 1 & 2 \\ 1 & 0 \end{bmatrix}, \ A_{12}B_2=\begin{bmatrix} 3 & 2 \\ 2 & 4  \end{bmatrix}\\
A_{21}B_1=\begin{bmatrix} 0 & 0 \end{bmatrix},\ A_{22}B_2=\begin{bmatrix} 1 & 3 \end{bmatrix}
$$


가 되어


$$
AB = \begin{bmatrix}4 & 4 \\ 3 & 4 \\ 1 & 3\end{bmatrix}
$$


가 됩니다.





<br/>



#### (3) Row Column expansion of $AB$



<br/>



Partitioned matrix를 이용하면 matrix multiplication.





<br/>



* **Theorem**



If $A$ is $m \times n$ matrix, and $B$ is $n \times p$, then


$$
AB = \begin{bmatrix}col_1{A} & col_2{A} & ... & col_n{A} \end{bmatrix} 
\begin{bmatrix}row_1(B)\\ row_2(B) \\ \vdots \\ row_n(B)\end{bmatrix}
\\
=col_1(A)row_1(B)+col_2(A)row_2(B)+\cdots+col_n(A)row_n(B)
$$




입니다. 이는 $A$의 column을 기준으로 partiton한 matrix, $B$의 row를 기준으로 partition한 matrix의 곱으로 생각해주면 됩니다. 

$col_k(A)row_k(B)$ matrix는 $m \times n$ matrix로, $(i, j)$ entry가 


$$
(col_k(A)row_k(B))_{ij}=a_{ik}b_{kj}
$$


입니다. 이를 모든 $k=1, 2, ..., n$까지 더한 값이 $AB$의 $(i, j)$th entry가 되고 이는


$$
\Sigma_{k=1}^na_{ik}b_{kj}
$$


입니다. 즉, $A$의 $i$th row와 $B$의 $j$th column의 같은 위치에 존재하는 성분의 곱을 다 더한 값이 됩니다. 



<br/>



#### (4) Inverse of partitioned matrix



<br/>



Partitioned matrix의 inverse 또한 partitoned matrix의 성질과 inverse의 정의를 이용하여 구할 수 있습니다.





<br/>



*example*


$$
A=\begin{bmatrix}A_{11} & A_{12} \\ 0 & A_{22}\end{bmatrix}
$$


where $A_{11}$ : $p \times p$, $A_22$ : $q \times q$ invertible matrix. 이 matrix의 inverse를 찾아보겠습니다.



inverse의 정의에 의해


$$
AA^{-1} = A^{-1}A=I
$$


를 만족합니다. 


$$
A^{-1}=\begin{bmatrix}X_{11} & X_{12} \\ X_{21} & X_{22} \end{bmatrix}
$$


일 때,


$$
AA^{-1}=\begin{bmatrix}A_{11}, & A_{12} \\ 0 & A_{22}\end{bmatrix}\begin{bmatrix}X_{11} & X_{12} \\ X_{21} & X_{22} \end{bmatrix} = \begin{bmatrix}A_{11}X_{11}+A_{12}X_{21} & A_{11}X_{12}+A_{12}X_{22} \\ A_{22}X_{21} & A_{22}X_{22} \end{bmatrix} =
\begin{bmatrix}I_p & 0 \\ 0 & I_q \end{bmatrix}
$$


를 만족합니다. 따라서


$$
A_{11}X_{11}+A_{12}X_{21}=I_p \\
A_{11}X_{12}+A_{12}X_{22}= 0 \\
A_{22}X_{21}=0 \\
A_{22}X_{22}=I_q
$$


를 만족하는 $X_{11}, X_{12}, X_{21}, X_{22}$가 $A^{-1}$의 submatrix가 됩니다.



$A_{22}$가 invertible하기 때문에,


$$
X_{22}=A_{22}^{-1},\  X_{21}=0
$$


입니다. 이를 $X_{21}$과 $X_{22}$에 대입하면


$$
X_{11}=A_{11}^{-1}
$$

$$
A_{11}X_{12}+A_{12}A_{22}^{-1}=0 \\

\Rightarrow X_{12}=-A_{11}^{-1}A_{12}A_{22}^{-1}
$$


이 됩니다. 따라서


$$
A^{-1} =\begin{bmatrix}A_{11}^{-1} & -A_{11}^{-1}A_{12}A_{22}^{-1} \\ 0 & A_{22}^{-1} \end{bmatrix}
$$


이 됩니다. 



<br/>



*example*


$$
A=\begin{bmatrix}B & 0 \\ 0 & C\end{bmatrix}
$$


where $B$ : $p \times p$, $C$ : $q \times q$ invertible matrix



$A^{-1}$를 구하기 위해


$$
A^{-1}=\begin{bmatrix}X_{11} & X_{12} \\ X_{21} & X_{22} \end{bmatrix}
$$


로 두고, $AA^{-1}$을 구하면


$$
AA^{-1} =\begin{bmatrix}B & 0 \\ 0 & C\end{bmatrix}\begin{bmatrix}X_{11} & X_{12} \\ X_{21} & X_{22} \end{bmatrix} = \begin{bmatrix}BX_{11} & BX_{12} \\ CX_{21} & CX_{22} \end{bmatrix} =
\begin{bmatrix}I_p & 0 \\ 0 & I_q \end{bmatrix}
$$


를 만족합니다.



$B, C$는 invertible하므로


$$
X_{11}=B^{-1}, X_{22}=C^{-1}, X_{12}=X_{21}=0
$$


가 되어 


$$
A^{-1} =\begin{bmatrix}B^{-1} & 0 \\ 0 & C^{-1}\end{bmatrix}
$$


이 됩니다. 여기서 $B, C$가 invertible하면 $A$의 inverse 또한 존재하는 것을 알 수 있습니다.



<br/>



지금까지 partitioned matrix에 대해 알아보았습니다. 다음 포스트에서는 determinant에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!