---
layout: single
title:  "2.2 Matrix Operation (2)"
categories: [Linear Algebra]
tag: [Linear Algebra, Matrix, Matrix operation]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---







이번 포스트에서는 matrix에서 정의되는 특별한 연산인 transpose와 trace에 대해 알아보겠습니다.

<br/>



### 1) Transpose

<br/>



**Definition : Transpose of matrix**



Given an $m \times n$ matrix $A$, the transpose of $A$ is the $n \times m$ matrix whose columns are formed from the corresponding rows of $A$



notation : $A^T$



Transpose를 하게 되면, 기존 matrix의 row는 column으로 바뀌게 됩니다. 이는 기존 matrix의 column이 row로 바뀌게 된다는 것과 같은 의미입니다. 즉, $A$의 $i$th row은 $A^T$의 $i$th column, $A$의 $j$th column은 $A^T$의 $j$th row가 됩니다.



*example*


$$
A=\begin{bmatrix}1 & 0 & 3 \\ 2 & 2 & 4\end{bmatrix}
$$


일 때, 


$$
A^T = \begin{bmatrix}1 & 2\\ 0 & 2 \\ 3 & 4\end{bmatrix}
$$


입니다. 



<br/>



**Properties of transpose of matrix**



transpose는 다음과 같은 성질을 가지고 있습니다.



* $(A^T)^T=A$
* $(A+B)^T=A^T+B^T$
* For any scalar $r$, $(rA)^T=rA^T$
* $(AB)^T=B^TA^T$ : matrix multiplication 후 transpose를 진행하면, transpose끼리의 곱 순서가 바뀝니다.





Transpose와 관련된 특별한 matrix인 symmetric matrix에 대해 알아보겠습니다.



<br/>



**Definition : Symmetric matrix**



A square matrix is called symmetric if $A^T=A$



즉, transpose를 취한 matrix와 취하기 전의 matrix와 같을 때, symmetric matrix라고 합니다. 또한 transpose를 취한 matrix와 취하기 전 matrix가 같아야 하기 때문에, symmetric matrix는 square matrix에서만 정의됩니다.



*example*


$$
A=\begin{bmatrix}1 & 0 & 3 \\ 0 & 2 & 2 \\ 3 & 2& 1\end{bmatrix}
$$


이 때, $A=A^T$이므로 symmetric matrix입니다.



<br/>



**Properties of symmetric matrix**



이러한 symmetric matrix는 특별한 성질을 가집니다.



$A, B$ are symmetric matrices with the same size ans $k$ is any scalar, than



* $A^T$ is symmetric
* $A+B$ and $A-B$ are symmetric
* $kA$ is symmetric



다음 성질은 symmetric matrix의 정의를 이용하여 쉽게 증명할 수 있습니다.





<br/>



### 2) Trace

<br/>





**Definition : Trace of matrix**



If $A$ is a square matrix, then the trace of $A$, denoted by $tr(A)$, is defined to be the sum of the entries on the main diagonal of $A$



trace 역시 square matrix에서 정의하고, square matrix의 main diagonal entries의 합으로 정의합니다.





*example*


$$
A=\begin{bmatrix}1 & 2& 3 & 4 \\ 2 & -1 & 2 & 3 \\ 0 & 1& 3& 4 \\ 2& 5& 1 & 9\end{bmatrix}
$$


일 때


$$
tr(A)=1+(-1)+3+9=12
$$


입니다.



<br/>



**Properties of Trace**



Trace는 다음과 같은 성질을 가지고 있습니다.



$A, B, C$  are square matrices with same size, and $k$ is scalar, $\boldsymbol{a}$ is a vector in $\mathbb{R}^n$, then



* $tr(A+B)=tr(A)+tr(B)$
* $tr(kA)=ktr(A)$
* $tr(AB)=tr(BA)$
* $tr(ABC)=tr(BCA)=tr(CAB)\neq tr(BAC)$
* $tr(A)=tr(A^T)$
* $tr(\boldsymbol{a}\boldsymbol{a}^T)=tr(\boldsymbol{a}^T\boldsymbol{a})$





지금까지 matrix에서의 특별한 연산인 transpose와 trace에 대해 알아보았습니다. 다음 포스트에서는 invertible matrix와 inverse에 대해 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!