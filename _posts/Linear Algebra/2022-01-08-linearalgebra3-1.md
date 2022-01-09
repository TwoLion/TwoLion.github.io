---
layout: single
title:  "3.1 Determinant"
categories: [Linear Algebra]
tag: [Linear Algebra, Determinant,]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---



d

d

d

d

d



이번 포스트에서는 determinant에 대해서 알아보겠습니다.



<br/>

### 1) Determinant





<br/>

$2 \times 2$​ matrix 


$$
A=\begin{bmatrix}a & b \\ c& d \end{bmatrix}
$$


의 determinant는


$$
detA = ad-bc
$$




였습니다. 만약 $detA \neq 0$이면 $A$는 invertible하고, $detA=0$이면 $A$는 invertible하지 않습니다.



이번 chapter에서는 일반적인 $n \times n $ matrix의 determinant를 구하는 방법에 대해 알아보겠습니다.





<br/>

*example* 



$3 \times 3$ matrix $A$가 다음과 같습니다.


$$
A=\begin{bmatrix}a_{11} & a_{12} & a_{13} \\ a_{21}& a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33}  \end{bmatrix}
$$


만약 $A$가 invertible하면, $A$의 pivot column의 수가 3개여야 합니다. 따라서 row operation을 통해 echelon form을 만들어주면


$$
A \sim \begin{bmatrix}a_{11} & a_{12} & a_{13} \\ 0 & a_{11}a_{22}-a_{11}a_{21} & a_{11}a_{23}-a_{11}a_{21} \\ 0 & a_{11}a_{32}-a_{11}a_{31} & a_{11}a_{33}-a_{11}a_{31}  \end{bmatrix} \sim \begin{bmatrix}a_{11} & a_{12} & a_{13} \\ 0 & a_{11}a_{22}-a_{11}a_{21} & a_{11}a_{23}-a_{11}a_{21} \\ 0 & 0 & a_{11}\Delta  \end{bmatrix} 
$$


로 만들어지고, $\Delta$는


$$
\begin{aligned}
\Delta &= a_{11}a_{22}a_{33}+a_{12}a_{23}a_{31}+a_{13}a_{21}a_{32}-a_{11}a_{23}a_{32}-a_{12}a_{21}a_{33}-a_{13}a_{22}a_{31} \\
&=a_{11}(a_{22}a_{33}-a_{23}a_{32})-a_{12}(a_{21}a_{33}-a_{23}a_{31})+a_{13}(a_{21}a_{32}-a_{22}a_{31})
\end{aligned}
$$


과 같이 정리됩니다. 이 때, $a_{11}$에 곱해진 $a_{22}a_{33}-a_{23}a_{32}$는


$$
A_1=\begin{bmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{bmatrix}
$$


의 determinant이며, 마찬가지로, $a_{12}$와 $a_{13}$에 곱해진 $a_{21}a_{33}-a_{23}a_{31}$, $a_{21}a_{32}-a_{22}a_{31}$은 각각


$$
A_2=\begin{bmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{bmatrix}, \\
A_3=\begin{bmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}
$$


의 determinant입니다. 위 세 행렬을 자세히 보면, $A_{1}$ matrix는 $A$에서 $a_{11}$에 해당하는 row와 column(1행 1열)을 제외한 나머지 matrix가 되고, $A_2$는 $a_{12}$에 해당하는 row와 column(1행 2열)을 제외한 나머지 matrix, $A_3$는 $a_{13}$에 해당하는 row와 column(1행 3열)을 제외한 나머지 matrix가 됩니다.



$A$가 invertible하려면 $\Delta \neq0$이어야 하기 때문에, 이 값을 $3 \times 3$ matrix의 determinant로 정의합니다. Determinant를 계산하였을 때, 하나의 행의 entry 값과 그 entry에 해당하는 row와 column을 제외한 나머지 matrix의 determinant의 조합으로 정의됩니다. 





<br/>



#### (1) Cofatctor



<br/>



* **Defintion : Minor, Cofactor**



$A : n \times n$ matrix일 때,



**Minor of entry **$a_{ij}$ : Determinant of submatrix that remains when $i$th row and $j$th column of $A$ are deleted



Notation : $M_{ij}$



**Cofactor of entry $a_{ij}$** : $C_{ij} = (-1)^{i+j}M_{ij}$



$(i, j)$ entry의 minor는 $A$에서 $i$행과 $j$ 열을 제외하고 만든 submatrix의 determinant입니다.

$(i, j)$ entry의 cofactor는 $(i, j)$ entry의 minor에 $(-1)^{i+j}$를 곱한 값으로 정의합니다. 





<br/>



*example*


$$
A=\begin{bmatrix} 1 & 2 & 3 \\ -4 & 5& 6 \\ 7 & -8 & 9 \end{bmatrix}
$$


일 때


$$
C_{11} = (-1)^{1+1} det(\begin{bmatrix}2 & 3 \\ 5 & 6 \end{bmatrix}) = 93 \\
C_{12} = (-1)^{1+2} det(\begin{bmatrix}-4 & 6 \\ 7 & 9 \end{bmatrix}) = 78 \\
C_{13} = (-1)^{1+3} det(\begin{bmatrix}-4 & 5 \\ 7 & -8 \end{bmatrix}) = -3 \\


$$


입니다. 나머지 entry의 cofactor 또한 같은 방법으로 구할 수 있습니다.



<br/>



#### (2) Determinant



<br/>



Cofactor를 이용하여 determinant를 정의할 수 있습니다.

<br/>



* **Definition : Determinant**



The determinant of an $n \times n $ matrix $A$ can be computed by multiplying the entries in any row (or column) by their cofactors and adding the resulting products



$det(A)=a_{i1}C_{i1}+a_{i2}C_{i2}+ \cdots + a_{in}C_{in}$

$det(A)=a_{1j}C_{1j}+a_{2j}C_{2j}+ \cdots + a_{nj}C_{nj}$



$A$의 determinant을 구하기 위해서는 먼저 $A$의 특정한 행 또는 열을 선택을 합니다. 선택을 한 행(또는 열)에 대해서 각각의 entry 값과, entry의 cofactor를 곱해준 뒤, 더한 값이 determinant입니다.



임의의 행 또는 열을 선택해서 계산을 하더라도 모두 같은 결과가 나오기 때문에, 실제로 determinant를 계산할 때는 계산이 간편해지는 row나 column을 선택하여 계산합니다.



<br/>



*example*


$$
A=\begin{bmatrix}1 & 2 & 3 \\ -4 & 5 &6 \\ 7 & -8 & 9 \end{bmatrix}
$$


$A$의 determinant을 구하기 위해, 1행을 기준으로, 1열을 기준으로 determinant를 구해보겠습니다.

먼저 1행을 기준으로 determinant를 구하면


$$
det(A)=1 \cdot C_{11} +2\cdot C_{12} +3 \cdot C_{13}
$$
 

이고, 위에서 계산한 cofactor를 이용하면


$$
det(A) = 1\cdot 93 + 2\cdot 78 + 3 \cdot(-3) = 240
$$


이 나옵니다. 1열을 기준으로 determinant를 계산하면


$$
det(A)=1 \cdot C_{11} -4\cdot C_{21} +7 \cdot C_{31}
$$


이고


$$
C_{21}=(-1)^{2+1}det(\begin{bmatrix}2 & 3 \\ -8 & 9  \end{bmatrix}) = -42 \\
C_{31} = (-1)^{3+1}det(\begin{bmatrix}2 & 3 \\ 5 &6\end{bmatrix})=-3
$$


임을 이용하면


$$
det(A)=1\cdot93 + (-4)\cdot(-42) + 7 \cdot(-3) = 240
$$


을 얻을 수 있습니다. 임의의 열이나 행을 선택하여 determinant를 구하더라도 결과는 같은 값으로 나옵니다.



