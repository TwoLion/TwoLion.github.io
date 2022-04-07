---
layout: single
title:  "4.6 Rank, Nullity"
categories: [Linear Algebra]
tag: [Linear Algebra, Rank]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---









이번 포스트에서는 Rank와 nullity에 대해서 알아보도록 하겠습니다.





<br/>

### 1) Rank, Nullity



<br/>

**Definition: Rank**



The rank of $A$ is the dimension of the column space of $A$



Matrix $A$의 rank는 $ColA$의 dimension입니다.



$RowA$는 $ColA^T$이므로, $RowA$의 dimension은 $A^T$의 rank와 같습니다.





<br/>

**Definition: Nullity**



The nullity of $A$ is the dimension of the null space of $A$



Matrix $A$의 nullity는 $NulA$의 dimension입니다.



<br/>

*example*


$$
A = \begin{bmatrix}2 & 4 & -2 & 1 \\ -2 & -5 & 7 & 3 \\ 3 & 7 & -8 & 6 \end{bmatrix}
$$




이전 포스트에서 $ColA$의 dimension은 3, $NulA$의 dimension은 1인 것을 계산을 통해 구했습니다. 따라서


$$
rankA=3, \ \ nullityA=1
$$


입니다.





<br/>



### 2) Rank Theorem



<br/>

Rank Theorem은 matrix $A$로 정의되는 vector space인 $RowA, ColA, NulA$의 dimension 간의 관계를 설명해줍니다.



<br/>

**Theorem : Rank Theorem**



The dimensions of the column space and row space of an $m \times n$ matrix $A$ are equal. This common dimension, the rank of $A$, also equals the number of pivot positions in $A$ and


$$
rankA +nullityA=n
$$




Rank theorem에 따르면, $A$의 column space와 row space의 dimension이 동일합니다. 따라서 교재마다 rank를 처음 소개할 때, row space의 dimension으로 소개하는 경우도 있습니다. Rank theorem에 의해 row space와 column space의 dimension은 동일하게 되어, rank를 column space의 dimension으로 말하기도 하고, row space의 dimension으로 말할 수 있습니다.

두 번째로, matrix $A$의 rank와 nullity의 합은 $A$의 column의 개수와 동일합니다. 



위 정리를 증명하기 위해, 사용되는 정리가 하나 있습니다. 



<br/>

**Theorem : Pivot Theorem**



The pivot columns of a matrix $A$ forms a basis for $ColA$



위 정리로 인해 $A$의 rank가 pivot position 개수와 같게 됩니다. 



두 정리에 대한 증명은 appendix에 남겨놓겠습니다.



*example*


$$
A = \begin{bmatrix} 2 & -1 & 1 & -6 & 8 \\ 1 & -2 & -4 &3 & -2 \\ -7 & 8 & 10 & 3 & -10 \\4 & -5 & -7 & 0 & 4 \end{bmatrix}
$$


$A$의 rank와 nulity를 구해보도록 하겠습니다.



$A$의 pivot position을 찾기 위해 row operation을 통해 echelon form을 구하면


$$
\begin{bmatrix} 2 & -1 & 1 & -6 & 8 \\ 1 & -2 & -4 &3 & -2 \\ -7 & 8 & 10 & 3 & -10 \\4 & -5 & -7 & 0 & 4   \end{bmatrix} \sim \begin{bmatrix} 1 & -2 & -4 & -3 & -2 \\ 0 & 3 & 9 & -12 & 12 \\0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 &0  \end{bmatrix}
$$


가 됩니다. Echelon form의 leadind entry가 첫 번째, 두 번째 column에만 존재하기 때문에, $ColA$의 basis는


$$
B= \{\begin{bmatrix}2 \\ 1 \\ -7 \\ 4\end{bmatrix}, \begin{bmatrix}-1 \\ -2 \\ 8 \\ -5\end{bmatrix}\}
$$


가 되고


$$
rankA=2
$$


임을 구할 수 있습니다.  Rank theorem에 의해서


$$
rankA + nullityA = 5
$$


가 되어야 하므로 


$$
nullityA=3
$$


임을 알 수 있습니다.





지금까지 rank와 nullity에 대해 알아보았습니다. 다음 포스트에서는 coordinate system에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



<br/>



### Appendix : Proof of Theorem



<br/>

**Theorem : Pivot Theorem**



The pivot columns of a matrix $A$ forms a basis for $ColA$



<br/>

* **Proof**


$$
A = \begin{bmatrix} \boldsymbol{a_1}, ..., \boldsymbol{a_n} \end{bmatrix}
$$


인 $m \times n$ matrix $A$에 대해서


$$
ColA = Span\{\boldsymbol{a_1}, ..., \boldsymbol{a_n}\}
$$


입니다.



$ColA$의 basis를 찾기 위해서는 $\{\boldsymbol{a_1}, ..., \boldsymbol{a_n}\}$이 linearly independent한지 확인을 하고, linearly dependent하다면 적절한 벡터를 지워서 linearly independent한 set를 만들어주어야 합니다.



만약 $\{\boldsymbol{a_1}, ..., \boldsymbol{a_n}\}$가 linearly independent이면  $\{\boldsymbol{a_1}, ..., \boldsymbol{a_n}\}$는 $ColA$의 basis가 됩니다. 또한 $A$의 모든 column이 pivot column이 되기 때문에, $A$의 pivot column이 $ColA$의 basis를 형성합니다.



한편, 만약  $\{\boldsymbol{a_1}, ..., \boldsymbol{a_n}\}$이 linearly dependent하다면, $A\boldsymbol{x}=0$이 non-trivial solution을 가지는 것을 의미합니다.

이는 $A$의 column 중 pivot column이 아닌 column이 존재합니다. 또한 해당 column은 pivot column들로 표현이 가능합니다. 

따라서, pivot column이 아닌 column을 제거한 $S'$ 집합은 linearly independent한 set이 됩니다. 동시에 $ColA$를 span하구요. 따라서 $A$의 pivot column만 가지는 집합 $S'$이 $ColA$의 basis가 됩니다.





<br/>

**Theorem : Rank Theorem**



The dimensions of the column space and row space of an $m \times n$ matrix $A$ are equal. This common dimension, the rank of $A$, also equals the number of pivot positions in $A$ and


$$
rankA +nullityA=n
$$



<br/>

* **Proof**



<br/>



위 정리에서 밝혀야 할 내용은 두 가지입니다.

1. Row space와 Column space의 dimension이 같다.
2. $RankA + NullityA = n$



<br/>

Proof of 1.



Pivot theorem에 의해 $rankA$는 $A$의 pivot column 개수입니다. 이를 다시 말하면 $rankA$는 $A$와 row equivalent한 echelon form matrix $B$의 leading entry를 포함하는 row의 수, 즉 non-zero row의 수와 같습니다. echelon form matrix의 row space의 basis는 non-zero row이기 때문에, $rowB$의 dimension은 nonzero row의 개수가 됩니다. 이 때, row equivalent한 matrix의 row space는 똑같기 때문에 $RowA$의 dimension은 $A$의 pivot column의 개수와 동일합니다. 따라서


$$
rankA = \dim RowA
$$


가 성립합니다.



<br/>

Proof of 2.



$rankA = k$라고 하면 



matrix $A$는 $k$개의 pivot column을 가집니다. 이는, 


$$
A\boldsymbol{x}=0
$$


equation이 $n-k$개의 free variable을 가지게 됩니다. 즉, 위 system의 solution이 $n-k$개의 vector들의 linear combination으로 표현되고, $n-k$개의 벡터들은 linearly independent합니다. 따라서


$$
nullityA = n-k
$$


가 되어


$$
rankA + nullityA =n
$$




을 만족합니다.
