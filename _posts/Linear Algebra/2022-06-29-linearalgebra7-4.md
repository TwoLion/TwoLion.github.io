---
layout: single
title:  "7.4 Properties with Singular Value Decomposition"
categories: [Linear Algebra]
tag: [Linear Algebra, Singular Value Decomposition]
toc: true
author_profile: true #프로필 생략 여부
use_math: true

---



이번 포스트에서는 Singular value decomposition을 이용하여 얻을 수 있는 다양한 matrix와 vector space의 성질에 대해서 알아보겠습니다.





<br/>



### 1) Bases for Fundamental Subspaces



<br/>

Vector space를 배우면서, 행렬 $A$가 정의되면 $A$와 관련된 3개의 subspace를 만들 수 있다고 하였습니다. 바로 $NulA, RowA, ColA$인데요. Singular value decomposition에서 사용된 정리를 이용하면 세 space간 특별한 관계를 가진다는 것을 알 수 있습니다.



 이전 포스트에서, $m\times n$ matrix $A$에 대해서 $A$의 singular value를 $\sigma_1, ..., \sigma_r$, A를 다음과 같이 


$$
A = U\Sigma V^T
$$


로 분해하였을 때, $U$의 column인 left singular vectors,


$$
\boldsymbol u_1, ..., \boldsymbol u_m
$$


$V$의 column인 right singular vectors


$$
\boldsymbol v_1, ..., \boldsymbol v_n
$$




을 정의할 수 있습니다. 여기서 left singular vectors 중,  $\boldsymbol u_i = \frac{1}{\sigma_i}A\boldsymbol v_i, \ \ i=1, ..., r$을 이용하면,


$$
\{\boldsymbol u_1, ..., \boldsymbol u_r\}
$$




은 $ColA$의 orthonormal basis가 됩니다(singular value decomposition 정리 참고). 또한 6장에 배운 내용인


$$
(ColA)^\perp = NulA^T
$$



$$
\{\boldsymbol u_{r+1}, ..., \boldsymbol u_m\}
$$


은 $NulA^T$의 orthonormal basis가 됩니다.



다음은 right singular vector를 살펴봅시다.


$$
\boldsymbol v_{r+1}, ..., \boldsymbol v_{n}
$$


 들에 대응하는 eigenvalue는 모두 $0$입니다. 즉


$$
A^TA\boldsymbol v_j = 0, \ \ j=r+1, ..., n
$$


입니다. 이는


$$
\|A\boldsymbol v_j\| = 0 \\
\Rightarrow A\boldsymbol v_j = 0
$$


을 뜻하기 때문에,


$$
\{\boldsymbol v_{r+1}, ..., \boldsymbol v_n\}
$$


 은 $NulA$의 orthonormal basis가 됩니다. 또한 $(NulA)^\perp = ColA^T = RowA$임을 이용하면 위 basis와 orthogonal한


$$
\{\boldsymbol v_1, ..., \boldsymbol v_r\}
$$


은 $RowA$의 orthonormal basis가 됩니다.




<br/>

### 2) Reduced SVD Pseudoinverse of A

<br/>

기본 SVD에서는, $\Sigma$ matrix의 size가 $A$의 size와 같아야 했습니다. 여기서, 만약 $A^TA$의 eigenvalue가 0을 포함한다면, 즉 $m \times n$ matrix $A$에 대해서 $r$개의 singular value가 존재한다면


$$
\Sigma = \begin{bmatrix} D & 0 \\ 0 & 0 \end{bmatrix}
$$


인 partitioned matrix로 표현이 가능합니다. 이 여기서, $A$를 결정짓는 중요한 요소는 $D$ matrix이므로, 2번 째 diagonal element matrix인 0 matrix를 제거한 matrix, 즉 $D$만을 사용하며 $A$를 표현할 수 있습니다. $A$의 SVD인


$$
A = U\Sigma V^T \\
U = \begin{bmatrix}\boldsymbol u_1 & ... & \boldsymbol u_m \end{bmatrix} = \begin{bmatrix}U_r & U_{m-r}\end{bmatrix}\\
V =\begin{bmatrix}\boldsymbol v_1 & ... & \boldsymbol v_n \end{bmatrix} = \begin{bmatrix}V_r & V_{n-r}\end{bmatrix}
$$


에서 $U$와 $V$를 다음의 두 matrix로 partition한다면


$$
U_r = \begin{bmatrix} \boldsymbol u_1 & ... \boldsymbol u_r \end{bmatrix} \\
V_r = \begin{bmatrix} \boldsymbol v_1 & ... \boldsymbol v_r \end{bmatrix}
$$


$A$는 다음과 같이 표현가능합니다.


$$
A = \begin{bmatrix}U_r & U_{m-r} \end{bmatrix} \begin{bmatrix} D & 0 \\ 0 & 0  \end{bmatrix}\begin{bmatrix}V_r^T \\ V_{n-r}^T \end{bmatrix} = U_rDV_r^T
$$


다음과 같이 


$$
A =U_rDV_r^T
$$


로 분해하는 것을 **reduced singular value decomposition**이라고 합니다. 여기서, $D$는 invertible하기 때문에, 다음의 matrix


$$
A^+ = V_rD^{-1}U_r^T
$$


를 **pseudoinverse of $A$** 라고 합니다.



Pseudoinverse는 inverse는 아니지만 비슷한(가짜) 역할을 한다고 하여 붙여진 이름입니다. 또한 invertible 여부와 관련없이 pseudoinverse는 모든 matrix에 대해서 존재합니다! 다음의 예시를 보면서 pseudoinverse의 의미를 확인해보도록 합시다.



<br/>



*Example*




$$
A\boldsymbol x = \boldsymbol b
$$


다음의 equation에서 pseudoinverse of $A$를 이용하면


$$
\hat {\boldsymbol x} = A^+\boldsymbol b = V_rD^{-1}U_r^T\boldsymbol b
$$


가 됩니다. $\hat{\boldsymbol x}$에 $A$를 곱하면


$$
\begin{aligned}
A\hat{\boldsymbol x} &= AV_rD^{-1}U_r^T\boldsymbol b \\
&=UDV^TV_rD^{-1}U_r^T\boldsymbol b = UDD^{-1}U_r^T\boldsymbol b \\
& = UU_r^T  \boldsymbol b

\end{aligned}
$$


가 됩니다. 이 때, 


$$
UU_r^T\boldsymbol b
$$


는 **Projection of $\boldsymbol b$ onto the $ColA$**가 됩니다. (이는 $U_r$의 column이 orthonormal basis of $ColA$이기 때문입니다.) 즉, 위 값이 


$$
A\boldsymbol x = \boldsymbol b
$$


의 least-squares solution이 됩니다.





<br/>



지금까지 singular value decomposition을 통해 얻을 수 있는 다양한 성질에 대해서 알아보았습니다. 다음 포스트에서는 Principa Component Analysis에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!
