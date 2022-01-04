---
layout: single
title:  "2.3 Inverse of Matrix (2)"
categories: [Linear Algebra]
tag: [Linear Algebra, Matrix, Inverse]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"

---







이번 포스트에서는 일반적인 square matrix에 대해 inverse를 구하는 방법에 대해서 알아보겠습니다.

<br/>

### 1) Elementary matrix

<br/>



Inverse를 구하는 과정에서 elementary matrix를 이용합니다.

<br/>



* **Definition : Elementary matrix**



Elementary matrix is the matrix that is obtained by performing single elementary row operation on an identity matrix



Elementary matrix를 특정한 matrix에 곱하게 되면, matrix에 row operation을 취한 matrix와 같게 만들어줍니다. 



다음의 예시 elementary를 살펴봅시다.

<br/>



*example*

$$
E_1=\begin{bmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ -4 & 0 & 1\end{bmatrix}, \
E_2 = \begin{bmatrix}0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1\end{bmatrix}, \
E_3 = \begin{bmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 3\end{bmatrix}, \
A=\begin{bmatrix}a & b & c \\ d & e & f \\ g & h & i\end{bmatrix}
$$


다음 4개의 matrix가 있습니다. 먼저 $E_1A$를 구하면


$$
E_1A=\begin{bmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ -4 & 0 & 1\end{bmatrix} \begin{bmatrix}a & b & c \\ d & e & f \\ g & h & i\end{bmatrix} =\begin{bmatrix}a & b & c \\ d & e & f \\-4a+ g & -4b+h &-4c+i\end{bmatrix}
$$


$E_1$을 $A$에 곱한 matrix는 $A$의 3행 대신 3행에 1행의 -4배를 더한 행으로 교체된 matrix입니다. 즉, $E_1$을 곱해준 matrix는 $A$에 **replacement**를 적용한 matrix가 됩니다. 



다음, $E_2A$를 구하면


$$
E_2A=\begin{bmatrix}0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1\end{bmatrix} \begin{bmatrix}a & b & c \\ d & e & f \\ g & h & i\end{bmatrix} =\begin{bmatrix}d & e & f \\ a & b & c \\g & h &i\end{bmatrix}
$$


$E_2A$는 $A$의 1행과 2행의 위치를 바꾼 matrix입니다. 즉, $E_2$를 곱해준 matrix는 $A$에 **interchange**를 적용한 matrix가 됩니다. 



마지막으로, $E_3A$를 구하면


$$
E_2A=\begin{bmatrix}1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 3\end{bmatrix} \begin{bmatrix}a & b & c \\ d & e & f \\ g & h & i\end{bmatrix} =\begin{bmatrix}a & b & c \\ d & e & f \\3g & 3h &3i\end{bmatrix}
$$


$E_3A$는 $A$의 3행에 3을 곱해준 matrix입니다. 즉, $E_3$를 곱해준 matrix는 $A$에 **scaling**을 적용한 matrix가 됩니다. 



$E_1, E_2, E_3$처럼, 곱하였을 때 **row operation과 같은 역할을 해주는 matrix를 elementary matrix**라고 합니다.



<br/>



#### (1) Elementary matrix and invertible matrix

<br/>



$m \times n$ matrix $A$에 elementary matrix $E$를 곱한 $EA$는, $A$에 특정한 row operation($EA$와 같은 결과를 만드는 row operation)을 취한 matrix입니다. 



이 때, **모든 row operation은 반대의 row operation 또한 가지기 때문에(reversible 하기 때문에), Elementary matrix는 invertible합니다.**



$EA$를 기존의 matrix $A$로 만들어주는 row operation이 존재하고, 이는 이에 해당하는 **elementary matrix** $F$가 존재한다는 것과 같은 의미를 지닙니다.



따라서, $EF = FE = I$를 만족하므로, $F=E^{-1}, E=F^{-1}$입니다.  



각각의 elementary matrix $E_1, E_2, ..., E_k$들이 invertible하므로, 이들의 곱 $E_kE_{k-1}\cdots E_2E_1$ 또한 invertible합니다.



이 때, $E_kE_{k-1}\cdots E_2E_1$는 $E_1$에 해당하는 row operation, $E_2$에 해당하는 row operation, ..., $E_k$에 해당하는 row operation을 차례대로 취한 결과와 같게 만들어주는 matrix입니다.



즉, $E_kE_{k-1}\cdots E_2E_1A$는 $A$에 각각에 elementary matrix에 해당하는 row operation을 순서대로 적용한 matrix로 생각할 수 있습니다.



만약, $A$**에 row operation을 취하여 Identity matrix** $$I$$(이 때 identity matrix는 $A$의 reduced echelon form입니다.)를 만들 수 있다면, 이는


$$
E_kE_{k-1}\cdots E_2E_1A=I
$$


를 뜻하고,  $A$가 invertible matrix인 것을 의미합니다. 또한 $A$의 inverse는


$$
A^{-1}=E_kE_{k-1}\cdots E_2E_1
$$


임을 알 수 있습니다. 이를 일반화한 정리가 다음 정리입니다.

<br/>



**Theorem**



An $n \times n $ matrix $A$ is invertible if and only if $A$ is row equivalent to $I_n$, and in this case, any sequence of elementary row operations that reduces $A$ to $I_n$ also transforms $I_n$ into $A^{-1}$



즉, $A$가 $I_n$과 row equivalent하면, $A$는 invertible하고, $A$를 $I_n$으로 만드는 row operation 과정을 $I_n$에 적용하였을 때 얻어지는 matrix가 $A^{-1}$이 됩니다. (증명은 appendix 참고)



이 정리를 이용하면, 임의의 square matrix $A$가 invertible matrix인지 아닌지, invertible하면 inverse가 무엇인지 바로 알 수 있습니다.

<br/>



### 2) Algorithm for Finding $A^{-1}$

<br/>



위의 정리를 이용하여 $A^{-1}$를 찾는 방법에 대해 알아보겠습니다.

<br/>



#### (1) Algorithm for finding $A^{-1}$

<br/>



Row reduce the augmented matrix $\begin{bmatrix}A & I\end{bmatrix}$.  If $A$ is row equivalent to $I$, then $\begin{bmatrix}A & I\end{bmatrix}$ is row equivalent to $\begin{bmatrix}I & A^{-1}\end{bmatrix}$ . Otherwise, $A$ does not have an inverse.



위의 정리 내용은 $A$가 $I$와 row equivalent하면 $A$는 invertible하다는 것이었습니다. 또한 $A$를 $I$로 만드는 row operation은 $I$를 $A^{-1}$로 만드는 row operation입니다. 그리고, row operation에 대응하는 elementary matrix가 존재합니다. 이 셋을 종합하면, $A$를 $I$로 만드는 row operation을


$$
E_kE_{k-1}\cdots E_2E_1
$$


elementary matrix의 곱으로 두면,


$$
E_kE_{k-1}\cdots E_2E_1A=I
$$


가 되어


$$
A^{-1}=E_kE_{k-1}\cdots E_2E_1 
$$


가 됩니다. 여기서, $E_kE_{k-1}\cdots E_2E_1$는,


$$
E_kE_{k-1}\cdots E_2E_1 = E_kE_{k-1}\cdots E_2E_1I
$$


입니다. 즉, $A^{-1}$은 **Identity matrix에 $A$를 $I$로 만드는 row operation을 적용한 matrix입니다.**



따라서, $A^{-1}$를 찾는 알고리즘은


$$
\begin{bmatrix}A & I\end{bmatrix}
$$




matrix를 생각합니다. 만약 $A$가 $n \times n$ matrix이면 위 matrix는 $n \times 2n$ matrix입니다. 여기서, $A$와 $I$는 row operation을 진행하는데 영향을 미치지 않고, $A$에서 $I$로 만드는 row operation이 $I$를 $A^{-1}$로 만드는 row operation이므로, 이 row operation을 진행해주면


$$
E_kE_{k-1}\cdots E_2E_1\begin{bmatrix}A & I\end{bmatrix} = \begin{bmatrix}I & A^{-1}\end{bmatrix}
$$


이 됩니다. 즉, $A$를 $I$로 만들었을 때, $I$는 $A^{-1}$이 됩니다. 만약, $A$가 $I$와 row equivalent하지 않으면, row operation을 취했을 때, $I$가 나오지 않을 것이고, 따라서 invertible하지 않은 것을 알 수 있습니다. 



정리하면


$$
\begin{bmatrix}A & I\end{bmatrix} \sim \begin{bmatrix}I & A^{-1}\end{bmatrix}
$$


로 만들어 $A$의 inverse를 구할 수 있습니다.

<br/>



*example*

$$
A=\begin{bmatrix}0 & 1 &2 \\ 1 & 0 & 3 \\ 4 & -3 & 8\end{bmatrix}
$$


의 inverse를 구하기 위해 $\begin{bmatrix}A & I\end{bmatrix} $를 만들어서 이를 $\begin{bmatrix}I & A^{-1}\end{bmatrix} $로 만들어주면


$$
\begin{bmatrix}0 & 1 &2 & 1 & 0 & 0 \\ 1 & 0 & 3 & 0 & 1 & 0 \\ 4 & -3 & 8 & 0 & 0 &1\end{bmatrix} \sim 
\begin{bmatrix}1 & 0 &0 & -\frac{9}{2} & 7 & -\frac{3}{2} \\ 0& 1 & 0 & -2 & 4 & -1 \\ 0 & 0 & 1 & \frac{3}{2} & -2 & \frac{1}{2}\end{bmatrix}
$$


가 되어, 


$$
A^{-1}= \begin{bmatrix} -\frac{9}{2} & 7 & -\frac{3}{2} \\-2 & 4 & -1 \\\frac{3}{2} & -2 & \frac{1}{2}\end{bmatrix}
$$


가 됩니다.



*example*


$$
B=\begin{bmatrix}1 & -2 &-1 \\ -1 & 5 & 6 \\ 5 & -4 & 5\end{bmatrix}
$$


의 inverse를 구하기 위하여,  $\begin{bmatrix}B & I\end{bmatrix} $를  $\begin{bmatrix}I & B^{-1}\end{bmatrix} $로 만들어주는 과정을 진행하면


$$
\begin{bmatrix}1 & -2 &-1 & 1 & 0 & 0 \\ -1 & 5 & 6 & 0 & 1 & 0 \\ 5 & -4 & 5 & 0 & 0 &1\end{bmatrix} \sim 
\begin{bmatrix}1 & -2 &-1 & 1 & 0 &0 \\ 0& 3 & 5 & 1 & 1 & 0 \\ 0 & 0 & 0 & -7 & -2 & 1\end{bmatrix}
$$


위 matrix가 나옵니다.  $\begin{bmatrix}B & I\end{bmatrix} $에서 $B$ 부분을 row operation을 취해주었을 때 zero row가 발생하여, $B$는 $I$와 row equivalent하지 않은 것을 알 수 있습니다. 따라서 $B$는 invertible하지 않습니다.



<br/>



#### (2) Another view of matrix inversion

<br/>



$A^{-1}$를 찾는 algorithm에서 사용되는 $\begin{bmatrix}A & I\end{bmatrix} $ matrix에 대한 다른 해석입니다.



위에서 정의된 identity matrix의 column을 $\boldsymbol{e_1}, \boldsymbol{e_2}, ..., \boldsymbol{e_n}$이라 하면


$$
I=\begin{bmatrix}\boldsymbol{e_1} & \boldsymbol{e_2} & ... & \boldsymbol{e_n}\end{bmatrix}
$$


으로 해석할 수 있습니다.



여기서


$$
\begin{bmatrix}A & I\end{bmatrix} \sim \begin{bmatrix}I & A^{-1}\end{bmatrix}
$$




를 찾는 과정을 $I$의 각각의 column에 대해서 해석을 하면


$$
\begin{bmatrix}A & \boldsymbol{e_j}\end{bmatrix} \sim \begin{bmatrix}I & ?\end{bmatrix}
$$


로 생각할 수 있습니다. $\begin{bmatrix}A & \boldsymbol{e_j}\end{bmatrix}  $를 어떤 linear system의 augmented matrix로 생각하면, 이 linear system은


$$
A\boldsymbol{x}=\boldsymbol{e_j}
$$


로 생각할 수 있습니다. 즉, 이러한 linear system, 또는 matrix equation을 $I$의 각 column에 대해 실시하므로


$$
A\boldsymbol{x}=\boldsymbol{e_1}, A\boldsymbol{x}=\boldsymbol{e_2}, ..., A\boldsymbol{x}=\boldsymbol{e_n}
$$


과 같이 **n개의 matrix equation의 solution을 이용하여 $A^{-1}$의 column을 구하게 됩니다.**



하지만 위 matrix equation을 풀 때, augemented matrix를 reduced echelon form으로 만드는 row operation이 n개의 matrix equation에 동일한 row operation이 적용되기 때문에,


$$
\begin{bmatrix}A & I\end{bmatrix} \sim \begin{bmatrix}I & A^{-1}\end{bmatrix}
$$


를 이용하여 한번에 구하게 됩니다.



지금까지 $A^{-1}$를 구하기 위해 elementary matrix와 algorithm에 대해 알아보았습니다. 다음 포스트에서는 Invertible Matrix Theorem에 대해 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!

<br/>



#### Appendix : Proof of Theorem

<br/>



#### (1) Elementary matrix and invertible matrix

<br/>



**Theorem**





An $n \times n $ matrix $A$ is invertible if and only if $A$ is row equivalent to $I_n$, and in this case, any sequence of elementary row operations that reduces $A$ to $I_n$ also transforms $I_n$ into $A^{-1}$

<br/>



* **proof**



* $\Rightarrow$ **방향**



Matrix $A$ 가 invertible할 때, $A$가 $I$와 row equivalent한 것을 밝혀야 합니다. 



$I_n$을 보게 되면, $I_n$은 reduced echelon form을 만족합니다. 즉, $A$가 $I_n$과 row equivalent하다는 것은, $A$의 reduced echelon form이 $I_n$인 것을 의미합니다. 



그런데 $I_n$을 보면 모든 diagonal entry 값이 1입니다. **즉, zero row** 가 존재하지 않습니다.



따라서 $A$를 row operation을 통해 reduced echelon form을 만들었을 때, $I_n$이 되거나, zero row가 존재하는 reduced echelon form $T$가 될 수도 있습니다.



$A$가 $I_n$이 아닌, $T$와 row equivalent하다고 가정해봅시다. 즉, $A$의 reduced echelon form이 zero row를 포함한다고 가정해봅시다.



그리고 다음의 matrix equation을 생각해봅시다.


$$
A\boldsymbol{x}=\boldsymbol{b}
$$


$A$가 invertible하기 때문에 $\mathbb{R}^n$에 존재하는 모든 $\boldsymbol{b}$에 대해서 위 equation은 unique한 solution을 가져야 합니다.



하지만 만약 $A$가 $T$와 row equivalent하다면, 위의 matrix equation과 같은 solution을 가지는 linear system의 augmented matrix가


$$
\begin{bmatrix}A &\boldsymbol{b}\end{bmatrix}
$$


가 되고, 이를 reduced echelon form을 만들면


$$
\begin{bmatrix}T &\boldsymbol{b'}\end{bmatrix}
$$


이 됩니다. 이 때, $\mathbb{R}^n$**에 속하는 모든** $\boldsymbol{b}$에 대해서 위 matrix equation이 solution이 존재해야 하는데, $T$는 zero row를 포함하기 때문에, 위 reduced echelon form에서 하나의  row가


$$
\begin{bmatrix}0 & 0 & ... & 0 & b''\end{bmatrix}, \ b''\neq0
$$


가 되는 $\boldsymbol{b}$가 존재합니다. 즉, 위 matrix equation의 solution이 존재하지 않는 $\boldsymbol{b}$가 존재합니다. $A$가 invertible하므로 모든 $\boldsymbol{b}$에 대해서 matrix equation이 solution이 존재해야하는데 모순이 발생하였기 때문에 증명에서 내린 가정



$A$가 $I_n$이 아닌, $T$와 row equivalent하다



가 틀린 가정이 됩니다. 따라서



$A$는 $I_n$와 row equivalent합니다.





* $\Leftarrow$ 방향



$A$가 $I_n$과 row equivalent하면 $A$는 invertible하다는 것을 밝혀야 합니다.



$A$가 $I_n$과 row equivalent하므로, $A$에서 row operation을 취하여 $I_n$을 만들 수 있습니다. row operation에 해당하는 elementary matrix를 $E_1, E_2, ..., E_k$라고 하면


$$
E_kE_{k-1}\cdots E_2E_1A=I
$$


를 만족합니다. elementary matrix는 invertible하고, 위의 식을 통해서 $A$ 또한 invertible하고


$$
A^{-1}=E_kE_{k-1}\cdots E_2E_1
$$


 임을 알 수 있습니다.



<br/>



* Theorem



위 정리에서 추가적으로, 



 any sequence of elementary row operations that reduces $A$ to $I_n$ also transforms $I_n$ into $A^{-1}$



의 증명은 다음과 같습니다.



* **proof**



앞선 증명에서


$$
A^{-1}=E_kE_{k-1}\cdots E_2E_1
$$


인 것을 알 수 있습니다.



이 때,


$$
E_kE_{k-1}\cdots E_2E_1=E_kE_{k-1}\cdots E_2E_1I
$$


로 표현할 수 있습니다. 즉,  identity matrix에 $A$를 $I$로 만드는 row operation을 적용한 matrix가 $A^{-1}$입니다.







