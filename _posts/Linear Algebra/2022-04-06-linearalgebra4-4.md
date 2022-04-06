---
layout: single
title:  "4.4 Linear independent sets ; Basis"
categories: [Linear Algebra]
tag: [Linear Algebra, Basis]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---





이번 포스트에서는 vector space에서의 Basis에 대해 알아보겠습니다.



<br/>

### 1) Basis

<br/>



**Definition : Basis**



Let $H$ be a subspace of a vector space $V$. An indexed set of vectors $B =\{\boldsymbol{b_1}, \boldsymbol{b_2}, ..., \boldsymbol{b_p} \}$ in $V$ is a basis for $H$ if



1. $B$ is a linera independent set
2. The subspace spanned by $B$ is $H$. That is


$$
H=Span\{\boldsymbol{b_1}, \boldsymbol{b_2}, ..., \boldsymbol{b_p} \}
$$


Subspace $H$의 basis는 basis에 속한 벡터가 linearly independent여야 하고, 두 번째로, basis를 이용하여 span한 set이 $H$가 되어야 합니다.



vector space $V$의 subset $H$이 subspace가 되기 위해서는 다음의 조건이 필요합니다.



1. $H\subset V$
2. For all $\boldsymbol{u, v} \in H, \boldsymbol{u+v} \in H$
3. For all $\boldsymbol{u} \in H$ and scalar $k$, $k\boldsymbol{u} \in H$



subspace $H$에 속한 벡터는 적을수도, 무수히 많을수도 있습니다. 만약 $H$에 속한 벡터가 무수히 많다면, $H$를 설명하거나 표현할 때 어려움이 존재할 수 있고, 이해가 힘들수도 있습니다. 따라서, **$H$를 설명할 수 있는 대표 벡터를 이용하여 $H$의 특징을 설명하고자 합니다.**여기서 말하는 대표 벡터들을 모아놓은 집합이 basis입니다.

그렇다면 $H$를 대표할 수 있다는 뜻은 무엇일까요? 첫 번째로, 대표 벡터만을 이용하여 $H$를 설명할 수 있어야 합니다. 이 조건이 basis 정의에서 두 번째 조건인


$$
H=Span\{\boldsymbol{b_1}, \boldsymbol{b_2}, ..., \boldsymbol{b_p} \}
$$


조건입니다. 즉, basis에 속한 벡터들의 linear combination으로 $H$에 속한 모든 벡터를 표현할 수 있습니다.

두 번째는 대표 벡터가 중복되게 너무 많으면 안된다는 점입니다. 만약 대표 벡터들끼리 관련이 있거나 다른 대표 벡터로 표현이 가능하다면, 다른 대표 벡터들로부터 표현되는 벡터는 없어도 상관이 없습니다. 따라서, **$H$를 설명할 수 있는 가장 최소한의 벡터들을 생각을 합니다.** 최소한의 벡터 집합을 정의하기 위해 linear independence 정의를 사용합니다. 즉, Basis가 linearly independent한 조건을 통해, 중복거나 서로 관련이 없는(linearly indepenent) 최소한의 벡터를 이용하여 $H$를 설명하게 됩니다. 



정리하면, **subspace $H$의 basis $B$는 $H$를 설명하는(span 조건) 가장 최소한의 벡터를 모아놓은(linear independence 조건) 집합입니다.**



추가적으로, subspace가 아닌 vector space 또한 subspace가 되기 때문에, vector space의 basis 또한 똑같이 정의됩니다. 



<br/>

*example*



Let $A$ be an invertible $n \times n$ matrix, say $A=\begin{bmatrix}\boldsymbol{a_1}, ..., \boldsymbol{a_n} \end{bmatrix}$ . Then the columns of $A$ form a basis for $\mathbb R^n$ 



$A$의 columns이 $R^n$의 basis가 성립되기 위해서는 두 가지 조건을 확인해야 합니다. 



1. linear independence

   $A$가 invertible하므로, $A$의 column들은 linearly independent합니다.

2. Span

   $A$가 invertible하므로, $\mathbb R^n$에 속하는 모든 벡터 $\boldsymbol{b}$에 대해 $A\boldsymbol{x}=\boldsymbol{b}$는 consistent합니다. 즉, $\mathbb R^n$에 속하는 모든 벡터는 $A$의 columns의 linear combination으로 표현이 가능합니다.



두 조건을 만족하기 때문에 $A$의 column은 $\mathbb R^n$의 basis가 됩니다.

위 예시를 통해 알 수 있는 점은 **특정 vector space의 basis는 하나로 고정되는 것이 아닌 여러개가 존재할 수 있습니다.** (invertible matrix는 무수히 많으니까요.) 하지만, **basis에 속한 벡터의 개수는 같습니다.** 





<br/>

*example*



The nonzero row vectors of a matrix in row echelon form form a basis for row space



echelon form인 matrix $A$가 다음과 같이 표현된다고 해봅시다.


$$
A =\begin{bmatrix}  * & \times & \cdots & \times \\ 0 & * & \cdots &  \times \\ \vdots  & \vdots & \vdots  & \vdots  \\ 0 & 0 & 0 & 0\end{bmatrix}
$$


여기서, non-zero row의 leading entry가 모두 다릅니다. 이는 특정 non-zero row를 이를 제외한 나머지 non-zero row의 linear combination으로 표현할 수 없다는 것을 뜻합니다.(leading entry 자리를 채울 수 없기 때문이죠. ) 따라서 non-zero row들은 linearly independent합니다. 또한 row space 정의가 row들의 linear combination 모두 모아놓은 집합이므로 $A$의 row space의 basis는 non-zero row를 모아놓은 집합이 됩니다.

추가적으로, row equivalent한 두 matrix의 row space는 동일하기 때문에, basis 또한 동일합니다. 따라서 어떤 matrix $B$의 row space의 basis를 구하기 위해서는, $B$와 row equivalent한 echelon form $A$를 만든 후, $A$의 non-zero row가 $RowB$의 basis가 됩니다.



<br/>

*example*





Let $\boldsymbol{e_1}, \boldsymbol{e_2}, ..., \boldsymbol{e_n}$ be the columns of the $n \times n$ matrix $I_n$. The set $\{\boldsymbol{e_1}, ..., \boldsymbol{e_n}\}$ is called the standard basis of $\mathbb R^n$


$$
\boldsymbol{e_1} = \begin{bmatrix}1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \ \ \boldsymbol{e_2} = \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \ \ ... , \boldsymbol{e_n} =\begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}
$$


Identity matrix는 invertible하므로 $I_n$의 column은 $\mathbb R^n$의 basis가 됩니다. 따라서 standard unit vector들 또한 basis가 될 수 있습니다. **$\mathbb R^2, \mathbb R^3$에서 좌표평면, 좌표공간을 그릴 때 x축, y축, z축을 이용하여 그리는데, 축이 standard basis의 벡터 방향을 표시한 것**으로 생각하면 되겠습니다. basis의 정의를 이용하면 일반적인 $\mathbb R^n$에서도 축의 개념(basis 벡터)을 생각해볼 수 있습니다.



<br/>

*example*


$$
\boldsymbol{v_1}=\begin{bmatrix} 3 \\ 0 \\ -6\end{bmatrix}, 
\boldsymbol{v_2}=\begin{bmatrix} -4 \\ 1 \\ 7\end{bmatrix}, 
\boldsymbol{v_3}=\begin{bmatrix} -2 \\ 1 \\ 5\end{bmatrix}
$$


다음 벡터들이 $\mathbb R^3$의 basis가 되는지 확인해봅시다.



* Linearly independent



세 벡터가 linearly independent임을 확인하기 위해서 vector equation이 trivial solution이 갖는지를 확인해보겠습니다. 


$$
x_1\boldsymbol{v_1} + x_2\boldsymbol{v_2} + x_3\boldsymbol{v_3} = 0
$$


다음 vector equation의 augmented matrix를 이용하여 equation을 풀면


$$
\begin{bmatrix}3 & -4 & -2 & 0 \\ 0 & 1 & 1 & 0 \\ -6 & 7 & 5 & 0 \end{bmatrix} \sim \begin{bmatrix}1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
$$


따라서 solution이 $x_1=x_2=x_3=0$, trivial solution만을 가지기 때문에 linearly independent합니다. 



* span



세 벡터로 $\mathbb R^3$를 span하는지 확인해보겠습니다. 임의의 $\boldsymbol b \in \mathbb R^3$ 에 대해서 vector equation


$$
x_1\boldsymbol{v_1} + x_2\boldsymbol{v_2} + x_3\boldsymbol{v_3} = \boldsymbol{b}
$$


가 consistent하여야 합니다. 위의 linear independence 계산과정에서 알 수 있듯이, 위 equation의 augmented matrix의 첫 번째, 두 번째, 세 번째 column에 pivot이 존재하기 때문에, 위 equation은 반드시 consistent합니다. 즉


$$
Span\{\boldsymbol{v_1}, \boldsymbol{v_2}, \boldsymbol{v_3}\} = \mathbb R^3
$$


가 성립됩니다. 따라서 $\{\boldsymbol{v_1}, \boldsymbol{v_2}, \boldsymbol{v_3}\}$은 $\mathbb R^3$의 basis가 됩니다.









<br/>

**Theorem**



Let $S$ be a finite set of vectors in a non-zero subspace $V$



If $S$ spans $V$, but is not a basis for $V$, then a basis for $V$ can be obatined by removing appropriate vectors from $S$

If $S$ is linearly independent, but is not abasis for $V$, then a basis for $V$ can be optained by adding appropriate vectors from $V$ to $S$



위 정리는 basis의 두 조건 중 하나만 만족되었을 때, basis를 찾는 방법을 알 수 있는 정리입니다. 

만약 $S$가 $V$를 span하지만, basis가 되지 않는다면, $S$가 linearly dependent하다는 것을 뜻합니다. 따라서 span 조건은 유지하면서 적절한 벡터를 제거하여 linearly independent한 set을 만들 수 있고, 그 집합이 basis가 됩니다. (벡터를 제거하여도 span 조건이 유지될 수 있는 이유는 $S$에 속한 벡터 중 하나 이상이 나머지 벡터들의 linear combination으로 표현되기 때문입니다.)

이를 통해 $V$의 basis는 **$V$를 span하는 집합 중 가장 작은 집합**인 것을 알 수 있습니다. 

만약 $S$가 linearly independent하지만, basis가 되지 않는다면, $S$가 $V$를 span하지 못한다는 것을 뜻합니다. 따라서, $V$에 있는 벡터 중에 $S$에 추가하여도 linearly independent 성질이 유지되는 벡터가 존재합니다. 따라서 이러한 벡터를 적절히 추가하여, $S$가 $V$를 span하도록 만들 수 있고, 그 집합이 basis가 됩니다.

즉, $V$의 basis는 **$V$에 속하는 linearly independent한 집합 중 가장 큰 집합**인 것을 알 수 있습니다.





지금까지 vector space의 basis에 대해 알아보았습니다. 다음 포스트에서는 dimension에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!