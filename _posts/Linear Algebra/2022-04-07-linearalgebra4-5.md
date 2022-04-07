---
layout: single
title:  "4.5 Dimension"
categories: [Linear Algebra]
tag: [Linear Algebra, Dimension]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---





이번 포스트에서는 vector space의 dimension에 대해 알아보겠습니다.





<br/>

### 1) Dimension

<br/>



**Definition : Dimension**



If $V$ is spanned by a finite set, then $V$ is said to be finite-dimensional, and the dimension of $V$, written as $\dim V$ is the number of vectors in a basis for $V$



The dimension of the zero vector space $\{0\}$  is defined to be zero.



If $V$ is not spanned by a finite set, then $V$ is said to be infinite-dimensional.



Vector space의 dimension은 basis의 벡터의 개수로 정의됩니다. 

Zero vector로만 이루어진 vector space의 경우 basis가 없기 때문에(linearly dependent하기 때문입니다.) dimension을 0으로 따로 정의합니다.

또한 basis가 무한히 많은 vector space는 infinite-dimensional하다고 정의합니다.

하나의 vector space의 basis는 여러개 존재할 수 있지만, dimension은 같은 값으로 고정됩니다.



<br/>

*example*



$\dim \mathbb R^n$



basis for $\mathbb R^n$ : $\{\boldsymbol{e_1}, \boldsymbol{e_2}, ..., \boldsymbol{e_n}\}$

basis에 속한 벡터의 개수가 $n$개이므로, dimension은 n입니다. 

우리가 일반적으로 좌표평면의 경우 2차원, 좌표 공간의 경우 3차원이라고 하는 이유 또한 basis와 dimension 정의로부터 알 수 있습니다.



<br/>

*example*


$$
H = \{\begin{bmatrix}a-3b+6c \\ 5a+4d \\ b-2c-d \\ 5d \end{bmatrix}\mid a, b, c, d \in \mathbb R \}
$$


 $H$는 다음과 같이 표현될 수 있습니다.


$$
H = \{a\begin{bmatrix}1 \\ 5 \\ 0 \\ 0 \end{bmatrix} + b\begin{bmatrix}-3 \\ 0 \\ 1 \\ 0 \end{bmatrix} 
+ c\begin{bmatrix}6 \\ 0 \\ -2 \\ 0 \end{bmatrix} + d\begin{bmatrix}0 \\ 4 \\ -1 \\ 5 \end{bmatrix}\mid a, b, c, d \in \mathbb R \}
$$


따라서


$$
H = Span\{\boldsymbol{v_1}, \boldsymbol{v_2}, \boldsymbol{v_3}, \boldsymbol{v_4}\} \\

\boldsymbol{v_1} =\begin{bmatrix}1 \\ 5 \\ 0 \\ 0 \end{bmatrix}, \boldsymbol{v_2}=\begin{bmatrix}-3 \\ 0 \\ 1 \\ 0 \end{bmatrix}, 
\boldsymbol{v_3}=\begin{bmatrix}6 \\ 0 \\ -2 \\ 0 \end{bmatrix}, \boldsymbol{v_4}=\begin{bmatrix}0 \\ 4 \\ -1 \\ 5 \end{bmatrix}\
$$


입니다. $H$의 dimension을 구하기 위해서는 $H$의 basis를 구해야 합니다. basis의 조건 중 span 조건은 만족했으니, linear independence 조건만 만족하면 됩니다. 하지만, $\boldsymbol{v_3} = -2\boldsymbol{v_2} $이기 때문에, 4개의 벡터는 linearly dependent합니다. 따라서, $\boldsymbol{v_2}, \boldsymbol{v_3}$ 중 하나를 제거한다면,


$$
B = \{\boldsymbol{v_1}, \boldsymbol{v_2}, \boldsymbol{v_4}\}
$$
$B$는 linearly independent하게 되어 $H$의 basis가 됩니다. 따라서


$$
\dim H=3
$$


이 됩니다.



<br/>

*example*





다음 linear system


$$
\begin{aligned}

2x_1 +4x_2 -2x_3+x_4&=0 \\
-2x_1-5x_2+7x_3+3x_4&=0 \\
3x_1+7x_2 -8x_3+6x_4&=0



\end{aligned}
$$




의 solution space를 생각해봅시다. 다음 system의 coefficient matrix를


$$
A = \begin{bmatrix} 2 & 4 & -2 & 1 \\-2 & -5 & 7 &3 \\ 3 & 7 & -8 & 6 \end{bmatrix}
$$


이 되고, 위 linear system의 solution space는 $A$의 null space가 됩니다. 따라서 $NulA$를 구해보면


$$
 \begin{bmatrix} 2 & 4 & -2 & 1 & 0\\-2 & -5 & 7 &3 &0\\ 3 & 7 & -8 & 6&0 \end{bmatrix} \sim \begin{bmatrix} 1 & 0 & 9 & 0 & 0\\0 & 1 & -5 &0 &0\\ 0 & 0 & 0 & 1&0 \end{bmatrix}
$$


 가 되어


$$
\boldsymbol{x} = x_3\begin{bmatrix} -9 \\ 5 \\ 1 \\0\end{bmatrix},\ \ x_3 :  free
$$


가 됩니다. 즉, basis는
$$
B = \{\begin{bmatrix} -9 \\ 5 \\ 1 \\0\end{bmatrix} \}
$$


가 되어, dimension은 1이 됩니다.





<br/>

### 2) Property of basis

<br/>



**Theorem**



If $V$ is a non-zero subspace of $\mathbb R^n$, then there exists a basis for $V$ that has at most $n$ vectors, i.e. $\dim V \leq n$



$\mathbb R^n$의 non-zero subspace는 반드시 basis가 존재하고, dimension은 $n$보다 작습니다. 이를 일반화한 정리는 다음 정리 입니다.



<br/>



**Theorem**



If $V$ and $W$ are subspaces of $\mathbb R^n$, and if $V$ is a subspace of $W$, then


$$
0\leq \dim V \leq \dim W \leq n
$$


$V=W$ if and only if $\dim V =\dim W$



어떤 vector space의 subspace는 dimension이 자신을 포함하는 vector space보다 작거나 같습니다. 같은 경우, 두 vector space는 같은 space가 됩니다.



<br/>



**Theorem**



Let $S$ be a nonempty set of vectos in a vector space $V$, and let $S'$ be a set that results by adding additional vectors in $V$ to $S$



* If the additional vectors are in $SpanS$, then $SpanS' = SpanS$
* If $SpanS'=SpanS$, then the additional vectors are in $SpanS$
* If $SpanS$ and $SpanS'$ have the same dimension, then the additional vectors are in $SpanS$ and $SpanS'=SpanS$



Span의 성질에 대해 다룬 정리입니다. span의 정의와 linear independence를 적용하면 쉽게 확인할 수 있습니다.





<br/>



**Theorem**



Le $V$ be a p-dimensional vector space, $p\geq 1$

Any linear independent set of exactly p elements in $V$ automatically a basis for $V$

Any set of exactly p elements that spans $V$ is automatically a basis for $V$



dimension을 알고 있다면, dimension의 수만큼의 벡터가 basis 조건 중 하나만 만족해도(span or linear independence) 그 집합은 basis가 됩니다.



정리에 대한 증명은 appendix를 통해 확인하면 되겠습니다.



<br/>



### 3) Finding dimension of $NulA$ and $ColA$



<br/>



Null space와 column space와의 dimension은 특수한 관계가 존재합니다. 다음의 예를 통해 알아보도록 하겠습니다.



<br/>

*example*


$$
A = \begin{bmatrix}2 & 4 & -2 & 1 \\ -2 & -5 & 7 & 3 \\ 3 & 7 & -8 & 6 \end{bmatrix}
$$


먼저 $NulA$의 dimension을 구해보겠습니다. $NulA$를 구하기 위해 $A\boldsymbol{x}=0$을 풀면


$$
\begin{bmatrix}2 & 4 & -2 & 1 & 0 \\ -2 & -5 & 7 & 3 & 0 \\ 3 & 7 & -8 & 6&0 \end{bmatrix} \sim 
\begin{bmatrix}1 & 0 & 9 & 0 & 0 \\ 0 & 1 & -5 & 0 & 0 \\ 0 & 0 & 0 & 1&0 \end{bmatrix}
$$


이 되어 


$$
\boldsymbol{x} = x_3\begin{bmatrix}-9 \\ 5 \\ 1 \\ 0 \end{bmatrix}, \ \ x_3 \ \ is \ \ free
$$
입니다. 따라서 null space의 basis는 


$$
B=\{ \begin{bmatrix}-9 \\ 5 \\ 1 \\ 0 \end{bmatrix} \}
$$


가 되고, 


$$
\dim NulA = 1
$$


이 됩니다. 



다음은 $ColA$의 dimension을 구해보겠습니다. $ColA$는


$$
ColA=Span\{\begin{bmatrix}2 \\ -2 \\ 3 \end{bmatrix}\, \begin{bmatrix}4 \\ -5 \\ 7 \end{bmatrix}, \begin{bmatrix}-2 \\ 7 \\ -8 \end{bmatrix}, \begin{bmatrix}1 \\ 3 \\ 6 \end{bmatrix}\}
$$


입니다. basis를 확인하기 위해서는 4개의 vector가 linearly independent한지 확인하면 됩니다. Null space를 구할 때 알았지만, $A$의 세 번째 column을 제외한 나머지 column들이 pivot column이므로, 3번 째 column을 제외하면, column들이 linearly independent한 것을 알 수 있습니다. 따라서,


$$
B=\{\begin{bmatrix}2 \\ -2 \\ 3 \end{bmatrix}\, \begin{bmatrix}4 \\ -5 \\ 7 \end{bmatrix}, \begin{bmatrix}1 \\ 3 \\ 6 \end{bmatrix}\}
$$




가 되고, 따라서 


$$
\dim ColA = 3
$$


가 됩니다.



여기서 중요한 점은


$$
dim NulA + dim ColA = 4
$$


가 되는데, **이는 $A$  matrix의 column의 개수가 됩니다.**



이에 대한 자세한 내용은 다음 포스트에서 다루도록 하겠습니다.



지금까지 dimension에 대해서 알아보았습니다. 다음 포스트에서는 rank에 대해서 알아보도록 하겠습니다. 질문이나 오류 있으시면 댓글 남겨주세요! 감사합니다!



<br/>



### Appendix : Proof of Theorem



<br/>

**Theorem**



If $V$ is a non-zero subspace of $\mathbb R^n$, then there exists a basis for $V$ that has at most $n$ vectors, i.e. $\dim V \leq n$

<br/>

* **Proof**



$V \subset \mathbb R^n$



let $\boldsymbol{v_1} \in V$, $\boldsymbol{v_1} \neq 0$



만약 $Span\{\boldsymbol{v_1}\} = V $ 이면, $\{\boldsymbol{v_1}\}$ 은 $V$의 basis가 됩니다.

만약 $Span\{\boldsymbol{v_1}\} \neq V$이면, span이 안된다는 뜻이므로, $\boldsymbol{v_1}$과 linearly independent한 어떤 벡터 $\boldsymbol{v_2}$가 $V$에 존재합니다. 따라서, 두 벡터를 포함한 집합을 이용하여 span을 해볼 수 있습니다. 



$Span\{\boldsymbol{v_1, v_2}\} = V$이면, $\{\boldsymbol{v_1, v_2}\}$는 $V$의 basis가 됩니다. 

$Span\{\boldsymbol{v_1, v_2}\} \neq V$이면, $\boldsymbol{v_1, v_2}$과 linearly independent한 어떤 벡터 $\boldsymbol{v_3}$가 $V$에 존재합니다. 따라서, 세 벡터를 포함한 집합을 이용하여 span 해볼 수 있습니다. 



위와 같은 방법을 계속 진행하다



$Span\{\boldsymbol{v_1, v_2, ..., v_n}\} \neq V$ 경우가 발생한다고 가정해봅시다. 그러면, $\{\boldsymbol{v_1, v_2, ..., v_n}\}$와 linearly independent한 $\boldsymbol{v_{n+1}}$이 $V$에 존재한다는 것을 뜻합니다. 하지만, $\mathbb R^n$의 dimension이 $n$이기 때문에, $\{\boldsymbol{v_1}, ..., \boldsymbol{v_{n+1}}\}$은 linearly dependent합니다. 따라서, $Span\{\boldsymbol{v_1, v_2, ..., v_n}\} \neq V$인 경우는 발생하지 않습니다.



따라서 


$$
\dim V \leq n
$$


입니다.





 

<br/>



**Theorem**



If $V$ and $W$ are subspaces of $\mathbb R^n$, and if $V$ is a subspace of $W$, then


$$
0\leq \dim V \leq \dim W \leq n
$$


$V=W$ if and only if $\dim V =\dim W$

<br/>

* **Proof**


$$
0\leq \dim V \leq \dim W \leq n
$$


이 부분은 앞선 정리의 증명과 동일합니다. $V \subset W \subset \mathbb R^n$이기 때문에, $W \subset \mathbb R^n$일 때의 dimension 관계 밝히는 방법을 똑같이 적용하면 알 수 있습니다. 



두 번째로 밝혀야 하는 부분은 


$$
V=W \iff \dim V=\dim W
$$


입니다. 



$V=W$ 이면, trivial하게 $\dim V = \dim W$입니다.

반대로, $\dim V = \dim W$이고, $V\subset W$인 상황을 생각해봅시다.


$$
\dim V = \dim W=k
$$


이라고 하고,


$$
S=\{\boldsymbol{v_1}, \boldsymbol{v_2}, ..., \boldsymbol{v_k}\}
$$


를 $V$의 basis라고 하면, S는 반드시 $W$의 basis가 되어야 합니다. 왜냐하면 $V \subset W$이고, $W$의 dimension이 $k$이며, $S$는 linearly independent하기 때문이죠. 따라서, 


$$
V=W
$$


를 만족합니다.



<br/>



**Theorem**



Let $S$ be a nonempty set of vectos in a vector space $V$, and let $S'$ be a set that results by adding additional vectors in $V$ to $S$



1. If the additional vectors are in $SpanS$, then $SpanS' = SpanS$

2. If $SpanS'=SpanS$, then the additional vectors are in $SpanS$

3. If $SpanS$ and $SpanS'$ have the same dimension, then the additional vectors are in $SpanS$ and $SpanS'=SpanS$



<br/>

* **Proof**



Proof of 1.



추가된 벡터가 $SpanS$에 있다는 것은 추가된 벡터는 $S$에 속한 벡터의 linear combination으로 표현이 가능하다는 것을 뜻합니다. 따라서


$$
Span S = Span S'
$$


을 만족합니다. $SpanS$는 $S$에 속한 벡터들의 linear combination 모두 모아놓은 집합을 뜻하기 때문입니다. 





Proof of 2



마찬가지로, $SpanS'=SpanS$이면, 추가된 벡터가 $S$에 속한 벡터들의 linear combination으로 표현된다는 것을 뜻합니다. (만약 표현되지 않으면 두 span이 같을 수 없습니다.) 따라서 추가된 벡터는 $SpanS$에 속하는 것을 알 수 있습니다.



Proof of 3



$SpanS$와 $SpanS'$를 보면, $S$에 벡터 하나를 추가하여 $S'$를 만들었기 때문에


$$
S \subseteq S'
$$


입니다. 따라서, 


$$
\dim SpanS \leq \dim SpanS'
$$


인데, dimension이 같은 경우 두 vector space가 동일합니다. 따라서, 


$$
SpanS =SpanS'
$$


임과 동시에, 추가된 벡터가 $SpanS$에 포함되는 것을 알 수 있습니다.



<br/>



**Theorem**



Le $V$ be a p-dimensional vector space, $p\geq 1$

Any linear independent set of exactly p elements in $V$ automatically a basis for $V$

Any set of exactly p elements that spans $V$ is automatically a basis for $V$

<br/>

* **Proof**



$\dim V =p$입니다. 여기서, linearly independent한 벡터의 수가 $p$개인 집합 $S$를 생각해봅시다.

만약 $S$가 basis가 되지 않는다면, Span 조건을 만족하지 않기 때문에, $S$에서 $V$에 있는 적절한 벡터를 추가한 $S'$가 basis가 되도록 만들 수 있습니다.

하지만, 이 때 $S'$에 속한 벡터의 개수가 $p$보다 커지기 때문에, 모순이 발생합니다.

따라서 $S$는 basis가 됩니다.



두 번째로 $SpanS=V$를 만족하는 벡터의 개수가 $p$개인 집합 $S$를 생각해봅시다.

만약 $S$가 basis가 되지 않는다면, linear independence 조건을 만족하지 않기 때문에, $S$에 속해 있는 벡터 중 적절한 벡터를 제거한 집합 $S'$가  $V$의 basis가 되도록 만들 수 있습니다. 

하지만, 이 때 $S'$에 속한 벡터의 개수가 $p$보다 작아지기 때문에, 모순이 발생합니다.

따라서 $S$는 basis가 됩니다.