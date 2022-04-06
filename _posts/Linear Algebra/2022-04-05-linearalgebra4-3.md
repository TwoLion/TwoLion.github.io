---
layout: single
title:  "4.3 Matrix and subspace"
categories: [Linear Algebra]
tag: [Linear Algebra, Column space, Null space, Row space]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---







이번 포스트에서는 Matrix를 이용하여 정의하는 null space, column space, row space에 대해서 알아보겠습니다. 



<br/>



### 1) Null Space

<br/>



**Definition : Null Space**



The null space of an $m \times n $ matrix $A$, written as $NulA$, is the set of all solutions of the homogeneous equation $A\boldsymbol{x}=0$


$$
NulA = \{\boldsymbol{x} \mid A\boldsymbol{x} = 0, \boldsymbol{x} \in \mathbb R^n\}
$$


matrix $A$의 null space는 $A\boldsymbol{x}=0$을 만족하는 $\boldsymbol{x}$ 를 모은 집합니다. 

matrix $A$를 standard matrix로 가지는 matrix transformation $T_A$를 생각해보면


$$
NulA = \{\boldsymbol{x} \mid A\boldsymbol{x} = 0, \boldsymbol{x} \in \mathbb R^n\} =  \{\boldsymbol{x} \mid T_A(\boldsymbol{x}) = 0, \boldsymbol{x} \in \mathbb R^n\} = Ker(T_A)
$$


$A$의 null space는 다름 아닌 $T_A$의 kernel임을 알 수 있습니다. 



<br/>

**Theorem**



The null space of an $m \times n$ matrix $A$ is a subspace of $\mathbb R^n$



Linear transformation의 kernel은 subspace가 되는 것을 통해 쉽게 알 수 있습니다.









<br/>

*example*


$$
A = \begin{bmatrix}1 & -1 \\ 2 & 5 \\ 3 & 4 \end{bmatrix}
$$


의 null space는


$$
NulA = \{\boldsymbol{x} \mid A\boldsymbol{x}=0\}
$$


가 되어 $A\boldsymbol{x}=0$의 augmented matrix를 이용하여 equation을 풀면


$$
\begin{bmatrix}1 & -1 & 0\\ 2 & 5 & 0 \\ 3 & 4 & 0 \end{bmatrix} \sim \begin{bmatrix}1 & 0 & 0\\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{bmatrix}
$$


가 되어


$$
NulA =\{\begin{bmatrix} 0\\  0 \end{bmatrix}\}
$$


이 됩니다. 



<br/>

### 2) Column Space

<br/>



**Definition : Column Space**



The column space of an $m \times n $ matrix $A$, written as $ColA$, is the set of all linear combinations of the columns of $A$. If $A=\begin{bmatrix}\boldsymbol{a}_1 & \boldsymbol{a}_2 & ... & \boldsymbol{a}_n  \end{bmatrix}$,


$$
ColA = Span \{\boldsymbol{a}_1, \boldsymbol{a}_2, ..., \boldsymbol{a}_n\}
$$


Matrix $A$의 column space는 $A$의 columns의 linear combination을 모은 집합입니다. 



vector들에 의해 span된 subset은 subspace가 되므로, column space 역시 subspace가 됩니다.



<br/>

**Theorem**



The column space of an $m \times n$ matrix $A$ is a subspace of $\mathbb R^m$



Matrix $A$의 column space에 속한 vector들은 $A$의 column의 linear combination으로 이루어져 있습니다. 따라서


$$
ColA = \{\boldsymbol{b} \mid \boldsymbol{b} = A\boldsymbol{x} \ \ for \ \ some \ \ \boldsymbol x \in \mathbb R^n\}
$$


으로도 해석할 수 있습니다.



<br/>

*example*


$$
A = \begin{bmatrix}1 & -1 \\ 2 & 5 \\ 3 & 4 \end{bmatrix}
$$


다음 matrix의 column space는 


$$
ColA = Span\{\begin{bmatrix}1 \\ 2 \\ 3  \end{bmatrix}, \begin{bmatrix}-1 \\  5 \\  4 \end{bmatrix}\}
$$


이 됩니다.



<br/>

### 3) The constrast between NulA and ColA

<br/>

다음 matrix를 통해서 Null space와 Column space가 가지는 특징을 비교해보겠습니다.

<br/>

*example*


$$
A=\begin{bmatrix}2 & 4& -2 & 1 \\ -2 & -5 & 7 & 3 \\ 3 & 7 & -8 & 6 \end{bmatrix}, \ \ \boldsymbol{u}=\begin{bmatrix}3 \\ -2 \\ -1 \\0 \end{bmatrix}, \ \ \boldsymbol{v} =\begin{bmatrix}3 \\ -1 \\3 \end{bmatrix}
$$
<br/>

* $NulA$?

<br/>

Matrix $A$의 null space를 구하기 위해서는


$$
A\boldsymbol{x }=0
$$


을 풀어야 합니다.  이를 풀면


$$
\begin{bmatrix} 2 & 4& -2 & 1 & 0 \\-2 & -5 & 7 & 3 & 0 \\3 & 7 & -8 & 6 & 0 \end{bmatrix} \sim  
\begin{bmatrix} 1 & 0& 9 & 0 & 0 \\0 & 1 & -5 & 0 & 0 \\0 & 0 & 0 & 1 & 0 \end{bmatrix}
$$


가 되어


$$
\boldsymbol{x} = x_3\begin{bmatrix}-9\\5\\1\\0 \end{bmatrix}, \ \ x_3 : free
$$


따라서


$$
NulA = Span\{\begin{bmatrix}-9\\5\\1\\0 \end{bmatrix}\}
$$


이 됩니다.

<br/>

* $ColA$?

<br/>

$A$의 column space는 column의 linear combination을 모두 모은 집합입니다. 따라서


$$
ColA = span\{\begin{bmatrix}2 \\ -2 \\ 3 \end{bmatrix}, \begin{bmatrix}4 \\ -5 \\ 7 \end{bmatrix}, \begin{bmatrix}-2 \\ 7 \\ -8 \end{bmatrix}, \begin{bmatrix}1 \\ 3 \\ 6 \end{bmatrix}\}
$$
이 됩니다.

<br/>

* $\boldsymbol{u} \in NulA$?

<br/>



 $\boldsymbol{u}$가 $NulA$에 속한다는 것은 $A\boldsymbol{u}=0$을 만족한다는 뜻입니다. 따라서


$$
A\boldsymbol{u} = \begin{bmatrix}6-8+2 \\-6+10-7+3 \\ 9-14+8 \end{bmatrix} = \begin{bmatrix}0 \\ 0\\3 \end{bmatrix} \neq 0
$$


따라서 $\boldsymbol{u} \notin NulA$입니다.

<br/>

* $\boldsymbol{v} \in ColA$?

<br/>

$\boldsymbol{v}$가 $ColA$에 속한다는 것은 $\boldsymbol{v}$가 $A$의 column의 linear combination으로 표현된다는 것을 뜻합니다. 따라서


$$
A\boldsymbol{x}=\boldsymbol{v}
$$


방정식이 consistent한지 확인을 해야 합니다. 다음 linear system의 augmented matrix를 이용하면


$$
\begin{bmatrix} 2 & 4& -2 & 1 & 3 \\-2 & -5 & 7 & 3 & -1 \\3 & 7 & -8 & 6 & 3 \end{bmatrix} \sim 
\begin{bmatrix} 2 & 4& -2 & 1 & 3 \\0 & -1 & 5 & 4 & 2 \\0 & 0 & 0 &\frac{31}{2} & \frac{1}{2} \end{bmatrix}
$$


echelon form의 모든 row에 leading entry가 있기 때문에, 다음 system의 solution이 존재합니다. 따라서 위 linear system은consistent하고 $\boldsymbol{v} \in ColA$입니다.



<br/>

Matrix $A$를 통해 null space와 column space를 구해보고, 특정 벡터가 column space, null space에 속하는지 확인도 해보았습니다. 

$m \times n$ matrix $A$에 대해 Null space는 다음의 특징을 가집니다.



1. $NulA$ is subspace of $\mathbb R^n$

2. $NulA$ is implicitly defined

   $NulA$를 구하려면 $A\boldsymbol{x}=0$을 풀어야 합니다. 따라서 linear system을 풀어야 하기 때문에 시간이 걸립니다. 

3. No obvious relation between $NulA$ and the entries in $A$

    $A$의 entry를 이용하여 $NulA$를 바로 확인할 수 없습니다. 

4. A typical vector $\boldsymbol{v}$ in $NulA$ has the property that $A\boldsymbol{v}=0$

5. It is easy to tell if $\boldsymbol{v}$ is in $NulA$

    $A\boldsymbol{v}$를 계산해서, $0$이 나오면 $NulA$에 속하고, $0$이 나오지 않으면 $NulA$에 속하지 않습니다.

6. $NulA=\{0\}$ if and only if $A\boldsymbol{x}=0$ has only the trivial solution

7. $NulA=\{0\}$ if and only if the linear transformation $\boldsymbol{x} \rightarrow A\boldsymbol x$ is one to one

   6과 7은 null space의 정의와 linear transformation이 one to one일 때 가지는 성질을 이용하여 쉽게 확인할 수 있습니다. 



<br/>



한편, $m \times n$ matrix $A$의 Column space는 다음의 특징을 가집니다.



1. $ColA$ is subspace of $\mathbb R^m$

2. $ColA$ is explicitly defined

   $ColA$는 $A$의 column의 linear combination을 모두 모은 집합입니다. 따라서 $A$를 통해 $ColA$를 바로 구할 수 있습니다.

3. Obvious relation between $ColA$ and the entries of $A$

   따라서, $A$의 entry와 $ColA$는 명확한 관계를 가집니다. 

4. A typical vector $\boldsymbol{u}$ in $ColA$ has the property that the equation $A\boldsymbol{x} = \boldsymbol{u}$ is consistent

5. It takes times to tell if $\boldsymbol u$ is in $ColA$

   $A\boldsymbol{x} = \boldsymbol{u}$ 가 consistent한지 확인을 해야 하기 때문에, linear system을 풀어야 합니다. 

6. $ColA =\mathbb R^m$ if and only if the equation $A\boldsymbol{x} =\boldsymbol{b}$ has a solution for every $\boldsymbol{b} \in \mathbb R^m$ 

7. $ColA =\mathbb R^m$ if and only if the linear transformation $\boldsymbol x \rightarrow A\boldsymbol{x}$ is onto

    6과 7은 column space의 정의와 linear transformation이 onto일 때 가지는 성질을 이용하여 쉽게 확인할 수 있습니다.



<br/>

### 4) Row Space

<br/>



**Definition : Row Space **



The row space of an $m \times n$ matrix $A$, written as $RowA$, is the set of all linear combinations of the rows of $A$



Matrix $A$의 row들의 linear combination을 모두 모은 집합이 $RowA$입니다. 

각각의 row는 $n$개의 entry를 가지기 때문에, $RowA \subset \mathbb R^n$입니다.

또한, matrix $A$의 row는 matrix $A^T$의 column과 같기 때문에


$$
RowA=ColA^T
$$


이 성립합니다.



Row space는 다음의 성질을 가집니다.



<br/>

**Theorem**



If two matrices $A$ and $B$ are row equivalent, then their row spaces are the same.



Row eqivalent하다면, row operation을 통하여 다른 matrix를 만들 수 있다는 뜻입니다. row operation은 linear combination를 모두 모은 집합에는 영향을 끼치지 않기 때문에, 두 matrix의 row space는 동일합니다.





<br/>

*example*


$$
A = \begin{bmatrix}-2 & -5 & 8 & 0 & -17 \\ 1 & 3 & -5 & 1 & 5 \\ 3 & 11 & -19 & 7 & 1 \\ 1 & 7 & -13 & 5 & -3 \end{bmatrix}
$$


다음 matrix의 row space는


$$
RowA = Span\{\boldsymbol{a_1}, \boldsymbol{a_2}, \boldsymbol{a_3}, \boldsymbol{a_4}\} \\
where \ \ \ \boldsymbol{a_1}=\begin{bmatrix}-2 \\ -5 \\8 \\0 \\ -17\end{bmatrix}, \ \ 
\boldsymbol{a_1}=\begin{bmatrix}1 \\ 3 \\ -5 \\ 1 \\ 5\end{bmatrix}, \ \
\boldsymbol{a_1}=\begin{bmatrix}3 \\ 11 \\ -19 \\ 7 \\ 1\end{bmatrix}, \ \ 
\boldsymbol{a_1}=\begin{bmatrix}1 \\ 7 \\ -13 \\ 5 \\ -3\end{bmatrix}
$$
이 됩니다.



지금까지 matrix를 통해 정의하는 vector space인 null space, column space, row space에 대해 알아보았습니다. 다음 포스트에서는 basis에 대해 알아보도록 하겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



<br/>



### Appendix : Proof of Theorem



<br/>

**Theorem**



If two matrices $A$ and $B$ are row equivalent, then their row spaces are the same.



* **Proof**



$m \times n $ matrix $A$를 다음과 같이 표현을 하면


$$
A = \begin{bmatrix}-\boldsymbol a_{1}-  \\ \vdots \\ -\boldsymbol a_{m}-\end{bmatrix}
$$


$\boldsymbol{a}_1, \boldsymbol{a}_2, ..., \boldsymbol{a}_m$은 $\mathbb R^n$에 속하는 벡터입니다. 

$A$와 $B$가 row equivalent하다는 것은 row operation을 통해서 $A$에서 $B$, $B$에서 $A$를 만들 수 있다는 뜻입니다. row operation인 replacement, interchange, scaling은 row space의 변화에 영향을 끼치지 않는 것을 통해 두 matrix의 row space가 같은 것을 확인할 수 있습니다.



* Replacement


$$
\begin{aligned}

RowA &= Span\{\boldsymbol{a}_1, ..., \boldsymbol{a}_m\} \\
&= Span\{\boldsymbol{a}_1+k\boldsymbol{a}_i, ..., \boldsymbol{a}_m\}\ \ \ i=1, ..., m


\end{aligned}
$$




* Interchange




$$
\begin{aligned}

RowA &= Span\{\boldsymbol{a}_1, ...,\boldsymbol{a}_i, ..., \boldsymbol{a}_j,  \boldsymbol{a}_m\} \\
&= Span\{\boldsymbol{a}_1, ..., \boldsymbol{a}_j, ..., \boldsymbol{a}_i,  \boldsymbol{a}_m\}\ \ \ i \neq j, \ \  i, j=1, ..., m

\end{aligned}
$$


* Scaling


$$
\begin{aligned}

RowA &= Span\{\boldsymbol{a}_1, ...,\boldsymbol{a}_i, \boldsymbol{a}_m\} \\
&= Span\{\boldsymbol{a}_1, ..., k\boldsymbol{a}_i,  \boldsymbol{a}_m\}\ \ \ i=1, ..., m

\end{aligned}
$$


Row operation을 하더라도 row space가 변화하지 않으므로, Row equivalent한 두 matrix의 row space는 동일합니다.