---
layout: single
title:  "4.2 Linear transformation (1)"
categories: [Linear Algebra]
tag: [Linear Algebra, Linear transformation]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---









이번 포스트에서는 Linear transformation에 대해서 알아보겠습니다.

<br/>

### 1) Linear Transformation

<br/>

#### (1) Transformation

<br/>

Transformation은 다음과 같이 정의됩니다.



<br/>

**Definition : Transformation**



Transformation is a function whose inputs and outputs are vectors


$$
T : \mathbb R^n \rightarrow \mathbb R^m
$$


A transformation 


$$
T : \mathbb R^n \rightarrow \mathbb R^n
$$


is called operator on $\mathbb R^n$



Transformation이란, input과 output이 vector인 함수입니다. 고등학교에서 배운 함수는 input과 output이 실수인, 또는 input이 벡터지만 output이 실수인 경우만 다루었다면, transformation은 이에서 확장하여 input과 output이 모두 vector인 경우를 뜻합니다. 또한, input과 output이 $\mathbb R^n$으로 같다면, 해당 transformation은 operator가 됩니다. Transformation 또한 함수이기 때문에, 기존의 함수에서 사용했던 개념인 정의역(domain), 공역(codomain), 치역(image) 또한 똑같이 정의됩니다.



<br/>

*Example*



$T$ : transformation that maps a vector $\boldsymbol{x}=(x_1, x_2)$ in $\mathbb R^2$ into the vector $2\boldsymbol{x} = (2x_1, 2x_2)$ in $\mathbb R^2$



input과 output 모두 $\mathbb R^2$에 속하는 함수입니다. 따라서 $T$는 transformation입니다. 또한, input과 output이 모두  $\mathbb R^2$에 속하므로, operator입니다. 



<br/>

#### (2) Linear transformation

<br/>

**Definition : Linear transformation**



A function $T : \mathbb R^n \rightarrow \mathbb R^m$ is called a linear transformation from $\mathbb R^n$ to $\mathbb R^m$ if following two properties hold for all vectors $\boldsymbol{u, v}$ in $\mathbb R^n$ and for all scalars $c$.



1. $T(c\boldsymbol u) = cT(\boldsymbol{u})$
2. $T(\boldsymbol{u+v}) = T(\boldsymbol{u}) + T(\boldsymbol{v})$



If $n=m$, then the linear transformation $T$ is called linear operator.



Linear transformation이 정의되기 위해서는 transformation이 scalar multiplication, vector addition에 대한 조건을 만족하여야 합니다. 또한 input과 output이 $\mathbb R^n$인 linear transformation을 linear operator라고 정의합니다. 어떤 transformation이 linear transformation임을 확인하기 위해서는 정의역에 존재하는 임의의 벡터에 해대서 위 두 조건을 만족하는지 확인하면 됩니다.



<br/>

*example*



$T$ : transformation that maps a vector $\boldsymbol{x}=(x_1, x_2)$ in $\mathbb R^2$ into the vector $2\boldsymbol{x} = (2x_1, 2x_2)$ in $\mathbb R^2$



위 transformation이 linear transformation을 만족하는지 확인해봅시다.



1. $\boldsymbol{u, v} \in \mathbb R^2, T(\boldsymbol{u+v}) = 2(\boldsymbol{u+v}) = 2\boldsymbol{u}+2\boldsymbol{v} = T(\boldsymbol{u}) + T(\boldsymbol{v})$, 
2. $\boldsymbol u \in \mathbb R^2, c \in \mathbb R, T(c\boldsymbol{u})=2c\boldsymbol{u}=c2\boldsymbol{u}=cT(\boldsymbol{u})$



$\mathbb R^2$에 속하는 임의의 vector와 scalar에 대해서 두 조건이 성립하기 때문에, 위 transformation은 linear transformation입니다. 또한 정의역과 공역이 $\mathbb R^2$이므로 linear operator입니다.



Linear transformation의 특징을 파악하기 위해서는 matrix transformation이 필요합니다. 



<br/>

**Definition : Matrix transformation**



If $A$ is $m \times n$ matrix, and if $\boldsymbol{x}$ is a column vector in $\mathbb R^n$, then the product $A\boldsymbol x$ is a vector in $\mathbb R^m$. Therefore, the transformation


$$
T_A = \mathbb R^n \rightarrow \mathbb R^m \\
T_A(\boldsymbol{x}) = A\boldsymbol{x}
$$
 

is called the multiplication by $A$ or the transformation $A$



즉 matrix product로 이루어지는 transformation을 matrix transformation이라고 합니다. 

<br/>

*example*


$$
A = \begin{bmatrix}1 & -1 \\ 2 & 5 \\ 3 & 4 \end{bmatrix}, \boldsymbol{b} = \begin{bmatrix}7 \\ 0 \\ 7 \end{bmatrix}
$$


이 때, $T_A(\boldsymbol x) = A\boldsymbol{x}$가 됩니다. $T(\boldsymbol{x}) =\boldsymbol{b}$를 만족시키는 $\boldsymbol{x}$는 $A\boldsymbol{x} = \boldsymbol{b}$를 만족합니다. 따라서 matrix equation을 풀면 
$$
\boldsymbol{x} = \begin{bmatrix} 6 \\ -1 \end{bmatrix}
$$
이 됩니다. 

<br/>

*example*

$T_0$ : Zero transformation 

output이 0인 transformation 또한 matrix transformation입니다. Zero matrix를 이용하여 transformation을 정의할 수 있습니다.
$$
T_0 : \mathbb R^n \rightarrow \mathbb R^m \\
T(\boldsymbol{x}) = 0\boldsymbol{x} = 0
$$


<br/>

*example*

$T_I$ : Identity transformation

Output이 input과 같도록 만드는 transformation 또한 matrix transformation입니다. Identity matrix를 이용하여 transformation을 정의할 수 있습니다.
$$
T_I = \mathbb R^n \rightarrow \mathbb R^n \\
T_I(\boldsymbol{x}) = I\boldsymbol{x} = \boldsymbol{x}
$$


이 때, 위 transfromation은 operator이기도 합니다. 



다음은 linear transformation과 matrix transformation 사이의 관계를 알아봅시다. 사실, 두 transformation은 같은 transformation입니다.



<br/>

**Theorem**



All linear transformation are matrix transformation



Let $T : \mathbb R^n \rightarrow \mathbb R^m$ be a linear transformation. If $\boldsymbol{e_1, e_2, ..., e_n}$ are standard unit vectors in $\mathbb R^n$, and $\boldsymbol{x}$ is any vector in $\mathbb R^n$, then $T(\boldsymbol{x})$ can be represented as


$$
T(\boldsymbol{x}) = Ax \\
where \ A =\begin{bmatrix}T(\boldsymbol{e_1}) & T(\boldsymbol{e_2})&...&T(\boldsymbol{e_n}) \end{bmatrix}
$$


$A$ : Standard matrix for $T$ and $A=\begin{bmatrix}T\end{bmatrix}$



즉 모든 linear transformation은 matrix transformation으로 나타낼 수 있습니다. 또한 matrix tranformation은 linear transformation이므로, 사실상 두 transformation은 같은 transfromation임을 알 수 있습니다. 



*example*


$$
T : \mathbb R^3 \rightarrow \mathbb R^2 \\
T : \begin{bmatrix}x_1 \\ x_2\\x_3 \end{bmatrix} \rightarrow \begin{bmatrix}x_1 + x_2  \\ x_2 - x_3\end{bmatrix}
$$


먼저 위 transformation이 linear transformation인지 확인해봅시다.



1. $\boldsymbol{u, v} \in \mathbb R^3, \boldsymbol{u} = \begin{bmatrix}u_1 \\ u_2 \\ u_3 \end{bmatrix}, \boldsymbol{v} = \begin{bmatrix}v_1 \\ v_2 \\ v_3 \end{bmatrix}$

   $T(\boldsymbol{u+v}) = \begin{bmatrix}(u_1+v_1) + (u_2 + v_2) \\ (u_2+v_2)-(u_3+v_3)\end{bmatrix} = \begin{bmatrix}u_1 + u_2 \\ u_2-u_3\end{bmatrix} + \begin{bmatrix}v_1 + v_2 \\ v_2-v_3\end{bmatrix} = T(\boldsymbol{u}) + T(\boldsymbol{v})$

2. $c \in \mathbb R$

   $T(c\boldsymbol u) = \begin{bmatrix}cu_1 +cu_2 \\ cu_2 - cu_3 \end{bmatrix} = c\begin{bmatrix}u_1 +u_2 \\ u_2 - u_3 \end{bmatrix} = cT(\boldsymbol{u})$



따라서 위 transformation은 linear transformation입니다. 위 linear transformation의 standard matrix를 찾기 위해 unit vector을 이용하면


$$
T(\begin{bmatrix}1 \\0 \\0  \end{bmatrix}) = \begin{bmatrix}1 \\ 0\end{bmatrix}, \ \ T(\begin{bmatrix}0 \\1 \\0  \end{bmatrix}) = \begin{bmatrix}1 \\ 1 \end{bmatrix}, \ \ T(\begin{bmatrix}0 \\0 \\1  \end{bmatrix}) = \begin{bmatrix}0 \\ -1 \end{bmatrix}
$$
 따라서


$$
[T] = \begin{bmatrix}1 & 1 & 0 \\ 0 & 1 & -1 \end{bmatrix}
$$
임을 알 수 있습니다.





지금까지 transformation과 linear transformation을 알아보았습니다. 다음 포스트에서는 kernel과 range에 대해서 알아보도록 하겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!

<br/>



### Appendix : Proof of Theorem

<br/>



Linear tranformation과 matrix transformation의 관계에 대한 정리에 대한 증명입니다.



<br/>

**Theorem**



Let $T : \mathbb R^n \rightarrow \mathbb R^m$ be a linear transformation. If $\boldsymbol{e_1, e_2, ..., e_n}$ are standard unit vectors in $\mathbb R^n$, and $\boldsymbol{x}$ is any vector in $\mathbb R^n$, then $T(\boldsymbol{x})$ can be represented as


$$
T(\boldsymbol{x}) = Ax \\
where \ A =\begin{bmatrix}T(\boldsymbol{e_1}) & T(\boldsymbol{e_2})&...&T(\boldsymbol{e_n}) \end{bmatrix}
$$


$A$ : Standard matrix for $T$ and $A=\begin{bmatrix}T\end{bmatrix}$





* **Proof**



$\boldsymbol x \in \mathbb R^n$이라고 하면


$$
\boldsymbol{x} = \begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = x_1\begin{bmatrix}1 \\ 0 \\ \vdots \\ 0 \end{bmatrix} + x_2 \begin{bmatrix} 0 \\ 1 \\   \vdots \\ 0 \end{bmatrix} + \cdots + x_n \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1\end{bmatrix} = x_1\boldsymbol{e_1} + x_2\boldsymbol{e_2} + \cdots + x_n\boldsymbol{e_n}
$$


, 즉 standard unit vector의 linear combination으로 표현됩니다.



$T$가 linear transformation이므로, $T(\boldsymbol{x})$은 다음과 같이 표현됩니다.


$$
T(\boldsymbol{x}) =T(x_1\boldsymbol{e_1} + x_2\boldsymbol{e_2} + \cdots + x_n\boldsymbol{e_n}) = x_1T(\boldsymbol{e_1}) + x_2T(\boldsymbol{e_2}) + \cdots + x_nT(\boldsymbol{e_n})
$$


이는 matrix product에 따라


$$
x_1T(\boldsymbol{e_1}) + x_2T(\boldsymbol{e_2}) + \cdots + x_nT(\boldsymbol{e_n}) = \begin{bmatrix}T(\boldsymbol{e_1}) & T(\boldsymbol{e_2}) & \cdots & T(\boldsymbol{e_n}) \end{bmatrix}\begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_n\end{bmatrix}
$$


이 되어,  다음 matrix
$$
[T] =\begin{bmatrix}T(\boldsymbol{e_1}) & T(\boldsymbol{e_2}) & \cdots & T(\boldsymbol{e_n}) \end{bmatrix}
$$


의 transformation이 됩니다. 즉


$$
T : \mathbb R^n \rightarrow \mathbb R^m \\
T(\boldsymbol{x}) = [T]\boldsymbol{x}
$$


따라서, linear transformation은 matrix transformation이고, standard unit vector를 통해 standard matrix를 구할 수 있습니다.