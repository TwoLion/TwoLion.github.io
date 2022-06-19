---
layout: single
title:  "7.2 Quadratic Form"
categories: [Linear Algebra]
tag: [Linear Algebra, Quadratic Form]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---



이번 포스트에서는 quadratic form에 대해 알아보겠습니다.





<br/>

### 1) Quadratic Form



<br/>

#### (1) Quadratic Form



Quadratic form이 무엇인지 알아보겠습니다.



<br/>

**Definition : Quadratic Form** 



A quadratic form on $\mathbb R^n$ is a function $Q$ defined on $\mathbb R^n$ whose valude at a vector $\boldsymbol x$ in $\mathbb R^n$ can be computed  by an expression of the form


$$
Q(\boldsymbol x) = \boldsymbol x ^T A \boldsymbol x
$$




where $A$ is $n\times n $ symmetric matrix 



The matrix $A$ is called the matrix of the quadratic form.



Quadratic form은 symmetric matrix $A$로 정의되는 정의역이 $\mathbb R^n$이고 공역이 $\mathbb R$인 함수입니다. 함수식은 위와 같이 정의되구요.



<br/>

*Example*


$$
Q(\boldsymbol x) = \boldsymbol x^T I \boldsymbol x = \|\boldsymbol x\|^2
$$


Norm, Length는 quadratic form입니다. 해당 quadratic form에 해당하는 matrix는 identity matrix이구요.



<br/>

*Example*


$$
\boldsymbol x= \begin{bmatrix}x_1 \\ x_2 \end{bmatrix}, \ \ A=\begin{bmatrix}4 & 0 \\ 0 & 3\end{bmatrix}, \ \ B=\begin{bmatrix}3 & -2 \\ -2 & 7\end{bmatrix}
$$


해당 matrix $A, B$를 이용하여 quadrtic form을 정의할 수 있습니다.


$$
Q_A(\boldsymbol x) = \boldsymbol x^TA\boldsymbol x = 4x_1^2+3x_2^2 \\
Q_B(\boldsymbol x) = \boldsymbol x^TB\boldsymbol x = 3x_1^2 - 4x_1x_2 + 7x_2^2
$$




두 예시에서 알 수 있듯이, 다음 함수를 quadratic form(이차 형식)으로 정의하는 이유 중 하나는, 결과값의 차수가 2이기 때문입니다. 

Quadratic form에서 사용되는 matrix가 symmetric matrix이므로, 이를 이용하여 quadratic form 계산을 빠르게 할 수 있습니다. 



<br/>

*Example*


$$
Q(\boldsymbol x) = 5x_1^2+3x_2^2 +2x_3^2 -2x_1x_2+8x_2x_3
$$


해당 식을 보면, $\boldsymbol x$ 는 $\mathbb R^3$에 속한 벡터입니다. 이를 이용하여 해당 quadratic form의 matrix를 구할 수 있습니다. 


$$
A = \begin{bmatrix}a & b & c \\ b & d & e \\ c & e & f\end{bmatrix}
$$


로 $A$를 정의하면


$$
Q_A(\boldsymbol x) = \boldsymbol x^T A \boldsymbol x = ax_1^2+dx_2^2+fx_3^2+ 2bx_1x_2 +2cx_1x_3+2ex_2x_3
$$


임을 알 수 있고, 이를 통해


$$
a=5, \ b=-1, \ c=0, \ d=3, \ e=4, \ f=2
$$


즉


$$
A = \begin{bmatrix} 5 & -1 & 0 \\ -1 & 3 & 4 \\ 0 & 4 & 2 \end{bmatrix}
$$


임을 알 수 있습니다. 





<br/>

#### (2) Types of Quadratic form



<br/>



Quadratic Form에 대응하는 matrix에 따라 결과값이 어떻게 변화하는지 확인할 수 있습니다. 예를 들어


$$
A=\begin{bmatrix}4 & 0 \\ 0 & 3\end{bmatrix}
$$


에 해당하는 quadratic form $Q_A(\boldsymbol x)$는


$$
Q_A(\boldsymbol x) = 4x_1^2 + 3x_2^2
$$


은 모든 $\boldsymbol x \in \mathbb R^2$에 대해서 **0보다 크거나 같은 값을 가집니다**. 한편


$$
B=\begin{bmatrix}-4 & 0 \\ 0 & -3\end{bmatrix}
$$


에 해당하는 quadratic form $Q_B(\boldsymbol x)$는 


$$
Q_B(\boldsymbol x) =-4x_1^2 -3x_2^2
$$


은 모든 $\boldsymbol x\in \mathbb R^2$에 대해서 **0보다 작거나 큰 값을 가집니다. **


$$
C=\begin{bmatrix}-4 & 0\\ 0 & 3\end{bmatrix}
$$


에 해당하는 quadratic form $Q_C(\boldsymbol x)$는


$$
Q_C(\boldsymbol x) = -4x_1^2+3x_2^2
$$


은 $\boldsymbol x$에 따라 **양의 값, 음의 값을 가질 수 있습니다.**



Quadratic form에서 위와 같이 모든 $\boldsymbol x \in \mathbb R^n$에 대해 양의 값을 가지는지, 음의 값을 가지는지는 중요한 특징 중 하나입니다. 따라서 이를 통해 quadratic form을 분류합니다. 



<br/>

**Definition : Classifying Quadratic Forms**



When $A$ is an $n \times n $ matrix, the quadratic form $Q(\boldsymbol x) = \boldsymbol x ^T A \boldsymbol x$ is a real-valued function with domain $\mathbb R^n$. A quadratic form $Q$ is



1. **Positive Definite** if $Q(\boldsymbol x)>0$ for all $\boldsymbol x \neq 0$
2. **Positive Semidefinite** if $Q(\boldsymbol x)\geq0$
3. **Negative Definite** if $Q(\boldsymbol x) <0$ for all $\boldsymbol x\neq 0$
4. **Negativve Semidefinite** if $Q(\boldsymbol x)\leq 0$
5. **Indefinite** if $Q(\boldsymbol x)$ assumes both positive and negative values





위의 예시에서 $Q_A$는 positive definite(or positive semidefinite), $Q_B$는 negative definite(or negative semidefinite), $Q_C$는 indefinite인 것을 알 수 있습니다. 



위의 quadratic form에 해당하는 matrix $A, B, C$가 가지는 특징은 모두 **diagonal matrix**라는 점입니다. **diagonal matrix에 대한 quadratic form을 계산하면 결과값에 제곱항에 대한 식만 남게 됩니다. 실수는 제곱하는 순간 0보다 크거나 같기 때문에, 제곱항의 계수에 따라 해당 quadratic form이 positive definite인지, negative definite인지, indefinite인지 확인할 수 있습니다.** 하지만, 다음과 같은 matrix $E$


$$
E=\begin{bmatrix}4 & 2 \\ 2 & 3\end{bmatrix}
$$


에 대응하는 quadratic form $Q_E(\boldsymbol x)$


$$
Q_D(\boldsymbol x) = 4x_1^2+4x_1x_2+3x_2^2
$$


은 모든 $\boldsymbol x \in \mathbb R^2$에 대해서 양의 값을 가지는지, 음의 값을 가지는지, 알 수 없는지 바로 확인을 하기가 어렵니다. 이는 $x_1x_2$항이 존재하기 때문입니다. $x_1, x_2$의 부호에 따라 양의 값을 가지기도, 음의 값을 가지기도 하기 때문입니다. 따라서 해당 matrix가 어떤 quadratic form을 가지는지 확인하기 위해, 이전 포스트에서 배웠떤 spectral decomposition을 이용합니다. 



<br/>

#### 3) Change of Variable in a Quadratic Form



<br/>

위의 예시에서 $Q_E$가 어떤 type인지 확인하는 쉬운 방법은 $x_1x_2$항을 없애주는 것입니다. 즉, $E$를 적절한 방법을 이용하여 diagonal matrix로 만들어주면 됩니다. Diagonal matrix로 만들어주기 위해 이전 포스트에서 배운 **Spectral Decomposition과 Change of Variable** 개념을 사용합니다. 







$\boldsymbol x \in \mathbb R^n$일 때, standard basis에 해당하는 coordinate vector가 $\boldsymbol x$가 됩니다. 즉


$$
\boldsymbol x = \left[\boldsymbol x\right]_{\epsilon}
$$


이를 standard basis가 아닌 $n\times n$ invertible matrix $P$의 column으로 이루어진 basis에 해당하는 coordinate vector는 다음과 같이 표현이 가능합니다. 


$$
\boldsymbol x = P\boldsymbol y
$$


$P$는 invertible하므로,


$$
P^{-1}\boldsymbol x = \boldsymbol y = [\boldsymbol x]_B \\
B = \{\boldsymbol p_1, ..., \boldsymbol p_n\}
$$


$\boldsymbol y$**가 $P$의 column으로 이루어진 basis를 이용하여 표현한 $\boldsymbol x$의 coordinate vector가 됩니다.**



한편, Quadratic Form $Q_E$에 대응하는 matrix $E$는 symmetric matrix이므로, **spectral decomposition**를 적용하면


$$
E = PDP^T, \ \  where \\

P = \begin{bmatrix}\boldsymbol p_1 & ... & \boldsymbol p_n \end{bmatrix}, \ \ D = diag(\lambda_1, ..., \lambda_n)
$$


으로 표현가능합니다. $\lambda_1, ..., \lambda_n$은 $E$의 eigenvalue이고, $\boldsymbol p_1, ..., \boldsymbol p_n$은 length가 1인 $\lambda_1, ..., \lambda_n$에 대응되는 eigenvector이구요. spectral decomposition을 이용하여 quadratic form을 표현하면


$$
Q_E(\boldsymbol x) = \boldsymbol x^T E \boldsymbol x = \boldsymbol x^TPDP^T\boldsymbol x = (P^T\boldsymbol x)^TD(P^T\boldsymbol x)
$$


가 됩니다. 이 때, $P^{-1}=P^T$이므로 (P는 orthogonal matrix입니다.) 위에서 정의한 


$$
\boldsymbol y = P^{-1}\boldsymbol x = P^T\boldsymbol x
$$


가 되어 이를 이용하여 표현하면


$$
Q_E(\boldsymbol x) = \boldsymbol y^TD\boldsymbol y \\
\boldsymbol y =P^T\boldsymbol x
$$


임을 알 수 있습니다. 이 때, $P$는 invertible하므로, $\boldsymbol x$와 $\boldsymbol y$는 일대일 대응이고, $\boldsymbol y$가 가질 수 있는 값은 $\mathbb R^n$ 전체입니다. 따라서 모든 $\boldsymbol x \in \mathbb R^n$에 대해 해당 quadratic form의 결과를 비교하는 것은, 모든 $\boldsymbol y\in \mathbb R^n$에 대해 


$$
\boldsymbol y^TD\boldsymbol y
$$


의 결과를 비교하는 것과 같습니다. 이제, 중앙의 matrix가 diagonal matrix이므로, 위의 quadratic form이 모든 $\boldsymbol x$(or $\boldsymbol y$)에 대해 양의 값을 가지는지, 음의 값을 가지는지, 알 수 없는지 확인할 수 있고, 따라서 positive definite인지, negative definite인지, indefinite인지 구별할 수 있습니다.



해당 개념을 보여주는 정리가 principal axes theorem입니다.



<br/> **Theorem : Principal Axes Theorem**



Let $A$ be an $n \times n$ symmetric matrix. Then there is an orthogonal change of variable, $\boldsymbol x = P\boldsymbol y$, that transforms the quadratic form $\boldsymbol x^TA\boldsymbol x$ into the quadratic form $\boldsymbol y^TD\boldsymbol y$ with no cross-product term.



The columns of $P$ are called the principal axes of the quadratic form $\boldsymbol x^T A\boldsymbol x$



The vector $\boldsymbol y$ is the coordinate vector of $\boldsymbol x$ relative to the orthonormal basis of $\mathbb R^n$ given by these principal axes.



$A$가 $n\times n$ symmetric matrix일 때, spectral decomposition을 이용하여 $A=PDP^T$로 orthogonally diagonalizable하고, 이를 통해 quadratic form을 


$$
Q_A(\boldsymbol x ) = \boldsymbol x^TA\boldsymbol x = (P^T\boldsymbol x)^TDP^T\boldsymbol x = \boldsymbol y^TD\boldsymbol y
$$


로 표현할 수 있습니다. 이 때, $D$는 diagonal matrix이므로, cross-product term(ex: $x_1x_2, x_2x_3, ...$)이 없고, 제곱항으로만 표현됩니다. 이 때, **$P$의 column을 principal axes라고 하고, $\boldsymbol y$는 Principal axes로 이루어진 basis를 이용한 $\boldsymbol x$의 coordinate vector**


$$
\boldsymbol y = [\boldsymbol x]_B \\
B = \{\boldsymbol p_1, ..., \boldsymbol p_n\}
$$


입니다.



<br/>

*Example*


$$
Q(\boldsymbol x) = x_1^2 -8x_1x_2 -5x_2^2
$$


해당 quadratic form에 해당하는 matrix는


$$
A = \begin{bmatrix}1 & -4 \\ -4 & -5 \end{bmatrix}
$$


인 것을 알 수 있습니다. $A$는 symmetric이므로, spectral decomposition을 위해 $A$의 eigenvalue와 eigenvector를 구해보면



* Eigenvalue


$$
\det(A-\lambda I) = (1-\lambda)(-5-\lambda)-16 = \lambda^2 +4\lambda -21 = 0 \\
\lambda = -7, 3
$$




* Eigenvector - $\lambda=3$

$$
A\boldsymbol x = 3\boldsymbol x \\

\begin{bmatrix}-2 & -4 & 0 \\ -4 & -8&0 \end{bmatrix} \sim \begin{bmatrix}1 & 2 & 0 \\ 0 & 0&0 \end{bmatrix}
$$


$$
\boldsymbol x = x_2\begin{bmatrix}-2 \\ 1\end{bmatrix}, \ \ x_2 : free
$$


크기가 1인 eigenvector는


$$
\boldsymbol p_1 =\begin{bmatrix}-\frac{2}{\sqrt 5} \\ \frac{1}{\sqrt 5} \end{bmatrix}
$$






* Eigenvector - $\lambda =-7$


$$
A\boldsymbol x = -7\boldsymbol x \\

\begin{bmatrix}8 & -4 & 0 \\ -4 & -2&0 \end{bmatrix} \sim \begin{bmatrix}1 & -\frac{1}{2} & 0 \\ 0 & 0&0 \end{bmatrix}
$$

$$
\boldsymbol x = x_2 \begin{bmatrix}\frac{1}{2} \\ 1 \end{bmatrix}, \ \ x_2 : free
$$


이므로, 크기가 1인 eigenvector는


$$
\boldsymbol p_2 =\begin{bmatrix}\frac{1}{\sqrt5} \\ \frac{2}{\sqrt5} \end{bmatrix}
$$


다음과 같이 eigenvalue와 eigenvector를 구할 수 있고, 이를 이용하여 diagonalization을 진행할 수 있습니다.


$$
A = PDP^T, \ \ where \\
P = \begin{bmatrix}\boldsymbol p_1 & \boldsymbol p_2  \end{bmatrix}, \ \ D = \begin{bmatrix}3 & 0 \\ 0 & -7 \end{bmatrix}
$$


따라서, $\boldsymbol y = P^T\boldsymbol x$로 정의하면, $Q_A$는  다음과 같이 바뀝니다.


$$
Q_A(\boldsymbol x ) = \boldsymbol y^TD\boldsymbol y = 3y_1^2 -7y_2^2
$$




$\boldsymbol y$에 따라서 해당 quadratic form이 양의 값도, 음의 값도 가질 수 있기 때문에, 해당 quadratic form은 indefinite입니다.



지금까지 spectral decomposition을 통하여 해당 quadratic form에서 cross-product term을 없애는 방법에 대해 알아보았습니다. cross-product term을 없앤 결과값에서의 각각의 제곱항에 해당하는 계수는 해당 matrix의 eigenvalue였습니다. 따라서, eigenvalue가 어떤지에 따라서 해당 quadratic form의 type을 분류할 수 있습니다.



<br/>

**Theorem**



Let $A$ be an $n \times n$ symmetric matrix. Then a quadratic form $\boldsymbol x^T A \boldsymbol x$ is 



1. **Positive definite** if and only if the eigenvalues of $A$ are **all positive**
2. **Negative definte** if and only if the eigenvalues of $A$ are **all negative**
3. **Indefinite** if and only if $A$ has both positive and negative eigenvalues



이는 


$$
\boldsymbol x^TA\boldsymbol x = \boldsymbol y^TD\boldsymbol y = \lambda_1y_1^2 + \lambda_2y_2^2 +\cdots + \lambda_ny_n^2
$$


인데, $y_1, ..., y_n$은 $\mathbb R$에 속하므로 제곱하면 항상 0보다 크거나 같기 때문에, 제곱항의 계수인 $\lambda_1, ..., \lambda_n$의 값에 따라 해당 quadratic form의 type을 결정할 수 있기 때문입니다.







<br/>

*Example*


$$
A = \begin{bmatrix}3 & 2 & 0 \\ 2 & 2 & 2 \\ 0 & 2 & 1\end{bmatrix}
$$


해당 matrix를 이용한 quadratic form $Q_A(\boldsymbol x)$가 positive definite인지, negative definite인지, indefinite인지 확인해봅시다. $A$의 eigenvalue를 구하면


$$
\begin{aligned}
\det(A-\lambda I) &= \begin{vmatrix}3-\lambda & 2 & 0 \\ 2 & 2-\lambda & 2 \\ 0 & 2 & 1-\lambda\end{vmatrix} = (3-\lambda)\begin{vmatrix}2-\lambda & 2 \\ 2 & 1-\lambda\end{vmatrix}-2\begin{vmatrix}2 & 2 \\ 0 & 1-\lambda\end{vmatrix} \\
&=(3-\lambda)\{(2-\lambda)(1-\lambda)-4\}-2\{2(1-\lambda)\} \\
&=(\lambda+1)(\lambda+2)(\lambda+5) = 0
\end{aligned}
$$


이므로


$$
\lambda = -1, -2, -5
$$


입니다. 모든 eigenvalue가 음의 값을 가지기 때문에 해당 quadratic form은 negative definite입니다.





<br/>



지금까지 quadratic form에 대해서 알아보았습니다. 다음 포스트에서는 Singular Value Decomposition에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!