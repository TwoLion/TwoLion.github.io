---
layout: single
title:  "Logistic Regression (2)"
categories: [Google Boot Camp]
tag: [Deep Learning]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---











이번 포스트에서는 computational graph와 vectorization에 대해서 알아보도록 하겠습니다. 또한 이를 이용하여 logistic regression에 적용해보도록 하겠습니다.





<br/>

### 1) Computation Graph

<br/>



#### (1) Computational Graph



<br/>



Computational Graph는 어떠한 식을 계산하는 일련의 과정을 순서에 맞게 나타낸 그래프입니다. 다음 예를 통해 computational graph에 대해 알아보도록 하겠습니다.



<br/>



*example*


$$
J(a, b, c) = 3(a+bc)
$$


다음의 식에서 $a, b, c$가 주어졌을 때, 계산하는 순서는 다음과 같습니다.



1. $b\times c$의 결과값을 구한다.
2. 1.의 결과인 $bc$에 $a$를 더한다.
3. 2.의 결과인 $a+bc$에 3을 곱한다.



이렇게 3번의 연산으로 $J$가 계산됩니다. 순서에 대한 결과값을 명확하게 하기 위해



1. $u = bc$
2. $v = a+bc = a+u$
3. $J = 3(a+bc) = 3v$



로 새로운 $u, v$를 정의하겠습니다. 위의 연산을 도식화하면 다음과 같습니다.



![딥러닝 computational graph](../../images/2022-07-15-Deeplearning1-2(2)/딥러닝 computational graph.png)







이처럼 연산의 과정을 도식화한 것을 computational graph입니다. Logistic regression에서도 같은 방법으로 computational graph를 정의할 수 있습니다. 



<br/>

#### (2) Computational Graph and Derivative

<br/>

그렇다면 computational graph를 통해 얻을 수 있는 이점은 무엇일까요? 바로 각 변수에 대한 derivative를 효율적으로 구할 수 있습니다.



미분에서 합성함수의 미분(chain rule)을 이용하면 여러 함수가 합성된 함수의 derivative 또는 gradient를 구할 수 있습니다.

만약, $u = f(x), \  y = g(u)=g(f(x))$와 같은 함수에서


$$
\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx} = g'(f(x))f'(x)
$$


와 같이 구할 수 있습니다.



이를 이용하면 각 단계에서 얻어지는 값의 derivative를 구할 수 있습니다.



1. $\frac{\partial J}{\partial v}$는 $J$를 $v$에 대해 미분해서 바로 얻을 수 있습니다. ($\frac{\partial J}{\partial v} = 3$)
2. $\frac{\partial J}{\partial u} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial u} = 3 \times 1 =3$
3. $\frac{\partial J}{\partial a} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial a} = 3$
4. $\frac{\partial J}{\partial b} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial u}\frac{\partial u}{\partial b} = 3c$
5. $\frac{\partial J}{\partial c} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial u}\frac{\partial u}{\partial c} = 3b$



만약 $a, b, c$에 어떤 값이 주어져 있다면, $\frac{\partial J}{\partial b}, \frac{\partial J}{\partial c}$에 주어진 $b, c$를 대입하여 얻을 수 있습니다.  이 과정을 도식도로 그리면 다음과 같습니다.



![딥러닝 역전파](../../images/2022-07-15-Deeplearning1-2(2)/딥러닝 역전파.png)



($\frac{\partial J}{\partial a}$ 기호를 간단하게 $da$로 표기하였습니다. $a$부분에 다른 변수를 넣을 수 있습니다.)



물론 $J(a, b, c)$에서 바로 $a, b, c$의 derivative를 구할수도 있습니다. 하지만 위의 경우는 $J$가 간단한 경우이고, $J$와 computational graph가 복잡해질수록 $J$를 통해 바로 원하는 변수의 derivative를 구하기 힘들 수 있습니다. 따라서, 다음과 같은 방법으로 derivative를 계산하게 됩니다. 

이렇게, **$J$에서 computation graph 연산 방향의 반대 방향으로 derivative를 구하는 과정을 딥러닝에서는 back propagation이라고 합니다. 반대로, computation graph 순서대로 연산을 진행하여 $J$를 얻는 과정을 forward propagation이라고 합니다.**





<br/>

#### (3) Logistic Regression and Gradient descent

<br/>



Logistic regression에서의 loss function을 구하는 과정은 다음과 같습니다.



1. $\boldsymbol w^T\boldsymbol x + b$를 계산한다.
2. 1.에서 구한 값에 sigmoid 함수를 합성한다.
3. loss function을 2.에서 구한 값과 $y$값을 이용하여 구한다.



Logistic regression에서의 loss function은 다음과 같습니다(이유는 이전 포스트 참고)


$$
L(a, y)  = -(y\log a + (1-y)\log (1-a))
$$


이를 computation graph로 나타내면 다음과 같습니다.



![로지스틱 computation graph](../../images/2022-07-15-Deeplearning1-2(2)/로지스틱 computation graph.png)





우리가 gradient descent를 이용하기 위해 필요한 값은 $d \boldsymbol w = \frac{\partial L}{\partial \boldsymbol w}$와 $d b = \frac{\partial L}{\partial b}$ 입니다. 이를 back propagation을 통해 구하면 다음과 같습니다.



1. $da$ 구하기

   
   $$
   da = \frac{\partial L(a, y)}{\partial a} = -\frac{y}{a}+\frac{1-y}{1-a} = \frac{a-y}{a(1-a)}
   $$

2. $dz$ 구하기
   
   $$
   \begin{aligned}
   dz &= \frac{\partial L(a, y)}{\partial a}\frac{\partial a}{\partial z} = \frac{a-y}{a(1-a)}\frac{e^z}{(1+e^z)^2} \\ &= \frac{a-y}{a(1-a)}\frac{e^z}{1+e^z}\frac{1}{1+e^z} = \frac{a-y}{a(1-a)}a(1-a)\\ &=a-y
   \end{aligned}
   $$
   

3. $d\boldsymbol w$ 구하기

   
   
   $$
   d\boldsymbol w = \frac{\partial L(a, y)}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial w} = dz \boldsymbol  x \\
   
   dw_i = dz x_i,\ \ \ i=1, ... n_x
   $$
   

4. $db$ 구하기
   $$
   db =  \frac{\partial L(a, y)}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial b} =dz
   $$
   







이를 computation graph로 나타낼 수 있습니다.



![로지스틱 역전파](../../images/2022-07-15-Deeplearning1-2(2)/로지스틱 역전파.png)







<br/>

### 2) Vectorizaton

<br/>



현재까지 하나의 observation에 대해서 computation graph를 그리고, 역전파를 이용하여 gradient descent를 적용하는 방법에 대해 알아보았습니다. 그럼 $m$개의 train examples에 대해서는 어떻게 적용할까요? 



<br/>



#### (1) Vectorization



<br/>



위의 logistic regression 파트에서 확인한 computation graph의 가장 마지막 결과는 loss function이었습니다. 즉 observation 하나에 대한 loss가 최종 단계였죠. 이를 cost function으로만 변경해주면 됩니다. 





![로지스틱 m개 데이터](../../images/2022-07-15-Deeplearning1-2(2)/로지스틱 m개 데이터.png)





다만 위의 식에서 $\hat y$ 부분, $z$, $\boldsymbol x$ 부분에 수정이 필요합니다. $m$개의 데이터를 사용해서 얻은 식이 cost function인데, 위 computation graph에서는 나오지 않으니까요. Observation마다 위 computation graph 각각의 순서를 통해 얻은 결과를 column-wise로 쌓은 matrix를 만들어서 $m$개의 데이터 연산 결과를 표현해줍니다. 


$$
X = \begin{bmatrix}\boldsymbol x^{(1)} & \boldsymbol x^{(2)} & ... & \boldsymbol x^{(m)} \end{bmatrix}, \ \ 
\boldsymbol b = \begin{bmatrix} b & b& ...  & b \end{bmatrix}, \\
Z = \begin{bmatrix}z^{(1)} & z^{(2)} & ... & z^{(m)} \end{bmatrix}, \ \ A = \begin{bmatrix}a^{(1)} & a^{(2)} & ... & a^{(m)} \end{bmatrix}
$$


다음과 같이 matrix를 정의해주면


$$
Z = \boldsymbol w^TX + b \\
A = \sigma(Z) \ \ 
$$


인 것을 알 수 있습니다. ($\sigma(Z)$는 $Z$의 모든 element에 sigmoid 함수를 도입하여 얻은 결과값을 나타낸 벡터입니다.)



위를 도입하였을 때 compuation graph는 다음과 같이 바뀝니다.



![로지스틱 vectorization](../../images/2022-07-15-Deeplearning1-2(2)/로지스틱 vectorization.png)



computation graph를 구했으니, 우리가 원하는 $d\boldsymbol w, db$를 구하기 위해 back -propagation을 적용할 수 있습니다. 여기서, $J(\boldsymbol w, b)$를 살펴보면


$$
J(\boldsymbol w, b) = \frac{1}{m}\sum_{i=1}^m L(\hat{y}^{(i)}, y^{(i)})
$$


인데, 우리는 $L(\hat{y}^{(i)}, y^{(i)})$에서의 back propagation을 구했습니다. 이를 $m$개의 observation에 대해 각각 진행을 해준 후 다 더한 뒤 $m$으로 나누어주면 됩니다. 즉


$$
dA = \frac{1}{m}\begin{bmatrix} da^{(1)} & da^{(2)} & ... & da^{(m)} \end{bmatrix}\\
dZ = \frac{1}{m}\begin{bmatrix} dz^{(1)} & dz^{(2)} & ... & dz^{(m)} \end{bmatrix} \\
d\boldsymbol w = \frac{1}{m} XdZ^T = \frac{1}{m}\begin{bmatrix}x^{(1)}dz^{(1)}+\cdots+x^{(m)}dz^{(m)}\end{bmatrix} \\
db = \frac{1}{m}\sum_{i=1}^mdz^{(i)}
$$




결과적으로 $d\boldsymbol w, db$를 구했으니, gradient descent 방법을 이용하여 $\boldsymbol w, b$를 업데이트하며 최적의 값을 찾아주면 됩니다.





<br/>

지금까지 Computational graph와 vectorization에 대해서 알아보았습니다. 다음 포스트에서는 Neural Network에 대해서 알아보도록 하겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!