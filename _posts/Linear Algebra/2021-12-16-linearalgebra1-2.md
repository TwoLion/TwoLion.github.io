---
layout: single
title:  "1.2 Solving a linear system"
categories: [Linear Algebra]
tag: [Linear Algebra]
toc: true
author_profile: false #프로필 생략 여부
use_math: true
sidebar:
    nav: "docs"
---





두 번째 포스트에서는 저번 포스트에서 정의한 linear equation와 linear system을 푸는 방법에 대해 다루어 보겠습니다.



### 1) Elementary Row Operation



linear system을 푸는 과정에서, linear equation에, 혹은 linear equation끼리 특정한 연산을 가하게 됩니다. 예를 들어



$x+y=1 $ - 1번 식

$x+2y=3$ - 2번 식

의 linear system을 푸는 것을 생각해봅시다. 위 system을 만족하는 $x,  y$를 찾기 위해서, 2번식에서 1번식을 빼게 되면

$(x+2y)-(x+y) = 3-1$

$\Rightarrow y=2$

을 얻게 되고, $x=-1$임을 알 수 있습니다. 

혹은, 1번식의 양변에 2를 곱한 후, 2번 식을 빼게 되면

$2x+2y=2$

$x+2y=3$

$\Rightarrow x=-1$

을 구할 수 있습니다. (기본적인 연립일차방정식 풀이 방법 중 하나입니다.)

위의 linear system을 풀 때, linear equation끼리 연산을 하여 변수를 하나만 남겨놓게 만들어 solution을 구했습니다.  linear equation끼리의 연산을 Row operation이라고 하고, linear system의 solution을 구할 때 사용하는 가장 기본적인 operation 3개를 **elementary row operation**이라고 합니다.

1. **Replacement** : 하나의 equation을 자신의 곱과 다른 식의 합 또는 차로 바꾸는 operation
2. **Interchange** : 하나의 equation과 다른 equation의 위치를 바꾸는 operation
3. **Scaling** : 하나의 equation을 자신의 실수배로 바꾸는 operation



마찬가지로, matrix에 대해서도 elementary row operation을 적용할 수 있습니다. 이전 포스트에서 linear system을 matrix로 표현하는 방법인 augmented matrix을 이용하여 linear system을 풀 경우, matrix의 행(row)에 위와 같은 operation을 취해서 solution을 구할 수 있습니다. 행렬에서의 elementary row operation은 다음과 같습니다.



1. **Replacement** : 하나의 row을 자신의 곱과 다른 row의 합 또는 차로 바꾸는 operation
2. **Interchange** : 하나의 row와 다른 row의 위치를 바꾸는 operation
3. **Scaling** : 하나의 row을 자신의 실수배로 바꾸는 operation



Linear system을 풀 때 elementary row operation만으로 solution을 구할 수 있습니다. 앞서 든 예시를 다시 한번 보면서 적용해보겠습니다.

$x+y=1 $ - 1번 식

$x+2y=3$ - 2번 식



여기서 2번식 대신에, 2번식에서 1번식을 뺀 새로운 식으로 대체하는 replacement를 적용하면



$\begin{aligned} x+y&=1 \\\ y&=2  \end{aligned}$

과 같이 나오게 됩니다. 

마지막으로 1번식에 새로운 2번식을 빼주는 replacement를 적용하면

$x=-1$

$y=1$

으로 solution을 구할 수 있습니다.



위의 linear system을 다른 elementary row operation을 통해서 solution을 구해보겠습니다.



1번식 전체에 2를 곱하여 대체하는 scaling을 적용하면

$\begin{aligned} 2x+2y&=2 \\\ x+2y&=3  \end{aligned}$

이 나옵니다. 이 후 1번식 대신 1번식에서 2번식을 뺀 새로운 식으로 대체하는 replacement를 적용하면

$\begin{aligned} x\qquad \,&=-1 \\\ x+2y&=3  \end{aligned}$

이 나오고, 2번식 대신 2번식에서 1번식을 뺀 새로운 식으로 대체하는 replacement를 적용하면

$\begin{aligned} x&=-1 \\\ 2y&=2  \end{aligned}$

마지막으로, 2번식 전체에 1/2를 곱하여 대체하는 scaling을 적용하여

$\begin{aligned} x&=-1 \\\ y&=1  \end{aligned}$

으로 solution을 구할 수 있습니다.



#### (1) Row Equivalant



앞서 다룬 예제를 augmented matrix를 통해서도 풀 수 있습니다. 여기서 확인해야 하는 점은 어떤 matrix $A$에 row operation을 통해 $B$라는 새로운 matrix를 만들었다면, 마찬가지로 matrix $B$에서 row operation을 통해 matrix $A$를 만들 수 있다는 점입니다. ($A$에서 $B$를 만드는 과정에서 사용한 row operation을 반대로 사용하면 만들 두 있습니다.)

이처럼 하나의 matrix에서 row operation을 통해 다른 matrix를 만들 수 있을 때, 두 matrix는 **Row equivalent**하다라고 합니다. 

앞서 다룬 에제를 augemented matrix를 통해 row operation을 적용했을 때 나타나는 모든 matrix가 row equivalent합니다. 

Row equivalent의 의미를 안다면,  다음의 명제를 얻을 수 있습니다.



**2개의 linear system의 augmented matrix가 row equivalent하다면, 두 linear system은 같은 solution set을 갖는다. **



즉,  row operation은 linear system의 solution에 영향을 주지 않습니다.  따라서 row operation을 통해서 linear system의 solution을 찾을 수 있는 것이구요.





### 2) Row Echelon Form



Augmented matrix를 이용하여 풀 때,  linear system의 solution을 바로 알 수 있는 augmented matrix의 모양이 있습니다. matrix에 적혀있는 숫자가 사다리꼴 모양으로 분포가 되어 있어 row echelon form이라고 하는 형식은, 다음과 같이 정의됩니다.



A rectangluar matrix is in row echelon form if it has the following three properties

1. All nonzero rows are above any rows of all zeros
2. Each leading entry of a row is in a column to the right of the leading entry of the row above it
3. All entries in a column below the leading entry are zeros.

