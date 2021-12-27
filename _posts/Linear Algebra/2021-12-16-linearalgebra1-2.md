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



$x+y=1 $ \\
$x+2y=3$



의 linear system을 푸는 것을 생각해봅시다. 위 system을 만족하는 $x,  y$를 찾기 위해서, 2번식에서 1번식을 빼게 되면



$(x+2y)-(x+y) = 3-1$ \\
$\Rightarrow y=2$



을 얻게 되고, $x=-1$임을 알 수 있습니다. 

혹은, 1번식의 양변에 2를 곱한 후, 2번 식을 빼게 되면  



$2x+2y=2$ \\
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



$x+y=1$  : 1번 식 

$x+2y=3$  : 2번 식



여기서 2번식 대신에, 2번식에서 1번식을 뺀 새로운 식으로 대체하는 replacement를 적용하면





$$
\begin{aligned} x+y&=1 \\\ y&=2  \end{aligned}
$$





과 같이 나오게 됩니다. 

마지막으로 1번식에 새로운 2번식을 빼주는 replacement를 적용하면



$$
x=-1\\
y=2
$$


으로 solution을 구할 수 있습니다.



위의 linear system을 다른 elementary row operation을 통해서 solution을 구해보겠습니다.



1번식 전체에 2를 곱하여 대체하는 scaling을 적용하면


$$
\begin{aligned} 2x+2y&=2 \\\ x+2y&=3  \end{aligned}
$$



이 나옵니다. 이 후 1번식 대신 1번식에서 2번식을 뺀 새로운 식으로 대체하는 replacement를 적용하면


$$
\begin{aligned} x\qquad \,&=-1 \\\ x+2y&=3  \end{aligned}
$$



이 나오고, 2번식 대신 2번식에서 1번식을 뺀 새로운 식으로 대체하는 replacement를 적용하면


$$
\begin{aligned} x&=-1 \\\ 2y&=2  \end{aligned}
$$



마지막으로, 2번식 전체에 1/2를 곱하여 대체하는 scaling을 적용하여


$$
\begin{aligned} x&=-1 \\\ y&=1  \end{aligned}
$$



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



용어 정리 : leading entry란 각 행에서 처음으로 0이 아닌 값이 나오는 entry를 뜻합니다. 



이를 정리하면 다음과 같습니다. 

1. 0만 존재하는 행은 0이 아닌 값이 존재하는 행보다 무조건 아래에 위치해야 한다.
2. 윗 행에 존재하는 leading entry의 위치는 아래 행에 존재하는 leading entry의 위치보다 왼쪽에 있어야 한다. 
3. leading entry 아래에 있는 성분(entry)의 값은 모두 0이어야 한다. 



다음의 예시를 보겠습니다. 


$$
\begin{bmatrix}1&2&3&3\\0&4&5&8\\0&0&9&3\\0&0&0&0\\ \end{bmatrix}
$$


1. 4번째 행은 0만 포함한 행이므로 0이 아닌 값을 포함한 행보다 아래에 위치해 있습니다.
2. leading entry는 1, 4, 9인데, 1은 4보다 왼쪽에, 4는 9보다 왼쪽에 위치합니다.
3. leading entry 1, 4, 9의 아랫 성분들은 모두 0입니다.

위 matrix는 row echelon form의 3가지 조건을 만족합니다. 따라서, 위의 matrix form은 echelon form입니다. 

echelon이라는 단어가 쓰이는 이유는 0이 아닌 entry 전체를 보면 사다리꼴 모양으로 생겨서 echelon이라는 단어를 사용하였습니다. 



위 echelon form 조건에서, 다음의 두 가지 조건을 추가적으로 만족한다면, 그 matrix는 **reduced echelon form**이라고 합니다.



4. The leading entry in each nonzero row is 1
5. Each leading 1 is the only nonzero entry in its column



정리하면,

4. leading entry의 값이 1이어야 한다.
5. leading entry를 포함하는 column에서 leading entry를 제외한 나머지 성분은 0이어야 한다.



위에 나온 예시 matrix는 leading entry가 1이 아니고, leading entry를 포함한 열에 다른 0이 아닌 성분이 존재하기 때문에, reduced echelon form이 아닙니다. 


$$
\begin{bmatrix}1&0&0&0\\0&1&0&-1\\0&0&1&3\\0&0&0&0\\ \end{bmatrix}
$$


다음의 matrix의 경우, echelon form임과 동시에, leading entry 값이 1이고, leading entry를 포함한 열의 다른 성분이 모두 0이므로, reduced echelon form입니다.



Linear system을 풀 때, linear system의 echelon form과 reduced echelon form을 구할 수 있다면, linear system의 solution을 바로 찾아낼 수 있습니다. 

이와 관련된 정리가 아래의 정리입니다. 





#### Theorem: Uniqueness of the Reduced Echelon Form

Each matrix is row equivalent to one and only one reduced echelon matrix



즉, 각각의 matrix는 오직 하나의 reduced echelon matrix와 row equivalent합니다. 

(각각의 matrix와 row equivalent한 echelon matrix는 여러개 존재할 수 있지만, reduced echelon matrix는 하나만 존재합니다.)



만약 $A$ matrix가 echelon matrix인 $U$와 row equivalent하면, $U$ 는 A의 echelon form이라고  합니다.

만약 $A$ matrix가 reduced echelon matrix인 $U$와 row equivalent하면, $U$는 A의 reduced echelon form이라고 합니다. 





#### (1) Pivot Position

Echelon form과 연결되어 사용이 되는 pivot position과 pivot column에 대한 정의입니다. 



A  pivot position in a matrix $A$ is the location in $A$ that corresponds to a leading 1 in the reduced echelon form of $A$

A pivot column is a column of $A$ that contains a pivot positions



즉, matrix $A$의 pivot position은 $A$의 reduced echelon form에서의 leading 1의 위치이고, pivot column은 pivot position을 가지는 column입니다. 



pivot position을 이용하여 free variable과 basic variable을 정의할 수 있습니다. 





#### (2) Free variable



Basic variable : The variables corresponding to pivot columns in the matrix

Free variable : The variables except the pivot columns in the matrix



linear system을 augmented matrix, 또는 coefficient matrix로 만든 후, reduced echelon form을 만들었을 때, pivot column에 해당하는 variable이 free variable이고, pivot column에 해당하지 않는 variable이 basic variable입니다. 





### 3) Row Reduction algorithm





위에서 정의한 echelon form과 pivot position, free variable과 basic variable은 linear system을 augmented matrix를 이용하여 해결할 때 사용됩니다. 알고리즘은 다음과 같습니다. 



1. Linear system을 augmented matrix로 바꾼다. 
2. augmented matrix를 row operation을 통해 reduced echelon form으로 바꾼다. 
3. reduced echelon form을 이용하여 각 variable 해당하는 solution을 찾는다.



즉, Reduced echelon form을 구하면 solution 또한 바로 나올 수 있게 됩니다. 예시를 통해 적용해보겠습니다. 
$$
\begin{aligned}
2x_3+4x_4+4x_5&=0\\
2x_1-4x_2-2x_3+2x_4+2x_5&=0\\
2x_1-4x_2+9x_4+6x_5&=0\\
3x_1-6x_2+9x_4+9x_5&=0
\end{aligned}
$$
위 linear system의 augmented matrix는 다음과 같습니다. 
$$
\begin{bmatrix}0&0&2&0&0&0\\2&-4&-2&2&2&0\\2&-4&0&9&6&0\\3&-6&0&9&9&0\\ \end{bmatrix}
$$


위 matrix를 row operation을 통하여 reduced echelon form으로 바꾸면


$$
\begin{bmatrix}1&-2&0&0&3&0\\0&0&1&0&2&0\\0&0&0&1&0&0\\0&0&0&0&0&0\\ \end{bmatrix}
$$
이 됩니다. 여기서 leading entry는 (1, 1), (2, 3), (3, 4) 성분이 되며, augmented matrix의 (1, 1), (2, 3), (3, 4) 위치가 pivot position이 되고, 1열, 3열, 4열에 해당하는 변수 $x_1, x_3, x_4$가 basic variable, $x_2, x_5$가 free variable이 됩니다. 

Reduced echelon form을 다시 linear system으로 변경시키면\\


$$
\begin{aligned}

x_1-2x_2+3x_5&=0\\
x_3+2x_5&=0\\
x_4&=0


\end{aligned}
$$




\\과 같이 나오게 됩니다. 이를 basic variable인 $x_1, x_3, x_4$로 나타내게 되면


$$
\begin{aligned}
x_1&=2x_2+3x_5\\
x_3&=-2x_5\\
x_4&=0\\
x_2, x_5&: free \ varible 
\end{aligned}
$$
이 됩니다. 즉, free variable인 $x_2, x_5$에는 아무 값을 넣더라도, $x_1, x_3, x_4$가 다음의 조건을 만족하면 성립합니다.



Reduced echelon form을 구할 수 있으면 linear system의 solution을 구할 수 있고, pivot position과 pivot column을 통해서 basic variable과 free variable을 구분할 수 있습니다. basic varible에 대한 solution은 free variable에 대한 식으로 나타내어지고, free variable의 solution은 variable의 domain (위의 예시의 경우 실수)가 됩니다. 





#### 4) Solution of Linear system





Linear system의 solution type은 3가지로 구분이 됩니다. 



1. solution이 없는 경우(inconsistent)
2. solution이 하나만 있는 경우
3. solution이 무수히 많은 경우



위 3가지 경우를 augmented matrix의 (reduced) echelon form과 pivot column을 통해서 확인할 수 있습니다.





#### Theorem : Existence and Uniquness theorem





A linear system is consistent if and only if the rightmost column of the augmented matrix is not a pivot column.

That is, if and only if an echelon form of the augmented matrix has no row of the form

$[0 \ 0 \ 0 \ .... \ b]$, where $b\neq0$

If a linear system is consistent, then the solution set contains either a unique solution, when there are no free varaibles, or infinitely many solutions, when there is at least one free variable.



첫 번째로, linear system이 inconsistent한 경우를 보면, echelon form에서의 어떤 행이



$[0 \ 0 \ 0 \ .... \ b]$, where $b \neq 0$



와 같이 생기게 된다면, 위의 row를 방정식으로 나타내면



$0=b$



가 됩니다. 이는 성립할 수가 없기 때문에, 위 linear system을 만족시키는 solution은 존재하지 않습니다.



두 번째로, linear system이 consistent한 경우, linear system에서 free variable이 존재하는 경우와 존재하지 않는 경우로 나눌 수 있습니다. Solution에서 free variable에 해당하는 부분은 특정한 조건이 없기 때문에, 실수 전체에 대해서 solution이 성립합니다. 따라서 free variable이 존재하는 경우는 solution이 무수히 많게 되고, free variable이 없는 경우는 solution이 하나만 존재하게 됩니다.






지금까지 linear system의 solution을 구하는 방법에 대해서 알아보았습니다. 다음 포스팅에서는 linear system을 표현하는 다양한 방법에 대해서 알아보겠습니다.
