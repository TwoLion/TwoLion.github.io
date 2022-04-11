---
layout: single
title:  "5.2 Characteristic Equation"
categories: [Linear Algebra]
tag: [Linear Algebra, Characteristic Equation, Similarity]
toc: true
author_profile: true #프로필 생략 여부
use_math: true
---









이번 포스트에서는 characteristic equation에 대해서 알아보도록 하겠습니다. 



<br/>

### 1) Characteristic Equation

<br/>



이전 포스트에서 eigenvector와 eigenvalue에 대해서 알아보았습니다. 어떤 $A$ matrix가 주어졌을 때, $A$의 eigenvalue를 알아야 이에 대응하는 eigenvector를 구할 수 있습니다. $A$  eigenvalue와 관련된 식이 바로 characteristic equation입니다. 



<br/>



**Definition : The characteristic equation**



Let $A$ be $n\times n$ matrix, then


$$
\det(A-\lambda I) =0
$$


is called the characteristic equation of $A$. And


$$
\det (A-\lambda I)
$$


is called the characteristic polynomial of $A$



즉, $A-\lambda I$의 determinant가 characteristic polynomial이고, 이를 이용한 방정식


$$
\det (A-\lambda I) =0
$$


이 characteristic equation of $A$라고 합니다. characteristic polynomial은 $\lambda$에 대한 식입니다. 



$A$의 eigenvalue가 $\lambda$라면, $\lambda$는


$$
A\boldsymbol{x} = \lambda \boldsymbol{x}
$$


가 non-trivial solution을 가져야 합니다. 이는


$$
(A-\lambda I)\boldsymbol{x} =0
$$


이 non-trivial solution을 가져야 한다는 것과 같고, 이는 $A-\lambda I$가 invertible하지 않아야 합니다. 즉


$$
\det (A-\lambda I) =0
$$


을 만족시키는 $\lambda$가 $A$의 eigenvalue가 될 수 있습니다. 만약, 위 equation의 solution이 muliplicity(중복도, 중근)을 가질 수 있고, 이 때 eigenvalue의 multiplicity는 characteristic equation의 solution의 multiplicity로 정의합니다.



</br>



*example*


$$
A = \begin{bmatrix} 5 & -2 & 6 & -1 \\ 0 & 3 & -8 & 0 \\ 0 & 0 & 5 & 4 \\ 0 & 0 & 0 & 1\end{bmatrix}
$$


$A$의 characteristic polynomial은


$$
\det(A-\lambda I) = (5-\lambda)(3-\lambda)(5-\lambda)(1-\lambda)
$$


가 되고, characteristic equation은


$$
\det(A-\lambda I) = (5-\lambda)(3-\lambda)(5-\lambda)(1-\lambda)=0
$$


 이 되어 위 equation의 solution인


$$
\lambda = 5, 3, 1
$$


이 $A$의 eigenvalue가 됩니다. 이 때 $\lambda=5$는 중근을 가지므로 mulitplicity가 2인 eigenvalue입니다.





</br>



### 2) Similarity



</br>



특정 matrix가 복잡할 때, 해당하는 matrix와 비슷하지만, 비교적 간단한 matrix를 이용할 수 있습니다. 여기서 두 matrix가 비슷하다는 것은 어떤걸 뜻할까요? 두 matrix가 similar하다는 것을 다음과 같이 정의합니다. 



</br>



**Definition : Similarity**



Let $A, B$ be $n \times n $ matrices. Then

$A$ is similar to $B$ if there is an invertible matrix $P$ such that


$$
P^{-1}AP=B, \ \ \ or \ \ \ A=PBP^{-1}
$$


$B$ is also similar to $A$, so we can say that $A$ and $B$ are similar



Changing $A$ into $P^{-1}AP=B$ is called similarity transformation





$A, B$가 similar하다는 것은 어떤 invertible matrix $P$가 존재하여


$$
A=PBP^{-1}
$$


을 만족함을 뜻합니다. 이 때 $A$에서 $B$로 바꿔주는 mapping을 similarity transformation이라고 합니다. 



$A$와 $B$가 similar하면 두 matrix는 다음의 성질을 공유합니다. 



<br/>

**Theorem**



1. Similar matrices have the same determinant
2. Similar matrices have the same rank
3. Similar matrices have the same nullity
4. Similar matrices have the same trace
5. Similar matrices have the same characteristic equation and have the same eigenvalues



두 matrix가 similar 하면 특정 성질을 공유합니다.  완전히 같지는 않지만 특정 성질을 공유하기 때문에, 특정한 상황에서 similarity transformation을 통해서 matrix에 대한 해석을 용이하게 할 수 있습니다.  similarity를 이용하는 방법은 다음 포스트에서 다룰 예정입니다.(증명은 appendix 참고해주시기 바랍니다.)



</br>

* **Caution**



Eigenvalue가 같다고 해서 두 matrix가 similar하지는 않습니다. (similar이면 eigenvalue가 같지만, 역은 성립하지 않습니다. )

Similarity와 row equivalent는 완전히 다른 개념입니다. 즉 similar한 matrix끼리 row equivalent하다고 말할 수 없습니다.



</br>



지금까지 Characteristic equation에 대해서 알아보았습니다. 다음 포스트에서는 diagonalization에 대해서 알아보겠습니다. 질문이나 오류 있으면 댓글 남겨주세요! 감사합니다!



</br>

### Appendix : Proof of Theorem

</br>



**Theorem**



1. Similar matrices have the same determinant
2. Similar matrices have the same rank
3. Similar matrices have the same nullity
4. Similar matrices have the same trace
5. Similar matrices have the same characteristic equation and have the same eigenvalues



<br/>



* **Proof**



<br/>



$n \times n$ matrix $A, B$가 similar하다고 가정해봅시다. 그럼 어떤 invertible matrix $P$가 존재하여


$$
A=PBP^{-1}
$$


을 만족합니다. 



<br/>



* Proof of 1


$$
A=PBP^{-1}
$$


이므로


$$
\det(A) = \det(PBP^{-1}) = \det(P)\det(B)\det(P^{-1}) = \det(B)\det(P)\det(P^{-1})=\det(B)
$$


가 되어 $A, B$의 determinant가 동일합니다.



<br/>



* Proof of 2, 3



$A$와 $B$의 rank, nullity가 같음을 밝히기 위해 다음을 밝힐 예정입니다.


$$
rankB =rankPB = rankAP = rankA
$$


두 matrix의 rank가 같으면 rank theorem에 의해 nulity도 같음을 알 수 있습니다. 



(1) $rankB = rankPB$



$P$가 invertible하므로


$$
NulB = NulPB
$$


가 성립합니다. 이는


$$
PB\boldsymbol{x} = 0 \iff B\boldsymbol{x} =0
$$


이기 때문입니다. $P$ 또한 $n \times n$ matrix이므로 rank theorem에 의해


$$
rank B = rank PB
$$


가 성립합니다. 





(2) $rankAP = rankA$



$P$가 invertible하므로, $P^T$ 또한 invertible합니다. 또한 rank의 정의에 의해


$$
rankAP = rank(AP)^T = rank(P^TA^T)
$$


가 성립합니다.  이 경우, (1)에서의 방법과 마찬가지로


$$
Nul(P^TA^T) = NulA^T
$$


가 성립하므로, 


$$
rankAP = rankP^TA^T = rankA^T = rankA
$$


가 됩니다. 



(3) $rankPB = rankAP$



현재 $A, B$가 similar하므로


$$
A =PBP^{-1}
$$


가 성립합니다. 양변에 $P$를 곱해주면


$$
AP = PB
$$


가 성립하여, 두 matrix가 같기 때문에 두 matrix의 rank 또한 같습니다. 따라서


$$
rankA = rankB
$$


 가 성립하고, rank theorem에 의해




$$
Nullity A =Nullity B
$$


가 성립합니다.





<br/>







* Proof of 3


$$
A =PBP^{-1}
$$


의 trace를 이용하면


$$
tr(A) = tr(PBP^{-1})=tr(BP^{-1}P)=tr(B)
$$


가 됩니다.





<br/>







* Proof of 4



$A$의 characteristic polynomial은


$$
\det(A-\lambda I)
$$


 입니다. 이 식은


$$
\det(A-\lambda I) = \det (PBP^{-1}-\lambda PP^{-1}) = \det(P(B-\lambda I)P^{-1})  \\
=\det(P)\det(B-\lambda I)\det(P^{-1}) = det(B-\lambda I)\det(P)\det(P^{-1}) = det(B-\lambda I)
$$


가 되어 


$$
\det(A-\lambda I) = \det(B-\lambda I)
$$


가 성립합니다. 즉 $A, B$의 characteristic polynomial이 같기 때문에, characteristic equation도 같고, 따라서 eigenvalue 또한 같습니다.



