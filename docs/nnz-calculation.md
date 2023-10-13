nnz = number of non-zeros

Calculating **number of non-zeros**:

$nnz = n+\sum_{i=1}^{k_{1}}n-i+\sum_{i=1}^{k_{2}}n-i $

symmetrical case:

$nnz = n+\sum_{i=1}^{(b-1)/2}2\cdot(n-i)$

Bandwidth of the product of two banded matrices:

$\text{bandwidth}_{\text{prod}} = (b_1-1)+(b_2-1)+1$

nnz of product:

$nnz = n+\sum_{i=1}^{((b_1-b_2-2))/2}2\cdot(n-i)$