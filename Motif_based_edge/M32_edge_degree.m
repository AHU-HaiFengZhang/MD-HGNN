function W=M32_edge_degree(A)%A为邻接矩阵
    W=sparse(length(A),length(A));
    [b1,b2]=find(tril(A));%tril()返回下三角矩阵,[row,col]=find()返回非0元素的行列下标，注意matlab矩阵为按列存储
    
    for i=1:length(b1)%非0元素的总数，每个非0元素都是一条边，可用（b1(i), b2(i)）确定；遍历每一条边
        lj1=setdiff(find(A(b1(i),:)),b2(i));%b1(i)的邻居，C=setdiff(A,B)函数返回在向量A中却不在向量B中的元素，并且C中不包含重复元素，并且从小到大排序：find(A(b1(i),:))为b1(i)的所有邻居，lj1为b1(i)不包含b2(i)的邻居
        for j=1:length(lj1)       
            if A(lj1(j),b2(i))==1
                W(b1(i),b2(i))=W(b1(i),b2(i))+1;      
            end
        end
   end
   W=W+W';
end