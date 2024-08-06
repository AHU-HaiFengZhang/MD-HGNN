function W=M42_edge_degree(A)%A为邻接矩阵
    W=sparse(length(A),length(A));
    [b1,b2]=find(tril(A));%tril()返回下三角矩阵,[row,col]=find()返回非0元素的行列下标，注意matlab矩阵为按列存储
    
    for i=1:length(b1)%非0元素的总数，每个非0元素都是一条边，可用（b1(i), b2(i)）确定；遍历每一条边
        lj1=setdiff(find(A(b1(i),:)),b2(i));%b1(i)邻居，C=setdiff(A,B)函数返回在向量A中却不在向量B中的元素，并且C中不包含重复元素，并且从小到大排序：find(A(b1(i),:))为b1(i)的所有邻居，lj1为b1(i)不包含b2(i)的邻居
        lj2=setdiff(find(A(b2(i),:)),b1(i));%b2(i)邻居
         for j=1:length(lj1)-1
             for k=j+1:length(lj1)
                 if lj1(j)~=lj1(k)&&A(lj1(j),lj1(k))==0&&A(lj1(j),b2(i))==0&&A(lj1(k),b2(i))==0
                    W(b1(i),b2(i))=W(b1(i),b2(i))+1;
                    W(lj1(j),lj1(k))=W(lj1(j),lj1(k))+1/3;
                    W(lj1(j),b2(i))=W(lj1(j),b2(i))+1/3;
                    W(lj1(k),b2(i))=W(lj1(k),b2(i))+1/3;
                 end
             end
         end
         for m=1:length(lj2)-1
             for n=m+1:length(lj2)
                 if lj2(m)~=lj2(n)&&A(lj2(m),lj2(n))==0&&A(lj2(m),b1(i))==0&&A(lj2(n),b1(i))==0
                    W(b1(i),b2(i))=W(b1(i),b2(i))+1;
                    W(lj2(m),lj2(n))=W(lj2(m),lj2(n))+1/3;
                    W(lj2(m),b1(i))=W(lj2(m),b1(i))+1/3;
                    W(lj2(n),b1(i))=W(lj2(n),b1(i))+1/3;
                 end
             end
         end
   end
   W=W+W';
end