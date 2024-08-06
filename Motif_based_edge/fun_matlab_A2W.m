function W=fun_matlab_A2W(A,M,bata)%A为邻接矩阵，M=(1,1,1)
    W=sparse(length(A),length(A));
    du=sum(A);
    A1=A;
    %不考虑节点小于平均度的情况
    nod_del=find(du<=bata);
    A1(nod_del,nod_del)=0;
    A1(:,nod_del)=0;
    A1(nod_del,:)=0;
    [b1,b2]=find(tril(A1));%tril()返回下三角矩阵,[row,col]=find()返回非0元素的行列下标，注意matlab矩阵为按列存储
    
    for i=1:length(b1)%非0元素的总数，每个非0元素都是一条边，可用（b1(i), b2(i)）确定
        lj1=setdiff(find(A(b1(i),:)),b2(i));%核1邻居，C=setdiff(A,B)函数返回在向量A中却不在向量B中的元素，并且C中不包含重复元素，并且从小到大排序：find(A(b1(i),:))为b1(i)的所有邻居，lj1为b1(i)不包含b2(i)的邻居
        lj2=setdiff(find(A(b2(i),:)),b1(i));%核2邻居
         for j=1:length(lj1)
             for k=1:length(lj2)
                 if lj1(j)~=lj2(k)&&du(lj1(j))<du(b1(i))&&du(lj2(k))<du(b1(i))&&du(lj1(j))<du(b2(i))&&du(lj2(k))<du(b2(i))&&A(lj1(j),lj2(k))==0
                     if A(lj1(j),b2(i))==1&&A(lj2(k),b1(i))==1   %模体1
                         W(b1(i),b2(i))=W(b1(i),b2(i))+0.5*M(1);
                         W(lj1(j),lj2(k))=W(lj1(j),lj2(k))+0.5*M(1);
                     elseif A(lj1(j),b2(i))==0&&A(lj2(k),b1(i))==0   %模体3
                         W(b1(i),b2(i))=W(b1(i),b2(i))+1*M(3);
                         W(lj1(j),lj2(k))=W(lj1(j),lj2(k))+1*M(3);   
                     else %模体2
                         W(b1(i),b2(i))=W(b1(i),b2(i))+1*M(2);
                         W(lj1(j),lj2(k))=W(lj1(j),lj2(k))+1*M(2);  
                     end
                  end


             end
         end
   end
   W=W+W';
end

































































































































































































































