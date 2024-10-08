function Motifs=M41_generator(A)%A为邻接矩阵
    [b1,b2]=find(tril(A));%tril()返回下三角矩阵,[row,col]=find()返回非0元素的行列下标，注意matlab矩阵为按列存储
    M=[];%储存模体
    for i=1:length(b1)%非0元素的总数，每个非0元素都是一条边，可用（b1(i), b2(i)）确定；遍历每一条边
        lj1=setdiff(find(A(b1(i),:)),b2(i));%b1(i)邻居，C=setdiff(A,B)函数返回在向量A中却不在向量B中的元素，并且C中不包含重复元素，并且从小到大排序：find(A(b1(i),:))为b1(i)的所有邻居，lj1为b1(i)不包含b2(i)的邻居
        lj2=setdiff(find(A(b2(i),:)),b1(i));%b2(i)邻居
         for j=1:length(lj1)
             for k=1:length(lj2)
                 if lj1(j)~=lj2(k)&&A(lj1(j),lj2(k))==0&&A(lj1(j),b2(i))==0&&A(lj2(k),b1(i))==0
                   M1=[b1(i);b2(i);lj1(j);lj2(k)]; %用三维列向量表示模体
                   M2=sort(M1);%模体节点从小到大排序
                   M=[M,M2];%添加到M
                 end   
            end
        end
    end
    Motifs=unique(M','rows');%去除重复模体
end