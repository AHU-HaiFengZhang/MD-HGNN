function W=fun_matlab_A2W(A,M,bata)%AΪ�ڽӾ���M=(1,1,1)
    W=sparse(length(A),length(A));
    du=sum(A);
    A1=A;
    %�����ǽڵ�С��ƽ���ȵ����
    nod_del=find(du<=bata);
    A1(nod_del,nod_del)=0;
    A1(:,nod_del)=0;
    A1(nod_del,:)=0;
    [b1,b2]=find(tril(A1));%tril()���������Ǿ���,[row,col]=find()���ط�0Ԫ�ص������±꣬ע��matlab����Ϊ���д洢
    
    for i=1:length(b1)%��0Ԫ�ص�������ÿ����0Ԫ�ض���һ���ߣ����ã�b1(i), b2(i)��ȷ��
        lj1=setdiff(find(A(b1(i),:)),b2(i));%��1�ھӣ�C=setdiff(A,B)��������������A��ȴ��������B�е�Ԫ�أ�����C�в������ظ�Ԫ�أ����Ҵ�С��������find(A(b1(i),:))Ϊb1(i)�������ھӣ�lj1Ϊb1(i)������b2(i)���ھ�
        lj2=setdiff(find(A(b2(i),:)),b1(i));%��2�ھ�
         for j=1:length(lj1)
             for k=1:length(lj2)
                 if lj1(j)~=lj2(k)&&du(lj1(j))<du(b1(i))&&du(lj2(k))<du(b1(i))&&du(lj1(j))<du(b2(i))&&du(lj2(k))<du(b2(i))&&A(lj1(j),lj2(k))==0
                     if A(lj1(j),b2(i))==1&&A(lj2(k),b1(i))==1   %ģ��1
                         W(b1(i),b2(i))=W(b1(i),b2(i))+0.5*M(1);
                         W(lj1(j),lj2(k))=W(lj1(j),lj2(k))+0.5*M(1);
                     elseif A(lj1(j),b2(i))==0&&A(lj2(k),b1(i))==0   %ģ��3
                         W(b1(i),b2(i))=W(b1(i),b2(i))+1*M(3);
                         W(lj1(j),lj2(k))=W(lj1(j),lj2(k))+1*M(3);   
                     else %ģ��2
                         W(b1(i),b2(i))=W(b1(i),b2(i))+1*M(2);
                         W(lj1(j),lj2(k))=W(lj1(j),lj2(k))+1*M(2);  
                     end
                  end


             end
         end
   end
   W=W+W';
end

































































































































































































































