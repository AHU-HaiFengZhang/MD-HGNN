function Motifs=M32_generator(A)%AΪ�ڽӾ���
    [b1,b2]=find(tril(A));%tril()���������Ǿ���,[row,col]=find()���ط�0Ԫ�ص������±꣬ע��matlab����Ϊ���д洢
    M=[];%����ģ��
    for i=1:length(b1)%��0Ԫ�ص�������ÿ����0Ԫ�ض���һ���ߣ����ã�b1(i), b2(i)��ȷ��������ÿһ����
        lj1=setdiff(find(A(b1(i),:)),b2(i));%b1(i)���ھӣ�C=setdiff(A,B)��������������A��ȴ��������B�е�Ԫ�أ�����C�в������ظ�Ԫ�أ����Ҵ�С��������find(A(b1(i),:))Ϊb1(i)�������ھӣ�lj1Ϊb1(i)������b2(i)���ھ�
        for j=1:length(lj1)       
            if A(lj1(j),b2(i))==1
                M1=[b1(i);b2(i);lj1(j)]; %����ά��������ʾģ��
                M2=sort(M1);%ģ��ڵ��С��������
                M=[M,M2];%��ӵ�M
            end
        end
    end
    Motifs=unique(M','rows');%ȥ���ظ�ģ��
end