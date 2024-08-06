edge=load('PB.txt');
edge(:,1:2)=edge(:,1:2)-1;
%edge(:,3)=[];
fid=fopen('PB.new.txt','wt');
    [M,N]=size(edge);
    for m=1:M
        for n=1:N
            fprintf(fid,' %s',mat2str(edge(m,n)));
        end
        fprintf(fid,'\n');
    end
    back=fclose(fid);
