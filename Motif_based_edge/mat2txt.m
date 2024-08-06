function back=mat2txt(file_name,matrix)
    fid=fopen(file_name,'wt');
    [M,N]=size(matrix);
    for m=1:M
        for n=1:N
            fprintf(fid,' %s',mat2str(matrix(m,n)));
        end
        fprintf(fid,'\n');
    end
    back=fclose(fid);
end
