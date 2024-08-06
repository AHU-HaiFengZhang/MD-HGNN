edge=load('test.txt');%输入为从点0开始的边列表edgelist
edge(:,:)=edge(:,:)+1;%mat矩阵从1开始
G = biograph(sparse(edge(:,1), edge(:,2), 1));
adj=adjacency(G);
%motifs=M32_generator(adj);
%motifs(:,:)=motifs(:,:)-1;%python矩阵从0开始
motifd=M31_edge_degree(adj);%选择模体度矩阵生成函数
motifd=full(motifd);
%save('online_motifs.mat','motifs');
%save('myspace_m31_motifd.mat','motifd');%修改输出文件名