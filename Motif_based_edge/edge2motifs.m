edge=load('test.txt');%����Ϊ�ӵ�0��ʼ�ı��б�edgelist
edge(:,:)=edge(:,:)+1;%mat�����1��ʼ
G = biograph(sparse(edge(:,1), edge(:,2), 1));
adj=adjacency(G);
%motifs=M32_generator(adj);
%motifs(:,:)=motifs(:,:)-1;%python�����0��ʼ
motifd=M31_edge_degree(adj);%ѡ��ģ��Ⱦ������ɺ���
motifd=full(motifd);
%save('online_motifs.mat','motifs');
%save('myspace_m31_motifd.mat','motifd');%�޸�����ļ���