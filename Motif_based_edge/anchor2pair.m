anchor=load('groundtruth.number');
print(length(anchor));
pair=mat(length(anchor),2);
pair(:,1)=anchor;
pair(:,2)=anchor;
savetxt('anchor','pair');