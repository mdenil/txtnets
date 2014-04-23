function [stack decodeInfo] = param2stack(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,gpu)
% fast gpu version

if gpu

    stack = gpuArray.zeros(numel(m1)+numel(m2)+numel(m3)+numel(m4)+numel(m5)+ ... 
        + numel(m6)+numel(m7)+numel(m8)+numel(m9)+numel(m10),1,'single');

    decodeInfo = gpuArray.zeros(10,2,'single'); %all inputs assumed to be matrices
else
     stack = zeros(numel(m1)+numel(m2)+numel(m3)+numel(m4)+numel(m5)+ ... 
        + numel(m6)+numel(m7)+numel(m8)+numel(m9)+numel(m10),1);
     decodeInfo = zeros(10,2); %all inputs assumed to be matrices
end
    
num=0;

stack(num+1:numel(m1)) = m1(:);
decodeInfo(1,:) = size(m1);
num = num+numel(m1);

stack(num+1:num+numel(m2)) = m2(:);
decodeInfo(2,:) = size(m2);
num = num+numel(m2);

stack(num+1:num+numel(m3)) = m3(:);
decodeInfo(3,:) = size(m3);
num = num+numel(m3);

stack(num+1:num+numel(m4)) = m4(:);
decodeInfo(4,:) = size(m4);
num = num+numel(m4);

stack(num+1:num+numel(m5)) = m5(:);
decodeInfo(5,:) = size(m5);
num = num+numel(m5);

stack(num+1:num+numel(m6)) = m6(:);
decodeInfo(6,:) = size(m6);
num = num+numel(m6);

stack(num+1:num+numel(m7)) = m7(:);
decodeInfo(7,:) = size(m7);
num = num+numel(m7);

stack(num+1:num+numel(m8)) = m8(:);
decodeInfo(8,:) = size(m8);
num = num+numel(m8);

stack(num+1:num+numel(m9)) = m9(:);
decodeInfo(9,:) = size(m9);
num = num+numel(m9);

stack(num+1:num+numel(m10)) = m10(:);
decodeInfo(10,:) = size(m10);
num = num+numel(m10);
