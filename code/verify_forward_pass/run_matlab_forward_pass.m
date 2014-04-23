clear;

% maxNumCompThreads(1); % force 1 core

data_f = 'data/SENT_vec_1_emb_ind_bin';

load(data_f, 'train','train_lbl', 'valid', 'valid_lbl',  'test', 'test_lbl', 'size_vocab','sent_length', 'vocab_emb', 'index');


%% ASSIGN P AND indices
p(1) = 42;          disp(strcat('Size word vectors:',num2str(p(1))));
p(2) = sent_length; disp(strcat('Max sent length:',num2str(p(2))));
p(3) = 5;           disp(strcat('Number feat maps in first layer:',num2str(p(3))));
p(5) = 10;           disp(strcat('Number feat maps in second layer:', num2str(p(5))));
p(37) = 18;          disp(strcat('Number feat maps in third layer:', num2str(p(37))));
p(4) = 2;           disp(strcat('Size of kernel in first layer:', num2str(p(4))));
p(6) = 5;           disp(strcat('Size of kernel in second layer:', num2str(p(6))));
p(36) = 3;          disp(strcat('Size of kernel in third layer:', num2str(p(36))));
p(8) = 0;           disp(strcat('Using relu:',num2str(p(8))));
p(9) = 6;           disp(strcat('Number of output classes:',num2str(p(9))));
p(10) = 1;          disp(strcat('Number of conv layers being used (1 or 2 or 3):',num2str(p(10))));
p(7) = 7;           disp(strcat('TOP POOLING width:',num2str(p(7))));
p(12) = 1;          disp(strcat('Folding in first layer:', num2str(p(12))));
p(13) = 1;          disp(strcat('Folding in second layer:', num2str(p(13))));
p(35) = 0;          disp(strcat('Folding in third layer:',num2str(p(35))));
p(30) = size_vocab; disp(strcat('Size vocab (and pad):',num2str(p(30))));
p(32) = 1;          disp(strcat('Word embedding learning ON:',num2str(p(32))));
p(33) = 199;        disp(strcat('if emb learn ON, after how many epochs OFF:',num2str(p(33))));
p(34) = 1;          disp(strcat('use preinitialized vocabulary:',num2str(p(34))));
p(40) = 1;          disp(strcat('Dropout ON:',num2str(p(40))));
%
%
disp(' ');
p(20) = 1e-4;       disp(strcat('Reg E (word vectors):',num2str(p(20))));
p(21) = 3e-5;       disp(strcat('Reg 1 (first conv layer):',num2str(p(21))));
p(22) = 3e-6;       disp(strcat('Reg 2 (second conv layer):',num2str(p(22))));
p(23) = 1e-5;       disp(strcat('Reg 3 (third conv layer):',num2str(p(23))));
p(24) = 1e-4;       disp(strcat('Reg Z (classification layer):',num2str(p(24))));
%
%
%
p(31) = 0;         disp(strcat('GPU and SINGLE on:',num2str(p(31))));



[train_msk, valid_msk, test_msk, p] = Masks(train, train_lbl, valid, valid_lbl, test, test_lbl, p);
CR = RCTM(p);
if p(34) %if use external vocabulary
    CR.E = vocab_emb(1:p(1),:);
    CR.E(:,p(30)) = zeros(size(CR.E,1),1);
end



%% TRAINING

[X, decodeInfo] = param2stack(CR.E, CR.one, CR.one_b, CR.two, CR.two_b, CR.three, CR.three_b, CR.Z, [], [], p);
[CR_E, CR_1, CR_1_b, CR_2, CR_2_b, CR_3, CR_3_b, CR_Z, ~, ~] = stack2param(X, decodeInfo);
CR_E(:,p(30)) = 0; %this and dropout and fft will affect gradient check
save('data/debugging_model_params.mat', ...
    'X', 'decodeInfo', 'CR_E', 'CR_1', 'CR_1_b', 'CR_2', 'CR_2_b', 'CR_3', 'CR_3_b', 'CR_Z');





train_lbl = train_lbl(:,1); %getting rid of length information for sentences
test_lbl = test_lbl(:,1);
valid_lbl = valid_lbl(:,1);




batchsize = 40;

num_batch_epochs = floor(size(train,1)/(batchsize)); %leaves last batch out at an iteration
indices = kron(1:p(1),ones(1,batchsize*p(2)+1)).'; %adding one value for consistent size of E_df

%%

batch_results = {};


p_orig = p;

for j=1:num_batch_epochs
    p = p_orig;
    
    minibatch = reshape(train((j-1)*batchsize+1:j*batchsize,:)',1,[]);%fixed size batches
    labels = train_lbl((j-1)*batchsize+1:j*batchsize);
    mini_msk = train_msk((j-1)*batchsize+1:j*batchsize,:);

    [CR_E, CR_1, CR_1_b, CR_2, CR_2_b, CR_3, CR_3_b, CR_Z, ~, ~] = stack2param(X, decodeInfo);
    CR_E(:,p(30)) = 0; %this and dropout and fft will affect gradient check


    size_mini = length(labels);
    %down
    data = reshape(permute(reshape(repmat(CR_E(:,minibatch),p(3),1),p(1)*p(3),p(2),size_mini),[1,3,2]),p(3)*p(1)*size_mini,p(2));

    %% Composition Maps One
    kernel_one = repmat(CR_1,size_mini,1);
    M_1 = fastConv(data,fliplr(kernel_one),'f',p(31));

    % folding
    %up sum down
    M_1 = reshape(permute(reshape(M_1',[],p(1),p(3)*size_mini),[2,1,3]),p(1),[]);
    p(1) = p(1)/2; %Temp
    M_1 = M_1(1:p(1),:) + M_1(p(1)+1:end,:);
    M_1 = reshape(permute(reshape(M_1,p(1),[],p(3)*size_mini),[1,3,2]),p(3)*p(1)*size_mini,[]);

    % here M_1 axes:
    % embedding_dim * n_feature_maps * minibatch_size, sentence_len
    % d, f, b, w

    %% K-Max pooling
    %apply -inf mask (1)
    mask_1_1 = repmat(reshape(mini_msk(:,1:p(25))',1,[]),p(1)*p(3),1);
    mask_1_1 = reshape(permute(reshape(mask_1_1,p(1)*p(3),[],size_mini),[1,3,2]),p(3)*p(1)*size_mini,[]);
    M_1(mask_1_1) = -Inf; 

    %apply index mask (2)
    [i, indx] = sort(M_1,2,'descend'); 
    mask_1_2 = repmat(reshape(mini_msk(:,p(25)+1:2*p(25))',1,[]),p(1)*p(3),1);
    mask_1_2 = reshape(permute(reshape(mask_1_2,p(1)*p(3),[],size_mini),[1,3,2]),p(3)*p(1)*size_mini,[]);
    indx(mask_1_2) = p(25)+1; 

    %sort masked indices (3)
    sorted_indx = sort(indx,2);

    %remask -inf to 0 and extend width by one zero column (4)
    M_1(mask_1_1) = 0;

    M_1 = [M_1,zeros(size(M_1,1),1)];
    sorted_indx = [sorted_indx,(p(25)+1)*ones(size(sorted_indx,1),1)];


    %pool elements in specified order
    subs_1 = sub2ind(size(M_1),repmat((1:(p(1)*p(3)*size_mini))',p(25)+1,1), sorted_indx(:));
    % embedding_dim * n_feature_maps * size_mini, k+1 (why +1?)
    M_1 = reshape(M_1(subs_1),(p(1)*p(3)*size_mini),p(25)+1);

    %truncate to max pooled width
    M_1 = M_1(:,1:p(26)-p(6)+1);
    map_wdt = p(26)-p(6)+1;
    mask_1_2 = mask_1_2(:,1:map_wdt); %for backprop

    % M_1 axes: d, f, b, w

    %% Nonlinearity after K-Max pooling
    M_1 = reshape(bsxfun(@plus,reshape(M_1,p(1)*p(3),[]),CR_1_b),p(1)*p(3)*size_mini,[]); %Apply bias on wrong reshape, but it is equivalent
    M_1(mask_1_2) = 0;

    M_1 = tanh(M_1);

    % up

    % At outset M_1 axes: d * f * b, w
    m_1 = reshape( ...
        permute( ...
            reshape( ...
                M_1', ... w, d * f * b
                map_wdt,p(1)*p(3),size_mini), ... w, d * f, b
            [2,1,3]), ... d * f, w, b
        p(1)*p(3),map_wdt*size_mini); % d * f, w * b


    %vectorize one-layer model for classification

    M_3 = reshape(...
        m_1, ...  d * f, w * b
        p(1)*p(3)*map_wdt,size_mini ... d * f * w, b
        );

    %%Classification
    % intercept term for softmax
    b_w = ones(1,size_mini);


    % CR_Z is n_classes x (rep_size+1), the +1 is the bias term
    % Z is n_classes x batch_size
    Z = exp(CR_Z*[M_3;b_w]);
    Z = bsxfun(@rdivide,Z,sum(Z)); % sum acts vertically (over classes here)


    
    
    
    
    % batch_results{end+1} = permute(reshape(M_1, p(1), p(3), size_mini, p(7)), [3,4,2,1]);

    batch_results{end+1} = Z;
    
    % ZZ = Z(Z_sub2ind) is a 1xbatch_size matrix where ZZ(i,j) = Z(labels(i), j)
    % Z_sub2ind = sub2ind(size(Z),labels,(1:size_mini)');
    % cost = sum(log(Z(Z_sub2ind)));
    
    % backproppin'
    % D_Z = 0-Z;
    % D_Z(Z_sub2ind) = 1-Z(Z_sub2ind);

    % Z_df = D_Z*[M_3;b_w]';

    % d_z = CR_Z'*D_Z;
    % D_p = d_z(1:end-1,:);

    if mod(j, 100) == 0
        fprintf('.');
    end
end

fprintf('\n');

%%

% batch_results = permute(cat(5, batch_results{:}), [5, 1, 2, 3, 4]);

batch_results = permute(cat(3, batch_results{:}), [3, 1, 2]);

save('data/batch_results_first_layer.mat', 'batch_results');


exit(0);