clc;
clear;
classifier_names = {'KNN', 'SVM', 'LDA'};
svm_para = {
    {'BoxConstraint', 5000, 'KernelFunction', 'rbf', 'KernelScale', 1}, ...
    {'BoxConstraint', 10000, 'KernelFunction', 'rbf', 'KernelScale', 1/sqrt(32)}, ...
    {'BoxConstraint', 10000, 'KernelFunction', 'rbf', 'KernelScale', 1}, ...
    {'BoxConstraint', 100, 'KernelFunction', 'rbf', 'KernelScale', 1/sqrt(32)}, ...
    {'BoxConstraint', 100, 'KernelFunction', 'rbf', 'KernelScale', 1/(2*sqrt(2))} ...
};
Dataset = get_data(dataset_names{2});
Dataset.train_ratio = 0.1;
Dataset.svm_para = svm_para{1, 1};
superpix_path={'Dataset/IP/Indian_Pines-40.mat','Dataset/PU/PaviaU-200.mat','Dataset/SA/Salinas-150','Dataset/BS/Botswana-X1.mat'};
dataset_name={'Indian','PaviaU','Salinas','Botswana'};
res=py.train_s3.run_band_selection(dataset_name{2},superpix_path{2},pyargs('num_epochs', int32(250)));
W_py = res.data;
if isa(W_py, 'py.torch.Tensor')
    % 确保张量在CPU上
    if logical(W_py.is_cuda)
        W_py = W_py.cpu();
    end
    % 分离计算图并转换为numpy
    W_np = W_py.detach().numpy();
    % 转换为MATLAB双精度矩阵
    W = double(W_np);
else
    W = double(W_py);  % 如果已经是numpy数组直接转换
end


feature_scores = vecnorm(W, 2, 1); % 沿行(第2维度)计算L2范数

% 对特征得分进行降序排序并获取索引
[~, selected_features] = sort(feature_scores, 'descend');

% 将索引转换为整数类型（可选）
selected_features = int32(selected_features);


num_bands = 30;  % 生成 [5,10,15,20,25,30,35,40,45,50]



    selected_bands = selected_features(1:n);  % 取前n个索引


    [acc,~] = test_bs_accu(selected_bands, Dataset,classifier_names{2}); 
    disp(acc);