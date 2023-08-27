
clear;clc;

% Create Info for IQA database


%{
load('/home/vista/Documents/ZhouZehong/WaDIQaM-master/data/LIVEfullinfo.mat');

dtype = {'jp2k', 'jpeg', 'wn/i', 'gblu', 'fast'};

ref_Names = cell(779, 1);
dtype_ids = [];

score_type='DMOS';

for i = 1:779
    rname = ref_names{ref_ids(i), 1};
    ref_Names{i} = ['refimgs/' rname];
    for j = 1:5
        if strcmp(dtype{j}, im_names{i}(1:4))
            dtype_ids = [dtype_ids; j];
        end
    end
end

ref_names = ref_Names;
subjective_scores_STD = subjective_scoresSTD;

save('LIVEinfo.mat', 'dtype_ids', 'im_names', 'index', 'ref_ids', 'ref_names', 'score_type', ...
    'subjective_scores', 'subjective_scores_STD');
%}

%{
load('/home/vista/Documents/ZhouZehong/WaDIQaM-master/data/TID2013fullinfo.mat');

score_type = 'MOS';
subjective_scores_STD = subjective_scoresSTD;
ref_Names = cell(3000,1);
dlevel = [];
dtype_ids = [];

for i = 1:3000
    rname = ref_names{ref_ids(i), 1};
    ref_Names{i} = ['reference_images/' rname];
    dtype_ids = [dtype_ids; str2double(im_names{i}(5:6))];
    dlevel = [dlevel; str2double(im_names{i}(8:9))];
    im_names{i} = ['distorted_images/' im_names{i}];
end

ref_names = ref_Names;

save('TIID2013info.mat', 'dtype_ids', 'im_names', 'index', 'ref_ids', 'ref_names', 'score_type', 'dlevel', ...
    'subjective_scores', 'subjective_scores_STD');
%}

%{
load('/home/vista/Documents/ZhouZehong/WaDIQaM-master/data/KADID-10K.mat');

score_type = 'DMOS';
subjective_scores_STD = subjective_scoresSTD;
ref_Names = cell(10125,1);
dlevel = [];
dtype_ids = [];

for i = 1:10125
    rname = ref_names{ref_ids(i), 1};
    ref_Names{i} = ['images/' rname];
    dtype_ids = [dtype_ids; str2double(im_names{i}(5:6))];
    dlevel = [dlevel; str2double(im_names{i}(8:9))];
    im_names{i} = ['images/' im_names{i}];
end

ref_names = ref_Names;

save('KADID10Kinfo.mat', 'dtype_ids', 'im_names', 'index', 'ref_ids', 'ref_names', 'score_type', 'dlevel', ...
    'subjective_scores', 'subjective_scores_STD');
%}

%{
load('/home/vista/Documents/ZhouZehong/WaDIQaM-master/data/CLIVEinfo.mat');

score_type = 'MOS';
subjective_scores_STD = subjective_scoresSTD;
dtype_ids = [];
dlevel = [];
ref_ids = (1:1162)';

for i = 1:1162
    im_names{i} = ['img/' im_names{i}];
end

ref_names = im_names;

save('LIVECinfo.mat', 'dtype_ids', 'im_names', 'index', 'ref_ids', 'ref_names', 'score_type', 'dlevel', ...
    'subjective_scores', 'subjective_scores_STD');
%}

%{
load('/home/vista/Documents/ZhouZehong/WaDIQaM-master/data/KonIQ-10k.mat');

score_type = 'MOS';
subjective_scores_STD = subjective_scoresSTD;
dtype_ids = [];
dlevel = [];

for i = 1:10073
    im_names{i} = ['512x384/' im_names{i}];
end

ref_names = im_names;

save('KonIQ10Kinfo.mat', 'dtype_ids', 'im_names', 'index', 'ref_ids', 'ref_names', 'score_type', 'dlevel', ...
    'subjective_scores', 'subjective_scores_STD');
%}

%{
load('CSIQInfo_old.mat');

% for i = 1:1000
%     sline = randperm(30);
%     index = [index; sline];
% end
% 
% score_type = 'DMOS';
% dtype_ids = double(dtype_ids');
% im_names = cell(866, 1);
% ref_names = cell(866, 1);
% ref_ids = double(org_ids');
% dlevel = [];
% 
% for i = 1:866
%     for idx = 3:39
%         if img_names(i, idx) == 'g' && img_names(i, idx-1) == 'n' && img_names(i, idx-2) == 'p'
%             im_names{i} = img_names(i,1:idx);
%             dlevel = [dlevel; str2double(img_names(i,idx-4))];
%         end
%     end
%     for idx = 3:28
%         if org_names(i, idx) == 'g' && org_names(i, idx-1) == 'n' && org_names(i, idx-2) == 'p'
%             ref_names{i} = org_names(i,1:idx);
%         end
%     end
% end
%}

%{
data = importdata('/media/vista/Samsung_T5/IQADataset/SPAQ/Annotations/MOS and Image attribute scores.xlsx');

dlevel = [];
dtype_ids = [];
score_type = 'MOS';
im_names = cell(11125,1);

for i = 1:11125
    im_names{i} = ['TestImage/' data.textdata{i+1, 1}];
end
ref_names = im_names;

index = [];
for i = 1:1000
    sline = randperm(11125);
    index = [index; sline];
end

ref_ids = (1:11125)';
subjective_scores = data.data(:,1);
subjective_scores_STD = [];

save('SPAQinfo.mat', 'dtype_ids', 'im_names', 'index', 'ref_ids', 'ref_names', 'score_type', 'dlevel', ...
    'subjective_scores', 'subjective_scores_STD');
%}





