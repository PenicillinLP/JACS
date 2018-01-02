function [Indicator,Code]=huffman_code(Signal,Probability)
% Record the length.
l_P = length(Probability);

% Create a indicator matrix in the form of CELL.
indicator = [num2cell(Signal);num2cell(Probability)];

% Re-arrange the inputs in an ascending order.
temp = indicator(2,:);
[~, index] = sort(cell2mat(temp));
indicator(1,:) = indicator(1,index);
indicator(2,:) = indicator(2,index);

% Add a row of Rank indicators
indicator = [indicator;num2cell(1:l_P)];

% Create a make matrix
mate = cell(1,l_P);

% Create a working matrix.
probability = sort(Probability);
working_matrix = [probability;1:l_P];

% Create a code matrix.
code_matrix = cell(1,l_P);

% Coding start
i=1;
while length(working_matrix(1,:))>1
    
    rank_one = working_matrix(2,1);
    rank_two = working_matrix(2,2);
    
    code_matrix = update_code_matrix(mate,code_matrix,rank_one,i,1);
    code_matrix = update_code_matrix(mate,code_matrix,rank_two,i,0);
    
    mate = update_mate(mate,indicator,rank_one,rank_two);
    
    working_matrix = update_working_matrix(working_matrix);
    
    i = i+1;
end

Indicator = indicator(1,:);
Code = code_matrix;
end


%%%%%%%

function C_M = update_code_matrix(mate,code_matrix,rank,i,j)

code_matrix{rank} = [num2str(j),code_matrix{rank}];
l_m=length(mate{rank});
k=1;
while k<=l_m
    code_matrix = update_code_matrix(mate,code_matrix,mate{rank}(k),i,j);
    k=k+1;
end
C_M = code_matrix;

end


%%%%%%%

function M = update_mate(mate,indicator,rank_one,rank_two)
% Add the first signal as the mate of the second signal.
mate{rank_two} = [mate{rank_two},indicator{3,rank_one}];
M = mate;
end

%%%%%%%

function W_M = update_working_matrix(working_matrix)

% Add the two probabilities.
working_matrix(1,2)=working_matrix(1,2)+working_matrix(1,1);
% Hide the first signal.
working_matrix(:,1)=[];
% Rearrange the matrix.
if length(working_matrix(1,:))>1
    index = 2;
    while index<=length(working_matrix(1,:)) && working_matrix(1,index)<=working_matrix(1,1)
        index = index+1;
    end
    working_matrix = insert(working_matrix,index-1);
end
W_M = working_matrix;

end

%%%%%%%

function M = insert(matrix,index)

if index~=1
    l = length(matrix(1,:));
    if index == l
        matrix = [matrix,matrix(:,1)];
        matrix(:,1)=[];
    else
        matrix = [matrix(:,2:index),matrix(:,1),matrix(:,index+1:l)];
    end
end

M = matrix;

end



