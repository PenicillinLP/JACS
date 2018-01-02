function scanned = Zscan( image,NY,NX,ZZ )
scanned = zeros(NY*NX,64);
temp = zeros(1,64);
for indexy = 1:NY
    for indexx = 1:NX
        tempimage = image(8*(indexy-1)+1:8*(indexy-1)+8,8*(indexx-1)+1:8*(indexx-1)+8);
        for i = 1:8
            for j = 1:8
                temp(ZZ(i,j)) = tempimage(i,j);
            end
        end
        scanned(NX*(indexy-1)+indexx,:) = temp;
    end
end


% function scanned = Zscan_X( image,NY,NX )
% Len = [1 2 3 4 5 6 7 8 7 6 5 4 3 2 1];
% starti = [1 1 3 1 5 1 7 1 8 3 8 5 8 7 8];
% startj = [1 2 1 4 1 6 1 8 2 8 4 8 6 8 8];
% Rou = 15;
% scanned = zeros(NY*NX,64);
% temp = zeros(1,64);
% for indexy = 1:NY
%     for indexx = 1:NX
%         tempimage = image(8*(indexy-1)+1:8*(indexy-1)+8,8*(indexx-1)+1:8*(indexx-1)+8);
%         temp(1) = tempimage(1,1);
%         direction = -1;
%         for rou = 2:Rou
%             i = starti(rou);
%             j = startj(rou);
%             tempLen = Len(rou);
%             len = 1;
%             temp(sum(Len(1:(rou-1)))+len) = tempimage(i,j);
%             while len < tempLen
%                 len = len + 1;
%                 i = i - direction;
%                 j = j + direction;
%                 temp(sum(Len(1:(rou-1)))+len) = tempimage(i,j);
%             end
%             direction = direction*(-1);
%         end
%         scanned(NX*(indexy-1)+indexx,:) = temp;
%     end
% end