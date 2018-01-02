function revimage = revZscan( scanned,NY,NX,ZZ )
revimage = zeros(NY*8,NX*8);
for indexy = 1:NY
    for indexx = 1:NX
        temp = scanned(64*(NX*(indexy-1)+indexx-1)+1:64*(NX*(indexy-1)+indexx-1)+64);
        for i = 1:8
            for j = 1:8
                revimage(8*(indexy-1)+i,8*(indexx-1)+j) = temp(ZZ(i,j));
            end
        end
    end
end

