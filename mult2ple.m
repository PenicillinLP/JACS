function output = mult2ple( a,b )

 output = zeros(1,length(b(1,:)));
 for i = 1:length(b(1,:))
     temp = a*b(:,i);
     output(i) = mod(temp,2);
 end
 
end

