function H = HGenerate(n)

N = 2^n;
H = zeros(N,n);

i = 2;

for runners = 1:floor(n/2)
    
    % Runners On Deck!!!
    
    % Let's imagine that they are runners
    pivot = ones(runners,3); % The first columns for start point
    % The second for current point
    % The third for destination
    pivot(:,1) = 1:runners;  % Initiate the start point
    pivot(:,2) = pivot(:,1); % Current point is the start point
    pivot(:,3) = n - runners + (1:runners); % Initiate the destination
    
    % Get! Set! Go!
    current = runners; % The first runner
    onestep = 0; % Move all the way down
    while(1) % While there are still runners
        
        while pivot(current,2) <= pivot(current,3) % Not over yet
            if onestep
                if pivot(current,2) == pivot(current,3) % If this runer also gets to the end
                    pivot(current,2) = pivot(current,2) + 1; % Break the while loop
                else
                    pivot(current,2) = pivot(current,2) + 1; % Only move one step
                    for k = current+1:runners
                        pivot(k,2) = pivot(current,2) + k - current; % All the former runners get back
                    end
                    current = runners; % Start with the first runner again
                    onestep = 0;
                end
            else
                for j=1:runners
                    H(i,pivot(j,2)) = 1; % Record the current positions of all runners
                end
                i = i + 1; % Next line
                pivot(current,2) = pivot(current,2) + 1; % Current runner keeps running
            end
        end
        
        % Now the current runner reachs its destination
        if current == 1 % If this is the last runner
            break
        else % If not
            current = current - 1; % Call the next runner
            onestep = 1;  % Only move one step
        end
        
    end
    
end

for i=N/2+1:N
    H(i,:) = bitxor(ones(1,n),H(N-i+1,:)); 
end

