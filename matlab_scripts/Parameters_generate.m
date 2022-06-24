% Generates the parameter vector of different types.
function theta = Parameters_generate(L, d, rows, cols, type, radius, val)
    switch type
        % The parameter vector is an operator of specified neighbourhood.
        case 'operator'
            center = val;
            values = zeros(1,radius);
            for r=1:radius
                if r<radius/2
                    values(r) = val;
                else
                    values(r) = -val;
                end
            end
            % TODO: change this.
            center = 0.8;
            values = [1 -1.8];
            theta = operator_pass(rows, cols, d, center, values, length(values));
        % The parameter vector is a sobel-like operator constructed from coordinates.
        case 'sobel'
            s = val;
            theta = sobel_operator(rows, cols, d, radius, s);
        case 'd-ball'
        otherwise
            density = 0.2;
            theta = sprandn(1, L*d*L, density);
            theta = reshape(theta, L, d*L);
    end
end

function [theta] = operator_pass(rows, cols, d, center, values, radius)
    L = rows*cols;
    theta = zeros(L, d*L);
    for i = 1:rows
        for j = 1:cols
            temp = zeros(rows, cols);
            temp(i,j)=center;
            for r=1:radius
                c = count_neighbours(i,j,rows,cols,r);
                temp = assign_values(i,j,rows,cols,r,values(r)/c,temp);
            end
            temp = reshape(temp.', 1, []);
            theta((i-1)*cols+j, :) = repmat(temp,1,d);
        end
    end
end

function c = count_neighbours(i,j,rows,cols,r)
    c = 0;
    if(rows > 1)
      for x = max(1, i-r):min(i+r, rows)
        for y = max(1, j-r):min(j+r, cols)
          if(x ~= i || y ~= j)
            if(x==(i+r) || x==(i-r) || y==(j+r) || y==(j-r))
                c=c+1;
            end
          end
        end
      end
    end
end

function temp = assign_values(i,j,rows,cols,r,value,temp)
    if(rows > 1)
      for x = max(1, i-r):min(i+r, rows)
        for y = max(1, j-r):min(j+r, cols)
          if(x ~= i || y ~= j)
            if(x==(i+r) || x==(i-r) || y==(j+r) || y==(j-r))
                temp(x,y)=value;
            end
          end
        end
      end
    end
end

function [theta] = sobel_operator(rows, cols, d, radius, s)
    L = rows*cols;
    theta = zeros(L, d*L);
    for i = 1:rows
        for j = 1:cols
            temp = zeros(rows, cols);
            for x = max(1, i-radius):min(i+radius, rows)
              for y = max(1, j-radius):min(j+radius, cols)
                temp(x,y) = (x-y)^s;
              end
            end
            temp = reshape(temp.', 1, []);
            theta((i-1)*cols+j, :) = repmat(temp,1,d);
        end
    end
end