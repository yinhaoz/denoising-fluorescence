function [I,counts] = scol2im(Z,patchSize,mm,nn,method,weights)
if isa(method,'function_handle')
    t = reshape(1:(mm*nn),[mm nn]);
    temp = im2col(t,[patchSize patchSize]);
    I = reshape(accumarray(temp(:), Z(:), [mm*nn 1], method),[mm nn]);
else
    switch lower(method)
        case 'average'
            t = reshape(1:(mm*nn),[mm nn]);
            temp = im2col(t,[patchSize patchSize]);
            I = reshape(accumarray(temp(:), Z(:), [mm*nn 1], @(x) mean(x)),[mm nn]);
        case 'sum'
            t = reshape(1:(mm*nn),[mm nn]);
            temp = im2col(t,[patchSize patchSize]);
            weights = ones(size(Z));
            counts = reshape(accumarray(temp(:), weights(:), [mm*nn 1], @sum)',[mm nn]);

            I = reshape(accumarray(temp(:), Z(:), [mm*nn 1], @(x) sum(x)),[mm nn]);
        case 'waverage'
            t = reshape(1:(mm*nn),[mm nn]);
            temp = im2col(t,[patchSize patchSize]);

            ws = reshape(accumarray(temp(:), weights(:), [mm*nn 1], @sum)',[mm nn]);
            weights = weights./im2col(ws,[patchSize patchSize]);

            I = reshape(accumarray(temp(:), Z(:).*weights(:), [mm*nn 1], @(x) sum(x)),[mm nn]);
        case 'center'
            I = zeros(mm,nn);

            width = nn;
            height = mm;
            mid = patchSize/2+1;
            mind = mid*patchSize + mid;

            % middle of image
            I(mid:end-mid+2,mid+1:end-mid+3) = reshape(Z(mind,:),[height-patchSize+1 width-patchSize+1]);

            % left bar
            I(mid:end-mid+2,1:mid) = reshape(Z((0:patchSize/2)*patchSize + mid,1:(height-patchSize+1)),[patchSize/2+1 height-patchSize+1])';

            % right bar
            I(mid:end-mid+2,end-mid+2:end) = reshape(Z((patchSize/2:patchSize-1)*patchSize + mid,(width-patchSize)*(height-patchSize+1) + (1:(height-patchSize+1))),[patchSize/2 height-patchSize+1])';

            % top bar
            I(1:mid,mid+1:end-mid+3) = reshape(Z(mid*patchSize + (1:mid),(height-patchSize+1)*(0:width-patchSize)+1),[patchSize/2+1 width-patchSize+1]);

            % bottom bar
            I(end-mid+2:end,mid+1:end-mid+3) = reshape(Z(mid*patchSize + mid - 1 + (1:patchSize/2),(height-patchSize+1)*(0:width-patchSize)+height-patchSize+1),[patchSize/2 width-patchSize+1]);

            % top left corner
            p = reshape(Z(:,1),[patchSize patchSize]);
            I(1:mid-1,1:mid) = p(1:mid-1,1:mid);

            % top right corner
            p = reshape(Z(:,(width-patchSize)*(height-patchSize+1)+1),[patchSize patchSize]);
            I(1:mid-1,end-mid+1:end) = p(1:mid-1,end-mid+1:end);

            % bottom left corner
            p = reshape(Z(:,height - patchSize + 1),[patchSize patchSize]);
            I(end-patchSize/2:end,1:mid) = p(end-patchSize/2:end,1:mid);

            % bottom right corner
            p = reshape(Z(:,end),[patchSize patchSize]);
            I(end-patchSize/2:end,end-mid+1:end) = p(end-patchSize/2:end,end-mid+1:end);
        otherwise
            I = 0;
    end
end