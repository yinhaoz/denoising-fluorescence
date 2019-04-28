function [img_indices] = avg_select(n_img, n_select, n_repeat)
    % function to generate indices corresponding to n_select images
    % selected from n_img images; this procedure is repeated for n_repeat
    % times
    
    % use the same seed for rng
    rng(0);
    
    img_indices = zeros(n_repeat, n_select);
    
    for i_repeat = 1:n_repeat
        
        img_indices(i_repeat, :) = randperm(n_img, n_select);
        
    end

end