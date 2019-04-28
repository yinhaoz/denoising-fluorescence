function [img_array] = import_img_array(directory)

    % import the noisy images
    files = dir([directory, '*.png']);      
    n_files = length(files); 
    cell_imgs = cell(n_files, 1);
    for i_img=1:n_files
       filename = files(i_img).name;
       % Read a grayscale image and scale its intensities in range [0,1]
       current_image = im2double(imread([directory, filename]));
       cell_imgs{i_img} = current_image;
    end


    % select the noisy image
    img_array = cat(3,cell_imgs{:});
    
end