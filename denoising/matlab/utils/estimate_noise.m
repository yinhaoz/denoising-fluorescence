function [fitparams] = estimate_noise(z)

    polyorder=1;    
    variance_power=1;
    clipping_below=1;
    clipping_above=1; 
    prior_density=1;  
    median_est=1;  
    LS_median_size=1;   
    tau_threshold_initial=1; 
    level_set_density_factor=1;   
    integral_resolution_factor=1;    
    speed_factor=1;   
    text_verbosity=0;       
    figure_verbosity=0;   
    lambda=1; 
    auto_lambda=1; 
    
    fitparams = function_ClipPoisGaus_stdEst2D(z,polyorder,variance_power,...
        clipping_below,clipping_above,prior_density,median_est,...
        LS_median_size,tau_threshold_initial,level_set_density_factor,...
        integral_resolution_factor,speed_factor,text_verbosity,...
        figure_verbosity,lambda,auto_lambda);
end