function  [E_Img]  =  Patch2Im( ImPat, WPat, PatSize, ImageH, ImageW )
    TempR        =   ImageH-PatSize+1;
    TempC        =   ImageW-PatSize+1;
    TempOffsetR  =   [1:TempR];
    TempOffsetC  =   [1:TempC];    

    E_Img  	=  zeros(ImageH,ImageW);
    W_Img 	=  zeros(ImageH,ImageW);
    k        =   0;
    for i  = 1:PatSize
        for j  = 1:PatSize
            k    =  k+1;
            E_Img(TempOffsetR-1+i,TempOffsetC-1+j)  =  E_Img(TempOffsetR-1+i,TempOffsetC-1+j) + reshape( ImPat(k,:)', [TempR TempC]);
            W_Img(TempOffsetR-1+i,TempOffsetC-1+j)  =  W_Img(TempOffsetR-1+i,TempOffsetC-1+j) + reshape( WPat(k,:)',  [TempR TempC]);
        end
    end
     E_Img  =  E_Img./(W_Img+eps);

